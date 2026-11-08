"""Unit tests for Stage D2: Predictive Autoscaling."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from autoscaling.base_scaler import ScalerConfig, ScalingDecision
from autoscaling.predictive_scaler import PredictiveScaler
from models.predictor import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_HORIZON,
    MIN_HISTORICAL_OBSERVATIONS,
)
from simulation.cloud_simulator import CloudSimulator, ServerConfig
from simulation.workload_generator import generate_normal_traffic


class TestPredictiveScaler(unittest.TestCase):
    """Test suite covering PredictiveScaler policy, forecasting integration, hysteresis, and boundaries."""

    def setUp(self) -> None:
        """Initialize simulator, configuration, and default test harness."""
        self.server_config = ServerConfig(
            capacity_per_server=1000,
            min_servers=2,
            max_servers=30,
            initial_servers=2,
        )
        self.simulator = CloudSimulator(config=self.server_config)

        # Standard 6-observation historical baseline for minute 5
        self.baseline_history = pd.DataFrame(
            {
                "minute": [0, 1, 2, 3, 4, 5],
                "viewers": [1000, 1050, 1100, 1150, 1200, 1250],
            }
        )

    # Test 1: Predictive scale-up when forecast demand requires more servers.
    def test_predictive_scale_up_when_forecast_demands_more_servers(self) -> None:
        """Verify proactive scale-up is triggered before load arrives when predicted demand exceeds capacity."""
        # 2 active servers (capacity 2,000, target 70% capacity = 1,400)
        # Predict 2,800 viewers at t+5 -> requires ceil(2800 / 700) = 4 servers
        scaler = PredictiveScaler(
            min_servers=2,
            max_servers=30,
            target_cpu_utilization=70.0,
            capacity_per_server=1000,
            predictor=lambda hist: 2800,
        )

        decision = scaler.make_decision(
            current_interval=5,
            current_viewers=1250,
            current_active_servers=2,
            current_cpu_utilization=62.5,
            simulator=self.simulator,
            viewer_history=self.baseline_history,
        )

        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.previous_servers, 2)
        self.assertEqual(decision.desired_servers, 4)
        self.assertEqual(decision.actual_servers, 4)
        self.assertEqual(decision.predicted_viewers, 2800)
        self.assertTrue(decision.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 4)
        self.assertEqual(scaler.scale_up_count, 1)
        self.assertEqual(scaler.last_scaling_interval, 5)
        self.assertIn("Scaled up proactively from 2 to 4 servers", decision.reason)

    # Test 2: Predictive scale-up is deferred during cooldown.
    def test_predictive_scale_up_deferred_during_cooldown(self) -> None:
        """Verify proactive scale-up is deferred if the cooldown period has not expired."""
        scaler = PredictiveScaler(
            cooldown_intervals=2,
            predictor=lambda hist: 3500,  # Demands 5 servers
        )

        # Action 1 at interval 10: scale up from 2 to 5 servers
        hist10 = pd.DataFrame({"minute": range(5, 11), "viewers": [1250] * 6})
        d0 = scaler.make_decision(
            current_interval=10,
            current_viewers=1250,
            current_active_servers=2,
            current_cpu_utilization=62.5,
            simulator=self.simulator,
            viewer_history=hist10,
        )
        self.assertEqual(d0.action, "scale_up")
        self.assertEqual(d0.actual_servers, 5)
        self.assertTrue(d0.scaled)

        # Interval 11: High forecast (e.g. 4,500 demands 7 servers), but in cooldown (elapsed = 1 < 2)
        scaler.predictor = lambda hist: 4500
        hist11 = pd.DataFrame({"minute": range(6, 12), "viewers": [1250] * 6})
        d1 = scaler.make_decision(
            current_interval=11,
            current_viewers=1250,
            current_active_servers=5,
            current_cpu_utilization=25.0,
            simulator=self.simulator,
            viewer_history=hist11,
        )
        self.assertEqual(d1.action, "no_action")
        self.assertEqual(d1.desired_servers, 7)
        self.assertEqual(d1.actual_servers, 5)
        self.assertFalse(d1.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 5)
        self.assertIn("deferred due to active cooldown", d1.reason)
        self.assertEqual(scaler.scale_up_count, 1)

        # Interval 12: Cooldown expired (elapsed = 2 >= 2) -> can scale up
        hist12 = pd.DataFrame({"minute": range(7, 13), "viewers": [1250] * 6})
        d2 = scaler.make_decision(
            current_interval=12,
            current_viewers=1250,
            current_active_servers=5,
            current_cpu_utilization=25.0,
            simulator=self.simulator,
            viewer_history=hist12,
        )
        self.assertEqual(d2.action, "scale_up")
        self.assertEqual(d2.actual_servers, 7)
        self.assertTrue(d2.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 7)
        self.assertEqual(scaler.scale_up_count, 2)

    # Test 3: Predictive scale-down after required consecutive low-demand intervals.
    def test_predictive_scale_down_after_consecutive_low_demand(self) -> None:
        """Verify scale-down executes after 3 consecutive intervals of low projected CPU."""
        self.simulator.add_server()  # 3 servers (capacity 3,000)
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

        # Low forecast: 600 viewers -> projected CPU = 600 / 3000 = 20.0% (< 30%)
        scaler = PredictiveScaler(
            min_servers=2,
            scale_down_cpu_threshold=30.0,
            scale_down_consecutive_intervals=3,
            predictor=lambda hist: 600,
        )

        # Intervals 10 and 11: low projected demand accumulates
        for idx, interval in enumerate([10, 11]):
            hist = pd.DataFrame({"minute": range(interval - 5, interval + 1), "viewers": [600] * 6})
            d = scaler.make_decision(interval, 600, 3, 20.0, simulator=self.simulator, viewer_history=hist)
            self.assertEqual(d.action, "no_action")
            self.assertFalse(d.scaled)
            self.assertEqual(scaler.consecutive_low_demand_count, idx + 1)
            self.assertEqual(len(self.simulator.get_active_servers()), 3)

        # Interval 12: 3rd low demand interval -> executes conservative scale down by 1
        hist12 = pd.DataFrame({"minute": range(7, 13), "viewers": [600] * 6})
        d2 = scaler.make_decision(12, 600, 3, 20.0, simulator=self.simulator, viewer_history=hist12)
        self.assertEqual(d2.action, "scale_down")
        self.assertEqual(d2.previous_servers, 3)
        self.assertEqual(d2.desired_servers, 2)
        self.assertEqual(d2.actual_servers, 2)
        self.assertTrue(d2.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        self.assertEqual(scaler.scale_down_count, 1)
        self.assertEqual(scaler.last_scaling_interval, 12)
        self.assertEqual(scaler.consecutive_low_demand_count, 0)
        self.assertIn("removed server-3", d2.reason)

    # Test 4: No scale-down before the hysteresis threshold is reached.
    def test_no_scale_down_before_hysteresis_threshold(self) -> None:
        """Verify scale-down is NOT triggered on 1st or 2nd low projected demand observation."""
        self.simulator.add_server()  # 3 servers
        scaler = PredictiveScaler(predictor=lambda hist: 500)

        hist10 = pd.DataFrame({"minute": range(5, 11), "viewers": [500] * 6})
        d0 = scaler.make_decision(10, 500, 3, 16.7, simulator=self.simulator, viewer_history=hist10)
        self.assertEqual(d0.action, "no_action")
        self.assertEqual(scaler.consecutive_low_demand_count, 1)

        hist11 = pd.DataFrame({"minute": range(6, 12), "viewers": [500] * 6})
        d1 = scaler.make_decision(11, 500, 3, 16.7, simulator=self.simulator, viewer_history=hist11)
        self.assertEqual(d1.action, "no_action")
        self.assertEqual(scaler.consecutive_low_demand_count, 2)
        self.assertEqual(scaler.scale_down_count, 0)
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

    # Test 5: No scale-down below minimum server count.
    def test_no_scale_down_below_minimum_server_count(self) -> None:
        """Verify scaler never scales below min_servers (2) even during prolonged low demand."""
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        scaler = PredictiveScaler(min_servers=2, predictor=lambda hist: 200)

        for minute in range(10, 15):
            hist = pd.DataFrame({"minute": range(minute - 5, minute + 1), "viewers": [200] * 6})
            decision = scaler.make_decision(
                current_interval=minute,
                current_viewers=200,
                current_active_servers=2,
                current_cpu_utilization=10.0,
                simulator=self.simulator,
                viewer_history=hist,
            )
            self.assertEqual(decision.action, "no_action")
            self.assertEqual(decision.actual_servers, 2)
            self.assertFalse(decision.scaled)
            self.assertEqual(len(self.simulator.get_active_servers()), 2)

        self.assertEqual(scaler.scale_down_count, 0)

    # Test 6: Respect maximum server count.
    def test_respect_maximum_server_count(self) -> None:
        """Verify proactive scale-up is strictly clamped at max_servers (30)."""
        for _ in range(27):
            self.simulator.add_server()  # 29 servers
        self.assertEqual(len(self.simulator.get_active_servers()), 29)

        # Huge forecast (60,000 viewers)
        scaler = PredictiveScaler(max_servers=30, predictor=lambda hist: 60000)

        hist10 = pd.DataFrame({"minute": range(5, 11), "viewers": [2000] * 6})
        decision = scaler.make_decision(
            current_interval=10,
            current_viewers=2000,
            current_active_servers=29,
            current_cpu_utilization=6.9,
            simulator=self.simulator,
            viewer_history=hist10,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.desired_servers, 30)
        self.assertEqual(decision.actual_servers, 30)
        self.assertTrue(decision.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 30)

        # Second massive interval while already at 30 servers
        scaler.last_scaling_interval = None  # clear cooldown to test boundary logic
        hist15 = pd.DataFrame({"minute": range(10, 16), "viewers": [2000] * 6})
        d2 = scaler.make_decision(
            current_interval=15,
            current_viewers=2000,
            current_active_servers=30,
            current_cpu_utilization=6.7,
            simulator=self.simulator,
            viewer_history=hist15,
        )
        self.assertEqual(d2.action, "no_action")
        self.assertEqual(d2.actual_servers, 30)
        self.assertFalse(d2.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 30)

    # Test 7: Scale-up with simulator=None does not modify actual server count or scaling counters.
    def test_scale_up_without_simulator(self) -> None:
        """Verify predictive scale-up without simulator returns requested target without claiming mutation."""
        scaler = PredictiveScaler(predictor=lambda hist: 2800)
        decision = scaler.make_decision(
            current_interval=5,
            current_viewers=1250,
            current_active_servers=2,
            current_cpu_utilization=62.5,
            simulator=None,
            viewer_history=self.baseline_history,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.previous_servers, 2)
        self.assertEqual(decision.desired_servers, 4)
        self.assertEqual(decision.actual_servers, 2)  # Unchanged
        self.assertFalse(decision.scaled)            # False because infrastructure did not change
        self.assertEqual(scaler.scale_up_count, 0)   # Unchanged
        self.assertIsNone(scaler.last_scaling_interval)  # Unchanged
        self.assertIn("no simulator provided", decision.reason)
        self.assertIn("unchanged", decision.reason)

    # Test 8: Scale-down with simulator=None does not modify actual server count or scaling counters.
    def test_scale_down_without_simulator(self) -> None:
        """Verify predictive scale-down without simulator returns requested target without claiming mutation."""
        scaler = PredictiveScaler(predictor=lambda hist: 500)

        # Accumulate 2 low demand intervals
        for m in [10, 11]:
            h = pd.DataFrame({"minute": range(m - 5, m + 1), "viewers": [500] * 6})
            scaler.make_decision(m, 500, 4, 12.5, simulator=None, viewer_history=h)

        # 3rd interval triggers scale_down decision
        h12 = pd.DataFrame({"minute": range(7, 13), "viewers": [500] * 6})
        decision = scaler.make_decision(
            current_interval=12,
            current_viewers=500,
            current_active_servers=4,
            current_cpu_utilization=12.5,
            simulator=None,
            viewer_history=h12,
        )
        self.assertEqual(decision.action, "scale_down")
        self.assertEqual(decision.previous_servers, 4)
        self.assertEqual(decision.desired_servers, 3)
        self.assertEqual(decision.actual_servers, 4)  # Unchanged
        self.assertFalse(decision.scaled)            # False because infrastructure did not change
        self.assertEqual(scaler.scale_down_count, 0)  # Unchanged
        self.assertIsNone(scaler.last_scaling_interval)  # Unchanged
        self.assertIn("no simulator provided", decision.reason)
        self.assertIn("unchanged", decision.reason)

    # Test 9: Successful scale-down removes exactly one server and identifies its ID.
    def test_scale_down_identifies_removed_server(self) -> None:
        """Verify scale-down with simulator decommissions exactly one server and states its ID."""
        self.simulator.add_server()  # add server-3
        active_ids_before = [s.server_id for s in self.simulator.get_active_servers()]
        self.assertEqual(len(active_ids_before), 3)
        self.assertIn("server-3", active_ids_before)

        scaler = PredictiveScaler(predictor=lambda hist: 500)

        # Accumulate 3 low demand intervals
        for m in [10, 11]:
            h = pd.DataFrame({"minute": range(m - 5, m + 1), "viewers": [500] * 6})
            scaler.make_decision(m, 500, 3, 16.7, simulator=self.simulator, viewer_history=h)

        h12 = pd.DataFrame({"minute": range(7, 13), "viewers": [500] * 6})
        decision = scaler.make_decision(12, 500, 3, 16.7, simulator=self.simulator, viewer_history=h12)

        self.assertEqual(decision.action, "scale_down")
        self.assertEqual(decision.previous_servers, 3)
        self.assertEqual(decision.desired_servers, 2)
        self.assertEqual(decision.actual_servers, 2)
        self.assertTrue(decision.scaled)
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        self.assertIn("removed server-3", decision.reason)

        active_ids_after = [s.server_id for s in self.simulator.get_active_servers()]
        self.assertNotIn("server-3", active_ids_after)
        self.assertEqual(active_ids_after, ["server-1", "server-2"])

    # Test 10: Prediction uses only historical observations available up to current interval.
    def test_prediction_uses_only_historical_observations(self) -> None:
        """Verify that the prediction function receives observations only up to interval t."""
        received_history = []

        def spy_predictor(hist: pd.DataFrame) -> int:
            received_history.append(hist.copy())
            return 2100

        scaler = PredictiveScaler(predictor=spy_predictor)
        current_minute = 10
        hist_df = pd.DataFrame(
            {"minute": list(range(5, 11)), "viewers": [1500 + i * 10 for i in range(6)]}
        )

        scaler.make_decision(
            current_interval=current_minute,
            current_viewers=hist_df["viewers"].iloc[-1],
            current_active_servers=2,
            current_cpu_utilization=78.0,
            simulator=self.simulator,
            viewer_history=hist_df,
        )

        self.assertEqual(len(received_history), 1)
        passed_df = received_history[0]
        self.assertEqual(passed_df["minute"].max(), current_minute)
        self.assertNotIn(current_minute + 1, passed_df["minute"].values)

    # Test 11: Insufficient history is handled safely without fabricating data.
    def test_insufficient_history_handled_safely(self) -> None:
        """Verify scaler returns safe no_action without predicting when history has < 6 rows."""
        scaler = PredictiveScaler(predictor=lambda hist: 5000)

        # 3 observations (< 6)
        short_history = pd.DataFrame({"minute": [0, 1, 2], "viewers": [1000, 1100, 1200]})

        decision = scaler.make_decision(
            current_interval=2,
            current_viewers=1200,
            current_active_servers=2,
            current_cpu_utilization=60.0,
            simulator=self.simulator,
            viewer_history=short_history,
        )

        self.assertEqual(decision.action, "no_action")
        self.assertIsNone(decision.predicted_viewers)
        self.assertEqual(decision.actual_servers, 2)
        self.assertFalse(decision.scaled)
        self.assertIn("Insufficient historical observations", decision.reason)
        self.assertIn("3/6", decision.reason)

        # None history
        decision_none = scaler.make_decision(
            current_interval=0,
            current_viewers=1000,
            current_active_servers=2,
            current_cpu_utilization=50.0,
            simulator=self.simulator,
            viewer_history=None,
        )
        self.assertEqual(decision_none.action, "no_action")
        self.assertIsNone(decision_none.predicted_viewers)
        self.assertFalse(decision_none.scaled)

    # Test 12: Sequence history format (list of ints) is supported.
    def test_sequence_history_support(self) -> None:
        """Verify list/sequence of integers is cleanly accepted as viewer_history."""
        scaler = PredictiveScaler(predictor=lambda hist: 2100)
        history_list = [1000, 1050, 1100, 1150, 1200, 1250]

        decision = scaler.make_decision(
            current_interval=5,
            current_viewers=1250,
            current_active_servers=2,
            current_cpu_utilization=62.5,
            simulator=self.simulator,
            viewer_history=history_list,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.predicted_viewers, 2100)
        self.assertTrue(decision.scaled)

    # Test 13: History mismatch triggers descriptive ValueError.
    def test_history_mismatch_validation(self) -> None:
        """Verify ValueError is raised if the latest history row contradicts current interval/viewers."""
        scaler = PredictiveScaler(predictor=lambda hist: 2000)

        # Viewers mismatch
        mismatch_viewers = pd.DataFrame({"minute": range(6), "viewers": [1000] * 5 + [9999]})
        with self.assertRaises(ValueError) as ctx:
            scaler.make_decision(5, 1250, 2, 62.5, viewer_history=mismatch_viewers)
        self.assertIn("must match current_viewers", str(ctx.exception))

        # Minute mismatch (history ending earlier than current_interval)
        mismatch_minute = pd.DataFrame({"minute": range(0, 5), "viewers": [1250] * 5})
        with self.assertRaises(ValueError) as ctx2:
            scaler.make_decision(5, 1250, 2, 62.5, viewer_history=mismatch_minute)
        self.assertIn("must match current_interval", str(ctx2.exception))

    # Test 14: Integration test with the real trained model artifact.
    def test_real_model_integration(self) -> None:
        """Verify end-to-end integration using the actual trained Random Forest model artifact."""
        if not DEFAULT_MODEL_PATH.is_file():
            self.skipTest(f"Model artifact not found at {DEFAULT_MODEL_PATH}")

        scaler = PredictiveScaler(model_path=DEFAULT_MODEL_PATH)

        # Generate realistic initial traffic history using Stage A normal scenario
        df = generate_normal_traffic(duration_minutes=20, random_seed=42)
        history_slice = df.iloc[:10].copy()
        current_minute = int(history_slice["minute"].iloc[-1])
        current_viewers = int(history_slice["viewers"].iloc[-1])

        decision = scaler.make_decision(
            current_interval=current_minute,
            current_viewers=current_viewers,
            current_active_servers=2,
            current_cpu_utilization=current_viewers / 20.0,
            simulator=self.simulator,
            viewer_history=history_slice,
        )

        self.assertIn(decision.action, {"scale_up", "scale_down", "no_action"})
        self.assertIsInstance(decision.predicted_viewers, int)
        self.assertGreater(decision.predicted_viewers, 0)
        self.assertEqual(decision.actual_servers, len(self.simulator.get_active_servers()))

    # --- Targeted Audit Regression Tests ---

    # Regression Test 15: DataFrame containing a future minute in a non-final row is rejected.
    def test_future_minute_in_history_rejected(self) -> None:
        """Verify DataFrame with a future minute value in any row is rejected."""
        scaler = PredictiveScaler(predictor=lambda hist: 2000)
        # Non-final row has minute 20 > current_interval (10)
        future_history = pd.DataFrame(
            {"minute": [5, 6, 20, 8, 9, 10], "viewers": [1000] * 6}
        )
        with self.assertRaises(ValueError) as ctx:
            scaler.make_decision(10, 1000, 2, 50.0, viewer_history=future_history)
        self.assertIn("Future data is prohibited", str(ctx.exception))

    # Regression Test 16: DataFrame with duplicate minutes is rejected.
    def test_duplicate_minutes_in_history_rejected(self) -> None:
        """Verify DataFrame containing duplicate minute values is rejected."""
        scaler = PredictiveScaler(predictor=lambda hist: 2000)
        duplicate_history = pd.DataFrame(
            {"minute": [5, 6, 7, 7, 9, 10], "viewers": [1000] * 6}
        )
        with self.assertRaises(ValueError) as ctx:
            scaler.make_decision(10, 1000, 2, 50.0, viewer_history=duplicate_history)
        self.assertIn("duplicate minute values", str(ctx.exception))

    # Regression Test 17: DataFrame with missing/non-consecutive minutes is rejected.
    def test_non_consecutive_minutes_in_history_rejected(self) -> None:
        """Verify DataFrame with missing intervals (gaps) is rejected."""
        scaler = PredictiveScaler(predictor=lambda hist: 2000)
        gapped_history = pd.DataFrame(
            {"minute": [5, 6, 7, 9, 10], "viewers": [1000] * 5}
        )
        with self.assertRaises(ValueError) as ctx:
            scaler.make_decision(10, 1000, 2, 50.0, viewer_history=gapped_history)
        self.assertIn("consecutive", str(ctx.exception).lower())

    # Regression Test 18: Unsorted but otherwise valid history is handled correctly.
    def test_unsorted_valid_history_handled_correctly(self) -> None:
        """Verify valid history provided in scrambled chronological order is sorted and accepted."""
        scaler = PredictiveScaler(predictor=lambda hist: 2100)
        unsorted_history = pd.DataFrame(
            {"minute": [8, 5, 10, 6, 9, 7], "viewers": [1080, 1050, 1250, 1060, 1090, 1070]}
        )
        decision = scaler.make_decision(
            current_interval=10,
            current_viewers=1250,
            current_active_servers=2,
            current_cpu_utilization=62.5,
            simulator=self.simulator,
            viewer_history=unsorted_history,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.predicted_viewers, 2100)

    # Regression Test 19: Sequence history that would imply a negative starting minute is rejected.
    def test_negative_starting_minute_in_sequence_rejected(self) -> None:
        """Verify sequence history of length > current_interval + 1 is rejected (prevents negative time)."""
        scaler = PredictiveScaler(predictor=lambda hist: 2000)
        # At interval 3, a sequence of 8 observations implies starting at 3 - 8 + 1 = -4 < 0
        long_seq = [1000] * 8
        with self.assertRaises(ValueError) as ctx:
            scaler.make_decision(3, 1000, 2, 50.0, viewer_history=long_seq)
        self.assertIn("negative starting minute", str(ctx.exception))

    # Regression Test 20: Sequence history with a valid non-negative starting minute remains supported.
    def test_valid_sequence_history_supported(self) -> None:
        """Verify sequence history whose implied start minute >= 0 succeeds without error."""
        scaler = PredictiveScaler(predictor=lambda hist: 2100)
        # At interval 5, a sequence of 6 observations implies start minute 5 - 6 + 1 = 0 >= 0
        valid_seq = [1000, 1050, 1100, 1150, 1200, 1250]
        decision = scaler.make_decision(
            current_interval=5,
            current_viewers=1250,
            current_active_servers=2,
            current_cpu_utilization=62.5,
            simulator=self.simulator,
            viewer_history=valid_seq,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.actual_servers, 3)

    # Regression Test 21: Unsupported prediction horizon is rejected.
    def test_unsupported_prediction_horizon_rejected(self) -> None:
        """Verify configuring any horizon other than 5 raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PredictiveScaler(prediction_horizon=10)
        self.assertIn("Unsupported prediction_horizon 10", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            PredictiveScaler(prediction_horizon=3)
        self.assertIn("Unsupported prediction_horizon 3", str(ctx2.exception))

        # Default 5 succeeds
        scaler = PredictiveScaler(prediction_horizon=DEFAULT_PREDICTION_HORIZON)
        self.assertEqual(scaler.prediction_horizon, 5)

    # Regression Test 22: ScalingDecision.scaled semantics for successful scaling vs requests without changes.
    def test_scaling_decision_scaled_semantics(self) -> None:
        """Verify ScalingDecision.scaled accurately reflects physical infrastructure mutation."""
        # Case 1: scale_up without infrastructure mutation -> scaled is False
        d_req_up = ScalingDecision(
            action="scale_up",
            previous_servers=2,
            desired_servers=4,
            actual_servers=2,
            reason="Scale up requested (no simulator)",
        )
        self.assertFalse(d_req_up.scaled)

        # Case 2: scale_down without infrastructure mutation -> scaled is False
        d_req_down = ScalingDecision(
            action="scale_down",
            previous_servers=4,
            desired_servers=3,
            actual_servers=4,
            reason="Scale down requested (no simulator)",
        )
        self.assertFalse(d_req_down.scaled)

        # Case 3: no_action -> scaled is False
        d_none = ScalingDecision(
            action="no_action",
            previous_servers=2,
            desired_servers=2,
            actual_servers=2,
            reason="Normal",
        )
        self.assertFalse(d_none.scaled)

        # Case 4: successful scale_up mutation -> scaled is True
        d_up = ScalingDecision(
            action="scale_up",
            previous_servers=2,
            desired_servers=4,
            actual_servers=4,
            reason="Scaled up",
        )
        self.assertTrue(d_up.scaled)

        # Case 5: successful scale_down mutation -> scaled is True
        d_down = ScalingDecision(
            action="scale_down",
            previous_servers=3,
            desired_servers=2,
            actual_servers=2,
            reason="Scaled down",
        )
        self.assertTrue(d_down.scaled)

    # Regression Test 23: Negative minute values in DataFrame history are rejected.
    def test_negative_minute_in_dataframe_history_rejected(self) -> None:
        """Verify DataFrame with explicit negative minute values is rejected."""
        scaler = PredictiveScaler(predictor=lambda hist: 2000)
        neg_df = pd.DataFrame({"minute": [-2, -1, 0, 1, 2, 3], "viewers": [1000] * 6})
        with self.assertRaises(ValueError) as ctx:
            scaler.make_decision(3, 1000, 2, 50.0, viewer_history=neg_df)
        self.assertIn("negative minute values", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
