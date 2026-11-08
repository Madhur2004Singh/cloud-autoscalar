"""Unit tests for Stage D3: Integrated Autoscaling Comparison Experiment."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from autoscaling.base_scaler import ScalerConfig, ScalingDecision
from autoscaling.predictive_scaler import PredictiveScaler
from autoscaling.reactive_scaler import ReactiveScaler
from run_autoscaling_comparison import (
    calculate_summary_metrics,
    generate_comparison_visualizations,
    load_raw_scenarios,
    run_autoscaling_comparison,
    run_strategy_simulation,
)
from simulation.cloud_simulator import CloudSimulator, ServerConfig


class TestAutoscalingExperiment(unittest.TestCase):
    """Test suite covering Stage D3 comparison experiment execution, integrity, and metrics."""

    def setUp(self) -> None:
        """Initialize reusable synthetic datasets and configurations."""
        self.server_config = ServerConfig(
            capacity_per_server=1000,
            min_servers=2,
            max_servers=30,
            initial_servers=2,
        )
        self.scaler_config = ScalerConfig(
            min_servers=2,
            max_servers=30,
            target_cpu_utilization=70.0,
            scale_down_cpu_threshold=30.0,
            cooldown_intervals=2,
            scale_down_consecutive_intervals=3,
            capacity_per_server=1000,
        )

        # Deterministic 15-minute synthetic workload trace
        # Minutes 0-4: steady 1200 viewers (insufficient history for predictive)
        # Minute 5-8: ramp to 3500 viewers (demands scale-up)
        # Minute 9-14: drop to 500 viewers (demands scale-down after cooldown/consecutive)
        self.sample_trace = pd.DataFrame(
            {
                "minute": list(range(15)),
                "viewers": [
                    1200,
                    1200,
                    1200,
                    1200,
                    1200,
                    2800,
                    3200,
                    3500,
                    3000,
                    500,
                    500,
                    500,
                    500,
                    500,
                    500,
                ],
                "scenario": ["Synthetic Test"] * 15,
            }
        )

    # 1. Test both strategies can run on all four scenarios
    def test_both_strategies_run_on_all_four_scenarios(self) -> None:
        """Verify that ReactiveScaler and PredictiveScaler successfully execute on all 4 scenarios."""
        scenarios = load_raw_scenarios()
        self.assertEqual(len(scenarios), 4)
        expected_keys = {"normal", "sudden_spike", "peak_event", "event_ending"}
        self.assertEqual(set(scenarios.keys()), expected_keys)

        # Run on a 10-minute slice of each scenario to verify end-to-end execution across all 4
        for name, df in scenarios.items():
            sub_df = df.iloc[:10].copy()
            sc_name = sub_df["scenario"].iloc[0]

            # Reactive execution
            m_react, d_react = run_strategy_simulation(
                scenario_name=sc_name,
                df=sub_df,
                strategy="Reactive",
                server_config=self.server_config,
                scaler_config=self.scaler_config,
            )
            self.assertEqual(len(m_react), 10)
            self.assertEqual(len(d_react), 10)

            # Predictive execution (uses mock predictor for fast unit testing)
            mock_scaler = PredictiveScaler(
                config=self.scaler_config,
                predictor=lambda hist: 2500,
            )
            m_pred, d_pred = run_strategy_simulation(
                scenario_name=sc_name,
                df=sub_df,
                strategy="Predictive",
                server_config=self.server_config,
                scaler_config=self.scaler_config,
                scaler_instance=mock_scaler,
            )
            self.assertEqual(len(m_pred), 10)
            self.assertEqual(len(d_pred), 10)

    # 2. Test each strategy starts from an independent simulator
    def test_each_strategy_starts_from_independent_simulator(self) -> None:
        """Verify that both strategies run on completely independent simulator instances."""
        sim_react = CloudSimulator(config=self.server_config)
        sim_pred = CloudSimulator(config=self.server_config)

        # Mutate reactive simulator to 5 servers
        sim_react.add_server()
        sim_react.add_server()
        sim_react.add_server()
        self.assertEqual(len(sim_react.get_active_servers()), 5)

        # Verify predictive simulator is completely unaffected and remains at initial 2 servers
        self.assertEqual(len(sim_pred.get_active_servers()), 2)
        self.assertIsNot(sim_react, sim_pred)

        # In run_strategy_simulation, if no simulator is passed, a fresh one is always created
        m_r, _ = run_strategy_simulation(
            "Test", self.sample_trace.iloc[:3], "Reactive", server_config=self.server_config
        )
        m_p, _ = run_strategy_simulation(
            "Test",
            self.sample_trace.iloc[:3],
            "Predictive",
            server_config=self.server_config,
            scaler_instance=PredictiveScaler(predictor=lambda h: 1000),
        )
        self.assertEqual(m_r[0]["active_servers"], 2)
        self.assertEqual(m_p[0]["active_servers"], 2)

    # 3. Test both strategies receive identical workload traces
    def test_both_strategies_receive_identical_workload_traces(self) -> None:
        """Verify that both strategies ingest identical minute-by-minute viewer workloads."""
        mock_pred = PredictiveScaler(predictor=lambda h: 2000)
        m_react, _ = run_strategy_simulation(
            scenario_name="Synthetic",
            df=self.sample_trace,
            strategy="Reactive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
        )
        m_pred, _ = run_strategy_simulation(
            scenario_name="Synthetic",
            df=self.sample_trace,
            strategy="Predictive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
            scaler_instance=mock_pred,
        )

        self.assertEqual(len(m_react), len(m_pred))
        for r, p in zip(m_react, m_pred):
            self.assertEqual(r["minute"], p["minute"])
            self.assertEqual(r["actual_viewers"], p["actual_viewers"])

    # 4. Test PredictiveScaler receives no future observations
    def test_predictive_scaler_receives_no_future_observations(self) -> None:
        """Verify that PredictiveScaler receives strictly history up to current interval t without leakage."""
        observed_histories: List[pd.DataFrame] = []

        def tracking_predictor(hist: pd.DataFrame) -> int:
            observed_histories.append(hist.copy())
            return int(hist["viewers"].iloc[-1])

        scaler = PredictiveScaler(
            config=self.scaler_config,
            predictor=tracking_predictor,
        )

        run_strategy_simulation(
            scenario_name="Synthetic",
            df=self.sample_trace,
            strategy="Predictive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
            scaler_instance=scaler,
        )

        # Observations start at minute 5 (after 6 observations)
        self.assertGreater(len(observed_histories), 0)
        for hist in observed_histories:
            latest_minute = int(hist["minute"].iloc[-1])
            # Assert all minutes in history are <= latest_minute
            self.assertTrue(all(m <= latest_minute for m in hist["minute"]))
            # Assert no future minute exists beyond latest_minute
            self.assertFalse(any(m > latest_minute for m in hist["minute"]))

    # 5. Test PredictiveScaler handles its initial insufficient-history intervals correctly
    def test_predictive_scaler_handles_insufficient_history_correctly(self) -> None:
        """Verify that PredictiveScaler returns no_action and predicted_viewers=None during warm-up."""
        scaler = PredictiveScaler(
            config=self.scaler_config,
            predictor=lambda hist: 5000,
        )

        m_pred, d_pred = run_strategy_simulation(
            scenario_name="Synthetic",
            df=self.sample_trace,
            strategy="Predictive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
            scaler_instance=scaler,
        )

        # Minutes 0 to 4 have insufficient history (< 6 observations)
        for t in range(5):
            self.assertEqual(d_pred[t]["action"], "no_action")
            self.assertIsNone(d_pred[t]["predicted_viewers"])
            self.assertIn("Insufficient historical observations", d_pred[t]["reason"])
            self.assertIsNone(m_pred[t]["predicted_viewers"])

        # Minute 5 has 6 observations (minutes 0, 1, 2, 3, 4, 5) -> forecast active
        self.assertEqual(d_pred[5]["predicted_viewers"], 5000)
        self.assertEqual(d_pred[5]["action"], "scale_up")

    # 6. Test scaling decisions affect subsequent intervals, not previously recorded metrics
    def test_scaling_decision_affects_subsequent_interval_not_current(self) -> None:
        """Verify discrete-time causality: decision at t changes servers at t+1 without altering t's metrics."""
        # Trace where minute 0 has 2500 viewers (causes ReactiveScaler to scale up from 2 to 4 servers)
        short_trace = pd.DataFrame(
            {
                "minute": [0, 1],
                "viewers": [2500, 2500],
                "scenario": ["Causality Test", "Causality Test"],
            }
        )

        metrics, decisions = run_strategy_simulation(
            scenario_name="Causality Test",
            df=short_trace,
            strategy="Reactive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
        )

        # Interval 0:
        # Load handled by initial 2 servers -> CPU = (2500 / 2000) * 100 = 125%
        self.assertEqual(metrics[0]["active_servers"], 2)
        self.assertEqual(metrics[0]["previous_server_count"], 2)
        self.assertEqual(metrics[0]["cpu_utilization"], 125.0)
        self.assertTrue(metrics[0]["overload_status"])
        self.assertEqual(metrics[0]["scaling_action"], "scale_up")
        self.assertEqual(metrics[0]["actual_server_count_after_decision"], 4)

        # Interval 1:
        # Load handled by 4 servers (decision took effect from next interval onward)
        # CPU = (2500 / 4000) * 100 = 62.5%
        self.assertEqual(metrics[1]["active_servers"], 4)
        self.assertEqual(metrics[1]["previous_server_count"], 4)
        self.assertEqual(metrics[1]["cpu_utilization"], 62.5)
        self.assertFalse(metrics[1]["overload_status"])

        # Verify interval 0's recorded metrics remain unmutated
        self.assertEqual(metrics[0]["active_servers"], 2)
        self.assertEqual(metrics[0]["cpu_utilization"], 125.0)

    # 7. Test summary metrics are calculated correctly using a small deterministic test dataset
    def test_summary_metrics_calculated_correctly_deterministic(self) -> None:
        """Verify exact calculation of all summary metrics on a handcrafted deterministic dataset."""
        synthetic_metrics = pd.DataFrame(
            {
                "scenario": ["TestScenario"] * 5,
                "strategy": ["Reactive"] * 5,
                "minute": [0, 1, 2, 3, 4],
                "actual_viewers": [1000, 2000, 3000, 1500, 1000],
                "active_servers": [2, 2, 4, 4, 3],
                "cpu_utilization": [50.0, 100.0, 75.0, 37.5, 33.33],
                "memory_utilization": [30.0, 50.0, 40.0, 25.0, 20.0],
                "overload_status": [False, True, False, False, False],
                "scaling_action": ["no_action", "scale_up", "no_action", "scale_down", "no_action"],
                "previous_server_count": [2, 2, 4, 4, 3],
                "desired_server_count": [2, 4, 4, 3, 3],
                "actual_server_count_after_decision": [2, 4, 4, 3, 3],
                "predicted_viewers": [None, None, None, None, None],
            }
        )

        summary = calculate_summary_metrics(synthetic_metrics)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]

        self.assertEqual(row["scenario"], "TestScenario")
        self.assertEqual(row["strategy"], "Reactive")
        self.assertEqual(row["total_overload_intervals"], 1)
        self.assertAlmostEqual(row["overload_percentage"], 20.0, places=2)  # 1/5 * 100
        self.assertAlmostEqual(row["average_cpu_utilization"], (50.0 + 100.0 + 75.0 + 37.5 + 33.33) / 5, places=2)
        self.assertEqual(row["maximum_cpu_utilization"], 100.0)
        self.assertAlmostEqual(row["average_active_servers"], (2 + 2 + 4 + 4 + 3) / 5, places=2)
        self.assertEqual(row["maximum_active_servers"], 4)
        self.assertEqual(row["total_server_minutes"], 15)  # 2 + 2 + 4 + 4 + 3
        self.assertEqual(row["scale_up_operations"], 1)
        self.assertEqual(row["scale_down_operations"], 1)
        self.assertEqual(row["total_scaling_operations"], 2)

    # 8. Test CSV outputs contain the expected columns
    def test_csv_outputs_contain_expected_columns(self) -> None:
        """Verify that metrics, decisions, and summary tables contain all required columns."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create minimal scenario dictionary
            scenarios = {"test_sc": self.sample_trace.iloc[:6].copy()}

            mock_pred = PredictiveScaler(predictor=lambda h: 2000)
            # Run comparison with mock scaler to avoid disk dependency
            metrics_df, decisions_df, summary_df = run_autoscaling_comparison(
                scenarios=scenarios,
                server_config=self.server_config,
                scaler_config=self.scaler_config,
                output_dir=tmp_path,
                generate_plots=False,
            )

            # Check expected columns for Detailed Metrics CSV
            expected_metrics_cols = [
                "scenario",
                "strategy",
                "minute",
                "actual_viewers",
                "active_servers",
                "cpu_utilization",
                "memory_utilization",
                "overload_status",
                "scaling_action",
                "previous_server_count",
                "desired_server_count",
                "actual_server_count_after_decision",
                "predicted_viewers",
            ]
            self.assertEqual(list(metrics_df.columns), expected_metrics_cols)

            # Check expected columns for Decisions CSV
            expected_decision_cols = [
                "scenario",
                "strategy",
                "minute",
                "action",
                "reason",
                "previous_server_count",
                "desired_server_count",
                "actual_server_count",
                "predicted_viewers",
            ]
            self.assertEqual(list(decisions_df.columns), expected_decision_cols)

            # Check expected columns for Summary CSV
            expected_summary_cols = [
                "scenario",
                "strategy",
                "total_overload_intervals",
                "overload_percentage",
                "average_cpu_utilization",
                "maximum_cpu_utilization",
                "average_active_servers",
                "maximum_active_servers",
                "total_server_minutes",
                "scale_up_operations",
                "scale_down_operations",
                "total_scaling_operations",
            ]
            self.assertEqual(list(summary_df.columns), expected_summary_cols)

            # Verify files were saved to disk
            self.assertTrue((tmp_path / "autoscaling_comparison_metrics.csv").is_file())
            self.assertTrue((tmp_path / "autoscaling_decisions.csv").is_file())
            self.assertTrue((tmp_path / "autoscaling_comparison_summary.csv").is_file())

    # 9. Test repeated runs with the same inputs produce consistent results
    def test_deterministic_reproducibility(self) -> None:
        """Verify that running the simulation twice with identical inputs yields strictly identical results."""
        m1, d1 = run_strategy_simulation(
            scenario_name="Reproducibility",
            df=self.sample_trace,
            strategy="Reactive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
        )
        m2, d2 = run_strategy_simulation(
            scenario_name="Reproducibility",
            df=self.sample_trace,
            strategy="Reactive",
            server_config=self.server_config,
            scaler_config=self.scaler_config,
        )

        df_m1 = pd.DataFrame(m1)
        df_m2 = pd.DataFrame(m2)
        pd.testing.assert_frame_equal(df_m1, df_m2)

        df_d1 = pd.DataFrame(d1)
        df_d2 = pd.DataFrame(d2)
        pd.testing.assert_frame_equal(df_d1, df_d2)

    # 10. Test visualization generation executes and creates all 4 PNG files
    def test_visualization_generation(self) -> None:
        """Verify that generate_comparison_visualizations produces all 4 expected PNG plots."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create synthetic metrics and summary DataFrames
            synthetic_metrics = pd.DataFrame(
                {
                    "scenario": ["Normal Traffic"] * 10 + ["Sudden Spike"] * 10,
                    "strategy": ["Reactive"] * 5 + ["Predictive"] * 5 + ["Reactive"] * 5 + ["Predictive"] * 5,
                    "minute": list(range(5)) * 4,
                    "actual_viewers": [1500] * 20,
                    "active_servers": [2] * 20,
                    "cpu_utilization": [75.0] * 20,
                    "memory_utilization": [40.0] * 20,
                    "overload_status": [False] * 20,
                    "scaling_action": ["no_action"] * 20,
                    "previous_server_count": [2] * 20,
                    "desired_server_count": [2] * 20,
                    "actual_server_count_after_decision": [2] * 20,
                    "predicted_viewers": [1500] * 20,
                }
            )
            summary_df = calculate_summary_metrics(synthetic_metrics)

            saved_plots = generate_comparison_visualizations(
                metrics_df=synthetic_metrics,
                summary_df=summary_df,
                output_dir=tmp_path,
            )

            expected_plots = [
                "overload_comparison.png",
                "average_cpu_comparison.png",
                "server_usage_comparison.png",
                "autoscaling_timeline_comparison.png",
            ]
            for plot_name in expected_plots:
                self.assertTrue((tmp_path / plot_name).is_file(), f"Missing plot: {plot_name}")
                self.assertGreater((tmp_path / plot_name).stat().st_size, 0)

    # 11. Test error handling for invalid strategy or malformed inputs
    def test_invalid_strategy_or_inputs_rejected(self) -> None:
        """Verify descriptive exceptions when invalid inputs or strategies are passed."""
        # Unsupported strategy
        with self.assertRaises(ValueError):
            run_strategy_simulation(
                scenario_name="Test",
                df=self.sample_trace,
                strategy="NonExistentStrategy",
            )

        # Missing required column
        bad_df = pd.DataFrame({"viewers": [1000, 2000]})
        with self.assertRaises(ValueError):
            run_strategy_simulation(
                scenario_name="Test",
                df=bad_df,
                strategy="Reactive",
            )

        # Missing files directory
        with self.assertRaises(FileNotFoundError):
            load_raw_scenarios(data_dir="/non/existent/path")


if __name__ == "__main__":
    unittest.main()
