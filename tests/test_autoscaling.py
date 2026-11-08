"""Unit tests for Stage D1: Base Autoscaler Interface and Reactive Autoscaling."""

import unittest
from autoscaling.base_scaler import BaseScaler, ScalerConfig, ScalingDecision
from autoscaling.reactive_scaler import ReactiveScaler
from simulation.cloud_simulator import CloudSimulator, ServerConfig


class DummyScaler(BaseScaler):
    """Concrete mock subclass of BaseScaler for interface and helper testing."""

    def make_decision(
        self,
        current_interval: int,
        current_viewers: int,
        current_active_servers: int,
        current_cpu_utilization: float,
        simulator=None,
    ) -> ScalingDecision:
        return ScalingDecision(
            action="no_action",
            previous_servers=current_active_servers,
            desired_servers=current_active_servers,
            actual_servers=current_active_servers,
            reason="dummy decision",
        )


class TestBaseScaler(unittest.TestCase):
    """Test suite covering BaseScaler interface, configuration, validation, and helpers."""

    def test_cannot_instantiate_abc(self) -> None:
        """Verify that BaseScaler cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseScaler()  # type: ignore

    def test_valid_configuration_accepted(self) -> None:
        """Verify valid default and custom configurations are accepted."""
        # Default configuration
        scaler = DummyScaler()
        self.assertEqual(scaler.min_servers, 2)
        self.assertEqual(scaler.max_servers, 30)
        self.assertEqual(scaler.target_cpu_utilization, 70.0)
        self.assertEqual(scaler.scale_down_cpu_threshold, 30.0)
        self.assertEqual(scaler.cooldown_intervals, 2)
        self.assertEqual(scaler.scale_down_consecutive_intervals, 3)
        self.assertEqual(scaler.capacity_per_server, 1000)

        # Custom configuration via ScalerConfig
        cfg = ScalerConfig(
            min_servers=3,
            max_servers=20,
            target_cpu_utilization=75.0,
            scale_down_cpu_threshold=35.0,
            cooldown_intervals=4,
            scale_down_consecutive_intervals=2,
            capacity_per_server=500,
        )
        custom_scaler = DummyScaler(config=cfg)
        self.assertEqual(custom_scaler.min_servers, 3)
        self.assertEqual(custom_scaler.max_servers, 20)
        self.assertEqual(custom_scaler.target_cpu_utilization, 75.0)
        self.assertEqual(custom_scaler.scale_down_cpu_threshold, 35.0)
        self.assertEqual(custom_scaler.cooldown_intervals, 4)
        self.assertEqual(custom_scaler.cooldown, 4)
        self.assertEqual(custom_scaler.scale_down_consecutive_intervals, 2)
        self.assertEqual(custom_scaler.capacity_per_server, 500)

    def test_invalid_min_max_servers_rejected(self) -> None:
        """Verify rejection of invalid min/max server combinations."""
        # min_servers > max_servers
        with self.assertRaises(ValueError):
            ScalerConfig(min_servers=10, max_servers=5)

        with self.assertRaises(ValueError):
            DummyScaler(min_servers=15, max_servers=10)

        # min_servers <= 0
        with self.assertRaises(ValueError):
            ScalerConfig(min_servers=0, max_servers=10)

        with self.assertRaises(ValueError):
            ScalerConfig(min_servers=-2, max_servers=10)

        # max_servers <= 0
        with self.assertRaises(ValueError):
            ScalerConfig(min_servers=1, max_servers=0)

    def test_invalid_cpu_thresholds_rejected(self) -> None:
        """Verify rejection of invalid CPU percentage thresholds."""
        # target_cpu <= 0 or > 100
        with self.assertRaises(ValueError):
            ScalerConfig(target_cpu_utilization=0.0)

        with self.assertRaises(ValueError):
            ScalerConfig(target_cpu_utilization=105.0)

        with self.assertRaises(ValueError):
            ScalerConfig(target_cpu_utilization=-10.0)

        # scale_down <= 0 or > 100
        with self.assertRaises(ValueError):
            ScalerConfig(scale_down_cpu_threshold=0.0)

        with self.assertRaises(ValueError):
            ScalerConfig(scale_down_cpu_threshold=110.0)

        # scale_down >= target_cpu
        with self.assertRaises(ValueError):
            ScalerConfig(target_cpu_utilization=60.0, scale_down_cpu_threshold=60.0)

        with self.assertRaises(ValueError):
            ScalerConfig(target_cpu_utilization=50.0, scale_down_cpu_threshold=70.0)

    def test_invalid_interval_settings_rejected(self) -> None:
        """Verify rejection of invalid cooldown, consecutive intervals, or capacity."""
        # cooldown < 0
        with self.assertRaises(ValueError):
            ScalerConfig(cooldown_intervals=-1)

        # scale_down_consecutive_intervals <= 0
        with self.assertRaises(ValueError):
            ScalerConfig(scale_down_consecutive_intervals=0)

        with self.assertRaises(ValueError):
            ScalerConfig(scale_down_consecutive_intervals=-2)

        # capacity_per_server <= 0
        with self.assertRaises(ValueError):
            ScalerConfig(capacity_per_server=0)

    def test_required_server_count_calculation(self) -> None:
        """Verify calculation of required server counts under target CPU utilization."""
        scaler = DummyScaler(
            capacity_per_server=1000,
            target_cpu_utilization=70.0,
            min_servers=2,
            max_servers=30,
        )
        # target capacity per server = 1000 * 0.70 = 700 viewers
        # 0 viewers -> 0 servers (unclamped) / 2 servers (clamped)
        self.assertEqual(scaler.calculate_required_servers(0, clamp=False), 0)
        self.assertEqual(scaler.calculate_required_servers(0, clamp=True), 2)

        # 700 viewers -> ceil(700 / 700) = 1 (unclamped) / 2 (clamped)
        self.assertEqual(scaler.calculate_required_servers(700, clamp=False), 1)
        self.assertEqual(scaler.calculate_required_servers(700, clamp=True), 2)

        # 701 viewers -> ceil(701 / 700) = 2
        self.assertEqual(scaler.calculate_required_servers(701, clamp=False), 2)
        self.assertEqual(scaler.calculate_required_servers(701, clamp=True), 2)

        # 2,000 viewers -> ceil(2000 / 700) = 3
        self.assertEqual(scaler.calculate_required_servers(2000, clamp=False), 3)

        # 21,000 viewers -> ceil(21000 / 700) = 30
        self.assertEqual(scaler.calculate_required_servers(21000, clamp=False), 30)

        # 50,000 viewers -> ceil(50000 / 700) = 72 (unclamped) / 30 (clamped)
        self.assertEqual(scaler.calculate_required_servers(50000, clamp=False), 72)
        self.assertEqual(scaler.calculate_required_servers(50000, clamp=True), 30)

        # Negative viewers should raise ValueError
        with self.assertRaises(ValueError):
            scaler.calculate_required_servers(-50)

    def test_server_count_clamping(self) -> None:
        """Verify clamp_server_count correctly enforces [min_servers, max_servers]."""
        scaler = DummyScaler(min_servers=2, max_servers=30)
        self.assertEqual(scaler.clamp_server_count(-5), 2)
        self.assertEqual(scaler.clamp_server_count(0), 2)
        self.assertEqual(scaler.clamp_server_count(1), 2)
        self.assertEqual(scaler.clamp_server_count(2), 2)
        self.assertEqual(scaler.clamp_server_count(15), 15)
        self.assertEqual(scaler.clamp_server_count(30), 30)
        self.assertEqual(scaler.clamp_server_count(31), 30)
        self.assertEqual(scaler.clamp_server_count(100), 30)

    def test_cooldown_behavior(self) -> None:
        """Verify cooldown expiration logic across simulation intervals."""
        scaler = DummyScaler(cooldown_intervals=2)

        # Initial state before any scaling: cooldown is expired
        self.assertTrue(scaler.is_cooldown_expired(0))
        self.assertTrue(scaler.is_cooldown_expired(10))

        # Action recorded at interval 10
        scaler.record_scaling_action("scale_up", interval=10)
        self.assertEqual(scaler.scale_up_count, 1)
        self.assertEqual(scaler.last_scaling_interval, 10)

        # Interval 10: cooldown active (elapsed = 0 < 2)
        self.assertFalse(scaler.is_cooldown_expired(10))

        # Interval 11: cooldown active (elapsed = 1 < 2)
        self.assertFalse(scaler.is_cooldown_expired(11))

        # Interval 12: cooldown expired (elapsed = 2 >= 2)
        self.assertTrue(scaler.is_cooldown_expired(12))

        # Interval 13: cooldown expired (elapsed = 3 >= 2)
        self.assertTrue(scaler.is_cooldown_expired(13))

        # Negative interval raises ValueError
        with self.assertRaises(ValueError):
            scaler.is_cooldown_expired(-1)

        # Reset clears tracking
        scaler.reset()
        self.assertIsNone(scaler.last_scaling_interval)
        self.assertEqual(scaler.scale_up_count, 0)
        self.assertTrue(scaler.is_cooldown_expired(10))

    def test_scaling_decision_dataclass(self) -> None:
        """Verify ScalingDecision fields, properties, and dictionary representation."""
        decision = ScalingDecision(
            action="scale_up",
            previous_servers=2,
            desired_servers=4,
            actual_servers=4,
            reason="High load",
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.previous_server_count, 2)
        self.assertEqual(decision.desired_server_count, 4)
        self.assertEqual(decision.actual_server_count, 4)
        self.assertTrue(decision.scaled)

        d_dict = decision.to_dict()
        self.assertEqual(d_dict["action"], "scale_up")
        self.assertEqual(d_dict["previous_servers"], 2)
        self.assertEqual(d_dict["desired_servers"], 4)
        self.assertEqual(d_dict["actual_servers"], 4)
        self.assertEqual(d_dict["reason"], "High load")

        # Invalid action raises ValueError
        with self.assertRaises(ValueError):
            ScalingDecision(
                action="invalid_action",
                previous_servers=2,
                desired_servers=3,
                actual_servers=3,
                reason="Invalid",
            )


class TestReactiveScaler(unittest.TestCase):
    """Test suite covering ReactiveScaler policy, thresholds, hysteresis, and simulation integration."""

    def setUp(self) -> None:
        """Initialize server config, simulator, and reactive scaler."""
        self.server_config = ServerConfig(
            capacity_per_server=1000,
            min_servers=2,
            max_servers=30,
            initial_servers=2,
        )
        self.simulator = CloudSimulator(config=self.server_config)
        self.scaler = ReactiveScaler(
            min_servers=2,
            max_servers=30,
            target_cpu_utilization=70.0,
            scale_down_cpu_threshold=30.0,
            cooldown_intervals=2,
            scale_down_consecutive_intervals=3,
            capacity_per_server=1000,
        )

    # Test: Scales up when CPU exceeds 70%.
    def test_scales_up_when_cpu_exceeds_target(self) -> None:
        """Verify scale-up is triggered when CPU utilization > 70%."""
        # 2 servers with 2,000 viewers -> 100% CPU > 70%
        # target capacity = 700 per server -> ceil(2000 / 700) = 3 servers desired
        decision = self.scaler.make_decision(
            current_interval=0,
            current_viewers=2000,
            current_active_servers=2,
            current_cpu_utilization=100.0,
            simulator=self.simulator,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.previous_servers, 2)
        self.assertEqual(decision.desired_servers, 3)
        self.assertEqual(decision.actual_servers, 3)
        self.assertEqual(len(self.simulator.get_active_servers()), 3)
        self.assertEqual(self.scaler.scale_up_count, 1)
        self.assertEqual(self.scaler.last_scaling_interval, 0)

    # Test: Does not scale up when CPU is exactly 70%.
    def test_does_not_scale_up_when_cpu_is_exactly_target(self) -> None:
        """Verify no scale-up occurs when CPU utilization is exactly 70.0%."""
        decision = self.scaler.make_decision(
            current_interval=0,
            current_viewers=1400,
            current_active_servers=2,
            current_cpu_utilization=70.0,
            simulator=self.simulator,
        )
        self.assertEqual(decision.action, "no_action")
        self.assertEqual(decision.actual_servers, 2)
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        self.assertEqual(self.scaler.scale_up_count, 0)

    # Test: Does not scale down before 3 consecutive low-CPU intervals.
    def test_does_not_scale_down_before_three_consecutive_intervals(self) -> None:
        """Verify scaler does not scale down on 1st or 2nd low-CPU interval."""
        # Start with 3 servers
        self.simulator.add_server()
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

        # Interval 0: 1st low CPU interval (e.g. 25% CPU < 30%)
        d0 = self.scaler.make_decision(
            current_interval=0,
            current_viewers=750,
            current_active_servers=3,
            current_cpu_utilization=25.0,
            simulator=self.simulator,
        )
        self.assertEqual(d0.action, "no_action")
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 1)
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

        # Interval 1: 2nd low CPU interval
        d1 = self.scaler.make_decision(
            current_interval=1,
            current_viewers=750,
            current_active_servers=3,
            current_cpu_utilization=25.0,
            simulator=self.simulator,
        )
        self.assertEqual(d1.action, "no_action")
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 2)
        self.assertEqual(len(self.simulator.get_active_servers()), 3)
        self.assertEqual(self.scaler.scale_down_count, 0)

    # Test: Scales down after 3 consecutive intervals below 30%.
    def test_scales_down_after_three_consecutive_intervals(self) -> None:
        """Verify scale-down is triggered on the 3rd consecutive low-CPU interval."""
        # Start with 3 servers
        self.simulator.add_server()
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

        # Intervals 0 and 1: low CPU
        self.scaler.make_decision(0, 750, 3, 25.0, simulator=self.simulator)
        self.scaler.make_decision(1, 750, 3, 25.0, simulator=self.simulator)

        # Interval 2: 3rd low CPU interval -> scale down by 1 server
        d2 = self.scaler.make_decision(2, 750, 3, 25.0, simulator=self.simulator)
        self.assertEqual(d2.action, "scale_down")
        self.assertEqual(d2.previous_servers, 3)
        self.assertEqual(d2.desired_servers, 2)
        self.assertEqual(d2.actual_servers, 2)
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        self.assertEqual(self.scaler.scale_down_count, 1)
        self.assertEqual(self.scaler.last_scaling_interval, 2)
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 0)

    # Test: Never scales below 2 servers.
    def test_never_scales_below_min_servers(self) -> None:
        """Verify scaler never reduces server pool below min_servers (2)."""
        self.assertEqual(len(self.simulator.get_active_servers()), 2)

        # Provide 5 consecutive low-CPU intervals
        for minute in range(5):
            decision = self.scaler.make_decision(
                current_interval=minute,
                current_viewers=200,
                current_active_servers=2,
                current_cpu_utilization=10.0,
                simulator=self.simulator,
            )
            self.assertEqual(decision.action, "no_action")
            self.assertEqual(decision.actual_servers, 2)
            self.assertEqual(len(self.simulator.get_active_servers()), 2)

        self.assertEqual(self.scaler.scale_down_count, 0)

    # Test: Does not exceed 30 servers.
    def test_does_not_exceed_max_servers(self) -> None:
        """Verify scaler never scales beyond max_servers (30)."""
        # Set simulator active servers to 29
        for _ in range(27):
            self.simulator.add_server()
        self.assertEqual(len(self.simulator.get_active_servers()), 29)

        # Massive viewership that would demand 50+ servers
        decision = self.scaler.make_decision(
            current_interval=0,
            current_viewers=40000,
            current_active_servers=29,
            current_cpu_utilization=137.9,
            simulator=self.simulator,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.desired_servers, 30)
        self.assertEqual(decision.actual_servers, 30)
        self.assertEqual(len(self.simulator.get_active_servers()), 30)

        # Another massive interval while at max_servers -> no_action
        self.scaler.last_scaling_interval = None  # bypass cooldown to test boundary check
        d2 = self.scaler.make_decision(
            current_interval=5,
            current_viewers=50000,
            current_active_servers=30,
            current_cpu_utilization=166.7,
            simulator=self.simulator,
        )
        self.assertEqual(d2.action, "no_action")
        self.assertEqual(d2.actual_servers, 30)
        self.assertEqual(len(self.simulator.get_active_servers()), 30)

    # Test: Resets the low-CPU counter when CPU rises above 30%.
    def test_resets_low_cpu_counter_when_cpu_rises(self) -> None:
        """Verify that consecutive low-CPU counter resets to 0 when CPU rises >= 30%."""
        self.simulator.add_server()  # 3 servers
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

        # 2 low-CPU intervals
        self.scaler.make_decision(0, 600, 3, 20.0, simulator=self.simulator)
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 1)
        self.scaler.make_decision(1, 600, 3, 20.0, simulator=self.simulator)
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 2)

        # CPU rises to 45% (within normal range [30%, 70%])
        d_reset = self.scaler.make_decision(2, 1350, 3, 45.0, simulator=self.simulator)
        self.assertEqual(d_reset.action, "no_action")
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 0)

        # Next interval is low CPU again: counter should restart at 1, not 3
        d_next = self.scaler.make_decision(3, 600, 3, 20.0, simulator=self.simulator)
        self.assertEqual(d_next.action, "no_action")
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 1)
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

    # Test: Respects cooldown for both scaling directions.
    def test_respects_cooldown_for_both_scaling_directions(self) -> None:
        """Verify scaling actions are prevented during the cooldown period."""
        # 1. Scale Up cooldown test
        # Scale up at interval 0 from 2 to 3 servers
        d0 = self.scaler.make_decision(0, 2000, 2, 100.0, simulator=self.simulator)
        self.assertEqual(d0.action, "scale_up")
        self.assertEqual(d0.actual_servers, 3)

        # Interval 1: High CPU (90%) but in cooldown (elapsed = 1 < 2)
        d1 = self.scaler.make_decision(1, 2700, 3, 90.0, simulator=self.simulator)
        self.assertEqual(d1.action, "no_action")
        self.assertEqual(d1.actual_servers, 3)
        self.assertIn("cooldown", d1.reason)

        # Interval 2: Cooldown expired (elapsed = 2 >= 2) -> can scale up
        d2 = self.scaler.make_decision(2, 2800, 3, 93.3, simulator=self.simulator)
        self.assertEqual(d2.action, "scale_up")
        self.assertEqual(d2.actual_servers, 4)

        # 2. Scale Down cooldown test
        # Scaled up at interval 2. Force consecutive low CPU
        self.scaler.consecutive_low_cpu_count = 2

        # Interval 3: 3rd low CPU condition reached, but interval 3 - 2 = 1 < 2 -> in cooldown!
        d3 = self.scaler.make_decision(3, 500, 4, 12.5, simulator=self.simulator)
        self.assertEqual(d3.action, "no_action")
        self.assertEqual(d3.actual_servers, 4)
        self.assertIn("cooldown", d3.reason)

        # Interval 4: Cooldown expired (elapsed = 4 - 2 = 2 >= 2) -> can scale down
        d4 = self.scaler.make_decision(4, 500, 4, 12.5, simulator=self.simulator)
        self.assertEqual(d4.action, "scale_down")
        self.assertEqual(d4.actual_servers, 3)

    # Test: Does not scale when CPU is between thresholds.
    def test_does_not_scale_when_cpu_between_thresholds(self) -> None:
        """Verify no action when CPU is in the neutral band [30%, 70%]."""
        test_cpus = [30.0, 30.1, 50.0, 65.0, 70.0]
        for idx, cpu_val in enumerate(test_cpus):
            with self.subTest(cpu=cpu_val):
                decision = self.scaler.make_decision(
                    current_interval=idx,
                    current_viewers=int(cpu_val * 20),
                    current_active_servers=2,
                    current_cpu_utilization=cpu_val,
                    simulator=self.simulator,
                )
                self.assertEqual(decision.action, "no_action")
                self.assertEqual(decision.actual_servers, 2)
                self.assertEqual(self.scaler.consecutive_low_cpu_count, 0)

    # Test: Returned decision result correctly reports action and server counts.
    def test_decision_result_structure_and_reporting(self) -> None:
        """Verify decision result returns correct structured fields."""
        decision = self.scaler.make_decision(
            current_interval=0,
            current_viewers=2100,
            current_active_servers=2,
            current_cpu_utilization=105.0,
            simulator=self.simulator,
        )
        self.assertIsInstance(decision, ScalingDecision)
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.previous_servers, 2)
        self.assertEqual(decision.desired_servers, 3)
        self.assertEqual(decision.actual_servers, 3)
        self.assertIn("Scaled up from 2 to 3 servers", decision.reason)

    # Test: Reset clears all operational state.
    def test_scaler_reset_clears_all_counters(self) -> None:
        """Verify reset() restores scaler to pristine initial state."""
        self.scaler.make_decision(0, 2100, 2, 105.0, simulator=self.simulator)
        self.assertEqual(self.scaler.scale_up_count, 1)
        self.assertEqual(self.scaler.last_scaling_interval, 0)

        # Trigger partial low CPU accumulation
        self.scaler.make_decision(3, 400, 3, 13.3, simulator=self.simulator)
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 1)

        self.scaler.reset()
        self.assertEqual(self.scaler.scale_up_count, 0)
        self.assertEqual(self.scaler.scale_down_count, 0)
        self.assertIsNone(self.scaler.last_scaling_interval)
        self.assertEqual(self.scaler.consecutive_low_cpu_count, 0)

    # Test: Operates without simulator instance (pure calculation).
    def test_scale_up_without_simulator(self) -> None:
        """Verify scale-up with simulator=None returns recommendation without modifying infrastructure."""
        initial_scale_up_count = self.scaler.scale_up_count
        initial_last_interval = self.scaler.last_scaling_interval

        decision = self.scaler.make_decision(
            current_interval=5,
            current_viewers=2800,
            current_active_servers=2,
            current_cpu_utilization=140.0,
            simulator=None,
        )
        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.previous_servers, 2)
        self.assertGreater(decision.desired_servers, decision.previous_servers)
        self.assertEqual(decision.desired_servers, 4)
        self.assertEqual(decision.actual_servers, 2)  # Unchanged
        self.assertEqual(self.scaler.scale_up_count, initial_scale_up_count)  # Unchanged
        self.assertEqual(self.scaler.last_scaling_interval, initial_last_interval)  # Unchanged
        self.assertIn("no simulator provided", decision.reason)
        self.assertIn("unchanged", decision.reason)

    def test_scale_down_without_simulator(self) -> None:
        """Verify scale-down with simulator=None returns recommendation without modifying infrastructure."""
        # Accumulate 2 low-CPU intervals
        self.scaler.make_decision(0, 500, 4, 12.5, simulator=None)
        self.scaler.make_decision(1, 500, 4, 12.5, simulator=None)

        initial_scale_down_count = self.scaler.scale_down_count
        initial_last_interval = self.scaler.last_scaling_interval

        # 3rd interval triggers scale_down decision
        decision = self.scaler.make_decision(
            current_interval=2,
            current_viewers=500,
            current_active_servers=4,
            current_cpu_utilization=12.5,
            simulator=None,
        )
        self.assertEqual(decision.action, "scale_down")
        self.assertEqual(decision.previous_servers, 4)
        self.assertLess(decision.desired_servers, decision.previous_servers)
        self.assertEqual(decision.desired_servers, 3)
        self.assertEqual(decision.actual_servers, 4)  # Unchanged
        self.assertEqual(self.scaler.scale_down_count, initial_scale_down_count)  # Unchanged
        self.assertEqual(self.scaler.last_scaling_interval, initial_last_interval)  # Unchanged
        self.assertIn("no simulator provided", decision.reason)
        self.assertIn("unchanged", decision.reason)

    def test_scale_down_with_simulator_server_identification(self) -> None:
        """Verify scale-down with simulator removes exactly 1 server, marks it inactive, and reports its ID."""
        self.simulator.add_server()  # add server-3 -> active servers: server-1, server-2, server-3
        active_ids_before = [s.server_id for s in self.simulator.get_active_servers()]
        self.assertEqual(len(active_ids_before), 3)
        self.assertIn("server-3", active_ids_before)

        # Accumulate 3 low-CPU intervals
        self.scaler.make_decision(0, 500, 3, 16.7, simulator=self.simulator)
        self.scaler.make_decision(1, 500, 3, 16.7, simulator=self.simulator)
        decision = self.scaler.make_decision(2, 500, 3, 16.7, simulator=self.simulator)

        self.assertEqual(decision.action, "scale_down")
        self.assertEqual(decision.previous_servers, 3)
        self.assertEqual(decision.desired_servers, 2)
        self.assertEqual(decision.actual_servers, 2)
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        # Removed server ID is identified in reason
        self.assertIn("removed server-3", decision.reason)
        # Verify server-3 is no longer in active servers
        active_ids_after = [s.server_id for s in self.simulator.get_active_servers()]
        self.assertNotIn("server-3", active_ids_after)
        self.assertEqual(active_ids_after, ["server-1", "server-2"])

    # Test: Full simulation integration loop.
    def test_simulation_integration_loop(self) -> None:
        """Verify seamless execution over a sequence of intervals with CloudSimulator."""
        # Workload sequence:
        # intervals 0-2: high workload (2,500 viewers) -> scales up
        # intervals 3-7: very low workload (200 viewers) -> cools down and scales down
        workload = [2500, 2500, 2500, 200, 200, 200, 200, 200]

        for minute, viewers in enumerate(workload):
            infra_m, _ = self.simulator.step(viewers=viewers, minute=minute)
            decision = self.scaler.make_decision(
                current_interval=minute,
                current_viewers=viewers,
                current_active_servers=infra_m.active_servers,
                current_cpu_utilization=infra_m.average_cpu_utilization,
                simulator=self.simulator,
            )
            self.assertIn(decision.action, {"scale_up", "scale_down", "no_action"})
            self.assertEqual(decision.actual_servers, len(self.simulator.get_active_servers()))


if __name__ == "__main__":
    unittest.main()
