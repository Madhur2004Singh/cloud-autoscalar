"""Unit tests for Stage C: Simulated Cloud Infrastructure."""

import unittest
import numpy as np
import pandas as pd

from models.predictor import predict_next_viewers
from simulation.cloud_simulator import CloudSimulator, ServerConfig, SimulatedServer
from simulation.metrics import InfrastructureMetrics, ServerMetrics, calculate_aggregate_metrics
from simulation.workload_generator import generate_all_scenarios


class TestCloudSimulator(unittest.TestCase):
    """Test suite covering Stage C simulated cloud infrastructure requirements."""

    def setUp(self) -> None:
        """Initialize default server configuration and simulator instance."""
        self.config = ServerConfig(
            capacity_per_server=1000,
            min_servers=2,
            max_servers=30,
            initial_servers=2,
            total_memory_gb=8.0,
            baseline_memory_gb=1.0,
            memory_per_viewer_mb=4.0,
        )
        self.simulator = CloudSimulator(config=self.config)

    # Test 1: Correct initial server count.
    def test_initial_server_count(self) -> None:
        """Verify simulator starts with configured initial server count."""
        active_servers = self.simulator.get_active_servers()
        self.assertEqual(len(active_servers), self.config.initial_servers)
        self.assertEqual(len(active_servers), 2)

    # Test 2: Unique server IDs.
    def test_unique_server_ids(self) -> None:
        """Verify all server IDs in the active pool are unique."""
        for _ in range(5):
            self.simulator.add_server()
        active_servers = self.simulator.get_active_servers()
        server_ids = [s.server_id for s in active_servers]
        self.assertEqual(len(server_ids), len(set(server_ids)))

    # Test 3: Balanced viewer distribution.
    def test_balanced_viewer_distribution(self) -> None:
        """Verify viewers are distributed as evenly as possible across active servers."""
        # 3 servers and 2,000 viewers -> allocations: 667, 667, 666
        self.simulator.add_server()  # now 3 servers
        self.simulator.distribute_load(2000)
        allocations = [s.assigned_viewers for s in self.simulator.get_active_servers()]
        self.assertEqual(len(allocations), 3)
        self.assertLessEqual(max(allocations) - min(allocations), 1)
        self.assertIn(667, allocations)
        self.assertIn(666, allocations)

        # 2 servers and 1,500 viewers -> allocations: 750, 750
        sim2 = CloudSimulator(config=self.config)
        sim2.distribute_load(1500)
        alloc2 = [s.assigned_viewers for s in sim2.get_active_servers()]
        self.assertEqual(alloc2, [750, 750])

    # Test 4: Correct total assigned viewer count.
    def test_correct_total_assigned_viewers(self) -> None:
        """Verify sum of assigned viewers strictly equals incoming viewer count."""
        test_counts = [0, 1, 7, 100, 1500, 2000, 9999, 55329]
        for count in test_counts:
            with self.subTest(viewers=count):
                self.simulator.distribute_load(count)
                assigned_total = sum(
                    s.assigned_viewers for s in self.simulator.get_active_servers()
                )
                self.assertEqual(assigned_total, count)

    # Test 5: CPU utilisation calculation.
    def test_cpu_utilization_calculation(self) -> None:
        """Verify CPU utilization follows formula: (assigned / capacity) * 100."""
        server = SimulatedServer(server_id="s1", config=self.config)

        server.assign_workload(250)
        self.assertEqual(server.calculate_cpu_utilization(), 25.0)

        server.assign_workload(700)
        self.assertEqual(server.calculate_cpu_utilization(), 70.0)

        server.assign_workload(1000)
        self.assertEqual(server.calculate_cpu_utilization(), 100.0)

    # Test 6: CPU utilisation exceeding 100% when overloaded.
    def test_cpu_utilization_exceeding_100(self) -> None:
        """Verify CPU utilization is not capped at 100% when overloaded."""
        server = SimulatedServer(server_id="s1", config=self.config)
        server.assign_workload(1200)
        self.assertEqual(server.calculate_cpu_utilization(), 120.0)
        self.assertTrue(server.is_overloaded())

    # Test 7: Memory utilisation calculation.
    def test_memory_utilization_calculation(self) -> None:
        """Verify memory calculation: (baseline + viewers * mb_per_viewer / 1024) / total_gb * 100."""
        server = SimulatedServer(server_id="s1", config=self.config)
        # 1000 viewers: memory used = 1.0 + (1000 * 4.0 / 1024) = 4.90625 GB
        # memory utilization = (4.90625 / 8.0) * 100 = 61.328125%
        server.assign_workload(1000)
        expected_mem_util = (1.0 + (1000 * 4.0 / 1024.0)) / 8.0 * 100.0
        self.assertAlmostEqual(server.calculate_memory_utilization(), expected_mem_util)

    # Test 8: Memory utilisation exceeding 100% when overloaded.
    def test_memory_utilization_exceeding_100(self) -> None:
        """Verify memory utilization exceeds 100% when memory usage exceeds total memory."""
        server = SimulatedServer(server_id="s1", config=self.config)
        # 2000 viewers: memory used = 1.0 + (2000 * 4 / 1024) = 8.8125 GB (> 8 GB)
        server.assign_workload(2000)
        self.assertGreater(server.calculate_memory_utilization(), 100.0)
        self.assertTrue(server.is_overloaded())

    def test_independent_memory_overload(self) -> None:
        """Verify that memory alone can trigger overload when CPU remains below 100%."""
        # Custom config: large capacity (2,000 viewers) but constrained RAM (4 GB)
        custom_config = ServerConfig(
            capacity_per_server=2000,
            total_memory_gb=4.0,
            baseline_memory_gb=1.0,
            memory_per_viewer_mb=8.0,
        )
        server = SimulatedServer(server_id="mem-test-server", config=custom_config)

        # 500 viewers:
        # CPU util: (500 / 2000) * 100 = 25.0% (< 100%)
        # Memory used: 1.0 + (500 * 8.0 / 1024) = 4.90625 GB (> 4.0 GB)
        # Memory util: (4.90625 / 4.0) * 100 = 122.66% (> 100%)
        server.assign_workload(500)

        cpu_util = server.calculate_cpu_utilization()
        mem_util = server.calculate_memory_utilization()

        self.assertLess(cpu_util, 100.0, f"CPU expected under 100%, got {cpu_util}%")
        self.assertGreater(mem_util, 100.0, f"Memory expected over 100%, got {mem_util}%")
        self.assertTrue(
            server.is_overloaded(),
            "Server must be overloaded when memory exceeds 100% even if CPU is under 100%",
        )

    # Test 9: Correct server addition.
    def test_server_addition(self) -> None:
        """Verify add_server increments active server count and returns valid server."""
        initial_count = len(self.simulator.get_active_servers())
        new_server = self.simulator.add_server()
        self.assertIsInstance(new_server, SimulatedServer)
        self.assertEqual(len(self.simulator.get_active_servers()), initial_count + 1)
        self.assertTrue(new_server.active)

    # Test 10: Maximum server limit enforcement.
    def test_maximum_server_limit_enforcement(self) -> None:
        """Verify add_server raises ValueError when maximum limit is reached."""
        sim = CloudSimulator(
            config=ServerConfig(min_servers=2, max_servers=4, initial_servers=2)
        )
        sim.add_server()  # 3
        sim.add_server()  # 4 (max)
        self.assertEqual(len(sim.get_active_servers()), 4)

        with self.assertRaises(ValueError) as ctx:
            sim.add_server()  # attempt 5th
        self.assertIn("maximum server limit", str(ctx.exception).lower())

    # Test 11: Correct server removal.
    def test_server_removal(self) -> None:
        """Verify remove_server decrements active count and marks server inactive."""
        self.simulator.add_server()  # 3 servers
        self.assertEqual(len(self.simulator.get_active_servers()), 3)

        removed = self.simulator.remove_server()
        self.assertIsInstance(removed, SimulatedServer)
        self.assertFalse(removed.active)
        self.assertEqual(len(self.simulator.get_active_servers()), 2)

    def test_targeted_server_removal(self) -> None:
        """Verify removing a specific server ID removes exactly that server and marks it inactive."""
        # Setup: start with 2 initial servers (server-1, server-2), add server-3 and server-4
        s3 = self.simulator.add_server()
        s4 = self.simulator.add_server()
        initial_active_ids = [s.server_id for s in self.simulator.get_active_servers()]
        self.assertEqual(initial_active_ids, ["server-1", "server-2", "server-3", "server-4"])

        # Target and remove server-2 specifically
        removed = self.simulator.remove_server(server_id="server-2")

        # 1. Returned server is exactly server-2
        self.assertEqual(removed.server_id, "server-2")
        # 2. Returned server is marked inactive
        self.assertFalse(removed.active)

        # 3. Other active servers remain unchanged and active
        surviving_servers = self.simulator.get_active_servers()
        surviving_ids = [s.server_id for s in surviving_servers]
        self.assertEqual(surviving_ids, ["server-1", "server-3", "server-4"])
        for s in surviving_servers:
            self.assertTrue(s.active)
            self.assertNotEqual(s.server_id, "server-2")

    def test_targeted_server_removal_nonexistent_id(self) -> None:
        """Verify attempting to remove a nonexistent server ID raises ValueError."""
        self.simulator.add_server()  # 3 servers
        with self.assertRaises(ValueError) as ctx:
            self.simulator.remove_server(server_id="server-nonexistent-999")
        self.assertIn("not found in active pool", str(ctx.exception))

    # Test 12: Minimum server limit enforcement.
    def test_minimum_server_limit_enforcement(self) -> None:
        """Verify remove_server raises ValueError when minimum limit is reached."""
        self.assertEqual(len(self.simulator.get_active_servers()), 2)
        with self.assertRaises(ValueError) as ctx:
            self.simulator.remove_server()
        self.assertIn("minimum server limit", str(ctx.exception).lower())

    # Test 13: Workload redistribution after server removal.
    def test_workload_redistribution_after_server_removal(self) -> None:
        """Verify workload is redistributed across remaining servers after removal."""
        self.simulator.add_server()  # 3 servers
        self.simulator.step(viewers=3000, minute=1)
        for s in self.simulator.get_active_servers():
            self.assertEqual(s.assigned_viewers, 1000)

        # Remove one server and simulate next interval
        self.simulator.remove_server()
        self.simulator.step(viewers=3000, minute=2)
        active = self.simulator.get_active_servers()
        self.assertEqual(len(active), 2)
        for s in active:
            self.assertEqual(s.assigned_viewers, 1500)

    # Test 14: Zero-viewer workload.
    def test_zero_viewer_workload(self) -> None:
        """Verify 0 incoming viewers assigns 0 to all servers and yields 0% CPU."""
        infra_m, server_ms = self.simulator.step(viewers=0, minute=0)
        self.assertEqual(infra_m.total_viewers, 0)
        self.assertEqual(infra_m.average_cpu_utilization, 0.0)
        self.assertFalse(infra_m.overload_status)
        for sm in server_ms:
            self.assertEqual(sm.assigned_viewers, 0)
            self.assertEqual(sm.cpu_utilization, 0.0)

    # Test 15: Invalid viewer count handling.
    def test_invalid_viewer_count_handling(self) -> None:
        """Verify negative or non-integer viewer counts raise ValueError."""
        with self.assertRaises(ValueError):
            self.simulator.step(viewers=-50, minute=1)

        with self.assertRaises(ValueError):
            self.simulator.step(viewers="many", minute=1)  # type: ignore

    # Test 16: Correct aggregate infrastructure metrics.
    def test_aggregate_infrastructure_metrics(self) -> None:
        """Verify calculate_aggregate_metrics produces mathematically correct values."""
        # 2 servers with 500 and 1500 viewers
        sm1 = ServerMetrics(
            minute=5, server_id="s1", assigned_viewers=500,
            cpu_utilization=50.0, memory_utilization=36.0, overloaded=False
        )
        sm2 = ServerMetrics(
            minute=5, server_id="s2", assigned_viewers=1500,
            cpu_utilization=150.0, memory_utilization=86.0, overloaded=True
        )
        agg = calculate_aggregate_metrics(
            minute=5, total_viewers=2000, capacity_per_server=1000,
            server_metrics_list=[sm1, sm2]
        )
        self.assertEqual(agg.active_servers, 2)
        self.assertEqual(agg.total_capacity, 2000)
        self.assertEqual(agg.average_cpu_utilization, 100.0)
        self.assertEqual(agg.maximum_cpu_utilization, 150.0)
        self.assertEqual(agg.average_memory_utilization, 61.0)
        self.assertEqual(agg.maximum_memory_utilization, 86.0)
        self.assertEqual(agg.overloaded_servers, 1)
        self.assertTrue(agg.overload_status)

    # Test 17: Correct overload detection.
    def test_overload_detection(self) -> None:
        """Verify overload detection triggers when either CPU or Memory exceeds 100%."""
        # 2 servers, capacity = 2,000 total. 2,500 viewers -> both servers have 1,250 viewers (125% CPU)
        infra_m, _ = self.simulator.step(viewers=2500, minute=10)
        self.assertTrue(infra_m.overload_status)
        self.assertEqual(infra_m.overloaded_servers, 2)

        # 1,600 viewers -> 800 per server (80% CPU) -> normal
        infra_normal, _ = self.simulator.step(viewers=1600, minute=11)
        self.assertFalse(infra_normal.overload_status)
        self.assertEqual(infra_normal.overloaded_servers, 0)

    # Test 18: Multiple simulation intervals reset previous workload correctly.
    def test_multiple_simulation_intervals_reset(self) -> None:
        """Verify subsequent intervals reset previous workloads without accumulation."""
        self.simulator.step(viewers=5000, minute=1)
        infra_m2, server_ms2 = self.simulator.step(viewers=1000, minute=2)

        self.assertEqual(infra_m2.total_viewers, 1000)
        total_assigned = sum(sm.assigned_viewers for sm in server_ms2)
        self.assertEqual(total_assigned, 1000)
        self.assertEqual(infra_m2.average_cpu_utilization, 50.0)

    # Test 19: Deterministic workload distribution.
    def test_deterministic_workload_distribution(self) -> None:
        """Verify identical inputs produce identical distribution across runs."""
        sim_a = CloudSimulator(config=self.config)
        sim_b = CloudSimulator(config=self.config)

        m_a, s_a = sim_a.step(viewers=1753, minute=5)
        m_b, s_b = sim_b.step(viewers=1753, minute=5)

        self.assertEqual(
            [s.assigned_viewers for s in s_a],
            [s.assigned_viewers for s in s_b],
        )
        self.assertEqual(m_a.average_cpu_utilization, m_b.average_cpu_utilization)

    # Test 20: No modification of Stage A or Stage B behavior.
    def test_stage_a_and_stage_b_integrity(self) -> None:
        """Verify Stage A generator and Stage B predictor still work without regression."""
        scenarios = generate_all_scenarios(duration_minutes=120, random_seed=42)
        self.assertEqual(len(scenarios), 4)
        for df in scenarios.values():
            self.assertEqual(len(df), 120)

        # Stage B predictor API
        history = scenarios["normal"].iloc[:7].copy()
        pred = predict_next_viewers(history)
        self.assertIsInstance(pred, (int, np.integer))
        self.assertGreaterEqual(pred, 0)


if __name__ == "__main__":
    unittest.main()
