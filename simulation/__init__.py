"""Simulation package for cloud autoscaler workload generation, modeling, and infrastructure."""

from simulation.cloud_simulator import (
    CloudSimulator,
    ServerConfig,
    SimulatedServer,
)
from simulation.metrics import (
    InfrastructureMetrics,
    ServerMetrics,
    calculate_aggregate_metrics,
)
from simulation.workload_generator import (
    generate_all_scenarios,
    generate_event_ending_traffic,
    generate_normal_traffic,
    generate_peak_event_traffic,
    generate_sudden_spike_traffic,
)

__all__ = [
    "generate_normal_traffic",
    "generate_sudden_spike_traffic",
    "generate_peak_event_traffic",
    "generate_event_ending_traffic",
    "generate_all_scenarios",
    "CloudSimulator",
    "SimulatedServer",
    "ServerConfig",
    "InfrastructureMetrics",
    "ServerMetrics",
    "calculate_aggregate_metrics",
]
