"""Simulation package for cloud autoscaler workload generation and modeling."""

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
]
