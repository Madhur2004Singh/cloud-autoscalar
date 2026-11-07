"""Infrastructure metrics calculation and aggregation module.

This module provides data structures and aggregation utilities for tracking
simulated cloud server performance, resource utilization, and overload conditions.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ServerMetrics:
    """Resource metrics for an individual simulated server."""

    minute: int
    server_id: str
    assigned_viewers: int
    cpu_utilization: float
    memory_utilization: float
    overloaded: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


@dataclass
class InfrastructureMetrics:
    """Aggregate infrastructure metrics across all active servers for a simulation step."""

    minute: int
    total_viewers: int
    active_servers: int
    total_capacity: int
    average_cpu_utilization: float
    maximum_cpu_utilization: float
    average_memory_utilization: float
    maximum_memory_utilization: float
    overloaded_servers: int
    overload_status: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregate metrics to dictionary."""
        return asdict(self)


def calculate_aggregate_metrics(
    minute: int,
    total_viewers: int,
    capacity_per_server: int,
    server_metrics_list: List[ServerMetrics],
) -> InfrastructureMetrics:
    """Calculate aggregate infrastructure metrics from a collection of server metrics.

    Formulas:
    - total_capacity = active_servers * capacity_per_server
    - average_cpu_utilization = mean(server cpu utilizations)
    - maximum_cpu_utilization = max(server cpu utilizations)
    - average_memory_utilization = mean(server memory utilizations)
    - maximum_memory_utilization = max(server memory utilizations)
    - overloaded_servers = count of servers where cpu > 100% or memory > 100%
    - overload_status = True if overloaded_servers > 0, else False

    Args:
        minute: Current simulation minute.
        total_viewers: Total incoming viewers in this interval.
        capacity_per_server: Configured viewer capacity per server.
        server_metrics_list: Metrics list from all active servers.

    Returns:
        Populated InfrastructureMetrics instance.
    """
    active_count = len(server_metrics_list)
    if active_count == 0:
        return InfrastructureMetrics(
            minute=minute,
            total_viewers=total_viewers,
            active_servers=0,
            total_capacity=0,
            average_cpu_utilization=0.0,
            maximum_cpu_utilization=0.0,
            average_memory_utilization=0.0,
            maximum_memory_utilization=0.0,
            overloaded_servers=0,
            overload_status=False,
        )

    total_capacity = active_count * capacity_per_server
    cpu_utils = [sm.cpu_utilization for sm in server_metrics_list]
    mem_utils = [sm.memory_utilization for sm in server_metrics_list]
    overloaded_count = sum(1 for sm in server_metrics_list if sm.overloaded)

    avg_cpu = sum(cpu_utils) / active_count
    max_cpu = max(cpu_utils)
    avg_mem = sum(mem_utils) / active_count
    max_mem = max(mem_utils)
    overload_status = overloaded_count > 0

    return InfrastructureMetrics(
        minute=minute,
        total_viewers=total_viewers,
        active_servers=active_count,
        total_capacity=total_capacity,
        average_cpu_utilization=avg_cpu,
        maximum_cpu_utilization=max_cpu,
        average_memory_utilization=avg_mem,
        maximum_memory_utilization=max_mem,
        overloaded_servers=overloaded_count,
        overload_status=overload_status,
    )
