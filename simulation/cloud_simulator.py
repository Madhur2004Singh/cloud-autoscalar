"""Simulated Cloud Infrastructure Module for Live Event Broadcasting.

This module models cloud server infrastructure that receives concurrent viewer traffic,
distributes workloads deterministically across active servers, models CPU and memory
utilization, detects resource overloads, and supports explicit server lifecycle management.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from simulation.metrics import (
    InfrastructureMetrics,
    ServerMetrics,
    calculate_aggregate_metrics,
)


@dataclass
class ServerConfig:
    """Configuration parameters for cloud server infrastructure simulation."""

    capacity_per_server: int = 1000
    min_servers: int = 2
    max_servers: int = 30
    initial_servers: int = 2
    total_memory_gb: float = 8.0
    baseline_memory_gb: float = 1.0
    memory_per_viewer_mb: float = 4.0

    def __post_init__(self) -> None:
        """Validate configuration settings."""
        if self.capacity_per_server <= 0:
            raise ValueError(
                f"capacity_per_server must be positive, got {self.capacity_per_server}"
            )
        if self.min_servers <= 0:
            raise ValueError(f"min_servers must be positive, got {self.min_servers}")
        if self.max_servers < self.min_servers:
            raise ValueError(
                f"max_servers ({self.max_servers}) cannot be less than min_servers ({self.min_servers})"
            )
        if not (self.min_servers <= self.initial_servers <= self.max_servers):
            raise ValueError(
                f"initial_servers ({self.initial_servers}) must be between "
                f"min_servers ({self.min_servers}) and max_servers ({self.max_servers})"
            )
        if self.total_memory_gb <= 0:
            raise ValueError(
                f"total_memory_gb must be positive, got {self.total_memory_gb}"
            )
        if self.baseline_memory_gb < 0:
            raise ValueError(
                f"baseline_memory_gb must be non-negative, got {self.baseline_memory_gb}"
            )
        if self.baseline_memory_gb >= self.total_memory_gb:
            raise ValueError(
                f"baseline_memory_gb ({self.baseline_memory_gb}) cannot exceed or equal "
                f"total_memory_gb ({self.total_memory_gb})"
            )
        if self.memory_per_viewer_mb <= 0:
            raise ValueError(
                f"memory_per_viewer_mb must be positive, got {self.memory_per_viewer_mb}"
            )


class SimulatedServer:
    """Represents an individual simulated virtual server node."""

    def __init__(self, server_id: str, config: ServerConfig) -> None:
        """Initialize a simulated server instance.

        Args:
            server_id: Unique and stable identifier for the server.
            config: Server infrastructure configuration parameters.
        """
        self.server_id: str = server_id
        self.config: ServerConfig = config
        self.active: bool = True
        self.assigned_viewers: int = 0

    def assign_workload(self, viewers: int) -> None:
        """Assign viewer workload to this server for the current interval.

        Args:
            viewers: Non-negative integer count of assigned viewers.

        Raises:
            ValueError: If viewers count is negative.
        """
        if viewers < 0:
            raise ValueError(
                f"Server {self.server_id}: cannot assign negative viewers ({viewers})"
            )
        self.assigned_viewers = int(viewers)

    def calculate_cpu_utilization(self) -> float:
        """Calculate CPU utilization percentage based on capacity.

        Formula:
            CPU utilisation (%) = (assigned viewers / server capacity) * 100

        Returns:
            CPU utilization as a float percentage (not capped at 100%).
        """
        if self.config.capacity_per_server <= 0:
            return 0.0
        return (float(self.assigned_viewers) / float(self.config.capacity_per_server)) * 100.0

    def calculate_memory_utilization(self) -> float:
        """Calculate memory utilization percentage.

        Formula:
            Memory used (GB) = baseline memory + (assigned viewers * memory per viewer / 1024)
            Memory utilisation (%) = (Memory used / total server memory) * 100

        Returns:
            Memory utilization as a float percentage (not capped at 100%).
        """
        memory_used_gb = self.config.baseline_memory_gb + (
            float(self.assigned_viewers) * (self.config.memory_per_viewer_mb / 1024.0)
        )
        return (memory_used_gb / self.config.total_memory_gb) * 100.0

    def is_overloaded(self) -> bool:
        """Determine if the server is in an overloaded state (>100% CPU or Memory).

        Returns:
            True if CPU utilization > 100% or Memory utilization > 100%, else False.
        """
        return (
            self.calculate_cpu_utilization() > 100.0
            or self.calculate_memory_utilization() > 100.0
        )

    def reset_workload(self) -> None:
        """Reset assigned workload to zero for the next simulation interval."""
        self.assigned_viewers = 0

    def get_metrics(self, minute: int = 0) -> ServerMetrics:
        """Retrieve current performance metrics for this server.

        Args:
            minute: Current simulation minute index.

        Returns:
            ServerMetrics dataclass instance.
        """
        return ServerMetrics(
            minute=minute,
            server_id=self.server_id,
            assigned_viewers=self.assigned_viewers,
            cpu_utilization=self.calculate_cpu_utilization(),
            memory_utilization=self.calculate_memory_utilization(),
            overloaded=self.is_overloaded(),
        )

    def __repr__(self) -> str:
        return (
            f"SimulatedServer(id={self.server_id}, viewers={self.assigned_viewers}, "
            f"cpu={self.calculate_cpu_utilization():.1f}%, active={self.active})"
        )


class CloudSimulator:
    """Manages active server pools, load distribution, and simulation steps."""

    def __init__(self, config: Optional[ServerConfig] = None) -> None:
        """Initialize cloud infrastructure simulator.

        Args:
            config: Optional ServerConfig. Uses default parameters if None.
        """
        self.config: ServerConfig = config or ServerConfig()
        self._server_counter: int = 0
        self.servers: Dict[str, SimulatedServer] = {}
        self.reset_simulation_state()

    def reset_simulation_state(self) -> None:
        """Reset simulator state to the initial server configuration."""
        self.servers.clear()
        self._server_counter = 0
        for _ in range(self.config.initial_servers):
            self._create_server()

    def _create_server(self) -> SimulatedServer:
        """Instantiate a new simulated server with a unique sequential ID."""
        self._server_counter += 1
        server_id = f"server-{self._server_counter}"
        server = SimulatedServer(server_id=server_id, config=self.config)
        self.servers[server_id] = server
        return server

    def add_server(self) -> SimulatedServer:
        """Explicitly add a server to the active infrastructure pool.

        Returns:
            The newly created SimulatedServer instance.

        Raises:
            ValueError: If adding a server exceeds max_servers limit.
        """
        active_count = len(self.get_active_servers())
        if active_count >= self.config.max_servers:
            raise ValueError(
                f"Cannot add server: maximum server limit ({self.config.max_servers}) reached."
            )
        return self._create_server()

    def remove_server(self, server_id: Optional[str] = None) -> SimulatedServer:
        """Explicitly remove a server from the active infrastructure pool.

        Discrete Time Semantics:
            In our discrete-time simulation, workload allocations belong to the
            interval in which step() executes. When a server is removed between
            simulation intervals, it is decommissioned from the active pool. The
            subsequent step() call resets allocations and distributes the complete
            incoming workload across the remaining active servers.

        Args:
            server_id: Specific server ID to remove. If None, removes the most recently added.

        Returns:
            The removed SimulatedServer instance (marked active=False).

        Raises:
            ValueError: If removal violates min_servers limit or server_id does not exist.
        """
        active_servers = self.get_active_servers()
        if len(active_servers) <= self.config.min_servers:
            raise ValueError(
                f"Cannot remove server: minimum server limit ({self.config.min_servers}) reached."
            )

        if server_id is not None:
            if server_id not in self.servers or not self.servers[server_id].active:
                raise ValueError(f"Cannot remove server: '{server_id}' not found in active pool.")
            target_server = self.servers[server_id]
        else:
            # Default to removing the latest added active server (LIFO)
            target_server = active_servers[-1]

        target_server.active = False
        del self.servers[target_server.server_id]
        return target_server

    def get_active_servers(self) -> List[SimulatedServer]:
        """Get ordered list of currently active simulated servers."""
        return [s for s in self.servers.values() if s.active]

    def get_server_metrics(self, minute: int = 0) -> List[ServerMetrics]:
        """Get current metrics for all active servers.

        Args:
            minute: Current simulation minute index.

        Returns:
            List of ServerMetrics for each active server.
        """
        return [s.get_metrics(minute=minute) for s in self.get_active_servers()]

    def distribute_load(self, viewers: int) -> None:
        """Distribute viewer workload across active servers deterministically.

        Ensures that:
        1. Viewers are distributed as evenly as possible (|server_i - server_j| <= 1).
        2. Total assigned viewers strictly equals incoming viewers.
        3. All active servers receive their allocated portion.

        Args:
            viewers: Non-negative integer incoming viewer count.

        Raises:
            ValueError: If viewers count is negative or non-integer.
            RuntimeError: If no active servers are available.
        """
        if not isinstance(viewers, (int, float)) or int(viewers) != viewers:
            raise ValueError(f"Viewer count must be an integer, got {type(viewers).__name__}: {viewers}")
        int_viewers = int(viewers)
        if int_viewers < 0:
            raise ValueError(f"Viewer count cannot be negative, got {int_viewers}")

        active_servers = self.get_active_servers()
        num_active = len(active_servers)
        if num_active == 0:
            raise RuntimeError("Cannot distribute workload: no active servers available.")

        # Deterministic balanced integer distribution
        base_allocation = int_viewers // num_active
        remainder = int_viewers % num_active

        for idx, server in enumerate(active_servers):
            # First 'remainder' servers receive 1 extra viewer to match exact total
            assigned = base_allocation + (1 if idx < remainder else 0)
            server.assign_workload(assigned)

    def step(
        self, viewers: int, minute: int = 0
    ) -> Tuple[InfrastructureMetrics, List[ServerMetrics]]:
        """Advance the infrastructure simulation by one interval.

        Steps:
        1. Validate incoming viewer count.
        2. Reset previous interval's server workloads.
        3. Distribute current viewers across active servers.
        4. Calculate CPU and memory utilisation for each server.
        5. Identify overloaded servers.
        6. Generate aggregate and per-server metrics records.

        Args:
            viewers: Incoming concurrent viewer count.
            minute: Current simulation minute index.

        Returns:
            Tuple of (InfrastructureMetrics, List[ServerMetrics]).
        """
        # Step 2: Reset previous interval's workload
        for server in self.get_active_servers():
            server.reset_workload()

        # Step 3: Distribute current workload (validates viewers)
        self.distribute_load(viewers)

        # Step 4-6: Collect per-server and aggregate metrics
        server_metrics_list = self.get_server_metrics(minute=minute)
        aggregate_metrics = calculate_aggregate_metrics(
            minute=minute,
            total_viewers=int(viewers),
            capacity_per_server=self.config.capacity_per_server,
            server_metrics_list=server_metrics_list,
        )

        return aggregate_metrics, server_metrics_list
