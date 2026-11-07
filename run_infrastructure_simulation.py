"""Stage C Execution Pipeline: Simulated Cloud Infrastructure Simulation.

This script executes the Stage C experiment:
1. Loads the four Stage A workload scenarios from data/raw/.
2. Simulates a fixed cloud server pool (default initial = 2 servers, 1,000 capacity each).
3. Evaluates infrastructure performance under normal, sudden spike, peak, and ending traffic.
4. Records aggregate infrastructure metrics and per-server metrics.
5. Saves results to data/processed/infrastructure_metrics.csv and server_metrics.csv.
6. Generates Matplotlib visualizations for CPU utilization, server loads, and overload tracking.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.cloud_simulator import CloudSimulator, ServerConfig
from simulation.metrics import InfrastructureMetrics, ServerMetrics

# Directory paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"


def load_raw_scenarios() -> Dict[str, pd.DataFrame]:
    """Load individual Stage A CSV scenarios from data/raw/.

    Returns:
        Dictionary mapping scenario keys to DataFrames.

    Raises:
        FileNotFoundError: If any scenario CSV is missing.
    """
    scenario_files = {
        "normal": DATA_RAW_DIR / "normal_traffic.csv",
        "sudden_spike": DATA_RAW_DIR / "sudden_spike_traffic.csv",
        "peak_event": DATA_RAW_DIR / "peak_event_traffic.csv",
        "event_ending": DATA_RAW_DIR / "event_ending_traffic.csv",
    }

    scenarios: Dict[str, pd.DataFrame] = {}
    for key, path in scenario_files.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Stage A dataset '{path.name}' not found under data/raw/. "
                "Ensure Stage A has been executed."
            )
        df = pd.read_csv(path)
        scenarios[key] = df

    return scenarios


def run_simulation(
    scenarios: Dict[str, pd.DataFrame],
    config: Optional[ServerConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run infrastructure simulation across all scenarios with fixed servers.

    Args:
        scenarios: Dictionary of raw scenario DataFrames.
        config: Infrastructure configuration. Uses defaults if None.

    Returns:
        Tuple of (infrastructure_metrics_df, server_metrics_df).
    """
    cfg = config or ServerConfig()
    simulator = CloudSimulator(config=cfg)

    all_infra_records: List[Dict] = []
    all_server_records: List[Dict] = []

    for key, df in scenarios.items():
        scenario_name = df["scenario"].iloc[0] if "scenario" in df.columns else key

        # Reset simulator state for fresh fixed-server baseline per scenario
        simulator.reset_simulation_state()

        for _, row in df.iterrows():
            minute = int(row["minute"])
            viewers = int(row["viewers"])

            # Step the infrastructure
            infra_metric, server_metrics = simulator.step(viewers=viewers, minute=minute)

            # Record aggregate metric with scenario label
            infra_dict = infra_metric.to_dict()
            infra_dict["scenario"] = scenario_name
            all_infra_records.append(infra_dict)

            # Record per-server metrics with scenario label
            for sm in server_metrics:
                sm_dict = sm.to_dict()
                sm_dict["scenario"] = scenario_name
                all_server_records.append(sm_dict)

    infra_df = pd.DataFrame(all_infra_records)
    server_df = pd.DataFrame(all_server_records)

    # Reorder columns so 'scenario' and 'minute' are up front
    infra_cols = ["scenario", "minute"] + [c for c in infra_df.columns if c not in ["scenario", "minute"]]
    server_cols = ["scenario", "minute", "server_id"] + [
        c for c in server_df.columns if c not in ["scenario", "minute", "server_id"]
    ]

    return infra_df[infra_cols], server_df[server_cols]


def generate_visualizations(
    infra_df: pd.DataFrame,
    server_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Generate Stage C Matplotlib visualizations saved under data/processed/.

    Creates:
    A. cpu_utilization_comparison.png: Average CPU utilization for all 4 scenarios.
    B. active_server_load.png: Individual server CPU loads for representative scenarios.
    C. infrastructure_overload.png: Overloaded server count over time for all 4 scenarios.

    Args:
        infra_df: Aggregate infrastructure metrics DataFrame.
        server_df: Per-server metrics DataFrame.

    Returns:
        Dictionary of plot names to saved paths.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, Path] = {}

    scenarios = infra_df["scenario"].unique()
    scenario_colors = {
        "Normal Traffic": "#1f77b4",
        "Sudden Traffic Spike": "#d62728",
        "Peak Broadcasting Event": "#2ca02c",
        "Event Ending": "#ff7f0e",
    }

    # Plot A: CPU Utilization Comparison across all scenarios
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
    for sc in scenarios:
        sub = infra_df[infra_df["scenario"] == sc]
        ax.plot(
            sub["minute"],
            sub["average_cpu_utilization"],
            label=f"{sc} (Avg CPU)",
            color=scenario_colors.get(sc, "#333333"),
            linewidth=2.0,
        )

    # Overload threshold line
    ax.axhline(
        100.0,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Overload Threshold (100% CPU)",
    )
    ax.set_title(
        "Stage C: Fixed Infrastructure CPU Utilization Comparison (2 Active Servers)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Simulation Minute", fontsize=11)
    ax.set_ylabel("Average CPU Utilization (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    path_a = DATA_PROCESSED_DIR / "cpu_utilization_comparison.png"
    fig.savefig(path_a)
    plt.close(fig)
    saved_plots["cpu_utilization_comparison"] = path_a

    # Plot B: Active Server Load (Individual Server Breakdown for Sudden Spike & Normal)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120, sharey=True)

    # Subplot 1: Sudden Spike individual servers
    spike_sub = server_df[server_df["scenario"] == "Sudden Traffic Spike"]
    for s_id in spike_sub["server_id"].unique():
        s_data = spike_sub[spike_sub["server_id"] == s_id]
        ax1.plot(s_data["minute"], s_data["cpu_utilization"], label=f"{s_id} CPU", linewidth=1.8)
    ax1.axhline(100.0, color="red", linestyle="--", alpha=0.7, label="Capacity Limit (100%)")
    ax1.set_title("Sudden Traffic Spike: Per-Server CPU Load", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Simulation Minute", fontsize=10)
    ax1.set_ylabel("CPU Utilization (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=8.5)

    # Subplot 2: Normal Traffic individual servers
    norm_sub = server_df[server_df["scenario"] == "Normal Traffic"]
    for s_id in norm_sub["server_id"].unique():
        s_data = norm_sub[norm_sub["server_id"] == s_id]
        ax2.plot(s_data["minute"], s_data["cpu_utilization"], label=f"{s_id} CPU", linewidth=1.8)
    ax2.axhline(100.0, color="red", linestyle="--", alpha=0.7, label="Capacity Limit (100%)")
    ax2.set_title("Normal Traffic: Per-Server CPU Load", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Simulation Minute", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "Stage C: Individual Server Load Balancing & Resource Consumption",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path_b = DATA_PROCESSED_DIR / "active_server_load.png"
    fig.savefig(path_b)
    plt.close(fig)
    saved_plots["active_server_load"] = path_b

    # Plot C: Overloaded Server Count over time (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120, sharex=True, sharey=True)
    for idx, sc in enumerate(scenarios):
        ax = axes[idx // 2, idx % 2]
        sub = infra_df[infra_df["scenario"] == sc]

        ax.fill_between(
            sub["minute"],
            0,
            sub["overloaded_servers"],
            color=scenario_colors.get(sc, "#333333"),
            alpha=0.35,
        )
        ax.plot(
            sub["minute"],
            sub["overloaded_servers"],
            color=scenario_colors.get(sc, "#333333"),
            linewidth=2.0,
            label="Overloaded Servers",
        )
        ax.set_title(f"{sc}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Overloaded Count", fontsize=10)
        ax.set_ylim(-0.1, 2.3)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5)
        if idx >= 2:
            ax.set_xlabel("Simulation Minute", fontsize=10)

    fig.suptitle(
        "Stage C: Infrastructure Overloaded Server Count (Fixed Pool = 2 Servers)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path_c = DATA_PROCESSED_DIR / "infrastructure_overload.png"
    fig.savefig(path_c)
    plt.close(fig)
    saved_plots["infrastructure_overload"] = path_c

    return saved_plots


def print_simulation_summaries(infra_df: pd.DataFrame) -> None:
    """Print readable summary statistics for each scenario simulation."""
    print("\n" + "=" * 128)
    print("STAGE C: SIMULATED CLOUD INFRASTRUCTURE EXPERIMENT SUMMARY (Fixed Servers = 2, Capacity = 2,000)")
    print("=" * 128)
    print(
        f"{'Scenario Name':<25} | {'Duration':>8} | {'Max Viewers':>11} | {'Overloaded Min':>14} | "
        f"{'Max Overloaded Servers':>22} | {'Avg CPU (%)':>11} | {'Max CPU (%)':>11} | {'Avg Mem (%)':>11} | {'Max Mem (%)':>11}"
    )
    print("-" * 128)

    for sc in infra_df["scenario"].unique():
        sub = infra_df[infra_df["scenario"] == sc]
        n_obs = len(sub)
        max_viewers = int(sub["total_viewers"].max())
        overload_mins = int((sub["overloaded_servers"] > 0).sum())
        overload_pct = (overload_mins / n_obs) * 100.0
        max_overloaded_servers = int(sub["overloaded_servers"].max())
        active_fleet = int(sub["active_servers"].iloc[0])

        avg_cpu = float(sub["average_cpu_utilization"].mean())
        max_cpu = float(sub["maximum_cpu_utilization"].max())
        avg_mem = float(sub["average_memory_utilization"].mean())
        max_mem = float(sub["maximum_memory_utilization"].max())

        overload_str = f"{overload_mins} ({overload_pct:.1f}%)"
        max_overload_str = f"{max_overloaded_servers} / {active_fleet}"
        print(
            f"{sc:<25} | {n_obs:>8d} | {max_viewers:>11,d} | {overload_str:>14} | "
            f"{max_overload_str:>22} | {avg_cpu:>11.1f}% | {max_cpu:>11.1f}% | {avg_mem:>11.1f}% | {max_mem:>11.1f}%"
        )

    print("=" * 128 + "\n")


def main() -> None:
    """Run the Stage C infrastructure simulation pipeline."""
    print("\n" + "=" * 80)
    print("STAGE C: SIMULATED CLOUD INFRASTRUCTURE SIMULATION")
    print("=" * 80)

    # 1. Load Stage A raw scenario data
    print("Loading Stage A scenario datasets from data/raw/...")
    scenarios = load_raw_scenarios()
    print(f"  - Loaded {len(scenarios)} scenarios.")

    # 2. Run simulation with default fixed configuration (2 servers)
    config = ServerConfig(
        capacity_per_server=1000,
        min_servers=2,
        max_servers=30,
        initial_servers=2,
        total_memory_gb=8.0,
        baseline_memory_gb=1.0,
        memory_per_viewer_mb=4.0,
    )
    print(
        f"Simulating infrastructure: {config.initial_servers} initial servers, "
        f"{config.capacity_per_server} viewer capacity each (total capacity = {config.initial_servers * config.capacity_per_server})..."
    )
    infra_df, server_df = run_simulation(scenarios, config=config)

    # 3. Save output CSV datasets
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    infra_csv_path = DATA_PROCESSED_DIR / "infrastructure_metrics.csv"
    server_csv_path = DATA_PROCESSED_DIR / "server_metrics.csv"

    infra_df.to_csv(infra_csv_path, index=False)
    server_df.to_csv(server_csv_path, index=False)
    print(f"  - Saved aggregate metrics to: {infra_csv_path.name} ({len(infra_df)} rows)")
    print(f"  - Saved per-server metrics to: {server_csv_path.name} ({len(server_df)} rows)")

    # 4. Generate visualizations
    print("Generating infrastructure visualizations under data/processed/...")
    plots = generate_visualizations(infra_df, server_df)
    for name, p in plots.items():
        print(f"  - Generated plot: {p.name}")

    # 5. Print summary
    print_simulation_summaries(infra_df)
    print("Stage C infrastructure simulation completed successfully!\n")


if __name__ == "__main__":
    main()
