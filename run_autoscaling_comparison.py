"""Stage D3: Integrated Autoscaling Comparison Experiment (Reactive vs Predictive).

This module runs a comprehensive comparative experiment between threshold-based Reactive
autoscaling (ReactiveScaler) and machine-learning-driven Predictive autoscaling
(PredictiveScaler) across four live-broadcast workload scenarios:
1. Normal Traffic
2. Sudden Traffic Spike
3. Peak Broadcasting Event
4. Event Ending

Discrete-Time Simulation Cycle (per interval t):
1. Ingest observed concurrent viewers for minute t.
2. Step CloudSimulator: reset allocations, balance load deterministically across active servers.
3. Observe infrastructure state: active servers, average CPU, memory utilization, overload status.
4. Record interval t infrastructure performance metrics.
5. Invoke autoscaler (Reactive or Predictive) with information available up to minute t.
6. Record scaling decision and reasoning.
7. Apply scaling adjustment (add_server / remove_server) on CloudSimulator. The resulting server
   pool handles traffic starting from interval t + 1 onward without retroactively altering
   interval t metrics.

Predictive Integrity:
- For PredictiveScaler, only historical observations from minute 0 to t are supplied.
- Never pass future viewer counts.
- Strict 6-observation minimum history requirement: minutes 0-4 return no_action; first
  prediction/scale-up decision occurs at minute 5 and takes effect at minute 6 (operating with
  the initial 2-server pool during intervals 0-5).
- Random Forest model predicts viewer demand 5 minutes ahead (t + 5).

Outputs:
- Detailed metrics CSV: data/processed/autoscaling_comparison_metrics.csv
- Scaling decisions CSV: data/processed/autoscaling_decisions.csv
- Summary metrics CSV: data/processed/autoscaling_comparison_summary.csv
- Visualizations:
  1. data/processed/overload_comparison.png
  2. data/processed/average_cpu_comparison.png
  3. data/processed/server_usage_comparison.png
  4. data/processed/autoscaling_timeline_comparison.png
"""

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress benign third-party threading/font warnings from sklearn/joblib/matplotlib
warnings.filterwarnings("ignore", category=UserWarning)

from autoscaling.base_scaler import BaseScaler, ScalerConfig, ScalingDecision
from autoscaling.predictive_scaler import PredictiveScaler
from autoscaling.reactive_scaler import ReactiveScaler
from simulation.cloud_simulator import CloudSimulator, ServerConfig

# Base project paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"


def load_raw_scenarios(
    data_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load and validate the four Stage A workload scenarios from CSV files.

    Args:
        data_dir: Directory containing raw scenario CSV files. Defaults to data/raw/.

    Returns:
        Dictionary mapping canonical scenario keys to validated DataFrames.

    Raises:
        FileNotFoundError: If any scenario CSV file is missing.
        ValueError: If any scenario dataset is empty, missing required columns,
            or contains invalid/malformed telemetry.
    """
    raw_path = Path(data_dir).resolve() if data_dir is not None else DATA_RAW_DIR

    scenario_files = {
        "normal": raw_path / "normal_traffic.csv",
        "sudden_spike": raw_path / "sudden_spike_traffic.csv",
        "peak_event": raw_path / "peak_event_traffic.csv",
        "event_ending": raw_path / "event_ending_traffic.csv",
    }

    scenarios: Dict[str, pd.DataFrame] = {}

    for key, file_path in scenario_files.items():
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Workload dataset '{file_path.name}' not found under '{raw_path}'. "
                "Ensure Stage A synthetic workloads have been generated."
            )

        df = pd.read_csv(file_path)
        if len(df) == 0:
            raise ValueError(f"Scenario dataset '{file_path.name}' is empty.")

        # Validate required columns
        for col in ["minute", "viewers"]:
            if col not in df.columns:
                raise ValueError(
                    f"Scenario dataset '{file_path.name}' missing required column '{col}'."
                )

        # Check for missing values in core columns
        if df["minute"].isna().any() or df["viewers"].isna().any():
            raise ValueError(
                f"Scenario dataset '{file_path.name}' contains NaN values in minute or viewers."
            )

        # Validate integer types and non-negative values
        try:
            minute_vals = df["minute"].to_numpy()
            viewer_vals = df["viewers"].to_numpy()
            if not np.all(np.equal(np.mod(minute_vals, 1), 0)):
                raise ValueError("Non-integer minute values detected.")
            if not np.all(np.equal(np.mod(viewer_vals, 1), 0)):
                raise ValueError("Non-integer viewer values detected.")
            if (minute_vals < 0).any():
                raise ValueError("Negative minute values detected.")
            if (viewer_vals < 0).any():
                raise ValueError("Negative viewer counts detected.")
        except (TypeError, ValueError) as err:
            raise ValueError(f"Malformed telemetry in '{file_path.name}': {err}") from err

        # Sort chronologically by minute
        df = df.sort_values(by="minute").reset_index(drop=True)

        # Verify contiguous minute step of 1
        if len(df) > 1:
            diffs = np.diff(df["minute"].to_numpy())
            if not np.all(diffs == 1):
                raise ValueError(
                    f"Gaps or non-consecutive minutes detected in '{file_path.name}'."
                )

        scenarios[key] = df

    return scenarios


def run_strategy_simulation(
    scenario_name: str,
    df: pd.DataFrame,
    strategy: str,
    server_config: Optional[ServerConfig] = None,
    scaler_config: Optional[ScalerConfig] = None,
    scaler_instance: Optional[BaseScaler] = None,
    simulator_instance: Optional[CloudSimulator] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run discrete-time simulation for a single strategy on a single workload trace.

    Execution Sequence for interval t:
    1. Read actual viewers V(t) and minute t.
    2. Step simulator with V(t), distributing load across current active servers.
    3. Read active server count and resulting CPU and memory utilization.
    4. Record infrastructure metrics for interval t (representing load handling state).
    5. Invoke autoscaler make_decision() with metrics available up to interval t.
       For PredictiveScaler, only historical rows up to t are provided.
    6. Record scaling decision and reason.
    7. Execute scaling action on CloudSimulator. The resulting server count will
       handle traffic in interval t + 1.

    Args:
        scenario_name: Human-readable scenario name (e.g., 'Normal Traffic').
        df: DataFrame containing 'minute' and 'viewers'.
        strategy: 'Reactive' or 'Predictive'.
        server_config: Infrastructure configuration parameters.
        scaler_config: Autoscaling policy configuration parameters.
        scaler_instance: Optional pre-configured scaler instance.
        simulator_instance: Optional pre-configured simulator instance.

    Returns:
        Tuple of (metrics_records_list, decision_records_list).

    Raises:
        ValueError: If strategy is unsupported or df is invalid.
    """
    if strategy not in {"Reactive", "Predictive"}:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Must be 'Reactive' or 'Predictive'."
        )

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"df must be a pandas DataFrame, got {type(df).__name__}.")

    for required_col in ["minute", "viewers"]:
        if required_col not in df.columns:
            raise ValueError(
                f"Workload DataFrame missing required column '{required_col}'."
            )

    # 1. Independent fresh simulator instance
    sim_cfg = server_config or ServerConfig()
    simulator = simulator_instance or CloudSimulator(config=sim_cfg)

    # 2. Independent scaler instance
    scale_cfg = scaler_config or ScalerConfig()
    if scaler_instance is not None:
        scaler = scaler_instance
    elif strategy == "Reactive":
        scaler = ReactiveScaler(config=scale_cfg)
    else:
        scaler = PredictiveScaler(config=scale_cfg)

    metrics_records: List[Dict[str, Any]] = []
    decision_records: List[Dict[str, Any]] = []

    # Ensure chronological order
    data = df.sort_values(by="minute").reset_index(drop=True)

    for idx in range(len(data)):
        row = data.iloc[idx]
        minute = int(row["minute"])
        viewers = int(row["viewers"])

        # Step 2: Step the simulator handling current workload
        infra_metric, _ = simulator.step(viewers=viewers, minute=minute)

        # Step 3 & 4: Read and capture current infrastructure state
        active_servers_during_interval = infra_metric.active_servers
        cpu_util = infra_metric.average_cpu_utilization
        mem_util = infra_metric.average_memory_utilization
        overload_status = infra_metric.overload_status

        # Step 5: Invoke appropriate autoscaler
        if strategy == "Reactive":
            # Reactive scaler receives only current interval's observed information
            decision = scaler.make_decision(
                current_interval=minute,
                current_viewers=viewers,
                current_active_servers=active_servers_during_interval,
                current_cpu_utilization=cpu_util,
                simulator=simulator,
            )
        else:
            # Predictive scaler receives strictly history up to current interval t
            # Never pass future viewer counts
            history_slice = data.iloc[: idx + 1][["minute", "viewers"]].copy()
            if "event_active" in data.columns:
                history_slice["event_active"] = data.iloc[: idx + 1]["event_active"].values

            decision = scaler.make_decision(
                current_interval=minute,
                current_viewers=viewers,
                current_active_servers=active_servers_during_interval,
                current_cpu_utilization=cpu_util,
                simulator=simulator,
                viewer_history=history_slice,
            )

        # Step 6: Record infrastructure metrics row for interval t
        # (reflects infrastructure state while handling interval t's workload)
        metrics_record = {
            "scenario": scenario_name,
            "strategy": strategy,
            "minute": minute,
            "actual_viewers": viewers,
            "active_servers": active_servers_during_interval,
            "cpu_utilization": cpu_util,
            "memory_utilization": mem_util,
            "overload_status": overload_status,
            "scaling_action": decision.action,
            "previous_server_count": decision.previous_servers,
            "desired_server_count": decision.desired_servers,
            "actual_server_count_after_decision": decision.actual_servers,
            "predicted_viewers": decision.predicted_viewers,
        }
        metrics_records.append(metrics_record)

        # Record scaling decision record
        decision_record = {
            "scenario": scenario_name,
            "strategy": strategy,
            "minute": minute,
            "action": decision.action,
            "reason": decision.reason,
            "previous_server_count": decision.previous_servers,
            "desired_server_count": decision.desired_servers,
            "actual_server_count": decision.actual_servers,
            "predicted_viewers": decision.predicted_viewers,
        }
        decision_records.append(decision_record)

    return metrics_records, decision_records


def calculate_summary_metrics(
    metrics_df: pd.DataFrame,
    decisions_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Calculate aggregated comparison summary metrics per scenario and strategy.

    Metrics computed:
    - Total overload intervals: Number of intervals where overload_status is True.
    - Overload percentage: (Total overload intervals / Total intervals) * 100.
    - Average CPU utilization: Mean average_cpu_utilization across all intervals.
    - Maximum CPU utilization: Peak average_cpu_utilization observed in any interval.
    - Average active servers: Mean active_servers handling workload across intervals.
    - Maximum active servers: Peak active_servers count observed in any interval.
    - Total server-minutes: Sum of active servers across all intervals (server resource consumption).
    - Scale-up operations: Count of executed scale_up actions.
    - Scale-down operations: Count of executed scale_down actions.
    - Total scaling operations: Sum of scale-up and scale-down operations.

    Args:
        metrics_df: Detailed metrics DataFrame.
        decisions_df: Optional decisions DataFrame (if omitted, extracted from metrics_df).

    Returns:
        Summary DataFrame indexed by scenario and strategy.
    """
    summary_rows: List[Dict[str, Any]] = []

    # Preserve order of scenarios and strategies as they appear
    scenarios = metrics_df["scenario"].unique()
    strategies = metrics_df["strategy"].unique()

    for sc in scenarios:
        for strat in strategies:
            sub_m = metrics_df[
                (metrics_df["scenario"] == sc) & (metrics_df["strategy"] == strat)
            ]
            if len(sub_m) == 0:
                continue

            total_intervals = len(sub_m)
            total_overload_intervals = int(sub_m["overload_status"].sum())
            overload_pct = (
                (float(total_overload_intervals) / float(total_intervals)) * 100.0
                if total_intervals > 0
                else 0.0
            )
            avg_cpu = float(sub_m["cpu_utilization"].mean()) if total_intervals > 0 else 0.0
            max_cpu = float(sub_m["cpu_utilization"].max()) if total_intervals > 0 else 0.0
            avg_servers = (
                float(sub_m["active_servers"].mean()) if total_intervals > 0 else 0.0
            )
            max_servers = int(sub_m["active_servers"].max()) if total_intervals > 0 else 0
            total_server_mins = int(sub_m["active_servers"].sum())

            # Count actual scaling actions executed
            # In our simulation loop, a decision with action == "scale_up" represents
            # an executed scale-up where actual_servers > previous_servers
            scale_ups = int((sub_m["scaling_action"] == "scale_up").sum())
            scale_downs = int((sub_m["scaling_action"] == "scale_down").sum())
            total_scaling = scale_ups + scale_downs

            summary_rows.append(
                {
                    "scenario": sc,
                    "strategy": strat,
                    "total_overload_intervals": total_overload_intervals,
                    "overload_percentage": round(overload_pct, 2),
                    "average_cpu_utilization": round(avg_cpu, 2),
                    "maximum_cpu_utilization": round(max_cpu, 2),
                    "average_active_servers": round(avg_servers, 2),
                    "maximum_active_servers": max_servers,
                    "total_server_minutes": total_server_mins,
                    "scale_up_operations": scale_ups,
                    "scale_down_operations": scale_downs,
                    "total_scaling_operations": total_scaling,
                }
            )

    return pd.DataFrame(summary_rows)


def generate_comparison_visualizations(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Generate high-quality comparison charts saved to data/processed/.

    Generated figures:
    1. overload_comparison.png: Grouped bar chart of overload intervals across scenarios.
    2. average_cpu_comparison.png: Grouped bar chart of average CPU utilization with threshold lines.
    3. server_usage_comparison.png: Resource consumption comparison (Total Server-Minutes & Avg Servers).
    4. autoscaling_timeline_comparison.png: Comprehensive 4-panel timeline showing viewer demand,
       server count trajectories, and scaling events for both strategies.

    Args:
        metrics_df: Detailed metrics DataFrame.
        summary_df: Summary comparison DataFrame.
        output_dir: Directory where PNG plots will be saved.

    Returns:
        Dictionary mapping plot identifiers to saved file paths.
    """
    out_dir = Path(output_dir).resolve() if output_dir is not None else DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, Path] = {}

    scenarios = list(summary_df["scenario"].unique())
    n_scenarios = len(scenarios)
    x = np.arange(n_scenarios)
    width = 0.35

    color_reactive = "#e74c3c"  # Coral / Red
    color_predictive = "#2980b9"  # Sapphire Blue

    # -------------------------------------------------------------
    # 1. OVERLOAD COMPARISON (overload_comparison.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    reactive_overloads = [
        int(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Reactive")
            ]["total_overload_intervals"].iloc[0]
        )
        for sc in scenarios
    ]
    predictive_overloads = [
        int(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Predictive")
            ]["total_overload_intervals"].iloc[0]
        )
        for sc in scenarios
    ]

    rects1 = ax.bar(
        x - width / 2,
        reactive_overloads,
        width,
        label="Reactive Autoscaler",
        color=color_reactive,
        alpha=0.9,
    )
    rects2 = ax.bar(
        x + width / 2,
        predictive_overloads,
        width,
        label="Predictive Autoscaler",
        color=color_predictive,
        alpha=0.9,
    )

    # Value annotations on bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(
            f"{int(height)} min",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(
            f"{int(height)} min",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(
        "Stage D3: Overload Comparison (Total Overloaded Minutes per Scenario)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Overloaded Simulation Minutes", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()

    plot1_path = out_dir / "overload_comparison.png"
    fig.savefig(plot1_path)
    plt.close(fig)
    saved_plots["overload_comparison"] = plot1_path

    # -------------------------------------------------------------
    # 2. AVERAGE CPU UTILIZATION (average_cpu_comparison.png)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    reactive_cpu = [
        float(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Reactive")
            ]["average_cpu_utilization"].iloc[0]
        )
        for sc in scenarios
    ]
    predictive_cpu = [
        float(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Predictive")
            ]["average_cpu_utilization"].iloc[0]
        )
        for sc in scenarios
    ]

    rects1 = ax.bar(
        x - width / 2,
        reactive_cpu,
        width,
        label="Reactive Autoscaler",
        color=color_reactive,
        alpha=0.9,
    )
    rects2 = ax.bar(
        x + width / 2,
        predictive_cpu,
        width,
        label="Predictive Autoscaler",
        color=color_predictive,
        alpha=0.9,
    )

    # Annotate CPU values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Reference policy thresholds
    ax.axhline(
        70.0,
        color="#27ae60",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Target CPU Threshold (70%)",
    )
    ax.axhline(
        30.0,
        color="#f39c12",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Scale-Down CPU Threshold (30%)",
    )
    ax.axhline(
        100.0,
        color="#c0392b",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label="Capacity Limit (100%)",
    )

    ax.set_title(
        "Stage D3: Average CPU Utilization Comparison Across Scenarios",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylabel("Average CPU Utilization (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.legend(framealpha=0.9, fontsize=9.5, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()

    plot2_path = out_dir / "average_cpu_comparison.png"
    fig.savefig(plot2_path)
    plt.close(fig)
    saved_plots["average_cpu_comparison"] = plot2_path

    # -------------------------------------------------------------
    # 3. SERVER RESOURCE CONSUMPTION (server_usage_comparison.png)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    reactive_mins = [
        int(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Reactive")
            ]["total_server_minutes"].iloc[0]
        )
        for sc in scenarios
    ]
    predictive_mins = [
        int(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Predictive")
            ]["total_server_minutes"].iloc[0]
        )
        for sc in scenarios
    ]

    # Panel 1: Total Server-Minutes
    ax1.bar(
        x - width / 2,
        reactive_mins,
        width,
        label="Reactive",
        color=color_reactive,
        alpha=0.9,
    )
    ax1.bar(
        x + width / 2,
        predictive_mins,
        width,
        label="Predictive",
        color=color_predictive,
        alpha=0.9,
    )
    for idx, (rm, pm) in enumerate(zip(reactive_mins, predictive_mins)):
        ax1.annotate(
            f"{rm}",
            xy=(idx - width / 2, rm),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
        ax1.annotate(
            f"{pm}",
            xy=(idx + width / 2, pm),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax1.set_title("Total Server-Minutes (Resource Consumption)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Server-Minutes", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=9, rotation=15)
    ax1.legend(fontsize=9.5)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Panel 2: Average Active Servers
    reactive_avg_s = [
        float(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Reactive")
            ]["average_active_servers"].iloc[0]
        )
        for sc in scenarios
    ]
    predictive_avg_s = [
        float(
            summary_df[
                (summary_df["scenario"] == sc) & (summary_df["strategy"] == "Predictive")
            ]["average_active_servers"].iloc[0]
        )
        for sc in scenarios
    ]

    ax2.bar(
        x - width / 2,
        reactive_avg_s,
        width,
        label="Reactive",
        color=color_reactive,
        alpha=0.9,
    )
    ax2.bar(
        x + width / 2,
        predictive_avg_s,
        width,
        label="Predictive",
        color=color_predictive,
        alpha=0.9,
    )
    for idx, (rs, ps) in enumerate(zip(reactive_avg_s, predictive_avg_s)):
        ax2.annotate(
            f"{rs:.1f}",
            xy=(idx - width / 2, rs),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
        ax2.annotate(
            f"{ps:.1f}",
            xy=(idx + width / 2, ps),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_title("Average Active Servers", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Servers", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, fontsize=9, rotation=15)
    ax2.legend(fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(
        "Stage D3: Cloud Resource Provisioning Comparison",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    plot3_path = out_dir / "server_usage_comparison.png"
    fig.savefig(plot3_path)
    plt.close(fig)
    saved_plots["server_usage_comparison"] = plot3_path

    # -------------------------------------------------------------
    # 4. AUTOSCALING TIMELINE COMPARISON (autoscaling_timeline_comparison.png)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(14, 3.8 * n_scenarios), dpi=120)
    if n_scenarios == 1:
        axes = [axes]

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        sub_react = metrics_df[
            (metrics_df["scenario"] == sc) & (metrics_df["strategy"] == "Reactive")
        ].sort_values("minute")
        sub_pred = metrics_df[
            (metrics_df["scenario"] == sc) & (metrics_df["strategy"] == "Predictive")
        ].sort_values("minute")

        minutes = sub_react["minute"].to_numpy()
        viewers = sub_react["actual_viewers"].to_numpy()

        # Twin axis: Left for Viewer Demand (shaded)
        ax_viewers = ax.twinx()
        ax_viewers.fill_between(
            minutes,
            0,
            viewers,
            color="#bdc3c7",
            alpha=0.25,
            label="Viewer Demand",
        )
        ax_viewers.plot(
            minutes,
            viewers,
            color="#7f8c8d",
            linestyle=":",
            linewidth=1.2,
            alpha=0.6,
        )
        ax_viewers.set_ylabel("Viewer Demand", color="#555555", fontsize=9.5)
        ax_viewers.set_ylim(0, max(viewers) * 1.15 if max(viewers) > 0 else 1000)

        # Primary axis: Server allocations and scaling events
        # Reactive servers
        line1 = ax.plot(
            minutes,
            sub_react["active_servers"],
            color=color_reactive,
            linewidth=2.0,
            label="Reactive Active Servers",
        )
        # Predictive servers
        line2 = ax.plot(
            minutes,
            sub_pred["active_servers"],
            color=color_predictive,
            linewidth=2.0,
            label="Predictive Active Servers",
        )

        # Scaling action markers
        # Reactive scale-ups and scale-downs
        r_up = sub_react[sub_react["scaling_action"] == "scale_up"]
        r_down = sub_react[sub_react["scaling_action"] == "scale_down"]
        ax.scatter(
            r_up["minute"],
            r_up["active_servers"],
            marker="^",
            color="#c0392b",
            s=55,
            zorder=4,
            label="Reactive Scale-Up",
        )
        ax.scatter(
            r_down["minute"],
            r_down["active_servers"],
            marker="v",
            color="#e67e22",
            s=55,
            zorder=4,
            label="Reactive Scale-Down",
        )

        # Predictive scale-ups and scale-downs
        p_up = sub_pred[sub_pred["scaling_action"] == "scale_up"]
        p_down = sub_pred[sub_pred["scaling_action"] == "scale_down"]
        ax.scatter(
            p_up["minute"],
            p_up["active_servers"],
            marker="^",
            color="#1b4f72",
            s=55,
            zorder=4,
            label="Predictive Scale-Up",
        )
        ax.scatter(
            p_down["minute"],
            p_down["active_servers"],
            marker="v",
            color="#2e86c1",
            s=55,
            zorder=4,
            label="Predictive Scale-Down",
        )

        ax.set_title(f"{sc}: Autoscaling Timeline Comparison", fontsize=11, fontweight="bold")
        ax.set_ylabel("Active Servers", fontsize=10)
        ax.set_ylim(0, 32)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlim(0, max(minutes))

        if idx == n_scenarios - 1:
            ax.set_xlabel("Simulation Minute", fontsize=11)

        # Clean combined legend for the first subplot
        if idx == 0:
            lines_a, labels_a = ax.get_legend_handles_labels()
            lines_b, labels_b = ax_viewers.get_legend_handles_labels()
            ax.legend(
                lines_b + lines_a,
                labels_b + labels_a,
                loc="upper left",
                fontsize=8.5,
                ncol=3,
                framealpha=0.9,
            )

    fig.suptitle(
        "Stage D3: Viewer Demand, Server Allocation, and Scaling Action Timelines",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()

    plot4_path = out_dir / "autoscaling_timeline_comparison.png"
    fig.savefig(plot4_path)
    plt.close(fig)
    saved_plots["autoscaling_timeline_comparison"] = plot4_path

    return saved_plots


def run_autoscaling_comparison(
    scenarios: Optional[Dict[str, pd.DataFrame]] = None,
    server_config: Optional[ServerConfig] = None,
    scaler_config: Optional[ScalerConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
    generate_plots: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the complete Stage D3 comparison experiment across all scenarios.

    Args:
        scenarios: Dictionary of raw scenario DataFrames. If None, loaded from data/raw/.
        server_config: Server infrastructure configuration. Uses defaults if None.
        scaler_config: Autoscaler policy configuration. Uses defaults if None.
        output_dir: Destination directory for CSV and visualization outputs.
        generate_plots: Whether to generate and save Matplotlib comparison charts.

    Returns:
        Tuple of (metrics_df, decisions_df, summary_df).
    """
    out_dir = Path(output_dir).resolve() if output_dir is not None else DATA_PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_scenarios = scenarios or load_raw_scenarios()

    sim_cfg = server_config or ServerConfig()
    scale_cfg = scaler_config or ScalerConfig()

    all_metrics: List[Dict[str, Any]] = []
    all_decisions: List[Dict[str, Any]] = []

    for sc_key, df in loaded_scenarios.items():
        scenario_name = df["scenario"].iloc[0] if "scenario" in df.columns else sc_key

        # Strategy 1: Reactive Autoscaling
        # Fresh simulator instance and fresh scaler instance
        m_reactive, d_reactive = run_strategy_simulation(
            scenario_name=scenario_name,
            df=df,
            strategy="Reactive",
            server_config=sim_cfg,
            scaler_config=scale_cfg,
        )
        all_metrics.extend(m_reactive)
        all_decisions.extend(d_reactive)

        # Strategy 2: Predictive Autoscaling
        # Fresh simulator instance and fresh scaler instance
        m_predictive, d_predictive = run_strategy_simulation(
            scenario_name=scenario_name,
            df=df,
            strategy="Predictive",
            server_config=sim_cfg,
            scaler_config=scale_cfg,
        )
        all_metrics.extend(m_predictive)
        all_decisions.extend(d_predictive)

    metrics_df = pd.DataFrame(all_metrics)
    decisions_df = pd.DataFrame(all_decisions)
    summary_df = calculate_summary_metrics(metrics_df=metrics_df, decisions_df=decisions_df)

    # Reorder columns explicitly to match project specifications
    metrics_cols = [
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
    metrics_df = metrics_df[metrics_cols]

    decision_cols = [
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
    decisions_df = decisions_df[decision_cols]

    # Save outputs
    metrics_path = out_dir / "autoscaling_comparison_metrics.csv"
    decisions_path = out_dir / "autoscaling_decisions.csv"
    summary_path = out_dir / "autoscaling_comparison_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    decisions_df.to_csv(decisions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    if generate_plots:
        generate_comparison_visualizations(
            metrics_df=metrics_df,
            summary_df=summary_df,
            output_dir=out_dir,
        )

    return metrics_df, decisions_df, summary_df


def main() -> None:
    """Main CLI entrypoint for running the Stage D3 autoscaling comparison."""
    print("=" * 80)
    print("STAGE D3: COMPARATIVE EXPERIMENT - REACTIVE VS PREDICTIVE AUTOSCALING")
    print("=" * 80)

    print("\n[1/4] Loading raw workload scenarios from data/raw/...")
    scenarios = load_raw_scenarios()
    for name, df in scenarios.items():
        print(f"  - Loaded {name:15s}: {len(df)} intervals (minutes 0 to {len(df)-1})")

    print("\n[2/4] Executing simulations for ReactiveScaler and PredictiveScaler...")
    metrics_df, decisions_df, summary_df = run_autoscaling_comparison(
        scenarios=scenarios,
        generate_plots=True,
    )
    print("  - Completed simulations across all 4 scenarios.")

    print("\n[3/4] Generated Artifacts:")
    print(f"  - Detailed Metrics:  {DATA_PROCESSED_DIR / 'autoscaling_comparison_metrics.csv'}")
    print(f"  - Scaling Decisions: {DATA_PROCESSED_DIR / 'autoscaling_decisions.csv'}")
    print(f"  - Summary Results:   {DATA_PROCESSED_DIR / 'autoscaling_comparison_summary.csv'}")
    print(f"  - Overload Chart:    {DATA_PROCESSED_DIR / 'overload_comparison.png'}")
    print(f"  - Average CPU Chart: {DATA_PROCESSED_DIR / 'average_cpu_comparison.png'}")
    print(f"  - Server Usage:      {DATA_PROCESSED_DIR / 'server_usage_comparison.png'}")
    print(f"  - Timeline Plot:     {DATA_PROCESSED_DIR / 'autoscaling_timeline_comparison.png'}")

    print("\n[4/4] SUMMARY RESULTS:")
    print("-" * 105)
    header = (
        f"{'Scenario':<25} | {'Strategy':<11} | {'Overloads':<10} | {'Avg CPU':<9} | "
        f"{'Max CPU':<9} | {'Avg Svr':<8} | {'Max Svr':<8} | {'Svr-Mins':<9} | {'Ups':<4} | {'Downs':<5}"
    )
    print(header)
    print("-" * 105)

    for _, row in summary_df.iterrows():
        print(
            f"{row['scenario']:<25} | {row['strategy']:<11} | "
            f"{row['total_overload_intervals']:>3d} ({row['overload_percentage']:>4.1f}%) | "
            f"{row['average_cpu_utilization']:>7.1f}% | {row['maximum_cpu_utilization']:>7.1f}% | "
            f"{row['average_active_servers']:>8.2f} | {row['maximum_active_servers']:>8d} | "
            f"{row['total_server_minutes']:>9d} | {row['scale_up_operations']:>4d} | "
            f"{row['scale_down_operations']:>5d}"
        )
    print("-" * 105)
    print("\nStage D3 Autoscaling Comparison complete successfully.")


if __name__ == "__main__":
    main()
