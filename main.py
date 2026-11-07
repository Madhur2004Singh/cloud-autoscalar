"""Stage A Entry Point: Synthetic Workload Generator for Live Broadcasting.

This script executes the Stage A pipeline:
1. Generates synthetic viewership workloads across four realistic scenarios.
2. Saves individual scenario datasets as CSV files under data/raw/.
3. Combines all four scenarios into a unified dataset under data/raw/.
4. Prints descriptive summary statistics (observations, min, max, average).
5. Generates high-quality visualization plots saved under data/processed/.
"""

import os
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd

from simulation.workload_generator import (
    DEFAULT_DURATION_MINUTES,
    DEFAULT_RANDOM_SEED,
    generate_all_scenarios,
)

# Project paths
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"


def ensure_directories() -> None:
    """Ensure required data output directories exist."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_scenarios_to_csv(
    scenarios: Dict[str, pd.DataFrame]
) -> Dict[str, Path]:
    """Save individual and combined scenario datasets to CSV under data/raw/.

    Args:
        scenarios: Dictionary of scenario DataFrames.

    Returns:
        Dictionary mapping scenario names to their saved file paths.
    """
    saved_paths: Dict[str, Path] = {}

    for key, df in scenarios.items():
        file_path = DATA_RAW_DIR / f"{key}_traffic.csv"
        df.to_csv(file_path, index=False)
        saved_paths[key] = file_path

    # Combined dataset containing all scenarios
    combined_df = pd.concat(scenarios.values(), ignore_index=True)
    combined_path = DATA_RAW_DIR / "all_scenarios_combined.csv"
    combined_df.to_csv(combined_path, index=False)
    saved_paths["combined"] = combined_path

    return saved_paths


def print_workload_summaries(scenarios: Dict[str, pd.DataFrame]) -> None:
    """Print clean summary statistics for each scenario.

    Args:
        scenarios: Dictionary of scenario DataFrames.
    """
    print("\n" + "=" * 76)
    print("STAGE A: SYNTHETIC WORKLOAD GENERATOR SUMMARY")
    print("=" * 76)
    print(
        f"{'Scenario Name':<28} | {'Observations':>12} | {'Min Viewers':>11} | {'Max Viewers':>11} | {'Avg Viewers':>11}"
    )
    print("-" * 76)

    for key, df in scenarios.items():
        name = df["scenario"].iloc[0]
        n_obs = len(df)
        min_v = int(df["viewers"].min())
        max_v = int(df["viewers"].max())
        avg_v = float(df["viewers"].mean())
        print(
            f"{name:<28} | {n_obs:>12} | {min_v:>11,d} | {max_v:>11,d} | {avg_v:>11,.1f}"
        )

    print("=" * 76 + "\n")


def generate_scenario_plots(scenarios: Dict[str, pd.DataFrame]) -> None:
    """Generate and save visual plots for each scenario and a comparison plot.

    Args:
        scenarios: Dictionary of scenario DataFrames.
    """
    colors = {
        "normal": "#1f77b4",
        "sudden_spike": "#d62728",
        "peak_event": "#2ca02c",
        "event_ending": "#ff7f0e",
    }

    # 1. Individual plots for each scenario
    for key, df in scenarios.items():
        scenario_name = df["scenario"].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

        # Plot viewer count curve
        ax.plot(
            df["minute"],
            df["viewers"],
            color=colors.get(key, "#333333"),
            linewidth=2,
            label="Concurrent Viewers",
        )

        # Highlight event active periods
        active_mask = df["event_active"].to_numpy()
        if active_mask.any():
            ax.fill_between(
                df["minute"],
                0,
                df["viewers"].max() * 1.05,
                where=active_mask,
                color="red",
                alpha=0.15,
                label="Event Active Period",
            )

        ax.set_title(
            f"Workload Simulation: {scenario_name}", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Simulation Time (minutes)", fontsize=11)
        ax.set_ylabel("Concurrent Viewers", fontsize=11)
        ax.set_xlim(0, len(df) - 1)
        ax.set_ylim(0, df["viewers"].max() * 1.1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9)

        plot_path = DATA_PROCESSED_DIR / f"{key}_workload.png"
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    # 2. Combined comparison plot (4-panel figure)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=120, sharex=True)
    scenario_items = list(scenarios.items())

    for idx, (key, df) in enumerate(scenario_items):
        ax = axes[idx // 2, idx % 2]
        scenario_name = df["scenario"].iloc[0]

        ax.plot(
            df["minute"],
            df["viewers"],
            color=colors.get(key, "#333333"),
            linewidth=2,
            label="Viewers",
        )

        active_mask = df["event_active"].to_numpy()
        if active_mask.any():
            ax.fill_between(
                df["minute"],
                0,
                df["viewers"].max() * 1.08,
                where=active_mask,
                color="red",
                alpha=0.15,
                label="Event Active",
            )

        ax.set_title(scenario_name, fontsize=11, fontweight="bold")
        ax.set_ylabel("Viewers", fontsize=10)
        ax.set_xlim(0, len(df) - 1)
        ax.set_ylim(0, df["viewers"].max() * 1.1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

        if idx >= 2:
            ax.set_xlabel("Simulation Time (minutes)", fontsize=10)

    fig.suptitle(
        "Stage A: Live Event Broadcasting Synthetic Workload Scenarios",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    comparison_path = DATA_PROCESSED_DIR / "all_scenarios_comparison.png"
    fig.savefig(comparison_path)
    plt.close(fig)


def main() -> None:
    """Run the Stage A workload generation pipeline."""
    print("Initializing Stage A: Synthetic Workload Generator...")
    ensure_directories()

    # Step 1: Generate all four scenarios
    print(
        f"Generating scenarios (duration={DEFAULT_DURATION_MINUTES} min, seed={DEFAULT_RANDOM_SEED})..."
    )
    scenarios = generate_all_scenarios(
        duration_minutes=DEFAULT_DURATION_MINUTES,
        random_seed=DEFAULT_RANDOM_SEED,
    )

    # Step 2 & 3: Save individual and combined CSV files
    print("Saving datasets to data/raw/...")
    saved_csvs = save_scenarios_to_csv(scenarios)
    for name, path in saved_csvs.items():
        print(f"  - Saved {name} dataset: {path.name}")

    # Step 4: Print summary table
    print_workload_summaries(scenarios)

    # Step 5: Generate and save workload visualization plots
    print("Generating workload plots under data/processed/...")
    generate_scenario_plots(scenarios)
    print("  - Saved 4 individual scenario plots.")
    print("  - Saved all_scenarios_comparison.png comparison overview.")

    print("\nStage A workload generation completed successfully!\n")


if __name__ == "__main__":
    main()
