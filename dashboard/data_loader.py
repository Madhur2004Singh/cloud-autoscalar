"""Data loading and aggregation utilities for the Streamlit dashboard."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# Base paths relative to project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"

SCENARIOS: List[str] = [
    "Normal Traffic",
    "Sudden Traffic Spike",
    "Peak Broadcasting Event",
    "Event Ending",
]

STRATEGIES: List[str] = ["Reactive", "Predictive"]

SCENARIO_FILE_MAP: Dict[str, str] = {
    "Normal Traffic": "normal_traffic.csv",
    "Sudden Traffic Spike": "sudden_spike_traffic.csv",
    "Peak Broadcasting Event": "peak_event_traffic.csv",
    "Event Ending": "event_ending_traffic.csv",
}


def get_project_root() -> Path:
    """Return the resolved project root directory path."""
    return PROJECT_ROOT


@st.cache_data
def load_workload_data(scenario: Optional[str] = None) -> pd.DataFrame:
    """Load raw workload scenario time-series data.

    Args:
        scenario: Optional scenario name filter.

    Returns:
        DataFrame containing workload records.

    Raises:
        FileNotFoundError: If the workload data CSV files are missing.
    """
    combined_path = DATA_RAW_DIR / "all_scenarios_combined.csv"
    if combined_path.exists():
        df = pd.read_csv(combined_path)
    else:
        # Fall back to individual scenario files if combined is missing
        dfs = []
        for name, fname in SCENARIO_FILE_MAP.items():
            fpath = DATA_RAW_DIR / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Workload dataset '{fname}' not found in {DATA_RAW_DIR}. "
                    "Run 'python main.py' to generate Stage A raw workload data."
                )
            dfs.append(pd.read_csv(fpath))
        df = pd.concat(dfs, ignore_index=True)

    if scenario:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {SCENARIOS}")
        df = df[df["scenario"] == scenario].reset_index(drop=True)

    return df


def compute_workload_summary_metrics(df_scenario: pd.DataFrame) -> Dict[str, Any]:
    """Compute aggregate viewer metrics for a workload scenario.

    Args:
        df_scenario: Scenario DataFrame containing 'viewers' column.

    Returns:
        Dictionary with peak, average, total, min, and duration metrics.
    """
    if df_scenario.empty or "viewers" not in df_scenario.columns:
        return {
            "peak_viewers": 0,
            "avg_viewers": 0.0,
            "total_viewers": 0,
            "min_viewers": 0,
            "duration_minutes": 0,
        }

    return {
        "peak_viewers": int(df_scenario["viewers"].max()),
        "avg_viewers": float(df_scenario["viewers"].mean()),
        "total_viewers": int(df_scenario["viewers"].sum()),
        "min_viewers": int(df_scenario["viewers"].min()),
        "duration_minutes": len(df_scenario),
    }


@st.cache_data
def load_prediction_data(scenario: Optional[str] = None) -> pd.DataFrame:
    """Load Stage B model prediction evaluation records.

    Args:
        scenario: Optional scenario name filter.

    Returns:
        DataFrame containing actual, predicted, and baseline viewer numbers.

    Raises:
        FileNotFoundError: If prediction_results.csv is missing.
    """
    path = DATA_PROCESSED_DIR / "prediction_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction results file not found at {path}. "
            "Run 'python train_model.py' to generate Stage B model predictions."
        )

    df = pd.read_csv(path)
    if scenario:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {SCENARIOS}")
        df = df[df["scenario"] == scenario].reset_index(drop=True)

    return df


def compute_prediction_metrics(df_pred: pd.DataFrame) -> Dict[str, float]:
    """Compute MAE and RMSE metrics for Random Forest and Baseline.

    Args:
        df_pred: DataFrame with actual_viewers, predicted_viewers, and baseline_viewers.

    Returns:
        Dictionary containing rf_mae, rf_rmse, base_mae, base_rmse, and improvement percentages.
    """
    if df_pred.empty:
        return {
            "rf_mae": 0.0,
            "rf_rmse": 0.0,
            "base_mae": 0.0,
            "base_rmse": 0.0,
            "mae_reduction_pct": 0.0,
            "rmse_reduction_pct": 0.0,
        }

    actual = df_pred["actual_viewers"].to_numpy(dtype=float)
    rf_pred = df_pred["predicted_viewers"].to_numpy(dtype=float)
    base_pred = df_pred["baseline_viewers"].to_numpy(dtype=float)

    rf_mae = float(np.mean(np.abs(actual - rf_pred)))
    rf_rmse = float(np.sqrt(np.mean((actual - rf_pred) ** 2)))
    base_mae = float(np.mean(np.abs(actual - base_pred)))
    base_rmse = float(np.sqrt(np.mean((actual - base_pred) ** 2)))

    mae_reduction = ((base_mae - rf_mae) / base_mae * 100.0) if base_mae > 0 else 0.0
    rmse_reduction = ((base_rmse - rf_rmse) / base_rmse * 100.0) if base_rmse > 0 else 0.0

    return {
        "rf_mae": rf_mae,
        "rf_rmse": rf_rmse,
        "base_mae": base_mae,
        "base_rmse": base_rmse,
        "mae_reduction_pct": mae_reduction,
        "rmse_reduction_pct": rmse_reduction,
    }


@st.cache_data
def load_autoscaling_summary() -> pd.DataFrame:
    """Load Stage D3 autoscaling comparison summary table.

    Returns:
        Summary DataFrame across all scenarios and strategies.

    Raises:
        FileNotFoundError: If autoscaling_comparison_summary.csv is missing.
    """
    path = DATA_PROCESSED_DIR / "autoscaling_comparison_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Autoscaling summary file not found at {path}. "
            "Run 'python run_autoscaling_comparison.py' to generate Stage D3 summary."
        )
    return pd.read_csv(path)


@st.cache_data
def load_autoscaling_metrics(
    scenario: Optional[str] = None, strategy: Optional[str] = None
) -> pd.DataFrame:
    """Load Stage D3 discrete-minute autoscaling comparison metrics.

    Args:
        scenario: Optional scenario name filter.
        strategy: Optional strategy filter ('Reactive' or 'Predictive').

    Returns:
        Filtered metrics DataFrame.

    Raises:
        FileNotFoundError: If autoscaling_comparison_metrics.csv is missing.
    """
    path = DATA_PROCESSED_DIR / "autoscaling_comparison_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Autoscaling metrics file not found at {path}. "
            "Run 'python run_autoscaling_comparison.py' to generate Stage D3 metrics."
        )

    df = pd.read_csv(path)
    if scenario:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario}'. Choose from {SCENARIOS}")
        df = df[df["scenario"] == scenario]

    if strategy:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")
        df = df[df["strategy"] == strategy]

    return df.reset_index(drop=True)


@st.cache_data
def load_autoscaling_decisions(
    scenario: Optional[str] = None, strategy: Optional[str] = None
) -> pd.DataFrame:
    """Load Stage D3 discrete scaling decision event logs.

    Args:
        scenario: Optional scenario filter.
        strategy: Optional strategy filter.

    Returns:
        Filtered decision logs DataFrame.

    Raises:
        FileNotFoundError: If autoscaling_decisions.csv is missing.
    """
    path = DATA_PROCESSED_DIR / "autoscaling_decisions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Autoscaling decisions file not found at {path}. "
            "Run 'python run_autoscaling_comparison.py' to generate Stage D3 decisions."
        )

    df = pd.read_csv(path)
    if scenario:
        df = df[df["scenario"] == scenario]
    if strategy:
        df = df[df["strategy"] == strategy]

    return df.reset_index(drop=True)


@st.cache_data
def load_infrastructure_metrics(scenario: Optional[str] = None) -> pd.DataFrame:
    """Load Stage C baseline static infrastructure simulation metrics.

    Args:
        scenario: Optional scenario name filter.

    Returns:
        DataFrame containing fixed-infrastructure metrics over time.

    Raises:
        FileNotFoundError: If infrastructure_metrics.csv is missing.
    """
    path = DATA_PROCESSED_DIR / "infrastructure_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Infrastructure metrics file not found at {path}. "
            "Run 'python run_infrastructure_simulation.py' to generate Stage C metrics."
        )

    df = pd.read_csv(path)
    if scenario:
        df = df[df["scenario"] == scenario]

    return df.reset_index(drop=True)


# Convenient alias
load_autoscaling_comparison_metrics = load_autoscaling_metrics
