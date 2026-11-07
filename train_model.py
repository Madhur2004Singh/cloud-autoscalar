"""Stage B Training Pipeline: Train and Evaluate Workload Prediction Engine.

This script executes the complete Stage B workflow:
1. Loads raw synthetic workload scenarios generated in Stage A.
2. Extracts chronological lag and rolling features.
3. Applies a purged chronological split (80% train, 20% test per scenario with a 5-minute purge gap).
4. Audits and verifies the purge condition: max(train target) < min(test feature).
5. Trains a RandomForestRegressor model on the purged training set.
6. Evaluates model predictions against a Persistence Baseline (overall and per scenario).
7. Executes a separate Sudden-Spike Stress Evaluation around the spike onset and peak.
8. Saves the trained model artifact to models/random_forest_model.joblib.
9. Saves evaluation test predictions to data/processed/prediction_results.csv.
10. Generates evaluation visualization figures under data/processed/.
"""

from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.predictor import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_HORIZON,
    FEATURE_COLUMNS,
    engineer_features,
    prepare_training_data,
    save_model,
    train_random_forest,
)

# Base directory definitions
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
MODELS_DIR: Path = BASE_DIR / "models"


def load_raw_scenarios() -> Dict[str, pd.DataFrame]:
    """Load individual Stage A CSV scenarios from data/raw/.

    Returns:
        Dictionary mapping scenario keys to raw DataFrames.

    Raises:
        FileNotFoundError: If any scenario CSV file is missing.
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


def audit_purged_split(
    meta_train: pd.DataFrame,
    meta_test: pd.DataFrame,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> bool:
    """Audit and verify the purged chronological split across all scenarios.

    Prints feature and target minute/timestamp ranges for training and testing,
    and explicitly verifies that max(train target) < min(test feature).

    Args:
        meta_train: Metadata DataFrame for training samples.
        meta_test: Metadata DataFrame for testing samples.
        prediction_horizon: Forecast horizon in minutes.

    Returns:
        True if all scenarios strictly satisfy the purge condition, False otherwise.
    """
    print("\n" + "=" * 90)
    print(f"CHRONOLOGICAL PURGED SPLIT AUDIT (Horizon = {prediction_horizon} minutes, Purge Gap = {prediction_horizon} min)")
    print("=" * 90)
    print(
        f"{'Scenario Name':<25} | {'Train Feature Min':>17} | {'Train Target Min':>16} | {'Test Feature Min':>16} | {'Purge Condition':>15}"
    )
    print("-" * 90)

    all_passed = True
    scenarios = meta_test["scenario"].unique()

    for sc_name in scenarios:
        train_sub = meta_train[meta_train["scenario"] == sc_name]
        test_sub = meta_test[meta_test["scenario"] == sc_name]

        train_feat_min = f"[{train_sub['minute'].min()}, {train_sub['minute'].max()}]"
        train_targ_min = f"[{train_sub['target_minute'].min()}, {train_sub['target_minute'].max()}]"
        test_feat_min = f"[{test_sub['minute'].min()}, {test_sub['minute'].max()}]"

        max_train_target = train_sub["target_minute"].max()
        min_test_feature = test_sub["minute"].min()

        condition_met = max_train_target < min_test_feature
        if not condition_met:
            all_passed = False

        status = "PASSED" if condition_met else "FAILED"
        print(
            f"{sc_name:<25} | {train_feat_min:>17} | {train_targ_min:>16} | {test_feat_min:>16} | {status:>15}"
        )

    print("-" * 90)
    print("TIMESTAMP RANGES:")
    for sc_name in scenarios:
        train_sub = meta_train[meta_train["scenario"] == sc_name]
        test_sub = meta_test[meta_test["scenario"] == sc_name]

        if "timestamp" in train_sub.columns and "target_timestamp" in train_sub.columns:
            tr_t_start = str(train_sub["timestamp"].min()).split()[1]
            tr_t_end = str(train_sub["timestamp"].max()).split()[1]
            tr_tgt_end = str(train_sub["target_timestamp"].max()).split()[1]
            te_t_start = str(test_sub["timestamp"].min()).split()[1]
            te_t_end = str(test_sub["timestamp"].max()).split()[1]

            print(
                f"  {sc_name:<25}: Train Feat [{tr_t_start}-{tr_t_end}], "
                f"Train Target Max: {tr_tgt_end} < Test Feat Min: {te_t_start} -> Test Feat [{te_t_start}-{te_t_end}]"
            )

    print("=" * 90)
    return all_passed


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    baseline_pred: pd.Series,
) -> Tuple[float, float, float, float]:
    """Compute MAE and RMSE metrics for Random Forest and Baseline.

    Returns:
        Tuple of (rf_mae, rf_rmse, base_mae, base_rmse).
    """
    rf_mae = float(mean_absolute_error(y_true, y_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    base_mae = float(mean_absolute_error(y_true, baseline_pred))
    base_rmse = float(np.sqrt(mean_squared_error(y_true, baseline_pred)))
    return rf_mae, rf_rmse, base_mae, base_rmse


def run_sudden_spike_stress_evaluation(
    raw_scenarios: Dict[str, pd.DataFrame],
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> Dict[str, Any]:
    """Execute a dedicated chronological stress evaluation for the Sudden-Spike scenario.

    Evaluates predictions around the actual spike window (minutes 45 to 70, target 50 to 75).
    To strictly prevent target leakage, the evaluation model is trained ONLY on:
    - Pre-spike Sudden Spike observations up to minute 39 (targets <= 44).
    - Other scenarios' training portions (minutes <= 87, targets <= 92).

    No evaluation targets appear in the training set.

    Args:
        raw_scenarios: Dictionary of raw scenario DataFrames.
        prediction_horizon: Forecasting horizon in minutes.

    Returns:
        Dictionary of stress evaluation results and metrics.
    """
    df_spike = raw_scenarios["sudden_spike"]
    feat_spike = engineer_features(
        df_spike, prediction_horizon=prediction_horizon, include_target=True
    ).dropna(subset=FEATURE_COLUMNS + ["target"])

    # Stress window: Feature minutes 45 to 70 (forecasting targets at minutes 50 to 75)
    stress_mask = (feat_spike["minute"] >= 45) & (feat_spike["minute"] <= 70)
    stress_test = feat_spike[stress_mask].copy()

    # Pre-spike training partition: feature minutes <= 39 -> target minutes <= 44
    # Purge gap of 5 minutes: max training target (44) < min stress feature (45)
    spike_train_pre = feat_spike[feat_spike["minute"] <= (45 - prediction_horizon - 1)].copy()

    # Other scenarios training data (minutes <= 87)
    other_train_list = [spike_train_pre]
    for key in ["normal", "peak_event", "event_ending"]:
        feat_other = engineer_features(
            raw_scenarios[key], prediction_horizon=prediction_horizon, include_target=True
        ).dropna(subset=FEATURE_COLUMNS + ["target"])
        # Use chronological training partition (minutes <= 87)
        other_train_list.append(feat_other[feat_other["minute"] <= 87].copy())

    train_combined = pd.concat(other_train_list, ignore_index=True)

    # Train dedicated stress-eval model without any evaluation target overlap
    rf_stress = RandomForestRegressor(
        n_estimators=200, random_state=42, min_samples_leaf=2, n_jobs=-1
    )
    rf_stress.fit(train_combined[FEATURE_COLUMNS], train_combined["target"])

    raw_preds = rf_stress.predict(stress_test[FEATURE_COLUMNS])
    rf_preds = np.maximum(0, np.round(raw_preds)).astype(int)
    actuals = stress_test["target"].values.astype(int)
    baseline_preds = stress_test["viewers"].values.astype(int)

    mae_rf, rmse_rf, mae_base, rmse_base = evaluate_predictions(
        pd.Series(actuals), rf_preds, pd.Series(baseline_preds)
    )

    return {
        "minutes": stress_test["minute"].values,
        "viewers": stress_test["viewers"].values,
        "actuals": actuals,
        "rf_preds": rf_preds,
        "baseline_preds": baseline_preds,
        "rf_mae": mae_rf,
        "rf_rmse": rmse_rf,
        "base_mae": mae_base,
        "base_rmse": rmse_base,
        "n_samples": len(stress_test),
    }


def plot_visualizations(
    test_results: pd.DataFrame,
    feature_importances: pd.Series,
) -> Dict[str, Path]:
    """Generate and save Stage B evaluation plots under data/processed/.

    Args:
        test_results: DataFrame containing actual, predicted, minute, and scenario.
        feature_importances: Series of feature importances sorted by value.

    Returns:
        Dictionary of plot names to saved paths.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    saved_plots: Dict[str, Path] = {}

    # Plot A: Actual vs Predicted across 4 separate scenario subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    target_scenarios = [
        "Normal Traffic",
        "Sudden Traffic Spike",
        "Peak Broadcasting Event",
        "Event Ending",
    ]

    for idx, sc_name in enumerate(target_scenarios):
        ax = axes[idx // 2, idx % 2]
        sc_data = test_results[test_results["scenario"] == sc_name]

        if not sc_data.empty:
            ax.plot(
                sc_data["minute"],
                sc_data["actual_viewers"],
                label="Actual Viewers (t + 5)",
                color="#1f77b4",
                linewidth=2.2,
            )
            ax.plot(
                sc_data["minute"],
                sc_data["predicted_viewers"],
                label="Random Forest Forecast (t + 5)",
                color="#d62728",
                linestyle="--",
                linewidth=2.0,
            )
            ax.plot(
                sc_data["minute"],
                sc_data["baseline_viewers"],
                label="Persistence Baseline (t)",
                color="#7f7f7f",
                linestyle=":",
                alpha=0.85,
                linewidth=1.6,
            )

        ax.set_title(f"Scenario: {sc_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Simulation Minute (t)", fontsize=10)
        ax.set_ylabel("Concurrent Viewers", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Actual vs Predicted Workload Across Independent Broadcasting Scenarios (Test Horizon = 5 min)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    path_a = DATA_PROCESSED_DIR / "actual_vs_predicted.png"
    fig.savefig(path_a)
    plt.close(fig)
    saved_plots["actual_vs_predicted"] = path_a

    # Plot B: Scenario-wise Prediction Comparison (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=120)
    unique_scenarios = test_results["scenario"].unique()

    for idx, sc_name in enumerate(unique_scenarios):
        ax = axes[idx // 2, idx % 2]
        sc_data = test_results[test_results["scenario"] == sc_name]

        ax.plot(
            sc_data["minute"],
            sc_data["actual_viewers"],
            label="Actual (t + 5)",
            color="#1f77b4",
            linewidth=2,
        )
        ax.plot(
            sc_data["minute"],
            sc_data["predicted_viewers"],
            label="RF Predicted",
            color="#d62728",
            linestyle="--",
            linewidth=2,
        )
        ax.plot(
            sc_data["minute"],
            sc_data["baseline_viewers"],
            label="Persistence",
            color="#7f7f7f",
            linestyle=":",
            alpha=0.8,
        )
        ax.set_title(sc_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Simulation Minute", fontsize=10)
        ax.set_ylabel("Viewers", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=9, framealpha=0.85)

    fig.suptitle(
        "Stage B: Workload Forecast Comparison Across Broadcasting Scenarios (Purged Split)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    path_b = DATA_PROCESSED_DIR / "scenario_prediction_comparison.png"
    fig.savefig(path_b)
    plt.close(fig)
    saved_plots["scenario_comparison"] = path_b

    # Plot C: Feature Importance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    sorted_features = feature_importances.sort_values(ascending=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_features)))

    ax.barh(sorted_features.index, sorted_features.values, color=colors, edgecolor="#2b5c8f")
    ax.set_title(
        "Random Forest Feature Importances (5-Minute Horizon)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Relative Importance (Gini Impurity Reduction)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    for i, v in enumerate(sorted_features.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    path_c = DATA_PROCESSED_DIR / "feature_importance.png"
    fig.savefig(path_c)
    plt.close(fig)
    saved_plots["feature_importance"] = path_c

    return saved_plots


def main() -> None:
    """Execute the Stage B model training and evaluation pipeline."""
    print("\n" + "=" * 90)
    print("STAGE B: ML-BASED WORKLOAD PREDICTION ENGINE AUDIT & TRAINING")
    print("=" * 90)

    # 1. Load Stage A raw scenario data
    print("Loading Stage A scenario datasets from data/raw/...")
    raw_scenarios = load_raw_scenarios()
    total_raw_rows = sum(len(df) for df in raw_scenarios.values())
    print(f"  - Loaded {len(raw_scenarios)} scenarios ({total_raw_rows} total observations).")

    # 2. Engineer features and apply purged chronological split
    print(f"Engineering features with {DEFAULT_PREDICTION_HORIZON}-minute horizon and purged split...")
    X_train, y_train, X_test, y_test, meta_train, meta_test = prepare_training_data(
        raw_scenarios,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        train_ratio=0.8,
    )

    # 3. Audit purged chronological split
    purge_ok = audit_purged_split(meta_train, meta_test, prediction_horizon=DEFAULT_PREDICTION_HORIZON)
    if not purge_ok:
        raise RuntimeError("Purged split audit failed: target timestamps overlap with test feature timestamps!")

    print(f"  - Total Training Samples (Purged): {len(X_train)} (83 per scenario)")
    print(f"  - Total Purged Samples (Embargo):   20 (5 per scenario)")
    print(f"  - Total Testing Samples:           {len(X_test)} (22 per scenario)")

    # 4. Train Random Forest model
    print("Training RandomForestRegressor (n_estimators=200, min_samples_leaf=2)...")
    rf_model = train_random_forest(
        X_train,
        y_train,
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    # 5. Predict on Test Set
    raw_rf_preds = rf_model.predict(X_test)
    rf_preds = np.maximum(0, np.round(raw_rf_preds)).astype(int)
    baseline_preds = X_test["viewers"].to_numpy().astype(int)

    # 6. Evaluate Overall Metrics
    rf_mae, rf_rmse, base_mae, base_rmse = evaluate_predictions(
        y_test, rf_preds, pd.Series(baseline_preds, index=y_test.index)
    )

    # Compile test results DataFrame
    test_results = pd.DataFrame(
        {
            "minute": meta_test["minute"].values,
            "scenario": meta_test["scenario"].values,
            "actual_viewers": y_test.values.astype(int),
            "predicted_viewers": rf_preds,
            "baseline_viewers": baseline_preds,
            "error_rf": rf_preds - y_test.values.astype(int),
            "error_baseline": baseline_preds - y_test.values.astype(int),
        }
    )

    # 7. Save Evaluation Results CSV
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results_csv_path = DATA_PROCESSED_DIR / "prediction_results.csv"
    test_results.to_csv(results_csv_path, index=False)
    print(f"  - Saved test predictions to: {results_csv_path.name}")

    # 8. Save Model Artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved_model_path = save_model(
        rf_model,
        filepath=DEFAULT_MODEL_PATH,
        prediction_horizon=DEFAULT_PREDICTION_HORIZON,
        feature_names=FEATURE_COLUMNS,
    )
    print(f"  - Saved model artifact to: {saved_model_path.name}")

    # 9. Feature Importances
    importances = pd.Series(
        rf_model.feature_importances_, index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    # 10. Generate Visualizations
    print("Generating prediction visualizations under data/processed/...")
    plots = plot_visualizations(test_results, importances)
    for name, p in plots.items():
        print(f"  - Generated plot: {p.name}")

    # 11. Primary Holdout Summary Reporting
    print("\n" + "-" * 90)
    print("PRIMARY HOLDOUT EVALUATION METRICS (Purged Split, Horizon = 5 minutes)")
    print("-" * 90)
    print(f"{'Metric':<25} | {'Random Forest':>16} | {'Persistence Baseline':>22}")
    print("-" * 90)
    print(f"{'Overall MAE (viewers)':<25} | {rf_mae:>16,.1f} | {base_mae:>22,.1f}")
    print(f"{'Overall RMSE (viewers)':<25} | {rf_rmse:>16,.1f} | {base_rmse:>22,.1f}")
    print("-" * 90)

    print("\nSCENARIO-WISE METRIC BREAKDOWN (Primary Holdout):")
    print(
        f"{'Scenario Name':<28} | {'RF MAE':>10} | {'Base MAE':>10} | {'RF RMSE':>10} | {'Base RMSE':>10}"
    )
    print("-" * 90)

    for sc_name in test_results["scenario"].unique():
        sub = test_results[test_results["scenario"] == sc_name]
        sc_rf_mae, sc_rf_rmse, sc_base_mae, sc_base_rmse = evaluate_predictions(
            sub["actual_viewers"],
            sub["predicted_viewers"].to_numpy(),
            sub["baseline_viewers"],
        )
        print(
            f"{sc_name:<28} | {sc_rf_mae:>10,.1f} | {sc_base_mae:>10,.1f} | {sc_rf_rmse:>10,.1f} | {sc_base_rmse:>10,.1f}"
        )

    print("=" * 90)

    # 12. Separate Sudden-Spike Stress Evaluation
    print("\n" + "=" * 90)
    print("SUDDEN-SPIKE STRESS EVALUATION (Evaluating Spike Window: Minutes 45–70)")
    print("=" * 90)
    stress_results = run_sudden_spike_stress_evaluation(
        raw_scenarios, prediction_horizon=DEFAULT_PREDICTION_HORIZON
    )
    print(
        f"Evaluation Window:       Minutes 45 to 70 (Targets: Minutes 50 to 75, n={stress_results['n_samples']})\n"
        f"Training Constraint:     No evaluation targets in training set (Spike train <= min 39, targets <= min 44)\n"
        f"Random Forest Stress MAE:        {stress_results['rf_mae']:>10,.1f} viewers\n"
        f"Persistence Baseline Stress MAE: {stress_results['base_mae']:>10,.1f} viewers\n"
        f"Random Forest Stress RMSE:       {stress_results['rf_rmse']:>10,.1f} viewers\n"
        f"Persistence Baseline Stress RMSE:{stress_results['base_rmse']:>10,.1f} viewers"
    )
    print("-" * 90)
    print("STRESS WINDOW SAMPLE PREDICTIONS (Onset & Peak):")
    print(
        f"{'Minute':>6} | {'Current Viewers':>15} | {'Actual Target (t+5)':>19} | {'RF Forecast':>12} | {'Base Forecast':>13} | {'RF Error':>9} | {'Base Error':>10}"
    )
    print("-" * 90)
    for m, cv, act, rf_p, bs_p in list(
        zip(
            stress_results["minutes"],
            stress_results["viewers"],
            stress_results["actuals"],
            stress_results["rf_preds"],
            stress_results["baseline_preds"],
        )
    )[:10]:  # print first 10 steps of onset
        print(
            f"{m:>6d} | {cv:>15,d} | {act:>19,d} | {rf_p:>12,d} | {bs_p:>13,d} | {rf_p-act:>9,d} | {bs_p-act:>10,d}"
        )
    print("=" * 90)
    print("\nStage B pipeline audit and execution completed successfully!\n")


if __name__ == "__main__":
    main()
