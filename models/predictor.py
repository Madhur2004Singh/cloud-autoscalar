"""Machine Learning Workload Prediction Engine for Live Event Broadcasting.

This module provides feature engineering, model training, persistence, and a reusable
prediction API for forecasting future concurrent viewership during live broadcasts.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Project directory paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH: Path = BASE_DIR / "models" / "random_forest_model.joblib"

# Default forecasting settings
DEFAULT_PREDICTION_HORIZON: int = 5
MIN_HISTORICAL_OBSERVATIONS: int = 6

# Standardized feature column list
FEATURE_COLUMNS: List[str] = [
    "viewers",
    "viewers_lag_1",
    "viewers_lag_2",
    "viewers_lag_5",
    "viewers_rolling_mean_5",
    "viewers_rolling_std_5",
    "viewer_growth_rate",
    "minute",
    "event_active",
]


def engineer_features(
    df: pd.DataFrame,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    include_target: bool = True,
) -> pd.DataFrame:
    """Engineer historical features and target variable for a single scenario.

    Features generated (strictly using information up to time t):
    1. Current viewer count: viewers
    2. Viewer count 1 minute ago: viewers_lag_1
    3. Viewer count 2 minutes ago: viewers_lag_2
    4. Viewer count 5 minutes ago: viewers_lag_5
    5. Rolling average of viewers over past 5 minutes including current: viewers_rolling_mean_5
    6. Rolling standard deviation over past 5 minutes: viewers_rolling_std_5
    7. Recent viewer growth rate: viewer_growth_rate (viewers - viewers_lag_1)
    8. Current simulation minute: minute
    9. Current event_active indicator: event_active (as integer 0 or 1)

    Target:
    - viewers at time t + prediction_horizon

    Args:
        df: Input DataFrame containing 'minute', 'viewers', and optional 'event_active'.
        prediction_horizon: Forecasting horizon in minutes (default 5).
        include_target: Whether to compute the future target variable.

    Returns:
        DataFrame with engineered feature columns (and 'target' if include_target=True).

    Raises:
        ValueError: If required columns ('minute', 'viewers') are missing.
    """
    if "viewers" not in df.columns or "minute" not in df.columns:
        raise ValueError("Input DataFrame must contain 'minute' and 'viewers' columns.")

    data = df.copy()

    # Ensure chronological order within scenario
    data = data.sort_values(by="minute").reset_index(drop=True)

    # 1. Current viewer count
    # 'viewers' is already present

    # 2-4. Lag features
    data["viewers_lag_1"] = data["viewers"].shift(1)
    data["viewers_lag_2"] = data["viewers"].shift(2)
    data["viewers_lag_5"] = data["viewers"].shift(5)

    # 5-6. Rolling features over 5-minute window including current minute (t-4 to t)
    data["viewers_rolling_mean_5"] = (
        data["viewers"].rolling(window=5, min_periods=5).mean()
    )
    data["viewers_rolling_std_5"] = (
        data["viewers"].rolling(window=5, min_periods=5).std().fillna(0.0)
    )

    # 7. Growth rate from previous minute
    data["viewer_growth_rate"] = data["viewers"] - data["viewers_lag_1"]

    # 8. Minute is already present

    # 9. event_active indicator (cast boolean to integer, or 0 if missing)
    if "event_active" in data.columns:
        data["event_active"] = data["event_active"].astype(int)
    else:
        data["event_active"] = 0

    # Target variable (strictly for training/evaluation)
    if include_target:
        if prediction_horizon <= 0:
            raise ValueError(
                f"prediction_horizon must be positive, got {prediction_horizon}"
            )
        data["target"] = data["viewers"].shift(-prediction_horizon)
        data["target_minute"] = data["minute"] + prediction_horizon
        if "timestamp" in data.columns:
            data["target_timestamp"] = pd.to_datetime(data["timestamp"]) + pd.Timedelta(
                minutes=prediction_horizon
            )

    return data


def prepare_training_data(
    scenario_dfs: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    train_ratio: float = 0.8,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Prepare and split training and testing datasets across independent scenarios.

    Each scenario is processed independently so no historical or future values
    leak across scenario boundaries. A purged chronological split is applied within
    each scenario before combining training and testing sets.

    To eliminate target-to-test leakage, a purge gap of `prediction_horizon` minutes
    is enforced such that:
        max(training target minute) < min(test feature minute)
    and equivalently:
        max(training target timestamp) < min(test feature timestamp)

    Args:
        scenario_dfs: Collection or dictionary of raw scenario DataFrames.
        prediction_horizon: Lead time for the target forecast in minutes.
        train_ratio: Proportion of earlier observations allocated to training.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, meta_train, meta_test).

    Raises:
        ValueError: If train_ratio is outside (0, 1) or scenario_dfs is empty.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    dfs = (
        list(scenario_dfs.values())
        if isinstance(scenario_dfs, dict)
        else list(scenario_dfs)
    )
    if not dfs:
        raise ValueError("scenario_dfs must contain at least one scenario DataFrame.")

    train_feature_list: List[pd.DataFrame] = []
    test_feature_list: List[pd.DataFrame] = []

    for raw_df in dfs:
        # Process features and target strictly within this scenario
        feat_df = engineer_features(
            raw_df, prediction_horizon=prediction_horizon, include_target=True
        )

        # Drop boundary rows containing NaN in features or target
        clean_df = feat_df.dropna(subset=FEATURE_COLUMNS + ["target"]).copy()

        if len(clean_df) == 0:
            continue

        # Chronological test split per scenario
        n_obs = len(clean_df)
        n_train_raw = int(n_obs * train_ratio)

        scenario_test = clean_df.iloc[n_train_raw:].copy()
        test_start_min = int(scenario_test["minute"].min())

        # Purged split: Remove training observations whose target falls into or after test_start_min
        # Enforces: target_minute < test_start_min <=> (minute + prediction_horizon) < test_start_min
        purge_mask = (clean_df["minute"] + prediction_horizon) < test_start_min
        scenario_train = clean_df[purge_mask].copy()

        train_feature_list.append(scenario_train)
        test_feature_list.append(scenario_test)

    if not train_feature_list or not test_feature_list:
        raise ValueError("Failed to create non-empty train or test partitions.")

    all_train = pd.concat(train_feature_list, ignore_index=True)
    all_test = pd.concat(test_feature_list, ignore_index=True)

    X_train = all_train[FEATURE_COLUMNS].copy()
    y_train = all_train["target"].copy()

    X_test = all_test[FEATURE_COLUMNS].copy()
    y_test = all_test["target"].copy()

    # Meta DataFrames retain scenario, minute, target_minute, timestamp, and target_timestamp
    meta_cols = [
        c
        for c in [
            "minute",
            "target_minute",
            "timestamp",
            "target_timestamp",
            "scenario",
            "viewers",
        ]
        if c in all_train.columns
    ]
    meta_train = all_train[meta_cols].copy()
    meta_test = all_test[meta_cols].copy()

    return X_train, y_train, X_test, y_test, meta_train, meta_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
    min_samples_leaf: int = 2,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor model on the training feature matrix.

    Args:
        X_train: Training features DataFrame.
        y_train: Target viewer values Series.
        n_estimators: Number of trees in the forest (default 200).
        random_state: Random state for reproducibility (default 42).
        min_samples_leaf: Minimum samples required at leaf node (default 2).
        n_jobs: Number of parallel jobs (-1 for all cores).
        **kwargs: Additional hyperparameters for RandomForestRegressor.

    Returns:
        Fitted RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def save_model(
    model: RandomForestRegressor,
    filepath: Union[str, Path] = DEFAULT_MODEL_PATH,
    prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    feature_names: Optional[List[str]] = None,
) -> Path:
    """Save the trained model and associated metadata artifact to disk using Joblib.

    Args:
        model: Trained RandomForestRegressor.
        filepath: Destination file path for the joblib artifact.
        prediction_horizon: Prediction horizon used during training.
        feature_names: Ordered list of feature names.

    Returns:
        Path of the saved artifact.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": feature_names or FEATURE_COLUMNS,
        "prediction_horizon": prediction_horizon,
        "model_type": "RandomForestRegressor",
    }
    joblib.dump(artifact, path)
    return path


def load_model(
    filepath: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    """Load a trained model and metadata artifact from disk.

    Args:
        filepath: Path to the joblib artifact.

    Returns:
        Tuple of (model, metadata_dict).

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the file content is invalid.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    loaded = joblib.load(path)
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        metadata = {k: v for k, v in loaded.items() if k != "model"}
        return model, metadata
    elif isinstance(loaded, RandomForestRegressor):
        return loaded, {"feature_names": FEATURE_COLUMNS, "prediction_horizon": DEFAULT_PREDICTION_HORIZON}
    else:
        raise ValueError(f"Unrecognized model artifact format at: {path}")


# Global cache for the loaded model artifact to avoid redundant disk reads
_CACHED_MODEL: Optional[RandomForestRegressor] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None
_CACHED_PATH: Optional[Path] = None


def predict_next_viewers(
    history: pd.DataFrame,
    model_artifact_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> int:
    """Reusable Prediction API: Predict viewer count for time t + prediction_horizon.

    This function is designed for seamless integration with future cloud simulation
    modules. It validates the input history, engineers features using only past
    and current observations, and returns a single non-negative integer prediction.

    Expected input DataFrame columns:
    - 'minute': integer simulation minute.
    - 'viewers': concurrent viewer count.
    - 'event_active' (optional): boolean or integer indicating active event.

    Args:
        history: Historical telemetry DataFrame containing at least 6 consecutive observations.
        model_artifact_path: Path to the trained joblib model artifact.

    Returns:
        Non-negative integer representing predicted viewers at time t + prediction_horizon.

    Raises:
        ValueError: If history has insufficient rows or missing required columns.
        FileNotFoundError: If the model artifact cannot be found.
    """
    global _CACHED_MODEL, _CACHED_METADATA, _CACHED_PATH

    if not isinstance(history, pd.DataFrame):
        raise ValueError(f"history must be a pandas DataFrame, got {type(history)}")

    if len(history) < MIN_HISTORICAL_OBSERVATIONS:
        raise ValueError(
            f"Insufficient historical observations: received {len(history)} rows, "
            f"but at least {MIN_HISTORICAL_OBSERVATIONS} are required to compute lag and rolling features."
        )

    for col in ["minute", "viewers"]:
        if col not in history.columns:
            raise ValueError(f"history DataFrame must contain '{col}' column.")

    # Validate that historical minutes are strictly consecutive
    sorted_minutes = history["minute"].sort_values().to_numpy()
    if len(sorted_minutes) > 1 and not np.all(np.diff(sorted_minutes) == 1):
        raise ValueError(
            "Historical observations must have strictly consecutive minute values "
            "with a constant time step of 1 minute (no gaps or duplicates)."
        )

    # Load and cache model
    model_path = Path(model_artifact_path).resolve()
    if _CACHED_MODEL is None or _CACHED_PATH != model_path:
        _CACHED_MODEL, _CACHED_METADATA = load_model(model_path)
        _CACHED_PATH = model_path

    # Engineer features without target
    features_df = engineer_features(history, include_target=False)

    # Select the latest observation (time t)
    latest_features = features_df.iloc[[-1]][FEATURE_COLUMNS]

    if latest_features.isna().any().any():
        raise ValueError("Engineered features for the latest observation contain NaN values.")

    # Predict
    raw_pred = _CACHED_MODEL.predict(latest_features)[0]

    # Constrain to non-negative integer
    return int(max(0, round(float(raw_pred))))


class WorkloadPredictor:
    """Class wrapper providing object-oriented access to workload prediction."""

    def __init__(
        self,
        model: Optional[RandomForestRegressor] = None,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Initialize the WorkloadPredictor instance."""
        self.model_path = Path(model_path)
        self.prediction_horizon = prediction_horizon
        self.feature_names = feature_names or FEATURE_COLUMNS

        if model is not None:
            self.model = model
        elif self.model_path.is_file():
            self.model, meta = load_model(self.model_path)
            self.prediction_horizon = meta.get("prediction_horizon", self.prediction_horizon)
            self.feature_names = meta.get("feature_names", self.feature_names)
        else:
            self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> "WorkloadPredictor":
        """Fit the internal RandomForestRegressor model."""
        self.model = train_random_forest(X_train, y_train, **kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for a feature matrix."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        preds = self.model.predict(X[self.feature_names])
        return np.maximum(0, np.round(preds)).astype(int)

    def predict_next(self, history: pd.DataFrame) -> int:
        """Predict next viewers from historical DataFrame."""
        if self.model is None:
            raise RuntimeError("Model has not been trained or loaded yet.")
        return predict_next_viewers(history, self.model_path)

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Save the model to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        target_path = Path(filepath) if filepath else self.model_path
        return save_model(
            self.model,
            filepath=target_path,
            prediction_horizon=self.prediction_horizon,
            feature_names=self.feature_names,
        )
