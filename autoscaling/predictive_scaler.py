"""Predictive autoscaler implementation using Machine Learning workload forecasting.

This module provides `PredictiveScaler`, a proactive autoscaling strategy that
utilizes a trained Random Forest model to forecast concurrent viewer demand 5 minutes
ahead (t + 5) and adjusts cloud infrastructure capacity before traffic changes occur.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from autoscaling.base_scaler import BaseScaler, ScalerConfig, ScalingDecision
from models.predictor import (
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTION_HORIZON,
    MIN_HISTORICAL_OBSERVATIONS,
    load_model,
    predict_next_viewers,
)

if TYPE_CHECKING:
    from simulation.cloud_simulator import CloudSimulator


class PredictiveScaler(BaseScaler):
    """Predictive autoscaler driven by Random Forest 5-minute workload forecasts.

    Scaling Policy:
    1. Proactive Scale-Up:
       - Triggered when predicted demand (viewers at t + 5) requires more capacity than
         is available under target CPU utilization (default 70%).
       - Required servers: ceil(predicted_viewers / (capacity_per_server * target_fraction)).
       - Clamped between min_servers and max_servers.
       - Executed if desired servers > current active servers and cooldown has expired.
       - Resets consecutive low-demand counter.

    2. Conservative Scale-Down:
       - Uses predicted demand to compute projected CPU utilization:
         projected_cpu = (predicted_viewers / (active_servers * capacity_per_server)) * 100
       - Triggered when projected CPU < scale_down_cpu_threshold (default 30%) for
         scale_down_consecutive_intervals (default 3) consecutive intervals.
       - Reduces server pool conservatively by 1 server at a time.
       - Clamped by min_servers.
       - Executed if current servers > min_servers and cooldown has expired.
       - Resets consecutive low-demand counter after scaling.

    3. Neutral / No Action:
       - Retains current server count when projected CPU is within normal bounds.
       - Also returned when scaling is deferred due to active cooldown, limit boundaries,
         or insufficient historical observations.
       - Resets consecutive low-demand counter when projected CPU rises above threshold.
    """

    def __init__(
        self,
        config: Optional[ScalerConfig] = None,
        min_servers: Optional[int] = None,
        max_servers: Optional[int] = None,
        target_cpu_utilization: Optional[float] = None,
        scale_down_cpu_threshold: Optional[float] = None,
        cooldown_intervals: Optional[int] = None,
        scale_down_consecutive_intervals: Optional[int] = None,
        capacity_per_server: Optional[int] = None,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        predictor: Optional[Callable[[pd.DataFrame], int]] = None,
        prediction_horizon: int = DEFAULT_PREDICTION_HORIZON,
    ) -> None:
        """Initialize predictive autoscaler with infrastructure and model settings.

        Args:
            config: Optional ScalerConfig instance.
            min_servers: Minimum server limit (default 2).
            max_servers: Maximum server limit (default 30).
            target_cpu_utilization: Target upper CPU threshold percentage (default 70.0%).
            scale_down_cpu_threshold: Target lower CPU threshold percentage (default 30.0%).
            cooldown_intervals: Inactive intervals required after scaling (default 2).
            scale_down_consecutive_intervals: Consecutive low-demand intervals for scale-down (default 3).
            capacity_per_server: Server capacity in viewers (default 1000).
            model_path: Path to the trained joblib model artifact.
            predictor: Optional callable `(pd.DataFrame) -> int` for custom or mocked forecasts.
            prediction_horizon: Lead forecasting time in minutes (default 5).
        """
        super().__init__(
            config=config,
            min_servers=min_servers,
            max_servers=max_servers,
            target_cpu_utilization=target_cpu_utilization,
            scale_down_cpu_threshold=scale_down_cpu_threshold,
            cooldown_intervals=cooldown_intervals,
            scale_down_consecutive_intervals=scale_down_consecutive_intervals,
            capacity_per_server=capacity_per_server,
        )
        if prediction_horizon != DEFAULT_PREDICTION_HORIZON:
            raise ValueError(
                f"Unsupported prediction_horizon {prediction_horizon}. The trained model "
                f"strictly forecasts {DEFAULT_PREDICTION_HORIZON} minutes ahead."
            )
        self.prediction_horizon: int = DEFAULT_PREDICTION_HORIZON
        self.consecutive_low_demand_count: int = 0

        # Model and predictor setup
        self.model_path: Path = Path(model_path).resolve()
        if predictor is not None:
            self.predictor: Callable[[pd.DataFrame], int] = predictor
        else:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model artifact not found at: {self.model_path}")
            # predict_next_viewers internally caches the model artifact to avoid disk thrashing
            self.predictor = lambda hist: predict_next_viewers(
                hist, model_artifact_path=self.model_path
            )

    @property
    def consecutive_low_cpu_count(self) -> int:
        """Alias for consecutive_low_demand_count matching reactive scaler interface."""
        return self.consecutive_low_demand_count

    @consecutive_low_cpu_count.setter
    def consecutive_low_cpu_count(self, value: int) -> None:
        self.consecutive_low_demand_count = int(value)

    def reset(self) -> None:
        """Reset operational tracking counters and low-demand hysteresis state."""
        super().reset()
        self.consecutive_low_demand_count = 0

    def predict(self, viewer_history: pd.DataFrame) -> int:
        """Generate a 5-minute ahead viewer forecast from historical observations.

        Args:
            viewer_history: Historical DataFrame containing at least 6 consecutive observations.

        Returns:
            Non-negative integer forecasted viewer count at t + 5.
        """
        return self.predictor(viewer_history)

    def make_decision(
        self,
        current_interval: int,
        current_viewers: int,
        current_active_servers: int,
        current_cpu_utilization: float,
        simulator: Optional["CloudSimulator"] = None,
        viewer_history: Optional[Union[pd.DataFrame, List[int], pd.Series]] = None,
    ) -> ScalingDecision:
        """Evaluate predicted future workload and execute/return a proactive scaling decision.

        Args:
            current_interval: Current simulation interval/minute (must be >= 0).
            current_viewers: Incoming concurrent viewer count at interval t.
            current_active_servers: Active server count currently handling traffic.
            current_cpu_utilization: Current observed average CPU utilization percentage.
            simulator: Optional CloudSimulator instance to perform direct scaling operations.
            viewer_history: Historical viewer observations up to and including interval t.
                Can be a DataFrame with 'minute' and 'viewers', or a sequence of viewer counts.

        Returns:
            Structured ScalingDecision record with action, server counts, reason, and predicted_viewers.

        Raises:
            ValueError: If input numeric ranges are invalid, or if viewer_history mismatches current state.
        """
        if current_interval < 0:
            raise ValueError(f"current_interval must be non-negative, got {current_interval}")
        if current_viewers < 0:
            raise ValueError(f"current_viewers must be non-negative, got {current_viewers}")
        if current_active_servers <= 0:
            raise ValueError(
                f"current_active_servers must be a positive integer, got {current_active_servers}"
            )
        if current_cpu_utilization < 0.0:
            raise ValueError(
                f"current_cpu_utilization cannot be negative, got {current_cpu_utilization}"
            )

        # Synchronize active count and server capacity with simulator instance if provided
        active_count = (
            len(simulator.get_active_servers())
            if simulator is not None
            else current_active_servers
        )
        cap = (
            simulator.config.capacity_per_server
            if simulator is not None
            else self.capacity_per_server
        )

        # -------------------------------------------------------------
        # 1. PROCESS AND VALIDATE HISTORICAL OBSERVATIONS
        # -------------------------------------------------------------
        history_df: Optional[pd.DataFrame] = None

        if viewer_history is not None:
            if isinstance(viewer_history, pd.DataFrame):
                data = viewer_history.copy()

                if "viewers" not in data.columns:
                    raise ValueError("viewer_history DataFrame must contain 'viewers' column.")

                if len(data) > 0:
                    if "minute" in data.columns:
                        # Validate minute values are non-null integers
                        if data["minute"].isna().any():
                            raise ValueError("viewer_history contains missing/NaN minute values.")
                        try:
                            minute_vals = data["minute"].to_numpy()
                            if not np.all(np.equal(np.mod(minute_vals, 1), 0)):
                                raise ValueError("viewer_history contains non-integer minute values.")
                            data["minute"] = data["minute"].astype(int)
                        except (TypeError, ValueError) as err:
                            raise ValueError(f"viewer_history contains invalid minute values: {err}") from err

                        # Reject duplicate minute values
                        if data["minute"].duplicated().any():
                            raise ValueError("viewer_history contains duplicate minute values.")

                        # Reject any minute greater than current_interval (including in unsorted rows)
                        if (data["minute"] > current_interval).any():
                            future_minutes = data.loc[data["minute"] > current_interval, "minute"].tolist()
                            raise ValueError(
                                f"viewer_history contains future minute values {future_minutes} "
                                f"exceeding current_interval ({current_interval}). Future data is prohibited."
                            )

                        # Reject negative minute values
                        if (data["minute"] < 0).any():
                            neg_minutes = data.loc[data["minute"] < 0, "minute"].tolist()
                            raise ValueError(
                                f"viewer_history contains negative minute values: {neg_minutes}."
                            )

                        # Sort chronologically by minute
                        data = data.sort_values(by="minute").reset_index(drop=True)

                        # Require the latest minute to equal current_interval
                        latest_minute = int(data["minute"].iloc[-1])
                        if latest_minute != current_interval:
                            raise ValueError(
                                f"Latest minute in viewer_history ({latest_minute}) must match "
                                f"current_interval ({current_interval})."
                            )

                        # Require observations to be strictly consecutive with no missing minute
                        if len(data) > 1:
                            minute_diffs = np.diff(data["minute"].to_numpy())
                            if not np.all(minute_diffs == 1):
                                raise ValueError(
                                    "Historical observations must have strictly consecutive minute values "
                                    "with a step of 1 minute (no gaps or missing minutes)."
                                )

                        # Verify latest sorted viewer count equals current_viewers
                        latest_viewers = int(data["viewers"].iloc[-1])
                        if latest_viewers != current_viewers:
                            raise ValueError(
                                f"Latest observation in viewer_history ({latest_viewers}) must match "
                                f"current_viewers ({current_viewers})."
                            )

                        history_df = data
                    else:
                        start_minute = current_interval - len(data) + 1
                        if start_minute < 0:
                            raise ValueError(
                                f"viewer_history of length {len(data)} at interval {current_interval} "
                                f"implies a negative starting minute ({start_minute}). Observations cannot precede minute 0."
                            )
                        latest_viewers = int(data["viewers"].iloc[-1])
                        if latest_viewers != current_viewers:
                            raise ValueError(
                                f"Latest observation in viewer_history ({latest_viewers}) must match "
                                f"current_viewers ({current_viewers})."
                            )
                        data["minute"] = list(range(start_minute, current_interval + 1))
                        history_df = data

            elif isinstance(viewer_history, (list, tuple, pd.Series, np.ndarray)):
                if len(viewer_history) > 0:
                    start_minute = current_interval - len(viewer_history) + 1
                    if start_minute < 0:
                        raise ValueError(
                            f"viewer_history of length {len(viewer_history)} at interval {current_interval} "
                            f"implies a negative starting minute ({start_minute}). Observations cannot precede minute 0."
                        )
                    latest_viewers = int(viewer_history[-1])
                    if latest_viewers != current_viewers:
                        raise ValueError(
                            f"Latest observation in viewer_history ({latest_viewers}) must match "
                            f"current_viewers ({current_viewers})."
                        )
                    minutes = list(range(start_minute, current_interval + 1))
                    history_df = pd.DataFrame(
                        {"minute": minutes, "viewers": [int(v) for v in viewer_history]}
                    )
            else:
                raise ValueError(
                    f"Unsupported viewer_history type '{type(viewer_history).__name__}'. "
                    f"Must be a DataFrame, Series, or sequence of integers."
                )

        # -------------------------------------------------------------
        # 2. INSUFFICIENT HISTORY CHECK
        # -------------------------------------------------------------
        # Feature engineering requires at least 6 consecutive observations for lag-5 and rolling-5
        if history_df is None or len(history_df) < MIN_HISTORICAL_OBSERVATIONS:
            obs_count = len(history_df) if history_df is not None else 0
            return ScalingDecision(
                action="no_action",
                previous_servers=active_count,
                desired_servers=active_count,
                actual_servers=active_count,
                reason=(
                    f"Insufficient historical observations ({obs_count}/{MIN_HISTORICAL_OBSERVATIONS}); "
                    f"maintaining {active_count} active servers until history reaches {MIN_HISTORICAL_OBSERVATIONS} intervals."
                ),
                predicted_viewers=None,
            )

        # -------------------------------------------------------------
        # 3. FORECAST FUTURE DEMAND (t + prediction_horizon)
        # -------------------------------------------------------------
        raw_prediction = self.predictor(history_df)
        predicted_viewers = max(0, int(round(float(raw_prediction))))

        # Calculate required servers from predicted demand
        raw_desired = self.calculate_required_servers(
            viewers=predicted_viewers,
            capacity_per_server=cap,
            target_cpu_utilization=self.target_cpu_utilization,
            clamp=False,
        )
        desired_servers = self.clamp_server_count(raw_desired)

        # Compute projected CPU utilization under current active server pool
        total_capacity = active_count * cap
        predicted_cpu = (
            (float(predicted_viewers) / float(total_capacity)) * 100.0
            if total_capacity > 0
            else 0.0
        )

        # -------------------------------------------------------------
        # 4. PROACTIVE SCALE-UP EVALUATION
        # -------------------------------------------------------------
        if desired_servers > active_count:
            # High future demand predicted: reset low-demand counter
            self.consecutive_low_demand_count = 0

            # Rule: Respect cooldown
            if not self.is_cooldown_expired(current_interval):
                return ScalingDecision(
                    action="no_action",
                    previous_servers=active_count,
                    desired_servers=desired_servers,
                    actual_servers=active_count,
                    reason=(
                        f"Predicted demand ({predicted_viewers} viewers at t+{self.prediction_horizon}) "
                        f"requires {desired_servers} servers, but predictive scale-up is deferred due to "
                        f"active cooldown (last scaled at interval {self.last_scaling_interval}, "
                        f"cooldown={self.cooldown_intervals} intervals)."
                    ),
                    predicted_viewers=predicted_viewers,
                )

            # Cooldown expired: execute proactive scale-up
            if simulator is not None:
                servers_to_add = desired_servers - active_count
                for _ in range(servers_to_add):
                    simulator.add_server()
                actual_servers = len(simulator.get_active_servers())
                self.record_scaling_action("scale_up", current_interval)
                reason = (
                    f"Predicted demand ({predicted_viewers} viewers at t+{self.prediction_horizon}) "
                    f"exceeded capacity. Scaled up proactively from {active_count} to {actual_servers} servers."
                )
            else:
                actual_servers = active_count
                reason = (
                    f"Predicted demand ({predicted_viewers} viewers at t+{self.prediction_horizon}) "
                    f"exceeds capacity. Scale-up requested to {desired_servers} servers (no simulator provided; "
                    f"infrastructure unchanged at {active_count} servers)."
                )

            self.consecutive_low_demand_count = 0

            return ScalingDecision(
                action="scale_up",
                previous_servers=active_count,
                desired_servers=desired_servers,
                actual_servers=actual_servers,
                reason=reason,
                predicted_viewers=predicted_viewers,
            )

        # -------------------------------------------------------------
        # 5. CONSERVATIVE SCALE-DOWN EVALUATION
        # -------------------------------------------------------------
        elif predicted_cpu < self.scale_down_cpu_threshold:
            self.consecutive_low_demand_count += 1

            # Rule: Wait for scale_down_consecutive_intervals (default 3)
            if self.consecutive_low_demand_count < self.scale_down_consecutive_intervals:
                return ScalingDecision(
                    action="no_action",
                    previous_servers=active_count,
                    desired_servers=active_count,
                    actual_servers=active_count,
                    reason=(
                        f"Predicted CPU ({predicted_cpu:.1f}%) below scale-down threshold "
                        f"({self.scale_down_cpu_threshold}%) for {self.consecutive_low_demand_count}/"
                        f"{self.scale_down_consecutive_intervals} consecutive intervals (predicted "
                        f"{predicted_viewers} viewers at t+{self.prediction_horizon})."
                    ),
                    predicted_viewers=predicted_viewers,
                )

            # Conservative reduction target (1 server)
            conservative_desired = max(self.min_servers, active_count - 1)

            # Rule: Never scale below configured minimum servers
            if active_count <= self.min_servers:
                return ScalingDecision(
                    action="no_action",
                    previous_servers=active_count,
                    desired_servers=self.min_servers,
                    actual_servers=active_count,
                    reason=(
                        f"Predicted CPU ({predicted_cpu:.1f}%) below threshold for "
                        f"{self.consecutive_low_demand_count} intervals, but active servers ({active_count}) "
                        f"already at minimum ({self.min_servers})."
                    ),
                    predicted_viewers=predicted_viewers,
                )

            # Rule: Respect cooldown
            if not self.is_cooldown_expired(current_interval):
                return ScalingDecision(
                    action="no_action",
                    previous_servers=active_count,
                    desired_servers=conservative_desired,
                    actual_servers=active_count,
                    reason=(
                        f"Predicted CPU ({predicted_cpu:.1f}%) below threshold for "
                        f"{self.consecutive_low_demand_count} intervals, but predictive scale-down is deferred "
                        f"due to active cooldown (last scaled at interval {self.last_scaling_interval}, "
                        f"cooldown={self.cooldown_intervals} intervals)."
                    ),
                    predicted_viewers=predicted_viewers,
                )

            # Cooldown expired: execute conservative scale-down
            if simulator is not None:
                removed_server = simulator.remove_server()
                actual_servers = len(simulator.get_active_servers())
                self.record_scaling_action("scale_down", current_interval)
                self.consecutive_low_demand_count = 0
                reason = (
                    f"Predicted CPU ({predicted_cpu:.1f}%) below {self.scale_down_cpu_threshold}% for "
                    f"{self.scale_down_consecutive_intervals} consecutive intervals (predicted {predicted_viewers} "
                    f"viewers at t+{self.prediction_horizon}). Scaled down conservatively from {active_count} "
                    f"to {actual_servers} servers (removed {removed_server.server_id})."
                )
            else:
                actual_servers = active_count
                reason = (
                    f"Predicted CPU ({predicted_cpu:.1f}%) below {self.scale_down_cpu_threshold}% for "
                    f"{self.scale_down_consecutive_intervals} consecutive intervals (predicted {predicted_viewers} "
                    f"viewers at t+{self.prediction_horizon}). Scale-down requested to {conservative_desired} servers "
                    f"(no simulator provided; infrastructure unchanged at {active_count} servers)."
                )

            return ScalingDecision(
                action="scale_down",
                previous_servers=active_count,
                desired_servers=conservative_desired,
                actual_servers=actual_servers,
                reason=reason,
                predicted_viewers=predicted_viewers,
            )

        # -------------------------------------------------------------
        # 6. NEUTRAL RANGE (Within Normal CPU Range)
        # -------------------------------------------------------------
        else:
            # Projected CPU is between scale_down_cpu_threshold and target_cpu_utilization
            self.consecutive_low_demand_count = 0

            return ScalingDecision(
                action="no_action",
                previous_servers=active_count,
                desired_servers=active_count,
                actual_servers=active_count,
                reason=(
                    f"Predicted demand ({predicted_viewers} viewers at t+{self.prediction_horizon}, "
                    f"projected CPU {predicted_cpu:.1f}%) is within normal operating range "
                    f"[{self.scale_down_cpu_threshold}%, {self.target_cpu_utilization}%]. "
                    f"Maintaining {active_count} servers."
                ),
                predicted_viewers=predicted_viewers,
            )
