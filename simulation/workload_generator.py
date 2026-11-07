"""Synthetic Workload Generator for Live Event Broadcasting Simulation.

This module models concurrent viewer traffic across four realistic broadcasting
scenarios using a mathematical combination of baseline traffic, planned event traffic,
sudden event spikes, and stochastic noise:

    V(t) = B(t) + S(t) + P(t) + noise(t)

Where:
    - V(t): Total concurrent viewers at time t.
    - B(t): Baseline viewer traffic.
    - S(t): Sudden event spike modeled with a Gaussian function:
            S(t) = A * exp(-((t - t0)^2) / (2 * sigma^2))
    - P(t): Planned event-related traffic (e.g. high-interest matches or finals).
    - noise(t): Stochastic fluctuations simulating natural viewer behavior.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

# Scenario name constants
SCENARIO_NORMAL: str = "Normal Traffic"
SCENARIO_SPIKE: str = "Sudden Traffic Spike"
SCENARIO_PEAK: str = "Peak Broadcasting Event"
SCENARIO_ENDING: str = "Event Ending"

# Default configuration parameters
DEFAULT_DURATION_MINUTES: int = 120
DEFAULT_TIME_INTERVAL: int = 1
DEFAULT_INITIAL_VIEWERS: int = 2000
DEFAULT_MIN_VIEWERS: int = 0
DEFAULT_MAX_VIEWERS: int = 100000
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_START_TIME: str = "2026-01-01 00:00:00"


def _validate_base_config(
    duration_minutes: int,
    time_interval: int,
    initial_viewers: int,
    min_viewers: int,
    max_viewers: int,
) -> None:
    """Validate core simulation parameters.

    Args:
        duration_minutes: Total simulation duration in minutes.
        time_interval: Step size between observations in minutes.
        initial_viewers: Viewer count at minute 0.
        min_viewers: Allowed lower bound for viewer count.
        max_viewers: Allowed upper bound for viewer count.

    Raises:
        ValueError: If any parameter is outside valid domains.
    """
    if duration_minutes <= 0:
        raise ValueError(f"duration_minutes must be positive, got {duration_minutes}")
    if time_interval <= 0:
        raise ValueError(f"time_interval must be positive, got {time_interval}")
    if min_viewers < 0:
        raise ValueError(f"min_viewers must be non-negative, got {min_viewers}")
    if max_viewers <= min_viewers:
        raise ValueError(
            f"max_viewers ({max_viewers}) must be strictly greater than min_viewers ({min_viewers})"
        )
    if not (min_viewers <= initial_viewers <= max_viewers):
        raise ValueError(
            f"initial_viewers ({initial_viewers}) must be between min_viewers ({min_viewers}) "
            f"and max_viewers ({max_viewers})"
        )


def _build_dataframe(
    minutes: np.ndarray,
    viewers: np.ndarray,
    scenario_name: str,
    event_active: np.ndarray,
    start_time: Union[str, pd.Timestamp],
    time_interval: int = 1,
) -> pd.DataFrame:
    """Construct a standardized Pandas DataFrame for a generated scenario.

    Args:
        minutes: 1D array of simulation minute indices.
        viewers: 1D array of integer concurrent viewer counts.
        scenario_name: Name of the scenario.
        event_active: 1D array of boolean event active indicators.
        start_time: Starting timestamp for the simulation.
        time_interval: Interval in minutes between observations.

    Returns:
        Standardized DataFrame with columns:
        ['minute', 'timestamp', 'viewers', 'scenario', 'event_active']
    """
    base_timestamp = pd.to_datetime(start_time)
    timestamps = [base_timestamp + pd.Timedelta(minutes=int(m)) for m in minutes]

    df = pd.DataFrame(
        {
            "minute": minutes.astype(int),
            "timestamp": timestamps,
            "viewers": viewers.astype(int),
            "scenario": scenario_name,
            "event_active": event_active.astype(bool),
        }
    )
    return df


def generate_normal_traffic(
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
    time_interval: int = DEFAULT_TIME_INTERVAL,
    initial_viewers: int = DEFAULT_INITIAL_VIEWERS,
    min_viewers: int = DEFAULT_MIN_VIEWERS,
    max_viewers: int = DEFAULT_MAX_VIEWERS,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    start_time: Union[str, pd.Timestamp] = DEFAULT_START_TIME,
    noise_std: float = 75.0,
) -> pd.DataFrame:
    """Generate Scenario A: Normal Traffic.

    Simulates ordinary broadcasting viewership:
    - Starts with approximately 2,000 viewers.
    - Gradually increases viewership to a moderate peak (~8,500 viewers).
    - Introduces small random fluctuations.
    - Gradually declines towards the end.
    - Exhibits smooth, non-linear progression.
    - event_active is False throughout (no unexpected/major event active).

    Args:
        duration_minutes: Total minutes of simulation (default 120).
        time_interval: Step size in minutes (default 1).
        initial_viewers: Starting viewer count (default 2,000).
        min_viewers: Minimum viewer clamp limit (default 0).
        max_viewers: Maximum viewer clamp limit (default 100,000).
        random_seed: Random seed for reproducibility (default 42).
        start_time: Reference starting timestamp.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        pd.DataFrame with simulation data.
    """
    _validate_base_config(
        duration_minutes, time_interval, initial_viewers, min_viewers, max_viewers
    )

    rng = np.random.default_rng(random_seed)
    minutes = np.arange(0, duration_minutes, time_interval)
    t = minutes.astype(float)
    total_time = float(duration_minutes)

    # Non-linear growth and decline using combined sinusoidal harmonics
    # Primary cycle rises to peak around mid-simulation
    norm_t = t / total_time
    growth = 6800.0 * np.sin(np.pi * norm_t) ** 1.3
    # Secondary harmonic adds subtle realistic asymmetry
    asymmetry = 700.0 * np.sin(2.0 * np.pi * norm_t)

    baseline = initial_viewers + growth + asymmetry
    noise = rng.normal(loc=0.0, scale=noise_std, size=len(t))

    raw_viewers = baseline + noise
    viewers = np.clip(np.round(raw_viewers), min_viewers, max_viewers).astype(int)

    # Normal traffic has no major anomaly or critical event window
    event_active = np.zeros(len(t), dtype=bool)

    return _build_dataframe(
        minutes=minutes,
        viewers=viewers,
        scenario_name=SCENARIO_NORMAL,
        event_active=event_active,
        start_time=start_time,
        time_interval=time_interval,
    )


def generate_sudden_spike_traffic(
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
    time_interval: int = DEFAULT_TIME_INTERVAL,
    initial_viewers: int = DEFAULT_INITIAL_VIEWERS,
    min_viewers: int = DEFAULT_MIN_VIEWERS,
    max_viewers: int = DEFAULT_MAX_VIEWERS,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    start_time: Union[str, pd.Timestamp] = DEFAULT_START_TIME,
    spike_magnitude: float = 25000.0,
    spike_time: float = 60.0,
    spike_sigma: float = 6.0,
    noise_std: float = 80.0,
) -> pd.DataFrame:
    """Generate Scenario B: Sudden Traffic Spike.

    Simulates an unexpected major occurrence during broadcasting (e.g., wicket or goal):
    - Starts with relatively low viewership (~2,000).
    - Maintains moderate baseline traffic initially.
    - Sharp Gaussian-shaped spike centered around spike_time (default min 60):
          S(t) = A * exp(-((t - t0)^2) / (2 * sigma^2))
    - Gradually returns towards baseline traffic.
    - event_active is True within the 2*sigma spike interval (|t - spike_time| <= 2 * spike_sigma),
      covering over 95% of the spike energy.

    Args:
        duration_minutes: Total minutes of simulation (default 120).
        time_interval: Step size in minutes (default 1).
        initial_viewers: Starting viewer count (default 2,000).
        min_viewers: Minimum viewer clamp limit (default 0).
        max_viewers: Maximum viewer clamp limit (default 100,000).
        random_seed: Random seed for reproducibility (default 42).
        start_time: Reference starting timestamp.
        spike_magnitude: Peak amplitude A of the Gaussian spike.
        spike_time: Center time t0 of the spike in minutes.
        spike_sigma: Spread parameter sigma of the Gaussian spike.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        pd.DataFrame with simulation data.
    """
    _validate_base_config(
        duration_minutes, time_interval, initial_viewers, min_viewers, max_viewers
    )
    if spike_sigma <= 0:
        raise ValueError(f"spike_sigma must be positive, got {spike_sigma}")
    if spike_magnitude < 0:
        raise ValueError(f"spike_magnitude must be non-negative, got {spike_magnitude}")

    rng = np.random.default_rng(random_seed)
    minutes = np.arange(0, duration_minutes, time_interval)
    t = minutes.astype(float)
    total_time = float(duration_minutes)

    # Moderate baseline traffic
    baseline = initial_viewers + 1200.0 * np.sin(np.pi * (t / total_time))

    # Gaussian spike: S(t) = A * exp(-((t - t0)^2) / (2 * sigma^2))
    spike = spike_magnitude * np.exp(-((t - spike_time) ** 2) / (2.0 * (spike_sigma**2)))

    noise = rng.normal(loc=0.0, scale=noise_std, size=len(t))

    raw_viewers = baseline + spike + noise
    viewers = np.clip(np.round(raw_viewers), min_viewers, max_viewers).astype(int)

    # Event active during the significant spike window (within 2 standard deviations)
    event_active = np.abs(t - spike_time) <= (2.0 * spike_sigma)

    return _build_dataframe(
        minutes=minutes,
        viewers=viewers,
        scenario_name=SCENARIO_SPIKE,
        event_active=event_active,
        start_time=start_time,
        time_interval=time_interval,
    )


def generate_peak_event_traffic(
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
    time_interval: int = DEFAULT_TIME_INTERVAL,
    initial_viewers: int = DEFAULT_INITIAL_VIEWERS,
    min_viewers: int = DEFAULT_MIN_VIEWERS,
    max_viewers: int = DEFAULT_MAX_VIEWERS,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    start_time: Union[str, pd.Timestamp] = DEFAULT_START_TIME,
    peak_viewers: int = 55000,
    peak_start: int = 48,
    peak_end: int = 72,
    residual_viewers: int = 8000,
    noise_std: float = 220.0,
) -> pd.DataFrame:
    """Generate Scenario C: Peak Broadcasting Event.

    Simulates a major scheduled live event (e.g. championship final):
    - Starts with relatively low viewership (~2,000).
    - Gradually increases viewership as the event progresses.
    - Reaches a massive peak around the middle of the simulation.
    - Maintains elevated plateau traffic during the critical event window (peak_start to peak_end).
    - Gradually decreases afterwards as the broadcast concludes.
    - event_active is True throughout the core peak window [peak_start, peak_end].

    Args:
        duration_minutes: Total minutes of simulation (default 120).
        time_interval: Step size in minutes (default 1).
        initial_viewers: Starting viewer count (default 2,000).
        min_viewers: Minimum viewer clamp limit (default 0).
        max_viewers: Maximum viewer clamp limit (default 100,000).
        random_seed: Random seed for reproducibility (default 42).
        start_time: Reference starting timestamp.
        peak_viewers: Target concurrent viewership at peak plateau.
        peak_start: Minute when peak plateau begins.
        peak_end: Minute when peak plateau ends.
        residual_viewers: Viewer count towards simulation conclusion.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        pd.DataFrame with simulation data.
    """
    _validate_base_config(
        duration_minutes, time_interval, initial_viewers, min_viewers, max_viewers
    )
    if peak_viewers <= initial_viewers:
        raise ValueError(
            f"peak_viewers ({peak_viewers}) must be greater than initial_viewers ({initial_viewers})"
        )
    if peak_viewers > max_viewers:
        raise ValueError(
            f"peak_viewers ({peak_viewers}) must not exceed max_viewers ({max_viewers})"
        )
    if not (0 < peak_start < peak_end < duration_minutes):
        raise ValueError(
            f"Invalid peak window [{peak_start}, {peak_end}] for duration {duration_minutes}"
        )
    if not (min_viewers <= residual_viewers < peak_viewers):
        raise ValueError(
            f"residual_viewers ({residual_viewers}) must be between min_viewers ({min_viewers}) "
            f"and peak_viewers ({peak_viewers})"
        )

    rng = np.random.default_rng(random_seed)
    minutes = np.arange(0, duration_minutes, time_interval)
    t = minutes.astype(float)
    total_steps = float(duration_minutes - 1)

    # Smooth piece-wise harmonic envelope:
    # 1. Ramp-up (0 -> peak_start): smooth harmonic rise from initial_viewers to peak_viewers
    # 2. Peak plateau (peak_start -> peak_end): elevated traffic sustained at peak_viewers
    # 3. Ramp-down (peak_end -> end): smooth harmonic decline to residual_viewers
    planned_traffic = np.zeros_like(t)

    # Ramp-up phase
    ramp_up_mask = t <= peak_start
    if peak_start > 0:
        up_progress = t[ramp_up_mask] / float(peak_start)
        ramp_up_factor = 0.5 * (1.0 - np.cos(np.pi * up_progress))
        planned_traffic[ramp_up_mask] = (
            initial_viewers + (peak_viewers - initial_viewers) * ramp_up_factor
        )

    # Sustained peak plateau phase
    plateau_mask = (t > peak_start) & (t <= peak_end)
    planned_traffic[plateau_mask] = float(peak_viewers)

    # Ramp-down phase
    ramp_down_mask = t > peak_end
    if np.any(ramp_down_mask):
        down_progress = (t[ramp_down_mask] - float(peak_end)) / (
            total_steps - float(peak_end)
        )
        ramp_down_factor = 0.5 * (1.0 + np.cos(np.pi * down_progress))
        planned_traffic[ramp_down_mask] = (
            residual_viewers + (peak_viewers - residual_viewers) * ramp_down_factor
        )

    noise = rng.normal(loc=0.0, scale=noise_std, size=len(t))

    raw_viewers = planned_traffic + noise
    viewers = np.clip(np.round(raw_viewers), min_viewers, max_viewers).astype(int)

    # Event is actively at peak during [peak_start, peak_end]
    event_active = (minutes >= peak_start) & (minutes <= peak_end)

    return _build_dataframe(
        minutes=minutes,
        viewers=viewers,
        scenario_name=SCENARIO_PEAK,
        event_active=event_active,
        start_time=start_time,
        time_interval=time_interval,
    )


def generate_event_ending_traffic(
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
    time_interval: int = DEFAULT_TIME_INTERVAL,
    initial_viewers: int = 50000,
    final_viewers: int = 3000,
    min_viewers: int = DEFAULT_MIN_VIEWERS,
    max_viewers: int = DEFAULT_MAX_VIEWERS,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    start_time: Union[str, pd.Timestamp] = DEFAULT_START_TIME,
    hold_duration: int = 18,
    noise_std: float = 180.0,
) -> pd.DataFrame:
    """Generate Scenario D: Event Ending.

    Simulates an event whose audience gradually departs:
    - Begins with high viewer traffic (~50,000).
    - Maintains a relatively stable audience initially during post-match wrap-up.
    - Gradually decreases viewers as the broadcast concludes.
    - Reaches low traffic towards the end (~3,000 viewers).
    - Finishes with fewer viewers than it starts with.
    - event_active is True during the initial wrap-up phase (minute <= hold_duration).

    Args:
        duration_minutes: Total minutes of simulation (default 120).
        time_interval: Step size in minutes (default 1).
        initial_viewers: High starting viewer count (default 50,000).
        final_viewers: Target low audience at conclusion (default 3,000).
        min_viewers: Minimum viewer clamp limit (default 0).
        max_viewers: Maximum viewer clamp limit (default 100,000).
        random_seed: Random seed for reproducibility (default 42).
        start_time: Reference starting timestamp.
        hold_duration: Minutes initial high viewership remains sustained.
        noise_std: Standard deviation of Gaussian noise.

    Returns:
        pd.DataFrame with simulation data.
    """
    _validate_base_config(
        duration_minutes, time_interval, initial_viewers, min_viewers, max_viewers
    )
    if final_viewers < min_viewers or final_viewers >= initial_viewers:
        raise ValueError(
            f"final_viewers ({final_viewers}) must be between min_viewers ({min_viewers}) "
            f"and initial_viewers ({initial_viewers})"
        )
    if hold_duration < 0 or hold_duration >= duration_minutes:
        raise ValueError(
            f"hold_duration must be between 0 and duration_minutes ({duration_minutes}), got {hold_duration}"
        )

    rng = np.random.default_rng(random_seed)
    minutes = np.arange(0, duration_minutes, time_interval)
    t = minutes.astype(float)
    total_steps = float(duration_minutes - 1)

    traffic_base = np.zeros_like(t)

    # Initial stable hold period
    hold_mask = t <= hold_duration
    traffic_base[hold_mask] = float(initial_viewers)

    # Smooth harmonic decay after hold_duration down to final_viewers
    decay_mask = t > hold_duration
    if np.any(decay_mask):
        decay_progress = (t[decay_mask] - float(hold_duration)) / (
            total_steps - float(hold_duration)
        )
        decay_factor = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
        traffic_base[decay_mask] = (
            final_viewers + (initial_viewers - final_viewers) * decay_factor
        )

    noise = rng.normal(loc=0.0, scale=noise_std, size=len(t))

    raw_viewers = traffic_base + noise
    viewers = np.clip(np.round(raw_viewers), min_viewers, max_viewers).astype(int)

    # Active event wrap-up during initial hold period
    event_active = minutes <= hold_duration

    return _build_dataframe(
        minutes=minutes,
        viewers=viewers,
        scenario_name=SCENARIO_ENDING,
        event_active=event_active,
        start_time=start_time,
        time_interval=time_interval,
    )


def generate_all_scenarios(
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
    time_interval: int = DEFAULT_TIME_INTERVAL,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    start_time: Union[str, pd.Timestamp] = DEFAULT_START_TIME,
) -> Dict[str, pd.DataFrame]:
    """Generate all four broadcasting workload scenarios.

    Args:
        duration_minutes: Simulation duration in minutes for each scenario (default 120).
        time_interval: Time interval between observations in minutes (default 1).
        random_seed: Random seed for reproducibility (default 42).
        start_time: Starting reference timestamp.

    Returns:
        Dictionary mapping scenario key names to corresponding DataFrames:
        - "normal": Scenario A (Normal Traffic)
        - "sudden_spike": Scenario B (Sudden Traffic Spike)
        - "peak_event": Scenario C (Peak Broadcasting Event)
        - "event_ending": Scenario D (Event Ending)
    """
    scenarios: Dict[str, pd.DataFrame] = {
        "normal": generate_normal_traffic(
            duration_minutes=duration_minutes,
            time_interval=time_interval,
            random_seed=random_seed,
            start_time=start_time,
        ),
        "sudden_spike": generate_sudden_spike_traffic(
            duration_minutes=duration_minutes,
            time_interval=time_interval,
            random_seed=random_seed,
            start_time=start_time,
        ),
        "peak_event": generate_peak_event_traffic(
            duration_minutes=duration_minutes,
            time_interval=time_interval,
            random_seed=random_seed,
            start_time=start_time,
        ),
        "event_ending": generate_event_ending_traffic(
            duration_minutes=duration_minutes,
            time_interval=time_interval,
            random_seed=random_seed,
            start_time=start_time,
        ),
    }
    return scenarios
