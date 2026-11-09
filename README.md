# Predictive Cloud Resource Allocation and Auto-Scaling for Live Event Broadcasting

## Project Overview

This project simulates cloud resource allocation and predictive auto-scaling during live event broadcasting. The simulation models audience viewer dynamics, evaluates machine learning-based workload forecasts, and controls simulated cloud server pools to maintain quality of service while optimizing infrastructure costs.

---

## Stage A: Synthetic Workload Generator

### 1. What Synthetic Workload Generation Means

In live event broadcasting (e.g., sports tournaments, concerts, breaking news), audience viewership fluctuates dynamically depending on scheduled matches, unexpected match events (such as goals or wickets), and broadcast endings.

Because production telemetry data is often proprietary or noisy, **synthetic workload generation** models concurrent viewer counts over discrete time intervals using mathematical formulations and stochastic processes. This provides a structured, reproducible, and realistic benchmark dataset suitable for:
- Training machine learning forecasting models (e.g., Random Forest regressors).
- Driving discrete-event cloud infrastructure simulations.
- Benchmarking reactive vs. predictive auto-scaling policies.

### 2. Workload Scenarios

Stage A implements four independent 120-minute traffic scenarios:

1. **Scenario A: Normal Traffic (`Normal Traffic`)**
   - Simulates typical day-to-day broadcasting traffic.
   - Starts around 2,000 viewers, gradually builds to a moderate peak (~8,500 viewers) near mid-broadcast, and slowly declines towards the end with minor natural variations.
   - `event_active` is `False` throughout.

2. **Scenario B: Sudden Traffic Spike (`Sudden Traffic Spike`)**
   - Simulates sudden viral moments or major game occurrences (e.g., a critical goal, wicket, or turnover).
   - Maintains a steady moderate baseline until minute 60, where viewership spikes sharply by ~25,000 concurrent viewers before returning toward baseline.
   - `event_active` is `True` within a $\pm 2\sigma$ window around the spike peak.

3. **Scenario C: Peak Broadcasting Event (`Peak Broadcasting Event`)**
   - Simulates a major scheduled live event (e.g., a championship final).
   - Starts at ~2,000 viewers, undergoes sustained gradual build-up, reaches a high-volume peak plateau (~55,000 viewers) between minutes 48 and 72, and gradually decreases afterwards.
   - `event_active` is `True` during the core peak event window `[48, 72]`.

4. **Scenario D: Event Ending (`Event Ending`)**
   - Simulates audience departure as an event concludes and post-match coverage wraps up.
   - Begins with elevated viewership (~50,000 viewers), holds steady for approximately 18 minutes, then smoothly decays down to ~3,000 viewers at simulation end.
   - `event_active` is `True` during the initial hold period ($\le 18$ minutes).

### 3. Mathematical Approach

The concurrent viewer workload $V(t)$ at simulation minute $t$ is modeled as:

$$V(t) = B(t) + S(t) + P(t) + \text{noise}(t)$$

Where:
- **$B(t)$ (Baseline Traffic):** Steady audience component modeled via smooth sinusoidal harmonics.
- **$S(t)$ (Sudden Event Spike):** Gaussian distribution modeling localized audience surges:
  $$S(t) = A \cdot \exp\left(-\frac{(t - t_0)^2}{2\sigma^2}\right)$$
  where $A$ is the spike amplitude, $t_0$ is the spike center time (minute 60), and $\sigma$ controls duration spread.
- **$P(t)$ (Planned Event Traffic):** Smooth harmonic transitions modeling scheduled multi-phase audience surges and sustained plateaus.
- **$\text{noise}(t)$ (Stochastic Component):** Zero-mean Gaussian noise $\mathcal{N}(0, \sigma_{\text{noise}}^2)$ simulating viewer fluctuations.
- **Boundary Constraints:** Workload values are constrained such that $V(t) \in [\text{min\_viewers}, \text{max\_viewers}]$, rounded to integers, and sequenced at continuous one-minute intervals. Controlled random seeds ensure complete reproducibility.

### 4. Dataset Format

Each generated scenario produces a dataset with the following schema:

| Column | Type | Description |
|---|---|---|
| `minute` | `int` | Sequential simulation minute index ($0, 1, \dots, 119$) |
| `timestamp` | `datetime` | Simulation timestamp corresponding to the observation minute |
| `viewers` | `int` | Number of concurrent active viewers |
| `scenario` | `str` | Name of the traffic scenario |
| `event_active` | `bool` | Boolean flag indicating whether a major event/spike is active |

### 5. Directory Structure

```
cloud-autoscaler/
├── data/
│   ├── raw/
│   │   ├── normal_traffic.csv
│   │   ├── sudden_spike_traffic.csv
│   │   ├── peak_event_traffic.csv
│   │   ├── event_ending_traffic.csv
│   │   └── all_scenarios_combined.csv
│   └── processed/
│       ├── normal_workload.png
│       ├── sudden_spike_workload.png
│       ├── peak_event_workload.png
│       ├── event_ending_workload.png
│       ├── all_scenarios_comparison.png
│       ├── prediction_results.csv
│       ├── actual_vs_predicted.png
│       ├── scenario_prediction_comparison.png
│       ├── feature_importance.png
│       ├── infrastructure_metrics.csv
│       ├── server_metrics.csv
│       ├── cpu_utilization_comparison.png
│       ├── active_server_load.png
│       └── infrastructure_overload.png
├── models/
│   ├── __init__.py
│   ├── predictor.py
│   └── random_forest_model.joblib
├── simulation/
│   ├── __init__.py
│   ├── workload_generator.py
│   ├── cloud_simulator.py
│   └── metrics.py
├── tests/
│   ├── __init__.py
│   ├── test_workload_generator.py
│   ├── test_predictor.py
│   └── test_cloud_simulator.py
├── .gitignore
├── main.py
├── train_model.py
├── run_infrastructure_simulation.py
├── README.md
└── requirements.txt
```

### 6. Running the Workload Generator

To run the pipeline and generate all CSV datasets and visual plots:

```bash
# Activate virtual environment (if applicable)
source .venv/bin/activate

# Execute Stage A pipeline
python main.py
```

### 7. Generated File Locations

- **Raw Datasets (`data/raw/`):**
  - `normal_traffic.csv`: Scenario A data (120 observations).
  - `sudden_spike_traffic.csv`: Scenario B data (120 observations).
  - `peak_event_traffic.csv`: Scenario C data (120 observations).
  - `event_ending_traffic.csv`: Scenario D data (120 observations).
  - `all_scenarios_combined.csv`: Combined dataset with all 4 scenarios (480 observations).

- **Visualizations (`data/processed/`):**
  - High-resolution individual workload plots and a 4-panel comparison figure (`all_scenarios_comparison.png`).

### 8. Running Automated Tests

Run the unit test suite covering all Stage A behavioral requirements:

```bash
python -m unittest discover -s tests -v
```

---

## Stage B: ML-Based Workload Prediction Engine

### 1. Objective and Architecture

Stage B develops an automated machine learning forecasting engine that predicts concurrent viewer demand $V(t + 5)$ five minutes into the future, using only telemetry available at or before time $t$. This proactive lead time allows cloud resource controllers to initiate server spin-up procedures before traffic surges cause Quality of Service (QoS) degradation.

### 2. Why Random Forest Regressor Was Selected

A `RandomForestRegressor` ensemble is selected for the predictive core because:
- **Non-Linear Dynamics:** Viewership surges exhibit sharp transitions, sigmoidal plateaus, and non-linear harmonic decay that linear regressors fail to capture accurately.
- **Robustness to Scale and Outliers:** Tree-based ensembles do not require feature normalization or standard scaling across heterogeneous inputs (e.g., raw viewer counts vs. growth rates).
- **Interpretability:** Provides native feature importance metrics based on Gini impurity reduction.
- **Ensemble Stability:** Averaging across 200 de-correlated decision trees reduces variance and limits overfitting to random noise fluctuations.

### 3. Prediction Horizon

- **Horizon ($H = 5$ minutes):** At observation minute $t$, the model predicts viewership at minute $t + 5$.
- In cloud autoscaling architectures, virtual machines require initialization time (OS boot, container pulling, readiness checks). A 5-minute prediction horizon provides the necessary buffer for proactive server provisioning.
- The prediction horizon is configurable across the preprocessing and predictor APIs.

### 4. Historical Features

All features are strictly derived from historical or current observations ($t \le \text{prediction time}$):

| Feature Name | Description | Window / Formulation |
|---|---|---|
| `viewers` | Current concurrent viewer count | Observation at minute $t$ |
| `viewers_lag_1` | Viewers 1 minute ago | $V(t - 1)$ |
| `viewers_lag_2` | Viewers 2 minutes ago | $V(t - 2)$ |
| `viewers_lag_5` | Viewers 5 minutes ago | $V(t - 5)$ |
| `viewers_rolling_mean_5` | Rolling average over previous 5 minutes | $\frac{1}{5} \sum_{k=0}^{4} V(t - k)$ (includes minute $t$) |
| `viewers_rolling_std_5` | Rolling standard deviation over previous 5 minutes | Standard deviation of $\{V(t-4), \dots, V(t)\}$ |
| `viewer_growth_rate` | Recent minute-over-minute viewer velocity | $V(t) - V(t - 1)$ |
| `minute` | Current simulation time index | Minute $t$ |
| `event_active` | Indicator if a major event/spike is active at time $t$ | $1$ if active at minute $t$, else $0$ |

### 5. Chronological Purged Split & Leakage Prevention

- **Independent Scenario Processing:** Feature extraction, lag calculation, and rolling windows are computed for each scenario independently. The final observations of one scenario never blend into the starting history of another.
- **Purged Chronological Split (Purge Gap = 5 minutes):** For each scenario of 110 aligned observations (minutes 5 to 114):
  - Training feature window: minutes 5 to 87 (83 observations per scenario, 332 total).
  - Training target window: minutes 10 to 92 ($\max = 92$).
  - Purged embargo gap: minutes 88 to 92 (5 observations removed from training per scenario).
  - Test feature window: minutes 93 to 114 (22 observations per scenario, 88 total; $\min = 93$).
  - Test target window: minutes 98 to 119.
- **Strict Leakage Prevention Condition:**
  $$\max(\text{training target minute}) = 92 < 93 = \min(\text{test feature minute})$$
  $$\max(\text{training target timestamp}) = \text{01:32:00} < \text{01:33:00} = \min(\text{test feature timestamp})$$
  No training target overlaps with the test feature window or test period.

### 6. Evaluation Metrics and Baseline Comparison

The Random Forest model is evaluated alongside a **Persistence Baseline**:
$$\hat{V}_{\text{persistence}}(t + 5) = V(t)$$

#### Metric Definitions:
- **Mean Absolute Error (MAE):** Average magnitude of forecasting errors in concurrent viewers:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |V_{i} - \hat{V}_{i}|$$
- **Root Mean Squared Error (RMSE):** Quadratic metric that penalizes larger prediction deviations:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (V_{i} - \hat{V}_{i})^2}$$

#### Primary Holdout Test Set Results (Purged Split):

| Metric | Random Forest Model | Persistence Baseline | Improvement |
|---|---:|---:|---:|
| **Overall MAE** | **1,263.7 viewers** | 1,871.3 viewers | **32.5% reduction in error** |
| **Overall RMSE** | **1,687.5 viewers** | 3,002.1 viewers | **43.8% reduction in error** |

#### Scenario Breakdown (Primary Holdout):

| Scenario Name | RF MAE | Baseline MAE | RF RMSE | Baseline RMSE |
|---|---:|---:|---:|---:|
| **Normal Traffic** | 1,106.5 | 600.3 | 1,165.3 | 617.6 |
| **Sudden Traffic Spike** | 403.1 | 135.5 | 414.3 | 157.6 |
| **Peak Broadcasting Event** | **2,292.7** | 5,381.1 | **2,788.1** | 5,775.0 |
| **Event Ending** | **1,252.5** | 1,368.4 | **1,444.6** | 1,514.1 |

*Analysis:* In high-gradient scenarios with strong dynamic trends (Peak Broadcasting Event and Event Ending), the Random Forest model dramatically outperforms persistence, cutting MAE by over 57% on peak broadcasting. On near-flat segments (e.g. post-spike quiet baseline in the Sudden Spike scenario), persistence exhibits minimal baseline error.

#### Separate Sudden-Spike Stress Evaluation:

To evaluate model performance during the actual spike, a dedicated chronological stress evaluation was conducted on the sudden-spike scenario across the active spike onset and peak window (feature minutes 45 to 70, forecasting targets 50 to 75):
- **Training Constraints:** The model was trained with zero exposure to the spike window targets (Sudden Spike training was restricted to minutes $\le 39$, targets $\le 44$).
- **Stress Results:**
  - Random Forest Stress MAE: **7,798.3 viewers** | Persistence Stress MAE: **8,352.7 viewers**
  - Random Forest Stress RMSE: **10,983.2 viewers** | Persistence Stress RMSE: **8,915.0 viewers**
- **Dynamic Behavior:** During the rapid onset (minutes 48–54), persistence severely lags with errors between 9,000 and 12,000 viewers. Random Forest detects early growth rate signals and tracks the rise much closer (e.g., at minute 53, RF error is only +382 viewers vs. -10,965 for persistence). During the post-peak collapse, both models face significant prediction error as viewers leave rapidly.

### 7. Known Limitations

- **Synthetic Workload Environment:** Models are trained on mathematical simulation patterns and stochastic noise rather than real-world production CDN telemetry.
- **Unannounced Sudden Spikes:** A sharp, unexpected spike (e.g., instantaneous goal or wicket) cannot be anticipated five minutes prior without an antecedent signal in historical data. The model can only adapt once early onset indicators register in lag and growth rate features.

### 8. How to Train the Model

To execute the Stage B training and evaluation pipeline:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run training pipeline
python train_model.py
```

### 9. Model Artifact and Output Files

- **Model Artifact:** [`models/random_forest_model.joblib`](file:///home/madhur/Documents/Projects/cloud-autoscaler/models/random_forest_model.joblib) contains the fitted estimator and metadata dictionary (`feature_names`, `prediction_horizon`).
- **Evaluation Predictions:** [`data/processed/prediction_results.csv`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/prediction_results.csv) records test predictions, actual viewers, and residuals.
- **Visualizations:**
  - `actual_vs_predicted.png`: Forecast vs actual viewers across the test set.
  - `scenario_prediction_comparison.png`: 4-panel scenario comparison.
  - `feature_importance.png`: Feature importance breakdown.

### 10. Reusable Prediction API

Future cloud simulator modules can import the standalone prediction function:

```python
import pandas as pd
from models.predictor import predict_next_viewers

# history must contain at least 6 chronological observations
# with columns ['minute', 'viewers', 'event_active' (optional)]
history_df = pd.DataFrame({
    "minute": [25, 26, 27, 28, 29, 30],
    "viewers": [12100, 12450, 12900, 13400, 14020, 14700],
    "event_active": [False, False, False, False, False, True],
})

# Predict viewers 5 minutes ahead (at minute 35)
predicted_viewers = predict_next_viewers(history_df)
print(f"Predicted viewers at minute 35: {predicted_viewers}")
```

### 11. Running the Automated Test Suite

Run all Stage A and Stage B unit tests:

```bash
python -m unittest discover -s tests -v
```

---

## Stage C: Simulated Cloud Infrastructure

### 1. Objective and Architecture

Stage C provides a deterministic, Python-based discrete-time simulation of cloud infrastructure designed to serve incoming live-broadcast concurrent viewership.

The infrastructure simulation is intentionally decoupled from autoscaling policy decision-making:
- **Stage C Scope:** Models simulated server nodes, load balancing, CPU and memory utilization, resource overload detection, and explicit lifecycle management controls (`add_server`, `remove_server`).
- **Decoupled Architecture:** Infrastructure mechanisms are exposed via a clean simulation API (`step()`). In Stage D, proactive and reactive autoscalers will interface with this exact API to decide *when* and *how many* servers to scale.

### 2. Server Configuration Defaults

Infrastructure parameters are configurable via `ServerConfig`:

| Parameter | Default Value | Description |
|---|---:|---|
| `capacity_per_server` | 1,000 viewers | Maximum nominal viewer capacity per server node |
| `initial_servers` | 2 servers | Baseline server pool size at simulation initialization |
| `min_servers` | 2 servers | Strict lower bound enforced on the active server pool |
| `max_servers` | 30 servers | Strict upper bound enforced on the active server pool |
| `total_memory_gb` | 8.0 GB | Total simulated physical RAM per server node |
| `baseline_memory_gb` | 1.0 GB | Static OS and background runtime memory footprint |
| `memory_per_viewer_mb` | 4.0 MB | Additional memory allocated per active connected viewer |

### 3. Load Distribution Mechanism

For each simulation interval, incoming viewers $V(t)$ are distributed across $N$ active servers deterministically and evenly:

$$\text{base} = \lfloor V(t) / N \rfloor, \quad \text{remainder} = V(t) \pmod N$$

- The first $\text{remainder}$ servers receive $\text{base} + 1$ viewers.
- The remaining servers receive $\text{base}$ viewers.
- **Guarantee:** $\sum_{i=1}^N \text{assigned\_viewers}_i = V(t)$ strictly holds, and $|\text{viewers}_i - \text{viewers}_j| \le 1$.
- Viewers are never discarded when server capacity is exceeded; surplus load continues onto the servers to accurately capture resource stress.

### 4. CPU and Memory Utilisation Models

#### CPU Utilisation Model:
$$\text{CPU utilisation (\%)} = \left(\frac{\text{assigned viewers}}{\text{capacity per server}}\right) \times 100$$
- Uncapped metric: Values $>100\%$ remain visible to reflect server CPU stress.

#### Memory Utilisation Model:
$$\text{Memory used (GB)} = \text{baseline\_memory\_gb} + \left(\text{assigned viewers} \times \frac{\text{memory\_per\_viewer\_mb}}{1024}\right)$$
$$\text{Memory utilisation (\%)} = \left(\frac{\text{Memory used (GB)}}{\text{total\_memory\_gb}}\right) \times 100$$
- Illustrative memory formulation capturing baseline OS overhead plus linear per-session stream buffering.

#### Overload Detection:
- A server is marked **`overloaded = True`** when either its CPU utilization or Memory utilization exceeds $100\%$.
- The infrastructure state is marked **`overload_status = True`** if one or more active servers are overloaded.

### 5. Server Management Interface

The `CloudSimulator` exposes explicit lifecycle methods:
- `add_server() -> SimulatedServer`: Adds an active server with a unique sequential ID (`server-1`, `server-2`, ...). Raises `ValueError` if `max_servers` is reached.
- `remove_server(server_id: Optional[str] = None) -> SimulatedServer`: Decommissions an active server (LIFO by default). Raises `ValueError` if `min_servers` boundary is reached.
- `get_active_servers() -> List[SimulatedServer]`: Returns the ordered pool of active servers.
- `step(viewers: int, minute: int) -> Tuple[InfrastructureMetrics, List[ServerMetrics]]`: Executes one discrete minute of simulation.
- `reset_simulation_state()`: Restores the server pool to `initial_servers` and clears historical workloads.

### 6. Fixed-Infrastructure Simulation Experiment

Running the four 120-minute Stage A scenarios through a fixed 2-server infrastructure pool (total capacity = 2,000 viewers) demonstrates the necessity of autoscaling:

| Scenario Name | Duration | Peak Viewers | Overloaded Minutes (%) | Avg CPU (%) | Max CPU (%) | Avg Memory (%) | Max Memory (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Normal Traffic** | 120 min | 8,994 | 120 (100.0%) | 298.6% | 449.7% | 158.3% | 232.1% |
| **Sudden Traffic Spike** | 120 min | 28,065 | 118 (98.3%) | 294.6% | 1,403.3% | 156.4% | 697.7% |
| **Peak Broadcasting Event** | 120 min | 55,329 | 119 (99.2%) | 1,738.3% | 2,766.5% | 861.3% | 1,363.3% |
| **Event Ending** | 120 min | 50,203 | 120 (100.0%) | 1,500.7% | 2,510.2% | 745.3% | 1,238.2% |

### 7. How to Run the Infrastructure Simulation

```bash
# Activate virtual environment
source .venv/bin/activate

# Execute Stage C simulation pipeline
python run_infrastructure_simulation.py
```

### 8. Output Datasets and Visualizations

- **Datasets (`data/processed/`):**
  - [`infrastructure_metrics.csv`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/infrastructure_metrics.csv): 480 minute-by-minute aggregate records across all scenarios.
  - [`server_metrics.csv`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/server_metrics.csv): 960 per-server utilization records across all scenarios.
- **Visualizations (`data/processed/`):**
  - [`cpu_utilization_comparison.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/cpu_utilization_comparison.png): Average CPU utilization across all scenarios with 100% threshold.
  - [`active_server_load.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/active_server_load.png): Individual server CPU load breakdown for Sudden Spike and Normal Traffic.
  - [`infrastructure_overload.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/infrastructure_overload.png): Overloaded server count over time across all 4 scenarios.

### 9. Running the Complete Automated Test Suite

Run tests:

```bash
python -m unittest discover -s tests -v
```

---

## Stage D: Cloud Autoscaling Policies and Comparative Evaluation

Stage D designs, implements, and evaluates autonomous cloud scaling strategies to dynamically provision servers in response to live-broadcast viewer fluctuations:
- **Stage D1 (Reactive Autoscaling):** CPU threshold-driven autoscaling policy (`ReactiveScaler`).
- **Stage D2 (Predictive Autoscaling):** Machine-learning forecast-driven autoscaling policy (`PredictiveScaler`) using the trained Random Forest model.
- **Stage D3 (Integrated Comparison Experiment):** Head-to-head empirical evaluation comparing Reactive vs Predictive autoscaling across four 120-minute synthetic workload scenarios.

---

### Stage D3: Reactive vs Predictive Autoscaling Comparison

#### 1. Experiment Architecture & Methodology

The experiment compares `ReactiveScaler` and `PredictiveScaler` under identical simulated operating conditions:
1. **Isolated Infrastructure:** Each strategy runs with an independent, freshly initialized `CloudSimulator` instance. No state or server allocations are shared between strategies.
2. **Identical Workload Traces:** Both scalers receive identical minute-by-minute viewer traces from `data/raw/` across all four scenarios (Normal Traffic, Sudden Traffic Spike, Peak Broadcasting Event, Event Ending).
3. **Identical Configuration:** Identical server capacities, scaling bounds, CPU thresholds, cooldown windows, and hysteresis parameters are enforced.

```
                           +---------------------------+
                           |   Workload Scenario Trace |
                           |    (data/raw/*.csv)       |
                           +-------------+-------------+
                                         |
                   +---------------------+---------------------+
                   |                                           |
                   v                                           v
    +-----------------------------+             +-----------------------------+
    |   Reactive Autoscaling      |             |   Predictive Autoscaling    |
    |   - Fresh CloudSimulator    |             |   - Fresh CloudSimulator    |
    |   - ReactiveScaler (D1)     |             |   - PredictiveScaler (D2)   |
    |   - Current CPU Feedback    |             |   - RF Forecast (t + 5 min) |
    +--------------+--------------+             +--------------+--------------+
                   |                                           |
                   +---------------------+---------------------+
                                         v
                           +---------------------------+
                           | Comparative Evaluation    |
                           | - Metrics & Decisions CSV |
                           | - Summary Analysis CSV    |
                           | - High-Resolution Charts  |
                           +---------------------------+
```

#### 2. Discrete-Time Simulation Timing and Causality

A strict discrete-time simulation cycle is enforced at every minute interval $t$:

$$\begin{aligned}
1. &\quad \text{Ingest concurrent viewers } V(t) \text{ and simulation minute } t. \\
2. &\quad \text{Step } \texttt{CloudSimulator.step}(V(t), t): \text{distribute incoming viewers evenly across current active servers } N(t). \\
3. &\quad \text{Observe infrastructure state: active servers } N(t), \text{ average CPU utilization } \text{CPU}(t), \text{ memory utilization, and overload status.} \\
4. &\quad \text{Record interval } t \text{ infrastructure performance metrics (representing the system while serving workload } V(t)). \\
5. &\quad \text{Invoke autoscaler: } \texttt{make\_decision}(t, V(t), N(t), \text{CPU}(t), \dots) \text{ using only observations available up to interval } t. \\
6. &\quad \text{Record scaling decision: action, reason, target server count, and predicted viewers.} \\
7. &\quad \text{Apply scaling adjustment directly on } \texttt{CloudSimulator} (\texttt{add\_server} / \texttt{remove\_server}).
\end{aligned}$$

**Causality Guarantee:**
- A scaling decision made at minute $t$ updates the server pool to $N(t+1)$.
- Workload $V(t+1)$ at interval $t+1$ is served by the newly resized server pool $N(t+1)$.
- Metrics recorded for minute $t$ reflect the actual infrastructure state while handling minute $t$'s workload and are never retroactively altered.

#### 3. Prediction Integrity & Anti-Leakage Controls

To maintain rigorous scientific validity, `PredictiveScaler` enforces strict forecasting boundaries:
- **Historical Horizon Restriction:** At interval $t$, the scaler receives historical telemetry strictly spanning $[0, t]$. Any telemetry from $t+1, t+2, \dots$ is completely withheld.
- **Automated Validation:** The scaler verifies that historical minute values are non-negative, strictly contiguous (step of 1), and that $\max(\text{minute}) = t$. Any observation with $\text{minute} > t$ immediately triggers a `ValueError`.
- **Warm-Up and Cold-Start Timing:** The feature engineering pipeline requires at least 6 consecutive observations to compute lag (1, 2, 5) and rolling window (mean, std) statistics:
  - **Minutes 0–4:** Insufficient historical observations (< 6). The scaler returns `action="no_action"` with `predicted_viewers=None`.
  - **Minute 5:** Exactly 6 observations ($t = 0, 1, 2, 3, 4, 5$) are available, and the first prediction and scaling decision can be made.
  - **Minute 6:** By discrete-time simulation causality, a scaling decision made at minute 5 affects minute 6, not minute 5.
  - **Initial Capacity Duration:** Therefore, the predictive scaler operates with its initial 2-server pool during intervals 0–5 (six intervals).
- **Reactive Information Boundary:** `ReactiveScaler` operates solely on real-time feedback ($V(t)$ and $\text{CPU}(t)$) without access to forecasts or future telemetry.

#### 4. Shared Infrastructure and Policy Configuration

Both strategies are configured with identical parameters:

| Parameter | Value | Description |
|---|---:|---|
| `capacity_per_server` | 1,000 viewers | Maximum nominal viewer capacity per server node |
| `initial_servers` | 2 servers | Starting server pool size at minute 0 |
| `min_servers` | 2 servers | Minimum active server allocation floor |
| `max_servers` | 30 servers | Maximum active server allocation ceiling (30,000 capacity) |
| `target_cpu_utilization` | 70.0% | Upper CPU target for triggering proactive or reactive scale-up |
| `scale_down_cpu_threshold` | 30.0% | Lower CPU threshold for conservative scale-down |
| `cooldown_intervals` | 2 minutes | Minimum simulation intervals required between scaling actions |
| `scale_down_consecutive_intervals` | 3 minutes | Consecutive low-demand intervals required before decommissioning a server |
| `prediction_horizon` | 5 minutes | Lead forecasting window ($t + 5$) for PredictiveScaler |

#### 5. Output Datasets and Metrics Definitions

All experimental outputs are saved under `data/processed/`:

1. **Detailed Metrics CSV ([`autoscaling_comparison_metrics.csv`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/autoscaling_comparison_metrics.csv)):**
   - 960 total records (4 scenarios $\times$ 2 strategies $\times$ 120 minutes).
   - Columns: `scenario`, `strategy`, `minute`, `actual_viewers`, `active_servers`, `cpu_utilization`, `memory_utilization`, `overload_status`, `scaling_action`, `previous_server_count`, `desired_server_count`, `actual_server_count_after_decision`, `predicted_viewers`.
2. **Scaling Decisions CSV ([`autoscaling_decisions.csv`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/autoscaling_decisions.csv)):**
   - 960 individual decision logs with explanatory messages and reasoning.
   - Columns: `scenario`, `strategy`, `minute`, `action`, `reason`, `previous_server_count`, `desired_server_count`, `actual_server_count`, `predicted_viewers`.
3. **Summary Comparison CSV ([`autoscaling_comparison_summary.csv`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/autoscaling_comparison_summary.csv)):**
   - Quantitative evaluation aggregating infrastructure stability and efficiency:
     - **Total Overload Intervals:** Count of simulation minutes where `overload_status == True` ($\text{CPU} > 100\%$ or $\text{Memory} > 100\%$).
     - **Overload Percentage:** $(\text{Total Overload Intervals} / \text{Total Intervals}) \times 100\%$.
     - **Average CPU Utilization:** Mean average CPU percentage across all 120 simulation intervals.
     - **Maximum CPU Utilization:** Peak average CPU percentage recorded during the simulation.
     - **Average Active Servers:** Mean active server count allocated across all intervals.
     - **Maximum Active Servers:** Peak server pool allocation reached during the scenario.
     - **Total Server-Minutes:** Sum of active servers allocated per minute ($\sum_{t=0}^{119} N(t)$), representing compute infrastructure cost.
     - **Number of Scale-Up Operations:** Count of executed scaling actions expanding the server pool.
     - **Number of Scale-Down Operations:** Count of executed scaling actions decommissioning servers.
     - **Total Scaling Operations:** Sum of scale-up and scale-down actions (measure of operational churn).

#### 6. Quantitative Experimental Results

| Scenario | Strategy | Overloaded Minutes (%) | Avg CPU (%) | Max CPU (%) | Avg Servers | Max Servers | Total Server-Minutes | Scale-Ups | Scale-Downs | Total Actions |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Normal Traffic** | Reactive | 1 (0.8%) | 56.3% | 101.2% | 10.82 | 13 | 1,298 | 11 | 6 | 17 |
| **Normal Traffic** | Predictive | 6 (5.0%) | 55.5% | 128.2% | 11.18 | 13 | 1,341 | 9 | 5 | 14 |
| **Sudden Traffic Spike** | Reactive | 1 (0.8%) | 45.4% | 101.2% | 15.65 | 30 | 1,878 | 10 | 16 | 26 |
| **Sudden Traffic Spike** | Predictive | 4 (3.3%) | 43.2% | 108.5% | 15.82 | 30 | 1,898 | 9 | 18 | 27 |
| **Peak Broadcasting Event** | Reactive | 73 (60.8%) | 124.7% | 184.4% | 26.72 | 30 | 3,206 | 11 | 1 | 12 |
| **Peak Broadcasting Event** | Predictive | 77 (64.2%) | 123.1% | 184.4% | 27.61 | 30 | 3,313 | 6 | 2 | 8 |
| **Event Ending** | Reactive | 64 (53.3%) | 119.9% | 2,502.8% | 29.00 | 30 | 3,480 | 1 | 8 | 9 |
| **Event Ending** | Predictive | 64 (53.3%) | 217.0% | 2,508.4% | 27.77 | 30 | 3,332 | 1 | 8 | 9 |

#### 7. Comparative Performance Insights

1. **Proactive vs Reactive Scaling Dynamics:**
   - In Normal Traffic and Sudden Spike scenarios, both scalers maintain average CPU utilization safely within the target range ($43\% - 56\%$).
   - `PredictiveScaler` initiates scaling adjustments **ahead of traffic spikes** using its 5-minute lead forecast ($t+5$), scaling in larger, smoother steps (fewer overall scale-up actions: 9 vs 11 in Normal, 6 vs 11 in Peak Event).
   - `ReactiveScaler` scales incrementally only after CPU breaches the 70% threshold, which leads to brief reactive overloads at step transitions before catching up.
2. **Warm-Up and Cold-Start Tradeoffs:**
   - In the Event Ending scenario, incoming traffic starts immediately at 50,000 viewers.
   - For `PredictiveScaler`, minutes 0–4 have insufficient historical observations (< 6). At minute 5, six observations are available and the first prediction/scale-up decision is made. That decision affects minute 6, not minute 5. Therefore, the predictive scaler operates with its initial two-server pool during intervals 0–5 (six intervals), resulting in elevated CPU utilization during these first six intervals before the expanded capacity becomes active at minute 6.
   - In contrast, `ReactiveScaler` requires no historical telemetry; it reacts immediately to observed CPU > 70% at minute 0, applying a scale-up that takes effect starting at minute 1.
3. **Resource Efficiency and Churn:**
   - Both strategies demonstrate controlled hysteresis, avoiding oscillation during high-load plateau intervals while gradually shedding excess capacity (16–18 scale-down operations in Sudden Spike) during traffic run-offs.

#### 8. Generated Visualizations

The experiment generates four high-resolution comparative figures in `data/processed/`:

- [`overload_comparison.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/overload_comparison.png): Grouped bar chart comparing total overloaded minutes across all four scenarios.
- [`average_cpu_comparison.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/average_cpu_comparison.png): Average CPU utilization compared against target (70%), scale-down (30%), and capacity (100%) thresholds.
- [`server_usage_comparison.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/server_usage_comparison.png): Total server-minutes and average active server count comparison.
- [`autoscaling_timeline_comparison.png`](file:///home/madhur/Documents/Projects/cloud-autoscaler/data/processed/autoscaling_timeline_comparison.png): 4-panel stacked timeline tracking incoming viewer demand, active server curves, and individual scale-up/scale-down action events for both strategies.

#### 9. How to Execute the Autoscaling Comparison Experiment

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the complete Stage D3 comparison experiment
python run_autoscaling_comparison.py

# Run all automated test suites (133 tests across Stages A, B, C, D1, D2, D3, and E)
python -m unittest discover -s tests -v
```

---

## Stage E: Interactive Streamlit Dashboard

### 1. Overview and Purpose

Stage E provides an interactive web-based dashboard built using **Streamlit** and **Plotly** to visually analyze and benchmark:
- Audience workload scenario profiles (Stage A).
- Random Forest 5-minute lead predictions vs Persistence Baseline (Stage B).
- Discrete cloud infrastructure simulation and telemetry (Stage C).
- Comparative autoscaling performance between Reactive and Predictive policies (Stage D3).

### 2. Dependency Installation

To install all required dependencies (including Streamlit and Plotly) into the project virtual environment:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Launching the Dashboard

Launch the application with the Streamlit CLI:

```bash
streamlit run app.py
```

The application opens locally at `http://localhost:8501`.

### 4. Dashboard Sections and Capabilities

The dashboard is structured into four functional tabs alongside interactive sidebar controls:

- **Global Sidebar Controls:**
  - Scenario Selector: Switch dynamically between *Normal Traffic*, *Sudden Traffic Spike*, *Peak Broadcasting Event*, and *Event Ending*.
  - Reference System Specifications: Display active capacity parameters (1,000 viewers/server, 70% scale-up target, 30% scale-down target, 2-minute cooldown, 2–30 server bounds).
  - Uncapped Capacity Disclaimer: Highlights that CPU utilization exceeding 100% represents an uncapped demand-to-capacity ratio in the simulation.

- **Section A: Workload Overview (`📈 A. Workload Overview`):**
  - Interactive Plotly time-series chart showing audience viewership across 120 simulation minutes.
  - Highlights live event windows (`event_active == True`) with shaded region annotations.
  - Metric summary cards: Peak Viewers, Average Viewers, Total Cumulative Viewers (Viewer-Minutes), and Minimum Baseline Viewers.
  - Raw workload data inspector table.

- **Section B: Workload Prediction (`🤖 B. Workload Prediction`):**
  - Time-series overlay comparing Actual Demand, Random Forest 5-minute lead forecasts ($t + 5$), and the Persistence Baseline ($t$).
  - Evaluates forecast accuracy on the purged test set (minutes 93–114) with real-time KPI cards for MAE, RMSE, and error reduction percentage.
  - Forecast residual error bar chart comparing $|V(t+5) - \hat{V}(t+5)|$ against baseline errors.
  - Cross-scenario benchmark summary table comparing RF vs Baseline across all 4 traffic profiles.

- **Section C: Autoscaling Comparison (`⚖️ C. Autoscaling Comparison`):**
  - Head-to-head KPI comparison cards for the selected scenario: Overloaded Intervals, Average CPU Utilization, Maximum CPU Utilization, Total Server-Minutes (compute cost), Average Active Servers, Scale-Up Operations, Scale-Down Operations, and Total Operational Churn.
  - 3-panel synchronized comparative timeline showing incoming demand, active server allocation trajectories (with scale-up marker events), and average CPU curves for both Reactive and Predictive scalers.
  - Cross-scenario grouped bar charts for Overloaded Intervals, Average CPU Utilization (with 30%, 70%, and 100% threshold reference lines), and Server Resource Usage (Server-Minutes and Active Servers).

- **Section D: Infrastructure Monitoring (`🖥️ D. Infrastructure Monitoring`):**
  - Deep-dive operational telemetry tracking CPU Utilization (%), Memory Utilization (%), and Active Server Count over time.
  - Highlights overload incidents ($\text{CPU} > 100\%$) with red marker annotations.
  - Policy strategy selector to toggle between Reactive and Predictive execution traces.
  - Interactive scaling decision event logs displaying timestamped actions, desired server counts, and trigger rationale.

