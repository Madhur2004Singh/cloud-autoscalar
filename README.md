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
│       └── all_scenarios_comparison.png
├── simulation/
│   ├── __init__.py
│   └── workload_generator.py
├── tests/
│   ├── __init__.py
│   └── test_workload_generator.py
├── .gitignore
├── main.py
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
