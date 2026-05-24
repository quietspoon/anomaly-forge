# AnomalyForge — Granular Implementation Plan

## Context

AnomalyForge is a 4-phase time-series anomaly detection and synthesis pipeline, built as a Streamlit demo for Rockfish (synthetic data startup). The goal is to **prove** that synthetic anomaly-augmented training data measurably improves rare-event detection in ML models — with before/after benchmarks shown live in the UI.

**Why this plan exists:** The project repo (`/Users/monoid/Documents/development/anomaly_forge/anomaly-forge/`) has only scaffolding (CLAUDE.md, README, LICENSE). All source code must be built from scratch. The build order is strict (types → utils → profiler → synthesizer → benchmarker → discoverer → app) because each module depends on the previous one.

**Target outcome:** All 4 pytest suites pass in Docker, Streamlit runs end-to-end on Yahoo S5 A1Benchmark CSV, and the benchmark shows measurable Recall improvement for augmented training.

**Scope decision:** Benchmark model is **IsolationForest only** (sklearn). The LSTM Autoencoder is removed to keep the implementation simpler and faster. No PyTorch training loop, no CUDA/CPU torch complexity in benchmarker.

---

## Repository Info

- **Repo path:** `/Users/monoid/Documents/development/anomaly_forge/anomaly-forge/`
- **Remote:** `https://github.com/quietspoon/anomaly-forge.git`
- **Branch:** `main`

---

## Orchestration Protocol (MANDATORY for every commit task)

The **main thread** owns git operations and orchestration. It **never writes code directly**.

For **each commit task**, the sequence is:

```
1. Main thread spawns WRITER AGENT
   → Writer reads ALL relevant existing files before writing anything
   → Writer returns complete file(s)
   → Main thread writes files to disk

2. Main thread spawns TESTER AGENT
   → Tester runs exact Docker command specified in task
   → Tester returns: PASS (with stdout) or FAIL (with full error output)

3. Decision:
   PASS → main thread runs: git add -A && git commit -m "<message>" && git push origin main
   FAIL → main thread sends error output back to Writer Agent for targeted fix → repeat from 1

4. Never proceed to next task until Docker tests are green.
```

**Main thread responsibilities:** Orchestration only + final E2E test after all commits.

**Writer agents:** Read existing code → implement → return files.

**Tester agents:** Run Docker tests → parse output → report PASS/FAIL with details.

---

## Docker Testing Policy

**All tests run inside Docker. Zero local test execution.**

Build the image once (Task 1), then all subsequent tasks mount source code as a volume so code changes don't require rebuilds during development.

```bash
# Build image (once):
docker-compose build

# Test command for each task (volume-mounted):
docker-compose run --rm test pytest tests/<test_file>.py -v --tb=short

# Final E2E test (all suites):
docker-compose run --rm test pytest tests/ -v --tb=short
```

---

## Tech Stack

```
Python 3.11+ | pandas | numpy | scipy | statsmodels
pyod (IForest + KNN ensemble) | scikit-learn
dtaidistance (DTW) | streamlit | matplotlib | seaborn | pytest
```
> **Note:** `torch` is NOT used in the benchmark. It can remain in `requirements.txt` for future use but is not imported by any module in scope.

**Key research findings applied to implementation:**
- `MPLBACKEND=Agg` in Docker environment for headless matplotlib
- DTW distance matrix via `dtaidistance.dtw.distance_matrix_fast` with C extension fallback
- `AgglomerativeClustering(metric='precomputed', linkage='average')` for shape clustering
- IsolationForest anomaly score = `-model.decision_function(windows)` (higher = more anomalous)
- STL period auto-inferred from `DatetimeIndex.freq` using `freq_to_period` map
- **No PyTorch dependency in benchmarker** — torch removed from the benchmarker entirely

---

## File Structure (Final State)

```
anomaly-forge/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── requirements.txt
├── app.py
├── plan.md                       ← written at the end by main thread
├── data/
│   ├── raw/.gitkeep
│   └── synthetic/.gitkeep
├── src/
│   ├── __init__.py
│   ├── types.py
│   ├── utils.py
│   ├── profiler.py
│   ├── discoverer.py
│   ├── synthesizer.py
│   └── benchmarker.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_utils.py
│   ├── test_profiler.py
│   ├── test_synthesizer.py
│   ├── test_benchmarker.py
│   └── test_discoverer.py
└── notebooks/
    └── exploration.ipynb
```

---

## Dependency Graph

```
Task 1 (Scaffolding)
  └── Task 2 (types.py)
        └── Task 3 (utils.py)
              ├── Task 4 (test_utils.py)
              └── Task 5 (profiler.py)
                    └── Task 6 (test_profiler.py)
                          └── Task 7 (synthesizer.py)
                                └── Task 8 (test_synthesizer.py)
                                      └── Task 9 (benchmarker.py)
                                            └── Task 10 (test_benchmarker.py)
                                                  └── Task 11 (discoverer.py)
                                                        └── Task 12 (test_discoverer.py + app.py)
                                                              └── FINAL E2E TEST (main thread)
```

Each task blocked on its predecessor being green in Docker.

---

## Task 1 — Project Scaffolding

**Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.gitignore`, `requirements.txt`, `src/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `data/raw/.gitkeep`, `data/synthetic/.gitkeep`, `notebooks/.gitkeep`

### Writer Agent Context
```
Read: CLAUDE.md, README.md

Create the following files:

1. requirements.txt — list all packages without pinned versions:
   pandas, numpy, scipy, statsmodels, pyod, scikit-learn, dtaidistance,
   streamlit, matplotlib, seaborn, pytest, pytest-cov
   torch is NOT included (benchmarker is IForest-only). Optionally add as a comment:
   # torch>=2.2.0  # optional; install with: pip install torch --index-url https://download.pytorch.org/whl/cpu

2. Dockerfile — (python:3.11-slim base):
   FROM python:3.11-slim
   RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libgomp1 \
       && rm -rf /var/lib/apt/lists/*
   ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=/app
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["pytest", "tests/", "-v", "--tb=short"]
   # Note: torch is NOT installed in Docker — benchmarker uses IForest only (sklearn).

3. docker-compose.yml:
   services:
     test:
       build: .
       volumes: [".:/app"]
       environment: [MPLBACKEND=Agg, PYTHONDONTWRITEBYTECODE=1]
       command: pytest tests/ -v --tb=short
     app:
       build: .
       ports: ["8501:8501"]
       volumes: [".:/app"]
       environment: [MPLBACKEND=Agg, STREAMLIT_SERVER_HEADLESS=true,
                     STREAMLIT_SERVER_FILE_WATCHER_TYPE=none]
       command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0

4. .dockerignore: __pycache__, .git, *.pyc, data/raw/*, data/synthetic/*,
   .DS_Store, notebooks/, .pytest_cache, .venv/

5. .gitignore: __pycache__, *.pyc, .env, data/raw/*, data/synthetic/*,
   .DS_Store, .pytest_cache, *.egg-info, dist/, build/, .venv/, venv/

6. src/__init__.py — empty
7. tests/__init__.py — empty
8. tests/conftest.py:
   import numpy as np, import pandas as pd, import pytest
   @pytest.fixture
   def sample_series() -> pd.Series:
       idx = pd.date_range("2020-01-01", periods=500, freq="D")
       rng = np.random.default_rng(42)
       values = np.sin(np.linspace(0, 8 * np.pi, 500)) + rng.normal(0, 0.1, 500)
       return pd.Series(values, index=idx, name="test_series")

9. data/raw/.gitkeep, data/synthetic/.gitkeep, notebooks/.gitkeep — empty
```

### Tester Agent Instructions
```
Run: docker-compose build
Then: docker-compose run --rm test python -c "import pandas, numpy, scipy, statsmodels, pyod, sklearn, dtaidistance, streamlit, matplotlib, seaborn, pytest, torch; print('ALL IMPORTS OK')"
Then: docker-compose run --rm test pytest tests/ -v --tb=short

Pass: docker build succeeds + import check prints "ALL IMPORTS OK" + pytest exits 0
Report full stdout/stderr for any failure.
```

### Success Criteria
- Docker image builds without error
- All required libraries importable inside container
- `pytest tests/` exits 0 (collects conftest, no tests yet)

### Git Commit Message
```
feat: project scaffolding — Dockerfile, docker-compose, requirements, conftest
```

---

## Task 2 — `src/types.py`

**Files:** `src/types.py`

### Writer Agent Context
```
Read: CLAUDE.md (full), tests/conftest.py

Create src/types.py with these 6 dataclasses (from __future__ import annotations at top):

@dataclass class TimeSeriesProfile:
  series_name: str, n_observations: int, start_time: pd.Timestamp, end_time: pd.Timestamp,
  inferred_freq: str, distribution_name: str, distribution_params: tuple[float, ...],
  has_seasonality: bool, seasonal_period: int, seasonal_strength: float, noise_std: float,
  acf_lags: list[float], pacf_lags: list[float], existing_anomaly_indices: list[int],
  trend_slope: float

@dataclass class AnomalyCandidate:
  candidate_id: str, anomaly_type: str, representative_window: np.ndarray,
  window_indices: list[int], frequency_count: int, severity_score: float,
  confirmed: bool = False

@dataclass class SynthesisConfig:
  n_spike_events: int = 5, n_drift_events: int = 3, n_flatline_events: int = 2,
  n_noise_burst_events: int = 3, magnitude_multiplier: float = 3.0,
  drift_duration: int = 20, flatline_duration: int = 15, noise_burst_duration: int = 10,
  noise_burst_multiplier: float = 5.0, random_seed: int = 42

@dataclass class SyntheticDataset:
  original_series: pd.Series, synthetic_series: pd.Series, labels: np.ndarray,
  anomaly_type_labels: list[str], injection_metadata: list[dict],
  n_injected_events: int, synthesis_config: SynthesisConfig

@dataclass class BenchmarkConfig:
  test_ratio: float = 0.2, window_size: int = 64, window_step: int = 1,
  iforest_contamination: float = 0.1, iforest_n_estimators: int = 100,
  iforest_threshold_percentile: float = 95.0,
  random_seed: int = 42

@dataclass class BenchmarkResult:
  # IsolationForest only — real-only vs augmented
  iforest_real_precision: float, iforest_real_recall: float, iforest_real_f1: float,
  iforest_real_auc: float,
  iforest_augmented_precision: float, iforest_augmented_recall: float,
  iforest_augmented_f1: float, iforest_augmented_auc: float,
  # Metadata
  n_test_windows: int, n_anomalous_test_windows: int,
  iforest_real_threshold: float, iforest_augmented_threshold: float,
  # ROC curve arrays (for plotting)
  real_fpr: list[float], real_tpr: list[float],
  augmented_fpr: list[float], augmented_tpr: list[float]

Module docstring: "Shared dataclasses for AnomalyForge inter-module data contracts."
No logic, no print statements, no methods beyond optional __post_init__.
Imports: from __future__ import annotations, from dataclasses import dataclass field,
         numpy as np, pandas as pd
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test python -c \
  "from src.types import TimeSeriesProfile, AnomalyCandidate, SynthesisConfig, \
   SyntheticDataset, BenchmarkConfig, BenchmarkResult; print('ALL 6 TYPES OK')"

Pass: prints "ALL 6 TYPES OK", exit code 0.
```

### Success Criteria
- All 6 dataclasses importable, all fields annotated, no syntax errors

### Git Commit Message
```
feat(types): define all 6 dataclasses — data contracts for inter-module interfaces
```

---

## Task 3 — `src/utils.py`

**Files:** `src/utils.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/types.py

Create src/utils.py with exactly 5 public functions. All functions:
- Complete type hints (inputs + return type)
- Docstring: one-line + Args + Returns
- No print statements; use: logger = logging.getLogger(__name__)
- No plt.show() calls anywhere

MODULE HEADER:
import logging, matplotlib, matplotlib.pyplot as plt, numpy as np
import pandas as pd, seaborn as sns
from sklearn.preprocessing import StandardScaler
matplotlib.use("Agg")  # REQUIRED for Docker headless
logger = logging.getLogger(__name__)

FUNCTION SIGNATURES:

def sliding_window(series: pd.Series, window_size: int, step: int = 1) -> np.ndarray:
  Use np.lib.stride_tricks.sliding_window_view then slice by step.
  Shape returned: (n_windows, window_size). n_windows = (len-window_size)//step + 1
  Raises ValueError if window_size > len(series) or step < 1.

def normalize(series: pd.Series) -> tuple[pd.Series, StandardScaler]:
  Z-score normalize. Use StandardScaler.fit_transform(values.reshape(-1,1)).flatten()
  Return (pd.Series with same index, fitted_scaler).
  Raises ValueError if series is empty.

def temporal_split(series: pd.Series, test_ratio: float = 0.2) -> tuple[pd.Series, pd.Series]:
  split_idx = int(len(series) * (1 - test_ratio)). No randomness.
  Return (series[:split_idx], series[split_idx:]).
  Raises ValueError if test_ratio not in (0,1) or len < 4.

def plot_series_with_anomalies(series: pd.Series, labels: np.ndarray, 
                                title: str = "Time Series with Anomalies") -> matplotlib.figure.Figure:
  Blue line for full series; red scatter at anomaly points (where labels==1).
  Figure size (14, 4). Return fig. Do NOT call plt.show().
  Raises ValueError if len(labels) != len(series).

def plot_roc_comparison(real_fpr, real_tpr, real_auc, augmented_fpr, augmented_tpr, 
                         augmented_auc, model_name="Model") -> matplotlib.figure.Figure:
  Two subplots side by side. Each: diagonal dashed line + ROC curve + AUC in legend.
  Figure size (12, 5). Return fig. Do NOT call plt.show().
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test python -c \
  "from src.utils import sliding_window, normalize, temporal_split, \
   plot_series_with_anomalies, plot_roc_comparison; print('ALL 5 UTILS OK')"

Pass: prints "ALL 5 UTILS OK", exit 0.
Fail: full traceback.
```

### Success Criteria
- All 5 functions import cleanly, no `plt.show()` calls, `matplotlib.use("Agg")` at module level

### Git Commit Message
```
feat(utils): sliding_window, normalize, temporal_split, plot helpers
```

---

## Task 4 — `tests/test_utils.py`

**Files:** `tests/test_utils.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/utils.py, src/types.py, tests/conftest.py

Create tests/test_utils.py. Use pytest. sample_series fixture auto-discovered.
At top: import numpy as np, import pandas as pd, import pytest, import matplotlib
from src.utils import sliding_window, normalize, temporal_split,
                      plot_series_with_anomalies, plot_roc_comparison

TEST CASES (all use sample_series fixture):

sliding_window:
  test_sliding_window_shape: windows.shape == (500-64+1, 64) for step=1
  test_sliding_window_step: check n_windows for step=32
  test_sliding_window_values: 5-element series, verify first/last windows
  test_sliding_window_raises_on_large_window: window_size=1000 → ValueError
  test_sliding_window_raises_on_bad_step: step=0 → ValueError

normalize:
  test_normalize_mean_zero: abs(normed.mean()) < 1e-10
  test_normalize_std_one: abs(normed.std() - 1.0) < 1e-6
  test_normalize_returns_scaler: hasattr(scaler, "inverse_transform")
  test_normalize_invert: inverse_transform matches original values (atol=1e-10)
  test_normalize_preserves_index: index unchanged
  test_normalize_raises_on_empty: pd.Series([], dtype=float) → ValueError

temporal_split:
  test_temporal_split_ratio: len(train)+len(test)==500, len(test)==100
  test_temporal_split_no_shuffle: train.index[-1] < test.index[0]
  test_temporal_split_coverage: pd.concat([train,test]) == sample_series
  test_temporal_split_raises_bad_ratio: 0.0 and 1.0 both raise ValueError
  test_temporal_split_raises_short_series: 2-element series → ValueError

plot_series_with_anomalies:
  test_plot_returns_figure: isinstance(fig, matplotlib.figure.Figure)
  test_plot_raises_mismatched_labels: len=10 labels for len=500 series → ValueError

plot_roc_comparison:
  test_roc_comparison_returns_figure: isinstance(fig, matplotlib.figure.Figure)
  test_roc_comparison_two_axes: len(fig.axes) == 2

Add plt.close("all") after each plotting test.
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test pytest tests/test_utils.py -v --tb=short

Pass: All ~18 tests collected, 0 failures, 0 errors.
Report: exact failure messages and tracebacks if any fail.
```

### Success Criteria
- All tests green in Docker, no display/matplotlib warnings

### Git Commit Message
```
test(utils): full unit test suite for all 5 utility functions
```

---

## Task 5 — `src/profiler.py`

**Files:** `src/profiler.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/types.py, src/utils.py, tests/conftest.py

Create src/profiler.py. Single public function + private helpers. All ≤ 40 lines each.

MODULE HEADER:
import logging, numpy as np, pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from src.types import TimeSeriesProfile
logger = logging.getLogger(__name__)

PUBLIC FUNCTION:
def profile_series(series: pd.Series) -> TimeSeriesProfile:
  Validates: DatetimeIndex, no NaN, len >= 50. Raises ValueError with descriptive message.
  Returns TimeSeriesProfile from helpers below.

PRIVATE HELPERS:

def _infer_period(series: pd.Series) -> int:
  freq_name = series.index.freqstr or series.index.inferred_freq or ""
  freq_to_period = {"D": 365, "H": 24, "W": 52, "MS": 12, "M": 12, "T": 60, "min": 60, "QS": 4}
  Map freq_name prefix to period. Default 7. Return int.

def _fit_distribution(values: np.ndarray) -> tuple[str, tuple]:
  Candidates: scipy_stats.norm, scipy_stats.lognorm, scipy_stats.expon
  For lognorm: shift values if min <= 0 (values += abs(min)+1)
  For each: fit params, compute log-likelihood, AIC = 2*k - 2*logL
  Return (name of lowest AIC, params).

def _run_stl(series: pd.Series, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  If period < 2: return (series.values, np.zeros_like(series), series.values - series.mean())
  stl = STL(series, period=period, robust=True).fit()
  Return (stl.trend, stl.seasonal, stl.resid).

def _compute_seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
  denom = np.var(seasonal) + np.var(residual)
  Return 0.0 if denom == 0 else float(np.clip(np.var(seasonal)/denom, 0.0, 1.0))

def _compute_acf_pacf(residual: np.ndarray, n_lags: int = 20) -> tuple[list, list]:
  acf_vals = acf(residual, nlags=n_lags, fft=True)[1:]  # exclude lag 0
  pacf_vals = pacf(residual, nlags=n_lags, method="ols")[1:]
  Return (list(acf_vals), list(pacf_vals))

def _flag_existing_anomalies(residual: np.ndarray, noise_std: float) -> list[int]:
  If noise_std == 0: return []
  Return [int(i) for i, v in enumerate(residual) if abs(v) > 3.5 * noise_std]

def _compute_trend_slope(trend: np.ndarray) -> float:
  Return float(np.polyfit(np.arange(len(trend)), trend, 1)[0])

ASSEMBLY in profile_series:
1. Validate (DatetimeIndex → "series must have a DatetimeIndex", NaN → "series contains NaN",
   len<50 → "series must have at least 50 observations")
2. freq_str = series.index.freqstr or series.index.inferred_freq or "unknown"
3. period = _infer_period(series)
4. dist_name, dist_params = _fit_distribution(series.values)
5. trend, seasonal, residual = _run_stl(series, period)
6. seasonal_strength = _compute_seasonal_strength(seasonal, residual)
7. has_seasonality = seasonal_strength > 0.1
8. acf_lags, pacf_lags = _compute_acf_pacf(residual)
9. noise_std = float(np.std(residual))
10. anomaly_indices = _flag_existing_anomalies(residual, noise_std)
11. trend_slope = _compute_trend_slope(trend)
12. Return TimeSeriesProfile(series_name=series.name or "", n_observations=len(series),
    start_time=series.index[0], end_time=series.index[-1], inferred_freq=freq_str,
    distribution_name=dist_name, distribution_params=dist_params,
    has_seasonality=has_seasonality, seasonal_period=period if has_seasonality else 0,
    seasonal_strength=seasonal_strength, noise_std=noise_std, acf_lags=acf_lags,
    pacf_lags=pacf_lags, existing_anomaly_indices=anomaly_indices, trend_slope=trend_slope)
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test python -c \
  "from src.profiler import profile_series; print('PROFILER IMPORT OK')"

Pass: prints "PROFILER IMPORT OK", exit 0.
```

### Success Criteria
- `profile_series` imports cleanly, returns `TimeSeriesProfile` on valid input

### Git Commit Message
```
feat(profiler): STL decomposition, distribution fitting, ACF/PACF, anomaly flagging
```

---

## Task 6 — `tests/test_profiler.py`

**Files:** `tests/test_profiler.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/profiler.py, src/types.py, tests/conftest.py

Create tests/test_profiler.py.
Imports: import numpy as np, import pandas as pd, import pytest
         from src.profiler import profile_series
         from src.types import TimeSeriesProfile

HAPPY PATH TESTS (use sample_series fixture):
  test_profile_returns_correct_type: isinstance(profile, TimeSeriesProfile)
  test_profile_n_observations: profile.n_observations == 500
  test_profile_inferred_freq: profile.inferred_freq in ("D", "day", "<Day>")
  test_profile_distribution_name: in ("norm", "lognorm", "expon")
  test_profile_distribution_params_not_empty: len(profile.distribution_params) >= 2
  test_profile_has_seasonality_bool: isinstance(profile.has_seasonality, bool)
  test_profile_seasonal_strength_range: 0.0 <= profile.seasonal_strength <= 1.0
  test_profile_noise_std_positive: profile.noise_std > 0.0
  test_profile_acf_length: len(profile.acf_lags) == 20
  test_profile_pacf_length: len(profile.pacf_lags) == 20
  test_profile_existing_anomaly_indices_list: isinstance(..., list)
  test_profile_start_end_time: matches series.index[0] and series.index[-1]
  test_profile_trend_slope_float: isinstance(profile.trend_slope, float)

ANOMALY DETECTION:
  test_profile_flags_obvious_spike:
    200-point daily series, values[100] = 50.0 (massive spike)
    assert 100 in profile.existing_anomaly_indices

ERROR CASES:
  test_profile_raises_on_non_datetime_index:
    pd.Series(np.random.randn(100)) (RangeIndex) → ValueError matching "DatetimeIndex"
  test_profile_raises_on_short_series:
    10-point DatetimeSeries → ValueError matching "50"
  test_profile_raises_on_nan:
    100-point series with s.iloc[50] = np.nan → ValueError matching "NaN"
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test pytest tests/test_profiler.py -v --tb=short

Pass: All ~16 tests collected, 0 failures.
MUST PASS: test_profile_flags_obvious_spike, all 3 error case tests.
Report full failure output if any fail.
```

### Success Criteria
- All tests green, spike detection test and all error tests pass

### Git Commit Message
```
test(profiler): happy path, anomaly detection, and all error cases
```

---

## Task 7 — `src/synthesizer.py`

**Files:** `src/synthesizer.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/types.py, src/utils.py, src/profiler.py

Create src/synthesizer.py. Single public function + private helpers.

MODULE HEADER:
import logging, numpy as np, pandas as pd
from src.types import TimeSeriesProfile, AnomalyCandidate, SynthesisConfig, SyntheticDataset
logger = logging.getLogger(__name__)

PUBLIC FUNCTION:
def synthesize_anomalies(series, profile, confirmed_candidates, config) -> SyntheticDataset:
  NEVER mutate input series. Work on a copy.
  Validates: series not empty, config magnitudes > 0.
  Returns SyntheticDataset.

CRITICAL INVARIANTS:
  - original_series in output == input series (no mutation)
  - len(synthetic_series) == len(series)
  - len(labels) == len(series)
  - every injected event appears in labels AND injection_metadata

PRIVATE HELPERS:

def _choose_injection_sites(series, n_events, duration, rng, occupied) -> list[int]:
  Valid start indices: positions where [i, i+duration) doesn't overlap occupied.
  Sample without replacement up to n_events. Return up to n_events sites (may be fewer).
  If 0 valid sites, return [].

def _inject_spike(values, labels, type_labels, start_idx, local_std, magnitude_multiplier, rng) -> dict:
  spike = values[start_idx] + magnitude_multiplier * local_std * rng.choice([-1, 1])
  values[start_idx] = spike; labels[start_idx] = 1; type_labels[start_idx] = "spike"
  Return {"type": "spike", "start_idx": start_idx, "end_idx": start_idx, "magnitude": magnitude_multiplier*local_std}

def _inject_drift(values, labels, type_labels, start_idx, duration, local_std, magnitude_multiplier) -> dict:
  end = min(start_idx+duration, len(values))
  ramp = np.linspace(values[start_idx], values[start_idx]+magnitude_multiplier*local_std, end-start_idx)
  values[start_idx:end] = ramp; labels[start_idx:end] = 1
  Set type_labels[start_idx:end] = "drift"
  Return {"type": "drift", "start_idx": start_idx, "end_idx": end-1, "magnitude": magnitude_multiplier*local_std}

def _inject_flatline(values, labels, type_labels, start_idx, duration) -> dict:
  end = min(start_idx+duration, len(values))
  values[start_idx:end] = values[start_idx]  # hold constant
  labels[start_idx:end] = 1; type_labels[start_idx:end] = "flatline"
  Return {"type": "flatline", "start_idx": start_idx, "end_idx": end-1, "magnitude": 0.0}

def _inject_noise_burst(values, labels, type_labels, start_idx, duration, local_std, noise_burst_multiplier, rng) -> dict:
  end = min(start_idx+duration, len(values))
  noise = rng.normal(0, noise_burst_multiplier*local_std, end-start_idx)
  values[start_idx:end] += noise; labels[start_idx:end] = 1
  type_labels[start_idx:end] = "noise_burst"
  Return {"type": "noise_burst", "start_idx": start_idx, "end_idx": end-1, "magnitude": noise_burst_multiplier*local_std}

ASSEMBLY in synthesize_anomalies:
1. Validate inputs
2. values = series.values.copy(); labels = np.zeros(len(series), dtype=int)
   type_labels = [""] * len(series)
3. rng = np.random.default_rng(config.random_seed); occupied = set(); all_metadata = []
4. local_std_arr = pd.Series(values).rolling(window=32, min_periods=1).std().fillna(profile.noise_std).values
5. For each type: sites = _choose_injection_sites(...); for each site: inject, update occupied, append metadata
   Injection order: spike, drift, flatline, noise_burst
   For spike: duration=1; for others: use config duration
6. After all injections: build SyntheticDataset and return
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test python -c \
  "from src.synthesizer import synthesize_anomalies; print('SYNTHESIZER IMPORT OK')"

Pass: prints "SYNTHESIZER IMPORT OK", exit 0.
```

### Success Criteria
- `synthesize_anomalies` imports cleanly, no mutation of input on call

### Git Commit Message
```
feat(synthesizer): 4 injection types (spike, drift, flatline, noise_burst), ground-truth labels
```

---

## Task 8 — `tests/test_synthesizer.py`

**Files:** `tests/test_synthesizer.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/synthesizer.py, src/types.py, src/profiler.py, tests/conftest.py

Create tests/test_synthesizer.py.
Imports: numpy as np, pandas as pd, pytest
         from src.synthesizer import synthesize_anomalies
         from src.profiler import profile_series
         from src.types import SyntheticDataset, SynthesisConfig

ADD module-level fixture:
@pytest.fixture
def synthesis_inputs(sample_series):
    profile = profile_series(sample_series)
    config = SynthesisConfig(n_spike_events=2, n_drift_events=1, n_flatline_events=1,
                              n_noise_burst_events=1, magnitude_multiplier=3.0,
                              drift_duration=10, flatline_duration=8,
                              noise_burst_duration=6, noise_burst_multiplier=5.0, random_seed=42)
    return sample_series, profile, [], config

INVARIANT TESTS:
  test_output_type: isinstance(result, SyntheticDataset)
  test_output_length_invariant: len(synthetic_series)==len(series), len(labels)==len(series),
                                  len(anomaly_type_labels)==len(series)
  test_no_mutation_of_input: original_values = series.values.copy(); after call, assert unchanged
  test_original_series_preserved_in_output: pd.testing.assert_series_equal(result.original_series, series)
  test_index_preserved: pd.testing.assert_index_equal(result.synthetic_series.index, series.index)

LABEL COVERAGE:
  test_labels_are_binary: set(np.unique(labels)).issubset({0, 1})
  test_at_least_one_anomaly_injected: labels.sum() > 0
  test_n_injected_events_matches_metadata: result.n_injected_events == len(result.injection_metadata)
  test_metadata_has_required_keys: each dict has "type", "start_idx", "end_idx"
  test_type_labels_at_anomaly_indices: anomaly type label non-empty wherever labels==1
  test_normal_indices_have_empty_type_label: spot check first 50 normal indices

INJECTION TYPE COVERAGE:
  test_spike_type_present: "spike" in [m["type"] for m in injection_metadata]
  test_drift_type_present: "drift" in [m["type"] for m in injection_metadata]

ERROR CASES:
  test_raises_on_empty_series: 0-length series → ValueError
  test_synthesis_config_zero_events_is_ok: all n_*_events=0 → labels.sum()==0, n_injected_events==0
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test pytest tests/test_synthesizer.py -v --tb=short

Pass: All ~15 tests, 0 failures.
MUST PASS: test_no_mutation_of_input, test_output_length_invariant.
Report full tracebacks on failure.
```

### Success Criteria
- All tests green; no-mutation and length invariant tests pass specifically

### Git Commit Message
```
test(synthesizer): no-mutation, length invariant, label coverage, all 4 injection types
```

---

## Task 9 — `src/benchmarker.py`

**Files:** `src/benchmarker.py`

**Scope:** IsolationForest only. No LSTM, no PyTorch. Two variants: real-only training vs real+synthetic training, evaluated on the same held-out test set.

### Writer Agent Context
```
Read: CLAUDE.md, src/types.py, src/utils.py

Create src/benchmarker.py. Single public function + private helpers. No torch, no LSTM.

MODULE HEADER:
import logging, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from src.types import SyntheticDataset, BenchmarkConfig, BenchmarkResult
from src.utils import sliding_window, temporal_split
logger = logging.getLogger(__name__)

PRIVATE HELPERS:

def _make_windows_and_labels(
    series: pd.Series,
    labels_arr: np.ndarray,
    window_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Create (windows, per-window binary labels) from series + sample-level labels.
  Window label = 1 if ANY sample in that window is labeled anomalous, else 0.
  windows shape: (n_windows, window_size).
  """
  windows = sliding_window(series, window_size, step)
  n = windows.shape[0]
  window_labels = np.array([
      int(labels_arr[i * step : i * step + window_size].any())
      for i in range(n)
  ])
  return windows, window_labels

def _train_iforest(
    train_windows: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[IsolationForest, float]:
  """Fit IsolationForest. Return (model, threshold).
  Threshold = percentile(train_anomaly_scores, iforest_threshold_percentile).
  Scores = -model.decision_function(train_windows) so higher = more anomalous.
  """
  model = IsolationForest(
      n_estimators=config.iforest_n_estimators,
      contamination=config.iforest_contamination,
      random_state=config.random_seed,
  )
  model.fit(train_windows)
  train_scores = -model.decision_function(train_windows)
  threshold = float(np.percentile(train_scores, config.iforest_threshold_percentile))
  return model, threshold

def _compute_metrics(
    scores: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float, list[float], list[float]]:
  """Compute precision, recall, F1, AUC-ROC, fpr, tpr.
  pred = (scores >= threshold).astype(int).
  Handle edge case: if true_labels are all one class, return auc=0.0.
  Return (precision, recall, f1, auc, fpr_list, tpr_list).
  """
  pred = (scores >= threshold).astype(int)
  try:
      auc = float(roc_auc_score(true_labels, scores))
      fpr_arr, tpr_arr, _ = roc_curve(true_labels, scores)
      fpr_list, tpr_list = fpr_arr.tolist(), tpr_arr.tolist()
  except ValueError:
      auc = 0.0
      fpr_list, tpr_list = [0.0, 1.0], [0.0, 1.0]
  precision = float(precision_score(true_labels, pred, zero_division=0))
  recall = float(recall_score(true_labels, pred, zero_division=0))
  f1 = float(f1_score(true_labels, pred, zero_division=0))
  return precision, recall, f1, auc, fpr_list, tpr_list

PUBLIC FUNCTION:
def run_benchmark(dataset: SyntheticDataset, config: BenchmarkConfig) -> BenchmarkResult:
  """Train and evaluate IsolationForest on real-only vs real+synthetic training data.

  Args:
      dataset: SyntheticDataset from synthesizer.py.
      config: Benchmark configuration.

  Returns:
      BenchmarkResult with before/after IForest metrics.

  Raises:
      ValueError: If dataset has fewer than 100 observations.
      ValueError: If test set has no anomalous windows (benchmark is meaningless).
  """
  ASSEMBLY:
  1. Validate len(dataset.original_series) >= 100
  2. split_idx = int(len(dataset.original_series) * (1 - config.test_ratio))
     train_orig = dataset.original_series.iloc[:split_idx]
     test_orig  = dataset.original_series.iloc[split_idx:]
     train_labels = dataset.labels[:split_idx]
     test_labels  = dataset.labels[split_idx:]
     synth_train  = dataset.synthetic_series.iloc[:split_idx]
  3. test_windows, test_window_labels = _make_windows_and_labels(test_orig, test_labels, ...)
  4. Validate test_window_labels.sum() > 0 else ValueError("no anomalous windows in test set")
  5. train_real_windows, _  = _make_windows_and_labels(train_orig, train_labels, ...)
     train_synth_windows, _ = _make_windows_and_labels(synth_train, train_labels, ...)
  6. iforest_real,  threshold_real  = _train_iforest(train_real_windows, config)
     iforest_synth, threshold_synth = _train_iforest(train_synth_windows, config)
  7. real_scores  = -iforest_real.decision_function(test_windows)
     synth_scores = -iforest_synth.decision_function(test_windows)
  8. p_r, r_r, f_r, auc_r, fpr_r, tpr_r = _compute_metrics(real_scores,  test_window_labels, threshold_real)
     p_a, r_a, f_a, auc_a, fpr_a, tpr_a = _compute_metrics(synth_scores, test_window_labels, threshold_synth)
  9. Return BenchmarkResult(
         iforest_real_precision=p_r, iforest_real_recall=r_r, iforest_real_f1=f_r, iforest_real_auc=auc_r,
         iforest_augmented_precision=p_a, iforest_augmented_recall=r_a, iforest_augmented_f1=f_a, iforest_augmented_auc=auc_a,
         n_test_windows=len(test_windows), n_anomalous_test_windows=int(test_window_labels.sum()),
         iforest_real_threshold=threshold_real, iforest_augmented_threshold=threshold_synth,
         real_fpr=fpr_r, real_tpr=tpr_r, augmented_fpr=fpr_a, augmented_tpr=tpr_a,
     )

CRITICAL ML CONSTRAINTS:
  - Test windows always from original_series (never synthetic)
  - No shuffling anywhere
  - Threshold calibrated on training anomaly scores (not test)
  - No torch import anywhere in this file
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test python -c \
  "from src.benchmarker import run_benchmark; print('BENCHMARKER IMPORT OK')"

Pass: prints "BENCHMARKER IMPORT OK", exit 0.
Also verify no torch dependency:
docker-compose run --rm test python -c \
  "import ast, pathlib
   src = pathlib.Path('src/benchmarker.py').read_text()
   assert 'torch' not in src, 'torch found in benchmarker.py!'
   print('No torch in benchmarker — OK')"
```

### Success Criteria
- `run_benchmark` imports cleanly, no `torch` in the file

### Git Commit Message
```
feat(benchmarker): IsolationForest real-only vs augmented, temporal split, before/after metrics
```

---

## Task 10 — `tests/test_benchmarker.py`

**Files:** `tests/test_benchmarker.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/benchmarker.py, src/types.py, src/synthesizer.py, src/profiler.py, tests/conftest.py

Create tests/test_benchmarker.py. IForest-only — no LSTM, no torch references.

FIXTURES (scope="module"):
@pytest.fixture(scope="module")
def benchmark_dataset(sample_series):
    short_series = sample_series.iloc[:300]
    profile = profile_series(short_series)
    config = SynthesisConfig(n_spike_events=5, n_drift_events=2, n_flatline_events=2,
                              n_noise_burst_events=2, random_seed=42)
    return synthesize_anomalies(short_series, profile, [], config)

@pytest.fixture(scope="module")
def benchmark_config():
    return BenchmarkConfig(test_ratio=0.2, window_size=32, window_step=4, random_seed=42)

STRUCTURE TESTS:
  test_benchmark_result_type: isinstance(result, BenchmarkResult)
  test_iforest_fields_present:
    assert hasattr(result, "iforest_real_f1")
    assert hasattr(result, "iforest_augmented_f1")
    assert hasattr(result, "iforest_real_auc")
    assert hasattr(result, "iforest_augmented_auc")
  test_no_lstm_fields:
    assert not hasattr(result, "lstm_real_f1"), "LSTM fields must not exist"
  test_precision_recall_f1_in_valid_range:
    for attr in ["iforest_real_precision", "iforest_real_recall", "iforest_real_f1",
                 "iforest_augmented_precision", "iforest_augmented_recall", "iforest_augmented_f1"]:
        assert 0.0 <= getattr(result, attr) <= 1.0
  test_auc_in_valid_range:
    for attr in ["iforest_real_auc", "iforest_augmented_auc"]:
        assert 0.0 <= getattr(result, attr) <= 1.0
  test_n_test_windows_positive: result.n_test_windows > 0
  test_n_anomalous_test_windows_positive: result.n_anomalous_test_windows > 0
  test_roc_arrays_present:
    assert isinstance(result.real_fpr, list) and len(result.real_fpr) >= 2
    assert isinstance(result.augmented_fpr, list) and len(result.augmented_fpr) >= 2

NO-LOOKAHEAD TEST:
  test_no_lookahead_temporal_split:
    Build SyntheticDataset where anomalies are ONLY in last 20% (positions 240-299).
    values[240:] += 10.0 (obviously anomalous, easy to detect).
    Run run_benchmark(dataset, BenchmarkConfig(test_ratio=0.2, window_size=16, window_step=1))
    assert result.n_anomalous_test_windows > 0
    (Confirms test set captures the anomalies from the final 20%, not training.)

ERROR CASES:
  test_raises_on_short_series:
    SyntheticDataset with 50-point series → ValueError
  test_raises_when_no_anomalies_in_test:
    SyntheticDataset with all labels = 0 → ValueError("no anomalous windows in test set")

All imports: numpy, pandas, pytest, from src.benchmarker import run_benchmark,
from src.types import BenchmarkResult, BenchmarkConfig, SyntheticDataset, SynthesisConfig,
from src.profiler import profile_series, from src.synthesizer import synthesize_anomalies.
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test pytest tests/test_benchmarker.py -v --tb=short

Pass: All tests collected, 0 failures.
MUST PASS: test_no_lookahead_temporal_split, test_no_lstm_fields.
Should complete in < 30 seconds (IForest is fast, no GPU/LSTM needed).
Report exact output.
```

### Success Criteria
- All tests green; no-lookahead and no-lstm-fields tests pass specifically
- Benchmark completes in < 30 seconds in Docker

### Git Commit Message
```
test(benchmarker): no-leakage, temporal split, IForest-only BenchmarkResult validation
```

---

## Task 11 — `src/discoverer.py`

**Files:** `src/discoverer.py`

### Writer Agent Context
```
Read: CLAUDE.md, src/types.py, src/utils.py, src/profiler.py

Create src/discoverer.py. Single public function + helpers.

MODULE HEADER:
import logging, uuid, numpy as np, pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.cluster import AgglomerativeClustering
from src.types import TimeSeriesProfile, AnomalyCandidate
from src.utils import sliding_window
logger = logging.getLogger(__name__)

DTW IMPORT (with fallback):
try:
    from dtaidistance import dtw as dtw_module
    _USE_FAST_DTW = True
except ImportError:
    _USE_FAST_DTW = False
    logger.warning("dtaidistance C extension not available, using slow DTW")

PUBLIC FUNCTION:
def discover_candidates(series, profile, window_size=64, step=1,
                         contamination=0.1, max_candidates=10,
                         dtw_threshold=0.5) -> list[AnomalyCandidate]:
  Validates: len(series) >= window_size else ValueError; contamination in (0, 0.5] else ValueError.
  Returns list, never raises on empty result. Caps at max_candidates.

PRIVATE HELPERS:

def _build_ensemble_scores(windows, contamination) -> np.ndarray:
  iforest = IForest(contamination=contamination, random_state=42, n_estimators=100)
  knn = KNN(contamination=contamination, n_neighbors=min(5, len(windows)-1))
  iforest.fit(windows); knn.fit(windows)
  scores = np.column_stack([iforest.decision_function(windows), knn.decision_function(windows)])
  raw = scores.mean(axis=1)
  # Min-max normalize to [0, 1]
  rng = raw.max() - raw.min()
  return (raw - raw.min()) / rng if rng > 0 else np.zeros_like(raw)

def _select_high_score_windows(windows, scores, contamination):
  threshold = np.quantile(scores, 1 - contamination)
  mask = scores >= threshold
  if mask.sum() == 0: return np.empty((0,windows.shape[1])), np.array([]), np.array([])
  indices = np.where(mask)[0]
  Return (windows[mask], scores[mask], indices)

def _cluster_windows_dtw(high_windows, dtw_threshold) -> np.ndarray:
  If len == 1: return np.array([0])
  windows_list = [w.astype(np.double) for w in high_windows]
  try:
    if _USE_FAST_DTW:
      dist_matrix = dtw_module.distance_matrix_fast(windows_list)
    else:
      dist_matrix = dtw_module.distance_matrix(windows_list)
  except: fallback to dtw_module.distance_matrix(windows_list)
  np.fill_diagonal(dist_matrix, 0)
  clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=dtw_threshold,
                                       linkage="average", metric="precomputed")
  Return clusterer.fit_predict(dist_matrix)

def _guess_anomaly_type(window) -> str:
  w_std = window.std(); w_mean = window.mean()
  if w_std < 1e-10: w_std = 1e-10
  # Spike: extreme single point
  peak = abs(window - w_mean).max()
  if peak > 3 * w_std: return "spike"
  # Flatline: very low variance
  if w_std < 0.05 * (abs(w_mean) + 1e-9): return "flatline"
  # Drift: large start-to-end change
  if abs(window[-1] - window[0]) > 2 * w_std: return "drift"
  # Noise burst: high variance with no clear pattern
  if w_std > 2.0 * abs(w_mean + 1e-9): return "noise_burst"
  return "unknown"

def _build_candidate(cluster_windows, cluster_scores, cluster_high_indices) -> AnomalyCandidate:
  rep_idx = np.argmax(cluster_scores)
  return AnomalyCandidate(
    candidate_id=str(uuid.uuid4()),
    anomaly_type=_guess_anomaly_type(cluster_windows[rep_idx]),
    representative_window=cluster_windows[rep_idx],
    window_indices=list(cluster_high_indices.astype(int)),
    frequency_count=len(cluster_windows),
    severity_score=float(cluster_scores.mean()),
    confirmed=False
  )

ASSEMBLY in discover_candidates:
1. Validate; windows = sliding_window(series, window_size, step)
2. scores = _build_ensemble_scores(windows, contamination)
3. high_windows, high_scores, high_indices = _select_high_score_windows(...)
4. If len(high_windows) == 0: return []
5. If len(high_windows) == 1: return [_build_candidate(high_windows, high_scores, high_indices)]
6. cluster_labels = _cluster_windows_dtw(high_windows, dtw_threshold)
7. candidates = []
8. For each unique label: extract cluster subset, build candidate, append
9. Sort by severity_score descending
10. Return candidates[:max_candidates]
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test python -c \
  "from src.discoverer import discover_candidates; print('DISCOVERER IMPORT OK')"

Pass: prints "DISCOVERER IMPORT OK", exit 0.
```

### Success Criteria
- `discover_candidates` imports cleanly, returns list on valid input

### Git Commit Message
```
feat(discoverer): PyOD ensemble (IForest+KNN), DTW clustering, AnomalyCandidate output
```

---

## Task 12 — `tests/test_discoverer.py` + `app.py`

**Files:** `tests/test_discoverer.py`, `app.py`

### Writer Agent Context (test_discoverer.py)
```
Read: CLAUDE.md, src/discoverer.py, src/types.py, src/profiler.py, tests/conftest.py

Create tests/test_discoverer.py.

FIXTURE:
@pytest.fixture
def discovery_inputs(sample_series):
    return sample_series, profile_series(sample_series)

RETURN TYPE & EMPTY:
  test_returns_list: isinstance(result, list)
  test_empty_on_flat_series: 200-point series of np.ones → result is list; all severity < 0.5
  test_caps_at_max_candidates: max_candidates=3 → len(result) <= 3
  test_caps_at_ten_by_default: len(result) <= 10

CANDIDATE STRUCTURE:
  test_candidate_type: each item is AnomalyCandidate
  test_candidate_anomaly_type_valid: type in {"spike","drift","flatline","noise_burst","unknown"}
  test_candidate_severity_range: 0.0 <= severity <= 1.0
  test_candidate_representative_window_shape: shape == (window_size,) → use window_size=32
  test_candidate_confirmed_default_false: confirmed == False
  test_candidates_sorted_by_severity: severity[i] >= severity[i+1] for all consecutive pairs

DETECTION:
  test_detects_obvious_spike:
    300-point series, values[50]=20.0, values[150]=-20.0, values[250]=20.0
    result = discover_candidates(series, profile, window_size=16, step=4)
    assert len(result) > 0

ERROR CASES:
  test_raises_on_series_shorter_than_window: 30-point series, window_size=64 → ValueError
  test_raises_on_bad_contamination: contamination=0.9 → ValueError
```

### Writer Agent Context (app.py)
```
Read: CLAUDE.md, src/types.py, src/profiler.py, src/discoverer.py,
      src/synthesizer.py, src/benchmarker.py, src/utils.py

Create app.py. NO business logic — only calls src/ functions.

TOP-OF-FILE:
import streamlit as st, import pandas as pd, import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.profiler import profile_series
from src.discoverer import discover_candidates
from src.synthesizer import synthesize_anomalies
from src.benchmarker import run_benchmark
from src.types import SynthesisConfig, BenchmarkConfig
from src.utils import plot_series_with_anomalies, plot_roc_comparison
st.set_page_config(page_title="AnomalyForge", layout="wide")
st.title("AnomalyForge — Synthetic Anomaly Pipeline")

NAVIGATION: st.sidebar.radio("Navigation", ["1. Upload & Profile", "2. Discover & Confirm",
                                              "3. Generate", "4. Benchmark"])

SCREEN 1: CSV upload → parse CSV (first col=datetime index, second col=values) →
  profile_series → store in st.session_state["profile"] + st.session_state["series"]
  Display: st.line_chart, summary card, existing anomaly warning

SCREEN 2: Guard (require session_state["profile"]) →
  contamination slider (0.01, 0.30, 0.10) → window_size slider (16, 128, 64) →
  "Run Discoverer" button → discover_candidates → session_state["candidates"]
  Display: st.expander per candidate with mini chart, type, frequency, severity, checkbox
  magnitude_multiplier slider, n_events number inputs → store in session_state

SCREEN 3: Guard (require session_state["candidates"]) →
  Build SynthesisConfig from sliders → "Generate Dataset" button →
  synthesize_anomalies → session_state["dataset"]
  Display: two-col original vs synthetic plots, metrics, download button

SCREEN 4: Guard (require session_state["dataset"]) →
  test_ratio slider (0.1–0.4, default 0.2) → "Run Benchmark" button →
  run_benchmark with st.spinner → session_state["result"]
  Display:
  - Metrics table (st.dataframe): columns = [Model, Condition, Precision, Recall, F1, AUC]
    Two rows: IForest real-only | IForest augmented
  - ROC curve comparison via st.pyplot (plot_roc_comparison with result.real_fpr/tpr and result.augmented_fpr/tpr)
  - Headline metric: st.metric("Recall Improvement (IForest)",
      f"+{result.iforest_augmented_recall - result.iforest_real_recall:.1%}")
  No LSTM metrics, no lstm_epochs slider.
```

### Tester Agent Instructions
```
Run: docker-compose run --rm test pytest tests/test_discoverer.py -v --tb=short

MUST PASS: test_detects_obvious_spike, test_caps_at_max_candidates, test_empty_on_flat_series.

Also run app.py syntax check:
docker-compose run --rm test python -c "import ast; ast.parse(open('app.py').read()); print('app.py syntax OK')"

Pass: All discoverer tests 0 failures AND app.py syntax OK.
```

### Success Criteria
- All discoverer tests green; app.py parses without syntax errors; all imports in app.py resolve

### Git Commit Message
```
feat(discoverer+app): PyOD+DTW tests, 4-screen Streamlit dashboard
```

---

## Final E2E Test (Main Thread Responsibility)

After Task 12 is committed and pushed, the main thread runs:

```bash
# Full test suite in Docker:
docker-compose run --rm test pytest tests/ -v --tb=short

# Syntax check all source files:
docker-compose run --rm test python -c "
import src.types, src.utils, src.profiler, src.synthesizer, src.benchmarker, src.discoverer
print('All module imports OK')
"

# app.py import check:
docker-compose run --rm test python -c "
import ast; ast.parse(open('app.py').read())
print('app.py syntax OK')
"
```

**Expected:** All tests collected, 0 failures, 0 errors. This is the final gate.

---

## Success Criteria Summary

| Criterion | Verified By |
|-----------|------------|
| All 4 pytest suites pass | `pytest tests/ -v` in Docker |
| `synthesizer.py` output length == input length | `test_output_length_invariant` |
| No mutation of input series | `test_no_mutation_of_input` |
| No data leakage (threshold on train only) | `test_no_lookahead_temporal_split` |
| No lookahead (temporal split confirmed) | `test_no_lookahead_temporal_split` |
| IForest-only in benchmarker (no LSTM) | `test_no_lstm_fields` |
| Benchmark completes in < 30s | Docker timing on Task 10 |
| Anomaly detection works (spike flagging) | `test_profile_flags_obvious_spike` |
| Discoverer caps at max_candidates | `test_caps_at_max_candidates` |
| app.py has no syntax errors | `ast.parse` smoke test |
| All module imports clean | Import smoke tests per task |

---

## Key Design Decisions

1. **Synthesizer before discoverer in build order** — benchmarker depends on SyntheticDataset; benchmarker must be solid before discoverer.

2. **Test windows from original_series only** — evaluating on synthetic test data would be distribution leakage; the benchmark measures detection on real data.

3. **`matplotlib.use("Agg")` at module level** — Docker has no display server; setting in both `utils.py` and `app.py` prevents `_tkinter` errors regardless of import order.

4. **IsolationForest only (no LSTM)** — scope reduction for simplicity and speed. IForest trains in milliseconds (no GPU, no epoch loops), Docker test suite completes in < 30 seconds, and the core pitch (synthetic augmentation improves recall) is fully demonstrable with IForest alone.

5. **DTW threshold as parameter** — different series have different absolute scales; exposing it lets the user tune and prevents over/under-clustering.

6. **Volume-mount source in docker-compose** — code changes don't require `docker build` during development; the test container always uses the latest source.

---

## Write plan.md After Completion

After all 12 tasks are committed and the final E2E test passes, the main thread writes the final plan to the project repository:

```bash
cp /Users/monoid/.claude/plans/we-need-to-build-delegated-honey.md \
   /Users/monoid/Documents/development/anomaly_forge/anomaly-forge/plan.md
git -C /Users/monoid/Documents/development/anomaly_forge/anomaly-forge \
    add plan.md && git commit -m "docs: add plan.md" && git push origin main
```
