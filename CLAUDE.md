# AnomalyForge — Claude Code Context

## What This Project Is
A demo pitch for Rockfish, a synthetic time-series data startup.
AnomalyForge demonstrates the core value of Rockfish's product:
synthetic anomaly-augmented training data improves rare-event detection
in ML models — provably, with before/after benchmarks.

Target audience: Rockfish engineering leadership.
Goal: show product instinct + technical depth in one runnable demo.

## Project Status
Early-stage implementation. The architecture described below serves as the implementation blueprint.

## Development Commands
(To be added once implementation begins)
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest tests/`
- Run dashboard: `streamlit run app.py`
- Run single test file: `pytest tests/test_profiler.py -v`

---

## The Four-Phase Pipeline

```
[Real Data Slice]
      │
      ▼
profiler.py         ← characterize the baseline (distribution, seasonality, noise)
      │
      ▼
discoverer.py       ← mine candidate anomaly patterns, surface for user confirmation
      │
      ▼
synthesizer.py      ← inject labeled rare events, preserve baseline fingerprint
      │
      ▼
benchmarker.py      ← train/eval on real-only vs real+synthetic, show delta
      │
      ▼
app.py (Streamlit)  ← demo dashboard, walks through all 4 phases visually
```

All business logic lives in `src/`. `app.py` only calls `src/` — no logic in the dashboard layer.

---

## File Structure

```
anomalyforge/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── app.py                        # Streamlit dashboard
├── data/
│   ├── raw/                      # Original time-series input
│   └── synthetic/                # Generated labeled datasets
├── src/
│   ├── types.py                  # Shared dataclasses and TypedDicts — source of truth for interfaces
│   ├── profiler.py               # Phase 1: statistical fingerprinting
│   ├── discoverer.py             # Phase 2: near-anomaly mining + candidate surfacing
│   ├── synthesizer.py            # Phase 3: labeled synthetic data generation
│   ├── benchmarker.py            # Phase 4: model training + before/after evaluation
│   └── utils.py                  # Shared helpers: windowing, normalization, plotting
├── tests/
│   ├── test_profiler.py
│   ├── test_discoverer.py
│   ├── test_synthesizer.py
│   └── test_benchmarker.py
└── notebooks/
    └── exploration.ipynb         # EDA scratch space only — not part of the pitch
```

---

## Data Contracts (types.py is the source of truth)

All inter-module data flows through typed dataclasses defined in `src/types.py`.
Never pass raw dicts between modules. Always import from types.py.

Key types (defined in types.py):
- `TimeSeriesProfile` — output of profiler.py
- `AnomalyCandidate` — output of discoverer.py, one per candidate pattern
- `SyntheticDataset` — output of synthesizer.py (series + labels + metadata)
- `BenchmarkResult` — output of benchmarker.py (metrics for real-only vs augmented)

If you need to add a field to a type, update types.py first, then update all downstream consumers.

---

## Module Responsibilities

### profiler.py
Input: `pd.Series` with a `DatetimeIndex`
Output: `TimeSeriesProfile`

Responsibilities:
- Fit a baseline distribution (normal, log-normal, or best fit via scipy)
- STL decomposition to detect seasonality and extract residuals
- Compute autocorrelation structure (ACF/PACF via statsmodels)
- Characterize noise floor (std of residuals)
- Flag any existing anomalies in the slice (do NOT remove them, just annotate)

Do NOT make any assumptions about sampling frequency — infer it from the index.

### discoverer.py
Input: `pd.Series`, `TimeSeriesProfile`
Output: `list[AnomalyCandidate]`

Responsibilities:
- Score every window using PyOD (IForest + KNN ensemble, majority vote)
- Cluster high-scoring windows by shape similarity (DTW distance + agglomerative clustering)
- Return one AnomalyCandidate per cluster with: representative window, anomaly type guess, frequency count, severity score
- Anomaly type guesses: "spike", "drift", "flatline", "noise_burst", "unknown"

Do NOT ask the user to specify anomaly types upfront — surface candidates, let them confirm.

### synthesizer.py
Input: `pd.Series`, `TimeSeriesProfile`, `list[AnomalyCandidate]` (confirmed), `SynthesisConfig`
Output: `SyntheticDataset`

Responsibilities:
- Preserve baseline statistical fingerprint outside injection windows
- Injection types: point spike, sustained drift, flatline, noise burst
- Spike amplitude = config.magnitude_multiplier * local_std at injection point
- Duration controlled per-event by config
- Never mutate input series — always return new arrays
- Every injected event has a ground-truth label in the output

Approach: statistical injection, NOT generative models (GAN/VAE).
Reason: explainability matters for a pitch — we need to show exactly what was injected and why.

### benchmarker.py
Input: `SyntheticDataset`, `BenchmarkConfig`
Output: `BenchmarkResult`

Responsibilities:
- Temporal train/test split — test set is always the chronologically latest portion
- NEVER shuffle time-series data — this causes lookahead leakage
- Train two models: IsolationForest (sklearn) and LSTM Autoencoder (PyTorch)
- Evaluate both on the same held-out test set
- Metrics: Precision, Recall, F1, AUC-ROC — computed separately for rare events vs. normal windows
- Return before (real-only) and after (real+synthetic) metrics for side-by-side comparison

The core story: synthetic augmentation improves rare-event Recall without
significantly hurting normal-data Precision.

### utils.py
- `sliding_window(series, window_size, step)` → np.ndarray of windows
- `normalize(series)` → normalized series + scaler (return both so we can invert)
- `temporal_split(series, test_ratio)` → (train, test) — NO shuffling
- `plot_series_with_anomalies(series, labels)` → matplotlib figure
- `plot_roc_comparison(real_result, augmented_result)` → matplotlib figure

---

## Code Style

- Python 3.11+
- Type hints on every function signature (inputs and return type)
- Docstring on every public function — one line summary + Args + Returns
- No global state — every function takes explicit inputs and returns explicit outputs
- No print statements in src/ — use Python `logging` module with `logger = logging.getLogger(__name__)`
- Raise typed exceptions with descriptive messages, never silent failures
- Keep functions small — if a function exceeds ~40 lines, consider splitting it

---

## ML Constraints (Non-Negotiable)

- Temporal train/test split always — test set = latest time window
- No data leakage: fit scalers/normalizers on train set only, apply to test set
- No lookahead features: features at time t can only use data from t and earlier
- Labels in SyntheticDataset are ground-truth injection labels — not model predictions

These are non-negotiable. If a code change would violate any of these, refuse and flag it.

---

## Dependencies

```
pandas
numpy
scipy
statsmodels
pyod
scikit-learn
torch
streamlit
matplotlib
seaborn
dtaidistance        # for DTW-based clustering in discoverer.py
```

---

## Dataset

Primary: Yahoo S5 Anomaly Detection Benchmark (A1Benchmark subset)
Fallback: NASA SMAP or MSL datasets

Both are publicly available, labeled, and well-known in the anomaly detection community —
good for credibility in a pitch context.

Data lives in `data/raw/`. Do NOT commit large data files — add to .gitignore.

---

## Build Order

Build and verify modules in this order. Do not move to the next until the current one has passing tests.

1. `src/types.py` — define all interfaces first
2. `src/utils.py` — shared helpers needed by everything
3. `src/profiler.py` + `tests/test_profiler.py`
4. `src/synthesizer.py` + `tests/test_synthesizer.py`
5. `src/benchmarker.py` + `tests/test_benchmarker.py`
6. `src/discoverer.py` + `tests/test_discoverer.py`
7. `app.py` — wire everything into Streamlit last

Note: synthesizer comes before discoverer in build order because the benchmark
pipeline needs synthetic data to validate against. Discoverer is built last because
it's the UX differentiator — it should be solid before it becomes the front door.

---

## Streamlit Dashboard Flow (app.py)

Screen 1 — Upload & Profile
- Upload CSV → plot raw series → display TimeSeriesProfile summary card
- Show: detected distribution, seasonality Y/N, noise level, any flagged existing anomalies

Screen 2 — Discover & Confirm
- Run discoverer → show candidate anomaly patterns with mini-plots
- Checkboxes to confirm which patterns matter
- Sliders: severity multiplier, synthetic frequency (events per 1000 samples)

Screen 3 — Generate
- "Generate Dataset" button → show original vs. synthetic side-by-side
- Injected events highlighted and color-coded by type
- Download button for the labeled CSV

Screen 4 — Benchmark
- "Run Benchmark" button → progress bar (LSTM takes a moment)
- Before/after comparison table (real-only vs real+synthetic)
- Side-by-side ROC curves
- Headline callout: Recall improvement on rare events

---

## Pitch Narrative (keep this in mind when making design decisions)

> "Rockfish asks users to define their rare events upfront — but most teams
> don't have that vocabulary yet. AnomalyForge closes that loop: it discovers
> candidate anomaly patterns from your existing data, lets you confirm and
> extend them, then proves the value of synthetic augmentation with
> before/after benchmarks — all in a single demo."

Every design decision should serve this narrative.
If a feature doesn't make the demo clearer or the results more compelling, cut it.
