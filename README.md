# 🔥 AnomalyForge

> Prove that synthetic anomaly augmentation improves rare-event detection — end to end, in a live Streamlit demo.

AnomalyForge is a 4-phase time-series anomaly detection and synthesis pipeline. It takes real-world time-series data (e.g. server CPU utilization), discovers anomaly patterns, synthesizes labeled training data, and benchmarks IsolationForest **before vs. after** augmentation — showing measurable recall improvement on a held-out test set.

---

## System Diagram

```mermaid
flowchart TD
    CSV["📄 CSV Upload\n(datetime + value)"]

    subgraph Phase1["Phase 1 — Profile"]
        PRO["profiler.py\n─────────────\nSTL decomposition\nDistribution fitting\nACF / PACF\nAnomaly flagging"]
        TP["TimeSeriesProfile\n─────────────\nfreq · noise_std\nseasonality · trend\nexisting anomalies"]
    end

    subgraph Phase2["Phase 2 — Discover"]
        DIS["discoverer.py\n─────────────\nSliding windows\nPyOD ensemble\n(IForest + KNN)\nDTW clustering"]
        AC["AnomalyCandidates\n─────────────\ntype · severity\nrepresentative window\nfrequency count"]
    end

    subgraph Phase3["Phase 3 — Synthesize"]
        SYN["synthesizer.py\n─────────────\nSpike injection\nDrift injection\nFlatline injection\nNoise burst injection"]
        SD["SyntheticDataset\n─────────────\noriginal_series\nsynthetic_series\nlabels (0/1)\ninjection_metadata"]
    end

    subgraph Phase4["Phase 4 — Benchmark"]
        BEN["benchmarker.py\n─────────────\nTemporal split\nIForest · real-only\nIForest · augmented\nROC · AUC · F1"]
        BR["BenchmarkResult\n─────────────\nPrecision · Recall · F1\nAUC · ROC arrays\nRecall Δ (before/after)"]
    end

    subgraph UI["Streamlit Dashboard (app.py)"]
        S1["① Upload & Profile"]
        S2["② Discover & Confirm"]
        S3["③ Generate"]
        S4["④ Benchmark"]
    end

    CSV --> S1 --> PRO --> TP
    TP  --> S2 --> DIS --> AC
    AC  --> S3 --> SYN --> SD
    SD  --> S4 --> BEN --> BR

    BR -->|"📈 Recall improvement\nReal-only vs Augmented"| S4
```

---

## Pipeline Phases

| Phase | Module | Input | Output |
|-------|--------|-------|--------|
| 1 · Profile | `profiler.py` | Raw `pd.Series` | `TimeSeriesProfile` |
| 2 · Discover | `discoverer.py` | Series + Profile | `list[AnomalyCandidate]` |
| 3 · Synthesize | `synthesizer.py` | Series + Candidates + Config | `SyntheticDataset` |
| 4 · Benchmark | `benchmarker.py` | SyntheticDataset + Config | `BenchmarkResult` |

---

## Anomaly Types Synthesized

| Type | Description | CPU Example |
|------|-------------|-------------|
| **Spike** | Single-point extreme deviation (±3σ) | Cron job firing, GC pause |
| **Drift** | Linear ramp over N samples | Memory leak, queue backup |
| **Flatline** | Segment held at constant value | Stuck sensor, frozen metric feed |
| **Noise Burst** | High-variance Gaussian segment | Retry storm, thundering herd |

---

## Quickstart

```bash
# Clone and set up
git clone https://github.com/quietspoon/anomaly-forge.git
cd anomaly-forge
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy statsmodels pyod scikit-learn \
            dtaidistance streamlit matplotlib seaborn pytest

# Run the app
streamlit run app.py
```

Open **http://localhost:8501** — upload any two-column datetime CSV and follow the 4-step flow.

### Run tests (Docker)

```bash
docker-compose run --rm test pytest tests/ -v --tb=short
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Data | `pandas` · `numpy` |
| Statistics | `scipy` · `statsmodels` (STL) |
| Anomaly detection | `pyod` (IForest + KNN) · `scikit-learn` (IsolationForest) |
| Clustering | `dtaidistance` (DTW) · `sklearn.AgglomerativeClustering` |
| Visualization | `matplotlib` · `seaborn` |
| Dashboard | `streamlit` |
| Testing | `pytest` · Docker |

---

## Project Structure

```
anomaly-forge/
├── src/
│   ├── types.py          # Shared dataclasses (data contracts)
│   ├── utils.py          # sliding_window, normalize, temporal_split, plot helpers
│   ├── profiler.py       # Phase 1 — STL, distribution fitting, ACF/PACF
│   ├── discoverer.py     # Phase 2 — PyOD ensemble + DTW clustering
│   ├── synthesizer.py    # Phase 3 — 4 anomaly injection types
│   └── benchmarker.py    # Phase 4 — IForest real-only vs augmented
├── tests/                # 89 pytest tests, all run in Docker
├── app.py                # Streamlit 4-screen dashboard
├── Dockerfile
└── docker-compose.yml
```
