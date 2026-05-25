"""AnomalyForge — Streamlit dashboard.

Four-screen guided flow:
  1. Upload & Profile  — load CSV, run profiler, display fingerprint
  2. Discover & Confirm — run discoverer, user confirms candidate patterns
  3. Generate          — synthesize anomalies, show side-by-side, download CSV
  4. Benchmark         — run IForest real-only vs augmented, show before/after

All business logic lives in src/. This file only calls src/ functions.
No logic belongs here — delegate everything to the pipeline modules.
"""

import io

import matplotlib
matplotlib.use("Agg")  # headless — must be called before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.benchmarker import run_benchmark
from src.discoverer import discover_candidates
from src.profiler import profile_series
from src.synthesizer import synthesize_anomalies
from src.types import BenchmarkConfig, SynthesisConfig
from src.utils import plot_roc_comparison, plot_series_with_anomalies

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AnomalyForge",
    page_icon="🔥",
    layout="wide",
)
st.title("🔥 AnomalyForge — Synthetic Anomaly Pipeline")
st.caption(
    "Discover → Confirm → Synthesize → Benchmark: "
    "prove that synthetic augmentation improves rare-event detection."
)

# ─── Sidebar navigation ───────────────────────────────────────────────────────

screen = st.sidebar.radio(
    "Navigation",
    options=[
        "1. Upload & Profile",
        "2. Discover & Confirm",
        "3. Generate",
        "4. Benchmark",
    ],
    index=0,
)

# ─── Screen 1 — Upload & Profile ─────────────────────────────────────────────

if screen == "1. Upload & Profile":
    st.header("Step 1 — Upload & Profile")
    st.markdown(
        "Upload a CSV with a datetime column (first) and a numeric value column (second). "
        "The profiler will characterise the baseline distribution, detect seasonality, "
        "and flag any pre-existing anomalies."
    )

    uploaded = st.file_uploader(
        "Upload time-series CSV", type=["csv"], key="csv_upload"
    )

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, parse_dates=[0], index_col=0)
            if df.shape[1] < 1:
                st.error("CSV must have at least two columns: datetime and value.")
                st.stop()

            series = df.iloc[:, 0].dropna()
            series.name = series.name or "value"

            st.subheader("Raw Series")
            st.line_chart(series)

            with st.spinner("Profiling series…"):
                profile = profile_series(series)

            st.session_state["series"] = series
            st.session_state["profile"] = profile

            st.subheader("Statistical Profile")
            col1, col2, col3 = st.columns(3)
            col1.metric("Observations", profile.n_observations)
            col1.metric("Frequency", profile.inferred_freq)
            col2.metric("Distribution", profile.distribution_name)
            col2.metric("Noise Std", f"{profile.noise_std:.4f}")
            col3.metric("Seasonality", "Yes" if profile.has_seasonality else "No")
            col3.metric("Trend Slope", f"{profile.trend_slope:.6f}")

            if profile.existing_anomaly_indices:
                st.warning(
                    f"⚠️ Found **{len(profile.existing_anomaly_indices)}** "
                    "pre-existing anomalies in the uploaded data. "
                    "These are annotated but not removed."
                )
            else:
                st.success("✅ No pre-existing anomalies detected.")

            st.caption("✅ Profile complete. Proceed to **Step 2 — Discover & Confirm**.")

        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load or profile the CSV: {exc}")

    else:
        st.info("⬆️ Upload a CSV file to get started.")

# ─── Screen 2 — Discover & Confirm ───────────────────────────────────────────

elif screen == "2. Discover & Confirm":
    st.header("Step 2 — Discover & Confirm")

    if "profile" not in st.session_state:
        st.info("⬅️ Complete Step 1 (Upload & Profile) first.")
        st.stop()

    series: pd.Series = st.session_state["series"]
    profile = st.session_state["profile"]

    st.markdown("Tune the discoverer parameters, then run it to surface candidate anomaly patterns.")

    with st.expander("Discoverer settings", expanded=True):
        contamination = st.slider(
            "Contamination (assumed anomaly fraction)", 0.01, 0.30, 0.10, step=0.01
        )
        window_size = st.slider("Window size (samples)", 16, 128, 64, step=8)
        dtw_threshold = st.slider("DTW clustering threshold", 0.1, 2.0, 0.5, step=0.1)

    if st.button("🔍 Run Discoverer"):
        with st.spinner("Scoring windows and clustering…"):
            try:
                candidates = discover_candidates(
                    series,
                    profile,
                    window_size=window_size,
                    step=max(1, window_size // 8),
                    contamination=contamination,
                    dtw_threshold=dtw_threshold,
                )
                st.session_state["candidates"] = candidates
                st.session_state["window_size"] = window_size
            except Exception as exc:  # noqa: BLE001
                st.error(f"Discovery failed: {exc}")
                st.stop()

    if "candidates" not in st.session_state:
        st.info("Click **Run Discoverer** to find anomaly candidates.")
        st.stop()

    candidates = st.session_state["candidates"]

    if not candidates:
        st.warning("No anomaly candidates found. Try lowering the contamination threshold.")
    else:
        st.success(f"Found **{len(candidates)}** candidate pattern(s).")

        confirmed_ids: set[str] = set()
        for i, candidate in enumerate(candidates):
            with st.expander(
                f"Candidate {i + 1}: **{candidate.anomaly_type}** — "
                f"severity {candidate.severity_score:.3f}, "
                f"frequency {candidate.frequency_count}",
                expanded=(i == 0),
            ):
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    fig, ax = plt.subplots(figsize=(6, 2))
                    ax.plot(candidate.representative_window, color="steelblue", linewidth=1.2)
                    ax.set_title(f"Representative window ({candidate.anomaly_type})")
                    ax.set_xlabel("Sample offset")
                    ax.set_ylabel("Value")
                    st.pyplot(fig)
                    plt.close(fig)
                with col_b:
                    st.metric("Type", candidate.anomaly_type)
                    st.metric("Severity", f"{candidate.severity_score:.3f}")
                    st.metric("Count", candidate.frequency_count)

                checked = st.checkbox(
                    "✅ Confirm this pattern",
                    key=f"confirm_{candidate.candidate_id}",
                    value=False,
                )
                if checked:
                    candidate.confirmed = True
                    confirmed_ids.add(candidate.candidate_id)
                else:
                    candidate.confirmed = False

    # Synthesis config sliders
    st.subheader("Synthesis Configuration")
    c1, c2, c3, c4 = st.columns(4)
    n_spike = c1.number_input("Spike events", min_value=0, max_value=20, value=5)
    n_drift = c2.number_input("Drift events", min_value=0, max_value=20, value=3)
    n_flat = c3.number_input("Flatline events", min_value=0, max_value=20, value=2)
    n_burst = c4.number_input("Noise burst events", min_value=0, max_value=20, value=3)
    magnitude = st.slider("Magnitude multiplier", 1.0, 10.0, 3.0, step=0.5)

    # Store config params in session_state for Screen 3
    st.session_state["synth_params"] = {
        "n_spike_events": int(n_spike),
        "n_drift_events": int(n_drift),
        "n_flatline_events": int(n_flat),
        "n_noise_burst_events": int(n_burst),
        "magnitude_multiplier": float(magnitude),
    }

    st.caption("✅ Confirm patterns above, then proceed to **Step 3 — Generate**.")

# ─── Screen 3 — Generate ─────────────────────────────────────────────────────

elif screen == "3. Generate":
    st.header("Step 3 — Generate Synthetic Dataset")

    if "candidates" not in st.session_state:
        st.info("⬅️ Complete Step 2 (Discover & Confirm) first.")
        st.stop()

    series: pd.Series = st.session_state["series"]
    profile = st.session_state["profile"]
    candidates = st.session_state["candidates"]
    synth_params: dict = st.session_state.get("synth_params", {})

    confirmed = [c for c in candidates if c.confirmed]
    st.info(
        f"**{len(confirmed)}** pattern(s) confirmed. "
        "The synthesizer will inject events of all configured types."
    )

    if st.button("⚗️ Generate Dataset"):
        config = SynthesisConfig(
            n_spike_events=synth_params.get("n_spike_events", 5),
            n_drift_events=synth_params.get("n_drift_events", 3),
            n_flatline_events=synth_params.get("n_flatline_events", 2),
            n_noise_burst_events=synth_params.get("n_noise_burst_events", 3),
            magnitude_multiplier=synth_params.get("magnitude_multiplier", 3.0),
        )
        with st.spinner("Synthesizing anomalies…"):
            try:
                dataset = synthesize_anomalies(series, profile, confirmed, config)
                st.session_state["dataset"] = dataset
            except Exception as exc:  # noqa: BLE001
                st.error(f"Synthesis failed: {exc}")
                st.stop()

    if "dataset" not in st.session_state:
        st.info("Click **Generate Dataset** to inject synthetic anomalies.")
        st.stop()

    dataset = st.session_state["dataset"]

    st.subheader("Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Events Injected", dataset.n_injected_events)
    m2.metric("Anomaly Rate", f"{float(dataset.labels.mean()) * 100:.1f}%")
    m3.metric("Series Length", len(dataset.synthetic_series))

    col_orig, col_synth = st.columns(2)

    with col_orig:
        st.markdown("**Original Series**")
        fig_orig = plot_series_with_anomalies(
            dataset.original_series,
            np.zeros(len(dataset.original_series), dtype=int),
            title="Original (no injections)",
        )
        st.pyplot(fig_orig)
        plt.close(fig_orig)

    with col_synth:
        st.markdown("**Synthetic Series (with injections)**")
        fig_synth = plot_series_with_anomalies(
            dataset.synthetic_series,
            dataset.labels,
            title="Synthetic (injected anomalies in red)",
        )
        st.pyplot(fig_synth)
        plt.close(fig_synth)

    # Download button
    out_df = pd.DataFrame({
        "datetime": dataset.synthetic_series.index,
        "value": dataset.synthetic_series.values,
        "label": dataset.labels,
        "anomaly_type": dataset.anomaly_type_labels,
    })
    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download labeled CSV",
        data=csv_bytes,
        file_name="anomalyforge_synthetic.csv",
        mime="text/csv",
    )

    st.caption("✅ Dataset ready. Proceed to **Step 4 — Benchmark**.")

# ─── Screen 4 — Benchmark ────────────────────────────────────────────────────

elif screen == "4. Benchmark":
    st.header("Step 4 — Benchmark")

    if "dataset" not in st.session_state:
        st.info("⬅️ Complete Step 3 (Generate) first.")
        st.stop()

    dataset = st.session_state["dataset"]

    st.markdown(
        "Compare IsolationForest trained on **real-only** data vs. "
        "**real+synthetic** (augmented) data. Both are evaluated on the "
        "same held-out chronological test set."
    )

    with st.expander("Benchmark settings", expanded=True):
        test_ratio = st.slider("Test ratio", 0.10, 0.40, 0.20, step=0.05)
        window_size_b = st.slider("Window size (samples)", 16, 128, 64, step=8)
        n_estimators = st.slider("IForest estimators", 50, 300, 100, step=50)
        contamination_b = st.slider(
            "Contamination (expected anomaly fraction)",
            0.01, 0.40, 0.10, step=0.01,
            help="Set this close to the actual anomaly rate in your data. "
                 "Lower = fewer windows flagged as anomalous.",
        )
        threshold_pct = st.slider(
            "Threshold percentile",
            50, 99, 80, step=1,
            help="Anomaly score percentile used as the classification threshold. "
                 "Lower values flag more windows as anomalous, increasing recall. "
                 "If Recall=0, try lowering this to 70–80.",
        )

    if st.button("🚀 Run Benchmark"):
        config = BenchmarkConfig(
            test_ratio=test_ratio,
            window_size=window_size_b,
            window_step=max(1, window_size_b // 8),
            iforest_n_estimators=n_estimators,
            iforest_contamination=contamination_b,
            iforest_threshold_percentile=float(threshold_pct),
        )
        with st.spinner("Training and evaluating models…"):
            try:
                result = run_benchmark(dataset, config)
                st.session_state["result"] = result
            except ValueError as exc:
                st.error(f"Benchmark error: {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error during benchmark: {exc}")
                st.stop()

    if "result" not in st.session_state:
        st.info("Click **Run Benchmark** to start.")
        st.stop()

    result = st.session_state["result"]

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.subheader("Before / After Metrics")
    metrics_df = pd.DataFrame([
        {
            "Model": "IsolationForest",
            "Training": "Real-only",
            "Precision": f"{result.iforest_real_precision:.3f}",
            "Recall": f"{result.iforest_real_recall:.3f}",
            "F1": f"{result.iforest_real_f1:.3f}",
            "AUC": f"{result.iforest_real_auc:.3f}",
        },
        {
            "Model": "IsolationForest",
            "Training": "Augmented (Real + Synthetic)",
            "Precision": f"{result.iforest_augmented_precision:.3f}",
            "Recall": f"{result.iforest_augmented_recall:.3f}",
            "F1": f"{result.iforest_augmented_f1:.3f}",
            "AUC": f"{result.iforest_augmented_auc:.3f}",
        },
    ])
    st.dataframe(metrics_df, use_container_width=True)

    # ── Headline metric ────────────────────────────────────────────────────────
    recall_delta = result.iforest_augmented_recall - result.iforest_real_recall
    st.metric(
        label="📈 Recall Improvement (Augmented vs Real-only)",
        value=f"{result.iforest_augmented_recall:.1%}",
        delta=f"{recall_delta:+.1%}",
    )

    # ── ROC curves ────────────────────────────────────────────────────────────
    st.subheader("ROC Curves")
    fig_roc = plot_roc_comparison(
        real_fpr=np.array(result.real_fpr),
        real_tpr=np.array(result.real_tpr),
        real_auc=result.iforest_real_auc,
        augmented_fpr=np.array(result.augmented_fpr),
        augmented_tpr=np.array(result.augmented_tpr),
        augmented_auc=result.iforest_augmented_auc,
        model_name="IsolationForest",
    )
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # ── Test set summary ───────────────────────────────────────────────────────
    st.caption(
        f"Test set: {result.n_test_windows} windows "
        f"({result.n_anomalous_test_windows} anomalous, "
        f"{result.n_test_windows - result.n_anomalous_test_windows} normal)"
    )
