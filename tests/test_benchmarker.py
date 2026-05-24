"""Unit tests for src/benchmarker.py — IsolationForest-only benchmark."""

import numpy as np
import pandas as pd
import pytest

from src.benchmarker import run_benchmark
from src.profiler import profile_series
from src.synthesizer import synthesize_anomalies
from src.types import (
    BenchmarkConfig,
    BenchmarkResult,
    SyntheticDataset,
    SynthesisConfig,
)


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def benchmark_dataset(sample_series: pd.Series) -> SyntheticDataset:
    """Build a minimal SyntheticDataset on a 300-point slice for fast tests."""
    short_series = sample_series.iloc[:300]
    profile = profile_series(short_series)
    config = SynthesisConfig(
        n_spike_events=5,
        n_drift_events=2,
        n_flatline_events=2,
        n_noise_burst_events=2,
        magnitude_multiplier=4.0,  # large magnitude for clearly-detectable anomalies
        random_seed=42,
    )
    return synthesize_anomalies(short_series, profile, [], config)


@pytest.fixture(scope="module")
def benchmark_config() -> BenchmarkConfig:
    """Fast benchmark config — small windows, no LSTM epochs."""
    return BenchmarkConfig(
        test_ratio=0.2,
        window_size=32,
        window_step=4,
        iforest_contamination=0.1,
        iforest_n_estimators=50,  # fewer trees for speed
        random_seed=42,
    )


@pytest.fixture(scope="module")
def benchmark_result(
    benchmark_dataset: SyntheticDataset,
    benchmark_config: BenchmarkConfig,
) -> BenchmarkResult:
    """Run benchmark once; reuse result across all structure tests."""
    return run_benchmark(benchmark_dataset, benchmark_config)


# ─── Result structure ─────────────────────────────────────────────────────────


def test_benchmark_result_type(benchmark_result: BenchmarkResult) -> None:
    """run_benchmark must return a BenchmarkResult instance."""
    assert isinstance(benchmark_result, BenchmarkResult)


def test_iforest_fields_present(benchmark_result: BenchmarkResult) -> None:
    """BenchmarkResult must have all four IForest field pairs."""
    assert hasattr(benchmark_result, "iforest_real_f1")
    assert hasattr(benchmark_result, "iforest_augmented_f1")
    assert hasattr(benchmark_result, "iforest_real_auc")
    assert hasattr(benchmark_result, "iforest_augmented_auc")


def test_no_lstm_fields(benchmark_result: BenchmarkResult) -> None:
    """BenchmarkResult must NOT have any LSTM fields (IForest-only scope)."""
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(benchmark_result)}
    lstm_fields = [f for f in field_names if "lstm" in f]
    assert not lstm_fields, f"Unexpected LSTM fields: {lstm_fields}"


def test_precision_recall_f1_in_valid_range(benchmark_result: BenchmarkResult) -> None:
    """All precision, recall, and F1 values must be in [0.0, 1.0]."""
    for attr in [
        "iforest_real_precision",
        "iforest_real_recall",
        "iforest_real_f1",
        "iforest_augmented_precision",
        "iforest_augmented_recall",
        "iforest_augmented_f1",
    ]:
        val = getattr(benchmark_result, attr)
        assert 0.0 <= val <= 1.0, f"{attr} = {val} is out of [0, 1]"


def test_auc_in_valid_range(benchmark_result: BenchmarkResult) -> None:
    """All AUC values must be in [0.0, 1.0]."""
    for attr in ["iforest_real_auc", "iforest_augmented_auc"]:
        val = getattr(benchmark_result, attr)
        assert 0.0 <= val <= 1.0, f"{attr} = {val} is out of [0, 1]"


def test_n_test_windows_positive(benchmark_result: BenchmarkResult) -> None:
    """n_test_windows must be greater than 0."""
    assert benchmark_result.n_test_windows > 0


def test_n_anomalous_test_windows_positive(benchmark_result: BenchmarkResult) -> None:
    """n_anomalous_test_windows must be greater than 0."""
    assert benchmark_result.n_anomalous_test_windows > 0


def test_roc_arrays_present(benchmark_result: BenchmarkResult) -> None:
    """ROC curve arrays must be lists with at least 2 elements."""
    assert isinstance(benchmark_result.real_fpr, list)
    assert isinstance(benchmark_result.real_tpr, list)
    assert isinstance(benchmark_result.augmented_fpr, list)
    assert isinstance(benchmark_result.augmented_tpr, list)
    assert len(benchmark_result.real_fpr) >= 2
    assert len(benchmark_result.augmented_fpr) >= 2


def test_roc_fpr_starts_at_zero(benchmark_result: BenchmarkResult) -> None:
    """ROC curve fpr must start at 0.0."""
    assert benchmark_result.real_fpr[0] == pytest.approx(0.0, abs=1e-6)
    assert benchmark_result.augmented_fpr[0] == pytest.approx(0.0, abs=1e-6)


def test_thresholds_are_floats(benchmark_result: BenchmarkResult) -> None:
    """IForest thresholds must be Python floats."""
    assert isinstance(benchmark_result.iforest_real_threshold, float)
    assert isinstance(benchmark_result.iforest_augmented_threshold, float)


# ─── No-lookahead test ────────────────────────────────────────────────────────


def test_no_lookahead_temporal_split() -> None:
    """Verify that anomalies placed only in the last 20% are detected in test.

    If the split is chronological (no lookahead), the test set contains
    the injected anomalies and n_anomalous_test_windows > 0.
    If the split accidentally includes future data in training, the test
    set would have no anomalies and this test would catch it.
    """
    n = 300
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(99)
    values = rng.normal(0, 1.0, n)
    labels = np.zeros(n, dtype=int)

    # Anomalies ONLY in the last 20% (indices 240-299)
    anomaly_start = int(n * 0.8)
    labels[anomaly_start:] = 1
    values[anomaly_start:] += 15.0  # very large shift — easy to detect

    series = pd.Series(values, index=idx, name="no_lookahead_test")
    syn_series = pd.Series(values.copy(), index=idx, name="no_lookahead_test")

    dataset = SyntheticDataset(
        original_series=series,
        synthetic_series=syn_series,
        labels=labels,
        anomaly_type_labels=["spike" if l == 1 else "" for l in labels],
        injection_metadata=[],
        n_injected_events=int(labels.sum()),
        synthesis_config=SynthesisConfig(),
    )
    config = BenchmarkConfig(
        test_ratio=0.2,
        window_size=16,
        window_step=1,
        random_seed=0,
    )
    result = run_benchmark(dataset, config)

    # If temporal split is correct, test set includes the anomalous tail
    assert result.n_anomalous_test_windows > 0, (
        "n_anomalous_test_windows == 0: temporal split may have lookahead or "
        "the anomaly region is not in the test set"
    )


# ─── Error cases ──────────────────────────────────────────────────────────────


def test_raises_on_short_series() -> None:
    """SyntheticDataset with fewer than 100 observations must raise ValueError."""
    n = 50
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    series = pd.Series(np.ones(n), index=idx)
    dataset = SyntheticDataset(
        original_series=series,
        synthetic_series=series,
        labels=np.zeros(n, dtype=int),
        anomaly_type_labels=[""] * n,
        injection_metadata=[],
        n_injected_events=0,
        synthesis_config=SynthesisConfig(),
    )
    with pytest.raises(ValueError, match="100"):
        run_benchmark(dataset, BenchmarkConfig())


def test_raises_when_no_anomalies_in_test() -> None:
    """ValueError must be raised when the test set has no anomalous windows."""
    n = 200
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1.0, n)
    series = pd.Series(values, index=idx)
    # All labels = 0 → test set will have no anomalies
    dataset = SyntheticDataset(
        original_series=series,
        synthetic_series=series,
        labels=np.zeros(n, dtype=int),
        anomaly_type_labels=[""] * n,
        injection_metadata=[],
        n_injected_events=0,
        synthesis_config=SynthesisConfig(),
    )
    with pytest.raises(ValueError):
        run_benchmark(dataset, BenchmarkConfig(test_ratio=0.2, window_size=16))
