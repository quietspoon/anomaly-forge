"""Unit tests for src/utils.py — all 5 utility functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.utils import (
    normalize,
    plot_roc_comparison,
    plot_series_with_anomalies,
    sliding_window,
    temporal_split,
)


# ─── sliding_window ───────────────────────────────────────────────────────────


def test_sliding_window_shape(sample_series: pd.Series) -> None:
    """Windows of size 64 with step=1 over 500 points -> 437 windows."""
    windows = sliding_window(sample_series, window_size=64, step=1)
    assert windows.shape == (500 - 64 + 1, 64)


def test_sliding_window_step(sample_series: pd.Series) -> None:
    """Step=32 should produce the correct number of windows."""
    windows = sliding_window(sample_series, window_size=64, step=32)
    expected_n = (500 - 64) // 32 + 1
    assert windows.shape[0] == expected_n
    assert windows.shape[1] == 64


def test_sliding_window_values() -> None:
    """Verify first and last window values for a tiny known series."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    windows = sliding_window(s, window_size=3, step=1)
    assert windows.shape == (3, 3)
    np.testing.assert_array_almost_equal(windows[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(windows[2], [3.0, 4.0, 5.0])


def test_sliding_window_raises_on_large_window(sample_series: pd.Series) -> None:
    """window_size > len(series) must raise ValueError."""
    with pytest.raises(ValueError, match="window_size"):
        sliding_window(sample_series, window_size=1000)


def test_sliding_window_raises_on_bad_step(sample_series: pd.Series) -> None:
    """step=0 must raise ValueError."""
    with pytest.raises(ValueError, match="step"):
        sliding_window(sample_series, window_size=10, step=0)


def test_sliding_window_returns_writable() -> None:
    """Returned array must be writable (not a read-only view)."""
    s = pd.Series(np.ones(20))
    windows = sliding_window(s, window_size=5, step=1)
    windows[0, 0] = 99.0  # should not raise


# ─── normalize ────────────────────────────────────────────────────────────────


def test_normalize_mean_zero(sample_series: pd.Series) -> None:
    """Normalized series must have zero mean."""
    normed, _ = normalize(sample_series)
    assert abs(normed.mean()) < 1e-10


def test_normalize_std_one(sample_series: pd.Series) -> None:
    """Normalized series must have unit std (population std, matching StandardScaler ddof=0)."""
    normed, _ = normalize(sample_series)
    # StandardScaler uses ddof=0 (population std); pandas .std() uses ddof=1 by default
    assert abs(normed.std(ddof=0) - 1.0) < 1e-6


def test_normalize_returns_scaler(sample_series: pd.Series) -> None:
    """normalize must return a fitted scaler with inverse_transform."""
    _, scaler = normalize(sample_series)
    assert hasattr(scaler, "inverse_transform")


def test_normalize_invert(sample_series: pd.Series) -> None:
    """Inverting the transform must recover the original values."""
    normed, scaler = normalize(sample_series)
    reconstructed = scaler.inverse_transform(
        normed.values.reshape(-1, 1)
    ).flatten()
    np.testing.assert_allclose(reconstructed, sample_series.values, atol=1e-10)


def test_normalize_preserves_index(sample_series: pd.Series) -> None:
    """Normalized series must have the same index as the input."""
    normed, _ = normalize(sample_series)
    pd.testing.assert_index_equal(normed.index, sample_series.index)


def test_normalize_raises_on_empty() -> None:
    """normalize must raise ValueError on an empty series."""
    with pytest.raises(ValueError):
        normalize(pd.Series([], dtype=float))


# ─── temporal_split ───────────────────────────────────────────────────────────


def test_temporal_split_ratio(sample_series: pd.Series) -> None:
    """train + test must cover the full series."""
    train, test = temporal_split(sample_series, test_ratio=0.2)
    assert len(train) + len(test) == len(sample_series)
    assert len(test) == int(len(sample_series) * 0.2)


def test_temporal_split_no_shuffle(sample_series: pd.Series) -> None:
    """Train must always precede test chronologically."""
    train, test = temporal_split(sample_series, test_ratio=0.2)
    assert train.index[-1] < test.index[0]


def test_temporal_split_coverage(sample_series: pd.Series) -> None:
    """Concatenating train and test must equal the original series."""
    train, test = temporal_split(sample_series, test_ratio=0.2)
    combined = pd.concat([train, test])
    pd.testing.assert_series_equal(combined, sample_series)


def test_temporal_split_raises_bad_ratio_zero(sample_series: pd.Series) -> None:
    """test_ratio=0.0 must raise ValueError."""
    with pytest.raises(ValueError):
        temporal_split(sample_series, test_ratio=0.0)


def test_temporal_split_raises_bad_ratio_one(sample_series: pd.Series) -> None:
    """test_ratio=1.0 must raise ValueError."""
    with pytest.raises(ValueError):
        temporal_split(sample_series, test_ratio=1.0)


def test_temporal_split_raises_short_series() -> None:
    """Series with fewer than 4 observations must raise ValueError."""
    tiny = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError):
        temporal_split(tiny, test_ratio=0.2)


# ─── plot_series_with_anomalies ───────────────────────────────────────────────


def test_plot_returns_figure(sample_series: pd.Series) -> None:
    """plot_series_with_anomalies must return a matplotlib Figure."""
    labels = np.zeros(len(sample_series), dtype=int)
    labels[100] = 1
    fig = plot_series_with_anomalies(sample_series, labels)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_plot_raises_mismatched_labels(sample_series: pd.Series) -> None:
    """Labels length != series length must raise ValueError."""
    with pytest.raises(ValueError):
        plot_series_with_anomalies(sample_series, np.zeros(10, dtype=int))
    plt.close("all")


def test_plot_no_anomalies_ok(sample_series: pd.Series) -> None:
    """All-zero labels (no anomalies) must still return a Figure."""
    labels = np.zeros(len(sample_series), dtype=int)
    fig = plot_series_with_anomalies(sample_series, labels, title="Clean Series")
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


# ─── plot_roc_comparison ──────────────────────────────────────────────────────


def test_roc_comparison_returns_figure() -> None:
    """plot_roc_comparison must return a matplotlib Figure."""
    fpr = np.linspace(0, 1, 100)
    tpr = np.linspace(0, 1, 100)
    fig = plot_roc_comparison(fpr, tpr, 0.5, fpr, tpr, 0.7)
    assert isinstance(fig, matplotlib.figure.Figure)
    plt.close("all")


def test_roc_comparison_two_axes() -> None:
    """plot_roc_comparison must produce exactly 2 subplots."""
    fpr = np.linspace(0, 1, 10)
    tpr = np.linspace(0, 1, 10)
    fig = plot_roc_comparison(fpr, tpr, 0.5, fpr, tpr, 0.7, model_name="IForest")
    assert len(fig.axes) == 2
    plt.close("all")
