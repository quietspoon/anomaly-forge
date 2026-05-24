"""Shared utility functions for AnomalyForge.

All functions are pure (no side effects), take explicit inputs,
and return explicit outputs. No global state.
"""

import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # Headless rendering — required for Docker (no display server)

logger = logging.getLogger(__name__)


def sliding_window(
    series: pd.Series,
    window_size: int,
    step: int = 1,
) -> np.ndarray:
    """Extract overlapping windows from a time series.

    Args:
        series: Input time series. Values are extracted in index order.
        window_size: Number of samples per window. Must be <= len(series).
        step: Number of samples to advance between window starts. Must be >= 1.

    Returns:
        Array of shape (n_windows, window_size).
        n_windows = max(0, (len(series) - window_size) // step + 1).

    Raises:
        ValueError: If window_size > len(series).
        ValueError: If step < 1.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if window_size > len(series):
        raise ValueError(
            f"window_size ({window_size}) must be <= len(series) ({len(series)})"
        )

    values = series.values.astype(np.float64)
    # sliding_window_view returns a read-only view; shape (n_possible, window_size)
    view = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size)
    # Apply step by slicing rows
    windowed = view[::step]
    logger.debug(
        "sliding_window: series_len=%d, window_size=%d, step=%d -> %d windows",
        len(series),
        window_size,
        step,
        len(windowed),
    )
    return windowed.copy()  # return a writable copy


def normalize(
    series: pd.Series,
) -> tuple[pd.Series, StandardScaler]:
    """Z-score normalize a series, returning both the result and the fitted scaler.

    The scaler is fit on the input series so it can be used to invert
    the transform later. Both normalized series and scaler are returned
    so the caller can reconstruct the original scale.

    Args:
        series: Input series to normalize. Must be non-empty.

    Returns:
        Tuple of (normalized_series, fitted_scaler).
        normalized_series has the same index as the input.
        fitted_scaler can be used with scaler.inverse_transform() later.

    Raises:
        ValueError: If series is empty.
    """
    if len(series) == 0:
        raise ValueError("series must be non-empty")

    scaler = StandardScaler()
    values = series.values.reshape(-1, 1)
    normalized_values = scaler.fit_transform(values).flatten()
    normalized_series = pd.Series(
        normalized_values, index=series.index, name=series.name
    )
    logger.debug(
        "normalize: mean=%.4f, std=%.4f -> normalized mean=%.6f, std=%.6f",
        series.mean(),
        series.std(),
        normalized_series.mean(),
        normalized_series.std(),
    )
    return normalized_series, scaler


def temporal_split(
    series: pd.Series,
    test_ratio: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """Split a time series into train and test sets preserving temporal order.

    NEVER shuffles. Test set is always the chronologically latest portion.
    Scaler / normalizer must be fit on the train set and applied to the test set.

    Args:
        series: Input time series. Must have at least 4 observations.
        test_ratio: Fraction of data for the test set. Must be in (0, 1).

    Returns:
        Tuple of (train_series, test_series).
        train_series contains the earliest (1 - test_ratio) fraction.
        test_series contains the chronologically latest test_ratio fraction.

    Raises:
        ValueError: If test_ratio is not in (0, 1).
        ValueError: If series has fewer than 4 observations.
    """
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    if len(series) < 4:
        raise ValueError(
            f"series must have at least 4 observations, got {len(series)}"
        )

    split_idx = int(len(series) * (1 - test_ratio))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    logger.debug(
        "temporal_split: total=%d, train=%d, test=%d (ratio=%.2f)",
        len(series),
        len(train),
        len(test),
        test_ratio,
    )
    return train, test


def plot_series_with_anomalies(
    series: pd.Series,
    labels: np.ndarray,
    title: str = "Time Series with Anomalies",
) -> matplotlib.figure.Figure:
    """Plot a time series with anomalous points highlighted in red.

    Args:
        series: Time series to plot. Values plotted as a blue line.
        labels: Binary array of shape (len(series),). 1 = anomaly, 0 = normal.
        title: Figure title string.

    Returns:
        Matplotlib Figure object. Does NOT call plt.show() — caller is
        responsible for displaying or saving.

    Raises:
        ValueError: If len(labels) != len(series).
    """
    if len(labels) != len(series):
        raise ValueError(
            f"len(labels) ({len(labels)}) must equal len(series) ({len(series)})"
        )

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 4))

    # Full series in blue
    ax.plot(series.index, series.values, color="steelblue", linewidth=0.8, label="Series")

    # Anomaly points in red
    anomaly_mask = labels == 1
    if anomaly_mask.any():
        ax.scatter(
            series.index[anomaly_mask],
            series.values[anomaly_mask],
            color="red",
            s=20,
            zorder=5,
            label="Anomaly",
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_roc_comparison(
    real_fpr: np.ndarray,
    real_tpr: np.ndarray,
    real_auc: float,
    augmented_fpr: np.ndarray,
    augmented_tpr: np.ndarray,
    augmented_auc: float,
    model_name: str = "Model",
) -> matplotlib.figure.Figure:
    """Plot side-by-side ROC curves for real-only vs augmented training.

    Args:
        real_fpr: False positive rates for the real-only trained model.
        real_tpr: True positive rates for the real-only trained model.
        real_auc: AUC score for the real-only model.
        augmented_fpr: False positive rates for the augmented trained model.
        augmented_tpr: True positive rates for the augmented trained model.
        augmented_auc: AUC score for the augmented model.
        model_name: Label prefix for legend entries (e.g. "IsolationForest").

    Returns:
        Matplotlib Figure with two side-by-side subplots.
        Does NOT call plt.show().
    """
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, fpr, tpr, auc, condition in [
        (axes[0], real_fpr, real_tpr, real_auc, "Real-Only"),
        (axes[1], augmented_fpr, augmented_tpr, augmented_auc, "Augmented"),
    ]:
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
        ax.plot(
            fpr,
            tpr,
            color="steelblue",
            linewidth=2,
            label=f"{model_name} ({condition})\nAUC = {auc:.3f}",
        )
        ax.set_title(f"ROC — {model_name} ({condition})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="lower right")

    fig.tight_layout()
    return fig
