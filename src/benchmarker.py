"""Phase 4 — Model benchmarking.

Trains IsolationForest on real-only vs real+synthetic training data and
evaluates both on the same held-out test set. The test set is always the
chronologically latest portion — no shuffling, no lookahead.

No PyTorch / LSTM — benchmark model is IsolationForest (sklearn) only.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.types import BenchmarkConfig, BenchmarkResult, SyntheticDataset
from src.utils import sliding_window

logger = logging.getLogger(__name__)


def run_benchmark(
    dataset: SyntheticDataset,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Train and evaluate IsolationForest on real-only vs augmented data.

    The benchmark answers: "does training on real+synthetic data improve
    anomaly detection on real held-out data?"

    Args:
        dataset: SyntheticDataset from synthesizer.py.
        config: Benchmark configuration controlling split ratio, window size,
                and IsolationForest hyperparameters.

    Returns:
        BenchmarkResult with before/after IsolationForest metrics and ROC
        curve arrays for plotting.

    Raises:
        ValueError: If dataset has fewer than 100 observations.
        ValueError: If the test set contains no anomalous windows (the
                    benchmark would be meaningless without ground-truth positives).
    """
    n = len(dataset.original_series)
    if n < 100:
        raise ValueError(
            f"dataset must have at least 100 observations, got {n}"
        )

    # ── Temporal split (NEVER shuffle) ────────────────────────────────────────
    split_idx = int(n * (1 - config.test_ratio))
    train_orig = dataset.original_series.iloc[:split_idx]
    test_orig = dataset.original_series.iloc[split_idx:]
    train_labels_arr = dataset.labels[:split_idx]
    test_labels_arr = dataset.labels[split_idx:]
    synth_train = dataset.synthetic_series.iloc[:split_idx]

    logger.info(
        "Temporal split: train=%d, test=%d samples", len(train_orig), len(test_orig)
    )

    # ── Build test windows (always from ORIGINAL series) ─────────────────────
    test_windows, test_window_labels = _make_windows_and_labels(
        test_orig, test_labels_arr, config.window_size, config.window_step
    )
    n_test = len(test_windows)
    n_anomalous = int(test_window_labels.sum())

    if n_anomalous == 0:
        raise ValueError(
            "No anomalous windows found in the test set. "
            "Increase the number of injected events or adjust test_ratio."
        )

    logger.info(
        "Test set: %d windows (%d anomalous, %d normal)",
        n_test, n_anomalous, n_test - n_anomalous,
    )

    # ── Build training windows ────────────────────────────────────────────────
    train_real_windows, _ = _make_windows_and_labels(
        train_orig, train_labels_arr, config.window_size, config.window_step
    )
    train_synth_windows, _ = _make_windows_and_labels(
        synth_train, train_labels_arr, config.window_size, config.window_step
    )

    # ── Train IsolationForest (real-only) ─────────────────────────────────────
    iforest_real, threshold_real = _train_iforest(train_real_windows, config)
    real_scores = _score_iforest(iforest_real, test_windows)

    # ── Train IsolationForest (augmented: real+synthetic) ─────────────────────
    iforest_aug, threshold_aug = _train_iforest(train_synth_windows, config)
    aug_scores = _score_iforest(iforest_aug, test_windows)

    # ── Evaluate both models on the same test set ─────────────────────────────
    p_r, r_r, f_r, auc_r, fpr_r, tpr_r = _compute_metrics(
        real_scores, test_window_labels, threshold_real
    )
    p_a, r_a, f_a, auc_a, fpr_a, tpr_a = _compute_metrics(
        aug_scores, test_window_labels, threshold_aug
    )

    logger.info(
        "IForest real-only   — P=%.3f R=%.3f F1=%.3f AUC=%.3f",
        p_r, r_r, f_r, auc_r,
    )
    logger.info(
        "IForest augmented   — P=%.3f R=%.3f F1=%.3f AUC=%.3f",
        p_a, r_a, f_a, auc_a,
    )

    return BenchmarkResult(
        iforest_real_precision=p_r,
        iforest_real_recall=r_r,
        iforest_real_f1=f_r,
        iforest_real_auc=auc_r,
        iforest_augmented_precision=p_a,
        iforest_augmented_recall=r_a,
        iforest_augmented_f1=f_a,
        iforest_augmented_auc=auc_a,
        n_test_windows=n_test,
        n_anomalous_test_windows=n_anomalous,
        iforest_real_threshold=threshold_real,
        iforest_augmented_threshold=threshold_aug,
        real_fpr=fpr_r,
        real_tpr=tpr_r,
        augmented_fpr=fpr_a,
        augmented_tpr=tpr_a,
    )


# ─── Private helpers ──────────────────────────────────────────────────────────


def _make_windows_and_labels(
    series: pd.Series,
    labels_arr: np.ndarray,
    window_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create (windows, per-window labels) from a series and sample-level labels.

    A window is labeled 1 (anomalous) if ANY sample within it is labeled 1.
    Windows shape: (n_windows, window_size).
    """
    if len(series) < window_size:
        # Return empty arrays rather than crashing; caller handles the empty case
        logger.warning(
            "Series length %d < window_size %d; returning empty windows",
            len(series), window_size,
        )
        return np.empty((0, window_size)), np.empty(0, dtype=int)

    windows = sliding_window(series, window_size, step)
    n_windows = windows.shape[0]

    window_labels = np.array(
        [
            int(labels_arr[i * step: i * step + window_size].any())
            for i in range(n_windows)
        ],
        dtype=int,
    )
    return windows, window_labels


def _train_iforest(
    train_windows: np.ndarray,
    config: BenchmarkConfig,
) -> tuple[IsolationForest, float]:
    """Fit IsolationForest; return (model, anomaly_score_threshold).

    Threshold = percentile(train_anomaly_scores, iforest_threshold_percentile).
    Higher anomaly scores = more anomalous (scores are negated decision_function).
    """
    model = IsolationForest(
        n_estimators=config.iforest_n_estimators,
        contamination=config.iforest_contamination,
        random_state=config.random_seed,
    )
    model.fit(train_windows)

    train_scores = _score_iforest(model, train_windows)
    threshold = float(np.percentile(train_scores, config.iforest_threshold_percentile))
    logger.debug("IForest trained on %d windows; threshold=%.4f", len(train_windows), threshold)
    return model, threshold


def _score_iforest(
    model: IsolationForest,
    windows: np.ndarray,
) -> np.ndarray:
    """Return per-window anomaly scores. Higher value = more anomalous."""
    return -model.decision_function(windows)


def _compute_metrics(
    scores: np.ndarray,
    true_labels: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float, list[float], list[float]]:
    """Compute precision, recall, F1, AUC-ROC, fpr, tpr.

    Handles the edge case where true_labels contains only one class
    (AUC is undefined; returns 0.0 in that case).

    Returns:
        (precision, recall, f1, auc, fpr_list, tpr_list)
    """
    pred = (scores >= threshold).astype(int)

    precision = float(precision_score(true_labels, pred, zero_division=0))
    recall = float(recall_score(true_labels, pred, zero_division=0))
    f1 = float(f1_score(true_labels, pred, zero_division=0))

    try:
        auc = float(roc_auc_score(true_labels, scores))
        fpr_arr, tpr_arr, _ = roc_curve(true_labels, scores)
        fpr_list = fpr_arr.tolist()
        tpr_list = tpr_arr.tolist()
    except ValueError:
        # Only one class present in true_labels
        logger.warning("AUC is undefined (only one class in test labels); returning 0.0")
        auc = 0.0
        fpr_list = [0.0, 1.0]
        tpr_list = [0.0, 1.0]

    return precision, recall, f1, auc, fpr_list, tpr_list
