"""Phase 2 — Anomaly candidate discovery.

Scores every sliding window using a PyOD ensemble (IForest + KNN, majority
vote by mean score), then clusters high-scoring windows by shape similarity
using DTW distance + agglomerative clustering.

Returns one AnomalyCandidate per cluster, sorted by severity descending,
capped at max_candidates. Never raises when no candidates are found —
returns an empty list instead.
"""

import logging
import uuid

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from sklearn.cluster import AgglomerativeClustering

from src.types import AnomalyCandidate, TimeSeriesProfile
from src.utils import sliding_window

logger = logging.getLogger(__name__)

# Try to import the fast C-extension DTW; fall back to pure-Python if unavailable
try:
    from dtaidistance import dtw as _dtw_module
    _dtw_fast_available = True
    logger.debug("dtaidistance C extension available — using fast DTW")
except Exception:  # noqa: BLE001
    _dtw_fast_available = False
    logger.warning("dtaidistance C extension not available; DTW will be slower")


def discover_candidates(
    series: pd.Series,
    profile: TimeSeriesProfile,
    window_size: int = 64,
    step: int = 1,
    contamination: float = 0.1,
    max_candidates: int = 10,
    dtw_threshold: float = 0.5,
) -> list[AnomalyCandidate]:
    """Discover candidate anomaly patterns using PyOD ensemble + DTW clustering.

    Does NOT require the user to specify anomaly types upfront. Surfaces
    candidate patterns for user confirmation in the Streamlit UI.

    Args:
        series: Input time series with DatetimeIndex.
        profile: Statistical profile from profiler.py (used for context only).
        window_size: Sliding window size for feature extraction.
        step: Step size between windows.
        contamination: Assumed anomaly fraction for PyOD models. Must be in (0, 0.5].
        max_candidates: Maximum number of AnomalyCandidates to return.
        dtw_threshold: Distance threshold for AgglomerativeClustering dendrogram cut.

    Returns:
        List of AnomalyCandidate, one per cluster, sorted by severity_score
        descending. Empty list when no anomalous windows are found.
        Never exceeds max_candidates.

    Raises:
        ValueError: If series has fewer than window_size observations.
        ValueError: If contamination is not in (0, 0.5].
    """
    if len(series) < window_size:
        raise ValueError(
            f"series length ({len(series)}) must be >= window_size ({window_size})"
        )
    if not (0 < contamination <= 0.5):
        raise ValueError(
            f"contamination must be in (0, 0.5], got {contamination}"
        )

    logger.info(
        "Discovering anomaly candidates: n=%d, window_size=%d, step=%d, "
        "contamination=%.2f, max_candidates=%d",
        len(series), window_size, step, contamination, max_candidates,
    )

    # ── Build windows ─────────────────────────────────────────────────────────
    windows = sliding_window(series, window_size, step)  # (n_windows, window_size)
    n_windows = windows.shape[0]
    logger.debug("Built %d windows", n_windows)

    if n_windows == 0:
        return []

    # ── Score windows with PyOD ensemble ──────────────────────────────────────
    scores = _build_ensemble_scores(windows, contamination)

    # ── Select high-scoring windows ───────────────────────────────────────────
    high_windows, high_scores, high_indices = _select_high_score_windows(
        windows, scores, contamination
    )
    if len(high_windows) == 0:
        logger.info("No high-scoring windows found; returning empty candidate list")
        return []

    logger.info("Selected %d high-scoring windows for clustering", len(high_windows))

    # ── Handle single-window edge case ────────────────────────────────────────
    if len(high_windows) == 1:
        candidate = _build_candidate(high_windows, high_scores, high_indices)
        return [candidate]

    # ── Cluster by shape (DTW + agglomerative) ────────────────────────────────
    cluster_labels = _cluster_windows_dtw(high_windows, dtw_threshold)
    unique_labels = np.unique(cluster_labels)
    logger.info("DTW clustering produced %d clusters", len(unique_labels))

    # ── Build one AnomalyCandidate per cluster ────────────────────────────────
    candidates: list[AnomalyCandidate] = []
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_windows = high_windows[mask]
        cluster_scores = high_scores[mask]
        cluster_indices = high_indices[mask]
        candidate = _build_candidate(cluster_windows, cluster_scores, cluster_indices)
        candidates.append(candidate)

    # ── Sort by severity descending, cap at max_candidates ────────────────────
    candidates.sort(key=lambda c: c.severity_score, reverse=True)
    return candidates[:max_candidates]


# ─── Private helpers ──────────────────────────────────────────────────────────


def _build_ensemble_scores(
    windows: np.ndarray,
    contamination: float,
) -> np.ndarray:
    """Fit IForest + KNN and return a min-max normalised mean ensemble score.

    Higher score = more likely anomalous. Output in [0, 1].
    """
    n_windows = windows.shape[0]
    n_neighbors = min(5, max(1, n_windows - 1))

    iforest = IForest(contamination=contamination, random_state=42, n_estimators=100)
    knn = KNN(contamination=contamination, n_neighbors=n_neighbors)

    iforest.fit(windows)
    knn.fit(windows)

    # decision_function: higher = more normal for IForest (we negate it)
    # KNN: higher decision score = more anomalous already
    iforest_scores = -iforest.decision_function(windows)  # negate so higher = worse
    knn_scores = knn.decision_function(windows)

    raw = np.column_stack([iforest_scores, knn_scores]).mean(axis=1)

    # Min-max normalise to [0, 1]
    score_range = raw.max() - raw.min()
    if score_range > 0:
        return (raw - raw.min()) / score_range
    return np.zeros_like(raw)


def _select_high_score_windows(
    windows: np.ndarray,
    scores: np.ndarray,
    contamination: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select windows with scores above the (1 - contamination) quantile.

    Returns (high_windows, high_scores, high_indices).
    All three arrays are empty when no windows exceed the threshold.
    """
    threshold = np.quantile(scores, 1.0 - contamination)
    mask = scores >= threshold
    if not mask.any():
        empty = np.empty((0, windows.shape[1]))
        return empty, np.array([]), np.array([], dtype=int)

    indices = np.where(mask)[0]
    return windows[mask], scores[mask], indices


def _cluster_windows_dtw(
    high_windows: np.ndarray,
    dtw_threshold: float,
) -> np.ndarray:
    """Cluster high-scoring windows by DTW distance.

    Uses AgglomerativeClustering with precomputed DTW distance matrix.
    Falls back gracefully when only one window is present.
    """
    n = len(high_windows)
    if n == 1:
        return np.array([0])

    # DTW requires double precision
    windows_list = [row.astype(np.double) for row in high_windows]

    try:
        if _dtw_fast_available:
            dist_matrix = _dtw_module.distance_matrix_fast(
                windows_list, use_c=True, parallel=False
            )
        else:
            dist_matrix = _dtw_module.distance_matrix(windows_list)
        dist_matrix = np.array(dist_matrix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fast DTW failed (%s); retrying with pure-Python DTW", exc)
        try:
            dist_matrix = np.array(_dtw_module.distance_matrix(windows_list))
        except Exception as exc2:  # noqa: BLE001
            logger.error("DTW distance matrix computation failed: %s", exc2)
            # Fall back: assign each window its own cluster
            return np.arange(n)

    # Ensure symmetry and zero diagonal
    np.fill_diagonal(dist_matrix, 0.0)
    dist_matrix = np.maximum(dist_matrix, dist_matrix.T)

    # Replace any NaN/inf with max finite value
    finite_max = np.nanmax(dist_matrix[np.isfinite(dist_matrix)]) if np.isfinite(dist_matrix).any() else 1.0
    dist_matrix = np.where(np.isfinite(dist_matrix), dist_matrix, finite_max)

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=dtw_threshold,
            linkage="average",
            metric="precomputed",
        )
        return clusterer.fit_predict(dist_matrix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("AgglomerativeClustering failed (%s); assigning one cluster", exc)
        return np.zeros(n, dtype=int)


def _guess_anomaly_type(window: np.ndarray) -> str:
    """Heuristic classification of anomaly type from window shape.

    Returns one of: "spike", "flatline", "drift", "noise_burst", "unknown".
    """
    w_std = window.std()
    w_mean = window.mean()

    # Guard against near-zero std
    if w_std < 1e-10:
        return "flatline"

    # Spike: a single extreme point dominates
    peak_deviation = float(np.abs(window - w_mean).max())
    if peak_deviation > 3.0 * w_std:
        return "spike"

    # Flatline: extremely low variance
    if w_std < 0.05 * (abs(w_mean) + 1e-9):
        return "flatline"

    # Drift: large monotonic change from start to end
    start_end_diff = abs(float(window[-1]) - float(window[0]))
    if start_end_diff > 2.0 * w_std:
        return "drift"

    # Noise burst: high variance relative to mean
    if w_std > 2.0 * (abs(w_mean) + 1e-9):
        return "noise_burst"

    return "unknown"


def _build_candidate(
    cluster_windows: np.ndarray,
    cluster_scores: np.ndarray,
    cluster_indices: np.ndarray,
) -> AnomalyCandidate:
    """Build a single AnomalyCandidate from a cluster of windows."""
    rep_idx = int(np.argmax(cluster_scores))
    representative_window = cluster_windows[rep_idx]
    anomaly_type = _guess_anomaly_type(representative_window)
    severity = float(cluster_scores.mean())

    return AnomalyCandidate(
        candidate_id=str(uuid.uuid4()),
        anomaly_type=anomaly_type,
        representative_window=representative_window.copy(),
        window_indices=[int(i) for i in cluster_indices],
        frequency_count=len(cluster_windows),
        severity_score=severity,
        confirmed=False,
    )
