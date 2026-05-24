"""Shared dataclasses for AnomalyForge inter-module data contracts.

All inter-module data flows through these typed dataclasses.
Never pass raw dicts between modules. Always import from this file.
If you need to add a field, update this file first, then all downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesProfile:
    """Statistical fingerprint of an input time series. Output of profiler.py."""

    series_name: str
    n_observations: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    inferred_freq: str  # e.g. "D", "H", "W"
    distribution_name: str  # "norm", "lognorm", "expon", etc.
    distribution_params: tuple[float, ...]  # scipy fit params
    has_seasonality: bool
    seasonal_period: int  # 0 if no seasonality detected
    seasonal_strength: float  # var(seasonal) / (var(seasonal) + var(residual)), 0.0–1.0
    noise_std: float  # std of STL residuals
    acf_lags: list[float]  # ACF values at lags 1..20
    pacf_lags: list[float]  # PACF values at lags 1..20
    existing_anomaly_indices: list[int]  # indices flagged as pre-existing anomalies
    trend_slope: float  # linear trend slope of STL trend component


@dataclass
class AnomalyCandidate:
    """A candidate anomaly pattern discovered by discoverer.py."""

    candidate_id: str  # UUID string
    anomaly_type: str  # "spike" | "drift" | "flatline" | "noise_burst" | "unknown"
    representative_window: np.ndarray  # shape (window_size,)
    window_indices: list[int]  # start indices of all windows in this cluster
    frequency_count: int  # number of similar windows found
    severity_score: float  # mean ensemble anomaly score, 0.0–1.0
    confirmed: bool = False  # set to True by user in Streamlit UI


@dataclass
class SynthesisConfig:
    """Configuration for synthetic anomaly injection. Input to synthesizer.py."""

    n_spike_events: int = 5
    n_drift_events: int = 3
    n_flatline_events: int = 2
    n_noise_burst_events: int = 3
    magnitude_multiplier: float = 3.0  # spike amplitude = multiplier * local_std
    drift_duration: int = 20  # samples for sustained drift events
    flatline_duration: int = 15  # samples for flatline events
    noise_burst_duration: int = 10  # samples for noise burst events
    noise_burst_multiplier: float = 5.0  # std multiplier for noise bursts
    random_seed: int = 42


@dataclass
class SyntheticDataset:
    """Output of synthesizer.py. Contains series + labels + metadata."""

    original_series: pd.Series
    synthetic_series: pd.Series  # same index as original, with injected anomalies
    labels: np.ndarray  # shape (n,), dtype int, 0 = normal, 1 = anomaly
    anomaly_type_labels: list[str]  # per-sample type string, "" for normal samples
    injection_metadata: list[dict[str, Any]]  # one dict per injection event with keys:
    #   type, start_idx, end_idx, magnitude
    n_injected_events: int
    synthesis_config: SynthesisConfig


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarker.py."""

    test_ratio: float = 0.2  # fraction of data held out for test (chronological)
    window_size: int = 64  # sliding window size for feature extraction
    window_step: int = 1  # step size for sliding window
    iforest_contamination: float = 0.1
    iforest_n_estimators: int = 100
    iforest_threshold_percentile: float = 95.0  # percentile of train scores for threshold
    random_seed: int = 42


@dataclass
class BenchmarkResult:
    """Output of benchmarker.py. IsolationForest before/after metrics.

    Compares real-only training vs. real+synthetic (augmented) training,
    both evaluated on the same held-out test set.
    """

    # IsolationForest — real-only training metrics
    iforest_real_precision: float
    iforest_real_recall: float
    iforest_real_f1: float
    iforest_real_auc: float

    # IsolationForest — augmented (real + synthetic) training metrics
    iforest_augmented_precision: float
    iforest_augmented_recall: float
    iforest_augmented_f1: float
    iforest_augmented_auc: float

    # Test set metadata
    n_test_windows: int
    n_anomalous_test_windows: int
    iforest_real_threshold: float
    iforest_augmented_threshold: float

    # ROC curve arrays (stored as lists for serialisation friendliness)
    real_fpr: list[float]
    real_tpr: list[float]
    augmented_fpr: list[float]
    augmented_tpr: list[float]
