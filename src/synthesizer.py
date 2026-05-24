"""Phase 3 — Synthetic anomaly injection.

Injects labeled rare events into a copy of the input series while
preserving the baseline statistical fingerprint outside injection windows.
The input series is NEVER mutated.

Injection types:
  spike       — single-point extreme value
  drift       — sustained linear ramp over a window
  flatline    — constant value held for a window
  noise_burst — high-amplitude Gaussian noise for a window
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.types import (
    AnomalyCandidate,
    SyntheticDataset,
    SynthesisConfig,
    TimeSeriesProfile,
)

logger = logging.getLogger(__name__)


def synthesize_anomalies(
    series: pd.Series,
    profile: TimeSeriesProfile,
    confirmed_candidates: list[AnomalyCandidate],
    config: SynthesisConfig,
) -> SyntheticDataset:
    """Inject labeled anomalies into a copy of the series.

    The original series is never modified. Every injected event has a
    ground-truth label in the output. The output series always has the
    same length as the input series.

    Args:
        series: Original time series. Never mutated.
        profile: Statistical profile from profiler.py. Used for noise_std fallback.
        confirmed_candidates: User-confirmed anomaly candidates. May be empty.
        config: Injection configuration (counts, magnitudes, durations).

    Returns:
        SyntheticDataset with synthetic_series, labels, and full injection metadata.
        len(synthetic_series) == len(series) is always guaranteed.

    Raises:
        ValueError: If series is empty.
        ValueError: If config.magnitude_multiplier <= 0.
    """
    if len(series) == 0:
        raise ValueError("series must be non-empty")
    if config.magnitude_multiplier <= 0:
        raise ValueError(
            f"config.magnitude_multiplier must be > 0, got {config.magnitude_multiplier}"
        )

    n = len(series)
    # Work on a copy — NEVER mutate the original
    values: np.ndarray = series.values.astype(np.float64).copy()
    labels: np.ndarray = np.zeros(n, dtype=int)
    type_labels: list[str] = [""] * n

    rng = np.random.default_rng(config.random_seed)
    occupied: set[int] = set()
    all_metadata: list[dict[str, Any]] = []

    # Per-sample local std via rolling window; fall back to profile.noise_std
    rolling_std = (
        pd.Series(values)
        .rolling(window=min(32, n), min_periods=1)
        .std()
        .fillna(profile.noise_std)
        .values
    )

    # ── Inject spikes ─────────────────────────────────────────────────────────
    spike_sites = _choose_injection_sites(n, config.n_spike_events, 1, rng, occupied)
    for site in spike_sites:
        meta = _inject_spike(
            values, labels, type_labels, site,
            float(rolling_std[site]), config.magnitude_multiplier, rng,
        )
        occupied.update(range(site, site + 1))
        all_metadata.append(meta)

    # ── Inject drifts ─────────────────────────────────────────────────────────
    drift_sites = _choose_injection_sites(
        n, config.n_drift_events, config.drift_duration, rng, occupied
    )
    for site in drift_sites:
        meta = _inject_drift(
            values, labels, type_labels, site,
            config.drift_duration, float(rolling_std[site]), config.magnitude_multiplier,
        )
        occupied.update(range(site, min(site + config.drift_duration, n)))
        all_metadata.append(meta)

    # ── Inject flatlines ──────────────────────────────────────────────────────
    flatline_sites = _choose_injection_sites(
        n, config.n_flatline_events, config.flatline_duration, rng, occupied
    )
    for site in flatline_sites:
        meta = _inject_flatline(values, labels, type_labels, site, config.flatline_duration)
        occupied.update(range(site, min(site + config.flatline_duration, n)))
        all_metadata.append(meta)

    # ── Inject noise bursts ───────────────────────────────────────────────────
    burst_sites = _choose_injection_sites(
        n, config.n_noise_burst_events, config.noise_burst_duration, rng, occupied
    )
    for site in burst_sites:
        meta = _inject_noise_burst(
            values, labels, type_labels, site,
            config.noise_burst_duration, float(rolling_std[site]),
            config.noise_burst_multiplier, rng,
        )
        occupied.update(range(site, min(site + config.noise_burst_duration, n)))
        all_metadata.append(meta)

    n_injected = len(all_metadata)
    logger.info(
        "Injected %d events (%d anomalous samples / %d total)",
        n_injected,
        int(labels.sum()),
        n,
    )

    synthetic_series = pd.Series(values, index=series.index, name=series.name)

    return SyntheticDataset(
        original_series=series,
        synthetic_series=synthetic_series,
        labels=labels,
        anomaly_type_labels=type_labels,
        injection_metadata=all_metadata,
        n_injected_events=n_injected,
        synthesis_config=config,
    )


# ─── Private helpers ──────────────────────────────────────────────────────────


def _choose_injection_sites(
    series_len: int,
    n_events: int,
    duration: int,
    rng: np.random.Generator,
    occupied: set[int],
) -> list[int]:
    """Select non-overlapping start indices for injection events.

    A candidate start index i is valid when no index in [i, i+duration) is
    already occupied. Returns at most n_events sites (may be fewer when the
    series is too short or fully occupied).
    """
    if n_events <= 0 or duration <= 0:
        return []
    max_start = series_len - duration
    if max_start < 0:
        logger.warning(
            "Series too short (len=%d) for injection duration=%d; skipping",
            series_len, duration,
        )
        return []

    # Build the list of valid start positions
    valid: list[int] = [
        i for i in range(max_start + 1)
        if not any(j in occupied for j in range(i, i + duration))
    ]
    if not valid:
        return []

    n_select = min(n_events, len(valid))
    chosen = rng.choice(len(valid), size=n_select, replace=False)
    return sorted(int(valid[c]) for c in chosen)


def _inject_spike(
    values: np.ndarray,
    labels: np.ndarray,
    type_labels: list[str],
    start_idx: int,
    local_std: float,
    magnitude_multiplier: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Inject a single-point spike. Returns injection metadata dict."""
    direction = rng.choice([-1.0, 1.0])
    magnitude = magnitude_multiplier * max(local_std, 1e-6)
    values[start_idx] += direction * magnitude
    labels[start_idx] = 1
    type_labels[start_idx] = "spike"
    return {
        "type": "spike",
        "start_idx": start_idx,
        "end_idx": start_idx,
        "magnitude": magnitude,
    }


def _inject_drift(
    values: np.ndarray,
    labels: np.ndarray,
    type_labels: list[str],
    start_idx: int,
    duration: int,
    local_std: float,
    magnitude_multiplier: float,
) -> dict[str, Any]:
    """Inject a sustained linear drift. Returns injection metadata dict."""
    end_idx = min(start_idx + duration, len(values))
    actual_len = end_idx - start_idx
    target = values[start_idx] + magnitude_multiplier * max(local_std, 1e-6)
    ramp = np.linspace(values[start_idx], target, actual_len)
    values[start_idx:end_idx] = ramp
    labels[start_idx:end_idx] = 1
    type_labels[start_idx:end_idx] = ["drift"] * actual_len
    magnitude = magnitude_multiplier * max(local_std, 1e-6)
    return {
        "type": "drift",
        "start_idx": start_idx,
        "end_idx": end_idx - 1,
        "magnitude": magnitude,
    }


def _inject_flatline(
    values: np.ndarray,
    labels: np.ndarray,
    type_labels: list[str],
    start_idx: int,
    duration: int,
) -> dict[str, Any]:
    """Hold values constant for the duration window. Returns metadata dict."""
    end_idx = min(start_idx + duration, len(values))
    actual_len = end_idx - start_idx
    values[start_idx:end_idx] = values[start_idx]
    labels[start_idx:end_idx] = 1
    type_labels[start_idx:end_idx] = ["flatline"] * actual_len
    return {
        "type": "flatline",
        "start_idx": start_idx,
        "end_idx": end_idx - 1,
        "magnitude": 0.0,
    }


def _inject_noise_burst(
    values: np.ndarray,
    labels: np.ndarray,
    type_labels: list[str],
    start_idx: int,
    duration: int,
    local_std: float,
    noise_burst_multiplier: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Add high-amplitude Gaussian noise for the duration window. Returns metadata dict."""
    end_idx = min(start_idx + duration, len(values))
    actual_len = end_idx - start_idx
    noise_std = noise_burst_multiplier * max(local_std, 1e-6)
    noise = rng.normal(0, noise_std, actual_len)
    values[start_idx:end_idx] += noise
    labels[start_idx:end_idx] = 1
    type_labels[start_idx:end_idx] = ["noise_burst"] * actual_len
    return {
        "type": "noise_burst",
        "start_idx": start_idx,
        "end_idx": end_idx - 1,
        "magnitude": noise_std,
    }
