"""Unit tests for src/discoverer.py."""

import numpy as np
import pandas as pd
import pytest

from src.discoverer import discover_candidates
from src.profiler import profile_series
from src.types import AnomalyCandidate


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def discovery_inputs(sample_series: pd.Series):
    """Return (series, profile) for standard discovery tests."""
    profile = profile_series(sample_series)
    return sample_series, profile


# ─── Return type and empty-list behaviour ─────────────────────────────────────


def test_returns_list(discovery_inputs) -> None:
    """discover_candidates must always return a list."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    assert isinstance(result, list)


def test_caps_at_max_candidates(discovery_inputs) -> None:
    """Result must never exceed max_candidates."""
    series, profile = discovery_inputs
    result = discover_candidates(
        series, profile, window_size=32, step=16, max_candidates=3
    )
    assert len(result) <= 3


def test_caps_at_ten_by_default(discovery_inputs) -> None:
    """Default max_candidates=10 must be respected."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    assert len(result) <= 10


def test_empty_on_flat_series() -> None:
    """A flat (constant) series should have no high-severity candidates."""
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    flat = pd.Series(np.ones(200) * 5.0, index=idx, name="flat")
    profile = profile_series(flat)
    result = discover_candidates(flat, profile, window_size=32, step=16)
    # Either empty or all severity scores are very low
    assert isinstance(result, list)
    for c in result:
        assert c.severity_score < 0.5, (
            f"High severity {c.severity_score} on flat series — unexpected"
        )


# ─── Candidate structure ──────────────────────────────────────────────────────


def test_candidate_instances(discovery_inputs) -> None:
    """Every returned item must be an AnomalyCandidate instance."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    for c in result:
        assert isinstance(c, AnomalyCandidate)


def test_candidate_anomaly_type_valid(discovery_inputs) -> None:
    """anomaly_type must be one of the five allowed strings."""
    valid_types = {"spike", "drift", "flatline", "noise_burst", "unknown"}
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    for c in result:
        assert c.anomaly_type in valid_types, (
            f"Unexpected anomaly_type: {c.anomaly_type!r}"
        )


def test_candidate_severity_range(discovery_inputs) -> None:
    """severity_score must be in [0.0, 1.0]."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    for c in result:
        assert 0.0 <= c.severity_score <= 1.0, (
            f"severity_score {c.severity_score} out of [0, 1]"
        )


def test_candidate_representative_window_shape(discovery_inputs) -> None:
    """representative_window must have shape (window_size,)."""
    window_size = 32
    series, profile = discovery_inputs
    result = discover_candidates(
        series, profile, window_size=window_size, step=16
    )
    for c in result:
        assert c.representative_window.shape == (window_size,), (
            f"Expected shape ({window_size},), got {c.representative_window.shape}"
        )


def test_candidate_confirmed_default_false(discovery_inputs) -> None:
    """confirmed must be False by default (user has not confirmed yet)."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    for c in result:
        assert c.confirmed is False


def test_candidates_sorted_by_severity(discovery_inputs) -> None:
    """Candidates must be sorted by severity_score in descending order."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    if len(result) > 1:
        for i in range(len(result) - 1):
            assert result[i].severity_score >= result[i + 1].severity_score, (
                f"Candidate {i} severity {result[i].severity_score} < "
                f"candidate {i+1} severity {result[i+1].severity_score}"
            )


def test_candidate_id_unique(discovery_inputs) -> None:
    """Each candidate must have a distinct candidate_id."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    ids = [c.candidate_id for c in result]
    assert len(ids) == len(set(ids)), "Duplicate candidate IDs found"


def test_window_indices_are_ints(discovery_inputs) -> None:
    """window_indices must be a list of Python ints."""
    series, profile = discovery_inputs
    result = discover_candidates(series, profile, window_size=32, step=16)
    for c in result:
        for idx in c.window_indices:
            assert isinstance(idx, int), f"window_index {idx!r} is not int"


# ─── Obvious anomaly detection ────────────────────────────────────────────────


def test_detects_obvious_spikes() -> None:
    """Massive spikes must be discovered as candidates."""
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    rng = np.random.default_rng(7)
    values = rng.normal(0, 0.1, 300)
    # Three well-separated large spikes
    values[50] = 20.0
    values[150] = -20.0
    values[250] = 20.0
    spiked = pd.Series(values, index=idx, name="spiked")
    profile = profile_series(spiked)
    result = discover_candidates(
        spiked, profile, window_size=16, step=4, contamination=0.1
    )
    assert len(result) > 0, "Expected at least one candidate from obvious spikes"


# ─── Error cases ──────────────────────────────────────────────────────────────


def test_raises_on_series_shorter_than_window() -> None:
    """Series shorter than window_size must raise ValueError."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    short = pd.Series(np.random.randn(30), index=idx)
    # Profile must come from a valid series
    valid_idx = pd.date_range("2020-01-01", periods=100, freq="D")
    valid_profile = profile_series(
        pd.Series(np.random.randn(100), index=valid_idx)
    )
    with pytest.raises(ValueError, match="window_size"):
        discover_candidates(short, valid_profile, window_size=64)


def test_raises_on_bad_contamination(discovery_inputs) -> None:
    """contamination outside (0, 0.5] must raise ValueError."""
    series, profile = discovery_inputs
    with pytest.raises(ValueError, match="contamination"):
        discover_candidates(series, profile, contamination=0.9)


def test_raises_on_zero_contamination(discovery_inputs) -> None:
    """contamination=0 must raise ValueError."""
    series, profile = discovery_inputs
    with pytest.raises(ValueError, match="contamination"):
        discover_candidates(series, profile, contamination=0.0)
