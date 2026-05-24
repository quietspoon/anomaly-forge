"""Unit tests for src/synthesizer.py."""

import numpy as np
import pandas as pd
import pytest

from src.profiler import profile_series
from src.synthesizer import synthesize_anomalies
from src.types import SyntheticDataset, SynthesisConfig


# ─── Shared fixture ───────────────────────────────────────────────────────────


@pytest.fixture
def synthesis_inputs(sample_series: pd.Series):
    """Return (series, profile, confirmed_candidates, config) for tests."""
    profile = profile_series(sample_series)
    config = SynthesisConfig(
        n_spike_events=2,
        n_drift_events=1,
        n_flatline_events=1,
        n_noise_burst_events=1,
        magnitude_multiplier=3.0,
        drift_duration=10,
        flatline_duration=8,
        noise_burst_duration=6,
        noise_burst_multiplier=5.0,
        random_seed=42,
    )
    return sample_series, profile, [], config


# ─── Output type and invariants ───────────────────────────────────────────────


def test_output_type(synthesis_inputs) -> None:
    """synthesize_anomalies must return a SyntheticDataset."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    assert isinstance(result, SyntheticDataset)


def test_output_length_invariant(synthesis_inputs) -> None:
    """Synthetic series, labels, and type_labels must all have the same length as the input."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    n = len(series)
    assert len(result.synthetic_series) == n, (
        f"synthetic_series length {len(result.synthetic_series)} != {n}"
    )
    assert len(result.labels) == n, f"labels length {len(result.labels)} != {n}"
    assert len(result.anomaly_type_labels) == n, (
        f"anomaly_type_labels length {len(result.anomaly_type_labels)} != {n}"
    )


def test_no_mutation_of_input(synthesis_inputs) -> None:
    """The original series must not be modified after calling synthesize_anomalies."""
    series, profile, candidates, config = synthesis_inputs
    original_values = series.values.copy()
    _ = synthesize_anomalies(series, profile, candidates, config)
    np.testing.assert_array_equal(
        series.values, original_values,
        err_msg="Original series was mutated by synthesize_anomalies",
    )


def test_original_series_preserved_in_output(synthesis_inputs) -> None:
    """SyntheticDataset.original_series must equal the input series (by value)."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    pd.testing.assert_series_equal(result.original_series, series)


def test_index_preserved(synthesis_inputs) -> None:
    """synthetic_series must share the DatetimeIndex of the original series."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    pd.testing.assert_index_equal(result.synthetic_series.index, series.index)


# ─── Label coverage ───────────────────────────────────────────────────────────


def test_labels_are_binary(synthesis_inputs) -> None:
    """All label values must be 0 or 1."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    unique = set(np.unique(result.labels))
    assert unique.issubset({0, 1}), f"Unexpected label values: {unique}"


def test_at_least_one_anomaly_injected(synthesis_inputs) -> None:
    """At least one sample must be labeled anomalous when events are requested."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    assert result.labels.sum() > 0, "No anomalies were injected"


def test_n_injected_events_matches_metadata(synthesis_inputs) -> None:
    """n_injected_events must equal len(injection_metadata)."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    assert result.n_injected_events == len(result.injection_metadata), (
        f"n_injected_events={result.n_injected_events} != "
        f"len(metadata)={len(result.injection_metadata)}"
    )


def test_metadata_has_required_keys(synthesis_inputs) -> None:
    """Each injection_metadata dict must contain 'type', 'start_idx', 'end_idx'."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    required_keys = {"type", "start_idx", "end_idx"}
    for i, event in enumerate(result.injection_metadata):
        missing = required_keys - event.keys()
        assert not missing, f"Event {i} missing keys: {missing}"


def test_type_labels_at_anomaly_indices(synthesis_inputs) -> None:
    """Wherever labels == 1, anomaly_type_labels must be non-empty."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    anomaly_indices = np.where(result.labels == 1)[0]
    for i in anomaly_indices:
        assert result.anomaly_type_labels[i] != "", (
            f"Index {i} has label=1 but empty type label"
        )


def test_normal_indices_have_empty_type_label(synthesis_inputs) -> None:
    """Wherever labels == 0, anomaly_type_labels must be empty string."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    normal_indices = np.where(result.labels == 0)[0]
    # Spot-check first 50 normal indices for speed
    for i in normal_indices[:50]:
        assert result.anomaly_type_labels[i] == "", (
            f"Index {i} has label=0 but non-empty type label: "
            f"'{result.anomaly_type_labels[i]}'"
        )


# ─── Injection type coverage ──────────────────────────────────────────────────


def test_spike_type_present(synthesis_inputs) -> None:
    """At least one event in injection_metadata must have type='spike'."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    types = [m["type"] for m in result.injection_metadata]
    assert "spike" in types, f"No spike events found; types: {types}"


def test_drift_type_present(synthesis_inputs) -> None:
    """At least one event in injection_metadata must have type='drift'."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    types = [m["type"] for m in result.injection_metadata]
    assert "drift" in types, f"No drift events found; types: {types}"


def test_flatline_type_present(synthesis_inputs) -> None:
    """At least one event in injection_metadata must have type='flatline'."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    types = [m["type"] for m in result.injection_metadata]
    assert "flatline" in types, f"No flatline events found; types: {types}"


def test_noise_burst_type_present(synthesis_inputs) -> None:
    """At least one event in injection_metadata must have type='noise_burst'."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    types = [m["type"] for m in result.injection_metadata]
    assert "noise_burst" in types, f"No noise_burst events found; types: {types}"


# ─── Synthesis config edge cases ──────────────────────────────────────────────


def test_synthesis_config_zero_events_is_ok(synthesis_inputs) -> None:
    """Config with all-zero event counts must return all-zero labels without error."""
    series, profile, candidates, _ = synthesis_inputs
    config_zero = SynthesisConfig(
        n_spike_events=0,
        n_drift_events=0,
        n_flatline_events=0,
        n_noise_burst_events=0,
    )
    result = synthesize_anomalies(series, profile, candidates, config_zero)
    assert result.labels.sum() == 0, "Expected zero anomaly labels"
    assert result.n_injected_events == 0, "Expected zero injected events"
    assert len(result.injection_metadata) == 0


def test_synthesis_config_name_preserved(synthesis_inputs) -> None:
    """The series name must be preserved in the synthetic output."""
    series, profile, candidates, config = synthesis_inputs
    result = synthesize_anomalies(series, profile, candidates, config)
    assert result.synthetic_series.name == series.name


# ─── Error cases ──────────────────────────────────────────────────────────────


def test_raises_on_empty_series(sample_series: pd.Series) -> None:
    """synthesize_anomalies must raise ValueError for an empty series."""
    idx = pd.date_range("2020-01-01", periods=0, freq="D")
    empty = pd.Series([], index=idx, dtype=float)
    # Need a profile — use the non-empty sample_series for that
    profile = profile_series(sample_series)
    with pytest.raises(ValueError):
        synthesize_anomalies(empty, profile, [], SynthesisConfig())


def test_raises_on_nonpositive_magnitude(synthesis_inputs) -> None:
    """magnitude_multiplier <= 0 must raise ValueError."""
    series, profile, candidates, _ = synthesis_inputs
    bad_config = SynthesisConfig(magnitude_multiplier=0.0)
    with pytest.raises(ValueError, match="magnitude_multiplier"):
        synthesize_anomalies(series, profile, candidates, bad_config)
