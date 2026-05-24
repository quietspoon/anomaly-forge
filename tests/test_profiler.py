"""Unit tests for src/profiler.py."""

import numpy as np
import pandas as pd
import pytest

from src.profiler import profile_series
from src.types import TimeSeriesProfile


# ─── Happy path ───────────────────────────────────────────────────────────────


def test_profile_returns_correct_type(sample_series: pd.Series) -> None:
    """profile_series must return a TimeSeriesProfile instance."""
    result = profile_series(sample_series)
    assert isinstance(result, TimeSeriesProfile)


def test_profile_n_observations(sample_series: pd.Series) -> None:
    """n_observations must equal len(series)."""
    result = profile_series(sample_series)
    assert result.n_observations == 500


def test_profile_inferred_freq(sample_series: pd.Series) -> None:
    """Inferred frequency must indicate daily data."""
    result = profile_series(sample_series)
    # Accept common representations of daily frequency
    assert result.inferred_freq in ("D", "day", "<Day>", "B", "24h"), (
        f"Unexpected freq: {result.inferred_freq!r}"
    )


def test_profile_distribution_name(sample_series: pd.Series) -> None:
    """distribution_name must be one of the supported candidates."""
    result = profile_series(sample_series)
    assert result.distribution_name in ("norm", "lognorm", "expon")


def test_profile_distribution_params_not_empty(sample_series: pd.Series) -> None:
    """distribution_params must contain at least 2 parameters."""
    result = profile_series(sample_series)
    assert len(result.distribution_params) >= 2


def test_profile_has_seasonality_bool(sample_series: pd.Series) -> None:
    """has_seasonality must be a bool."""
    result = profile_series(sample_series)
    assert isinstance(result.has_seasonality, bool)


def test_profile_seasonal_strength_range(sample_series: pd.Series) -> None:
    """seasonal_strength must be in [0.0, 1.0]."""
    result = profile_series(sample_series)
    assert 0.0 <= result.seasonal_strength <= 1.0


def test_profile_noise_std_positive(sample_series: pd.Series) -> None:
    """noise_std must be strictly positive for a noisy series."""
    result = profile_series(sample_series)
    assert result.noise_std > 0.0


def test_profile_acf_length(sample_series: pd.Series) -> None:
    """acf_lags must contain exactly 20 values (lags 1..20)."""
    result = profile_series(sample_series)
    assert len(result.acf_lags) == 20


def test_profile_pacf_length(sample_series: pd.Series) -> None:
    """pacf_lags must contain exactly 20 values (lags 1..20)."""
    result = profile_series(sample_series)
    assert len(result.pacf_lags) == 20


def test_profile_existing_anomaly_indices_list(sample_series: pd.Series) -> None:
    """existing_anomaly_indices must be a list of ints."""
    result = profile_series(sample_series)
    assert isinstance(result.existing_anomaly_indices, list)


def test_profile_start_end_time(sample_series: pd.Series) -> None:
    """start_time and end_time must match the series index boundaries."""
    result = profile_series(sample_series)
    assert result.start_time == sample_series.index[0]
    assert result.end_time == sample_series.index[-1]


def test_profile_trend_slope_float(sample_series: pd.Series) -> None:
    """trend_slope must be a Python float."""
    result = profile_series(sample_series)
    assert isinstance(result.trend_slope, float)


def test_profile_seasonal_period_zero_when_no_seasonality() -> None:
    """seasonal_period must be 0 when has_seasonality is False."""
    # Flat + random noise — unlikely to have meaningful seasonality
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=200, freq="D")
    values = rng.normal(0, 1, 200)  # pure noise, no seasonal pattern
    s = pd.Series(values, index=idx)
    result = profile_series(s)
    if not result.has_seasonality:
        assert result.seasonal_period == 0


# ─── Anomaly detection ────────────────────────────────────────────────────────


def test_profile_flags_obvious_spike() -> None:
    """A massive spike must appear in existing_anomaly_indices."""
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    values = rng.normal(0, 0.1, 200)
    values[100] = 50.0  # unmistakable spike
    spiked = pd.Series(values, index=idx, name="spiked")
    result = profile_series(spiked)
    assert 100 in result.existing_anomaly_indices, (
        f"Expected index 100 in anomaly list, got {result.existing_anomaly_indices}"
    )


# ─── Error cases ──────────────────────────────────────────────────────────────


def test_profile_raises_on_non_datetime_index() -> None:
    """series with a RangeIndex must raise ValueError mentioning DatetimeIndex."""
    s = pd.Series(np.random.randn(100))
    with pytest.raises(ValueError, match="DatetimeIndex"):
        profile_series(s)


def test_profile_raises_on_short_series() -> None:
    """series with fewer than 50 observations must raise ValueError."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    s = pd.Series(np.ones(10), index=idx)
    with pytest.raises(ValueError, match="50"):
        profile_series(s)


def test_profile_raises_on_nan() -> None:
    """series containing NaN must raise ValueError mentioning NaN."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    s = pd.Series(np.random.randn(100), index=idx)
    s.iloc[50] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        profile_series(s)
