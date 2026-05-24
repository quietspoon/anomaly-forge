"""Phase 1 — Time series statistical profiling.

Characterises the baseline distribution, seasonality, noise floor,
autocorrelation structure, and flags pre-existing anomalies.
All business logic is in private helper functions; profile_series
is the single public entry point.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf

from src.types import TimeSeriesProfile

logger = logging.getLogger(__name__)

# Mapping from common pandas frequency strings to seasonal periods
_FREQ_TO_PERIOD: dict[str, int] = {
    "D": 365,
    "H": 24,
    "h": 24,
    "W": 52,
    "W-SUN": 52,
    "MS": 12,
    "M": 12,
    "ME": 12,
    "QS": 4,
    "Q": 4,
    "T": 60,
    "min": 60,
    "s": 3600,
    "S": 3600,
    "B": 365,
}


def profile_series(series: pd.Series) -> TimeSeriesProfile:
    """Compute a statistical fingerprint for a time series.

    Args:
        series: Input time series with a DatetimeIndex and at least 50
                non-NaN observations. The series is read-only — it is
                never modified.

    Returns:
        TimeSeriesProfile populated with distribution, seasonality,
        noise, autocorrelation, and anomaly-flagging fields.

    Raises:
        ValueError: If series does not have a DatetimeIndex.
        ValueError: If series contains NaN values.
        ValueError: If series has fewer than 50 observations.
    """
    # ── Validate inputs ───────────────────────────────────────────────────────
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            "series must have a DatetimeIndex, "
            f"got {type(series.index).__name__}"
        )
    if series.isna().any():
        raise ValueError(
            "series contains NaN values. Remove or impute NaNs before profiling."
        )
    if len(series) < 50:
        raise ValueError(
            f"series must have at least 50 observations, got {len(series)}"
        )

    logger.info("Profiling series '%s' (%d observations)", series.name, len(series))

    # ── Infer frequency ───────────────────────────────────────────────────────
    freq_str = _get_freq_str(series)

    # ── Seasonal period ───────────────────────────────────────────────────────
    period = _infer_period(series)
    logger.debug("Inferred seasonal period: %d (freq=%s)", period, freq_str)

    # ── Distribution fit ──────────────────────────────────────────────────────
    dist_name, dist_params = _fit_distribution(series.values)
    logger.debug("Best-fit distribution: %s %s", dist_name, dist_params)

    # ── STL decomposition ─────────────────────────────────────────────────────
    trend, seasonal, residual = _run_stl(series, period)

    # ── Seasonal strength & flag ──────────────────────────────────────────────
    seasonal_strength = _compute_seasonal_strength(seasonal, residual)
    has_seasonality = seasonal_strength > 0.1

    # ── ACF / PACF ────────────────────────────────────────────────────────────
    acf_lags, pacf_lags = _compute_acf_pacf(residual)

    # ── Noise floor ───────────────────────────────────────────────────────────
    noise_std = float(np.std(residual))

    # ── Pre-existing anomaly detection ────────────────────────────────────────
    anomaly_indices = _flag_existing_anomalies(residual, noise_std)
    if anomaly_indices:
        logger.info("Flagged %d pre-existing anomalies", len(anomaly_indices))

    # ── Trend slope ───────────────────────────────────────────────────────────
    trend_slope = _compute_trend_slope(trend)

    return TimeSeriesProfile(
        series_name=series.name if series.name is not None else "",
        n_observations=len(series),
        start_time=series.index[0],
        end_time=series.index[-1],
        inferred_freq=freq_str,
        distribution_name=dist_name,
        distribution_params=dist_params,
        has_seasonality=has_seasonality,
        seasonal_period=period if has_seasonality else 0,
        seasonal_strength=seasonal_strength,
        noise_std=noise_std,
        acf_lags=acf_lags,
        pacf_lags=pacf_lags,
        existing_anomaly_indices=anomaly_indices,
        trend_slope=trend_slope,
    )


# ─── Private helpers ──────────────────────────────────────────────────────────


def _get_freq_str(series: pd.Series) -> str:
    """Return a human-readable frequency string from the DatetimeIndex."""
    idx = series.index
    if idx.freq is not None:
        return str(idx.freqstr)
    inferred = pd.infer_freq(idx)
    if inferred:
        return inferred
    return "unknown"


def _infer_period(series: pd.Series) -> int:
    """Map DatetimeIndex frequency to a seasonal period integer.

    Returns 7 (weekly) as default when frequency is unknown.
    Returns 0 only when the series is too short for any meaningful period.
    """
    freq_str = _get_freq_str(series)
    # Try exact match first, then prefix match
    period = _FREQ_TO_PERIOD.get(freq_str)
    if period is None:
        # Prefix match (e.g. "W-MON" -> "W")
        for key, val in _FREQ_TO_PERIOD.items():
            if freq_str.startswith(key):
                period = val
                break
    if period is None:
        period = 7  # default fallback
    return period


def _fit_distribution(values: np.ndarray) -> tuple[str, tuple[float, ...]]:
    """Fit norm, lognorm, and expon distributions; return the best by AIC.

    For lognorm, values are shifted to be positive if necessary.
    """
    candidates = [
        ("norm", scipy_stats.norm),
        ("expon", scipy_stats.expon),
    ]

    # For lognorm, values must be strictly positive
    shifted = values.copy()
    min_val = shifted.min()
    if min_val <= 0:
        shifted = shifted - min_val + 1.0
    candidates.append(("lognorm", scipy_stats.lognorm))

    best_name = "norm"
    best_params: tuple[float, ...] = ()
    best_aic = np.inf

    for name, dist in candidates:
        try:
            data = shifted if name == "lognorm" else values
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            if aic < best_aic:
                best_aic = aic
                best_name = name
                best_params = tuple(float(p) for p in params)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Distribution fit failed for %s: %s", name, exc)

    return best_name, best_params


def _run_stl(
    series: pd.Series,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run STL decomposition, returning (trend, seasonal, residual) arrays.

    Falls back to trivial decomposition when period < 2 (e.g. irregular data).
    """
    if period < 2 or len(series) < 2 * period:
        logger.debug("Skipping STL (period=%d, n=%d); using trivial decomposition", period, len(series))
        trend = np.full(len(series), series.mean())
        seasonal = np.zeros(len(series))
        residual = series.values - trend
        return trend, seasonal, residual

    # seasonal window must be odd and >= 7
    seasonal_window = max(7, period | 1)  # bitwise OR 1 ensures odd
    try:
        stl = STL(series, period=period, seasonal=seasonal_window, robust=True)
        result = stl.fit()
        return result.trend, result.seasonal, result.resid
    except Exception as exc:  # noqa: BLE001
        logger.warning("STL decomposition failed (%s); using trivial fallback", exc)
        trend = np.full(len(series), series.mean())
        seasonal = np.zeros(len(series))
        residual = series.values - trend
        return trend, seasonal, residual


def _compute_seasonal_strength(
    seasonal: np.ndarray,
    residual: np.ndarray,
) -> float:
    """Compute seasonal strength as var(seasonal) / (var(seasonal) + var(residual)).

    Returns 0.0 if the denominator is zero (e.g. flat series).
    Result is clamped to [0.0, 1.0].
    """
    var_seasonal = np.var(seasonal)
    var_residual = np.var(residual)
    denom = var_seasonal + var_residual
    if denom == 0:
        return 0.0
    return float(np.clip(var_seasonal / denom, 0.0, 1.0))


def _compute_acf_pacf(
    residual: np.ndarray,
    n_lags: int = 20,
) -> tuple[list[float], list[float]]:
    """Compute ACF and PACF of the residual at lags 1..n_lags.

    Lag 0 (always 1.0 for ACF) is excluded from the returned lists.
    """
    acf_values = acf(residual, nlags=n_lags, fft=True)
    pacf_values = pacf(residual, nlags=n_lags, method="ols")
    # Exclude lag 0
    return list(acf_values[1:].astype(float)), list(pacf_values[1:].astype(float))


def _flag_existing_anomalies(
    residual: np.ndarray,
    noise_std: float,
) -> list[int]:
    """Flag indices where abs(residual) > 3.5 * noise_std as pre-existing anomalies.

    Returns an empty list when noise_std is zero (flat series).
    """
    if noise_std == 0:
        return []
    threshold = 3.5 * noise_std
    return [int(i) for i, v in enumerate(residual) if abs(v) > threshold]


def _compute_trend_slope(trend: np.ndarray) -> float:
    """Fit a linear regression to the trend component; return the slope."""
    x = np.arange(len(trend), dtype=float)
    coeffs = np.polyfit(x, trend, 1)
    return float(coeffs[0])
