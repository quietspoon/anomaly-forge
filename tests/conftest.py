"""Shared pytest fixtures for AnomalyForge test suite."""

import os

import numpy as np
import pandas as pd
import pytest

# Ensure matplotlib uses a non-interactive backend in all environments
# (matches MPLBACKEND=Agg set in docker-compose.yml)
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="module")
def sample_series() -> pd.Series:
    """Return a 500-point daily DatetimeIndex sine-wave series for testing.

    The series has a clear sinusoidal pattern with small Gaussian noise,
    making it suitable for profiler, synthesizer, and discoverer tests.
    """
    idx = pd.date_range("2020-01-01", periods=500, freq="D")
    rng = np.random.default_rng(42)
    values = np.sin(np.linspace(0, 8 * np.pi, 500)) + rng.normal(0, 0.1, 500)
    return pd.Series(values, index=idx, name="test_series")
