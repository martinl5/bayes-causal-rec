"""Tests for the Expected Calibration Error implementation."""

from __future__ import annotations

import numpy as np

from bcr.evaluation.calibration import expected_calibration_error


def test_ece_zero_for_perfectly_calibrated():
    # In each bin the predicted probability equals the empirical frequency.
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.uniform(0, 1, n)
    actuals = (rng.uniform(0, 1, n) < p).astype(float)
    ece = expected_calibration_error(p, actuals, n_bins=10)
    assert ece < 0.02  # near zero up to sampling noise


def test_ece_large_for_systematically_overconfident():
    # Predict 0.9 everywhere but only 10% are positive -> ECE near 0.8.
    p = np.full(1000, 0.9)
    actuals = np.zeros(1000)
    actuals[:100] = 1.0
    ece = expected_calibration_error(p, actuals, n_bins=10)
    assert ece > 0.7


def test_ece_in_unit_interval():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    actuals = rng.integers(0, 2, 500).astype(float)
    ece = expected_calibration_error(p, actuals, n_bins=10)
    assert 0.0 <= ece <= 1.0
