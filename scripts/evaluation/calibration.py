"""Propensity calibration utilities.

A well-calibrated propensity model is critical for valid IPS correction:
if predicted probabilities are systematically off, IPW estimates are biased.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(
    propensities: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) for a propensity model.

    ECE = sum_b (|B_b| / N) * |mean_predicted(B_b) - mean_actual(B_b)|

    where B_b is the set of predictions falling in bin b.

    Args:
        propensities: Predicted P(observed), shape (N,) — pass as
                      propensity_matrix.flatten().
        actuals: Actual observation indicators (0/1), shape (N,).
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE as a float in [0, 1].  Lower is better.
    """
    propensities = np.asarray(propensities, dtype=float).ravel()
    actuals = np.asarray(actuals, dtype=float).ravel()
    n = len(propensities)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (propensities >= lo) & (propensities < hi)
        if not in_bin.any():
            continue
        mean_pred = propensities[in_bin].mean()
        mean_actual = actuals[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(mean_pred - mean_actual)

    return float(ece)


def plot_calibration_curve(
    propensities: np.ndarray,
    actuals: np.ndarray,
    save_path: str,
    n_bins: int = 10,
) -> None:
    """Plot predicted probability vs actual frequency (reliability diagram).

    Also prints ECE alongside the plot.

    Args:
        propensities: Predicted P(observed), shape (N,).
        actuals: Actual 0/1 observations, shape (N,).
        save_path: File path to save the figure.
        n_bins: Number of bins for the diagram.
    """
    propensities = np.asarray(propensities, dtype=float).ravel()
    actuals = np.asarray(actuals, dtype=float).ravel()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids, mean_preds, mean_acts, counts = [], [], [], []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (propensities >= lo) & (propensities < hi)
        if not in_bin.any():
            continue
        bin_mids.append((lo + hi) / 2)
        mean_preds.append(propensities[in_bin].mean())
        mean_acts.append(actuals[in_bin].mean())
        counts.append(in_bin.sum())

    ece = expected_calibration_error(propensities, actuals, n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.scatter(mean_preds, mean_acts, s=np.array(counts) / max(counts) * 200 + 20,
               color="steelblue", alpha=0.8, zorder=3, label="Bin means")
    for mp, ma, cnt in zip(mean_preds, mean_acts, counts):
        ax.vlines(mp, min(mp, ma), max(mp, ma), color="tomato", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title(f"Calibration curve  (ECE = {ece:.4f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Confidence histogram
    ax2 = axes[1]
    ax2.hist(propensities, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Predicted propensity P(observed)")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of predicted propensities")

    plt.suptitle(f"Propensity model calibration — ECE = {ece:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Calibration curve saved to {save_path}")
