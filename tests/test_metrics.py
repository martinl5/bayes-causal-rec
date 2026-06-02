"""Tests for ranking/error metrics.

These use a tiny hand-constructed score matrix so the expected values can be
computed by hand; no model fitting is involved.
"""

from __future__ import annotations

import numpy as np

from bcr.evaluation.metrics import (
    doubly_robust_ndcg,
    ndcg_at_k,
    recall_at_k,
    rmse_on_test,
)


class _StubModel:
    """Minimal stand-in exposing the `_score_matrix` interface metrics need."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = scores

    def _score_matrix(self) -> np.ndarray:
        return self._scores


def test_ndcg_perfect_ranking_is_one():
    # One user; predicted order matches the true rating order exactly.
    scores = np.array([[3.0, 2.0, 1.0]])
    test_ratings = np.array([[3.0, 2.0, 1.0]])
    test_mask = np.array([[True, True, True]])
    assert ndcg_at_k(_StubModel(scores), test_ratings, test_mask, k=3) == 1.0


def test_ndcg_reversed_ranking_below_one():
    # Predicted order is the reverse of the ideal order -> NDCG < 1.
    scores = np.array([[1.0, 2.0, 3.0]])
    test_ratings = np.array([[3.0, 2.0, 1.0]])
    test_mask = np.array([[True, True, True]])
    val = ndcg_at_k(_StubModel(scores), test_ratings, test_mask, k=3)
    assert 0.0 < val < 1.0


def test_ndcg_matches_hand_computation():
    # Ideal: ratings [3,2,1]; predicted ranks them as item0,item2,item1.
    scores = np.array([[3.0, 1.0, 2.0]])
    test_ratings = np.array([[3.0, 2.0, 1.0]])
    test_mask = np.array([[True, True, True]])
    # DCG: 3/log2(2) + 1/log2(3) + 2/log2(4) = 3 + 0.6309 + 1.0
    dcg = 3 / np.log2(2) + 1 / np.log2(3) + 2 / np.log2(4)
    idcg = 3 / np.log2(2) + 2 / np.log2(3) + 1 / np.log2(4)
    expected = dcg / idcg
    got = ndcg_at_k(_StubModel(scores), test_ratings, test_mask, k=3)
    assert abs(got - expected) < 1e-9


def test_recall_at_k_counts_relevant_in_topk():
    # Two relevant items (rating >= 3): items 0 and 3. Top-2 by score: items 0, 1.
    scores = np.array([[5.0, 4.0, 1.0, 0.5]])
    test_ratings = np.array([[3.0, 1.0, 1.0, 4.0]])
    test_mask = np.array([[True, True, True, True]])
    # relevant = {0, 3}; top-2 = {0, 1}; overlap = {0} -> recall = 1/2
    val = recall_at_k(_StubModel(scores), test_ratings, test_mask, k=2, threshold=3.0)
    assert abs(val - 0.5) < 1e-9


def test_rmse_zero_for_exact_predictions():
    scores = np.array([[3.0, 4.0], [2.0, 5.0]])
    test_ratings = scores.copy()
    test_mask = np.array([[True, True], [True, True]])
    assert rmse_on_test(_StubModel(scores), test_ratings, test_mask) == 0.0


def test_rmse_known_value():
    scores = np.array([[3.0]])
    test_ratings = np.array([[5.0]])
    test_mask = np.array([[True]])
    assert abs(rmse_on_test(_StubModel(scores), test_ratings, test_mask) - 2.0) < 1e-9


def test_dr_ndcg_is_not_degenerate():
    # The corrected DR-NDCG ranks by the direct model, so a direct model that
    # already ranks well should score high but a misranked one should not be 1.
    direct = np.array([[1.0, 2.0, 3.0]])  # predicts item2 best
    test_ratings = np.array([[3.0, 2.0, 1.0]])  # truth: item0 best
    test_mask = np.array([[True, True, True]])
    propensities = np.array([[0.5, 0.5, 0.5]])
    val = doubly_robust_ndcg(direct, propensities, test_ratings, test_mask, k=3)
    # A badly-ranked direct model must not self-rank to a perfect 1.0
    assert 0.0 <= val < 1.0
