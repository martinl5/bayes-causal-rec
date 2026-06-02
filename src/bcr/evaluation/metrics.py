"""Evaluation metrics for recommendation quality.

All metrics are evaluated exclusively on the **unbiased test set**.
Never pass biased training ratings as test_ratings.
"""

from __future__ import annotations

import numpy as np

from bcr.models.bayesian_pmf import BayesianPMF

_RELEVANCE_THRESHOLD = 3.0  # ratings >= this count as "relevant" for Recall


def ndcg_at_k(
    model: BayesianPMF,
    test_ratings: np.ndarray,
    test_mask: np.ndarray,
    k: int = 10,
) -> float:
    """Normalised Discounted Cumulative Gain at k on the unbiased test set.

    Uses graded relevance (raw rating values).  Averages over users that have
    at least one observed test rating.

    Args:
        model: Fitted BayesianPMF with posterior samples.
        test_ratings: (n_users, n_items) unbiased ratings; 0 = unobserved.
        test_mask: (n_users, n_items) bool; True = observed in test set.
        k: Cutoff rank.

    Returns:
        Mean NDCG@k across users.
    """
    scores = model._score_matrix()
    n_users = test_ratings.shape[0]
    ndcgs: list[float] = []

    for u in range(n_users):
        test_items = np.where(test_mask[u])[0]
        if len(test_items) == 0:
            continue

        pred = scores[u, test_items]
        rel = test_ratings[u, test_items]

        ranked = np.argsort(pred)[::-1][:k]
        dcg = sum(rel[ranked[i]] / np.log2(i + 2) for i in range(len(ranked)))

        ideal = np.argsort(rel)[::-1][:k]
        idcg = sum(rel[ideal[i]] / np.log2(i + 2) for i in range(len(ideal)))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def recall_at_k(
    model: BayesianPMF,
    test_ratings: np.ndarray,
    test_mask: np.ndarray,
    k: int = 10,
    threshold: float = _RELEVANCE_THRESHOLD,
) -> float:
    """Recall@k on the unbiased test set.

    'Relevant' items are those with test rating >= threshold.
    Averages over users that have at least one relevant test item.

    Args:
        model: Fitted BayesianPMF.
        test_ratings: (n_users, n_items) unbiased ratings; 0 = unobserved.
        test_mask: (n_users, n_items) bool; True = observed in test set.
        k: Cutoff rank.
        threshold: Minimum rating to be considered relevant.

    Returns:
        Mean Recall@k across users.
    """
    scores = model._score_matrix()
    n_users = test_ratings.shape[0]
    recalls: list[float] = []

    for u in range(n_users):
        relevant = set(np.where(test_mask[u] & (test_ratings[u] >= threshold))[0].tolist())
        if not relevant:
            continue

        # Rank ALL items by predicted score, recommend top-k
        top_k = set(np.argsort(scores[u])[::-1][:k].tolist())
        recalls.append(len(relevant & top_k) / len(relevant))

    return float(np.mean(recalls)) if recalls else 0.0


def rmse_on_test(
    model: BayesianPMF,
    test_ratings: np.ndarray,
    test_mask: np.ndarray,
) -> float:
    """Root Mean Squared Error on observed entries of the unbiased test set.

    Args:
        model: Fitted BayesianPMF.
        test_ratings: (n_users, n_items) unbiased ratings; 0 = unobserved.
        test_mask: (n_users, n_items) bool; True = observed in test set.

    Returns:
        RMSE (float).
    """
    scores = model._score_matrix()
    user_idx, item_idx = np.where(test_mask)
    preds = scores[user_idx, item_idx]
    actuals = test_ratings[user_idx, item_idx]
    return float(np.sqrt(np.mean((preds - actuals) ** 2)))


def doubly_robust_ndcg(
    direct_model_predictions: np.ndarray,
    propensities: np.ndarray,
    test_ratings: np.ndarray,
    test_mask: np.ndarray,
    k: int = 10,
    clip_max: float = 10.0,
) -> float:
    """Doubly-robust NDCG of the policy induced by the direct model.

    Estimates the NDCG that the *direct model's ranking* achieves, using
    doubly-robust relevance labels.  Crucially, the ranking and the gain come
    from different quantities:

      - **Ranking** is fixed by the direct model (the deployed policy): items
        are ordered by ``direct_model_predictions[u]``.
      - **Gain** uses the DR-corrected relevance
            dr[u, i] = dm_pred[u, i]
                       + O_{ui} * (r_{ui} - dm_pred[u, i]) * clip(1 / p_{ui})
        which is unbiased if either the propensity model OR the rating model is
        correct (Dudík et al., 2011; Saito & Joachims, 2020).

    The earlier version ranked *and* scored by ``dr`` itself, which trivially
    self-ranks to 1.0 — corrected here so the number is informative.

    Caveat: with very small propensities the IPS residual has high variance;
    the inverse propensity is clipped (``clip_max``).  A fully trustworthy
    DR-NDCG needs a real randomised test split (e.g. Coat) or cross-fitting
    of the direct model.

    Args:
        direct_model_predictions: (n_users, n_items) predicted ratings for ALL items.
        propensities: (n_users, n_items) P(O_{ui} = 1) — must be > 0.
        test_ratings: (n_users, n_items) unbiased ratings; 0 = unobserved.
        test_mask: (n_users, n_items) bool; True = observed in test set.
        k: Cutoff rank.
        clip_max: Maximum inverse-propensity weight (variance control).

    Returns:
        Mean DR-NDCG@k across users with at least one test observation.
    """
    ndcgs: list[float] = []
    n_users = test_ratings.shape[0]

    for u in range(n_users):
        test_items = np.where(test_mask[u])[0]
        if len(test_items) == 0:
            continue

        # DR-corrected relevance labels (start from direct-model imputation)
        dr_rel = direct_model_predictions[u].copy()
        for i in test_items:
            inv_p = min(1.0 / max(float(propensities[u, i]), 1e-6), clip_max)
            dr_rel[i] += (test_ratings[u, i] - direct_model_predictions[u, i]) * inv_p
        rel_clipped = np.maximum(dr_rel, 0.0)

        # Rank by the DIRECT MODEL (the deployed policy), not by dr_rel
        ranked = np.argsort(direct_model_predictions[u])[::-1][:k]
        dcg = sum(rel_clipped[ranked[i]] / np.log2(i + 2) for i in range(len(ranked)))

        # Ideal ranking uses the (DR-corrected) relevance labels
        ideal = np.argsort(rel_clipped)[::-1][:k]
        idcg = sum(rel_clipped[ideal[i]] / np.log2(i + 2) for i in range(len(ideal)))

        if idcg > 0:
            ndcgs.append(dcg / idcg)

    return float(np.mean(ndcgs)) if ndcgs else 0.0
