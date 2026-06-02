"""Thompson Sampling recommenders and feedback-loop simulation.

Phase 3: uses posterior uncertainty from BayesianPMF or NumPyroPMF to drive
exploration, then simulates T rounds to show Thompson sampling resists the
popularity-bias amplification produced by greedy point-estimate recommenders.
"""

from __future__ import annotations

import time
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from scripts.models.bayesian_pmf import BayesianPMF, NumPyroPMF


class BayesianThompsonSampler:
    """Thompson Sampling using exact NUTS posterior samples from BayesianPMF.

    At each recommendation step for user u:
      1. Draw ONE sample (chain c, draw d) uniformly from the NUTS trace.
      2. Score all items: score_i = U_sample[u] · V_sample[i]
      3. Recommend top-k items by sampled score.

    This balances exploration (high-variance items occasionally score high)
    and exploitation (consistently good items dominate most draws) without
    any explicit exploration parameter to tune.

    Args:
        bayesian_pmf: Fitted BayesianPMF with self.trace populated.
        random_seed: Controls which (chain, draw) indices are selected.
    """

    def __init__(self, bayesian_pmf: BayesianPMF, random_seed: int = 42) -> None:
        if bayesian_pmf.trace is None:
            raise RuntimeError("BayesianPMF must be fitted before passing to Thompson sampler.")
        self._model = bayesian_pmf
        self._rng = np.random.default_rng(random_seed)

    def recommend(self, user_idx: int, k: int = 10) -> list[int]:
        """Thompson recommendation: sample one posterior draw, score all items.

        Args:
            user_idx: Zero-based user index.
            k: Number of items to recommend.

        Returns:
            Item indices sorted descending by sampled score.
        """
        trace = self._model.trace
        n_chains, n_draws = trace.posterior["U"].shape[:2]
        c = int(self._rng.integers(n_chains))
        d = int(self._rng.integers(n_draws))

        U_sample = trace.posterior["U"].values[c, d, user_idx, :]  # (K,)
        V_sample = trace.posterior["V"].values[c, d]               # (n_items, K)
        scores = V_sample @ U_sample
        return list(np.argsort(scores)[::-1][:k])

    def recommend_greedy(self, user_idx: int, k: int = 10) -> list[int]:
        """Greedy baseline: always recommend by posterior mean score.

        Args:
            user_idx: Zero-based user index.
            k: Number of items to recommend.

        Returns:
            Top-k item indices by posterior mean score.
        """
        return self._model.recommend_topk(user_idx, k)


# ── Feedback-loop simulation ─────────────────────────────────────────────────

class FeedbackLoopSimulator:
    """Simulates a sequential recommendation feedback loop over T rounds.

    Each round:
      1. A recommender (Thompson / Greedy / Random) selects k items per user.
      2. Users "interact": true oracle ratings revealed for recommended items.
      3. New observations appended to training pool.
      4. Model re-fitted on expanded data via NumPyroPMF SVI (fast).
      5. NDCG@10, per-round coverage, cumulative Gini recorded on the fixed
         unbiased test set.

    Key hypothesis tested:
        Greedy → popularity amplifies each round (rich-get-richer feedback loop).
        Thompson → posterior uncertainty on long-tail items drives exploration;
                   recommendation distribution stays broader.
        Random → maximum coverage but poor relevance.

    Why SVI not NUTS:
        Full NUTS per round costs ~8 min on CPU; 10 rounds × 3 strategies = 4 h.
        NumPyroPMF (SVI, mean-field) fits in ~30 s per round.  The trade-off
        (underestimated uncertainty → less Thompson exploration than exact NUTS)
        is documented in the notebook comparison table.

    Args:
        true_ratings: (n_users, n_items) oracle ratings — used as interaction
                      ground truth when a recommended item is "clicked".
        test_ratings: (n_users, n_items) fixed unbiased test set; 0 = unobserved.
        test_mask:    (n_users, n_items) bool; True = in test set.
        n_rounds:     Number of feedback rounds (default 10).
        random_seed:  Controls all simulator stochasticity.
    """

    def __init__(
        self,
        true_ratings: np.ndarray,
        test_ratings: np.ndarray,
        test_mask: np.ndarray,
        n_rounds: int = 10,
        random_seed: int = 42,
    ) -> None:
        self.true_ratings = true_ratings
        self.test_ratings = test_ratings
        self.test_mask = test_mask
        self.n_rounds = n_rounds
        self._rng = np.random.default_rng(random_seed)

    def run(
        self,
        strategy: Literal["thompson", "greedy", "random"],
        initial_train_ratings: np.ndarray,
        initial_train_mask: np.ndarray,
        k: int = 10,
        n_svi_steps: int = 800,
        n_factors: int = 10,
    ) -> dict:
        """Run the feedback loop for self.n_rounds rounds.

        Args:
            strategy:              "thompson", "greedy", or "random".
            initial_train_ratings: Starting (n_users, n_items) rating matrix.
            initial_train_mask:    Starting observation mask.
            k:                     Items recommended per user per round.
            n_svi_steps:           NumPyroPMF SVI training steps per round.
            n_factors:             Latent factor dimension.

        Returns:
            dict with per-round lists:
              'ndcg'    : NDCG@10 on the fixed unbiased test set
              'coverage': fraction of item catalogue recommended that round
              'gini'    : Gini coefficient of cumulative rec-frequency distribution
              'fit_times': wall-clock seconds for each SVI fit
        """
        n_users, n_items = self.true_ratings.shape
        train_r = initial_train_ratings.copy().astype(float)
        train_mask = initial_train_mask.copy()

        cumulative_counts = np.zeros(n_items)
        results: dict[str, list] = {
            "ndcg": [], "coverage": [], "gini": [], "fit_times": []
        }

        print(f"\n── Strategy: {strategy.upper()} ──")

        for t in range(self.n_rounds):
            t0 = time.time()

            # ── Fit SVI model on current training data ────────────────────
            svi = NumPyroPMF(
                n_factors=n_factors,
                random_seed=int(self._rng.integers(0, 2**31)),
                n_steps=n_svi_steps,
            )
            svi.fit(train_r, train_mask)
            fit_time = time.time() - t0

            # ── Recommend for all users (batch to avoid per-user JAX recompile)
            all_recs: dict[int, list[int]] = {}
            round_counts = np.zeros(n_items)
            jax_key = jax.random.PRNGKey(int(self._rng.integers(0, 2**31)))

            if strategy == "greedy":
                # Compute full score matrix once; slice per user
                score_mat = svi._score_matrix(n_samples=10)
                for u in range(n_users):
                    recs = list(np.argsort(score_mat[u])[::-1][:k])
                    all_recs[u] = recs
                    for i in recs:
                        round_counts[i] += 1

            elif strategy == "thompson":
                # Draw one posterior sample per user; vectorised over users
                jax_key, subkey = jax.random.split(jax_key)
                samples = svi._posterior_samples(n_samples=n_users, rng_key=subkey)
                U_s = samples["U"]  # (n_users, n_users_dim, K) — sample i for user i
                V_s = samples["V"]  # (n_users, n_items, K)
                for u in range(n_users):
                    scores = V_s[u] @ U_s[u, u, :]  # user u's sample for themselves
                    recs = list(np.argsort(scores)[::-1][:k])
                    all_recs[u] = recs
                    for i in recs:
                        round_counts[i] += 1

            else:  # random
                for u in range(n_users):
                    recs = self._rng.choice(n_items, size=k, replace=False).tolist()
                    all_recs[u] = recs
                    for i in recs:
                        round_counts[i] += 1

            # ── Observe oracle ratings for newly recommended items ────────
            for u, recs in all_recs.items():
                for i in recs:
                    if not train_mask[u, i]:
                        train_r[u, i] = float(self.true_ratings[u, i])
                        train_mask[u, i] = True

            # ── Compute metrics on fixed test set ─────────────────────────
            cumulative_counts += round_counts
            scores = svi._score_matrix(n_samples=10)
            ndcg = _ndcg_from_scores(scores, self.test_ratings, self.test_mask, k=10)
            coverage = float((round_counts > 0).sum()) / n_items
            gini = _gini(cumulative_counts)

            results["ndcg"].append(ndcg)
            results["coverage"].append(coverage)
            results["gini"].append(gini)
            results["fit_times"].append(fit_time)

            print(
                f"  Round {t+1:2d}/{self.n_rounds}: "
                f"NDCG={ndcg:.4f}  Coverage={coverage:.3f}  "
                f"Gini={gini:.3f}  SVI={fit_time:.0f}s"
            )

        return results


# ── Shared metric helpers ────────────────────────────────────────────────────

def _ndcg_from_scores(
    scores: np.ndarray,
    test_ratings: np.ndarray,
    test_mask: np.ndarray,
    k: int = 10,
) -> float:
    """NDCG@k from a pre-computed (n_users × n_items) score matrix."""
    ndcgs: list[float] = []
    for u in range(test_ratings.shape[0]):
        items = np.where(test_mask[u])[0]
        if len(items) == 0:
            continue
        pred = scores[u, items]
        rel = test_ratings[u, items]
        ranked = np.argsort(pred)[::-1][:k]
        dcg = sum(rel[ranked[i]] / np.log2(i + 2) for i in range(len(ranked)))
        ideal = np.argsort(rel)[::-1][:k]
        idcg = sum(rel[ideal[i]] / np.log2(i + 2) for i in range(len(ideal)))
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient of a recommendation-frequency distribution.

    Returns 0 when all items are recommended equally, 1 when one item
    monopolises all recommendations.
    """
    n = len(counts)
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    sorted_c = np.sort(counts)
    return float(
        (2.0 * np.dot(np.arange(1, n + 1), sorted_c)) / (n * total) - (n + 1.0) / n
    )
