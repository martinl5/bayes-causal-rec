"""Deconfounded Recommender — two-stage causal debiasing.

Implements the approach of Wang, Liang, Charlin & Blei (2018)
"The Deconfounded Recommender: A Causal Inference Approach to Recommendation".

Stage 1 — Factor model on the exposure matrix:
    Fits a low-rank Bayesian model on the binary observation mask to extract
    a latent substitute confounder Z ∈ R^{n_users × n_z}.  Z approximates
    the unmeasured confounder (session context, user mood, browsing history)
    that simultaneously drives exposure and true ratings.

Stage 2 — Outcome model conditioned on Z:
    Fits a Bayesian PMF-style rating model that includes Z as an additional
    user-side feature, implementing approximate backdoor adjustment:
        P(R_{ui} | do(A_{ui}=a)) ≈ E_Z[ P(R_{ui} | A_{ui}=a, Z) ]

IMPORTANT CAVEAT (Ogburn et al., 2020):
    The substitute-confounder identification strategy requires strong
    assumptions — specifically that Z contains all common causes of
    (A_{u·}, R_{ui}).  If unmeasured item-level confounders exist,
    identification may fail even with a perfect factor model.
    This approach is most trustworthy when evaluated on a genuinely
    unbiased test set (e.g., Coat's uniform-random test split).
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


class DeconfoundedRecommender:
    """Two-stage causal debiasing via substitute confounder.

    Args:
        n_factors: Latent-factor dimension for the rating model.
        n_z_factors: Dimension of the substitute confounder Z.
        random_seed: Seed for all PyMC sampling.
    """

    def __init__(
        self,
        n_factors: int = 10,
        n_z_factors: int = 5,
        random_seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.n_z_factors = n_z_factors
        self.random_seed = random_seed

        self._Z: Optional[np.ndarray] = None          # (n_users, n_z_factors)
        self.outcome_model: Optional[pm.Model] = None
        self.outcome_trace: Optional[az.InferenceData] = None
        self._n_users: Optional[int] = None
        self._n_items: Optional[int] = None

    # ------------------------------------------------------------------
    # Stage 1 — Substitute confounder via exposure factor model
    # ------------------------------------------------------------------

    def fit_factor_model(self, observation_mask: np.ndarray) -> np.ndarray:
        """Stage 1: Fit a Bayesian factor model on the binary exposure matrix.

        Uses a Gaussian PMF on the 0/1 mask as a tractable approximation to
        the Poisson factorisation in the original paper.  The posterior mean
        user factors U serve as the substitute confounder Z.

        Args:
            observation_mask: Boolean (n_users, n_items); True = observed.

        Returns:
            Z array of shape (n_users, n_z_factors), the substitute confounders.
        """
        from scripts.models.bayesian_pmf import BayesianPMF

        n_users, n_items = observation_mask.shape
        self._n_users = n_users
        self._n_items = n_items

        # Treat binary mask as a 0/1 rating matrix
        mask_ratings = observation_mask.astype(np.float32)

        # All entries are "observed" for the factor model
        all_mask = np.ones((n_users, n_items), dtype=bool)

        factor_model = BayesianPMF(n_factors=self.n_z_factors, random_seed=self.random_seed)
        factor_model.build_model(mask_ratings, all_mask)

        print(f"[Deconfounder Stage 1] Fitting exposure factor model "
              f"({n_users} users × {n_items} items, K={self.n_z_factors}) …")
        factor_trace = factor_model.fit(draws=300, tune=300, chains=2, target_accept=0.9)

        # Z = posterior mean user factors from the exposure model
        Z = factor_trace.posterior["U"].values.mean(axis=(0, 1))  # (n_users, K)
        self._Z = Z

        print(f"[Deconfounder Stage 1] Z extracted — shape: {Z.shape}")
        return Z

    # ------------------------------------------------------------------
    # Stage 2 — Outcome model conditioned on Z
    # ------------------------------------------------------------------

    def fit_outcome_model(
        self,
        ratings: np.ndarray,
        mask: np.ndarray,
        Z: np.ndarray,
    ) -> az.InferenceData:
        """Stage 2: Fit rating model conditioned on the substitute confounder Z.

        The model includes Z[u] as a fixed user-side feature alongside the
        learnable latent factor U[u], implementing approximate backdoor
        adjustment.

        Model:
            sigma_u, sigma_v, tau ~ HalfNormal(1)
            U_offset[u] ~ Normal(0, 1),  U = U_offset * sigma_u
            V_offset[i] ~ Normal(0, 1),  V = V_offset * sigma_v
            gamma ~ Normal(0, 1)  shape: (n_z_factors,)   confounder coefficient
            r_hat[u,i] = U[u]·V[i] + Z[u]·gamma           backdoor adjustment
            R[u,i] ~ Normal(r_hat[u,i], 1/tau)

        Args:
            ratings: Dense (n_users, n_items) matrix; 0 where unobserved.
            mask: Boolean (n_users, n_items); True where observed.
            Z: (n_users, n_z_factors) substitute confounders from Stage 1.

        Returns:
            ArviZ InferenceData stored in self.outcome_trace.
        """
        n_users, n_items = ratings.shape
        n_z = Z.shape[1]
        user_idx, item_idx = np.where(mask)
        ratings_obs = ratings[user_idx, item_idx].astype(float)

        # Z values for observed entries
        Z_obs = Z[user_idx]  # (n_obs, n_z_factors)

        with pm.Model() as model:
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
            tau = pm.HalfNormal("tau", sigma=1.0)

            # Preference factors (non-centred)
            U_offset = pm.Normal(
                "U_offset", mu=0.0, sigma=1.0, shape=(n_users, self.n_factors)
            )
            V_offset = pm.Normal(
                "V_offset", mu=0.0, sigma=1.0, shape=(n_items, self.n_factors)
            )
            U = pm.Deterministic("U", U_offset * sigma_u)
            V = pm.Deterministic("V", V_offset * sigma_v)

            # Confounder coefficient — one scalar per Z dimension
            gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, shape=n_z)

            # Predicted ratings: preference term + confounder adjustment
            pref_term = pt.sum(U[user_idx] * V[item_idx], axis=1)
            conf_term = pt.dot(pm.Data("Z_obs", Z_obs), gamma)
            r_hat = pref_term + conf_term

            ratings_data = pm.Data("ratings_obs", ratings_obs)
            pm.Normal("obs", mu=r_hat, sigma=1.0 / (tau + 1e-6), observed=ratings_data)

        self.outcome_model = model

        print(f"[Deconfounder Stage 2] Fitting outcome model conditioned on Z …")
        with model:
            self.outcome_trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                target_accept=0.9,
                random_seed=self.random_seed,
                progressbar=True,
            )

        # Convergence check
        summary = az.summary(
            self.outcome_trace,
            var_names=["sigma_u", "sigma_v", "tau", "gamma"],
        )
        for param in summary.index:
            if summary.loc[param, "r_hat"] >= 1.05:
                print(
                    f"⚠️  Convergence warning: R-hat ≥ 1.05 for {param}."
                )

        return self.outcome_trace

    # ------------------------------------------------------------------
    # Counterfactual relevance — interventional predictions
    # ------------------------------------------------------------------

    def counterfactual_relevance(
        self,
        user_idx: int,
        item_indices: list[int],
    ) -> np.ndarray:
        """Posterior predictive relevance under do(A_{ui} = 1) for each item.

        After conditioning on Z (Stage 2), predictions are approximately
        interventional: we have blocked the backdoor path from the confounder
        to ratings via the substitute confounder Z[u].

        E[R_{ui} | do(A_{ui} = 1)] ≈ posterior_mean(U[u]·V[i] + Z[u]·gamma)

        Args:
            user_idx: Zero-based user index.
            item_indices: List of item indices to evaluate.

        Returns:
            Array of shape (n_items, 2) where column 0 = posterior mean,
            column 1 = posterior std.  These are the interventional
            relevance estimates with uncertainty.
        """
        if self.outcome_trace is None or self._Z is None:
            raise RuntimeError("Call fit_factor_model() and fit_outcome_model() first.")

        n_items_req = len(item_indices)
        item_indices = np.asarray(item_indices)

        # Posterior samples: (chains, draws, ...)
        U_s = self.outcome_trace.posterior["U"].values[:, :, user_idx, :]  # (C, D, K)
        V_s = self.outcome_trace.posterior["V"].values[:, :, item_indices, :]  # (C, D, n, K)
        gamma_s = self.outcome_trace.posterior["gamma"].values  # (C, D, n_z)

        # Flatten chain/draw dimensions
        n_s = U_s.shape[0] * U_s.shape[1]
        U_flat = U_s.reshape(n_s, -1)                   # (S, K)
        V_flat = V_s.reshape(n_s, n_items_req, -1)       # (S, n, K)
        gamma_flat = gamma_s.reshape(n_s, -1)            # (S, n_z)

        # Preference term: U[u] · V[i] for each sample and item
        pref = np.einsum("sk,snk->sn", U_flat, V_flat)  # (S, n)

        # Confounder term: Z[u] · gamma (same for all items)
        Z_u = self._Z[user_idx]                          # (n_z,)
        conf = np.einsum("sz,z->s", gamma_flat, Z_u)    # (S,)

        r_samples = pref + conf[:, None]                 # (S, n)

        means = r_samples.mean(axis=0)
        stds = r_samples.std(axis=0)

        return np.stack([means, stds], axis=1)  # (n_items, 2)

    def _score_matrix(self) -> np.ndarray:
        """Return posterior-mean (n_users × n_items) score matrix (deconfounded)."""
        if self.outcome_trace is None or self._Z is None:
            raise RuntimeError("Model not fitted.")
        U_mean = self.outcome_trace.posterior["U"].values.mean(axis=(0, 1))   # (n_u, K)
        V_mean = self.outcome_trace.posterior["V"].values.mean(axis=(0, 1))   # (n_i, K)
        gamma_mean = self.outcome_trace.posterior["gamma"].values.mean(axis=(0, 1))  # (n_z,)

        pref_scores = U_mean @ V_mean.T            # (n_u, n_i)
        conf_adj = self._Z @ gamma_mean            # (n_u,)
        return pref_scores + conf_adj[:, None]

    def recommend_topk(self, user_idx: int, k: int = 10) -> list[int]:
        """Return top-k items by deconfounded posterior mean score.

        Args:
            user_idx: Zero-based user index.
            k: Number of items.

        Returns:
            List of item indices sorted descending by score.
        """
        scores = self._score_matrix()[user_idx]
        return list(np.argsort(scores)[::-1][:k])
