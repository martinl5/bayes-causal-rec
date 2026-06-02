"""Bayesian Probabilistic Matrix Factorization.

Phase 1 model: full-posterior NUTS via PyMC.
Phase 3 extension: NumPyroPMF with SVI (added in Phase 3).
"""

from __future__ import annotations

import warnings
from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


class BayesianPMF:
    """Bayesian Probabilistic Matrix Factorization with MAP/NUTS inference.

    Model specification:
        sigma_u ~ HalfNormal(1)
        sigma_v ~ HalfNormal(1)
        tau     ~ HalfNormal(1)            # observation noise precision
        U_offset[u] ~ Normal(0, 1)         shape: (n_users, n_factors)
        V_offset[i] ~ Normal(0, 1)         shape: (n_items, n_factors)
        U = U_offset * sigma_u             (non-centered reparameterisation)
        V = V_offset * sigma_v
        R[u,i] ~ Normal(U[u] · V[i], 1/tau)  for observed (u,i)

    Non-centred parameterisation improves NUTS mixing when sigma_u, sigma_v
    are small relative to the data scale.

    Args:
        n_factors: Latent factor dimension.
        random_seed: Seed for PyMC sampling for reproducibility.
    """

    def __init__(self, n_factors: int = 10, random_seed: int = 42) -> None:
        self.n_factors = n_factors
        self.random_seed = random_seed
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None
        self._n_users: Optional[int] = None
        self._n_items: Optional[int] = None
        self._user_idx_obs: Optional[np.ndarray] = None
        self._item_idx_obs: Optional[np.ndarray] = None

    def build_model(
        self,
        ratings: np.ndarray,
        mask: np.ndarray,
    ) -> pm.Model:
        """Compile a PyMC model for observed (user, item) rating pairs.

        Args:
            ratings: Dense (n_users, n_items) matrix; 0 where unobserved.
            mask: Boolean (n_users, n_items) array; True where observed.

        Returns:
            Compiled pm.Model stored in self.model.
        """
        n_users, n_items = ratings.shape
        user_idx, item_idx = np.where(mask)
        ratings_obs = ratings[user_idx, item_idx].astype(float)

        self._n_users = n_users
        self._n_items = n_items
        self._user_idx_obs = user_idx
        self._item_idx_obs = item_idx

        with pm.Model() as model:
            # Hyperpriors on factor scales
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
            tau = pm.HalfNormal("tau", sigma=1.0)

            # Non-centred latent factors
            U_offset = pm.Normal(
                "U_offset", mu=0.0, sigma=1.0, shape=(n_users, self.n_factors)
            )
            V_offset = pm.Normal(
                "V_offset", mu=0.0, sigma=1.0, shape=(n_items, self.n_factors)
            )

            U = pm.Deterministic("U", U_offset * sigma_u)
            V = pm.Deterministic("V", V_offset * sigma_v)

            # Predicted ratings for observed entries
            r_hat = pt.sum(U[user_idx] * V[item_idx], axis=1)

            # Likelihood via pm.Data so the array is mutable post-build
            ratings_data = pm.Data("ratings_obs", ratings_obs)
            pm.Normal("obs", mu=r_hat, sigma=1.0 / (tau + 1e-6), observed=ratings_data)

        self.model = model
        return model

    def fit(
        self,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.9,
    ) -> az.InferenceData:
        """Run NUTS sampling.  Check R-hat and warn on convergence issues.

        Args:
            draws: Posterior draws per chain.
            tune: Tuning (warm-up) steps per chain.
            chains: Number of independent chains.
            target_accept: NUTS acceptance probability target.

        Returns:
            ArviZ InferenceData stored in self.trace.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() before fit().")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=self.random_seed,
                progressbar=True,
            )

        # Convergence diagnostics on global hyperparameters
        summary = az.summary(self.trace, var_names=["sigma_u", "sigma_v", "tau"])
        for param in summary.index:
            r_hat_val = summary.loc[param, "r_hat"]
            if r_hat_val >= 1.05:
                print(
                    f"⚠️  Convergence warning: R-hat ≥ 1.05 for {param}. "
                    "Consider increasing tune/draws."
                )

        return self.trace

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _score_matrix(self) -> np.ndarray:
        """Return (n_users × n_items) posterior-mean score matrix."""
        if self.trace is None:
            raise RuntimeError("Model not yet fitted.")
        U_mean = self.trace.posterior["U"].values.mean(axis=(0, 1))  # (n_users, K)
        V_mean = self.trace.posterior["V"].values.mean(axis=(0, 1))  # (n_items, K)
        return U_mean @ V_mean.T  # (n_users, n_items)

    def predict(self, user_idx: int, item_idx: int) -> dict:
        """Posterior predictive statistics for a single (user, item) pair.

        Args:
            user_idx: Zero-based user index.
            item_idx: Zero-based item index.

        Returns:
            dict with keys 'mean' (float), 'std' (float), 'hdi_94' (tuple).
        """
        if self.trace is None:
            raise RuntimeError("Model not yet fitted.")

        # Gather posterior samples: shape (chains, draws, n_factors)
        U_s = self.trace.posterior["U"].values[:, :, user_idx, :]
        V_s = self.trace.posterior["V"].values[:, :, item_idx, :]

        # Flatten to (n_samples, n_factors) and compute dot-product
        U_flat = U_s.reshape(-1, self.n_factors)
        V_flat = V_s.reshape(-1, self.n_factors)
        r_samples = (U_flat * V_flat).sum(axis=1)

        hdi = az.hdi(r_samples, hdi_prob=0.94)
        return {
            "mean": float(r_samples.mean()),
            "std": float(r_samples.std()),
            "hdi_94": (float(hdi[0]), float(hdi[1])),
        }

    def predict_all_items(self, user_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (means, stds) over all items for one user.

        Args:
            user_idx: Zero-based user index.

        Returns:
            Tuple of arrays, each shape (n_items,).
        """
        if self.trace is None:
            raise RuntimeError("Model not yet fitted.")
        # U posterior samples: (chains*draws, K)
        U_s = self.trace.posterior["U"].values[:, :, user_idx, :].reshape(
            -1, self.n_factors
        )
        # V posterior samples: (chains*draws, n_items, K)
        V_s = self.trace.posterior["V"].values.reshape(-1, self._n_items, self.n_factors)

        # r_samples: (n_samples, n_items)
        r_samples = np.einsum("sk,sik->si", U_s, V_s)
        return r_samples.mean(axis=0), r_samples.std(axis=0)

    def recommend_topk(self, user_idx: int, k: int = 10) -> list[int]:
        """Top-k item indices by posterior mean score.

        Args:
            user_idx: Zero-based user index.
            k: Number of items to recommend.

        Returns:
            List of item indices sorted descending by score.
        """
        scores = self._score_matrix()[user_idx]
        return list(np.argsort(scores)[::-1][:k])


class IPSBayesianPMF(BayesianPMF):
    """BayesianPMF extended with Inverse Propensity Score (IPS) weighting.

    The IPS-corrected likelihood down-weights frequently-observed (high-propensity)
    items so the model learns preference estimates closer to the unbiased
    population.

    Modification: the observation noise precision for entry (u, i) is scaled
    by the IPS weight w_{ui} = clip(1 / P(O_{ui}=1), 1, clip_max).  Items
    exposed with low probability contribute higher effective precision —
    i.e., the model trusts those rare signals more, correcting for the
    selection mechanism.

    Formally:
        R_{ui} ~ Normal(U[u]·V[i], 1/sqrt(w_{ui} * tau))

    Args:
        n_factors: Latent factor dimension.
        random_seed: Seed for NUTS sampling.
    """

    def build_model(  # type: ignore[override]
        self,
        ratings: np.ndarray,
        mask: np.ndarray,
        propensities: np.ndarray,
        clip_max: float = 5.0,
    ) -> pm.Model:
        """Build IPS-weighted PyMC model.

        Args:
            ratings: Dense (n_users, n_items) matrix; 0 where unobserved.
            mask: Boolean (n_users, n_items); True where observed.
            propensities: (n_users, n_items) P(observed) from BayesianPropensityModel.
            clip_max: Maximum IPS weight (variance-reduction clipping).

        Returns:
            Compiled pm.Model stored in self.model.
        """
        n_users, n_items = ratings.shape
        user_idx, item_idx = np.where(mask)
        ratings_obs = ratings[user_idx, item_idx].astype(float)

        # IPS weights for observed entries
        prop_obs = np.clip(propensities[user_idx, item_idx], 1e-6, 1.0)
        ips_weights = np.clip(1.0 / prop_obs, 1.0, clip_max).astype(float)

        self._n_users = n_users
        self._n_items = n_items
        self._user_idx_obs = user_idx
        self._item_idx_obs = item_idx

        with pm.Model() as model:
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
            tau = pm.HalfNormal("tau", sigma=1.0)

            U_offset = pm.Normal(
                "U_offset", mu=0.0, sigma=1.0, shape=(n_users, self.n_factors)
            )
            V_offset = pm.Normal(
                "V_offset", mu=0.0, sigma=1.0, shape=(n_items, self.n_factors)
            )

            U = pm.Deterministic("U", U_offset * sigma_u)
            V = pm.Deterministic("V", V_offset * sigma_v)

            r_hat = pt.sum(U[user_idx] * V[item_idx], axis=1)

            # IPS-scaled sigma: lower propensity → higher weight → tighter noise
            ips_w = pm.Data("ips_weights", ips_weights)
            effective_sigma = 1.0 / (pt.sqrt(ips_w * tau) + 1e-6)

            ratings_data = pm.Data("ratings_obs", ratings_obs)
            pm.Normal("obs", mu=r_hat, sigma=effective_sigma, observed=ratings_data)

        self.model = model
        return model
