"""Bayesian propensity model for P(O_{ui} = 1 | user u, item i).

Models the exposure mechanism as hierarchical logistic regression on the
observation (binary) mask.  Propensity scores are required by IPSBayesianPMF
and DoublyRobust evaluation.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm
import scipy.special


class BayesianPropensityModel:
    """Bayesian hierarchical logistic-regression propensity model.

    Models whether a (user, item) pair is observed in the training data.

    Model:
        mu_alpha, mu_beta ~ Normal(0, 1)       # global intercept means
        sigma_alpha, sigma_beta ~ HalfNormal(1) # hierarchical scales
        alpha_raw[u] ~ Normal(0, 1)            # user offsets (non-centred)
        beta_raw[i]  ~ Normal(0, 1)            # item offsets (non-centred)
        alpha[u] = mu_alpha + alpha_raw[u] * sigma_alpha
        beta[i]  = mu_beta  + beta_raw[i]  * sigma_beta
        logit(p_{ui}) = alpha[u] + beta[i]
        O_{ui} ~ Bernoulli(p_{ui})

    Args:
        random_seed: Seed for PyMC sampling.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.random_seed = random_seed
        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None
        self._n_users: int | None = None
        self._n_items: int | None = None

    def build_model(self, train_mask: np.ndarray) -> pm.Model:
        """Build the propensity model on the binary observation mask.

        Args:
            train_mask: Boolean (n_users, n_items); True where rating is observed.

        Returns:
            Compiled pm.Model stored in self.model.
        """
        n_users, n_items = train_mask.shape
        self._n_users = n_users
        self._n_items = n_items

        obs_flat = train_mask.astype(np.int8).flatten()

        with pm.Model() as model:
            # Global intercept means
            mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0)
            mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0)

            # Hierarchical scales
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)

            # Non-centred user/item offsets
            alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, shape=n_users)
            beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=1.0, shape=n_items)

            alpha = pm.Deterministic("alpha", mu_alpha + alpha_raw * sigma_alpha)
            beta = pm.Deterministic("beta", mu_beta + beta_raw * sigma_beta)

            # Logit probabilities for all (u, i) pairs, flattened
            logit_p = (alpha[:, None] + beta[None, :]).flatten()

            mask_data = pm.Data("mask_obs", obs_flat)
            pm.Bernoulli("obs", logit_p=logit_p, observed=mask_data)

        self.model = model
        return model

    def fit(
        self,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        target_accept: float = 0.9,
    ) -> az.InferenceData:
        """Run NUTS sampling.  Checks R-hat on global parameters.

        Args:
            draws: Posterior draws per chain.
            tune: Warm-up steps per chain.
            chains: Number of independent chains.
            target_accept: NUTS acceptance target.

        Returns:
            ArviZ InferenceData stored in self.trace.
        """
        if self.model is None:
            raise RuntimeError("Call build_model() first.")

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=self.random_seed,
                progressbar=True,
            )

        summary = az.summary(
            self.trace,
            var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"],
        )
        for param in summary.index:
            r_hat_val = summary.loc[param, "r_hat"]
            if r_hat_val >= 1.05:
                print(
                    f"⚠️  Convergence warning: R-hat ≥ 1.05 for {param}. "
                    "Consider increasing tune/draws."
                )

        return self.trace

    def propensity_scores(self) -> np.ndarray:
        """Return (n_users, n_items) matrix of posterior-mean P(observed).

        Computed from posterior means of alpha and beta (memory-efficient:
        avoids storing the full n_users×n_items p matrix in the trace).

        Returns:
            Float32 array of shape (n_users, n_items), values in (0, 1).
        """
        if self.trace is None:
            raise RuntimeError("Model not yet fitted.")
        alpha_mean = self.trace.posterior["alpha"].values.mean(axis=(0, 1))  # (n_users,)
        beta_mean = self.trace.posterior["beta"].values.mean(axis=(0, 1))  # (n_items,)
        logit_p = alpha_mean[:, None] + beta_mean[None, :]
        return scipy.special.expit(logit_p).astype(np.float32)
