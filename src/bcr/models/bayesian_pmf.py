"""Bayesian Probabilistic Matrix Factorization.

Phase 1 model: full-posterior NUTS via PyMC.
Phase 3 extension: NumPyroPMF with SVI (added in Phase 3).
"""

from __future__ import annotations

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pymc as pm
import pytensor.tensor as pt
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal


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
        self.model: pm.Model | None = None
        self.trace: az.InferenceData | None = None
        self._n_users: int | None = None
        self._n_items: int | None = None
        self._user_idx_obs: np.ndarray | None = None
        self._item_idx_obs: np.ndarray | None = None

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
        global_mean = float(ratings_obs.mean()) if ratings_obs.size else 0.0

        self._n_users = n_users
        self._n_items = n_items
        self._user_idx_obs = user_idx
        self._item_idx_obs = item_idx

        with pm.Model() as model:
            # Hyperpriors on factor scales
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
            tau = pm.HalfNormal("tau", sigma=1.0)

            # Global mean + user/item bias intercepts.  Without these, the
            # zero-mean factor priors force predictions toward 0 while ratings
            # live in [1, 5], producing RMSE ~= the mean rating.  The biases
            # anchor predictions to the rating scale (standard PMF practice).
            mu_global = pm.Normal("mu_global", mu=global_mean, sigma=1.0)
            sigma_bu = pm.HalfNormal("sigma_bu", sigma=1.0)
            sigma_bi = pm.HalfNormal("sigma_bi", sigma=1.0)
            bias_u_offset = pm.Normal("bias_u_offset", mu=0.0, sigma=1.0, shape=n_users)
            bias_i_offset = pm.Normal("bias_i_offset", mu=0.0, sigma=1.0, shape=n_items)
            bias_u = pm.Deterministic("bias_u", bias_u_offset * sigma_bu)
            bias_i = pm.Deterministic("bias_i", bias_i_offset * sigma_bi)

            # Non-centred latent factors
            U_offset = pm.Normal("U_offset", mu=0.0, sigma=1.0, shape=(n_users, self.n_factors))
            V_offset = pm.Normal("V_offset", mu=0.0, sigma=1.0, shape=(n_items, self.n_factors))

            U = pm.Deterministic("U", U_offset * sigma_u)
            V = pm.Deterministic("V", V_offset * sigma_v)

            # Predicted ratings for observed entries
            r_hat = (
                mu_global
                + bias_u[user_idx]
                + bias_i[item_idx]
                + pt.sum(U[user_idx] * V[item_idx], axis=1)
            )

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
        """Return (n_users × n_items) posterior-mean score matrix.

        Includes the global mean and user/item bias intercepts so that scores
        are on the rating scale, not just the zero-centred factor interaction.
        """
        if self.trace is None:
            raise RuntimeError("Model not yet fitted.")
        post = self.trace.posterior
        U_mean = post["U"].values.mean(axis=(0, 1))  # (n_users, K)
        V_mean = post["V"].values.mean(axis=(0, 1))  # (n_items, K)
        scores = U_mean @ V_mean.T  # (n_users, n_items)

        # Add intercepts if present (models trained before bias terms omit them)
        if "mu_global" in post:
            mu_global = float(post["mu_global"].values.mean())
            bias_u = post["bias_u"].values.mean(axis=(0, 1))  # (n_users,)
            bias_i = post["bias_i"].values.mean(axis=(0, 1))  # (n_items,)
            scores = scores + mu_global + bias_u[:, None] + bias_i[None, :]
        return scores

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

        post = self.trace.posterior
        # Gather posterior samples: shape (chains, draws, n_factors)
        U_s = post["U"].values[:, :, user_idx, :]
        V_s = post["V"].values[:, :, item_idx, :]

        # Flatten to (n_samples, n_factors) and compute dot-product
        U_flat = U_s.reshape(-1, self.n_factors)
        V_flat = V_s.reshape(-1, self.n_factors)
        r_samples = (U_flat * V_flat).sum(axis=1)

        # Add intercepts per posterior sample (preserves uncertainty)
        if "mu_global" in post:
            r_samples = (
                r_samples
                + post["mu_global"].values.reshape(-1)
                + post["bias_u"].values[:, :, user_idx].reshape(-1)
                + post["bias_i"].values[:, :, item_idx].reshape(-1)
            )

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
        post = self.trace.posterior
        # U posterior samples: (chains*draws, K)
        U_s = post["U"].values[:, :, user_idx, :].reshape(-1, self.n_factors)
        # V posterior samples: (chains*draws, n_items, K)
        V_s = post["V"].values.reshape(-1, self._n_items, self.n_factors)

        # r_samples: (n_samples, n_items)
        r_samples = np.einsum("sk,sik->si", U_s, V_s)

        # Add intercepts per posterior sample (preserves uncertainty)
        if "mu_global" in post:
            mu_s = post["mu_global"].values.reshape(-1, 1)  # (S, 1)
            bu_s = post["bias_u"].values[:, :, user_idx].reshape(-1, 1)  # (S, 1)
            bi_s = post["bias_i"].values.reshape(-1, self._n_items)  # (S, n_items)
            r_samples = r_samples + mu_s + bu_s + bi_s

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
        global_mean = float(ratings_obs.mean()) if ratings_obs.size else 0.0

        # IPS weights for observed entries, clipped to control variance, then
        # self-normalised to mean 1 (SNIPS-style).  Normalisation keeps the
        # effective observation precision on the same scale as the naive model,
        # so IPS reweights *relative* importance without inflating overall noise.
        prop_obs = np.clip(propensities[user_idx, item_idx], 1e-6, 1.0)
        ips_weights = np.clip(1.0 / prop_obs, 1.0, clip_max)
        ips_weights = (ips_weights / ips_weights.mean()).astype(float)

        self._n_users = n_users
        self._n_items = n_items
        self._user_idx_obs = user_idx
        self._item_idx_obs = item_idx

        with pm.Model() as model:
            sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
            tau = pm.HalfNormal("tau", sigma=1.0)

            # Global mean + user/item bias intercepts (see BayesianPMF)
            mu_global = pm.Normal("mu_global", mu=global_mean, sigma=1.0)
            sigma_bu = pm.HalfNormal("sigma_bu", sigma=1.0)
            sigma_bi = pm.HalfNormal("sigma_bi", sigma=1.0)
            bias_u_offset = pm.Normal("bias_u_offset", mu=0.0, sigma=1.0, shape=n_users)
            bias_i_offset = pm.Normal("bias_i_offset", mu=0.0, sigma=1.0, shape=n_items)
            bias_u = pm.Deterministic("bias_u", bias_u_offset * sigma_bu)
            bias_i = pm.Deterministic("bias_i", bias_i_offset * sigma_bi)

            U_offset = pm.Normal("U_offset", mu=0.0, sigma=1.0, shape=(n_users, self.n_factors))
            V_offset = pm.Normal("V_offset", mu=0.0, sigma=1.0, shape=(n_items, self.n_factors))

            U = pm.Deterministic("U", U_offset * sigma_u)
            V = pm.Deterministic("V", V_offset * sigma_v)

            r_hat = (
                mu_global
                + bias_u[user_idx]
                + bias_i[item_idx]
                + pt.sum(U[user_idx] * V[item_idx], axis=1)
            )

            # IPS-scaled sigma: lower propensity → higher weight → tighter noise
            ips_w = pm.Data("ips_weights", ips_weights)
            effective_sigma = 1.0 / (pt.sqrt(ips_w * tau) + 1e-6)

            ratings_data = pm.Data("ratings_obs", ratings_obs)
            pm.Normal("obs", mu=r_hat, sigma=effective_sigma, observed=ratings_data)

        self.model = model
        return model


class NumPyroPMF:
    """Bayesian PMF via NumPyro SVI for fast approximate inference.

    Used inside the feedback-loop simulation where running full NUTS per round
    would take hours.  A mean-field AutoNormal guide (diagonal-covariance
    normal over U and V) is trained with Adam + ELBO for n_steps iterations.

    Trade-off vs PyMC NUTS:
        PyMC NUTS: asymptotically exact posterior, calibrated uncertainty, slow.
        NumPyro SVI: mean-field approximation (underestimates posterior variance),
                     fast (~30s on CPU), scales via subsampling.

    This underestimation means Thompson Sampling via SVI explores less
    aggressively than exact NUTS would, but it is the practical choice for
    interactive simulation (10 rounds × 3 strategies).

    Args:
        n_factors: Latent factor dimension.
        random_seed: Seed for JAX PRNG.
        n_steps: SVI training iterations.
        learning_rate: Adam learning rate.
    """

    def __init__(
        self,
        n_factors: int = 10,
        random_seed: int = 42,
        n_steps: int = 1000,
        learning_rate: float = 1e-2,
    ) -> None:
        self.n_factors = n_factors
        self.random_seed = random_seed
        self.n_steps = n_steps
        self.learning_rate = learning_rate

        self._n_users: int | None = None
        self._n_items: int | None = None
        self._svi_result = None
        self._guide = None
        self._model_fn = None

    # ------------------------------------------------------------------
    # NumPyro model
    # ------------------------------------------------------------------

    @staticmethod
    def _pmf_model(
        user_idx: np.ndarray,
        item_idx: np.ndarray,
        ratings_obs: np.ndarray,
        n_users: int,
        n_items: int,
        n_factors: int,
        global_mean: float = 0.0,
    ) -> None:
        """NumPyro generative model for Bayesian PMF with bias intercepts."""
        sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(1.0))
        sigma_v = numpyro.sample("sigma_v", dist.HalfNormal(1.0))
        tau = numpyro.sample("tau", dist.HalfNormal(1.0))

        # Global mean + user/item bias intercepts (anchor to rating scale)
        mu_global = numpyro.sample("mu_global", dist.Normal(global_mean, 1.0))
        sigma_bu = numpyro.sample("sigma_bu", dist.HalfNormal(1.0))
        sigma_bi = numpyro.sample("sigma_bi", dist.HalfNormal(1.0))
        bias_u = numpyro.sample("bias_u", dist.Normal(jnp.zeros(n_users), sigma_bu))
        bias_i = numpyro.sample("bias_i", dist.Normal(jnp.zeros(n_items), sigma_bi))

        U = numpyro.sample("U", dist.Normal(jnp.zeros((n_users, n_factors)), sigma_u))
        V = numpyro.sample("V", dist.Normal(jnp.zeros((n_items, n_factors)), sigma_v))

        r_hat = (
            mu_global
            + bias_u[user_idx]
            + bias_i[item_idx]
            + jnp.sum(U[user_idx] * V[item_idx], axis=-1)
        )
        numpyro.sample(
            "obs",
            dist.Normal(r_hat, 1.0 / (tau + 1e-6)),
            obs=ratings_obs,
        )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, ratings: np.ndarray, mask: np.ndarray) -> None:
        """Fit via SVI (Adam + ELBO) on the observed (user, item) ratings.

        Args:
            ratings: Dense (n_users, n_items) matrix; 0 where unobserved.
            mask: Boolean (n_users, n_items); True where observed.
        """
        n_users, n_items = ratings.shape
        self._n_users = n_users
        self._n_items = n_items

        user_idx, item_idx = np.where(mask)
        ratings_obs = ratings[user_idx, item_idx].astype(np.float32)
        global_mean = float(ratings_obs.mean()) if ratings_obs.size else 0.0

        user_idx_jnp = jnp.array(user_idx)
        item_idx_jnp = jnp.array(item_idx)
        ratings_jnp = jnp.array(ratings_obs)

        def model():
            return NumPyroPMF._pmf_model(
                user_idx_jnp,
                item_idx_jnp,
                ratings_jnp,
                n_users,
                n_items,
                self.n_factors,
                global_mean,
            )

        guide = AutoNormal(model)
        optimizer = numpyro.optim.Adam(self.learning_rate)
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        rng_key = jax.random.PRNGKey(self.random_seed)
        # svi.run() uses lax.scan internally — JIT-compiled, much faster than
        # a Python for-loop over svi.update() on first call and all subsequent.
        svi_result = svi.run(rng_key, self.n_steps, progress_bar=False)

        self._svi_result = svi_result.params
        self._guide = guide
        self._model_fn = model

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _posterior_samples(self, n_samples: int = 50, rng_key: jax.Array | None = None) -> dict:
        """Draw samples from the variational posterior.

        Args:
            n_samples: Number of samples to draw.
            rng_key: JAX PRNGKey; defaults to a key derived from random_seed.

        Returns:
            Dict with keys 'U' (n_samples, n_users, K) and 'V' (n_samples, n_items, K).
        """
        if self._guide is None:
            raise RuntimeError("Call fit() before sampling.")
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.random_seed + 1)

        predictive = numpyro.infer.Predictive(
            self._guide, params=self._svi_result, num_samples=n_samples
        )
        samples = predictive(rng_key)
        return {
            "U": np.array(samples["U"]),  # (n_samples, n_users, K)
            "V": np.array(samples["V"]),  # (n_samples, n_items, K)
            "mu_global": np.array(samples["mu_global"]),  # (n_samples,)
            "bias_u": np.array(samples["bias_u"]),  # (n_samples, n_users)
            "bias_i": np.array(samples["bias_i"]),  # (n_samples, n_items)
        }

    def _score_matrix(self, n_samples: int = 50) -> np.ndarray:
        """(n_users × n_items) score matrix averaged over posterior samples.

        Includes the global mean and bias intercepts so scores are on the
        rating scale.

        Args:
            n_samples: Number of variational posterior samples to average.

        Returns:
            Float array of shape (n_users, n_items).
        """
        samples = self._posterior_samples(n_samples)
        U_mean = samples["U"].mean(axis=0)  # (n_users, K)
        V_mean = samples["V"].mean(axis=0)  # (n_items, K)
        mu = float(samples["mu_global"].mean())
        bu = samples["bias_u"].mean(axis=0)  # (n_users,)
        bi = samples["bias_i"].mean(axis=0)  # (n_items,)
        return U_mean @ V_mean.T + mu + bu[:, None] + bi[None, :]

    def recommend_greedy(self, user_idx: int, k: int = 10) -> list[int]:
        """Top-k items by posterior-mean score (exploitation baseline).

        Args:
            user_idx: Zero-based user index.
            k: Number of items to recommend.

        Returns:
            Item indices sorted descending by mean score.
        """
        scores = self._score_matrix()[user_idx]
        return list(np.argsort(scores)[::-1][:k])

    def recommend_thompson(
        self,
        user_idx: int,
        k: int = 10,
        rng_key: jax.Array | None = None,
    ) -> list[int]:
        """Thompson Sampling: draw one posterior sample, score all items.

        Args:
            user_idx: Zero-based user index.
            k: Number of items to recommend.
            rng_key: JAX PRNGKey for this recommendation step.

        Returns:
            Item indices sorted descending by sampled score.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.random_seed)

        samples = self._posterior_samples(n_samples=1, rng_key=rng_key)
        U_sample = samples["U"][0, user_idx, :]  # (K,)
        V_sample = samples["V"][0]  # (n_items, K)
        # bias_i affects within-user ranking; mu_global and bias_u are constant
        # shifts for a fixed user and do not change the argsort.
        bias_i = samples["bias_i"][0]  # (n_items,)
        scores = V_sample @ U_sample + bias_i
        return list(np.argsort(scores)[::-1][:k])
