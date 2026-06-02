"""Fast, sampling-free tests of model construction.

Building a PyMC model compiles the graph but does not run NUTS, so these
stay quick enough for CI.
"""

from __future__ import annotations

import numpy as np

from bcr.data.preprocess import make_synthetic_mnar
from bcr.models.bayesian_pmf import BayesianPMF, IPSBayesianPMF


def test_pmf_model_has_bias_terms():
    d = make_synthetic_mnar(n_users=20, n_items=15, n_factors=3, random_seed=0)
    m = BayesianPMF(n_factors=3, random_seed=0)
    model = m.build_model(d["train_ratings"], d["train_mask"])
    names = set(model.named_vars)
    # The fix for the rating-scale bug adds these intercept variables.
    for v in ("mu_global", "bias_u", "bias_i"):
        assert v in names


def test_ips_weights_are_self_normalised_to_mean_one():
    d = make_synthetic_mnar(n_users=30, n_items=25, n_factors=3, random_seed=1)
    propensities = d["true_propensities"]
    m = IPSBayesianPMF(n_factors=3, random_seed=1)
    model = m.build_model(d["train_ratings"], d["train_mask"], propensities, clip_max=5.0)
    weights = model.named_vars["ips_weights"].get_value()
    assert abs(float(weights.mean()) - 1.0) < 1e-6
    assert np.all(weights > 0)
