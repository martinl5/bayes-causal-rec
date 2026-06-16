"""Tests for the synthetic MNAR generator and its ground-truth propensities."""

from __future__ import annotations

import numpy as np

from bcr.data.preprocess import make_synthetic_mnar


def test_shapes_and_keys():
    d = make_synthetic_mnar(n_users=40, n_items=30, n_factors=5, random_seed=0)
    for key in (
        "train_ratings",
        "train_mask",
        "test_ratings",
        "test_mask",
        "true_ratings",
        "true_propensities",
    ):
        assert key in d
        assert d[key].shape == (40, 30)


def test_ratings_in_scale():
    d = make_synthetic_mnar(n_users=50, n_items=40, random_seed=1)
    true = d["true_ratings"]
    assert true.min() >= 1.0 - 1e-6
    assert true.max() <= 5.0 + 1e-6


def test_propensities_are_probabilities():
    d = make_synthetic_mnar(n_users=50, n_items=40, random_seed=2)
    p = d["true_propensities"]
    assert p.min() >= 0.0
    assert p.max() <= 1.0


def test_train_and_test_masks_disjoint():
    # Test entries are held out before MNAR exposure is sampled, so train and
    # test masks must never overlap (no leakage).
    d = make_synthetic_mnar(n_users=60, n_items=50, random_seed=3)
    overlap = d["train_mask"] & d["test_mask"]
    assert not overlap.any()


def test_test_set_is_mcar_unbiased():
    # The test set is a uniform (MCAR) sample over ALL (user, item) pairs, so
    # the mean true rating over test cells should match the global mean.  The
    # MNAR training set, by contrast, is skewed toward higher ratings because
    # well-liked items are more likely to be observed.
    d = make_synthetic_mnar(
        n_users=200, n_items=150, alpha_relevance=1.0, random_seed=11
    )
    true = d["true_ratings"]
    global_mean = float(true.mean())
    test_mean = float(true[d["test_mask"]].mean())
    train_mean = float(true[d["train_mask"]].mean())
    # MCAR test set: unbiased w.r.t. the full population.
    assert abs(test_mean - global_mean) < 0.05
    # MNAR train set: selection bias inflates observed ratings.
    assert train_mean > test_mean


def test_observed_train_entries_are_nonzero():
    d = make_synthetic_mnar(n_users=40, n_items=30, random_seed=4)
    assert np.all(d["train_ratings"][d["train_mask"]] > 0)
    # Unobserved entries are encoded as 0.
    assert np.all(d["train_ratings"][~d["train_mask"]] == 0)


def test_reproducible_with_seed():
    a = make_synthetic_mnar(n_users=30, n_items=20, random_seed=7)
    b = make_synthetic_mnar(n_users=30, n_items=20, random_seed=7)
    assert np.array_equal(a["true_ratings"], b["true_ratings"])
    assert np.array_equal(a["train_mask"], b["train_mask"])
