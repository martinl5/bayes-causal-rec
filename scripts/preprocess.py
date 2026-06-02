"""Data loading and synthetic data generation.

Usage:
    from scripts.preprocess import load_coat, make_synthetic_mnar
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_coat(data_dir: str = "data/raw") -> dict:
    """Load Coat train/test matrices from ASCII files.

    The Coat dataset has 290 users and 300 items.  Train ratings are
    biased (users chose what to rate); test ratings are from a uniform
    random exposure design and form the unbiased evaluation set.

    Args:
        data_dir: Directory containing data/raw/coat/train.ascii and test.ascii.

    Returns:
        dict with keys:
            'train'      : np.ndarray (290, 300), biased ratings; 0 = unobserved
            'test'       : np.ndarray (290, 300), unbiased ratings; 0 = unobserved
            'train_mask' : bool array (290, 300), True where rating observed
            'test_mask'  : bool array (290, 300), True where rating observed
    """
    coat_dir = Path(data_dir) / "coat"
    train_path = coat_dir / "train.ascii"
    test_path = coat_dir / "test.ascii"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Coat files not found in {coat_dir}. "
            "Run `python scripts/download_data.py` first."
        )

    train = np.loadtxt(train_path, dtype=np.float32)
    test = np.loadtxt(test_path, dtype=np.float32)

    return {
        "train": train,
        "test": test,
        "train_mask": train > 0,
        "test_mask": test > 0,
    }


def make_synthetic_mnar(
    n_users: int = 500,
    n_items: int = 200,
    n_factors: int = 10,
    alpha_popularity: float = 2.0,
    alpha_relevance: float = 1.0,
    min_obs_prob: float = 0.01,
    test_fraction: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Generate a synthetic MNAR dataset with known ground-truth propensities.

    Exposure model: logit P(observe|u,i) = alpha_popularity * log(pop_i)
                                          + alpha_relevance * true_rating_{u,i}
    This creates Missing-Not-At-Random data: popular, well-liked items are
    more likely to be observed in training, introducing selection bias.

    Args:
        n_users: Number of simulated users.
        n_items: Number of simulated items.
        n_factors: Dimension of latent factor space.
        alpha_popularity: Strength of popularity effect on exposure.
        alpha_relevance: Strength of relevance (MNAR) effect on exposure.
        min_obs_prob: Floor on observation probability to avoid zero weights.
        test_fraction: Fraction of users held out for unbiased test evaluation.
        random_seed: NumPy random seed.

    Returns:
        dict with keys:
            'train_ratings'     : np.ndarray (n_users, n_items), 0 = unobserved
            'train_mask'        : bool array (n_users, n_items)
            'test_ratings'      : np.ndarray (n_users, n_items), 0 = unobserved
            'test_mask'         : bool array (n_users, n_items)
            'true_ratings'      : np.ndarray (n_users, n_items), full ground truth
            'true_propensities' : np.ndarray (n_users, n_items), P(observe)
    """
    rng = np.random.default_rng(random_seed)

    # Ground-truth latent factors
    U_true = rng.standard_normal((n_users, n_factors)) / np.sqrt(n_factors)
    V_true = rng.standard_normal((n_items, n_factors)) / np.sqrt(n_factors)

    # True ratings in [1, 5]
    raw_ratings = U_true @ V_true.T  # (n_users, n_items)
    raw_ratings = (raw_ratings - raw_ratings.mean()) / (raw_ratings.std() + 1e-8)
    true_ratings = np.clip(3.0 + raw_ratings * 1.0, 1.0, 5.0)

    # Item popularity (Zipf-like)
    item_pop = rng.exponential(1.0, n_items)
    item_pop = item_pop / item_pop.max()

    # MNAR exposure logits
    log_odds = (
        alpha_popularity * np.log(item_pop + 1e-6)[np.newaxis, :]
        + alpha_relevance * (true_ratings - true_ratings.mean())
    )
    propensities = 1.0 / (1.0 + np.exp(-log_odds))
    propensities = np.clip(propensities, min_obs_prob, 1.0)

    # Sample biased training observations
    observed = rng.random((n_users, n_items)) < propensities
    train_ratings = np.where(observed, true_ratings, 0.0).astype(np.float32)

    # Unbiased test: uniform random sample over unobserved entries
    unobserved = ~observed
    n_unobs = unobserved.sum()
    n_test = int(n_unobs * test_fraction)
    unobs_idx = np.argwhere(unobserved)
    test_sel = rng.choice(len(unobs_idx), size=n_test, replace=False)
    test_mask = np.zeros((n_users, n_items), dtype=bool)
    test_mask[unobs_idx[test_sel, 0], unobs_idx[test_sel, 1]] = True
    test_ratings = np.where(test_mask, true_ratings, 0.0).astype(np.float32)

    return {
        "train_ratings": train_ratings,
        "train_mask": observed,
        "test_ratings": test_ratings,
        "test_mask": test_mask,
        "true_ratings": true_ratings.astype(np.float32),
        "true_propensities": propensities.astype(np.float32),
    }
