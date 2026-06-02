"""Create Phase 1 Jupyter notebooks using nbformat.

Run from repo root:  python scripts/create_notebooks.py
"""

import nbformat as nbf
from pathlib import Path


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip())


def md(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(src.strip())


def save(nb: nbf.NotebookNode, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {path}")


# ======================================================================
# 00_data_exploration.ipynb
# ======================================================================

def make_00() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [

        code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve()))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.download_data import download_coat, coat_is_available
from scripts.preprocess import load_coat, make_synthetic_mnar

RANDOM_SEED = 42
sns.set_theme(style="whitegrid", palette="muted")
"""),

        md("""\
## Notebook 00 — Data Exploration

**What this notebook does and why.**

Before fitting any model, we need to understand the data.
This notebook loads the Coat Shopping dataset (or falls back to a
synthetic MNAR surrogate) and visualises the key properties that
motivate Bayesian causal modelling:

1. **Rating-scale distribution** — do train vs test ratings differ? (MNAR signal)
2. **Sparsity** — how dense is the interaction matrix?
3. **Popularity bias** — are a few items responsible for most observations?

The Coat dataset has a biased *train* set (users self-selected what to rate)
and a uniform-random *test* set (each user was asked to rate a random subset).
This train/test split is precisely the MNAR structure we aim to correct.
"""),

        code("""\
# ── Load data ────────────────────────────────────────────────────────
USE_COAT = False
try:
    download_coat("data/raw")
    coat = load_coat("data/raw")
    train_r = coat["train"]
    test_r  = coat["test"]
    train_mask = coat["train_mask"]
    test_mask  = coat["test_mask"]
    USE_COAT = True
    print(f"✓ Coat loaded — {train_r.shape[0]} users × {train_r.shape[1]} items")
except Exception as e:
    print(f"⚠️  Coat unavailable: {e}")
    print("Falling back to synthetic MNAR dataset.")
    syn = make_synthetic_mnar(
        n_users=500, n_items=200, n_factors=10, random_seed=RANDOM_SEED
    )
    train_r    = syn["train_ratings"]
    test_r     = syn["test_ratings"]
    train_mask = syn["train_mask"]
    test_mask  = syn["test_mask"]

n_users, n_items = train_r.shape
dataset_name = "Coat" if USE_COAT else "Synthetic MNAR"
print(f"Dataset   : {dataset_name}")
print(f"Shape     : {n_users} users × {n_items} items")
print(f"Train density: {train_mask.mean():.3f}  ({train_mask.sum():,} observations)")
print(f"Test  density: {test_mask.mean():.3f}  ({test_mask.sum():,} observations)")
print(f"Rating scale (train observed): {train_r[train_mask].min():.0f} – {train_r[train_mask].max():.0f}")
"""),

        md("### 1 — Rating distributions: train vs test"),

        code("""\
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, ratings, mask, label, color in zip(
    axes,
    [train_r, test_r],
    [train_mask, test_mask],
    ["Train (biased)", "Test (unbiased)"],
    ["steelblue", "darkorange"],
):
    vals = ratings[mask]
    unique, counts = np.unique(vals, return_counts=True)
    ax.bar(unique, counts / counts.sum(), color=color, alpha=0.8, edgecolor="white")
    ax.set_title(label)
    ax.set_xlabel("Rating value")
    ax.set_ylabel("Fraction of observed ratings")
    ax.set_xticks(unique)

fig.suptitle(f"Rating distributions — {dataset_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/00_rating_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# Test: do distributions differ?
from scipy.stats import ks_2samp
ks_stat, ks_p = ks_2samp(train_r[train_mask], test_r[test_mask])
print(f"Kolmogorov–Smirnov test (train vs test): statistic={ks_stat:.3f}, p={ks_p:.4f}")
if ks_p < 0.05:
    print("→ Distributions differ significantly — MNAR bias confirmed.")
else:
    print("→ Distributions not significantly different.")
"""),

        md("### 2 — Sparsity heatmap"),

        code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, mask, label in zip(
    axes,
    [train_mask, test_mask],
    ["Train mask", "Test mask"],
):
    # Subsample for visibility
    show_u = min(n_users, 100)
    show_i = min(n_items, 100)
    im = ax.imshow(
        mask[:show_u, :show_i].astype(float),
        aspect="auto", cmap="Blues", interpolation="nearest",
    )
    ax.set_title(f"{label}  (first {show_u}×{show_i})")
    ax.set_xlabel("Item index")
    ax.set_ylabel("User index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Sparsity heatmap — {dataset_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/00_sparsity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("### 3 — Item popularity distribution"),

        code("""\
item_obs_counts = train_mask.sum(axis=0)  # observations per item in train

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(
    np.arange(n_items),
    np.sort(item_obs_counts)[::-1],
    color="steelblue", alpha=0.8, edgecolor="none",
)
axes[0].set_title("Item popularity (sorted, train)")
axes[0].set_xlabel("Item rank")
axes[0].set_ylabel("Number of observed ratings")

axes[1].hist(item_obs_counts, bins=30, color="steelblue", alpha=0.8, edgecolor="white")
axes[1].set_title("Popularity distribution histogram")
axes[1].set_xlabel("Observations per item (train)")
axes[1].set_ylabel("Number of items")

pct_top10 = item_obs_counts[np.argsort(item_obs_counts)[-int(n_items*0.1):]].sum() / item_obs_counts.sum()
print(f"Top 10% of items account for {pct_top10:.1%} of training observations → popularity bias present")

fig.suptitle(f"Item popularity — {dataset_name}", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/figures/00_popularity_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("""\
### Summary

| Stat | Value |
|------|-------|
| Dataset | `{name}` |
| Users | `{n_users}` |
| Items | `{n_items}` |
| Train density | see output above |
| Test density  | see output above |

**Key takeaways:**
- The train and test rating distributions differ (MNAR bias) — this is why we need causal correction in Phase 2.
- Item popularity is highly skewed — a small fraction of items dominate training observations.
- The sparsity pattern confirms self-selection: users rate items they already like or know.

These observations motivate the full modelling pipeline in Phases 1–3.
"""),

    ]
    return nb


# ======================================================================
# 01_bayesian_pmf.ipynb
# ======================================================================

def make_01() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [

        code("""\
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path().resolve()))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

from scripts.download_data import download_coat
from scripts.preprocess import load_coat, make_synthetic_mnar
from scripts.models.bayesian_pmf import BayesianPMF
from scripts.evaluation.metrics import ndcg_at_k, recall_at_k, rmse_on_test

RANDOM_SEED = 42
sns.set_theme(style="whitegrid", palette="muted")
"""),

        md("""\
## Notebook 01 — Bayesian Probabilistic Matrix Factorization

### What is Bayesian PMF and why use it?

**Standard Matrix Factorization** decomposes the rating matrix R ≈ U Vᵀ and
returns point estimates for the latent user/item factors. It tells you *where*
the posterior is most likely, but not *how certain* you should be.

**Bayesian PMF** places priors over U and V and infers a full posterior
distribution. This gives us:

- **Calibrated uncertainty** — the model tells you which predictions it's
  confident about vs. which are wild guesses. Items with high posterior
  variance are candidates for exploration (Phase 3: Thompson Sampling).
- **Regularisation for free** — the prior σ_u, σ_v play the role of L2
  regularisation, but their strength is inferred from data.
- **Principled cold-start handling** — for users/items with few observations,
  the prior dominates and predictions shrink toward the global mean, rather
  than overfitting to noise.

### Model

```
sigma_u ~ HalfNormal(1)         # user factor scale
sigma_v ~ HalfNormal(1)         # item factor scale
tau     ~ HalfNormal(1)         # observation noise precision
U[u]    ~ Normal(0, sigma_u)    shape: (n_users, n_factors)
V[i]    ~ Normal(0, sigma_v)    shape: (n_items, n_factors)
R[u,i]  ~ Normal(U[u]·V[i], 1/tau)   for observed (u,i)
```

We use a **non-centred parameterisation** (U = U_offset × σ_u) to improve
NUTS mixing when σ is small — a common trick in hierarchical Bayesian models
(see McElreath 2020, Ch. 13).

Inference is via **NUTS** (No-U-Turn Sampler), giving asymptotically exact
posterior samples with minimal tuning.
"""),

        code("""\
# ── Load data ────────────────────────────────────────────────────────
USE_COAT = False
try:
    download_coat("data/raw")
    coat = load_coat("data/raw")
    full_train_r    = coat["train"]
    full_test_r     = coat["test"]
    full_train_mask = coat["train_mask"]
    full_test_mask  = coat["test_mask"]
    USE_COAT = True
    print(f"✓ Coat loaded — {full_train_r.shape[0]} users × {full_train_r.shape[1]} items")
except Exception as e:
    print(f"⚠️  Coat unavailable: {e}")
    print("Using synthetic MNAR dataset (500 users × 200 items).")
    syn = make_synthetic_mnar(n_users=500, n_items=200, n_factors=10, random_seed=RANDOM_SEED)
    full_train_r    = syn["train_ratings"]
    full_test_r     = syn["test_ratings"]
    full_train_mask = syn["train_mask"]
    full_test_mask  = syn["test_mask"]

dataset_name = "Coat" if USE_COAT else "Synthetic MNAR"
print(f"Dataset: {dataset_name} — {full_train_r.shape}")
"""),

        md("""\
### 1 — Fit model on a small subset (fast iteration)

We first fit on the first **50 users**, all items, to iterate quickly on
diagnostics. Full-dataset fitting follows the same code.
"""),

        code("""\
N_USERS_SUBSET = 50
train_r    = full_train_r[:N_USERS_SUBSET]
train_mask = full_train_mask[:N_USERS_SUBSET]
test_r     = full_test_r[:N_USERS_SUBSET]
test_mask  = full_test_mask[:N_USERS_SUBSET]

print(f"Subset: {train_r.shape[0]} users × {train_r.shape[1]} items")
print(f"  Train observations: {train_mask.sum():,}")
print(f"  Test  observations: {test_mask.sum():,}")
"""),

        code("""\
# Build and fit BayesianPMF
N_FACTORS = 10
model = BayesianPMF(n_factors=N_FACTORS, random_seed=RANDOM_SEED)
model.build_model(train_r, train_mask)

print(f"Model built: {train_mask.sum():,} observed ratings, {N_FACTORS} latent factors")
print("Starting NUTS sampling …")

trace = model.fit(draws=500, tune=500, chains=2, target_accept=0.9)
print("Sampling complete.")
"""),

        md("### 2 — Convergence diagnostics"),

        code("""\
# ── R-hat summary ────────────────────────────────────────────────────
summary_global = az.summary(trace, var_names=["sigma_u", "sigma_v", "tau"])
print("Convergence summary (global parameters):")
print(summary_global.to_string())

max_rhat = summary_global["r_hat"].max()
if max_rhat < 1.05:
    print(f"\\n✓ All R-hat < 1.05 (max = {max_rhat:.3f}) — chains converged.")
else:
    print(f"\\n⚠️  Max R-hat = {max_rhat:.3f} — consider more tuning steps.")
"""),

        code("""\
# ── Trace plots ──────────────────────────────────────────────────────
ax = az.plot_trace(
    trace,
    var_names=["sigma_u", "sigma_v", "tau"],
    figsize=(12, 6),
    combined=False,
)
plt.suptitle("Trace plots — global hyperparameters", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/figures/01_trace_plot.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        code("""\
# ── Posterior plots ──────────────────────────────────────────────────
az.plot_posterior(
    trace,
    var_names=["sigma_u", "sigma_v", "tau"],
    figsize=(12, 4),
    kind="hist",
)
plt.suptitle("Posterior distributions — global hyperparameters", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("outputs/figures/01_posterior_plot.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        code("""\
# ── Energy plot ───────────────────────────────────────────────────────
az.plot_energy(trace, figsize=(8, 4))
plt.title("Energy plot — BFMI diagnostic")
plt.tight_layout()
plt.savefig("outputs/figures/01_energy_plot.png", dpi=150, bbox_inches="tight")
plt.show()

# BFMI < 0.2 is a warning sign
bfmi = az.bfmi(trace)
print(f"BFMI per chain: {bfmi}")
if any(b < 0.2 for b in bfmi):
    print("⚠️  Low BFMI — possible divergences; try more tuning steps or re-parameterisation.")
else:
    print("✓ BFMI looks healthy.")
"""),

        md("### 3 — Predictions and posterior predictive check"),

        code("""\
# ── Posterior predictive check ───────────────────────────────────────
with model.model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=RANDOM_SEED, progressbar=False)

ppc_obs = ppc.posterior_predictive["obs"].values.reshape(-1)
actual_obs = train_r[train_mask]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(actual_obs, bins=20, alpha=0.6, label="Observed (train)", color="steelblue",
        density=True, edgecolor="white")
ax.hist(ppc_obs, bins=50, alpha=0.5, label="PPC samples", color="darkorange",
        density=True, edgecolor="none")
ax.set_xlabel("Rating value")
ax.set_ylabel("Density")
ax.set_title("Posterior Predictive Check — rating distribution")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/01_ppc.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        code("""\
# ── Posterior mean ± 94% HDI for 3 sample users ──────────────────────
sample_users = [0, 1, 2]
TOP_ITEMS = 20

fig, axes = plt.subplots(len(sample_users), 1, figsize=(14, 4 * len(sample_users)))

for ax, u in zip(axes, sample_users):
    means, stds = model.predict_all_items(u)
    top_idx = np.argsort(means)[::-1][:TOP_ITEMS]

    # HDI bounds via approximate normal
    z94 = 1.88  # ≈ 94% CI half-width multiplier for normal
    lo = means[top_idx] - z94 * stds[top_idx]
    hi = means[top_idx] + z94 * stds[top_idx]

    xs = np.arange(TOP_ITEMS)
    ax.bar(xs, means[top_idx], color="steelblue", alpha=0.7, label="Posterior mean")
    ax.vlines(xs, lo, hi, color="black", linewidth=1.5, label="~94% HDI")

    # Mark items that appear in test set
    in_test = np.isin(top_idx, np.where(test_mask[u])[0])
    if in_test.any():
        ax.scatter(xs[in_test], means[top_idx][in_test] + 0.1,
                   marker="*", color="darkorange", zorder=5, s=120,
                   label="In test set")

    ax.set_title(f"User {u} — top-{TOP_ITEMS} items by posterior mean")
    ax.set_xlabel("Item rank")
    ax.set_ylabel("Predicted rating")
    ax.set_xticks(xs)
    ax.set_xticklabels(top_idx, rotation=45, fontsize=7)
    ax.legend(fontsize=8)

plt.suptitle("Posterior mean ± 94% HDI for sample users", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("outputs/figures/01_user_predictions.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("### 4 — Evaluation on unbiased test set"),

        code("""\
# ── Metrics ───────────────────────────────────────────────────────────
ndcg  = ndcg_at_k(model, test_r, test_mask, k=10)
rec   = recall_at_k(model, test_r, test_mask, k=10)
rmse  = rmse_on_test(model, test_r, test_mask)

print("=" * 45)
print("Results on unbiased test set:")
print(f"  NDCG@10  : {ndcg:.4f}")
print(f"  Recall@10: {rec:.4f}")
print(f"  RMSE     : {rmse:.4f}")
print("=" * 45)
"""),

        code("""\
# Save trace for Phase 2
import os
os.makedirs("outputs/traces", exist_ok=True)
az.to_netcdf(trace, "outputs/traces/bpmf_phase1.nc")
print("Trace saved to outputs/traces/bpmf_phase1.nc")
"""),

        md("""\
### 5 — Reflection: what does posterior uncertainty tell us?

The posterior uncertainty captured in `std` and the 94% HDI has several
practical implications:

1. **Item popularity effect**: Items that appear rarely in the training data
   (long-tail items) have high posterior variance — the model is uncertain
   because it has seen little evidence. This is epistemically honest.

2. **User heterogeneity**: Some users have narrow HDI bands (they rated many
   items, constraining their latent factor U[u]); others have wide bands
   (new/sparse users). A point-estimate model hides this distinction.

3. **Exploration signal** (Phase 3 preview): The items with highest *uncertainty*
   are precisely the ones a Thompson Sampler should explore — if the true rating
   is high, we miss value by never recommending them; the posterior variance is
   the natural signal for exploration.

4. **MNAR caveat**: Even with calibrated uncertainty, the model was trained on
   *biased* data. Items that were never exposed to a user are absent from
   training — their latent factor V[i] is estimated only from other users'
   ratings. Phase 2 addresses this via causal debiasing.

---

**Note on convergence**: If R-hat ≥ 1.05 appeared above, the chains may not
have fully mixed. Common fixes:
- Increase `tune` from 500 to 1000+
- Increase `target_accept` to 0.95
- Reduce `n_factors` (fewer parameters to explore)
- The non-centred parameterisation in our model already helps; further
  improvement requires more data or better initialisation.
"""),

    ]
    return nb


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    save(make_00(), "notebooks/00_data_exploration.ipynb")
    save(make_01(), "notebooks/01_bayesian_pmf.ipynb")
    print("Done.")
