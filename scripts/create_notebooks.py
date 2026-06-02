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



# ======================================================================
# 02_causal_debiasing.ipynb
# ======================================================================

def make_02() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [

        code("""\
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path().resolve()))
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pymc as pm
import arviz as az

from scripts.download_data import download_coat
from scripts.preprocess import load_coat, make_synthetic_mnar
from scripts.models.bayesian_pmf import BayesianPMF, IPSBayesianPMF
from scripts.models.propensity import BayesianPropensityModel
from scripts.models.deconfounder import DeconfoundedRecommender
from scripts.evaluation.metrics import ndcg_at_k, recall_at_k, rmse_on_test, doubly_robust_ndcg
from scripts.evaluation.calibration import expected_calibration_error, plot_calibration_curve
import os

RANDOM_SEED = 42
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/traces", exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")
"""),

        md("""\
## Notebook 02 — Causal Debiasing with PyMC

### The MNAR Problem: Why Observed Ratings Are Not a Random Sample

When users choose what to rate, the resulting data is **Missing Not At Random (MNAR)**.
Popular items receive more ratings, and users tend to rate items they already like.
A naive recommender trained on this biased sample will:

- **Over-recommend popular items** — the model sees them more and gets more confident
- **Ignore long-tail items** — sparse ratings → high uncertainty → model avoids them
- **Report inflated accuracy** — measured on biased training data, not true preferences

### The Causal DAG

```
     Popularity ─────────────────────────────┐
          │                                  │
          ▼                                  ▼
     User ─────► Exposure (O_{ui}) ─────► Rating (R_{ui})
          └─────────────────────────────────►
                    (direct preference)
```

`Exposure` is a collider: conditioning on it (i.e., only observing rated items)
opens a spurious path between `Popularity` and `Rating`.  To recover the true
causal effect of the item on the user's satisfaction, we must account for
the exposure mechanism.

### Three Strategies Implemented

1. **IPS-PMF**: Inverse Propensity Score weighting — reweight observations by
   1/P(exposed), down-weighting popular items in the likelihood.
2. **DR-NDCG**: Doubly-Robust evaluation — combines IPS correction with a
   direct-model imputation term.  Unbiased if either model is correct.
3. **Deconfounded Recommender** (Wang et al. 2018): Extract a substitute
   confounder Z from the exposure pattern; condition the rating model on Z
   to block the backdoor confounding path.
"""),

        code("""\
# ── Draw the causal DAG ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

nodes = {
    "Popularity": (2, 5),
    "User\\nfeatures": (2, 2),
    "Exposure\\n(O_{ui})": (5, 3.5),
    "Rating\\n(R_{ui})": (8, 3.5),
    "Confounder\\n(Z)": (5, 1),
}
for label, (x, y) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.7, color="steelblue", alpha=0.8, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8, color="white",
            fontweight="bold", zorder=4)

edges = [
    ("Popularity", "Exposure\\n(O_{ui})", "black"),
    ("Popularity", "Rating\\n(R_{ui})", "gray"),
    ("User\\nfeatures", "Exposure\\n(O_{ui})", "black"),
    ("User\\nfeatures", "Rating\\n(R_{ui})", "black"),
    ("Exposure\\n(O_{ui})", "Rating\\n(R_{ui})", "black"),
    ("Confounder\\n(Z)", "Exposure\\n(O_{ui})", "tomato"),
    ("Confounder\\n(Z)", "Rating\\n(R_{ui})", "tomato"),
]
for src, dst, color in edges:
    x1, y1 = nodes[src]; x2, y2 = nodes[dst]
    dx, dy = x2 - x1, y2 - y1
    length = (dx**2 + dy**2)**0.5
    ax.annotate("", xy=(x2 - 0.75*dx/length, y2 - 0.75*dy/length),
                xytext=(x1 + 0.75*dx/length, y1 + 0.75*dy/length),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

ax.text(5, 5.5, "Causal DAG — MNAR Recommendation", ha="center", fontsize=12, fontweight="bold")
ax.text(5, 0.3, "Red arrows = unmeasured confounder paths that Z approximates",
        ha="center", fontsize=8, color="tomato")
plt.tight_layout()
plt.savefig("outputs/figures/02_causal_dag.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("### 1 — Load data"),

        code("""\
USE_COAT = False
try:
    download_coat("data/raw")
    coat = load_coat("data/raw")
    full_train_r    = coat["train"]
    full_test_r     = coat["test"]
    full_train_mask = coat["train_mask"]
    full_test_mask  = coat["test_mask"]
    USE_COAT = True
    print(f"✓ Coat — {full_train_r.shape}")
except Exception as e:
    print(f"⚠️  Coat unavailable: {e}")
    syn = make_synthetic_mnar(n_users=500, n_items=200, n_factors=10, random_seed=RANDOM_SEED)
    full_train_r    = syn["train_ratings"]
    full_test_r     = syn["test_ratings"]
    full_train_mask = syn["train_mask"]
    full_test_mask  = syn["test_mask"]

N_USERS = 50
train_r    = full_train_r[:N_USERS]
train_mask = full_train_mask[:N_USERS]
test_r     = full_test_r[:N_USERS]
test_mask  = full_test_mask[:N_USERS]

dataset_name = "Coat" if USE_COAT else "Synthetic MNAR"
print(f"Dataset: {dataset_name}  |  {N_USERS} users × {train_r.shape[1]} items")
print(f"Train density: {train_mask.mean():.3f}  |  Test density: {test_mask.mean():.3f}")
"""),

        md("### 2 — Bayesian Propensity Model"),

        code("""\
print("Fitting BayesianPropensityModel …")
prop_model = BayesianPropensityModel(random_seed=RANDOM_SEED)
prop_model.build_model(train_mask)
prop_trace = prop_model.fit(draws=500, tune=500, chains=2, target_accept=0.9)
propensities = prop_model.propensity_scores()

ece = expected_calibration_error(propensities.flatten(), train_mask.flatten().astype(int))
print(f"\\nExpected Calibration Error (ECE): {ece:.4f}")
print("(ECE closer to 0 = better calibration — essential for valid IPS correction)")
"""),

        code("""\
# Propensity convergence diagnostics
prop_summary = az.summary(prop_trace, var_names=["mu_alpha", "mu_beta", "sigma_alpha", "sigma_beta"])
print("Propensity model convergence:")
print(prop_summary[["mean","sd","r_hat","ess_bulk"]].to_string())
"""),

        code("""\
plot_calibration_curve(
    propensities.flatten(),
    train_mask.flatten().astype(int),
    "outputs/figures/calibration.png",
)

# Also show propensity heatmap for first 30 users / 50 items
fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(propensities[:30, :50], aspect="auto", cmap="YlOrRd")
ax.set_xlabel("Item index (first 50)")
ax.set_ylabel("User index (first 30)")
ax.set_title("Posterior mean propensity P(O_{ui}=1) — heatmap")
plt.colorbar(im, ax=ax, label="P(observed)")
plt.tight_layout()
plt.savefig("outputs/figures/02_propensity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("### 3 — Model comparison: Naive PMF vs IPS-PMF vs DR-PMF vs Deconfounded"),

        code("""\
results = {}

# ── Naive PMF ────────────────────────────────────────────────────────
print("Fitting Naive BayesianPMF …")
naive = BayesianPMF(n_factors=10, random_seed=RANDOM_SEED)
naive.build_model(train_r, train_mask)
naive.fit(draws=500, tune=500, chains=2, target_accept=0.9)
results["Naive PMF"] = {
    "ndcg":   ndcg_at_k(naive, test_r, test_mask, k=10),
    "recall": recall_at_k(naive, test_r, test_mask, k=10),
    "rmse":   rmse_on_test(naive, test_r, test_mask),
}
naive_preds = naive._score_matrix()
print(f"  Naive PMF — NDCG@10: {results['Naive PMF']['ndcg']:.4f}")
"""),

        code("""\
# ── IPS-PMF ──────────────────────────────────────────────────────────
print("Fitting IPS-BayesianPMF …")
ips = IPSBayesianPMF(n_factors=10, random_seed=RANDOM_SEED)
ips.build_model(train_r, train_mask, propensities, clip_max=5.0)
ips.fit(draws=500, tune=500, chains=2, target_accept=0.9)
results["IPS-PMF"] = {
    "ndcg":   ndcg_at_k(ips, test_r, test_mask, k=10),
    "recall": recall_at_k(ips, test_r, test_mask, k=10),
    "rmse":   rmse_on_test(ips, test_r, test_mask),
}
print(f"  IPS-PMF — NDCG@10: {results['IPS-PMF']['ndcg']:.4f}")

# DR-NDCG uses naive model predictions + IPS propensities
dr_ndcg = doubly_robust_ndcg(naive_preds, propensities, test_r, test_mask, k=10)
results["DR-NDCG (eval)"] = {"ndcg": dr_ndcg, "recall": None, "rmse": None}
print(f"  DR-NDCG@10: {dr_ndcg:.4f}")
"""),

        code("""\
# ── Deconfounded Recommender ─────────────────────────────────────────
print("Fitting DeconfoundedRecommender …")
deconf = DeconfoundedRecommender(n_factors=10, n_z_factors=5, random_seed=RANDOM_SEED)
Z = deconf.fit_factor_model(train_mask)
deconf.fit_outcome_model(train_r, train_mask, Z)

class _DeconfWrapper:
    def _score_matrix(self): return deconf._score_matrix()

results["Deconfounded"] = {
    "ndcg":   ndcg_at_k(_DeconfWrapper(), test_r, test_mask, k=10),
    "recall": recall_at_k(_DeconfWrapper(), test_r, test_mask, k=10),
    "rmse":   rmse_on_test(_DeconfWrapper(), test_r, test_mask),
}
print(f"  Deconfounded — NDCG@10: {results['Deconfounded']['ndcg']:.4f}")
"""),

        code("""\
# ── Results table ────────────────────────────────────────────────────
import pandas as pd
rows = []
for model_name, m in results.items():
    rows.append({
        "Model":     model_name,
        "NDCG@10":  f"{m['ndcg']:.4f}",
        "Recall@10": f"{m['recall']:.4f}" if m["recall"] is not None else "—",
        "RMSE":      f"{m['rmse']:.4f}"  if m["rmse"]   is not None else "—",
    })
df_results = pd.DataFrame(rows).set_index("Model")
print("\\nResults on unbiased test set:")
print(df_results.to_string())
print(f"\\nPropensity ECE: {ece:.4f}")
"""),

        code("""\
# ── Bar chart comparison ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
models = [m for m in results if results[m]["ndcg"] is not None]
ndcgs  = [results[m]["ndcg"] for m in models]
colors = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
bars = ax.bar(models, ndcgs, color=colors[:len(models)], alpha=0.85, edgecolor="white")
for bar, val in zip(bars, ndcgs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("NDCG@10")
ax.set_title(f"Model comparison — {dataset_name} unbiased test set")
ax.set_ylim(0, max(ndcgs) * 1.15)
plt.tight_layout()
plt.savefig("outputs/figures/02_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("### 4 — Counterfactual relevance from the Deconfounded Recommender"),

        code("""\
# Show counterfactual relevance scores for 3 users
sample_users = [0, 1, 2]
n_items_show = 20

fig, axes = plt.subplots(len(sample_users), 1, figsize=(14, 4*len(sample_users)))

for ax, u in zip(axes, sample_users):
    cf_scores = deconf.counterfactual_relevance(u, list(range(n_items_show)))
    means, stds = cf_scores[:, 0], cf_scores[:, 1]

    xs = np.arange(n_items_show)
    ax.bar(xs, means, color="mediumpurple", alpha=0.7, label="E[R | do(A=1)] mean")
    ax.vlines(xs, means - 1.88*stds, means + 1.88*stds, color="black",
              linewidth=1.5, label="~94% credible interval")

    # Highlight items in test set
    in_test = np.where(test_mask[u, :n_items_show])[0]
    if len(in_test) > 0:
        ax.scatter(in_test, means[in_test] + 0.1, marker="*", color="darkorange",
                   zorder=5, s=120, label="In test set")

    ax.set_title(f"User {u} — Counterfactual relevance E[R | do(A=1)]")
    ax.set_xlabel("Item index")
    ax.set_ylabel("Interventional rating")
    ax.set_xticks(xs)
    ax.legend(fontsize=8)

plt.suptitle("Counterfactual relevance scores (deconfounded)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("outputs/figures/02_counterfactual_relevance.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("""\
### 5 — Identifiability caveat

**⚠️  Important: The substitute-confounder approach has known limitations.**

The deconfounded recommender assumes that Z (derived from the exposure
matrix) captures *all* common causes of exposure and ratings.  In practice:

- **Item-level confounders** (e.g., a marketing campaign affecting both
  who sees an item and how they rate it) are not captured by user-factor Z.
- **Ogburn et al. (2020)** showed that the substitute-confounder test
  (checking that Z renders A independent of R) is not a valid falsification
  test for causal identification — a model can pass the test and still
  be causally mis-specified.

**Why we still use it:**
- It's an *improvement* over naive PMF even if not fully identified.
- Evaluating on Coat's **uniform-random test set** (or Yahoo R3's random
  subset) gives honest error estimates that don't suffer from MNAR bias —
  this is the critical safeguard.
- The approach is presented as an approximation with documented limitations,
  not as ground-truth causal identification.

**For production:** combine with A/B testing or bandit logs with logging
propensities to achieve stronger causal guarantees (logged bandit feedback).
"""),

    ]
    return nb


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    save(make_00(), "notebooks/00_data_exploration.ipynb")
    save(make_01(), "notebooks/01_bayesian_pmf.ipynb")
    save(make_02(), "notebooks/02_causal_debiasing.ipynb")
    print("Done.")
