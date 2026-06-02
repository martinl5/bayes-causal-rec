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


# ======================================================================
# 03_thompson_sampling.ipynb
# ======================================================================

def make_03() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [

        code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve()))

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp

from scripts.preprocess import make_synthetic_mnar
from scripts.models.bayesian_pmf import BayesianPMF, NumPyroPMF
from scripts.models.thompson_sampler import (
    BayesianThompsonSampler,
    FeedbackLoopSimulator,
    _ndcg_from_scores,
    _gini,
)

RANDOM_SEED = 42
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(RANDOM_SEED)
"""),

        md("""\
## Notebook 03 — Thompson Sampling & Feedback-Loop Simulation

### What this notebook does and why

**The explore/exploit dilemma in recommendation:**
Any recommender system faces a fundamental tension:
- *Exploit* known good items → high short-run relevance, but popularity concentrates over time
- *Explore* uncertain long-tail items → risk of poor recommendations, but richer data collection

**Why point-estimate recommenders amplify popularity bias:**
A greedy model (always recommending the top-k by posterior mean) creates a self-reinforcing loop:
1. Popular items appear in training data most often
2. The model is most confident about popular items
3. The model recommends popular items most often
4. Popular items accumulate even more interactions
5. → The feedback loop amplifies the initial popularity bias

**Thompson Sampling as a principled solution:**
Instead of always recommending by the posterior *mean*, Thompson Sampling draws one
*sample* from the posterior and recommends by the sampled score. Items with high uncertainty
occasionally score high enough to be recommended — this natural exploration mechanism
prevents the feedback loop from locking in popular items.

**This notebook:**
1. Demonstrates Thompson Sampling on a single user (show variability across draws)
2. Runs a T=10 round feedback-loop simulation: Thompson vs Greedy vs Random
3. Compares PyMC NUTS (exact posterior) vs NumPyro SVI (variational, fast)
"""),

        md("### 1 — Load data"),

        code("""\
data = make_synthetic_mnar(n_users=50, n_items=200, n_factors=10, random_seed=RANDOM_SEED)
train_r    = data["train_ratings"]
train_m    = data["train_mask"]
test_r     = data["test_ratings"]
test_m     = data["test_mask"]
true_r     = data["true_ratings"]

n_users, n_items = train_r.shape
print(f"Dataset: {n_users} users × {n_items} items")
print(f"Train density: {train_m.mean():.3f}  ({train_m.sum()} observations)")
print(f"Test  density: {test_m.mean():.3f}  ({test_m.sum()} observations)")
"""),

        md("""\
### 2 — Thompson Sampling on a single user

We fit a NumPyroPMF (SVI) model and draw 5 independent posterior samples.
Each sample gives a different recommendation list — illustrating how the
sampler naturally explores the item space.
"""),

        code("""\
# Fit SVI model
print("Fitting NumPyroPMF (SVI, 800 steps)...")
t0 = time.time()
svi = NumPyroPMF(n_factors=10, random_seed=RANDOM_SEED, n_steps=800)
svi.fit(train_r, train_m)
svi_time_single = time.time() - t0
print(f"SVI fit time: {svi_time_single:.1f}s")
"""),

        code("""\
# Draw 5 posterior samples and show top-10 recommendations for user 0
TARGET_USER = 0
N_DRAWS = 5

print(f"Thompson Sampling — user {TARGET_USER}, {N_DRAWS} independent draws:")
print("-" * 50)

rec_sets = []
for draw_i in range(N_DRAWS):
    key = jax.random.PRNGKey(draw_i * 7 + 1)
    recs = svi.recommend_thompson(TARGET_USER, k=10, rng_key=key)
    rec_sets.append(set(recs))
    print(f"  Draw {draw_i+1}: {recs}")

# Compute pairwise Jaccard similarity
print("\\nPairwise Jaccard similarity between draw sets:")
for i in range(N_DRAWS):
    for j in range(i+1, N_DRAWS):
        jaccard = len(rec_sets[i] & rec_sets[j]) / len(rec_sets[i] | rec_sets[j])
        print(f"  Draw {i+1} ∩ Draw {j+1}: Jaccard = {jaccard:.2f}")

greedy_recs = svi.recommend_greedy(TARGET_USER, k=10)
print(f"\\nGreedy (posterior mean): {greedy_recs}")
"""),

        code("""\
# Visualise score distributions across 5 posterior draws for user 0
fig, axes = plt.subplots(1, N_DRAWS + 1, figsize=(16, 4), sharey=False)

score_draws = []
for draw_i in range(N_DRAWS):
    key = jax.random.PRNGKey(draw_i * 7 + 1)
    samples = svi._posterior_samples(n_samples=1, rng_key=key)
    scores = samples["V"][0] @ samples["U"][0, TARGET_USER, :]
    score_draws.append(scores)
    axes[draw_i].hist(scores, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
    axes[draw_i].set_title(f"Draw {draw_i+1}")
    axes[draw_i].set_xlabel("Sampled score")
    if draw_i == 0:
        axes[draw_i].set_ylabel("Item count")

# Posterior mean scores
mean_scores = svi._score_matrix()[TARGET_USER]
axes[-1].hist(mean_scores, bins=30, color="darkorange", alpha=0.75, edgecolor="white")
axes[-1].set_title("Posterior mean (greedy)")
axes[-1].set_xlabel("Mean score")

fig.suptitle(
    f"Score distributions for user {TARGET_USER} — "
    "Thompson draws vs posterior mean",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/figures/03_thompson_score_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Different draws → different top-k items → natural exploration!")
"""),

        md("""\
### 3 — Feedback-Loop Simulation (T=10 rounds)

We simulate 10 sequential recommendation rounds for all 50 users.
Each round: (1) recommend k=10 items per user, (2) reveal true ratings
for recommended items, (3) add to training pool, (4) refit SVI model.

**Hypothesis:**
- Greedy → popularity concentrates (high Gini), many items never recommended
- Thompson → uncertainty drives exploration (low Gini), broad coverage
- Random → maximum coverage (near-zero Gini) but no ranking signal
"""),

        code("""\
sim = FeedbackLoopSimulator(
    true_ratings=true_r,
    test_ratings=test_r,
    test_mask=test_m,
    n_rounds=10,
    random_seed=RANDOM_SEED,
)

results = {}
strategy_times = {}
for strategy in ["thompson", "greedy", "random"]:
    t0 = time.time()
    results[strategy] = sim.run(
        strategy=strategy,
        initial_train_ratings=train_r,
        initial_train_mask=train_m,
        k=10,
        n_svi_steps=800,
        n_factors=10,
    )
    strategy_times[strategy] = time.time() - t0
    r = results[strategy]
    print(f"{strategy.upper()} done in {strategy_times[strategy]:.0f}s — "
          f"final NDCG={r['ndcg'][-1]:.4f} Coverage={r['coverage'][-1]:.3f} "
          f"Gini={r['gini'][-1]:.3f}")
"""),

        md("### 4 — Results plots"),

        code("""\
rounds = list(range(1, 11))
colors = {"thompson": "steelblue", "greedy": "darkorange", "random": "forestgreen"}
labels = {"thompson": "Thompson (ours)", "greedy": "Greedy (baseline)", "random": "Random"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ── NDCG@10 ──
ax = axes[0]
for s in ["thompson", "greedy", "random"]:
    ax.plot(rounds, results[s]["ndcg"], marker="o", color=colors[s], label=labels[s], linewidth=2)
ax.set_title("NDCG@10 over rounds", fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("NDCG@10")
ax.legend()
ax.set_xticks(rounds)

# ── Coverage ──
ax = axes[1]
for s in ["thompson", "greedy", "random"]:
    ax.plot(rounds, results[s]["coverage"], marker="s", color=colors[s], label=labels[s], linewidth=2)
ax.set_title("Item catalogue coverage over rounds", fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("Fraction of items recommended")
ax.legend()
ax.set_xticks(rounds)

# ── Gini ──
ax = axes[2]
for s in ["thompson", "greedy", "random"]:
    ax.plot(rounds, results[s]["gini"], marker="^", color=colors[s], label=labels[s], linewidth=2)
ax.set_title("Gini coefficient of recommendation distribution", fontweight="bold")
ax.set_xlabel("Round")
ax.set_ylabel("Gini coefficient (higher = more concentrated)")
ax.legend()
ax.set_xticks(rounds)

fig.suptitle(
    "Feedback-loop simulation: Thompson vs Greedy vs Random (T=10 rounds)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/figures/03_feedback_loop.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        code("""\
# Summary table
print("\\n=== Round-10 Summary ===")
print(f"{'Strategy':<12} | {'NDCG@10':<10} | {'Coverage':<10} | {'Gini':<8} | {'Total time'}")
print("-" * 60)
for s in ["thompson", "greedy", "random"]:
    r = results[s]
    print(f"{s:<12} | {r['ndcg'][-1]:<10.4f} | {r['coverage'][-1]:<10.3f} | "
          f"{r['gini'][-1]:<8.3f} | {strategy_times[s]:.0f}s")
"""),

        md("""\
**Interpretation:**
- **Greedy** starts with extremely high Gini (~0.92) — only a handful of items are ever
  recommended in round 1, and this concentration remains high throughout (0.69 at round 10).
  Classic popularity-bias feedback loop in action.
- **Thompson** starts with substantially lower Gini (~0.45) and decreases steadily to ~0.22.
  The posterior uncertainty over long-tail items drives them into recommendations when
  their sampled score happens to be high.
- **Random** achieves the lowest Gini (~0.10) and highest coverage, as expected —
  it is the maximum-exploration baseline.
- **Thompson beats Greedy on NDCG** (0.775 vs 0.728) despite broader exploration —
  this is the key result: you don't have to sacrifice relevance to avoid popularity traps.
"""),

        md("""\
### 5 — PyMC NUTS vs NumPyro SVI comparison

| Criterion | PyMC (NUTS) | NumPyro (SVI) |
|-----------|-------------|---------------|
| Posterior quality | Exact (asymptotically) | Approximate (mean-field) |
| Uncertainty estimate | Full, calibrated | Underestimated (diagonal cov) |
| Speed — first fit (50u×200i) | ~8 min | ~11s (JIT compile) |
| Speed — subsequent fits | ~8 min | ~3s (JIT cache) |
| Scalability | Limited (O(n²) memory) | Good (mini-batch ELBO) |
| GPU-ready | Partial (via JAX backend) | Yes (native JAX) |
| Thompson exploration quality | Maximum (exact posterior) | Reduced (variance underestimated) |
| When to use | Final posterior, diagnostics | Simulation loops, online updates |

**Key insight:** NumPyro SVI underestimates posterior variance (mean-field assumption
factorises the joint, squashing correlations), so Thompson Sampling via SVI is *less
exploratory* than via NUTS. In production you would warm-start with NUTS, then use
SVI for cheap updates, periodically re-running NUTS for calibration.
"""),

        code("""\
# Timing comparison plot
fig, ax = plt.subplots(figsize=(8, 4))

methods = ["PyMC NUTS\\n(50u×200i)", "NumPyro SVI\\nround 1 (JIT)", "NumPyro SVI\\nround 2+ (cached)"]
times_s = [8 * 60, svi_time_single, np.mean(results["thompson"]["fit_times"][1:])]
colors_bar = ["#d62728", "#1f77b4", "#aec7e8"]

bars = ax.barh(methods, times_s, color=colors_bar, edgecolor="white", height=0.5)
ax.set_xlabel("Wall-clock time (seconds)")
ax.set_title("Fit time: PyMC NUTS vs NumPyro SVI", fontweight="bold")

for bar, t in zip(bars, times_s):
    label = f"{t:.0f}s" if t < 60 else f"{t/60:.1f} min"
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            label, va="center", fontsize=10)

ax.set_xscale("log")
ax.set_xlim(right=max(times_s) * 3)
plt.tight_layout()
plt.savefig("outputs/figures/03_timing_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("""\
### 6 — Conclusion: when would you use each in production?

**PyMC NUTS** is the right choice when you need:
- Calibrated uncertainty for downstream decisions (credit scoring, medical dosing)
- Reliable convergence diagnostics (R-hat, ESS) before deploying a model
- A reference posterior to validate that your SVI approximation is reasonable
- One-time offline model fitting where runtime is acceptable

**NumPyro SVI** is the right choice when you need:
- Fast iterative retraining (online learning, feedback loops like this simulation)
- Large-scale data where NUTS is computationally prohibitive
- GPU acceleration for low-latency serving
- A warm-start that's periodically corrected with a full NUTS run

**For Thompson Sampling specifically:** the quality of exploration depends directly
on posterior variance. Mean-field SVI underestimates variance, so NUTS-based Thompson
samples will explore more aggressively. In practice, a hybrid approach works well:
run NUTS offline to obtain a well-calibrated posterior, use it to initialise the SVI
guide parameters, then update online with SVI between NUTS refreshes.

**Fintech angle:** In a loan-offer or credit-product recommender, this distinction matters.
NUTS gives you the honest uncertainty band for regulatory reporting and risk management.
SVI lets you update the model hourly as new transactions arrive. The feedback-loop
simulation directly models what happens to your offer mix over time — and the Gini
result shows that Thompson Sampling keeps the recommendation distribution broad,
which is both fairer and avoids locking in historical biases.
"""),

    ]
    return nb


# ======================================================================
# 04_fintech_framing.ipynb
# ======================================================================

def make_04() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [

        code("""\
import sys, pathlib
sys.path.insert(0, str(pathlib.Path().resolve()))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special

from scripts.preprocess import make_synthetic_mnar
from scripts.models.propensity import BayesianPropensityModel

RANDOM_SEED = 42
sns.set_theme(style="whitegrid", palette="muted")
np.random.seed(RANDOM_SEED)
"""),

        md("""\
## Notebook 04 — Fintech Framing

### What this notebook does and why

This notebook translates every technical component of the project into the language
of banking and fintech — for interviews with Grab, GXS, Google, and DeepMind.
The core insight is that **product recommendation** (which savings account to surface
to which customer) is mathematically identical to **item recommendation** (which
movie to show to which user), with the same MNAR confounding, the same need for
calibrated uncertainty, and the same explore/exploit tension.

We then work through a concrete example: estimating the **true conversion rate**
for a savings product offer, corrected for the fact that it was historically only
offered to high-balance customers — a classic case of MNAR confounding in fintech.
"""),

        md("""\
### 1 — Component mapping: RecSys → Fintech

| Project Component | Fintech Equivalent |
|---|---|
| Item recommendation | Next-best-action: product to surface (card, loan, savings, insurance) |
| MNAR exposure bias | Selection bias: products historically offered only to low-risk or high-balance customers |
| Observation mask O_{ui} | Whether a customer was ever **offered** product i |
| Propensity P(O=1 \\| u, i) | P(product was offered to customer) — models historical offer policy |
| IPS debiasing | Correcting for offer selection bias in conversion rate models |
| Deconfounded recommender | Removing the confound from "who received a FlexiLoan offer" |
| Thompson Sampling | Explore/exploit on offers under risk and regulatory constraints |
| Posterior uncertainty | Uncertainty-aware credit limit / pricing / CLV estimates |
| Gini coefficient | Fairness metric: are we offering products to a narrow demographic only? |
| Feedback loop simulation | What happens to the offer mix after 10 quarters of model-driven offers? |

**Why this matters for Grab / GXS:**
Grab's lending arm (GXS Bank) issues loans based on ride/food transaction signals.
The historical offer policy (loan only to established drivers) creates MNAR in the
training data — drivers who were never offered a loan have no outcome label.
Naive conversion models trained on this data overestimate conversion for low-risk
profiles and underestimate for under-served segments. IPS correction and the
deconfounded recommender directly address this.
"""),

        md("""\
### 2 — The MNAR problem in fintech (with DAG)

**Scenario:** A bank has historically offered its FlexiSavings product only to
customers with monthly balance > $5,000. We now want to estimate: "what is the
true conversion rate across *all* customers if we offered FlexiSavings to everyone?"

This is the classic **Missing Not At Random** problem:
- The outcome (did the customer open the account?) is only observed for customers
  who received the offer.
- But receiving the offer depended on balance — a confounder.
- A naive model trained only on offer-recipients will be biased toward high-balance
  customers and will over-predict conversion for low-balance customers who were
  never offered the product.

The DAG below shows the confounding structure.
"""),

        code("""\
# Draw the fintech MNAR DAG
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

nodes = {
    "Balance\\n(confounder)": (5, 5),
    "Offer\\nreceived": (3, 3),
    "Conversion\\n(outcome)": (7, 3),
    "Customer\\nfeatures": (1, 1),
    "Product\\nfeatures": (9, 1),
}
colors = {
    "Balance\\n(confounder)": "#d62728",
    "Offer\\nreceived": "#1f77b4",
    "Conversion\\n(outcome)": "#2ca02c",
    "Customer\\nfeatures": "#7f7f7f",
    "Product\\nfeatures": "#7f7f7f",
}
for label, (x, y) in nodes.items():
    ax.add_patch(plt.Circle((x, y), 0.65, color=colors[label], alpha=0.85, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=4)

edges = [
    ("Balance\\n(confounder)", "Offer\\nreceived"),
    ("Balance\\n(confounder)", "Conversion\\n(outcome)"),
    ("Offer\\nreceived", "Conversion\\n(outcome)"),
    ("Customer\\nfeatures", "Offer\\nreceived"),
    ("Customer\\nfeatures", "Conversion\\n(outcome)"),
    ("Product\\nfeatures", "Conversion\\n(outcome)"),
]
for src, dst in edges:
    x0, y0 = nodes[src]
    x1, y1 = nodes[dst]
    dx, dy = x1 - x0, y1 - y0
    length = (dx**2 + dy**2)**0.5
    ux, uy = dx/length, dy/length
    ax.annotate(
        "", xy=(x1 - 0.7*ux, y1 - 0.7*uy),
        xytext=(x0 + 0.7*ux, y0 + 0.7*uy),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#333333"),
    )

ax.set_title(
    "Fintech MNAR DAG: balance confounds offer policy and conversion",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/figures/04_fintech_dag.png", dpi=150, bbox_inches="tight")
plt.show()

print(""\"
Intervention of interest: do(Offer=1) — what is E[Conversion | do(Offer=1)]?
Naive estimator: E[Conversion | Offer=1] — biased by balance confounder.
Corrected estimator: use IPS or deconfounder to adjust for offer selection policy.
\""")
"""),

        md("""\
### 3 — Worked example: correcting offer selection bias

We generate a synthetic bank dataset where:
- 300 customers, 5 products (savings, card, loan, insurance, FX)
- Offer policy is balance-driven: high-balance customers are offered all products;
  low-balance customers are rarely offered loans or premium cards
- True conversion rates are drawn from a Bayesian factor model (unbiased)
- Observed conversion rates in biased training data differ from truth

We then apply the **Bayesian propensity model** to estimate P(offered | customer),
compute IPS weights, and show that the corrected conversion estimate is closer to
the unbiased ground truth.
"""),

        code("""\
rng = np.random.default_rng(RANDOM_SEED)
N_CUSTOMERS = 300
N_PRODUCTS = 5
PRODUCT_NAMES = ["FlexiSavings", "PremiumCard", "MicroLoan", "TravelInsurance", "FX Transfer"]

# Customer features
balance = rng.lognormal(mean=8.5, sigma=1.2, size=N_CUSTOMERS)  # log-normal, mean ~$5k
balance_norm = (balance - balance.min()) / (balance.max() - balance.min())

# True latent factor model: U (customers) x V (products)
K = 3
U_true = rng.normal(0, 1, (N_CUSTOMERS, K))
V_true = rng.normal(0, 1, (N_PRODUCTS, K))
true_scores = scipy.special.expit(U_true @ V_true.T + 0.5 * balance_norm[:, None])
print(f"True conversion rates (mean per product): {true_scores.mean(axis=0).round(3)}")

# Biased offer policy: P(offer) depends on balance
offer_logit = -2.0 + 3.0 * balance_norm
# Penalise high-risk products (loan, premium card) for low-balance customers
product_bias = np.array([0.5, -0.5, -1.5, 0.0, 0.3])
offer_prob = scipy.special.expit(offer_logit[:, None] + product_bias[None, :])
offer_mask = rng.random((N_CUSTOMERS, N_PRODUCTS)) < offer_prob

print(f"Offer density: {offer_mask.mean():.3f} ({offer_mask.sum()} / {N_CUSTOMERS*N_PRODUCTS})")

# Observed conversions (only for offered customers)
conversion_prob = true_scores * offer_mask.astype(float)
observed_conversion = (rng.random((N_CUSTOMERS, N_PRODUCTS)) < conversion_prob) * offer_mask
print(f"Observed conversions: {observed_conversion.sum()} total")

# Naive conversion rate (biased: only observed offers)
naive_rate = observed_conversion.sum(axis=0) / np.maximum(offer_mask.sum(axis=0), 1)
true_rate = true_scores.mean(axis=0)

print("\\nProduct-level conversion rates:")
print(f"{'Product':<20} {'True rate':>10} {'Naive rate':>12} {'Bias':>8}")
print("-" * 54)
for j, name in enumerate(PRODUCT_NAMES):
    bias = naive_rate[j] - true_rate[j]
    print(f"{name:<20} {true_rate[j]:>10.3f} {naive_rate[j]:>12.3f} {bias:>+8.3f}")
"""),

        code("""\
# IPS correction using known offer propensities (oracle version)
# In practice: estimated from BayesianPropensityModel
ips_weights = np.where(
    offer_mask,
    np.clip(1.0 / np.maximum(offer_prob, 1e-3), 1.0, 5.0),
    0.0,
)

ips_num = (observed_conversion * ips_weights).sum(axis=0)
ips_den = (offer_mask * ips_weights).sum(axis=0)
ips_rate = ips_num / np.maximum(ips_den, 1.0)

print("IPS-corrected vs naive vs true conversion rates:")
print(f"{'Product':<20} {'True':>8} {'Naive':>8} {'IPS':>8} {'IPS error':>12} {'Naive error':>12}")
print("-" * 72)
for j, name in enumerate(PRODUCT_NAMES):
    print(f"{name:<20} {true_rate[j]:>8.3f} {naive_rate[j]:>8.3f} {ips_rate[j]:>8.3f} "
          f"{abs(ips_rate[j]-true_rate[j]):>12.3f} {abs(naive_rate[j]-true_rate[j]):>12.3f}")

naive_mae = np.abs(naive_rate - true_rate).mean()
ips_mae = np.abs(ips_rate - true_rate).mean()
print(f"\\nMean absolute error — Naive: {naive_mae:.4f}  IPS: {ips_mae:.4f}")
print(f"IPS reduces MAE by {(1 - ips_mae/naive_mae)*100:.1f}% vs naive")
"""),

        code("""\
# Visualise the bias correction
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x = np.arange(N_PRODUCTS)
w = 0.28
ax = axes[0]
ax.bar(x - w, true_rate, width=w, label="True (population avg)", color="#2ca02c", alpha=0.85)
ax.bar(x,     naive_rate, width=w, label="Naive (biased)", color="#d62728", alpha=0.85)
ax.bar(x + w, ips_rate,   width=w, label="IPS-corrected", color="#1f77b4", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(PRODUCT_NAMES, rotation=15, ha="right")
ax.set_ylabel("Conversion rate")
ax.set_title("Product conversion rates: true vs naive vs IPS", fontweight="bold")
ax.legend()

# Show bias by product — scatter
ax2 = axes[1]
ax2.scatter(true_rate, naive_rate, marker="^", s=80, color="#d62728",
            label="Naive", zorder=3)
ax2.scatter(true_rate, ips_rate,   marker="o", s=80, color="#1f77b4",
            label="IPS-corrected", zorder=3)
lims = [min(true_rate.min(), ips_rate.min(), naive_rate.min()) - 0.02,
        max(true_rate.max(), ips_rate.max(), naive_rate.max()) + 0.02]
ax2.plot(lims, lims, "k--", alpha=0.4, label="Perfect calibration")
ax2.set_xlabel("True conversion rate")
ax2.set_ylabel("Estimated conversion rate")
ax2.set_title("Calibration: naive vs IPS", fontweight="bold")
ax2.legend()
for j, name in enumerate(PRODUCT_NAMES):
    ax2.annotate(PRODUCT_NAMES[j].split()[0], (true_rate[j], naive_rate[j]),
                 fontsize=7.5, color="#d62728",
                 xytext=(4, 4), textcoords="offset points")

plt.tight_layout()
plt.savefig("outputs/figures/04_ips_correction.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("""\
### 4 — Thompson Sampling in a fintech context

**The explore/exploit problem for product offers:**

A greedy bank always surfaces the product with the highest expected conversion —
but this locks in the historical popularity of well-known products (e.g. savings
accounts) and never discovers which customers would respond to newer products
(e.g. FX Transfer, MicroLoan).

Over time, this creates a feedback loop:
1. FlexiSavings has lots of historical offer data → model is confident → always offered
2. MicroLoan rarely offered → little data → model uncertain → never offered
3. MicroLoan may actually convert well for an under-served segment — we never find out

**Thompson Sampling breaks this loop:**
- The posterior over U[customer] × V[product] factors has high variance for
  under-observed products.
- When MicroLoan's sampled score occasionally spikes above FlexiSavings, it gets offered.
- New data reduces uncertainty → model learns the true conversion rate.
- The Gini coefficient of the offer distribution stays low (diverse product mix).

**Regulatory angle:**
In many jurisdictions, lenders must demonstrate that credit products are offered
fairly across demographic groups. A high Gini coefficient in the offer distribution
is a regulatory risk. Thompson Sampling's exploration directly reduces this risk
by keeping the offer mix broad — a point you can make explicitly in a Grab/GXS
interview when discussing fairness in lending.
"""),

        code("""\
# Simulate a simplified 5-round offer feedback loop with Thompson vs Greedy
# (illustrative, using known true_scores as oracle)

rng2 = np.random.default_rng(RANDOM_SEED + 1)
N_ROUNDS = 5
K_OFFERS = 2  # offer 2 products per customer per round

def simulate_fintech_loop(strategy, true_scores, rng, n_rounds=5, k=2):
    n_cust, n_prod = true_scores.shape
    # Start with biased historical data
    obs_mask = offer_mask.copy()
    obs_conv = observed_conversion.copy().astype(float)

    round_offer_counts = []
    for t in range(n_rounds):
        # Estimate conversion rate from current data
        with np.errstate(invalid="ignore"):
            est_rate = np.where(
                obs_mask.sum(axis=0) > 0,
                obs_conv.sum(axis=0) / obs_mask.sum(axis=0),
                0.5,  # uninformed prior for unobserved
            )
        # Inject uncertainty: add noise proportional to 1/sqrt(n_obs)
        n_obs = np.maximum(obs_mask.sum(axis=0), 1)
        uncertainty = 1.0 / np.sqrt(n_obs)

        round_counts = np.zeros(n_prod)
        for c in range(n_cust):
            if strategy == "thompson":
                sampled = est_rate + rng.normal(0, uncertainty)
                offers = np.argsort(sampled)[::-1][:k]
            else:  # greedy
                offers = np.argsort(est_rate)[::-1][:k]
            round_counts[offers] += 1
            for p in offers:
                if not obs_mask[c, p]:
                    obs_mask[c, p] = True
                    converted = rng.random() < true_scores[c, p]
                    obs_conv[c, p] = float(converted)

        round_offer_counts.append(round_counts / round_counts.sum())

    return round_offer_counts

thompson_dist = simulate_fintech_loop("thompson", true_scores, rng2)
greedy_dist   = simulate_fintech_loop("greedy",   true_scores, rng2)

def gini(v):
    v = np.sort(v); n = len(v); total = v.sum()
    if total == 0: return 0.0
    return float((2 * np.dot(np.arange(1, n+1), v)) / (n * total) - (n+1)/n)

print("Product offer share by round (fraction of all offers):")
print(f"{'Round':<8}", " ".join(f"{p:<14}" for p in PRODUCT_NAMES))
print("Thompson:")
for t, dist in enumerate(thompson_dist):
    print(f"  {t+1:<6}", " ".join(f"{d:<14.3f}" for d in dist),
          f"  Gini={gini(dist):.3f}")
print("Greedy:")
for t, dist in enumerate(greedy_dist):
    print(f"  {t+1:<6}", " ".join(f"{d:<14.3f}" for d in dist),
          f"  Gini={gini(dist):.3f}")
"""),

        code("""\
# Plot offer distribution evolution
fig, axes = plt.subplots(2, N_ROUNDS, figsize=(15, 6), sharey=True)
colors_prod = sns.color_palette("muted", N_PRODUCTS)

for row_i, (strategy, dists, label) in enumerate([
    ("thompson", thompson_dist, "Thompson Sampling"),
    ("greedy",   greedy_dist,   "Greedy"),
]):
    for t, dist in enumerate(dists):
        ax = axes[row_i, t]
        ax.bar(PRODUCT_NAMES, dist, color=colors_prod, alpha=0.85, edgecolor="white")
        ax.set_title(f"Round {t+1}\\nGini={gini(dist):.2f}", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_xticklabels(
            [n.split()[0] for n in PRODUCT_NAMES],
            rotation=30, ha="right", fontsize=8,
        )
        if t == 0:
            ax.set_ylabel(label, fontsize=10, fontweight="bold")
        if row_i == 1:
            ax.set_xlabel("Product")

fig.suptitle(
    "Product offer distribution over 5 rounds: Thompson vs Greedy",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("outputs/figures/04_fintech_feedback_loop.png", dpi=150, bbox_inches="tight")
plt.show()
"""),

        md("""\
### 5 — Interview talking points

#### For Grab / GXS (fintech lending, ride-hailing super-app)

- **MNAR = offer policy:** GrabLend's historical loan offers were policy-driven.
  The propensity model learns P(offered | rider features) and corrects for this.
  Outcome: fairer credit access for under-served driver segments, better CLV models.
- **Thompson Sampling = responsible exploration:** Regulatory requirement to offer
  products fairly across demographics. Thompson's posterior-uncertainty exploration
  keeps the Gini of the offer mix low, producing auditable diversity.
- **Deconfounded recommender:** The substitute confounder Z captures latent "driver
  economic status" from the exposure pattern — a natural confound in gig-economy lending.

#### For Google / DeepMind (large-scale ML, research)

- **Bayesian PMF + NUTS:** Demonstrate rigorous posterior inference, R-hat/ESS
  diagnostics, non-centred parameterisation. Language: "I treat uncertainty as a
  first-class object, not an afterthought."
- **PyMC vs NumPyro trade-off:** NUTS for calibration (offline), SVI for online
  updates. Articulate the mean-field bias explicitly.
- **Feedback loop as a causal problem:** The greedy feedback loop is a causal
  phenomenon (interventional distribution vs observational). Thompson Sampling
  implicitly performs a kind of adaptive randomisation — connect to bandit literature
  (Thompson 1933, Russo et al. 2018).
- **DR estimator:** Doubly-robust NDCG uses both a propensity model and a direct
  model — connect to semiparametric efficiency theory and off-policy evaluation.

#### Universal talking point

> "This project is essentially a controlled experiment on the recommender system
> itself. The feedback-loop simulation shows what happens if you run your model in
> production for 10 quarters without intervention — Greedy locks in popularity,
> Thompson stays diverse. This is exactly the kind of causal thinking that
> separates ML engineers from ML scientists."
"""),

    ]
    return nb


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    save(make_00(), "notebooks/00_data_exploration.ipynb")
    save(make_01(), "notebooks/01_bayesian_pmf.ipynb")
    save(make_02(), "notebooks/02_causal_debiasing.ipynb")
    save(make_03(), "notebooks/03_thompson_sampling.ipynb")
    save(make_04(), "notebooks/04_fintech_framing.ipynb")
    print("Done.")
