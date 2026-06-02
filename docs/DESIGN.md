# Design Notes

This document records the methodology and the design decisions behind the
project. It is the "why" companion to the README's "what" and "how".

## Goal

Build a recommender that does three things most production collaborative
filters do not:

1. **Quantify uncertainty** over user/item preferences with a full Bayesian
   posterior, rather than a single point estimate.
2. **Correct for MNAR exposure bias** — the fact that the ratings we observe
   are a biased sample driven by what the system chose to show.
3. **Explore responsibly** — use the posterior directly (Thompson Sampling) to
   avoid the popularity-amplifying feedback loop that greedy point-estimate
   recommenders create.

The work is organised in four stages, each building on the last.

## Stage 1 — Bayesian Probabilistic Matrix Factorization

A PMF model in PyMC with a non-centred parameterisation for the latent factors
and **global-mean + user/item bias intercepts**. The biases matter: without
them the zero-mean factor priors pull predictions toward 0 while ratings live
in [1, 5], which inflates RMSE to roughly the mean rating even when the ranking
is fine. The intercepts anchor predictions to the rating scale (standard PMF
practice, e.g. Koren et al. 2009).

Inference is NUTS. We check R-hat and ESS and surface convergence warnings
rather than suppressing them; divergences are reported honestly.

## Stage 2 — Causal debiasing

- **Propensity model.** A hierarchical logistic regression on the observation
  mask, `logit P(O_ui = 1) = alpha_u + beta_i`, gives a calibrated estimate of
  how likely each (user, item) pair was to be observed. Calibration (ECE +
  reliability diagram) is checked *before* the propensities are used — an
  uncalibrated propensity model produces IPS weights that add variance instead
  of removing bias.
- **IPS-PMF.** Inverse-propensity weighting via the observation precision, with
  the weights clipped and then self-normalised to mean 1 (SNIPS-style) so the
  reweighting changes *relative* importance without inflating overall noise.
- **Doubly-robust NDCG.** Estimates the NDCG of the direct model's ranking using
  DR-corrected relevance labels. The ranking and the gain deliberately come from
  different quantities so the estimator is informative rather than degenerate.
  On synthetic data with tiny propensities the IPS residual is high-variance; a
  trustworthy DR number needs a real randomised split (Coat) or cross-fitting.
- **Deconfounded recommender** (Wang, Liang, Charlin & Blei 2018). A two-stage
  substitute-confounder approach: a factor model on the exposure matrix yields a
  per-user latent `Z`, and the outcome model conditions on `Z` for approximate
  backdoor adjustment.

  **Honesty about identifiability.** The substitute-confounder strategy has been
  critiqued (Ogburn et al.; D'Amour): in general it is *not* identified, and a
  factor model is not a substitute for a real randomised experiment. We treat any
  improvement as predictive performance on an unbiased test split, not as proof of
  causal identification, and we always evaluate on the randomised test set.

## Stage 3 — Thompson Sampling and the feedback loop

Thompson Sampling draws one sample from the posterior at recommendation time and
ranks by the sampled scores. High-uncertainty (often long-tail) items
occasionally surface, giving targeted exploration with no exploration
hyperparameter.

The feedback-loop simulator runs T rounds of recommend → observe → refit and
tracks NDCG, catalogue coverage, and the Gini coefficient of the recommendation
distribution. The honest reading of the result is a **trade-off**, not a single
winner: Greedy concentrates (low coverage, high Gini), Random is the exploration
upper bound, and Thompson is the Pareto-attractive point — near-Random coverage
with far better ranking than Greedy. Per-round NDCG on a small synthetic test set
is noisy, so we lead with the robust coverage/Gini separation, not a noisy NDCG
ranking.

**PyMC vs NumPyro.** Full NUTS per round is too slow for a 10-round, 3-strategy
simulation, so the loop uses NumPyro SVI (mean-field). Mean-field underestimates
posterior variance, which makes SVI-based Thompson explore *less* than exact NUTS
would — a trade-off we document rather than hide.

## Stage 4 — Fintech framing

A translation of every component into banking terms (next-best-action, offer
selection bias, offer-policy propensities, fairness via the offer-distribution
Gini), with a worked synthetic example.

## Evaluation discipline

All accuracy numbers are reported on the **unbiased test split only** — Coat's
uniform-random held-out ratings, or the uniform-random test sample of the
synthetic generator. Training-set metrics are never reported as accuracy.

## Reproducing the numbers

`python experiments/run_all.py` regenerates the Phase 2 table, the propensity
ECE, the DR-NDCG, and the Phase 3 simulation from seed 42 and writes them to
`outputs/`. On CPU this takes roughly 30-45 minutes (NUTS dominates).

## Key references

- Wang, Liang, Charlin & Blei (2018), *The Deconfounded Recommender*.
- Schnabel et al. (2016), *Recommendations as Treatments*.
- Kawale et al. (2015), *Efficient Thompson Sampling for Online Matrix-Factorization*.
- Koren, Bell & Volinsky (2009), *Matrix Factorization Techniques for Recommender Systems* (bias terms).
- McElreath (2020), *Statistical Rethinking*.
- Ogburn, Shpitser & Tchetgen Tchetgen (2022), critique of the deconfounder's identifiability.
