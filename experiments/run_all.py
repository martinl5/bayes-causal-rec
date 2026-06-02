"""Reproduce all reported results from a fixed seed.

Regenerates the Phase 2 model-comparison table, the propensity ECE, the
doubly-robust NDCG, and the Phase 3 feedback-loop simulation, then writes
the numbers to ``outputs/phase2_results.txt``, ``outputs/phase3_results.txt``,
and ``outputs/phase3_results.json``.

Run:  python experiments/run_all.py
On CPU this takes roughly 30-45 minutes (NUTS dominates).  Use a smaller
``--users`` for a faster smoke run.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from bcr.data.preprocess import make_synthetic_mnar
from bcr.evaluation.calibration import expected_calibration_error, plot_calibration_curve
from bcr.evaluation.metrics import (
    doubly_robust_ndcg,
    ndcg_at_k,
    recall_at_k,
    rmse_on_test,
)
from bcr.models.bayesian_pmf import BayesianPMF, IPSBayesianPMF
from bcr.models.deconfounder import DeconfoundedRecommender
from bcr.models.propensity import BayesianPropensityModel
from bcr.models.thompson_sampler import FeedbackLoopSimulator

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce all results.")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--items", type=int, default=200)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    args = parser.parse_args()

    Path("outputs/figures").mkdir(parents=True, exist_ok=True)

    data = make_synthetic_mnar(
        n_users=args.users, n_items=args.items, n_factors=10, random_seed=SEED
    )
    train_r, train_m = data["train_ratings"], data["train_mask"]
    test_r, test_m = data["test_ratings"], data["test_mask"]
    true_r = data["true_ratings"]

    print("=" * 60)
    print(f"Reproduce — synthetic MNAR ({args.users}u x {args.items}i)")
    print("=" * 60)

    # ── Propensity model + calibration ────────────────────────────────
    print("\n[1] Bayesian propensity model ...")
    prop = BayesianPropensityModel(random_seed=SEED)
    prop.build_model(train_m)
    prop.fit(draws=args.draws, tune=args.tune, chains=2)
    propensities = prop.propensity_scores()
    ece = expected_calibration_error(propensities.flatten(), train_m.flatten().astype(float))
    plot_calibration_curve(
        propensities.flatten(),
        train_m.flatten().astype(float),
        "outputs/figures/calibration.png",
    )
    print(f"    ECE = {ece:.4f}")

    results: dict[str, dict] = {}

    # ── Naive PMF ─────────────────────────────────────────────────────
    print("\n[2] Naive Bayesian PMF ...")
    naive = BayesianPMF(n_factors=10, random_seed=SEED)
    naive.build_model(train_r, train_m)
    naive.fit(draws=args.draws, tune=args.tune, chains=2)
    results["Naive PMF"] = {
        "ndcg": ndcg_at_k(naive, test_r, test_m),
        "recall": recall_at_k(naive, test_r, test_m),
        "rmse": rmse_on_test(naive, test_r, test_m),
    }

    # ── IPS-PMF ───────────────────────────────────────────────────────
    print("\n[3] IPS-PMF ...")
    ips = IPSBayesianPMF(n_factors=10, random_seed=SEED)
    ips.build_model(train_r, train_m, propensities, clip_max=5.0)
    ips.fit(draws=args.draws, tune=args.tune, chains=2, target_accept=0.95)
    results["IPS-PMF"] = {
        "ndcg": ndcg_at_k(ips, test_r, test_m),
        "recall": recall_at_k(ips, test_r, test_m),
        "rmse": rmse_on_test(ips, test_r, test_m),
    }

    # ── Doubly-robust NDCG (uses naive predictions + propensities) ────
    dr_ndcg = doubly_robust_ndcg(naive._score_matrix(), propensities, test_r, test_m)

    # ── Deconfounded recommender ──────────────────────────────────────
    print("\n[4] Deconfounded recommender ...")
    dec = DeconfoundedRecommender(n_factors=10, n_z_factors=5, random_seed=SEED)
    Z = dec.fit_factor_model(train_m)
    dec.fit_outcome_model(train_r, train_m, Z)

    # Deconfounded metrics via a thin adapter exposing _score_matrix
    class _Adapter:
        def __init__(self, m):
            self._m = m

        def _score_matrix(self):
            return self._m._score_matrix()

    dec_ad = _Adapter(dec)
    results["Deconfounded"] = {
        "ndcg": ndcg_at_k(dec_ad, test_r, test_m),
        "recall": recall_at_k(dec_ad, test_r, test_m),
        "rmse": rmse_on_test(dec_ad, test_r, test_m),
    }

    # ── Write Phase 2 results ─────────────────────────────────────────
    lines = [
        f"Phase 2 Results — Synthetic MNAR ({args.users} users x {args.items} items, "
        f"{args.draws} draws x 2 chains)",
        "Model comparison on unbiased test set:",
    ]
    for name, r in results.items():
        lines.append(
            f"  {name:<13}: NDCG@10={r['ndcg']:.4f}  "
            f"Recall@10={r['recall']:.4f}  RMSE={r['rmse']:.4f}"
        )
    lines.append(f"  DR-NDCG (eval): {dr_ndcg:.4f}")
    lines.append(f"\nPropensity calibration ECE: {ece:.4f}")
    Path("outputs/phase2_results.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # ── Phase 3 feedback-loop simulation ──────────────────────────────
    print("\n[5] Feedback-loop simulation (T=10) ...")
    sim = FeedbackLoopSimulator(true_r, test_r, test_m, n_rounds=10, random_seed=SEED)
    sim_results: dict[str, dict] = {}
    for strategy in ["thompson", "greedy", "random"]:
        t0 = time.time()
        sim_results[strategy] = sim.run(
            strategy=strategy,
            initial_train_ratings=train_r,
            initial_train_mask=train_m,
            k=10,
            n_svi_steps=800,
            n_factors=10,
        )
        sim_results[strategy]["total_time_s"] = time.time() - t0

    with open("outputs/phase3_results.json", "w") as f:
        json.dump(
            {
                k: {
                    kk: ([float(x) for x in vv] if isinstance(vv, list) else vv)
                    for kk, vv in v.items()
                }
                for k, v in sim_results.items()
            },
            f,
            indent=2,
        )

    p3 = [
        "Phase 3 Results — Thompson Sampling Feedback-Loop Simulation",
        f"Dataset: Synthetic MNAR ({args.users}u x {args.items}i), T=10, k=10",
        "",
        f"{'Strategy':<12} | {'Final NDCG@10':<14} | {'Final Coverage':<15} | {'Final Gini'}",
        "-" * 60,
    ]
    for s in ["thompson", "greedy", "random"]:
        r = sim_results[s]
        p3.append(
            f"{s:<12} | {r['ndcg'][-1]:<14.4f} | {r['coverage'][-1]:<15.3f} | {r['gini'][-1]:.3f}"
        )
    Path("outputs/phase3_results.txt").write_text("\n".join(p3) + "\n")
    print("\n" + "\n".join(p3))

    print("\n=== reproduce complete ===")


if __name__ == "__main__":
    main()
