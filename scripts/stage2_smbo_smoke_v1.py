#!/usr/bin/env python3
"""Stage2 P2: SMBO closed-loop smoke (Phase 1, offline on the 60 known widths).

Goal (per method_zh_ieee / 2_design_cost_model / 3_design_exploring_tvm):
demonstrate that the AutoTVM/ALT-style closed loop
  seed -> surrogate predict (+uncertainty) -> acquisition pick top-K
       -> "measure" (reveal true) -> feedback+retrain -> repeat
(a) improves the surrogate on held-out configs, and
(b) finds better (lower-latency) points faster than random sampling.

Universe = original60's 60 width configs at a fixed precision (fp16 by default),
whose true latency/energy are already measured -> we can reveal ground truth
offline to validate loop logic before spending H800 on NEW widths (Phase 2).

Surrogate: LightGBM log1p value model (the validated v2 approach). Uncertainty:
quantile LGBM (alpha=0.16/0.84) -> sigma_hat=(q84-q16)/2. Acquisition for
minimizing latency: LCB = mu_hat - kappa*sigma_hat (explore low-latency).

Output: cost_model/reports/stage2_smbo_smoke_v1_report_latest.md + metrics json.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.stats import spearmanr
import lightgbm as lgb

FEATURE_ORDER = ["w0", "w1", "w2", "w_sum", "w_prod_norm", "precision_bits",
                 "is_fp16", "is_int8", "fam_frontier", "fam_lhc", "fam_s0",
                 "fam_s1", "fam_s2", "fam_other", "lat_sched_default"]
LGB = dict(n_estimators=300, num_leaves=7, min_child_samples=5, learning_rate=0.05,
           subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
           verbosity=-1, random_state=0, n_jobs=1, num_threads=1)

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fit_predict(Xtr, ytr, Xte):
    """log1p value model + quantile sigma."""
    ylog = np.log1p(ytr)
    mv = lgb.LGBMRegressor(objective="regression", **LGB); mv.fit(Xtr, ylog)
    mu = np.expm1(mv.predict(Xte))
    qlo = lgb.LGBMRegressor(objective="quantile", alpha=0.16, **LGB); qlo.fit(Xtr, ylog)
    qhi = lgb.LGBMRegressor(objective="quantile", alpha=0.84, **LGB); qhi.fit(Xtr, ylog)
    lo = np.expm1(qlo.predict(Xte)); hi = np.expm1(qhi.predict(Xte))
    sigma = np.maximum((hi - lo) / 2.0, 1e-9)
    return mu, sigma

def run_loop(X, y, seed_idx, pool_idx, strategy, kappa, K, rng):
    """One SMBO run. Returns per-round dict: test_mae, test_spearman, best_found."""
    seen = list(seed_idx); pool = list(pool_idx)
    hist = []
    true_best = float(y.min())
    while pool:
        Xtr = X[seen]; ytr = y[seen]
        Xpool = X[pool]; ypool = y[pool]
        if len(set(ytr.tolist())) < 2:
            mu = np.full(len(pool), ytr.mean()); sigma = np.ones(len(pool))
        else:
            mu, sigma = fit_predict(Xtr, ytr, Xpool)
        # test metrics on current pool (held-out)
        tmae = float(np.mean(np.abs(mu - ypool)))
        tsp = float(spearmanr(ypool, mu).correlation) if len(pool) > 2 else float("nan")
        best_found = float(y[seen].min())
        hist.append({"n_seen": len(seen), "test_mae": tmae, "test_spearman": tsp,
                     "best_found": best_found, "regret": best_found - true_best})
        # acquisition: pick K from pool
        if strategy == "random":
            order = rng.permutation(len(pool))
        elif strategy == "lcb":      # minimize latency: lower conf bound
            order = np.argsort(mu - kappa * sigma)
        elif strategy == "uncertainty":  # improve model globally
            order = np.argsort(-sigma)
        else:
            raise ValueError(strategy)
        take = [pool[i] for i in order[:K]]
        for t in take:
            seen.append(t); pool.remove(t)
    # final full-data metric already implicit; append terminal best
    hist.append({"n_seen": len(seen), "test_mae": 0.0, "test_spearman": float("nan"),
                 "best_found": float(y[seen].min()), "regret": float(y[seen].min()) - true_best})
    return hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path,
                    default=Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"))
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int8", "fp32"])
    ap.add_argument("--target", default="latency_ms", choices=["latency_ms", "energy_j"])
    ap.add_argument("--seed-n", type=int, default=10)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    root = args.output_root
    tab = json.load(open(root / "cost_model/train/original60_training_table_latest.json"))
    rows = [r for r in tab["rows"] if r["precision"] == args.precision
            and r.get(f"{args.target}_status") == "measured" and r.get(args.target) is not None]
    X = np.array([[r[f] for f in FEATURE_ORDER] for r in rows], float)
    y = np.array([float(r[args.target]) for r in rows], float)
    n = len(rows)

    strategies = ["lcb", "uncertainty", "random"]
    agg = {s: {"test_mae": [], "test_spearman": [], "regret": []} for s in strategies}
    # align rounds by n_seen; collect curves per repeat
    curves = {s: [] for s in strategies}
    for rep in range(args.repeats):
        rng = np.random.default_rng(1000 + rep)
        seed_idx = rng.choice(n, size=args.seen_n if hasattr(args, "seen_n") else args.seed_n,
                              replace=False).tolist()
        pool_idx = [i for i in range(n) if i not in seed_idx]
        for s in strategies:
            hist = run_loop(X, y, seed_idx, pool_idx, s, args.kappa, args.K, rng)
            curves[s].append(hist)

    # aggregate: report metrics at first round (seed only) and mid + final
    def at_round(hist_list, key, n_seen_target):
        vals = []
        for h in hist_list:
            cand = [d[key] for d in h if d["n_seen"] == n_seen_target]
            if cand and cand[0] == cand[0]:  # not nan
                vals.append(cand[0])
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

    seed_n = args.seed_n
    mid_n = seed_n + args.K * ((n - seed_n) // args.K // 2)
    metrics = {"schema": "stage2_smbo_smoke_v1", "generated_at": utc(),
               "precision": args.precision, "target": args.target, "n_universe": n,
               "seed_n": seed_n, "K": args.K, "kappa": args.kappa, "repeats": args.repeats,
               "rounds": {}}
    for s in strategies:
        metrics["rounds"][s] = {
            "seed_test_mae": at_round(curves[s], "test_mae", seed_n),
            "seed_test_spearman": at_round(curves[s], "test_spearman", seed_n),
            "mid_test_mae": at_round(curves[s], "test_mae", mid_n),
            "mid_test_spearman": at_round(curves[s], "test_spearman", mid_n),
            "seed_regret": at_round(curves[s], "regret", seed_n),
            "mid_regret": at_round(curves[s], "regret", mid_n),
        }
    (root / "cost_model/reports").mkdir(parents=True, exist_ok=True)
    json.dump(metrics, open(root / "cost_model/reports/stage2_smbo_smoke_v1_metrics_latest.json", "w"),
              ensure_ascii=False, indent=1)

    def f(t): return f"{t[0]:.4f}±{t[1]:.4f}"
    L = ["# stage2 SMBO closed-loop smoke v1 (Phase 1, offline)", "",
         f"generated_at: {utc()}",
         f"universe={n} widths @ {args.precision}; target={args.target}; "
         f"seed={seed_n}; K={args.K}; kappa={args.kappa}; repeats={args.repeats}",
         "", "Closed loop: seed -> log1p surrogate + quantile sigma -> acquisition -> reveal true -> retrain.",
         "Compares LCB (find-min), uncertainty-sampling (improve-model), random (baseline).",
         f"Rounds compared: seed(n={seed_n}) vs mid(n={mid_n}).", "",
         "| strategy | test_MAE seed→mid | test_Spearman seed→mid | regret seed→mid |",
         "|---|---|---|---|"]
    for s in strategies:
        r = metrics["rounds"][s]
        L.append(f"| {s} | {f(r['seed_test_mae'])} → {f(r['mid_test_mae'])} | "
                 f"{f(r['seed_test_spearman'])} → {f(r['mid_test_spearman'])} | "
                 f"{f(r['seed_regret'])} → {f(r['mid_regret'])} |")
    L += ["", "Interpretation:",
          "- test_MAE/Spearman improving seed→mid = closed loop calibrates the surrogate.",
          "- regret→0 faster for LCB vs random = acquisition finds better points faster.",
          "- Phase 2 (real): let surrogate propose NEW widths (outside original60), measure on H800, feed back."]
    (root / "cost_model/reports/stage2_smbo_smoke_v1_report_latest.md").write_text("\n".join(L))
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
