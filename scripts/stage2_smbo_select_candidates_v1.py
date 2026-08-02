#!/usr/bin/env python3
"""Stage2 P2 real loop -- step 1: candidate generation + surrogate predict + acquisition.

Full search space = 7x7x7 = 343 backbone width grid:
  w0 in {16,24,32,40,48,56,64}, w1 in {32,48,64,80,96,112,128},
  w2 in {64,96,128,160,192,224,256}.
original60 measured 60 of them -> 283 UNMEASURED candidates = the real space.

Trains log1p latency surrogate (+ quantile sigma) on the 60 measured fp32 points,
predicts the 283 candidates, and selects top-K by acquisition for REAL H800 TVM
measurement (step 2). Emits a candidate-queue JSONL consumable by
scripts/stage2_original60_export_onnx.py.
"""
from __future__ import annotations
import argparse, json, itertools
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import lightgbm as lgb

W0 = [16, 24, 32, 40, 48, 56, 64]
W1 = [32, 48, 64, 80, 96, 112, 128]
W2 = [64, 96, 128, 160, 192, 224, 256]
FAMS = ["frontier", "lhc", "s0", "s1", "s2", "other"]
LGB = dict(n_estimators=300, num_leaves=7, min_child_samples=5, learning_rate=0.05,
           subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
           verbosity=-1, random_state=0, n_jobs=1, num_threads=1)

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def feats(w, precision="fp32"):
    fam = "other"  # new candidates have no family label
    d = {"w0": w[0], "w1": w[1], "w2": w[2], "w_sum": sum(w),
         "w_prod_norm": (w[0]*w[1]*w[2])/1e5,
         "precision_bits": {"fp32": 32.0, "fp16": 16.0, "int8": 8.0}[precision],
         "is_fp16": 1.0 if precision == "fp16" else 0.0,
         "is_int8": 1.0 if precision == "int8" else 0.0}
    for f in FAMS:
        d[f"fam_{f}"] = 1.0 if f == fam else 0.0
    d["lat_sched_default"] = 0.0
    return d

FEATURE_ORDER = ["w0", "w1", "w2", "w_sum", "w_prod_norm", "precision_bits",
                 "is_fp16", "is_int8"] + [f"fam_{f}" for f in FAMS] + ["lat_sched_default"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path,
                    default=Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"))
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--target", default="latency_ms")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--strategy", default="mixed", choices=["lcb", "uncertainty", "mixed"])
    ap.add_argument("--queue-out", type=Path,
                    default=Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/jobs/smbo_round1_candidate_queue_v1.jsonl"))
    args = ap.parse_args()
    root = args.output_root

    tab = json.load(open(root / "cost_model/train/original60_training_table_latest.json"))
    measured = {}
    train = []
    for r in tab["rows"]:
        if r["precision"] != args.precision:
            continue
        w = tuple(int(x) for x in (r["width"] if isinstance(r["width"], list)
                                   else str(r["width"]).replace("x", ",").split(",")))
        measured[w] = r
        if r.get(f"{args.target}_status") == "measured" and r.get(args.target) is not None:
            train.append((w, float(r[args.target])))

    Xtr = np.array([[feats(w, args.precision)[f] for f in FEATURE_ORDER] for w, _ in train], float)
    ytr = np.array([v for _, v in train], float)
    ylog = np.log1p(ytr)
    mv = lgb.LGBMRegressor(objective="regression", **LGB); mv.fit(Xtr, ylog)
    qlo = lgb.LGBMRegressor(objective="quantile", alpha=0.16, **LGB); qlo.fit(Xtr, ylog)
    qhi = lgb.LGBMRegressor(objective="quantile", alpha=0.84, **LGB); qhi.fit(Xtr, ylog)

    # full grid minus measured
    cands = [w for w in itertools.product(W0, W1, W2) if w not in measured]
    Xc = np.array([[feats(w, args.precision)[f] for f in FEATURE_ORDER] for w in cands], float)
    mu = np.expm1(mv.predict(Xc))
    lo = np.expm1(qlo.predict(Xc)); hi = np.expm1(qhi.predict(Xc))
    sigma = np.maximum((hi - lo) / 2.0, 1e-9)
    lcb = mu - args.kappa * sigma

    idx_lcb = list(np.argsort(lcb))          # lowest predicted latency (find-min)
    idx_unc = list(np.argsort(-sigma))       # highest uncertainty (improve-model)
    if args.strategy == "lcb":
        pick = idx_lcb[:args.K]
    elif args.strategy == "uncertainty":
        pick = idx_unc[:args.K]
    else:  # mixed: half exploit (lcb) + half explore (uncertainty), dedup
        pick = []
        for a, b in zip(idx_lcb, idx_unc):
            for x in (a, b):
                if x not in pick:
                    pick.append(x)
            if len(pick) >= args.K:
                break
        pick = pick[:args.K]

    selected = []
    for i in pick:
        w = cands[i]
        wsafe = f"{w[0]}x{w[1]}x{w[2]}"
        selected.append({
            "candidate_id": f"smbo_round1:pyramid_lidar:w{wsafe}:{args.precision}",
            "label": f"smbo1_{wsafe}",
            "width": list(w),
            "precision": args.precision,
            "pred_latency_ms": float(mu[i]), "pred_sigma_ms": float(sigma[i]),
            "pred_lcb_ms": float(lcb[i]),
        })

    args.queue_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.queue_out, "w") as fh:
        for s in selected:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    summary = {"schema": "smbo_round1_selection_v1", "generated_at": utc(),
               "space_total": 343, "measured": len(measured),
               "candidates_unmeasured": len(cands), "K": args.K,
               "strategy": args.strategy, "target": args.target, "precision": args.precision,
               "surrogate_train_n": len(train),
               "pred_range_over_candidates": [float(mu.min()), float(mu.max())],
               "selected": selected, "queue_out": str(args.queue_out)}
    json.dump(summary, open(str(args.queue_out).replace(".jsonl", "_summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
