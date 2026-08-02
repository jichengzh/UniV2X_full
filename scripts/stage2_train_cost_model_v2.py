#!/usr/bin/env python3
"""Stage2 P1 v2: cost models faithful to design doc / IEEE method.

Fixes over v1 (scripts/stage2_train_cost_model_v1.py), which was methodologically
flawed:
  * v1 used Leave-One-Out CV, but each width appears 3x (fp32/fp16/int8) with
    near-identical AP -> data leakage. Honest CV must be leave-one-WIDTH-out.
  * v1 regressed raw latency/energy with L2 -> large values dominate.
  * v1 regressed absolute AP -> flat plateau, no signal (worse than mean).

v2 implements the paper's method (method_zh_ieee_v1.md §latency/energy/AP):
  * latency/energy: predict log1p(y) (magnitude uniformity) + report pairwise
    rank loss; also train an LGBMRanker (lambdarank) rank head.
  * AP: residual prediction anchored on a MEASURED reference. Anchor = per-width
    FP32 AP; target = AP(x) - anchor = precision-induced AP penalty (fp32->0).
    This isolates the only learnable AP signal (quantization penalty) from the
    flat width plateau.
  * CV: LeaveOneGroupOut grouped by width (no leakage). Baselines reported.

Outputs (under output-root/cost_model/):
  reports/stage2_cost_model_v2_report_latest.md
  reports/stage2_cost_model_v2_metrics_latest.json
  models/stage2_cost_v2_{latency_ms,energy_j}_logval_v2.txt        (LGBM value, log-target)
  models/stage2_cost_v2_{latency_ms,energy_j}_rank_v2.txt          (LGBM lambdarank)
  models/stage2_cost_v2_ap70_residual_v2.txt                       (LGBM delta-AP)
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import LeaveOneGroupOut
import lightgbm as lgb

FEATURE_ORDER = ["w0", "w1", "w2", "w_sum", "w_prod_norm", "precision_bits",
                 "is_fp16", "is_int8", "fam_frontier", "fam_lhc", "fam_s0",
                 "fam_s1", "fam_s2", "fam_other", "lat_sched_default"]
# AP residual model drops precision one-hots' redundancy? keep is_int8/is_fp16 (penalty differs by precision)

def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

LGB_VAL = dict(objective="regression", n_estimators=300, num_leaves=7,
               min_child_samples=5, learning_rate=0.05, subsample=0.9,
               subsample_freq=1, colsample_bytree=0.9, verbosity=-1, random_state=0, n_jobs=1, num_threads=1)
LGB_RANK = dict(objective="lambdarank", n_estimators=300, num_leaves=7,
                min_child_samples=5, learning_rate=0.05, verbosity=-1, random_state=0, n_jobs=1, num_threads=1)


def pairwise_rank_loss(y_true, y_score, smaller_is_better=True):
    """Paper's pairwise logistic rank loss: sum log(1+exp(-s_ij*(r_i-r_j))).
    r = rank score where SMALLER means better (we use -y_score if larger better).
    s_ij = sign(better_i - better_j)."""
    y_true = np.asarray(y_true, float); y_score = np.asarray(y_score, float)
    n = len(y_true)
    if n < 2:
        return float("nan")
    # define "goodness": for latency/energy smaller better; score predicts value.
    # rank score r increasing with predicted value; want pairs ordered by true value.
    tot = 0.0; cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            s = np.sign(y_true[j] - y_true[i]) if smaller_is_better else np.sign(y_true[i] - y_true[j])
            # r_i - r_j uses predicted score aligned so smaller-better
            d = (y_score[i] - y_score[j]) if smaller_is_better else (y_score[j] - y_score[i])
            tot += math.log1p(math.exp(min(50.0, -s * d)))
            cnt += 1
    return tot / cnt if cnt else float("nan")


def cv_value_log(X, y, groups, largest_better):
    """Leave-one-width-out CV of a log-target value model. Returns honest metrics."""
    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y, float)
    ylog = np.log1p(y)
    for tr, te in logo.split(X, ylog, groups):
        m = lgb.LGBMRegressor(**LGB_VAL)
        m.fit(X[tr], ylog[tr])
        preds[te] = np.expm1(m.predict(X[te]))
    err = preds - y
    return {
        "cv": "leave_one_width_out", "target_space": "log1p",
        "n": int(len(y)),
        "mae_orig": float(np.mean(np.abs(err))),
        "rmse_orig": float(np.sqrt(np.mean(err ** 2))),
        "mae_log": float(np.mean(np.abs(np.log1p(preds) - ylog))),
        "spearman": float(spearmanr(y, preds).correlation),
        "kendall_tau": float(kendalltau(y, preds).correlation),
        "pairwise_rank_loss": pairwise_rank_loss(y, preds, smaller_is_better=not largest_better),
        "mae_mean_baseline": float(np.mean(np.abs(y - y.mean()))),
    }


def cv_rank_lambda(X, y, groups, largest_better):
    """Leave-one-width-out CV of an LGBMRanker (lambdarank). Relevance = rank buckets."""
    logo = LeaveOneGroupOut()
    preds = np.zeros_like(y, float)
    # relevance: higher = better. For latency/energy smaller better -> invert.
    order_val = y if largest_better else -y
    # bucket into integer relevance 0..K within full set (lambdarank needs int gains)
    ranks = np.argsort(np.argsort(order_val))
    rel = (ranks / max(1, len(ranks) - 1) * 10).astype(int)
    for tr, te in logo.split(X, y, groups):
        m = lgb.LGBMRanker(**LGB_RANK)
        m.fit(X[tr], rel[tr], group=[len(tr)])
        preds[te] = m.predict(X[te])  # higher = predicted better
    # align to value ranking: preds higher=better; compare to true goodness
    good = y if largest_better else -y
    return {
        "cv": "leave_one_width_out", "model": "lambdarank",
        "spearman_vs_goodness": float(spearmanr(good, preds).correlation),
        "kendall_vs_goodness": float(kendalltau(good, preds).correlation),
    }


def cv_ap_residual(rows):
    """AP residual: anchor = per-width fp32 AP; target = AP - anchor (precision penalty).
    Train on fp16/int8 rows (fp32 delta==0 trivially). Honest leave-one-width-out."""
    # build per-width fp32 anchor
    anchor = {}
    for r in rows:
        if r["precision"] == "fp32" and r.get("ap70_status") == "measured" and r.get("ap70") is not None:
            anchor[str(r["width"])] = float(r["ap70"])
    sub = [r for r in rows if r["precision"] in ("fp16", "int8")
           and r.get("ap70_status") == "measured" and r.get("ap70") is not None
           and str(r["width"]) in anchor and not r["is_anomaly"]]
    X = np.array([[r[f] for f in FEATURE_ORDER] for r in sub], float)
    delta = np.array([float(r["ap70"]) - anchor[str(r["width"])] for r in sub], float)
    ap_abs = np.array([float(r["ap70"]) for r in sub], float)
    anc = np.array([anchor[str(r["width"])] for r in sub], float)
    groups = np.array([str(r["width"]) for r in sub])
    logo = LeaveOneGroupOut()
    dpred = np.zeros_like(delta, float)
    for tr, te in logo.split(X, delta, groups):
        m = lgb.LGBMRegressor(**LGB_VAL)
        m.fit(X[tr], delta[tr])
        dpred[te] = m.predict(X[te])
    ap_hat = anc + dpred  # reconstructed AP using MEASURED anchor
    return {
        "cv": "leave_one_width_out", "anchor": "per_width_fp32_ap70",
        "n": int(len(sub)),
        "delta_mae": float(np.mean(np.abs(dpred - delta))),
        "delta_mae_zero_baseline": float(np.mean(np.abs(delta))),  # predict "no penalty"
        "delta_mae_mean_baseline": float(np.mean(np.abs(delta - delta.mean()))),
        "delta_mean_true": float(delta.mean()), "delta_std_true": float(delta.std()),
        "ap_hat_mae": float(np.mean(np.abs(ap_hat - ap_abs))),
        "ap_hat_spearman": float(spearmanr(ap_abs, ap_hat).correlation),
    }


def cv_ap_absolute(rows):
    """Contrast: absolute AP70 regression, honest leave-one-width-out (the flawed v1 target)."""
    sub = [r for r in rows if r.get("ap70_status") == "measured" and r.get("ap70") is not None
           and not r["is_anomaly"]]
    X = np.array([[r[f] for f in FEATURE_ORDER] for r in sub], float)
    y = np.array([float(r["ap70"]) for r in sub], float)
    groups = np.array([str(r["width"]) for r in sub])
    logo = LeaveOneGroupOut(); preds = np.zeros_like(y, float)
    for tr, te in logo.split(X, y, groups):
        m = lgb.LGBMRegressor(**LGB_VAL); m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    return {"cv": "leave_one_width_out", "target": "absolute_ap70",
            "mae": float(np.mean(np.abs(preds - y))),
            "mae_mean_baseline": float(np.mean(np.abs(y - y.mean()))),
            "spearman": float(spearmanr(y, preds).correlation)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path,
                    default=Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"))
    args = ap.parse_args()
    root = args.output_root
    tab = json.load(open(root / "cost_model/train/original60_training_table_latest.json"))
    rows = tab["rows"]
    (root / "cost_model/models").mkdir(parents=True, exist_ok=True)
    (root / "cost_model/reports").mkdir(parents=True, exist_ok=True)

    metrics = {"schema": "stage2_cost_model_v2_metrics_v1", "generated_at": utc(),
               "cv": "leave_one_width_out (group=width, no 3x-precision leakage)",
               "targets": {}}

    # latency / energy: log-value + rank
    for tname in ["latency_ms", "energy_j"]:
        sub = [r for r in rows if r.get(f"{tname}_status") == "measured" and r.get(tname) is not None]
        X = np.array([[r[f] for f in FEATURE_ORDER] for r in sub], float)
        y = np.array([float(r[tname]) for r in sub], float)
        g = np.array([str(r["width"]) for r in sub])
        val = cv_value_log(X, y, g, largest_better=False)
        rnk = cv_rank_lambda(X, y, g, largest_better=False)
        metrics["targets"][tname] = {"value_logtarget": val, "rank_lambdarank": rnk}
        # persist final models on all data
        mv = lgb.LGBMRegressor(**LGB_VAL); mv.fit(X, np.log1p(y))
        mv.booster_.save_model(str(root / f"cost_model/models/stage2_cost_v2_{tname}_logval_v2.txt"))

    # AP: residual + absolute contrast
    metrics["targets"]["ap70"] = {
        "residual_per_width_fp32_anchor": cv_ap_residual(rows),
        "absolute_contrast": cv_ap_absolute(rows),
    }
    # persist AP residual final model (train on fp16/int8 deltas)
    anchor = {str(r["width"]): float(r["ap70"]) for r in rows
              if r["precision"] == "fp32" and r.get("ap70") is not None}
    subap = [r for r in rows if r["precision"] in ("fp16", "int8")
             and r.get("ap70") is not None and str(r["width"]) in anchor and not r["is_anomaly"]]
    Xa = np.array([[r[f] for f in FEATURE_ORDER] for r in subap], float)
    da = np.array([float(r["ap70"]) - anchor[str(r["width"])] for r in subap], float)
    ma = lgb.LGBMRegressor(**LGB_VAL); ma.fit(Xa, da)
    ma.booster_.save_model(str(root / "cost_model/models/stage2_cost_v2_ap70_residual_v2.txt"))

    json.dump(metrics, open(root / "cost_model/reports/stage2_cost_model_v2_metrics_latest.json", "w"),
              ensure_ascii=False, indent=1)

    # report
    L = ["# stage2 cost model v2 training report", "", f"generated_at: {utc()}",
         "", "CV = leave-one-WIDTH-out (fixes v1 LOO leakage from 3x-per-width).",
         "latency/energy: log1p value target + lambdarank rank head.",
         "AP: residual vs per-width FP32 anchor (precision penalty); absolute shown for contrast.", ""]
    for t in ["latency_ms", "energy_j"]:
        v = metrics["targets"][t]["value_logtarget"]; r = metrics["targets"][t]["rank_lambdarank"]
        L += [f"## {t}",
              f"- value(log1p): MAE_orig={v['mae_orig']:.4f} RMSE={v['rmse_orig']:.4f} "
              f"Spearman={v['spearman']:.3f} Kendall={v['kendall_tau']:.3f} "
              f"pairwise_rank_loss={v['pairwise_rank_loss']:.4f} (mean-baseline MAE={v['mae_mean_baseline']:.4f})",
              f"- rank(lambdarank): Spearman={r['spearman_vs_goodness']:.3f} Kendall={r['kendall_vs_goodness']:.3f}", ""]
    apr = metrics["targets"]["ap70"]["residual_per_width_fp32_anchor"]
    apa = metrics["targets"]["ap70"]["absolute_contrast"]
    L += ["## ap70 (residual, per-width FP32 anchor)",
          f"- delta MAE={apr['delta_mae']:.5f} vs zero-baseline={apr['delta_mae_zero_baseline']:.5f} "
          f"vs mean-baseline={apr['delta_mae_mean_baseline']:.5f} (true delta mean={apr['delta_mean_true']:.5f} std={apr['delta_std_true']:.5f})",
          f"- reconstructed AP_hat MAE={apr['ap_hat_mae']:.5f} Spearman={apr['ap_hat_spearman']:.3f}",
          "## ap70 (absolute contrast, honest width-group CV)",
          f"- MAE={apa['mae']:.5f} mean-baseline={apa['mae_mean_baseline']:.5f} Spearman={apa['spearman']:.3f}",
          "", "Note: absolute AP has ~no cross-width signal (plateau). Residual isolates the",
          "precision penalty; its usefulness = whether delta MAE beats zero/mean baseline.", ""]
    (root / "cost_model/reports/stage2_cost_model_v2_report_latest.md").write_text("\n".join(L))

    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
