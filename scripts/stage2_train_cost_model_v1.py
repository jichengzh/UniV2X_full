#!/usr/bin/env python3
"""Stage2 P1: train AP/latency/energy cost models on the frozen original60 table.

Data role: original60 three-precision three-metric table (60 widths x 3 precisions
= 180 measured cells) is the cost-model TRAINING/CALIBRATION data, not a direct
Pareto source. See progress/7_2/5_7_2_P0冻结记录.

Anomaly policy (progress/7_2/4_7_2): frontier_01 (width 24x64x128) FP16/INT8 AP
collapse to 0.0 (reduced-precision detection cliff, FP32 normal). AP regressor
trains an exclude-anomaly variant; latency/energy keep all points.

Outputs (under output-root):
  cost_model/train/original60_training_table_latest.{csv,json}
  cost_model/schema/stage2_cost_model_feature_schema_v1.{md,json}
  cost_model/models/stage2_cost_{ap70,latency_ms,energy_j}_{allpts,excl_anom}_v1.joblib
  cost_model/reports/stage2_cost_model_training_report_latest.md
  cost_model/reports/stage2_cost_model_metrics_latest.json
  exports/original60_cost_model_training_freeze_latest.{json,md}
  exports/original60_quant_anomaly_review_20260702.{json,md}
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut

PRECISIONS = ["fp32", "fp16", "int8"]
PRECISION_BITS = {"fp32": 32.0, "fp16": 16.0, "int8": 8.0}
ANOMALY_LABELS = {"frontier_01"}  # reduced-precision AP collapse

# ------------------------------------------------------------------ helpers
def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "MISSING"

def parse_width(w) -> list[float]:
    if isinstance(w, list):
        return [float(x) for x in w]
    return [float(x) for x in str(w).replace("x", ",").split(",")]

def label_family(label: str) -> str:
    for fam in ("frontier", "lhc", "s0", "s1", "s2"):
        if label.startswith(fam):
            return fam
    return "other"

FAMILIES = ["frontier", "lhc", "s0", "s1", "s2", "other"]

# ------------------------------------------------------------------ features
def build_features(row: dict) -> dict:
    w = parse_width(row["width"])
    prec = row["precision"]
    fam = label_family(row["label"])
    feat = {
        "w0": w[0], "w1": w[1], "w2": w[2],
        "w_sum": sum(w), "w_prod_norm": (w[0] * w[1] * w[2]) / 1e5,
        "precision_bits": PRECISION_BITS[prec],
        "is_fp16": 1.0 if prec == "fp16" else 0.0,
        "is_int8": 1.0 if prec == "int8" else 0.0,
    }
    for f in FAMILIES:
        feat[f"fam_{f}"] = 1.0 if fam == f else 0.0
    # latency/energy carry schedule confound (tuned vs default salvage)
    feat["lat_sched_default"] = 1.0 if row.get("latency_schedule_policy") == "default" else 0.0
    return feat

FEATURE_ORDER = (["w0", "w1", "w2", "w_sum", "w_prod_norm", "precision_bits",
                  "is_fp16", "is_int8"] + [f"fam_{f}" for f in FAMILIES] +
                 ["lat_sched_default"])

TARGETS = {
    "ap70": ("ap70", "ap_status"),
    "latency_ms": ("latency_ms", "latency_status"),
    "energy_j": ("energy_j_per_inference", "energy_status"),
}

# ------------------------------------------------------------------ metrics
def eval_loo(X: np.ndarray, y: np.ndarray, seed: int = 0) -> dict:
    """Leave-one-out CV; returns MAE/RMSE + rank metrics (Spearman/Kendall)."""
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.9,
                                      random_state=seed)
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    err = preds - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sp = float(spearmanr(y, preds).correlation) if len(y) > 2 else float("nan")
    kt = float(kendalltau(y, preds).correlation) if len(y) > 2 else float("nan")
    # top-K overlap: does predicted best-K set match true best-K (smaller=better for lat/energy; larger=better for AP handled by caller sign)
    return {"n": int(len(y)), "mae": mae, "rmse": rmse,
            "spearman": sp, "kendall_tau": kt,
            "y_min": float(y.min()), "y_max": float(y.max()),
            "y_mean": float(y.mean())}

def topk_overlap(y: np.ndarray, preds: np.ndarray, k: int, largest: bool) -> float:
    if len(y) < k:
        return float("nan")
    order = np.argsort(y)[::-1] if largest else np.argsort(y)
    porder = np.argsort(preds)[::-1] if largest else np.argsort(preds)
    true_k = set(order[:k].tolist()); pred_k = set(porder[:k].tolist())
    return len(true_k & pred_k) / k

def loo_preds(X, y, seed=0):
    loo = LeaveOneOut(); preds = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.9, random_state=seed)
        m.fit(X[tr], y[tr]); preds[te] = m.predict(X[te])
    return preds

# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path,
                    default=Path("multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"))
    args = ap.parse_args()
    root = args.output_root
    summ = json.load(open(root / "exports/original60_quant_three_metric_summary_latest.json"))
    rows = summ["rows"]

    # ---- training table ----
    table = []
    for r in rows:
        feat = build_features(r)
        rec = {"label": r["label"], "precision": r["precision"], "width": r["width"],
               "family": label_family(r["label"]),
               "is_anomaly": r["label"] in ANOMALY_LABELS,
               **feat}
        for tname, (col, stat) in TARGETS.items():
            rec[tname] = r.get(col)
            rec[f"{tname}_status"] = r.get(stat)
        table.append(rec)

    (root / "cost_model/train").mkdir(parents=True, exist_ok=True)
    (root / "cost_model/schema").mkdir(parents=True, exist_ok=True)
    (root / "cost_model/models").mkdir(parents=True, exist_ok=True)
    (root / "cost_model/reports").mkdir(parents=True, exist_ok=True)

    # CSV
    cols = (["label", "precision", "width", "family", "is_anomaly"] + FEATURE_ORDER +
            list(TARGETS.keys()) + [f"{t}_status" for t in TARGETS])
    import csv
    with open(root / "cost_model/train/original60_training_table_latest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for rec in table:
            w.writerow({c: rec.get(c) for c in cols})
    json.dump({"schema": "original60_cost_model_training_table_v1", "generated_at": utc(),
               "n_rows": len(table), "columns": cols, "rows": table},
              open(root / "cost_model/train/original60_training_table_latest.json", "w"),
              ensure_ascii=False, indent=1)

    # ---- feature schema ----
    schema = {"schema": "stage2_cost_model_feature_schema_v1", "generated_at": utc(),
              "feature_order": FEATURE_ORDER,
              "features": {
                  "w0/w1/w2": "backbone stage channel widths (parsed from width triple)",
                  "w_sum/w_prod_norm": "aggregate capacity features",
                  "precision_bits": "32/16/8 numeric precision",
                  "is_fp16/is_int8": "precision one-hot (fp32=baseline)",
                  "fam_*": "label family one-hot (frontier/lhc/s0/s1/s2/other)",
                  "lat_sched_default": "1 if latency measured via default-salvage schedule (tuned-path CUDA failure)",
              },
              "targets": {"ap70": "AP@0.7 IoU (true_eval)",
                          "latency_ms": "backbone latency ms (H800 TVM)",
                          "energy_j": "energy J/inference (threaded_window)"},
              "anomaly_labels": sorted(ANOMALY_LABELS),
              "model": "sklearn.GradientBoostingRegressor(n_estimators=200,max_depth=3,lr=0.05,subsample=0.9)",
              "cv": "LeaveOneOut"}
    json.dump(schema, open(root / "cost_model/schema/stage2_cost_model_feature_schema_v1.json", "w"),
              ensure_ascii=False, indent=1)
    (root / "cost_model/schema/stage2_cost_model_feature_schema_v1.md").write_text(
        "# stage2 cost model feature schema v1\n\n"
        f"generated_at: {utc()}\n\n## features ({len(FEATURE_ORDER)})\n" +
        "\n".join(f"- `{k}`: {v}" for k, v in schema["features"].items()) +
        "\n\n## targets\n" + "\n".join(f"- `{k}`: {v}" for k, v in schema["targets"].items()) +
        f"\n\n## model\n{schema['model']}, CV={schema['cv']}\n"
        f"\n## anomaly labels (AP excl variant)\n{sorted(ANOMALY_LABELS)}\n")

    # ---- train + eval per target, two variants ----
    import joblib
    metrics = {"schema": "stage2_cost_model_metrics_v1", "generated_at": utc(), "targets": {}}
    for tname, (col, stat) in TARGETS.items():
        largest = (tname == "ap70")  # AP: larger better; lat/energy: smaller better
        metrics["targets"][tname] = {}
        for variant in ["allpts", "excl_anom"]:
            sub = [rec for rec in table
                   if rec.get(f"{tname}_status") == "measured"
                   and rec.get(tname) is not None
                   and (variant == "allpts" or not rec["is_anomaly"])]
            X = np.array([[rec[f] for f in FEATURE_ORDER] for rec in sub], dtype=float)
            y = np.array([float(rec[tname]) for rec in sub], dtype=float)
            m = eval_loo(X, y)
            preds = loo_preds(X, y)
            m["topk5_overlap"] = topk_overlap(y, preds, 5, largest)
            m["topk10_overlap"] = topk_overlap(y, preds, 10, largest)
            metrics["targets"][tname][variant] = m
            # fit final model on all rows of this variant, persist
            fm = GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                           learning_rate=0.05, subsample=0.9, random_state=0)
            fm.fit(X, y)
            joblib.dump({"model": fm, "feature_order": FEATURE_ORDER, "target": tname,
                         "variant": variant, "n_train": len(sub)},
                        root / f"cost_model/models/stage2_cost_{tname}_{variant}_v1.joblib")
    json.dump(metrics, open(root / "cost_model/reports/stage2_cost_model_metrics_latest.json", "w"),
              ensure_ascii=False, indent=1)

    # ---- training report ----
    def fmt(m):
        return (f"n={m['n']} MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} "
                f"Spearman={m['spearman']:.3f} Kendall={m['kendall_tau']:.3f} "
                f"top5={m['topk5_overlap']:.2f} top10={m['topk10_overlap']:.2f} "
                f"(y {m['y_min']:.3f}~{m['y_max']:.3f})")
    lines = ["# stage2 cost model training report (P1)", "", f"generated_at: {utc()}",
             "", "data: original60 frozen 180 cells; model GBM; CV LeaveOneOut.",
             "AP uses AP70. AP excl_anom drops frontier_01 (reduced-precision cliff).", ""]
    for tname in TARGETS:
        lines.append(f"## {tname}")
        for variant in ["allpts", "excl_anom"]:
            lines.append(f"- **{variant}**: {fmt(metrics['targets'][tname][variant])}")
        lines.append("")
    (root / "cost_model/reports/stage2_cost_model_training_report_latest.md").write_text("\n".join(lines))

    # ---- P0 machine-readable freeze + anomaly review ----
    freeze_files = ["exports/original60_quant_three_metric_summary_latest.json",
                    "exports/original60_quant_three_metric_summary_latest.csv",
                    "rows/native_int8_original60_ap_rows_v1.jsonl",
                    "rows/fp16_true_original60_ap_rows_v1.jsonl",
                    "rows/fp32_true_original60_ap_rows_v1.jsonl",
                    "rows/fp16_rewritten_tensorcore_full60_latency_rows_v1.jsonl",
                    "rows/fp32_original60_energy_threaded60_rows_v1.jsonl",
                    "rows/fp32_latency_gap2_default_salvage_rows_20260702.jsonl"]
    cov = {p: {"latency": sum(1 for r in rows if r["precision"] == p and r["latency_status"] == "measured"),
               "ap": sum(1 for r in rows if r["precision"] == p and r["ap_status"] == "measured"),
               "energy": sum(1 for r in rows if r["precision"] == p and r["energy_status"] == "measured")}
           for p in PRECISIONS}
    freeze = {"schema": "original60_cost_model_training_freeze_v1", "frozen_at": utc(),
              "role": "cost_model_training_calibration_data",
              "coverage": cov, "total_cells": len(rows),
              "anomaly_labels": sorted(ANOMALY_LABELS),
              "salvage_labels": ["lhc_17", "s2_096"],
              "manifest": {f: {"sha256_16": sha16(root / f)} for f in freeze_files}}
    json.dump(freeze, open(root / "exports/original60_cost_model_training_freeze_latest.json", "w"),
              ensure_ascii=False, indent=1)
    (root / "exports/original60_cost_model_training_freeze_latest.md").write_text(
        "# original60 cost_model training freeze (latest)\n\n"
        f"frozen_at: {freeze['frozen_at']}\nrole: {freeze['role']}\ntotal_cells: {len(rows)}\n\n"
        "## coverage\n" + "\n".join(f"- {p}: latency {cov[p]['latency']}/60 ap {cov[p]['ap']}/60 energy {cov[p]['energy']}/60" for p in PRECISIONS) +
        f"\n\n## anomaly: {sorted(ANOMALY_LABELS)}\n## salvage(default): ['lhc_17','s2_096']\n\n"
        "## manifest (sha256_16)\n" + "\n".join(f"- `{f}`: {sha16(root/f)}" for f in freeze_files) + "\n")

    # anomaly review
    def apof(label, prec):
        for r in rows:
            if r["label"] == label and r["precision"] == prec:
                return r.get("ap70")
        return None
    review = {"schema": "original60_quant_anomaly_review_v1", "reviewed_at": utc(),
              "items": [
                  {"label": "frontier_01", "verdict": "blocked_suspect_anomaly",
                   "ap70": {p: apof("frontier_01", p) for p in PRECISIONS},
                   "reason": "FP16/INT8 AP collapse to 0.0 (reduced-precision detection cliff at width 24x64x128); FP32 normal; INT8 full-val 1789 samples empty predictions. Excluded from AP regressor training; retained as cliff evidence.",
                   "policy": "AP regressor excl_anom variant drops it; latency/energy keep it."},
                  {"label": "frontier_31", "verdict": "rechecked_normal",
                   "ap70": {p: apof("frontier_31", p) for p in PRECISIONS},
                   "reason": "Recheck 20260702 confirmed INT8 AP70=0.5642 consistent with FP32 0.575 / FP16 0.574. Not an anomaly.",
                   "policy": "Kept in all training variants."},
              ]}
    json.dump(review, open(root / "exports/original60_quant_anomaly_review_20260702.json", "w"),
              ensure_ascii=False, indent=1)
    (root / "exports/original60_quant_anomaly_review_20260702.md").write_text(
        "# original60 quant anomaly review (2026-07-02)\n\n" +
        "\n\n".join(f"## {it['label']} — {it['verdict']}\n- ap70: {it['ap70']}\n- {it['reason']}\n- policy: {it['policy']}"
                    for it in review["items"]) + "\n")

    print(json.dumps({"status": "ok", "n_rows": len(table),
                      "metrics_summary": {t: {v: {"mae": metrics["targets"][t][v]["mae"],
                                                  "spearman": metrics["targets"][t][v]["spearman"]}
                                              for v in ["allpts", "excl_anom"]} for t in TARGETS}},
                     ensure_ascii=False, indent=1))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
