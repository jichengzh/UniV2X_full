"""Custom tracking eval that bypasses AMOTA min_recall=0.1 truncation.

Reports the *real* per-class recall / precision based on direct spatial matching
between predictions in `results_nusc.json` and GT in `sample_annotation.json`,
applying nuScenes-tracking class_range filter (50m for car etc.) but no AMOTA
threshold.

Usage:
    python tools/eval_real_recall.py \\
        --pred /path/to/results_nusc.json \\
        --ann-root datasets/V2X-Seq-SPD-New/cooperative \\
        --info data/infos/V2X-Seq-SPD-New/cooperative/spd_infos_temporal_val.pkl \\
        [--score-min 0.0] [--dist-thresh 2.0] [--out report.json]
"""
import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


# nuScenes tracking eligible classes (subset of detection classes)
TRACK_CLASSES = ["bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"]

# Default class_range from coop_tiny config (matches nuScenes tracking eval)
CLASS_RANGE_M = {
    "car": 50.0, "truck": 50.0, "bus": 50.0, "trailer": 50.0,
    "construction_vehicle": 50.0,
    "pedestrian": 40.0, "motorcycle": 40.0, "bicycle": 40.0,
    "traffic_cone": 30.0, "barrier": 30.0,
}


def load_categories(ann_root: Path):
    cat_json = ann_root / "v1.0-trainval" / "category.json"
    inst_json = ann_root / "v1.0-trainval" / "instance.json"
    with open(cat_json) as f:
        cats = json.load(f)
    with open(inst_json) as f:
        insts = json.load(f)
    cat_map = {c["token"]: c["name"] for c in cats}
    inst_to_cat = {i["token"]: cat_map.get(i.get("category_token"), "?") for i in insts}
    return inst_to_cat


def load_annos_for_samples(ann_root: Path, sample_tokens):
    """Return dict[sample_token][class_name] -> list of {translation, instance_token}."""
    anno_json = ann_root / "v1.0-trainval" / "sample_annotation.json"
    with open(anno_json) as f:
        annos = json.load(f)
    inst_to_cat = load_categories(ann_root)
    out = defaultdict(lambda: defaultdict(list))
    sample_set = set(sample_tokens)
    for a in annos:
        if a["sample_token"] not in sample_set:
            continue
        cls = inst_to_cat.get(a.get("instance_token"), "?")
        if cls not in TRACK_CLASSES:
            continue
        out[a["sample_token"]][cls].append({
            "translation": a["translation"],
            "instance_token": a["instance_token"],
        })
    return out


def load_ego_translations(info_pkl: Path):
    """sample_token -> ego2global_translation."""
    with open(info_pkl, "rb") as f:
        info = pickle.load(f)
    out = {}
    for inf in info["infos"]:
        out[inf["token"]] = np.array(inf["ego2global_translation"], dtype=np.float64)
    return out


def filter_by_class_range(items, ego_xyz, cls):
    """Keep only items within class_range from ego center (xy distance, nuScenes convention)."""
    r = CLASS_RANGE_M.get(cls, 50.0)
    out = []
    for it in items:
        t = np.array(it["translation"], dtype=np.float64)
        d_xy = math.hypot(t[0] - ego_xyz[0], t[1] - ego_xyz[1])
        if d_xy <= r:
            out.append(it)
    return out


def hungarian_match_one_frame(preds, gts, dist_thresh):
    """Per-frame Hungarian matching by 2D center distance (xy). Returns (n_tp, n_unmatched_pred, n_unmatched_gt, matches)."""
    if not preds or not gts:
        return 0, len(preds), len(gts), []
    cost = np.full((len(preds), len(gts)), 1e6, dtype=np.float64)
    for i, p in enumerate(preds):
        pp = np.array(p["translation"], dtype=np.float64)
        for j, g in enumerate(gts):
            gg = np.array(g["translation"], dtype=np.float64)
            d = math.hypot(pp[0] - gg[0], pp[1] - gg[1])
            if d <= dist_thresh:
                cost[i, j] = d
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(rows, cols):
        if cost[r, c] < 1e5:
            matches.append((r, c, cost[r, c]))
    n_tp = len(matches)
    matched_p = {m[0] for m in matches}
    matched_g = {m[1] for m in matches}
    return n_tp, len(preds) - len(matched_p), len(gts) - len(matched_g), matches


def evaluate(pred_json, ann_root, info_pkl, score_min=0.0, dist_thresh=2.0):
    with open(pred_json) as f:
        pred_data = json.load(f)
    sample_tokens = list(pred_data["results"].keys())
    print(f"[eval] {len(sample_tokens)} samples in pred file")

    gt_by_sample = load_annos_for_samples(Path(ann_root), sample_tokens)
    ego_by_sample = load_ego_translations(Path(info_pkl))

    per_class = {cls: {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "n_gt": 0} for cls in TRACK_CLASSES}

    for sample in sample_tokens:
        ego_xyz = ego_by_sample.get(sample)
        if ego_xyz is None:
            continue
        # group predictions by class for this sample
        preds_by_cls = defaultdict(list)
        for p in pred_data["results"][sample]:
            cls = p.get("tracking_name")
            score = p.get("tracking_score", 0.0)
            if score < score_min:
                continue
            if cls not in TRACK_CLASSES:
                continue
            preds_by_cls[cls].append(p)

        gts = gt_by_sample.get(sample, {})

        for cls in TRACK_CLASSES:
            preds_filt = filter_by_class_range(preds_by_cls.get(cls, []), ego_xyz, cls)
            gts_filt = filter_by_class_range(gts.get(cls, []), ego_xyz, cls)
            per_class[cls]["n_pred"] += len(preds_filt)
            per_class[cls]["n_gt"] += len(gts_filt)
            tp, fp, fn, _ = hungarian_match_one_frame(preds_filt, gts_filt, dist_thresh)
            per_class[cls]["tp"] += tp
            per_class[cls]["fp"] += fp
            per_class[cls]["fn"] += fn

    # Aggregate
    total = {"tp": 0, "fp": 0, "fn": 0, "n_pred": 0, "n_gt": 0}
    for cls in TRACK_CLASSES:
        for k in total:
            total[k] += per_class[cls][k]

    def safe_div(a, b):
        return a / b if b > 0 else float("nan")

    summary = {}
    for cls in TRACK_CLASSES:
        s = per_class[cls]
        if s["n_gt"] == 0 and s["n_pred"] == 0:
            continue
        recall = safe_div(s["tp"], s["tp"] + s["fn"])
        precision = safe_div(s["tp"], s["tp"] + s["fp"])
        f1 = safe_div(2 * recall * precision, recall + precision) if recall and precision else float("nan")
        summary[cls] = {**s, "recall": recall, "precision": precision, "f1": f1}

    overall_recall = safe_div(total["tp"], total["tp"] + total["fn"])
    overall_precision = safe_div(total["tp"], total["tp"] + total["fp"])
    summary["__overall__"] = {
        **total,
        "recall": overall_recall,
        "precision": overall_precision,
        "f1": safe_div(2 * overall_recall * overall_precision, overall_recall + overall_precision)
              if overall_recall and overall_precision else float("nan"),
    }
    return summary


def fmt(v, k):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "  -  "
    if k in {"recall", "precision", "f1"}:
        return f"{v*100:6.2f}%"
    return f"{int(v):5d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="results_nusc.json from test.py")
    ap.add_argument("--ann-root", required=True, help="datasets/.../{vehicle-side,cooperative}")
    ap.add_argument("--info", required=True, help="spd_infos_temporal_val.pkl")
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--dist-thresh", type=float, default=2.0, help="nuScenes tracking default 2m")
    ap.add_argument("--out", default=None)
    ap.add_argument("--label", default="ckpt", help="label for the run")
    args = ap.parse_args()

    summary = evaluate(args.pred, args.ann_root, args.info, args.score_min, args.dist_thresh)

    print(f"\n=== Real-recall eval [{args.label}] (dist≤{args.dist_thresh}m, score≥{args.score_min}) ===")
    print(f"{'class':12s} {'n_pred':>6s} {'n_gt':>6s} {'tp':>5s} {'fp':>5s} {'fn':>5s} {'recall':>7s} {'precision':>9s} {'f1':>7s}")
    for cls in TRACK_CLASSES + ["__overall__"]:
        if cls not in summary:
            continue
        s = summary[cls]
        print(f"{cls:12s} {s['n_pred']:6d} {s['n_gt']:6d} {s['tp']:5d} {s['fp']:5d} {s['fn']:5d} "
              f"{fmt(s['recall'], 'recall')} {fmt(s['precision'], 'precision')} {fmt(s['f1'], 'f1')}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "args": vars(args), "summary": summary}, f, indent=2, default=str)
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
