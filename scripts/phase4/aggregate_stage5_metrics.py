"""Aggregate stage5 metrics JSON files into a CSV row per config.

Reads:
  - data/phase2/active_samples_plan.csv  (config feature schema for B1..E12)
  - data/phase4/stage5_metrics/<id>.json (real-recall metrics)

Writes:
  - data/phase4/stage5_baseline_v2.csv  (config_id, features..., F1, recall, precision, n_pred, tp)
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLAN_CSV = ROOT / "data/phase2/active_samples_plan.csv"
METRICS_DIR = ROOT / "data/phase4/stage5_metrics"
OUT_CSV = ROOT / "data/phase4/stage5_baseline_v2.csv"


def main() -> None:
    plan = pd.read_csv(PLAN_CSV)
    plan_by_id = {row["id"]: row for _, row in plan.iterrows()}

    rows = []
    for json_path in sorted(METRICS_DIR.glob("*.json")):
        cfg_id = json_path.stem
        with open(json_path) as f:
            data = json.load(f)
        if "error" in data:
            print(f"skip {cfg_id}: {data['error']}", file=sys.stderr)
            continue
        summary = data.get("summary", {})
        overall = summary.get("__overall__", {})
        car = summary.get("car", {})
        bike = summary.get("bicycle", {})

        if cfg_id == "baseline":
            features = {
                "id": "baseline",
                "goal": "no pruning, merge pipeline reference",
                "physical_ok": True,
                "is_exploratory": False,
                "prune_object": "none",
                "prune_rate__backbone": 0.0,
                "prune_rate__encoder": 0.0,
                "prune_rate__decoder": 0.0,
                "prune_rate__heads": 0.0,
                "prune_rate__v2x_comm": 0.0,
                "q_bits__backbone": "FP32",
                "q_bits__encoder": "FP32",
                "q_bits__decoder": "FP32",
                "q_bits__heads": "FP32",
                "q_bits__v2x_comm": "FP32",
                "q_granularity__encoder": "none",
                "q_object__encoder": "none",
                "d_routing__backbone": "GPU",
            }
        elif cfg_id in plan_by_id:
            row = plan_by_id[cfg_id]
            features = {k: row[k] for k in plan.columns}
        else:
            print(f"skip {cfg_id}: no plan entry", file=sys.stderr)
            continue

        features.update({
            "config_id": cfg_id,
            "source": "stage5_active",
            "real_recall_overall": overall.get("recall", float("nan")),
            "real_precision_overall": overall.get("precision", float("nan")),
            "real_f1_overall": overall.get("f1", float("nan")),
            "real_recall_car": car.get("recall", float("nan")),
            "real_precision_car": car.get("precision", float("nan")),
            "real_f1_car": car.get("f1", float("nan")),
            "n_pred_overall": overall.get("n_pred", 0),
            "tp_overall": overall.get("tp", 0),
            "n_gt_overall": overall.get("n_gt", 0),
        })
        rows.append(features)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print(df[["config_id", "prune_rate__encoder", "prune_rate__decoder", "prune_rate__heads",
              "real_f1_overall", "real_recall_car", "n_pred_overall", "tp_overall"]].to_string(index=False))


if __name__ == "__main__":
    main()
