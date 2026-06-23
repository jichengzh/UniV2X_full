"""Extract AMOTA / mAP / recall etc from Plan B per-config eval logs.

Reads logs/plan_b/<config>.log and writes data/phase4/stage5_v3/<config>.json.
Both pruning (test_with_pruning.py) and quantization (quick_eval_quant.py) logs
are supported — they share the same dataset.evaluate() output format.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


METRICS_PATTERNS = {
    # Format A: nuScenes "Aggregated results" table (legacy)
    "amota": re.compile(r"^\s*AMOTA[\s\t:=]+([0-9.]+)", re.IGNORECASE | re.MULTILINE),
    "amotp": re.compile(r"^\s*AMOTP[\s\t:=]+([0-9.]+)", re.IGNORECASE | re.MULTILINE),
    "recall": re.compile(r"^\s*RECALL[\s\t:=]+([0-9.]+)", re.IGNORECASE | re.MULTILINE),
    "mota": re.compile(r"^\s*MOTA[\s\t:=]+([0-9.]+)", re.IGNORECASE | re.MULTILINE),
    "motp": re.compile(r"^\s*MOTP[\s\t:=]+([0-9.]+)", re.IGNORECASE | re.MULTILINE),
    "tp": re.compile(r"^\s*TP[\s\t:=]+(\d+)", re.MULTILINE),
    "fp": re.compile(r"^\s*FP[\s\t:=]+(\d+)", re.MULTILINE),
    "fn": re.compile(r"^\s*FN[\s\t:=]+(\d+)", re.MULTILINE),
    "ml": re.compile(r"^\s*ML[\s\t:=]+(\d+)", re.MULTILINE),
    "mt": re.compile(r"^\s*MT[\s\t:=]+(\d+)", re.MULTILINE),
    "ids": re.compile(r"^\s*IDS[\s\t:=]+(\d+)", re.MULTILINE),
    "gt": re.compile(r"^\s*GT[\s\t:=]+(\d+)", re.MULTILINE),
    "car_amota": re.compile(r"^\s*car\s+([0-9.]+)\s+", re.MULTILINE),
}

# Format B: test_with_pruning.py "最终指标" block / dict-style output
DICT_PATTERNS = {
    "amota": re.compile(r"pts_bbox_NuScenes/amota[\s:'\"]+([0-9.]+)"),
    "amotp": re.compile(r"pts_bbox_NuScenes/amotp[\s:'\"]+([0-9.]+)"),
    "recall": re.compile(r"pts_bbox_NuScenes/recall[\s:'\"]+([0-9.]+)"),
    "mota": re.compile(r"pts_bbox_NuScenes/mota[\s:'\"]+([0-9.]+)"),
    "tp": re.compile(r"pts_bbox_NuScenes/tp[\s:'\"]+([0-9.]+)"),
    "fp": re.compile(r"pts_bbox_NuScenes/fp[\s:'\"]+([0-9.]+)"),
    "fn": re.compile(r"pts_bbox_NuScenes/fn[\s:'\"]+([0-9.]+)"),
    "gt": re.compile(r"pts_bbox_NuScenes/gt[\s:'\"]+([0-9.]+)"),
    "mAP": re.compile(r"pts_bbox_NuScenes/mAP[\s:'\"]+([0-9.]+)"),
    "NDS": re.compile(r"pts_bbox_NuScenes/NDS[\s:'\"]+([0-9.]+)"),
    "car_ap_4m": re.compile(r"pts_bbox_NuScenes/car_AP_dist_4\.0[\s:'\"]+([0-9.]+)"),
}

# Multi-line table parsing for nuScenes-style output
TABLE_HEADER_RE = re.compile(r"^\s*Per-class results:", re.IGNORECASE)


def parse_log(log_path: Path) -> dict:
    """Pull metrics out of a single log file (tries both formats)."""
    if not log_path.exists():
        return {"error": f"log not found: {log_path}"}

    text = log_path.read_text(errors="ignore")
    out = {}

    # Format B (dict-style "pts_bbox_NuScenes/amota: 0.330") — preferred
    for key, pat in DICT_PATTERNS.items():
        matches = pat.findall(text)
        if matches:
            try:
                out[key] = float(matches[-1])
            except ValueError:
                pass

    # Format A (table-style) — fallback if dict format didn't match
    if "amota" not in out:
        for key, pat in METRICS_PATTERNS.items():
            matches = pat.findall(text)
            if matches:
                try:
                    out[key] = float(matches[-1])
                except ValueError:
                    pass

    # Look for failed/errored runs
    if "Traceback" in text or "FAILED" in text or "RuntimeError" in text:
        out["has_error"] = True
        lines = text.strip().split("\n")
        out["error_tail"] = "\n".join(lines[-5:])

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--log-dir", default="logs/plan_b")
    ap.add_argument("--out-dir", default="data/phase4/stage5_v3")
    args = ap.parse_args()

    log_path = Path(args.log_dir) / f"{args.config_id}.log"
    out_path = Path(args.out_dir) / f"{args.config_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = parse_log(log_path)
    metrics["config_id"] = args.config_id

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"{args.config_id}: {metrics}")


if __name__ == "__main__":
    main()
