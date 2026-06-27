"""Evaluate existing attention-p50 checkpoints on a larger/full split.

This script reuses the T1 attention surgery and evaluation helpers. It does not
train; it loads baseline, no-finetune p50, and short-finetuned p50 checkpoints
under one latency/AP protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from t1_attention_e2e_pq import (  # noqa: E402
    CKPT_FILE,
    LOG_DIR,
    REPO_ROOT,
    _round,
    apply_attention_p50_surgery,
    build_dataset_from_hypes,
    evaluate_ap,
    load_model_and_hypes,
    load_state_dict_compatible,
    measure_model_forward_latency,
)


NOFT_CKPT = REPO_ROOT / "models/v2xvit_attention_t1/attention_p50_noft_epoch17.pth"
NOFT_MANIFEST = REPO_ROOT / "models/v2xvit_attention_t1/attention_p50_manifest_v1.json"
SHORTFT_CKPT = Path(
    os.environ.get(
        "V2XVIT_ATTENTION_SHORTFT_CKPT",
        REPO_ROOT / "models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth",
    )
)
SHORTFT_MANIFEST = Path(
    os.environ.get(
        "V2XVIT_ATTENTION_SHORTFT_MANIFEST",
        REPO_ROOT / "models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json",
    )
)
SHORTFT_FINETUNE = os.environ.get(
    "V2XVIT_ATTENTION_SHORTFT_FINETUNE",
    "100_steps_lr1e-4_seed20260623_all_params",
)
OUT_JSON = REPO_ROOT / "results/attention_e2e_checkpoint_eval_v1.json"
OUT_CSV = REPO_ROOT / "results/attention_e2e_checkpoint_eval_v1.csv"
OUT_MD = REPO_ROOT / "results/attention_e2e_checkpoint_eval_v1.md"


def default_eval_configs() -> list[dict[str, Any]]:
    return [
        {
            "config": "baseline",
            "checkpoint_path": str(CKPT_FILE),
            "manifest_path": None,
            "pruned": False,
            "attention_prune_pct": 0,
            "finetune": "official_ckpt",
        },
        {
            "config": "attention-p50-fp16",
            "checkpoint_path": str(NOFT_CKPT),
            "manifest_path": str(NOFT_MANIFEST),
            "pruned": True,
            "attention_prune_pct": 50,
            "finetune": "none",
        },
        {
            "config": "attention-p50-shortft-fp16",
            "checkpoint_path": str(SHORTFT_CKPT),
            "manifest_path": str(SHORTFT_MANIFEST),
            "pruned": True,
            "attention_prune_pct": 50,
            "finetune": SHORTFT_FINETUNE,
        },
    ]


def _load_checkpoint_state(path: str) -> dict[str, Any]:
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"unsupported checkpoint object at {path}: {type(obj).__name__}")


def eval_log_tag(eval_samples: int) -> str:
    return "full" if int(eval_samples) >= 1789 else f"eval{int(eval_samples)}"


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_eval_model(config: dict[str, Any], device: str):
    model, hypes = load_model_and_hypes(device)
    load_info = {"status": "baseline_loaded_from_official_checkpoint"}
    if config.get("pruned"):
        apply_attention_p50_surgery(model)
        state = _load_checkpoint_state(config["checkpoint_path"])
        load_info = load_state_dict_compatible(model, state)
    return model.eval(), hypes, load_info


def build_comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(row for row in rows if row["config"] == "baseline")
    base_latency = float(baseline["latency_p50_ms"])
    base_ap50 = float(baseline["ap50"])
    base_ap70 = float(baseline["ap70"])
    out = []
    for row in rows:
        enriched = dict(row)
        latency = float(row["latency_p50_ms"])
        enriched["speedup_vs_baseline"] = _round(base_latency / latency, 4) if latency > 0 else None
        enriched["delta_ap50"] = _round(float(row["ap50"]) - base_ap50, 4)
        enriched["delta_ap70"] = _round(float(row["ap70"]) - base_ap70, 4)
        out.append(enriched)
    return out


def evaluate_config(config: dict[str, Any], *, device: str, precision: str, eval_samples: int, latency_samples: int, latency_warmup: int, num_workers: int) -> dict[str, Any]:
    model, hypes, load_info = load_eval_model(config, device)
    dataset = build_dataset_from_hypes(hypes)
    tag = config["config"].replace("/", "_")
    sample_tag = eval_log_tag(eval_samples)
    lat_log = LOG_DIR / f"{tag}_{sample_tag}_latency_v1.json"
    ap_log = LOG_DIR / f"{tag}_{sample_tag}_ap_v1.json"
    latency = measure_model_forward_latency(
        model,
        dataset,
        device,
        precision,
        latency_warmup,
        latency_samples,
        lat_log,
        num_workers,
    )
    latency["log_path"] = str(lat_log)
    ap = evaluate_ap(model, dataset, device, precision, eval_samples, ap_log, num_workers)
    ap["log_path"] = str(ap_log)
    return {
        "config": config["config"],
        "checkpoint_path": config["checkpoint_path"],
        "manifest_path": config.get("manifest_path"),
        "attention_prune_pct": config["attention_prune_pct"],
        "quant": "fp16" if precision == "fp16" else "fp32",
        "finetune": config["finetune"],
        "latency_scope": latency.get("latency_scope"),
        "latency_p50_ms": latency.get("p50_ms"),
        "latency_mean_ms": latency.get("mean_ms"),
        "ap50": ap.get("ap50"),
        "ap70": ap.get("ap70"),
        "accuracy_status": ap.get("accuracy_status"),
        "ap_scope": ap.get("ap_scope"),
        "dataset_split": ap.get("dataset_split", "DAIR val"),
        "eval_samples": ap.get("sample_count"),
        "latency_samples": latency.get("sample_count"),
        "latency_log": latency.get("log_path"),
        "ap_log": ap.get("log_path"),
        "load_info": load_info,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attention E2E Checkpoint Eval v1",
        "",
        f"Eval samples: {report.get('eval_samples')}. This is not Stop-A unless it is full-val and includes the TVM quantized row.",
        "",
        "| config | prune | finetune | latency p50 ms | speedup | AP50 | AP70 | dAP50 | dAP70 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['config']} | {row.get('attention_prune_pct', '')}% | {row.get('finetune', '')} | "
            f"{row['latency_p50_ms']} | {row['speedup_vs_baseline']}x | {row['ap50']} | "
            f"{row['ap70']} | {row['delta_ap50']} | {row['delta_ap70']} |"
        )
    lines += [
        "",
        "Caveat: this table compares PyTorch FP16 checkpoints. It does not include a complete TVM mixed-INT8 e2e row yet.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--latency-samples", type=int, default=30)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    parser.add_argument("--out-md", default=str(OUT_MD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        evaluate_config(
            config,
            device=args.device,
            precision=args.precision,
            eval_samples=args.eval_samples,
            latency_samples=args.latency_samples,
            latency_warmup=args.latency_warmup,
            num_workers=args.num_workers,
        )
        for config in default_eval_configs()
    ]
    rows = build_comparison_rows(rows)
    report = {
        "schema_version": "attention_e2e_checkpoint_eval_v1",
        "precision": args.precision,
        "eval_samples": args.eval_samples,
        "latency_samples": args.latency_samples,
        "rows": rows,
        "stop_a_ready": False,
        "reason": "missing complete TVM mixed-INT8 e2e row",
    }
    out_json = resolve_repo_path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_csv(rows, resolve_repo_path(args.out_csv))
    resolve_repo_path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
