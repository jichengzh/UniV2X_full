#!/usr/bin/env python3
"""Generate H800 launch-plan artifacts for original60 FP16 AP backfill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
DEFAULT_GPUS = (3, 4, 5, 6)
DEFAULT_MASTER_PORT_BASE = 29730
DEFAULT_CKPT_ROOT = Path("/exdata/jichengzhi/heal_research/checkpoints/stage1")
DEFAULT_BASE_CKPT_DIR = (
    "/exdata/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/exdata/jichengzhi/heal_research/HEAL"
SCHEMA = "stage2_original60_fp16_ap_launch_plan_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--completion-queue", default=None)
    parser.add_argument("--ap-queue", default=None)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--gpus", default="3,4,5,6")
    parser.add_argument("--pilot-size", type=int, default=4)
    parser.add_argument("--master-port-base", type=int, default=DEFAULT_MASTER_PORT_BASE)
    parser.add_argument("--ckpt-root", default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--base-ckpt-dir", default=DEFAULT_BASE_CKPT_DIR)
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--epoches", type=int, default=31)
    parser.add_argument("--width-per-group", type=int, default=4)
    parser.add_argument("--groups", type=int, default=32)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_gpus(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item.strip()]


def ckpt_dir_for(ckpt_root: Path, label: str) -> Path:
    return ckpt_root / f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_28"


def raw_dir_for(output_root: Path, label: str, gpu_id: int) -> Path:
    return output_root / "raw/ap_eval_original60" / f"fp16_ckpt_gen_{label}_gpu{gpu_id}_v1"


def master_port_for(base: int, index: int) -> int:
    return base + index


def train_command(args: argparse.Namespace, *, label: str, width: list[int], gpu_id: int, master_port: int, ckpt_dir: Path, raw_dir: Path) -> list[str]:
    return [
        args.env_python,
        str(ROOT / "scripts/stage2_original60_fp16_train_launcher.py"),
        "--label",
        label,
        "--width",
        ",".join(str(x) for x in width),
        "--gpu-id",
        str(gpu_id),
        "--master-port",
        str(master_port),
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(raw_dir),
        "--env-python",
        args.env_python,
        "--heal-root",
        args.heal_root,
        "--base-ckpt-dir",
        args.base_ckpt_dir,
        "--epoches",
        str(args.epoches),
        "--width-per-group",
        str(args.width_per_group),
        "--groups",
        str(args.groups),
    ]


def eval_watcher_command(args: argparse.Namespace, *, label: str, width: list[int], gpu_id: int, ckpt_dir: Path, raw_dir: Path) -> list[str]:
    return [
        args.env_python,
        str(ROOT / "scripts/stage2_original60_fp16_eval_when_ready.py"),
        "--label",
        label,
        "--width",
        ",".join(str(x) for x in width),
        "--gpu-id",
        str(gpu_id),
        "--ckpt-dir",
        str(ckpt_dir),
        "--raw-dir",
        str(raw_dir),
        "--train-pid",
        "<train_runner.pid>",
        "--rows-out",
        str(Path(args.output_root) / "rows/fp16_true_original60_ap_rows_v1.jsonl"),
    ]


def summary_md(plan: dict[str, Any]) -> str:
    lines = [
        "# FP16 Original60 AP Launch Plan",
        "",
        f"- created_at: `{plan['created_at']}`",
        f"- fp16_gap_labels: `{plan['summary']['fp16_gap_labels']}`",
        f"- pilot_label_count: `{plan['summary']['pilot_label_count']}`",
        f"- bulk_label_count: `{plan['summary']['bulk_label_count']}`",
        f"- gpus: `{plan['summary']['gpus']}`",
        "",
        "## Pilot Batch",
        "",
        f"- labels: `{', '.join(plan['pilot_batch']['labels'])}`",
    ]
    for gpu, labels in plan["pilot_batch"]["gpu_map"].items():
        lines.append(f"- GPU{gpu}: `{', '.join(labels) if labels else '-'}`")
    lines.extend(["", "## Bulk Batch"])
    for gpu, labels in plan["bulk_batches"]["gpu_map"].items():
        lines.append(f"- GPU{gpu}: `{', '.join(labels) if labels else '-'}`")
    lines.extend(["", "## Labels", "", "| label | width | gpu | port | phase |", "|---|---|---:|---:|---|"])
    for item in plan["labels"]:
        width = "x".join(str(x) for x in item["width"])
        lines.append(
            f"| {item['label']} | {width} | {item['gpu_id']} | {item['master_port']} | {item['phase']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    completion_queue = Path(args.completion_queue) if args.completion_queue else output_root / "jobs/fp16_int8_original60_completion_queue_v1.jsonl"
    ap_queue = Path(args.ap_queue) if args.ap_queue else output_root / "jobs/fp16_int8_original60_ap_true_eval_queue_v1.jsonl"
    created_at = args.created_at or "unknown"
    gpus = parse_gpus(args.gpus)
    ckpt_root = Path(args.ckpt_root)

    completion_jobs = {
        str(row["label"]): row
        for row in read_jsonl(completion_queue)
        if str(row.get("precision") or "") == "fp16"
        and str(row.get("ap_status") or "") == "no_claim"
    }
    fp16_gap_labels = [
        str(row["label"])
        for row in read_jsonl(ap_queue)
        if str(row.get("precision") or "") == "fp16"
        and str(row.get("ap_eval_status") or "") == "blocked"
        and str(row.get("blocker") or "") == "no_compliant_true_fp16_model_eval_source"
        and str(row.get("label") or "") in completion_jobs
    ]

    labels: list[dict[str, Any]] = []
    for index, label in enumerate(fp16_gap_labels):
        job = completion_jobs[label]
        gpu_id = gpus[index % len(gpus)]
        master_port = master_port_for(args.master_port_base, index)
        width = [int(x) for x in job.get("width") or []]
        ckpt_dir = ckpt_dir_for(ckpt_root, label)
        raw_dir = raw_dir_for(output_root, label, gpu_id)
        phase = "pilot" if index < args.pilot_size else "bulk"
        labels.append(
            {
                "index": index,
                "phase": phase,
                "label": label,
                "width": width,
                "gpu_id": gpu_id,
                "master_port": master_port,
                "ckpt_dir": str(ckpt_dir),
                "raw_dir": str(raw_dir),
                "workdir": job.get("workdir"),
                "recovery_search_roots": [
                    str(ckpt_root),
                    "/home/jichengzhi/heal_research/checkpoints/stage1",
                ],
                "train_command": train_command(
                    args,
                    label=label,
                    width=width,
                    gpu_id=gpu_id,
                    master_port=master_port,
                    ckpt_dir=ckpt_dir,
                    raw_dir=raw_dir,
                ),
                "eval_watcher_command": eval_watcher_command(
                    args,
                    label=label,
                    width=width,
                    gpu_id=gpu_id,
                    ckpt_dir=ckpt_dir,
                    raw_dir=raw_dir,
                ),
            }
        )

    def gpu_map(phase: str) -> dict[str, list[str]]:
        mapping = {str(gpu): [] for gpu in gpus}
        for item in labels:
            if item["phase"] != phase:
                continue
            mapping[str(item["gpu_id"])].append(item["label"])
        return mapping

    plan = {
        "schema": SCHEMA,
        "created_at": created_at,
        "inputs": {
            "completion_queue": str(completion_queue),
            "ap_queue": str(ap_queue),
        },
        "summary": {
            "fp16_gap_labels": len(fp16_gap_labels),
            "pilot_label_count": min(args.pilot_size, len(fp16_gap_labels)),
            "bulk_label_count": max(0, len(fp16_gap_labels) - args.pilot_size),
            "gpus": gpus,
            "epoches": args.epoches,
            "width_per_group": args.width_per_group,
            "groups": args.groups,
        },
        "pilot_batch": {
            "labels": [item["label"] for item in labels if item["phase"] == "pilot"],
            "gpu_map": gpu_map("pilot"),
        },
        "bulk_batches": {
            "labels": [item["label"] for item in labels if item["phase"] == "bulk"],
            "gpu_map": gpu_map("bulk"),
        },
        "labels": labels,
    }

    json_path = output_root / "exports/fp16_original60_ap_launch_plan_latest.json"
    md_path = output_root / "exports/fp16_original60_ap_launch_plan_latest.md"
    write_json(json_path, plan)
    md_path.write_text(summary_md(plan), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "plan_json": str(json_path),
                "fp16_gap_labels": len(fp16_gap_labels),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
