#!/usr/bin/env python3
"""Build four cross-time full-AP repeat jobs bound to fresh H800 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "stage35_gold128_ap_repeat_plan_v1"
AP_ANCHORS = (
    ("base", "pyramid", "32x32x64", "tvm_fp16"),
    ("small_channel", "codriving", "16x32x64", "trt_int8"),
    ("alignment_trap", "pyramid", "48x64x128", "tvm_int8"),
    ("large_model", "codriving", "64x128x256", "trt_fp16"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _option(command: Sequence[str], flag: str) -> str:
    if flag not in command:
        raise ValueError(f"repeat command lacks {flag}")
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"repeat command lacks value for {flag}")
    return str(command[index + 1])


def _compiled_artifact(job: Mapping[str, Any]) -> Path:
    command = list(map(str, job.get("command") or []))
    runner = str(job.get("runner_key"))
    if runner == "tvm_fp16":
        return Path(_option(command, "--out-dir")) / _option(command, "--label") / "route_b_fp16_auto.so"
    if runner == "tvm_int8":
        return Path(_option(command, "--out-dir")) / _option(command, "--label") / "route_b_int8_auto_decomp.vmexec"
    if runner in {"trt_fp16", "trt_int8"}:
        return Path(_option(command, "--artifact-dir")) / "compiled.engine"
    raise ValueError(f"unsupported repeat runner: {runner}")


def _pyramid_model_dir(width_key: str) -> str:
    widths = [int(value) for value in width_key.split("x")]
    if len(widths) != 3:
        raise ValueError(f"invalid width key: {width_key}")
    return "/home/jichengzhi/V2X/models/dataset_a_cache/ft_" + "_".join(f"{value:03d}" for value in widths)


def _full_command(job: Mapping[str, Any], artifact: Path, output_dir: Path) -> tuple[str, list[str]]:
    model = str(job["model"])
    width = str(job["width_key"])
    runner = str(job["runner_key"])
    report = output_dir / "full_ap_eval_report.json"
    if model == "pyramid" and runner == "tvm_fp16":
        return "pyramid_tvm_fp16_bridge", [
            "python3", "scripts/stage2_h800_fp16_rewritten_activation_bridge.py",
            "--label", f"stage35_repeat_pyramid_{width}_fp16",
            "--ckpt-dir", _pyramid_model_dir(width),
            "--raw-dir", str(output_dir), "--eval-range", "102.4,51.2",
            "--artifact-path", str(artifact), "--artifact-input-dtype", "float32",
            "--persistent-worker", "--num-samples", "1789", "--full-ap-min-samples", "1789",
            "--export-report-json", str(report),
        ]
    if model == "pyramid" and runner == "tvm_int8":
        return "pyramid_tvm_int8_numeric_gate", [
            "python3", "scripts/stage3_pyramid_tvm_int8_ap_numeric_gate_v3.py",
            "--compiled-artifact", str(artifact), "--precision-tag", "int8",
            "--num-samples", "1789", "--full-ap-min-samples", "1789",
            "--output-dir", str(output_dir), "--model-dir", _pyramid_model_dir(width),
            "--eval-range", "102.4,51.2", "--report-json", str(report),
        ]
    if model == "codriving" and runner in {"trt_fp16", "trt_int8"}:
        precision = "int8" if runner == "trt_int8" else "fp16"
        model_dir = f"/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709/{width}"
        return "codriving_trt_multiscale_bridge", [
            "python3", "scripts/stage3_codriving_trt_multiscale_ap_bridge_v3.py",
            "--repo-root", "/exdata/jichengzhi/V2Xverse_pyramid",
            "--model-dir", model_dir, "--engine", str(artifact), "--precision-tag", precision,
            "--gate", "full", "--n-samples", "1789", "--num-workers", "0",
            "--eval-dir", str(output_dir / "eval"), "--out-json", str(report),
        ]
    raise ValueError(f"unsupported AP repeat anchor: {model}/{runner}")


def _replace_option(command: Sequence[str], option: str, value: str) -> list[str]:
    updated = list(command)
    if option not in updated:
        return updated
    index = updated.index(option)
    return [*updated[: index + 1], value, *updated[index + 2 :]]


def _sanity_command(full_command: Sequence[str], output_dir: Path) -> list[str]:
    sanity_dir = output_dir.parent / f"{output_dir.name}_sanity16"
    command = [str(token).replace(str(output_dir), str(sanity_dir)) for token in full_command]
    for option in ("--num-samples", "--n-samples"):
        command = _replace_option(command, option, "16")
    command = _replace_option(command, "--gate", "sanity")
    return command


def build_ap_repeat_plan(
    repeat_jobs: Sequence[Mapping[str, Any]],
    gold_rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_anchor = {
        (str(row.get("repeat_category")), str(row.get("model")), str(row.get("width_key")), str(row.get("runner_key"))): row
        for row in repeat_jobs
    }
    gold_by_id = {str(row.get("manifest_job_id")): row for row in gold_rows}
    plans: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    for category, model, width, runner in AP_ANCHORS:
        job = by_anchor.get((category, model, width, runner))
        if job is None:
            raise ValueError(f"missing AP repeat anchor: {category}/{model}/{width}/{runner}")
        artifact = _compiled_artifact(job)
        if not artifact.is_file():
            raise FileNotFoundError(f"compiled artifact is missing: {artifact}")
        manifest_id = str(job["manifest_job_id"])
        baseline = gold_by_id.get(manifest_id)
        if baseline is None:
            raise ValueError(f"missing Gold AP baseline: {manifest_id}")
        ap_job_id = f"ap-repeat|{category}|{model}|{width}|{runner}"
        output_dir = Path(output_root) / category / ap_job_id.replace("|", "__")
        runner_key, command = _full_command(job, artifact, output_dir)
        plan = {
            "schema_version": SCHEMA,
            "job_id": ap_job_id,
            "manifest_job_id": ap_job_id,
            "source_manifest_job_id": manifest_id,
            "repeat_category": category,
            "model": model,
            "width_key": width,
            "q_mode": str(job.get("q_mode")),
            "runner_key": runner_key,
            "ap_terminal": "ready",
            "compiled_artifact_path": str(artifact),
            "compiled_artifact_digest": sha256_file(artifact),
            "sanity_command": _sanity_command(command, output_dir),
            "full_command": command,
            "baseline_ap30": baseline.get("ap30"),
            "baseline_ap50": baseline.get("ap50"),
            "baseline_ap70": baseline.get("ap70"),
            "baseline_ap_report_sha256": baseline.get("ap_report_sha256"),
        }
        plans.append(plan)
    return plans, seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-jobs-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--gold-json", type=Path, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-plan-jsonl", type=Path, required=True)
    parser.add_argument("--output-seed-state-jsonl", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = [row for path in args.repeat_jobs_jsonl for row in read_jsonl(path)]
    gold = json.loads(args.gold_json.read_text(encoding="utf-8"))
    plans, seeds = build_ap_repeat_plan(jobs, gold, output_root=args.output_root)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    selected_ids = {
        row["job_id"] for index, row in enumerate(plans) if index % args.shard_count == args.shard_index
    }
    plans = [row for row in plans if row["job_id"] in selected_ids]
    seeds = [row for row in seeds if row["job_id"] in selected_ids]
    args.output_plan_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plans), encoding="utf-8")
    args.output_seed_state_jsonl.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in seeds), encoding="utf-8")
    print(json.dumps({
        "jobs": len(plans), "categories": [row["repeat_category"] for row in plans],
        "shard_count": args.shard_count, "shard_index": args.shard_index,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
