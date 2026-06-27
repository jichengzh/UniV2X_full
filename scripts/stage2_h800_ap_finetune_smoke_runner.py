#!/usr/bin/env python3
"""Run Stage2 Phase C AP finetune smoke jobs on H800.

This consumes ``ap_finetune_smoke_queue_v1.jsonl`` and performs the full
claimable AP chain for each job:

structural_prune_pyramid -> flat ckpt check -> train_ddp --half -> ONNX export
-> TRT FP16 build -> DAIR-V2X val_1789 AP eval -> canonical AP row.

The script is fail-closed: predicted/model-fit/interpolated AP is never written
as measured. A failed config writes job_state + quarantine and the rest of the
batch continues.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    ap_anchor_row,
    stable_config_id,
    utc_timestamp,
    validate_lut_row,
)


DEFAULT_BASE = (
    "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626"
)
DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_BASE_CKPT_DIR = (
    "/home/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)
AP_PROTOCOL = "structural_prune_pyramid_train_ddp_half_epoches31_TRT_FP16_DAIR_val_1789"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=DEFAULT_BASE)
    parser.add_argument("--job-queue", default=None)
    parser.add_argument("--job-state", default=None)
    parser.add_argument("--rows-out", default=None)
    parser.add_argument("--quarantine-out", default=None)
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--base-ckpt-dir", default=DEFAULT_BASE_CKPT_DIR)
    parser.add_argument("--labels", default="", help="Comma-separated labels to run.")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--created-at", default=None)
    parser.add_argument("--execute", action="store_true", help="Actually run jobs.")
    parser.add_argument("--force", action="store_true", help="Rerun even if row already exists.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(values):
        h.update(value.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def command_env(gpu_id: int, heal_root: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = heal_root + os.pathsep + env.get("PYTHONPATH", "")
    return env


def save_command(path: Path, command: list[str], env: dict[str, str], cwd: Path, timeout_s: int) -> None:
    write_json(
        path,
        {
            "command": command,
            "cwd": str(cwd),
            "timeout_s": timeout_s,
            "env": {
                "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
                "PYTHONPATH": env.get("PYTHONPATH"),
            },
        },
    )


def run_logged(
    *,
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    raw_dir: Path,
    timeout_s: int,
) -> subprocess.CompletedProcess[str]:
    save_command(raw_dir / f"{name}_command.json", command, env, cwd, timeout_s)
    stdout_path = raw_dir / f"{name}_stdout.log"
    stderr_path = raw_dir / f"{name}_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            timeout=timeout_s,
            check=False,
        )


def record_gpu_preflight(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "gpu_preflight.log").open("w", encoding="utf-8") as handle:
        for command in (
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate", "--format=csv"],
            ["nvidia-smi", "pmon", "-c", "1"],
        ):
            handle.write("$ " + " ".join(command) + "\n")
            proc = subprocess.run(command, text=True, capture_output=True, check=False)
            handle.write(proc.stdout)
            if proc.stderr:
                handle.write("\n[stderr]\n" + proc.stderr)
            handle.write(f"\n[returncode] {proc.returncode}\n\n")


def patch_epoches(config_path: Path, epoches: int) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    patched = re.sub(r"(^\s*epoches:\s*)\d+", rf"\g<1>{epoches}", text, flags=re.M)
    changed = patched != text
    if changed:
        config_path.write_text(patched, encoding="utf-8")
    return {"config_path": str(config_path), "target_epoches": epoches, "changed": changed}


def flat_ckpt_check(ckpt_path: Path) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {"ckpt_path": str(ckpt_path), "exists": ckpt_path.exists(), "action": "none"}
    if not ckpt_path.exists():
        payload["status"] = "missing"
        return payload
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        torch.save(obj["model_state_dict"], ckpt_path)
        payload.update({"status": "unwrapped", "action": "unwrap_model_state_dict"})
    elif isinstance(obj, dict):
        payload.update({"status": "flat", "num_keys": len(obj)})
    else:
        payload.update({"status": "unexpected_type", "type": type(obj).__name__})
    return payload


def epoch_from_ckpt(path: Path) -> int:
    match = re.search(r"(?:bestval_at|net_epoch)(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_best_ckpt(ckpt_dir: Path) -> Path | None:
    bestvals = list(ckpt_dir.glob("net_epoch_bestval_at*.pth"))
    if bestvals:
        return max(bestvals, key=epoch_from_ckpt)
    epochs = [p for p in ckpt_dir.glob("net_epoch*.pth") if "bestval" not in p.name]
    if epochs:
        return max(epochs, key=epoch_from_ckpt)
    return None


def checkpoint_manifest(ckpt_dir: Path, best_ckpt: Path, config_path: Path) -> dict[str, Any]:
    ckpts = sorted(ckpt_dir.glob("net_epoch*.pth"), key=lambda p: (epoch_from_ckpt(p), p.name))
    rows = [
        {
            "path": str(path),
            "name": path.name,
            "epoch": epoch_from_ckpt(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in ckpts
    ]
    return {
        "ckpt_dir": str(ckpt_dir),
        "best_ckpt": str(best_ckpt),
        "best_ckpt_sha256": sha256_file(best_ckpt),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoints": rows,
    }


def build_success_row(
    *,
    job: dict[str, Any],
    report: dict[str, Any],
    manifest: dict[str, Any],
    raw_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    label = str(job["label"])
    width = [int(item) for item in job["width"]]
    software_point_id = f"pyramid_lidar:full_model_ap:w{width[0]}x{width[1]}x{width[2]}:fp16"
    config_id = stable_config_id(
        model="pyramid_lidar",
        candidate_id=f"ap_stability:{label}",
        software_point_id=software_point_id,
        quant_policy="fp16",
        schedule_policy="not_applicable",
    )
    digest = source_digest(
        [
            manifest["best_ckpt_sha256"],
            manifest["config_sha256"],
            str(report.get("ap70")),
            str(report.get("n_samples")),
        ]
    )
    row = ap_anchor_row(
        config_id=config_id,
        model="pyramid_lidar",
        manifest_digest=digest,
        candidate_id=f"ap_stability:pyramid_lidar:{label}",
        software_point_id=software_point_id,
        dense_stage="backbone",
        optimized_scope="full_model_ap_eval",
        width=width,
        quant_policy="fp16",
        schedule_policy="not_applicable",
        hardware_target="not_hardware_specific",
        backend="model_eval",
        measurement_status="measured",
        metric="AP70",
        metric_value=float(report["ap70"]),
        secondary_metrics={"AP30": float(report["ap30"]), "AP50": float(report["ap50"])},
        dataset="DAIR-V2X",
        eval_split="val_1789",
        num_samples=int(report.get("n_samples", 1789)),
        ckpt_path=str(manifest["best_ckpt"]),
        ckpt_digest=str(manifest["best_ckpt_sha256"]),
        config_path=str(manifest["config_path"]),
        config_digest=str(manifest["config_sha256"]),
        finetune_protocol=AP_PROTOCOL,
        training_budget="init23_to_epoch31_8ep",
        eval_command="scripts/phase1/m4_8_hybrid_infer_ap.py --engine-collab ... --n-samples 1789 --dataset dair",
        run_id=f"ap_stability_20260626_finetune_{label}",
        created_at=created_at,
        source_files=[str(raw_dir / "ap_eval_report.json"), str(raw_dir / "best_ckpt_manifest.json")],
        raw_artifact=str(raw_dir),
        provenance="stage2_ap_stability_h800_finetune_smoke",
        notes="Fresh true_eval AP row from Phase C stable smoke; not predicted/model-fit/interpolated.",
        claim_status="claimable_true_eval",
        ap_source_kind="true_eval",
        gpu_id=int(job["gpu_id"]),
        finetune_runs=int(job.get("finetune_runs", 1)),
        epoches=int(job.get("epoches", 31)),
    )
    validate_lut_row(row)
    return row


class JobFailure(RuntimeError):
    pass


def run_job(
    job: dict[str, Any],
    *,
    args: argparse.Namespace,
    created_at: str,
    rows_out: Path,
    state_out: Path,
    quarantine_out: Path,
    lock: threading.Lock,
) -> dict[str, Any]:
    label = str(job["label"])
    gpu_id = int(job["gpu_id"])
    width = [int(item) for item in job["width"]]
    raw_dir = ROOT / str(job["raw_dir"])
    ckpt_dir = Path(str(job["ckpt_dir"]))
    config_path = ckpt_dir / "config.yaml"
    env = command_env(gpu_id, args.heal_root)

    def state(status: str, **extra: Any) -> None:
        append_jsonl(
            state_out,
            {
                "schema": "stage2_ap_finetune_smoke_job_state_v1",
                "job_id": job["job_id"],
                "label": label,
                "width": width,
                "gpu_id": gpu_id,
                "status": status,
                "attempt": 1,
                "created_at": utc_timestamp(),
                **extra,
            },
            lock,
        )

    def quarantine(reason: str, **extra: Any) -> None:
        append_jsonl(
            quarantine_out,
            {
                "schema": "stage2_ap_quarantine_row_v1",
                "job_id": job["job_id"],
                "label": label,
                "width": width,
                "quarantine_status": "active",
                "claim_status": "no_claim_missing_source",
                "failure_reason": reason,
                "created_at": utc_timestamp(),
                "raw_artifact": str(raw_dir),
                **extra,
            },
            lock,
        )

    state("running", stage="start", raw_artifact=str(raw_dir))
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
        record_gpu_preflight(raw_dir)

        prune_cmd = [
            args.env_python,
            "tools/structural_prune_pyramid.py",
            "--orig-dir",
            args.base_ckpt_dir,
            "--out-dir",
            str(ckpt_dir),
            "--num-filters-new",
            ",".join(str(item) for item in width),
            "--width-per-group",
            "4",
            "--groups",
            "32",
        ]
        proc = run_logged(name="prune", command=prune_cmd, cwd=ROOT, env=env, raw_dir=raw_dir, timeout_s=900)
        if proc.returncode != 0 or not (ckpt_dir / "net_epoch_bestval_at23.pth").exists():
            raise JobFailure(f"prune_failed_rc_{proc.returncode}")

        flat = flat_ckpt_check(ckpt_dir / "net_epoch_bestval_at23.pth")
        write_json(raw_dir / "flat_ckpt_check.json", flat)
        if flat["status"] not in {"flat", "unwrapped"}:
            raise JobFailure(f"flat_ckpt_check_failed:{flat['status']}")

        patch = patch_epoches(config_path, int(job.get("epoches", 31)))
        write_json(raw_dir / "config_patch.json", patch)

        train_cmd = [
            args.env_python,
            "-m",
            "torch.distributed.launch",
            "--nproc_per_node=1",
            "--use_env",
            f"--master_port={int(job['master_port'])}",
            str(Path(args.heal_root) / "opencood/tools/train_ddp.py"),
            "--hypes_yaml",
            str(config_path),
            "--model_dir",
            str(ckpt_dir),
            "--half",
        ]
        proc = run_logged(name="train", command=train_cmd, cwd=Path(args.heal_root), env=env, raw_dir=raw_dir, timeout_s=28800)
        post_ckpts = [p for p in ckpt_dir.glob("net_epoch*.pth") if epoch_from_ckpt(p) > int(job.get("init_epoch", 23))]
        if proc.returncode != 0 and not post_ckpts:
            raise JobFailure(f"train_failed_rc_{proc.returncode}")

        best_ckpt = find_best_ckpt(ckpt_dir)
        if best_ckpt is None:
            raise JobFailure("no_checkpoint_after_train")

        onnx_path = raw_dir / f"{label}.onnx"
        export_cmd = [
            args.env_python,
            str(ROOT / "tools/export_onnx_pyramid_collab.py"),
            "--ckpt",
            str(best_ckpt),
            "--hypes",
            str(config_path),
            "--out",
            str(onnx_path),
            "--feat-h",
            "128",
            "--feat-w",
            "256",
        ]
        proc = run_logged(name="export", command=export_cmd, cwd=ROOT, env=env, raw_dir=raw_dir, timeout_s=600)
        if proc.returncode != 0 or not onnx_path.exists():
            raise JobFailure(f"onnx_export_failed_rc_{proc.returncode}")

        engine_path = raw_dir / f"{label}_fp16.engine"
        trt_report = raw_dir / "trt_build_report.json"
        trt_cmd = [
            args.env_python,
            str(ROOT / "scripts/phase1/m4_8_trt_build_bench.py"),
            "--onnx",
            str(onnx_path),
            "--precision",
            "fp16",
            "--engine",
            str(engine_path),
            "--report",
            str(trt_report),
            "--input-shape",
            "2,64,128,256",
            "--extra-input-shape",
            "t_ego:2,2,3",
            "--n-warmup",
            "100",
            "--n-measure",
            "200",
        ]
        proc = run_logged(name="trt_build", command=trt_cmd, cwd=ROOT, env=env, raw_dir=raw_dir, timeout_s=900)
        if proc.returncode != 0 or not engine_path.exists() or not trt_report.exists():
            raise JobFailure(f"trt_build_failed_rc_{proc.returncode}")

        ap_report = raw_dir / "ap_eval_report.json"
        ap_cmd = [
            args.env_python,
            str(ROOT / "scripts/phase1/m4_8_hybrid_infer_ap.py"),
            "--engine-collab",
            str(engine_path),
            "--tag",
            f"stage2_ap_{label}_fp16",
            "--model-dir",
            str(ckpt_dir),
            "--n-samples",
            "1789",
            "--dataset",
            "dair",
            "--range",
            "102.4,51.2",
            "--collab-spatial-shape",
            "2,64,128,256",
            "--collab-tego-shape",
            "2,2,3",
            "--report",
            str(ap_report),
        ]
        proc = run_logged(name="ap_eval", command=ap_cmd, cwd=Path(args.heal_root), env=env, raw_dir=raw_dir, timeout_s=3600)
        if proc.returncode != 0 or not ap_report.exists():
            raise JobFailure(f"ap_eval_failed_rc_{proc.returncode}")

        report = json.loads(ap_report.read_text(encoding="utf-8"))
        for key in ("ap30", "ap50", "ap70"):
            if key not in report:
                raise JobFailure(f"ap_report_missing_{key}")
        if int(report.get("n_samples", 0)) != 1789:
            raise JobFailure(f"ap_report_wrong_n_samples:{report.get('n_samples')}")

        manifest = checkpoint_manifest(ckpt_dir, best_ckpt, config_path)
        write_json(raw_dir / "best_ckpt_manifest.json", manifest)
        row = build_success_row(job=job, report=report, manifest=manifest, raw_dir=raw_dir, created_at=created_at)
        append_jsonl(rows_out, row, lock)
        state("succeeded", stage="done", raw_artifact=str(raw_dir), ap70=float(report["ap70"]), row_id=row["row_id"])
        return {"label": label, "status": "succeeded", "ap70": float(report["ap70"])}
    except Exception as exc:
        reason = f"{type(exc).__name__}:{exc}"
        state("failed", stage="failed", failure_reason=reason, raw_artifact=str(raw_dir))
        quarantine(reason)
        return {"label": label, "status": "failed", "failure_reason": reason}


def main() -> int:
    args = parse_args()
    base_dir = ROOT / args.base_dir
    job_queue = Path(args.job_queue) if args.job_queue else base_dir / "jobs/ap_finetune_smoke_queue_v1.jsonl"
    job_state = Path(args.job_state) if args.job_state else base_dir / "jobs/ap_finetune_smoke_job_state_v1.jsonl"
    rows_out = Path(args.rows_out) if args.rows_out else base_dir / "rows/ap_anchor_rows_v1.jsonl"
    quarantine_out = Path(args.quarantine_out) if args.quarantine_out else base_dir / "quarantine/ap_unstable_or_unclaimable_v1.jsonl"
    created_at = args.created_at or utc_timestamp()
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}

    jobs = load_jsonl(job_queue)
    if labels:
        jobs = [job for job in jobs if str(job.get("label")) in labels]
    existing_labels = {
        str(row.get("candidate_id", "")).split(":")[-1]
        for row in load_jsonl(rows_out)
        if row.get("schema") == "ap_anchor_row_v1"
    }
    if not args.force:
        jobs = [job for job in jobs if str(job.get("label")) not in existing_labels]

    print(json.dumps({"execute": args.execute, "jobs_to_run": [job.get("label") for job in jobs]}, ensure_ascii=False))
    if not args.execute:
        return 0
    if not jobs:
        return 0

    lock = threading.Lock()
    max_workers = min(max(1, int(args.max_workers)), len(jobs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_job,
                job,
                args=args,
                created_at=created_at,
                rows_out=rows_out,
                state_out=job_state,
                quarantine_out=quarantine_out,
                lock=lock,
            )
            for job in jobs
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    print(json.dumps({"results": sorted(results, key=lambda row: row["label"])}, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "succeeded" for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
