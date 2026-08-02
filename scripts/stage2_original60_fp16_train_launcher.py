#!/usr/bin/env python3
"""Launch one original60 FP16 checkpoint-generation finetune job."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_BASE_CKPT_DIR = (
    "/home/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--width", required=True, help="comma-separated widths, e.g. 24,64,128")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--base-ckpt-dir", default=DEFAULT_BASE_CKPT_DIR)
    parser.add_argument("--epoches", type=int, default=31)
    parser.add_argument("--width-per-group", type=int, default=4)
    parser.add_argument("--groups", type=int, default=32)
    parser.add_argument("--force-prune", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def patch_epoches(config_path: Path, epoches: int) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    patched = re.sub(r"(^\s*epoches:\s*)\d+", rf"\g<1>{epoches}", text, flags=re.M)
    changed = patched != text
    if changed:
        config_path.write_text(patched, encoding="utf-8")
    return {"config_path": str(config_path), "target_epoches": epoches, "changed": changed}


def flatten_init_ckpt(ckpt_path: Path) -> dict[str, Any]:
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


def run_logged(command: list[str], cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path, timeout_s: int) -> subprocess.CompletedProcess[str]:
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


def post_init_checkpoint_exists(ckpt_dir: Path) -> bool:
    for path in ckpt_dir.glob("net_epoch*.pth"):
        match = re.search(r"(?:bestval_at|net_epoch)(\d+)", path.name)
        if match and int(match.group(1)) > 23:
            return True
    return False


def build_env(*, gpu_id: int, heal_root: Path, env_python: str) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(heal_root) + os.pathsep + env.get("PYTHONPATH", "")
    python_bin_dir = str(Path(env_python).resolve().parent)
    current_path = env.get("PATH", "")
    env["PATH"] = python_bin_dir if not current_path else python_bin_dir + os.pathsep + current_path
    return env


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    heal_root = Path(args.heal_root).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(gpu_id=args.gpu_id, heal_root=heal_root, env_python=args.env_python)

    write_json(
        raw_dir / "launcher_command.json",
        {
            "schema": "stage2_original60_fp16_train_launcher_v1",
            "label": args.label,
            "width": args.width,
            "gpu_id": args.gpu_id,
            "master_port": args.master_port,
            "ckpt_dir": str(ckpt_dir),
            "raw_dir": str(raw_dir),
            "heal_root": str(heal_root),
            "base_ckpt_dir": args.base_ckpt_dir,
            "epoches": args.epoches,
            "width_per_group": args.width_per_group,
            "groups": args.groups,
            "cwd": str(heal_root),
            "pythonpath": env["PYTHONPATH"],
        },
    )

    init_ckpt = ckpt_dir / "net_epoch_bestval_at23.pth"
    config_path = ckpt_dir / "config.yaml"
    if args.force_prune or not init_ckpt.exists():
        prune_cmd = [
            args.env_python,
            str(ROOT / "tools/structural_prune_pyramid.py"),
            "--orig-dir",
            args.base_ckpt_dir,
            "--out-dir",
            str(ckpt_dir),
            "--num-filters-new",
            args.width,
            "--width-per-group",
            str(args.width_per_group),
            "--groups",
            str(args.groups),
        ]
        write_json(raw_dir / "structural_prune_command.json", {"command": prune_cmd, "cwd": str(ROOT)})
        prune_proc = run_logged(
            prune_cmd,
            cwd=ROOT,
            env=env,
            stdout_path=raw_dir / "structural_prune_stdout.txt",
            stderr_path=raw_dir / "structural_prune_stderr.txt",
            timeout_s=1800,
        )
        if prune_proc.returncode != 0 or not init_ckpt.exists():
            return 2

    flatten_payload = flatten_init_ckpt(init_ckpt)
    write_json(raw_dir / "init_ckpt_flatten.json", flatten_payload)
    patch_payload = patch_epoches(config_path, args.epoches)
    write_json(raw_dir / "config_patch.json", patch_payload)

    if post_init_checkpoint_exists(ckpt_dir):
        write_json(
            raw_dir / "train_skip.json",
            {
                "reason": "post_init_checkpoint_exists",
                "ckpt_dir": str(ckpt_dir),
            },
        )
        return 0

    train_cmd = [
        args.env_python,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node=1",
        "--use_env",
        f"--master_port={args.master_port}",
        str(heal_root / "opencood/tools/train_ddp.py"),
        "--hypes_yaml",
        str(config_path),
        "--model_dir",
        str(ckpt_dir),
        "--half",
    ]
    write_json(
        raw_dir / "train_command.json",
        {"command": train_cmd, "cwd": str(heal_root), "env": {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"], "PYTHONPATH": env["PYTHONPATH"]}},
    )
    stdout_handle = (raw_dir / "train_stdout.txt").open("w", encoding="utf-8")
    stderr_handle = (raw_dir / "train_stderr.txt").open("w", encoding="utf-8")
    process = subprocess.Popen(
        train_cmd,
        cwd=str(heal_root),
        env=env,
        stdout=stdout_handle,
        stderr=stderr_handle,
        start_new_session=True,
    )
    (raw_dir / "train_runner.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
