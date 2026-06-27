#!/usr/bin/env python3
"""Preflight H800 AP finetune/eval environment for Stage2 AP stability.

The script intentionally checks the full AP chain dependencies, not only GPU
availability. A passing result means it is reasonable to start Phase C AP
finetune smoke; a blocked result must not produce measured AP rows.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ENV_PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
DEFAULT_HEAL_ROOT = "/home/jichengzhi/heal_research/HEAL"
DEFAULT_BASE_CKPT_DIR = (
    "/home/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)
DEFAULT_REPO_ROOT = "/home/jichengzhi/V2X"
DEFAULT_DATASET_ROOT = (
    "/home/jichengzhi/heal_research/HEAL/dataset/my_dair_v2x/"
    "v2x_c/cooperative-vehicle-infrastructure"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-python", default=DEFAULT_ENV_PYTHON)
    parser.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    parser.add_argument("--base-ckpt-dir", default=DEFAULT_BASE_CKPT_DIR)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args()


def exists_record(path: str, *, kind: str) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    kind_ok = p.is_dir() if kind == "dir" else p.is_file()
    return {
        "path": path,
        "exists": exists,
        "kind": kind,
        "kind_ok": bool(kind_ok),
        "resolved": str(p.resolve()) if exists else None,
    }


def run_command(command: list[str], env: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
            env=env,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive on remote hosts.
        return {
            "command": command,
            "returncode": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }


def python_import_check(env_python: str, heal_root: str) -> dict[str, Any]:
    code = (
        "import json\n"
        "mods=['torch','numpy','yaml','opencood']\n"
        "out={}\n"
        "for m in mods:\n"
        "    try:\n"
        "        mod=__import__(m)\n"
        "        out[m]={'ok': True, 'version': getattr(mod, '__version__', ''), 'file': getattr(mod, '__file__', '')}\n"
        "    except Exception as exc:\n"
        "        out[m]={'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}\n"
        "print(json.dumps(out, sort_keys=True))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = heal_root + os.pathsep + env.get("PYTHONPATH", "")
    result = run_command([env_python, "-c", code], env=env)
    try:
        imports = json.loads(result["stdout"]) if result["stdout"] else {}
    except json.JSONDecodeError:
        imports = {}
    return {"process": result, "imports": imports}


def gpu_check() -> dict[str, Any]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate",
            "--format=csv",
        ]
    )
    pmon = run_command(["nvidia-smi", "pmon", "-c", "1"])
    return {"query": query, "pmon": pmon}


def main() -> int:
    args = parse_args()
    paths = {
        "env_python": exists_record(args.env_python, kind="file"),
        "heal_root": exists_record(args.heal_root, kind="dir"),
        "train_ddp": exists_record(
            str(Path(args.heal_root) / "opencood/tools/train_ddp.py"),
            kind="file",
        ),
        "base_ckpt_dir": exists_record(args.base_ckpt_dir, kind="dir"),
        "base_ckpt": exists_record(
            str(Path(args.base_ckpt_dir) / "net_epoch_bestval_at23.pth"),
            kind="file",
        ),
        "base_config": exists_record(str(Path(args.base_ckpt_dir) / "config.yaml"), kind="file"),
        "structural_prune": exists_record(
            str(Path(args.repo_root) / "tools/structural_prune_pyramid.py"),
            kind="file",
        ),
        "dataset_root": exists_record(args.dataset_root, kind="dir"),
        "dataset_train": exists_record(str(Path(args.dataset_root) / "train.json"), kind="file"),
        "dataset_val": exists_record(str(Path(args.dataset_root) / "val.json"), kind="file"),
    }
    path_ok = all(item["exists"] and item["kind_ok"] for item in paths.values())
    imports = python_import_check(args.env_python, args.heal_root) if paths["env_python"]["kind_ok"] else {
        "process": {"returncode": -1, "stdout": "", "stderr": "env python missing"},
        "imports": {},
    }
    import_ok = bool(imports["imports"]) and all(
        item.get("ok") for item in imports["imports"].values()
    )
    gpu = None if args.skip_gpu else gpu_check()
    gpu_ok = True if args.skip_gpu else bool(gpu and gpu["query"]["returncode"] == 0)
    blocked_reasons = []
    for name, item in paths.items():
        if not (item["exists"] and item["kind_ok"]):
            blocked_reasons.append(f"missing_or_wrong_kind:{name}:{item['path']}")
    if not import_ok:
        blocked_reasons.append("python_import_check_failed")
    if not gpu_ok:
        blocked_reasons.append("gpu_query_failed")
    payload = {
        "schema": "stage2_h800_ap_env_preflight_v1",
        "status": "pass" if not blocked_reasons else "blocked",
        "blocked_reasons": blocked_reasons,
        "paths": paths,
        "python_import_check": imports,
        "gpu_check": gpu,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
