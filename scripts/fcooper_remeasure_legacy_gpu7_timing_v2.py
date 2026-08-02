#!/usr/bin/env python3
"""Remeasure missing monotonic GPU7 timing without replacing selected metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence


PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"
SUCCESS = "completed_exclusive"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_option(command: list[str], option: str, value: Path) -> bool:
    if option not in command:
        return False
    index = command.index(option) + 1
    if index >= len(command):
        raise ValueError(f"{option} has no value")
    command[index] = str(value)
    return True


def rewrite_command_outputs(
    command: Sequence[str],
    *,
    output_root: Path,
    replacement_engine: Path | None,
) -> list[str]:
    rewritten = [str(value) for value in command]
    replaced = 0
    replaced += int(
        _replace_option(rewritten, "--output-json", output_root / "repeat.json")
    )
    replaced += int(
        _replace_option(rewritten, "--artifact-dir", output_root / "engine")
    )
    replaced += int(
        _replace_option(rewritten, "--out", output_root / "performance.json")
    )
    if replacement_engine is not None:
        if not _replace_option(rewritten, "--engine", replacement_engine):
            raise ValueError("AP timing command is missing --engine")
        if "--output-json" in rewritten:
            _replace_option(rewritten, "--output-json", output_root / "ap.json")
    if replaced < 1:
        raise ValueError("guarded command has no recognized output option")
    return rewritten


def _atomic_write(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _target_specs(root: Path) -> list[dict[str, object]]:
    old = root / "controls/resource_audit/gpu7_exclusivity"
    new = root / "controls/resource_audit/gpu7_timing_remeasurement"
    specs: list[dict[str, object]] = []
    for label in ("original_default", "schedule_only"):
        for index in range(3):
            specs.append(
                {
                    "label": f"{label}_repeat_{index}",
                    "old_audit": old / f"{label}_repeat_{index}_guard.json",
                    "output_root": new / f"{label}_repeat_{index}",
                    "replacement_engine": None,
                    "cwd": None,
                    "environment": (
                        {
                            "CUDA_VISIBLE_DEVICES": "7",
                            "PYTHONPATH": (
                                "/home/jichengzhi/V2X:"
                                "/exdata/jichengzhi/heal_research/HEAL"
                            ),
                        }
                        if label == "original_default"
                        else {}
                    ),
                }
            )
    specs.append(
        {
            "label": "schedule_only_full_ap",
            "old_audit": old / "schedule_only_full_ap_guard.json",
            "output_root": new / "schedule_only_full_ap",
            "replacement_engine": (
                new / "schedule_only_repeat_0/engine/compiled.engine"
            ),
            "cwd": Path("/exdata/jichengzhi/heal_research/HEAL"),
            "environment": {
                "CUDA_VISIBLE_DEVICES": "7",
                "PYTHONPATH": (
                    "/home/jichengzhi/V2X:"
                    "/exdata/jichengzhi/heal_research/HEAL"
                ),
            },
        }
    )
    return specs


def _validate_old_audit(path: Path) -> dict[str, Any]:
    audit = _read(path)
    if (
        audit.get("schema_version") != "fcooper_gpu_exclusivity_gate_v1"
        or audit.get("status") != SUCCESS
        or audit.get("evidence_scope") != "sampled_process_exclusivity"
        or int(audit.get("gpu_index", -1)) != 7
        or int(audit.get("return_code", -1)) != 0
        or not isinstance(audit.get("runtime_sample_count"), int)
        or int(audit["runtime_sample_count"]) < 1
        or list(audit.get("runtime_observations") or [])
        or list(audit.get("residual_processes") or [])
        or not list(audit.get("command") or [])
    ):
        raise ValueError(f"legacy GPU7 audit is invalid: {path}")
    if PILOT_FRAGMENT in json.dumps(audit, sort_keys=True):
        raise ValueError("legacy GPU7 audit references pilot data")
    return audit


def validate_timing_guard(
    audit: Mapping[str, Any],
    *,
    expected_runtime: float | None = None,
) -> float:
    runtime = audit.get("runtime_seconds")
    valid_runtime = (
        isinstance(runtime, (int, float))
        and math.isfinite(float(runtime))
        and float(runtime) > 0.0
    )
    if expected_runtime is not None:
        valid_runtime = valid_runtime and math.isclose(
            float(runtime),
            float(expected_runtime),
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
    if not valid_runtime:
        raise ValueError("GPU7 timing guard runtime is invalid")
    if (
        audit.get("schema_version") != "fcooper_gpu_exclusivity_gate_v1"
        or audit.get("status") != SUCCESS
        or audit.get("evidence_scope") != "sampled_process_exclusivity"
        or int(audit.get("gpu_index", -1)) != 7
        or int(audit.get("return_code", -1)) != 0
        or not isinstance(audit.get("runtime_sample_count"), int)
        or int(audit["runtime_sample_count"]) < 1
        or list(audit.get("runtime_observations") or [])
        or list(audit.get("residual_processes") or [])
        or not list(audit.get("command") or [])
    ):
        raise ValueError("GPU7 timing guard is not valid exclusive evidence")
    return float(runtime)


def _run_guard(
    *,
    python: Path,
    gate: Path,
    lock_file: Path,
    audit_path: Path,
    log_path: Path,
    command: Sequence[str],
    cwd: Path | None,
    environment: Mapping[str, str],
) -> None:
    invocation = [
        str(python),
        str(gate),
        "guard",
        "--gpu-index",
        "7",
        "--timeout-seconds",
        "43200",
        "--poll-seconds",
        "30",
        "--quiet-seconds",
        "15",
        "--monitor-seconds",
        "0.5",
        "--runtime-timeout-seconds",
        "7200",
        "--lock-file",
        str(lock_file),
        "--audit-json",
        str(audit_path),
        "--log-file",
        str(log_path),
    ]
    if cwd is not None:
        invocation.extend(["--cwd", str(cwd)])
    for key, value in sorted(environment.items()):
        invocation.extend(["--env", f"{key}={value}"])
    invocation.extend(["--", *command])
    subprocess.run(invocation, stdin=subprocess.DEVNULL, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    expected_output = (
        root
        / "controls/resource_audit/gpu7_timing_remeasurement/summary.json"
    )
    if args.output_json.resolve() != expected_output:
        raise ValueError(f"--output-json must be {expected_output}")
    if args.output_json.exists():
        raise FileExistsError(f"timing summary already exists: {args.output_json}")
    gate = args.code_root.resolve() / "scripts/fcooper_gpu_exclusivity_gate_v1.py"
    lock = root / "controls/resource_audit/gpu7_exclusivity/gpu7.guard.lock"
    records: list[dict[str, object]] = []
    for spec in _target_specs(root):
        old_path = Path(spec["old_audit"])
        old_audit = _validate_old_audit(old_path)
        output_root = Path(spec["output_root"])
        output_root.mkdir(parents=True, exist_ok=False)
        new_root = expected_output.parent
        audit_path = new_root / f"{spec['label']}_guard.json"
        log_path = new_root / f"{spec['label']}.log"
        command = rewrite_command_outputs(
            old_audit["command"],
            output_root=output_root,
            replacement_engine=spec["replacement_engine"],
        )
        _run_guard(
            python=args.python.resolve(),
            gate=gate,
            lock_file=lock,
            audit_path=audit_path,
            log_path=log_path,
            command=command,
            cwd=spec["cwd"],
            environment=spec["environment"],
        )
        new_audit = _read(audit_path)
        runtime = validate_timing_guard(new_audit)
        records.append(
            {
                "label": spec["label"],
                "timing_kind": "independent_gpu7_monotonic_remeasurement",
                "runtime_seconds": runtime,
                "legacy_audit_path": str(old_path),
                "legacy_audit_sha256": _sha(old_path),
                "timing_audit_path": str(audit_path),
                "timing_audit_sha256": _sha(audit_path),
                "log_path": str(log_path),
                "log_sha256": _sha(log_path),
            }
        )
    payload = {
        "schema_version": "fcooper_legacy_gpu7_timing_remeasurement_v2",
        "status": "passed",
        "does_not_replace_selected_metrics": True,
        "records": records,
        "total_runtime_seconds": sum(
            float(record["runtime_seconds"]) for record in records
        ),
    }
    _atomic_write(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
