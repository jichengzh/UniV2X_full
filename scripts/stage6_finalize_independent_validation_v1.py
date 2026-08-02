#!/usr/bin/env python3
"""Finalize arm-scoped Stage6 independent reruns into a SHA audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Stage5's finalizer owns the normalized measurement/result contract. Reuse
# those pure helpers here; the Stage5 request module only constructs requests.
from scripts.stage5_finalize_independent_validation_v1 import (  # noqa: E402
    evaluate_consistency,
    normalize_ap_report,
    normalize_performance_repeat,
    successful_state_for_configuration,
    unique_successful_ap_terminal,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalized_file(root: Path, name: str, payload: Mapping[str, Any]) -> dict[str, str]:
    path = root / f"{name}.json"
    _write(path, payload)
    return {"path": str(path), "sha256": _sha(path)}


def _validation_task_id(plan: Mapping[str, Any]) -> str:
    backend = str(plan.get("backend") or "")
    arm_id = str(plan.get("arm_id") or "")
    if backend not in {"tvm", "trt"} or not arm_id:
        raise ValueError("Stage6 validation plan has an invalid arm scope")
    return f"{backend}:{arm_id}"


def _configuration(
    *,
    plan: Mapping[str, Any],
    configuration_id: str,
    reference: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    arm_root = Path(str(plan["root"]))
    repeats = []
    normalized_root = output_root / "normalized" / str(plan["backend"]) / str(plan["arm_id"])
    safe_id = hashlib.sha256(configuration_id.encode()).hexdigest()[:16]
    normalized_root = normalized_root / safe_id
    for repeat_index in range(3):
        repeat_root = arm_root / f"repeat_{repeat_index}"
        jobs = _jsonl(repeat_root / "performance_jobs.jsonl")
        states = _jsonl(repeat_root / "performance_state.jsonl")
        state = successful_state_for_configuration(
            configuration_id, jobs=jobs, states=states
        )
        normalized = normalize_performance_repeat(
            task_id=_validation_task_id(plan),
            configuration_id=configuration_id,
            state=state,
            repeat_index=repeat_index,
        )
        bound = _normalized_file(
            normalized_root, f"performance_repeat_{repeat_index}", normalized
        )
        repeats.append(
            {
                "repeat_id": normalized["repeat_id"],
                "performance_result_json": bound["path"],
                "performance_result_sha256": bound["sha256"],
            }
        )

    ap_states = _jsonl(arm_root / "ap/ap_state.jsonl")
    terminal = unique_successful_ap_terminal(configuration_id, ap_states)
    ap_normalized = normalize_ap_report(
        task_id=_validation_task_id(plan),
        configuration_id=configuration_id,
        terminal=terminal,
    )
    ap_bound = _normalized_file(normalized_root, "full_ap", ap_normalized)
    normalized_repeats = [
        _read(Path(item["performance_result_json"])) for item in repeats
    ]
    consistency = evaluate_consistency(
        {
            "latency_ms": float(reference.get("search_latency_ms", reference["latency_ms"])),
            "energy_j": float(reference.get("search_energy_j", reference["energy_j"])),
            "ap70": float(reference.get("search_AP70", reference["AP70"])),
        },
        normalized_repeats,
        ap_normalized,
        raise_on_failure=False,
    )
    source_path = Path(str((reference.get("evidence_files") or {}).get("source") or ""))
    if not source_path.is_file():
        raise ValueError(f"independent source evidence is missing: {configuration_id}")
    return {
        "arm_id": str(plan["arm_id"]),
        "pipeline_id": str(plan["pipeline_id"]),
        "configuration_id": configuration_id,
        "consistency": consistency,
        "performance_repeats": repeats,
        "ap_report_path": ap_bound["path"],
        "ap_report_sha256": ap_bound["sha256"],
        "evidence_path": str(source_path),
        "evidence_sha256": _sha(source_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _read(args.plan_json)
    if manifest.get("schema_version") != "stage6_independent_validation_plan_v1":
        raise ValueError("unexpected Stage6 independent validation plan schema")
    tasks = []
    for plan in manifest.get("plans") or []:
        references = {
            str(point["manifest_job_id"]): point
            for point in _read(Path(str(plan["root"])) / "reference_points.json")
        }
        configurations = [
            _configuration(
                plan=plan,
                configuration_id=str(configuration_id),
                reference=references[str(configuration_id)],
                output_root=args.output_json.parent,
            )
            for configuration_id in plan["configuration_ids"]
        ]
        tasks.append(
            {
                "task_id": f"{plan['backend']}:{plan['arm_id']}",
                "arm_id": plan["arm_id"],
                "backend": plan["backend"],
                "passed": all(
                    configuration["consistency"]["passed"]
                    for configuration in configurations
                ),
                "configurations": configurations,
            }
        )
    audit = {
        "schema_version": "stage6_independent_validation_audit_v1",
        "all_tasks_passed": bool(tasks) and all(task["passed"] for task in tasks),
        "task_count": len(tasks),
        "configuration_count": sum(len(task["configurations"]) for task in tasks),
        "tasks": tasks,
    }
    _write(args.output_json, audit)
    print(json.dumps({"output_json": str(args.output_json), **{key: audit[key] for key in ("all_tasks_passed", "task_count", "configuration_count")}}, sort_keys=True))
    return 0 if audit["all_tasks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
