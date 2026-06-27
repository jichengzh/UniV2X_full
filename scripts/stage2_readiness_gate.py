#!/usr/bin/env python3
"""Offline readiness dry-run gate for Stage2 P/Q/S LUT production."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.artifact_registry import (  # noqa: E402
    artifact_ready_for_config,
    validate_artifact_registry_row,
)
from framework.stage2.evidence_registry import (  # noqa: E402
    EvidenceRegistryError,
    Stage2EvidenceRegistry,
)
from framework.stage2.lut_productization import (  # noqa: E402
    energy_claim_allowed_from_rows,
    read_jsonl,
    validate_job_plan_row,
    validate_lut_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-registry", required=True)
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--ap-rows", action="append", default=[])
    parser.add_argument("--energy-rows", action="append", default=[])
    parser.add_argument("--job-plan")
    parser.add_argument("--evidence-registry")
    parser.add_argument("--outlier-report")
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def _read_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _gate(status: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {"status": status, **detail}


def _schema_gate(
    *,
    artifact_rows: list[dict[str, Any]],
    lut_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    for row in artifact_rows:
        try:
            validate_artifact_registry_row(row)
        except Exception as exc:  # pragma: no cover - detail is reported
            errors.append(str(exc))
    for row in lut_rows:
        try:
            validate_lut_row(row)
        except Exception as exc:  # pragma: no cover - detail is reported
            errors.append(str(exc))
    for row in job_rows:
        try:
            validate_job_plan_row(row)
        except Exception as exc:  # pragma: no cover - detail is reported
            errors.append(str(exc))
    return _gate(
        "pass" if not errors else "fail",
        {
            "artifact_rows": len(artifact_rows),
            "lut_rows": len(lut_rows),
            "job_rows": len(job_rows),
            "errors": errors[:20],
        },
    )


def _artifact_gate(
    artifact_rows: list[dict[str, Any]],
    job_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    not_ready = [
        row
        for row in artifact_rows
        if row.get("artifact_status") != "ready"
    ]
    blocked_jobs = [
        str(job.get("job_id"))
        for job in job_rows
        if not artifact_ready_for_config(artifact_rows, str(job.get("config_id")))
    ]
    status = "pass" if not not_ready and not blocked_jobs else "fail"
    return _gate(
        status,
        {
            "total_artifacts": len(artifact_rows),
            "not_ready_artifacts": len(not_ready),
            "blocked_jobs": blocked_jobs,
        },
    )


def _energy_gate(energy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    claimable = energy_claim_allowed_from_rows(energy_rows)
    return _gate(
        "pass" if claimable else "fail",
        {
            "energy_rows": len(energy_rows),
            "claimable": claimable,
        },
    )


def _outlier_gate(path: str | None) -> dict[str, Any]:
    if not path:
        return _gate("fail", {"reason": "missing_outlier_report"})
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    unstable = int(report.get("unstable_rows", 0))
    return _gate(
        "pass" if unstable == 0 else "conditional",
        {
            "unstable_rows": unstable,
            "claimable_rows": int(report.get("claimable_rows", 0)),
            "report": path,
        },
    )


def _evidence_registry_gate(path: str | None) -> dict[str, Any]:
    if not path:
        return _gate("not_applicable", {"reason": "not_provided"})
    try:
        registry = Stage2EvidenceRegistry.from_file(path)
    except (EvidenceRegistryError, OSError, ValueError) as exc:
        return _gate("fail", {"path": path, "error": str(exc)})
    return _gate(
        "pass",
        {
            "path": path,
            "model": registry.model,
            "energy_claim_allowed": registry.energy_claim_allowed,
            "coverage_summary": registry.coverage_summary(),
            "unsupported_conclusions": registry.unsupported_conclusions,
        },
    )


def _decision(gates: dict[str, dict[str, Any]]) -> str:
    if any(gate["status"] == "fail" for gate in gates.values()):
        return "NO_GO"
    if any(gate["status"] == "conditional" for gate in gates.values()):
        return "CONDITIONAL_GO"
    return "GO"


def main() -> int:
    args = parse_args()
    artifact_rows = read_jsonl(args.artifact_registry)
    latency_rows = _read_many(args.latency_rows)
    ap_rows = _read_many(args.ap_rows)
    energy_rows = _read_many(args.energy_rows)
    job_rows = read_jsonl(args.job_plan) if args.job_plan else []
    gates = {
        "schema": _schema_gate(
            artifact_rows=artifact_rows,
            lut_rows=[*latency_rows, *ap_rows, *energy_rows],
            job_rows=job_rows,
        ),
        "artifact_registry": _artifact_gate(artifact_rows, job_rows),
        "energy": _energy_gate(energy_rows),
        "outlier": _outlier_gate(args.outlier_report),
        "evidence_registry": _evidence_registry_gate(args.evidence_registry),
    }
    report = {
        "schema": "stage2_three_arm_readiness_gate_v1",
        "decision": _decision(gates),
        "gates": gates,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
