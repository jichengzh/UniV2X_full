#!/usr/bin/env python3
"""Collect TVM control feedback and prepare the automatic tuned four."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.stage5_advance_fcooper_round_v1 import (
    validate_fcooper_feedback_evidence,
)


SUCCESS = "measured_success_gold"
TUNED_TASK_ID = "S6-FCO-TVM-COMPRESS-THEN-TUNE-TUNED-V1"


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite drifted artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_request_rows(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    recorded = manifest.pop("manifest_sha256", None)
    if recorded != canonical_sha256(manifest):
        raise ValueError("control request manifest SHA drift")
    rows: list[dict[str, Any]] = []
    for entry in manifest.get("requests") or []:
        request_path = Path(str(entry["path"]))
        request = read_json(request_path)
        request_sha = request.pop("measurement_request_sha256", None)
        if (
            request_sha != canonical_sha256(request)
            or request_sha != entry.get("measurement_request_sha256")
        ):
            raise ValueError("control measurement request SHA drift")
        rows.extend(dict(row) for row in request.get("rows") or [])
    if len(rows) != int(manifest.get("row_count", -1)):
        raise ValueError("control request row count drift")
    return rows


def collect_feedback(
    request_rows: Sequence[Mapping[str, Any]],
    artifact_root: Path,
) -> list[dict[str, Any]]:
    expected = [str(row["row_id"]) for row in request_rows]
    by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(artifact_root).glob("execution/*/feedback_row.json")):
        try:
            row = read_json(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        row_id = str(row.get("row_id") or "")
        if row_id not in expected:
            continue
        if row_id in by_id:
            raise ValueError(f"duplicate control feedback row: {row_id}")
        by_id[row_id] = row
    missing = [row_id for row_id in expected if row_id not in by_id]
    if missing:
        raise ValueError(f"missing control feedback rows: {missing}")
    rows = [by_id[row_id] for row_id in expected]
    validate_fcooper_feedback_evidence(rows)
    return rows


def select_tuned_screen_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    ap70_ref: float,
    max_ap_drop: float,
) -> list[dict[str, Any]]:
    if len(rows) != 12 or len({str(row.get("row_id")) for row in rows}) != 12:
        raise ValueError("Compress -> Tune requires exactly 12 unique screen rows")
    successful = [dict(row) for row in rows if row.get("terminal_status") == SUCCESS]
    if len(successful) < 4:
        raise ValueError("fewer than four successful screen rows can be tuned")
    for row in successful:
        for metric in ("ap70", "latency_ms", "energy_j"):
            value = float(row.get(metric, math.nan))
            if not math.isfinite(value):
                raise ValueError(f"screen row has invalid {metric}")
    floor = float(ap70_ref) - float(max_ap_drop)
    feasible = [row for row in successful if float(row["ap70"]) >= floor]
    candidates = feasible if len(feasible) >= 4 else successful
    return sorted(
        candidates,
        key=lambda row: (
            float(row["latency_ms"]),
            float(row["energy_j"]),
            -float(row["ap70"]),
            str(row["row_id"]),
        ),
    )[:4]


def build_tuned_request(screen_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(screen_rows) != 4:
        raise ValueError("tuned remeasurement requires four selected screen rows")
    profile_ids = {str(row["capability_profile_id"]) for row in screen_rows}
    capability_digests = {str(row["capability_digest"]) for row in screen_rows}
    if len(profile_ids) != 1 or len(capability_digests) != 1:
        raise ValueError("selected screen rows have capability drift")
    task_contract = {
        "schema_version": "stage6_fcooper_tvm_control_task_contract_v1",
        "task_id": TUNED_TASK_ID,
        "arm_id": "compress_then_tune",
        "phase": "tuned_remeasurement",
        "model": "fcooper",
        "hardware_id": "h800",
        "backend": "tvm",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": next(iter(profile_ids)),
        "capability_digest": next(iter(capability_digests)),
        "tvm_trials": 64,
        "runner_request_kind": "stage6-tvm-control",
        "recovery_contract_required_for_pruned_rows": True,
    }
    task_sha = canonical_sha256(task_contract)
    rows = [
        {
            **dict(source),
            "task_id": TUNED_TASK_ID,
            "task_sha256": task_sha,
            "phase": "tuned_remeasurement",
            "tvm_trials": 64,
        }
        for source in screen_rows
    ]
    request = {
        "schema_version": "stage6_fcooper_tvm_control_measurement_request_v1",
        "task_id": TUNED_TASK_ID,
        "task_sha256": task_sha,
        "task_contract": task_contract,
        "arm_id": "compress_then_tune",
        "phase": "tuned_remeasurement",
        "request_index": 0,
        "batch_size": 4,
        "atomic_feedback": True,
        "real_h800_measurement_required": True,
        "backend": "tvm",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": next(iter(profile_ids)),
        "capability_digest": next(iter(capability_digests)),
        "tvm_trials": 64,
        "row_sha256": {
            str(row["row_id"]): canonical_sha256(row) for row in rows
        },
        "rows": rows,
    }
    request["measurement_request_sha256"] = canonical_sha256(request)
    return request


def prepare_tuned(
    *,
    screen_manifest: Path,
    artifact_root: Path,
    ap70_ref: float,
    screen_pool_json: Path,
    tuned_request_json: Path,
) -> dict[str, Any]:
    request_rows = manifest_request_rows(screen_manifest)
    feedback = collect_feedback(request_rows, artifact_root)
    screen_pool = {
        "schema_version": "stage6_fcooper_tvm_terminal_pool_v1",
        "arm_id": "compress_then_tune",
        "phase": "screen",
        "row_count": 12,
        "rows": feedback,
    }
    write_immutable(screen_pool_json, screen_pool)
    selected = select_tuned_screen_rows(
        feedback,
        ap70_ref=ap70_ref,
        max_ap_drop=0.10,
    )
    selected_ids = [str(row["row_id"]) for row in selected]
    source_by_id = {str(row["row_id"]): dict(row) for row in request_rows}
    request = build_tuned_request([source_by_id[row_id] for row_id in selected_ids])
    write_immutable(tuned_request_json, request)
    selection = {
        "schema_version": "stage6_fcooper_tvm_tuned_selection_v1",
        "automatic": True,
        "source_pool_path": str(screen_pool_json.resolve()),
        "source_pool_sha256": file_sha256(screen_pool_json),
        "selected_row_ids": selected_ids,
        "tuned_request_path": str(tuned_request_json.resolve()),
        "tuned_request_sha256": file_sha256(tuned_request_json),
    }
    write_immutable(tuned_request_json.parent / "automatic_selection.json", selection)
    return selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--ap70-ref", type=float, required=True)
    parser.add_argument("--screen-pool-json", type=Path, required=True)
    parser.add_argument("--tuned-request-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = prepare_tuned(
        screen_manifest=args.screen_manifest,
        artifact_root=args.artifact_root,
        ap70_ref=args.ap70_ref,
        screen_pool_json=args.screen_pool_json,
        tuned_request_json=args.tuned_request_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
