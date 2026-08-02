#!/usr/bin/env python3
"""Promote one terminal Stage5 batch to actual-feature online feedback."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage35_extract_onnx_graph_features_v1 import (  # noqa: E402
    graph_features as extract_graph_features,
)


SURROGATE_PROVENANCE = "coldstart_width_conditioned_surrogate_v1"
ACTUAL_PROVENANCE = "materialized_onnx_extracted_v1"
Extractor = Callable[[Path], Mapping[str, Any]]


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _verify_file(path_value: Any, sha_value: Any, label: str) -> Path:
    path = Path(str(path_value or ""))
    expected = str(sha_value or "")
    if not path.is_file():
        raise ValueError(f"{label} artifact missing: {path}")
    if len(expected) != 64 or _sha_file(path) != expected:
        raise ValueError(f"{label} SHA mismatch: {path}")
    return path


def _feedback_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("actual feedback promotion requires one four-row batch")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("feedback batch rows must be objects")
    return [dict(row) for row in rows]


def _validate_batch(
    request: Mapping[str, Any], feedback_rows: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    request_rows = request.get("rows")
    row_sha = request.get("row_sha256")
    if request.get("schema_version") != "stage5_measurement_request_v2":
        raise ValueError("unexpected measurement request schema")
    if not isinstance(request_rows, list) or len(request_rows) != 4:
        raise ValueError("measurement request must contain four rows")
    if not isinstance(row_sha, Mapping):
        raise ValueError("measurement request row SHA map missing")
    requested = {_row_id(row): dict(row) for row in request_rows}
    feedback = {_row_id(row): row for row in feedback_rows}
    if "" in requested or set(requested) != set(feedback) or set(row_sha) != set(requested):
        raise ValueError("request/feedback identity mismatch")
    pairs = []
    for row_id, request_row in requested.items():
        if str(row_sha[row_id]) != _sha_payload(request_row):
            raise ValueError(f"request row SHA mismatch: {row_id}")
        feedback_row = feedback[row_id]
        if feedback_row.get("measurement_request_row_sha256") != row_sha[row_id]:
            raise ValueError(f"feedback request SHA mismatch: {row_id}")
        candidate = request_row.get("graph_features")
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate graph features missing: {row_id}")
        if candidate.get("graph_feature_provenance") != SURROGATE_PROVENANCE:
            raise ValueError(f"unexpected candidate graph provenance: {row_id}")
        if feedback_row.get("graph_features") != candidate:
            raise ValueError(f"feedback graph snapshot drift: {row_id}")
        pairs.append((request_row, feedback_row))
    return pairs


def _materialized_graph(
    request_row: Mapping[str, Any], feedback_row: Mapping[str, Any], extractor: Extractor
) -> dict[str, Any]:
    row_id = _row_id(request_row)
    source_path = _verify_file(
        feedback_row.get("materialized_source_evidence_path"),
        feedback_row.get("materialized_source_evidence_sha256"),
        f"{row_id} source evidence",
    )
    source = _read(source_path)
    expected_width = "x".join(map(str, request_row.get("width") or []))
    expected = {
        "schema_version": "stage5_source_materialization_evidence_v1",
        "status": "ready",
        "group_id": request_row.get("group_id"),
        "model": request_row.get("model"),
        "width": expected_width,
        "source_plan_sha256": request_row.get("source_evidence_sha256"),
    }
    if not isinstance(source, Mapping) or any(source.get(key) != value for key, value in expected.items()):
        raise ValueError(f"materialized source contract mismatch: {row_id}")
    onnx_path = _verify_file(source.get("onnx_path"), source.get("onnx_sha256"), f"{row_id} ONNX")
    extracted = dict(extractor(onnx_path))
    if extracted.get("onnx_sha256") != source.get("onnx_sha256"):
        raise ValueError(f"extracted ONNX SHA mismatch: {row_id}")
    return {
        "schema": "stage5_actual_graph_features_v1",
        "group_id": request_row["group_id"],
        "model": request_row["model"],
        "width": list(request_row["width"]),
        **extracted,
        "graph_feature_provenance": ACTUAL_PROVENANCE,
    }


def promote_feedback_batch(
    request_path: Path,
    feedback_path: Path,
    *,
    extractor: Extractor = extract_graph_features,
) -> dict[str, Any]:
    request = _read(request_path)
    feedback_rows = _feedback_rows(_read(feedback_path))
    pairs = _validate_batch(request, feedback_rows)
    actual_by_group: dict[str, dict[str, Any]] = {}
    promoted_rows = []
    row_audit = []
    for requested, feedback in pairs:
        group_id = str(requested["group_id"])
        extracted_actual = _materialized_graph(requested, feedback, extractor)
        actual = actual_by_group.get(group_id)
        if actual is None:
            actual = extracted_actual
            actual_by_group[group_id] = actual
        elif actual != extracted_actual:
            raise ValueError(f"materialized graph identity drift: {group_id}")
        candidate = copy.deepcopy(dict(requested["graph_features"]))
        promoted = {
            **copy.deepcopy(feedback),
            "candidate_graph_features": candidate,
            "candidate_graph_features_sha256": _sha_payload(candidate),
            "graph_features": copy.deepcopy(actual),
            "materialized_graph_features_sha256": _sha_payload(actual),
            "historical_feedback_row_sha256": _sha_payload(feedback),
            "feedback_feature_contract": "actual_feedback_v3",
            "graph_feature_promotion_schema": "stage5_actual_feedback_promotion_v3",
        }
        promoted["actual_feedback_row_sha256"] = _sha_payload(promoted)
        promoted_rows.append(promoted)
        row_audit.append(
            {
                "manifest_job_id": _row_id(promoted),
                "group_id": group_id,
                "candidate_graph_features_sha256": promoted[
                    "candidate_graph_features_sha256"
                ],
                "materialized_graph_features_sha256": promoted[
                    "materialized_graph_features_sha256"
                ],
                "actual_feedback_row_sha256": promoted["actual_feedback_row_sha256"],
            }
        )
    audit = {
        "schema_version": "stage5_actual_feedback_batch_audit_v3",
        "task_id": request.get("task_id"),
        "round_index": request.get("round_index"),
        "promoted_row_count": len(promoted_rows),
        "actual_group_count": len(actual_by_group),
        "silent_surrogate_fallback_count": sum(
            row["graph_features"].get("graph_feature_provenance") != ACTUAL_PROVENANCE
            for row in promoted_rows
        ),
        "measurement_request_file_sha256": _sha_file(request_path),
        "historical_feedback_file_sha256": _sha_file(feedback_path),
        "rows": row_audit,
    }
    return {"rows": promoted_rows, "audit": audit}


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-request-json", type=Path, required=True)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = promote_feedback_batch(args.measurement_request_json, args.feedback_json)
    _write(args.output_dir / "stage5_feedback_v3_actual.json", result["rows"])
    _write(args.output_dir / "actual_feedback_batch_audit_v3.json", result["audit"])
    print(json.dumps(result["audit"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
