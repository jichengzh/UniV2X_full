#!/usr/bin/env python3
"""Backfill materialized ONNX graph features without rewriting Stage5 history."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from stage35_extract_onnx_graph_features_v1 import (  # noqa: E402
    graph_features as extract_graph_features,
)
from framework.stage5.measurement_plan_v1 import (  # noqa: E402
    _source_plan_sha as source_plan_sha,
)
from framework.stage5.measurement_plan_v2 import (  # noqa: E402
    _validate_request as validate_measurement_request,
)
from framework.stage5.single_target_search_v2 import (  # noqa: E402
    finalize_atomic_batch,
)


TASKS = ("S5-PYR-TVM", "S5-PYR-TRT", "S5-COD-TVM", "S5-COD-TRT")
ROUND_COUNT = 4
BATCH_SIZE = 4
ONLINE_ROW_COUNT = 64
SURROGATE_PROVENANCE = "coldstart_width_conditioned_surrogate_v1"
ACTUAL_PROVENANCE = "materialized_onnx_extracted_v1"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required artifact: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}: {error.msg}") from error


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _rows(payload: Any, *, label: str) -> list[dict[str, Any]]:
    source = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(source, list) or not all(isinstance(row, Mapping) for row in source):
        raise ValueError(f"{label} must contain a list of objects")
    return [dict(row) for row in source]


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("manifest_job_id") or row.get("row_id") or "")


def _verify_file(path_value: Any, sha_value: Any, *, label: str) -> Path:
    path = Path(str(path_value or ""))
    expected = str(sha_value or "")
    if not path.is_file():
        raise ValueError(f"{label} artifact missing: {path}")
    if len(expected) != 64 or _sha_file(path) != expected:
        raise ValueError(f"{label} SHA mismatch: {path}")
    return path


def _numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _drift(candidate: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    names = sorted(
        name
        for name in set(candidate) & set(actual)
        if _numeric(candidate[name]) and _numeric(actual[name])
    )
    features = []
    for name in names:
        predicted = float(candidate[name])
        measured = float(actual[name])
        absolute = abs(predicted - measured)
        relative = absolute / max(abs(measured), 1.0)
        features.append(
            {
                "feature": name,
                "surrogate": predicted,
                "actual": measured,
                "absolute_error": absolute,
                "relative_error": relative,
            }
        )
    relative = [float(item["relative_error"]) for item in features]
    return {
        "compared_numeric_feature_count": len(features),
        "changed_numeric_feature_count": sum(value > 0.0 for value in relative),
        "mean_relative_error": float(sum(relative) / len(relative)) if relative else 0.0,
        "max_relative_error": float(max(relative, default=0.0)),
        "features": features,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relative = [
        float(feature["relative_error"])
        for row in rows
        for feature in row["drift"]["features"]
    ]
    return {
        "row_count": len(rows),
        "numeric_comparison_count": len(relative),
        "mean_relative_error": float(sum(relative) / len(relative)) if relative else 0.0,
        "max_relative_error": float(max(relative, default=0.0)),
    }


def build_backfill(formal_root: Path) -> dict[str, Any]:
    online_records: list[dict[str, Any]] = []
    seen_row_ids: set[str] = set()
    for task_id in TASKS:
        for round_index in range(ROUND_COUNT):
            round_root = formal_root / task_id / f"round_{round_index:02d}"
            request = _read_json(round_root / "measurement_request.json")
            if (
                not isinstance(request, Mapping)
                or request.get("task_id") != task_id
                or request.get("round_index") != round_index
            ):
                raise ValueError(f"directory task/round identity drift: {task_id}:{round_index}")
            request_rows = validate_measurement_request(request)
            feedback_rows = _rows(
                _read_json(round_root / "final/stage5_feedback_v2_final.json"),
                label=f"{task_id} round {round_index} feedback",
            )
            if len(request_rows) != BATCH_SIZE or len(feedback_rows) != BATCH_SIZE:
                raise ValueError(f"Stage5 batch must contain four rows: {task_id}:{round_index}")
            atomic = finalize_atomic_batch(request, feedback_rows)
            if (
                atomic.get("feedback_released") is not True
                or atomic.get("budget_consumed") != BATCH_SIZE
            ):
                raise ValueError(f"Stage5 feedback batch is not terminal: {task_id}:{round_index}")
            request_by_id = {_row_id(row): row for row in request_rows}
            feedback_by_id = {_row_id(row): row for row in feedback_rows}
            if "" in request_by_id or set(request_by_id) != set(feedback_by_id):
                raise ValueError(f"request/feedback identity mismatch: {task_id}:{round_index}")
            row_sha = request.get("row_sha256")
            if not isinstance(row_sha, Mapping) or set(row_sha) != set(request_by_id):
                raise ValueError(f"request row SHA map mismatch: {task_id}:{round_index}")
            for row_id, requested in request_by_id.items():
                if row_id in seen_row_ids:
                    raise ValueError(f"duplicate online row identity: {row_id}")
                seen_row_ids.add(row_id)
                if str(row_sha[row_id]) != _sha_payload(requested):
                    raise ValueError(f"request row SHA mismatch: {row_id}")
                feedback = feedback_by_id[row_id]
                if feedback.get("measurement_request_row_sha256") != row_sha[row_id]:
                    raise ValueError(f"feedback request SHA mismatch: {row_id}")
                drifted_fields = sorted(
                    key
                    for key, value in requested.items()
                    if key not in {"schema_version", "graph_features"}
                    and feedback.get(key) != value
                )
                if drifted_fields:
                    raise ValueError(
                        f"feedback request identity drift: {row_id}:{drifted_fields}"
                    )
                candidate_graph = requested.get("graph_features")
                if not isinstance(candidate_graph, Mapping):
                    raise ValueError(f"candidate graph features missing: {row_id}")
                if feedback.get("graph_features") != candidate_graph:
                    raise ValueError(f"feedback graph snapshot drift: {row_id}")
                if candidate_graph.get("graph_feature_provenance") != SURROGATE_PROVENANCE:
                    raise ValueError(f"unexpected candidate graph provenance: {row_id}")
                source_path = _verify_file(
                    feedback.get("materialized_source_evidence_path"),
                    feedback.get("materialized_source_evidence_sha256"),
                    label=f"{row_id} source evidence",
                )
                source = _read_json(source_path)
                if (
                    not isinstance(source, Mapping)
                    or source.get("schema_version")
                    != "stage5_source_materialization_evidence_v1"
                    or source.get("status") != "ready"
                    or source.get("group_id") != requested.get("group_id")
                    or source.get("model") != requested.get("model")
                    or str(source.get("width") or "")
                    != "x".join(map(str, requested.get("width") or []))
                    or source.get("source_plan_sha256")
                    != requested.get("source_evidence_sha256")
                ):
                    raise ValueError(f"materialized source contract mismatch: {row_id}")
                onnx_path = _verify_file(
                    source.get("onnx_path"), source.get("onnx_sha256"), label=f"{row_id} ONNX"
                )
                online_records.append(
                    {
                        "task_id": task_id,
                        "round_index": round_index,
                        "row_id": row_id,
                        "request_row_sha256": str(row_sha[row_id]),
                        "request_path": str(round_root / "measurement_request.json"),
                        "feedback_path": str(
                            round_root / "final/stage5_feedback_v2_final.json"
                        ),
                        "requested": copy.deepcopy(requested),
                        "feedback": copy.deepcopy(feedback),
                        "source_evidence_path": str(source_path),
                        "source_evidence_sha256": _sha_file(source_path),
                        "onnx_path": str(onnx_path),
                        "onnx_sha256": _sha_file(onnx_path),
                    }
                )
    if len(online_records) != ONLINE_ROW_COUNT:
        raise ValueError(f"actual graph backfill requires exactly 64 rows, got {len(online_records)}")

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in online_records:
        by_group[str(record["requested"]["group_id"])].append(record)
    actual_by_group: dict[str, dict[str, Any]] = {}
    group_audit: list[dict[str, Any]] = []
    for group_id, records in sorted(by_group.items()):
        models = {str(record["requested"].get("model") or "") for record in records}
        widths = {tuple(record["requested"].get("width") or []) for record in records}
        onnx_bindings = {
            (str(record["onnx_path"]), str(record["onnx_sha256"])) for record in records
        }
        if len(models) != 1 or len(widths) != 1 or len(onnx_bindings) != 1:
            raise ValueError(f"materialized graph identity drift: {group_id}")
        onnx_path, onnx_sha = next(iter(onnx_bindings))
        extracted = extract_graph_features(Path(onnx_path))
        if extracted.get("onnx_sha256") != onnx_sha:
            raise ValueError(f"extracted ONNX SHA mismatch: {group_id}")
        actual_by_group[group_id] = {
            "schema": "stage5_actual_graph_features_v1",
            "group_id": group_id,
            "model": next(iter(models)),
            "width": list(next(iter(widths))),
            **extracted,
            "graph_feature_provenance": ACTUAL_PROVENANCE,
            "extractor_script_sha256": _sha_file(
                REPO_ROOT / "scripts/stage35_extract_onnx_graph_features_v1.py"
            ),
        }
        group_audit.append(
            {
                "group_id": group_id,
                "row_count": len(records),
                "manifest_job_ids": sorted(str(record["row_id"]) for record in records),
                "task_ids": sorted({str(record["task_id"]) for record in records}),
                "round_indices": sorted({int(record["round_index"]) for record in records}),
                "q_modes": sorted(
                    {str(record["requested"]["q_mode"]) for record in records}
                ),
                "capability_profile_ids": sorted(
                    {
                        str(record["requested"]["capability_profile_id"])
                        for record in records
                    }
                ),
                "onnx_sha256": onnx_sha,
            }
        )

    feedback_view = []
    drift_rows = []
    for record in online_records:
        requested = record["requested"]
        candidate = copy.deepcopy(dict(requested["graph_features"]))
        actual = copy.deepcopy(actual_by_group[str(requested["group_id"])])
        candidate_sha = _sha_payload(candidate)
        actual_sha = _sha_payload(actual)
        derived = {
            **record["feedback"],
            "candidate_graph_features": candidate,
            "candidate_graph_features_sha256": candidate_sha,
            "graph_features": actual,
            "materialized_graph_features_sha256": actual_sha,
            "graph_feature_backfill_schema": "stage5_online_feedback_actual_graph_v1",
            "graph_feature_backfill_round_index": record["round_index"],
            "derived_view_only": True,
            "historical_feedback_row_sha256": _sha_payload(record["feedback"]),
            "historical_measurement_request_row_sha256": record[
                "request_row_sha256"
            ],
            "historical_request_identity_matches_visible_row": False,
        }
        derived["derived_view_row_sha256"] = _sha_payload(derived)
        feedback_view.append(derived)
        drift_rows.append(
            {
                "task_id": record["task_id"],
                "round_index": record["round_index"],
                "manifest_job_id": record["row_id"],
                "group_id": requested["group_id"],
                "model": requested["model"],
                "dispatch_key": requested["dispatch_key"],
                "q_mode": requested["q_mode"],
                "candidate_graph_features_sha256": candidate_sha,
                "materialized_graph_features_sha256": actual_sha,
                "drift": _drift(candidate, actual),
            }
        )

    grouped: dict[str, dict[str, Any]] = {}
    for field in ("model", "dispatch_key", "q_mode", "task_id"):
        for value in sorted({str(row[field]) for row in drift_rows}):
            selected = [row for row in drift_rows if str(row[field]) == value]
            grouped[f"{field}={value}"] = _aggregate(selected)
    return {
        "actual": {
            "schema_version": "stage5_actual_graph_features_v1",
            "group_count": len(actual_by_group),
            "graph_features": list(actual_by_group.values()),
            "group_audit": group_audit,
        },
        "feedback_view": {
            "schema_version": "stage5_online_feedback_actual_graph_view_v1",
            "row_count": len(feedback_view),
            "rows": feedback_view,
        },
        "drift": {
            "schema_version": "stage5_graph_feature_drift_audit_v1",
            "row_count": len(drift_rows),
            "group_count": len(actual_by_group),
            "historical_requests_mutated": False,
            "summary": _aggregate(drift_rows),
            "grouped_summary": grouped,
            "rows": drift_rows,
        },
    }


def _write_idempotent(path: Path, payload: Any) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    paths = {
        "actual": output_dir / "stage5_actual_graph_features_v1.json",
        "feedback_view": output_dir / "stage5_online_feedback_actual_graph_view_v1.json",
        "drift": output_dir / "stage5_graph_feature_drift_audit_v1.json",
    }
    for key, path in paths.items():
        _write_idempotent(path, result[key])
    return {key: str(path) for key, path in paths.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_backfill(args.formal_root)
    paths = write_outputs(result, args.output_dir)
    print(
        json.dumps(
            {
                "status": "success",
                "row_count": result["feedback_view"]["row_count"],
                "group_count": result["actual"]["group_count"],
                "outputs": paths,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
