#!/usr/bin/env python3
"""Build an auditable Stage4 training view from cold-start and online feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}


def _ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("manifest_job_id") or "") for row in rows]


def _groups(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("group_id") or "") for row in rows}


def _validate_rows(
    rows: Sequence[Mapping[str, Any]], *, allowed_splits: set[str], source_name: str
) -> None:
    row_ids = _ids(rows)
    if any(not row_id for row_id in row_ids):
        raise ValueError(f"{source_name} rows contain an empty manifest_job_id")
    duplicates = sorted(row_id for row_id, count in Counter(row_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate manifest_job_id values: {duplicates}")
    observed_splits = {str(row.get("split") or "") for row in rows}
    if not observed_splits <= allowed_splits:
        raise ValueError(
            f"{source_name} rows must retain splits {sorted(allowed_splits)}, "
            f"got {sorted(observed_splits)}"
        )

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get("group_id") or "")
        if not group_id:
            raise ValueError(f"{source_name} row is missing group_id")
        by_group[group_id].append(row)
    for group_id, group_rows in by_group.items():
        arms = {
            (str(row.get("dispatch_key") or ""), str(row.get("q_mode") or ""))
            for row in group_rows
        }
        if len(group_rows) != 4 or arms != ARMS:
            raise ValueError(f"group is not a complete four-arm group: {group_id}")


def _validate_graph_features(
    features: Sequence[Mapping[str, Any]], expected_groups: set[str]
) -> None:
    feature_groups = [str(item.get("group_id") or "") for item in features]
    duplicates = sorted(
        group_id for group_id, count in Counter(feature_groups).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate graph feature groups: {duplicates}")
    if set(feature_groups) != expected_groups:
        missing = sorted(expected_groups - set(feature_groups))
        extra = sorted(set(feature_groups) - expected_groups)
        raise ValueError(f"graph feature coverage mismatch: missing={missing}, extra={extra}")


def merge_feedback_view(
    cold_rows: Sequence[Mapping[str, Any]],
    feedback_rows: Sequence[Mapping[str, Any]],
    cold_graph_features: Sequence[Mapping[str, Any]],
    feedback_graph_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_rows(
        cold_rows,
        allowed_splits={"train", "locked_holdout"},
        source_name="initial_coldstart",
    )
    _validate_rows(
        feedback_rows,
        allowed_splits={"online_feedback"},
        source_name="online_feedback",
    )
    cold_groups = _groups(cold_rows)
    feedback_groups = _groups(feedback_rows)
    overlap = sorted(cold_groups & feedback_groups)
    if overlap:
        raise ValueError(f"cold-start and feedback group overlap: {overlap}")

    cold_features = [dict(item) for item in cold_graph_features]
    feedback_features = [dict(item) for item in feedback_graph_features]
    _validate_graph_features(cold_features, cold_groups)
    _validate_graph_features(feedback_features, feedback_groups)

    rows = [
        {**dict(row), "training_source": "initial_coldstart"}
        for row in cold_rows
    ] + [
        {**dict(row), "training_source": "online_feedback"}
        for row in feedback_rows
    ]
    features = sorted(
        [*cold_features, *feedback_features], key=lambda item: str(item["group_id"])
    )
    return {
        "schema_version": "stage4_feedback_training_view_v1",
        "rows": rows,
        "graph_features": features,
        "audit": {
            "row_count": len(rows),
            "group_count": len(cold_groups | feedback_groups),
            "training_source_rows": dict(sorted(Counter(
                str(row["training_source"]) for row in rows
            ).items())),
            "training_source_groups": {
                "initial_coldstart": len(cold_groups),
                "online_feedback": len(feedback_groups),
            },
            "group_overlap": overlap,
            "graph_feature_group_count": len(features),
        },
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-json", type=Path, required=True)
    parser.add_argument("--feedback-json", type=Path, required=True)
    parser.add_argument("--cold-graph-features-json", type=Path, required=True)
    parser.add_argument("--feedback-graph-features-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = merge_feedback_view(
        _load(args.cold_json),
        _load(args.feedback_json),
        _load(args.cold_graph_features_json),
        _load(args.feedback_graph_features_json),
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "gold176_feedback16_rows.json"
    graph_path = output_dir / "gold176_feedback16_graph_features.json"
    rows_path.write_text(json.dumps(result["rows"], indent=2) + "\n", encoding="utf-8")
    graph_path.write_text(
        json.dumps(result["graph_features"], indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        **result["audit"],
        "schema_version": result["schema_version"],
        "input_sha256": {
            "cold_rows": _sha256(args.cold_json),
            "feedback_rows": _sha256(args.feedback_json),
            "cold_graph_features": _sha256(args.cold_graph_features_json),
            "feedback_graph_features": _sha256(args.feedback_graph_features_json),
        },
        "output_sha256": {
            "rows": _sha256(rows_path),
            "graph_features": _sha256(graph_path),
        },
    }
    audit_path = output_dir / "gold176_feedback16_merge_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
