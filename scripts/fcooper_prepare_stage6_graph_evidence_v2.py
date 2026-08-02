#!/usr/bin/env python3
"""Build graph-only Stage6 evidence from formal T16 and the original probe ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASK_ID = "S5-FCO-TRT-V2"
PILOT_FRAGMENT = "fcooper_workpackage_a_20260723"
BASE_WIDTH = [64, 128, 256, 128, 256]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def merge_graph_evidence(
    t16_rows: Sequence[Mapping[str, Any]],
    base_graph: Mapping[str, Any],
) -> dict[str, Any]:
    if len(t16_rows) != 16:
        raise ValueError("Stage6 graph evidence requires the complete formal T16")
    identities = [str(row.get("row_id") or "") for row in t16_rows]
    if any(not row_id for row_id in identities) or len(set(identities)) != 16:
        raise ValueError("Stage6 graph evidence requires 16 unique T16 row identities")
    actual = []
    bindings = []
    for row in t16_rows:
        graph = row.get("graph_features")
        feedback_without_sha = dict(row)
        recorded_feedback_sha = feedback_without_sha.pop(
            "actual_feedback_row_sha256", None
        )
        if (
            row.get("task_id") != TASK_ID
            or row.get("training_source") != "online_feedback"
            or not isinstance(graph, Mapping)
            or graph.get("graph_feature_provenance")
            != "materialized_onnx_extracted_v1"
        ):
            raise ValueError("Stage6 graph evidence contains non-formal feedback")
        if row.get("materialized_graph_features_sha256") != _sha_payload(graph):
            raise ValueError("Stage6 graph feature SHA is not bound to its T16 row")
        if recorded_feedback_sha != _sha_payload(feedback_without_sha):
            raise ValueError("Stage6 graph evidence feedback row SHA drift")
        actual.append(dict(graph))
        bindings.append(
            {
                "row_id": str(row["row_id"]),
                "actual_feedback_row_sha256": recorded_feedback_sha,
                "materialized_graph_features_sha256": row[
                    "materialized_graph_features_sha256"
                ],
                "graph_payload_sha256": _sha_payload(graph),
            }
        )
    base = dict(base_graph)
    if list(base.get("width") or []) != BASE_WIDTH:
        raise ValueError("graph-only anchor is not the original F-Cooper width")
    return {
        "schema_version": "fcooper_stage6_observed_graph_evidence_v2",
        "task_id": TASK_ID,
        "t16_actual_graph_count": 16,
        "base_graph_source": "capability_probe_onnx_graph_only",
        "probe_metric_labels_used": [],
        "t16_graph_bindings": bindings,
        "graph_features": [*actual, base],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t16-feedback-json", type=Path, required=True)
    parser.add_argument("--base-probe-onnx", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    for path in vars(args).values():
        if isinstance(path, Path) and PILOT_FRAGMENT in str(path):
            raise ValueError(f"Stage6 graph evidence cannot use pilot path: {path}")
    payload = json.loads(args.t16_feedback_json.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        raise ValueError("formal T16 feedback must contain rows")
    from scripts.stage35_extract_onnx_graph_features_v1 import graph_features

    base = {
        "schema": "stage35_actual_onnx_graph_features_v1",
        "group_id": "fcooper|original_graph_anchor",
        "model": "fcooper",
        "width": BASE_WIDTH,
        "graph_feature_provenance": "probe_graph_only_not_a_label",
        **graph_features(args.base_probe_onnx),
    }
    result = merge_graph_evidence(rows, base)
    result["t16_feedback_path"] = str(args.t16_feedback_json.resolve())
    result["t16_feedback_sha256"] = _sha_file(args.t16_feedback_json)
    result["base_probe_onnx_path"] = str(args.base_probe_onnx.resolve())
    result["base_probe_onnx_sha256"] = _sha_file(args.base_probe_onnx)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output_json),
                "graph_count": len(result["graph_features"]),
                "probe_metric_labels_used": [],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
