#!/usr/bin/env python3
"""Recompute all file and payload SHAs for F-Cooper feedback rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    feedback_paths = sorted(args.feedback_root.glob("execution/*/feedback_row.json"))
    if not feedback_paths:
        raise ValueError("no feedback rows found")
    audits = []
    for feedback_path in feedback_paths:
        row = json.loads(feedback_path.read_text())
        recorded_row_sha = row.pop("actual_feedback_row_sha256")
        if sha256_payload(row) != recorded_row_sha:
            raise ValueError(f"feedback row SHA drift: {feedback_path}")
        graph = row["graph_features"]
        if sha256_payload(graph) != row["materialized_graph_features_sha256"]:
            raise ValueError(f"graph payload SHA drift: {feedback_path}")
        checks = {}
        for path_field, sha_field in (
            ("performance_result_json", "performance_result_sha256"),
            ("ap_report_path", "ap_report_sha256"),
            ("materialized_source_evidence_path", "materialized_source_evidence_sha256"),
            ("engine_path", "engine_sha256"),
        ):
            path = Path(row[path_field])
            actual = sha256_file(path)
            if actual != row[sha_field]:
                raise ValueError(f"{sha_field} drift: {feedback_path}")
            checks[sha_field] = actual
        source = json.loads(Path(row["materialized_source_evidence_path"]).read_text())
        graph_path = Path(source["actual_graph_features_path"])
        if sha256_file(graph_path) != source["actual_graph_features_sha256"]:
            raise ValueError(f"actual graph file SHA drift: {feedback_path}")
        ap = json.loads(Path(row["ap_report_path"]).read_text())
        if ap.get("status") != "success_full" or ap.get("fallback_samples") != 0:
            raise ValueError(f"AP execution contract drift: {feedback_path}")
        audits.append(
            {
                "row_id": row["row_id"],
                "feedback_path": str(feedback_path),
                "feedback_row_sha256": recorded_row_sha,
                "graph_payload_sha256": row["materialized_graph_features_sha256"],
                "file_sha256": checks,
                "ap_processed_samples": ap.get("processed_samples"),
                "passed": True,
            }
        )
    payload = {
        "schema_version": "fcooper_feedback_file_integrity_audit_v1",
        "feedback_root": str(args.feedback_root),
        "row_count": len(audits),
        "passed": True,
        "rows": audits,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "row_count": len(audits)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
