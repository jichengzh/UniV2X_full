#!/usr/bin/env python3
"""Build Stage4 Feedback16 online-feedback performance jobs."""

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

from framework.stage2.canonical_search_v3 import build_active_manifest
from scripts.stage35_gold32_performance_plan_v1 import _build_job


CANDIDATE_SCHEMA = "stage4_feedback16_candidate_v1"
MANIFEST_SCHEMA = "stage4_feedback16_manifest_v1"
PERFORMANCE_JOB_SCHEMA = "stage4_feedback16_performance_job_v1"
EXPECTED_ARMS = {
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hashes(
    candidate: Mapping[str, Any],
    *,
    gold176_path: Path,
    manifest176_path: Path,
    graph_features_path: Path,
) -> None:
    checks = (
        ("Gold176 SHA", "source_gold176_sha256", gold176_path),
        ("Gold176 manifest SHA", "source_gold176_manifest_sha256", manifest176_path),
        ("graph features SHA", "source_graph_features_sha256", graph_features_path),
    )
    for label, field, path in checks:
        expected = str(candidate.get(field) or "")
        actual = sha256_file(path)
        if expected != actual:
            raise ValueError(f"{label} mismatch: expected {expected}, got {actual}")


def _parse_width(width: Sequence[Any]) -> list[int]:
    values = [int(value) for value in width]
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"invalid width: {width!r}")
    return values


def _width_key(width: Sequence[Any]) -> str:
    return "x".join(str(value) for value in _parse_width(width))


def _padded_width(width: Sequence[Any]) -> str:
    return "x".join(f"{value:03d}" for value in _parse_width(width))


def _validate_candidate(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    if candidate.get("schema_version") != CANDIDATE_SCHEMA:
        raise ValueError(f"expected candidate schema {CANDIDATE_SCHEMA}")
    groups = candidate.get("groups")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ValueError("Feedback16 candidate must contain four groups")
    if candidate.get("group_count") != 4 or candidate.get("row_count") != 16:
        raise ValueError("Feedback16 candidate count contract must be 4 groups and 16 rows")
    normalized = [dict(group) for group in groups]
    group_ids = {str(group.get("group_id") or "") for group in normalized}
    if len(group_ids) != 4:
        raise ValueError("Feedback16 candidate group ids must be unique")
    for group in normalized:
        width_key = _width_key(group["width"])
        if str(group.get("group_id") or "") != f"{group['model']}|{width_key}":
            raise ValueError(f"candidate group identity mismatch: {group.get('group_id')}")
    return normalized


def _source_contract(group: Mapping[str, Any], remote_result_root: str | Path) -> dict[str, Any]:
    model = str(group["model"])
    width_key = _width_key(group["width"])
    root = Path(remote_result_root)
    if model == "pyramid":
        padded = _padded_width(group["width"])
        source_dir = root / "pyramid_sources" / padded
        calibration_dir = root / "pyramid_calibration" / padded
        return {
            "checkpoint_path": str(group["checkpoint_path"]),
            "checkpoint_dir": str(Path(str(group["checkpoint_path"])).parent),
            "onnx_path": str(source_dir / f"pyramid_{padded}_multiscale.onnx"),
            "calibration_root": str(calibration_dir),
            "calibration_npz": str(calibration_dir / "spatial_features_train16.npz"),
            "calibration_summary": str(calibration_dir / "summary.json"),
            "source_done_marker": str(root / "source_prep" / f"pyramid_{width_key}.done"),
        }
    if model == "codriving":
        model_dir = Path(str(group["model_dir"]))
        return {
            "model_dir": str(model_dir),
            "onnx_path": str(model_dir / f"resnet_multiscale_{width_key}_final_fp32.onnx"),
            "calibration_root": str(model_dir / "calibration_source"),
            "calibration_npz": str(model_dir / "stage3_calib_train_n16_float32.npz"),
            "calibration_summary": str(model_dir / "stage3_calib_train_n16_float32_summary.json"),
            "training_done_marker": str(model_dir / "stage4_feedback16_training_complete.json"),
            "source_done_marker": str(root / "source_prep" / f"codriving_{width_key}.done"),
        }
    raise ValueError(f"unsupported model: {model}")


def build_plan(
    candidate: Mapping[str, Any],
    *,
    gold176_rows: Sequence[Mapping[str, Any]],
    gold176_manifest: Mapping[str, Any],
    capability_profiles: Sequence[Mapping[str, Any]],
    remote_result_root: str | Path,
    gpus: Sequence[int],
    graph_features_sha256: str,
    capability_profiles_sha256: str,
) -> dict[str, Any]:
    groups = _validate_candidate(candidate)
    if len(gold176_rows) != 176:
        raise ValueError("source Gold176 must contain 176 rows")
    base_jobs = gold176_manifest.get("jobs")
    if not isinstance(base_jobs, list) or len(base_jobs) != 176:
        raise ValueError("source Gold176 manifest must contain 176 jobs")
    if not gpus:
        raise ValueError("gpus must not be empty")
    if str(candidate.get("source_graph_features_sha256") or "") != graph_features_sha256:
        raise ValueError("graph features SHA mismatch")

    existing_group_ids = {str(row["group_id"]) for row in base_jobs}
    candidate_group_ids = {str(group["group_id"]) for group in groups}
    if existing_group_ids & candidate_group_ids:
        raise ValueError("Feedback16 candidate overlaps the frozen Gold176 manifest")

    widths_by_model: dict[str, list[list[int]]] = {}
    groups_by_id: dict[str, dict[str, Any]] = {}
    for group in groups:
        widths_by_model.setdefault(str(group["model"]), []).append(_parse_width(group["width"]))
        groups_by_id[str(group["group_id"])] = group
    active = build_active_manifest(
        widths_by_model=widths_by_model,
        capability_profiles=capability_profiles,
    )

    rows: list[dict[str, Any]] = []
    for source in active["jobs"]:
        source_group = groups_by_id[str(source["group_id"])]
        row = {
            **source,
            "schema_version": MANIFEST_SCHEMA,
            "genome": [*source["width"], source["q_mode"]],
            "split": "online_feedback",
            "source_pool": "stage4_feedback16_online_feedback",
            "width_stratum": "online_feedback",
            "selection_reason": source_group["selection_reason"],
            "feedback_batch_id": candidate["feedback_batch_id"],
            "required_metrics": ["latency", "energy", "ap"],
            "source_contract": _source_contract(source_group, remote_result_root),
            "terminal_status": "pending",
        }
        rows.append(row)
    rows.sort(key=lambda row: (str(row["group_id"]), str(row["dispatch_key"]), str(row["q_mode"])))

    grouped_arms: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        grouped_arms.setdefault(str(row["group_id"]), set()).add(
            (str(row["dispatch_key"]), str(row["q_mode"]))
        )
    if len(rows) != 16 or any(arms != EXPECTED_ARMS for arms in grouped_arms.values()):
        raise ValueError("Feedback16 plan is not four complete four-arm groups")

    performance_jobs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        job = _build_job(
            row,
            batch_index=1,
            row_index=index,
            remote_artifact_root=Path(remote_result_root) / "performance_execution",
            gpus=gpus,
        )
        performance_jobs.append({**job, "schema_version": PERFORMANCE_JOB_SCHEMA})

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_pool": "online_feedback",
        "feedback_batch_id": candidate["feedback_batch_id"],
        "group_count": 4,
        "row_count": 16,
        "source_sha256": {
            "gold176": candidate["source_gold176_sha256"],
            "gold176_manifest": candidate["source_gold176_manifest_sha256"],
            "graph_features": candidate["source_graph_features_sha256"],
            "capability_profiles": capability_profiles_sha256,
        },
        "jobs": rows,
    }
    return {"manifest": manifest, "performance_jobs": performance_jobs}


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative GPU ids")
    return gpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--gold176-json", type=Path, required=True)
    parser.add_argument("--gold176-manifest-json", type=Path, required=True)
    parser.add_argument("--graph-features-json", type=Path, required=True)
    parser.add_argument("--capability-profiles-json", type=Path, required=True)
    parser.add_argument("--remote-result-root", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="6,7")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = json.loads(args.candidate_json.read_text(encoding="utf-8"))
    validate_source_hashes(
        candidate,
        gold176_path=args.gold176_json,
        manifest176_path=args.gold176_manifest_json,
        graph_features_path=args.graph_features_json,
    )
    capabilities_payload = json.loads(args.capability_profiles_json.read_text(encoding="utf-8"))
    capabilities = (
        capabilities_payload["capability_profiles"]
        if isinstance(capabilities_payload, Mapping)
        else capabilities_payload
    )
    result = build_plan(
        candidate,
        gold176_rows=json.loads(args.gold176_json.read_text(encoding="utf-8")),
        gold176_manifest=json.loads(args.gold176_manifest_json.read_text(encoding="utf-8")),
        capability_profiles=capabilities,
        remote_result_root=args.remote_result_root,
        gpus=_parse_gpus(args.gpus),
        graph_features_sha256=sha256_file(args.graph_features_json),
        capability_profiles_sha256=sha256_file(args.capability_profiles_json),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "feedback16_manifest.json"
    jobs_path = args.output_dir / "feedback16_performance_jobs.jsonl"
    manifest_path.write_text(
        json.dumps(result["manifest"], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    jobs_path.write_text(
        "".join(
            json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n"
            for job in result["performance_jobs"]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "jobs": str(jobs_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
