#!/usr/bin/env python3
"""Build the frozen four-group Gold128 targeted supplement execution plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage2.canonical_search_v3 import build_active_manifest
from scripts.stage35_gold32_performance_plan_v1 import _build_job


SCHEMA = "stage35_gold128_targeted_supplement_manifest_v1"
JOB_SCHEMA = "stage35_gold128_targeted_performance_job_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _padded(width: Sequence[int]) -> str:
    if len(width) != 3:
        raise ValueError("targeted supplement widths must have three stages")
    return "x".join(f"{int(value):03d}" for value in width)


def _parse_gpus(value: str) -> list[int]:
    gpus = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative GPU IDs")
    return gpus


def build_targeted_plan(
    candidate_plan: Mapping[str, Any],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    source_prep_audit: Mapping[str, Any],
    source_prep_audit_sha256: str,
    base_gold_manifest: Mapping[str, Any],
    base_gold_sha256: str,
    remote_result_root: str | Path,
    gpus: Sequence[int],
) -> dict[str, Any]:
    groups = candidate_plan.get("groups")
    if (
        candidate_plan.get("schema_version")
        != "stage35_gold128_targeted_supplement_candidate_plan_v1"
        or candidate_plan.get("selection_policy") != "freeze_before_targeted_measurement"
        or candidate_plan.get("locked_holdout_policy")
        != "retain_the_original_six_groups_unchanged"
        or not isinstance(groups, list)
        or len(groups) != 4
        or not gpus
    ):
        raise ValueError("invalid frozen targeted-supplement candidate plan")
    if not SHA256_PATTERN.fullmatch(source_prep_audit_sha256):
        raise ValueError("source-preparation audit SHA256 must be 64 hexadecimal characters")
    if not SHA256_PATTERN.fullmatch(base_gold_sha256):
        raise ValueError("base Gold128 SHA256 must be 64 hexadecimal characters")
    if str(candidate_plan.get("source_gold128_sha256") or "") != base_gold_sha256:
        raise ValueError("candidate source_gold128_sha256 does not match base Gold128 SHA256")

    base_jobs = base_gold_manifest.get("jobs")
    if not isinstance(base_jobs, list):
        raise ValueError("base Gold manifest must contain jobs")
    locked_groups = {
        str(job.get("group_id"))
        for job in base_jobs
        if isinstance(job, Mapping) and str(job.get("split")) == "locked_holdout"
    }
    if len(locked_groups) != 6:
        raise ValueError("base Gold manifest must contain exactly six locked holdout groups")
    candidate_groups = {str(group.get("group_id")) for group in groups}
    overlap = candidate_groups & locked_groups
    if overlap:
        raise ValueError(f"targeted candidate groups overlap locked holdout: {sorted(overlap)}")

    audit_groups = source_prep_audit.get("groups")
    if (
        source_prep_audit.get("schema_version")
        != "stage35_gold128_targeted_source_prep_audit_v1"
        or not isinstance(audit_groups, list)
    ):
        raise ValueError("invalid targeted source-preparation audit")
    evidence_by_group = {str(row.get("group_id")): row for row in audit_groups}
    if len(evidence_by_group) != len(audit_groups) or set(evidence_by_group) != candidate_groups:
        raise ValueError("source-preparation audit groups do not match frozen candidates")
    required_evidence_files = {
        "checkpoint_path", "onnx_path", "onnx_report_path", "calibration_npz",
        "calibration_summary", "trt_npy_manifest", "done_marker",
    }
    root = Path(remote_result_root)
    group_by_id = {str(group["group_id"]): group for group in groups}
    for group_id, evidence in evidence_by_group.items():
        files = evidence.get("files")
        if (
            evidence.get("status") != "prepared"
            or evidence.get("trt_npy_sample_count") != 15
            or not isinstance(files, Mapping)
            or set(files) != required_evidence_files
            or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("path"), str)
                or not SHA256_PATTERN.fullmatch(str(item.get("sha256") or ""))
                for item in files.values()
            )
        ):
            raise ValueError(f"source-preparation evidence is incomplete for {group_id}")
        group = group_by_id[group_id]
        padded = _padded(group["width"])
        width_key = "x".join(str(int(value)) for value in group["width"])
        source_root = root / "pyramid_sources" / padded
        calibration_root = root / "pyramid_calibration" / padded
        expected_paths = {
            "checkpoint_path": str(group["checkpoint_path"]),
            "onnx_path": str(source_root / f"pyramid_{padded}_multiscale.onnx"),
            "onnx_report_path": str(source_root / "onnx_export_report.json"),
            "calibration_npz": str(calibration_root / "spatial_features_train16.npz"),
            "calibration_summary": str(calibration_root / "summary.json"),
            "trt_npy_manifest": str(calibration_root / "trt_npy" / "trt_npy_manifest.json"),
            "done_marker": str(root / "source_prep" / f"pyramid_{width_key}.done"),
        }
        mismatches = [
            name for name, expected_path in expected_paths.items()
            if str(files[name]["path"]) != expected_path
        ]
        if mismatches:
            raise ValueError(
                f"source-preparation audit path mismatch for {group_id}: {mismatches}"
            )
    widths = [group["width"] for group in groups]
    active = build_active_manifest(
        widths_by_model={"pyramid": widths}, capability_profiles=capability_profiles
    )
    jobs = []
    for source in active["jobs"]:
        group = group_by_id[str(source["group_id"])]
        padded = _padded(source["width"])
        source_root = root / "pyramid_sources" / padded
        calibration_root = root / "pyramid_calibration" / padded
        jobs.append({
            **dict(source),
            "schema_version": SCHEMA,
            "split": "train",
            "width_stratum": "targeted_failure_boundary",
            "failure_boundary": group["failure_boundary"],
            "source_status": "checkpoint_consistent_source_prepared",
            "source_prep_evidence": {
                **dict(evidence_by_group[str(source["group_id"])]),
                "execution_host": source_prep_audit.get("execution_host"),
            },
            "source_contract": {
                "source_kind": "checkpoint_consistent_pyramid_multiscale_targeted_v1",
                "checkpoint_dir": group["checkpoint_dir"],
                "checkpoint_path": group["checkpoint_path"],
                "onnx_path": str(source_root / f"pyramid_{padded}_multiscale.onnx"),
                "onnx_report_path": str(source_root / "onnx_export_report.json"),
                "calibration_root": str(calibration_root),
                "calibration_npz": str(calibration_root / "spatial_features_train16.npz"),
                "calibration_summary": str(calibration_root / "summary.json"),
            },
            "required_metrics": ["latency_ms", "energy_j", "ap30", "ap50", "ap70"],
            "terminal_status": "pending",
        })
    if len(jobs) != 16:
        raise ValueError("targeted supplement must contain 16 four-arm rows")
    jobs.sort(key=lambda row: (str(row["group_id"]), str(row["dispatch_key"]), str(row["q_mode"])))
    performance_jobs = []
    for index, row in enumerate(jobs):
        job = _build_job(
            row,
            batch_index=1,
            row_index=index,
            remote_artifact_root=root / "performance",
            gpus=gpus,
        )
        performance_jobs.append({**job, "schema_version": JOB_SCHEMA})
    manifest = {
        **active,
        "schema_version": SCHEMA,
        "source_gold128_sha256": candidate_plan["source_gold128_sha256"],
        "source_prep_audit_sha256": source_prep_audit_sha256,
        "source_sufficiency_report_sha256": candidate_plan["source_sufficiency_report_sha256"],
        "locked_holdout_policy": candidate_plan["locked_holdout_policy"],
        "jobs": jobs,
    }
    return {"manifest": manifest, "performance_jobs": performance_jobs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-plan-json", type=Path, required=True)
    parser.add_argument("--capability-manifest-json", type=Path, required=True)
    parser.add_argument("--source-prep-audit-json", type=Path, required=True)
    parser.add_argument("--base-gold-manifest-json", type=Path, required=True)
    parser.add_argument("--base-gold-json", type=Path, required=True)
    parser.add_argument("--remote-result-root", required=True)
    parser.add_argument("--gpus", default="6,7")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = json.loads(args.candidate_plan_json.read_text(encoding="utf-8"))
    capabilities = json.loads(args.capability_manifest_json.read_text(encoding="utf-8"))
    result = build_targeted_plan(
        candidates,
        capabilities["capability_profiles"],
        source_prep_audit=json.loads(args.source_prep_audit_json.read_text(encoding="utf-8")),
        source_prep_audit_sha256=sha256_file(args.source_prep_audit_json),
        base_gold_manifest=json.loads(args.base_gold_manifest_json.read_text(encoding="utf-8")),
        base_gold_sha256=sha256_file(args.base_gold_json),
        remote_result_root=args.remote_result_root,
        gpus=_parse_gpus(args.gpus),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "targeted_supplement_manifest.json"
    jobs_path = args.output_dir / "targeted_performance_jobs.jsonl"
    manifest_path.write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jobs_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in result["performance_jobs"]),
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest_json": str(manifest_path.resolve()),
        "performance_jobs_jsonl": str(jobs_path.resolve()),
        "rows": len(result["manifest"]["jobs"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
