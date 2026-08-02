#!/usr/bin/env python3
"""Materialize a relocated Stage7 source from its frozen registry checkpoint.

This is an operational recovery tool. It never edits a canonical Stage7
request. Instead, it verifies the frozen registry, copies the exact checkpoint
and configuration into the Stage7 source root, builds a temporary authenticated
request that carries the copied checkpoint's real SHA, and invokes the existing
Stage5 source-only materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


EXPECTED_REGISTRY_SHA256 = (
    "033cb9e38af39009f9919233f03c5ff77894651f6192d292dee501d974493b51"
)
EXPECTED_GROUPS = frozenset({"pyramid|24x32x64", "pyramid|56x128x256"})
EXPORT_CHECKPOINT_NAME = "net_epoch1.pth"
EXPECTED_CONFIG_SHA256 = {
    "pyramid|24x32x64": (
        "58d4204d7ccd59d07faa971543d860abfe0eae80861f5eca2f96506aaf4aa76d"
    ),
    "pyramid|56x128x256": (
        "0c44a8effea209bf0c6478a3769eb1892ed8c7af8b33731422b06f262a71c571"
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def source_plan_sha256(row: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "kind": row["materialization_kind"],
            "width": [int(value) for value in row["width"]],
            "contract": row["source_contract"],
        }
    )


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 5:
        raise RuntimeError("unexpected nvidia-smi GPU snapshot")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "index": int(fields[0]),
        "uuid": fields[1],
        "model": fields[2],
        "memory_used_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
        "compute_processes": processes.splitlines() if processes else [],
        "wall_time": time.time(),
    }


def require_idle_gpu(gpu: int) -> list[dict[str, Any]]:
    snapshots = [gpu_snapshot(gpu)]
    time.sleep(2)
    snapshots.append(gpu_snapshot(gpu))
    if any(
        snapshot["model"] != "NVIDIA H800"
        or snapshot["memory_used_mib"] > 1024
        or snapshot["utilization_percent"] > 5
        or snapshot["compute_processes"]
        for snapshot in snapshots
    ):
        raise RuntimeError("selected H800 GPU is not continuously idle")
    return snapshots


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def copy_immutable(source: Path, destination: Path, expected_sha256: str) -> None:
    if not source.is_file() or source.is_symlink():
        raise RuntimeError(f"source file is missing or unsafe: {source}")
    actual = file_sha256(source)
    if actual != expected_sha256:
        raise RuntimeError(f"source SHA drift: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            not destination.is_file()
            or destination.is_symlink()
            or file_sha256(destination) != expected_sha256
        ):
            raise RuntimeError(f"destination conflict: {destination}")
        return
    temporary = destination.with_name(
        f".{destination.name}.stage7-import-{os.getpid()}.tmp"
    )
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if file_sha256(temporary) != expected_sha256:
            raise RuntimeError(f"copied SHA drift: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def authenticate_request(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != "stage5_measurement_request_v2"
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise RuntimeError("canonical request shape drift")
    result = {**payload, "rows": [dict(row) for row in rows]}
    result.pop("measurement_request_sha256", None)
    result["row_sha256"] = {
        str(row["row_id"]): canonical_sha256(row) for row in result["rows"]
    }
    result["measurement_request_sha256"] = canonical_sha256(result)
    return result


def evidence_valid(evidence_path: Path, group_id: str, source_plan_sha: str) -> bool:
    if not evidence_path.is_file() or evidence_path.is_symlink():
        return False
    evidence = read_object(evidence_path)
    if (
        evidence.get("schema_version")
        != "stage5_source_materialization_evidence_v1"
        or evidence.get("group_id") != group_id
        or evidence.get("source_plan_sha256") != source_plan_sha
        or evidence.get("status") != "ready"
    ):
        return False
    pairs = (
        ("checkpoint_path", "checkpoint_sha256"),
        ("onnx_path", "onnx_sha256"),
        ("calibration_path", "calibration_sha256"),
        ("calibration_summary_path", "calibration_summary_sha256"),
    )
    return all(
        isinstance(evidence.get(path_key), str)
        and isinstance(evidence.get(sha_key), str)
        and Path(evidence[path_key]).is_file()
        and file_sha256(Path(evidence[path_key])) == evidence[sha_key]
        for path_key, sha_key in pairs
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--group-id", choices=sorted(EXPECTED_GROUPS), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--materializer", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--frozen-repo-root", type=Path, required=True)
    parser.add_argument("--model-repo-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.v2_root.resolve(strict=True)
    sources_root = (root / "sources").resolve(strict=True)
    registry_path = args.registry.resolve(strict=True)
    request_path = args.request.resolve(strict=True)
    materializer = args.materializer.resolve(strict=True)
    code_root = args.code_root.resolve(strict=True)
    frozen_repo_root = args.frozen_repo_root.resolve(strict=True)
    model_repo_root = args.model_repo_root.resolve(strict=True)
    audit_dir = args.audit_dir
    try:
        materializer.relative_to(code_root)
    except ValueError as error:
        raise RuntimeError("materializer escapes deployed code root") from error
    for relative in (
        "tools/structural_prune_pyramid.py",
        "scripts/stage2_h800_export_checkpoint_multiscale_onnx.py",
        "scripts/stage3_pyramid_calibration_export_v3.py",
        "scripts/stage35_prepare_pyramid_trt_calibration_v1.py",
    ):
        candidate = model_repo_root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(f"model source helper is missing or unsafe: {relative}")
    audit_root = (root / "audits/speed_priority_recovery").resolve(strict=True)
    try:
        audit_dir.resolve().relative_to(audit_root)
    except ValueError as error:
        raise RuntimeError("audit directory escapes recovery audit root") from error
    audit_dir.mkdir(parents=True, exist_ok=False)

    if file_sha256(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("frozen source registry SHA drift")
    registry = read_object(registry_path)
    groups = [
        group
        for group in registry.get("groups", [])
        if isinstance(group, dict) and group.get("group_id") == args.group_id
    ]
    if len(groups) != 1:
        raise RuntimeError("frozen registry group is missing or ambiguous")
    registry_group = groups[0]
    original = registry_group.get("source_contract")
    if (
        not isinstance(original, dict)
        or registry_group.get("model") != "pyramid"
        or registry_group.get("materialization_kind") != "pyramid_checkpoint_export"
        or original.get("training_required") is not False
    ):
        raise RuntimeError("frozen registry source contract drift")

    canonical_request = read_object(request_path)
    matching = [
        row
        for row in canonical_request.get("rows", [])
        if isinstance(row, dict) and row.get("group_id") == args.group_id
    ]
    if not matching:
        raise RuntimeError("group absent from canonical request")
    source_plans = {row.get("source_evidence_sha256") for row in matching}
    contracts = {canonical_sha256(row.get("source_contract")) for row in matching}
    if len(source_plans) != 1 or len(contracts) != 1:
        raise RuntimeError("canonical group source identity is inconsistent")
    canonical_source_plan_sha = next(iter(source_plans))
    if not is_sha256(canonical_source_plan_sha):
        raise RuntimeError("canonical source plan SHA is invalid")
    canonical_contract = matching[0].get("source_contract")
    if (
        not isinstance(canonical_contract, dict)
        or canonical_contract.get("training_required") is not False
        or canonical_contract.get("checkpoint_sha256") is not None
    ):
        raise RuntimeError("canonical relocated source contract drift")
    for field in (
        "checkpoint_path",
        "checkpoint_dir",
        "config_path",
        "onnx_path",
        "calibration_npz",
        "calibration_summary",
        "trt_calibration_dir",
        "source_done_marker",
    ):
        value = canonical_contract.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"canonical relocated path missing: {field}")
        try:
            Path(value).resolve().relative_to(sources_root)
        except ValueError as error:
            raise RuntimeError(f"canonical path escapes Stage7 source root: {field}") from error

    original_checkpoint = Path(str(original["checkpoint_path"])).resolve(strict=True)
    original_sha = str(original["checkpoint_sha256"])
    target_checkpoint = Path(canonical_contract["checkpoint_path"])
    copy_immutable(original_checkpoint, target_checkpoint, original_sha)
    export_checkpoint = target_checkpoint.with_name(EXPORT_CHECKPOINT_NAME)
    copy_immutable(original_checkpoint, export_checkpoint, original_sha)

    original_config = Path(str(original["checkpoint_dir"])) / "config.yaml"
    target_config = Path(canonical_contract["config_path"])
    config_sha = file_sha256(original_config.resolve(strict=True))
    if config_sha != EXPECTED_CONFIG_SHA256[args.group_id]:
        raise RuntimeError("frozen source config SHA drift")
    copy_immutable(original_config, target_config, config_sha)

    rebound_rows = []
    temporary_source_plan_sha = None
    for row in canonical_request["rows"]:
        copied = json.loads(json.dumps(row))
        if copied.get("group_id") == args.group_id:
            copied["source_contract"]["checkpoint_path"] = str(export_checkpoint)
            copied["source_contract"]["checkpoint_sha256"] = original_sha
            copied["source_evidence_sha256"] = source_plan_sha256(copied)
            temporary_source_plan_sha = copied["source_evidence_sha256"]
        rebound_rows.append(copied)
    if not is_sha256(temporary_source_plan_sha):
        raise RuntimeError("temporary source plan SHA is invalid")
    temporary_request = authenticate_request(
        {**canonical_request, "rows": rebound_rows}
    )
    temporary_path = audit_dir / "temporary_source_only_request.json"
    temporary_path.write_text(
        json.dumps(temporary_request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(code_root), str(frozen_repo_root))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "REPO": str(model_repo_root),
    }
    gpu_snapshots = require_idle_gpu(args.gpu)
    started = time.time()
    completed = subprocess.run(
        [
            str(materializer),
            "--request",
            str(temporary_path),
            "--model",
            "pyramid",
            "--group-id",
            args.group_id,
            "--gpu",
            str(args.gpu),
        ],
        cwd=str(audit_dir),
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (audit_dir / "materializer.stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (audit_dir / "materializer.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    source_marker = Path(canonical_contract["source_done_marker"])
    evidence_path = source_marker.with_name(
        source_marker.name.removesuffix(".done") + "_evidence.json"
    )
    temporary_valid = evidence_valid(
        evidence_path, args.group_id, temporary_source_plan_sha
    )
    if completed.returncode == 0 and temporary_valid:
        evidence = read_object(evidence_path)
        evidence["source_plan_sha256"] = canonical_source_plan_sha
        evidence["checkpoint_path"] = str(target_checkpoint)
        evidence["checkpoint_sha256"] = original_sha
        temporary_evidence_path = evidence_path.with_name(
            f".{evidence_path.name}.canonical-{os.getpid()}.tmp"
        )
        temporary_evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_evidence_path, evidence_path)
    valid = evidence_valid(
        evidence_path, args.group_id, canonical_source_plan_sha
    )
    audit = {
        "schema_version": "stage7_frozen_checkpoint_source_import_v1",
        "status": "success" if completed.returncode == 0 and valid else "failed",
        "group_id": args.group_id,
        "canonical_request_path": str(request_path),
        "canonical_request_file_sha256": file_sha256(request_path),
        "frozen_registry_path": str(registry_path),
        "frozen_registry_sha256": EXPECTED_REGISTRY_SHA256,
        "original_checkpoint_path": str(original_checkpoint),
        "original_checkpoint_sha256": original_sha,
        "target_checkpoint_path": str(target_checkpoint),
        "target_checkpoint_sha256": file_sha256(target_checkpoint),
        "export_checkpoint_path": str(export_checkpoint),
        "export_checkpoint_sha256": file_sha256(export_checkpoint),
        "target_config_path": str(target_config),
        "target_config_sha256": file_sha256(target_config),
        "canonical_source_plan_sha256": canonical_source_plan_sha,
        "temporary_source_plan_sha256": temporary_source_plan_sha,
        "temporary_source_evidence_valid": temporary_valid,
        "source_plan_rebound_to_canonical_after_materialization": True,
        "source_evidence_path": str(evidence_path),
        "source_evidence_valid": valid,
        "materializer_path": str(materializer),
        "materializer_sha256": file_sha256(materializer),
        "model_repo_root": str(model_repo_root),
        "materializer_returncode": completed.returncode,
        "gpu_index": args.gpu,
        "gpu_idle_snapshots": gpu_snapshots,
        "elapsed_seconds": time.time() - started,
        "canonical_request_modified": False,
        "selected_ids_changed": False,
        "selected_event_budget_delta": 0,
        "performance_measurement_jobs_launched": 0,
        "ap_jobs_launched": 0,
        "scientific_contract_changed": False,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    (audit_dir / "import_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if audit["status"] != "success":
        raise RuntimeError("source import/materialization did not close")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
