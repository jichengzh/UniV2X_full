"""Deployment-safe filesystem and identity helpers for the Stage7 no-GPU gate."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage7 import deployment_bundle_v2


JSON = dict[str, Any]


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_mapping(path: Path) -> JSON:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid dry-run artifact: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"dry-run artifact must be an object: {path}")
    return copy.deepcopy(dict(payload))


def write_isolated_json(path: Path, payload: Any) -> None:
    """Write only the isolated dry-run namespace and never overwrite drift."""
    encoded = json_bytes(payload)
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & stat.S_IWOTH
        ):
            raise ValueError(f"dry-run artifact is not a regular file: {path}")
        if path.read_bytes() != encoded:
            raise ValueError(f"dry-run artifact recovery drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() or temporary.is_symlink():
        metadata = temporary.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & stat.S_IWOTH
            or temporary.read_bytes() != encoded
        ):
            raise ValueError("dry-run temporary artifact drift")
        os.replace(temporary, path)
        descriptor = os.open(str(path.parent), os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def require_no_gpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("no-GPU dry-run requires CUDA_VISIBLE_DEVICES='' explicitly")


def exact_admission_receipts_closed(
    receipts: Any,
    *,
    variants: Sequence[str],
    events_per_round: int,
) -> bool:
    return (
        isinstance(receipts, list)
        and len(receipts) == len(variants)
        and all(isinstance(receipt, Mapping) for receipt in receipts)
        and [receipt.get("variant") for receipt in receipts] == list(variants)
        and all(
            receipt.get("round0_miss_count") == events_per_round
            and receipt.get("miss_admission_passed") is True
            for receipt in receipts
        )
    )


def validate_no_gpu_integration_result(
    result: Mapping[str, Any],
    *,
    variants: Sequence[str],
    events_per_round: int,
) -> None:
    if result.get("formal_v2_gpu_jobs_launched") != 0:
        raise ValueError("no-GPU gate launched a GPU job")
    if (
        result.get("canonical_round0_requests_written") != len(variants)
        or result.get("round0_miss_count") != len(variants) * events_per_round
        or result.get("integration_closed") is not True
        or result.get("blocking_reason") not in (None, "")
        or not exact_admission_receipts_closed(
            result.get("variant_receipts"),
            variants=variants,
            events_per_round=events_per_round,
        )
    ):
        raise ValueError("formal no-GPU integration gate is incomplete")


def no_gpu_integration_result_closed(
    result: Mapping[str, Any],
    *,
    variants: Sequence[str],
    events_per_round: int,
) -> bool:
    try:
        validate_no_gpu_integration_result(
            result, variants=variants, events_per_round=events_per_round
        )
    except ValueError:
        return False
    return True


def deployment_binding(
    v2_root: Path,
    *,
    frozen_repo_root: Path,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
) -> JSON:
    """Require prepare state and deployment bytes to name the same bundle."""
    root = Path(v2_root).resolve(strict=True)
    state = read_mapping(root / "prepare_state.json")
    verification = deployment_bundle_v2.validate_deployment_bundle(
        root,
        frozen_repo_root=Path(frozen_repo_root).resolve(strict=True),
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        expected_owner=(
            str(state["owner"]) if isinstance(state.get("owner"), str) else None
        ),
        expected_owner_uid=(
            int(state["owner_uid"])
            if isinstance(state.get("owner_uid"), int)
            and not isinstance(state.get("owner_uid"), bool)
            else None
        ),
    )
    if (
        state.get("deployment_manifest_sha256")
        != verification["deployment_manifest_sha256"]
        or state.get("deployment_manifest_file_sha256")
        != verification["deployment_manifest_file_sha256"]
        or state.get("deployment_bundle_sha256")
        != verification["deployment_bundle_sha256"]
        or state.get("deployment_release_sha256")
        != verification["deployment_release_sha256"]
    ):
        raise ValueError("no-GPU deployment bundle is not bound by prepare state")
    return verification


def close_synthetic_source_protocol(
    root: Path,
    *,
    variant: str,
    prepared: Mapping[str, Any],
    repo_root: Path,
    audit_dir: Path,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    online_module: Any,
) -> JSON:
    """Close source identity only through the existing quarantined no-GPU API."""
    directory = root / "variants" / variant / "seed_20260718" / "round_00"
    request = read_mapping(directory / "logical_request.json")
    plan = read_mapping(directory / "source_resolution_plan.json")
    rows = request.get("rows")
    candidate_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in rows
    ] if isinstance(rows, list) else []
    request_sha = request.get("measurement_request_sha256")
    if (
        len(candidate_ids) != 4
        or any(not value for value in candidate_ids)
        or request_sha != prepared.get("logical_request_sha256")
        or plan.get("logical_request_sha256") != request_sha
        or plan.get("source_resolution_plan_sha256")
        != prepared.get("source_resolution_plan_sha256")
        or plan.get("ordered_row_ids") != candidate_ids
        or plan.get("row_count") != 4
        or prepared.get("cache_membership_observed") is not False
        or prepared.get("exact_binding_frozen") is not False
    ):
        raise ValueError("SOURCE_PLAN_FROZEN no-GPU lineage drift")
    synthetic = (
        online_module.source_resolution
        .build_synthetic_dryrun_source_resolution_result(plan)
    )
    source_input = audit_dir / variant / "synthetic_source_resolution_input.json"
    write_isolated_json(source_input, synthetic)
    receipt = online_module.bind_reveal_after_source_ready(
        root,
        variant=variant,
        seed=20260718,
        round_index=0,
        source_result_path=source_input,
        synthetic_no_gpu_dryrun=True,
        repo_root=repo_root,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    source_result = read_mapping(
        directory / "synthetic_source_resolution_result.json"
    )
    protocol = read_mapping(directory / "synthetic_physical_protocol.json")
    forbidden_formal = (
        "source_resolution_result.json",
        "exact_selection_binding.json",
        "cache_snapshot_before_reveal.json",
        "cache_reveal.json",
        "miss_only_physical_request.json",
        "executor_admission.json",
        "exact_binding_reveal_receipt.json",
        "atomic_feedback_barrier.json",
    )
    if (
        receipt.get("controller_state") != "SYNTHETIC_PROTOCOL_CLOSED"
        or receipt.get("synthetic_nonfinal") is not True
        or receipt.get("cache_membership_observed") is not False
        or receipt.get("eligible_for_cache_append") is not False
        or receipt.get("eligible_for_finalization") is not False
        or receipt.get("gpu_launch_allowed") is not False
        or source_result.get("actual_v3_hardware_evidence") is not False
        or source_result.get("eligible_for_cache_append") is not False
        or source_result.get("eligible_for_finalization") is not False
        or protocol.get("formal_miss_plan_allowed") is not False
        or protocol.get("executor_admission_allowed") is not False
        or protocol.get("cache_reveal_allowed") is not False
        or protocol.get("gpu_launch_allowed") is not False
        or any((directory / name).exists() for name in forbidden_formal)
    ):
        raise ValueError("synthetic no-GPU source protocol escaped quarantine")
    closure = {
        "schema_version": "stage7_no_gpu_source_protocol_closure_v2",
        "variant": variant,
        "logical_request_sha256": request_sha,
        "source_resolution_plan_sha256": plan["source_resolution_plan_sha256"],
        "synthetic_source_resolution_result_sha256": source_result[
            "synthetic_dryrun_result_sha256"
        ],
        "synthetic_physical_protocol_sha256": protocol[
            "synthetic_physical_protocol_sha256"
        ],
        "formal_miss_plan_allowed": False,
        "executor_admission_allowed": False,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
        "canonical_barrier_written": False,
        "formal_v2_gpu_jobs_launched": 0,
    }
    closure = {
        **closure,
        "source_protocol_closure_sha256": canonical_sha256(closure),
    }
    write_isolated_json(
        audit_dir / variant / "synthetic_source_protocol_closure.json", closure
    )
    return {"request": request, "receipt": dict(receipt), "closure": closure}


__all__ = [
    "canonical_sha256",
    "close_synthetic_source_protocol",
    "deployment_binding",
    "deployment_bundle_v2",
    "exact_admission_receipts_closed",
    "file_sha256",
    "json_bytes",
    "no_gpu_integration_result_closed",
    "read_mapping",
    "require_no_gpu_environment",
    "validate_no_gpu_integration_result",
    "write_isolated_json",
]
