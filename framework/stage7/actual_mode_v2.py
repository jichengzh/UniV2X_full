"""Recoverable Stage7 physical-attempt orchestration.

This module owns only Stage7 execution layout, immutable lineage and recovery.
Stage5/Stage3 remain the owners of measurement, quantization and AP semantics.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

from framework.stage7 import deployment_bundle_v2


JSON = dict[str, Any]
STAGES = ("quant", "performance", "sanity", "full", "final")
EMPTY_TERMINAL_SCHEMA = "stage7_actual_v3_empty_physical_terminal_v2"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable(path: Path, payload: object) -> None:
    content = _json_bytes(payload)
    _assert_safe_path(path, anchor=Path(path.anchor))
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"immutable artifact drift: {path.name}")
        return
    if path.exists():
        raise ValueError(f"immutable artifact path is not a file: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _assert_safe_path(path: Path, *, anchor: Path) -> None:
    candidate = Path(path).absolute()
    boundary = Path(anchor).absolute()
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as error:
        raise ValueError("execution artifact escapes its authenticated root") from error
    current = boundary
    for part in relative.parts:
        current = current / part
        if not current.exists() and not current.is_symlink():
            continue
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("execution artifact path contains a symlink")
        if current == candidate and stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise ValueError("execution artifact has hardlink drift")
        elif current != candidate and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("execution artifact parent is not a directory")


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    frozen_repo_root: Path,
    deployment_bundle_sha256: str,
) -> JSON:
    if not isinstance(receipt, Mapping):
        raise ValueError("authenticated no-GPU receipt is required")
    copied = dict(receipt)
    recorded = copied.pop("dry_run_receipt_sha256", None)
    if (
        copied.get("schema_version") != "stage7_core_no_gpu_dry_run_v2"
        or copied.get("formal_v2_root") != str(root)
        or copied.get("frozen_repo_root") != str(frozen_repo_root)
        or copied.get("formal_v2_gpu_jobs_launched") != 0
        or copied.get("cuda_visible_devices") != ""
        or copied.get("deployment_bundle_sha256") != deployment_bundle_sha256
        or copied.get("deployment_primitive_pins_sha256")
        != deployment_bundle_v2.primitive_pins_sha256()
        or copied.get("synthetic_non_measurement") is not True
        or copied.get("actual_v3_hardware_evidence") is not False
        or copied.get("eligible_for_cache_append") is not False
        or copied.get("eligible_for_formal_finalization") is not False
        or recorded != _sha(copied)
    ):
        raise ValueError("authenticated no-GPU receipt drift")
    return {**copied, "dry_run_receipt_sha256": recorded}


def validate_no_gpu_receipt(
    receipt: Mapping[str, Any],
    *,
    root: Path,
    frozen_repo_root: Path,
    deployment_bundle_sha256: str,
) -> JSON:
    """Public admission boundary shared by the shell and recovery engine."""
    return _validate_receipt(
        receipt,
        root=Path(root).resolve(),
        frozen_repo_root=Path(frozen_repo_root).resolve(),
        deployment_bundle_sha256=deployment_bundle_sha256,
    )


def _validate_projection(
    projection: Mapping[str, Any],
    *,
    physical_request_sha256: str,
    logical_request_sha256: str,
    deployment_bundle_sha256: str,
) -> JSON:
    if not isinstance(projection, Mapping):
        raise ValueError("independent projection is required")
    value = dict(projection)
    lineage = value.get("stage7_projection_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("independent projection lineage is missing")
    if (
        lineage.get("physical_request_sha256") != physical_request_sha256
        or lineage.get("logical_request_sha256") != logical_request_sha256
        or lineage.get("deployment_bundle_sha256") != deployment_bundle_sha256
    ):
        raise ValueError("independent projection lineage drift")
    if value.get("schema_version") == EMPTY_TERMINAL_SCHEMA:
        unsigned = {
            key: item
            for key, item in value.items()
            if key != "empty_physical_terminal_sha256"
        }
        if (
            value.get("empty_physical_terminal_sha256") != _sha(unsigned)
            or value.get("rows") != []
            or value.get("gpu_subprocess_count") != 0
        ):
            raise ValueError("empty independent projection drift")
    return value


def _result(execution: Path, terminal: Mapping[str, Any]) -> JSON:
    return {
        "schema_version": "stage7_actual_v3_execution_result_v2",
        "execution_root": str(execution),
        "terminal_payload_path": str(execution / "terminal_payload.json"),
        "terminal_payload_sha256": _sha(terminal),
        "gpu_subprocess_count": int(terminal.get("gpu_subprocess_count", 0)),
        "retry_required": False,
    }


def _read_json(path: Path, *, label: str) -> JSON:
    _assert_safe_path(path, anchor=Path(path.anchor))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unavailable or invalid") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _stage_path(attempt: Path, stage: str) -> Path:
    parent = "ap" if stage in {"sanity", "full"} else stage
    return attempt / parent / f"{stage}.json"


def _validated_stage_receipt(path: Path, *, expected_stage: str) -> JSON:
    receipt = _read_json(path, label="stage receipt")
    unsigned = {
        key: value for key, value in receipt.items() if key != "stage_receipt_sha256"
    }
    if (
        receipt.get("schema_version") != "stage7_actual_v3_stage_receipt_v2"
        or receipt.get("stage") != expected_stage
        or receipt.get("stage_receipt_sha256") != _sha(unsigned)
        or receipt.get("status")
        not in {
            "success",
            "candidate_terminal",
            "infrastructure_failure",
            "evidence_failure",
        }
    ):
        raise ValueError("stage receipt authentication failed")
    return receipt


def _write_attempt_state(
    attempt: Path,
    *,
    status: str,
    completed_stages: list[str],
    contract_sha256: str,
    parent_failure_sha256: str | None,
    failure_receipt_sha256: str | None = None,
) -> JSON:
    unsigned = {
        "schema_version": "stage7_actual_v3_attempt_state_v2",
        "attempt_id": attempt.name,
        "status": status,
        "completed_stages": completed_stages,
        "execution_contract_sha256": contract_sha256,
        "parent_failure_sha256": parent_failure_sha256,
        "failure_receipt_sha256": failure_receipt_sha256,
    }
    state = {**unsigned, "attempt_state_sha256": _sha(unsigned)}
    _write_immutable(attempt / "attempt_state.json", state)
    return state


def _attempt_for_resume(execution: Path) -> tuple[Path, str | None]:
    attempts = sorted(
        path for path in execution.glob("attempt_[0-9][0-9][0-9]") if path.is_dir()
    )
    if not attempts:
        return execution / "attempt_000", None
    latest = attempts[-1]
    state_path = latest / "attempt_state.json"
    if not state_path.is_file():
        return latest, None
    state = _read_json(state_path, label="attempt state")
    unsigned = {
        key: value for key, value in state.items() if key != "attempt_state_sha256"
    }
    if state.get("attempt_state_sha256") != _sha(unsigned):
        raise ValueError("attempt state authentication failed")
    if state.get("status") in {"retryable_infrastructure", "retryable_evidence"}:
        index = int(latest.name.removeprefix("attempt_")) + 1
        return execution / f"attempt_{index:03d}", str(state["attempt_state_sha256"])
    return latest, None


def execute_recoverable_attempt(
    *,
    round_dir: Path,
    physical_request_sha256: str,
    logical_request_sha256: str,
    projection: Mapping[str, Any],
    deployment_bundle_sha256: str,
    no_gpu_receipt: Mapping[str, Any],
    frozen_repo_root: Path,
    dry_run: bool,
    stage_runner: Callable[[str, Path], Mapping[str, Any] | None],
    terminal_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> JSON:
    """Create or replay one authenticated execution attempt."""
    if not all(
        _is_sha(value)
        for value in (
            physical_request_sha256,
            logical_request_sha256,
            deployment_bundle_sha256,
        )
    ):
        raise ValueError("execution identity requires lowercase SHA256 values")
    directory = Path(round_dir).absolute()
    root = directory.parents[3]
    if root.resolve(strict=False) != root:
        raise ValueError("formal execution root must be canonical and non-symlinked")
    _assert_safe_path(directory, anchor=root)
    frozen = Path(frozen_repo_root).resolve(strict=True)
    receipt = _validate_receipt(
        no_gpu_receipt,
        root=root,
        frozen_repo_root=frozen,
        deployment_bundle_sha256=deployment_bundle_sha256,
    )
    authenticated = _validate_projection(
        projection,
        physical_request_sha256=physical_request_sha256,
        logical_request_sha256=logical_request_sha256,
        deployment_bundle_sha256=deployment_bundle_sha256,
    )
    execution = directory / "actual_v3_execution" / physical_request_sha256
    contract_unsigned = {
        "schema_version": "stage7_actual_v3_execution_contract_v2",
        "logical_request_sha256": logical_request_sha256,
        "physical_request_sha256": physical_request_sha256,
        "deployment_bundle_sha256": deployment_bundle_sha256,
        "no_gpu_receipt_sha256": receipt["dry_run_receipt_sha256"],
        "dry_run": bool(dry_run),
    }
    contract = {
        **contract_unsigned,
        "execution_contract_sha256": _sha(contract_unsigned),
    }
    _write_immutable(execution / "execution_contract.json", contract)
    _write_immutable(execution / "independent_request.json", authenticated)
    terminal_path = execution / "terminal_payload.json"
    if terminal_path.is_file():
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        if terminal.get("schema_version") == EMPTY_TERMINAL_SCHEMA:
            _validate_projection(
                terminal,
                physical_request_sha256=physical_request_sha256,
                logical_request_sha256=logical_request_sha256,
                deployment_bundle_sha256=deployment_bundle_sha256,
            )
        else:
            validator = terminal_validator
            if validator is None:
                from framework.stage7.physical_feedback_v2 import (
                    validate_physical_terminal_batch,
                )

                validator = validate_physical_terminal_batch
            terminal = dict(validator(terminal))
        _write_immutable(terminal_path, terminal)
        return _result(execution, terminal)

    attempt, parent_failure_sha = _attempt_for_resume(execution)
    for child in ("quant", "performance", "ap", "final"):
        child_path = attempt / child
        _assert_safe_path(child_path, anchor=execution)
        child_path.mkdir(parents=True, exist_ok=True)
    if authenticated.get("schema_version") == EMPTY_TERMINAL_SCHEMA:
        terminal = dict(authenticated)
        state_unsigned = {
            "schema_version": "stage7_actual_v3_attempt_state_v2",
            "attempt_id": "attempt_000",
            "status": "terminal_empty",
            "completed_stages": [],
            "gpu_subprocess_count": 0,
            "execution_contract_sha256": contract["execution_contract_sha256"],
            "parent_failure_sha256": parent_failure_sha,
            "failure_receipt_sha256": None,
        }
        state = {**state_unsigned, "attempt_state_sha256": _sha(state_unsigned)}
        _write_immutable(attempt / "attempt_state.json", state)
        _write_immutable(terminal_path, terminal)
        return _result(execution, terminal)

    completed: list[str] = []
    for stage in STAGES:
        receipt_path = _stage_path(attempt, stage)
        if receipt_path.is_file():
            receipt = _validated_stage_receipt(receipt_path, expected_stage=stage)
        else:
            result = stage_runner(stage, attempt)
            if not isinstance(result, Mapping):
                raise ValueError(f"{stage} runner did not return a mapping")
            status = str(result.get("status") or "")
            if status not in {
                "success",
                "candidate_terminal",
                "infrastructure_failure",
                "evidence_failure",
            }:
                raise ValueError(f"{stage} runner status is invalid")
            unsigned_receipt = {
                "schema_version": "stage7_actual_v3_stage_receipt_v2",
                "stage": stage,
                "status": status,
                "result": dict(result),
                "execution_contract_sha256": contract["execution_contract_sha256"],
            }
            receipt = {
                **unsigned_receipt,
                "stage_receipt_sha256": _sha(unsigned_receipt),
            }
            _write_immutable(receipt_path, receipt)
        status = str(receipt["status"])
        if status in {"infrastructure_failure", "evidence_failure"}:
            state = _write_attempt_state(
                attempt,
                status=(
                    "retryable_infrastructure"
                    if status == "infrastructure_failure"
                    else "retryable_evidence"
                ),
                completed_stages=completed,
                contract_sha256=contract["execution_contract_sha256"],
                parent_failure_sha256=parent_failure_sha,
                failure_receipt_sha256=receipt["stage_receipt_sha256"],
            )
            return {
                "schema_version": "stage7_actual_v3_execution_result_v2",
                "execution_root": str(execution),
                "terminal_payload_path": None,
                "retry_required": True,
                "attempt_failure_sha256": state["attempt_state_sha256"],
                "gpu_subprocess_count": 0,
            }
        completed.append(stage)
        if status == "candidate_terminal" or stage == "final":
            terminal = receipt["result"].get("terminal_payload")
            if not isinstance(terminal, Mapping):
                raise ValueError("terminal stage did not return a physical terminal")
            validator = terminal_validator
            if validator is None:
                from framework.stage7.physical_feedback_v2 import (
                    validate_physical_terminal_batch,
                )

                validator = validate_physical_terminal_batch
            validated_terminal = dict(validator(terminal))
            state = _write_attempt_state(
                attempt,
                status=(
                    "terminal_candidate"
                    if status == "candidate_terminal"
                    else "terminal_success"
                ),
                completed_stages=completed,
                contract_sha256=contract["execution_contract_sha256"],
                parent_failure_sha256=parent_failure_sha,
            )
            del state
            _write_immutable(terminal_path, validated_terminal)
            return _result(execution, validated_terminal)
    raise RuntimeError("physical execution ended without a terminal stage")


__all__ = [
    "STAGES",
    "execute_recoverable_attempt",
    "validate_no_gpu_receipt",
]
