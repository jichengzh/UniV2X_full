#!/usr/bin/env python3
"""Resolve one Stage7 source batch through the frozen Stage5 materializer.

This wrapper owns argv-safe source orchestration only.  It never performs
measurement, AP, cache reveal, feedback promotion, or terminal finalization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pwd
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from framework.stage7 import source_execution_v2 as source_execution
from framework.stage7 import source_resolution_v2 as source_resolution


REPO_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER_PATH = REPO_ROOT / "scripts" / "stage5_materialize_round_sources_v1.sh"
MATERIALIZER_SHA256 = "d978e6287afc239c63471cadf729b37b4617bf46d1b8b8ea13cb4e2433bf4abe"
READ_ONLY_DEPENDENCIES = {
    REPO_ROOT
    / "framework/stage5/measurement_plan_v1.py": "7da96d705831ec834cee6d9e519753ba0c7c587b78e13a81ae6761f8dc67b15c",
    REPO_ROOT
    / "framework/stage5/measurement_plan_v2.py": "4395418d9a0aedf4668bf045e1e23883779971ad823d75ef7a7ec92cc41196b8",
    REPO_ROOT
    / "framework/stage7/source_resolution_v2.py": "6ea22a278f97cf559a07e1b676e3424b097504300c99395452241a148ba94c2d",
    REPO_ROOT
    / "framework/stage7/source_round_orchestration_v2.py": "4873e30770019b29f80446c744ae56b7f416218b5043863827c580ce8c44418f",
}
ATTEMPT_SCHEMA = "stage7_source_execution_attempt_v2"
RECEIPT_SCHEMA = "stage7_source_execution_receipt_v2"
DRYRUN_SCHEMA = "stage7_source_resolution_no_gpu_dryrun_v2"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_safe_output_directory(path: Path, *, allowed_root: Path) -> Path:
    root = Path(allowed_root).resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("source output directory must be absolute")
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("source output directory escapes its round") from error
    if ".." in relative.parts:
        raise ValueError("source output directory escapes its round")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("source output directory contains a symlink")
        if current.exists():
            if not current.is_dir():
                raise ValueError("source output directory is not a directory")
        else:
            try:
                current.mkdir()
            except OSError as error:
                raise ValueError(
                    "source output directory cannot be created safely"
                ) from error
        if current.is_symlink():
            raise ValueError("source output directory contains a symlink")
    try:
        current.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as error:
        raise ValueError("source output directory escapes its round") from error
    return current


def _verify_frozen_dependencies() -> None:
    expected = {
        **READ_ONLY_DEPENDENCIES,
        MATERIALIZER_PATH: MATERIALIZER_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or _file_sha(path) != digest:
            raise ValueError(f"frozen source dependency SHA drift: {path}")


def _read_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{label} must be an absolute regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return copy.deepcopy(dict(payload))


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _write_immutable_bytes(path, _encoded_json(payload))


def _write_immutable_bytes(path: Path, encoded: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != encoded:
            raise ValueError(f"immutable source artifact conflict: {path.name}")
        return
    if path.exists():
        raise ValueError(f"immutable source artifact path conflict: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _is_no_gpu_audit_path(path: Path) -> bool:
    parts = path.parts
    return any(
        parts[index : index + 2] == ("audits", "no_gpu_dry_run")
        for index in range(len(parts) - 1)
    )


def _default_inventory_probe() -> dict[str, dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    inventory: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3 or not parts[0].isdigit() or not parts[1]:
            raise ValueError("H800 inventory probe output is malformed")
        inventory[parts[1]] = {
            "physical_index": int(parts[0]),
            "model": parts[2],
        }
    return inventory


def _default_process_identity(pid: int) -> dict[str, Any] | None:
    process = Path("/proc") / str(pid)
    try:
        stat = (process / "stat").read_text(encoding="utf-8").split()
        owner = pwd.getpwuid(process.stat().st_uid).pw_name
    except (OSError, KeyError, IndexError):
        return None
    return {
        "pid": pid,
        "owner": owner,
        "start_time": stat[21],
        "alive": True,
    }


def _verify_runtime(
    lease: Mapping[str, Any],
    *,
    no_gpu_dryrun: bool,
    advertised_gpu_uuids: Sequence[str],
    environment: Mapping[str, str],
    hostname_probe: Callable[[], str],
    inventory_probe: Callable[[], Mapping[str, Mapping[str, Any]]],
    process_identity_probe: Callable[[int], Mapping[str, Any] | None],
) -> None:
    expected_mode = "no_gpu_dryrun" if no_gpu_dryrun else "formal"
    if lease["execution_mode"] != expected_mode:
        raise ValueError("source lease execution mode drift")
    cuda = str(environment.get("CUDA_VISIBLE_DEVICES", ""))
    if no_gpu_dryrun:
        if cuda or list(advertised_gpu_uuids):
            raise ValueError("CUDA_VISIBLE_DEVICES must be empty for no-GPU dry-run")
        return
    if list(advertised_gpu_uuids) != lease["gpu_uuids"]:
        raise ValueError("advertised GPU UUID order differs from signed lease")
    expected_cuda = ",".join(lease["gpu_uuids"])
    if cuda != expected_cuda:
        raise ValueError("CUDA_VISIBLE_DEVICES differs from signed UUID order")
    if hostname_probe() != lease["hostname"]:
        raise ValueError("formal H800 hostname drift")
    current = inventory_probe()
    for binding in lease["group_bindings"]:
        actual = current.get(binding["gpu_uuid"])
        if (
            not isinstance(actual, Mapping)
            or actual.get("physical_index") != binding["physical_index"]
            or actual.get("model") != binding["gpu_model"]
        ):
            raise ValueError("formal H800 UUID/index/model inventory drift")
    owner = lease["lock_owner"]
    actual_owner = process_identity_probe(int(owner["pid"]))
    if (
        not isinstance(actual_owner, Mapping)
        or actual_owner.get("alive") is not True
        or actual_owner.get("pid") != owner["pid"]
        or actual_owner.get("owner") != owner["owner"]
        or str(actual_owner.get("start_time")) != owner["start_time"]
        or lease["source_locks_held"] is not True
    ):
        raise ValueError("formal source lock owner identity drift")


def _group_commands(
    execution_plan: Mapping[str, Any],
    lease: Mapping[str, Any],
    request_path: Path,
    *,
    dry_run: bool,
) -> list[list[str]]:
    lease_by_key = {
        row["source_group_execution_key"]: row for row in lease["group_bindings"]
    }
    commands: list[list[str]] = []
    for group in execution_plan["group_jobs"]:
        binding = lease_by_key.get(group["source_group_execution_key"])
        if not isinstance(binding, Mapping):
            raise ValueError("source group has no signed GPU assignment")
        command = [
            str(MATERIALIZER_PATH),
            "--request",
            str(request_path),
            "--model",
            str(group["model"]),
            "--group-id",
            str(group["group_id"]),
            "--gpu",
            str(binding["physical_index"]),
        ]
        if dry_run:
            command.append("--dry-run")
        commands.append(command)
    return commands


def _formal_result(
    source_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
) -> dict[str, Any]:
    result = source_resolution.build_formal_source_resolution_result(
        source_plan,
        evidence_paths_by_candidate=source_execution.candidate_evidence_paths(
            execution_plan
        ),
    )
    return source_resolution.validate_formal_source_resolution_result(
        result, source_plan
    )


def _run_materializer_batch(
    commands: Sequence[Sequence[str]],
    *,
    run_command: Callable[..., Any],
    destination: Path,
    runtime_environment: Mapping[str, str],
    source_attempt_sha256: str,
) -> list[tuple[Any | None, OSError | None]]:
    """Launch one bounded source batch and await every submitted child."""
    if not commands:
        return []
    child_environment = {
        **runtime_environment,
        "STAGE7_SOURCE_ATTEMPT_SHA256": source_attempt_sha256,
    }

    def invoke(command: Sequence[str]) -> Any:
        return run_command(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            cwd=str(destination),
            env=child_environment,
        )

    with ThreadPoolExecutor(
        max_workers=min(4, len(commands)),
        thread_name_prefix="stage7-source",
    ) as executor:
        futures = [executor.submit(invoke, command) for command in commands]
        outcomes: list[tuple[Any | None, OSError | None]] = []
        for future in futures:
            try:
                outcomes.append((future.result(), None))
            except OSError as error:
                outcomes.append((None, error))
    return outcomes


def _receipt(
    execution_plan: Mapping[str, Any],
    lease: Mapping[str, Any],
    *,
    status: str,
    invocation_count: int,
    formal_result_written: bool,
    result_path: Path | None = None,
    retry_path: Path | None = None,
    retry_reason: str | None = None,
    source_resolution_result_sha256: str | None = None,
    source_resolution_retry_sha256: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "logical_request_sha256": execution_plan["logical_request_sha256"],
        "source_resolution_plan_sha256": execution_plan[
            "source_resolution_plan_sha256"
        ],
        "source_execution_plan_sha256": execution_plan["source_execution_plan_sha256"],
        "source_gpu_lease_sha256": lease["source_gpu_lease_sha256"],
        "source_attempt_sha256": lease["source_attempt_sha256"],
        "materializer_sha256": MATERIALIZER_SHA256,
        "materializer_invocation_count": invocation_count,
        "formal_result_written": formal_result_written,
        "source_result_path": str(result_path) if result_path else None,
        "retry_path": str(retry_path) if retry_path else None,
        "retry_reason_code": retry_reason,
        "source_resolution_result_sha256": source_resolution_result_sha256,
        "source_resolution_retry_sha256": source_resolution_retry_sha256,
        "selected_event_budget_delta": 0,
        "partial_reveal_allowed": False,
        "eligible_for_exact_cache_reveal": False,
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def _retry_result(
    output_dir: Path,
    source_plan: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    lease: Mapping[str, Any],
    *,
    failed_group_keys: Sequence[str],
    reason: str,
    invocation_count: int,
) -> dict[str, Any]:
    retry = source_execution.build_zero_budget_source_retry(
        source_plan,
        execution_plan,
        failed_group_keys=failed_group_keys,
        reason_code=reason,
    )
    retry_path = output_dir / "source_resolution_retry.json"
    _write_immutable(retry_path, retry)
    receipt = _receipt(
        execution_plan,
        lease,
        status="retry_required",
        invocation_count=invocation_count,
        formal_result_written=False,
        retry_path=retry_path,
        retry_reason=reason,
        source_resolution_retry_sha256=retry["source_resolution_retry_sha256"],
    )
    receipt_path = output_dir / "source_execution_receipt.json"
    _write_immutable(receipt_path, receipt)
    return {
        **receipt,
        "source_execution_receipt_path": str(receipt_path),
    }


def run_source_resolution(
    *,
    logical_request_path: Path,
    selection_binding_path: Path,
    source_plan_path: Path,
    source_lease_path: Path,
    output_dir: Path,
    no_gpu_dryrun: bool,
    advertised_gpu_uuids: Sequence[str],
    run_command: Callable[..., Any] = subprocess.run,
    environment: Mapping[str, str] | None = None,
    hostname_probe: Callable[[], str] = socket.gethostname,
    inventory_probe: Callable[
        [], Mapping[str, Mapping[str, Any]]
    ] = _default_inventory_probe,
    process_identity_probe: Callable[
        [int], Mapping[str, Any] | None
    ] = _default_process_identity,
    pre_gate_source_only: bool = False,
) -> dict[str, Any]:
    """Resolve or audit one immutable four-row source batch."""
    _verify_frozen_dependencies()
    request_path = Path(logical_request_path).resolve()
    request_bytes = request_path.read_bytes()
    binding = _read_mapping(
        Path(selection_binding_path).resolve(), label="selection binding"
    )
    source_plan = _read_mapping(Path(source_plan_path).resolve(), label="source plan")
    lease_payload = _read_mapping(
        Path(source_lease_path).resolve(), label="source GPU lease"
    )
    execution_plan = source_execution.build_source_execution_plan(
        request_bytes, binding, source_plan
    )
    source_execution.validate_source_execution_plan(
        execution_plan, request_bytes, binding, source_plan
    )
    lease = source_execution.validate_source_gpu_lease(lease_payload, execution_plan)
    runtime_environment = dict(os.environ if environment is None else environment)
    if pre_gate_source_only and no_gpu_dryrun:
        raise ValueError("pre-gate source-only mode cannot be a synthetic dry-run")
    raw_destination = Path(output_dir)
    destination = (
        _ensure_safe_output_directory(
            raw_destination,
            allowed_root=request_path.parent,
        )
        if not no_gpu_dryrun
        else raw_destination.resolve()
    )
    if no_gpu_dryrun and not _is_no_gpu_audit_path(destination):
        raise ValueError("no-GPU dry-run output must be under audits/no_gpu_dry_run")
    try:
        _verify_runtime(
            lease,
            no_gpu_dryrun=no_gpu_dryrun,
            advertised_gpu_uuids=advertised_gpu_uuids,
            environment=runtime_environment,
            hostname_probe=hostname_probe,
            inventory_probe=inventory_probe,
            process_identity_probe=process_identity_probe,
        )
    except ValueError:
        if no_gpu_dryrun:
            raise
        return _retry_result(
            destination,
            source_plan,
            execution_plan,
            lease,
            failed_group_keys=[
                group["source_group_execution_key"]
                for group in execution_plan["group_jobs"]
            ],
            reason="infrastructure_unavailable",
            invocation_count=0,
        )
    commands = _group_commands(
        execution_plan, lease, request_path, dry_run=no_gpu_dryrun
    )
    attempt_payload = {
        "schema_version": ATTEMPT_SCHEMA,
        "status": "frozen",
        "execution_mode": lease["execution_mode"],
        "logical_request_file_sha256": execution_plan["logical_request_file_sha256"],
        "logical_request_sha256": execution_plan["logical_request_sha256"],
        "source_resolution_plan_sha256": execution_plan[
            "source_resolution_plan_sha256"
        ],
        "source_execution_plan_sha256": execution_plan["source_execution_plan_sha256"],
        "source_gpu_lease_sha256": lease["source_gpu_lease_sha256"],
        "source_attempt_sha256": lease["source_attempt_sha256"],
        "advertised_gpu_uuids": list(advertised_gpu_uuids),
        "materializer_path": str(MATERIALIZER_PATH),
        "materializer_sha256": MATERIALIZER_SHA256,
        "commands": commands,
    }
    attempt = {**attempt_payload, "attempt_manifest_sha256": _sha(attempt_payload)}
    _write_immutable(destination / "source_attempt.json", attempt)
    if no_gpu_dryrun:
        synthetic = source_resolution.build_synthetic_dryrun_source_resolution_result(
            source_plan
        )
        synthetic_path = destination / "synthetic_source_resolution_result.json"
        _write_immutable(synthetic_path, synthetic)
        audit_payload = {
            "schema_version": DRYRUN_SCHEMA,
            "status": "synthetic_protocol_ready",
            "synthetic_nonfinal": True,
            "logical_request_sha256": execution_plan["logical_request_sha256"],
            "source_resolution_plan_sha256": execution_plan[
                "source_resolution_plan_sha256"
            ],
            "source_attempt_sha256": lease["source_attempt_sha256"],
            "would_run_commands": commands,
            "gpu_jobs_launched": 0,
            "materializer_invocation_count": 0,
            "formal_result_written": False,
            "eligible_for_cache_append": False,
            "eligible_for_finalization": False,
            "synthetic_result_path": str(synthetic_path),
        }
        audit = {**audit_payload, "dryrun_audit_sha256": _sha(audit_payload)}
        audit_path = destination / "no_gpu_dryrun_audit.json"
        _write_immutable(audit_path, audit)
        return {**audit, "dryrun_audit_path": str(audit_path)}
    try:
        ready = _formal_result(source_plan, execution_plan)
    except (OSError, ValueError):
        ready = None
    result_path = destination / "source_resolution_result.staged.json"
    if ready is not None:
        _write_immutable(result_path, ready)
        receipt = _receipt(
            execution_plan,
            lease,
            status="source_ready",
            invocation_count=0,
            formal_result_written=True,
            result_path=result_path,
            source_resolution_result_sha256=ready["source_resolution_result_sha256"],
        )
        receipt_path = destination / "source_execution_receipt.json"
        _write_immutable(receipt_path, receipt)
        return {**receipt, "source_execution_receipt_path": str(receipt_path)}
    try:
        _verify_runtime(
            lease,
            no_gpu_dryrun=False,
            advertised_gpu_uuids=advertised_gpu_uuids,
            environment=runtime_environment,
            hostname_probe=hostname_probe,
            inventory_probe=inventory_probe,
            process_identity_probe=process_identity_probe,
        )
    except ValueError:
        return _retry_result(
            destination,
            source_plan,
            execution_plan,
            lease,
            failed_group_keys=[
                group["source_group_execution_key"]
                for group in execution_plan["group_jobs"]
            ],
            reason="infrastructure_unavailable",
            invocation_count=0,
        )
    outcomes = _run_materializer_batch(
        commands,
        run_command=run_command,
        destination=destination,
        runtime_environment=runtime_environment,
        source_attempt_sha256=lease["source_attempt_sha256"],
    )
    invocation_count = len(commands)
    infrastructure_failures: set[str] = set()
    evidence_failures: set[str] = set()
    for group, (completed, launch_error) in zip(
        execution_plan["group_jobs"], outcomes
    ):
        group_key = group["source_group_execution_key"]
        if launch_error is not None:
            infrastructure_failures.add(group_key)
            continue
        if completed is None:
            raise AssertionError("materializer outcome is missing")
        group_prefix = f"group_{group['source_group_execution_key'][:16]}"
        stdout = getattr(completed, "stdout", "")
        stderr = getattr(completed, "stderr", "")
        stdout_bytes = (
            stdout if isinstance(stdout, bytes) else str(stdout).encode("utf-8")
        )
        stderr_bytes = (
            stderr if isinstance(stderr, bytes) else str(stderr).encode("utf-8")
        )
        stdout_path = destination / f"{group_prefix}.stdout.log"
        stderr_path = destination / f"{group_prefix}.stderr.log"
        if not pre_gate_source_only:
            _write_immutable_bytes(stdout_path, stdout_bytes)
            _write_immutable_bytes(stderr_path, stderr_bytes)
        log_payload = {
            "schema_version": "stage7_source_group_process_v2",
            "source_group_execution_key": group["source_group_execution_key"],
            "returncode": int(completed.returncode),
            "stdout_path": None if pre_gate_source_only else str(stdout_path),
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stdout_byte_count": len(stdout_bytes),
            "stderr_path": None if pre_gate_source_only else str(stderr_path),
            "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
            "stderr_byte_count": len(stderr_bytes),
            "raw_process_output_persisted": not pre_gate_source_only,
        }
        _write_immutable(
            destination / f"{group_prefix}_process.json",
            {**log_payload, "process_receipt_sha256": _sha(log_payload)},
        )
        if int(completed.returncode) != 0:
            evidence_failures.add(group_key)
    if infrastructure_failures or evidence_failures:
        failed_group_keys = [
            group["source_group_execution_key"]
            for group in execution_plan["group_jobs"]
            if group["source_group_execution_key"]
            in infrastructure_failures | evidence_failures
        ]
        return _retry_result(
            destination,
            source_plan,
            execution_plan,
            lease,
            failed_group_keys=failed_group_keys,
            reason=(
                "infrastructure_unavailable"
                if infrastructure_failures
                else "evidence_unavailable"
            ),
            invocation_count=invocation_count,
        )
    try:
        ready = _formal_result(source_plan, execution_plan)
    except (OSError, ValueError):
        return _retry_result(
            destination,
            source_plan,
            execution_plan,
            lease,
            failed_group_keys=[
                group["source_group_execution_key"]
                for group in execution_plan["group_jobs"]
            ],
            reason="evidence_invalid",
            invocation_count=invocation_count,
        )
    _write_immutable(result_path, ready)
    receipt = _receipt(
        execution_plan,
        lease,
        status="source_ready",
        invocation_count=invocation_count,
        formal_result_written=True,
        result_path=result_path,
        source_resolution_result_sha256=ready["source_resolution_result_sha256"],
    )
    receipt_path = destination / "source_execution_receipt.json"
    _write_immutable(receipt_path, receipt)
    return {**receipt, "source_execution_receipt_path": str(receipt_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-request-json", type=Path, required=True)
    parser.add_argument("--selection-binding-json", type=Path, required=True)
    parser.add_argument("--source-plan-json", type=Path, required=True)
    parser.add_argument("--source-lease-json", type=Path, required=True)
    parser.add_argument("--gpu-uuids", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-gpu-dryrun", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_source_resolution(
        logical_request_path=args.logical_request_json,
        selection_binding_path=args.selection_binding_json,
        source_plan_path=args.source_plan_json,
        source_lease_path=args.source_lease_json,
        output_dir=args.output_dir,
        no_gpu_dryrun=args.no_gpu_dryrun,
        advertised_gpu_uuids=tuple(
            value for value in args.gpu_uuids.split(",") if value
        ),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = ["run_source_resolution", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
