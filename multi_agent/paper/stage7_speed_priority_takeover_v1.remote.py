#!/usr/bin/env python3
"""Operational Stage7 takeover shim for speed-priority GPU scheduling.

The formal deployment bundle and scientific request identities remain unchanged.
This shim only rebinds the dead orchestrator's runtime PID and disables
command-line-only GPU reservations in the live scheduler process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence


JSON = dict[str, Any]


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path, *, label: str) -> JSON:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".takeover.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"takeover temporary artifact already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_pid_exists(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _require_sha256(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _value_after_flag(argv: Sequence[str], flag: str) -> str | None:
    try:
        index = list(argv).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise ValueError(f"{flag} value is missing")
    return str(argv[index + 1])


def rebind_operational_identity(
    v2_root: Path,
    *,
    current_pid: int,
    capture_identity: Callable[[int], Mapping[str, Any]],
    pid_exists: Callable[[int], bool] = _default_pid_exists,
    current_uid: int | None = None,
    deployment_release_sha256: str | None = None,
    deployment_manifest_file_sha256: str | None = None,
) -> JSON:
    """Rebind operational PID state, optionally accepting a reviewed deployment."""
    root = Path(v2_root).resolve(strict=True)
    prepare_path = root / "prepare_state.json"
    initialize_path = root / "initialize_state.json"
    prepare = _read_mapping(prepare_path, label="prepare state")
    initialize = _read_mapping(initialize_path, label="initialize state")
    if (
        prepare.get("schema_version") != "stage7_core_ablation_v2_prepare_state"
        or prepare.get("status") != "prepared"
        or prepare.get("v2_root") not in (None, str(root))
        or initialize.get("schema_version")
        != "stage7_core_ablation_v2_initialize_result"
    ):
        raise ValueError("formal v2 root is not prepared and initialized")
    uid = os.getuid() if current_uid is None else current_uid
    if prepare.get("owner_uid") != uid:
        raise ValueError("takeover owner UID does not match prepared root")
    old_pid = prepare.get("orchestrator_pid")
    if isinstance(old_pid, bool) or not isinstance(old_pid, int) or old_pid <= 0:
        raise ValueError("prepared orchestrator PID is invalid")
    if old_pid != current_pid and pid_exists(old_pid):
        raise ValueError("prepared orchestrator PID is still alive")
    old_recovery_sha256 = prepare.get("recovery_contract_sha256")
    if (
        not isinstance(old_recovery_sha256, str)
        or initialize.get("prepare_recovery_contract_sha256")
        != old_recovery_sha256
    ):
        raise ValueError("prepare/initialize recovery binding is inconsistent")
    identity = dict(capture_identity(current_pid))
    if identity.get("pid") != current_pid or identity.get("uid") != uid:
        raise ValueError("captured takeover process identity is invalid")
    pin_updates: dict[str, str] = {}
    if deployment_release_sha256 is not None:
        pin_updates["deployment_release_sha256"] = _require_sha256(
            deployment_release_sha256, label="deployment release SHA256"
        )
    if deployment_manifest_file_sha256 is not None:
        pin_updates["deployment_manifest_file_sha256"] = _require_sha256(
            deployment_manifest_file_sha256,
            label="deployment manifest-file SHA256",
        )
    prepare_unsigned = {
        **{
            key: value
            for key, value in prepare.items()
            if key != "recovery_contract_sha256"
        },
        **pin_updates,
        "orchestrator_pid": current_pid,
        "orchestrator_process_identity": identity,
    }
    rebound_prepare = {
        **prepare_unsigned,
        "recovery_contract_sha256": _canonical_sha256(prepare_unsigned),
    }
    rebound_initialize = {
        **initialize,
        "prepare_recovery_contract_sha256": rebound_prepare[
            "recovery_contract_sha256"
        ],
    }
    _atomic_write_json(prepare_path, rebound_prepare)
    _atomic_write_json(initialize_path, rebound_initialize)
    if (
        _read_mapping(prepare_path, label="rebound prepare state")
        != rebound_prepare
        or _read_mapping(initialize_path, label="rebound initialize state")
        != rebound_initialize
    ):
        raise ValueError("operational PID rebind did not persist atomically")
    return {
        "old_orchestrator_pid": old_pid,
        "new_orchestrator_pid": current_pid,
        "deployment_release_sha256": rebound_prepare.get(
            "deployment_release_sha256"
        ),
        "deployment_manifest_file_sha256": rebound_prepare.get(
            "deployment_manifest_file_sha256"
        ),
        "old_prepare_recovery_contract_sha256": old_recovery_sha256,
        "new_prepare_recovery_contract_sha256": rebound_prepare[
            "recovery_contract_sha256"
        ],
    }


def install_speed_priority_policy(scheduler_module: Any) -> JSON:
    """Use physical occupancy, not command-line intent, for GPU availability."""

    def no_command_line_reservations(_processes: Sequence[Any]) -> dict[str, tuple]:
        return {}

    def no_unconditional_gpu7_exclusion(_scheduler: Any) -> None:
        return None

    scheduler_module.discover_process_reservations = no_command_line_reservations
    scheduler_module.V2Stage7Scheduler._gpu7_uuid = no_unconditional_gpu7_exclusion
    scheduler_module.MAX_PARALLEL_BATCHES = 8
    return {
        "command_line_reservations_blocking": False,
        "gpu7_unconditional_exclusion": False,
        "max_parallel_controllers": 8,
    }


def install_all_seed_fair_policy(orchestrator_module: Any) -> JSON:
    """Open a fair all-seed queue after the first authenticated Full barrier."""
    original_next = orchestrator_module._next_round_operation
    cursor = 0

    def first_incomplete_action(
        snapshot: Mapping[str, Any], variant: str, seed: int
    ) -> tuple[str, str, int, int] | None:
        for round_index in orchestrator_module.ROUNDS:
            record = orchestrator_module._round_record(
                snapshot, variant, seed, round_index
            )
            if round_index and (
                orchestrator_module._round_record(
                    snapshot, variant, seed, round_index - 1
                ).get("barrier")
                is not True
            ):
                return None
            if record.get("barrier") is True:
                continue
            if record.get("request_frozen") is not True:
                action = orchestrator_module.FREEZE_SOURCE_PLAN
            elif record.get("source_ready") is not True:
                action = orchestrator_module.RESOLVE_SOURCE
            elif record.get("exact_bound") is not True:
                action = orchestrator_module.BIND_EXACT
            elif record.get("terminal") is not True:
                action = orchestrator_module.EXECUTE_MEASUREMENT
            else:
                action = orchestrator_module.FINALIZE_TERMINAL_BARRIER
            return action, variant, seed, round_index
        return None

    def fair_next(
        snapshot: Mapping[str, Any], seed: int
    ) -> tuple[str, str, int, int] | None:
        nonlocal cursor
        first_full = orchestrator_module._round_record(
            snapshot,
            "full",
            orchestrator_module.PILOT_SEED,
            0,
        )
        if first_full.get("barrier") is not True:
            return original_next(snapshot, seed)
        operations = [
            operation
            for variant in orchestrator_module.CORE_VARIANTS
            for selected_seed in orchestrator_module.SEEDS
            if (
                operation := first_incomplete_action(
                    snapshot, variant, selected_seed
                )
            )
            is not None
        ]
        terminal = [
            operation
            for operation in operations
            if operation[0]
            == orchestrator_module.FINALIZE_TERMINAL_BARRIER
        ]
        if terminal:
            return terminal[0]
        if not operations:
            return None
        selected = operations[cursor % len(operations)]
        cursor = (cursor + 1) % len(operations)
        return selected

    orchestrator_module._next_round_operation = fair_next
    return {
        "all_seed_queue_after_first_full_barrier": True,
        "terminal_finalization_priority": True,
        "trajectory_internal_feedback_barrier": True,
    }


def _load_deployment_modules(v2_root: Path) -> tuple[Any, Any, Any]:
    code_root = (v2_root / "deployment" / "code").resolve(strict=True)
    if code_root != v2_root / "deployment" / "code" or not code_root.is_dir():
        raise ValueError("deployment code root is not canonical")
    prepare = _read_mapping(v2_root / "prepare_state.json", label="prepare state")
    raw_frozen_root = prepare.get("executor_repo_root")
    if not isinstance(raw_frozen_root, str) or not raw_frozen_root:
        raise ValueError("prepare state executor repo root is missing")
    frozen_root = Path(raw_frozen_root).resolve(strict=True)
    if not frozen_root.is_dir():
        raise ValueError("prepare state executor repo root is unavailable")
    for module_name in tuple(sys.modules):
        if (
            module_name == "framework"
            or module_name.startswith("framework.")
            or module_name == "scripts"
            or module_name.startswith("scripts.")
        ):
            sys.modules.pop(module_name, None)
    for runtime_root in (str(frozen_root), str(code_root)):
        if runtime_root in sys.path:
            sys.path.remove(runtime_root)
        sys.path.insert(0, runtime_root)
    from framework.stage7 import executor_admission_v2
    from scripts import stage7_core_ablation_orchestrator_v2
    from scripts import stage7_core_ablation_scheduler_v2

    return (
        executor_admission_v2,
        stage7_core_ablation_orchestrator_v2,
        stage7_core_ablation_scheduler_v2,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speed-priority operational takeover for Stage7"
    )
    parser.add_argument("--takeover-v2-root", type=Path, required=True)
    parser.add_argument("--takeover-expected-scheduler-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args, orchestrator_argv = build_parser().parse_known_args(argv)
    root = args.takeover_v2_root.resolve(strict=True)
    expected_scheduler_sha256 = args.takeover_expected_scheduler_sha256
    if (
        len(expected_scheduler_sha256) != 64
        or any(value not in "0123456789abcdef" for value in expected_scheduler_sha256)
    ):
        raise ValueError("expected scheduler SHA256 is invalid")
    executor, orchestrator, scheduler = _load_deployment_modules(root)
    scheduler_path = Path(scheduler.__file__).resolve(strict=True)
    actual_scheduler_sha256 = _file_sha256(scheduler_path)
    if actual_scheduler_sha256 != expected_scheduler_sha256:
        raise ValueError("deployed scheduler SHA256 drift")
    rebind = rebind_operational_identity(
        root,
        current_pid=os.getpid(),
        capture_identity=lambda pid: executor.capture_process_identity(
            executor.probe_linux_process_identity, pid
        ),
        deployment_release_sha256=_value_after_flag(
            orchestrator_argv, "--expected-release-sha256"
        ),
        deployment_manifest_file_sha256=_value_after_flag(
            orchestrator_argv, "--expected-manifest-file-sha256"
        ),
    )
    policy = install_speed_priority_policy(scheduler)
    orchestration_policy = install_all_seed_fair_policy(orchestrator)
    audit = {
        "schema_version": "stage7_speed_priority_takeover_v1",
        "v2_root": str(root),
        "takeover_wrapper_sha256": _file_sha256(Path(__file__).resolve()),
        "deployed_scheduler_sha256": actual_scheduler_sha256,
        "rebind": rebind,
        "policy": policy,
        "orchestration_policy": orchestration_policy,
        "scientific_contract_changed": False,
    }
    _atomic_write_json(
        root / "audits" / f"speed_priority_takeover_pid_{os.getpid()}.json",
        audit,
    )
    return int(orchestrator.main(orchestrator_argv))


if __name__ == "__main__":
    raise SystemExit(main())
