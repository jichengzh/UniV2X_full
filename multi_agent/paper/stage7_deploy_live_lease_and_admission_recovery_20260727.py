#!/usr/bin/env python3
"""Atomically deploy Stage7 live-lease and admission-retry recovery fixes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

from stage7_deploy_completed_round_rehydrate_patch_20260727 import (
    _atomic_replace,
    _rebind_scheduler_controller_pins,
)


TARGETS = (
    "framework/stage7/actual_pipeline_v2.py",
    "scripts/stage7_core_ablation_orchestrator_v2.py",
    "scripts/stage7_execute_actual_v3_misses_v2.sh",
)
EXPECTED_BUNDLE_MODULE_SHA256 = (
    "7a8030a3598e5e0f79492367de4e89d9f8d309da66949a392da70ba4b848ae08"
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _shell(command: str) -> str:
    return subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _active_experiment_processes(root: Path) -> str:
    return _shell(
        "ps -eo pid,ppid,state,etimes,args "
        "| grep -E 'stage7_(core_round_worker|execute_actual|core_ablation_orchestrator)|"
        "stage7_speed_priority_takeover_v1|"
        "stage3_execute_(performance|ap)_plan|stage2_route_b_' "
        f"| grep {str(root)!r} | grep -v grep || true"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_replace(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_expected(name: str) -> dict[str, str]:
    try:
        payload = json.loads(os.environ[name])
    except (KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"{name} must be a JSON object") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"{name} must be a JSON object")
    normalized = {str(key): str(value) for key, value in payload.items()}
    if set(normalized) != set(TARGETS):
        raise SystemExit(f"{name} target set mismatch")
    for value in normalized.values():
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise SystemExit(f"{name} contains a malformed SHA256")
    return normalized


def _backup_name(name: str) -> str:
    return name.replace("/", "__").replace(":", "_") + ".before"


def _recover_incomplete_transaction(
    transaction_dir: Path,
    *,
    fixed_paths: dict[str, Path],
) -> None:
    journal_path = transaction_dir / "journal.json"
    if not journal_path.is_file():
        raise SystemExit("deployment transaction exists without journal")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    status = journal.get("status")
    if status == "committed":
        raise SystemExit("recovery patch is already committed")
    if status not in {"prepared", "mutating"}:
        raise SystemExit("deployment transaction journal status is invalid")
    for name, destination in fixed_paths.items():
        backup = transaction_dir / _backup_name(name)
        if not backup.is_file():
            raise SystemExit(f"deployment rollback backup missing: {name}")
        _atomic_replace(destination, backup.read_bytes())
    journal["status"] = "rolled_back_on_resume"
    journal["recovered_wall_time"] = time.time()
    _write_json(journal_path, journal)
    recovered = transaction_dir.with_name(
        f"{transaction_dir.name}.recovered.{int(time.time())}"
    )
    os.replace(transaction_dir, recovered)
    _fsync_directory(transaction_dir.parent)


def main() -> None:
    root = Path(os.environ["ROOT"]).resolve(strict=True)
    patch_root = Path(os.environ["PATCH_ROOT"]).resolve(strict=True)
    audit_path = Path(os.environ["AUDIT"])
    expected_old = _load_expected("EXPECTED_OLD_SHAS_JSON")
    expected_new = _load_expected("EXPECTED_NEW_SHAS_JSON")
    code_root = root / "deployment/code"
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    state_path = root / "deployment/deployment_state_v2.json"
    scheduler_state_path = root / "status/scheduler_state.json"
    prepare_state_path = root / "prepare_state.json"
    initialize_state_path = root / "initialize_state.json"

    if audit_path.exists():
        raise SystemExit("audit path already exists")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    active = _active_experiment_processes(root)
    if active:
        raise SystemExit(f"refuse deploy while an experiment process is active\n{active}")

    bundle_path = code_root / "framework/stage7/deployment_bundle_v2.py"
    if _sha256(bundle_path.read_bytes()) != EXPECTED_BUNDLE_MODULE_SHA256:
        raise SystemExit("trusted deployment bundle module SHA mismatch")
    sys.path.insert(0, str(code_root))
    from framework.stage7 import deployment_bundle_v2 as bundle

    manifest = bundle._read_mapping(manifest_path, label="manifest")
    frozen_root = Path(str(manifest["frozen_repo_root"])).resolve(strict=True)
    target_paths: dict[str, Path] = {}
    for relative in TARGETS:
        target_paths[f"deployed::{relative}"] = code_root / relative
        target_paths[f"frozen::{relative}"] = frozen_root / relative
    fixed_paths = {
        **target_paths,
        "manifest": manifest_path,
        "state": state_path,
        "scheduler_state": scheduler_state_path,
        "prepare_state": prepare_state_path,
        "initialize_state": initialize_state_path,
    }
    transaction_dir = (
        root
        / "audits/speed_priority_recovery/.stage7_live_lease_admission_transaction"
    )
    if transaction_dir.exists():
        _recover_incomplete_transaction(
            transaction_dir,
            fixed_paths=fixed_paths,
        )

    patch_bytes: dict[str, bytes] = {}
    for relative in TARGETS:
        content = (patch_root / relative).read_bytes()
        if _sha256(content) != expected_new[relative]:
            raise SystemExit(f"reviewed patch SHA mismatch: {relative}")
        for destination in (code_root / relative, frozen_root / relative):
            if _sha256(destination.read_bytes()) != expected_old[relative]:
                raise SystemExit(f"old deployed SHA drift: {relative}")
        patch_bytes[relative] = content

    originals = {name: path.read_bytes() for name, path in fixed_paths.items()}
    old_prepare = json.loads(originals["prepare_state"])
    old_initialize = json.loads(originals["initialize_state"])
    old_prepare_unsigned = {
        key: value
        for key, value in old_prepare.items()
        if key != "recovery_contract_sha256"
    }
    old_recovery_sha = old_prepare.get("recovery_contract_sha256")
    if (
        old_prepare.get("schema_version")
        != "stage7_core_ablation_v2_prepare_state"
        or old_prepare.get("status") != "prepared"
        or old_prepare.get("v2_root") != str(root)
        or old_prepare.get("owner_uid") != os.getuid()
        or old_recovery_sha != bundle.canonical_sha256(old_prepare_unsigned)
        or old_initialize.get("schema_version")
        != "stage7_core_ablation_v2_initialize_result"
        or old_initialize.get("prepare_recovery_contract_sha256")
        != old_recovery_sha
    ):
        raise SystemExit("existing prepare/initialize recovery contract drift")

    transaction_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(transaction_dir.parent)
    for name, content in originals.items():
        backup = transaction_dir / _backup_name(name)
        backup.write_bytes(content)
        with backup.open("rb") as stream:
            os.fsync(stream.fileno())
    journal_path = transaction_dir / "journal.json"
    _write_json(
        journal_path,
        {
            "schema_version": "stage7_live_lease_admission_transaction_v1",
            "status": "prepared",
            "root": str(root),
            "old_shas": expected_old,
            "new_shas": expected_new,
            "created_wall_time": time.time(),
        },
    )

    before = {
        "target_sha256": expected_old,
        "release_sha256": manifest["deployment_release_sha256"],
        "manifest_file_sha256": bundle._file_sha256(manifest_path),
        "bundle_sha256": manifest["deployment_bundle_sha256"],
    }
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["status"] = "mutating"
        _write_json(journal_path, journal)
        for relative, content in patch_bytes.items():
            _atomic_replace(frozen_root / relative, content)
            _atomic_replace(code_root / relative, content)
        records = []
        patched = set()
        for raw_record in manifest["files"]:
            record = dict(raw_record)
            relative = record["repo_relative_path"]
            if relative in patch_bytes:
                patched.add(relative)
                record.update(
                    byte_count=len(patch_bytes[relative]),
                    sha256=expected_new[relative],
                    mode=stat.S_IMODE((code_root / relative).stat().st_mode),
                )
            records.append(record)
        if patched != set(TARGETS):
            raise RuntimeError("deployment manifest omits a patched target")
        manifest["files"] = records
        manifest["ordered_file_list_sha256"] = bundle.canonical_sha256(records)
        manifest["deployment_release_sha256"] = bundle.canonical_sha256(
            bundle._release_records(records)
        )
        unsigned_manifest = bundle._manifest_unsigned(manifest)
        manifest["manifest_unsigned_payload_sha256"] = bundle.canonical_sha256(
            unsigned_manifest
        )
        manifest["deployment_bundle_sha256"] = bundle.canonical_sha256(
            unsigned_manifest
        )
        _atomic_replace(manifest_path, bundle._json_bytes(manifest))

        state = bundle._read_mapping(state_path, label="state")
        state.update(
            deployment_manifest_sha256=manifest[
                "manifest_unsigned_payload_sha256"
            ],
            deployment_manifest_file_sha256=bundle._file_sha256(manifest_path),
            deployment_bundle_sha256=manifest["deployment_bundle_sha256"],
            deployment_release_sha256=manifest["deployment_release_sha256"],
        )
        unsigned_state = dict(state)
        unsigned_state.pop("state_sha256", None)
        state["state_sha256"] = bundle.canonical_sha256(unsigned_state)
        _atomic_replace(state_path, bundle._json_bytes(state))

        prepare_unsigned = {
            **{
                key: value
                for key, value in old_prepare.items()
                if key != "recovery_contract_sha256"
            },
            "deployment_manifest_sha256": manifest[
                "manifest_unsigned_payload_sha256"
            ],
            "deployment_manifest_file_sha256": bundle._file_sha256(
                manifest_path
            ),
            "deployment_bundle_sha256": manifest["deployment_bundle_sha256"],
            "deployment_release_sha256": manifest[
                "deployment_release_sha256"
            ],
        }
        prepare_state = {
            **prepare_unsigned,
            "recovery_contract_sha256": bundle.canonical_sha256(
                prepare_unsigned
            ),
        }
        initialize_state = old_initialize
        initialize_state["prepare_recovery_contract_sha256"] = prepare_state[
            "recovery_contract_sha256"
        ]
        _atomic_replace(prepare_state_path, bundle._json_bytes(prepare_state))
        _atomic_replace(
            initialize_state_path, bundle._json_bytes(initialize_state)
        )

        scheduler_state = json.loads(originals["scheduler_state"])
        rebound = _rebind_scheduler_controller_pins(
            scheduler_state,
            release_sha256=manifest["deployment_release_sha256"],
            manifest_file_sha256=bundle._file_sha256(manifest_path),
            wall_time=time.time(),
        )
        _atomic_replace(scheduler_state_path, bundle._json_bytes(rebound))
        verified = bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=frozen_root,
            expected_release_sha256=manifest["deployment_release_sha256"],
            expected_manifest_file_sha256=bundle._file_sha256(manifest_path),
        )
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["status"] = "committed"
        journal["committed_wall_time"] = time.time()
        _write_json(journal_path, journal)
    except BaseException as error:
        for name, destination in fixed_paths.items():
            _atomic_replace(destination, originals[name])
        audit_path.write_text(
            json.dumps(
                {
                    "schema_version": "stage7_live_lease_admission_failure_v1",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "rolled_back": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise

    audit = {
        "schema_version": "stage7_live_lease_admission_deployment_audit_v1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "before": before,
        "after": {
            "target_sha256": {
                relative: bundle._file_sha256(code_root / relative)
                for relative in TARGETS
            },
            "frozen_target_sha256": {
                relative: bundle._file_sha256(frozen_root / relative)
                for relative in TARGETS
            },
            "release_sha256": manifest["deployment_release_sha256"],
            "manifest_file_sha256": bundle._file_sha256(manifest_path),
            "bundle_sha256": manifest["deployment_bundle_sha256"],
        },
        "validated": verified,
        "science_identity_changed": False,
        "selected_ids_changed": False,
        "candidate_pool_changed": False,
        "contract_changed": "committed_control_recovery_only",
        "transaction_journal": str(journal_path),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
