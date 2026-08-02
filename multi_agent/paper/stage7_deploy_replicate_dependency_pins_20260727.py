#!/usr/bin/env python3
"""Transactionally deploy the reviewed exact-cache replicate pin chain."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any


EXPECTED_BUNDLE_MODULE_SHA256 = (
    "7a8030a3598e5e0f79492367de4e89d9f8d309da66949a392da70ba4b848ae08"
)
TARGETS = {
    "scripts/stage7_resolve_round_sources_v2.py": {
        "old": "90d6d8fa690b17a6ca67e8a7294bf74b03cd99249f4ed6e6b0954e50ff114f31",
        "new": "fc8dd87e70eecfc89a8c59eea135e8a2d33f892ad1d1207d7ca30f8552669ea3",
    },
    "scripts/stage7_source_scheduler_v2.py": {
        "old": "de30f7e31aed868d402537f7584909a6f6cd9a815fca40c0051e7bfaf9712855",
        "new": "c17437bfbcade17d19e25f0f3329516324bfb2834ad67aaa71117124bac8bf03",
    },
}
TRANSACTION_SCHEMA = "stage7_replicate_dependency_pin_transaction_v1"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _atomic_replace(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.replicate-pins.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_replace(
        path,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _active_experiment_processes(root: Path) -> str:
    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                "ps -eo pid,ppid,state,etimes,args "
                "| grep -E 'stage7_(core_round_worker|execute_actual|"
                "core_ablation_orchestrator)|stage7_speed_priority_takeover_v1|"
                "stage3_execute_(performance|ap)_plan|stage2_route_b_' "
                f"| grep {str(root)!r} | grep -v grep || true"
            ),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _replace_flag(command: object, flag: str, value: str) -> list[str]:
    if not isinstance(command, list) or command.count(flag) != 1:
        raise ValueError(f"controller command must contain one {flag}")
    updated = [str(part) for part in command]
    index = updated.index(flag)
    if index + 1 >= len(updated):
        raise ValueError(f"controller command has no value after {flag}")
    updated[index + 1] = value
    return updated


def _rebind_controller_pins(
    state: dict[str, Any],
    *,
    release_sha256: str,
    manifest_file_sha256: str,
    wall_time: float,
) -> dict[str, Any]:
    controllers = state.get("controllers")
    if not isinstance(controllers, dict):
        raise ValueError("scheduler state has no controller mapping")
    rebound: dict[str, Any] = {}
    for controller_id, raw_controller in controllers.items():
        if not isinstance(raw_controller, dict):
            raise ValueError("scheduler controller is not an object")
        controller = dict(raw_controller)
        old_release = str(controller.get("expected_release_sha256") or "")
        old_manifest = str(
            controller.get("expected_manifest_file_sha256") or ""
        )
        command = _replace_flag(
            controller.get("command"),
            "--expected-release-sha256",
            release_sha256,
        )
        command = _replace_flag(
            command,
            "--expected-manifest-file-sha256",
            manifest_file_sha256,
        )
        rebound[str(controller_id)] = {
            **controller,
            "command": command,
            "expected_release_sha256": release_sha256,
            "expected_manifest_file_sha256": manifest_file_sha256,
            "deployment_pin_rebind": {
                "old_expected_release_sha256": old_release,
                "old_expected_manifest_file_sha256": old_manifest,
                "new_expected_release_sha256": release_sha256,
                "new_expected_manifest_file_sha256": manifest_file_sha256,
                "wall_time": wall_time,
                "scientific_contract_changed": False,
            },
        }
    return {**state, "controllers": rebound, "updated_wall_time": wall_time}


def _recover(
    transaction_dir: Path,
    destinations: dict[str, Path],
) -> None:
    journal_path = transaction_dir / "journal.json"
    if not journal_path.is_file():
        raise SystemExit("incomplete transaction has no journal")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if journal.get("status") == "committed":
        raise SystemExit("replicate dependency pins are already committed")
    if journal.get("status") not in {"prepared", "mutating"}:
        raise SystemExit("invalid prior transaction status")
    for name, destination in destinations.items():
        backup = transaction_dir / f"{name}.before"
        if not backup.is_file():
            raise SystemExit(f"rollback backup is missing: {name}")
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
    code_root = root / "deployment/code"
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    deployment_state_path = root / "deployment/deployment_state_v2.json"
    scheduler_state_path = root / "status/scheduler_state.json"
    prepare_path = root / "prepare_state.json"
    initialize_path = root / "initialize_state.json"
    transaction_name = os.environ.get(
        "TRANSACTION_NAME",
        ".stage7_replicate_dependency_pin_transaction",
    )
    if (
        Path(transaction_name).name != transaction_name
        or not transaction_name.startswith(".stage7_")
    ):
        raise SystemExit("unsafe deployment transaction name")
    transaction_dir = (
        root / "audits/speed_priority_recovery" / transaction_name
    )

    if audit_path.exists():
        raise SystemExit("audit path already exists")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(audit_path.parent.parent)
    if _active_experiment_processes(root):
        raise SystemExit("refuse deployment while experiment processes are active")

    bundle_path = code_root / "framework/stage7/deployment_bundle_v2.py"
    if _sha256(bundle_path.read_bytes()) != EXPECTED_BUNDLE_MODULE_SHA256:
        raise SystemExit("trusted deployment bundle module SHA mismatch")
    sys.path.insert(0, str(code_root))
    from framework.stage7 import deployment_bundle_v2 as bundle

    manifest = bundle._read_mapping(manifest_path, label="manifest")
    frozen_root = Path(str(manifest["frozen_repo_root"])).resolve(strict=True)
    patch_bytes: dict[str, bytes] = {}
    destinations: dict[str, Path] = {
        "manifest": manifest_path,
        "deployment_state": deployment_state_path,
        "scheduler_state": scheduler_state_path,
        "prepare": prepare_path,
        "initialize": initialize_path,
    }
    for index, (relative, pins) in enumerate(TARGETS.items()):
        content = (patch_root / relative).read_bytes()
        if _sha256(content) != pins["new"]:
            raise SystemExit(f"reviewed patch SHA mismatch: {relative}")
        patch_bytes[relative] = content
        for location, base in (("frozen", frozen_root), ("deployed", code_root)):
            destination = base / relative
            destinations[f"target_{index}_{location}"] = destination

    if transaction_dir.exists():
        _recover(transaction_dir, destinations)
    for index, (relative, pins) in enumerate(TARGETS.items()):
        for location in ("frozen", "deployed"):
            destination = destinations[f"target_{index}_{location}"]
            if _sha256(destination.read_bytes()) != pins["old"]:
                raise SystemExit(
                    f"old dependency SHA drift: {location}/{relative}"
                )
    originals = {name: path.read_bytes() for name, path in destinations.items()}
    old_prepare = json.loads(originals["prepare"])
    old_initialize = json.loads(originals["initialize"])
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
        raise SystemExit("prepare/initialize recovery contract drift")

    transaction_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(transaction_dir.parent)
    for name, content in originals.items():
        backup = transaction_dir / f"{name}.before"
        backup.write_bytes(content)
        with backup.open("rb") as stream:
            os.fsync(stream.fileno())
    _write_json(
        transaction_dir / "journal.json",
        {
            "schema_version": TRANSACTION_SCHEMA,
            "status": "prepared",
            "root": str(root),
            "target_pins": TARGETS,
            "created_wall_time": time.time(),
        },
    )

    before = {
        "release_sha256": manifest["deployment_release_sha256"],
        "manifest_file_sha256": bundle._file_sha256(manifest_path),
        "bundle_sha256": manifest["deployment_bundle_sha256"],
    }
    try:
        journal_path = transaction_dir / "journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["status"] = "mutating"
        _write_json(journal_path, journal)
        for index, (relative, content) in enumerate(patch_bytes.items()):
            _atomic_replace(destinations[f"target_{index}_frozen"], content)
            _atomic_replace(destinations[f"target_{index}_deployed"], content)

        records: list[dict[str, Any]] = []
        found: set[str] = set()
        for raw_record in manifest["files"]:
            record = dict(raw_record)
            relative = record["repo_relative_path"]
            content = patch_bytes.get(relative)
            if content is not None:
                found.add(relative)
                deployed = code_root / relative
                record.update(
                    byte_count=len(content),
                    sha256=_sha256(content),
                    mode=stat.S_IMODE(deployed.stat().st_mode),
                )
            records.append(record)
        if found != set(TARGETS):
            raise RuntimeError("deployment manifest target coverage mismatch")
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

        state = bundle._read_mapping(deployment_state_path, label="state")
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
        _atomic_replace(deployment_state_path, bundle._json_bytes(state))

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
            "deployment_release_sha256": manifest["deployment_release_sha256"],
        }
        prepare = {
            **prepare_unsigned,
            "recovery_contract_sha256": bundle.canonical_sha256(
                prepare_unsigned
            ),
        }
        initialize = {
            **old_initialize,
            "prepare_recovery_contract_sha256": prepare[
                "recovery_contract_sha256"
            ],
        }
        _atomic_replace(prepare_path, bundle._json_bytes(prepare))
        _atomic_replace(initialize_path, bundle._json_bytes(initialize))

        scheduler = json.loads(originals["scheduler_state"])
        rebound = _rebind_controller_pins(
            scheduler,
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
    except BaseException:
        for name, destination in destinations.items():
            _atomic_replace(destination, originals[name])
        raise

    audit = {
        "schema_version": "stage7_replicate_dependency_pin_deployment_audit_v1",
        "root": str(root),
        "before": before,
        "after": {
            "release_sha256": manifest["deployment_release_sha256"],
            "manifest_file_sha256": bundle._file_sha256(manifest_path),
            "bundle_sha256": manifest["deployment_bundle_sha256"],
            "target_sha256": {
                relative: _sha256((code_root / relative).read_bytes())
                for relative in TARGETS
            },
        },
        "validated": verified,
        "science_identity_changed": False,
        "selected_ids_changed": False,
        "terminal_or_barrier_changed": False,
        "contract_changed": "runtime_dependency_pins_only",
        "transaction_journal": str(transaction_dir / "journal.json"),
    }
    _write_json(audit_path, audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
