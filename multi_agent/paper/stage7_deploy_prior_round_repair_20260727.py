#!/usr/bin/env python3
"""Atomically deploy the reviewed Stage7 prior-round recovery fix."""

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


RELATIVE_TARGET = "scripts/stage7_core_ablation_scheduler_v2.py"
EXPECTED_OLD_SHA256 = (
    "74010d9070b51129ca1c4500968191f1a6a7d076415391cebba8fdafde58984b"
)
EXPECTED_PATCH_SHA256 = (
    "aa8bd0191db2873ec48d62c372a5a29a6b5eab7aea2cc211a367d96c58523d20"
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
        raise SystemExit("reviewed scheduler patch is already committed")
    if status not in {"prepared", "mutating"}:
        raise SystemExit("deployment transaction journal status is invalid")
    for name, destination in fixed_paths.items():
        backup = transaction_dir / f"{name}.before"
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
    patch = Path(os.environ["PATCH"]).resolve(strict=True)
    audit_path = Path(os.environ["AUDIT"])
    code_root = root / "deployment/code"
    deployed_target = code_root / RELATIVE_TARGET
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    state_path = root / "deployment/deployment_state_v2.json"
    scheduler_state_path = root / "status/scheduler_state.json"
    prepare_state_path = root / "prepare_state.json"
    initialize_state_path = root / "initialize_state.json"

    if audit_path.exists():
        raise SystemExit("audit path already exists")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if _active_experiment_processes(root):
        raise SystemExit("refuse deploy while an experiment process is active")

    bundle_path = code_root / "framework/stage7/deployment_bundle_v2.py"
    if _sha256(bundle_path.read_bytes()) != EXPECTED_BUNDLE_MODULE_SHA256:
        raise SystemExit("trusted deployment bundle module SHA mismatch")
    sys.path.insert(0, str(code_root))
    from framework.stage7 import deployment_bundle_v2 as bundle

    manifest = bundle._read_mapping(manifest_path, label="manifest")
    frozen_root = Path(str(manifest["frozen_repo_root"])).resolve(strict=True)
    frozen_target = frozen_root / RELATIVE_TARGET
    fixed_paths = {
        "deployed_target": deployed_target,
        "frozen_target": frozen_target,
        "manifest": manifest_path,
        "state": state_path,
        "scheduler_state": scheduler_state_path,
        "prepare_state": prepare_state_path,
        "initialize_state": initialize_state_path,
    }
    transaction_dir = (
        root
        / "audits/speed_priority_recovery/.stage7_prior_round_repair_transaction"
    )
    if transaction_dir.exists():
        _recover_incomplete_transaction(
            transaction_dir,
            fixed_paths=fixed_paths,
        )
    patch_bytes = patch.read_bytes()
    if _sha256(patch_bytes) != EXPECTED_PATCH_SHA256:
        raise SystemExit("reviewed patch SHA mismatch")
    marker = b"request = self._controller_request(controller)"
    if marker not in patch_bytes:
        raise SystemExit("prior-round recovery marker missing")
    if (
        _sha256(deployed_target.read_bytes()) != EXPECTED_OLD_SHA256
        or _sha256(frozen_target.read_bytes()) != EXPECTED_OLD_SHA256
    ):
        raise SystemExit("old scheduler SHA drift")

    originals = {
        name: path.read_bytes() for name, path in fixed_paths.items()
    }
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
        backup = transaction_dir / f"{name}.before"
        backup.write_bytes(content)
        with backup.open("rb") as stream:
            os.fsync(stream.fileno())
    journal_path = transaction_dir / "journal.json"
    _write_json(
        journal_path,
        {
            "schema_version": "stage7_prior_round_repair_transaction_v1",
            "status": "prepared",
            "root": str(root),
            "old_scheduler_sha256": EXPECTED_OLD_SHA256,
            "new_scheduler_sha256": EXPECTED_PATCH_SHA256,
            "created_wall_time": time.time(),
        },
    )

    before = {
        "scheduler_sha256": EXPECTED_OLD_SHA256,
        "release_sha256": manifest["deployment_release_sha256"],
        "manifest_file_sha256": bundle._file_sha256(manifest_path),
        "bundle_sha256": manifest["deployment_bundle_sha256"],
    }
    try:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        journal["status"] = "mutating"
        _write_json(journal_path, journal)
        _atomic_replace(frozen_target, patch_bytes)
        _atomic_replace(deployed_target, patch_bytes)
        records = []
        found = False
        for raw_record in manifest["files"]:
            record = dict(raw_record)
            if record["repo_relative_path"] == RELATIVE_TARGET:
                found = True
                record.update(
                    byte_count=len(patch_bytes),
                    sha256=EXPECTED_PATCH_SHA256,
                    mode=stat.S_IMODE(deployed_target.stat().st_mode),
                )
            records.append(record)
        if not found:
            raise RuntimeError("deployment manifest omits scheduler")
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

        prepare_state = old_prepare
        if (
            prepare_state.get("schema_version")
            != "stage7_core_ablation_v2_prepare_state"
            or prepare_state.get("status") != "prepared"
        ):
            raise RuntimeError("prepare state identity drift")
        prepare_unsigned = {
            **{
                key: value
                for key, value in prepare_state.items()
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
        if (
            initialize_state.get("schema_version")
            != "stage7_core_ablation_v2_initialize_result"
        ):
            raise RuntimeError("initialize state identity drift")
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
                    "schema_version": "stage7_prior_round_repair_failure_v1",
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
        "schema_version": "stage7_prior_round_repair_deployment_audit_v1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "before": before,
        "after": {
            "scheduler_sha256": bundle._file_sha256(deployed_target),
            "frozen_scheduler_sha256": bundle._file_sha256(frozen_target),
            "release_sha256": manifest["deployment_release_sha256"],
            "manifest_file_sha256": bundle._file_sha256(manifest_path),
            "bundle_sha256": manifest["deployment_bundle_sha256"],
        },
        "validated": verified,
        "science_identity_changed": False,
        "selected_ids_changed": False,
        "candidate_pool_changed": False,
        "contract_changed": "committed_prior_round_recovery_only",
        "transaction_journal": str(journal_path),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
