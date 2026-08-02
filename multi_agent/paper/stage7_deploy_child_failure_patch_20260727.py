from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time


EXPECTED_PATCH_SHA256 = (
    "3ba59673f4ad6f9b4bebd5ed65ceff29916cc25b2a311db982f499a4af6b95c8"
)


def main() -> None:
    root = Path(os.environ["ROOT"])
    patch = Path(os.environ["PATCH"])
    audit_path = Path(os.environ["AUDIT"])
    code = root / "deployment" / "code"
    target = code / "framework/stage7/actual_pipeline_v2.py"
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    state_path = root / "deployment/deployment_state_v2.json"

    sys.path.insert(0, str(code))
    from framework.stage7 import deployment_bundle_v2 as bundle

    def sh(command: str) -> str:
        return subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    active = sh(
        "ps -eo pid,ppid,state,etimes,args "
        "| grep -E 'stage7_(core_round_worker|execute_actual)|"
        "stage3_execute_(performance|ap)_plan|stage2_route_b_' "
        f"| grep {str(root)!r} | grep -v grep || true"
    )
    if active:
        raise SystemExit(f"refuse deploy while active worker exists: {active}")

    patch_bytes = patch.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != EXPECTED_PATCH_SHA256:
        raise SystemExit("patch sha mismatch")
    if b"stage7_isolated_row_failure_v1" not in patch_bytes:
        raise SystemExit("patch marker missing")

    before_manifest = bundle._read_mapping(manifest_path, label="manifest")
    before = {
        "target_sha256": bundle._file_sha256(target),
        "manifest_file_sha256": bundle._file_sha256(manifest_path),
        "release_sha256": before_manifest["deployment_release_sha256"],
        "bundle_sha256": before_manifest["deployment_bundle_sha256"],
    }

    stale = sh(
        "ps -eo pid,ppid,state,etimes,args "
        "| grep 'stage7_speed_priority_takeover_v1.py' "
        f"| grep {str(root)!r} | grep -v grep || true"
    )
    stopped = []
    for line in stale.splitlines():
        if not line.strip():
            continue
        pid = int(line.split()[0])
        os.kill(pid, 15)
        stopped.append({"pid": pid, "line": line})
    if stopped:
        time.sleep(2)
        for item in stopped:
            try:
                os.kill(int(item["pid"]), 0)
            except ProcessLookupError:
                item["terminated"] = True
            else:
                os.kill(int(item["pid"]), 9)
                item["terminated"] = "SIGKILL_after_SIGTERM"

    temporary = target.with_name(target.name + ".child_failure.tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(patch_bytes)
    os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
    os.replace(temporary, target)

    manifest = bundle._read_mapping(manifest_path, label="manifest")
    files = []
    for original in manifest["files"]:
        record = dict(original)
        if record["repo_relative_path"] == "framework/stage7/actual_pipeline_v2.py":
            target_bytes = target.read_bytes()
            record["byte_count"] = len(target_bytes)
            record["sha256"] = hashlib.sha256(target_bytes).hexdigest()
            record["mode"] = stat.S_IMODE(target.stat().st_mode)
        files.append(record)
    manifest["files"] = files
    manifest["ordered_file_list_sha256"] = bundle.canonical_sha256(files)
    manifest["deployment_release_sha256"] = bundle.canonical_sha256(
        bundle._release_records(files)
    )
    unsigned = bundle._manifest_unsigned(manifest)
    manifest["manifest_unsigned_payload_sha256"] = bundle.canonical_sha256(unsigned)
    manifest["deployment_bundle_sha256"] = bundle.canonical_sha256(unsigned)
    manifest_path.write_bytes(bundle._json_bytes(manifest))

    state = bundle._read_mapping(state_path, label="state")
    state["deployment_manifest_sha256"] = manifest[
        "manifest_unsigned_payload_sha256"
    ]
    state["deployment_manifest_file_sha256"] = bundle._file_sha256(manifest_path)
    state["deployment_bundle_sha256"] = manifest["deployment_bundle_sha256"]
    state["deployment_release_sha256"] = manifest["deployment_release_sha256"]
    state_unsigned = dict(state)
    state_unsigned.pop("state_sha256", None)
    state["state_sha256"] = bundle.canonical_sha256(state_unsigned)
    state_path.write_bytes(bundle._json_bytes(state))

    verified = bundle.validate_deployment_bundle(
        root,
        frozen_repo_root=Path(manifest["frozen_repo_root"]),
        expected_release_sha256=manifest["deployment_release_sha256"],
        expected_manifest_file_sha256=bundle._file_sha256(manifest_path),
    )
    audit = {
        "schema_version": "stage7_child_failure_patch_deployment_audit_v1",
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "root": str(root),
        "stopped_stale_supervisors": stopped,
        "before": before,
        "after": {
            "target_sha256": bundle._file_sha256(target),
            "release_sha256": manifest["deployment_release_sha256"],
            "manifest_file_sha256": bundle._file_sha256(manifest_path),
            "bundle_sha256": manifest["deployment_bundle_sha256"],
            "deployment_manifest_sha256": manifest[
                "manifest_unsigned_payload_sha256"
            ],
        },
        "validated": verified,
        "science_identity_changed": False,
        "selected_ids_changed": False,
        "candidate_pool_changed": False,
        "contract_changed": "runtime_failure_packaging_only",
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
