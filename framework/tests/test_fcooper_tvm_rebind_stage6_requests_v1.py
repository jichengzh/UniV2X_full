from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from framework.stage2.canonical_search_v3 import build_capability_profile
from scripts.fcooper_tvm_rebind_stage6_requests_v1 import (
    canonical_sha256,
    rebind_stage6_requests,
)

ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _row(index: int, *, arm: str) -> dict:
    row_id = f"trt-{arm}-{index:02d}"
    return {
        "row_id": row_id,
        "manifest_job_id": row_id,
        "task_id": "S6-FCO-TRT-OLD",
        "task_sha256": "a" * 64,
        "model": "fcooper",
        "hardware_id": "h800",
        "group_id": f"fcooper|group-{index:02d}",
        "width": [16 + index, 32, 64, 32, 64],
        "width_schema": [
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        ],
        "q_mode": ("fp16", "int8")[index % 2],
        "capability_profile_id": "h800-trt-old",
        "capability_digest": "b" * 64,
        "dispatch_key": "trt_engine",
        "builder_optimization_level": 0,
        "source_contract": {
            "schema_version": "fcooper_source_contract_v2",
            "checkpoint_sha256": f"{index + 1:064x}",
        },
        "source_evidence": {
            "schema_version": "fcooper_source_evidence_v2",
            "recovery_report_sha256": f"{index + 101:064x}",
        },
        "source_evidence_sha256": f"{index + 201:064x}",
    }


def _write_manifest(
    root: Path,
    *,
    arm: str,
    phase: str,
    count: int,
    builder_level: int,
) -> Path:
    rows = [_row(index, arm=arm) for index in range(count)]
    task_contract = {
        "schema_version": "stage6_fcooper_control_task_contract_v2",
        "task_id": "S6-FCO-TRT-OLD",
        "arm_id": arm,
        "phase": phase,
        "model": "fcooper",
        "hardware_id": "h800",
        "backend": "trt",
        "builder_optimization_level": builder_level,
    }
    task_sha = canonical_sha256(task_contract)
    entries = []
    for request_index, offset in enumerate(range(0, count, 4)):
        batch = []
        for source in rows[offset : offset + 4]:
            row = {
                **source,
                "task_sha256": task_sha,
                "arm_id": arm,
                "phase": phase,
                "builder_optimization_level": builder_level,
            }
            batch.append(row)
        request = {
            "schema_version": "stage6_fcooper_control_measurement_request_v2",
            "task_id": task_contract["task_id"],
            "task_sha256": task_sha,
            "task_contract": task_contract,
            "arm_id": arm,
            "phase": phase,
            "request_index": request_index,
            "batch_size": len(batch),
            "atomic_feedback": True,
            "real_h800_measurement_required": True,
            "builder_optimization_level": builder_level,
            "row_sha256": {
                row["row_id"]: canonical_sha256(row) for row in batch
            },
            "rows": batch,
        }
        request["measurement_request_sha256"] = canonical_sha256(request)
        request_path = root / f"request_{request_index:02d}.json"
        _write_json(request_path, request)
        entries.append(
            {
                "request_index": request_index,
                "path": str(request_path),
                "measurement_request_sha256": request[
                    "measurement_request_sha256"
                ],
                "batch_size": len(batch),
            }
        )
    manifest = {
        "schema_version": "stage6_fcooper_control_request_manifest_v2",
        "task_id": task_contract["task_id"],
        "arm_id": arm,
        "phase": phase,
        "request_count": len(entries),
        "row_count": count,
        "requests": entries,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path = root / "request_manifest.json"
    _write_json(path, manifest)
    return path


def _profile() -> dict:
    return build_capability_profile(
        capability_profile_id="h800-tvm-fcooper-v1",
        hardware_target="h800",
        compiler_fingerprint="c" * 64,
        dispatch_key="tvm_auto",
        features={"fp16_build_coverage": 1.0, "int8_tensorized_conv": 23},
    )


def _flatten(manifest: dict) -> list[dict]:
    rows = []
    for entry in manifest["requests"]:
        request = json.loads(Path(entry["path"]).read_text())
        rows.extend(request["rows"])
    return rows


def test_rebind_preserves_genomes_sources_and_recomputes_all_identities(
    tmp_path: Path,
) -> None:
    manifests = {
        "compression_only": _write_manifest(
            tmp_path / "trt-compression",
            arm="compression_only",
            phase="measure",
            count=16,
            builder_level=0,
        ),
        "compress_then_tune": _write_manifest(
            tmp_path / "trt-compress-tune",
            arm="compress_then_tune",
            phase="screen",
            count=12,
            builder_level=0,
        ),
        "schedule_only": _write_manifest(
            tmp_path / "trt-schedule",
            arm="schedule_only",
            phase="measure",
            count=1,
            builder_level=5,
        ),
    }
    original_bytes = {key: path.read_bytes() for key, path in manifests.items()}

    result = rebind_stage6_requests(
        compression_manifest=manifests["compression_only"],
        compress_tune_manifest=manifests["compress_then_tune"],
        schedule_manifest=manifests["schedule_only"],
        capability_profile=_profile(),
        output_dir=tmp_path / "tvm",
    )

    assert result["audit"]["candidate_counts"] == {
        "compression_only": 16,
        "compress_then_tune": 12,
        "schedule_only": 1,
    }
    assert result["audit"]["genome_sequence_equal"] is True
    assert result["audit"]["source_contract_sequence_equal"] is True
    assert result["audit"]["source_evidence_sequence_equal"] is True
    assert result["audit"]["trt_performance_artifact_fields"] == []
    assert {key: path.read_bytes() for key, path in manifests.items()} == original_bytes

    expected_trials = {
        "compression_only": 0,
        "compress_then_tune": 0,
        "schedule_only": 64,
    }
    for arm, manifest in result["manifests"].items():
        rows = _flatten(manifest)
        assert len({row["row_id"] for row in rows}) == len(rows)
        assert all(row["backend"] == "tvm" for row in rows)
        assert all(row["dispatch_key"] == "tvm_auto" for row in rows)
        assert all(row["capability_profile_id"] == "h800-tvm-fcooper-v1" for row in rows)
        assert all(row["tvm_trials"] == expected_trials[arm] for row in rows)
        assert all("builder_optimization_level" not in row for row in rows)
        for entry in manifest["requests"]:
            request = json.loads(Path(entry["path"]).read_text())
            payload = {
                key: value
                for key, value in request.items()
                if key != "measurement_request_sha256"
            }
            assert request["measurement_request_sha256"] == canonical_sha256(payload)
            assert request["task_contract"]["backend"] == "tvm"
            assert request["task_contract"]["dispatch_key"] == "tvm_auto"
            assert request["tvm_trials"] == expected_trials[arm]
            assert request["task_sha256"] == canonical_sha256(
                request["task_contract"]
            )
            assert request["row_sha256"] == {
                row["row_id"]: canonical_sha256(row) for row in request["rows"]
            }


def test_rebind_rejects_wrong_counts_non_tvm_profile_and_measurement_labels(
    tmp_path: Path,
) -> None:
    compression = _write_manifest(
        tmp_path / "compression",
        arm="compression_only",
        phase="measure",
        count=15,
        builder_level=0,
    )
    compress_tune = _write_manifest(
        tmp_path / "compress-tune",
        arm="compress_then_tune",
        phase="screen",
        count=12,
        builder_level=0,
    )
    schedule = _write_manifest(
        tmp_path / "schedule",
        arm="schedule_only",
        phase="measure",
        count=1,
        builder_level=5,
    )
    with pytest.raises(ValueError, match="expected 16"):
        rebind_stage6_requests(
            compression_manifest=compression,
            compress_tune_manifest=compress_tune,
            schedule_manifest=schedule,
            capability_profile=_profile(),
            output_dir=tmp_path / "out-count",
        )

    non_tvm = {**_profile(), "dispatch_key": "trt_engine"}
    with pytest.raises(ValueError, match="tvm_auto"):
        rebind_stage6_requests(
            compression_manifest=_write_manifest(
                tmp_path / "compression-16",
                arm="compression_only",
                phase="measure",
                count=16,
                builder_level=0,
            ),
            compress_tune_manifest=compress_tune,
            schedule_manifest=schedule,
            capability_profile=non_tvm,
            output_dir=tmp_path / "out-profile",
        )

    request_path = next((tmp_path / "compression-16").glob("request_00.json"))
    request = json.loads(request_path.read_text())
    request["rows"][0]["latency_ms"] = 1.23
    request["row_sha256"][request["rows"][0]["row_id"]] = canonical_sha256(
        request["rows"][0]
    )
    request["measurement_request_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in request.items()
            if key != "measurement_request_sha256"
        }
    )
    _write_json(request_path, request)
    manifest_path = tmp_path / "compression-16" / "request_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["requests"][0]["measurement_request_sha256"] = request[
        "measurement_request_sha256"
    ]
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="performance artifact"):
        rebind_stage6_requests(
            compression_manifest=manifest_path,
            compress_tune_manifest=compress_tune,
            schedule_manifest=schedule,
            capability_profile=_profile(),
            output_dir=tmp_path / "out-label",
        )


def test_rebind_does_not_mutate_profile_or_nested_source_objects(
    tmp_path: Path,
) -> None:
    profile = _profile()
    frozen = copy.deepcopy(profile)
    compression = _write_manifest(
        tmp_path / "compression",
        arm="compression_only",
        phase="measure",
        count=16,
        builder_level=0,
    )
    result = rebind_stage6_requests(
        compression_manifest=compression,
        compress_tune_manifest=_write_manifest(
            tmp_path / "compress-tune",
            arm="compress_then_tune",
            phase="screen",
            count=12,
            builder_level=0,
        ),
        schedule_manifest=_write_manifest(
            tmp_path / "schedule",
            arm="schedule_only",
            phase="measure",
            count=1,
            builder_level=5,
        ),
        capability_profile=profile,
        output_dir=tmp_path / "out",
    )

    assert profile == frozen
    audit_path = Path(result["audit_path"])
    audit = json.loads(audit_path.read_text())
    assert audit["audit_sha256"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "audit_sha256"}
    )


def test_cli_resolves_repo_modules_outside_repo_cwd(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/fcooper_tvm_rebind_stage6_requests_v1.py"),
            "--help",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--compression-manifest" in completed.stdout
