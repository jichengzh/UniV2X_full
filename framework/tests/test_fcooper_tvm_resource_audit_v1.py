import hashlib
import json
from pathlib import Path

from scripts.fcooper_tvm_resource_audit_v1 import build_resource_audit
from scripts.fcooper_tvm_resource_audit_v1 import formal_measurement_summary


def _write(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resource_audit_reports_all_gpus_and_source_reuse(tmp_path: Path) -> None:
    scheduler = tmp_path / "schedulers/round_00"
    request = _write(
        tmp_path / "search/round_00/measurement_request.json",
        {
            "rows": [{"row_id": "row"}],
            "batch_size": 1,
        },
    )
    manifest = {
        "schema_version": "fcooper_tvm_gpu_job_manifest_v1",
        "barrier_order": ["round_00"],
        "gpu_pool": list(range(8)),
        "jobs": [
            {
                "job_id": "gear-r00-0",
                "request_json": str(request),
                "row_index": 0,
                "request_kind": "t16",
                "max_trials": 64,
            }
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write(scheduler / "manifest.json", manifest)
    _write(
        scheduler / "audit.json",
        {
            "schema_version": "fcooper_tvm_gpu_scheduler_audit_v1",
            "wall_clock_seconds": 20.0,
            "gpu_hours": 10.0 / 3600.0,
            "effective_parallelism": {
                "average_gpu_jobs": 0.5,
                "peak_running_jobs": 1,
            },
            "queue": {
                "queued_jobs": 1,
                "maximum_start_delay_seconds": 0.0,
            },
            "status_counts": {
                "succeeded": 1,
                "pending": 0,
                "running": 0,
                "failed_infrastructure": 0,
                "failed_terminal": 0,
                "blocked_dependency": 0,
            },
            "retries": {
                "infrastructure_retry_count": 0,
                "jobs_retried": 0,
            },
            "per_gpu": {
                "0": {
                    "busy_seconds": 10.0,
                    "jobs_started": 1,
                    "job_attempts": [
                        {
                            "job_id": "gear-r00-0",
                            "attempt": 1,
                            "gpu": 0,
                            "gpu_seconds": 10.0,
                            "returncode": 0,
                        }
                    ],
                }
            },
        },
    )
    _write(
        scheduler / "state.json",
        {
            "started_wall_time": 100.0,
            "jobs": {
                "gear-r00-0": {
                    "status": "succeeded",
                    "input_snapshot": {"row_id": "row"},
                    "attempt_history": [
                        {
                            "attempt": 1,
                            "gpu": 0,
                            "gpu_seconds": 10.0,
                            "returncode": 0,
                            "started_wall_time": 100.0,
                        }
                    ],
                    "finished_wall_time": 120.0,
                }
            },
        },
    )
    stale = tmp_path / "schedulers/round_00_stale"
    stale_manifest = {
        **manifest,
        "jobs": [{**manifest["jobs"][0], "job_id": "gear-r00-stale"}],
    }
    stale_manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in stale_manifest.items()
                if key != "manifest_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write(stale / "manifest.json", stale_manifest)
    _write(
        stale / "audit.json",
        {
            "schema_version": "fcooper_tvm_gpu_scheduler_audit_v1",
            "status_counts": {
                "succeeded": 0,
                "pending": 0,
                "running": 1,
                "failed_infrastructure": 0,
                "failed_terminal": 0,
                "blocked_dependency": 0,
            },
            "effective_parallelism": {
                "average_gpu_jobs": 1.0,
                "peak_running_jobs": 1,
            },
            "queue": {
                "queued_jobs": 1,
                "maximum_start_delay_seconds": 0.0,
            },
        },
    )
    _write(
        stale / "state.json",
        {
            "started_wall_time": 90.0,
            "jobs": {
                "gear-r00-stale": {
                    "status": "running",
                    "input_snapshot": {"row_id": "row"},
                    "attempt_history": [
                        {
                            "attempt": 1,
                            "gpu": 1,
                            "gpu_seconds": 0.0,
                            "returncode": None,
                            "started_wall_time": 90.0,
                        }
                    ],
                    "finished_wall_time": None,
                }
            },
        },
    )
    source = _write(
        tmp_path / "source.json",
        {
            "schema_version": "fcooper_tvm_backend_neutral_source_reuse_audit_v1",
            "passed": True,
            "backend_neutral_only": True,
            "source_reused_from_trt_v2": True,
            "source_generated_in_tvm_v1": False,
            "trt_compiled_artifacts_reused": False,
            "trt_performance_labels_reused": False,
            "trt_predictions_or_ap_reused": False,
            "recovery_training_seconds_saved": 42.0,
        },
    )
    pool = _write(
        tmp_path / "pool.json",
        {
            "schema_version": "fcooper_tvm_stage6_evidence_pool_v1",
            "pool_name": "gear",
            "rows": [
                {
                    "row_id": "row",
                    "tvm_trials": 64,
                    "terminal_status": "measured_success_gold",
                    "artifacts": {
                        "source_provenance": {
                            "path": str(source),
                            "sha256": _sha(source),
                        }
                    },
                }
            ],
        },
    )

    audit = build_resource_audit(
        evidence_root=tmp_path,
        pools={"gear": pool},
        outer_budget=16,
    )

    assert audit["passed"]
    assert set(audit["per_gpu"]) == {str(index) for index in range(8)}
    assert audit["per_gpu"]["0"]["jobs_started"] == 1
    assert audit["per_gpu"]["7"]["jobs_started"] == 0
    assert audit["formal_measurement"]["tvm_trials"] == 64
    assert audit["scheduler"]["formal_pair_coverage"]["expected"] == 1
    assert audit["scheduler"]["formal_pair_coverage"]["successful"] == 1
    assert audit["scheduler"]["formal_pair_coverage"]["missing"] == []
    assert audit["scheduler"]["nonterminal_or_abandoned_components"] == []
    assert audit["scheduler"]["superseded_abandoned_components"] == [
        "round_00_stale"
    ]
    assert audit["scheduler"]["queue"]["queued_jobs"] == 2
    assert audit["source_reuse"]["reused_from_trt_v2_rows"] == 1
    assert audit["source_reuse"]["recovery_training_seconds_saved"] == 42.0


def test_cross_pool_measurement_identity_reuse_is_rejected(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "source.json",
        {
            "passed": True,
            "backend_neutral_only": True,
            "source_reused_from_trt_v2": True,
            "source_generated_in_tvm_v1": False,
            "trt_compiled_artifacts_reused": False,
            "trt_performance_labels_reused": False,
            "trt_predictions_or_ap_reused": False,
        },
    )
    row = {
        "row_id": "same-row",
        "tvm_trials": 64,
        "terminal_status": "measured_success_gold",
        "artifacts": {
            "source_provenance": {
                "path": str(source),
                "sha256": _sha(source),
            }
        },
    }
    pools = {}
    for name in ("gear", "compress_then_tune_tuned"):
        pools[name] = _write(
            tmp_path / f"{name}.json",
            {
                "schema_version": "fcooper_tvm_stage6_evidence_pool_v1",
                "pool_name": name,
                "rows": [row],
            },
        )
    try:
        formal_measurement_summary(pools)
    except ValueError as exc:
        assert "measurement identity" in str(exc)
    else:
        raise AssertionError("cross-pool measurement identity reuse was accepted")
