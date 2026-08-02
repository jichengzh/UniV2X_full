from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from framework.stage7 import core_ablation_v2 as core
from framework.stage7 import core_cache_v2
from framework.stage7.online_component_ablation_v1 import SEEDS
from scripts import stage7_finalize_core_ablation_v2 as finalizer


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _events() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in core.CORE_VARIANTS:
        for seed in SEEDS:
            for round_index in range(4):
                for row_index in range(4):
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "round_index": round_index,
                            "event_index": round_index * 4 + row_index,
                            "candidate_id": (
                                f"{variant}-{seed}-{round_index}-{row_index}"
                            ),
                            "terminal_status": "measured_success_gold",
                            "terminal_evidence_kind": "actual_v3_success",
                            "cache_disposition": "miss",
                            "latency_ms": 2.0 + row_index,
                            "energy_j": 1.0 + row_index,
                            "ap70": 0.7 + round_index / 100,
                            "q_mode": "fp16",
                            "logical_request_sha256": (f"{round_index + 1:x}" * 64)[
                                :64
                            ],
                            "silent_surrogate_fallback_count": 0,
                        }
                    )
    return rows


def _evidence(events: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": finalizer.FORMAL_EVIDENCE_SCHEMA,
        "events": events,
        "trajectory_count": 12,
        "selected_event_count": len(events),
        "miss_count": len(events),
        "actual_v3_miss_evidence_count": len(events),
        "silent_surrogate_fallback_count": 0,
        "ordered_pre_scan_sha256": core.EXPECTED_ORDERED_PRE_SCAN_SHA256,
        "contract_sha256": "a" * 64,
        "audits": {name: True for name in finalizer.REQUIRED_AUDIT_FLAGS},
        "formal_v2_gpu_jobs_launched": len(events),
        "initial_gold176_rows": [
            {
                "row_id": "gold-baseline",
                "latency_ms": 8.0,
                "energy_j": 6.0,
                "ap70": 0.65,
            }
        ],
        "hv_reference": [10.0, 8.0, 0.0],
    }


def _patch_collection(
    monkeypatch: pytest.MonkeyPatch, evidence: dict[str, object]
) -> None:
    monkeypatch.setattr(
        finalizer,
        "collect_formal_evidence",
        lambda *_args, **_kwargs: evidence,
    )


@pytest.mark.parametrize("delta", [-1, 1])
def test_finalizer_rejects_any_count_other_than_192(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delta: int
) -> None:
    events = _events()
    if delta < 0:
        events.pop()
    else:
        events.append(dict(events[-1]))
    _patch_collection(monkeypatch, _evidence(events))

    with pytest.raises(finalizer.FinalizationError, match="192"):
        finalizer.finalize_v2(tmp_path / "input", tmp_path / "output")


def test_finalizer_rejects_missing_actual_v3_evidence_and_silent_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = _events()
    events[0] = {
        **events[0],
        "terminal_evidence_kind": "legacy_terminal_evidence_v1",
    }
    _patch_collection(monkeypatch, _evidence(events))
    with pytest.raises(finalizer.FinalizationError, match="actual-v3"):
        finalizer.finalize_v2(tmp_path / "input", tmp_path / "output")

    events = _events()
    evidence = _evidence(events)
    evidence["silent_surrogate_fallback_count"] = 1
    _patch_collection(monkeypatch, evidence)
    with pytest.raises(finalizer.FinalizationError, match="silent"):
        finalizer.finalize_v2(tmp_path / "input", tmp_path / "output")


def test_finalizer_emits_complete_paper_and_audit_artifacts_only_after_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(_events())
    _patch_collection(monkeypatch, evidence)
    output = tmp_path / "output"

    result = finalizer.finalize_v2(tmp_path / "input", output)

    assert result["core_ablation_ready"] is True
    assert result["paper_ready"] is True
    assert result["full_gear_s7_ready"] is False
    assert result["scanner_component_status"] == "deferred_important_fix"
    expected = {
        "aggregate/stage7_core_events_raw.json",
        "aggregate/stage7_core_events_raw.csv",
        "aggregate/stage7_core_trajectories_raw.json",
        "aggregate/stage7_core_trajectories_raw.csv",
        "aggregate/stage7_core_audit_bundle.json",
        "aggregate/stage7_core_descriptive_paired_statistics.json",
        "aggregate/stage7_core_root_cause_summary.md",
        "paper/table_stage7_core_component_ablation.csv",
        "paper/table_stage7_core_component_ablation.md",
        "status/finalization_status.json",
    }
    assert expected == {
        str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
    }
    stats = json.loads(
        (
            output / "aggregate/stage7_core_descriptive_paired_statistics.json"
        ).read_text()
    )
    encoded = json.dumps(stats).lower()
    assert "wilcoxon" not in encoded
    assert "p_value" not in encoded
    assert stats["statistics_policy"] == "descriptive_paired_only"


def test_finalizer_rejects_failed_global_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(_events())
    evidence["audits"] = {
        **evidence["audits"],
        "gpu_exclusive": False,
    }
    _patch_collection(monkeypatch, evidence)

    with pytest.raises(finalizer.FinalizationError, match="audit"):
        finalizer.finalize_v2(tmp_path / "input", tmp_path / "output")


def _dimensions(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "model": "pyramid",
        "capability_profile_id": "h800-tvm-auto-v3",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "measurement_scope": "formal_h800_latency_energy_ap",
        "input_protocol_sha256": "1" * 64,
        "batch_size": 1,
        "genome": [16, 32, 64, "fp16"],
        "q_mode": "fp16",
        "source_checkpoint_sha256": "2" * 64,
        "onnx_sha256": "3" * 64,
        "build_protocol_sha256": "4" * 64,
        "tuning_protocol_sha256": "5" * 64,
        "measurement_protocol_sha256": "6" * 64,
        "ap_protocol_sha256": "7" * 64,
        "runtime_contract_sha256": "8" * 64,
    }


def _formal_round(root: Path, variant: str, seed: int, round_index: int) -> Path:
    directory = (
        root / "variants" / variant / f"seed_{seed}" / f"round_{round_index:02d}"
    )
    rows = [
        {
            "row_id": f"{variant}-{seed}-{round_index}-{index}",
            "manifest_job_id": f"{variant}-{seed}-{round_index}-{index}",
            "q_mode": "fp16",
        }
        for index in range(4)
    ]
    request = {"rows": rows, "measurement_request_sha256": "9" * 64}
    selected = [
        {
            "candidate_id": row["row_id"],
            "exact_key_dimensions": _dimensions(row["row_id"]),
            "exact_cache_key_sha256": f"{index + 1:x}" * 64,
        }
        for index, row in enumerate(rows)
    ]
    binding = {"selected_candidates": selected}
    snapshot = {
        "schema_version": core_cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    reveal = {
        "entries": [
            {"candidate_id": row["row_id"], "disposition": "miss"} for row in rows
        ]
    }
    historical = [{**row, "terminal_status": "measured_success_gold"} for row in rows]
    promoted = [
        {
            **row,
            "terminal_status": "measured_success_gold",
            "feedback_feature_contract": "actual_feedback_v3",
        }
        for row in rows
    ]
    wrappers = {
        selected[index]["exact_cache_key_sha256"]: {
            "candidate_id": row["row_id"],
            "exact_key_dimensions": selected[index]["exact_key_dimensions"],
            "exact_cache_key_sha256": selected[index]["exact_cache_key_sha256"],
            "latency_ms": 1.0,
            "energy_j": 0.5,
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.7,
        }
        for index, row in enumerate(rows)
    }
    artifacts = {
        "logical_request.json": request,
        "selection_binding.json": binding,
        "cache_snapshot_before_reveal.json": snapshot,
        "cache_reveal.json": reveal,
        "miss_only_physical_request.json": {"physical_row_count": 4},
        "executor_admission.json": {"admission_passed": True},
        "historical_feedback.json": historical,
        "promoted_feedback.json": promoted,
        "promotion_audit.json": {
            "promoted_row_count": 4,
            "silent_surrogate_fallback_count": 0,
        },
        "cache_after_round.json": {
            "schema_version": core_cache_v2.CACHE_SCHEMA,
            "entries": wrappers,
            "lineage": [],
        },
    }
    for name, payload in artifacts.items():
        _write_json(directory / name, payload)
    return directory


def _patch_round_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        finalizer.online, "validate_committed_barrier", lambda _path: {}
    )
    monkeypatch.setattr(
        finalizer.core_cache_v2,
        "validate_selection_binding",
        lambda _request, binding: binding,
    )
    monkeypatch.setattr(
        finalizer.core_cache_v2,
        "reveal_v2_cache_after_selection",
        lambda _request, binding, _snapshot: {
            "entries": [
                {
                    "candidate_id": item["candidate_id"],
                    "disposition": "miss",
                }
                for item in binding["selected_candidates"]
            ]
        },
    )
    monkeypatch.setattr(
        finalizer.actual_adapter,
        "build_miss_only_physical_plan",
        lambda *_args, **_kwargs: {"physical_row_count": 4},
    )
    monkeypatch.setattr(
        finalizer.online,
        "validate_executor_admission",
        lambda *_args, **_kwargs: {"admission_passed": True},
    )
    monkeypatch.setattr(
        finalizer.core_cache_v2, "validate_actual_v3_cache", lambda cache: cache
    )
    monkeypatch.setattr(
        finalizer.actual_adapter,
        "validate_actual_v3_terminal_wrapper",
        lambda wrapper: wrapper,
    )


def test_formal_round_collector_uses_actual_v3_schema_and_exact_dimensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _formal_round(tmp_path, "full", 20260718, 0)
    _patch_round_validators(monkeypatch)

    events, evidence_count = finalizer._audit_formal_round(
        directory,
        variant="full",
        seed=20260718,
        round_index=0,
        contract_sha256="a" * 64,
    )

    assert evidence_count == 4
    assert len(events) == 4
    assert all(row["terminal_evidence_kind"] == "actual_v3_success" for row in events)


def test_formal_round_requires_authenticated_candidate_failure_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _formal_round(tmp_path, "full", 20260718, 0)
    _patch_round_validators(monkeypatch)
    historical_path = directory / "historical_feedback.json"
    promoted_path = directory / "promoted_feedback.json"
    historical = json.loads(historical_path.read_text())
    promoted = json.loads(promoted_path.read_text())
    candidate_id = historical[0]["row_id"]
    historical[0]["terminal_status"] = "feasibility_failure"
    promoted[0]["terminal_status"] = "feasibility_failure"
    _write_json(historical_path, historical)
    _write_json(promoted_path, promoted)
    wrapper = {"candidate_id": candidate_id}
    _write_json(
        directory / "actual_v3_failure_wrappers.json",
        {
            "schema_version": "stage7_actual_v3_failure_wrapper_batch_v2",
            "wrappers": [wrapper],
        },
    )
    monkeypatch.setattr(
        finalizer.actual_adapter,
        "validate_actual_v3_failure",
        lambda *_args, **_kwargs: {
            "candidate_id": candidate_id,
            "failure_class": "candidate",
            "consumes_selected_event_budget": True,
        },
    )

    events, evidence_count = finalizer._audit_formal_round(
        directory,
        variant="full",
        seed=20260718,
        round_index=0,
        contract_sha256="a" * 64,
    )

    assert evidence_count == 4
    assert events[0]["terminal_evidence_kind"] == "actual_v3_candidate_failure"
    assert events[0]["latency_ms"] is None


def test_collect_formal_evidence_requires_bound_closure_and_exact_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "formal"
    for variant in core.CORE_VARIANTS:
        for seed in SEEDS:
            for round_index in range(4):
                _formal_round(root, variant, seed, round_index)
    bound_artifacts = []
    for kind in finalizer.REQUIRED_AUDIT_FLAGS:
        source_suffix = ".jsonl" if kind == "gpu_exclusive" else ".json"
        source_path = root / f"status/{kind}_source{source_suffix}"
        source_schema = sorted(finalizer.AUDIT_SOURCE_SCHEMAS[kind])[0]
        if source_suffix == ".jsonl":
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(
                "\n".join(
                    json.dumps(
                        {"schema_version": source_schema, "sample": index},
                        sort_keys=True,
                    )
                    for index in range(2)
                )
                + "\n"
            )
        else:
            _write_json(source_path, {"schema_version": source_schema})
        audit_path = root / f"status/{kind}_audit.json"
        _write_json(
            audit_path,
            {
                "schema_version": finalizer.REQUIRED_AUDIT_SCHEMAS[kind],
                "passed": True,
                "evidence_origin": "formal_h800_actual_v3",
                "synthetic_non_measurement": False,
                "eligible_for_formal_finalization": True,
                "source_artifact_count": 1,
                "source_artifacts": [
                    {
                        "path": str(source_path.relative_to(root)),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "schema_version": source_schema,
                    }
                ],
            },
        )
        bound_artifacts.append(
            {
                "kind": kind,
                "path": f"status/{kind}_audit.json",
                "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            }
        )
    closure_payload = {
        "schema_version": finalizer.EXECUTION_CLOSURE_SCHEMA,
        "passed": True,
        "trajectory_count": 12,
        "selected_event_count": 192,
        "miss_count": 192,
        "actual_v3_miss_evidence_count": 192,
        "silent_surrogate_fallback_count": 0,
        "formal_v2_gpu_jobs_launched": 192,
        "audits": {name: True for name in finalizer.REQUIRED_AUDIT_FLAGS},
        "bound_audit_artifacts": bound_artifacts,
    }
    _write_json(
        root / "status/formal_execution_closure_v2.json",
        {
            **closure_payload,
            "execution_closure_sha256": finalizer._sha(closure_payload),
        },
    )
    monkeypatch.setattr(
        finalizer.online,
        "validate_formal_v2_root",
        lambda *_args, **_kwargs: {
            "contract": {
                "contract_sha256": "a" * 64,
                "ordered_pre_scan_sha256": core.EXPECTED_ORDERED_PRE_SCAN_SHA256,
            }
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_audit_formal_round",
        lambda directory, variant, seed, round_index, **_kwargs: (
            [
                {
                    "variant": variant,
                    "seed": seed,
                    "round_index": round_index,
                    "event_index": round_index * 4 + index,
                    "candidate_id": f"{variant}-{seed}-{round_index}-{index}",
                    "terminal_status": "measured_success_gold",
                    "terminal_evidence_kind": "actual_v3_success",
                    "cache_disposition": "miss",
                    "q_mode": "fp16",
                    "latency_ms": 1.0,
                    "energy_j": 0.5,
                    "ap30": 0.9,
                    "ap50": 0.8,
                    "ap70": 0.7,
                    "logical_request_sha256": "9" * 64,
                    "silent_surrogate_fallback_count": 0,
                }
                for index in range(4)
            ],
            4,
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_load_metric_context",
        lambda _contract: (
            [{"latency_ms": 8.0, "energy_j": 6.0, "ap70": 0.65}],
            (10.0, 8.0, 0.0),
        ),
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_semantic_audit_source",
        lambda *_args, **_kwargs: None,
    )

    evidence = finalizer.collect_formal_evidence(root, repo_root=tmp_path)

    assert evidence["trajectory_count"] == 12
    assert evidence["selected_event_count"] == 192
    assert evidence["actual_v3_miss_evidence_count"] == 192
    assert {row["kind"] for row in evidence["bound_audit_artifacts"]} == set(
        finalizer.REQUIRED_AUDIT_FLAGS
    )
    assert evidence["hv_reference"] == [10.0, 8.0, 0.0]


def test_bound_formal_audits_reject_synthetic_dry_run_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audits/no_gpu_dry_run/cache.json"
    _write_json(
        path,
        {
            "schema_version": finalizer.REQUIRED_AUDIT_SCHEMAS["cache"],
            "passed": True,
            "evidence_origin": "formal_h800_actual_v3",
            "synthetic_non_measurement": True,
            "eligible_for_formal_finalization": False,
        },
    )
    records = [
        {
            "kind": "cache",
            "path": "audits/no_gpu_dry_run/cache.json",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ]

    with pytest.raises(finalizer.FinalizationError, match="audit"):
        finalizer._validate_bound_audit_artifacts(tmp_path, records)


def test_semantic_audit_sources_run_real_content_validators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = {"schema_version": finalizer.actual_adapter.TERMINAL_WRAPPER_SCHEMA}
    terminal_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        finalizer.actual_adapter,
        "validate_actual_v3_terminal_wrapper",
        lambda row: terminal_calls.append(dict(row)) or dict(row),
    )
    monkeypatch.setattr(
        finalizer.core_cache_v2,
        "validate_actual_v3_cache",
        lambda _row: {"entries": {"key": terminal}, "lineage": [{"record": 1}]},
    )
    cache_path = tmp_path / "cache.json"
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="cache",
        path=cache_path,
        schema=core_cache_v2.CACHE_SCHEMA,
        rows=[{"schema_version": core_cache_v2.CACHE_SCHEMA}],
    )
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="numerical",
        path=tmp_path / "terminal.json",
        schema=finalizer.actual_adapter.TERMINAL_WRAPPER_SCHEMA,
        rows=[terminal],
    )
    assert len(terminal_calls) == 2

    barrier_path = tmp_path / "round_00/atomic_feedback_barrier.json"
    barrier_calls: list[Path] = []
    monkeypatch.setattr(
        finalizer.online,
        "validate_committed_barrier",
        lambda path: barrier_calls.append(path) or {},
    )
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="budget",
        path=barrier_path,
        schema=finalizer.online.BARRIER_SCHEMA,
        rows=[{"schema_version": finalizer.online.BARRIER_SCHEMA}],
    )
    assert barrier_calls == [barrier_path.parent]

    atomic = {
        "schema_version": "stage5_atomic_batch_audit_v2",
        "feedback_released": True,
        "batch_quarantined": False,
        "budget_consumed": 4,
        "released_feedback_rows": [{}, {}, {}, {}],
        "resume_row_ids": [],
    }
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="budget",
        path=tmp_path / "atomic.json",
        schema="stage5_atomic_batch_audit_v2",
        rows=[atomic],
    )

    logical_path = tmp_path / "selection/logical_request.json"
    _write_json(logical_path, {"schema_version": "logical"})
    selection_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        finalizer.core_cache_v2,
        "validate_selection_binding",
        lambda request, binding: selection_calls.append((request, binding)) or {},
    )
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="cache",
        path=logical_path.parent / "selection_binding.json",
        schema=core_cache_v2.SELECTION_BINDING_SCHEMA,
        rows=[{"schema_version": core_cache_v2.SELECTION_BINDING_SCHEMA}],
    )
    assert len(selection_calls) == 1

    selected = ["a", "b", "c", "d"]
    selector = {
        "schema_version": "stage7_actual_v3_selector_audit_v2",
        "variant": "full",
        "round_index": 0,
        "ordered_pre_scan_sha256": core.EXPECTED_ORDERED_PRE_SCAN_SHA256,
        "candidate_pool": "pre_scan",
        "scanner_deferred": True,
        "cache_visible_during_selection": False,
        "cache_input_accepted": False,
        "selected_row_ids": selected,
        "selected_row_ids_sha256": finalizer._sha(selected),
        "logical_request_sha256": "a" * 64,
    }
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="label_leakage",
        path=tmp_path / "selector.json",
        schema="stage7_actual_v3_selector_audit_v2",
        rows=[selector],
    )

    deployed_source = tmp_path / "source.py"
    deployed_destination = tmp_path / "deployed.py"
    deployed_source.write_text("reviewed")
    deployed_destination.write_text("reviewed")
    digest = hashlib.sha256(deployed_source.read_bytes()).hexdigest()
    deploy = {
        "schema_version": "stage7_deploy_manifest_v1",
        "files": [
            {
                "source": str(deployed_source.resolve()),
                "destination": str(deployed_destination.resolve()),
                "sha256": digest,
            }
        ],
        "python_environment": "formal",
        "git": {"commit": "reviewed"},
        "frozen_evidence": {"contract": "a" * 64},
    }
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="deployment",
        path=tmp_path / "deploy.json",
        schema="stage7_deploy_manifest_v1",
        rows=[deploy],
    )

    gpu_sources = {
        "stage7_gpu_lease_audit_v1": {
            "schema_version": "stage7_gpu_lease_audit_v1",
            "event": "batch_leased",
            "wall_time": 1.0,
            "gpu_uuids": ["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
        },
        "stage7_gpu_occupancy_snapshot_v1": {
            "schema_version": "stage7_gpu_occupancy_snapshot_v1",
            "event": "sample",
            "wall_time": 1.0,
            "gpus": [{"uuid": "GPU-a", "compute_pids": []}],
        },
        "stage7_h800_runtime_admission_v2": {
            "schema_version": "stage7_h800_runtime_admission_v2",
            "event": "runtime_context_sample",
            "wall_time": 1.0,
            "hostname": "zs-nj-tap-gpu18",
            "gpu_models": {"GPU-a": "NVIDIA H800"},
            "reservations": {},
            "processes": [],
        },
    }
    for schema, row in gpu_sources.items():
        finalizer._validate_semantic_audit_source(
            tmp_path,
            kind="gpu_exclusive",
            path=tmp_path / f"{schema}.jsonl",
            schema=schema,
            rows=[row],
        )

    scheduler = {
        "schema_version": "stage7_ablation_scheduler_state_v1",
        "scheduler_status": "complete",
        "controllers": {},
        "selected_event_budget_consumed": 192,
        "updated_wall_time": 2.0,
    }
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="lock",
        path=tmp_path / "scheduler.json",
        schema="stage7_ablation_scheduler_state_v1",
        rows=[scheduler],
    )

    contract_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        finalizer.contracts,
        "validate_v2_contract",
        lambda payload: contract_calls.append(dict(payload)) or dict(payload),
    )
    finalizer._validate_semantic_audit_source(
        tmp_path,
        kind="path_isolation",
        path=tmp_path / "contract.json",
        schema=core.SCHEMA_VERSION,
        rows=[{"schema_version": core.SCHEMA_VERSION}],
    )
    assert len(contract_calls) == 1


def test_metric_context_is_bound_to_frozen_gold_and_task_hv_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gold_path = tmp_path / "gold176.json"
    reference_path = tmp_path / "hv_reference.json"
    gold_row = {
        "row_id": "gold",
        "model": "pyramid",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "terminal_status": "measured_success_gold",
        "latency_ms": 8.0,
        "energy_j": 6.0,
        "ap70": 0.65,
    }
    _write_json(gold_path, {"rows": [gold_row]})
    _write_json(
        reference_path,
        {
            "values": {
                "latency_ms": 10.0,
                "energy_j": 8.0,
                "negative_ap70": 0.0,
            }
        },
    )
    monkeypatch.setattr(
        finalizer.metric_helpers,
        "_audit_result_row",
        lambda row: dict(row),
    )
    contract = {
        "immutable_inputs": {
            "gold176": {
                "path": str(gold_path.resolve()),
                "sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            },
            "hv_reference": {
                "path": str(reference_path.resolve()),
                "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            },
        }
    }

    initial, reference = finalizer._load_metric_context(contract)

    assert initial == [gold_row]
    assert reference == (10.0, 8.0, 0.0)
