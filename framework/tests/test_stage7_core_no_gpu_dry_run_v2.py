from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts import stage7_core_no_gpu_dry_run_v2 as dry_run


RELEASE_PIN = "d" * 64
MANIFEST_FILE_PIN = "e" * 64


def _mock_authenticated_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dry_run,
        "deployment_binding",
        lambda *_args, **_kwargs: {
            "deployment_manifest_sha256": "a" * 64,
            "deployment_manifest_file_sha256": MANIFEST_FILE_PIN,
            "deployment_bundle_sha256": "c" * 64,
            "deployment_release_sha256": RELEASE_PIN,
        },
    )


def _request() -> dict[str, object]:
    rows = [
        {
            "row_id": f"candidate-{index}",
            "manifest_job_id": f"candidate-{index}",
            "model": "pyramid",
            "width": [16 + index, 32, 64],
            "group_id": f"pyramid|{16 + index}x32x64",
            "graph_features": {
                "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
            },
        }
        for index in range(4)
    ]
    return {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "round_index": 0,
        "rows": rows,
        "row_sha256": {row["row_id"]: dry_run.canonical_sha256(row) for row in rows},
        "measurement_request_sha256": "a" * 64,
    }


def test_synthetic_fixture_is_permanently_ineligible_for_truth_or_cache(
    tmp_path: Path,
) -> None:
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")

    fixture = dry_run.build_synthetic_protocol_fixture(
        _request(),
        frozen_onnx=onnx,
        output_dir=tmp_path / "synthetic",
    )

    assert fixture["synthetic_non_measurement"] is True
    assert fixture["eligible_for_cache_append"] is False
    assert fixture["eligible_for_formal_finalization"] is False
    assert fixture["actual_v3_hardware_evidence"] is False
    assert fixture["formal_v2_gpu_jobs_launched"] == 0
    assert len(fixture["historical_feedback"]) == 4
    assert all(
        row["metric_source"] == "synthetic_protocol_fixture"
        for row in fixture["historical_feedback"]
    )
    assert all(
        row["training_source"] == "online_feedback"
        and row["synthetic_training_source_contract_only"] is True
        for row in fixture["historical_feedback"]
    )


def test_synthetic_fixture_never_uses_candidate_id_as_a_path_component(
    tmp_path: Path,
) -> None:
    request = _request()
    request["rows"][0]["row_id"] = "x/../../../escaped"
    request["rows"][0]["manifest_job_id"] = "x/../../../escaped"
    request["row_sha256"] = {
        row["row_id"]: dry_run.canonical_sha256(row) for row in request["rows"]
    }
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")
    output = tmp_path / "isolated"

    dry_run.build_synthetic_protocol_fixture(
        request,
        frozen_onnx=onnx,
        output_dir=output,
    )

    assert not (tmp_path / "escaped.json").exists()
    source_paths = [
        Path(record["path"])
        for record in json.loads(
            (output / "synthetic_protocol_fixture.json").read_text()
        )["source_records"]
    ]
    assert all(path.parent == output / "synthetic_sources" for path in source_paths)
    assert all("escaped" not in path.name for path in source_paths)


def test_environment_gate_requires_explicit_empty_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        dry_run.require_no_gpu_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        dry_run.require_no_gpu_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    dry_run.require_no_gpu_environment()


def test_deployment_binding_authenticates_prepare_state_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "formal-v2"
    root.mkdir()
    state = {
        "owner": "runner",
        "owner_uid": 4242,
        "deployment_manifest_sha256": "a" * 64,
        "deployment_manifest_file_sha256": "b" * 64,
        "deployment_bundle_sha256": "c" * 64,
        "deployment_release_sha256": RELEASE_PIN,
    }
    (root / "prepare_state.json").write_text(json.dumps(state))
    calls: list[dict[str, object]] = []

    def validate(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {
            key: state[key]
            for key in (
                "deployment_manifest_sha256",
                "deployment_manifest_file_sha256",
                "deployment_bundle_sha256",
                "deployment_release_sha256",
            )
        }

    monkeypatch.setattr(
        dry_run.deployment_bundle_v2, "validate_deployment_bundle", validate
    )
    result = dry_run.deployment_binding(
        root,
        frozen_repo_root=tmp_path,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
    )

    assert calls == [{
        "frozen_repo_root": tmp_path,
        "expected_release_sha256": RELEASE_PIN,
        "expected_manifest_file_sha256": MANIFEST_FILE_PIN,
        "expected_owner": "runner",
        "expected_owner_uid": 4242,
    }]
    assert result["deployment_bundle_sha256"] == "c" * 64


def test_isolated_json_writer_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        dry_run.os, "fsync", lambda descriptor: calls.append(descriptor)
    )
    dry_run._write_isolated_json(tmp_path / "nested" / "receipt.json", {"ok": True})
    assert len(calls) == 2


def test_fresh_root_entry_prepares_embedded_contracts_before_initialization(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-v2"
    source_manifest = tmp_path / "frozen-sources/manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text('{"schema_version":"frozen"}\n')
    calls: list[str] = []

    def prepare_root(**kwargs: object) -> dict[str, object]:
        calls.append("prepare")
        assert kwargs["v2_root"] == root
        assert kwargs["embedded_source_manifest"]["schema_version"] == "frozen"
        contracts = root / "contracts"
        contracts.mkdir(parents=True)
        (contracts / "scope_input_batch.json").write_text("{}\n")
        (contracts / "measurement_ap.json").write_text("{}\n")
        return {
            "schema_version": "stage7_core_ablation_v2_prepare_result",
            "embedded_immutable_contracts_written": [
                "measurement_ap",
                "scope_input_batch",
            ],
        }

    def initialize(v2_root: Path, **_kwargs: object) -> dict[str, object]:
        calls.append("initialize")
        assert (v2_root / "contracts/scope_input_batch.json").is_file()
        assert (v2_root / "contracts/measurement_ap.json").is_file()
        return {"trajectory_count": 12}

    result = dry_run.prepare_and_initialize_no_gpu_root(
        v1_root=tmp_path / "v1",
        v2_root=root,
        prepare_inputs={"pre_scan_registry": []},
        embedded_source_manifest=json.loads(source_manifest.read_text()),
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        orchestrator_pid=4242,
        repo_root=tmp_path,
        prepare_root=prepare_root,
        initialize=initialize,
    )

    assert calls == ["prepare", "initialize"]
    assert result["trajectory_count"] == 12
    assert result["embedded_immutable_contracts_written"] == [
        "measurement_ap",
        "scope_input_batch",
    ]
    assert not (root / "frozen-sources").exists()


def test_fresh_root_forwards_persistent_orchestrator_pid_to_prepare(
    tmp_path: Path,
) -> None:
    root = tmp_path / "formal-v2"
    calls: list[dict[str, object]] = []

    def prepare_root(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"orchestrator_pid": kwargs["orchestrator_pid"]}

    result = dry_run.prepare_and_initialize_no_gpu_root(
        v1_root=tmp_path / "v1",
        v2_root=root,
        prepare_inputs={"pre_scan_registry": []},
        embedded_source_manifest={"schema_version": "frozen"},
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        orchestrator_pid=4242,
        repo_root=tmp_path,
        prepare_root=prepare_root,
        initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
    )

    assert calls[0]["orchestrator_pid"] == 4242
    assert result["orchestrator_pid"] == 4242


def test_dry_run_orchestrator_closes_four_variants_without_formal_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _mock_authenticated_deployment(monkeypatch)
    calls: list[tuple[str, str]] = []

    def prepare(_root: Path, *, variant: str, **_kwargs: object) -> dict:
        assert _kwargs["expected_release_sha256"] == RELEASE_PIN
        assert _kwargs["expected_manifest_file_sha256"] == MANIFEST_FILE_PIN
        calls.append(("prepare", variant))
        return {"logical_request_sha256": variant.ljust(64, "0")[:64]}

    def inspect(_root: Path, variant: str) -> dict:
        calls.append(("inspect", variant))
        return {
            "request": _request(),
            "cache_snapshot": {
                "schema_version": "stage7_core_cache_v2",
                "entries": {},
                "lineage": [],
            },
            "cache_reveal": {
                "entries": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "disposition": "miss",
                    }
                    for index in range(4)
                ]
            },
            "physical_plan": {"physical_row_count": 4},
            "executor_admission": {
                "admission_passed": True,
                "gpu_jobs_launched": 0,
            },
        }

    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")
    result = dry_run.run_no_gpu_dry_run(
        tmp_path / "formal-v2",
        repo_root=tmp_path,
        frozen_onnx=onnx,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
        validate_root=lambda *_args, **_kwargs: {
            "contract": {"contract_sha256": "c" * 64}
        },
        prepare_round=prepare,
        inspect_round=inspect,
        admit_misses=lambda *_args, variant, **_kwargs: calls.append(("admit", variant))
        or {"admission_passed": True, "gpu_jobs_launched": 0},
        promote_synthetic=lambda fixture, **_kwargs: {
            "rows": fixture["historical_feedback"],
            "audit": {
                "promoted_row_count": 4,
                "silent_surrogate_fallback_count": 0,
            },
            "barrier": {"feedback_released": True, "budget_consumed": 4},
        },
        select_next=lambda *_args, variant, **_kwargs: {
            "variant": variant,
            "round_index": 1,
            "rows": [f"next-{index}" for index in range(4)],
        },
    )

    assert result["trajectory_count"] == 12
    assert result["round0_request_count"] == 4
    assert result["round0_miss_count"] == 16
    assert result["integration_closed"] is True
    assert result["blocking_reason"] is None
    assert all(
        receipt["miss_admission_passed"] is True
        and receipt["round0_miss_count"] == 4
        for receipt in result["variant_receipts"]
    )
    assert result["synthetic_fixture_count"] == 4
    assert result["next_round_request_count"] == 4
    assert result["formal_v2_gpu_jobs_launched"] == 0
    assert result["paper_ready"] is False
    assert result["core_ablation_ready"] is False
    assert result["canonical_initialization_artifacts_written"] is True
    assert result["canonical_round0_requests_written"] == 4
    assert result["canonical_terminal_artifacts_written"] == 0
    assert result["canonical_barriers_written"] == 0
    assert result["cache_appends"] == 0
    assert [value for action, value in calls if action == "prepare"] == [
        "full",
        "without_surrogate",
        "without_measured_feedback",
        "backend_blind",
    ]
    receipt = json.loads(
        (tmp_path / "formal-v2/audits/no_gpu_dry_run/dry_run_receipt.json").read_text()
    )
    assert receipt["synthetic_non_measurement"] is True
    assert receipt["eligible_for_formal_finalization"] is False
    assert receipt["formal_v2_root"] == str((tmp_path / "formal-v2").resolve())
    assert receipt["deployment_primitive_pins_sha256"] == (
        dry_run.deployment_bundle_v2.primitive_pins_sha256()
    )


def test_dry_run_surfaces_source_resolution_blocker_before_cache_reveal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _mock_authenticated_deployment(monkeypatch)
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")
    audit = {
        "schema_version": "stage7_round0_source_resolution_preflight_v2",
        "status": "source_resolution_required_before_exact_cache_reveal",
        "selected_candidate_count": 4,
        "unresolved_candidate_count": 4,
        "unresolved_candidates": [
            {
                "candidate_id": f"candidate-{index}",
                "checkpoint_sha256": None,
                "onnx_sha256": None,
            }
            for index in range(4)
        ],
        "placeholder_sha_inserted": False,
        "cache_reveal_allowed": False,
    }

    def prepare(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise dry_run.online.SourceResolutionRequiredBeforeExactCacheReveal(audit)

    result = dry_run.run_no_gpu_dry_run(
        tmp_path / "formal-v2",
        repo_root=tmp_path,
        frozen_onnx=onnx,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
        validate_root=lambda *_args, **_kwargs: {
            "contract": {"contract_sha256": "c" * 64}
        },
        prepare_round=prepare,
        inspect_round=lambda *_args, **_kwargs: pytest.fail(
            "cache reveal must not run while selected sources are unresolved"
        ),
    )

    assert result["blocking_reason"] == (
        "source_resolution_required_before_exact_cache_reveal"
    )
    assert result["formal_v2_gpu_jobs_launched"] == 0
    assert result["canonical_round0_requests_written"] == 0
    assert result["paper_ready"] is False
    assert result["core_ablation_ready"] is False
    assert result["source_resolution_audit"] == audit
    blocker_path = (
        tmp_path / "formal-v2/audits/no_gpu_dry_run/source_resolution_blocker.json"
    )
    blocker = json.loads(blocker_path.read_text())
    assert blocker["cache_reveal_allowed"] is False
    assert blocker["placeholder_sha_inserted"] is False
    assert not (
        tmp_path / "formal-v2/audits/no_gpu_dry_run/synthetic_protocol_fixture.json"
    ).exists()


def test_no_gpu_gate_binds_existing_pre_gate_source_before_exact_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _mock_authenticated_deployment(monkeypatch)
    root = tmp_path / "formal-v2"
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")
    calls: list[tuple[str, str]] = []

    def prepare(
        _root: Path, *, variant: str, seed: int, round_index: int, **_kwargs: object
    ) -> dict[str, object]:
        directory = root / "variants" / variant / f"seed_{seed}" / "round_00"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "source_resolution_result.json").write_text(
            json.dumps({"schema_version": "stage7_source_resolution_result_v2"})
        )
        return {"controller_state": "SOURCE_PLAN_FROZEN"}

    def bind_formal(
        _root: Path,
        *,
        variant: str,
        source_result_path: Path,
        synthetic_no_gpu_dryrun: bool,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert source_result_path == (
            root
            / "variants"
            / variant
            / "seed_20260718"
            / "round_00"
            / "source_resolution_result.json"
        )
        assert synthetic_no_gpu_dryrun is False
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        calls.append(("bind", variant))
        directory = source_result_path.parent
        for name in (
            "exact_selection_binding.json",
            "cache_snapshot_before_reveal.json",
            "cache_reveal.json",
            "miss_only_physical_request.json",
            "executor_admission.json",
        ):
            (directory / name).write_text("{}")
        return {"controller_state": "CACHE_REVEALED"}

    def inspect(_root: Path, variant: str) -> dict[str, object]:
        calls.append(("inspect", variant))
        return {
            "request": _request(),
            "cache_snapshot": {
                "schema_version": "stage7_core_cache_v2",
                "entries": {},
                "lineage": [],
            },
            "cache_reveal": {
                "entries": [
                    {"candidate_id": f"candidate-{index}", "disposition": "miss"}
                    for index in range(4)
                ]
            },
            "physical_plan": {"physical_row_count": 4},
            "executor_admission": {
                "admission_passed": True,
                "gpu_jobs_launched": 0,
            },
        }

    result = dry_run.run_no_gpu_dry_run(
        root,
        repo_root=tmp_path,
        frozen_onnx=onnx,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
        validate_root=lambda *_args, **_kwargs: {
            "contract": {"contract_sha256": "c" * 64}
        },
        prepare_round=prepare,
        bind_formal_source=bind_formal,
        inspect_round=inspect,
        admit_misses=lambda *_args, **_kwargs: {
            "admission_passed": True,
            "gpu_jobs_launched": 0,
        },
        promote_synthetic=lambda fixture, **_kwargs: {
            "rows": fixture["historical_feedback"],
            "audit": {
                "promoted_row_count": 4,
                "silent_surrogate_fallback_count": 0,
            },
            "barrier": {"feedback_released": True, "budget_consumed": 4},
        },
        select_next=lambda *_args, variant, **_kwargs: {
            "variant": variant,
            "round_index": 1,
            "rows": [f"next-{index}" for index in range(4)],
        },
    )

    assert result["integration_closed"] is True
    assert result["round0_miss_count"] == 16
    assert result["formal_v2_gpu_jobs_launched"] == 0
    assert result["pre_gate_source_results_bound"] == 4
    for variant in dry_run.CORE_VARIANTS:
        assert calls.index(("bind", variant)) < calls.index(("inspect", variant))


def test_valid_source_plan_frozen_path_closes_all_quarantined_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _mock_authenticated_deployment(monkeypatch)
    root = tmp_path / "formal-v2"
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")
    request = _request()

    def prepare(
        _root: Path, *, variant: str, seed: int, round_index: int, **_kwargs: object
    ) -> dict[str, object]:
        assert seed == 20260718
        assert round_index == 0
        directory = root / "variants" / variant / f"seed_{seed}" / "round_00"
        directory.mkdir(parents=True)
        source_plan_payload = {
            "schema_version": "stage7_source_resolution_plan_v2",
            "status": "frozen",
            "logical_request_sha256": request["measurement_request_sha256"],
            "ordered_row_ids": [row["row_id"] for row in request["rows"]],
            "row_count": 4,
        }
        source_plan = {
            **source_plan_payload,
            "source_resolution_plan_sha256": dry_run.canonical_sha256(
                source_plan_payload
            ),
        }
        (directory / "logical_request.json").write_text(json.dumps(request))
        (directory / "source_resolution_plan.json").write_text(
            json.dumps(source_plan)
        )
        return {
            "controller_state": "SOURCE_PLAN_FROZEN",
            "logical_request_sha256": request["measurement_request_sha256"],
            "source_resolution_plan_sha256": source_plan[
                "source_resolution_plan_sha256"
            ],
            "cache_membership_observed": False,
            "exact_binding_frozen": False,
        }

    def bind_synthetic(
        _root: Path,
        *,
        variant: str,
        prepared: dict[str, object],
        audit_dir: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        directory = root / "variants" / variant / "seed_20260718/round_00"
        protocol = {
            "formal_miss_plan_allowed": False,
            "executor_admission_allowed": False,
            "actual_v3_hardware_evidence": False,
            "eligible_for_cache_append": False,
            "eligible_for_finalization": False,
        }
        for name, payload in {
            "synthetic_source_resolution_result.json": {
                "actual_v3_hardware_evidence": False,
            },
            "synthetic_physical_protocol.json": protocol,
            "synthetic_protocol_receipt.json": {
                **protocol,
                "controller_state": "SYNTHETIC_PROTOCOL_CLOSED",
            },
        }.items():
            (directory / name).write_text(json.dumps(payload))
        assert prepared["controller_state"] == "SOURCE_PLAN_FROZEN"
        assert audit_dir == root / "audits/no_gpu_dry_run"
        return {
            "request": json.loads((directory / "logical_request.json").read_text()),
            "receipt": {
                **protocol,
                "controller_state": "SYNTHETIC_PROTOCOL_CLOSED",
            },
        }

    result = dry_run.run_no_gpu_dry_run(
        root,
        repo_root=tmp_path,
        frozen_onnx=onnx,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
        validate_root=lambda *_args, **_kwargs: {
            "contract": {"contract_sha256": "c" * 64}
        },
        prepare_round=prepare,
        bind_synthetic_source_protocol=bind_synthetic,
        promote_synthetic=lambda fixture, **_kwargs: {
            "rows": fixture["historical_feedback"],
            "audit": {
                "promoted_row_count": 4,
                "silent_surrogate_fallback_count": 0,
            },
            "barrier": {"feedback_released": True, "budget_consumed": 4},
        },
        select_next=lambda *_args, variant, **_kwargs: {
            "variant": variant,
            "round_index": 1,
            "rows": [f"next-{index}" for index in range(4)],
        },
    )

    assert result["canonical_round0_requests_written"] == 4
    assert result["synthetic_fixture_count"] == 4
    assert result["next_round_request_count"] == 4
    assert result["formal_v2_gpu_jobs_launched"] == 0
    assert result["cache_appends"] == 0
    assert result["canonical_terminal_artifacts_written"] == 0
    assert result["canonical_barriers_written"] == 0
    assert result["paper_ready"] is False
    assert result["core_ablation_ready"] is False
    assert result["integration_closed"] is False
    assert result["blocking_reason"] == (
        "stage7_to_actual_feedback_v3_integration_not_yet_closed"
    )
    assert result["round0_miss_count"] == 0
    for variant in dry_run.CORE_VARIANTS:
        round_dir = root / "variants" / variant / "seed_20260718/round_00"
        for formal_name in (
            "source_resolution_result.json",
            "exact_selection_binding.json",
            "cache_snapshot_before_reveal.json",
            "cache_reveal.json",
            "miss_only_physical_request.json",
            "executor_admission.json",
            "atomic_feedback_barrier.json",
        ):
            assert not (round_dir / formal_name).exists()
        protocol = json.loads(
            (round_dir / "synthetic_physical_protocol.json").read_text()
        )
        assert protocol["formal_miss_plan_allowed"] is False
        assert protocol["executor_admission_allowed"] is False
        assert protocol["actual_v3_hardware_evidence"] is False


def test_source_plan_frozen_rejects_preexisting_formal_exact_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _mock_authenticated_deployment(monkeypatch)
    root = tmp_path / "formal-v2"
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real-frozen-input")

    def prepare(
        _root: Path, *, variant: str, seed: int, **_kwargs: object
    ) -> dict[str, object]:
        directory = root / "variants" / variant / f"seed_{seed}" / "round_00"
        directory.mkdir(parents=True)
        for name in (
            "cache_snapshot_before_reveal.json",
            "cache_reveal.json",
            "miss_only_physical_request.json",
            "executor_admission.json",
        ):
            (directory / name).write_text("{}\n")
        return {"controller_state": "SOURCE_PLAN_FROZEN"}

    with pytest.raises(ValueError, match="formal exact-reveal artifacts"):
        dry_run.run_no_gpu_dry_run(
            root,
            repo_root=tmp_path,
            frozen_onnx=onnx,
            expected_release_sha256=RELEASE_PIN,
            expected_manifest_file_sha256=MANIFEST_FILE_PIN,
            initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
            validate_root=lambda *_args, **_kwargs: {
                "contract": {"contract_sha256": "c" * 64}
            },
            prepare_round=prepare,
        )


def test_dry_run_rejects_any_cache_hit_or_gpu_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _mock_authenticated_deployment(monkeypatch)
    onnx = tmp_path / "frozen.onnx"
    onnx.write_bytes(b"real")

    def inspect(_root: Path, _variant: str) -> dict:
        payload = {
            "request": _request(),
            "cache_snapshot": {
                "schema_version": "stage7_core_cache_v2",
                "entries": {},
                "lineage": [],
            },
            "cache_reveal": {
                "entries": [{"candidate_id": "candidate-0", "disposition": "hit"}]
            },
            "physical_plan": {"physical_row_count": 3},
            "executor_admission": {
                "admission_passed": True,
                "gpu_jobs_launched": 0,
            },
        }
        return payload

    with pytest.raises(ValueError, match="four misses"):
        dry_run.run_no_gpu_dry_run(
            tmp_path / "root",
            repo_root=tmp_path,
            frozen_onnx=onnx,
            expected_release_sha256=RELEASE_PIN,
            expected_manifest_file_sha256=MANIFEST_FILE_PIN,
            initialize=lambda *_args, **_kwargs: {"trajectory_count": 12},
            validate_root=lambda *_args, **_kwargs: {
                "contract": {"contract_sha256": "c" * 64}
            },
            prepare_round=lambda *_args, **_kwargs: {},
            inspect_round=inspect,
            admit_misses=lambda *_args, **_kwargs: {},
            promote_synthetic=lambda *_args, **_kwargs: {},
            select_next=lambda *_args, **_kwargs: {},
        )


def test_default_round_inspection_reads_only_canonical_task4_artifacts(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "variants/full/seed_20260718/round_00"
    for name, payload in {
        "logical_request.json": {"kind": "request"},
        "cache_snapshot_before_reveal.json": {"kind": "cache"},
        "cache_reveal.json": {"kind": "reveal"},
        "miss_only_physical_request.json": {"kind": "plan"},
        "executor_admission.json": {"kind": "admission"},
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(json.dumps(payload))

    inspected = dry_run._inspect_round(tmp_path, "full")

    assert inspected["round_dir"] == directory
    assert inspected["request"]["kind"] == "request"
    assert inspected["physical_plan"]["kind"] == "plan"


def test_default_miss_admission_invokes_reviewed_dry_run_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    round_dir = tmp_path / "variants/full/seed_20260718/round_00"
    round_dir.mkdir(parents=True)
    inspected = {
        "round_dir": round_dir,
        "request": {"measurement_request_sha256": "a" * 64},
    }

    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        admission_path = Path(command[command.index("--admission-json") + 1])
        admission_path.parent.mkdir(parents=True, exist_ok=True)
        admission_path.write_text(
            json.dumps(
                {
                    "admission_passed": True,
                    "dry_run": True,
                    "gpu_jobs_launched": 0,
                }
            )
        )
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr(dry_run.subprocess, "run", run)
    admission = dry_run._admit_misses(
        tmp_path,
        variant="full",
        inspected=inspected,
        repo_root=tmp_path,
        contract_sha256="c" * 64,
        expected_release_sha256=RELEASE_PIN,
        expected_manifest_file_sha256=MANIFEST_FILE_PIN,
        audit_dir=tmp_path / "audits/no_gpu_dry_run",
    )

    assert admission["gpu_jobs_launched"] == 0
    assert calls[0][calls[0].index("--expected-release-sha256") + 1] == RELEASE_PIN
    assert (
        calls[0][calls[0].index("--expected-manifest-file-sha256") + 1]
        == MANIFEST_FILE_PIN
    )


def test_default_synthetic_promotion_calls_v3_promoter_and_atomic_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dry_run._ensure_runtime_modules(dry_run.REPO_ROOT)
    request_path = tmp_path / "logical_request.json"
    request_path.write_text(json.dumps(_request()))
    fixture = {"historical_feedback": [{"row_id": str(index)} for index in range(4)]}
    monkeypatch.setattr(
        dry_run.stage5_promote,
        "promote_feedback_batch",
        lambda *_args, **_kwargs: {
            "rows": fixture["historical_feedback"],
            "audit": {
                "promoted_row_count": 4,
                "silent_surrogate_fallback_count": 0,
            },
        },
    )
    monkeypatch.setattr(
        dry_run.stage5_search,
        "finalize_atomic_batch",
        lambda *_args, **_kwargs: {
            "feedback_released": True,
            "budget_consumed": 4,
        },
    )

    result = dry_run._promote_synthetic(
        fixture,
        request_path=request_path,
        output_dir=tmp_path / "synthetic",
    )

    assert result["audit"]["promoted_row_count"] == 4
    wrapper = json.loads(
        (tmp_path / "synthetic/synthetic_promotion_barrier.json").read_text()
    )
    assert wrapper["canonical_barrier_written"] is False
    assert wrapper["eligible_for_formal_finalization"] is False


def test_default_next_selector_uses_actual_v3_selector_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dry_run._ensure_runtime_modules(dry_run.REPO_ROOT)
    round_dir = tmp_path / "variants/full/seed_20260718/round_00"
    round_dir.mkdir(parents=True)
    (round_dir / "selector_output.json").write_text(
        json.dumps({"a2_frozen": {"bundle": "frozen"}})
    )
    (round_dir / "logical_request.json").write_text(json.dumps(_request()))
    monkeypatch.setattr(
        dry_run.online,
        "_selector_inputs",
        lambda _validated: {
            "capability_profiles": [{"profile_id": "p"}],
            "initial_rows": [],
            "initial_graph_features": [],
            "closure": {},
        },
    )
    monkeypatch.setattr(
        dry_run.search_policy, "build_stage7_task", lambda _profile: "task"
    )

    class Result:
        selection = {"measurement_request": {"rows": [1, 2, 3, 4]}}
        audit = {"selector": "actual-v3"}

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        dry_run.actual_selector,
        "select_actual_v3_pre_scan_round",
        lambda **kwargs: calls.append(kwargs) or Result(),
    )

    result = dry_run._select_next(
        tmp_path,
        variant="full",
        validated_root={"pre_scan_pool": [{"row_id": "candidate"}]},
        promoted_rows=[{"row_id": "feedback"}],
        output_dir=tmp_path / "audits",
    )

    assert calls[0]["round_index"] == 1
    assert calls[0]["prior_selected_ids"] == [
        f"candidate-{index}" for index in range(4)
    ]
    assert result["selection"]["measurement_request"]["rows"] == [1, 2, 3, 4]
