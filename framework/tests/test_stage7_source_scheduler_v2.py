from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from framework.tests import test_stage7_execute_actual_v3_misses_v2 as miss_fixture


def _module():
    return importlib.import_module("scripts.stage7_source_scheduler_v2")


class FakeLock:
    def __init__(self, name: str, released: list[str]) -> None:
        self.name = name
        self.released = released

    def release(self) -> None:
        self.released.append(self.name)


def test_source_round_state_starts_from_frozen_plan_and_never_measurement(
    tmp_path: Path,
) -> None:
    module = _module()
    (tmp_path / "logical_request.json").write_text("{}")
    (tmp_path / "selection_binding.json").write_text("{}")
    (tmp_path / "source_resolution_plan.json").write_text("{}")

    state = module.detect_round_state(tmp_path)

    assert state == "SOURCE_PLAN_FROZEN"
    assert module.measurement_ready(tmp_path) is False


def test_source_state_progression_requires_signed_lease_before_materializing(
    tmp_path: Path,
) -> None:
    module = _module()
    for name in (
        "logical_request.json",
        "selection_binding.json",
        "source_resolution_plan.json",
        "source_execution_plan.json",
    ):
        (tmp_path / name).write_text("{}")
    assert module.detect_round_state(tmp_path) == "SOURCE_LEASE_PENDING"
    (tmp_path / "source_gpu_lease.json").write_text("{}")
    assert module.detect_round_state(tmp_path) == "SOURCE_MATERIALIZING"


def test_measurement_is_rejected_before_formal_source_exact_and_cache(
    tmp_path: Path,
) -> None:
    module = _module()
    for name in (
        "logical_request.json",
        "selection_binding.json",
        "source_resolution_plan.json",
        "source_execution_plan.json",
        "source_gpu_lease.json",
    ):
        (tmp_path / name).write_text("{}")

    with pytest.raises(ValueError, match="SOURCE_READY"):
        module.require_measurement_ready(tmp_path)


def test_synthetic_source_result_never_admits_measurement(tmp_path: Path) -> None:
    module = _module()
    for name in (
        "logical_request.json",
        "source_resolution_plan.json",
        "synthetic_source_resolution_result.json",
        "synthetic_exact_selection_binding.json",
        "synthetic_physical_protocol.json",
    ):
        (tmp_path / name).write_text("{}")

    with pytest.raises(ValueError, match="synthetic"):
        module.require_measurement_ready(tmp_path)


def test_source_retry_preserves_request_plan_and_zero_budget() -> None:
    module = _module()
    retry = {
        "logical_request_sha256": "a" * 64,
        "source_resolution_plan_sha256": "b" * 64,
        "selected_event_budget_delta": 0,
        "partial_reveal_allowed": False,
    }
    validated = module.validate_zero_budget_retry(
        retry,
        logical_request_sha256="a" * 64,
        source_resolution_plan_sha256="b" * 64,
    )
    assert validated == retry


def test_source_then_uuid_locks_are_sorted_and_partial_failure_releases_all() -> None:
    module = _module()
    order: list[str] = []
    released: list[str] = []

    def source_factory(key: str):
        order.append(f"source:{key}")
        return FakeLock(f"source:{key}", released)

    def uuid_factory(uuid: str):
        order.append(f"uuid:{uuid}")
        return None if uuid == "GPU-b" else FakeLock(f"uuid:{uuid}", released)

    result = module.acquire_source_then_uuid_locks(
        ("z", "a"), ("GPU-a", "GPU-b"), source_factory, uuid_factory
    )
    assert result is None
    assert order == ["source:a", "source:z", "uuid:GPU-a", "uuid:GPU-b"]
    assert released == ["uuid:GPU-a", "source:a", "source:z"]


def test_source_batch_capacity_counts_source_and_measurement_together() -> None:
    module = _module()
    controllers = {
        "source": {"status": "SOURCE_MATERIALIZING"},
        "measurement": {"status": "MEASURING"},
        "done": {"status": "FEEDBACK_COMPLETE"},
    }
    assert module.available_batch_slots(controllers) == 0


def test_uuid_lease_recheck_rejects_changed_or_duplicate_physical_index() -> None:
    module = _module()
    lease = {
        "group_bindings": [
            {"gpu_uuid": "GPU-a", "physical_index": 0, "gpu_model": "NVIDIA H800"},
            {"gpu_uuid": "GPU-b", "physical_index": 1, "gpu_model": "NVIDIA H800"},
        ]
    }
    current = {
        "GPU-a": {"physical_index": 0, "model": "NVIDIA H800"},
        "GPU-b": {"physical_index": 0, "model": "NVIDIA H800"},
    }
    with pytest.raises(ValueError, match="index"):
        module.recheck_source_lease_inventory(lease, current)


def test_gpu7_and_foreign_process_reservations_are_never_source_ready() -> None:
    module = _module()
    inventory = {
        f"GPU-{index}": SimpleNamespace(
            uuid=f"GPU-{index}",
            index=index,
            memory_used_mib=0,
            utilization_percent=0,
            compute_pids=(),
        )
        for index in range(8)
    }
    ready = module.ready_source_uuids(
        inventory,
        models={uuid: "NVIDIA H800" for uuid in inventory},
        reservations={"GPU-2": ("foreign_process",)},
        fcooper_gpu7_active=True,
    )
    assert "GPU-2" not in ready
    assert "GPU-7" not in ready
    ready_without_fcooper_process = module.ready_source_uuids(
        inventory,
        models={uuid: "NVIDIA H800" for uuid in inventory},
        reservations={},
        fcooper_gpu7_active=False,
    )
    assert "GPU-7" not in ready_without_fcooper_process


def test_mixed_synthetic_and_formal_round_is_explicitly_invalid(
    tmp_path: Path,
) -> None:
    module = _module()
    round_dir = tmp_path / "round"
    round_dir.mkdir()
    for name in (*module.FORMAL_MEASUREMENT_ARTIFACTS, "synthetic_protocol_receipt.json"):
        (round_dir / name).write_text("{}\n")

    assert module.detect_round_state(round_dir) == "INVALID_MIXED_SOURCE_STATE"


def test_infrastructure_failure_is_retry_not_candidate_capability_failure() -> None:
    module = _module()
    assert module.classify_source_failure("uuid_index_drift") == (
        "infrastructure_unavailable"
    )
    assert module.classify_source_failure("child_nonzero") == "evidence_unavailable"


def test_complete_validated_source_evidence_skips_lease_gpu_and_child() -> None:
    module = _module()
    calls: list[str] = []

    result = module.advance_source_prelease(
        validate_ready_evidence=lambda: {
            "schema_version": "stage7_source_resolution_result_v2",
            "row_count": 4,
            "eligible_for_exact_cache_reveal": True,
            "source_resolution_result_sha256": "a" * 64,
        },
        lease_factory=lambda: calls.append("lease"),
        launch_wrapper=lambda _lease: calls.append("child"),
    )

    assert result["state"] == "SOURCE_READY"
    assert result["fast_path"] is True
    assert calls == []


def test_incomplete_source_evidence_enters_lease_then_materialization() -> None:
    module = _module()
    calls: list[str] = []

    def unavailable():
        raise FileNotFoundError("evidence absent")

    result = module.advance_source_prelease(
        validate_ready_evidence=unavailable,
        lease_factory=lambda: calls.append("lease") or {"lease": "signed"},
        launch_wrapper=lambda lease: calls.append(f"child:{lease['lease']}"),
    )

    assert result == {
        "state": "SOURCE_MATERIALIZING",
        "fast_path": False,
        "lease_acquired": True,
    }
    assert calls == ["lease", "child:signed"]


def test_phase3a_replacement_dependencies_are_sha_pinned_before_source_work(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source.py"
    resolver = tmp_path / "resolver.py"
    source.write_text("source")
    resolver.write_text("resolver")
    pins = {
        "source.py": module.sha256_file(source),
        "resolver.py": module.sha256_file(resolver),
    }

    assert module.validate_dependency_pins(tmp_path, pins) == pins
    resolver.write_text("drift")
    with pytest.raises(ValueError, match="dependency SHA drift"):
        module.validate_dependency_pins(tmp_path, pins)


def test_remote_deploy_manifest_must_include_phase3a_replacement_pins() -> None:
    module = _module()
    pins = {"framework/source.py": "a" * 64, "scripts/resolver.py": "b" * 64}
    complete = {
        "files": [
            {
                "destination": "/remote/framework/source.py",
                "sha256": "a" * 64,
            },
            {
                "destination": "/remote/scripts/resolver.py",
                "sha256": "b" * 64,
            },
        ]
    }
    assert module.validate_deploy_manifest_pins(complete, pins) == pins
    with pytest.raises(ValueError, match="deployment pin missing"):
        module.validate_deploy_manifest_pins({"files": complete["files"][:1]}, pins)


def test_measurement_artifacts_are_formally_revalidated_before_lease(
    tmp_path: Path,
) -> None:
    module = _module()
    artifacts = miss_fixture._real_task4_artifacts(tmp_path / "formal-v2")

    validated = module.validate_measurement_artifacts(Path(artifacts["round_dir"]))

    assert (
        validated["source_result"]["source_resolution_result_sha256"]
        == artifacts["source_result_sha"]
    )
    assert validated["executor_admission"]["admission_passed"] is True


def test_tampered_cache_or_executor_admission_fails_before_measurement_lease(
    tmp_path: Path,
) -> None:
    module = _module()
    artifacts = miss_fixture._real_task4_artifacts(tmp_path / "formal-v2")
    cache_reveal = Path(artifacts["reveal"])
    payload = __import__("json").loads(cache_reveal.read_text())
    payload["cache_hits"] = []
    cache_reveal.write_text(__import__("json").dumps(payload))

    with pytest.raises(ValueError, match="measurement artifact authentication"):
        module.validate_measurement_artifacts(Path(artifacts["round_dir"]))


def test_required_deployment_pins_cover_all_phase3b_owned_production_files() -> None:
    module = _module()
    repo = Path(__file__).resolve().parents[2]

    pins = module.required_deployment_pins(repo)

    assert set(module.PHASE3B_DEPLOY_FILES).issubset(pins)
    assert set(module.PHASE3A_REPLACEMENT_PINS).issubset(pins)
    phase3a_only = {
        "files": [
            {
                "destination": "/remote/" + relative,
                "sha256": digest,
            }
            for relative, digest in module.PHASE3A_REPLACEMENT_PINS.items()
        ]
    }
    with pytest.raises(ValueError, match="deployment pin missing"):
        module.validate_deploy_manifest_pins(phase3a_only, pins)
