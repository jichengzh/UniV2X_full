from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest


def _module() -> Any:
    from framework.stage7 import physical_runtime_v2

    return physical_runtime_v2


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


LEASES = (
    "GPU-00000000-0000-0000-0000-000000000000",
    "GPU-11111111-1111-1111-1111-111111111111",
    "GPU-22222222-2222-2222-2222-222222222222",
    "GPU-33333333-3333-3333-3333-333333333333",
)
OWNER = "stage7-controller:pid-1234"


def _inventory(
    *,
    leases: tuple[str, ...] = LEASES,
    indices: tuple[int, ...] = (0, 2, 4, 6),
    model: str = "NVIDIA H800",
) -> list[dict[str, Any]]:
    return [
        {"global_index": index, "uuid": uuid, "name": model}
        for index, uuid in zip(indices, leases)
    ]


def _assignments(
    *,
    inventory: list[dict[str, Any]] | None = None,
    parent_mask: str | None = None,
    related_processes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    module = _module()
    return module.build_uuid_assignments(
        physical_bindings=[
            {
                "candidate_id": "candidate-1",
                "logical_row_index": 1,
                "disposition": "miss",
            },
            {
                "candidate_id": "candidate-3",
                "logical_row_index": 3,
                "disposition": "miss",
            },
        ],
        ordered_lease_uuids=LEASES,
        parent_cuda_visible_devices=parent_mask or ",".join(LEASES),
        inventory=inventory or _inventory(),
        lock_owners={uuid: OWNER for uuid in LEASES},
        expected_lock_owner=OWNER,
        related_processes=related_processes or [],
    )


def _assignment(index: int = 1) -> dict[str, Any]:
    batch = _assignments()
    return copy.deepcopy(batch["assignments"][0 if index == 1 else 1])


def _job() -> dict[str, Any]:
    return {
        "job_id": "pyramid|candidate-1|tvm_auto",
        "manifest_job_id": "candidate-1",
        "candidate_id": "candidate-1",
        "runner_key": "tvm_fp16",
        "runtime_gpu": 2,
        "command": [
            "/python",
            "scripts/stage2_route_b_fp16_auto_runner.py",
            "--gpu",
            "2",
            "--out",
            "/tmp/result.json",
        ],
        "expected_result_json": "/tmp/result.json",
        "max_attempts": 2,
    }


def _pin_probe(expected: dict[str, str]) -> Any:
    return lambda relative: expected[relative]


def _runtime_probes(
    assignment: dict[str, Any],
    *,
    post_inventory: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any, Any]:
    calls = {"inventory": 0}

    def inventory_probe() -> list[dict[str, Any]]:
        calls["inventory"] += 1
        if calls["inventory"] == 1 or post_inventory is None:
            return _inventory()
        return copy.deepcopy(post_inventory)

    def lock_probe(uuid: str) -> str | None:
        return OWNER if uuid in LEASES else None

    def process_probe() -> list[dict[str, Any]]:
        return [
            {
                "pid": 1234,
                "gpu_uuid": assignment["leased_uuid"],
                "related": True,
                "owner": OWNER,
                "controller_owned": True,
                "formal_job_id": assignment["candidate_id"],
            }
        ]

    return inventory_probe, lock_probe, process_probe


def _write_ap_shard(path: Path, *, manifest_job_id: str = "candidate-1") -> None:
    row = {
        "schema_version": "stage5_ap_plan_v2",
        "manifest_job_id": manifest_job_id,
        "model": "pyramid",
        "runner_key": "pyramid_tvm_fp16_bridge",
        "sanity_command": ["/python", "sanity.py"],
        "full_command": ["/python", "full.py"],
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def test_logical_misses_keep_original_lease_slots() -> None:
    batch = _assignments()

    assert [row["logical_row_index"] for row in batch["assignments"]] == [1, 3]
    assert [row["lease_slot"] for row in batch["assignments"]] == [1, 3]
    assert [row["leased_uuid"] for row in batch["assignments"]] == [
        LEASES[1],
        LEASES[3],
    ]
    assert [row["global_index"] for row in batch["assignments"]] == [2, 6]
    assert {row["runtime_visible_ordinal"] for row in batch["assignments"]} == {0}


def test_runtime_primitive_pins_match_current_frozen_source_bytes() -> None:
    module = _module()
    repository = Path(__file__).resolve().parents[2]

    observed = module.verify_runtime_primitives(
        lambda relative: hashlib.sha256(
            (repository / relative).read_bytes()
        ).hexdigest()
    )

    assert observed == module.RUNTIME_PRIMITIVE_SHA256


@pytest.mark.parametrize(
    "change",
    (
        "missing",
        "duplicate",
        "not_h800",
        "mask_order",
        "unleased_process",
        "lock",
        "duplicate_slot",
    ),
)
def test_uuid_assignment_rejects_invalid_inventory_or_parent_state(change: str) -> None:
    inventory = _inventory()
    leases = LEASES
    parent_mask = ",".join(LEASES)
    processes: list[dict[str, Any]] = []
    lock_owners = {uuid: OWNER for uuid in leases}
    bindings = [
        {
            "candidate_id": "candidate-1",
            "logical_row_index": 1,
            "disposition": "miss",
        }
    ]
    if change == "missing":
        inventory = inventory[:-1]
    elif change == "duplicate":
        leases = (LEASES[0], LEASES[1], LEASES[1], LEASES[3])
    elif change == "not_h800":
        inventory[1]["name"] = "NVIDIA A100"
    elif change == "mask_order":
        parent_mask = ",".join(reversed(LEASES))
    elif change == "unleased_process":
        processes = [
            {
                "pid": 999,
                "gpu_uuid": "GPU-UNLEASED",
                "related": True,
            }
        ]
    elif change == "lock":
        lock_owners = {uuid: None for uuid in leases}
    else:
        bindings.append({**bindings[0], "candidate_id": "candidate-other"})

    module = _module()
    with pytest.raises(ValueError):
        module.build_uuid_assignments(
            physical_bindings=bindings,
            ordered_lease_uuids=leases,
            parent_cuda_visible_devices=parent_mask,
            inventory=inventory,
            lock_owners=lock_owners,
            expected_lock_owner=OWNER,
            related_processes=processes,
        )


def test_performance_wrapper_uses_one_uuid_without_mutating_authenticated_job(
    tmp_path: Path,
) -> None:
    module = _module()
    assignment = _assignment()
    authenticated_job = _job()
    original = copy.deepcopy(authenticated_job)
    observed: dict[str, Any] = {}
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)

    def fake_run_single_job(
        *, job: dict[str, Any], state_jsonl: Path, attempt_root: Path
    ) -> dict[str, Any]:
        observed["job"] = copy.deepcopy(job)
        observed["env"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        observed["state_jsonl"] = state_jsonl
        observed["attempt_root"] = attempt_root
        return {
            "job_id": job["job_id"],
            "status": "success",
            "attempt": 1,
            "result_json": str(tmp_path / "result.json"),
            "result_sha256": "a" * 64,
        }

    result = module.execute_bound_performance_row(
        authenticated_job=authenticated_job,
        assignment=assignment,
        state_jsonl=tmp_path / "state.jsonl",
        attempt_root=tmp_path / "attempts",
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
        run_single_job=fake_run_single_job,
    )

    assert authenticated_job == original
    assert "runtime_gpu" not in observed["job"]
    assert observed["job"]["command"] == [
        "/python",
        "scripts/stage2_route_b_fp16_auto_runner.py",
        "--gpu",
        assignment["leased_uuid"],
        "--out",
        "/tmp/result.json",
    ]
    assert {
        key: value
        for key, value in observed["job"].items()
        if key not in {"runtime_gpu", "command"}
    } == {
        key: value
        for key, value in original.items()
        if key not in {"runtime_gpu", "command"}
    }
    assert observed["env"] == assignment["leased_uuid"]
    assert result["assignment"]["global_index"] == 2
    assert result["assignment"]["runtime_visible_ordinal"] == 0
    assert result["authenticated_job_sha256"] == _sha(original)
    assert result["execution_overlay_sha256"] == _sha(observed["job"])
    assert result["stage3_state_row"]["status"] == "success"


@pytest.mark.parametrize("change", ("assignment", "candidate", "command", "gpus"))
def test_performance_rejects_tampered_assignment_or_job_before_primitive(
    change: str, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    job = _job()
    if change == "assignment":
        assignment["global_index"] = 5
    elif change == "candidate":
        job["candidate_id"] = "candidate-other"
    elif change == "command":
        job["command"] = "--gpu 2"
    else:
        job["command"] = ["/python", "runner.py", "--gpus", "0,1", "--gpu", "2"]
    inventory_probe, lock_probe, process_probe = _runtime_probes(_assignment())

    with pytest.raises((TypeError, ValueError)):
        module.execute_bound_performance_row(
            authenticated_job=job,
            assignment=assignment,
            state_jsonl=tmp_path / "state.jsonl",
            attempt_root=tmp_path / "attempts",
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_single_job=lambda **_: pytest.fail("primitive must not run"),
        )


def test_performance_post_mapping_drift_blocks_evidence_even_after_stage3_success(
    tmp_path: Path,
) -> None:
    module = _module()
    assignment = _assignment()
    drifted = _inventory()
    drifted[1]["global_index"] = 7
    inventory_probe, lock_probe, process_probe = _runtime_probes(
        assignment, post_inventory=drifted
    )
    observed = {"calls": 0}

    def fake_run_single_job(**_: Any) -> dict[str, Any]:
        observed["calls"] += 1
        return {"job_id": _job()["job_id"], "status": "success"}

    with pytest.raises(RuntimeError, match="post-runtime"):
        module.execute_bound_performance_row(
            authenticated_job=_job(),
            assignment=assignment,
            state_jsonl=tmp_path / "state.jsonl",
            attempt_root=tmp_path / "attempts",
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_single_job=fake_run_single_job,
        )
    assert observed["calls"] == 1


def test_runtime_rejects_lost_lock_and_unleased_related_process(tmp_path: Path) -> None:
    module = _module()
    assignment = _assignment()

    with pytest.raises(RuntimeError):
        module.execute_bound_performance_row(
            authenticated_job=_job(),
            assignment=assignment,
            state_jsonl=tmp_path / "state.jsonl",
            attempt_root=tmp_path / "attempts",
            inventory_probe=_inventory,
            lock_probe=lambda _: None,
            process_probe=lambda: [],
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_single_job=lambda **_: pytest.fail("primitive must not run"),
        )

    with pytest.raises(RuntimeError):
        module.execute_bound_performance_row(
            authenticated_job=_job(),
            assignment=assignment,
            state_jsonl=tmp_path / "state.jsonl",
            attempt_root=tmp_path / "attempts",
            inventory_probe=_inventory,
            lock_probe=lambda _: OWNER,
            process_probe=lambda: [
                {"pid": 9, "gpu_uuid": "GPU-UNLEASED", "related": True}
            ],
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_single_job=lambda **_: pytest.fail("primitive must not run"),
        )


@pytest.mark.parametrize("process_state", ("foreign", "two_formal_jobs"))
def test_runtime_rejects_foreign_or_duplicate_job_on_the_leased_uuid(
    process_state: str, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    processes = [
        {
            "pid": 777,
            "gpu_uuid": assignment["leased_uuid"],
            "related": True,
            "owner": "foreign-controller",
            "controller_owned": False,
            "formal_job_id": "foreign-job",
        }
    ]
    if process_state == "two_formal_jobs":
        processes = [
            {
                "pid": 777 + index,
                "gpu_uuid": assignment["leased_uuid"],
                "related": True,
                "owner": OWNER,
                "controller_owned": True,
                "formal_job_id": formal_job_id,
            }
            for index, formal_job_id in enumerate(("candidate-1", "candidate-other"))
        ]

    with pytest.raises(RuntimeError, match="process"):
        module.execute_bound_performance_row(
            authenticated_job=_job(),
            assignment=assignment,
            state_jsonl=tmp_path / "state.jsonl",
            attempt_root=tmp_path / "attempts",
            inventory_probe=_inventory,
            lock_probe=lambda _: OWNER,
            process_probe=lambda: processes,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_single_job=lambda **_: pytest.fail("primitive must not run"),
        )


@pytest.mark.parametrize("q_mode,expected_calls", [("int8", 1), ("fp16", 0)])
def test_quant_invokes_only_pinned_int8_primitive(
    q_mode: str, expected_calls: int, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
        calls.append((copy.deepcopy(argv), copy.deepcopy(env)))
        return {"returncode": 0, "output_json": str(tmp_path / "quant.json")}

    result = module.execute_bound_quant_row(
        candidate_id="candidate-1",
        q_mode=q_mode,
        assignment=assignment,
        frozen_repo_root=tmp_path / "frozen",
        python_executable=Path("/python"),
        quant_arguments=[
            "--onnx",
            "/model.onnx",
            "--output-json",
            str(tmp_path / "quant.json"),
        ],
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
        run_argv=fake_run,
    )

    assert len(calls) == expected_calls
    if q_mode == "int8":
        argv, env = calls[0]
        assert argv[:2] == [
            "/python",
            str(tmp_path / "frozen" / "scripts/stage3_tvm_int8_quant_contract_v3.py"),
        ]
        assert env["CUDA_VISIBLE_DEVICES"] == assignment["leased_uuid"]
        assert result["status"] == "success"
    else:
        assert result["status"] == "skipped_fp16"


def test_fp16_quant_skip_still_authenticates_candidate_and_assignment(
    tmp_path: Path,
) -> None:
    module = _module()
    assignment = _assignment()
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)

    with pytest.raises(ValueError, match="candidate"):
        module.execute_bound_quant_row(
            candidate_id="candidate-other",
            q_mode="fp16",
            assignment=assignment,
            frozen_repo_root=tmp_path / "frozen",
            python_executable=Path("/python"),
            quant_arguments=[],
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_argv=lambda *_: pytest.fail("FP16 must not invoke quant"),
        )


@pytest.mark.parametrize("change", ("mode", "candidate", "gpu_flag", "string_argv"))
def test_quant_rejects_invalid_mode_identity_or_gpu_arguments(
    change: str, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    q_mode = "int8"
    candidate_id = "candidate-1"
    arguments: Any = ["--onnx-path", "/model.onnx"]
    if change == "mode":
        q_mode = "int4"
    elif change == "candidate":
        candidate_id = "candidate-other"
    elif change == "gpu_flag":
        arguments = ["--gpu", "2"]
    else:
        arguments = "--onnx-path /model.onnx"
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)

    with pytest.raises((TypeError, ValueError)):
        module.execute_bound_quant_row(
            candidate_id=candidate_id,
            q_mode=q_mode,
            assignment=assignment,
            frozen_repo_root=tmp_path / "frozen",
            python_executable=Path("/python"),
            quant_arguments=arguments,
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_argv=lambda *_: pytest.fail("quant primitive must not run"),
        )


@pytest.mark.parametrize("stage", ("sanity", "full"))
def test_ap_uses_one_fixed_global_gpu_and_never_idle_reselects(
    stage: str, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    ap_plan_jsonl = tmp_path / "ap.jsonl"
    _write_ap_shard(ap_plan_jsonl)
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
        calls.append((copy.deepcopy(argv), copy.deepcopy(env)))
        return {"returncode": 0, "stage": stage}

    result = module.execute_bound_ap_row(
        candidate_id="candidate-1",
        stage=stage,
        assignment=assignment,
        frozen_repo_root=tmp_path / "frozen",
        python_executable=Path("/python"),
        ap_plan_jsonl=ap_plan_jsonl,
        expected_manifest_job_id="candidate-1",
        state_jsonl=tmp_path / "ap-state.jsonl",
        artifact_root=tmp_path / "ap-artifacts",
        inventory_probe=inventory_probe,
        lock_probe=lock_probe,
        process_probe=process_probe,
        primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
        run_argv=fake_run,
    )

    assert len(calls) == 1
    argv, env = calls[0]
    assert argv == [
        "/python",
        str(tmp_path / "frozen" / "scripts/stage3_execute_ap_plan_v3.py"),
        "--ap-plan-jsonl",
        str(ap_plan_jsonl),
        "--stage",
        stage,
        "--state-jsonl",
        str(tmp_path / "ap-state.jsonl"),
        "--gpu",
        "2",
        "--artifact-root",
        str(tmp_path / "ap-artifacts"),
    ]
    assert "--gpus" not in argv
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert result["stage"] == stage


def test_ap_rejects_gpu_flags_and_string_commands(tmp_path: Path) -> None:
    module = _module()
    assignment = _assignment()
    ap_plan_jsonl = tmp_path / "ap.jsonl"
    _write_ap_shard(ap_plan_jsonl)
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)
    common = {
        "candidate_id": "candidate-1",
        "stage": "sanity",
        "assignment": assignment,
        "frozen_repo_root": tmp_path / "frozen",
        "python_executable": Path("/python"),
        "ap_plan_jsonl": ap_plan_jsonl,
        "expected_manifest_job_id": "candidate-1",
        "state_jsonl": tmp_path / "state.jsonl",
        "artifact_root": tmp_path / "artifacts",
        "inventory_probe": inventory_probe,
        "lock_probe": lock_probe,
        "process_probe": process_probe,
        "primitive_sha_probe": _pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
        "run_argv": lambda *_: pytest.fail("AP primitive must not run"),
    }
    with pytest.raises(ValueError):
        module.execute_bound_ap_row(**common, extra_arguments=["--gpus", "0,1"])
    with pytest.raises(TypeError):
        module.execute_bound_ap_row(**common, extra_arguments="--gpu 0")


@pytest.mark.parametrize("change", ("multi_row", "wrong_manifest"))
def test_ap_rejects_non_single_or_wrong_candidate_shard(
    change: str, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    ap_plan_jsonl = tmp_path / "ap.jsonl"
    _write_ap_shard(
        ap_plan_jsonl,
        manifest_job_id=(
            "candidate-other" if change == "wrong_manifest" else "candidate-1"
        ),
    )
    if change == "multi_row":
        with ap_plan_jsonl.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema_version": "stage5_ap_plan_v2",
                        "manifest_job_id": "candidate-other",
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    inventory_probe, lock_probe, process_probe = _runtime_probes(assignment)

    with pytest.raises(ValueError, match="AP plan"):
        module.execute_bound_ap_row(
            candidate_id="candidate-1",
            stage="sanity",
            assignment=assignment,
            frozen_repo_root=tmp_path / "frozen",
            python_executable=Path("/python"),
            ap_plan_jsonl=ap_plan_jsonl,
            expected_manifest_job_id="candidate-1",
            state_jsonl=tmp_path / "state.jsonl",
            artifact_root=tmp_path / "artifacts",
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_argv=lambda *_: pytest.fail("AP primitive must not run"),
        )


@pytest.mark.parametrize("change", ("stage", "candidate", "post_mapping"))
def test_ap_rejects_invalid_binding_or_post_execution_mapping(
    change: str, tmp_path: Path
) -> None:
    module = _module()
    assignment = _assignment()
    ap_plan_jsonl = tmp_path / "ap.jsonl"
    _write_ap_shard(ap_plan_jsonl)
    stage = "sanity"
    candidate_id = "candidate-1"
    post_inventory = None
    if change == "stage":
        stage = "automatic"
    elif change == "candidate":
        candidate_id = "candidate-other"
    else:
        post_inventory = _inventory()
        post_inventory[1]["global_index"] = 7
    inventory_probe, lock_probe, process_probe = _runtime_probes(
        assignment, post_inventory=post_inventory
    )

    with pytest.raises((ValueError, RuntimeError)):
        module.execute_bound_ap_row(
            candidate_id=candidate_id,
            stage=stage,
            assignment=assignment,
            frozen_repo_root=tmp_path / "frozen",
            python_executable=Path("/python"),
            ap_plan_jsonl=ap_plan_jsonl,
            expected_manifest_job_id="candidate-1",
            state_jsonl=tmp_path / "state.jsonl",
            artifact_root=tmp_path / "artifacts",
            inventory_probe=inventory_probe,
            lock_probe=lock_probe,
            process_probe=process_probe,
            primitive_sha_probe=_pin_probe(module.RUNTIME_PRIMITIVE_SHA256),
            run_argv=lambda *_: {"returncode": 0},
        )


def test_primitive_sha_drift_fails_before_any_gpu_or_primitive_call(
    tmp_path: Path,
) -> None:
    module = _module()
    observed = {"inventory": 0, "primitive": 0}

    def inventory_probe() -> list[dict[str, Any]]:
        observed["inventory"] += 1
        return _inventory()

    def run_single_job(**_: Any) -> dict[str, Any]:
        observed["primitive"] += 1
        return {"status": "success"}

    with pytest.raises(RuntimeError, match="primitive SHA drift"):
        module.execute_bound_performance_row(
            authenticated_job=_job(),
            assignment=_assignment(),
            state_jsonl=tmp_path / "state.jsonl",
            attempt_root=tmp_path / "attempts",
            inventory_probe=inventory_probe,
            lock_probe=lambda _: OWNER,
            process_probe=lambda: [],
            primitive_sha_probe=lambda _: "0" * 64,
            run_single_job=run_single_job,
        )

    assert observed == {"inventory": 0, "primitive": 0}


def test_zero_miss_runtime_does_not_probe_or_launch() -> None:
    module = _module()

    def explode(*_: Any, **__: Any) -> Any:
        raise AssertionError("zero miss must not observe GPU or a primitive")

    result = module.execute_runtime_rows(
        runtime_rows=[],
        inventory_probe=explode,
        primitive_runner=explode,
        forbidden_components={
            name: explode
            for name in (
                "source_materializer",
                "selector",
                "advance",
                "online_finalizer",
                "promoter",
                "legacy_parent_mask_executor",
            )
        },
    )

    assert result == {
        "schema_version": "stage7_empty_physical_runtime_v2",
        "physical_row_count": 0,
        "gpu_subprocess_count": 0,
        "rows": [],
    }


def test_runtime_dry_plan_never_calls_forbidden_search_or_feedback_components() -> None:
    module = _module()

    def explode(*_: Any, **__: Any) -> Any:
        raise AssertionError("forbidden component was called")

    result = module.execute_runtime_rows(
        runtime_rows=[{"candidate_id": "candidate-1", "action": "planned"}],
        inventory_probe=lambda: _inventory(),
        primitive_runner=lambda row: {**row, "status": "dry_planned"},
        forbidden_components={
            name: explode
            for name in (
                "source_materializer",
                "selector",
                "advance",
                "online_finalizer",
                "promoter",
                "legacy_parent_mask_executor",
            )
        },
    )

    assert result["physical_row_count"] == 1
    assert result["rows"][0]["status"] == "dry_planned"
