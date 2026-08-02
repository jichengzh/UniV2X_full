from __future__ import annotations

import hashlib
import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from framework.stage5.measurement_plan_v1 import _source_plan_sha
from framework.stage7.source_resolution_v2 import build_source_resolution_plan
from framework.stage7.source_round_orchestration_v2 import freeze_selection_identity
from framework.tests.test_stage7_source_execution_v2 import _inputs, _lease_args


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _artifacts_for_group(
    request: dict[str, Any],
    group_id: str,
    *,
    valid: bool = True,
) -> Path:
    row = next(row for row in request["rows"] if row["group_id"] == group_id)
    source = row["source_contract"]
    artifacts = {
        "checkpoint_path": b"checkpoint",
        "onnx_path": b"onnx",
        "calibration_npz": b"calibration",
        "calibration_summary": b'{"status":"success"}',
    }
    for field, content in artifacts.items():
        path = Path(source[field])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content + group_id.encode())
    evidence_path = Path(
        str(source["source_done_marker"]).removesuffix(".done") + "_evidence.json"
    )

    def file_sha(field: str) -> str:
        return hashlib.sha256(Path(source[field]).read_bytes()).hexdigest()

    evidence = {
        "schema_version": "stage5_source_materialization_evidence_v1",
        "group_id": group_id,
        "model": row["model"],
        "width": "x".join(map(str, row["width"])),
        "source_plan_sha256": row["source_evidence_sha256"],
        "checkpoint_path": source["checkpoint_path"],
        "checkpoint_sha256": file_sha("checkpoint_path"),
        "onnx_path": source["onnx_path"],
        "onnx_sha256": file_sha("onnx_path"),
        "calibration_path": source["calibration_npz"],
        "calibration_sha256": file_sha("calibration_npz"),
        "calibration_summary_path": source["calibration_summary"],
        "calibration_summary_sha256": file_sha("calibration_summary"),
        "status": "ready",
    }
    if not valid:
        evidence["onnx_sha256"] = "12" * 32
    _write_json(evidence_path, evidence)
    return evidence_path


def _four_group_inputs(root: Path) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    request_bytes, _binding, _source_plan = _inputs(root)
    request = json.loads(request_bytes)
    widths = (
        [16, 32, 64],
        [17, 33, 65],
        [24, 48, 96],
        [25, 49, 97],
    )
    for index, (row, width) in enumerate(zip(request["rows"], widths)):
        tag = "x".join(map(str, width))
        candidate_id = (
            f"pyramid|{tag}|q={row['q_mode']}|profile=h800-v3"
        )
        source_root = root / tag
        row["row_id"] = candidate_id
        row["manifest_job_id"] = candidate_id
        row["group_id"] = f"pyramid|{tag}"
        row["width"] = width
        row["genome"] = [*width, row["q_mode"]]
        row["source_contract"] = {
            "checkpoint_path": str(source_root / "checkpoint.pth"),
            "checkpoint_sha256": None,
            "onnx_path": str(source_root / "model.onnx"),
            "onnx_sha256": None,
            "calibration_npz": str(source_root / "calibration.npz"),
            "calibration_summary": str(source_root / "summary.json"),
            "source_done_marker": str(source_root / "source.done"),
            "trt_calibration_dir": str(source_root / "trt"),
        }
        row["source_evidence_sha256"] = _source_plan_sha(row)
    request["row_sha256"] = {
        row["row_id"]: _sha(row) for row in request["rows"]
    }
    request_without_sha = {
        key: value
        for key, value in request.items()
        if key != "measurement_request_sha256"
    }
    request = {
        **request_without_sha,
        "measurement_request_sha256": _sha(request_without_sha),
    }
    selection = {
        "measurement_request": request,
        "acquisition": {
            "selected_row_ids": [row["row_id"] for row in request["rows"]]
        },
    }
    binding = freeze_selection_identity(selection)
    source_plan = build_source_resolution_plan(request)
    encoded = (
        json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return encoded, binding, source_plan


def _case(
    tmp_path: Path,
    *,
    mode: str = "formal",
    four_unique_groups: bool = False,
) -> dict[str, Any]:
    from framework.stage7 import source_execution_v2

    inputs = _four_group_inputs if four_unique_groups else _inputs
    request_bytes, binding, source_plan = inputs(tmp_path / "sources")
    execution = source_execution_v2.build_source_execution_plan(
        request_bytes, binding, source_plan
    )
    lease = source_execution_v2.build_source_gpu_lease(
        execution,
        **_lease_args(execution),
        execution_mode=mode,
        source_locks_held=mode == "formal",
    )
    paths = {
        "logical_request_path": tmp_path / "round" / "logical_request.json",
        "selection_binding_path": tmp_path / "round" / "selection_binding.json",
        "source_plan_path": tmp_path / "round" / "source_resolution_plan.json",
        "source_lease_path": tmp_path / "round" / "source_gpu_lease.json",
        "output_dir": (
            tmp_path
            / (
                "audits/no_gpu_dry_run/source_materialization/source_attempt_00"
                if mode == "no_gpu_dryrun"
                else "round/source_attempt_00"
            )
        ),
    }
    paths["logical_request_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["logical_request_path"].write_bytes(request_bytes)
    _write_json(paths["selection_binding_path"], binding)
    _write_json(paths["source_plan_path"], source_plan)
    _write_json(paths["source_lease_path"], lease)
    return {
        **paths,
        "request": json.loads(request_bytes),
        "execution": execution,
        "lease": lease,
    }


def _probes(case: dict[str, Any]) -> dict[str, Any]:
    lease = case["lease"]
    inventory = {
        row["gpu_uuid"]: {
            "physical_index": row["physical_index"],
            "model": row["gpu_model"],
        }
        for row in lease["group_bindings"]
    }
    owner = lease["lock_owner"]
    return {
        "hostname_probe": lambda: "zs-nj-tap-gpu18",
        "inventory_probe": lambda: inventory,
        "process_identity_probe": lambda pid: {
            "pid": pid,
            "owner": owner["owner"],
            "start_time": owner["start_time"],
            "alive": True,
        },
    }


def _run(
    case: dict[str, Any],
    *,
    no_gpu_dryrun: bool,
    run_command: Any,
    environment: dict[str, str],
    **overrides: Any,
) -> dict[str, Any]:
    module = importlib.import_module("scripts.stage7_resolve_round_sources_v2")
    arguments = {
        key: case[key]
        for key in (
            "logical_request_path",
            "selection_binding_path",
            "source_plan_path",
            "source_lease_path",
            "output_dir",
        )
    }
    probes = {**_probes(case), **overrides}
    return module.run_source_resolution(
        **arguments,
        no_gpu_dryrun=no_gpu_dryrun,
        advertised_gpu_uuids=(
            tuple(case["lease"]["gpu_uuids"]) if not no_gpu_dryrun else ()
        ),
        run_command=run_command,
        environment=environment,
        **probes,
    )


def test_actual_fast_path_validates_existing_evidence_without_process(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    for group in case["execution"]["group_jobs"]:
        _artifacts_for_group(case["request"], group["group_id"])

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("materializer must not run for ready evidence")

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=forbidden,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
    )

    assert result["status"] == "source_ready"
    assert result["materializer_invocation_count"] == 0
    assert result["formal_result_written"] is True
    formal_path = Path(result["source_result_path"])
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    assert formal["schema_version"] == "stage7_source_resolution_result_v2"
    assert len(formal["rows"]) == 4


def test_actual_invokes_frozen_materializer_once_per_unique_group_with_argv(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        calls.append((command, kwargs))
        group_id = command[command.index("--group-id") + 1]
        _artifacts_for_group(case["request"], group_id)
        return SimpleNamespace(returncode=0, stdout="ready\n", stderr="")

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=fake_run,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
    )

    assert result["status"] == "source_ready"
    assert result["materializer_invocation_count"] == 2
    assert len(calls) == 2
    for command, kwargs in calls:
        assert command[0].endswith("stage5_materialize_round_sources_v1.sh")
        assert command[command.index("--request") + 1] == str(
            case["logical_request_path"].resolve()
        )
        assert command[command.index("--model") + 1] == "pyramid"
        assert command[command.index("--gpu") + 1].isdigit()
        assert kwargs["shell"] is False
    for group in case["execution"]["group_jobs"]:
        receipt_path = (
            case["output_dir"]
            / f"group_{group['source_group_execution_key'][:16]}_process.json"
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert "stdout" not in receipt
        assert "stderr" not in receipt
        assert Path(receipt["stdout_path"]).read_text(encoding="utf-8") == "ready\n"
        assert receipt["stdout_sha256"] == hashlib.sha256(b"ready\n").hexdigest()
        assert Path(receipt["stderr_path"]).read_bytes() == b""
        assert receipt["stderr_sha256"] == hashlib.sha256(b"").hexdigest()


def test_actual_launches_all_four_frozen_groups_in_one_parallel_batch(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, four_unique_groups=True)
    group_count = len(case["execution"]["group_jobs"])
    assert group_count == 4
    barrier = threading.Barrier(group_count)
    state_lock = threading.Lock()
    active = 0
    max_active = 0
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(command)
        barrier.wait(timeout=2)
        group_id = command[command.index("--group-id") + 1]
        _artifacts_for_group(case["request"], group_id)
        with state_lock:
            active -= 1
        return SimpleNamespace(
            returncode=0,
            stdout=f"{group_id}\n",
            stderr="",
        )

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=fake_run,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
    )

    assert result["status"] == "source_ready"
    assert result["materializer_invocation_count"] == 4
    assert max_active == 4
    assert len(calls) == 4
    assert {
        command[command.index("--gpu") + 1] for command in calls
    } == {"0", "1", "2", "3"}
    formal = json.loads(Path(result["source_result_path"]).read_text())
    assert [row["candidate_id"] for row in formal["rows"]] == [
        row["row_id"] for row in case["request"]["rows"]
    ]


def test_actual_parallel_batch_aggregates_failed_groups_into_one_zero_budget_retry(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, four_unique_groups=True)
    groups = case["execution"]["group_jobs"]
    barrier = threading.Barrier(len(groups))
    calls: list[str] = []
    failed_keys = {
        groups[1]["source_group_execution_key"],
        groups[3]["source_group_execution_key"],
    }

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        group_id = command[command.index("--group-id") + 1]
        group = next(row for row in groups if row["group_id"] == group_id)
        calls.append(group["source_group_execution_key"])
        barrier.wait(timeout=2)
        if group["source_group_execution_key"] in failed_keys:
            return SimpleNamespace(returncode=7, stdout="", stderr="failed")
        _artifacts_for_group(case["request"], group_id)
        return SimpleNamespace(returncode=0, stdout="ready", stderr="")

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=fake_run,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
    )

    assert len(calls) == 4
    assert result["status"] == "retry_required"
    assert result["retry_reason_code"] == "evidence_unavailable"
    assert result["materializer_invocation_count"] == 4
    assert result["selected_event_budget_delta"] == 0
    assert result["formal_result_written"] is False
    retry = json.loads(Path(result["retry_path"]).read_text())
    failed_candidate_ids = {
        candidate_id
        for group in groups
        if group["source_group_execution_key"] in failed_keys
        for candidate_id in group["candidate_ids"]
    }
    assert retry["retry_candidate_ids"] == [
        candidate_id
        for candidate_id in case["execution"]["ordered_candidate_ids"]
        if candidate_id in failed_candidate_ids
    ]
    assert not (case["output_dir"] / "source_resolution_result.staged.json").exists()


def test_pre_gate_source_only_persists_hashes_but_not_raw_process_logs(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    forbidden = "latency_ms=1.0 AP=0.5 cache_hit=true feedback=online"

    def fake_run(command: list[str], **_kwargs: Any) -> Any:
        group_id = command[command.index("--group-id") + 1]
        _artifacts_for_group(case["request"], group_id)
        return SimpleNamespace(returncode=0, stdout=forbidden, stderr=forbidden)

    result = _run(
        case,
        no_gpu_dryrun=False,
        pre_gate_source_only=True,
        run_command=fake_run,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
    )

    assert result["status"] == "source_ready"
    for group in case["execution"]["group_jobs"]:
        prefix = f"group_{group['source_group_execution_key'][:16]}"
        receipt = json.loads(
            (case["output_dir"] / f"{prefix}_process.json").read_text()
        )
        assert receipt["stdout_path"] is None
        assert receipt["stderr_path"] is None
        assert receipt["raw_process_output_persisted"] is False
        assert receipt["stdout_sha256"] == hashlib.sha256(forbidden.encode()).hexdigest()
        assert receipt["stderr_sha256"] == hashlib.sha256(forbidden.encode()).hexdigest()
        assert not (case["output_dir"] / f"{prefix}.stdout.log").exists()
        assert not (case["output_dir"] / f"{prefix}.stderr.log").exists()


@pytest.mark.parametrize("broken", (False, True))
def test_formal_output_parent_symlink_cannot_escape_round(
    broken: bool, tmp_path: Path
) -> None:
    case = _case(tmp_path)
    outside = tmp_path / "outside"
    if not broken:
        outside.mkdir()
    escape = case["logical_request_path"].parent / "escape"
    escape.symlink_to(outside, target_is_directory=True)
    case["output_dir"] = escape / "attempt_000"

    with pytest.raises(ValueError, match="output directory"):
        _run(
            case,
            no_gpu_dryrun=False,
            run_command=lambda *_args, **_kwargs: pytest.fail(
                "unsafe output must fail before materializer launch"
            ),
            environment={
                "CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])
            },
        )

    if outside.exists():
        assert list(outside.iterdir()) == []


@pytest.mark.parametrize(
    ("failure", "reason"),
    (
        ("returncode", "evidence_unavailable"),
        ("missing_evidence", "evidence_invalid"),
        ("exception", "infrastructure_unavailable"),
    ),
)
def test_actual_partial_or_infrastructure_failure_emits_zero_budget_retry_only(
    failure: str,
    reason: str,
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        if failure == "exception":
            raise OSError("launcher unavailable")
        if failure == "returncode":
            return SimpleNamespace(returncode=7, stdout="", stderr="failed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=fake_run,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
    )

    assert result["status"] == "retry_required"
    assert result["retry_reason_code"] == reason
    assert result["selected_event_budget_delta"] == 0
    assert result["formal_result_written"] is False
    assert result["eligible_for_exact_cache_reveal"] is False
    assert result["eligible_for_cache_append"] is False
    assert result["eligible_for_finalization"] is False
    assert len(result["source_resolution_retry_sha256"]) == 64
    assert not (case["output_dir"] / "source_resolution_result.staged.json").exists()
    retry = json.loads(
        (case["output_dir"] / "source_resolution_retry.json").read_text()
    )
    assert retry["partial_reveal_allowed"] is False


def test_no_gpu_dryrun_generates_synthetic_protocol_without_spawning(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, mode="no_gpu_dryrun")

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dry-run must not start a process")

    result = _run(
        case,
        no_gpu_dryrun=True,
        run_command=forbidden,
        environment={"CUDA_VISIBLE_DEVICES": ""},
    )

    assert result["status"] == "synthetic_protocol_ready"
    assert result["gpu_jobs_launched"] == 0
    assert result["materializer_invocation_count"] == 0
    assert len(result["would_run_commands"]) == 2
    assert all(command[-1] == "--dry-run" for command in result["would_run_commands"])
    assert Path(result["synthetic_result_path"]).is_file()
    assert not (case["output_dir"] / "source_resolution_result.staged.json").exists()


@pytest.mark.parametrize(
    ("mode", "cuda"),
    (
        ("formal", ""),
        ("formal", "GPU-wrong"),
        ("no_gpu_dryrun", "GPU-0"),
    ),
)
def test_cuda_environment_must_match_execution_mode_and_signed_uuid_order(
    mode: str,
    cuda: str,
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, mode=mode)

    if mode == "no_gpu_dryrun":
        with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
            _run(
                case,
                no_gpu_dryrun=True,
                run_command=lambda *args, **kwargs: None,
                environment={"CUDA_VISIBLE_DEVICES": cuda},
            )
        return

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=lambda *args, **kwargs: None,
        environment={"CUDA_VISIBLE_DEVICES": cuda},
    )
    assert result["status"] == "retry_required"
    assert result["retry_reason_code"] == "infrastructure_unavailable"
    assert result["materializer_invocation_count"] == 0


def test_actual_rejects_lock_owner_or_inventory_drift_before_process(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=lambda *args, **kwargs: None,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
        process_identity_probe=lambda pid: None,
    )
    assert result["status"] == "retry_required"
    assert result["retry_reason_code"] == "infrastructure_unavailable"
    assert result["materializer_invocation_count"] == 0


def test_actual_rechecks_inventory_once_before_parallel_batch_and_launches_none_on_drift(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    inventory_calls = 0
    process_calls = 0
    valid_inventory = case["lease"]["inventory"]

    def changing_inventory() -> dict[str, Any]:
        nonlocal inventory_calls
        inventory_calls += 1
        if inventory_calls < 2:
            return valid_inventory
        drifted = json.loads(json.dumps(valid_inventory))
        second_uuid = case["lease"]["gpu_uuids"][1]
        drifted[second_uuid]["physical_index"] = 7
        return drifted

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        nonlocal process_calls
        process_calls += 1
        group_id = command[command.index("--group-id") + 1]
        _artifacts_for_group(case["request"], group_id)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = _run(
        case,
        no_gpu_dryrun=False,
        run_command=fake_run,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
        inventory_probe=changing_inventory,
    )

    assert process_calls == 0
    assert result["status"] == "retry_required"
    assert result["retry_reason_code"] == "infrastructure_unavailable"
    assert result["materializer_invocation_count"] == 0
    assert result["formal_result_written"] is False


def test_advertised_gpu_uuids_must_match_lease_and_cuda(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)

    module = importlib.import_module("scripts.stage7_resolve_round_sources_v2")
    result = module.run_source_resolution(
        **{
            key: case[key]
            for key in (
                "logical_request_path",
                "selection_binding_path",
                "source_plan_path",
                "source_lease_path",
                "output_dir",
            )
        },
        no_gpu_dryrun=False,
        advertised_gpu_uuids=("GPU-wrong",),
        run_command=lambda *args, **kwargs: None,
        environment={"CUDA_VISIBLE_DEVICES": ",".join(case["lease"]["gpu_uuids"])},
        **_probes(case),
    )
    assert result["status"] == "retry_required"
    assert result["retry_reason_code"] == "infrastructure_unavailable"
    assert result["materializer_invocation_count"] == 0


def test_no_gpu_dryrun_rejects_non_audit_output_directory(tmp_path: Path) -> None:
    case = _case(tmp_path, mode="no_gpu_dryrun")
    case["output_dir"] = tmp_path / "round" / "source_attempt_00"

    with pytest.raises(ValueError, match="audits/no_gpu_dry_run"):
        _run(
            case,
            no_gpu_dryrun=True,
            run_command=lambda *args, **kwargs: None,
            environment={"CUDA_VISIBLE_DEVICES": ""},
        )


def test_no_gpu_dryrun_rejects_lookalike_audit_component(tmp_path: Path) -> None:
    case = _case(tmp_path, mode="no_gpu_dryrun")
    case["output_dir"] = (
        tmp_path / "audits" / "no_gpu_dry_run_evil" / "source_attempt_00"
    )

    with pytest.raises(ValueError, match="audits/no_gpu_dry_run"):
        _run(
            case,
            no_gpu_dryrun=True,
            run_command=lambda *args, **kwargs: None,
            environment={"CUDA_VISIBLE_DEVICES": ""},
        )


def test_cli_requires_explicit_gpu_uuid_visibility() -> None:
    module = importlib.import_module("scripts.stage7_resolve_round_sources_v2")

    with pytest.raises(SystemExit):
        module.build_parser().parse_args(
            [
                "--logical-request-json",
                "/tmp/request.json",
                "--selection-binding-json",
                "/tmp/binding.json",
                "--source-plan-json",
                "/tmp/plan.json",
                "--source-lease-json",
                "/tmp/lease.json",
                "--output-dir",
                "/tmp/output",
            ]
        )


def test_existing_conflicting_attempt_artifact_is_not_overwritten(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, mode="no_gpu_dryrun")
    case["output_dir"].mkdir(parents=True)
    conflict = case["output_dir"] / "source_attempt.json"
    conflict.write_text('{"conflict":true}\n', encoding="utf-8")
    before = conflict.read_bytes()

    with pytest.raises(ValueError, match="conflict"):
        _run(
            case,
            no_gpu_dryrun=True,
            run_command=lambda *args, **kwargs: None,
            environment={"CUDA_VISIBLE_DEVICES": ""},
        )

    assert conflict.read_bytes() == before


def test_wrapper_public_api_is_explicit() -> None:
    module = importlib.import_module("scripts.stage7_resolve_round_sources_v2")
    assert set(module.__all__) == {"run_source_resolution", "main"}
