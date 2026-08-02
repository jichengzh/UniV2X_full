from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from framework.stage7 import core_ablation_v2 as core
from framework.stage7 import core_cache_v2 as cache_v2
from framework.stage7 import deployment_bundle_v2
from framework.stage7 import source_resolution_v2
from framework.tests import test_stage7_actual_v3_adapter_v2 as adapter_fixture
from framework.tests import test_stage7_core_online_ablation_v2 as phase2_fixture
from scripts import stage7_core_online_ablation_v2 as online
from scripts import stage7_scheduler_requests_v2 as scheduler_requests


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage7_execute_actual_v3_misses_v2.sh"
PYTHON = Path("/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
INVALID_EXTERNAL_PIN = "9" * 64


def _sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _physical_plan(row_count: int = 4) -> dict[str, object]:
    rows = [
        {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S7-PYR-TVM",
            "task_sha256": "a" * 64,
            "row_id": f"candidate-{index}",
            "manifest_job_id": f"candidate-{index}",
            "model": "pyramid",
            "hardware_id": "h800",
            "dispatch_key": "tvm_auto",
            "capability_profile_id": "h800-tvm-auto-v3",
            "width": [16 + index, 32, 64],
            "q_mode": "int8" if index % 2 else "fp16",
            "group_id": f"pyramid|{16 + index}x32x64",
            "source_contract": {
                "source_done_marker": f"/tmp/source-{index}.done",
                "onnx_path": f"/tmp/source-{index}.onnx",
                "calibration_npz": f"/tmp/calibration-{index}.npz",
                "calibration_summary": f"/tmp/calibration-{index}.json",
            },
        }
        for index in range(row_count)
    ]
    payload = {
        "schema_version": "stage7_actual_v3_miss_only_physical_request_v2",
        "logical_request_sha256": "1" * 64,
        "selection_binding_sha256": "2" * 64,
        "cache_snapshot_sha256": "3" * 64,
        "lineage_head_sha256": "4" * 64,
        "cache_reveal_sha256": "5" * 64,
        "logical_row_count": 4,
        "physical_row_count": row_count,
        "row_sha256": {str(row["row_id"]): _sha(row) for row in rows},
        "rows": rows,
    }
    bindings = [
        {
            "logical_row_index": index,
            "candidate_id": f"candidate-{index}",
            "logical_row_sha256": str(index + 6) * 64,
            "logical_exact_binding_sha256": str(index + 1) * 64,
            "exact_cache_key_sha256": str(index + 2) * 64,
            "disposition": "miss" if index < row_count else "hit",
            "physical_row_sha256": (
                payload["row_sha256"].get(f"candidate-{index}")
                if index < row_count
                else None
            ),
        }
        for index in range(4)
    ]
    payload["logical_row_bindings"] = bindings
    return {**payload, "physical_request_sha256": _sha(payload)}


def _formal_contract(root: Path) -> dict[str, object]:
    unsigned = {
        "schema_version": core.SCHEMA_VERSION,
        "v1_root": str(core.V1_ROOT),
        "v2_root": str(root.resolve()),
        "source_registry_sha256": core.EXPECTED_SOURCE_REGISTRY_SHA256,
        "pre_scan_registry_sha256": core.EXPECTED_PRE_SCAN_REGISTRY_SHA256,
        "ordered_pre_scan_sha256": core.EXPECTED_ORDERED_PRE_SCAN_SHA256,
        "candidate_pool_count": 686,
        "variants": list(core.CORE_VARIANTS),
        "scanner_deferred": True,
        "scanner_claim_allowed": False,
        "variant_contracts": core._variant_contracts(
            core.EXPECTED_ORDERED_PRE_SCAN_SHA256
        ),
        "v1_blocker": {
            "path": str(core.V1_ROOT / "status/blocker.json"),
            "sha256": core.EXPECTED_BLOCKER_SHA256,
            "status": "blocked_missing_candidate_level_scanner",
        },
        "v1_expanded_admission": {
            "path": str(core.V1_ROOT / "audits/admission.json"),
            "sha256": core.EXPECTED_ADMISSION_SHA256,
            "admission_passed": False,
            "false_positive_unique_count": 3,
        },
        "immutable_inputs": {},
        "immutable_inputs_sha256": core.canonical_sha256({}),
        "actual_feedback_v3_executors": [
            dict(record) for record in core.FROZEN_ACTUAL_V3_EXECUTORS
        ],
    }
    return {
        **unsigned,
        "contract_sha256": core.canonical_sha256(unsigned),
    }


def _real_task4_artifacts(root: Path) -> dict[str, Path | str]:
    deployment = deployment_bundle_v2.build_deployment_bundle(
        v2_root=root, frozen_repo_root=ROOT
    )
    root.mkdir(parents=True, exist_ok=True)
    selection = phase2_fixture._source_selection(root)
    request = selection["measurement_request"]
    source_plan = source_resolution_v2.build_source_resolution_plan(request)
    source_result = phase2_fixture._write_source_result(selection, source_plan)
    resolved_by_id = {row["candidate_id"]: row for row in source_result["rows"]}
    dimensions = {}
    for row in request["rows"]:
        candidate_id = row["row_id"]
        resolved = resolved_by_id[candidate_id]
        dimensions[candidate_id] = {
            **adapter_fixture._dimensions(candidate_id, row["genome"]),
            "source_checkpoint_sha256": resolved["checkpoint_sha256"],
            "onnx_sha256": resolved["onnx_sha256"],
        }
    adapter = adapter_fixture._adapter()
    frozen = adapter.bind_selector_output(
        selection,
        exact_dimensions_by_candidate=dimensions,
    )
    binding = frozen["selection_binding"]
    cache = adapter_fixture._empty_cache()
    reveal = cache_v2.reveal_v2_cache_after_selection(request, binding, cache)
    plan = adapter.build_miss_only_physical_plan(
        request, binding, reveal, cache_snapshot=cache
    )
    contract = _formal_contract(root)
    admission = online.validate_executor_admission(
        plan,
        logical_request=request,
        selection_binding=binding,
        cache_reveal=reveal,
        cache_snapshot=cache,
        contract_sha256=str(contract["contract_sha256"]),
    )
    contract_path = root / "contracts/core_ablation_v2.json"
    round_dir = root / "variants/full/seed_20260718/round_00"
    contract_path.parent.mkdir(parents=True)
    round_dir.mkdir(parents=True)
    artifacts = {
        "contract": (contract_path, contract),
        "plan": (round_dir / "miss_only_physical_request.json", plan),
        "logical": (round_dir / "logical_request.json", request),
        "binding": (round_dir / "exact_selection_binding.json", binding),
        "reveal": (round_dir / "cache_reveal.json", reveal),
        "cache": (round_dir / "cache_snapshot_before_reveal.json", cache),
        "executor_admission": (round_dir / "executor_admission.json", admission),
        "source_plan": (round_dir / "source_resolution_plan.json", source_plan),
        "source_result": (
            round_dir / "source_resolution_result.json",
            source_result,
        ),
    }
    for path, payload in artifacts.values():
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    receipt_unsigned = {
        "schema_version": "stage7_core_no_gpu_dry_run_v2",
        "formal_v2_root": str(root.resolve()),
        "frozen_repo_root": str(ROOT.resolve()),
        "formal_v2_gpu_jobs_launched": 0,
        "cuda_visible_devices": "",
        "deployment_bundle_sha256": deployment["deployment_bundle_sha256"],
        "deployment_primitive_pins_sha256": (
            deployment_bundle_v2.primitive_pins_sha256()
        ),
        "synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
    }
    receipt = {
        **receipt_unsigned,
        "dry_run_receipt_sha256": _sha(receipt_unsigned),
    }
    receipt_path = root / "audits/no_gpu_dry_run/dry_run_receipt.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return {
        "root": root,
        "round_dir": round_dir,
        "contract_sha": str(contract["contract_sha256"]),
        "source_plan_sha": str(source_plan["source_resolution_plan_sha256"]),
        "source_result_sha": str(source_result["source_resolution_result_sha256"]),
        "resolved_source_lock_sha": scheduler_requests.resolved_source_lock_sha256(
            plan, source_result
        ),
        "expected_release_sha256": deployment["deployment_release_sha256"],
        "expected_manifest_file_sha256": deployment[
            "deployment_manifest_file_sha256"
        ],
        "no_gpu_receipt": receipt_path,
        **{name: path for name, (path, _) in artifacts.items()},
    }


def _run(
    tmp_path: Path,
    *,
    dry_run: bool = True,
    cuda_visible_devices: str = "",
    bundle_root_override: Path | None = None,
    tamper_manifest: bool = False,
    tamper_state: bool = False,
    remove_receipt: bool = False,
    allow_parent_bytecode: bool = False,
    omit_external_pins: bool = False,
    release_pin_override: str | None = None,
    manifest_pin_override: str | None = None,
) -> subprocess.CompletedProcess[str]:
    artifacts = _real_task4_artifacts(tmp_path / "formal-v2")
    deployment = Path(artifacts["root"]) / "deployment"
    if tamper_manifest:
        (deployment / "deployment_manifest_v2.json").write_bytes(b"{}\n")
    if tamper_state:
        state_path = deployment / "deployment_state_v2.json"
        state = json.loads(state_path.read_text())
        state["deployment_bundle_sha256"] = "f" * 64
        state_path.write_text(json.dumps(state) + "\n")
    if remove_receipt:
        Path(artifacts["no_gpu_receipt"]).unlink()
    physical = json.loads(Path(artifacts["plan"]).read_text())
    output = tmp_path / "admission.json"
    command = [
        str(
            Path(artifacts["root"])
            / "deployment/code/scripts/stage7_execute_actual_v3_misses_v2.sh"
        ),
        "--physical-plan-json",
        str(artifacts["plan"]),
        "--formal-v2-root",
        str(artifacts["root"]),
        "--logical-request-json",
        str(artifacts["logical"]),
        "--selection-binding-json",
        str(artifacts["binding"]),
        "--cache-reveal-json",
        str(artifacts["reveal"]),
        "--cache-snapshot-json",
        str(artifacts["cache"]),
        "--output-dir",
        str(tmp_path / "execution"),
        "--admission-json",
        str(output),
        "--executor-admission-json",
        str(artifacts["executor_admission"]),
        "--source-plan-json",
        str(artifacts["source_plan"]),
        "--source-result-json",
        str(artifacts["source_result"]),
        "--source-plan-sha256",
        str(artifacts["source_plan_sha"]),
        "--source-result-sha256",
        str(artifacts["source_result_sha"]),
        "--resolved-source-lock-sha256",
        str(artifacts["resolved_source_lock_sha"]),
        "--contract-sha256",
        str(artifacts["contract_sha"]),
        "--request-sha256",
        str(physical["logical_request_sha256"]),
        "--gpu-uuids",
        "GPU-a,GPU-b,GPU-c,GPU-d",
        "--no-gpu-receipt-json",
        str(artifacts["no_gpu_receipt"]),
    ]
    if not omit_external_pins:
        command.extend(
            [
                "--expected-release-sha256",
                release_pin_override
                or str(artifacts["expected_release_sha256"]),
                "--expected-manifest-file-sha256",
                manifest_pin_override
                or str(artifacts["expected_manifest_file_sha256"]),
            ]
        )
    if dry_run:
        command.append("--dry-run")
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "STAGE7_CONFIG_PYTHON": str(PYTHON),
        "STAGE7_BUNDLE_CODE_ROOT": str(
            bundle_root_override or (Path(artifacts["root"]) / "deployment/code")
        ),
        "STAGE7_FROZEN_REPO_ROOT": str(ROOT),
    }
    if allow_parent_bytecode:
        environment.pop("PYTHONDONTWRITEBYTECODE", None)
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_dry_run_rejects_wrong_bundle_code_root_before_admission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, bundle_root_override=tmp_path / "wrong-bundle")
    assert result.returncode != 0
    assert not (tmp_path / "admission.json").exists()


def test_bundled_shell_rejects_tampered_deployment_manifest_before_admission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, tamper_manifest=True)
    assert result.returncode != 0
    assert not (tmp_path / "admission.json").exists()


def test_bundled_shell_rejects_tampered_deployment_state_before_admission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, tamper_state=True)
    assert result.returncode != 0
    assert not (tmp_path / "admission.json").exists()


def test_bundled_shell_requires_external_release_and_manifest_pins_before_admission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, omit_external_pins=True)
    assert result.returncode != 0
    assert "external deployment SHA arguments" in result.stderr
    assert not (tmp_path / "admission.json").exists()


def test_bundled_shell_rejects_external_release_pin_drift_before_admission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, release_pin_override=INVALID_EXTERNAL_PIN)
    assert result.returncode != 0
    assert "external release pin mismatch" in result.stderr
    assert not (tmp_path / "admission.json").exists()


def test_bundled_shell_rejects_malformed_external_manifest_pin_before_admission(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, manifest_pin_override="BAD")
    assert result.returncode != 0
    assert "external deployment SHA arguments" in result.stderr
    assert not (tmp_path / "admission.json").exists()


def test_dry_run_maps_misses_to_existing_v3_primitives_without_cuda_launch(
    tmp_path: Path,
):
    result = _run(tmp_path, dry_run=True)

    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "admission.json").read_text())
    assert audit["admission_passed"] is True
    assert audit["gpu_jobs_launched"] == 0
    assert audit["logical_row_count"] == 4
    assert audit["physical_row_count"] == 4
    assert audit["dry_run"] is True
    assert audit["cuda_visible_devices"] == ""
    assert [step["primitive"] for step in audit["execution_steps"]] == [
        "stage3_v3_quant_contract",
        "stage5_v2_performance_plan_builder",
        "stage3_v3_performance_executor",
        "stage5_v2_ap_plan_builder",
        "stage3_v3_ap_executor",
        "stage5_v2_finalizer_input",
        "stage5_actual_feedback_v3_promoter_input",
    ]
    assert all(step["exists"] is True for step in audit["execution_steps"])
    assert audit["selected_only_physical_request"] is True
    assert audit["measurement_semantics_copied"] is False
    assert audit["source_materializer_invocation_count"] == 0
    assert audit["source_verification_count"] == 1
    assert audit["source_evidence_producer"]["mode"] == "preverified_source_input"
    assert (
        audit["independent_projection_schema"]
        == "stage5_independent_validation_request_v1"
    )
    assert len(audit["independent_projection_sha256"]) == 64
    assert audit["actual_mode_recovery_engine"] == (
        "framework.stage7.actual_mode_v2.execute_recoverable_attempt"
    )


def test_non_dry_run_refuses_empty_cuda_visibility_before_any_output(
    tmp_path: Path,
):
    result = _run(tmp_path, dry_run=False, cuda_visible_devices="")

    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES is empty" in result.stderr
    assert not (tmp_path / "admission.json").exists()


def test_actual_shell_refuses_missing_authenticated_no_gpu_receipt(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, remove_receipt=True)

    assert result.returncode != 0


def test_bundled_shell_never_contaminates_deployment_with_python_bytecode(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, allow_parent_bytecode=True)

    assert result.returncode == 0, result.stderr
    deployment = tmp_path / "formal-v2/deployment"
    assert list(deployment.rglob("__pycache__")) == []
    assert list(deployment.rglob("*.pyc")) == []


def test_dry_run_rejects_plan_not_bound_by_task4_executor_admission(
    tmp_path: Path,
):
    artifacts = _real_task4_artifacts(tmp_path / "formal-v2")
    admission = Path(artifacts["executor_admission"])
    payload = json.loads(admission.read_text())
    payload["physical_request_sha256"] = "f" * 64
    unsigned = {
        key: value for key, value in payload.items() if key != "admission_sha256"
    }
    payload["admission_sha256"] = _sha(unsigned)
    admission.write_text(json.dumps(payload) + "\n")
    result = subprocess.run(
        [
            str(SCRIPT),
            "--physical-plan-json",
            str(artifacts["plan"]),
            "--formal-v2-root",
            str(artifacts["root"]),
            "--expected-release-sha256",
            str(artifacts["expected_release_sha256"]),
            "--expected-manifest-file-sha256",
            str(artifacts["expected_manifest_file_sha256"]),
            "--logical-request-json",
            str(artifacts["logical"]),
            "--selection-binding-json",
            str(artifacts["binding"]),
            "--cache-reveal-json",
            str(artifacts["reveal"]),
            "--cache-snapshot-json",
            str(artifacts["cache"]),
            "--output-dir",
            str(tmp_path / "execution"),
            "--admission-json",
            str(tmp_path / "admission.json"),
            "--executor-admission-json",
            str(admission),
            "--source-plan-json",
            str(artifacts["source_plan"]),
            "--source-result-json",
            str(artifacts["source_result"]),
            "--source-plan-sha256",
            str(artifacts["source_plan_sha"]),
            "--source-result-sha256",
            str(artifacts["source_result_sha"]),
            "--resolved-source-lock-sha256",
            str(artifacts["resolved_source_lock_sha"]),
            "--contract-sha256",
            str(artifacts["contract_sha"]),
            "--request-sha256",
            str(
                json.loads(Path(artifacts["plan"]).read_text())[
                    "logical_request_sha256"
                ]
            ),
            "--gpu-uuids",
            "GPU-a,GPU-b,GPU-c,GPU-d",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "STAGE7_CONFIG_PYTHON": str(PYTHON),
            "STAGE7_BUNDLE_CODE_ROOT": str(Path(artifacts["root"]) / "deployment/code"),
            "STAGE7_FROZEN_REPO_ROOT": str(ROOT),
        },
        check=False,
    )

    assert result.returncode != 0
    assert "Task4" in result.stderr


def test_shell_uses_no_eval_and_passes_bash_syntax_check():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "eval " not in source
    assert "set -euo pipefail" in source
    result = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_miss_executor_requires_formal_source_plan_result_and_sha_flags():
    source = SCRIPT.read_text(encoding="utf-8")
    for flag in (
        "--source-plan-json",
        "--source-result-json",
        "--source-plan-sha256",
        "--source-result-sha256",
        "--resolved-source-lock-sha256",
        "--expected-release-sha256",
        "--expected-manifest-file-sha256",
    ):
        assert flag in source


def test_miss_executor_never_invokes_source_materializer():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '("stage5_source_materializer",' not in source
    assert '"source_materializer_invocation_count": 0' in source


def test_miss_executor_reports_one_formal_source_verification():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '"source_verification_count": 1' in source
    assert "validate_formal_source_resolution_result" in source
    assert "resolved_source_bindings" in source


def test_miss_executor_rejects_synthetic_source_artifacts_by_schema():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "stage7_source_resolution_synthetic_dryrun_v2" in source
    assert "synthetic source result cannot admit measurement" in source


def test_actual_mode_invokes_reviewed_pipeline_and_recovery_driver():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "actual execution remains disabled" not in source
    assert "validate_actual_runtime_admission" in source
    assert "build_concrete_stage_runner" in source
    assert "execute_recoverable_attempt" in source
    assert "--scheduler-state-json" in source
    assert "--controller-id" in source
    assert "source_materializer_invocation_count" in source
