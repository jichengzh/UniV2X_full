#!/usr/bin/env python3
"""Formal Stage7-to-actual-v3 integration dry-run with no GPU truth claims."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage7.core_ablation_v2 import CORE_VARIANTS
from framework.stage7.no_gpu_support_v2 import (
    canonical_sha256,
    close_synthetic_source_protocol,
    deployment_binding,
    deployment_bundle_v2,
    exact_admission_receipts_closed,
    file_sha256 as _file_sha256,
    read_mapping as _read_mapping,
    require_no_gpu_environment,
    write_isolated_json as _write_isolated_json,
)

stage5_search = None
actual_selector = None
core_cache_v2 = None
search_policy = None
stage5_promote = None
online = None
finalizer = None
prepare_v2 = None

DRY_RUN_SCHEMA = "stage7_core_no_gpu_dry_run_v2"
SYNTHETIC_FIXTURE_SCHEMA = "stage7_synthetic_protocol_fixture_v2"
SYNTHETIC_SUCCESS_STATUS = "measured_success_gold"
DRY_RUN_UUIDS = tuple(f"DRYRUN-H800-UUID-{index}" for index in range(4))
JSON = dict[str, Any]
Initialize = Callable[..., Mapping[str, Any]]
ValidateRoot = Callable[..., Mapping[str, Any]]
PrepareRound = Callable[..., Mapping[str, Any]]
InspectRound = Callable[[Path, str], Mapping[str, Any]]
AdmitMisses = Callable[..., Mapping[str, Any]]
PromoteSynthetic = Callable[..., Mapping[str, Any]]
SelectNext = Callable[..., Mapping[str, Any]]
PrepareRoot = Callable[..., Mapping[str, Any]]
BindSyntheticSource = Callable[..., Mapping[str, Any]]
BindFormalSource = Callable[..., Mapping[str, Any]]


def _insert_runtime_repo_root(repo_root: Path) -> None:
    repo = str(Path(repo_root).resolve())
    if repo not in sys.path:
        insert_at = 1 if str(REPO_ROOT) in sys.path else 0
        sys.path.insert(insert_at, repo)
    framework = sys.modules.get("framework")
    if framework is not None and hasattr(framework, "__path__"):
        from pkgutil import extend_path

        framework.__path__ = extend_path(framework.__path__, framework.__name__)


def _ensure_runtime_modules(repo_root: Path = REPO_ROOT) -> None:
    """Load actual-v3 runtime modules only after the frozen repo root is known."""
    global actual_selector
    global core_cache_v2
    global finalizer
    global online
    global prepare_v2
    global search_policy
    global stage5_promote
    global stage5_search
    if (
        stage5_search is not None
        and actual_selector is not None
        and core_cache_v2 is not None
        and search_policy is not None
        and stage5_promote is not None
        and online is not None
        and finalizer is not None
        and prepare_v2 is not None
    ):
        return
    _insert_runtime_repo_root(repo_root)
    stage5_search = importlib.import_module(
        "framework.stage5.single_target_search_v2"
    )
    actual_selector = importlib.import_module(
        "framework.stage7.actual_v3_selector_v2"
    )
    core_cache_v2 = importlib.import_module("framework.stage7.core_cache_v2")
    search_policy = importlib.import_module("framework.stage7.search_policy_v1")
    stage5_promote = importlib.import_module(
        "scripts.stage5_promote_actual_feedback_v3"
    )
    online = importlib.import_module("scripts.stage7_core_online_ablation_v2")
    finalizer = importlib.import_module("scripts.stage7_finalize_core_ablation_v2")
    prepare_v2 = importlib.import_module("scripts.stage7_prepare_core_ablation_v2")


def prepare_and_initialize_no_gpu_root(
    *,
    v1_root: Path,
    v2_root: Path,
    prepare_inputs: Mapping[str, Any],
    embedded_source_manifest: Mapping[str, Any],
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    orchestrator_pid: int | None = None,
    repo_root: Path = REPO_ROOT,
    prepare_root: PrepareRoot | None = None,
    initialize: Initialize | None = None,
) -> JSON:
    """Prepare the exact v2 base tree and initialize its twelve seed roots once."""
    root = Path(v2_root).resolve()
    _ensure_runtime_modules(repo_root)
    active_prepare = prepare_v2.prepare_v2_root if prepare_root is None else prepare_root
    active_initialize = online.initialize_formal_v2 if initialize is None else initialize
    prepared = active_prepare(
        v1_root=Path(v1_root).resolve(),
        v2_root=root,
        inputs=copy.deepcopy(dict(prepare_inputs)),
        embedded_source_manifest=copy.deepcopy(dict(embedded_source_manifest)),
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        orchestrator_pid=orchestrator_pid,
        repo_root=Path(repo_root).resolve(),
    )
    initialized = active_initialize(
        root,
        repo_root=Path(repo_root).resolve(),
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    if initialized.get("trajectory_count") != 12:
        raise ValueError("no-GPU fresh root did not initialize 12 trajectories")
    return {
        **copy.deepcopy(dict(prepared)),
        **copy.deepcopy(dict(initialized)),
    }


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def build_synthetic_protocol_fixture(
    request: Mapping[str, Any],
    *,
    frozen_onnx: Path,
    output_dir: Path,
) -> JSON:
    """Build quarantined terminal-shaped rows over a real frozen ONNX file."""
    onnx = Path(frozen_onnx).resolve(strict=True)
    onnx_sha = _file_sha256(onnx)
    rows = request.get("rows")
    row_sha = request.get("row_sha256")
    if (
        request.get("schema_version") != "stage5_measurement_request_v2"
        or not isinstance(rows, list)
        or len(rows) != 4
        or not isinstance(row_sha, Mapping)
    ):
        raise ValueError("synthetic fixture requires a real four-row request")
    historical: list[JSON] = []
    source_records: list[JSON] = []
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("synthetic fixture request row is invalid")
        requested = copy.deepcopy(dict(raw))
        candidate_id = _row_id(requested)
        if not candidate_id or row_sha.get(candidate_id) != canonical_sha256(requested):
            raise ValueError("synthetic fixture request row SHA drift")
        width = list(requested.get("width") or [])
        source_payload = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "status": "ready",
            "group_id": requested.get("group_id"),
            "model": requested.get("model"),
            "width": "x".join(map(str, width)),
            "source_plan_sha256": requested.get("source_evidence_sha256"),
            "onnx_path": str(onnx),
            "onnx_sha256": onnx_sha,
            "dry_run_synthetic_source_binding": True,
            "eligible_for_formal_finalization": False,
        }
        source_dir = (output_dir / "synthetic_sources").resolve(strict=False)
        safe_name = f"{ordinal:02d}_{str(row_sha[candidate_id])[:16]}.json"
        source_path = (source_dir / safe_name).resolve(strict=False)
        if source_path.parent != source_dir:
            raise ValueError("synthetic source evidence path escapes output")
        _write_isolated_json(source_path, source_payload)
        source_sha = _file_sha256(source_path)
        source_records.append({"path": str(source_path), "sha256": source_sha})
        historical.append(
            {
                **requested,
                "terminal_status": SYNTHETIC_SUCCESS_STATUS,
                "measurement_request_row_sha256": row_sha[candidate_id],
                "latency_ms": 1.0 + ordinal,
                "energy_j": 0.5 + ordinal,
                "ap30": 0.90,
                "ap50": 0.80,
                "ap70": 0.70,
                "metric_source": "synthetic_protocol_fixture",
                "training_source": "online_feedback",
                "synthetic_training_source_contract_only": True,
                "materialized_source_evidence_path": str(source_path),
                "materialized_source_evidence_sha256": source_sha,
                "synthetic_non_measurement": True,
                "eligible_for_cache_append": False,
                "eligible_for_formal_finalization": False,
            }
        )
    payload = {
        "schema_version": SYNTHETIC_FIXTURE_SCHEMA,
        "synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
        "formal_v2_gpu_jobs_launched": 0,
        "frozen_onnx_path": str(onnx),
        "frozen_onnx_sha256": onnx_sha,
        "logical_request_sha256": request.get("measurement_request_sha256"),
        "source_records": source_records,
        "historical_feedback": historical,
    }
    fixture = {
        **payload,
        "synthetic_fixture_sha256": canonical_sha256(payload),
    }
    _write_isolated_json(output_dir / "synthetic_protocol_fixture.json", fixture)
    return fixture


def _inspect_round(root: Path, variant: str) -> JSON:
    directory = root / "variants" / variant / "seed_20260718" / "round_00"
    return {
        "round_dir": directory,
        "request": _read_mapping(directory / "logical_request.json"),
        "cache_snapshot": _read_mapping(
            directory / "cache_snapshot_before_reveal.json"
        ),
        "cache_reveal": _read_mapping(directory / "cache_reveal.json"),
        "physical_plan": _read_mapping(directory / "miss_only_physical_request.json"),
        "executor_admission": _read_mapping(directory / "executor_admission.json"),
    }


def _admit_misses(
    root: Path,
    *,
    variant: str,
    inspected: Mapping[str, Any],
    repo_root: Path,
    contract_sha256: str,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    audit_dir: Path,
) -> JSON:
    round_dir = Path(str(inspected["round_dir"]))
    admission_path = audit_dir / variant / "miss_execution_admission.json"
    command = [
        str(
            root
            / "deployment"
            / "code"
            / "scripts/stage7_execute_actual_v3_misses_v2.sh"
        ),
        "--physical-plan-json",
        str(round_dir / "miss_only_physical_request.json"),
        "--formal-v2-root",
        str(root),
        "--logical-request-json",
        str(round_dir / "logical_request.json"),
        "--selection-binding-json",
        str(round_dir / "selection_binding.json"),
        "--cache-reveal-json",
        str(round_dir / "cache_reveal.json"),
        "--cache-snapshot-json",
        str(round_dir / "cache_snapshot_before_reveal.json"),
        "--output-dir",
        str(audit_dir / variant / "physical_execution_not_started"),
        "--admission-json",
        str(admission_path),
        "--executor-admission-json",
        str(round_dir / "executor_admission.json"),
        "--contract-sha256",
        contract_sha256,
        "--expected-release-sha256",
        expected_release_sha256,
        "--expected-manifest-file-sha256",
        expected_manifest_file_sha256,
        "--request-sha256",
        str(inspected["request"]["measurement_request_sha256"]),
        "--gpu-uuids",
        ",".join(DRY_RUN_UUIDS),
        "--dry-run",
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "STAGE7_BUNDLE_CODE_ROOT": str(root / "deployment" / "code"),
        "STAGE7_FROZEN_REPO_ROOT": str(repo_root),
        "PYTHONPATH": str(root / "deployment" / "code") + os.pathsep + str(repo_root),
    }
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise ValueError(
            "actual-v3 miss admission dry-run failed: " + completed.stderr.strip()
        )
    admission = _read_mapping(admission_path)
    if (
        admission.get("admission_passed") is not True
        or admission.get("dry_run") is not True
        or admission.get("gpu_jobs_launched") != 0
    ):
        raise ValueError("actual-v3 miss admission did not remain no-GPU")
    return admission


def _synthetic_extractor(path: Path) -> JSON:
    return {
        "schema": "stage7_synthetic_onnx_graph_probe_v2",
        "onnx_sha256": _file_sha256(path),
        "node_count": 0,
        "conv_count": 0,
        "dry_run_synthetic_non_measurement": True,
    }


def _promote_synthetic(
    fixture: Mapping[str, Any],
    *,
    request_path: Path,
    output_dir: Path,
) -> JSON:
    feedback_path = output_dir / "synthetic_historical_feedback.json"
    _write_isolated_json(feedback_path, fixture["historical_feedback"])
    promoted = stage5_promote.promote_feedback_batch(
        request_path, feedback_path, extractor=_synthetic_extractor
    )
    barrier = stage5_search.finalize_atomic_batch(
        _read_mapping(request_path), promoted["rows"]
    )
    wrapper = {
        "schema_version": "stage7_synthetic_promotion_barrier_fixture_v2",
        "synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
        "canonical_barrier_written": False,
        "promoted_rows": promoted["rows"],
        "audit": promoted["audit"],
        "barrier": barrier,
    }
    _write_isolated_json(output_dir / "synthetic_promotion_barrier.json", wrapper)
    return {
        "rows": promoted["rows"],
        "audit": promoted["audit"],
        "barrier": barrier,
    }


def _select_next(
    root: Path,
    *,
    variant: str,
    validated_root: Mapping[str, Any],
    promoted_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> JSON:
    inputs = online._selector_inputs(validated_root)
    profile = inputs["capability_profiles"][0]
    task = search_policy.build_stage7_task(profile)
    round_dir = root / "variants" / variant / "seed_20260718" / "round_00"
    round_zero = _read_mapping(round_dir / "selector_output.json")
    request = _read_mapping(round_dir / "logical_request.json")
    selected_ids = [_row_id(row) for row in request["rows"]]
    selected = actual_selector.select_actual_v3_pre_scan_round(
        variant=variant,
        seed=20260718,
        round_index=1,
        task=task,
        pre_scan_pool=validated_root["pre_scan_pool"],
        initial_rows=inputs["initial_rows"],
        initial_graph_features=inputs["initial_graph_features"],
        capability_profiles=inputs["capability_profiles"],
        closure=inputs["closure"],
        prior_selected_ids=selected_ids,
        promoted_feedback=promoted_rows,
        a2_frozen=round_zero.get("a2_frozen"),
    )
    payload = {
        "schema_version": "stage7_synthetic_next_round_request_v2",
        "synthetic_preparation_only": True,
        "eligible_for_formal_execution": False,
        "variant": variant,
        "round_index": 1,
        "selection": copy.deepcopy(dict(selected.selection)),
        "audit": copy.deepcopy(dict(selected.audit)),
    }
    _write_isolated_json(output_dir / "synthetic_next_round_request.json", payload)
    return payload


def _validate_round_zero(inspected: Mapping[str, Any]) -> None:
    cache = core_cache_v2.validate_actual_v3_cache(inspected["cache_snapshot"])
    reveal = inspected["cache_reveal"]
    plan = inspected["physical_plan"]
    admission = inspected["executor_admission"]
    entries = reveal.get("entries") if isinstance(reveal, Mapping) else None
    if cache.get("entries") != {} or cache.get("lineage") != []:
        raise ValueError("no-GPU dry-run initial cache must be exactly empty")
    if (
        not isinstance(entries, list)
        or len(entries) != 4
        or any(entry.get("disposition") != "miss" for entry in entries)
        or plan.get("physical_row_count") != 4
    ):
        raise ValueError("no-GPU dry-run requires exactly four misses")
    if (
        admission.get("admission_passed") is not True
        or admission.get("gpu_jobs_launched") != 0
    ):
        raise ValueError("Task4 admission launched a GPU or failed")


def _selected_count(next_request: Mapping[str, Any]) -> int:
    rows = next_request.get("rows")
    if isinstance(rows, list):
        return len(rows)
    selection = next_request.get("selection")
    if isinstance(selection, Mapping):
        request = selection.get("measurement_request")
        if isinstance(request, Mapping) and isinstance(request.get("rows"), list):
            return len(request["rows"])
    return 0


def run_no_gpu_dry_run(
    v2_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    frozen_onnx: Path,
    expected_release_sha256: str,
    expected_manifest_file_sha256: str,
    expected_frozen_onnx_sha256: str | None = None,
    initialize: Initialize | None = None,
    validate_root: ValidateRoot | None = None,
    prepare_round: PrepareRound | None = None,
    inspect_round: InspectRound = _inspect_round,
    admit_misses: AdmitMisses = _admit_misses,
    bind_synthetic_source_protocol: BindSyntheticSource | None = None,
    bind_formal_source: BindFormalSource | None = None,
    promote_synthetic: PromoteSynthetic = _promote_synthetic,
    select_next: SelectNext = _select_next,
) -> JSON:
    """Exercise formal initialization/selection with synthetic state quarantined."""
    require_no_gpu_environment()
    root = Path(v2_root).resolve()
    repo = Path(repo_root).resolve()
    _ensure_runtime_modules(repo)
    active_initialize = online.initialize_formal_v2 if initialize is None else initialize
    active_validate_root = (
        online.validate_formal_v2_root if validate_root is None else validate_root
    )
    active_prepare_round = online.prepare_round if prepare_round is None else prepare_round
    try:
        deployment = deployment_binding(
            root,
            frozen_repo_root=repo,
            expected_release_sha256=expected_release_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "no-GPU dry-run requires an authenticated deployment bundle"
        ) from error
    onnx = Path(frozen_onnx).resolve(strict=True)
    onnx_sha = _file_sha256(onnx)
    if (
        expected_frozen_onnx_sha256 is not None
        and onnx_sha != expected_frozen_onnx_sha256
    ):
        raise ValueError("frozen dry-run ONNX SHA drift")
    if (root / "initialize_state.json").is_file():
        initialized = _read_mapping(root / "initialize_state.json")
    else:
        initialized = active_initialize(
            root,
            repo_root=repo,
            expected_release_sha256=expected_release_sha256,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
        )
    validated = active_validate_root(
        root,
        repo_root=repo,
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    if initialized.get("trajectory_count") != 12:
        raise ValueError("no-GPU dry-run did not initialize 12 trajectories")
    audit_dir = root / "audits/no_gpu_dry_run"
    contract_sha = str(validated["contract"]["contract_sha256"])
    variant_receipts: list[JSON] = []
    active_bind_synthetic = (
        (
            lambda target, **kwargs: close_synthetic_source_protocol(
                target,
                online_module=online,
                expected_release_sha256=expected_release_sha256,
                expected_manifest_file_sha256=expected_manifest_file_sha256,
                **kwargs,
            )
        )
        if bind_synthetic_source_protocol is None
        else bind_synthetic_source_protocol
    )
    active_bind_formal = (
        online.bind_reveal_after_source_ready
        if bind_formal_source is None
        else bind_formal_source
    )
    pre_gate_source_results_bound = 0
    for variant in CORE_VARIANTS:
        synthetic_source: Mapping[str, Any] | None = None
        try:
            prepared = active_prepare_round(
                root,
                variant=variant,
                seed=20260718,
                round_index=0,
                repo_root=repo,
                expected_release_sha256=expected_release_sha256,
                expected_manifest_file_sha256=expected_manifest_file_sha256,
            )
            round_dir = root / "variants" / variant / "seed_20260718" / "round_00"
            exact_artifacts = (
                "exact_selection_binding.json",
                "cache_snapshot_before_reveal.json",
                "cache_reveal.json",
                "miss_only_physical_request.json",
                "executor_admission.json",
            )
            exact_presence = [(round_dir / name).is_file()
                              for name in exact_artifacts]
            if any(exact_presence) and not all(exact_presence):
                raise ValueError(
                    "partial formal exact-reveal artifacts after source plan"
                )
            source_result_path = round_dir / "source_resolution_result.json"
            if source_result_path.is_file() and not any(exact_presence):
                bound = copy.deepcopy(
                    dict(
                        active_bind_formal(
                            root,
                            variant=variant,
                            seed=20260718,
                            round_index=0,
                            source_result_path=source_result_path,
                            synthetic_no_gpu_dryrun=False,
                            repo_root=repo,
                            expected_release_sha256=expected_release_sha256,
                            expected_manifest_file_sha256=(
                                expected_manifest_file_sha256
                            ),
                        )
                    )
                )
                if bound.get("controller_state") != "CACHE_REVEALED":
                    raise ValueError(
                        "pre-gate source bind did not close exact cache reveal"
                    )
                exact_presence = [
                    (round_dir / name).is_file() for name in exact_artifacts
                ]
                if not all(exact_presence):
                    raise ValueError(
                        "pre-gate source bind left exact artifacts incomplete"
                    )
                pre_gate_source_results_bound += 1
            if (
                prepared.get("controller_state") == "SOURCE_PLAN_FROZEN"
                and not source_result_path.is_file()
            ):
                synthetic_source = active_bind_synthetic(
                    root,
                    variant=variant,
                    prepared=prepared,
                    repo_root=repo,
                    audit_dir=audit_dir,
                )
        except online.SourceResolutionRequiredBeforeExactCacheReveal as error:
            source_audit = copy.deepcopy(error.audit)
            _write_isolated_json(
                audit_dir / "source_resolution_blocker.json", source_audit
            )
            current_request_written = int(source_audit.get(
                "canonical_round0_request_written") is True)
            summary = {
                "schema_version": DRY_RUN_SCHEMA,
                "formal_v2_root": str(root),
                "frozen_repo_root": str(repo),
                "trajectory_count": 12,
                "round0_request_count":
                    len(variant_receipts) + current_request_written,
                "round0_miss_count": 4 * len(variant_receipts),
                "integration_closed": False,
                "synthetic_fixture_count": len(variant_receipts),
                "next_round_request_count": len(variant_receipts),
                "formal_v2_gpu_jobs_launched": 0,
                "pre_gate_source_results_bound": pre_gate_source_results_bound,
                "paper_ready": False,
                "core_ablation_ready": False,
                "full_gear_s7_ready": False,
                "scanner_component_status": "deferred_important_fix",
                "blocking_reason": source_audit.get(
                    "blocking_reason",
                    "source_resolution_required_before_exact_cache_reveal",
                ),
                "source_resolution_blocked_variant": variant,
                "source_resolution_audit": source_audit,
                "synthetic_non_measurement": True,
                "actual_v3_hardware_evidence": False,
                "eligible_for_cache_append": False,
                "eligible_for_formal_finalization": False,
                "canonical_initialization_artifacts_written": True,
                "canonical_round0_requests_written":
                    len(variant_receipts) + current_request_written,
                "canonical_terminal_artifacts_written": 0,
                "canonical_barriers_written": 0,
                "cache_appends": 0,
                "cuda_visible_devices": "",
                "frozen_onnx_path": str(onnx),
                "frozen_onnx_sha256": onnx_sha,
                "variant_receipts": variant_receipts,
                "deployment_manifest_sha256":
                    deployment["deployment_manifest_sha256"],
                "deployment_manifest_file_sha256":
                    deployment["deployment_manifest_file_sha256"],
                "deployment_bundle_sha256":
                    deployment["deployment_bundle_sha256"],
                "deployment_release_sha256":
                    deployment["deployment_release_sha256"],
                "deployment_primitive_pins_sha256": (
                    deployment_bundle_v2.primitive_pins_sha256()
                ),
            }
            receipt = {
                **summary,
                "dry_run_receipt_sha256": canonical_sha256(summary),
            }
            _write_isolated_json(audit_dir / "dry_run_receipt.json", receipt)
            finalizer.emit_non_final_dry_run(audit_dir / "finalizer_schema", summary)
            return summary
        if synthetic_source is None:
            inspected = copy.deepcopy(dict(inspect_round(root, variant)))
            _validate_round_zero(inspected)
            admitted = admit_misses(
                root,
                variant=variant,
                inspected=inspected,
                repo_root=repo,
                contract_sha256=contract_sha,
                expected_release_sha256=expected_release_sha256,
                expected_manifest_file_sha256=expected_manifest_file_sha256,
                audit_dir=audit_dir,
            )
            if admitted.get("gpu_jobs_launched") != 0:
                raise ValueError("dry-run miss admission launched a GPU")
        else:
            inspected = {
                "request": copy.deepcopy(dict(synthetic_source["request"])),
                "round_dir": round_dir,
            }
        variant_dir = audit_dir / variant
        fixture = build_synthetic_protocol_fixture(
            inspected["request"],
            frozen_onnx=onnx,
            output_dir=variant_dir,
        )
        promoted = promote_synthetic(
            fixture,
            request_path=(
                Path(str(inspected.get("round_dir"))) / "logical_request.json"
                if inspected.get("round_dir") is not None
                else variant_dir / "logical_request.synthetic-reference.json"
            ),
            output_dir=variant_dir,
        )
        audit = promoted.get("audit")
        barrier = promoted.get("barrier")
        if (
            not isinstance(audit, Mapping)
            or audit.get("promoted_row_count") != 4
            or audit.get("silent_surrogate_fallback_count") != 0
            or not isinstance(barrier, Mapping)
            or barrier.get("feedback_released") is not True
            or barrier.get("budget_consumed") != 4
        ):
            raise ValueError("synthetic promotion/barrier protocol did not close")
        next_request = select_next(
            root,
            variant=variant,
            validated_root=validated,
            promoted_rows=promoted["rows"],
            output_dir=variant_dir,
        )
        if _selected_count(next_request) != 4:
            raise ValueError("synthetic next-round request is not four rows")
        variant_receipts.append(
            {
                "variant": variant,
                "round0_miss_count": 0 if synthetic_source is not None else 4,
                "miss_admission_passed": synthetic_source is None,
                "synthetic_source_protocol_closed": synthetic_source is not None,
                "synthetic_promotion_rows": 4,
                "next_round_selected_count": 4,
            }
        )
    integration_closed = exact_admission_receipts_closed(
        variant_receipts, variants=CORE_VARIANTS, events_per_round=4
    )
    summary = {
        "schema_version": DRY_RUN_SCHEMA,
        "formal_v2_root": str(root),
        "frozen_repo_root": str(repo),
        "trajectory_count": 12,
        "round0_request_count": 4,
        "round0_miss_count": sum(
            int(receipt["round0_miss_count"]) for receipt in variant_receipts
        ),
        "integration_closed": integration_closed,
        "synthetic_fixture_count": 4,
        "next_round_request_count": 4,
        "formal_v2_gpu_jobs_launched": 0,
        "pre_gate_source_results_bound": pre_gate_source_results_bound,
        "paper_ready": False,
        "core_ablation_ready": False,
        "full_gear_s7_ready": False,
        "scanner_component_status": "deferred_important_fix",
        "blocking_reason": (
            None if integration_closed else finalizer.INTEGRATION_BLOCKING_REASON
        ),
        "synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
        "canonical_initialization_artifacts_written": True,
        "canonical_round0_requests_written": 4,
        "canonical_terminal_artifacts_written": 0,
        "canonical_barriers_written": 0,
        "cache_appends": 0,
        "cuda_visible_devices": "",
        "frozen_onnx_path": str(onnx),
        "frozen_onnx_sha256": onnx_sha,
        "variant_receipts": variant_receipts,
        "deployment_manifest_sha256": deployment["deployment_manifest_sha256"],
        "deployment_manifest_file_sha256":
            deployment["deployment_manifest_file_sha256"],
        "deployment_bundle_sha256": deployment["deployment_bundle_sha256"],
        "deployment_release_sha256": deployment["deployment_release_sha256"],
        "deployment_primitive_pins_sha256": (
            deployment_bundle_v2.primitive_pins_sha256()
        ),
    }
    receipt = {
        **summary,
        "dry_run_receipt_sha256": canonical_sha256(summary),
    }
    _write_isolated_json(audit_dir / "dry_run_receipt.json", receipt)
    finalizer.emit_non_final_dry_run(audit_dir / "finalizer_schema", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--frozen-onnx", type=Path, required=True)
    parser.add_argument("--frozen-onnx-sha256", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    parser.add_argument("--v1-root", type=Path)
    parser.add_argument("--prepare-inputs-json", type=Path)
    parser.add_argument("--embedded-source-manifest-json", type=Path)
    parser.add_argument("--orchestrator-pid", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prepare_arguments = (
        args.v1_root,
        args.prepare_inputs_json,
        args.embedded_source_manifest_json,
    )
    if any(value is not None for value in prepare_arguments):
        if any(value is None for value in prepare_arguments):
            raise ValueError(
                "fresh-root dry-run requires v1 root, prepare inputs and "
                "embedded source manifest together"
            )
        if args.orchestrator_pid is None:
            raise ValueError("fresh-root dry-run requires a persistent "
                             "--orchestrator-pid; the transient no-GPU CLI "
                             "PID cannot own recovery state")
        prepare_and_initialize_no_gpu_root(
            v1_root=args.v1_root,
            v2_root=args.v2_root,
            prepare_inputs=_read_mapping(args.prepare_inputs_json),
            embedded_source_manifest=_read_mapping(args.embedded_source_manifest_json),
            expected_release_sha256=args.expected_release_sha256,
            expected_manifest_file_sha256=args.expected_manifest_file_sha256,
            orchestrator_pid=args.orchestrator_pid,
            repo_root=args.repo_root,
        )
    result = run_no_gpu_dry_run(
        args.v2_root,
        repo_root=args.repo_root,
        frozen_onnx=args.frozen_onnx,
        expected_release_sha256=args.expected_release_sha256,
        expected_manifest_file_sha256=args.expected_manifest_file_sha256,
        expected_frozen_onnx_sha256=args.frozen_onnx_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
