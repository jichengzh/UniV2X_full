#!/usr/bin/env python3
"""Fail-closed paper finalizer for the Stage7 actual-feedback-v3 core ablation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5 import single_target_search_v2 as stage5_search
from framework.stage7 import actual_v3_adapter_v2 as actual_adapter
from framework.stage7 import ablation_statistics_v2 as statistics_v2
from framework.stage7 import core_ablation_v2 as contracts
from framework.stage7 import core_cache_v2
from framework.stage7 import formal_evidence_v2 as formal_evidence
from framework.stage7 import paper_outputs_v2 as paper_outputs
from framework.stage7.online_component_ablation_v1 import SEEDS
from scripts import stage7_core_online_ablation_v2 as online
from scripts import stage7_finalize_ablation_v1 as metric_helpers


SCHEMA_VERSION = formal_evidence.SCHEMA_VERSION
FORMAL_EVIDENCE_SCHEMA = formal_evidence.FORMAL_EVIDENCE_SCHEMA
EXECUTION_CLOSURE_SCHEMA = formal_evidence.EXECUTION_CLOSURE_SCHEMA
INTEGRATION_BLOCKING_REASON = formal_evidence.INTEGRATION_BLOCKING_REASON
CORE_VARIANTS = formal_evidence.CORE_VARIANTS
ROUND_COUNT = formal_evidence.ROUND_COUNT
BATCH_SIZE = formal_evidence.BATCH_SIZE
TOTAL_TRAJECTORIES = formal_evidence.TOTAL_TRAJECTORIES
TOTAL_EVENTS = formal_evidence.TOTAL_EVENTS
SUCCESS_STATUSES = formal_evidence.SUCCESS_STATUSES
TRUE_FAILURE_STATUSES = formal_evidence.TRUE_FAILURE_STATUSES
REQUIRED_AUDIT_FLAGS = formal_evidence.REQUIRED_AUDIT_FLAGS
REQUIRED_AUDIT_SCHEMAS = formal_evidence.REQUIRED_AUDIT_SCHEMAS
AUDIT_SOURCE_SCHEMAS = formal_evidence.AUDIT_SOURCE_SCHEMAS
FORMAL_OUTPUT_PATHS = formal_evidence.FORMAL_OUTPUT_PATHS
EVIDENCE_KINDS = formal_evidence.EVIDENCE_KINDS
JSON = formal_evidence.JSON
FinalizationError = formal_evidence.FinalizationError

_fail = formal_evidence.fail
_sha = formal_evidence.sha
_file_sha = formal_evidence.file_sha
_is_sha = formal_evidence.is_sha
_read_json = formal_evidence.read_json
_read_mapping = formal_evidence.read_mapping
_read_source_rows = formal_evidence.read_source_rows


def _rows(payload: Any, label: str) -> list[JSON]:
    value = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        _fail(f"{label} must contain object rows")
    return [copy.deepcopy(dict(row)) for row in value]


def _row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("row_id") or row.get("manifest_job_id") or "")


def _validate_embedded_sha(payload: Mapping[str, Any], field: str, label: str) -> JSON:
    copied = copy.deepcopy(dict(payload))
    recorded = copied.pop(field, None)
    if not _is_sha(recorded) or recorded != _sha(copied):
        _fail(f"{label} canonical SHA drift")
    return {**copied, field: recorded}


_validate_deploy_manifest = formal_evidence.validate_deploy_manifest
_validate_selector_audit = formal_evidence.validate_selector_audit
_validate_scheduler_state = formal_evidence.validate_scheduler_state
_validate_gpu_audit_rows = formal_evidence.validate_gpu_audit_rows
_validate_stage5_atomic_audit = formal_evidence.validate_stage5_atomic_audit
_validate_semantic_audit_source = formal_evidence.validate_semantic_audit_source


def _validate_bound_audit_artifacts(root: Path, records: Any) -> list[JSON]:
    return formal_evidence.validate_bound_audit_artifacts(
        root,
        records,
        semantic_validator=_validate_semantic_audit_source,
    )


def _load_execution_closure(root: Path) -> JSON:
    path = root / "status/formal_execution_closure_v2.json"
    closure = _read_mapping(path, "formal execution closure")
    _validate_embedded_sha(
        closure, "execution_closure_sha256", "formal execution closure"
    )
    audits = closure.get("audits")
    if (
        closure.get("schema_version") != EXECUTION_CLOSURE_SCHEMA
        or closure.get("passed") is not True
        or not isinstance(audits, Mapping)
        or set(audits) != set(REQUIRED_AUDIT_FLAGS)
        or any(value is not True for value in audits.values())
    ):
        _fail("formal execution closure audit is incomplete")
    return {
        **closure,
        "bound_audit_artifacts": _validate_bound_audit_artifacts(
            root, closure.get("bound_audit_artifacts")
        ),
    }


def _load_metric_context(
    contract: Mapping[str, Any],
) -> tuple[list[JSON], tuple[float, float, float]]:
    immutable = contract.get("immutable_inputs")
    if not isinstance(immutable, Mapping):
        _fail("formal contract lacks immutable metric inputs")
    gold_record = immutable.get("gold176")
    reference_record = immutable.get("hv_reference")
    if not isinstance(gold_record, Mapping) or not isinstance(
        reference_record, Mapping
    ):
        _fail("formal Gold176 or HV reference binding is missing")

    def load_bound(record: Mapping[str, Any], label: str) -> Any:
        path = Path(str(record.get("path") or ""))
        expected = str(record.get("sha256") or "")
        if (
            not path.is_absolute()
            or not path.is_file()
            or not _is_sha(expected)
            or _file_sha(path) != expected
        ):
            _fail(f"formal {label} binding drift")
        return _read_json(path, label)

    gold = load_bound(gold_record, "Gold176")
    raw_rows = gold.get("rows") if isinstance(gold, Mapping) else None
    if not isinstance(raw_rows, list):
        _fail("formal Gold176 rows are missing")
    initial: list[JSON] = []
    for raw in raw_rows:
        if (
            isinstance(raw, Mapping)
            and raw.get("model") == "pyramid"
            and raw.get("hardware_id") == "h800"
            and raw.get("dispatch_key") == "tvm_auto"
            and raw.get("terminal_status") in SUCCESS_STATUSES
        ):
            try:
                initial.append(metric_helpers._audit_result_row(dict(raw)))
            except (TypeError, ValueError, metric_helpers.FinalizationError) as error:
                _fail(f"formal Gold176 row is invalid: {error}")
    if not initial:
        _fail("formal Gold176 has no Pyramid/H800/tvm_auto success rows")
    reference_payload = load_bound(reference_record, "HV reference")
    values = (
        reference_payload.get("values")
        if isinstance(reference_payload, Mapping)
        else None
    )
    if not isinstance(values, Mapping):
        _fail("formal task-specific HV reference is missing")
    try:
        reference = tuple(
            float(values[field])
            for field in ("latency_ms", "energy_j", "negative_ap70")
        )
    except (KeyError, TypeError, ValueError) as error:
        _fail(f"formal task-specific HV reference is invalid: {error}")
    if len(reference) != 3 or not all(math.isfinite(value) for value in reference):
        _fail("formal task-specific HV reference is non-finite")
    return initial, reference


def _validate_failure_wrappers(
    round_dir: Path,
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, JSON]:
    path = round_dir / "actual_v3_failure_wrappers.json"
    if not path.is_file():
        return {}
    payload = _read_mapping(path, "actual-v3 failure wrappers")
    if payload.get("schema_version") != "stage7_actual_v3_failure_wrapper_batch_v2":
        _fail("actual-v3 failure wrapper batch schema drift")
    wrappers = payload.get("wrappers")
    if not isinstance(wrappers, list):
        _fail("actual-v3 failure wrapper list is invalid")
    validated: dict[str, JSON] = {}
    for wrapper in wrappers:
        try:
            item = actual_adapter.validate_actual_v3_failure(
                wrapper, request, binding, plan
            )
        except (OSError, ValueError) as error:
            _fail(f"actual-v3 candidate failure evidence is invalid: {error}")
        candidate_id = str(item["candidate_id"])
        if candidate_id in validated:
            _fail("duplicate actual-v3 candidate failure wrapper")
        validated[candidate_id] = item
    return validated


def _formal_exact_binding(
    binding: Mapping[str, Any],
) -> dict[str, JSON]:
    selected = binding.get("selected_candidates")
    if not isinstance(selected, list) or len(selected) != BATCH_SIZE:
        _fail("formal selection binding does not contain four candidates")
    result: dict[str, JSON] = {}
    for item in selected:
        if not isinstance(item, Mapping):
            _fail("formal selection row binding is invalid")
        try:
            dimensions = core_cache_v2.validate_formal_exact_dimensions(
                item.get("exact_key_dimensions")
            )
        except ValueError as error:
            _fail(f"formal exact-key dimensions are invalid: {error}")
        candidate_id = str(item.get("candidate_id") or "")
        if (
            not candidate_id
            or dimensions["candidate_id"] != candidate_id
            or candidate_id in result
        ):
            _fail("formal exact-key candidate identity drift")
        result[candidate_id] = {
            **copy.deepcopy(dict(item)),
            "exact_key_dimensions": dimensions,
        }
    return result


def _success_wrapper(
    *,
    disposition: str,
    reveal_entry: Mapping[str, Any],
    cache_after: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> JSON:
    terminal = (
        reveal_entry.get("terminal_evidence")
        if disposition == "hit"
        else (cache_after.get("entries") or {}).get(selected["exact_cache_key_sha256"])
    )
    try:
        validated = actual_adapter.validate_actual_v3_terminal_wrapper(terminal)
    except (OSError, ValueError, TypeError) as error:
        _fail(f"actual-v3 terminal evidence is invalid: {error}")
    if (
        validated["candidate_id"] != selected["candidate_id"]
        or validated["exact_key_dimensions"] != selected["exact_key_dimensions"]
        or validated["exact_cache_key_sha256"] != selected["exact_cache_key_sha256"]
    ):
        _fail("actual-v3 terminal exact-key binding drift")
    return validated


def _audit_formal_round(
    round_dir: Path,
    *,
    variant: str,
    seed: int,
    round_index: int,
    contract_sha256: str,
) -> tuple[list[JSON], int]:
    try:
        online.validate_committed_barrier(round_dir)
    except (OSError, ValueError) as error:
        _fail(f"formal atomic barrier validation failed: {error}")
    request = _read_mapping(round_dir / "logical_request.json", "logical request")
    binding = _read_mapping(round_dir / "selection_binding.json", "selection binding")
    snapshot = _read_mapping(
        round_dir / "cache_snapshot_before_reveal.json", "cache snapshot"
    )
    reveal = _read_mapping(round_dir / "cache_reveal.json", "cache reveal")
    plan = _read_mapping(
        round_dir / "miss_only_physical_request.json", "physical request"
    )
    stored_admission = _read_mapping(
        round_dir / "executor_admission.json", "executor admission"
    )
    historical = _rows(
        _read_json(round_dir / "historical_feedback.json", "historical feedback"),
        "historical feedback",
    )
    promoted = _rows(
        _read_json(round_dir / "promoted_feedback.json", "promoted feedback"),
        "promoted feedback",
    )
    promotion_audit = _read_mapping(
        round_dir / "promotion_audit.json", "promotion audit"
    )
    cache_after = core_cache_v2.validate_actual_v3_cache(
        _read_mapping(round_dir / "cache_after_round.json", "cache after round")
    )
    try:
        verified_binding = core_cache_v2.validate_selection_binding(request, binding)
        expected_reveal = core_cache_v2.reveal_v2_cache_after_selection(
            request, binding, snapshot
        )
        expected_plan = actual_adapter.build_miss_only_physical_plan(
            request, binding, reveal, cache_snapshot=snapshot
        )
        expected_admission = online.validate_executor_admission(
            plan,
            logical_request=request,
            selection_binding=binding,
            cache_reveal=reveal,
            cache_snapshot=snapshot,
            contract_sha256=contract_sha256,
        )
    except (OSError, ValueError) as error:
        _fail(f"formal request/cache/plan lineage is invalid: {error}")
    if (
        expected_reveal != reveal
        or expected_plan != plan
        or expected_admission != stored_admission
    ):
        _fail("formal request/cache/plan lineage drift")
    if (
        promotion_audit.get("promoted_row_count") != BATCH_SIZE
        or promotion_audit.get("silent_surrogate_fallback_count") != 0
    ):
        _fail("silent surrogate fallback or incomplete promotion detected")

    selected_by_id = _formal_exact_binding(verified_binding)
    reveal_entries = reveal.get("entries")
    if not isinstance(reveal_entries, list) or len(reveal_entries) != BATCH_SIZE:
        _fail("formal cache reveal must contain four rows")
    reveal_by_id = {
        str(entry.get("candidate_id") or ""): entry
        for entry in reveal_entries
        if isinstance(entry, Mapping)
    }
    historical_by_id = {_row_id(row): row for row in historical}
    promoted_by_id = {_row_id(row): row for row in promoted}
    if not (
        set(selected_by_id)
        == set(reveal_by_id)
        == set(historical_by_id)
        == set(promoted_by_id)
    ):
        _fail("formal selected/cache/feedback identity matrix drift")
    failures = _validate_failure_wrappers(round_dir, request, binding, plan)
    events: list[JSON] = []
    miss_evidence_count = 0
    for event_index, requested in enumerate(request["rows"]):
        candidate_id = _row_id(requested)
        selected = selected_by_id[candidate_id]
        reveal_entry = reveal_by_id[candidate_id]
        historical_row = historical_by_id[candidate_id]
        promoted_row = promoted_by_id[candidate_id]
        disposition = str(reveal_entry.get("disposition") or "")
        if disposition not in {"hit", "miss"}:
            _fail("formal cache disposition is invalid")
        status = str(historical_row.get("terminal_status") or "")
        if status in SUCCESS_STATUSES:
            terminal = _success_wrapper(
                disposition=disposition,
                reveal_entry=reveal_entry,
                cache_after=cache_after,
                selected=selected,
            )
            evidence_kind = (
                "cross_request_exact_hit"
                if disposition == "hit"
                else "actual_v3_success"
            )
            metrics = {
                field: terminal[field]
                for field in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
            }
            if disposition == "miss":
                miss_evidence_count += 1
        elif status in TRUE_FAILURE_STATUSES:
            if disposition != "miss":
                _fail("cached hit cannot be a newly consumed candidate failure")
            failure = failures.get(candidate_id)
            if (
                failure is None
                or failure.get("failure_class") != "candidate"
                or failure.get("consumes_selected_event_budget") is not True
            ):
                _fail("actual-v3 candidate failure evidence is missing")
            evidence_kind = "actual_v3_candidate_failure"
            metrics = {
                field: None
                for field in ("latency_ms", "energy_j", "ap30", "ap50", "ap70")
            }
            miss_evidence_count += 1
        else:
            _fail("formal row is not a terminal selected event")
        if promoted_row.get("feedback_feature_contract") != "actual_feedback_v3":
            _fail("promoted row is not actual-feedback-v3")
        events.append(
            {
                "variant": variant,
                "seed": seed,
                "round_index": round_index,
                "event_index": round_index * BATCH_SIZE + event_index,
                "candidate_id": candidate_id,
                "terminal_status": status,
                "terminal_evidence_kind": evidence_kind,
                "cache_disposition": disposition,
                "q_mode": requested.get("q_mode"),
                "logical_request_sha256": request["measurement_request_sha256"],
                "silent_surrogate_fallback_count": 0,
                **metrics,
            }
        )
    if set(failures) != {
        event["candidate_id"]
        for event in events
        if event["terminal_evidence_kind"] == "actual_v3_candidate_failure"
    }:
        _fail("unselected or unused actual-v3 failure evidence detected")
    return events, miss_evidence_count


def collect_formal_evidence(
    input_root: Path | str, *, repo_root: Path = REPO_ROOT
) -> JSON:
    """Authenticate Task1-5 formal artifacts and collect exactly 192 events."""
    try:
        root = Path(input_root).resolve(strict=True)
    except OSError as error:
        _fail(f"formal input root is unavailable or invalid: {error}")
    try:
        validated = online.validate_formal_v2_root(root, repo_root=repo_root)
    except (OSError, ValueError) as error:
        _fail(f"formal v2 root validation failed: {error}")
    closure = _load_execution_closure(root)
    contract = validated["contract"]
    initial_rows, hv_reference = _load_metric_context(contract)
    events: list[JSON] = []
    miss_evidence_count = 0
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            trajectory = root / "variants" / variant / f"seed_{seed}"
            discovered = {
                path.name for path in trajectory.glob("round_*") if path.is_dir()
            }
            expected = {f"round_{index:02d}" for index in range(ROUND_COUNT)}
            if discovered != expected:
                _fail("formal trajectory must contain exactly four rounds")
            selected_ids: list[str] = []
            for round_index in range(ROUND_COUNT):
                round_events, miss_count = _audit_formal_round(
                    trajectory / f"round_{round_index:02d}",
                    variant=variant,
                    seed=seed,
                    round_index=round_index,
                    contract_sha256=str(contract["contract_sha256"]),
                )
                events.extend(round_events)
                miss_evidence_count += miss_count
                selected_ids.extend(
                    str(event["candidate_id"]) for event in round_events
                )
            if len(selected_ids) != 16 or len(set(selected_ids)) != 16:
                _fail("trajectory selected identities are not 16 unique events")
    miss_count = sum(event["cache_disposition"] == "miss" for event in events)
    evidence = {
        "schema_version": FORMAL_EVIDENCE_SCHEMA,
        "events": events,
        "trajectory_count": TOTAL_TRAJECTORIES,
        "selected_event_count": len(events),
        "miss_count": miss_count,
        "actual_v3_miss_evidence_count": miss_evidence_count,
        "silent_surrogate_fallback_count": sum(
            int(event["silent_surrogate_fallback_count"]) for event in events
        ),
        "ordered_pre_scan_sha256": contract["ordered_pre_scan_sha256"],
        "contract_sha256": contract["contract_sha256"],
        "audits": copy.deepcopy(dict(closure["audits"])),
        "bound_audit_artifacts": closure["bound_audit_artifacts"],
        "formal_v2_gpu_jobs_launched": closure["formal_v2_gpu_jobs_launched"],
        "initial_gold176_rows": initial_rows,
        "hv_reference": list(hv_reference),
    }
    for field in (
        "trajectory_count",
        "selected_event_count",
        "miss_count",
        "actual_v3_miss_evidence_count",
        "silent_surrogate_fallback_count",
        "formal_v2_gpu_jobs_launched",
    ):
        if closure.get(field) != evidence[field]:
            _fail(f"formal execution closure count drift: {field}")
    return evidence


_validate_ready_evidence = statistics_v2.validate_ready_evidence
_trajectory_summaries = statistics_v2.trajectory_summaries
_describe = statistics_v2.describe
_paired_statistics = statistics_v2.paired_statistics
_paper_rows = statistics_v2.paper_rows
_atomic_write = paper_outputs.atomic_write
_json_text = paper_outputs.json_text
_csv_text = paper_outputs.csv_text
_markdown_table = paper_outputs.markdown_table
_validate_output_root = paper_outputs.validate_output_root
_emit_formal_outputs = paper_outputs.emit_formal_outputs


def finalize_v2(
    input_root: Path | str,
    output_root: Path | str,
    *,
    repo_root: Path = REPO_ROOT,
) -> JSON:
    source = Path(input_root).resolve()
    destination = Path(output_root).resolve()
    if source == destination:
        _fail("input and output roots must differ")
    _validate_output_root(destination, FORMAL_OUTPUT_PATHS)
    evidence = collect_formal_evidence(source, repo_root=repo_root)
    events = _validate_ready_evidence(evidence)
    trajectories = _trajectory_summaries(
        events,
        initial_rows=evidence["initial_gold176_rows"],
        reference=tuple(float(value) for value in evidence["hv_reference"]),
    )
    if len(trajectories) != TOTAL_TRAJECTORIES:
        _fail("formal finalization requires exactly 12 trajectories")
    stats = _paired_statistics(trajectories)
    _emit_formal_outputs(
        destination,
        evidence=evidence,
        events=events,
        trajectories=trajectories,
        stats=stats,
    )
    return {
        "paper_ready": True,
        "core_ablation_ready": True,
        "full_gear_s7_ready": False,
        "scanner_component_status": "deferred_important_fix",
        "trajectory_count": TOTAL_TRAJECTORIES,
        "selected_event_count": TOTAL_EVENTS,
        "output_root": str(destination),
    }


finalize = finalize_v2


def emit_non_final_dry_run(
    output_root: Path | str, dry_run_evidence: Mapping[str, Any]
) -> JSON:
    """Emit schema evidence for integration only; it can never claim truth."""
    root = Path(output_root).resolve()
    allowed = ("status/finalization_status.json",)
    _validate_output_root(root, allowed)
    status = {
        "schema_version": SCHEMA_VERSION,
        "paper_ready": False,
        "core_ablation_ready": False,
        "full_gear_s7_ready": False,
        "scanner_component_status": "deferred_important_fix",
        "formal_v2_gpu_jobs_launched": 0,
        "blocking_reason": INTEGRATION_BLOCKING_REASON,
        "dry_run_synthetic_non_measurement": True,
        "actual_v3_hardware_evidence": False,
        "eligible_for_cache_append": False,
        "eligible_for_formal_finalization": False,
        "dry_run_evidence": copy.deepcopy(dict(dry_run_evidence)),
    }
    _atomic_write(root / allowed[0], _json_text(status))
    return {
        key: status[key]
        for key in (
            "paper_ready",
            "core_ablation_ready",
            "full_gear_s7_ready",
            "scanner_component_status",
            "formal_v2_gpu_jobs_launched",
            "blocking_reason",
        )
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = finalize_v2(args.input_root, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
