#!/usr/bin/env python3
"""Freeze Stage7 inputs, audit the scanner, and initialize round zero."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage7 import search_policy_v1 as policy
from framework.stage7.online_component_ablation_v1 import (
    SEEDS,
    audit_single_variable_isolation,
    frozen_experiment_contracts,
    selection_candidate_view,
)


DEFAULT_GOLD176 = REPO_ROOT / (
    "results/stage35_gold144_targeted_supplement_v2_20260714/"
    "final_gold176_v1/gold176_final.json"
)
DEFAULT_GRAPH_FEATURES = REPO_ROOT / (
    "results/stage35_gold144_targeted_supplement_v2_20260714/"
    "final_gold176_v1/graph_features.json"
)
DEFAULT_PROFILES = (
    REPO_ROOT
    / "results/s1_profile_final_v3_20260711/capability_profiles_v3.json"
)
DEFAULT_SOURCE_REGISTRY = REPO_ROOT / (
    "results/stage5_single_target_search_v2_gold176_20260718/"
    "candidate_source_registry_full.json"
)
DEFAULT_STAGE4_CLOSURE = (
    REPO_ROOT
    / "results/stage4_p1_p3_closure_v1_20260716/stage4_p1_p3_closure_audit.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "results/stage7_pyramid_tvm_online_ablation_quick_v1_20260725"
)
FROZEN_SHA256 = policy.FORMAL_INPUT_SHA256
POLICY_SHA256 = policy.POLICY_EVIDENCE_SHA256
VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
    "without_capability_scan",
)


@dataclass(frozen=True)
class FrozenInputPaths:
    gold176: Path
    graph_features: Path
    capability_profiles: Path
    source_registry: Path
    stage4_closure: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("gold176", self.gold176),
            ("graph_features", self.graph_features),
            ("capability_profiles", self.capability_profiles),
            ("source_registry", self.source_registry),
            ("stage4_closure", self.stage4_closure),
        )


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_frozen_bytes(path: Path, content: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite drifted Stage7 checkpoint: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_frozen_json(path: Path, payload: Any) -> None:
    _write_frozen_bytes(path, _json_bytes(payload))


def _write_sha_sidecar(path: Path) -> None:
    _write_frozen_bytes(
        path.with_suffix(".sha256"),
        (_file_sha(path) + "\n").encode("ascii"),
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [copy.deepcopy(dict(row)) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [copy.deepcopy(dict(row)) for row in payload[field]]
    raise ValueError(f"expected a list or object containing {field}")


def _verify_input_paths(
    inputs: FrozenInputPaths,
    expected_sha256: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    audit = {}
    for name, path in inputs.items():
        if not path.is_absolute():
            raise ValueError(f"{name} path must be absolute")
        if not path.is_file():
            raise ValueError(f"frozen input is missing: {path}")
        digest = _file_sha(path)
        expected = expected_sha256.get(name)
        if expected is not None and digest != expected:
            raise ValueError(f"{name} SHA mismatch: {digest}")
        audit[name] = {"path": str(path), "sha256": digest}
    for relative, expected in POLICY_SHA256.items():
        path = REPO_ROOT / relative
        digest = _file_sha(path)
        if digest != expected:
            raise ValueError(f"policy evidence SHA mismatch: {relative}")
        audit[relative] = {"path": str(path), "sha256": digest}
    return audit


def _tvm_profile(profiles_payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profiles = _rows(profiles_payload, "capability_profiles")
    matching = [row for row in profiles if row.get("dispatch_key") == "tvm_auto"]
    if len(matching) != 1:
        raise ValueError("frozen capability profiles require exactly one tvm_auto profile")
    return matching[0], profiles


def _scanner_csv(decisions: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=("row_id", "decision", "reasons"),
        lineterminator="\n",
    )
    writer.writeheader()
    for decision in decisions:
        writer.writerow(
            {
                "row_id": decision["row_id"],
                "decision": decision["decision"],
                "reasons": json.dumps(
                    decision.get("reasons") or [],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return output.getvalue().encode("utf-8")


def _objective_reference(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [
        row
        for row in rows
        if row.get("terminal_status") == "measured_success_gold"
        and all(
            isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
            and math.isfinite(float(row[name]))
            for name in ("latency_ms", "energy_j", "ap70")
        )
    ]
    if not successful:
        raise ValueError("Gold input has no successful finite objective points")
    maxima = {
        "latency_ms": max(float(row["latency_ms"]) for row in successful),
        "energy_j": max(float(row["energy_j"]) for row in successful),
        "negative_ap70": max(-float(row["ap70"]) for row in successful),
    }
    values = {
        name: value + max(abs(value) * 0.05, 1e-9)
        for name, value in maxima.items()
    }
    payload = {
        "schema_version": "stage7_raw_objective_reference_v1",
        "formula": {
            "objectives": [
                "minimize(latency_ms)",
                "minimize(energy_j)",
                "minimize(-ap70)",
            ],
            "reference": "max_successful_gold_objective + max(abs(max)*0.05, 1e-9)",
            "strictly_worse_than_every_successful_gold_point": True,
        },
        "successful_gold_rows": len(successful),
        "values": values,
    }
    return {**payload, "reference_sha256": policy.canonical_sha256(payload)}


def _measured_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row.get("row_id") or row.get("manifest_job_id"))
        for row in rows
        if row.get("model") == "pyramid"
        and row.get("dispatch_key") == "tvm_auto"
        and str(row.get("row_id") or row.get("manifest_job_id") or "")
    }


def freeze_contracts(
    *,
    output_root: Path,
    inputs: FrozenInputPaths,
    expected_sha256: Mapping[str, str] = FROZEN_SHA256,
    enforce_formal_counts: bool = True,
) -> dict[str, Any]:
    if not output_root.is_absolute():
        raise ValueError("output root must be absolute")
    input_audit = _verify_input_paths(inputs, expected_sha256)
    contracts_dir = output_root / "contracts"
    profiles_payload = _load_json(inputs.capability_profiles)
    raw_profile, _ = _tvm_profile(profiles_payload)
    registry = _load_json(inputs.source_registry)
    frozen = policy.prepare_frozen_candidate_pools(
        registry,
        raw_profile=raw_profile,
        measured_row_ids=set(),
    )
    if enforce_formal_counts:
        pyramid_groups = sum(
            row.get("model") == "pyramid" for row in registry.get("groups") or []
        )
        codriving_groups = sum(
            row.get("model") == "codriving" for row in registry.get("groups") or []
        )
        if (pyramid_groups, codriving_groups, frozen["pre_scan_count"]) != (
            343,
            343,
            686,
        ):
            raise ValueError("formal source registry must be 343+343 groups and 686 genomes")
    frozen_inputs = {
        "schema_version": "stage7_frozen_inputs_v1",
        "inputs": input_audit,
        "formal_counts_enforced": enforce_formal_counts,
        "policy_evidence": {
            key: input_audit[key]
            for key in POLICY_SHA256
        },
    }
    frozen_inputs["frozen_inputs_sha256"] = policy.canonical_sha256(frozen_inputs)
    write_frozen_json(contracts_dir / "frozen_inputs.json", frozen_inputs)
    pre_scan_payload = {
        "schema_version": "stage7_pre_scan_candidate_registry_v1",
        "row_count": frozen["pre_scan_count"],
        "rows": frozen["pre_scan_candidates"],
    }
    write_frozen_json(
        contracts_dir / "pre_scan_candidate_registry.json",
        pre_scan_payload,
    )
    _write_sha_sidecar(contracts_dir / "pre_scan_candidate_registry.json")
    rule = frozen["scanner_profile"]["rule_manifest"]
    write_frozen_json(contracts_dir / "scanner_rule_manifest.json", rule)
    _write_sha_sidecar(contracts_dir / "scanner_rule_manifest.json")
    decision_path = contracts_dir / "scanner_decision_by_candidate.csv"
    _write_frozen_bytes(
        decision_path,
        _scanner_csv(frozen["scanner_audit"]["decisions"]),
    )
    _write_sha_sidecar(decision_path)
    scan_audit = {
        "schema_version": "stage7_pre_scan_to_scan_pass_audit_v1",
        "pre_scan_count": frozen["pre_scan_count"],
        "scan_pass_count": frozen["scanner_audit"]["scan_pass_count"],
        "scanner_note": frozen["scanner_audit"]["scanner_note"],
        "rule_manifest_file_sha256": _file_sha(
            contracts_dir / "scanner_rule_manifest.json"
        ),
        "scanner_decision_file_sha256": _file_sha(decision_path),
        "decision_sha256": frozen["scanner_audit"]["decision_sha256"],
    }
    write_frozen_json(
        contracts_dir / "pre_scan_to_scan_pass_audit.json",
        scan_audit,
    )
    full, variants = frozen_experiment_contracts()
    variant_contracts = {"full": full, **variants}
    write_frozen_json(
        contracts_dir / "variant_contracts.json",
        variant_contracts,
    )
    isolation = {
        "schema_version": "stage7_single_variable_isolation_v1",
        "audits": [
            audit_single_variable_isolation(full, variants[name])
            for name in sorted(variants)
        ],
    }
    write_frozen_json(
        output_root / "audits/single_variable_isolation.json",
        isolation,
    )

    # The scanner rule and every candidate decision are durable before this
    # first parse of the historical Gold terminal-label artifact.
    gold_rows = _rows(_load_json(inputs.gold176), "rows")
    measured_all = _measured_ids(gold_rows)
    pre_scan_ids = {
        str(row.get("row_id") or row.get("manifest_job_id"))
        for row in frozen["pre_scan_candidates"]
    }
    measured = measured_all & pre_scan_ids
    measured_outside_pool = measured_all - pre_scan_ids
    filtered_pools = {
        variant: [
            copy.deepcopy(dict(row))
            for row in rows
            if str(row.get("row_id") or row.get("manifest_job_id")) not in measured
        ]
        for variant, rows in frozen["variant_pools"].items()
    }
    pools_payload = {
        "schema_version": "stage7_frozen_candidate_pools_v1",
        "pre_scan_count": frozen["pre_scan_count"],
        "scan_pass_count": frozen["scanner_audit"]["scan_pass_count"],
        "pre_scan_sha256": policy.canonical_sha256(
            frozen["pre_scan_candidates"]
        ),
        "scan_pass_sha256": policy.canonical_sha256(
            frozen["variant_pools"]["full"]
        ),
        "scanner_rule_sha256": _file_sha(
            contracts_dir / "scanner_rule_manifest.json"
        ),
        "scanner_decision_sha256": _file_sha(decision_path),
        "gold_measured_row_ids": sorted(measured),
        "gold_excluded_count": len(measured),
        "gold_measured_ids_outside_frozen_pool": sorted(measured_outside_pool),
        "scan_pass_candidates": frozen["variant_pools"]["full"],
        "variant_pools": filtered_pools,
        "variant_pool_sha256": {
            variant: policy.canonical_sha256(rows)
            for variant, rows in filtered_pools.items()
        },
    }
    pools_payload["frozen_candidate_pools_sha256"] = policy.canonical_sha256(
        pools_payload
    )
    write_frozen_json(
        contracts_dir / "frozen_candidate_pools.json",
        pools_payload,
    )
    freeze_root = {
        "schema_version": "stage7_deterministic_freeze_root_v1",
        "inputs": copy.deepcopy(input_audit),
        "formal_counts_enforced": enforce_formal_counts,
        "formal_input_sha256": (
            copy.deepcopy(dict(FROZEN_SHA256))
            if enforce_formal_counts
            else None
        ),
        "policy_evidence_sha256": copy.deepcopy(dict(POLICY_SHA256)),
        "static_scanner_rule": copy.deepcopy(policy.STATIC_SCANNER_RULE),
        "pre_scan_sha256": pools_payload["pre_scan_sha256"],
        "scan_pass_sha256": pools_payload["scan_pass_sha256"],
        "variant_pool_sha256": copy.deepcopy(
            pools_payload["variant_pool_sha256"]
        ),
    }
    freeze_root["freeze_root_sha256"] = policy.canonical_sha256(freeze_root)
    write_frozen_json(
        contracts_dir / "freeze_state.json",
        freeze_root,
    )
    write_frozen_json(
        contracts_dir / "raw_capability_profile.json",
        raw_profile,
    )
    write_frozen_json(
        contracts_dir / "raw_objective_reference.json",
        _objective_reference(gold_rows),
    )
    result = {
        "schema_version": "stage7_freeze_result_v1",
        "status": "frozen",
        "output_root": str(output_root),
        "pre_scan_count": frozen["pre_scan_count"],
        "scan_pass_count": frozen["scanner_audit"]["scan_pass_count"],
        "gold_excluded_count": len(measured),
        "frozen_inputs_sha256": frozen_inputs["frozen_inputs_sha256"],
    }
    write_frozen_json(output_root / "freeze_state.json", result)
    return result


def _verify_sidecar(path: Path, label: str) -> str:
    sidecar = path.with_suffix(".sha256")
    recorded = sidecar.read_text(encoding="ascii").strip()
    actual = _file_sha(path)
    if recorded != actual:
        raise ValueError(f"{label} SHA drift")
    return actual


def audit_scanner(
    *,
    output_root: Path,
    terminal_evidence_json: Path,
) -> dict[str, Any]:
    contracts = output_root / "contracts"
    rule_path = contracts / "scanner_rule_manifest.json"
    decision_path = contracts / "scanner_decision_by_candidate.csv"
    rule_sha = _verify_sidecar(rule_path, "scanner rule")
    decision_sha = _verify_sidecar(decision_path, "scanner decision")
    decisions = {
        row["row_id"]: row["decision"]
        for row in csv.DictReader(
            io.StringIO(decision_path.read_text(encoding="utf-8"))
        )
    }
    # Historical labels are not parsed until the frozen rule and decisions
    # have both passed their independent byte-SHA checks above.
    terminal_rows = _rows(_load_json(terminal_evidence_json), "rows")
    successes = {
        str(row.get("row_id") or row.get("manifest_job_id"))
        for row in terminal_rows
        if row.get("terminal_status")
        in {"success", "measured_success", "measured_success_gold"}
    }
    candidate_failures = {
        str(row.get("row_id") or row.get("manifest_job_id"))
        for row in terminal_rows
        if row.get("terminal_status")
        in {"feasibility_failure", "numerical_feasibility_failure"}
        or row.get("failure_kind")
        in {
            "backend_failure",
            "build_failure",
            "unsupported_precision",
            "quantization_failure",
            "numerical_failure",
            "candidate_runtime_capability_failure",
        }
    }
    known_success_rejected = sorted(
        row_id for row_id in successes if decisions.get(row_id) == "reject"
    )
    true_failure_passed = sorted(
        row_id
        for row_id in candidate_failures
        if decisions.get(row_id) == "pass"
    )
    unknown = sorted(
        str(row.get("row_id") or row.get("manifest_job_id"))
        for row in terminal_rows
        if str(row.get("row_id") or row.get("manifest_job_id"))
        not in successes | candidate_failures
    )
    admitted = not known_success_rejected and not true_failure_passed
    any_reject = any(value == "reject" for value in decisions.values())
    status = (
        "blocked_scanner_admission_failed"
        if not admitted
        else (
            "scanner_ready_discriminative"
            if any_reject
            else "scanner_ready_all_pass"
        )
    )
    audit = {
        "schema_version": "stage7_scanner_admission_v1",
        "status": status,
        "admission_passed": admitted,
        "scanner_rule_file_sha256": rule_sha,
        "scanner_decision_file_sha256": decision_sha,
        "known_success_rejected": known_success_rejected,
        "true_candidate_capability_failure_passed": true_failure_passed,
        "false_negative_count": len(known_success_rejected),
        "false_positive_count": len(true_failure_passed),
        "unknown_row_ids": unknown,
        "terminal_evidence_path": str(terminal_evidence_json.resolve()),
        "terminal_evidence_sha256": _file_sha(terminal_evidence_json),
    }
    write_frozen_json(output_root / "audits/scanner_admission.json", audit)
    return audit


def _select_with_cache_invariance(
    candidates: Sequence[Mapping[str, Any]],
    selector: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    empty_candidates = [copy.deepcopy(dict(row)) for row in candidates]
    decorated = [
        {
            **copy.deepcopy(dict(row)),
            "cache_membership": index % 2 == 0,
            "cache_key": f"adversarial-exact-membership-{index}",
            "cache_disposition": (
                "exact_candidate_hit" if index % 2 == 0 else "exact_candidate_miss"
            ),
            "cached_labels": {
                "latency_ms": -1_000_000.0 + index,
                "energy_j": 1_000_000.0 - index,
                "ap70": 10_000.0 + index,
            },
        }
        for index, row in enumerate(candidates)
    ]
    empty_view = selection_candidate_view(empty_candidates)
    populated_view = selection_candidate_view(decorated)
    if empty_view != populated_view:
        raise ValueError("cache selection invariance drift")
    empty_result = selector(empty_candidates)
    populated_result = selector(decorated)
    empty_ids = list(
        empty_result.get("acquisition", {}).get("selected_row_ids") or []
    )
    populated_ids = list(
        populated_result.get("acquisition", {}).get("selected_row_ids") or []
    )
    if empty_ids != populated_ids:
        raise ValueError("cache population changed selector output")
    payload = {
        "verdict": "pass",
        "selection_input_sha256": policy.canonical_sha256(empty_view),
        "exact_candidate_membership_sha256": policy.canonical_sha256(
            [
                {
                    "row_id": row.get("row_id")
                    or row.get("manifest_job_id"),
                    "cache_membership": row["cache_membership"],
                    "cache_key": row["cache_key"],
                    "cache_disposition": row["cache_disposition"],
                }
                for row in decorated
            ]
        ),
        "exact_candidate_count": len(decorated),
        "selector_invocation_count": 2,
        "empty_cache_selected_ids": empty_ids,
        "populated_cache_selected_ids": populated_ids,
        "cache_labels_released_to_selector": False,
    }
    audit = {**payload, "audit_sha256": policy.canonical_sha256(payload)}
    return empty_result, audit


def initialize_trajectories(
    *,
    output_root: Path,
    variant: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    policy.validate_frozen_contracts(output_root)
    scanner_audit = _load_json(output_root / "audits/scanner_admission.json")
    if not str(scanner_audit.get("status") or "").startswith("scanner_ready_"):
        raise ValueError("scanner admission has not passed")
    frozen_inputs = _load_json(output_root / "contracts/frozen_inputs.json")
    pools = _load_json(output_root / "contracts/frozen_candidate_pools.json")
    paths = frozen_inputs["inputs"]
    gold_rows = _rows(_load_json(Path(paths["gold176"]["path"])), "rows")
    graphs = _rows(
        _load_json(Path(paths["graph_features"]["path"])),
        "graph_features",
    )
    profiles_payload = _load_json(Path(paths["capability_profiles"]["path"]))
    raw_profile, profiles = _tvm_profile(profiles_payload)
    closure = _load_json(Path(paths["stage4_closure"]["path"]))
    task = policy.build_stage7_task(raw_profile)
    chosen_variants = (variant,) if variant is not None else VARIANTS
    chosen_seeds = (seed,) if seed is not None else SEEDS
    if any(name not in VARIANTS for name in chosen_variants):
        raise ValueError("unknown Stage7 variant")
    if any(value not in SEEDS for value in chosen_seeds):
        raise ValueError("unknown Stage7 seed")
    generated = []
    for name in chosen_variants:
        candidate_pool = pools["variant_pools"][name]
        pool_sha = pools["variant_pool_sha256"][name]
        for frozen_seed in chosen_seeds:
            trajectory_dir = (
                output_root / "variants" / name / f"seed_{frozen_seed}"
            )
            trajectory = policy.build_trajectory_contract(
                variant=name,
                seed=frozen_seed,
                result_root=output_root,
                frozen_input_sha256=frozen_inputs["frozen_inputs_sha256"],
                candidate_pool_sha256=pool_sha,
                pre_scan_sha256=pools["pre_scan_sha256"],
                scan_pass_sha256=pools["scan_pass_sha256"],
                scanner_rule_sha256=pools["scanner_rule_sha256"],
                scanner_decision_sha256=pools["scanner_decision_sha256"],
            )
            def select_round_zero(
                candidates: list[dict[str, Any]],
            ) -> dict[str, Any]:
                return policy.select_stage7_round(
                    variant=name,
                    seed=frozen_seed,
                    round_index=0,
                    task=task,
                    candidate_pool=candidates,
                    initial_rows=gold_rows,
                    initial_graph_features=graphs,
                    capability_profiles=profiles,
                    closure=closure,
                )

            selection, cache_audit = _select_with_cache_invariance(
                candidate_pool,
                select_round_zero,
            )
            request = selection["measurement_request"]
            binding = policy.build_request_binding(
                trajectory,
                request,
                round_index=0,
                trajectory_dir=trajectory_dir,
                a2_frozen_contract=selection.get("a2_frozen"),
            )
            write_frozen_json(
                trajectory_dir / "trajectory_contract.json",
                trajectory,
            )
            round_dir = trajectory_dir / "round_00"
            payloads = {
                "candidate_manifest.json": selection["candidate_manifest"],
                "acquisition.json": selection["acquisition"],
                "measurement_request.json": request,
                "stage7_request_binding.json": binding,
                "cache_selection_invariance_audit.json": cache_audit,
            }
            if "model_bundle_manifest" in selection:
                payloads["model_bundle_manifest.json"] = selection[
                    "model_bundle_manifest"
                ]
            if "predicted_candidates" in selection:
                payloads["predicted_candidates.json"] = selection[
                    "predicted_candidates"
                ]
            if "a2_frozen" in selection:
                payloads["a2_frozen_bundle_manifest.json"] = selection[
                    "a2_frozen"
                ]["bundle_manifest"]
                payloads["a2_frozen_candidate_predictions.json"] = {
                    "rows": selection["a2_frozen"]["candidate_predictions"]
                }
                payloads["a2_frozen_anchor.json"] = selection["a2_frozen"][
                    "anchor"
                ]
                payloads["a2_frozen_contract.json"] = selection["a2_frozen"]
            if "backend_blind_audit" in selection:
                payloads["backend_blind_audit.json"] = selection[
                    "backend_blind_audit"
                ]
            state = policy.build_round_state(
                trajectory,
                binding,
                status="awaiting_terminal_feedback",
                completed_feedback_rows=0,
            )
            payloads["round_state.json"] = state
            for filename, payload in payloads.items():
                write_frozen_json(round_dir / filename, payload)
            generated.append(
                {"variant": name, "seed": frozen_seed, "round_dir": str(round_dir)}
            )
    result = {
        "schema_version": "stage7_initialize_result_v1",
        "trajectory_count": len(generated),
        "trajectories": generated,
    }
    write_frozen_json(output_root / "initialize_state.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    freeze.add_argument("--gold176", type=Path, default=DEFAULT_GOLD176)
    freeze.add_argument("--graph-features", type=Path, default=DEFAULT_GRAPH_FEATURES)
    freeze.add_argument("--capability-profiles", type=Path, default=DEFAULT_PROFILES)
    freeze.add_argument("--source-registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    freeze.add_argument("--stage4-closure", type=Path, default=DEFAULT_STAGE4_CLOSURE)
    audit = subparsers.add_parser("audit-scanner")
    audit.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    audit.add_argument("--terminal-evidence-json", type=Path, required=True)
    initialize = subparsers.add_parser("initialize")
    initialize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    initialize.add_argument("--variant", choices=VARIANTS)
    initialize.add_argument("--seed", type=int, choices=SEEDS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "freeze":
        result = freeze_contracts(
            output_root=args.output_root.resolve(),
            inputs=FrozenInputPaths(
                gold176=args.gold176.resolve(),
                graph_features=args.graph_features.resolve(),
                capability_profiles=args.capability_profiles.resolve(),
                source_registry=args.source_registry.resolve(),
                stage4_closure=args.stage4_closure.resolve(),
            ),
        )
    elif args.command == "audit-scanner":
        result = audit_scanner(
            output_root=args.output_root.resolve(),
            terminal_evidence_json=args.terminal_evidence_json.resolve(),
        )
    else:
        result = initialize_trajectories(
            output_root=args.output_root.resolve(),
            variant=args.variant,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
