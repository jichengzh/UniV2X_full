"""Stage7 search-policy adapters over the frozen Stage5 implementation."""

from __future__ import annotations

import copy
import csv
import hashlib
import inspect
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from framework.stage4.cost_model_selection_v1 import encode_rows
from framework.stage5 import production_search_v1 as stage5_production
from framework.stage5.production_search_v1 import (
    TARGETS,
    _extra_trees,
    _lgbm_huber,
    _predict_model,
    _quantile,
    _stable_calibration_groups,
    fit_production_bundle,
    predict_candidate_rows,
    validate_stage5_contract,
)

from framework.stage5.single_target_search_v2 import (
    SearchTask,
    build_measurement_request,
    build_task_candidate_manifest,
    fit_online_bundle,
    select_task_batch,
    validate_search_task,
)
from framework.stage7.online_component_ablation_v1 import (
    SEEDS,
    a1_select_unselected,
    derive_scanner_profile,
    frozen_experiment_contracts,
    project_a2_feedback,
    scan_candidate_pool,
    selection_candidate_view,
    variant_candidate_pools,
)


SCHEMA_VERSION = "stage7_search_policy_v1"
STATIC_SCANNER_RULE = {
    "width_alignment": [8, 16, 32],
    "width_min": [16, 32, 64],
    "width_max": [64, 128, 256],
}
STAGE5_PRODUCTION_SOURCE_SHA256 = (
    "6d373c312e236222d7630d91deb8cb122d5c928b464e5d1b60f7070ea33db2e2"
)
FORMAL_INPUT_SHA256 = {
    "gold176": "9880d625e1ac2c5e336a5de3bc1d861072d58e05d4b1bea6c79ef1cd0e93ca19",
    "graph_features": "c5f03e19daba4779cb187f036d7c4bf612d3479a00453149f4eaf213412536cd",
    "capability_profiles": "caac02a50dad5367adcb37b4b57bb0ce23113764ff46faaeda648e221b7c8563",
    "source_registry": "033cb9e38af39009f9919233f03c5ff77894651f6192d292dee501d974493b51",
}
POLICY_EVIDENCE_SHA256 = {
    "framework/stage5/single_target_search_v2.py": (
        "7d3694a82cee4e9f9265471fd32371d1012279a71a6c150bf8443240b870df35"
    ),
    "scripts/stage5_advance_task_round_v2.py": (
        "309a7f3889be01655702d0c97217c32fbbdc4ac7b3cb873937cb89f0dd059eb0"
    ),
    "framework/stage5/production_search_v1.py": (
        STAGE5_PRODUCTION_SOURCE_SHA256
    ),
}
_STAGE5_HELPER_SIGNATURES = {
    "_extra_trees": ("seed",),
    "_lgbm_huber": ("seed",),
    "_quantile": ("alpha", "seed"),
    "_stable_calibration_groups": ("rows", "seed"),
    "_predict_model": ("model", "matrix"),
}
_A2_BINDING_FIELDS = (
    "a2_frozen_contract_sha256",
    "a2_bundle_sha256",
    "a2_prediction_view_sha256",
    "a2_training_view_sha256",
    "a2_graph_feature_view_sha256",
    "a2_anchor_sha256",
)


@dataclass(frozen=True)
class BackendBlindBundle:
    manifest: Mapping[str, Any]
    full_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    graph_feature_names: tuple[str, ...]
    value_heads: Mapping[str, Any]
    interval_heads: Mapping[str, tuple[Any, Any, Any]]
    conformal_corrections: Mapping[str, Mapping[str, float]]
    model_anchors: Mapping[str, float]


@dataclass(frozen=True)
class V2PreScanSelection:
    """V2 selector output plus the observable single-variable audit."""

    selection: Mapping[str, Any]
    audit: Mapping[str, Any]


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _deterministic_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _deterministic_scanner_csv(
    decisions: Sequence[Mapping[str, Any]],
) -> bytes:
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


def _validate_embedded_sha(
    payload: Mapping[str, Any], field: str, label: str
) -> str:
    copied = copy.deepcopy(dict(payload))
    recorded = copied.pop(field, None)
    if recorded != canonical_sha256(copied):
        raise ValueError(f"{label} SHA drift")
    return str(recorded)


def _validate_file_sidecar(path: Path, label: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    recorded = path.with_suffix(".sha256").read_text(encoding="ascii").strip()
    if recorded != digest:
        raise ValueError(f"{label} sidecar SHA drift")
    return digest


def _row_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in rows
    ]


def validate_frozen_contracts(output_root: str | Path) -> dict[str, Any]:
    """Rehash every frozen input and derived candidate view before consumption."""
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("output root must be absolute")
    contracts = root / "contracts"
    frozen_inputs = _load_json_file(contracts / "frozen_inputs.json")
    frozen_inputs_sha = _validate_embedded_sha(
        frozen_inputs, "frozen_inputs_sha256", "frozen inputs manifest"
    )
    input_records = frozen_inputs.get("inputs")
    if not isinstance(input_records, Mapping):
        raise ValueError("frozen inputs manifest lacks input records")
    freeze_root = _load_json_file(contracts / "freeze_state.json")
    _validate_embedded_sha(
        freeze_root,
        "freeze_root_sha256",
        "deterministic freeze root",
    )
    if freeze_root.get("inputs") != input_records:
        raise ValueError("freeze root input manifest drift")
    if (
        freeze_root.get("formal_counts_enforced")
        != frozen_inputs.get("formal_counts_enforced")
    ):
        raise ValueError("freeze root formal-count mode drift")
    if freeze_root.get("static_scanner_rule") != STATIC_SCANNER_RULE:
        raise ValueError("freeze root scanner rule drift")
    if freeze_root.get("policy_evidence_sha256") != POLICY_EVIDENCE_SHA256:
        raise ValueError("freeze root Stage5 policy SHA drift")
    if frozen_inputs.get("formal_counts_enforced") is True:
        if freeze_root.get("formal_input_sha256") != FORMAL_INPUT_SHA256:
            raise ValueError("formal input root-of-trust drift")
        for name, digest in FORMAL_INPUT_SHA256.items():
            if input_records.get(name, {}).get("sha256") != digest:
                raise ValueError(f"{name} formal root-of-trust SHA drift")
    elif freeze_root.get("formal_input_sha256") is not None:
        raise ValueError("non-formal freeze root unexpectedly claims formal inputs")
    for relative, digest in POLICY_EVIDENCE_SHA256.items():
        record = input_records.get(relative)
        if not isinstance(record, Mapping) or record.get("sha256") != digest:
            raise ValueError(f"{relative} Stage5 policy root drift")
    expected_frozen_inputs = {
        "schema_version": "stage7_frozen_inputs_v1",
        "inputs": copy.deepcopy(dict(input_records)),
        "formal_counts_enforced": frozen_inputs["formal_counts_enforced"],
        "policy_evidence": {
            key: copy.deepcopy(dict(input_records[key]))
            for key in POLICY_EVIDENCE_SHA256
        },
    }
    expected_frozen_inputs["frozen_inputs_sha256"] = canonical_sha256(
        expected_frozen_inputs
    )
    if (
        frozen_inputs != expected_frozen_inputs
        or (contracts / "frozen_inputs.json").read_bytes()
        != _deterministic_json_bytes(expected_frozen_inputs)
    ):
        raise ValueError("frozen inputs deterministic bytes drift")
    for name, source in input_records.items():
        if not isinstance(source, Mapping):
            raise ValueError(f"{name} frozen input record is invalid")
        path = Path(str(source.get("path") or ""))
        expected = str(source.get("sha256") or "")
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"{name} frozen input path drift")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{name} file SHA drift")

    pre_scan_path = contracts / "pre_scan_candidate_registry.json"
    _validate_file_sidecar(pre_scan_path, "pre-scan registry")
    pre_scan = _load_json_file(pre_scan_path)
    pre_scan_rows = pre_scan.get("rows")
    if not isinstance(pre_scan_rows, list):
        raise ValueError("pre-scan registry rows are invalid")
    if pre_scan.get("row_count") != len(pre_scan_rows):
        raise ValueError("pre-scan registry row count drift")
    pre_ids = _row_ids(pre_scan_rows)
    if any(not row_id for row_id in pre_ids) or len(set(pre_ids)) != len(pre_ids):
        raise ValueError("pre-scan identities drift")

    rule_path = contracts / "scanner_rule_manifest.json"
    rule_file_sha = _validate_file_sidecar(rule_path, "scanner rule")
    decision_path = contracts / "scanner_decision_by_candidate.csv"
    decision_file_sha = _validate_file_sidecar(
        decision_path, "scanner decision"
    )
    decisions = list(
        csv.DictReader(io.StringIO(decision_path.read_text(encoding="utf-8")))
    )
    decision_ids = [str(row.get("row_id") or "") for row in decisions]
    if decision_ids != pre_ids:
        raise ValueError("scanner decision row order/identity drift")
    passed = {
        str(row["row_id"])
        for row in decisions
        if row.get("decision") == "pass"
    }
    if any(row.get("decision") not in {"pass", "reject"} for row in decisions):
        raise ValueError("scanner decision value drift")

    pools = _load_json_file(contracts / "frozen_candidate_pools.json")
    pools_sha = _validate_embedded_sha(
        pools, "frozen_candidate_pools_sha256", "frozen candidate pools"
    )
    scan_pass = pools.get("scan_pass_candidates")
    variant_pools = pools.get("variant_pools")
    pool_shas = pools.get("variant_pool_sha256")
    if (
        not isinstance(scan_pass, list)
        or not isinstance(variant_pools, Mapping)
        or not isinstance(pool_shas, Mapping)
    ):
        raise ValueError("frozen candidate pools structure drift")
    if pools.get("pre_scan_count") != len(pre_scan_rows):
        raise ValueError("frozen candidate pools pre-scan count drift")
    if pools.get("scan_pass_count") != len(scan_pass):
        raise ValueError("frozen candidate pools scan-pass count drift")
    if pools.get("pre_scan_sha256") != canonical_sha256(pre_scan_rows):
        raise ValueError("pre-scan rows SHA drift")
    if pools.get("scan_pass_sha256") != canonical_sha256(scan_pass):
        raise ValueError("scan-pass rows SHA drift")
    if pools.get("scanner_rule_sha256") != rule_file_sha:
        raise ValueError("scanner rule file SHA drift")
    if pools.get("scanner_decision_sha256") != decision_file_sha:
        raise ValueError("scanner decision file SHA drift")
    if set(_row_ids(scan_pass)) != passed:
        raise ValueError("scan-pass rows drift from scanner decisions")
    measured = set(map(str, pools.get("gold_measured_row_ids") or []))
    pre_by_id = dict(zip(pre_ids, pre_scan_rows))
    scan_by_id = dict(zip(_row_ids(scan_pass), scan_pass))
    expected_pool_ids = {
        "full": set(scan_by_id) - measured,
        "without_surrogate": set(scan_by_id) - measured,
        "without_measured_feedback": set(scan_by_id) - measured,
        "backend_blind": set(scan_by_id) - measured,
        "without_capability_scan": set(pre_by_id) - measured,
    }
    counts = {}
    for variant, expected_ids in expected_pool_ids.items():
        rows = variant_pools.get(variant)
        if not isinstance(rows, list):
            raise ValueError(f"{variant} candidate pool structure drift")
        actual_ids = _row_ids(rows)
        expected_order = [
            row_id
            for row_id in (
                pre_ids
                if variant == "without_capability_scan"
                else _row_ids(scan_pass)
            )
            if row_id in expected_ids
        ]
        if actual_ids != expected_order:
            raise ValueError(f"{variant} ordered deterministic pool drift")
        if pool_shas.get(variant) != canonical_sha256(rows):
            raise ValueError(f"{variant} candidate pool rows SHA drift")
        counts[variant] = len(rows)

    profile_payload = _load_json_file(
        Path(str(input_records["capability_profiles"]["path"]))
    )
    profiles = (
        profile_payload
        if isinstance(profile_payload, list)
        else profile_payload.get("capability_profiles")
    )
    if not isinstance(profiles, list):
        raise ValueError("capability profile root input structure drift")
    tvm_profiles = [
        row for row in profiles if row.get("dispatch_key") == "tvm_auto"
    ]
    if len(tvm_profiles) != 1:
        raise ValueError("deterministic rebuild requires one tvm_auto profile")
    source_registry = _load_json_file(
        Path(str(input_records["source_registry"]["path"]))
    )
    rebuilt = prepare_frozen_candidate_pools(
        source_registry,
        raw_profile=tvm_profiles[0],
        measured_row_ids=set(),
    )
    if rebuilt["pre_scan_candidates"] != pre_scan_rows:
        raise ValueError("pre-scan ordered deterministic rows drift")
    expected_pre_scan_payload = {
        "schema_version": "stage7_pre_scan_candidate_registry_v1",
        "row_count": len(rebuilt["pre_scan_candidates"]),
        "rows": rebuilt["pre_scan_candidates"],
    }
    if pre_scan_path.read_bytes() != _deterministic_json_bytes(
        expected_pre_scan_payload
    ):
        raise ValueError("pre-scan deterministic bytes drift")
    expected_rule = rebuilt["scanner_profile"]["rule_manifest"]
    stored_rule = _load_json_file(rule_path)
    if (
        stored_rule != expected_rule
        or rule_path.read_bytes() != _deterministic_json_bytes(expected_rule)
    ):
        raise ValueError("scanner rule deterministic rebuild drift")
    expected_decisions = [
        {
            "row_id": str(row["row_id"]),
            "decision": str(row["decision"]),
            "reasons": json.dumps(
                row.get("reasons") or [],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        for row in rebuilt["scanner_audit"]["decisions"]
    ]
    if decisions != expected_decisions:
        raise ValueError("scanner decision ordered deterministic rebuild drift")
    if decision_path.read_bytes() != _deterministic_scanner_csv(
        rebuilt["scanner_audit"]["decisions"]
    ):
        raise ValueError("scanner decision deterministic bytes drift")
    expected_scan_pass = rebuilt["variant_pools"]["full"]
    if scan_pass != expected_scan_pass:
        raise ValueError("scan-pass ordered deterministic rows drift")
    gold_payload = _load_json_file(
        Path(str(input_records["gold176"]["path"]))
    )
    gold_rows = (
        gold_payload
        if isinstance(gold_payload, list)
        else gold_payload.get("rows")
    )
    if not isinstance(gold_rows, list):
        raise ValueError("Gold root input structure drift")
    measured_all = {
        str(row.get("row_id") or row.get("manifest_job_id"))
        for row in gold_rows
        if row.get("model") == "pyramid"
        and row.get("dispatch_key") == "tvm_auto"
        and str(row.get("row_id") or row.get("manifest_job_id") or "")
    }
    expected_measured = measured_all & set(pre_ids)
    if measured != expected_measured:
        raise ValueError("Gold measured identity deterministic rebuild drift")
    if pools.get("gold_measured_ids_outside_frozen_pool") != sorted(
        measured_all - set(pre_ids)
    ):
        raise ValueError("Gold outside-pool identity deterministic rebuild drift")
    for variant, source_rows in rebuilt["variant_pools"].items():
        expected_rows = [
            copy.deepcopy(dict(row))
            for row in source_rows
            if str(row.get("row_id") or row.get("manifest_job_id"))
            not in expected_measured
        ]
        if variant_pools[variant] != expected_rows:
            raise ValueError(f"{variant} ordered deterministic pool drift")
    expected_variant_pools = {
        variant: [
            copy.deepcopy(dict(row))
            for row in source_rows
            if str(row.get("row_id") or row.get("manifest_job_id"))
            not in expected_measured
        ]
        for variant, source_rows in rebuilt["variant_pools"].items()
    }
    expected_pools = {
        "schema_version": "stage7_frozen_candidate_pools_v1",
        "pre_scan_count": len(rebuilt["pre_scan_candidates"]),
        "scan_pass_count": rebuilt["scanner_audit"]["scan_pass_count"],
        "pre_scan_sha256": canonical_sha256(
            rebuilt["pre_scan_candidates"]
        ),
        "scan_pass_sha256": canonical_sha256(expected_scan_pass),
        "scanner_rule_sha256": rule_file_sha,
        "scanner_decision_sha256": decision_file_sha,
        "gold_measured_row_ids": sorted(expected_measured),
        "gold_excluded_count": len(expected_measured),
        "gold_measured_ids_outside_frozen_pool": sorted(
            measured_all - set(pre_ids)
        ),
        "scan_pass_candidates": expected_scan_pass,
        "variant_pools": expected_variant_pools,
        "variant_pool_sha256": {
            variant: canonical_sha256(rows)
            for variant, rows in expected_variant_pools.items()
        },
    }
    expected_pools["frozen_candidate_pools_sha256"] = canonical_sha256(
        expected_pools
    )
    pools_path = contracts / "frozen_candidate_pools.json"
    if (
        pools != expected_pools
        or pools_path.read_bytes() != _deterministic_json_bytes(expected_pools)
    ):
        raise ValueError("frozen candidate pools deterministic bytes drift")
    expected_root_hashes = {
        "pre_scan_sha256": canonical_sha256(pre_scan_rows),
        "scan_pass_sha256": canonical_sha256(expected_scan_pass),
        "variant_pool_sha256": {
            variant: canonical_sha256(rows)
            for variant, rows in variant_pools.items()
        },
    }
    if any(
        freeze_root.get(field) != value
        for field, value in expected_root_hashes.items()
    ):
        raise ValueError("deterministic freeze root derivation drift")
    expected_freeze_root = {
        "schema_version": "stage7_deterministic_freeze_root_v1",
        "inputs": copy.deepcopy(dict(input_records)),
        "formal_counts_enforced": frozen_inputs["formal_counts_enforced"],
        "formal_input_sha256": (
            copy.deepcopy(FORMAL_INPUT_SHA256)
            if frozen_inputs["formal_counts_enforced"]
            else None
        ),
        "policy_evidence_sha256": copy.deepcopy(POLICY_EVIDENCE_SHA256),
        "static_scanner_rule": copy.deepcopy(STATIC_SCANNER_RULE),
        **expected_root_hashes,
    }
    expected_freeze_root["freeze_root_sha256"] = canonical_sha256(
        expected_freeze_root
    )
    freeze_root_path = contracts / "freeze_state.json"
    if (
        freeze_root != expected_freeze_root
        or freeze_root_path.read_bytes()
        != _deterministic_json_bytes(expected_freeze_root)
    ):
        raise ValueError("deterministic freeze root bytes drift")
    return {
        "verdict": "pass",
        "frozen_inputs_sha256": frozen_inputs_sha,
        "frozen_candidate_pools_sha256": pools_sha,
        "pre_scan_count": len(pre_scan_rows),
        "scan_pass_count": len(scan_pass),
        "variant_pool_counts": counts,
    }


def build_stage7_task(raw_profile: Mapping[str, Any]) -> SearchTask:
    task = SearchTask(
        task_id="S7-PYR-TVM",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=copy.deepcopy(dict(raw_profile)),
        sample_budget=16,
        batch_size=4,
        round_count=4,
    )
    validate_search_task(task)
    return task


def prepare_frozen_candidate_pools(
    source_registry: Mapping[str, Any],
    *,
    raw_profile: Mapping[str, Any],
    measured_row_ids: set[str],
) -> dict[str, Any]:
    """Scan every Pyramid genome before excluding already observed identities."""
    task = build_stage7_task(raw_profile)
    pre_scan_manifest = build_task_candidate_manifest(
        source_registry,
        task=task,
        measured_row_ids=set(),
    )
    pre_scan = [copy.deepcopy(dict(row)) for row in pre_scan_manifest["rows"]]
    scanner_profile = derive_scanner_profile(raw_profile, STATIC_SCANNER_RULE)
    scanner_audit = scan_candidate_pool(
        pre_scan,
        capability_profile=scanner_profile,
    )
    projected_pools = variant_candidate_pools(pre_scan, scanner_audit)
    rows_by_id = {
        str(row.get("row_id") or row.get("manifest_job_id")): row
        for row in pre_scan
    }
    pools = {
        variant: [
            copy.deepcopy(rows_by_id[str(row.get("row_id") or row.get("manifest_job_id"))])
            for row in projected_rows
        ]
        for variant, projected_rows in projected_pools.items()
    }
    selectable_pools = {
        variant: [
            copy.deepcopy(dict(row))
            for row in rows
            if str(row.get("row_id") or row.get("manifest_job_id") or "")
            not in measured_row_ids
        ]
        for variant, rows in pools.items()
    }
    return {
        "schema_version": SCHEMA_VERSION + "_frozen_candidate_pools",
        "task_contract": validate_search_task(task),
        "scanner_profile": scanner_profile,
        "pre_scan_manifest": pre_scan_manifest,
        "pre_scan_candidates": pre_scan,
        "pre_scan_count": len(pre_scan),
        "scanner_audit": scanner_audit,
        "selectable_pre_scan_count": len(
            selectable_pools["without_capability_scan"]
        ),
        "measured_row_ids": sorted(measured_row_ids),
        "variant_pools": selectable_pools,
    }


def _variant_contract(variant: str) -> dict[str, Any]:
    full, variants = frozen_experiment_contracts()
    contracts = {"full": full, **variants}
    if variant not in contracts:
        raise ValueError("unknown Stage7 variant")
    return copy.deepcopy(contracts[variant])


def build_trajectory_contract(
    *,
    variant: str,
    seed: int,
    result_root: str | Path,
    frozen_input_sha256: str,
    candidate_pool_sha256: str,
    pre_scan_sha256: str,
    scan_pass_sha256: str,
    scanner_rule_sha256: str,
    scanner_decision_sha256: str,
) -> dict[str, Any]:
    root = Path(result_root)
    if not root.is_absolute():
        raise ValueError("result_root must be absolute")
    if seed not in SEEDS:
        raise ValueError("seed is not one of the frozen Stage7 seeds")
    bound_digests = {
        "frozen_input_sha256": frozen_input_sha256,
        "candidate_pool_sha256": candidate_pool_sha256,
        "pre_scan_sha256": pre_scan_sha256,
        "scan_pass_sha256": scan_pass_sha256,
        "scanner_rule_sha256": scanner_rule_sha256,
        "scanner_decision_sha256": scanner_decision_sha256,
    }
    if any(not _is_sha256(value) for value in bound_digests.values()):
        raise ValueError("trajectory inputs require SHA256 digests")
    experiment = _variant_contract(variant)
    payload = {
        "schema_version": SCHEMA_VERSION + "_trajectory_contract",
        "task_id": "S7-PYR-TVM",
        "variant": variant,
        "seed": seed,
        "policy_name": experiment["policy_name"],
        "result_root": str(root),
        "trajectory_dir": str(root / "variants" / variant / f"seed_{seed}"),
        **bound_digests,
    }
    return {**payload, "trajectory_contract_sha256": canonical_sha256(payload)}


def build_request_binding(
    trajectory_contract: Mapping[str, Any],
    measurement_request: Mapping[str, Any],
    *,
    round_index: int,
    trajectory_dir: str | Path,
    previous_released_feedback_sha256: str | None = None,
    previous_request_binding_sha256: str | None = None,
    prior_chain_head_sha256: str | None = None,
    a2_frozen_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_dir = Path(str(trajectory_contract.get("trajectory_dir") or ""))
    actual_dir = Path(trajectory_dir)
    if actual_dir != expected_dir:
        raise ValueError("trajectory directory drift")
    if round_index not in range(4):
        raise ValueError("round index outside frozen budget")
    request_sha = measurement_request.get("measurement_request_sha256")
    if not _is_sha256(request_sha):
        raise ValueError("measurement request SHA missing")
    rows = measurement_request.get("rows")
    if not isinstance(rows, list):
        raise ValueError("measurement request rows missing")
    row_ids = [
        str(row.get("row_id") or row.get("manifest_job_id") or "")
        for row in rows
    ]
    if len(row_ids) != 4 or any(not row_id for row_id in row_ids):
        raise ValueError("measurement request must bind exactly four rows")
    if len(set(row_ids)) != 4:
        raise ValueError("measurement request row identities must be unique")
    if round_index == 0:
        if previous_released_feedback_sha256 is not None:
            raise ValueError("round zero cannot bind previous released feedback")
        if previous_request_binding_sha256 is not None:
            raise ValueError("round zero cannot bind a previous request")
        if prior_chain_head_sha256 is None:
            prior_chain_head_sha256 = trajectory_contract.get(
                "trajectory_contract_sha256"
            )
    elif (
        not _is_sha256(previous_released_feedback_sha256)
        or not _is_sha256(previous_request_binding_sha256)
        or not _is_sha256(prior_chain_head_sha256)
    ):
        raise ValueError("later rounds require a complete prior chain")
    if not _is_sha256(prior_chain_head_sha256):
        raise ValueError("prior chain head SHA missing")
    payload = {
        "schema_version": SCHEMA_VERSION + "_request_binding",
        "task_id": trajectory_contract["task_id"],
        "variant": trajectory_contract["variant"],
        "seed": trajectory_contract["seed"],
        "round_index": round_index,
        "trajectory_dir": str(actual_dir),
        "trajectory_contract_sha256": trajectory_contract[
            "trajectory_contract_sha256"
        ],
        "measurement_request_sha256": request_sha,
        "selected_row_ids": row_ids,
        "selected_row_ids_sha256": canonical_sha256(row_ids),
        "previous_released_feedback_sha256": previous_released_feedback_sha256,
        "previous_request_binding_sha256": previous_request_binding_sha256,
        "prior_chain_head_sha256": prior_chain_head_sha256,
    }
    if trajectory_contract["variant"] == "without_measured_feedback":
        if a2_frozen_contract is None:
            raise ValueError("A2 request binding requires its frozen contract")
        payload.update(a2_frozen_binding(a2_frozen_contract))
    elif a2_frozen_contract is not None:
        raise ValueError("non-A2 request binding cannot bind an A2 contract")
    return {**payload, "request_binding_sha256": canonical_sha256(payload)}


def build_round_chain_head(
    trajectory_contract: Mapping[str, Any],
    request_binding: Mapping[str, Any],
) -> str:
    payload = {
        "trajectory_contract_sha256": trajectory_contract.get(
            "trajectory_contract_sha256"
        ),
        "round_index": request_binding.get("round_index"),
        "measurement_request_sha256": request_binding.get(
            "measurement_request_sha256"
        ),
        "request_binding_sha256": request_binding.get(
            "request_binding_sha256"
        ),
        "previous_released_feedback_sha256": request_binding.get(
            "previous_released_feedback_sha256"
        ),
        "previous_request_binding_sha256": request_binding.get(
            "previous_request_binding_sha256"
        ),
        "prior_chain_head_sha256": request_binding.get(
            "prior_chain_head_sha256"
        ),
        "a2_frozen_binding": {
            field: request_binding.get(field)
            for field in _A2_BINDING_FIELDS
        },
    }
    required = (
        payload["trajectory_contract_sha256"],
        payload["measurement_request_sha256"],
        payload["request_binding_sha256"],
        payload["prior_chain_head_sha256"],
    )
    if any(not _is_sha256(value) for value in required):
        raise ValueError("round chain inputs require SHA256 digests")
    return canonical_sha256(payload)


def build_round_state(
    trajectory_contract: Mapping[str, Any],
    request_binding: Mapping[str, Any],
    *,
    status: str,
    completed_feedback_rows: int,
) -> dict[str, Any]:
    selected_ids = list(request_binding.get("selected_row_ids") or [])
    round_index = int(request_binding.get("round_index", -1))
    payload = {
        "schema_version": "stage7_round_state_v1",
        "variant": trajectory_contract["variant"],
        "seed": trajectory_contract["seed"],
        "round_index": round_index,
        "status": status,
        "completed_feedback_rows": completed_feedback_rows,
        "selected_event_count_after_request": 4 * (round_index + 1),
        "selected_row_ids": selected_ids,
        "measurement_request_sha256": request_binding[
            "measurement_request_sha256"
        ],
        "request_binding_sha256": request_binding["request_binding_sha256"],
        "previous_released_feedback_sha256": request_binding[
            "previous_released_feedback_sha256"
        ],
        "previous_request_binding_sha256": request_binding[
            "previous_request_binding_sha256"
        ],
        "prior_chain_head_sha256": request_binding["prior_chain_head_sha256"],
        "current_chain_head_sha256": build_round_chain_head(
            trajectory_contract,
            request_binding,
        ),
    }
    if trajectory_contract["variant"] == "without_measured_feedback":
        payload.update(
            {
                field: request_binding.get(field)
                for field in _A2_BINDING_FIELDS
            }
        )
    return {**payload, "round_state_sha256": canonical_sha256(payload)}


def validate_request_binding(
    binding: Mapping[str, Any],
    measurement_request: Mapping[str, Any],
    trajectory_contract: Mapping[str, Any],
) -> dict[str, Any]:
    recorded = dict(binding)
    binding_sha = recorded.pop("request_binding_sha256", None)
    if binding_sha != canonical_sha256(recorded):
        raise ValueError("request binding SHA drift")
    if (
        binding.get("measurement_request_sha256")
        != measurement_request.get("measurement_request_sha256")
    ):
        raise ValueError("measurement request SHA drift")
    if (
        binding.get("trajectory_contract_sha256")
        != trajectory_contract.get("trajectory_contract_sha256")
    ):
        raise ValueError("trajectory contract SHA drift")
    expected_identity = {
        "task_id": trajectory_contract.get("task_id"),
        "variant": trajectory_contract.get("variant"),
        "seed": trajectory_contract.get("seed"),
        "trajectory_dir": trajectory_contract.get("trajectory_dir"),
        "round_index": measurement_request.get("round_index"),
    }
    if any(binding.get(field) != value for field, value in expected_identity.items()):
        raise ValueError("request binding identity drift")
    request_row_ids = _row_ids(measurement_request.get("rows") or [])
    if (
        binding.get("selected_row_ids") != request_row_ids
        or binding.get("selected_row_ids_sha256")
        != canonical_sha256(request_row_ids)
    ):
        raise ValueError("request binding selected-row identity drift")
    if trajectory_contract.get("variant") == "without_measured_feedback":
        if any(not _is_sha256(binding.get(field)) for field in _A2_BINDING_FIELDS):
            raise ValueError("A2 request binding SHA identity drift")
    return {"verdict": "pass", "request_binding_sha256": binding_sha}


def audit_stage5_model_api(
    *,
    expected_source_sha256: str = STAGE5_PRODUCTION_SOURCE_SHA256,
) -> dict[str, Any]:
    source_path = Path(stage5_production.__file__).resolve()
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    signatures = {}
    for name, expected_parameters in _STAGE5_HELPER_SIGNATURES.items():
        helper = getattr(stage5_production, name, None)
        if helper is None:
            raise ValueError(f"blocked_backend_blind_model_api_drift: missing {name}")
        parameters = tuple(inspect.signature(helper).parameters)
        signatures[name] = str(inspect.signature(helper))
        if parameters != expected_parameters:
            raise ValueError(
                f"blocked_backend_blind_model_api_drift: {name} model API drift"
            )
    if source_sha != expected_source_sha256:
        raise ValueError("blocked_backend_blind_model_api_drift: source model API drift")
    payload = {
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "helper_signatures": signatures,
    }
    return {**payload, "audit_sha256": canonical_sha256(payload), "verdict": "pass"}


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _blind_encoded_rows(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    encoded = encode_rows(rows, graph_features, capability_profiles)
    retained = tuple(
        index
        for index, name in enumerate(encoded.feature_names)
        if not name.startswith(("cap:", "cap_x_q:"))
    )
    blind_names = tuple(encoded.feature_names[index] for index in retained)
    if any(name.startswith(("cap:", "cap_x_q:")) for name in blind_names):
        raise ValueError("backend-blind feature leakage")
    return encoded.matrix[:, retained], encoded.feature_names, blind_names


def _successful_value_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = [
        copy.deepcopy(dict(row))
        for row in rows
        if row.get("terminal_status") == "measured_success_gold"
        and all(_finite(row.get(target)) for target in TARGETS)
    ]
    if len(selected) < 4:
        raise ValueError("backend-blind fit requires measured successful rows")
    return selected


def fit_backend_blind_bundle(
    rows: Sequence[Mapping[str, Any]],
    graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    *,
    closure: Mapping[str, Any] | None,
    seed: int,
    training_view_policy: str = "inherit_stage4_feedback",
) -> BackendBlindBundle:
    api_audit = audit_stage5_model_api()
    if closure is not None:
        validate_stage5_contract(
            closure,
            rows,
            graph_features,
            capability_profiles,
            training_view_policy=training_view_policy,
        )
    source_rows = _successful_value_rows(rows)
    matrix, full_names, blind_names = _blind_encoded_rows(
        source_rows, graph_features, capability_profiles
    )
    anchors = {
        model: float(
            np.median(
                [
                    float(row["ap70"])
                    for row in source_rows
                    if str(row["model"]) == model
                ]
            )
        )
        for model in sorted({str(row["model"]) for row in source_rows})
    }
    value_heads: dict[str, Any] = {}
    for target in ("latency_ms", "energy_j"):
        model = _extra_trees(seed)
        model.fit(
            matrix,
            np.log1p([float(row[target]) for row in source_rows]),
        )
        value_heads[target] = model
    ap_model = _lgbm_huber(seed)
    ap_model.fit(
        matrix,
        np.asarray(
            [
                float(row["ap70"]) - anchors[str(row["model"])]
                for row in source_rows
            ],
            dtype=float,
        ),
    )
    value_heads["ap70"] = ap_model
    calibration_groups = _stable_calibration_groups(source_rows, seed)
    fit_indices = [
        index
        for index, row in enumerate(source_rows)
        if str(row["group_id"]) not in calibration_groups
    ]
    calibration_indices = [
        index
        for index, row in enumerate(source_rows)
        if str(row["group_id"]) in calibration_groups
    ]
    if not fit_indices or not calibration_indices:
        raise ValueError("backend-blind calibration split is empty")
    interval_heads: dict[str, tuple[Any, Any, Any]] = {}
    corrections: dict[str, dict[str, float]] = {}
    for target_index, target in enumerate(TARGETS):
        models = tuple(
            _quantile(alpha, seed + target_index)
            for alpha in (0.05, 0.50, 0.95)
        )
        truth = np.asarray(
            [float(source_rows[index][target]) for index in fit_indices],
            dtype=float,
        )
        for model in models:
            model.fit(matrix[fit_indices], truth)
        interval_heads[target] = models
        predicted = np.sort(
            np.vstack(
                [
                    _predict_model(model, matrix[calibration_indices])
                    for model in models
                ]
            ),
            axis=0,
        )
        scores_by_model: dict[str, list[float]] = {}
        for local_index, row_index in enumerate(calibration_indices):
            row = source_rows[row_index]
            score = max(
                float(predicted[0, local_index]) - float(row[target]),
                float(row[target]) - float(predicted[2, local_index]),
                0.0,
            )
            scores_by_model.setdefault(str(row["model"]), []).append(score)
        corrections[target] = {
            model: float(max(values))
            for model, values in scores_by_model.items()
        }
    full_matrix_sha = canonical_sha256(
        encode_rows(source_rows, graph_features, capability_profiles).matrix.tolist()
    )
    blind_matrix_sha = canonical_sha256(matrix.tolist())
    removed = [
        name
        for name in full_names
        if name.startswith(("cap:", "cap_x_q:"))
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION + "_backend_blind_bundle",
        "seed": seed,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "acquisition_policy": "predicted_frontier_diversity",
        "full_feature_names": list(full_names),
        "feature_names": list(blind_names),
        "removed_names": removed,
        "full_training_matrix_sha256": full_matrix_sha,
        "blind_training_matrix_sha256": blind_matrix_sha,
        "leakage_verdict": "no_forbidden_features",
        "stage5_model_api_audit": api_audit,
        "training_row_count": len(source_rows),
        "calibration_groups": sorted(calibration_groups),
    }
    manifest["bundle_config_sha256"] = canonical_sha256(manifest)
    return BackendBlindBundle(
        manifest=manifest,
        full_feature_names=full_names,
        feature_names=blind_names,
        graph_feature_names=tuple(
            name.removeprefix("graph:")
            for name in blind_names
            if name.startswith("graph:")
        ),
        value_heads=value_heads,
        interval_heads=interval_heads,
        conformal_corrections=corrections,
        model_anchors=anchors,
    )


def _candidate_graphs(
    rows: Sequence[Mapping[str, Any]],
    graph_feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    graphs = []
    observed: set[str] = set()
    for row in rows:
        group_id = str(row["group_id"])
        if group_id in observed:
            continue
        observed.add(group_id)
        source = row.get("graph_features") or {}
        graphs.append(
            {
                "group_id": group_id,
                "model": row["model"],
                "width": list(row["width"]),
                **{
                    name: (
                        float(source.get(name))
                        if _finite(source.get(name))
                        else 0.0
                    )
                    for name in graph_feature_names
                },
            }
        )
    return graphs


def _backend_blind_candidate_matrix_audit(
    bundle: BackendBlindBundle,
    candidates: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graphs = _candidate_graphs(candidates, bundle.graph_feature_names)
    full = encode_rows(candidates, graphs, capability_profiles)
    blind, full_names, blind_names = _blind_encoded_rows(
        candidates, graphs, capability_profiles
    )
    if full_names != bundle.full_feature_names or blind_names != bundle.feature_names:
        raise ValueError("backend-blind candidate schema drift")
    return {
        "full_candidate_matrix_sha256": canonical_sha256(full.matrix.tolist()),
        "blind_candidate_matrix_sha256": canonical_sha256(blind.tolist()),
    }


def predict_backend_blind_rows(
    bundle: BackendBlindBundle,
    candidate_rows: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    audit_stage5_model_api()
    candidates = [copy.deepcopy(dict(row)) for row in candidate_rows]
    graphs = _candidate_graphs(candidates, bundle.graph_feature_names)
    matrix, full_names, blind_names = _blind_encoded_rows(
        candidates, graphs, capability_profiles
    )
    if full_names != bundle.full_feature_names or blind_names != bundle.feature_names:
        raise ValueError("backend-blind candidate feature schema drift")
    value_predictions = {
        target: np.maximum(
            0.0,
            np.expm1(_predict_model(bundle.value_heads[target], matrix)),
        )
        for target in ("latency_ms", "energy_j")
    }
    residual = _predict_model(bundle.value_heads["ap70"], matrix)
    value_predictions["ap70"] = np.asarray(
        [
            float(residual[index])
            + float(bundle.model_anchors[str(row["model"])])
            for index, row in enumerate(candidates)
        ]
    )
    interval_predictions = {
        target: np.sort(
            np.vstack(
                [_predict_model(model, matrix) for model in models]
            ),
            axis=0,
        )
        for target, models in bundle.interval_heads.items()
    }
    result = []
    for index, row in enumerate(candidates):
        model_name = str(row["model"])
        intervals = {}
        for target in TARGETS:
            values = interval_predictions[target]
            correction = float(
                bundle.conformal_corrections[target].get(model_name, 0.0)
            )
            intervals[target] = {
                "lower": float(values[0, index] - correction),
                "median": float(values[1, index]),
                "upper": float(values[2, index] + correction),
            }
        result.append(
            {
                **row,
                "predictions": {
                    target: float(value_predictions[target][index])
                    for target in TARGETS
                },
                "prediction_intervals": intervals,
                "prediction_bundle_sha256": bundle.manifest[
                    "bundle_config_sha256"
                ],
            }
        )
    return result


def _graph_view_with_feedback(
    initial: Sequence[Mapping[str, Any]],
    feedback: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_group = {
        str(row["group_id"]): copy.deepcopy(dict(row))
        for row in initial
    }
    for row in feedback:
        graph = row.get("graph_features")
        if isinstance(graph, Mapping):
            by_group[str(row["group_id"])] = copy.deepcopy(dict(graph))
    return [by_group[group_id] for group_id in sorted(by_group)]


def _candidate_manifest(
    candidate_pool: Sequence[Mapping[str, Any]],
    selected_ids: set[str],
    task: SearchTask,
) -> dict[str, Any]:
    rows = [
        copy.deepcopy(dict(row))
        for row in selection_candidate_view(candidate_pool)
        if str(row.get("row_id") or row.get("manifest_job_id") or "")
        not in selected_ids
    ]
    if len(rows) < task.batch_size:
        raise ValueError("fewer than four unselected candidates remain")
    payload = {
        "schema_version": SCHEMA_VERSION + "_candidate_manifest",
        "task_id": task.task_id,
        "eligible_row_count": len(rows),
        "excluded_selected_row_ids": sorted(selected_ids),
        "rows": rows,
    }
    return {**payload, "candidate_manifest_sha256": canonical_sha256(payload)}


def _a2_frozen_payload(
    bundle_manifest: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION + "_a2_frozen",
        "bundle_manifest": copy.deepcopy(dict(bundle_manifest)),
        "bundle_sha256": str(bundle_manifest["bundle_config_sha256"]),
        "candidate_predictions": copy.deepcopy([dict(row) for row in predictions]),
        "prediction_view_sha256": canonical_sha256(list(predictions)),
        "training_view_sha256": canonical_sha256(list(initial_rows)),
        "graph_feature_view_sha256": canonical_sha256(
            list(initial_graph_features)
        ),
        "anchor": copy.deepcopy(dict(anchors)),
        "anchor_sha256": canonical_sha256(dict(anchors)),
    }
    return {**payload, "frozen_payload_sha256": canonical_sha256(payload)}


def derive_a2_anchor(
    initial_rows: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    stage5_production._validate_complete_groups(initial_rows)
    source_rows, _ = stage5_production._value_training_view(initial_rows)
    return {
        model: float(
            np.median(
                [
                    float(row["ap70"])
                    for row in source_rows
                    if str(row["model"]) == model
                ]
            )
        )
        for model in sorted({str(row["model"]) for row in source_rows})
    }


def a2_frozen_binding(payload: Mapping[str, Any]) -> dict[str, str]:
    mapping = {
        "a2_frozen_contract_sha256": payload.get(
            "frozen_payload_sha256"
        ),
        "a2_bundle_sha256": payload.get("bundle_sha256"),
        "a2_prediction_view_sha256": payload.get(
            "prediction_view_sha256"
        ),
        "a2_training_view_sha256": payload.get("training_view_sha256"),
        "a2_graph_feature_view_sha256": payload.get(
            "graph_feature_view_sha256"
        ),
        "a2_anchor_sha256": payload.get("anchor_sha256"),
    }
    if any(not _is_sha256(value) for value in mapping.values()):
        raise ValueError("A2 frozen binding requires complete SHA256 fields")
    return {key: str(value) for key, value in mapping.items()}


def validate_a2_frozen_contract(
    payload: Mapping[str, Any],
    *,
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    expected_anchor: Mapping[str, Any],
    artifact_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen = copy.deepcopy(dict(payload))
    recorded = frozen.pop("frozen_payload_sha256", None)
    if recorded != canonical_sha256(frozen):
        raise ValueError("A2 frozen payload SHA drift")
    if frozen.get("bundle_sha256") != frozen.get("bundle_manifest", {}).get(
        "bundle_config_sha256"
    ):
        raise ValueError("A2 bundle SHA drift")
    if frozen.get("prediction_view_sha256") != canonical_sha256(
        frozen.get("candidate_predictions") or []
    ):
        raise ValueError("A2 prediction SHA drift")
    if frozen.get("training_view_sha256") != canonical_sha256(
        list(initial_rows)
    ):
        raise ValueError("A2 training view deterministic drift")
    if frozen.get("graph_feature_view_sha256") != canonical_sha256(
        list(initial_graph_features)
    ):
        raise ValueError("A2 graph view deterministic drift")
    if (
        frozen.get("anchor") != dict(expected_anchor)
        or frozen.get("anchor_sha256")
        != canonical_sha256(dict(expected_anchor))
    ):
        raise ValueError("A2 anchor deterministic drift")
    predictions = frozen.get("candidate_predictions")
    if not isinstance(predictions, list):
        raise ValueError("A2 frozen predictions are missing")
    if predictions != list(artifact_predictions):
        raise ValueError("A2 prediction artifact drift")
    prediction_ids = _row_ids(predictions)
    if (
        any(not row_id for row_id in prediction_ids)
        or len(set(prediction_ids)) != len(prediction_ids)
    ):
        raise ValueError("A2 prediction row identity drift")
    a2_frozen_binding(payload)
    return copy.deepcopy(dict(payload))


def _validate_a2_frozen(
    payload: Mapping[str, Any],
    *,
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    artifact_predictions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    return validate_a2_frozen_contract(
        payload,
        initial_rows=initial_rows,
        initial_graph_features=initial_graph_features,
        expected_anchor=(
            derive_a2_anchor(initial_rows)
            if initial_rows
            else dict(payload.get("anchor") or {})
        ),
        artifact_predictions=(
            list(artifact_predictions)
            if artifact_predictions is not None
            else list(payload.get("candidate_predictions") or [])
        ),
    )


def select_stage7_round(
    *,
    variant: str,
    seed: int,
    round_index: int,
    task: SearchTask,
    candidate_pool: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
    selected_ids: set[str] | None = None,
    feedback_rows: Sequence[Mapping[str, Any]] = (),
    a2_frozen: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _variant_contract(variant)
    if seed not in SEEDS or round_index not in range(4):
        raise ValueError("Stage7 seed or round index is outside the frozen contract")
    if validate_search_task(task)["task_id"] != "S7-PYR-TVM":
        raise ValueError("search task is not S7-PYR-TVM")
    selected = set(selected_ids or set())
    manifest = _candidate_manifest(candidate_pool, selected, task)
    eligible = manifest["rows"]
    base = {
        "schema_version": SCHEMA_VERSION + "_round_selection",
        "variant": variant,
        "seed": seed,
        "round_index": round_index,
        "candidate_manifest": manifest,
    }
    if variant == "without_surrogate":
        chosen_ids = a1_select_unselected(
            eligible,
            set(),
            seed=seed,
            batch_size=task.batch_size,
        )
        by_id = {
            str(row.get("row_id") or row.get("manifest_job_id")): row
            for row in eligible
        }
        chosen_rows = [copy.deepcopy(by_id[row_id]) for row_id in chosen_ids]
        acquisition = {
            "schema_version": SCHEMA_VERSION + "_a1_acquisition",
            "policy": "uniform_random_without_replacement",
            "candidate_labels_visible_before_measurement": False,
            "selected_row_count": len(chosen_rows),
            "selected_row_ids": chosen_ids,
            "selected_rows": chosen_rows,
        }
        request = build_measurement_request(
            task=task,
            selected_rows=chosen_rows,
            round_index=round_index,
        )
        return {
            **base,
            "acquisition": acquisition,
            "measurement_request": request,
        }

    measured_rows = [*map(dict, initial_rows), *map(dict, feedback_rows)]
    measured_graphs = _graph_view_with_feedback(
        initial_graph_features, feedback_rows
    )
    if variant == "without_measured_feedback":
        measured_rows = [*map(dict, initial_rows)]
        measured_graphs = [*map(dict, initial_graph_features)]
        if round_index == 0:
            if a2_frozen is not None:
                raise ValueError("A2 round zero cannot accept a prior frozen payload")
            bundle = fit_production_bundle(
                initial_rows,
                initial_graph_features,
                capability_profiles,
                closure,
                seed=seed,
                training_view_policy="initial_coldstart_only",
            )
            frozen_predictions = predict_candidate_rows(
                bundle, candidate_pool, capability_profiles
            )
            anchors = getattr(bundle, "model_anchors", {})
            a2_payload = _a2_frozen_payload(
                bundle.manifest,
                frozen_predictions,
                initial_rows,
                initial_graph_features,
                anchors,
            )
        else:
            if a2_frozen is None:
                raise ValueError("A2 later round requires the frozen payload")
            a2_payload = _validate_a2_frozen(
                a2_frozen,
                initial_rows=initial_rows,
                initial_graph_features=initial_graph_features,
            )
        predicted_by_id = {
            str(row.get("row_id") or row.get("manifest_job_id")): row
            for row in a2_payload["candidate_predictions"]
        }
        predicted = [
            copy.deepcopy(predicted_by_id[str(row["row_id"])])
            for row in eligible
            if str(row["row_id"]) in predicted_by_id
        ]
        bundle_manifest = a2_payload["bundle_manifest"]
    elif variant == "backend_blind":
        if round_index == 0:
            blind_bundle = fit_backend_blind_bundle(
                initial_rows,
                initial_graph_features,
                capability_profiles,
                closure=closure,
                seed=seed,
                training_view_policy="initial_coldstart_only",
            )
            full_bundle = fit_production_bundle(
                initial_rows,
                initial_graph_features,
                capability_profiles,
                closure,
                seed=seed,
                training_view_policy="initial_coldstart_only",
            )
        else:
            blind_bundle = fit_backend_blind_bundle(
                measured_rows,
                measured_graphs,
                capability_profiles,
                closure=None,
                seed=seed + round_index,
            )
            full_bundle = fit_online_bundle(
                measured_rows,
                measured_graphs,
                capability_profiles,
                seed=seed + round_index,
            )
        predicted = predict_backend_blind_rows(
            blind_bundle, eligible, capability_profiles
        )
        full_predictions = predict_candidate_rows(
            full_bundle, eligible, capability_profiles
        )
        full_acquisition = select_task_batch(
            full_predictions,
            measured_rows,
            measured_graphs,
            task=task,
        )
        full_by_id = {
            str(row.get("row_id") or row.get("manifest_job_id")): row
            for row in full_predictions
        }
        blind_comparison = {
            "full_feature_names": list(blind_bundle.full_feature_names),
            "blind_feature_names": list(blind_bundle.feature_names),
            "removed_names": list(blind_bundle.manifest["removed_names"]),
            "full_training_matrix_sha256": blind_bundle.manifest[
                "full_training_matrix_sha256"
            ],
            "blind_training_matrix_sha256": blind_bundle.manifest[
                "blind_training_matrix_sha256"
            ],
            **_backend_blind_candidate_matrix_audit(
                blind_bundle, eligible, capability_profiles
            ),
            "leakage_verdict": blind_bundle.manifest["leakage_verdict"],
            "prediction_deltas": [
                {
                    "row_id": str(
                        row.get("row_id") or row.get("manifest_job_id")
                    ),
                    **{
                        f"{target}_blind_minus_full": (
                            float(row["predictions"][target])
                            - float(
                                full_by_id[
                                    str(
                                        row.get("row_id")
                                        or row.get("manifest_job_id")
                                    )
                                ]["predictions"][target]
                            )
                        )
                        for target in TARGETS
                    },
                }
                for row in predicted
            ],
            "full_selected_row_ids": full_acquisition["selected_row_ids"],
        }
        bundle_manifest = blind_bundle.manifest
    else:
        if round_index == 0:
            bundle = fit_production_bundle(
                initial_rows,
                initial_graph_features,
                capability_profiles,
                closure,
                seed=seed,
                training_view_policy="initial_coldstart_only",
            )
        else:
            bundle = fit_online_bundle(
                measured_rows,
                measured_graphs,
                capability_profiles,
                seed=seed + round_index,
            )
        predicted = predict_candidate_rows(
            bundle, eligible, capability_profiles
        )
        bundle_manifest = bundle.manifest
    acquisition = select_task_batch(
        predicted,
        measured_rows,
        measured_graphs,
        task=task,
    )
    request = build_measurement_request(
        task=task,
        selected_rows=acquisition["selected_rows"],
        round_index=round_index,
    )
    result = {
        **base,
        "model_bundle_manifest": copy.deepcopy(dict(bundle_manifest)),
        "predicted_candidates": {
            "schema_version": SCHEMA_VERSION + "_predictions",
            "rows": predicted,
        },
        "acquisition": acquisition,
        "measurement_request": request,
    }
    if variant == "without_measured_feedback":
        projection = project_a2_feedback(
            initial_rows,
            {
                str(row["group_id"]): dict(row)
                for row in initial_graph_features
            },
            anchor=a2_payload["anchor"],
            bundle_sha256=a2_payload["bundle_sha256"],
            selected_results=feedback_rows,
        )
        result["a2_frozen"] = a2_payload
        result["a2_feedback_projection"] = {
            "bundle_sha256": projection["bundle_sha256"],
            "feedback_refit": projection["feedback_refit"],
            "actual_graph_feature_feedback": projection[
                "actual_graph_feature_feedback"
            ],
            "recorded_row_ids": [
                str(row.get("row_id") or row.get("manifest_job_id"))
                for row in projection["recorded_results"]
            ],
            "training_view_sha256": a2_payload["training_view_sha256"],
            "graph_feature_view_sha256": a2_payload[
                "graph_feature_view_sha256"
            ],
        }
    if variant == "backend_blind":
        blind_ids = acquisition["selected_row_ids"]
        full_ids = blind_comparison["full_selected_row_ids"]
        result["backend_blind_audit"] = {
            **blind_comparison,
            "selected_id_overlap": {
                "full_selected_row_ids": full_ids,
                "blind_selected_row_ids": blind_ids,
                "overlap_row_ids": sorted(set(full_ids) & set(blind_ids)),
                "overlap_count": len(set(full_ids) & set(blind_ids)),
            },
        }
    return result


def select_v2_pre_scan_round(
    *,
    variant: str,
    seed: int,
    task: SearchTask,
    pre_scan_pool: Sequence[Mapping[str, Any]],
    initial_rows: Sequence[Mapping[str, Any]],
    initial_graph_features: Sequence[Mapping[str, Any]],
    capability_profiles: Sequence[Mapping[str, Any]],
    closure: Mapping[str, Any],
) -> V2PreScanSelection:
    """Select from one ordered pre-scan pool and expose v2 isolation evidence.

    This adapter intentionally does not invoke a scanner; v2's formal contract
    binds scanner admission to a deferred future stage.
    """
    from framework.stage7.core_ablation_v2 import CORE_VARIANTS

    if variant not in CORE_VARIANTS:
        raise ValueError("unknown v2 core-ablation variant")
    pool = [copy.deepcopy(dict(row)) for row in pre_scan_pool]
    trace = {
        "production_bundle_calls": 0,
        "online_bundle_calls": 0,
        "backend_blind_bundle_calls": 0,
        "candidate_prediction_calls": 0,
        "backend_blind_prediction_calls": 0,
        "predicted_frontier_calls": 0,
        "actual_graph_feedback_rows": 0,
    }
    hooks = {
        "fit_production_bundle": "production_bundle_calls",
        "fit_online_bundle": "online_bundle_calls",
        "fit_backend_blind_bundle": "backend_blind_bundle_calls",
        "predict_candidate_rows": "candidate_prediction_calls",
        "predict_backend_blind_rows": "backend_blind_prediction_calls",
        "select_task_batch": "predicted_frontier_calls",
        "project_a2_feedback": None,
    }
    original_hooks = {name: globals()[name] for name in hooks}

    def recorder(name: str, counter: str | None) -> Any:
        original = original_hooks[name]

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if counter is not None:
                trace[counter] += 1
            if name == "project_a2_feedback":
                selected_results = kwargs.get("selected_results", ())
                if not isinstance(selected_results, Sequence):
                    raise ValueError("v2 feedback hook received invalid rows")
                trace["actual_graph_feedback_rows"] += len(selected_results)
            return original(*args, **kwargs)

        return wrapped

    try:
        for name, counter in hooks.items():
            globals()[name] = recorder(name, counter)
        selection = select_stage7_round(
            variant=variant,
            seed=seed,
            round_index=0,
            task=task,
            candidate_pool=pool,
            initial_rows=initial_rows,
            initial_graph_features=initial_graph_features,
            capability_profiles=capability_profiles,
            closure=closure,
        )
    finally:
        for name, original in original_hooks.items():
            globals()[name] = original
    backend = selection.get("backend_blind_audit", {})
    blind_features = list(backend.get("blind_feature_names") or ())
    full_features = list(backend.get("full_feature_names") or ())
    audit = {
        "ordered_pre_scan_sha256": canonical_sha256(pool),
        "candidate_pool": "pre_scan",
        "scanner_deferred": True,
        "surrogate_calls": (
            trace["production_bundle_calls"]
            + trace["online_bundle_calls"]
            + trace["backend_blind_bundle_calls"]
        ),
        "uncertainty_calls": (
            trace["candidate_prediction_calls"]
            + trace["backend_blind_prediction_calls"]
        ),
        "predicted_frontier_calls": trace["predicted_frontier_calls"],
        "bundle_refit_calls_after_initial": trace["online_bundle_calls"],
        "actual_graph_feedback_rows": trace["actual_graph_feedback_rows"],
        "fixed_dispatch_key": _v2_backend_dispatch_key(pool, variant),
        "removed_feature_names": list(backend.get("removed_names") or ()),
        "model_feature_names": blind_features,
        "forbidden_backend_feature_names": [
            name for name in full_features if name.startswith(("cap:", "cap_x_q:"))
        ],
    }
    _validate_v2_selector_trace(variant, audit)
    return V2PreScanSelection(
        selection=copy.deepcopy(selection), audit=copy.deepcopy(audit)
    )


def _v2_backend_dispatch_key(
    pool: Sequence[Mapping[str, Any]], variant: str
) -> str | None:
    if variant != "backend_blind":
        return None
    dispatch_keys = {str(row.get("dispatch_key") or "") for row in pool}
    if dispatch_keys != {"tvm_auto"}:
        raise ValueError("v2 backend-blind pre-scan dispatch drift")
    return "tvm_auto"


def _validate_v2_selector_trace(variant: str, audit: Mapping[str, Any]) -> None:
    surrogate = int(audit["surrogate_calls"])
    uncertainty = int(audit["uncertainty_calls"])
    frontier = int(audit["predicted_frontier_calls"])
    if variant == "without_surrogate":
        if any((surrogate, uncertainty, frontier)):
            raise ValueError("v2 selector trace drift for without_surrogate")
        return
    if min(surrogate, uncertainty, frontier) <= 0:
        raise ValueError("v2 selector trace drift for model-guided variant")
    if variant == "without_measured_feedback" and (
        int(audit["bundle_refit_calls_after_initial"]) != 0
        or int(audit["actual_graph_feedback_rows"]) != 0
    ):
        raise ValueError("v2 selector trace drift for without_measured_feedback")
    if variant == "backend_blind" and (
        audit["fixed_dispatch_key"] != "tvm_auto"
        or not audit["removed_feature_names"]
        or set(audit["model_feature_names"])
        & set(audit["forbidden_backend_feature_names"])
    ):
        raise ValueError("v2 selector trace drift for backend_blind")


__all__ = [
    "STATIC_SCANNER_RULE",
    "FORMAL_INPUT_SHA256",
    "POLICY_EVIDENCE_SHA256",
    "BackendBlindBundle",
    "V2PreScanSelection",
    "a2_frozen_binding",
    "audit_stage5_model_api",
    "build_request_binding",
    "build_round_chain_head",
    "build_round_state",
    "build_stage7_task",
    "build_trajectory_contract",
    "canonical_sha256",
    "derive_a2_anchor",
    "fit_backend_blind_bundle",
    "predict_backend_blind_rows",
    "prepare_frozen_candidate_pools",
    "select_stage7_round",
    "select_v2_pre_scan_round",
    "validate_frozen_contracts",
    "validate_a2_frozen_contract",
    "validate_request_binding",
]
