"""Immutable-input contract for the scanner-deferred Stage7 core ablation v2."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional, Sequence


SCHEMA_VERSION = "stage7_core_ablation_v2"
V1_ROOT = Path(
    "/home/jichengzhi/V2X/results/stage7_pyramid_tvm_online_ablation_quick_v1_20260725"
)
V2_ROOT = Path(
    "/home/jichengzhi/V2X/results/stage7_pyramid_tvm_online_core_ablation_quick_v2_20260725"
)
CORE_VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
)
EXPECTED_SOURCE_REGISTRY_SHA256 = (
    "033cb9e38af39009f9919233f03c5ff77894651f6192d292dee501d974493b51"
)
EXPECTED_PRE_SCAN_REGISTRY_SHA256 = (
    "3e584a4d06afe3ca0ba6cedce9e0a99a20aafae8b1ce7ab62b4fd00da07ac4b8"
)
EXPECTED_ORDERED_PRE_SCAN_SHA256 = (
    "3294a3e66671af2778bffd4d719ab9f45963d948cabdfb21315012274daf96f4"
)
EXPECTED_BLOCKER_SHA256 = (
    "20cadfee33e2cbbffd3e2c58ececd89282c3e3733ad5924dadc885b0f9efc5da"
)
EXPECTED_ADMISSION_SHA256 = (
    "06761a3fdba4a945b9323941924d332e05b9f7a20321009a193696cce81ad170"
)
REQUIRED_IMMUTABLE_INPUT_KEYS = (
    "gold176",
    "graph_features",
    "capability_profiles",
    "source_registry",
    "scope_input_batch",
    "measurement_ap",
    "hv_reference",
)
FROZEN_ACTUAL_V3_EXECUTORS = (
    {
        "path": "scripts/stage5_task_round_controller_v3.sh",
        "sha256": "132321077fc1a308dc43fe3e5af3f6d74725088fb526e05e57bd31530ed40170",
    },
    {
        "path": "scripts/stage5_advance_task_round_v2.py",
        "sha256": "309a7f3889be01655702d0c97217c32fbbdc4ac7b3cb873937cb89f0dd059eb0",
    },
    {
        "path": "scripts/stage5_promote_actual_feedback_v3.py",
        "sha256": "fbbd295cc881ebac6db50ee3764eb667bc921168a1981f96ec7c9d3fcb440187",
    },
    {
        "path": "framework/stage5/single_target_search_v2.py",
        "sha256": "7d3694a82cee4e9f9265471fd32371d1012279a71a6c150bf8443240b870df35",
    },
    {
        "path": "framework/stage5/production_search_v1.py",
        "sha256": "6d373c312e236222d7630d91deb8cb122d5c928b464e5d1b60f7070ea33db2e2",
    },
)
V2Contract = Mapping[str, object]
_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "v1_root",
        "v2_root",
        "source_registry_sha256",
        "pre_scan_registry_sha256",
        "ordered_pre_scan_sha256",
        "candidate_pool_count",
        "variants",
        "scanner_deferred",
        "scanner_claim_allowed",
        "variant_contracts",
        "v1_blocker",
        "v1_expanded_admission",
        "immutable_inputs",
        "immutable_inputs_sha256",
        "actual_feedback_v3_executors",
        "contract_sha256",
    }
)


class _FrozenDict(dict):
    """JSON-serializable mapping that refuses every mutating dict operation."""

    @staticmethod
    def _immutable(*_args: object, **_kwargs: object) -> None:
        raise TypeError("v2 contract is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def canonical_sha256(payload: object) -> str:
    """Hash deterministic candidate/contract content without filesystem formatting."""
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_mapping(payload: object, label: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} payload is invalid")
    return payload


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def validate_immutable_inputs(
    payload: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    """Validate the complete path+SHA formal-input binding."""
    inputs = _thaw(_require_mapping(payload, "immutable inputs"))
    if not isinstance(inputs, dict) or set(inputs) != set(
        REQUIRED_IMMUTABLE_INPUT_KEYS
    ):
        raise ValueError("immutable input schema drift")
    validated: dict[str, dict[str, object]] = {}
    for key in REQUIRED_IMMUTABLE_INPUT_KEYS:
        record = _thaw(_require_mapping(inputs.get(key), f"immutable input {key}"))
        if not isinstance(record, dict):
            raise ValueError(f"immutable input {key} payload is invalid")
        path = record.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"immutable input {key} path must be absolute")
        if not _is_sha256(record.get("sha256")):
            raise ValueError(f"immutable input {key} SHA is invalid")
        validated[key] = copy.deepcopy(record)
    if (
        validated["source_registry"]["sha256"]
        != EXPECTED_SOURCE_REGISTRY_SHA256
    ):
        raise ValueError("v2 source registry SHA drift")
    return validated


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise ValueError(f"{label} artifact is missing")
    try:
        return _require_mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} payload is invalid") from error


def load_and_validate_v1_blocker(v1_root: Path) -> dict[str, object]:
    """Read v1's frozen failure evidence without changing either source artifact."""
    root = Path(v1_root)
    blocker_path = root / "status" / "without_capability_scan_blocked.json"
    admission_path = root / "audits" / "scanner_admission_expanded_v1.json"
    blocker_sha = _file_sha256(blocker_path) if blocker_path.is_file() else None
    if blocker_sha != EXPECTED_BLOCKER_SHA256:
        raise ValueError("v1 blocker SHA drift")
    admission_sha = _file_sha256(admission_path) if admission_path.is_file() else None
    if admission_sha != EXPECTED_ADMISSION_SHA256:
        raise ValueError("v1 expanded admission SHA drift")

    blocker = _read_json(blocker_path, "v1 blocker")
    admission = _read_json(admission_path, "v1 expanded admission")
    if blocker.get("status") != "blocked_missing_candidate_level_scanner":
        raise ValueError("v1 blocker status drift")
    if admission.get("admission_passed") is not False:
        raise ValueError("v1 expanded admission status drift")
    if admission.get("false_positive_unique_count") != 3:
        raise ValueError("v1 expanded admission false-positive count drift")
    return {
        "v1_blocker": {
            "path": str(blocker_path),
            "sha256": blocker_sha,
            "status": str(blocker["status"]),
        },
        "v1_expanded_admission": {
            "path": str(admission_path),
            "sha256": admission_sha,
            "admission_passed": False,
            "false_positive_unique_count": 3,
        },
    }


def _ordered_row_ids(pre_scan_registry: Sequence[Mapping[str, object]]) -> list[str]:
    row_ids: list[str] = []
    for row in pre_scan_registry:
        if not isinstance(row, Mapping):
            raise ValueError("pre-scan registry row is invalid")
        row_id = str(row.get("row_id") or row.get("manifest_job_id") or "")
        if not row_id:
            raise ValueError("pre-scan registry identities are invalid")
        row_ids.append(row_id)
    if len(row_ids) != 686:
        raise ValueError("pre-scan registry count drift")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("pre-scan registry identities drift")
    return row_ids


def _variant_contracts(ordered_pre_scan_sha256: str) -> list[dict[str, object]]:
    common = {
        "candidate_pool": "pre_scan",
        "candidate_pool_count": 686,
        "ordered_pre_scan_sha256": ordered_pre_scan_sha256,
        "scanner": "deferred",
        "scanner_deferred": True,
        "scanner_claim_allowed": False,
        "candidate_failure_accounting": "consume_selected_event",
        "infrastructure_failure_accounting": "same_request_retry_without_budget",
    }
    return [
        {
            **common,
            "variant": "full",
            "policy_name": "predicted_frontier_diversity",
            "surrogate_acquisition": "enabled",
            "feedback_refit": "enabled",
            "actual_graph_feature_feedback": "enabled",
            "backend_model_features": "enabled",
        },
        {
            **common,
            "variant": "without_surrogate",
            "policy_name": "predicted_frontier_diversity",
            "surrogate_acquisition": "uniform_random_without_replacement",
            "feedback_refit": "enabled",
            "actual_graph_feature_feedback": "enabled",
            "backend_model_features": "enabled",
        },
        {
            **common,
            "variant": "without_measured_feedback",
            "policy_name": "predicted_frontier_diversity",
            "surrogate_acquisition": "enabled",
            "feedback_refit": "frozen_initial_bundle",
            "actual_graph_feature_feedback": "off",
            "backend_model_features": "enabled",
        },
        {
            **common,
            "variant": "backend_blind",
            "policy_name": "predicted_frontier_diversity",
            "surrogate_acquisition": "enabled",
            "feedback_refit": "enabled",
            "actual_graph_feature_feedback": "enabled",
            "backend_model_features": "off",
        },
    ]


def build_v2_contract(
    pre_scan_registry: Sequence[Mapping[str, object]],
    v1_root: Path,
    *,
    immutable_inputs: Optional[Mapping[str, object]] = None,
    v2_root: Path = V2_ROOT,
) -> V2Contract:
    """Bind the four v2 selectors to one formal pre-scan registry and v1 blocker."""
    rows = [copy.deepcopy(dict(row)) for row in pre_scan_registry]
    _ordered_row_ids(rows)
    ordered_pre_scan_sha256 = canonical_sha256(rows)
    if ordered_pre_scan_sha256 != EXPECTED_ORDERED_PRE_SCAN_SHA256:
        raise ValueError("ordered pre-scan SHA drift")
    v1_evidence = load_and_validate_v1_blocker(Path(v1_root))
    formal_inputs = (
        validate_immutable_inputs(immutable_inputs)
        if immutable_inputs is not None
        else {}
    )
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "v1_root": str(Path(v1_root)),
        "v2_root": str(Path(v2_root).resolve(strict=False)),
        "source_registry_sha256": EXPECTED_SOURCE_REGISTRY_SHA256,
        "pre_scan_registry_sha256": EXPECTED_PRE_SCAN_REGISTRY_SHA256,
        "ordered_pre_scan_sha256": ordered_pre_scan_sha256,
        "candidate_pool_count": 686,
        "variants": list(CORE_VARIANTS),
        "scanner_deferred": True,
        "scanner_claim_allowed": False,
        "variant_contracts": _variant_contracts(ordered_pre_scan_sha256),
        "immutable_inputs": formal_inputs,
        "immutable_inputs_sha256": canonical_sha256(formal_inputs),
        "actual_feedback_v3_executors": [
            copy.deepcopy(record) for record in FROZEN_ACTUAL_V3_EXECUTORS
        ],
        **v1_evidence,
    }
    payload = {**unsigned, "contract_sha256": canonical_sha256(unsigned)}
    return _freeze(validate_v2_contract(payload))


def validate_v2_contract(payload: Mapping[str, object]) -> dict[str, object]:
    """Fail closed unless a serialized v2 contract still matches every frozen value."""
    contract = _thaw(_require_mapping(payload, "v2 contract"))
    if not isinstance(contract, dict):
        raise ValueError("v2 contract payload is invalid")
    if set(contract) != _CONTRACT_FIELDS:
        raise ValueError("v2 contract exact schema drift")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v2 contract schema drift")
    if tuple(contract.get("variants") or ()) != CORE_VARIANTS:
        raise ValueError("v2 contract variants drift")
    if contract.get("source_registry_sha256") != EXPECTED_SOURCE_REGISTRY_SHA256:
        raise ValueError("v2 source registry SHA drift")
    if contract.get("pre_scan_registry_sha256") != EXPECTED_PRE_SCAN_REGISTRY_SHA256:
        raise ValueError("v2 pre-scan registry SHA drift")
    if contract.get("ordered_pre_scan_sha256") != EXPECTED_ORDERED_PRE_SCAN_SHA256:
        raise ValueError("ordered pre-scan SHA drift")
    if contract.get("candidate_pool_count") != 686:
        raise ValueError("v2 candidate pool count drift")
    if contract.get("scanner_deferred") is not True:
        raise ValueError("v2 scanner deferred status drift")
    if contract.get("scanner_claim_allowed") is not False:
        raise ValueError("v2 scanner claim status drift")
    immutable_inputs = contract.get("immutable_inputs")
    if not isinstance(immutable_inputs, Mapping):
        raise ValueError("v2 immutable input schema drift")
    if immutable_inputs:
        validate_immutable_inputs(immutable_inputs)
    if contract.get("immutable_inputs_sha256") != canonical_sha256(
        immutable_inputs
    ):
        raise ValueError("v2 immutable input SHA drift")
    if contract.get("actual_feedback_v3_executors") != [
        dict(record) for record in FROZEN_ACTUAL_V3_EXECUTORS
    ]:
        raise ValueError("actual-feedback v3 executor SHA drift")

    variants = contract.get("variant_contracts")
    if not isinstance(variants, list):
        raise ValueError("v2 variant contract drift")
    if any(
        not isinstance(row, Mapping) or row.get("scanner_claim_allowed") is not False
        for row in variants
    ):
        raise ValueError("v2 scanner claim status drift")
    if variants != _variant_contracts(EXPECTED_ORDERED_PRE_SCAN_SHA256):
        raise ValueError("v2 variant contract drift")
    blocker = _require_mapping(contract.get("v1_blocker"), "v1 blocker")
    if blocker.get("sha256") != EXPECTED_BLOCKER_SHA256:
        raise ValueError("v1 blocker SHA drift")
    if blocker.get("status") != "blocked_missing_candidate_level_scanner":
        raise ValueError("v1 blocker status drift")
    admission = _require_mapping(contract.get("v1_expanded_admission"), "v1 expanded admission")
    if admission.get("sha256") != EXPECTED_ADMISSION_SHA256:
        raise ValueError("v1 expanded admission SHA drift")
    if admission.get("admission_passed") is not False:
        raise ValueError("v1 expanded admission status drift")
    if admission.get("false_positive_unique_count") != 3:
        raise ValueError("v1 expanded admission false-positive count drift")

    stored_sha = contract.pop("contract_sha256", None)
    if stored_sha != canonical_sha256(contract):
        raise ValueError("v2 contract SHA drift")
    return copy.deepcopy(contract | {"contract_sha256": stored_sha})
