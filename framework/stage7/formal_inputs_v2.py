"""Pinned formal-input and embedded-contract validation for Stage7 v2."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from framework.stage7.core_ablation_v2 import canonical_sha256
from framework.stage7.executor_admission_v2 import (
    file_sha256,
    is_sha256,
    protocol_hashes,
    verified_primitive_bindings,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDED_IMMUTABLE_KEYS = frozenset({"scope_input_batch", "measurement_ap"})
EMBEDDED_INPUT_KEY = "embedded_immutable_payloads"
EMBEDDED_SOURCE_SCHEMA = "stage7_embedded_immutable_source_manifest_v1"
EXPECTED_SUCCESSFUL_GOLD_ROWS = 174
FORMAL_EMBEDDED_SOURCE_IDENTITIES = {
    "task_contract": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage5_single_target_search_v2_gold176_20260718/"
            "S5-PYR-TVM/task_contract.json"
        ),
        "sha256": "8a7eaaa9b7797a2b35e0642d9b30960420fe0d82c0c47a1eb0411460193a5be2",
    },
    "stage4_closure": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage4_p1_p3_closure_v1_20260716/"
            "stage4_p1_p3_closure_audit.json"
        ),
        "sha256": "dff33fcd4aaa2342352b46fca94e9d91489b471b80f29d4cb58fde05fb03ee43",
    },
    "raw_objective_reference": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage7_pyramid_tvm_online_ablation_quick_v1_20260725/"
            "contracts/raw_objective_reference.json"
        ),
        "sha256": "227f2e3fb22f9265aa01e2c696c6ed88e909214c1648ea2abbb90c32a8e727ee",
    },
}
FORMAL_INPUT_IDENTITIES = {
    "gold176": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage35_gold144_targeted_supplement_v2_20260714/"
            "final_gold176_v1/gold176_final.json"
        ),
        "sha256": "9880d625e1ac2c5e336a5de3bc1d861072d58e05d4b1bea6c79ef1cd0e93ca19",
    },
    "graph_features": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage35_gold144_targeted_supplement_v2_20260714/"
            "final_gold176_v1/graph_features.json"
        ),
        "sha256": "c5f03e19daba4779cb187f036d7c4bf612d3479a00453149f4eaf213412536cd",
    },
    "capability_profiles": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "s1_profile_final_v3_20260711/capability_profiles_v3.json"
        ),
        "sha256": "caac02a50dad5367adcb37b4b57bb0ce23113764ff46faaeda648e221b7c8563",
    },
    "source_registry": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage5_single_target_search_v2_gold176_20260718/"
            "candidate_source_registry_full.json"
        ),
        "sha256": "033cb9e38af39009f9919233f03c5ff77894651f6192d292dee501d974493b51",
    },
    "hv_reference": {
        "path": (
            "/home/jichengzhi/V2X/results/"
            "stage7_pyramid_tvm_online_ablation_quick_v1_20260725/"
            "contracts/raw_objective_reference.json"
        ),
        "sha256": "227f2e3fb22f9265aa01e2c696c6ed88e909214c1648ea2abbb90c32a8e727ee",
    },
}
PRIMITIVE_PATHS = {
    "materializer": "scripts/stage5_materialize_round_sources_v1.sh",
    "quant_contract": "scripts/stage3_tvm_int8_quant_contract_v3.py",
    "performance_plan_builder": "scripts/stage5_build_performance_plan_v2.py",
    "performance_executor": "scripts/stage3_execute_performance_plan_v3.py",
    "ap_plan_builder": "scripts/stage5_ap_plan_v2.py",
    "ap_executor": "scripts/stage3_execute_ap_plan_v3.py",
    "finalizer": "scripts/stage5_finalize_feedback_v2.py",
    "promoter": "scripts/stage5_promote_actual_feedback_v3.py",
    "stage5_controller": "scripts/stage5_task_round_controller_v3.sh",
    "stage5_runtime": "scripts/stage5_advance_task_round_v2.py",
}
PROTOCOL_PRIMITIVES = {
    "build_protocol_sha256": (
        "materializer",
        "quant_contract",
        "performance_plan_builder",
    ),
    "tuning_protocol_sha256": (
        "performance_plan_builder",
        "performance_executor",
        "stage5_controller",
    ),
    "measurement_protocol_sha256": (
        "performance_plan_builder",
        "performance_executor",
        "finalizer",
    ),
    "ap_protocol_sha256": (
        "ap_plan_builder",
        "ap_executor",
        "finalizer",
    ),
    "runtime_contract_sha256": tuple(PRIMITIVE_PATHS),
}


def json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"v2 JSON artifact is invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"v2 JSON artifact is not an object: {path}")
    return copy.deepcopy(dict(payload))


def _verified_json_source(
    record: object,
    *,
    label: str,
    pinned_identities: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, str]]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"embedded source record shape drift: {label}")
    normalized = {
        "path": str(Path(str(record.get("path") or "")).resolve(strict=False)),
        "sha256": record.get("sha256"),
    }
    pinned = pinned_identities.get(label)
    if normalized != pinned:
        raise ValueError(f"pinned embedded source identity drift: {label}")
    path = Path(normalized["path"])
    expected = normalized["sha256"]
    if (
        not path.is_absolute()
        or not path.is_file()
        or not is_sha256(expected)
        or file_sha256(path) != expected
    ):
        raise ValueError(f"embedded source path/SHA drift: {label}")
    return read_json(path), {"path": str(path.resolve()), "sha256": str(expected)}


def _verify_task_contract(payload: Mapping[str, object]) -> None:
    unsigned = copy.deepcopy(dict(payload))
    task_sha = unsigned.pop("task_sha256", None)
    expected_keys = {
        "batch_size",
        "capability_digest",
        "capability_profile_id",
        "dispatch_key",
        "genome_schema",
        "hardware_id",
        "main_search_early_stopping",
        "round_count",
        "sample_budget",
        "schema_version",
        "target_model",
        "task_id",
        "task_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != "stage5_search_task_contract_v2"
        or payload.get("task_id") != "S5-PYR-TVM"
        or payload.get("target_model") != "pyramid"
        or payload.get("hardware_id") != "h800"
        or payload.get("dispatch_key") != "tvm_auto"
        or payload.get("batch_size") != 4
        or payload.get("round_count") != 4
        or payload.get("sample_budget") != 16
        or payload.get("genome_schema") != ["w0", "w1", "w2", "q_mode"]
        or payload.get("main_search_early_stopping") is not False
        or task_sha != canonical_sha256(unsigned)
    ):
        raise ValueError("frozen S5-PYR-TVM task contract content drift")


def _verify_stage4_closure(payload: Mapping[str, object]) -> None:
    heads = payload.get("canonical_value_heads")
    expected_keys = {
        "schema_version",
        "stage4_closed",
        "stage5_search_ready",
        "gates",
        "canonical_value_heads",
        "ranker_policy",
        "uncertainty_policy",
        "selected_acquisition_policy",
        "eligible_acquisition_policies",
        "acquisition_comparison",
        "feedback_update_scope",
        "reproducibility_exact_match",
        "feedback_improved_targets",
        "training_source_rows",
        "independent_holdout",
        "input_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != "stage4_p1_p3_closure_audit_v1"
        or payload.get("stage4_closed") is not True
        or payload.get("stage5_search_ready") is not True
        or not isinstance(heads, Mapping)
        or set(heads) != {"latency_ms", "energy_j", "ap70"}
        or payload.get("selected_acquisition_policy") != "predicted_frontier_diversity"
    ):
        raise ValueError("frozen Stage4 closure content drift")


def _verify_objective_reference(payload: Mapping[str, object]) -> None:
    unsigned = copy.deepcopy(dict(payload))
    reference_sha = unsigned.pop("reference_sha256", None)
    values = payload.get("values")
    required = {"latency_ms", "energy_j", "negative_ap70"}
    if (
        set(payload)
        != {
            "schema_version",
            "formula",
            "successful_gold_rows",
            "values",
            "reference_sha256",
        }
        or payload.get("schema_version") != "stage7_raw_objective_reference_v1"
        or payload.get("successful_gold_rows") != EXPECTED_SUCCESSFUL_GOLD_ROWS
        or reference_sha != canonical_sha256(unsigned)
        or not isinstance(values, Mapping)
        or set(values) != required
        or any(
            isinstance(values[field], bool)
            or not isinstance(values[field], (int, float))
            or not math.isfinite(float(values[field]))
            for field in required
        )
    ):
        raise ValueError("frozen raw objective reference content drift")


def verify_pinned_formal_input_identities(
    inputs: Mapping[str, object],
    *,
    pinned_identities: Mapping[str, Mapping[str, object]] = FORMAL_INPUT_IDENTITIES,
) -> None:
    """Reject alternate Gold/features/profile/registry records before writes."""
    for label, pinned in pinned_identities.items():
        record = inputs.get(label)
        if not isinstance(record, Mapping) or set(record) < {"path", "sha256"}:
            raise ValueError(
                f"formal frozen input provenance identity is missing: {label}"
            )
        normalized = {
            "path": str(Path(str(record.get("path") or "")).resolve(strict=False)),
            "sha256": record.get("sha256"),
        }
        if normalized != pinned:
            raise ValueError(f"formal frozen input provenance identity drift: {label}")
        path = Path(normalized["path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"formal frozen input is unavailable: {label}")
        if file_sha256(path) != normalized["sha256"]:
            raise ValueError(f"formal frozen input SHA drift: {label}")


def build_embedded_immutable_contracts(
    source_manifest: Mapping[str, object],
    *,
    repo_root: Path = REPO_ROOT,
    pinned_source_identities: Mapping[
        str, Mapping[str, object]
    ] = FORMAL_EMBEDDED_SOURCE_IDENTITIES,
) -> dict[str, dict[str, object]]:
    """Derive both embedded contracts only from reviewed frozen artifacts."""
    if (
        not isinstance(source_manifest, Mapping)
        or set(source_manifest)
        != {
            "schema_version",
            "task_contract",
            "stage4_closure",
            "raw_objective_reference",
            "primitives",
        }
        or source_manifest.get("schema_version") != EMBEDDED_SOURCE_SCHEMA
    ):
        raise ValueError("embedded source manifest schema drift")
    task, task_ref = _verified_json_source(
        source_manifest["task_contract"],
        label="task_contract",
        pinned_identities=pinned_source_identities,
    )
    closure, closure_ref = _verified_json_source(
        source_manifest["stage4_closure"],
        label="stage4_closure",
        pinned_identities=pinned_source_identities,
    )
    objective, objective_ref = _verified_json_source(
        source_manifest["raw_objective_reference"],
        label="raw_objective_reference",
        pinned_identities=pinned_source_identities,
    )
    _verify_task_contract(task)
    _verify_stage4_closure(closure)
    _verify_objective_reference(objective)
    primitives = verified_primitive_bindings(
        source_manifest["primitives"],
        repo_root=repo_root,
        primitive_paths=PRIMITIVE_PATHS,
    )
    scope = {
        "schema_version": "stage7_scope_input_batch_v2",
        "task_id": "S7-PYR-TVM",
        "source_task_id": "S5-PYR-TVM",
        "model": "pyramid",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "inference_batch": 1,
        "measurement_scope": "stage5_actual_v3_end_to_end_latency_energy_full_ap",
        "source_task_contract": task_ref,
        "task_contract": task,
    }
    measurement = {
        "schema_version": "stage7_measurement_ap_v2",
        "task_id": "S7-PYR-TVM",
        "source_stage4_closure": closure_ref,
        "closure": closure,
        "source_raw_objective_reference": objective_ref,
        "raw_objective_reference": objective,
        "primitive_bindings": primitives,
        "exact_key_protocol_sha256": protocol_hashes(primitives, PROTOCOL_PRIMITIVES),
    }
    return {
        "scope_input_batch": scope,
        "measurement_ap": measurement,
    }


def _embedded_source_manifest_from_payloads(
    payloads: Mapping[str, object],
) -> dict[str, object]:
    scope = payloads.get("scope_input_batch")
    measurement = payloads.get("measurement_ap")
    if not isinstance(scope, Mapping) or not isinstance(measurement, Mapping):
        raise ValueError("embedded immutable payload shape drift")
    primitives = measurement.get("primitive_bindings")
    if not isinstance(primitives, Mapping):
        raise ValueError("embedded primitive bindings are missing")
    return {
        "schema_version": EMBEDDED_SOURCE_SCHEMA,
        "task_contract": copy.deepcopy(scope.get("source_task_contract")),
        "stage4_closure": copy.deepcopy(measurement.get("source_stage4_closure")),
        "raw_objective_reference": copy.deepcopy(
            measurement.get("source_raw_objective_reference")
        ),
        "primitives": {
            name: {
                "path": record.get("path"),
                "sha256": record.get("sha256"),
            }
            for name, record in primitives.items()
            if isinstance(record, Mapping)
        },
    }


def resolve_embedded_inputs(
    inputs: Mapping[str, object],
    *,
    v2_root: Path,
    repo_root: Path,
    source_manifest: Mapping[str, object] | None,
    pinned_source_identities: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[Path, object]]:
    copied = copy.deepcopy(dict(inputs))
    supplied = copied.pop(EMBEDDED_INPUT_KEY, None)
    if source_manifest is not None and supplied is not None:
        raise ValueError("embedded caller content conflicts with source manifest")
    payloads = (
        build_embedded_immutable_contracts(
            source_manifest,
            repo_root=repo_root,
            pinned_source_identities=pinned_source_identities,
        )
        if source_manifest is not None
        else supplied
    )
    if payloads is None:
        return copied, {}
    if not isinstance(payloads, Mapping) or set(payloads) != EMBEDDED_IMMUTABLE_KEYS:
        raise ValueError("embedded immutable payload keys must be exactly two")
    expected = build_embedded_immutable_contracts(
        _embedded_source_manifest_from_payloads(payloads),
        repo_root=repo_root,
        pinned_source_identities=pinned_source_identities,
    )
    if dict(payloads) != expected:
        raise ValueError("embedded immutable payload content mismatch")
    artifacts: dict[Path, object] = {}
    contracts = v2_root / "contracts"
    for key in sorted(EMBEDDED_IMMUTABLE_KEYS):
        path = (contracts / f"{key}.json").resolve(strict=False)
        record = {
            "path": str(path),
            "sha256": hashlib.sha256(json_bytes(expected[key])).hexdigest(),
        }
        if key in copied and copied[key] != record:
            raise ValueError(f"embedded caller path/SHA mismatch: {key}")
        copied[key] = record
        artifacts[path] = expected[key]
    return copied, artifacts


__all__ = [
    "EMBEDDED_IMMUTABLE_KEYS",
    "EMBEDDED_INPUT_KEY",
    "EMBEDDED_SOURCE_SCHEMA",
    "FORMAL_EMBEDDED_SOURCE_IDENTITIES",
    "FORMAL_INPUT_IDENTITIES",
    "PRIMITIVE_PATHS",
    "PROTOCOL_PRIMITIVES",
    "REPO_ROOT",
    "build_embedded_immutable_contracts",
    "json_bytes",
    "read_json",
    "resolve_embedded_inputs",
    "verify_pinned_formal_input_identities",
]
