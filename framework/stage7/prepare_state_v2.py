"""Immutable root preparation and recovery state for Stage7 v2."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence

from framework.stage7.core_ablation_v2 import (
    CORE_VARIANTS,
    REQUIRED_IMMUTABLE_INPUT_KEYS,
    build_v2_contract,
    canonical_sha256,
    validate_immutable_inputs,
    validate_v2_contract,
)
from framework.stage7.deployment_bundle_v2 import (
    deployment_tree_files,
    validate_deployment_bundle,
)
from framework.stage7.executor_admission_v2 import (
    ProcessIdentityProbe,
    capture_process_identity,
    core_executor_records,
    file_sha256,
    probe_linux_process_identity,
    verify_frozen_actual_v3_executors,
)
from framework.stage7.formal_inputs_v2 import (
    EMBEDDED_IMMUTABLE_KEYS,
    FORMAL_EMBEDDED_SOURCE_IDENTITIES,
    FORMAL_INPUT_IDENTITIES,
    REPO_ROOT,
    json_bytes,
    read_json,
    resolve_embedded_inputs,
    verify_pinned_formal_input_identities,
)
from framework.stage7.online_component_ablation_v1 import SEEDS


IMMUTABLE_PROVENANCE_KEYS = frozenset(REQUIRED_IMMUTABLE_INPUT_KEYS)
_STATE_TOKENS = ("trajectory", "cache", "terminal", "event", "feedback")


def _require_external_sha256(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} external deployment pin is missing or invalid")
    return value


def _write_frozen_json(path: Path, payload: object) -> None:
    content = json_bytes(payload)
    if path.is_file():
        if path.read_bytes() != content:
            raise ValueError(f"refusing to overwrite drifted v2 artifact: {path}")
        return
    if path.exists():
        raise ValueError(f"v2 artifact path is not a file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def initial_measurement_cache() -> dict[str, object]:
    """Return the exact empty cache committed before any v2 selection."""
    return {
        "schema_version": "stage7_core_cache_v2",
        "entries": {},
        "lineage": [],
    }


def _pre_scan_rows(inputs: Mapping[str, object]) -> list[dict[str, object]]:
    rows = inputs.get("pre_scan_registry")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("v2 inputs require a pre_scan_registry row list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("v2 pre-scan registry rows are invalid")
    return [copy.deepcopy(dict(row)) for row in rows]


def _immutable_provenance(inputs: Mapping[str, object]) -> dict[str, object]:
    unexpected = set(inputs) - IMMUTABLE_PROVENANCE_KEYS - {"pre_scan_registry"}
    if unexpected:
        raise ValueError("v2 provenance contains unsupported input keys")
    provenance: dict[str, object] = {}
    for key, value in inputs.items():
        if key == "pre_scan_registry":
            continue
        _reject_state_shaped_provenance(value, key)
        provenance[key] = copy.deepcopy(value)
    validated = validate_immutable_inputs(provenance)
    payload = {
        "schema_version": "stage7_core_ablation_v2_immutable_provenance",
        "inputs": validated,
    }
    return {**payload, "inputs_sha256": canonical_sha256(payload)}


def _reject_state_shaped_provenance(value: object, location: str) -> None:
    lowered_location = location.lower()
    if any(token in lowered_location for token in _STATE_TOKENS):
        raise ValueError("v2 provenance contains mutable state shape")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_state_shaped_provenance(nested, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_state_shaped_provenance(nested, f"{location}[{index}]")
    elif isinstance(value, (str, Path)):
        lowered_value = str(value).lower()
        if any(token in lowered_value for token in _STATE_TOKENS):
            raise ValueError("v2 provenance contains mutable state value")


def _resolved_roots_do_not_overlap(v1_root: Path, v2_root: Path) -> tuple[Path, Path]:
    v1 = v1_root.resolve(strict=False)
    v2 = v2_root.resolve(strict=False)
    try:
        v2.relative_to(v1)
    except ValueError:
        try:
            v1.relative_to(v2)
        except ValueError:
            return v1, v2
    raise ValueError("v1 and v2 roots overlap after resolution")


def _prepare_state(
    *,
    v1_root: Path,
    v2_root: Path,
    contract: Mapping[str, object],
    cache: Mapping[str, object],
    process_identity: Mapping[str, object],
    executor_repo_root: Path,
    executor_verification: Sequence[Mapping[str, object]],
    embedded_immutable_keys: Sequence[str],
    deployment: Mapping[str, object] | None,
) -> dict[str, object]:
    unsigned = {
        "schema_version": "stage7_core_ablation_v2_prepare_state",
        "status": "prepared",
        "owner": process_identity["username"],
        "owner_uid": process_identity["uid"],
        "orchestrator_pid": process_identity["pid"],
        "orchestrator_process_identity": copy.deepcopy(process_identity),
        "executor_repo_root": str(executor_repo_root),
        "executor_verification": [
            copy.deepcopy(record) for record in executor_verification
        ],
        "v1_root": str(v1_root),
        "v2_root": str(v2_root),
        "core_ablation_contract_sha256": contract["contract_sha256"],
        "immutable_inputs_sha256": contract["immutable_inputs_sha256"],
        "actual_feedback_v3_executors": copy.deepcopy(
            contract["actual_feedback_v3_executors"]
        ),
        "initial_cache_sha256": canonical_sha256(cache),
        "initial_cache_lineage_sha256": canonical_sha256(cache["lineage"]),
        "embedded_immutable_contracts": list(embedded_immutable_keys),
        "variants": list(CORE_VARIANTS),
        "seeds": list(SEEDS),
        "deployment_release_sha256": (
            None if deployment is None else deployment["deployment_release_sha256"]
        ),
        "deployment_manifest_sha256": (
            None if deployment is None else deployment["deployment_manifest_sha256"]
        ),
        "deployment_manifest_file_sha256": (
            None
            if deployment is None
            else deployment["deployment_manifest_file_sha256"]
        ),
        "deployment_bundle_sha256": (
            None if deployment is None else deployment["deployment_bundle_sha256"]
        ),
    }
    return {
        **unsigned,
        "recovery_contract_sha256": canonical_sha256(unsigned),
    }


def _trajectory_contract(
    *,
    output_root: Path,
    contract: Mapping[str, object],
    variant: str,
    seed: int,
) -> dict[str, object]:
    trajectory_dir = output_root / "variants" / variant / f"seed_{seed}"
    payload = {
        "schema_version": "stage7_core_ablation_v2_trajectory_contract",
        "variant": variant,
        "seed": seed,
        "trajectory_dir": str(trajectory_dir),
        "round_artifact_pattern": str(trajectory_dir / "round_NN"),
        "candidate_pool": "pre_scan",
        "candidate_pool_count": contract["candidate_pool_count"],
        "ordered_pre_scan_sha256": contract["ordered_pre_scan_sha256"],
        "core_ablation_contract_sha256": contract["contract_sha256"],
        "immutable_inputs_sha256": contract["immutable_inputs_sha256"],
        "actual_feedback_v3_executors": copy.deepcopy(
            contract["actual_feedback_v3_executors"]
        ),
        "scanner": "deferred",
        "scanner_deferred": True,
        "scanner_claim_allowed": False,
    }
    return {**payload, "trajectory_contract_sha256": canonical_sha256(payload)}


def _expected_initialization_artifacts(
    root: Path,
    contract: Mapping[str, object],
    prepare_state: Mapping[str, object],
) -> dict[Path, object]:
    artifacts: dict[Path, object] = {}
    records: list[dict[str, object]] = []
    for variant in CORE_VARIANTS:
        for seed in SEEDS:
            trajectory = _trajectory_contract(
                output_root=root,
                contract=contract,
                variant=variant,
                seed=seed,
            )
            path = (
                root
                / "variants"
                / variant
                / f"seed_{seed}"
                / "trajectory_contract.json"
            )
            artifacts[path] = trajectory
            records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "candidate_pool": "pre_scan",
                    "scanner_deferred": True,
                    "trajectory_contract_path": str(path),
                }
            )
    artifacts[root / "initialize_state.json"] = {
        "schema_version": "stage7_core_ablation_v2_initialize_result",
        "trajectory_count": len(records),
        "trajectories": records,
        "prepare_recovery_contract_sha256": prepare_state["recovery_contract_sha256"],
    }
    return artifacts


def _validate_existing_recovery(
    root: Path,
    artifacts: Mapping[Path, object],
    contract: Mapping[str, object],
    prepare_state: Mapping[str, object],
    deployment_files: set[Path],
) -> bool:
    if not root.exists():
        return False
    if not root.is_dir():
        raise ValueError("v2 recovery root is not a directory")
    if not any(root.iterdir()):
        raise ValueError("refusing pre-existing empty v2 recovery root")
    existing_files = {path for path in root.rglob("*") if path.is_file()}
    if existing_files == deployment_files:
        return False
    for path, payload in artifacts.items():
        if not path.is_file() or path.read_bytes() != json_bytes(payload):
            raise ValueError(f"v2 recovery contract drift: {path}")
    cache = read_json(root / "contracts" / "measurement_cache_initial.json")
    if cache.get("entries") != {} or cache.get("lineage") != []:
        raise ValueError("v2 recovery cache lineage drift")
    base_files = set(artifacts)
    initialized_files = _expected_initialization_artifacts(
        root, contract, prepare_state
    )
    if existing_files == base_files:
        allowed_files = base_files
    elif existing_files == base_files | deployment_files:
        allowed_files = base_files | deployment_files
    elif existing_files == base_files | set(initialized_files) | deployment_files:
        allowed_files = base_files | set(initialized_files) | deployment_files
        for path, payload in initialized_files.items():
            if path.read_bytes() != json_bytes(payload):
                raise ValueError(f"v2 recovery canonical trajectory drift: {path}")
    else:
        raise ValueError("v2 recovery contains noncanonical or mutable state")
    allowed_dirs = {root}
    for path in allowed_files:
        parent = path.parent
        while parent != root:
            allowed_dirs.add(parent)
            parent = parent.parent
    existing_dirs = {path for path in root.rglob("*") if path.is_dir()} | {root}
    if existing_dirs != allowed_dirs:
        raise ValueError("v2 recovery contains noncanonical directories")
    return True


def _validate_prepare_state(
    root: Path,
    contract: Mapping[str, object],
    cache: Mapping[str, object],
    process_identity_probe: ProcessIdentityProbe,
    repo_root: Path,
    executor_records: Sequence[Mapping[str, object]] | None,
    deployment: Mapping[str, object] | None,
) -> dict[str, object]:
    state = read_json(root / "prepare_state.json")
    unsigned = copy.deepcopy(state)
    stored_sha = unsigned.pop("recovery_contract_sha256", None)
    if stored_sha != canonical_sha256(unsigned):
        raise ValueError("v2 prepare recovery contract SHA drift")
    if state.get("owner") in (None, ""):
        raise ValueError("v2 prepare recovery owner drift")
    pid = state.get("orchestrator_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("v2 prepare recovery orchestrator PID drift")
    live_identity = capture_process_identity(process_identity_probe, pid)
    if live_identity != state.get("orchestrator_process_identity"):
        raise ValueError("v2 prepare recovery process identity drift")
    if (
        state.get("owner") != live_identity["username"]
        or state.get("owner_uid") != live_identity["uid"]
    ):
        raise ValueError("v2 prepare recovery owner drift")
    resolved_repo_root = Path(repo_root).resolve(strict=True)
    verification = verify_frozen_actual_v3_executors(
        resolved_repo_root, executor_records=executor_records
    )
    if (
        state.get("executor_repo_root") != str(resolved_repo_root)
        or state.get("executor_verification") != verification
    ):
        raise ValueError("v2 prepare recovery executor verification drift")
    if state.get("v2_root") != str(root):
        raise ValueError("v2 prepare recovery root drift")
    if state.get("core_ablation_contract_sha256") != contract.get("contract_sha256"):
        raise ValueError("v2 prepare recovery contract drift")
    if state.get("initial_cache_sha256") != canonical_sha256(cache):
        raise ValueError("v2 prepare recovery cache drift")
    if state.get("initial_cache_lineage_sha256") != canonical_sha256(
        cache.get("lineage")
    ):
        raise ValueError("v2 prepare recovery cache lineage drift")
    if state.get("actual_feedback_v3_executors") != contract.get(
        "actual_feedback_v3_executors"
    ):
        raise ValueError("v2 prepare recovery executor SHA drift")
    if (
        state.get("deployment_release_sha256")
        != (None if deployment is None else deployment["deployment_release_sha256"])
        or state.get("deployment_manifest_sha256")
        != (None if deployment is None else deployment["deployment_manifest_sha256"])
        or state.get("deployment_manifest_file_sha256")
        != (
            None
            if deployment is None
            else deployment["deployment_manifest_file_sha256"]
        )
        or state.get("deployment_bundle_sha256")
        != (None if deployment is None else deployment["deployment_bundle_sha256"])
    ):
        raise ValueError("v2 prepare recovery deployment bundle drift")
    embedded = state.get("embedded_immutable_contracts")
    if not isinstance(embedded, list) or any(
        key not in EMBEDDED_IMMUTABLE_KEYS for key in embedded
    ):
        raise ValueError("v2 prepare recovery embedded contract drift")
    for key in embedded:
        record = contract["immutable_inputs"][key]
        path = (root / "contracts" / f"{key}.json").resolve(strict=False)
        if (
            record.get("path") != str(path)
            or not path.is_file()
            or record.get("sha256") != file_sha256(path)
        ):
            raise ValueError(f"v2 prepare recovery embedded SHA drift: {key}")
    return state


def _validate_prepared_base_root(
    root: Path,
    contract: Mapping[str, object],
    cache: Mapping[str, object],
    prepare_state: Mapping[str, object],
    deployment_files: set[Path],
) -> None:
    """Require the exact prepare-only tree immediately before initialization."""
    contracts = root / "contracts"
    provenance = read_json(contracts / "immutable_input_provenance.json")
    expected_provenance_unsigned = {
        "schema_version": "stage7_core_ablation_v2_immutable_provenance",
        "inputs": copy.deepcopy(contract["immutable_inputs"]),
    }
    expected_provenance = {
        **expected_provenance_unsigned,
        "inputs_sha256": canonical_sha256(expected_provenance_unsigned),
    }
    if provenance != expected_provenance:
        raise ValueError("v2 prepared base root provenance drift")
    registry = read_json(contracts / "pre_scan_candidate_registry.json")
    if (
        registry.get("schema_version") != "stage7_core_ablation_v2_pre_scan_registry"
        or registry.get("row_count") != contract.get("candidate_pool_count")
        or registry.get("ordered_pre_scan_sha256")
        != contract.get("ordered_pre_scan_sha256")
        or canonical_sha256(registry.get("rows"))
        != contract.get("ordered_pre_scan_sha256")
    ):
        raise ValueError("v2 prepared base root registry drift")
    expected_files = {
        contracts / "core_ablation_v2.json",
        contracts / "pre_scan_candidate_registry.json",
        contracts / "immutable_input_provenance.json",
        contracts / "measurement_cache_initial.json",
        root / "prepare_state.json",
    }
    embedded = prepare_state.get("embedded_immutable_contracts")
    if not isinstance(embedded, list) or any(
        key not in EMBEDDED_IMMUTABLE_KEYS for key in embedded
    ):
        raise ValueError("v2 prepared embedded immutable state drift")
    expected_files.update(contracts / f"{key}.json" for key in embedded)
    expected_files.update(deployment_files)
    if {path for path in root.rglob("*") if path.is_file()} != expected_files:
        raise ValueError("v2 prepared base root is contaminated or noncanonical")
    allowed_dirs = {root, contracts}
    for path in deployment_files:
        parent = path.parent
        while parent != root:
            allowed_dirs.add(parent)
            parent = parent.parent
    if ({path for path in root.rglob("*") if path.is_dir()} | {root}) != allowed_dirs:
        raise ValueError("v2 prepared base root contains noncanonical directories")
    if (
        read_json(contracts / "core_ablation_v2.json") != contract
        or read_json(contracts / "measurement_cache_initial.json") != cache
        or read_json(root / "prepare_state.json") != prepare_state
    ):
        raise ValueError("v2 prepared base root artifact drift")


def prepare_v2_root(
    v1_root: Path,
    v2_root: Path,
    inputs: Mapping[str, object],
    *,
    owner: Optional[str] = None,
    orchestrator_pid: Optional[int] = None,
    repo_root: Path = REPO_ROOT,
    process_identity_probe: ProcessIdentityProbe = probe_linux_process_identity,
    embedded_source_manifest: Mapping[str, object] | None = None,
    formal_input_identities: Mapping[
        str, Mapping[str, object]
    ] = FORMAL_INPUT_IDENTITIES,
    embedded_source_identities: Mapping[
        str, Mapping[str, object]
    ] = FORMAL_EMBEDDED_SOURCE_IDENTITIES,
    executor_records: Sequence[Mapping[str, object]] | None = None,
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> dict[str, object]:
    """Create only v2 immutable inputs, provenance, and its empty cache."""
    release_pin = _require_external_sha256(
        expected_release_sha256, label="expected release SHA256"
    )
    manifest_file_pin = _require_external_sha256(
        expected_manifest_file_sha256, label="expected manifest-file SHA256"
    )
    v1 = Path(v1_root)
    v2 = Path(v2_root)
    if not v1.is_absolute() or not v2.is_absolute():
        raise ValueError("v1_root and v2_root must be absolute")
    if not isinstance(inputs, Mapping):
        raise ValueError("v2 inputs must be a mapping")
    v1, v2 = _resolved_roots_do_not_overlap(v1, v2)
    verify_pinned_formal_input_identities(
        inputs, pinned_identities=formal_input_identities
    )
    resolved_repo_root = Path(repo_root).resolve(strict=True)
    records = core_executor_records() if executor_records is None else executor_records
    executor_verification = verify_frozen_actual_v3_executors(
        resolved_repo_root, executor_records=records
    )
    prepare_pid = orchestrator_pid if orchestrator_pid is not None else os.getpid()
    process_identity = capture_process_identity(process_identity_probe, prepare_pid)
    prepare_owner = owner if owner is not None else str(process_identity["username"])
    if prepare_owner != process_identity["username"]:
        raise ValueError("v2 recovery owner does not match live process identity")
    deployment = None
    deployment_files: set[Path] = set()
    if (v2 / "deployment").exists():
        deployment = validate_deployment_bundle(
            v2,
            frozen_repo_root=resolved_repo_root,
            expected_release_sha256=release_pin,
            expected_manifest_file_sha256=manifest_file_pin,
            expected_owner=str(process_identity["username"]),
            expected_owner_uid=int(process_identity["uid"]),
        )
        deployment_files = deployment_tree_files(v2, deployment)
    resolved_inputs, embedded_artifacts = resolve_embedded_inputs(
        inputs,
        v2_root=v2,
        repo_root=resolved_repo_root,
        source_manifest=embedded_source_manifest,
        pinned_source_identities=embedded_source_identities,
    )
    rows = _pre_scan_rows(resolved_inputs)
    provenance = _immutable_provenance(resolved_inputs)
    contract = validate_v2_contract(
        build_v2_contract(
            rows,
            v1,
            immutable_inputs=provenance["inputs"],
            v2_root=v2,
        )
    )
    cache = initial_measurement_cache()
    contracts = v2 / "contracts"
    registry = {
        "schema_version": "stage7_core_ablation_v2_pre_scan_registry",
        "row_count": len(rows),
        "ordered_pre_scan_sha256": contract["ordered_pre_scan_sha256"],
        "rows": rows,
    }
    state = _prepare_state(
        v1_root=v1,
        v2_root=v2,
        contract=contract,
        cache=cache,
        process_identity=process_identity,
        executor_repo_root=resolved_repo_root,
        executor_verification=executor_verification,
        embedded_immutable_keys=sorted(path.stem for path in embedded_artifacts),
        deployment=deployment,
    )
    artifacts = {
        contracts / "core_ablation_v2.json": contract,
        contracts / "pre_scan_candidate_registry.json": registry,
        contracts / "immutable_input_provenance.json": provenance,
        contracts / "measurement_cache_initial.json": cache,
        v2 / "prepare_state.json": state,
        **embedded_artifacts,
    }
    recovery_validated = _validate_existing_recovery(
        v2, artifacts, contract, state, deployment_files
    )
    for path, payload in artifacts.items():
        _write_frozen_json(path, payload)
    return {
        "schema_version": "stage7_core_ablation_v2_prepare_result",
        "v1_root": str(v1),
        "v2_root": str(v2),
        "ordered_pre_scan_sha256": contract["ordered_pre_scan_sha256"],
        "candidate_pool_count": contract["candidate_pool_count"],
        "scanner_deferred": True,
        "owner": prepare_owner,
        "orchestrator_pid": prepare_pid,
        "orchestrator_process_identity_sha256": process_identity[
            "process_identity_sha256"
        ],
        "executor_repo_root": str(resolved_repo_root),
        "recovery_validated": recovery_validated,
        "recovery_contract_sha256": state["recovery_contract_sha256"],
        "deployment_release_sha256": (
            None if deployment is None else deployment["deployment_release_sha256"]
        ),
        "deployment_manifest_sha256": (
            None if deployment is None else deployment["deployment_manifest_sha256"]
        ),
        "deployment_manifest_file_sha256": (
            None
            if deployment is None
            else deployment["deployment_manifest_file_sha256"]
        ),
        "deployment_bundle_sha256": (
            None if deployment is None else deployment["deployment_bundle_sha256"]
        ),
        "embedded_immutable_contracts_written": sorted(
            path.stem for path in embedded_artifacts
        ),
    }


def initialize_v2_trajectories(
    output_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    process_identity_probe: ProcessIdentityProbe = probe_linux_process_identity,
    executor_records: Sequence[Mapping[str, object]] | None = None,
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> dict[str, object]:
    """Create the four-by-three v2 round-zero contracts without scanner admission."""
    release_pin = _require_external_sha256(
        expected_release_sha256, label="expected release SHA256"
    )
    manifest_file_pin = _require_external_sha256(
        expected_manifest_file_sha256, label="expected manifest-file SHA256"
    )
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("v2 output root must be absolute")
    contract = validate_v2_contract(read_json(root / "contracts/core_ablation_v2.json"))
    if not contract.get("immutable_inputs"):
        raise ValueError("v2 formal immutable inputs are missing")
    cache = read_json(root / "contracts/measurement_cache_initial.json")
    if cache != initial_measurement_cache():
        raise ValueError("v2 initial measurement cache drift")
    records = core_executor_records() if executor_records is None else executor_records
    deployment = None
    deployment_files: set[Path] = set()
    if (root / "deployment").exists():
        state_owner = read_json(root / "prepare_state.json")
        deployment = validate_deployment_bundle(
            root,
            frozen_repo_root=Path(repo_root).resolve(strict=True),
            expected_release_sha256=release_pin,
            expected_manifest_file_sha256=manifest_file_pin,
            expected_owner=(
                state_owner.get("owner")
                if isinstance(state_owner.get("owner"), str)
                else None
            ),
            expected_owner_uid=(
                state_owner.get("owner_uid")
                if isinstance(state_owner.get("owner_uid"), int)
                else None
            ),
        )
        deployment_files = deployment_tree_files(root, deployment)
    prepare_state = _validate_prepare_state(
        root,
        contract,
        cache,
        process_identity_probe,
        repo_root,
        records,
        deployment,
    )
    _validate_prepared_base_root(root, contract, cache, prepare_state, deployment_files)
    if (root / "trajectories").exists():
        raise ValueError("legacy v2 trajectory path drift")
    if list(root.glob("variants/*/seed_*/round_*/trajectory_contract.json")):
        raise ValueError("v2 trajectory contract must be stored at seed root")
    artifacts = _expected_initialization_artifacts(root, contract, prepare_state)
    for path, payload in artifacts.items():
        _write_frozen_json(path, payload)
    return copy.deepcopy(dict(artifacts[root / "initialize_state.json"]))


__all__ = [
    "IMMUTABLE_PROVENANCE_KEYS",
    "initial_measurement_cache",
    "initialize_v2_trajectories",
    "prepare_v2_root",
]
