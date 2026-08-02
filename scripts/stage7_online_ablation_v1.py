#!/usr/bin/env python3
"""Unified, hardware-independent Stage7 online-ablation CLI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from framework.stage5.single_target_search_v2 import (
    finalize_atomic_batch as stage5_finalize_atomic_batch,
)
from framework.stage7 import cache_feedback_v1 as cache_api
from framework.stage7 import search_policy_v1 as policy_api
from scripts import stage5_promote_actual_feedback_v3 as promotion_api
from scripts import stage7_advance_online_ablation_v1 as advance_api
from scripts import stage7_prepare_online_ablation_v1 as prepare_api


RECEIPT_SCHEMA = "stage7_cli_command_receipt_v1"
AUDIT_SCHEMA = "stage7_trajectory_audit_v1"
COMMANDS = (
    "prepare",
    "init-round",
    "advance-round",
    "reveal-cache",
    "build-miss-plan",
    "finalize-round",
    "audit-trajectory",
)
PREPARE_RELATIVE_OUTPUTS = (
    "contracts/frozen_inputs.json",
    "contracts/freeze_state.json",
    "contracts/pre_scan_candidate_registry.json",
    "contracts/pre_scan_candidate_registry.sha256",
    "contracts/scanner_rule_manifest.json",
    "contracts/scanner_rule_manifest.sha256",
    "contracts/scanner_decision_by_candidate.csv",
    "contracts/scanner_decision_by_candidate.sha256",
    "contracts/pre_scan_to_scan_pass_audit.json",
    "contracts/frozen_candidate_pools.json",
    "contracts/raw_capability_profile.json",
    "contracts/raw_objective_reference.json",
    "contracts/variant_contracts.json",
    "audits/single_variable_isolation.json",
    "freeze_state.json",
)


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, payload: Any) -> str:
    """Durably create one immutable JSON artifact, or accept identical bytes."""
    destination = _absolute(path, "output JSON")
    content = _json_bytes(payload)
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != content:
            raise ValueError(f"refusing to overwrite drifted artifact: {destination}")
        return file_sha256(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_sha256(destination)


def _absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    return path


def _input_file(path: Path, label: str) -> Path:
    checked = _absolute(path, label)
    if not checked.is_file():
        raise ValueError(f"{label} file is missing: {checked}")
    return checked


def _read_json(path: Path, label: str) -> Any:
    checked = _input_file(path, label)
    try:
        return json.loads(checked.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON: {checked}") from error


def _object(path: Path, label: str) -> dict[str, Any]:
    payload = _read_json(path, label)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    return copy.deepcopy(dict(payload))


def _rows(path: Path, label: str) -> list[dict[str, Any]]:
    payload = _read_json(path, label)
    value = payload.get("rows") if isinstance(payload, Mapping) else payload
    if not isinstance(value, list) or not all(
        isinstance(row, Mapping) for row in value
    ):
        raise ValueError(f"{label} must contain a JSON row list")
    return [copy.deepcopy(dict(row)) for row in value]


def _nested_mapping(
    path: Path, label: str, field: str
) -> dict[str, Mapping[str, Any]]:
    payload = _object(path, label)
    value: Any = payload.get(field, payload)
    if not isinstance(value, Mapping) or not all(
        isinstance(item, Mapping) for item in value.values()
    ):
        raise ValueError(f"{label} must contain a {field} mapping")
    return {
        str(key): copy.deepcopy(dict(item))
        for key, item in value.items()
    }


def _validate_persisted_request_chain(
    trajectory: Mapping[str, Any],
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[str, str]:
    trajectory_copy = copy.deepcopy(dict(trajectory))
    trajectory_sha = trajectory_copy.pop("trajectory_contract_sha256", None)
    request_copy = copy.deepcopy(dict(request))
    request_sha = request_copy.pop("measurement_request_sha256", None)
    if trajectory_sha != policy_api.canonical_sha256(trajectory_copy):
        raise ValueError("trajectory contract SHA drift")
    if request_sha != policy_api.canonical_sha256(request_copy):
        raise ValueError("measurement request SHA drift")
    expected_binding_schema = policy_api.SCHEMA_VERSION + "_request_binding"
    if binding.get("schema_version") != expected_binding_schema:
        raise ValueError("unexpected Stage7 policy request binding schema")
    verdict = policy_api.validate_request_binding(binding, request, trajectory)
    cache_binding = _cache_binding(binding)
    # Reuse the reviewed public cache boundary as the canonical request
    # validator. Empty key/cache views produce four misses without revealing
    # external cache state, after validating schema, rows, row SHAs and binding.
    cache_api.reveal_selected_batch_cache(
        request,
        cache_binding,
        trajectory_contract_sha256=str(trajectory_sha),
        key_dimensions_by_row={},
        cache={},
    )
    return str(request_sha), str(verdict["request_binding_sha256"])


def _cache_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": cache_api.BINDING_SCHEMA,
        "measurement_request_sha256": binding.get("measurement_request_sha256"),
        "trajectory_contract_sha256": binding.get("trajectory_contract_sha256"),
        "round_index": binding.get("round_index"),
        "selected_row_ids": copy.deepcopy(binding.get("selected_row_ids")),
        "selected_row_ids_sha256": binding.get("selected_row_ids_sha256"),
    }
    return {
        **payload,
        "request_binding_sha256": policy_api.canonical_sha256(payload),
    }


def _validated_cache_binding(
    request: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, Any]:
    expected_schema = policy_api.SCHEMA_VERSION + "_request_binding"
    if binding.get("schema_version") != expected_schema:
        raise ValueError("unexpected Stage7 policy request binding schema")
    trajectory_view = {
        field: copy.deepcopy(binding.get(field))
        for field in (
            "task_id",
            "variant",
            "seed",
            "trajectory_dir",
            "trajectory_contract_sha256",
        )
    }
    policy_api.validate_request_binding(binding, request, trajectory_view)
    return _cache_binding(binding)


def prepare_managed_outputs(
    output_root: Path, *, include_scanner_audit: bool = True
) -> tuple[Path, ...]:
    paths = [output_root / relative for relative in PREPARE_RELATIVE_OUTPUTS]
    if include_scanner_audit:
        paths.append(output_root / "audits/scanner_admission.json")
    return tuple(paths)


def _round_selection_outputs(
    trajectory_dir: Path,
    *,
    variant: str,
    round_index: int,
    initialized: bool,
) -> tuple[Path, ...]:
    round_dir = trajectory_dir / f"round_{round_index:02d}"
    names = [
        "candidate_manifest.json",
        "acquisition.json",
        "measurement_request.json",
        "stage7_request_binding.json",
        "cache_selection_invariance_audit.json",
        "round_state.json",
    ]
    if variant != "without_surrogate":
        names.extend(("model_bundle_manifest.json", "predicted_candidates.json"))
    if variant == "without_measured_feedback":
        if initialized:
            names.extend(
                (
                    "a2_frozen_bundle_manifest.json",
                    "a2_frozen_candidate_predictions.json",
                    "a2_frozen_anchor.json",
                )
            )
        names.append("a2_frozen_contract.json")
    if variant == "backend_blind":
        names.append("backend_blind_audit.json")
    return tuple(round_dir / name for name in names)


def _initialize_managed_outputs(
    output_root: Path, result: Mapping[str, Any]
) -> tuple[Path, ...]:
    paths = [output_root / "initialize_state.json"]
    records = result.get("trajectories")
    if not isinstance(records, list):
        return tuple(paths)
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("initialize result trajectory record is invalid")
        variant = str(record.get("variant") or "")
        round_dir = _absolute(Path(str(record.get("round_dir") or "")), "round dir")
        if variant not in prepare_api.VARIANTS or round_dir.name != "round_00":
            raise ValueError("initialize result trajectory identity is invalid")
        trajectory_dir = round_dir.parent
        paths.append(trajectory_dir / "trajectory_contract.json")
        paths.extend(
            _round_selection_outputs(
                trajectory_dir,
                variant=variant,
                round_index=0,
                initialized=True,
            )
        )
    return tuple(paths)


def _advance_managed_outputs(
    output_root: Path,
    *,
    variant: str,
    seed: int,
    completed_round_index: int,
    result: Mapping[str, Any],
) -> tuple[Path, ...]:
    trajectory_dir = output_root / "variants" / variant / f"seed_{seed}"
    round_dir = trajectory_dir / f"round_{completed_round_index:02d}"
    if result.get("status") == "infrastructure_retry_required":
        return (round_dir / "infrastructure_retry.json",)
    paths = [
        round_dir / "released_feedback.json",
        round_dir / "atomic_feedback_audit.json",
    ]
    if result.get("schema_version") == "stage7_trajectory_terminal_v1":
        paths.append(trajectory_dir / "trajectory_terminal.json")
    elif result.get("status") == "next_round_generated":
        paths.extend(
            _round_selection_outputs(
                trajectory_dir,
                variant=variant,
                round_index=completed_round_index + 1,
                initialized=False,
            )
        )
    else:
        raise ValueError("advance result status is not recognized")
    return tuple(paths)


def _receipt_path(output_root: Path, command: str, identity: str) -> Path:
    safe_identity = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in identity
    )
    return output_root / "status" / "cli_receipts" / f"{command}_{safe_identity}.json"


def _write_result(
    *,
    command: str,
    output_root: Path,
    result: Mapping[str, Any],
    result_path: Path,
    inputs: Sequence[Path],
    managed_outputs: Sequence[Path],
    receipt_identity: str,
) -> dict[str, Any]:
    result_file_sha = atomic_write_json(result_path, result)
    input_records = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in inputs
    ]
    output_paths = sorted(
        {_absolute(path, "managed output") for path in (*managed_outputs, result_path)},
        key=str,
    )
    outputs = []
    for path in output_paths:
        if not path.is_file():
            raise ValueError(f"managed command output is missing: {path}")
        outputs.append({"absolute_path": str(path), "sha256": file_sha256(path)})
    receipt_payload = {
        "schema_version": RECEIPT_SCHEMA,
        "command": command,
        "output_root": str(output_root),
        "inputs": input_records,
        "input_set_sha256": canonical_sha256(input_records),
        "result_schema_version": result.get("schema_version"),
        "result_path": str(result_path),
        "result_payload_sha256": canonical_sha256(result),
        "output_file_sha256": result_file_sha,
        "outputs": outputs,
        "output_set_sha256": canonical_sha256(outputs),
    }
    receipt = {
        **receipt_payload,
        "command_receipt_sha256": canonical_sha256(receipt_payload),
    }
    atomic_write_json(
        _receipt_path(output_root, command, receipt_identity),
        receipt,
    )
    return receipt


def _prepare(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    inputs = prepare_api.FrozenInputPaths(
        gold176=_input_file(args.gold176, "Gold input"),
        graph_features=_input_file(args.graph_features, "graph feature input"),
        capability_profiles=_input_file(
            args.capability_profiles, "capability profile input"
        ),
        source_registry=_input_file(args.source_registry, "source registry input"),
        stage4_closure=_input_file(args.stage4_closure, "Stage4 closure input"),
    )
    terminal_evidence = _input_file(
        args.terminal_evidence_json, "scanner terminal evidence"
    )
    freeze = prepare_api.freeze_contracts(
        output_root=args.output_root,
        inputs=inputs,
        enforce_formal_counts=not args.synthetic,
    )
    scanner = prepare_api.audit_scanner(
        output_root=args.output_root,
        terminal_evidence_json=terminal_evidence,
    )
    payload = {
        "schema_version": "stage7_prepare_result_v1",
        "freeze": freeze,
        "scanner_admission": scanner,
    }
    paths = [path for _, path in inputs.items()]
    paths.append(terminal_evidence)
    return (
        payload,
        args.output_root / "prepare_result.json",
        paths,
        prepare_managed_outputs(args.output_root),
        "global",
    )


def _init_round(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    if args.round_index != 0:
        raise ValueError("init-round only accepts round index zero")
    result = prepare_api.initialize_trajectories(
        output_root=args.output_root,
        variant=args.variant,
        seed=args.seed,
    )
    path = (
        args.output_root
        / "status"
        / f"init_{args.variant}_seed_{args.seed}_round_00.json"
    )
    return (
        result,
        path,
        _frozen_contract_inputs(args.output_root),
        _initialize_managed_outputs(args.output_root, result),
        f"{args.variant}_{args.seed}_00",
    )


def _advance_round(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    feedback = _input_file(args.feedback_json, "feedback input")
    result = advance_api.advance_trajectory(
        output_root=args.output_root,
        variant=args.variant,
        seed=args.seed,
        completed_round_index=args.completed_round_index,
        feedback_json=feedback,
    )
    path = (
        args.output_root
        / "status"
        / (
            f"advance_{args.variant}_seed_{args.seed}_"
            f"round_{args.completed_round_index:02d}.json"
        )
    )
    return (
        result,
        path,
        [feedback, *_frozen_contract_inputs(args.output_root)],
        _advance_managed_outputs(
            args.output_root,
            variant=args.variant,
            seed=args.seed,
            completed_round_index=args.completed_round_index,
            result=result,
        ),
        f"{args.variant}_{args.seed}_{args.completed_round_index:02d}",
    )


def _reveal_cache(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    # Request and binding are read and validated before either cache input is opened.
    request = _object(args.request_json, "measurement request")
    binding = _object(args.request_binding_json, "request binding")
    trajectory = _object(args.trajectory_contract_json, "trajectory contract")
    _validate_persisted_request_chain(trajectory, request, binding)
    trajectory_sha = str(trajectory.get("trajectory_contract_sha256") or "")
    key_dimensions = _nested_mapping(
        args.key_dimensions_json, "cache key dimensions", "rows"
    )
    cache = _nested_mapping(args.cache_json, "measurement cache", "entries")
    result = cache_api.reveal_selected_batch_cache(
        request,
        _cache_binding(binding),
        trajectory_contract_sha256=trajectory_sha,
        key_dimensions_by_row=key_dimensions,
        cache=cache,
    )
    inputs = [
        args.request_json,
        args.request_binding_json,
        args.trajectory_contract_json,
        args.key_dimensions_json,
        args.cache_json,
    ]
    return (
        result,
        args.output_json,
        inputs,
        (),
        result["measurement_request_sha256"],
    )


def _build_miss_plan(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    inputs = [
        args.full_plan_json,
        args.request_json,
        args.request_binding_json,
        args.cache_reveal_json,
    ]
    request = _object(args.request_json, "measurement request")
    binding = _object(args.request_binding_json, "request binding")
    result = cache_api.derive_miss_only_plan(
        _object(args.full_plan_json, "full performance plan"),
        request,
        _validated_cache_binding(request, binding),
        _object(args.cache_reveal_json, "cache reveal"),
    )
    return (
        result,
        args.output_json,
        inputs,
        (),
        result["original_measurement_request_sha256"],
    )


def _finalize_round(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    request = _object(args.request_json, "measurement request")
    binding = _object(args.request_binding_json, "request binding")
    reveal = _object(args.cache_reveal_json, "cache reveal")
    misses = _rows(args.miss_results_json, "miss results")
    work_dir = _absolute(args.promotion_work_dir, "promotion work directory")
    work_dir.mkdir(parents=True, exist_ok=True)

    def promote(
        request_payload: Mapping[str, Any],
        feedback_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request_path = work_dir / "promotion_request.json"
        feedback_path = work_dir / "promotion_input_feedback.json"
        atomic_write_json(request_path, request_payload)
        atomic_write_json(feedback_path, list(feedback_rows))
        result = promotion_api.promote_feedback_batch(request_path, feedback_path)
        atomic_write_json(work_dir / "promoted_feedback.json", result["rows"])
        atomic_write_json(work_dir / "promotion_audit.json", result["audit"])
        return result

    result = cache_api.finalize_stage7_atomic_batch(
        request,
        _validated_cache_binding(request, binding),
        reveal,
        misses,
        promote_feedback_batch=promote,
        finalize_atomic_batch=stage5_finalize_atomic_batch,
    )
    managed_outputs: list[Path] = []
    if (work_dir / "promotion_request.json").is_file():
        managed_outputs.extend(
            (
                work_dir / "promotion_request.json",
                work_dir / "promotion_input_feedback.json",
                work_dir / "promoted_feedback.json",
                work_dir / "promotion_audit.json",
            )
        )
    if result.get("feedback_released") is True:
        released_path = (
            args.released_feedback_json
            if args.released_feedback_json is not None
            else args.output_json.with_name("final_feedback.json")
        )
        atomic_write_json(released_path, {"rows": result["released_feedback_rows"]})
        managed_outputs.append(released_path)
    inputs = [
        args.request_json,
        args.request_binding_json,
        args.cache_reveal_json,
        args.miss_results_json,
    ]
    return (
        result,
        args.output_json,
        inputs,
        tuple(managed_outputs),
        result["measurement_request_sha256"],
    )


def _audit_trajectory(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Path, list[Path], tuple[Path, ...], str]:
    trajectory = _object(args.trajectory_contract_json, "trajectory contract")
    request = _object(args.request_json, "measurement request")
    binding = _object(args.request_binding_json, "request binding")
    request_sha, binding_sha = _validate_persisted_request_chain(
        trajectory, request, binding
    )
    trajectory_sha = trajectory["trajectory_contract_sha256"]
    if args.expected_request_sha256 is not None and (
        request_sha != args.expected_request_sha256
    ):
        raise ValueError("measurement request SHA does not match expected request")
    payload = {
        "schema_version": AUDIT_SCHEMA,
        "verdict": "pass",
        "task_id": trajectory.get("task_id"),
        "variant": trajectory.get("variant"),
        "seed": trajectory.get("seed"),
        "round_index": request.get("round_index"),
        "selected_row_ids": list(binding.get("selected_row_ids") or []),
        "measurement_request_sha256": request_sha,
        "request_binding_sha256": binding_sha,
        "trajectory_contract_sha256": trajectory_sha,
    }
    result = {**payload, "trajectory_audit_sha256": canonical_sha256(payload)}
    inputs = [
        args.trajectory_contract_json,
        args.request_json,
        args.request_binding_json,
    ]
    return result, args.output_json, inputs, (), str(request_sha)


def _frozen_contract_inputs(output_root: Path) -> list[Path]:
    paths = (
        output_root / "contracts" / "frozen_inputs.json",
        output_root / "contracts" / "frozen_candidate_pools.json",
    )
    return [path for path in paths if path.is_file()]


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, required=True)


def _add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=prepare_api.VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=prepare_api.SEEDS, required=True)


def _add_request_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--request-binding-json", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    _add_output_root(prepare)
    prepare.add_argument("--gold176", type=Path, required=True)
    prepare.add_argument("--graph-features", type=Path, required=True)
    prepare.add_argument("--capability-profiles", type=Path, required=True)
    prepare.add_argument("--source-registry", type=Path, required=True)
    prepare.add_argument("--stage4-closure", type=Path, required=True)
    prepare.add_argument("--terminal-evidence-json", type=Path, required=True)
    prepare.add_argument("--synthetic", action="store_true")
    prepare.set_defaults(handler=_prepare)

    initialize = subparsers.add_parser("init-round")
    _add_output_root(initialize)
    _add_identity(initialize)
    initialize.add_argument("--round-index", type=int, choices=range(4), required=True)
    initialize.set_defaults(handler=_init_round)

    advance = subparsers.add_parser("advance-round")
    _add_output_root(advance)
    _add_identity(advance)
    advance.add_argument(
        "--completed-round-index", type=int, choices=range(4), required=True
    )
    advance.add_argument("--feedback-json", type=Path, required=True)
    advance.set_defaults(handler=_advance_round)

    reveal = subparsers.add_parser("reveal-cache")
    _add_output_root(reveal)
    _add_request_paths(reveal)
    reveal.add_argument("--trajectory-contract-json", type=Path, required=True)
    reveal.add_argument("--key-dimensions-json", type=Path, required=True)
    reveal.add_argument("--cache-json", type=Path, required=True)
    reveal.add_argument("--output-json", type=Path, required=True)
    reveal.set_defaults(handler=_reveal_cache)

    miss_plan = subparsers.add_parser("build-miss-plan")
    _add_output_root(miss_plan)
    _add_request_paths(miss_plan)
    miss_plan.add_argument("--full-plan-json", type=Path, required=True)
    miss_plan.add_argument("--cache-reveal-json", type=Path, required=True)
    miss_plan.add_argument("--output-json", type=Path, required=True)
    miss_plan.set_defaults(handler=_build_miss_plan)

    finalize = subparsers.add_parser("finalize-round")
    _add_output_root(finalize)
    _add_request_paths(finalize)
    finalize.add_argument("--cache-reveal-json", type=Path, required=True)
    finalize.add_argument("--miss-results-json", type=Path, required=True)
    finalize.add_argument("--promotion-work-dir", type=Path, required=True)
    finalize.add_argument("--released-feedback-json", type=Path)
    finalize.add_argument("--output-json", type=Path, required=True)
    finalize.set_defaults(handler=_finalize_round)

    audit = subparsers.add_parser("audit-trajectory")
    _add_output_root(audit)
    _add_request_paths(audit)
    audit.add_argument("--trajectory-contract-json", type=Path, required=True)
    audit.add_argument("--expected-request-sha256")
    audit.add_argument("--output-json", type=Path, required=True)
    audit.set_defaults(handler=_audit_trajectory)
    return parser


def _validate_namespace_paths(args: argparse.Namespace) -> None:
    args.output_root = _absolute(args.output_root, "output root")
    for name, value in vars(args).items():
        if name == "output_root" or not isinstance(value, Path):
            continue
        setattr(args, name, _absolute(value, name.replace("_", " ")))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_namespace_paths(args)
    result, result_path, inputs, managed_outputs, identity = args.handler(args)
    receipt = _write_result(
        command=args.command,
        output_root=args.output_root,
        result=result,
        result_path=result_path,
        inputs=[_input_file(path, "command input") for path in inputs],
        managed_outputs=managed_outputs,
        receipt_identity=identity,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
