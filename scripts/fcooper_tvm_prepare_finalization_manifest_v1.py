#!/usr/bin/env python3
"""Create the SHA-bound input manifest for F-Cooper TVM Stage6 finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


POOL_NAMES = (
    "compression_only",
    "schedule_only",
    "compress_then_tune_screen",
    "compress_then_tune_tuned",
    "gear",
)
VALIDATION_NAMES = (
    "original_default",
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "gear",
)
PRECONDITION_NAMES = (
    "capability_probe",
    "probe_isolation",
    "recovery_numeric_gate",
    "control_provenance",
)
SEARCH_INITIALIZATION_NAMES = (
    "task_contract",
    "initialization_summary",
    "capability_profile",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": file_sha256(resolved),
    }


def reference_binding(path: Path, *, model: str) -> dict[str, str]:
    result = binding(path)
    expected_token = f"stage6_{model}_formal"
    if expected_token not in result["path"].lower():
        raise ValueError(f"{model} reference path lacks model identity")
    return {**result, "model": model}


def build_manifest(
    *,
    evidence_root: Path,
    ap70_ref: float,
    original_contract: Path,
    pools: Mapping[str, Path],
    winner_validations: Mapping[str, Path],
    pyramid_reference: Path,
    codriving_reference: Path,
    resource_audit: Path,
    gear_round_audits: Sequence[Path],
    gear_round_states: Sequence[Path],
    precondition_audits: Mapping[str, Path],
    search_initialization: Mapping[str, Path],
) -> dict[str, Any]:
    if set(pools) != set(POOL_NAMES):
        raise ValueError("all five evidence pools are required")
    if set(winner_validations) != set(VALIDATION_NAMES):
        raise ValueError("all five winner validations are required")
    if not math.isfinite(float(ap70_ref)):
        raise ValueError("ap70_ref must be finite")
    if len(gear_round_audits) != 4:
        raise ValueError("exactly four GEAR atomic round audits are required")
    if len(gear_round_states) != 3:
        raise ValueError("exactly three post-round GEAR states are required")
    if set(precondition_audits) != set(PRECONDITION_NAMES):
        raise ValueError("all four precondition audits are required")
    if set(search_initialization) != set(SEARCH_INITIALIZATION_NAMES):
        raise ValueError("all three search initialization artifacts are required")
    return {
        "schema_version": "fcooper_tvm_stage6_finalize_manifest_v1",
        "evidence_root": str(Path(evidence_root).resolve()),
        "ap70_ref": float(ap70_ref),
        "original_contract": binding(original_contract),
        "resource_audit": binding(resource_audit),
        "gear_round_audits": [
            binding(path) for path in gear_round_audits
        ],
        "gear_round_states": [
            binding(path) for path in gear_round_states
        ],
        "precondition_audits": {
            name: binding(precondition_audits[name])
            for name in PRECONDITION_NAMES
        },
        "search_initialization": {
            name: binding(search_initialization[name])
            for name in SEARCH_INITIALIZATION_NAMES
        },
        "pools": {name: binding(pools[name]) for name in POOL_NAMES},
        "winner_validations": {
            name: binding(winner_validations[name])
            for name in VALIDATION_NAMES
        },
        "reference_artifacts": {
            "pyramid_tvm": reference_binding(
                pyramid_reference,
                model="pyramid",
            ),
            "codriving_tvm": reference_binding(
                codriving_reference,
                model="codriving",
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--ap70-ref", type=float, required=True)
    parser.add_argument("--original-contract", type=Path, required=True)
    for name in POOL_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}-pool",
            dest=f"{name}_pool",
            type=Path,
            required=True,
        )
    for name in VALIDATION_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}-validation",
            dest=f"{name}_validation",
            type=Path,
            required=True,
        )
    parser.add_argument("--pyramid-reference", type=Path, required=True)
    parser.add_argument("--codriving-reference", type=Path, required=True)
    parser.add_argument("--resource-audit", type=Path, required=True)
    parser.add_argument(
        "--gear-round-audit",
        type=Path,
        action="append",
        required=True,
        help="SHA-bind one atomic round audit; pass exactly four in round order",
    )
    parser.add_argument(
        "--gear-round-state",
        type=Path,
        action="append",
        required=True,
        help="SHA-bind round states 01, 02, and 03 in order",
    )
    for name in PRECONDITION_NAMES:
        parser.add_argument(
            f"--{name.replace('_', '-')}-audit",
            dest=f"{name}_audit",
            type=Path,
            required=True,
        )
    for name in SEARCH_INITIALIZATION_NAMES:
        parser.add_argument(
            f"--search-{name.replace('_', '-')}",
            dest=f"search_{name}",
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pools = {
        name: getattr(args, f"{name}_pool")
        for name in POOL_NAMES
    }
    validations = {
        name: getattr(args, f"{name}_validation")
        for name in VALIDATION_NAMES
    }
    preconditions = {
        name: getattr(args, f"{name}_audit")
        for name in PRECONDITION_NAMES
    }
    search_initialization = {
        name: getattr(args, f"search_{name}")
        for name in SEARCH_INITIALIZATION_NAMES
    }
    manifest = build_manifest(
        evidence_root=args.evidence_root,
        ap70_ref=args.ap70_ref,
        original_contract=args.original_contract,
        pools=pools,
        winner_validations=validations,
        pyramid_reference=args.pyramid_reference,
        codriving_reference=args.codriving_reference,
        resource_audit=args.resource_audit,
        gear_round_audits=args.gear_round_audit,
        gear_round_states=args.gear_round_state,
        precondition_audits=preconditions,
        search_initialization=search_initialization,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
