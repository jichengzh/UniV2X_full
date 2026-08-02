#!/usr/bin/env python3
"""Prepare the scanner-deferred Stage7 core-ablation v2 contracts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import sys
from typing import Any, Callable, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage7 import executor_admission_v2 as _admission
from framework.stage7 import deployment_bundle_v2 as _deployment
from framework.stage7 import formal_inputs_v2 as _formal
from framework.stage7 import prepare_state_v2 as _state
from framework.stage7.core_ablation_v2 import (
    CORE_VARIANTS,
    REQUIRED_IMMUTABLE_INPUT_KEYS,
    build_v2_contract,
    canonical_sha256,
    validate_immutable_inputs,
    validate_v2_contract,
)
from framework.stage7.executor_admission_v2 import ProcessIdentityProbe
from framework.stage7.online_component_ablation_v1 import SEEDS


IMMUTABLE_PROVENANCE_KEYS = _state.IMMUTABLE_PROVENANCE_KEYS
EMBEDDED_IMMUTABLE_KEYS = _formal.EMBEDDED_IMMUTABLE_KEYS
EMBEDDED_INPUT_KEY = _formal.EMBEDDED_INPUT_KEY
EMBEDDED_SOURCE_SCHEMA = _formal.EMBEDDED_SOURCE_SCHEMA
FORMAL_EMBEDDED_SOURCE_IDENTITIES = _formal.FORMAL_EMBEDDED_SOURCE_IDENTITIES
FORMAL_INPUT_IDENTITIES = _formal.FORMAL_INPUT_IDENTITIES
PRIMITIVE_PATHS = _formal.PRIMITIVE_PATHS
PROTOCOL_PRIMITIVES = _formal.PROTOCOL_PRIMITIVES
_PINNED_EMBEDDED_SOURCE_IDENTITIES = FORMAL_EMBEDDED_SOURCE_IDENTITIES
_PINNED_FORMAL_INPUT_IDENTITIES = FORMAL_INPUT_IDENTITIES
_STATE_TOKENS = ("trajectory", "cache", "terminal", "event", "feedback")


def _json_bytes(payload: object) -> bytes:
    return _formal.json_bytes(payload)


def read_json(path: Path) -> dict[str, object]:
    return _formal.read_json(path)


def initial_measurement_cache() -> dict[str, object]:
    return _state.initial_measurement_cache()


def _file_sha256(path: Path) -> str:
    return _admission.file_sha256(path)


def _is_sha256(value: object) -> bool:
    return _admission.is_sha256(value)


def verify_pinned_formal_input_identities(
    inputs: Mapping[str, object],
) -> None:
    return _formal.verify_pinned_formal_input_identities(
        inputs,
        pinned_identities=_PINNED_FORMAL_INPUT_IDENTITIES,
    )


def build_embedded_immutable_contracts(
    source_manifest: Mapping[str, object],
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[str, object]]:
    return _formal.build_embedded_immutable_contracts(
        source_manifest,
        repo_root=repo_root,
        pinned_source_identities=_PINNED_EMBEDDED_SOURCE_IDENTITIES,
    )


def verify_frozen_actual_v3_executors(
    repo_root: Path,
) -> list[dict[str, object]]:
    return _admission.verify_frozen_actual_v3_executors(
        repo_root,
        executor_records=core_executor_records(),
    )


def core_executor_records() -> list[dict[str, object]]:
    return _admission.core_executor_records()


def _normalize_process_identity(
    payload: Mapping[str, object], expected_pid: int
) -> dict[str, object]:
    return _admission.normalize_process_identity(payload, expected_pid)


def _capture_process_identity(
    probe: ProcessIdentityProbe, pid: int
) -> dict[str, object]:
    return _admission.capture_process_identity(probe, pid)


def probe_linux_process_identity(pid: int) -> dict[str, object]:
    return _admission.probe_linux_process_identity(pid)


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
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> dict[str, object]:
    """Compatibility wrapper retaining old-module monkeypatch seams."""
    return _state.prepare_v2_root(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=inputs,
        owner=owner,
        orchestrator_pid=orchestrator_pid,
        repo_root=repo_root,
        process_identity_probe=process_identity_probe,
        embedded_source_manifest=embedded_source_manifest,
        formal_input_identities=_PINNED_FORMAL_INPUT_IDENTITIES,
        embedded_source_identities=_PINNED_EMBEDDED_SOURCE_IDENTITIES,
        executor_records=core_executor_records(),
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )


def initialize_v2_trajectories(
    output_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    process_identity_probe: ProcessIdentityProbe = probe_linux_process_identity,
    expected_release_sha256: str | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> dict[str, object]:
    return _state.initialize_v2_trajectories(
        output_root,
        repo_root=repo_root,
        process_identity_probe=process_identity_probe,
        executor_records=core_executor_records(),
        expected_release_sha256=expected_release_sha256,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )


def build_deployment_bundle(
    *,
    v2_root: Path,
    frozen_repo_root: Path,
    owner: str | None = None,
) -> dict[str, object]:
    """Create the deployment-only root before immutable prepare artifacts."""
    return _deployment.build_deployment_bundle(
        v2_root=v2_root,
        frozen_repo_root=frozen_repo_root,
        owner=owner,
    )


def _parse_inputs(path: Path) -> dict[str, object]:
    return read_json(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-core-v2")
    prepare.add_argument("--v1-root", type=Path, required=True)
    prepare.add_argument("--v2-root", type=Path, required=True)
    prepare.add_argument("--inputs-json", type=Path, required=True)
    prepare.add_argument("--owner")
    prepare.add_argument("--orchestrator-pid", type=int, required=True)
    prepare.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    prepare.add_argument("--embedded-source-manifest-json", type=Path)
    prepare.add_argument("--expected-release-sha256", required=True)
    prepare.add_argument("--expected-manifest-file-sha256", required=True)
    initialize = commands.add_parser("init-core-v2")
    initialize.add_argument("--v2-root", type=Path, required=True)
    initialize.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    initialize.add_argument("--expected-release-sha256", required=True)
    initialize.add_argument("--expected-manifest-file-sha256", required=True)
    bundle = commands.add_parser("bundle-core-v2")
    bundle.add_argument("--v2-root", type=Path, required=True)
    bundle.add_argument("--frozen-repo-root", type=Path, required=True)
    bundle.add_argument("--owner")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "bundle-core-v2":
        result = build_deployment_bundle(
            v2_root=args.v2_root.resolve(),
            frozen_repo_root=args.frozen_repo_root.resolve(),
            owner=args.owner,
        )
    elif args.command == "prepare-core-v2":
        result = prepare_v2_root(
            v1_root=args.v1_root.resolve(),
            v2_root=args.v2_root.resolve(),
            inputs=_parse_inputs(args.inputs_json.resolve()),
            owner=args.owner,
            orchestrator_pid=args.orchestrator_pid,
            repo_root=args.repo_root.resolve(),
            expected_release_sha256=args.expected_release_sha256,
            expected_manifest_file_sha256=args.expected_manifest_file_sha256,
            embedded_source_manifest=(
                _parse_inputs(args.embedded_source_manifest_json.resolve())
                if args.embedded_source_manifest_json is not None
                else None
            ),
        )
    else:
        result = initialize_v2_trajectories(
            args.v2_root.resolve(),
            repo_root=args.repo_root.resolve(),
            expected_release_sha256=args.expected_release_sha256,
            expected_manifest_file_sha256=args.expected_manifest_file_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CORE_VARIANTS",
    "EMBEDDED_IMMUTABLE_KEYS",
    "EMBEDDED_INPUT_KEY",
    "EMBEDDED_SOURCE_SCHEMA",
    "FORMAL_EMBEDDED_SOURCE_IDENTITIES",
    "FORMAL_INPUT_IDENTITIES",
    "IMMUTABLE_PROVENANCE_KEYS",
    "PRIMITIVE_PATHS",
    "PROTOCOL_PRIMITIVES",
    "ProcessIdentityProbe",
    "REPO_ROOT",
    "REQUIRED_IMMUTABLE_INPUT_KEYS",
    "SEEDS",
    "build_embedded_immutable_contracts",
    "build_deployment_bundle",
    "build_parser",
    "build_v2_contract",
    "canonical_sha256",
    "core_executor_records",
    "initial_measurement_cache",
    "initialize_v2_trajectories",
    "main",
    "prepare_v2_root",
    "probe_linux_process_identity",
    "read_json",
    "validate_immutable_inputs",
    "validate_v2_contract",
    "verify_frozen_actual_v3_executors",
    "verify_pinned_formal_input_identities",
]


if __name__ == "__main__":
    raise SystemExit(main())
