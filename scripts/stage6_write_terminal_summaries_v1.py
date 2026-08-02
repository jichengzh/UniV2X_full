#!/usr/bin/env python3
"""Materialize Stage6 slot terminals from the final verified evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ARMS = (
    "compression_only",
    "schedule_only",
    "compress_then_tune",
    "tune_then_compress",
    "joint_shcosearch",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_terminals(bundle: Mapping[str, Any], bundle_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if bundle.get("schema_version") != "stage6_paper_evidence_bundle_v1":
        raise ValueError("unexpected Stage6 evidence bundle schema")
    digest = _sha(bundle_path)
    baseline = bundle.get("baseline") or {}
    if int(baseline.get("independent_repeat_count") or 0) != 3:
        raise ValueError("native baseline lacks three independent repeats")
    terminals = {
        ("pytorch_eager", "original_default"): {
            "schema_version": "stage6_terminal_summary_v1",
            "status": "complete",
            "backend": "pytorch_eager",
            "arm_id": "original_default",
            "evidence_bundle_path": str(bundle_path),
            "evidence_bundle_sha256": digest,
        }
    }
    for backend in ("tvm", "trt"):
        for arm_id in ARMS:
            arm = bundle["backends"][backend][arm_id]
            status = str(arm.get("status") or "")
            if status == "complete":
                if arm.get("independent_validation_complete") is not True:
                    raise ValueError(f"independent validation incomplete: {backend}/{arm_id}")
            elif status == "complete_failure":
                if arm.get("failure_evidence_sha_verified") is not True:
                    raise ValueError(f"failure evidence incomplete: {backend}/{arm_id}")
            else:
                raise ValueError(f"nonterminal Stage6 arm: {backend}/{arm_id}={status}")
            terminals[(backend, arm_id)] = {
                "schema_version": "stage6_terminal_summary_v1",
                "status": status,
                "backend": backend,
                "arm_id": arm_id,
                "point_count": len(arm.get("points") or []),
                "failure_count": int(arm.get("failure_count") or 0),
                "evidence_bundle_path": str(bundle_path),
                "evidence_bundle_sha256": digest,
            }
    return terminals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.evidence_bundle.read_text(encoding="utf-8"))
    terminals = build_terminals(bundle, args.evidence_bundle)
    for (backend, arm_id), terminal in terminals.items():
        _write(args.formal_root / backend / arm_id / "terminal_summary.json", terminal)
    print(json.dumps({"terminal_count": len(terminals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
