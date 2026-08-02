#!/usr/bin/env python3
"""Build canonical v3 capability, active-search, split, and historical-prior artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage2.canonical_search_v3 import (  # noqa: E402
    assign_grouped_split,
    build_active_manifest,
    build_capability_profile,
    route_historical_180,
)


DEFAULT_WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
    "24x64x128",
    "40x64x128",
    "48x64x128",
    "64x64x128",
)
DEFAULT_180_AUDIT = REPO / "results/coldstart_180_precision_audit_20260707.json"
DEFAULT_OUT_DIR = REPO / "results/canonical_search_v3_20260711"

def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_capability_profiles() -> list[dict[str, Any]]:
    """Build a leakage-free bootstrap profile before fresh compiler probes run."""

    bootstrap_features = {
        "supports_fp16_tensorcore": 1.0,
        "supports_int8_tensorcore": 1.0,
        "int8_channel_alignment": 16.0,
        "fp16_channel_alignment": 8.0,
        "int8_precision_propagation_ratio": None,
        "int8_precision_propagation_probe_conv_count": 0.0,
        "qdq_fold_ratio": None,
        "qdq_fold_ratio_observed": 0.0,
        "reformat_rate": None,
        "reformat_rate_observed": 0.0,
        "fusion_coverage": None,
        "compiler_structural_probe_observed": 0.0,
    }
    return [
        build_capability_profile(
            capability_profile_id="h800-tvm-auto-v3",
            hardware_target="h800",
            compiler_fingerprint=_digest({"runtime": "tvm", "profile": "bootstrap-unprobed"}),
            dispatch_key="tvm_auto",
            features=bootstrap_features,
        ),
        build_capability_profile(
            capability_profile_id="h800-trt-v3",
            hardware_target="h800",
            compiler_fingerprint=_digest({"runtime": "tensorrt", "profile": "bootstrap-unprobed"}),
            dispatch_key="trt_engine",
            features=bootstrap_features,
        ),
    ]


def build_outputs(
    *,
    original180_audit_path: Path,
    seed: int,
) -> dict[str, Any]:
    audit = json.loads(original180_audit_path.read_text(encoding="utf-8"))
    profiles = build_capability_profiles()
    manifest = build_active_manifest(
        widths_by_model={"pyramid": DEFAULT_WIDTHS, "codriving": DEFAULT_WIDTHS},
        capability_profiles=profiles,
    )
    split = assign_grouped_split(manifest["jobs"], holdout_group_count=4, seed=seed)
    routes = route_historical_180(audit["rows"], disagreement_ratio=1.5)
    return {
        "capability_profiles": profiles,
        "active_manifest": manifest,
        "split": {"schema_version": "stage2_grouped_split_v3", "seed": seed, **split},
        "historical180_routes": {
            "schema_version": "stage2_historical180_routes_v3",
            **routes,
        },
    }


def write_outputs(outputs: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "capability_profiles": out_dir / "capability_profiles_v3.json",
        "active_manifest": out_dir / "active_manifest_v3.json",
        "split": out_dir / "grouped_split_80_16_v3.json",
        "historical180_routes": out_dir / "historical180_routes_v3.json",
    }
    for name, path in paths.items():
        path.write_text(
            json.dumps(outputs[name], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original180-audit-json", type=Path, default=DEFAULT_180_AUDIT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = build_outputs(
        original180_audit_path=args.original180_audit_json,
        seed=args.seed,
    )
    paths = write_outputs(outputs, args.out_dir)
    summary = {
        "profiles": len(outputs["capability_profiles"]),
        "jobs": len(outputs["active_manifest"]["jobs"]),
        "train": len(outputs["split"]["train"]),
        "holdout": len(outputs["split"]["holdout"]),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
