#!/usr/bin/env python3
"""Audit the eight F-Cooper capability/coverage probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROFILE_ID = "h800-trt-probe-conditioned-v3"
WIDTHS = {
    "base": [64, 128, 256, 128, 256],
    "boundary": [32, 32, 32, 32, 64],
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def precision_coverage(path: Path) -> dict[str, Any]:
    payload = load(path)
    layers = payload.get("Layers") or []
    counts: Counter[str] = Counter()
    output_dtypes: Counter[str] = Counter()
    execution_layers = 0
    int8_execution_layers = 0
    for layer in layers:
        text = json.dumps(layer).lower()
        for name, tokens in {
            "int8": ("int8",),
            "fp16": ("half", "fp16"),
            "fp32": ("float", "fp32"),
            "reformat": ("reformat",),
        }.items():
            if any(token in text for token in tokens):
                counts[name] += 1
        if isinstance(layer, dict):
            layer_type = str(layer.get("LayerType") or "")
            is_execution = layer_type not in {"Constant", "Reformat"}
            execution_layers += int(is_execution)
            formats = [
                str(output.get("Format/Datatype") or "")
                for output in layer.get("Outputs") or []
                if isinstance(output, dict)
            ]
            for dtype in formats:
                output_dtypes[dtype] += 1
            int8_execution_layers += int(
                is_execution and any(dtype == "Int8" for dtype in formats)
            )
    return {
        "layer_count": len(layers),
        "int8_layer_mentions": counts["int8"],
        "fp16_layer_mentions": counts["fp16"],
        "fp32_layer_mentions": counts["fp32"],
        "reformat_layer_mentions": counts["reformat"],
        "execution_layer_count": execution_layers,
        "int8_execution_layer_count": int8_execution_layers,
        "int8_execution_propagation_ratio": (
            int8_execution_layers / execution_layers if execution_layers else 0.0
        ),
        "output_datatypes": dict(output_dtypes),
        "int8_propagation_observed": int8_execution_layers > 0,
    }


def group_id(width: list[int]) -> str:
    names = (
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    )
    return "fcooper|" + "|".join(
        f"{name}={value}" for name, value in zip(names, width)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for precision in ("fp16", "int8"):
        variants = (
            ("base_default", "base"),
            ("base_tuned", "base"),
            ("boundary_default", "boundary"),
            ("boundary_tuned", "boundary"),
        ) if precision == "fp16" else (("base", "base"), ("boundary", "boundary"))
        for suffix, width_key in variants:
            probe_id = f"{precision}_{suffix}"
            probe = args.root / "probes" / probe_id
            result_path = (
                probe / "result_detailed.json"
                if precision == "int8"
                else probe / "result.json"
            )
            artifact = (
                probe / "artifact_detailed"
                if precision == "int8"
                else probe / "artifact"
            )
            numeric_path = probe / "numeric_sanity16.json"
            result = load(result_path)
            numeric = load(numeric_path)
            width = WIDTHS[width_key]
            rows.append(
                {
                    "probe_id": probe_id,
                    "row_id": (
                        f"{group_id(width)}|q={precision}|profile={PROFILE_ID}"
                    ),
                    "status": "success",
                    "width": width,
                    "q_mode": precision,
                    "build_success": result.get("build_success") is True,
                    "latency_ms": result["lat_p50_ms"],
                    "energy_j": result["energy_j"],
                    "numerical_contract": numeric["numerical_contract"],
                    "engine_calls": numeric["engine_calls"],
                    "fallback_samples": numeric["fallback_samples"],
                    "engine_sha256": sha256_file(artifact / "compiled.engine"),
                    "result_sha256": sha256_file(result_path),
                    "inspector_sha256": sha256_file(
                        artifact / "engine_inspector.json"
                    ),
                    "precision_coverage": precision_coverage(
                        artifact / "engine_inspector.json"
                    ),
                }
            )
    for suffix, width_key in (("base", "base"), ("boundary", "boundary")):
        probe_id = f"maxfusion_{suffix}"
        path = args.root / "probes" / probe_id / "result.json"
        result = load(path)
        rows.append(
            {
                "probe_id": probe_id,
                "status": result["status"],
                "width": WIDTHS[width_key],
                "latency_ms": result["latency_ms"],
                "energy_j": result["energy_j"],
                "numerical_contract": {
                    "repeat_max_abs": result["numerical_repeat_max_abs"],
                    "output_sha256": result["output_sha256"],
                },
                "actual_graph_features": result["actual_graph_features"],
                "result_sha256": sha256_file(path),
            }
        )
    output = {
        "schema_version": "fcooper_probe_audit_v1",
        "probe_count": len(rows),
        "all_probes_terminal": len(rows) == 8
        and all(row["status"] == "success" for row in rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
