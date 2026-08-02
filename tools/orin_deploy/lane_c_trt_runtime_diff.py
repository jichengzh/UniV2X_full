#!/usr/bin/env python3
"""Compare Lane C TensorRT calibration caches and engine inspectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_calibration_cache(payload: bytes) -> dict[str, Any]:
    text = payload.decode("utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError("empty TensorRT calibration cache")
    entries: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        if ": " not in line:
            raise ValueError(f"invalid TensorRT calibration cache row: {line!r}")
        name, hex_value = line.rsplit(": ", 1)
        normalized = hex_value.lower()
        if len(normalized) != 8 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError(f"invalid calibration scale encoding: {hex_value!r}")
        if name in entries:
            raise ValueError(f"duplicate calibration tensor row: {name}")
        entries[name] = {
            "hex": normalized,
            "float32_be": float(struct.unpack(">f", bytes.fromhex(normalized))[0]),
        }
    normalized_payload = "\n".join(
        f"{name}: {entries[name]['hex']}" for name in sorted(entries)
    ).encode("utf-8")
    return {
        "raw_sha256": sha256_bytes(payload),
        "header": lines[0],
        "entry_count": len(entries),
        "normalized_entries_sha256": sha256_bytes(normalized_payload),
        "entries": entries,
    }


def _layers(inspector: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, layer in enumerate(inspector.get("Layers", [])):
        name = str(layer.get("Name", f"unnamed_{index}"))
        if name in result:
            name = f"{name}#{index}"
        result[name] = layer
    return result


def _dtype_token(value: Any) -> str:
    text = str(value).lower()
    if "int8" in text:
        return "int8"
    if "half" in text or "fp16" in text:
        return "fp16"
    if "float" in text or "fp32" in text:
        return "fp32"
    return "other"


def _precision_signature(layer: dict[str, Any]) -> list[str]:
    values = []
    for side in ("Inputs", "Outputs"):
        for tensor in layer.get(side, []):
            values.append(
                f"{side[:-1].lower()}:{_dtype_token(tensor.get('Format/Datatype'))}"
            )
    return values


def _format_signature(layer: dict[str, Any]) -> list[str]:
    values = []
    for side in ("Inputs", "Outputs"):
        for tensor in layer.get(side, []):
            values.append(f"{side[:-1].lower()}:{tensor.get('Format/Datatype')}")
    return values


def build_runtime_diff(
    *,
    h800_inspector: dict[str, Any],
    orin_inspector: dict[str, Any],
    h800_cache: dict[str, Any],
    orin_cache: dict[str, Any],
) -> dict[str, Any]:
    h800_entries = h800_cache["entries"]
    orin_entries = orin_cache["entries"]
    shared_tensors = sorted(set(h800_entries) & set(orin_entries))
    scale_differences = [
        {
            "tensor": name,
            "h800_hex": h800_entries[name]["hex"],
            "orin_hex": orin_entries[name]["hex"],
        }
        for name in shared_tensors
        if h800_entries[name]["hex"] != orin_entries[name]["hex"]
    ]
    h800_layers = _layers(h800_inspector)
    orin_layers = _layers(orin_inspector)
    shared_layers = sorted(set(h800_layers) & set(orin_layers))
    precision_differences = []
    format_differences = []
    tactic_differences = []
    for name in shared_layers:
        h800_layer = h800_layers[name]
        orin_layer = orin_layers[name]
        h800_precision = _precision_signature(h800_layer)
        orin_precision = _precision_signature(orin_layer)
        if h800_precision != orin_precision:
            precision_differences.append(
                {
                    "layer": name,
                    "h800": h800_precision,
                    "orin": orin_precision,
                }
            )
        h800_format = _format_signature(h800_layer)
        orin_format = _format_signature(orin_layer)
        if h800_format != orin_format:
            format_differences.append(
                {
                    "layer": name,
                    "h800": h800_format,
                    "orin": orin_format,
                }
            )
        h800_tactic = h800_layer.get("TacticName", h800_layer.get("TacticValue"))
        orin_tactic = orin_layer.get("TacticName", orin_layer.get("TacticValue"))
        if h800_tactic != orin_tactic:
            tactic_differences.append(
                {
                    "layer": name,
                    "h800": h800_tactic,
                    "orin": orin_tactic,
                }
            )
    key_sets_identical = set(h800_entries) == set(orin_entries)
    scales_identical = key_sets_identical and not scale_differences
    return {
        "schema_version": "lane_c_trt_runtime_difference_audit_v1",
        "calibration_cache": {
            "h800_header": h800_cache["header"],
            "orin_header": orin_cache["header"],
            "h800_raw_sha256": h800_cache["raw_sha256"],
            "orin_raw_sha256": orin_cache["raw_sha256"],
            "h800_entry_count": h800_cache["entry_count"],
            "orin_entry_count": orin_cache["entry_count"],
            "tensor_key_sets_identical": key_sets_identical,
            "tensor_scales_identical": scales_identical,
            "differing_tensor_scale_count": len(scale_differences),
            "scale_differences": scale_differences,
            "normalized_h800_entries_sha256": h800_cache[
                "normalized_entries_sha256"
            ],
            "normalized_orin_entries_sha256": orin_cache[
                "normalized_entries_sha256"
            ],
        },
        "inspector": {
            "h800_layer_count": len(h800_layers),
            "orin_layer_count": len(orin_layers),
            "shared_layer_count": len(shared_layers),
            "h800_only_layer_count": len(set(h800_layers) - set(orin_layers)),
            "orin_only_layer_count": len(set(orin_layers) - set(h800_layers)),
            "shared_layers_with_precision_difference": len(precision_differences),
            "shared_layers_with_format_difference": len(format_differences),
            "shared_layers_with_tactic_difference": len(tactic_differences),
            "precision_differences": precision_differences,
            "format_differences": format_differences,
            "tactic_differences": tactic_differences,
        },
        "dynamic_range_visibility": (
            "cache_tensor_scales_visible_internal_fused_layer_ranges_not_exposed"
        ),
        "causal_limit": (
            "same calibration tensor scales do not lock TensorRT precision "
            "placement, fusion, formats, kernels, or tactics"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h800-cache", type=Path, required=True)
    parser.add_argument("--orin-cache", type=Path, required=True)
    parser.add_argument("--h800-inspector", type=Path, required=True)
    parser.add_argument("--orin-inspector", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    h800_cache = parse_calibration_cache(args.h800_cache.read_bytes())
    orin_cache = parse_calibration_cache(args.orin_cache.read_bytes())
    h800_inspector = json.loads(args.h800_inspector.read_text(encoding="utf-8"))
    orin_inspector = json.loads(args.orin_inspector.read_text(encoding="utf-8"))
    report = build_runtime_diff(
        h800_inspector=h800_inspector,
        orin_inspector=orin_inspector,
        h800_cache=h800_cache,
        orin_cache=orin_cache,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
