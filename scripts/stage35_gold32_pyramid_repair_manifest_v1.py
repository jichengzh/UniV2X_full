#!/usr/bin/env python3
"""Overlay corrected Pyramid ONNX paths onto the frozen Gold32 manifest."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def _padded_width(width: list[int]) -> str:
    if len(width) != 3:
        raise ValueError(f"expected three Pyramid widths, got {width}")
    return "x".join(f"{int(value):03d}" for value in width)


def apply_pyramid_shape_repair(manifest: Mapping[str, Any], repaired_root: str | Path) -> dict[str, Any]:
    if manifest.get("schema_version") != "stage35_gold32_supplement_manifest_v1":
        raise ValueError("unexpected Gold32 manifest schema")
    repaired = copy.deepcopy(dict(manifest))
    rows = repaired.get("jobs")
    if not isinstance(rows, list) or len(rows) != 32:
        raise ValueError("Gold32 manifest must contain 32 rows")
    root = Path(repaired_root)
    repaired_count = 0
    for row in rows:
        if row.get("model") != "pyramid":
            continue
        width = [int(value) for value in row["width"]]
        padded = _padded_width(width)
        source = row["source_contract"]
        old_path = str(source["onnx_path"])
        source["supersedes_onnx_path"] = old_path
        source["onnx_path"] = str(root / padded / f"pyramid_{padded}_multiscale.onnx")
        source["onnx_report_path"] = str(root / padded / "onnx_export_report.json")
        source["source_kind"] = "checkpoint_consistent_pyramid_multiscale_shape_repaired_v1"
        row["source_status"] = "shape_contract_repaired_pending_measurement"
        repaired_count += 1
    repaired["repair_overlay"] = {
        "repair_id": "pyramid_static_input_2x64x128x256_v1",
        "reason": "original Pyramid supplement ONNX used synthetic 1x64x256x256 input",
        "input_shape": [2, 64, 128, 256],
        "repaired_model": "pyramid",
        "repaired_row_count": repaired_count,
        "repaired_root": str(root),
        "historical_performance_policy": "superseded_for_all_pyramid_rows",
    }
    if repaired_count != 16:
        raise ValueError(f"expected 16 Pyramid rows, repaired {repaired_count}")
    return repaired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--repaired-root", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    repaired = apply_pyramid_shape_repair(source, args.repaired_root)
    content = json.dumps(repaired, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(content, encoding="utf-8")
    print(json.dumps({
        "output_json": str(args.output_json.resolve()),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "repaired_row_count": repaired["repair_overlay"]["repaired_row_count"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
