#!/usr/bin/env python3
"""Write Stage2 H800 TVM INT8 artifact route status rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import utc_timestamp  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)
LABELS = {
    "base": [64, 128, 256],
    "s0_024": [24, 128, 256],
    "s1_048": [64, 48, 256],
}
QUANT_METHOD = "h800_tvm_int8_backbone_subnet_experimental"
QUANT_SCOPE = "backbone_only_requested"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--probe-json")
    parser.add_argument("--created-at")
    return parser.parse_args()


def read_probe(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    item = Path(path)
    if not item.exists():
        return {}
    payload = json.loads(item.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def sha256_file(path: str | None) -> str:
    if not path:
        return "missing"
    item = Path(path)
    if not item.is_file():
        return "unknown"
    digest = hashlib.sha256()
    with item.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_probe(probe: dict[str, Any], label: str) -> dict[str, Any]:
    status = str(probe.get("status") or "")
    if status and status != "succeeded":
        detail = probe.get("stderr") or probe.get("error") or probe.get("stdout_tail") or ""
        return {
            "probe_source": probe.get("probe_source") or "h800_probe",
            "error": f"{status}: {detail}".strip(),
            "build_status": "probe_failed",
            "validation_status": "not_run_probe_failed",
        }
    labels = probe.get("labels")
    if isinstance(labels, dict) and isinstance(labels.get(label), dict):
        return dict(labels[label])
    if isinstance(probe.get(label), dict):
        return dict(probe[label])
    return {}


def first_existing_path(data: dict[str, Any]) -> str | None:
    for key in (
        "compiled_artifact_path",
        "quantized_model_path",
        "tvm_compiled_artifact_path",
        "artifact_path",
    ):
        value = data.get(key)
        if value:
            return str(value)
    return None


def has_true(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "ready", "exists"}


def artifact_row(label: str, width: list[int], probe: dict[str, Any], created_at: str) -> dict[str, Any]:
    data = label_probe(probe, label)
    artifact_path = first_existing_path(data)
    has_artifact = bool(artifact_path) and (
        has_true(data, "artifact_exists")
        or has_true(data, "compiled_artifact_exists")
        or has_true(data, "quantized_model_exists")
    )
    has_calibration = bool(data.get("calibration_manifest_path")) and has_true(
        data, "calibration_manifest_exists"
    )
    has_recipe = bool(data.get("quant_recipe_path")) and has_true(data, "quant_recipe_exists")
    has_layer_summary = bool(data.get("layer_precision_summary_path")) and has_true(
        data, "layer_precision_summary_exists"
    )
    ready = has_artifact and has_calibration and has_recipe and has_layer_summary
    explicit_error = str(data.get("error") or data.get("blocker") or "")
    if ready:
        artifact_status = "ready"
        build_status = str(data.get("build_status") or "succeeded")
        validation_status = str(data.get("validation_status") or "artifact_files_present")
        blocker = ""
    elif explicit_error:
        artifact_status = "quarantine"
        build_status = str(data.get("build_status") or "failed")
        validation_status = str(data.get("validation_status") or "not_run")
        blocker = explicit_error
    else:
        artifact_status = "missing"
        build_status = str(data.get("build_status") or "missing_tvm_int8_backbone_subnet_route")
        validation_status = str(data.get("validation_status") or "not_run_missing_artifact")
        blocker = "missing calibration_manifest/quant_recipe/layer_precision_summary/quantized_or_compiled_artifact"
    return {
        "schema": "tvm_int8_artifact_status_row_v1",
        "label": label,
        "model": "pyramid_lidar",
        "width": width,
        "precision": "int8",
        "backend": "h800_tvm",
        "engine_kind": "tvm_vm",
        "quant_method": QUANT_METHOD,
        "quant_scope": QUANT_SCOPE,
        "full_network_claim": False,
        "artifact_status": artifact_status,
        "artifact_path": artifact_path,
        "artifact_digest": str(data.get("artifact_digest") or sha256_file(artifact_path)),
        "engine_digest": str(data.get("engine_digest") or data.get("artifact_digest") or sha256_file(artifact_path)),
        "build_status": build_status,
        "validation_status": validation_status,
        "calibration_manifest_path": data.get("calibration_manifest_path"),
        "calibration_manifest_digest": str(
            data.get("calibration_manifest_digest") or sha256_file(data.get("calibration_manifest_path"))
        ),
        "quant_recipe_path": data.get("quant_recipe_path"),
        "quant_recipe_digest": str(data.get("quant_recipe_digest") or sha256_file(data.get("quant_recipe_path"))),
        "layer_precision_summary_path": data.get("layer_precision_summary_path"),
        "layer_precision_summary_digest": str(
            data.get("layer_precision_summary_digest")
            or sha256_file(data.get("layer_precision_summary_path"))
        ),
        "quantized_model_path": data.get("quantized_model_path"),
        "quantized_onnx_digest": str(
            data.get("quantized_onnx_digest") or sha256_file(data.get("quantized_model_path"))
        ),
        "compiled_artifact_path": data.get("compiled_artifact_path"),
        "blocker": blocker,
        "created_at": created_at,
        "probe_source": probe.get("probe_source") or data.get("probe_source") or "local_default_missing_probe",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_exports(output_root: Path, rows: list[dict[str, Any]]) -> None:
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "width",
        "precision",
        "quant_method",
        "quant_scope",
        "artifact_status",
        "artifact_path",
        "artifact_digest",
        "build_status",
        "validation_status",
        "blocker",
    ]
    with (exports / "tvm_int8_artifact_status_latest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    payload = {
        "schema": "tvm_int8_artifact_status_summary_v1",
        "labels": [row["label"] for row in rows],
        "total_artifacts": len(rows),
        "artifact_status_counts": dict(sorted(Counter(row["artifact_status"] for row in rows).items())),
        "build_status_counts": dict(sorted(Counter(row["build_status"] for row in rows).items())),
        "rows": rows,
    }
    (exports / "tvm_int8_artifact_status_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TVM INT8 Artifact Route Status",
        "",
        "| label | width | artifact_status | build_status | validation_status | artifact_path | blocker |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {width} | {artifact_status} | {build_status} | {validation_status} | {artifact_path} | {blocker} |".format(
                label=row["label"],
                width="x".join(str(item) for item in row["width"]),
                artifact_status=row["artifact_status"],
                build_status=row["build_status"],
                validation_status=row["validation_status"],
                artifact_path=row.get("artifact_path") or "",
                blocker=row.get("blocker") or "",
            )
        )
    (exports / "tvm_int8_artifact_status_latest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    probe = read_probe(args.probe_json)
    created_at = args.created_at or utc_timestamp()
    rows = [artifact_row(label, width, probe, created_at) for label, width in LABELS.items()]
    write_jsonl(output_root / "artifacts/tvm_int8_artifact_registry_v1.jsonl", rows)
    write_exports(output_root, rows)
    print(
        json.dumps(
            {
                "schema": "tvm_int8_artifact_route_probe_result_v1",
                "output_root": str(output_root),
                "rows": len(rows),
                "status_counts": dict(sorted(Counter(row["artifact_status"] for row in rows).items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
