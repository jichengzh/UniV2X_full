#!/usr/bin/env python3
"""Record Stage2 FP32 H800 TVM latency smoke preflight status."""

from __future__ import annotations

import argparse
import csv
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
    return payload if isinstance(payload, dict) else {}


def preflight_status(probe: dict[str, Any]) -> tuple[str, str]:
    status = str(probe.get("status") or "")
    if status and status != "succeeded":
        detail = probe.get("stderr") or probe.get("error") or probe.get("stdout_tail") or ""
        return "ssh_failed_preflight_blocked", f"{status}: {detail}".strip()
    if not probe:
        return "not_run_missing_probe", "no h800 probe json supplied"
    gpu_query = probe.get("gpu_query")
    if isinstance(gpu_query, dict) and int(gpu_query.get("returncode") or 0) != 0:
        return "gpu_query_failed", str(gpu_query.get("stderr") or "")
    return "ready_for_manual_fp32_smoke", "readonly probe succeeded; measured job not launched"


def build_rows(probe: dict[str, Any], created_at: str) -> list[dict[str, Any]]:
    status, blocker = preflight_status(probe)
    rows = []
    for label, width in LABELS.items():
        rows.append(
            {
                "schema": "fp32_latency_smoke_preflight_row_v1",
                "label": label,
                "model": "pyramid_lidar",
                "width": width,
                "precision": "fp32",
                "backend": "h800_tvm",
                "engine_kind": "tvm_vm",
                "quant_method": "h800_tvm_relax_fp32",
                "quant_scope": "backbone_only",
                "full_network_claim": False,
                "preflight_status": status,
                "gpu_idle_gate_status": "not_reached" if status != "ready_for_manual_fp32_smoke" else "readonly_probe_only",
                "measurement_status": "not_launched",
                "blocker": blocker,
                "created_at": created_at,
            }
        )
    return rows


def write_exports(output_root: Path, rows: list[dict[str, Any]]) -> None:
    exports = output_root / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "width",
        "precision",
        "backend",
        "quant_method",
        "quant_scope",
        "preflight_status",
        "gpu_idle_gate_status",
        "measurement_status",
        "blocker",
    ]
    with (exports / "fp32_latency_smoke_preflight_latest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    payload = {
        "schema": "fp32_latency_smoke_preflight_summary_v1",
        "labels": [row["label"] for row in rows],
        "total_preflight_rows": len(rows),
        "status_counts": dict(sorted(Counter(row["preflight_status"] for row in rows).items())),
        "rows": rows,
    }
    (exports / "fp32_latency_smoke_preflight_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# FP32 H800 TVM Latency Smoke Preflight",
        "",
        "| label | width | preflight_status | gpu_idle_gate_status | measurement_status | blocker |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {width} | {preflight_status} | {gpu_idle_gate_status} | {measurement_status} | {blocker} |".format(
                label=row["label"],
                width="x".join(str(item) for item in row["width"]),
                preflight_status=row["preflight_status"],
                gpu_idle_gate_status=row["gpu_idle_gate_status"],
                measurement_status=row["measurement_status"],
                blocker=row["blocker"],
            )
        )
    (exports / "fp32_latency_smoke_preflight_latest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    probe = read_probe(args.probe_json)
    rows = build_rows(probe, args.created_at or utc_timestamp())
    write_exports(output_root, rows)
    print(
        json.dumps(
            {
                "schema": "fp32_latency_smoke_preflight_result_v1",
                "output_root": str(output_root),
                "rows": len(rows),
                "status_counts": dict(sorted(Counter(row["preflight_status"] for row in rows).items())),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
