#!/usr/bin/env python3
"""Build separate TVM/TRT Stage6 paper tables at frozen AP-loss thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage6.paper_table_v1 import build_backend_tables


FIELDS = (
    "method", "backend", "config", "AP70", "latency_ms", "energy_j", "speedup",
    "HV", "pareto_count", "outer_genomes", "tuning_trials", "gpu_hours",
    "wallclock_s", "failure_rate", "selection_status", "outcome", "failure_reason",
    "ap_constraint_violated", "latency_rank",
)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, list):
        return "(" + ",".join(map(str, value)) + ")"
    return str(value)


def _markdown(rows: list[dict[str, Any]], backend: str, delta: float, floor: float) -> str:
    lines = [
        f"# Pyramid Stage6 {backend.upper()} main table (DeltaAP_max={delta:.2f})",
        "",
        f"AP70 floor: `{floor:.6f}`. Only independently validated, SHA-verified measurements are eligible.",
        "",
        "| Method | Outcome | Config | AP70 ↑ | Latency ms ↓ | Energy J ↓ | Speedup ↑ | HV ↑ | Budget | Fail rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        method = str(row["method"])
        if row.get("latency_rank") == 1:
            method = f"**{method}**"
        elif row.get("latency_rank") == 2:
            method = f"<u>{method}</u>"
        ap70 = _fmt(row.get("AP70"))
        if row.get("ap_constraint_violated") is True:
            ap70 = f"**{ap70}**"
        lines.append(
            "| " + " | ".join(
                [
                    method,
                    _fmt(row.get("outcome")),
                    _fmt(row.get("config")),
                    ap70,
                    _fmt(row.get("latency_ms")),
                    _fmt(row.get("energy_j")),
                    (_fmt(row.get("speedup")) + "x") if row.get("speedup") is not None else "-",
                    _fmt(row.get("HV")),
                    _fmt(row.get("outer_genomes"), 0),
                    (_fmt(100 * row["failure_rate"], 2) + "%") if row.get("failure_rate") is not None else "-",
                ]
            ) + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--delta-ap-max",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20),
    )
    args = parser.parse_args()
    bundle = json.loads(args.evidence_bundle.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "stage6_paper_evidence_bundle_v1":
        raise ValueError("unexpected evidence bundle schema")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    all_ready = True
    for backend in ("tvm", "trt"):
        baseline = (bundle.get("backend_baselines") or {}).get(
            backend,
            bundle["baseline"],
        )
        result = build_backend_tables(
            backend=backend,
            baseline=baseline,
            arms=bundle["backends"][backend],
            deltas=tuple(args.delta_ap_max),
        )
        outputs[backend] = result
        all_ready = all_ready and result["paper_ready"]
        for key, rows in result["tables"].items():
            delta = float(key.removeprefix("delta_"))
            stem = f"pyramid_stage6_{backend}_delta_ap_{delta:.2f}"
            with (args.output_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            (args.output_dir / f"{stem}.md").write_text(
                _markdown(rows, backend, delta, result["baseline_ap70"] - delta), encoding="utf-8"
            )
    audit = {
        "schema_version": "stage6_paper_main_table_audit_v1",
        "paper_ready": all_ready,
        "selection_contract": {
            "delta_ap_max": list(args.delta_ap_max),
            "latency_close_fraction": 0.01,
            "tie_breaker": "minimum_energy_within_one_percent_of_minimum_latency",
            "trusted_points_only": True,
        },
        "backends": outputs,
    }
    (args.output_dir / "stage6_paper_main_table_audit_v1.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"paper_ready": all_ready, "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if all_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
