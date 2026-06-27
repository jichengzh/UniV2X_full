#!/usr/bin/env python3
"""Export a quick review table from artifact registry and canonical LUT rows."""

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

from framework.stage2.artifact_registry import validate_artifact_registry_row  # noqa: E402
from framework.stage2.lut_productization import (  # noqa: E402
    energy_claim_allowed_from_rows,
    read_jsonl,
)


CSV_FIELDS = [
    "artifact_id",
    "config_id",
    "label",
    "arm",
    "optimized_scope",
    "artifact_status",
    "ap70",
    "latency_p50_ms",
    "energy_joule_per_inference",
    "latency_quality_flag",
    "latency_claim_status",
    "energy_claim_status",
    "missing_artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-registry", required=True)
    parser.add_argument("--latency-rows", action="append", default=[])
    parser.add_argument("--ap-rows", action="append", default=[])
    parser.add_argument("--energy-rows", action="append", default=[])
    parser.add_argument("--outlier-report")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json")
    return parser.parse_args()


def _read_many(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _by_row_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("row_id")): row for row in rows if row.get("row_id")}


def _outlier_by_row_id(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(row.get("row_id")): row
        for row in report.get("rows", [])
        if row.get("row_id")
    }


def _first_row(ids: list[str], by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    for row_id in ids:
        row = by_id.get(str(row_id))
        if row is not None:
            return row
    return None


def _latency_review(
    ids: list[str],
    by_id: dict[str, dict[str, Any]],
    outlier_rows: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, str]:
    rows = [by_id[row_id] for row_id in ids if row_id in by_id]
    if not rows:
        return None, None, "no_claim"
    quality_rows = [
        outlier_rows.get(str(row.get("row_id")))
        for row in rows
        if outlier_rows.get(str(row.get("row_id"))) is not None
    ]
    claimable = [
        row
        for row in rows
        if outlier_rows.get(str(row.get("row_id")), {}).get("claim_status")
        == "claimable"
    ]
    selected = claimable[0] if claimable else rows[0]
    unstable_quality = [
        row
        for row in quality_rows
        if row.get("claim_status") != "claimable" or row.get("quality_flag") != "stable"
    ]
    selected_quality = outlier_rows.get(str(selected.get("row_id")))
    if unstable_quality and claimable:
        return (
            selected,
            f"has_{unstable_quality[0].get('quality_flag')}",
            "conditional",
        )
    if selected_quality:
        return (
            selected,
            str(selected_quality.get("quality_flag")),
            str(selected_quality.get("claim_status")),
        )
    return selected, None, "unknown"


def _ap70(rows: list[dict[str, Any]]) -> float | None:
    for row in rows:
        if row.get("metric") == "AP70":
            return float(row["metric_value"])
    return float(rows[0]["metric_value"]) if rows else None


def _label(row: dict[str, Any]) -> str:
    candidate_id = str(row.get("candidate_id") or "")
    if ":" in candidate_id:
        return candidate_id.split(":")[-1]
    return str(row.get("label") or row.get("config_id") or "")


def _alias_key(row: dict[str, Any]) -> tuple[str, tuple[int, ...], str, str]:
    return (
        _label(row),
        tuple(int(item) for item in row.get("width", []) or []),
        str(row.get("quant_policy") or ""),
        str(row.get("optimized_scope") or ""),
    )


def _ap_alias_index(ap_rows: list[dict[str, Any]]) -> dict[tuple[str, tuple[int, ...], str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, tuple[int, ...], str, str], list[dict[str, Any]]] = {}
    for row in ap_rows:
        index.setdefault(_alias_key(row), []).append(row)
    return index


def _latency_ms(row: dict[str, Any] | None) -> float | None:
    if row is None or row.get("latency_p50_us") is None:
        return None
    return float(row["latency_p50_us"]) / 1000.0


def _energy_joule(row: dict[str, Any] | None) -> float | None:
    if row is None or row.get("joule_per_inference") is None:
        return None
    return float(row["joule_per_inference"])


def build_quick_review_rows(
    *,
    artifact_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
    energy_rows: list[dict[str, Any]],
    outlier_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    latency_by_id = _by_row_id(latency_rows)
    ap_by_id = _by_row_id(ap_rows)
    ap_by_alias = _ap_alias_index(ap_rows)
    energy_by_id = _by_row_id(energy_rows)
    rows: list[dict[str, Any]] = []
    for artifact in artifact_rows:
        validate_artifact_registry_row(artifact)
        latency, latency_quality_flag, latency_claim_status = _latency_review(
            list(artifact["latency_row_ids"]),
            latency_by_id,
            outlier_rows,
        )
        energy = _first_row(list(artifact["energy_row_ids"]), energy_by_id)
        artifact_ap_rows = [
            ap_by_id[row_id]
            for row_id in artifact["ap_row_ids"]
            if row_id in ap_by_id
        ]
        if not artifact_ap_rows:
            artifact_ap_rows = ap_by_alias.get(
                (
                    str(artifact["label"]),
                    tuple(int(item) for item in artifact.get("width", []) or []),
                    str(latency.get("quant_policy") if latency else ""),
                    str(artifact.get("optimized_scope") or ""),
                ),
                [],
            )
        energy_claim_status = (
            "claimable"
            if energy is not None and energy_claim_allowed_from_rows([energy])
            else "no_claim"
        )
        rows.append(
            {
                "artifact_id": artifact["artifact_id"],
                "config_id": artifact["config_id"],
                "label": artifact["label"],
                "arm": artifact["arm"],
                "optimized_scope": artifact["optimized_scope"],
                "artifact_status": artifact["artifact_status"],
                "ap70": _ap70(artifact_ap_rows),
                "latency_p50_ms": _latency_ms(latency),
                "energy_joule_per_inference": _energy_joule(energy),
                "latency_quality_flag": latency_quality_flag,
                "latency_claim_status": latency_claim_status,
                "energy_claim_status": energy_claim_status,
                "missing_artifacts": ";".join(artifact["missing_artifacts"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def main() -> int:
    args = parse_args()
    artifact_rows = read_jsonl(args.artifact_registry)
    rows = build_quick_review_rows(
        artifact_rows=artifact_rows,
        latency_rows=_read_many(args.latency_rows),
        ap_rows=_read_many(args.ap_rows),
        energy_rows=_read_many(args.energy_rows),
        outlier_rows=_outlier_by_row_id(args.outlier_report),
    )
    out_csv = Path(args.out_csv)
    _write_csv(out_csv, rows)
    payload = {
        "schema": "stage2_quick_review_v1",
        "rows": rows,
        "total_rows": len(rows),
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "schema": "stage2_quick_review_export_summary_v1",
                "out_csv": str(out_csv),
                "rows": len(rows),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
