#!/usr/bin/env python3
"""Phase1 QxS v2 search entry.

V2 keeps compiler/backend choices out of the genome.  The genome is
``(w0,w1,w2,strategy_id)`` where ``strategy_id`` is built only from
``q_mode`` and ``mixed_policy_id``.  Compiler backends are context produced by
the H800 capability/evidence scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path("/home/jichengzhi/V2X")
DEFAULT_BACKEND_SCAN_OUT = REPO / "results/h800_compiler_backend_capability_scan_20260708.json"
DEFAULT_STRATEGY_MANIFEST_OUT = REPO / "results/phase1_qxs_strategy_space_manifest_20260708.json"
DEFAULT_TRAINING_TABLE_OUT = REPO / "results/phase1_qxs_training_table_context_backend_20260708.json"
DEFAULT_ENTRY_OUT = REPO / "results/phase1_qxs_smbo_entry_v2_20260708.json"
DEFAULT_COST_MODEL_SMOKE_OUT = REPO / "results/phase1_qxs_v2_cost_model_smoke_20260708.json"

DEFAULT_TVM_MIXED_SUMMARY = REPO / "results/codriving_routeb_mixed_qxs8_summary_20260708.json"
DEFAULT_TRT_QXS_DIR = REPO / "results/codriving_trt_qxs8_20260708"
DEFAULT_CUTLASS_FP16_SUMMARY = REPO / "results/codriving_qxs_g4_s5_buildrun_summary_20260708.json"
DEFAULT_CUTLASS_INT8_PROBE_SUMMARY = REPO / "results/codriving_qxs_g4_s5_int8_probe_summary_20260708.json"

DEFAULT_WIDTHS = ("16x32x64", "32x32x128", "64x128x256")
MIXED_POLICY_IDS = (
    "none",
    "top10_flops",
    "top25_flops",
    "top50_flops",
    "top75_flops",
    "all_eligible_conv",
)
ACTIVE_INT8_MIXED_POLICIES = tuple(policy for policy in MIXED_POLICY_IDS if policy != "none")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rows(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return list(payload)
    return list(payload.get("rows", []))


def parse_width(value: Any) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        parts = [int(item) for item in value]
    elif isinstance(value, str):
        parts = [int(item.strip()) for item in value.replace("x", ",").split(",") if item.strip()]
    else:
        raise TypeError(f"unsupported width value: {value!r}")
    if len(parts) != 3:
        raise ValueError(f"width must have 3 parts, got {value!r}")
    return parts[0], parts[1], parts[2]


def width_str(width: Any) -> str:
    w0, w1, w2 = parse_width(width)
    return f"{w0}x{w1}x{w2}"


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.lower() in {"true", "conditional", "yes", "1"}
    return False


def strategy_id(q_mode: str, mixed_policy_id: str) -> str:
    q = str(q_mode).lower()
    mixed = str(mixed_policy_id)
    if q == "fp16":
        mixed = "none"
    return f"q={q}|mixed={mixed}"


def _profile_by_id(backend_scan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(profile["compiler_profile_id"]): profile
        for profile in backend_scan.get("compiler_backend_profiles", [])
    }


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _support_as_float(value: Any) -> float:
    if value is True:
        return 1.0
    if value is False or value is None:
        return 0.0
    text = str(value).lower()
    if text in {"true", "pass", "success", "yes"}:
        return 1.0
    if text in {"partial", "unknown_or_partial", "backend_managed_qdq"}:
        return 0.5
    return 0.0


def build_h800_compiler_backend_scan_from_results(results_dir: Path = REPO / "results") -> dict[str, Any]:
    tvm_summary_path = results_dir / DEFAULT_TVM_MIXED_SUMMARY.relative_to(REPO / "results")
    trt_dir = results_dir / DEFAULT_TRT_QXS_DIR.relative_to(REPO / "results")
    cutlass_fp16_path = results_dir / DEFAULT_CUTLASS_FP16_SUMMARY.relative_to(REPO / "results")
    cutlass_int8_probe_path = results_dir / DEFAULT_CUTLASS_INT8_PROBE_SUMMARY.relative_to(REPO / "results")

    tvm_rows = _rows(load_json(tvm_summary_path)) if tvm_summary_path.is_file() else []
    trt_files = sorted(trt_dir.glob("codriving_trt_*_*.json")) if trt_dir.is_dir() else []
    cutlass_fp16 = load_json(cutlass_fp16_path) if cutlass_fp16_path.is_file() else {}
    cutlass_int8_probe = load_json(cutlass_int8_probe_path) if cutlass_int8_probe_path.is_file() else {}

    tvm_int8_slower = sum(
        1
        for row in tvm_rows
        if _float_or_none(row.get("pure_int8_over_fp16")) is not None
        and float(row["pure_int8_over_fp16"]) > 1.0
    )
    tvm_top25_faster = sum(
        1
        for row in tvm_rows
        if _float_or_none(row.get("mixed_top25_flops_over_fp16")) is not None
        and float(row["mixed_top25_flops_over_fp16"]) < 1.0
    )

    trt_fp16_files = [path for path in trt_files if "_fp16_" in path.name]
    trt_int8_files = [path for path in trt_files if "_int8_" in path.name]
    cutlass_probe_rows = _rows(cutlass_int8_probe)
    cutlass_int8_entry_ok = any(
        row.get("status") == "success" and str(row.get("case", "")).startswith(("s8", "u8", "int8_single"))
        for row in cutlass_probe_rows
    )
    cutlass_boundary_probe_ok = any(
        row.get("status") == "success" and "boundary" in str(row.get("case", ""))
        for row in cutlass_probe_rows
    )

    return {
        "schema_version": "h800_compiler_backend_capability_scan_v1",
        "created_at_utc": utc_now(),
        "scan_mode": "evidence_scan_from_existing_h800_measurements",
        "hardware_target": "h800",
        "note": "Profiles summarize existing H800 build/run evidence; they are context features, not search arms.",
        "compiler_backend_profiles": [
            {
                "compiler_profile_id": "h800_tvm_routeb_20260708",
                "backend": "tvm",
                "backend_family": "tvm_routeb",
                "supports_fp16": bool(tvm_rows),
                "supports_int8": bool(tvm_rows),
                "supports_mixed_int8": any(row.get("mixed_top25_flops_ms") not in (None, "") for row in tvm_rows),
                "supports_cutlass_byoc": False,
                "supports_group_conv": "unknown_for_codriving_routeb",
                "supports_conv2d_transpose": "not_in_routeb_backbone_only_probe",
                "supports_requant_fusion": "partial",
                "supports_cast_fusion": "partial",
                "measurement_confidence": "real_measurement_summary",
                "probe_results": {
                    "source": str(tvm_summary_path),
                    "n_widths": len(tvm_rows),
                    "pure_int8_slower_than_fp16_count": tvm_int8_slower,
                    "mixed_top25_faster_than_fp16_count": tvm_top25_faster,
                },
            },
            {
                "compiler_profile_id": "h800_trt_20260708",
                "backend": "tensorrt",
                "backend_family": "tensorrt_engine",
                "supports_fp16": bool(trt_fp16_files),
                "supports_int8": bool(trt_int8_files),
                "supports_mixed_int8": "unknown",
                "supports_cutlass_byoc": False,
                "supports_group_conv": "backend_managed",
                "supports_conv2d_transpose": "backend_managed",
                "supports_requant_fusion": "backend_managed_qdq",
                "supports_cast_fusion": "backend_managed_qdq",
                "measurement_confidence": "real_measurement_summary",
                "probe_results": {
                    "source_dir": str(trt_dir),
                    "n_fp16_measurements": len(trt_fp16_files),
                    "n_int8_measurements": len(trt_int8_files),
                },
            },
            {
                "compiler_profile_id": "h800_tvm_cutlass_byoc_20260708",
                "backend": "tvm_cutlass_byoc",
                "backend_family": "tvm_cutlass_byoc",
                "supports_fp16": bool(cutlass_fp16.get("all_build_and_run")),
                "supports_int8": bool(cutlass_int8_entry_ok),
                "supports_mixed_int8": "probe_only_not_fullgraph",
                "supports_cutlass_byoc": True,
                "supports_group_conv": "unknown",
                "supports_conv2d_transpose": False,
                "supports_requant_fusion": "probe_only",
                "supports_cast_fusion": "probe_only",
                "measurement_confidence": "probe_not_final_frontier",
                "probe_results": {
                    "fp16_source": str(cutlass_fp16_path),
                    "int8_probe_source": str(cutlass_int8_probe_path),
                    "fp16_fullgraph_success_count": cutlass_fp16.get("n_success"),
                    "int8_entry_probe_success": bool(cutlass_int8_entry_ok),
                    "int8_boundary_probe_success": bool(cutlass_boundary_probe_ok),
                    "fullgraph_int8_status": "not_closed",
                },
            },
        ],
    }


def build_strategy_manifest(
    widths: Iterable[Any],
    backend_scan: dict[str, Any],
    *,
    mixed_policy_ids: Iterable[str] = MIXED_POLICY_IDS,
) -> dict[str, Any]:
    strategies = [{"strategy_id": strategy_id("fp16", "none"), "q_mode": "fp16", "mixed_policy_id": "none"}]
    for policy in mixed_policy_ids:
        if policy == "none":
            continue
        strategies.append({"strategy_id": strategy_id("int8", policy), "q_mode": "int8", "mixed_policy_id": policy})

    return {
        "schema_version": "phase1_qxs_strategy_space_manifest_v2",
        "created_at_utc": utc_now(),
        "genome_schema": ["w0", "w1", "w2", "strategy_id"],
        "widths": [width_str(width) for width in widths],
        "strategy_space": {
            "q_mode": ["fp16", "int8"],
            "mixed_policy_id": list(mixed_policy_ids),
            "valid_strategy_constraints": [
                "q_mode=fp16 -> mixed_policy_id=none",
                "q_mode=int8 -> mixed_policy_id!=none",
            ],
        },
        "graph_boundary_policy": {
            "primary_search_axis": False,
            "status": "derived_diagnostic_feature",
            "derived_features": [
                "dtype_boundary_count",
                "requant_count",
                "cast_count",
                "materialized_bytes",
                "fusion_break_count",
                "partition_count",
            ],
        },
        "strategies": strategies,
        "compiler_backend_profiles": list(backend_scan.get("compiler_backend_profiles", [])),
        "forbidden_strategy_id_fields": [
            "backend",
            "compiler_profile_id",
            "schedule_route",
            "graph_boundary",
            "trusted_for_final_frontier",
            "source_confidence",
            "historical_prior_flag",
        ],
    }


def _default_derived_features() -> dict[str, Any]:
    return {
        "dtype_boundary_count": None,
        "requant_count": None,
        "cast_count": None,
        "materialized_bytes": None,
        "fusion_break_count": None,
        "partition_count": None,
        "mixed_int8_conv_count": None,
        "mixed_fp16_conv_count": None,
        "feature_status": "not_recorded_in_raw_result",
    }


def _row(
    *,
    width: str,
    q_mode: str,
    mixed_policy_id: str,
    compiler_profile_id: str,
    backend: str,
    latency_ms: float,
    source_file: str,
    realized_schedule_route: str,
    derived_graph_features: dict[str, Any] | None = None,
    measurement_source: str = "real",
    trusted_for_final_frontier: bool | str = True,
) -> dict[str, Any]:
    w0, w1, w2 = parse_width(width)
    return {
        "width": f"{w0}x{w1}x{w2}",
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "strategy_id": strategy_id(q_mode, mixed_policy_id),
        "q_mode": q_mode,
        "mixed_policy_id": "none" if q_mode == "fp16" else mixed_policy_id,
        "compiler_profile_id": compiler_profile_id,
        "backend": backend,
        "latency_ms": float(latency_ms),
        "measurement_source": measurement_source,
        "trusted_for_final_frontier": trusted_for_final_frontier,
        "source_file": source_file,
        "realized_schedule_route": realized_schedule_route,
        "derived_graph_features": derived_graph_features or _default_derived_features(),
    }


def _pure_features(int8_convs: int | None = None, fp16_convs: int | None = None) -> dict[str, Any]:
    features = _default_derived_features()
    features.update(
        {
            "mixed_int8_conv_count": int8_convs,
            "mixed_fp16_conv_count": fp16_convs,
            "feature_status": "pure_policy_no_mixed_boundary_counts_recorded",
        }
    )
    return features


def _mixed_features(summary_row: dict[str, Any], prefix: str) -> dict[str, Any]:
    features = _default_derived_features()
    features.update(
        {
            "mixed_int8_conv_count": _int_or_none(summary_row.get(f"{prefix}_int8_convs")),
            "mixed_fp16_conv_count": _int_or_none(summary_row.get(f"{prefix}_fp16_convs")),
            "feature_status": "partial_raw_counts_only",
        }
    )
    return features


def training_rows_from_tvm_mixed_summary_rows(
    summary_rows: list[dict[str, Any]],
    *,
    compiler_profile_id: str = "h800_tvm_routeb_20260708",
    backend: str = "tvm",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary_rows:
        width = width_str(item["width_key"])
        total_convs = None
        top25_int8 = _int_or_none(item.get("mixed_top25_flops_int8_convs"))
        top25_fp16 = _int_or_none(item.get("mixed_top25_flops_fp16_convs"))
        if top25_int8 is not None and top25_fp16 is not None:
            total_convs = top25_int8 + top25_fp16

        fp16_ms = _float_or_none(item.get("fp16_ms"))
        if fp16_ms is not None:
            rows.append(
                _row(
                    width=width,
                    q_mode="fp16",
                    mixed_policy_id="none",
                    compiler_profile_id=compiler_profile_id,
                    backend=backend,
                    latency_ms=fp16_ms,
                    source_file=str(item.get("fp16_source") or "codriving_routeb_mixed_qxs8_summary_20260708"),
                    realized_schedule_route="routeb_fp16_auto_tensorcore",
                    derived_graph_features=_pure_features(0, total_convs),
                )
            )

        int8_ms = _float_or_none(item.get("int8_ms"))
        if int8_ms is not None:
            rows.append(
                _row(
                    width=width,
                    q_mode="int8",
                    mixed_policy_id="all_eligible_conv",
                    compiler_profile_id=compiler_profile_id,
                    backend=backend,
                    latency_ms=int8_ms,
                    source_file=str(item.get("int8_source") or "codriving_routeb_mixed_qxs8_summary_20260708"),
                    realized_schedule_route="routeb_int8_auto_decomp_tensorcore",
                    derived_graph_features=_pure_features(total_convs, 0),
                )
            )

        top25_ms = _float_or_none(item.get("mixed_top25_flops_ms"))
        if top25_ms is not None:
            rows.append(
                _row(
                    width=width,
                    q_mode="int8",
                    mixed_policy_id="top25_flops",
                    compiler_profile_id=compiler_profile_id,
                    backend=backend,
                    latency_ms=top25_ms,
                    source_file=str(item.get("mixed_top25_flops_source") or "codriving_routeb_mixed_qxs8_summary_20260708"),
                    realized_schedule_route="routeb_mixed_top25_flops_tensorcore",
                    derived_graph_features=_mixed_features(item, "mixed_top25_flops"),
                )
            )

        top50_ms = _float_or_none(item.get("mixed_top50_ms"))
        if top50_ms is not None:
            rows.append(
                _row(
                    width=width,
                    q_mode="int8",
                    mixed_policy_id="top50_flops",
                    compiler_profile_id=compiler_profile_id,
                    backend=backend,
                    latency_ms=top50_ms,
                    source_file=str(item.get("mixed_top50_source") or "codriving_routeb_mixed_qxs8_summary_20260708"),
                    realized_schedule_route="routeb_mixed_top50_flops_tensorcore",
                    derived_graph_features=_mixed_features(item, "mixed_top50"),
                )
            )
    return rows


_TRT_NAME_RE = re.compile(r"codriving_trt_(?P<width>\d+x\d+x\d+)_(?P<precision>fp16|int8)_20260708\.json$")


def training_rows_from_trt_qxs_dir(
    trt_dir: Path,
    *,
    compiler_profile_id: str = "h800_trt_20260708",
    backend: str = "tensorrt",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(trt_dir.glob("codriving_trt_*_*.json")):
        match = _TRT_NAME_RE.match(path.name)
        if not match:
            continue
        payload = load_json(path)
        if not payload.get("build_success", True):
            continue
        latency_ms = _float_or_none(payload.get("lat_p50_ms"))
        if latency_ms is None:
            continue
        precision = match.group("precision")
        q_mode = "fp16" if precision == "fp16" else "int8"
        mixed_policy_id = "none" if precision == "fp16" else "all_eligible_conv"
        rows.append(
            _row(
                width=match.group("width"),
                q_mode=q_mode,
                mixed_policy_id=mixed_policy_id,
                compiler_profile_id=compiler_profile_id,
                backend=backend,
                latency_ms=latency_ms,
                source_file=str(path),
                realized_schedule_route=f"trt_{precision}_engine",
                derived_graph_features=_default_derived_features(),
            )
        )
    return rows


def _filter_widths(rows: list[dict[str, Any]], selected_widths: Iterable[Any]) -> list[dict[str, Any]]:
    allowed = {width_str(item) for item in selected_widths}
    return [row for row in rows if width_str(row["width"]) in allowed]


def build_context_training_rows(
    *,
    tvm_summary: dict[str, Any],
    trt_dir: Path,
    selected_widths: Iterable[Any] = DEFAULT_WIDTHS,
) -> list[dict[str, Any]]:
    tvm_rows = training_rows_from_tvm_mixed_summary_rows(_rows(tvm_summary))
    trt_rows = training_rows_from_trt_qxs_dir(trt_dir)
    return _filter_widths(tvm_rows + trt_rows, selected_widths)


def build_feature_spec(backend_scan: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = _profile_by_id(backend_scan)
    q_modes = _ordered_unique(["fp16", "int8"] + [row.get("q_mode") for row in rows])
    mixed_policies = _ordered_unique(list(MIXED_POLICY_IDS) + [row.get("mixed_policy_id") for row in rows])
    backends = _ordered_unique([profile.get("backend") for profile in profiles.values()] + [row.get("backend") for row in rows])
    feature_order = (
        ["w0", "w1", "w2"]
        + [f"q_mode={item}" for item in q_modes]
        + [f"mixed_policy_id={item}" for item in mixed_policies]
        + [f"backend={item}" for item in backends]
        + ["supports_fp16", "supports_int8", "supports_mixed_int8", "supports_cutlass_byoc"]
        + ["source_confidence"]
    )
    return {
        "genome_schema": ["w0", "w1", "w2", "strategy_id"],
        "feature_order": feature_order,
        "q_mode_categories": q_modes,
        "mixed_policy_categories": mixed_policies,
        "backend_categories": backends,
        "compiler_profile_ids": sorted(profiles),
    }


def _source_confidence(row: dict[str, Any], profile: dict[str, Any] | None) -> float:
    if row.get("source_confidence") is not None:
        return float(row["source_confidence"])
    confidence = str((profile or {}).get("measurement_confidence", "")).lower()
    if "real_measurement" in confidence:
        return 1.0
    if "probe" in confidence:
        return 0.65
    return 0.5


def encode_training_row(row: dict[str, Any], spec: dict[str, Any], backend_scan: dict[str, Any] | None = None) -> dict[str, Any]:
    w0, w1, w2 = parse_width(row.get("width", [row.get("w0"), row.get("w1"), row.get("w2")]))
    q_mode = str(row.get("q_mode", "unknown")).lower()
    mixed_policy_id = str(row.get("mixed_policy_id", "none"))
    sid = str(row.get("strategy_id") or strategy_id(q_mode, mixed_policy_id))
    profiles = _profile_by_id(backend_scan or {"compiler_backend_profiles": []})
    profile = profiles.get(str(row.get("compiler_profile_id")), {})
    backend = str(row.get("backend") or profile.get("backend") or "unknown")

    values: list[float] = [float(w0), float(w1), float(w2)]
    values.extend(1.0 if item == q_mode else 0.0 for item in spec["q_mode_categories"])
    values.extend(1.0 if item == mixed_policy_id else 0.0 for item in spec["mixed_policy_categories"])
    values.extend(1.0 if item == backend else 0.0 for item in spec["backend_categories"])
    values.extend(
        [
            _support_as_float(profile.get("supports_fp16")),
            _support_as_float(profile.get("supports_int8")),
            _support_as_float(profile.get("supports_mixed_int8")),
            _support_as_float(profile.get("supports_cutlass_byoc")),
        ]
    )
    values.append(_source_confidence(row, profile))

    return {
        "width": f"{w0}x{w1}x{w2}",
        "genome": [w0, w1, w2, sid],
        "features": values,
        "target": {
            "latency_ms": row.get("latency_ms"),
            "energy_j": row.get("energy_j"),
            "ap70": row.get("ap70"),
        },
        "source": {
            "strategy_id": sid,
            "q_mode": q_mode,
            "mixed_policy_id": mixed_policy_id,
            "compiler_profile_id": row.get("compiler_profile_id"),
            "backend": backend,
            "measurement_source": row.get("measurement_source"),
            "trusted_for_final_frontier": row.get("trusted_for_final_frontier"),
            "source_file": row.get("source_file"),
            "realized_schedule_route": row.get("realized_schedule_route"),
        },
        "derived_graph_features": row.get("derived_graph_features") or _default_derived_features(),
    }


def final_frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if _truthy(row.get("trusted_for_final_frontier")) and row.get("latency_ms") is not None
    ]


def build_entry(
    *,
    backend_scan: dict[str, Any],
    strategy_manifest: dict[str, Any],
    training_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    spec = build_feature_spec(backend_scan, training_rows)
    encoded = [encode_training_row(row, spec, backend_scan) for row in training_rows]
    final_rows = final_frontier_rows(training_rows)
    return {
        "schema_version": "phase1_qxs_smbo_entry_v2",
        "created_at_utc": utc_now(),
        "genome_schema": spec["genome_schema"],
        "feature_order": spec["feature_order"],
        "strategy_manifest_schema": strategy_manifest.get("schema_version"),
        "backend_scan_schema": backend_scan.get("schema_version"),
        "policy": {
            "backend_in_strategy_id": False,
            "graph_boundary_primary_search_axis": False,
            "graph_boundary_handling": "derived_diagnostic_feature_from_mixed_mask_or_lowering_record",
            "final_frontier_rule": "trusted real rows only; backend context allowed in features but not genome",
        },
        "summary": {
            "n_compiler_profiles": len(backend_scan.get("compiler_backend_profiles", [])),
            "n_strategies": len(strategy_manifest.get("strategies", [])),
            "n_training_rows": len(training_rows),
            "n_encoded_training_rows": len(encoded),
            "n_final_frontier_rows": len(final_rows),
            "n_widths": len({row["width"] for row in training_rows}),
        },
        "compiler_backend_profiles": backend_scan.get("compiler_backend_profiles", []),
        "strategies": strategy_manifest.get("strategies", []),
        "training_rows": training_rows,
        "training_feature_rows": encoded,
        "final_frontier_gold_rows": final_rows,
    }


def _ridge_predict(x_train: Any, y_train: Any, x_test: Any, alpha: float = 1e-3) -> Any:
    import numpy as np

    xtx = x_train.T @ x_train
    reg = np.eye(xtx.shape[0]) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.solve(xtx + reg, x_train.T @ y_train)
    return x_test @ beta, beta


def evaluate_latency_learning_smoke(entry: dict[str, Any]) -> dict[str, Any]:
    """Train a tiny ridge/log-latency smoke model on v2 feature rows.

    This is not the final paper cost model.  It is an executable contract check:
    the v2 table can be consumed by a model while backend remains a context
    feature instead of a genome component.
    """
    import numpy as np

    rows = [
        row for row in entry.get("training_feature_rows", [])
        if (row.get("target") or {}).get("latency_ms") is not None
    ]
    if not rows:
        raise ValueError("no latency rows available for v2 cost-model smoke")

    x = np.array([[1.0] + [float(v) for v in row["features"]] for row in rows], dtype=float)
    y = np.log1p(np.array([float(row["target"]["latency_ms"]) for row in rows], dtype=float))
    widths = np.array([str(row["width"]) for row in rows])
    pred_log, beta = _ridge_predict(x, y, x)
    pred = np.expm1(pred_log)
    y_ms = np.expm1(y)
    train_mae = float(np.mean(np.abs(pred - y_ms)))
    mean_baseline_mae = float(np.mean(np.abs(y_ms - np.mean(y_ms))))

    cv_preds = np.zeros_like(y_ms)
    unique_widths = sorted(set(widths.tolist()))
    if len(unique_widths) >= 2:
        for width in unique_widths:
            train_mask = widths != width
            test_mask = widths == width
            if not np.any(train_mask) or not np.any(test_mask):
                continue
            pred_log_fold, _ = _ridge_predict(x[train_mask], y[train_mask], x[test_mask])
            cv_preds[test_mask] = np.expm1(pred_log_fold)
        cv_mae = float(np.mean(np.abs(cv_preds - y_ms)))
    else:
        cv_mae = None

    feature_order = ["intercept"] + list(entry.get("feature_order", []))
    top_coefficients = sorted(
        (
            {
                "feature": feature_order[idx],
                "coefficient_abs": float(abs(coef)),
                "coefficient": float(coef),
            }
            for idx, coef in enumerate(beta)
        ),
        key=lambda item: item["coefficient_abs"],
        reverse=True,
    )[:10]
    return {
        "schema_version": "phase1_qxs_v2_cost_model_smoke_v1",
        "created_at_utc": utc_now(),
        "model": "ridge_linear_regression_on_log1p_latency",
        "purpose": "executable smoke that v2 context-aware feature rows are model-consumable",
        "n_rows": int(len(rows)),
        "n_features": int(len(feature_order) - 1),
        "cv_group": "width",
        "n_width_groups": int(len(unique_widths)),
        "train_mae_ms": train_mae,
        "mean_baseline_mae_ms": mean_baseline_mae,
        "leave_one_width_out_mae_ms": cv_mae,
        "feature_order": list(entry.get("feature_order", [])),
        "top_abs_coefficients": top_coefficients,
    }


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--tvm-mixed-summary", type=Path, default=DEFAULT_TVM_MIXED_SUMMARY)
    parser.add_argument("--trt-qxs-dir", type=Path, default=DEFAULT_TRT_QXS_DIR)
    parser.add_argument("--widths", default=",".join(DEFAULT_WIDTHS))
    parser.add_argument("--backend-scan-out", type=Path, default=DEFAULT_BACKEND_SCAN_OUT)
    parser.add_argument("--strategy-manifest-out", type=Path, default=DEFAULT_STRATEGY_MANIFEST_OUT)
    parser.add_argument("--training-table-out", type=Path, default=DEFAULT_TRAINING_TABLE_OUT)
    parser.add_argument("--entry-out", type=Path, default=DEFAULT_ENTRY_OUT)
    parser.add_argument("--cost-model-smoke-out", type=Path, default=DEFAULT_COST_MODEL_SMOKE_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_widths = [item.strip() for item in args.widths.split(",") if item.strip()]
    backend_scan = build_h800_compiler_backend_scan_from_results(args.results_dir)
    tvm_summary = load_json(args.tvm_mixed_summary)
    training_rows = build_context_training_rows(
        tvm_summary=tvm_summary,
        trt_dir=args.trt_qxs_dir,
        selected_widths=selected_widths,
    )
    strategy_manifest = build_strategy_manifest(selected_widths, backend_scan)
    entry = build_entry(
        backend_scan=backend_scan,
        strategy_manifest=strategy_manifest,
        training_rows=training_rows,
    )
    cost_model_smoke = evaluate_latency_learning_smoke(entry)

    write_json(args.backend_scan_out, backend_scan)
    write_json(args.strategy_manifest_out, strategy_manifest)
    write_json(args.training_table_out, {
        "schema_version": "phase1_qxs_training_table_context_backend_v2",
        "created_at_utc": utc_now(),
        "rows": training_rows,
    })
    write_json(args.entry_out, entry)
    write_json(args.cost_model_smoke_out, cost_model_smoke)
    print(json.dumps({
        "backend_scan_out": str(args.backend_scan_out),
        "cost_model_smoke_out": str(args.cost_model_smoke_out),
        "strategy_manifest_out": str(args.strategy_manifest_out),
        "training_table_out": str(args.training_table_out),
        "entry_out": str(args.entry_out),
        "summary": entry["summary"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
