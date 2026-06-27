#!/usr/bin/env python3
"""Bind S2.5 coverage gates and S3 quant-sensitivity evidence.

This script does not run new kernels. It converts already available source,
timing, AP, and latency evidence into explicit Stage1 gate reports so the safe
predictor can distinguish measured facts, static proofs, and blockers.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = ROOT / "results/stage1_model_predict"
S2_5_DIR = STAGE1_DIR / "s2_5_coverage_gates"
S3_DIR = STAGE1_DIR / "s3_quant_sensitivity"

HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
MAXFUSION_SOURCE = HEAL_ROOT / "opencood/models/fuse_modules/fusion_in_one.py"
HETER_BASELINE_SOURCE = HEAL_ROOT / "opencood/models/heter_model_baseline.py"
WHERE2COMM_SOURCE = HEAL_ROOT / "opencood/models/comm_modules/where2comm.py"
V2VNET_SOURCE = HEAL_ROOT / "opencood/models/fuse_modules/v2v_fuse.py"

FCOOPER_TIMING = ROOT / "paper_learning/2. AAAI最终故事/model/fcooper/分段耗时实测_v1.md"
AP_CSV = ROOT / "results/perstage_quant_AP_real_v2.csv"
LAT_CSV = ROOT / "results/perstage_quant_latency_v2.csv"
LAT_GLOBAL_CSV = ROOT / "results/perstage_quant_latency_global_v2.csv"
PARETO_MD = ROOT / "results/perstage_quant_pareto_verdict_v2.md"
V2XVIT_C4 = ROOT / "results/coupling_map/C4_QgranxP_v2xvit.json"
V2XVIT_C5 = ROOT / "results/coupling_map/C5_routing_v2xvit.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 4) -> float:
    return round(_f(value), digits)


def _extract_fcooper_maxfusion_timing() -> dict[str, Any]:
    text = FCOOPER_TIMING.read_text(encoding="utf-8") if FCOOPER_TIMING.exists() else ""
    row = None
    for line in text.splitlines():
        if "fusion_net" in line and "MaxFusion" in line and "|" in line:
            cells = [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 7 and cells[1] == "fusion_net":
                row = cells
                break

    parsed = {}
    if row:
        parsed = {
            "mean_ms": _round(row[3], 3),
            "p50_ms": _round(row[4], 3),
            "p99_ms": _round(row[5], 3),
            "pct_of_e2e": row[6].replace("%", "").strip(),
            "raw_row": " | ".join(row),
        }

    return {
        "source": _rel(FCOOPER_TIMING),
        "status": "FOUND" if parsed else "MISSING_OR_UNPARSED",
        "timing": parsed,
        "interpretation": (
            "Existing true-weight F-Cooper timing bounds MaxFusion as small "
            "relative to e2e, but this is not a new S2 H800 TVM schedule anchor."
        ),
    }


def _fusion_sidecar(model: str, filename: str) -> dict[str, Any]:
    path = STAGE1_DIR / "fusion_timing_probe" / filename
    data = _read_json(path)
    stages = data.get("forward_stages_cuda_event", {}) or {}
    fusion = stages.get("fusion_net", {}) or {}
    e2e = data.get("e2e_walltime", {}) or {}
    stage_sum = sum(_f(item.get("mean_ms")) for item in stages.values() if isinstance(item, dict))
    fusion_mean = _f(fusion.get("mean_ms"))
    return {
        "model": model,
        "source": _rel(path),
        "random_init": bool(data.get("random_init")),
        "record_len_mean": _round(data.get("record_len_mean"), 3),
        "measure": int(data.get("measure", 0) or 0),
        "device": data.get("device"),
        "e2e_mean_ms": _round(e2e.get("mean_ms"), 3),
        "fusion_mean_ms": _round(fusion_mean, 3),
        "fusion_p50_ms": _round(fusion.get("p50_ms"), 3),
        "fusion_p99_ms": _round(fusion.get("p99_ms"), 3),
        "fusion_pct_of_stage_sum": round(fusion_mean / stage_sum, 4) if stage_sum else None,
        "fusion_pct_of_e2e_walltime": round(fusion_mean / _f(e2e.get("mean_ms")), 4)
        if _f(e2e.get("mean_ms"))
        else None,
    }


def build_s2_5_report() -> dict[str, Any]:
    maxfusion_source_exists = MAXFUSION_SOURCE.exists()
    maxfusion_static_proof = {
        "source": _rel(MAXFUSION_SOURCE),
        "source_available": maxfusion_source_exists,
        "line_range": "87-124",
        "input_contract": "x=(sum(n_cav), C, H, W), record_len=(B), affine_matrix=(B, L, L, 2, 3)",
        "output_contract": "torch.stack(out) -> (B, C, H, W)",
        "shape_preserving_axes": ["C", "H", "W"],
        "reduced_axis": "agent dimension: sum(n_cav)/N -> B ego features",
        "operators": ["warp_affine_simple(..., (H, W))", "torch.max(neighbor_feature, dim=0)[0]"],
        "learned_parameters": 0,
        "status": "STATIC_SHAPE_PRESERVING_PROVEN" if maxfusion_source_exists else "SOURCE_MISSING",
    }
    fcooper_timing = _extract_fcooper_maxfusion_timing()

    sidecars = [
        _fusion_sidecar("where2comm", "stage1probe_where2comm_random.json"),
        _fusion_sidecar("v2vnet", "stage1probe_v2vnet_random.json"),
        _fusion_sidecar("disconet", "stage1probe_disconet_random.json"),
    ]
    heavy_models = [
        item["model"]
        for item in sidecars
        if (item.get("fusion_pct_of_stage_sum") or 0) >= 0.25
        or (item.get("fusion_pct_of_e2e_walltime") or 0) >= 0.25
    ]

    return {
        "schema": "stage1_s2_5_coverage_gate_closure_v1",
        "generated_on": "2026-06-23",
        "scope": "S2.5 targeted coverage gates only; no new broad P/Q/S/batch enumeration",
        "policy": {
            "do_not_expand_nine_probes_into_full_matrix": True,
            "new_probe_rule": "Only add acceptance-driven targeted probes that close a named blocker.",
            "proceed_to_s3_after_s2_5": True,
        },
        "gates": {
            "maxfusion_coverage_anchor": {
                "status": "STATIC_SHAPE_PROOF_PLUS_EXISTING_TIMING_BOUND",
                "verdict": "MAXFUSION_COVERAGE_BOUNDED_FOR_FCOOPER_ONLY",
                "scope": "F-Cooper MaxFusion module; not a generic fusion or full-model separability pass",
                "static_proof": maxfusion_static_proof,
                "existing_timing": fcooper_timing,
                "allowed_predictor_effect": (
                    "Remove 'MaxFusion unbounded' wording for F-Cooper, while keeping "
                    "standard-conv schedule coupling and routing/fusion gates explicit."
                ),
                "not_allowed": [
                    "Do not mark all fusion modules as negligible.",
                    "Do not infer full-model separability from MaxFusion alone.",
                    "Do not treat this as an H800 TVM schedule-anchor measurement.",
                ],
            },
            "routing_fusion_coverage_anchor": {
                "status": "MODEL_LEVEL_COVERAGE_GATE_REMAINS",
                "verdict": "ROUTING_FUSION_NOT_GLOBALLY_IGNORABLE",
                "scope": "AttFuse/Where2comm/V2VNet/DiscoNet/V2X-ViT routing or learned fusion modules",
                "source_contracts": [
                    {
                        "model": "heter_model_baseline",
                        "source": _rel(HETER_BASELINE_SOURCE),
                        "lines": "96-111, 223-230",
                        "finding": "fusion_method selects different fusion_net modules and calls fusion before heads",
                    },
                    {
                        "model": "where2comm",
                        "source": _rel(WHERE2COMM_SOURCE),
                        "lines": "34-78",
                        "finding": "confidence threshold masks and communication rates are data-dependent",
                    },
                    {
                        "model": "v2vnet",
                        "source": _rel(V2VNET_SOURCE),
                        "lines": "54-169",
                        "finding": "iterative message passing, warp_affine masks, msg_cnn/conv_gru, and aggregation alter fusion cost and feasible routing",
                    },
                ],
                "fusion_timing_sidecar": sidecars,
                "heavy_or_unbounded_models": heavy_models,
                "allowed_predictor_effect": (
                    "Keep full-model verdict capped for routing/fusion models unless a per-model "
                    "shape, latency, and feasible-set proof exists."
                ),
                "not_allowed": [
                    "Do not treat random-init timing as trained-checkpoint AP evidence.",
                    "Do not treat sidecar timing as a TVM schedule anchor.",
                    "Do not collapse routing/fusion into a constant overhead rule.",
                ],
            },
        },
        "sources": [
            _rel(MAXFUSION_SOURCE),
            _rel(HETER_BASELINE_SOURCE),
            _rel(WHERE2COMM_SOURCE),
            _rel(V2VNET_SOURCE),
            _rel(FCOOPER_TIMING),
            "results/stage1_model_predict/fusion_timing_probe/stage1probe_where2comm_random.json",
            "results/stage1_model_predict/fusion_timing_probe/stage1probe_v2vnet_random.json",
            "results/stage1_model_predict/fusion_timing_probe/stage1probe_disconet_random.json",
        ],
    }


@dataclass(frozen=True)
class JoinedRow:
    triplet: str
    config_label: str
    ap50: float
    delta_vs_forced: float
    delta_vs_automix: float
    lat_p50_ms: float | None


TRIPLET_MAP = {
    "T_baseline": "base",
    "T_prune50p": "prune50p",
    "T_prune75": "prune75",
}


def _joined_pyramid_rows() -> list[JoinedRow]:
    ap_rows = _read_csv(AP_CSV)
    lat_rows = _read_csv(LAT_CSV) + _read_csv(LAT_GLOBAL_CSV)
    lat_by_key = {
        (row.get("triplet"), row.get("config_label")): _f(row.get("lat_p50_ms"))
        for row in lat_rows
    }
    joined = []
    for row in ap_rows:
        triplet = TRIPLET_MAP.get(row.get("triplet", ""), row.get("triplet", ""))
        config = row.get("config_label", "")
        joined.append(
            JoinedRow(
                triplet=triplet,
                config_label=config,
                ap50=_f(row.get("ap50")),
                delta_vs_forced=_f(row.get("delta_ap50_vs_forced_int8")),
                delta_vs_automix=_f(row.get("delta_ap50_vs_automix_int8")),
                lat_p50_ms=lat_by_key.get((triplet, config)),
            )
        )
    return joined


def _pyramid_s3_summary() -> dict[str, Any]:
    rows = _joined_pyramid_rows()
    by_triplet: dict[str, list[JoinedRow]] = {}
    for row in rows:
        by_triplet.setdefault(row.triplet, []).append(row)

    triplet_summaries = []
    for triplet, items in sorted(by_triplet.items()):
        forced = next((item for item in items if item.config_label == "c_all_int8_forced"), None)
        automix = next((item for item in items if item.config_label == "global_int8_automix"), None)
        mixed = [
            item
            for item in items
            if item.config_label not in {"c_all_int8_forced", "global_int8_automix", "global_fp16_automix"}
        ]
        best_mixed = max(mixed, key=lambda item: item.delta_vs_forced, default=None)
        dominated_by_automix = []
        if automix and automix.lat_p50_ms is not None:
            for item in mixed:
                if item.lat_p50_ms is None:
                    continue
                if automix.lat_p50_ms <= item.lat_p50_ms and automix.ap50 >= item.ap50:
                    dominated_by_automix.append(item.config_label)

        triplet_summaries.append(
            {
                "triplet": triplet,
                "forced_int8_ap50": _round(forced.ap50) if forced else None,
                "forced_int8_lat_p50_ms": _round(forced.lat_p50_ms) if forced else None,
                "global_int8_ap50": _round(automix.ap50) if automix else None,
                "global_int8_lat_p50_ms": _round(automix.lat_p50_ms) if automix else None,
                "best_mixed_vs_forced": {
                    "config_label": best_mixed.config_label if best_mixed else None,
                    "ap50": _round(best_mixed.ap50) if best_mixed else None,
                    "delta_ap50_vs_forced": _round(best_mixed.delta_vs_forced) if best_mixed else None,
                    "delta_ap50_vs_global_int8": _round(best_mixed.delta_vs_automix)
                    if best_mixed
                    else None,
                    "lat_p50_ms": _round(best_mixed.lat_p50_ms) if best_mixed else None,
                },
                "mixed_configs_dominated_by_global_int8": dominated_by_automix,
            }
        )

    return {
        "status": "MEASURED_AP_AND_LATENCY_BOUND",
        "evidence_level": "true_trt_ap_plus_clean_gpu_latency",
        "rows": len(rows),
        "ap_source": _rel(AP_CSV),
        "latency_sources": [_rel(LAT_CSV), _rel(LAT_GLOBAL_CSV)],
        "pareto_source": _rel(PARETO_MD),
        "triplet_summary": triplet_summaries,
        "verdict": (
            "Forced per-stage INT8 can expose AP sensitivity at pruned widths, "
            "but hand-forced per-stage mixed precision is not Pareto-positive "
            "once TRT global automix baselines are included."
        ),
        "predictor_implication": (
            "Keep Pyramid as Q/AP-sensitive and P-hub-context blocked, but do not "
            "expand the search space toward arbitrary hand-forced per-stage Q rules."
        ),
    }


def _v2xvit_s3_summary() -> dict[str, Any]:
    c4 = _read_json(V2XVIT_C4)
    c5 = _read_json(V2XVIT_C5)
    c4_summary = c4.get("summary", {}) if isinstance(c4.get("summary"), dict) else {}
    open_followup = c4.get("open_followup", {}) if isinstance(c4.get("open_followup"), dict) else {}
    granularity = (
        c4.get("granularity_coupling_verdict", {})
        if isinstance(c4.get("granularity_coupling_verdict"), dict)
        else {}
    )
    c5_implication = (
        c5.get("coupling_map_implication", {})
        if isinstance(c5.get("coupling_map_implication"), dict)
        else {}
    )
    return {
        "status": "WEAK_FAKE_QUANT_GATE_ONLY_TRUE_TRT_BLOCKED",
        "evidence_level": "simulated_fake_quant_plus_structural_routing_analysis",
        "sources": [_rel(V2XVIT_C4), _rel(V2XVIT_C5)],
        "q_axis_signal": c4_summary.get("coupling_verdict", {}).get("Q_axis_signal"),
        "granularity_verdict": granularity.get("verdict"),
        "granularity_coupling_signal": granularity.get("coupling_signal"),
        "true_trt_int8_ap": open_followup.get("true_trt_int8_ap", {}),
        "true_trt_per_channel_int8": open_followup.get("true_trt_per_channel_int8", {}),
        "c5_routing_status": c5.get("status"),
        "c5_conclusion": c5_implication.get("verdict"),
        "verdict": (
            "V2X-ViT remains pair/joint-gated by fake-quant QxP signal and structural "
            "routing blacklist, but true TRT INT8 AP/per-channel closure is unresolved."
        ),
        "not_allowed": [
            "Do not call C4 a real TRT INT8 measurement.",
            "Do not call C5 an Orin latency measurement.",
        ],
    }


def build_s3_report() -> dict[str, Any]:
    return {
        "schema": "stage1_s3_quant_sensitivity_v1",
        "generated_on": "2026-06-23",
        "scope": "S3 evidence binding and executable closure; unresolved export/hardware blockers remain explicit",
        "models": {
            "pyramid": _pyramid_s3_summary(),
            "v2xvit": _v2xvit_s3_summary(),
        },
        "stage_status": {
            "s3_executable_evidence_binding": "DONE",
            "s3_true_trt_v2xvit": "BLOCKED_BY_FUSION_ONNX_EXPORT",
            "s4_three_arm_validation": "NOT_STARTED",
        },
        "stop_rule": (
            "S3 may close for available evidence after Pyramid true AP/latency and "
            "V2X-ViT fake-quant/blocker evidence are bound; S4 is required before "
            "any final joint-vs-pair search rule."
        ),
    }


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def render_s2_5_markdown(report: dict[str, Any]) -> str:
    maxfusion = report["gates"]["maxfusion_coverage_anchor"]
    routing = report["gates"]["routing_fusion_coverage_anchor"]
    timing = maxfusion["existing_timing"]["timing"]
    sidecar_rows = []
    for item in routing["fusion_timing_sidecar"]:
        sidecar_rows.append(
            {
                "model": item["model"],
                "random_init": item["random_init"],
                "record_len": item["record_len_mean"],
                "fusion_mean_ms": item["fusion_mean_ms"],
                "fusion_pct_stage_sum": item["fusion_pct_of_stage_sum"],
                "verdict": "cannot_ignore" if item["model"] in routing["heavy_or_unbounded_models"] else "bounded_warning",
            }
        )

    lines = [
        "# Stage1 S2.5 Coverage Gate Closure v1",
        "",
        "S2.5 只关闭两个 targeted coverage gate；它不是新的 P/Q/S/batch 枚举，也不是 S2/S3 全部通过。",
        "",
        f"- policy: `{report['policy']['new_probe_rule']}`",
        f"- proceed_to_s3_after_s2_5: `{report['policy']['proceed_to_s3_after_s2_5']}`",
        "",
        "## maxfusion_coverage_anchor",
        "",
        f"- status: `{maxfusion['status']}`",
        f"- verdict: `{maxfusion['verdict']}`",
        f"- scope: `{maxfusion['scope']}`",
        f"- static proof: `{maxfusion['static_proof']['status']}`; input `{maxfusion['static_proof']['input_contract']}`; output `{maxfusion['static_proof']['output_contract']}`",
        f"- existing timing: `{maxfusion['existing_timing']['status']}`; mean `{timing.get('mean_ms')}` ms; p50 `{timing.get('p50_ms')}` ms; p99 `{timing.get('p99_ms')}` ms; e2e pct `{timing.get('pct_of_e2e')}`",
        "",
        "限制：MaxFusion 只能解除 F-Cooper MaxFusion 未定界这一点，不能推出 full-model separable。",
        "",
        "## routing_fusion_coverage_anchor",
        "",
        f"- status: `{routing['status']}`",
        f"- verdict: `{routing['verdict']}`",
        f"- scope: `{routing['scope']}`",
        "",
    ]
    lines.extend(_md_table(sidecar_rows, ["model", "random_init", "record_len", "fusion_mean_ms", "fusion_pct_stage_sum", "verdict"]))
    lines.extend(
        [
            "",
            "结论：Where2comm/V2VNet 的 fusion 不能全局忽略；DiscoNet 当前只是单 agent DAIR sidecar，不能代表多 agent。",
            "",
        ]
    )
    return "\n".join(lines)


def render_s3_markdown(report: dict[str, Any]) -> str:
    pyramid = report["models"]["pyramid"]
    v2xvit = report["models"]["v2xvit"]
    pyramid_rows = []
    for item in pyramid["triplet_summary"]:
        best = item["best_mixed_vs_forced"]
        pyramid_rows.append(
            {
                "triplet": item["triplet"],
                "forced_ap50": item["forced_int8_ap50"],
                "global_int8_ap50": item["global_int8_ap50"],
                "best_mixed": best["config_label"],
                "best_delta_forced": best["delta_ap50_vs_forced"],
                "best_delta_global_int8": best["delta_ap50_vs_global_int8"],
            }
        )

    lines = [
        "# Stage1 S3 Quant Sensitivity v1",
        "",
        "S3 的目标是绑定 Q/AP sensitivity 证据，而不是继续扩大 S2 schedule anchor 数量。",
        "",
        "## Pyramid",
        "",
        f"- status: `{pyramid['status']}`",
        f"- evidence_level: `{pyramid['evidence_level']}`",
        f"- verdict: {pyramid['verdict']}",
        "",
    ]
    lines.extend(
        _md_table(
            pyramid_rows,
            [
                "triplet",
                "forced_ap50",
                "global_int8_ap50",
                "best_mixed",
                "best_delta_forced",
                "best_delta_global_int8",
            ],
        )
    )
    lines.extend(
        [
            "",
            "关键边界：混精相对 forced-all-INT8 有 AP 正向点，但被 TRT global_int8_automix 纳入后不构成手工 per-stage Q 搜索扩张依据。",
            "",
            "## V2X-ViT",
            "",
            f"- status: `{v2xvit['status']}`",
            f"- evidence_level: `{v2xvit['evidence_level']}`",
            f"- granularity verdict: `{v2xvit['granularity_verdict']}`; signal `{v2xvit['granularity_coupling_signal']}`",
            f"- true_trt_int8_ap: `{v2xvit['true_trt_int8_ap'].get('status')}`; blocker `{v2xvit['true_trt_int8_ap'].get('blocker')}`",
            f"- c5 routing: `{v2xvit['c5_conclusion']}`; Orin latency remains not measured",
            "",
            "结论：V2X-ViT 只能作为 fake-quant + structural routing gate，不能写成 true TRT closure。",
            "",
            "## Stage Status",
            "",
            f"- s3_executable_evidence_binding: `{report['stage_status']['s3_executable_evidence_binding']}`",
            f"- s3_true_trt_v2xvit: `{report['stage_status']['s3_true_trt_v2xvit']}`",
            f"- s4_three_arm_validation: `{report['stage_status']['s4_three_arm_validation']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    s2_5_report = build_s2_5_report()
    s3_report = build_s3_report()

    _write_json(S2_5_DIR / "stage1_s2_5_coverage_gate_closure_v1.json", s2_5_report)
    _write_text(S2_5_DIR / "stage1_s2_5_coverage_gate_closure_v1.md", render_s2_5_markdown(s2_5_report))
    _write_json(S3_DIR / "stage1_s3_quant_sensitivity_v1.json", s3_report)
    _write_text(S3_DIR / "stage1_s3_quant_sensitivity_v1.md", render_s3_markdown(s3_report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
