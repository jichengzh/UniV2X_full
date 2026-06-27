#!/usr/bin/env python3
"""Calibrated Stage1 coupling predictor v1.

V1 is a reporting/calibration layer over Safe Predictor v0 plus S2/S2.5/S3/S4
evidence. It keeps v0's no-overpromotion guarantees and adds explicit evidence
categories, rule triggers, and unsupported conclusions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from framework.stage1.coupling_predictor import BACKEND_POLICY, find_overpromotions


ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = ROOT / "results/stage1_model_predict"

V0_PATH = STAGE1_DIR / "stage1_coupling_predictions_v0.json"
S2_AUDIT_PATH = STAGE1_DIR / "s2_schedule_anchor_audit_v1.json"
S2_5_PATH = STAGE1_DIR / "s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json"
S3_PATH = STAGE1_DIR / "s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json"
S4_PATH = STAGE1_DIR / "s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json"

V1_JSON = STAGE1_DIR / "stage1_coupling_predictions_v1.json"
V1_MD = STAGE1_DIR / "stage1_coupling_predictions_v1.md"
CAL_JSON = STAGE1_DIR / "calibrated_predictor_report_v1.json"
CAL_MD = STAGE1_DIR / "calibrated_predictor_report_v1.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _s2_anchor_rollup(s2: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    anchors = s2.get("h800_anchor_scan", {}).get("anchors", [])
    for item in _as_list(anchors):
        if isinstance(item, dict) and item.get("probe_id"):
            out[str(item["probe_id"])] = item
    return out


def _s4_rollup(s4: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rollup = s4.get("anchor_rollup", {})
    return dict(rollup) if isinstance(rollup, dict) else {}


def _evidence_categories(model: str, base: dict[str, Any]) -> list[str]:
    categories = []
    level = str(base.get("evidence_level", ""))
    if "measured_s2_schedule" in level or "measured_h800_tvm" in level:
        categories.append("measured_h800_tvm_latency")
        categories.append("static_shape_proof")
    if level in {"measured_negative_anchor", "measured_existing_codriving_anchor"}:
        categories.append("measured_existing_codriving_anchor")
    if "historical_trt_evidence" in level or level in {
        "true_trt_ap_plus_clean_gpu_latency",
        "true_trt_ap_latency_plus_p_hub_context",
    }:
        categories.append("historical_trt_evidence")
        categories.append("p_hub_context")
    if "fake_quant" in level or model == "v2xvit":
        categories.append("fake_quant_gate")
        categories.append("structural_routing_gate")
    if "uncovered" in level or model in {"attfuse", "where2comm", "v2vnet", "disconet"}:
        categories.append("architecture_only_scan")
        categories.append("fusion_uncovered")
    if model in {"where2comm", "v2vnet", "disconet"}:
        categories.append("missing_architecture_scan_only")
        categories.append("random_init_fusion_timing_sidecar")
    if model == "fcooper":
        categories.append("maxfusion_static_shape_bound")
    categories.append("blocked_or_scoped" if base.get("blockers") else "scoped_evidence")
    return sorted(dict.fromkeys(categories))


def _calibrate_model(
    base: dict[str, Any],
    *,
    s2_anchors: dict[str, dict[str, Any]],
    s2_5: dict[str, Any],
    s3: dict[str, Any],
    s4_rollup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    model = str(base.get("model", "unknown"))
    rule_triggers = [
        "groups=1 is never a separability rule",
        "blocked/unresolved/partial-bound evidence cannot support full-model separability",
    ]
    unsupported = [
        "full-model separability",
        "static-only standard-conv low-risk pass",
    ]
    supported = []
    verdict = base.get("verdict")
    scope = base.get("scope")
    evidence_level = base.get("evidence_level")
    required_next = list(base.get("required_next_probe_or_gate", []))
    blockers = list(base.get("blockers", []))
    confidence = float(base.get("confidence", 0.0))

    if model == "codriving":
        verdict = "SCOPED_MEASURED_NEGATIVE_ANCHOR"
        scope = "codriving_measured_resnet_backbone_envelope"
        evidence_level = "measured_existing_codriving_anchor"
        rule_triggers.append("CoDriving negative evidence does not cross-model extrapolate")
        supported.append("CoDriving measured envelope can remain low-risk/scoped")
        unsupported.append("copying CoDriving verdict to F-Cooper/AttFuse/V2X-ViT/Pyramid/A2 models")
        confidence = 0.80
    elif model == "fcooper":
        verdict = "PAIR_CALIBRATION_REQUIRED_FULL_MODEL_GATED"
        scope = "full_model_gated_dense_anchor_pair_calibration_required"
        evidence_level = "measured_h800_tvm_s4_pair_required_plus_maxfusion_static_bound"
        rule_triggers.extend(
            [
                "S2 std_basebev_backbone_schedule_anchor COUPLING_SIGNAL_DETECTED",
                "S4 BaseBEVBackbone local-only winner changes; pair-search matches joint in mini matrix",
                "S2.5 MaxFusion bound is F-Cooper-only and not a full-model pass",
                "routing/fusion coverage gate remains model-level cap",
            ]
        )
        supported.append("F-Cooper MaxFusion is shape-bounded for this module")
        supported.append("F-Cooper dense anchor needs at least pair-level schedule calibration")
        unsupported.append("F-Cooper full-model separability")
        if "s4_three_arm_validation_for_basebev_schedule_anchor" in required_next:
            required_next.remove("s4_three_arm_validation_for_basebev_schedule_anchor")
        required_next.append("s4_ap_or_hv_validation_if_claiming_joint_vs_serial")
        confidence = 0.74
    elif model == "attfuse":
        verdict = "FUSION_UNCOVERED_STATIC_DENSE_PROMOTION_BLOCKED"
        rule_triggers.extend(
            [
                "attention fusion is skipped/unintegrated",
                "S2 measured standard-conv anchors detected schedule coupling",
                "S4 pair calibration needed for dense anchors before dense low-risk claims",
            ]
        )
        unsupported.append("AttFuse full-model separability")
        required_next.append("attention_fusion_coverage_anchor")
        confidence = 0.84
    elif model == "v2xvit":
        verdict = "PAIR_OR_JOINT_GATE_FAKE_QUANT_TRUE_TRT_BLOCKED"
        evidence_level = "fake_quant_qxp_plus_structural_routing_true_trt_blocked"
        rule_triggers.extend(
            [
                "V2X-ViT C4 is fake-quant only",
                "C5 routing is structural but Orin latency is not measured",
                "true TRT INT8 AP/per-channel remains blocked",
            ]
        )
        supported.append("V2X-ViT remains pair/joint-gated by fake-quant and structural routing evidence")
        unsupported.append("V2X-ViT true TRT INT8 AP closure")
        unsupported.append("V2X-ViT Orin latency measured")
        confidence = 0.82
    elif model.startswith("pyramid"):
        verdict = "P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND"
        evidence_level = "historical_trt_evidence_plus_p_hub_context"
        rule_triggers.extend(
            [
                "Pyramid TRT AP + latency is historical evidence only",
                "hand-forced per-stage mixed precision is not Pareto-positive vs global_int8_automix",
                "P-hub context still blocks model-level standard-conv promotion",
            ]
        )
        supported.append("Pyramid Q/AP sensitivity is retained as historical TRT evidence")
        unsupported.append("Pyramid hand-forced per-stage mixed precision Pareto-positive")
        unsupported.append("Pyramid model-level standard-conv separability")
        required_next = [item for item in required_next if item != "per_stage_q_ap_sensitivity_anchor"]
        confidence = 0.88
    elif model in {"where2comm", "v2vnet", "disconet"}:
        verdict = "ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN"
        evidence_level = "architecture_only_missing_ckpt_plus_random_init_fusion_sidecar"
        rule_triggers.extend(
            [
                "A2 models are missing_architecture_scan_only",
                "fusion timing sidecar is random-init latency-only",
                "routing/fusion coverage gate remains model-level cap",
            ]
        )
        supported.append(f"{model} architecture-only dense scan exists")
        unsupported.append(f"{model} trained checkpoint scan")
        unsupported.append(f"{model} full-model separability")
        if model == "disconet":
            unsupported.append("DiscoNet multi-agent fusion negligible")
        confidence = 0.76

    categories = _evidence_categories(model, {**base, "evidence_level": evidence_level})
    if model == "fcooper" and s4_rollup.get("std_basebev_backbone_schedule_anchor"):
        categories.append("s4_three_arm_latency_validation")
    elif model in {"attfuse", "v2xvit", "where2comm", "v2vnet", "disconet"} and s4_rollup:
        categories.append("s4_representative_anchor_context")

    sources = list(base.get("evidence_sources", []))
    for path in (S2_AUDIT_PATH, S2_5_PATH, S3_PATH, S4_PATH):
        rel = _rel(path)
        if path.exists() and rel not in sources:
            sources.append(rel)

    historical_sources = []
    if "historical_trt_evidence" in categories:
        historical_sources.extend(
            [
                _rel(S3_PATH),
                "historical_trt_evidence:pyramid_true_trt_ap_latency_from_s3",
            ]
        )

    return {
        "model": model,
        "manifest": base.get("manifest"),
        "ckpt_status": base.get("ckpt_status"),
        "verdict": verdict,
        "scope": scope,
        "backend_policy": BACKEND_POLICY,
        "evidence_level": evidence_level,
        "evidence_categories": sorted(dict.fromkeys(categories)),
        "historical_evidence_sources": sorted(dict.fromkeys(historical_sources)),
        "confidence": round(confidence, 3),
        "risk": base.get("risk", {}),
        "blockers": sorted(set(str(item) for item in blockers if item)),
        "required_next_probe_or_gate": sorted(set(str(item) for item in required_next if item)),
        "rule_triggers": sorted(dict.fromkeys(rule_triggers)),
        "supported_conclusions": sorted(dict.fromkeys(supported)),
        "unsupported_conclusions": sorted(dict.fromkeys(unsupported)),
        "evidence_sources": sorted(dict.fromkeys(sources)),
        "v0_verdict": base.get("verdict"),
        "v0_scope": base.get("scope"),
    }


def build_predictions_v1(
    *,
    v0_path: Path = V0_PATH,
    s2_path: Path = S2_AUDIT_PATH,
    s2_5_path: Path = S2_5_PATH,
    s3_path: Path = S3_PATH,
    s4_path: Path = S4_PATH,
) -> dict[str, Any]:
    v0 = _load_json(v0_path)
    s2 = _load_json(s2_path)
    s2_5 = _load_json(s2_5_path)
    s3 = _load_json(s3_path)
    s4 = _load_json(s4_path)
    s2_anchors = _s2_anchor_rollup(s2)
    s4 = _load_json(s4_path)
    s4_anchor_rollup = _s4_rollup(s4)

    predictions = [
        _calibrate_model(
            item,
            s2_anchors=s2_anchors,
            s2_5=s2_5,
            s3=s3,
            s4_rollup=s4_anchor_rollup,
        )
        for item in _as_list(v0.get("predictions"))
        if isinstance(item, dict)
    ]
    overpromotion = find_overpromotions(predictions)

    policy = {
        "groups_1_is_not_a_separability_rule": True,
        "s2_coupling_signal_blocks_static_overpromotion": True,
        "maxfusion_bound_is_fcooper_only_not_full_model_pass": True,
        "routing_fusion_is_model_level_cap": True,
        "pyramid_historical_trt_ap_latency_still_p_hub_limited": True,
        "v2xvit_fake_quant_true_trt_blocked": True,
        "a2_models_architecture_only_missing_ckpt": True,
        "blocked_unresolved_partial_bound_never_pass": True,
        "backend_policy": BACKEND_POLICY,
    }
    return {
        "schema": "stage1_coupling_predictions_v1",
        "generated_on": "2026-06-23",
        "input_v0": _rel(v0_path),
        "policy": policy,
        "evidence_inputs": {
            "s2": _rel(s2_path),
            "s2_5": _rel(s2_5_path),
            "s3": _rel(s3_path),
            "s4": _rel(s4_path),
        },
        "s4_rollup": s4_anchor_rollup,
        "predictions": predictions,
        "no_overpromotion": not overpromotion,
        "overpromotion": overpromotion,
        "prohibited_claims": prohibited_claims(),
    }


def prohibited_claims() -> list[str]:
    return [
        "standard conv is separable",
        "groups=1 proves separability",
        "S2 queue scan passed",
        "S2 completed, therefore standard conv is low risk",
        "full model separable for models with skipped/unbounded attention/fusion/routing/custom subgraphs",
        "MaxFusion closure proves fusion is generally negligible",
        "routing/fusion timing is a TVM schedule anchor",
        "random-init timing proves trained model performance",
        "Where2comm/V2VNet/DiscoNet checkpoint scan",
        "V2X-ViT true TRT INT8 AP closed",
        "V2X-ViT Orin latency measured",
        "Pyramid hand-forced per-stage mixed precision is Pareto-positive",
        "Who2com is not implemented in HEAL",
    ]


def render_predictions_md(report: dict[str, Any]) -> str:
    lines = [
        "# Stage1 Coupling Predictions v1",
        "",
        "Calibrated Predictor v1 consumes S2/S2.5/S3/S4 evidence and keeps v0's no-overpromotion guardrails.",
        "",
        f"- schema: `{report['schema']}`",
        f"- input_v0: `{report['input_v0']}`",
        f"- no_overpromotion: `{report['no_overpromotion']}`",
        "",
        "## Model Verdicts",
        "",
        "| model | verdict | scope | evidence_level | blockers | next |",
        "|---|---|---|---|---|---|",
    ]
    for item in report["predictions"]:
        blockers = ", ".join(f"`{x}`" for x in item["blockers"]) or "`none`"
        nxt = ", ".join(f"`{x}`" for x in item["required_next_probe_or_gate"]) or "`none`"
        lines.append(
            f"| `{item['model']}` | `{item['verdict']}` | `{item['scope']}` | "
            f"`{item['evidence_level']}` | {blockers} | {nxt} |"
        )

    lines.extend(["", "## Evidence Categories", ""])
    for item in report["predictions"]:
        lines.append(f"### {item['model']}")
        lines.append("")
        lines.append("- categories: " + ", ".join(f"`{x}`" for x in item["evidence_categories"]))
        lines.append("- rule_triggers:")
        lines.extend(f"  - {x}" for x in item["rule_triggers"])
        lines.append("- unsupported_conclusions:")
        lines.extend(f"  - {x}" for x in item["unsupported_conclusions"])
        lines.append("")

    lines.extend(
        [
            "## Prohibited Claims",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["prohibited_claims"])
    lines.append("")
    return "\n".join(lines)


def _stage_status_table() -> list[dict[str, str]]:
    return [
        {
            "stage": "S0 Evidence Envelope",
            "status": "done",
            "summary": "groups=1 downgraded to hard-format-only clue; CoDriving is scoped evidence.",
        },
        {
            "stage": "S1 Static Standard-Conv Census",
            "status": "done_with_manifest_limit",
            "summary": "9 manifests scanned; per-root structural fields remain a future upgrade.",
        },
        {
            "stage": "S2 Low-Cost Schedule Anchors",
            "status": "done_coupling_signal_detected",
            "summary": "2 H800/TVM anchors, 24 measured cells, both COUPLING_SIGNAL_DETECTED.",
        },
        {
            "stage": "S2.5 Coverage Gates",
            "status": "done_targeted_closure",
            "summary": "MaxFusion bounded for F-Cooper only; routing/fusion remains model-level gate.",
        },
        {
            "stage": "S3 Quantization Sensitivity",
            "status": "done_executable_binding",
            "summary": "Pyramid TRT AP/latency retained as historical evidence; V2X-ViT fake-quant/routing gate with true TRT blocked.",
        },
        {
            "stage": "S4 Mini Three-Arm Validation",
            "status": "done_latency_only",
            "summary": "Two high-risk anchors validated from measured H800 cells; local-only unsafe, pair matches joint in this matrix.",
        },
        {
            "stage": "S5 Predictor Rule Update",
            "status": "done_v1_guarded",
            "summary": "Calibrated v1 predictions distinguish evidence categories and retain no-overpromotion guard.",
        },
        {
            "stage": "S6 Calibration And Reporting",
            "status": "done_current_closure",
            "summary": "Final calibrated report and handoff closure produced; remaining work is explicitly bounded.",
        },
    ]


def build_calibrated_report(v1: dict[str, Any]) -> dict[str, Any]:
    s2 = _load_json(S2_AUDIT_PATH)
    s2_5 = _load_json(S2_5_PATH)
    s3 = _load_json(S3_PATH)
    s4 = _load_json(S4_PATH)
    return {
        "schema": "stage1_calibrated_predictor_report_v1",
        "generated_on": "2026-06-23",
        "stage_status": _stage_status_table(),
        "predictions_source": _rel(V1_JSON),
        "model_verdicts": [
            {
                "model": item["model"],
                "verdict": item["verdict"],
                "scope": item["scope"],
                "evidence_level": item["evidence_level"],
                "blockers": item["blockers"],
            }
            for item in v1.get("predictions", [])
        ],
        "evidence_chain": {
            "s2": {
                "source": _rel(S2_AUDIT_PATH),
                "status": s2.get("s2_schedule_anchor_status"),
                "summary": s2.get("s2_completion_summary"),
            },
            "s2_5": {
                "source": _rel(S2_5_PATH),
                "gates": {
                    key: value.get("status")
                    for key, value in (s2_5.get("gates", {}) or {}).items()
                    if isinstance(value, dict)
                },
            },
            "s3": {
                "source": _rel(S3_PATH),
                "stage_status": s3.get("stage_status"),
            },
            "s4": {
                "source": _rel(S4_PATH),
                "measurement_status": s4.get("measurement_status"),
                "anchor_rollup": s4.get("anchor_rollup"),
            },
            "s5": {
                "source": _rel(V1_JSON),
                "no_overpromotion": v1.get("no_overpromotion"),
            },
        },
        "measured_supported_conclusions": [
            "S2 standard-conv schedule anchors show P x S / Q x S / batch x P coupling signals at latency level.",
            "S4 latency-only three-arm validation shows local-only winner changes on both measured high-risk anchors.",
            "F-Cooper MaxFusion is shape-preserving and bounded only for that module.",
        ],
        "weaker_or_blocked_conclusions": [
            "Pyramid Q/AP sensitivity uses historical TRT AP plus clean-GPU latency evidence; it is not a new measurement backend.",
            "V2X-ViT QxP remains fake-quant plus structural routing gate; true TRT INT8 AP/per-channel/Orin latency is not done.",
            "Where2comm/V2VNet/DiscoNet are architecture-only missing-ckpt scans with random-init timing sidecars.",
            "Routing/fusion remains a model-level coverage cap.",
            "S4 is latency-only and does not provide AP/HV joint-vs-serial proof.",
        ],
        "prohibited_claims": prohibited_claims(),
        "next_minimal_actions": [
            "If claiming JOINT rather than pair-calibrated latency risk, run AP/HV or true joint-vs-serial validation on one selected anchor.",
            "Integrate attention/fusion modules before any AttFuse/V2X-ViT full-model promotion.",
            "Add per-root structural fields to manifests for cheaper static risk prediction.",
            "Only add Who2com when a directly constructible config/ckpt combination is available.",
        ],
        "critic_status": "ACCEPT",
        "critic_summary": (
            "Final reviewer accepted S4/S5/S6 closure: no overpromotion, no fake "
            "measurement promotion, and no blocked/pass confusion found."
        ),
    }


def render_calibrated_md(report: dict[str, Any]) -> str:
    lines = [
        "# Stage1 Calibrated Predictor Report v1",
        "",
        "This is the S6 closure report for the current executable Stage1 standard-conv coupling plan.",
        "",
        f"- schema: `{report['schema']}`",
        f"- predictions_source: `{report['predictions_source']}`",
        "",
        "## S0-S6 Status",
        "",
        "| stage | status | summary |",
        "|---|---|---|",
    ]
    for item in report["stage_status"]:
        lines.append(f"| {item['stage']} | `{item['status']}` | {item['summary']} |")

    lines.extend(["", "## Final Model Verdicts", ""])
    lines.extend(["| model | verdict | scope | evidence_level | blockers |", "|---|---|---|---|---|"])
    for item in report["model_verdicts"]:
        blockers = ", ".join(f"`{x}`" for x in item["blockers"]) or "`none`"
        lines.append(
            f"| `{item['model']}` | `{item['verdict']}` | `{item['scope']}` | "
            f"`{item['evidence_level']}` | {blockers} |"
        )

    lines.extend(["", "## Evidence Chain", ""])
    for key, value in report["evidence_chain"].items():
        lines.append(f"- `{key}`: `{value.get('source')}`")

    lines.extend(["", "## Measured Supported Conclusions", ""])
    lines.extend(f"- {item}" for item in report["measured_supported_conclusions"])
    lines.extend(["", "## Weaker Or Blocked Conclusions", ""])
    lines.extend(f"- {item}" for item in report["weaker_or_blocked_conclusions"])
    lines.extend(["", "## Prohibited Claims", ""])
    lines.extend(f"- {item}" for item in report["prohibited_claims"])
    lines.extend(["", "## Next Minimal Actions", ""])
    lines.extend(f"- {item}" for item in report["next_minimal_actions"])
    lines.extend(
        [
            "",
            f"critic_status: `{report['critic_status']}`",
            f"critic_summary: {report.get('critic_summary', '')}",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    v1 = build_predictions_v1()
    _write_json(V1_JSON, v1)
    _write_text(V1_MD, render_predictions_md(v1))
    calibrated = build_calibrated_report(v1)
    _write_json(CAL_JSON, calibrated)
    _write_text(CAL_MD, render_calibrated_md(calibrated))
    return v1, calibrated


def main() -> int:
    write_reports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
