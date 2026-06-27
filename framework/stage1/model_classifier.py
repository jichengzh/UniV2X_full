#!/usr/bin/env python3
"""End-to-end Stage1 model classifier.

The classifier consumes Stage1 partition manifests plus existing S2/S2.5/S3/S4
evidence. It does not run new probes. New measured evidence is H800 TVM only;
TRT is retained as historical evidence.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from framework.stage1.coupling_predictor import (
    BACKEND_POLICY,
    _normalize_skipped_subgraphs,
    find_overpromotions,
    predict_manifests,
)
from framework.stage1.trace_plan import legacy_trace_plan_from_manifest


ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = ROOT / "results/stage1_model_predict"

S2_AUDIT_PATH = STAGE1_DIR / "s2_schedule_anchor_audit_v1.json"
S2_COMPLETION_PATH = STAGE1_DIR / "s2_probe_results/stage1_s2_probe_completion_v1.json"
S2_5_PATH = STAGE1_DIR / "s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json"
S3_PATH = STAGE1_DIR / "s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json"
S4_PATH = STAGE1_DIR / "s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json"
STAGE2_DELTA_DIR = STAGE1_DIR / "stage2_evidence_delta"

DEFAULT_OUT_JSON = (
    STAGE1_DIR / "model_classifier/stage1_model_classification_v1.json"
)
DEFAULT_OUT_MD = (
    STAGE1_DIR / "model_classifier/stage1_model_classification_v1.md"
)

DEFAULT_MANIFESTS = [
    ROOT / "framework/partitions/codriving_partition.yaml",
    ROOT / "results/autoscan_fcooper_partition.yaml",
    ROOT / "results/autoscan_attfuse_partition.yaml",
    ROOT / "framework/partitions/v2xvit_partition.yaml",
    ROOT / "framework/partitions/pyramid_lidar_partition.yaml",
    ROOT / "framework/partitions/pyramid_camera_partition.yaml",
    ROOT / "results/autoscan_where2comm_partition.yaml",
    ROOT / "results/autoscan_v2vnet_partition.yaml",
    ROOT / "results/autoscan_disconet_partition.yaml",
]

MODEL_ORDER = [
    "codriving",
    "fcooper",
    "attfuse",
    "v2xvit",
    "pyramid_lidar",
    "pyramid_camera",
    "where2comm",
    "v2vnet",
    "disconet",
]

ACCELERATION_CLASS_LABELS = {
    "CO_ACCELERATION_REQUIRED": "需要协同加速",
    "SEPARABLE_ACCELERATION": "可分离加速",
    "SCAN_FAILED": "扫描失败",
}


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_yaml(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _op_types_from_layers(member_layers: Iterable[str]) -> list[str]:
    op_types = set()
    for name in member_layers:
        low = str(name).lower()
        if "bn" in low or "batchnorm" in low:
            op_types.add("BatchNorm2d")
        if "deblock" in low or "deconv" in low:
            op_types.add("ConvTranspose2d")
        if "conv" in low or "backbone" in low or "shrink" in low or "head" in low:
            op_types.add("Conv2d")
        if "linear" in low or "fc" in low:
            op_types.add("Linear")
    return sorted(op_types) or ["unknown"]


def _fallback_b1_feature(group: dict[str, Any]) -> dict[str, Any]:
    member_layers = [str(item) for item in _as_list(group.get("member_layers"))]
    groups = int(group.get("grouped_conv_g", group.get("groups", 1)) or 1)
    cout = group.get("cur_width")
    fanout = group.get("coupled_buckets")
    if fanout is None:
        fanout = [group.get("bucket")] if group.get("bucket") else []
    return {
        "cin": None,
        "cout": int(cout) if isinstance(cout, int) else cout,
        "groups": groups,
        "ic_bn": None,
        "kernel": None,
        "stride": None,
        "op_types": _op_types_from_layers(member_layers),
        "fanout_buckets": sorted({str(item) for item in _as_list(fanout) if item}),
    }


def _rollup_feature(groups: list[dict[str, Any]]) -> dict[str, Any]:
    features = [g.get("feature", {}) or {} for g in groups]
    ic_bn = [
        float(f["ic_bn"])
        for f in features
        if f.get("ic_bn") is not None
    ]
    max_groups = max(
        [int(f.get("groups") or 1) for f in features] or [1]
    )
    op_types = sorted(
        {
            str(op)
            for feature in features
            for op in _as_list(feature.get("op_types"))
            if op
        }
    )
    fanout = sorted(
        {
            str(bucket)
            for feature in features
            for bucket in _as_list(feature.get("fanout_buckets"))
            if bucket
        }
    )
    return {
        "min_ic_bn": min(ic_bn) if ic_bn else None,
        "max_groups": max_groups,
        "op_types": op_types or ["unknown"],
        "fanout_buckets": fanout,
    }


def _coverage_block(status: str | None) -> dict[str, Any]:
    trace_pct = None if status in {None, "skipped"} else 100.0
    return {
        "coverage_scope": "trace_net_only",
        "trace_net_latency_pct": trace_pct,
        "full_model_latency_pct": None,
        "skipped_subgraphs_accounted_separately": True,
        "note": (
            "Trace-net latency coverage only; this is not full-model coverage. "
            "Skipped sparse/fusion/attention/custom subgraphs are accounted "
            "separately by trace.skipped_subgraphs."
        ),
    }


def normalize_manifest_for_predictor(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a classifier-ready manifest without mutating the input."""

    data = deepcopy(manifest)
    trace_plan = data.get("trace_plan")
    if not isinstance(trace_plan, dict):
        trace_plan = legacy_trace_plan_from_manifest(data)
    data["trace_plan"] = trace_plan

    trace = dict(data.get("trace", {}) or {})
    trace.setdefault("skipped_modules", [])
    plan_skips = [
        dict(item)
        for item in _as_list(trace_plan.get("skipped_subgraphs"))
        if isinstance(item, dict)
    ]
    if plan_skips:
        trace["skipped_subgraphs"] = [
            *[
                dict(item)
                for item in _as_list(trace.get("skipped_subgraphs"))
                if isinstance(item, dict)
            ],
            *plan_skips,
        ]
    trace["skipped_subgraphs"] = _normalize_skipped_subgraphs({**data, "trace": trace})
    data["trace"] = trace

    b1_groups = []
    for group in _as_list(data.get("view_b1_prune_groups")):
        if not isinstance(group, dict):
            continue
        item = dict(group)
        item.setdefault("feature", _fallback_b1_feature(item))
        b1_groups.append(item)
    data["view_b1_prune_groups"] = b1_groups

    by_group_id = {str(group.get("group_id")): group for group in b1_groups}
    search_groups = []
    for group in _as_list(data.get("view_b1_search_groups")):
        if not isinstance(group, dict):
            continue
        item = dict(group)
        members = [
            by_group_id[str(group_id)]
            for group_id in _as_list(item.get("member_b1_groups"))
            if str(group_id) in by_group_id
        ]
        if not members and b1_groups:
            members = [
                g
                for g in b1_groups
                if g.get("bucket") == item.get("bucket")
            ]
        item.setdefault("feature", _rollup_feature(members))
        search_groups.append(item)
    data["view_b1_search_groups"] = search_groups

    latency = dict(data.get("view_latency", {}) or {})
    status = latency.get("status") or data.get("search_space_summary", {}).get("latency_status")
    latency.setdefault("status", status or "unknown")
    latency.setdefault("coverage", _coverage_block(latency.get("status")))
    data["view_latency"] = latency
    return data


def _evidence_inputs(evidence_dir: Path) -> dict[str, str]:
    paths = {
        "s2": evidence_dir / "s2_schedule_anchor_audit_v1.json",
        "s2_completion": evidence_dir
        / "s2_probe_results/stage1_s2_probe_completion_v1.json",
        "s2_5": evidence_dir
        / "s2_5_coverage_gates/stage1_s2_5_coverage_gate_closure_v1.json",
        "s3": evidence_dir
        / "s3_quant_sensitivity/stage1_s3_quant_sensitivity_v1.json",
        "s4": evidence_dir
        / "s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json",
        "stage2_evidence_delta": evidence_dir / "stage2_evidence_delta",
    }
    return {key: _rel(path) for key, path in paths.items()}


def _common_sources(base: dict[str, Any], evidence_dir: Path) -> list[str]:
    sources = list(base.get("evidence_sources", []))
    sources.extend(_evidence_inputs(evidence_dir).values())
    return sorted(dict.fromkeys(str(item) for item in sources if item))


def _historical_trt_sources(model: str) -> list[str]:
    if not model.startswith("pyramid"):
        return []
    return [
        _rel(S3_PATH),
        "historical_trt_evidence:pyramid_true_trt_ap_latency_from_s3",
    ]


def _measured_h800_for_model(model: str, evidence_dir: Path) -> dict[str, Any]:
    if model in {"fcooper", "attfuse", "v2xvit"}:
        return {
            "backend": "h800_tvm",
            "measurement_type": "latency",
            "sources": [
                _rel(evidence_dir / "s2_schedule_anchor_audit_v1.json"),
                _rel(evidence_dir / "s4_three_arm_validation/stage1_s4_three_arm_validation_v1.json"),
            ],
            "scope": "synthetic_dense_anchor_latency_only",
        }
    if model == "codriving":
        return {
            "backend": "h800_tvm",
            "measurement_type": "scoped_existing_anchor_latency",
            "sources": [
                "results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json",
                _rel(evidence_dir / "s2_schedule_anchor_audit_v1.json"),
            ],
            "scope": "codriving_measured_resnet_backbone_envelope",
        }
    return {}


def _classification_for_model(model: str) -> str:
    if model == "codriving":
        return "SCOPED_MEASURED_NEGATIVE_ANCHOR"
    if model == "fcooper":
        return "PAIR_CALIBRATION_REQUIRED_FULL_MODEL_GATED"
    if model == "attfuse":
        return "FUSION_UNCOVERED_STATIC_DENSE_PROMOTION_BLOCKED"
    if model == "v2xvit":
        return "PAIR_OR_JOINT_GATE_FAKE_QUANT_TRUE_TRT_BLOCKED"
    if model.startswith("pyramid"):
        return "P_HUB_CONTEXT_WITH_HISTORICAL_TRT_Q_AP_BOUND"
    if model in {"where2comm", "v2vnet", "disconet"}:
        return "ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN"
    return "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE"


def _acceleration_class_for_model(model: str, base: dict[str, Any]) -> str:
    ckpt_status = str(base.get("ckpt_status") or "")
    trace_plan = base.get("trace_plan", {}) if isinstance(base.get("trace_plan"), dict) else {}
    trace_ckpt_status = str(trace_plan.get("ckpt_status") or "")
    if ckpt_status == "missing_architecture_scan_only" or "missing" in trace_ckpt_status:
        return "SCAN_FAILED"
    if model == "codriving":
        return "SEPARABLE_ACCELERATION"
    return "CO_ACCELERATION_REQUIRED"


def _acceleration_class_reason(model: str, accel_class: str) -> str:
    if accel_class == "SCAN_FAILED":
        return (
            "No trained checkpoint classification is available; only an "
            "architecture-only dense-core scan plus random-init fusion sidecar "
            "exists, so this is not accepted as a model scan."
        )
    if accel_class == "SEPARABLE_ACCELERATION":
        return (
            "Scoped to the CoDriving measured dense ResNet backbone envelope; "
            "the scan has no grouped-conv int8 buildability cliff in the traced "
            "dense space, but this is not a cross-model or skipped-subgraph claim."
        )
    if model.startswith("pyramid"):
        return (
            "Pyramid keeps the P-hub/grouped-conv context and historical TRT "
            "Q/AP evidence only as context, so model-level separable acceleration "
            "is blocked."
        )
    return (
        "The model remains gated by measured H800 TVM latency coupling context "
        "and/or unclosed fusion, attention, routing, or true-INT8 evidence."
    )


def _evidence_level_for_model(model: str, base: dict[str, Any]) -> str:
    if model == "fcooper":
        return "measured_h800_tvm_latency_pair_calibration_required"
    if model == "attfuse":
        return "measured_h800_tvm_latency_context_plus_fusion_uncovered"
    if model in {"where2comm", "v2vnet", "disconet"}:
        return "architecture_only_missing_ckpt_plus_random_init_fusion_sidecar"
    if model == "v2xvit":
        return "fake_quant_qxp_plus_structural_routing_true_trt_blocked"
    if model.startswith("pyramid"):
        return "historical_trt_evidence_plus_p_hub_context"
    if model == "codriving":
        return "scoped_existing_anchor_plus_h800_tvm_context"
    return str(base.get("evidence_level") or "low_confidence")


def _extra_blockers(model: str) -> list[str]:
    if model == "v2xvit":
        return ["true TRT INT8 AP not done / blocked"]
    if model.startswith("pyramid"):
        return ["pyramid_mixed_p_hub_context_anchor"]
    return []


def _unsupported_for_model(model: str, base: dict[str, Any]) -> list[str]:
    unsupported = set(str(item) for item in _as_list(base.get("unsupported_conclusions")))
    unsupported.update(
        {
            "full-model separability",
            "groups=1 as a model-level separability proof",
            "no int8 buildability cliff as a model-level separability proof",
        }
    )
    if model == "codriving":
        unsupported.add("cross-model extrapolation from CoDriving to other models")
    if model == "v2xvit":
        unsupported.add("V2X-ViT true TRT INT8 AP closure")
        unsupported.add("V2X-ViT true TRT INT8 AP not done")
    if model.startswith("pyramid"):
        unsupported.add("Pyramid historical TRT evidence as a new measurement backend")
        unsupported.add("Pyramid model-level standard-conv separability")
    if model in {"where2comm", "v2vnet", "disconet"}:
        unsupported.add(f"{model} trained checkpoint scan")
        unsupported.add("random-init fusion timing as trained model evidence")
    if model in {"attfuse", "v2xvit", "where2comm", "v2vnet", "disconet"}:
        unsupported.add(f"{model} full-model separability before fusion/attention trace closure")
    return sorted(unsupported)


def _model_record(base: dict[str, Any], evidence_dir: Path) -> dict[str, Any]:
    model = str(base.get("model", "unknown"))
    trace_plan = base.get("trace_plan", {}) if isinstance(base.get("trace_plan"), dict) else {}
    classification = _classification_for_model(model)
    blockers = sorted(
        set(str(item) for item in _as_list(base.get("blockers")) + _extra_blockers(model) if item)
    )
    historical_sources = _historical_trt_sources(model)
    historical_labels = ["historical_trt_evidence"] if historical_sources else []
    measured = _measured_h800_for_model(model, evidence_dir)
    evidence_level = _evidence_level_for_model(model, base)
    acceleration_class = _acceleration_class_for_model(model, base)
    return {
        "model": model,
        "manifest": base.get("manifest"),
        "ckpt_status": base.get("ckpt_status"),
        "acceleration_class": acceleration_class,
        "acceleration_class_label": ACCELERATION_CLASS_LABELS[acceleration_class],
        "acceleration_class_reason": _acceleration_class_reason(model, acceleration_class),
        "classification": classification,
        "verdict": classification,
        "scope": base.get("scope"),
        "backend_policy": BACKEND_POLICY,
        "evidence_level": evidence_level,
        "trace_confidence": trace_plan.get("trace_confidence"),
        "coverage_scope": trace_plan.get("coverage_scope"),
        "manual_override_used": bool(trace_plan.get("manual_override_used", False)),
        "review_required": bool(trace_plan.get("review_required", False)),
        "included_modules": trace_plan.get("included_modules", []),
        "ignored_layers": trace_plan.get("ignored_layers", []),
        "skipped_subgraphs": trace_plan.get("skipped_subgraphs", []),
        "rejected_candidates": trace_plan.get("rejected_candidates", []),
        "trace_plan": {
            "schema": trace_plan.get("schema"),
            "detector": trace_plan.get("detector"),
            "trace_confidence": trace_plan.get("trace_confidence"),
            "coverage_scope": trace_plan.get("coverage_scope"),
            "manual_override_used": bool(trace_plan.get("manual_override_used", False)),
            "review_required": bool(trace_plan.get("review_required", False)),
            "review_reasons": _as_list(trace_plan.get("review_reasons")),
            "selected_candidate": trace_plan.get("selected_candidate"),
            "included_modules": trace_plan.get("included_modules", []),
            "ignored_layers": trace_plan.get("ignored_layers", []),
            "skipped_subgraphs": trace_plan.get("skipped_subgraphs", []),
            "rejected_candidates": trace_plan.get("rejected_candidates", []),
        },
        "evidence_sources": _common_sources(base, evidence_dir),
        "historical_evidence_sources": historical_sources,
        "historical_labels": historical_labels,
        "measured_h800_tvm": measured,
        "blockers": blockers,
        "required_next_probe_or_gate": sorted(
            set(str(item) for item in _as_list(base.get("required_next_probe_or_gate")) if item)
        ),
        "unsupported_conclusions": _unsupported_for_model(model, base),
        "no_overpromotion": True,
    }


def build_classification_report(
    manifests: Iterable[Path | str] | None = None,
    *,
    evidence_dir: Path | str = STAGE1_DIR,
) -> dict[str, Any]:
    evidence_dir = Path(evidence_dir)
    manifest_paths = [Path(path) for path in (manifests or DEFAULT_MANIFESTS)]
    v0 = predict_manifests(manifest_paths, evidence_dir=evidence_dir)
    models = [_model_record(item, evidence_dir) for item in v0.get("predictions", [])]
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    models.sort(key=lambda item: order.get(str(item.get("model")), 999))
    overpromotion = find_overpromotions(models)
    return {
        "schema": "stage1_model_classification_v1",
        "generated_on": "2026-06-24",
        "backend_policy": BACKEND_POLICY,
        "acceleration_class_labels": ACCELERATION_CLASS_LABELS,
        "evidence_inputs": _evidence_inputs(evidence_dir),
        "source_manifests": [_rel(path) for path in manifest_paths],
        "models": models,
        "no_overpromotion": not overpromotion,
        "overpromotion": overpromotion,
        "unsupported_global_conclusions": [
            "TRT as a new measurement backend",
            "TRT as the classifier default backend",
            "groups=1 proves model-level separability",
            "no int8 buildability cliff proves model-level separability",
            "S4 latency-only evidence proves AP/HV joint-vs-serial behavior",
        ],
    }


def render_classification_md(report: dict[str, Any]) -> str:
    lines = [
        "# Stage1 Model Classification v1",
        "",
        "This report classifies Stage1 model-level evidence without running new probes.",
        "",
        f"- schema: `{report['schema']}`",
        f"- default_new_measurement_backend: `{report['backend_policy']['default_new_measurement_backend']}`",
        f"- allowed_new_measurement_backends: `{', '.join(report['backend_policy']['allowed_new_measurement_backends'])}`",
        f"- historical_trt_evidence_only: `{report['backend_policy']['historical_trt_evidence_only']}`",
        f"- no_overpromotion: `{report['no_overpromotion']}`",
        "",
        "## Models",
        "",
        "| model | acceleration_class | classification | scope | evidence_level | blockers | next |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report["models"]:
        blockers = ", ".join(f"`{x}`" for x in item["blockers"]) or "`none`"
        nxt = ", ".join(f"`{x}`" for x in item["required_next_probe_or_gate"]) or "`none`"
        lines.append(
            f"| `{item['model']}` | `{item['acceleration_class_label']}` | "
            f"`{item['classification']}` | `{item['scope']}` | "
            f"`{item['evidence_level']}` | {blockers} | {nxt} |"
        )

    lines.extend(["", "## Three-Class Summary", ""])
    lines.extend([
        "| model | acceleration_class | label | reason |",
        "|---|---|---|---|",
    ])
    for item in report["models"]:
        lines.append(
            f"| `{item['model']}` | `{item['acceleration_class']}` | "
            f"{item['acceleration_class_label']} | {item['acceleration_class_reason']} |"
        )

    lines.extend(["", "## Trace Plan Summary", ""])
    lines.extend([
        "| model | detector | candidate | confidence | coverage | manual_override | review_required | skipped | rejected |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for item in report["models"]:
        plan = item.get("trace_plan", {}) or {}
        candidate = plan.get("selected_candidate", {}) or {}
        skipped = len(plan.get("skipped_subgraphs", []) or [])
        rejected = len(plan.get("rejected_candidates", []) or [])
        lines.append(
            f"| `{item['model']}` | `{plan.get('detector')}` | "
            f"`{candidate.get('candidate_id')}` | `{item.get('trace_confidence')}` | "
            f"`{item.get('coverage_scope')}` | `{item.get('manual_override_used')}` | "
            f"`{item.get('review_required')}` | `{skipped}` | `{rejected}` |"
        )

    lines.extend(["", "## Historical Evidence", ""])
    for item in report["models"]:
        if item["historical_evidence_sources"]:
            lines.append(
                f"- `{item['model']}`: "
                + ", ".join(f"`{src}`" for src in item["historical_evidence_sources"])
            )

    lines.extend(["", "## Unsupported Global Conclusions", ""])
    lines.extend(f"- {item}" for item in report["unsupported_global_conclusions"])
    lines.append("")
    return "\n".join(lines)


def write_classification_report(
    *,
    manifests: Iterable[Path | str] | None = None,
    evidence_dir: Path | str = STAGE1_DIR,
    out_json: Path | str = DEFAULT_OUT_JSON,
    out_md: Path | str = DEFAULT_OUT_MD,
) -> dict[str, Any]:
    report = build_classification_report(manifests, evidence_dir=evidence_dir)
    _write_json(Path(out_json), report)
    _write_text(Path(out_md), render_classification_md(report))
    return report


__all__ = [
    "BACKEND_POLICY",
    "DEFAULT_MANIFESTS",
    "DEFAULT_OUT_JSON",
    "DEFAULT_OUT_MD",
    "ACCELERATION_CLASS_LABELS",
    "build_classification_report",
    "normalize_manifest_for_predictor",
    "render_classification_md",
    "write_classification_report",
]
