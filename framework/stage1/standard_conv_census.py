#!/usr/bin/env python3
"""Build a low-cost standard-conv coupling census from Stage1 manifests.

This script is intentionally static-only. It reads existing partition YAML files,
summarizes standard-convolution regimes, and selects anchor-probe candidates.
It does not run TVM, CUDA, model construction, or full enumeration.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DESIGN_DIR = ROOT / "multi_agent/methods/design/stage1-model-predict"
RESULTS_DIR = ROOT / "results/stage1_model_predict"
DEFAULT_MANIFESTS = [
    ROOT / "framework/partitions/codriving_partition.yaml",
    ROOT / "results/autoscan_fcooper_partition.yaml",
    ROOT / "results/autoscan_attfuse_partition.yaml",
    ROOT / "results/autoscan_where2comm_partition.yaml",
    ROOT / "results/autoscan_v2vnet_partition.yaml",
    ROOT / "results/autoscan_disconet_partition.yaml",
    ROOT / "framework/partitions/v2xvit_partition.yaml",
    ROOT / "framework/partitions/pyramid_lidar_partition.yaml",
    ROOT / "framework/partitions/pyramid_camera_partition.yaml",
]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def _classify_layer_family(layer: str) -> str:
    layer = layer or ""
    if "resnet.layer" in layer:
        if ".downsample." in layer:
            return "resnet_downsample_1x1_or_projection"
        return "resnet_basicblock_3x3"
    if ".blocks." in layer or "backbone_m1.blocks." in layer:
        return "base_bev_backbone_conv"
    if ".deblocks." in layer:
        return "deconv_neck"
    if "shrink" in layer or "shrinker" in layer:
        return "neck_double_conv"
    if "head" in layer:
        return "prediction_head"
    if "pyramid" in layer or "resnext" in layer:
        return "pyramid_or_resnext"
    return "other_conv"


def _width_regime(widths: list[int]) -> str:
    if not widths:
        return "unknown"
    max_width = max(widths)
    if max_width <= 64:
        return "small_<=64"
    if max_width <= 128:
        return "medium_<=128"
    if max_width <= 256:
        return "large_<=256"
    return "xlarge_>256"


def _skipped_types(skipped_modules: list[str]) -> list[str]:
    text = " ".join(str(item).lower() for item in skipped_modules)
    result: list[str] = []
    channel_preserving = (
        "0-param" in text
        or "channel-preserving" in text
        or "maxfusion" in text
        or "max pooling" in text
        or "无参数" in text
    )
    if (
        "attention" in text
        or "attfusion" in text
        or "att fusion" in text
        or "transformer" in text
        or "hmsa" in text
        or "mswin" in text
        or "注意力" in text
    ):
        result.append("attention_or_transformer")
    if "fusion" in text:
        result.append("fusion")
    if "sparse" in text or "pillar" in text or "scatter" in text:
        result.append("sparse_frontend")
    if channel_preserving:
        result.append("channel_preserving_or_pooling_fusion")
    return sorted(set(result))


def _measurement_status(model: str, family: str) -> str:
    if model == "codriving" and family.startswith("resnet"):
        return "measured_negative_anchor_but_b4_has_estimated_p25_p75_latency"
    if model == "codriving":
        return "partially_covered_by_codriving_negative_result"
    if model in {"fcooper", "attfuse", "v2xvit", "where2comm", "v2vnet", "disconet"}:
        return "unmeasured_standard_conv_family"
    if model.startswith("pyramid"):
        return "pyramid_context_mixed_grouped_and_standard"
    return "unknown"


def _anchor_priority(
    model: str,
    is_standard: bool,
    primary_families: list[str],
    coupled_families: list[str],
    skipped: list[str],
    measurement_status: str,
    contains_convtranspose: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not is_standard:
        return "not_standard_conv", reasons

    if "unmeasured_standard_conv_family" in measurement_status:
        reasons.append("standard_conv_family_has_no_direct_three_arm_or_anchor_probe")
    if "attention_or_transformer" in skipped:
        reasons.append("full_model_has_skipped_fusion_or_attention")
    elif "fusion" in skipped and "channel_preserving_or_pooling_fusion" not in skipped:
        reasons.append("full_model_has_skipped_unknown_fusion")
    elif "channel_preserving_or_pooling_fusion" in skipped:
        reasons.append("skipped_channel_preserving_fusion_needs_coverage_check")
    if "deconv_neck" in primary_families:
        reasons.append("deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet")
    if contains_convtranspose:
        reasons.append("mixed_conv2d_convtranspose_group_requires_subgroup_split")
    if model == "codriving" and "estimated_p25_p75" in measurement_status:
        reasons.append("codriving_negative_anchor_uses_estimated_latency_for_some_widths")
    if "prediction_head" in primary_families or "prediction_head" in coupled_families:
        reasons.append("heads_are_accuracy_sensitive_and_often_quantization_locked")

    if any("attention" in reason or "unknown_fusion" in reason for reason in reasons):
        return "high", reasons
    if any("unmeasured" in reason or "deconv" in reason for reason in reasons):
        return "medium", reasons
    if reasons:
        return "medium", reasons
    return "low", ["groups=1_only_removes_hard_ic_bn_format_cliff_not_all_coupling"]


def analyze_manifest(path: Path) -> dict[str, Any]:
    data = _load_yaml(path)
    model = str(data.get("model", path.stem))
    trace = data.get("trace", {}) or {}
    skipped_modules = [str(x) for x in _as_list(trace.get("skipped_modules"))]
    skipped = _skipped_types(skipped_modules)

    structural_groups = {
        str(group.get("group_id")): group
        for group in _as_list(data.get("view_b1_prune_groups"))
        if isinstance(group, dict) and group.get("group_id") is not None
    }
    layer_ops = {
        str(item.get("node")): str(item.get("op_type"))
        for item in _as_list(data.get("view_d_routing"))
        if isinstance(item, dict) and item.get("node")
    }

    candidates: list[dict[str, Any]] = []
    for search_group in _as_list(data.get("view_b1_search_groups")):
        if not isinstance(search_group, dict):
            continue

        member_ids = [str(x) for x in _as_list(search_group.get("member_b1_groups"))]
        missing_member_ids = [mid for mid in member_ids if mid not in structural_groups]
        members = [structural_groups[mid] for mid in member_ids if mid in structural_groups]
        roots = sorted({str(group.get("root_layer")) for group in members if group.get("root_layer")})
        member_layers = sorted(
            {
                str(layer)
                for group in members
                for layer in _as_list(group.get("member_layers"))
            }
        )
        op_types = sorted({layer_ops.get(layer, "unknown") for layer in member_layers})
        primary_families = sorted({_classify_layer_family(layer) for layer in roots})
        coupled_families = sorted({_classify_layer_family(layer) for layer in member_layers})
        families = sorted(set(primary_families + coupled_families))
        missing_group_metadata = [str(group.get("group_id", "unknown")) for group in members if "grouped_conv_g" not in group]
        grouped_values = [
            int(group.get("grouped_conv_g") or 1)
            for group in members
            if "grouped_conv_g" in group
        ]
        max_group = max(grouped_values, default=None)
        widths = [int(width) for width in _as_list(search_group.get("widths")) if width is not None]
        round_to = int(search_group.get("round_to", 0) or 0)
        int8_align = int(search_group.get("int8_buildable_align", 0) or 0)
        has_unknown_metadata = bool(missing_member_ids or missing_group_metadata or "unknown" in op_types)
        contains_convtranspose = "ConvTranspose2d" in op_types
        is_standard = (
            not has_unknown_metadata
            and max_group == 1
            and "Conv2d" in op_types
        )
        measurement_status = _measurement_status(model, families[0] if families else "")
        priority, reasons = _anchor_priority(
            model=model,
            is_standard=is_standard,
            primary_families=primary_families,
            coupled_families=coupled_families,
            skipped=skipped,
            measurement_status=measurement_status,
            contains_convtranspose=contains_convtranspose,
        )

        candidates.append(
            {
                "model": model,
                "manifest": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                "search_group_id": search_group.get("search_group_id"),
                "bucket": search_group.get("bucket"),
                "widths": widths,
                "width_regime": _width_regime(widths),
                "round_to": round_to,
                "int8_buildable_align": int8_align,
                "max_group": max_group,
                "is_standard_conv_candidate": is_standard,
                "standard_conv_candidate_scope": (
                    "mixed_conv2d_convtranspose_requires_split"
                    if is_standard and contains_convtranspose
                    else "pure_conv2d_group"
                    if is_standard
                    else "not_standard_conv_or_unknown"
                ),
                "has_unknown_group_metadata": has_unknown_metadata,
                "missing_member_ids": missing_member_ids,
                "missing_grouped_conv_g_ids": missing_group_metadata,
                "contains_convtranspose": contains_convtranspose,
                "primary_families": primary_families,
                "coupled_families": coupled_families,
                "families": families,
                "op_types": op_types,
                "root_layers": roots,
                "n_structural_groups": len(members),
                "skipped_subgraph_types": skipped,
                "measurement_status": measurement_status,
                "anchor_priority": priority,
                "risk_reasons": reasons,
                "manifest_shape_fields": "missing_cin_cout_kernel_hw",
            }
        )

    return {
        "model": model,
        "manifest": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "scan_status": data.get("scan_status"),
        "trace_entry_shape": trace.get("entry_shape"),
        "skipped_modules": skipped_modules,
        "skipped_subgraph_types": skipped,
        "n_candidates": len(candidates),
        "candidates": candidates,
    }


def build_report(manifests: list[Path]) -> dict[str, Any]:
    models = [analyze_manifest(path) for path in manifests if path.exists()]
    all_candidates = [item for model in models for item in model["candidates"]]

    counts = {
        "models": len(models),
        "search_groups": len(all_candidates),
        "standard_conv_candidates": sum(1 for item in all_candidates if item["is_standard_conv_candidate"]),
        "high_priority_anchors": sum(1 for item in all_candidates if item["anchor_priority"] == "high"),
        "medium_priority_anchors": sum(1 for item in all_candidates if item["anchor_priority"] == "medium"),
    }

    family_counts = Counter(
        family
        for item in all_candidates
        if item["is_standard_conv_candidate"]
        for family in item["primary_families"]
    )
    model_counts = Counter(
        item["model"]
        for item in all_candidates
        if item["is_standard_conv_candidate"]
    )
    priority_counts = Counter(
        item["anchor_priority"]
        for item in all_candidates
        if item["is_standard_conv_candidate"]
    )

    anchors = [
        item
        for item in all_candidates
        if item["is_standard_conv_candidate"] and item["anchor_priority"] in {"high", "medium"}
    ]
    anchors.sort(key=lambda item: ({"high": 0, "medium": 1, "low": 2}.get(item["anchor_priority"], 9), item["model"], str(item["search_group_id"])))

    probe_queue = [
        {
            "probe_id": "std_basebev_backbone_schedule_anchor",
            "priority": "high",
            "representative_models": sorted(
                {
                    item["model"]
                    for item in anchors
                    if "base_bev_backbone_conv" in item["primary_families"]
                }
            ),
            "search_groups": ["backbone.s0", "backbone.s1", "backbone.s2"],
            "question": "Does OpenCOOD BaseBEVBackbone standard Conv2d have schedule-gain drift across widths, or does it behave like CoDriving ResNet?",
            "minimal_measurement": "For one representative dense backbone, measure base and one pruned/boundary width under fp16/int8 default and tuned schedules.",
            "stop_condition": "If tuning-ratio variance is small and no argmin drift appears, mark this family anchor-probed low schedule risk.",
        },
        {
            "probe_id": "attention_fusion_coverage_anchor",
            "priority": "high",
            "representative_models": sorted(
                {
                    item["model"]
                    for item in anchors
                    if "attention_or_transformer" in item["skipped_subgraph_types"]
                }
            ),
            "search_groups": ["fusion_net", "traced_dense_backbone"],
            "question": "Can dense-backbone separability be promoted to full-model separability when attention/transformer fusion is skipped?",
            "minimal_measurement": "No attention measurement in safe predictor v0: attach typed skipped-subgraph metadata and keep full-model verdict blocked until attention trace/export exists.",
            "stop_condition": "If attention/fusion is skipped or not integrated, full-model verdict remains FUSION_UNCOVERED_UNKNOWN.",
        },
        {
            "probe_id": "v2xvit_qgranularity_p_anchor",
            "priority": "high",
            "representative_models": ["v2xvit"],
            "search_groups": ["backbone", "fusion_net", "quant_units"],
            "question": "Does the known V2X-ViT Q-granularity x P interaction block a cheap full-model separability prediction?",
            "minimal_measurement": "Bind C4-style per-channel/per-tensor AP and latency evidence to one base/pruned P pair; do not infer from buildability alone.",
            "stop_condition": "If per-channel vs per-tensor rank or AP delta changes across P beyond the noise floor, keep V2X-ViT in JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND.",
        },
        {
            "probe_id": "routing_fusion_coverage_anchor",
            "priority": "high",
            "representative_models": sorted(
                {
                    "v2xvit",
                    "attfuse",
                    "fcooper",
                    *(
                        item["model"]
                        for item in anchors
                        if "attention_or_transformer" in item["skipped_subgraph_types"]
                        or (
                            "fusion" in item["skipped_subgraph_types"]
                            and "channel_preserving_or_pooling_fusion" not in item["skipped_subgraph_types"]
                        )
                    ),
                }
            ),
            "search_groups": ["fusion_net", "view_d_routing", "traced_dense_backbone"],
            "question": "Do skipped fusion/routing operators change the feasible Q/S action set or only add a constant uncovered cost?",
            "minimal_measurement": "Bind existing C5-style routing feasibility/op-blacklist evidence where available; otherwise attach typed skip metadata and keep fusion/routing as a blocker.",
            "stop_condition": "If routing feasibility differs by Q/S or fusion remains unbounded/unintegrated, full-model verdict cannot exceed uncovered/low-confidence.",
        },
        {
            "probe_id": "codriving_resnet_completion_anchor",
            "priority": "medium",
            "representative_models": ["codriving"],
            "search_groups": ["backbone.s0", "backbone.s1", "backbone.s2"],
            "question": "Does CoDriving remain negative when estimated p25/p75 latencies are replaced by real anchor measurements?",
            "minimal_measurement": "Remeasure the missing/pruned CoDriving widths with the same H800 TVM protocol used for base/p50.",
            "stop_condition": "If tuning ratios remain near-constant and serial HV still equals joint HV, keep CoDriving as measured negative anchor.",
        },
        {
            "probe_id": "standard_neck_deconv_anchor",
            "priority": "medium",
            "representative_models": sorted(
                {
                    item["model"]
                    for item in anchors
                    if "deconv_neck" in item["primary_families"]
                    or "neck_double_conv" in item["primary_families"]
                }
            ),
            "search_groups": ["neck"],
            "question": "Do deconv/neck standard-conv regimes share CoDriving ResNet separability, or do they need separate S-axis treatment?",
            "minimal_measurement": "Probe one neck/deconv representative with default vs tuned schedule and fp16/int8 schedule-swap.",
            "stop_condition": "If no drift appears, mark neck/deconv as anchor-probed low schedule risk.",
        },
        {
            "probe_id": "pyramid_mixed_p_hub_context_anchor",
            "priority": "medium",
            "representative_models": ["pyramid_camera", "pyramid_lidar"],
            "search_groups": ["bev_encoder", "backbone", "neck"],
            "question": "Does a local standard-conv group sit inside a model whose grouped-conv IC_BN P-hub dominates the architecture-level verdict?",
            "minimal_measurement": "Use existing C2/C3/C7 Pyramid evidence plus manifest adjacency to mark local standard Conv2d as dense-subgraph-only.",
            "stop_condition": "If grouped-conv P-hub neighbors share pruning/fanout or downstream heads, block any model-level standard-conv low-risk promotion.",
        },
        {
            "probe_id": "per_stage_q_ap_sensitivity_anchor",
            "priority": "medium",
            "representative_models": ["v2xvit", "pyramid_camera", "pyramid_lidar"],
            "search_groups": ["backbone", "bev_encoder", "quant_units"],
            "question": "Does pruning change which stage should keep higher-precision or per-channel quantization for AP?",
            "minimal_measurement": "One per-stage Q/AP sensitivity check after pruning, using existing AP-noise thresholds and no full P x Q grid.",
            "stop_condition": "If stage-wise Q choice or AP delta changes across P, require pair/joint treatment for Q decisions even when latency is stable.",
        },
        {
            "probe_id": "maxfusion_coverage_anchor",
            "priority": "medium",
            "representative_models": ["fcooper"],
            "search_groups": ["fusion_net", "traced_dense_backbone"],
            "question": "Is F-Cooper MaxFusion truly non-blocking for full-model separability?",
            "minimal_measurement": "Measure or statically prove MaxFusion latency/shape preservation once; do not use it as a blanket standard-conv rule.",
            "stop_condition": "If MaxFusion cost is negligible and shape-preserving, allow dense-subgraph low-risk wording but keep evidence level explicit.",
        },
    ]

    return {
        "schema": "standard_conv_coupling_census_v1",
        "source_manifests": [str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path) for path in manifests if path.exists()],
        "summary": counts,
        "counts_by_model": dict(sorted(model_counts.items())),
        "counts_by_family": dict(sorted(family_counts.items())),
        "counts_by_anchor_priority": dict(sorted(priority_counts.items())),
        "models": models,
        "anchor_probe_candidates": anchors,
        "recommended_probe_queue": probe_queue,
        "interpretation": {
            "groups_1_is_not_a_verdict": "groups=1 removes the grouped-conv IC_BN hard-cliff mechanism, but does not prove P/Q/S separability.",
            "current_gap": "Existing manifests lack cin/cout/kernel/HW, so this census is a regime selector; Stage1 must add those fields before calibrated prediction.",
            "next_step": "Run low-cost anchor probes only for high/medium candidates, then calibrate predictor verdicts.",
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Standard Conv Coupling Census v1")
    lines.append("")
    lines.append("This is a static-only census from existing Stage1 partition manifests. It is not a coupling verdict.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Counts By Model")
    lines.append("")
    for key, value in report["counts_by_model"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Counts By Family")
    lines.append("")
    for key, value in report["counts_by_family"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Anchor Probe Candidates")
    lines.append("")
    lines.append("| priority | model | search_group | bucket | widths | families | reasons |")
    lines.append("|---|---|---|---|---|---|---|")
    for item in report["anchor_probe_candidates"]:
        reasons = "<br>".join(item["risk_reasons"]) if item["risk_reasons"] else "-"
        families = "<br>".join(item["primary_families"]) if item["primary_families"] else "-"
        lines.append(
            "| {priority} | {model} | {group} | {bucket} | {widths} | {families} | {reasons} |".format(
                priority=item["anchor_priority"],
                model=item["model"],
                group=item["search_group_id"],
                bucket=item["bucket"],
                widths=",".join(str(x) for x in item["widths"]),
                families=families,
                reasons=reasons,
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for value in report["interpretation"].values():
        lines.append(f"- {value}")
    lines.append("")
    lines.append("## Recommended Low-Cost Probe Queue")
    lines.append("")
    lines.append("| priority | probe_id | representative_models | question | minimal_measurement |")
    lines.append("|---|---|---|---|---|")
    for item in report["recommended_probe_queue"]:
        lines.append(
            "| {priority} | {probe_id} | {models} | {question} | {measurement} |".format(
                priority=item["priority"],
                probe_id=item["probe_id"],
                models=", ".join(item["representative_models"]) or "-",
                question=item["question"],
                measurement=item["minimal_measurement"],
            )
        )
    lines.append("")
    lines.append("## Required Manifest Upgrade")
    lines.append("")
    lines.append("Every Conv2d/ConvTranspose2d group needs cin, cout, groups, kernel, stride, input HW, output HW, and fanout fields.")
    lines.append("Without these fields, the predictor can only select probe candidates; it cannot make calibrated architecture-level claims.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, help="Partition YAML path. Repeatable.")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "standard_conv_census_v1.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DESIGN_DIR / "standard_conv_census_v1.md",
    )
    args = parser.parse_args()

    manifests = args.manifest if args.manifest else DEFAULT_MANIFESTS
    report = build_report([path.resolve() for path in manifests])
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
