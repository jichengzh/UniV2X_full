#!/usr/bin/env python3
"""Build Stage1 S2 probe completion artifacts from measured anchors and gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MEASURED_PROBES = {
    "std_basebev_backbone_schedule_anchor",
    "standard_neck_deconv_anchor",
}

EXPECTED_P_LABELS = ("base", "boundary", "moderate")
EXPECTED_PRECISIONS = ("fp16", "int8")


NON_MEASURED_STATUS = {
    "attention_fusion_coverage_anchor": {
        "scan_status": "BLOCKED_NOT_RUNNABLE",
        "verdict": "BLOCKER_FULL_MODEL_NOT_PROMOTABLE",
        "evidence_kind": "typed_skip_guardrail",
        "summary": (
            "Attention/transformer fusion is not integrated into the current "
            "trace/export path, so the valid scan result is a blocker rather "
            "than a latency pass."
        ),
        "next_action": "Do not run a fake attention probe; integrate attention trace/export first.",
    },
    "routing_fusion_coverage_anchor": {
        "scan_status": "PARTIAL_BOUND_EXISTING_EVIDENCE",
        "verdict": "FUSION_ROUTING_REMAINS_MODEL_LEVEL_GATE",
        "evidence_kind": "typed_skip_plus_fusion_timing_sidecar",
        "summary": (
            "Fusion/routing cannot be ignored globally: V2VNet is fusion-dominant "
            "in the timing sidecar and Where2comm fusion is comparable to dense stages."
        ),
        "next_action": "Keep full-model verdict capped unless fusion/routing is bounded per model.",
    },
    "v2xvit_qgranularity_p_anchor": {
        "scan_status": "BOUND_EXISTING_EVIDENCE",
        "verdict": "PAIR_OR_JOINT_QP_REQUIRED_UNTIL_BOUND",
        "evidence_kind": "existing_quantization_sensitivity_gate",
        "summary": (
            "Known V2X-ViT Q-granularity and P interaction remains a predictor gate; "
            "manifest buildability alone is insufficient."
        ),
        "next_action": "Bind C4/C5 AP and latency evidence before promoting V2X-ViT.",
    },
    "codriving_resnet_completion_anchor": {
        "scan_status": "BOUND_EXISTING_MEASURED",
        "verdict": "SCOPED_NEGATIVE_ANCHOR_ONLY",
        "evidence_kind": "existing_h800_tvm_codriving_measurements",
        "summary": (
            "CoDriving stays a measured negative anchor only inside its ResNet/backbone "
            "envelope; it is not a universal standard-conv rule."
        ),
        "next_action": "Use as calibration baseline, not as evidence for other architectures.",
    },
    "maxfusion_coverage_anchor": {
        "scan_status": "STATIC_CANDIDATE_UNRESOLVED",
        "verdict": "UNKNOWN_UNBOUNDED_FOR_FULL_MODEL",
        "evidence_kind": "static_candidate_without_latency_bound",
        "summary": (
            "MaxFusion has not been measured or statically proven negligible in this "
            "S2 pass, so F-Cooper full-model promotion remains blocked."
        ),
        "next_action": "Add one MaxFusion shape/latency or shape-preservation proof later.",
    },
    "per_stage_q_ap_sensitivity_anchor": {
        "scan_status": "BOUND_EXISTING_OR_BACKLOG",
        "verdict": "AP_SENSITIVE_Q_GATE_REMAINS",
        "evidence_kind": "existing_or_pending_ap_sensitivity_gate",
        "summary": (
            "Latency-only S2 anchors do not settle per-stage Q/AP sensitivity; this "
            "gate remains separate from schedule-anchor completion."
        ),
        "next_action": "Run or bind S3 AP-sensitive Q evidence before AP-level separability claims.",
    },
    "pyramid_mixed_p_hub_context_anchor": {
        "scan_status": "BOUND_EXISTING_EVIDENCE",
        "verdict": "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_PROMOTION",
        "evidence_kind": "existing_pyramid_grouped_conv_context",
        "summary": (
            "Pyramid local standard-conv evidence is dense-subgraph-only because grouped "
            "conv P-hub context dominates the model-level coupling decision."
        ),
        "next_action": "Keep Pyramid under P-hub context rules.",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def measured_summary(scan: dict[str, Any], probe_id: str) -> dict[str, Any]:
    cells = [c for c in scan["cells"] if c["probe_id"] == probe_id]
    analysis = scan["analysis"][probe_id]
    expected_batches = tuple(scan.get("batches") or [1, 2])
    expected_keys = {
        (p_label, precision, batch)
        for p_label in EXPECTED_P_LABELS
        for precision in EXPECTED_PRECISIONS
        for batch in expected_batches
    }
    observed_keys = {
        (cell["p_label"], cell["precision"], cell["batch"])
        for cell in cells
    }
    duplicate_cell_count = len(cells) - len(observed_keys)
    expected_cell_count = len(expected_keys)
    expected_schedule_swap_count = sum(1 for _, precision, _ in expected_keys if precision == "int8")
    default_ok = sum(1 for c in cells if c["default"].get("ok"))
    own_tuned_ok = sum(1 for c in cells if c["own_tuned"].get("ok"))
    schedule_swap_ok = sum(
        1
        for c in cells
        if c["precision"] == "int8" and c["schedule_swap"].get("ok")
    )
    missing_cells = sorted(expected_keys - observed_keys)
    unexpected_cells = sorted(observed_keys - expected_keys)
    matrix_complete = (
        len(cells) == expected_cell_count
        and duplicate_cell_count == 0
        and not missing_cells
        and not unexpected_cells
        and default_ok == expected_cell_count
        and own_tuned_ok == expected_cell_count
        and schedule_swap_ok == expected_schedule_swap_count
    )
    failures: list[dict[str, Any]] = []
    for cell in cells:
        if not cell["default"].get("ok"):
            failures.append({"cell": cell_ref(cell), "arm": "default"})
        if not cell["own_tuned"].get("ok"):
            failures.append({"cell": cell_ref(cell), "arm": "own_tuned"})
        if cell["precision"] == "int8" and not cell["schedule_swap"].get("ok"):
            failures.append({"cell": cell_ref(cell), "arm": "schedule_swap"})

    comparisons = analysis["comparisons"]
    rank_flip_count = sum(1 for c in comparisons if c["default_vs_tuned_rank_flip"])
    gains = [
        gain["gain"]
        for comp in comparisons
        for gain in comp.get("schedule_gains", [])
    ]
    swap_ratios = [
        gap["swap_over_own"]
        for batch_gaps in analysis.get("schedule_swap_gaps", [])
        for gap in batch_gaps.get("gaps", [])
    ]
    batch_best = batch_best_summary(comparisons)
    flags = {
        "default_vs_tuned_rank_flip": rank_flip_count > 0,
        "schedule_gain_drift_gt_1p5x": gain_span(gains) > 1.5,
        "schedule_swap_gap_gt_1p25x_or_lt_0p8x": any(
            ratio > 1.25 or ratio < 0.8 for ratio in swap_ratios
        ),
        "batch_best_width_flip": any(item["best_width_changes"] for item in batch_best),
    }
    if not matrix_complete:
        verdict = "MEASUREMENT_INCOMPLETE_OR_FAILED"
    elif any(flags.values()):
        verdict = "COUPLING_SIGNAL_DETECTED"
    else:
        verdict = "LOW_RISK_ANCHOR"
    if matrix_complete:
        scan_status = "MEASURED_COMPLETE"
    elif failures:
        scan_status = "MEASURED_WITH_FAILURES"
    else:
        scan_status = "MEASURED_INCOMPLETE"
    return {
        "scan_status": scan_status,
        "verdict": verdict,
        "evidence_kind": "h800_tvm_relax_metaschedule_anchor",
        "cell_count": len(cells),
        "expected_cell_count": expected_cell_count,
        "matrix": {
            "p_labels": sorted({c["p_label"] for c in cells}),
            "precision": sorted({c["precision"] for c in cells}),
            "batches": sorted({c["batch"] for c in cells}),
            "default_ok": default_ok,
            "own_tuned_ok": own_tuned_ok,
            "schedule_swap_ok": schedule_swap_ok,
            "expected_schedule_swap_count": expected_schedule_swap_count,
            "matrix_complete": matrix_complete,
            "missing_cells": [
                {"p_label": p_label, "precision": precision, "batch": batch}
                for p_label, precision, batch in missing_cells
            ],
            "unexpected_cells": [
                {"p_label": p_label, "precision": precision, "batch": batch}
                for p_label, precision, batch in unexpected_cells
            ],
            "duplicate_cell_count": duplicate_cell_count,
        },
        "shape_contracts": sorted(
            {
                tuple(
                    [
                        c["op_kind"],
                        c["width"],
                        tuple(c["input_hw"]),
                        c["kernel"],
                        c["stride"],
                        c["padding"],
                    ]
                )
                for c in cells
            },
            key=str,
        ),
        "risk_flags": flags,
        "rank_flip_count": rank_flip_count,
        "gain_min": min(gains) if gains else None,
        "gain_max": max(gains) if gains else None,
        "schedule_swap_min": min(swap_ratios) if swap_ratios else None,
        "schedule_swap_max": max(swap_ratios) if swap_ratios else None,
        "batch_best_width": batch_best,
        "comparisons": comparisons,
        "schedule_swap_gaps": analysis.get("schedule_swap_gaps", []),
        "failures": failures,
        "summary": measured_text_summary(probe_id, flags, matrix_complete),
        "next_action": "Use as S2 risk signal; do not promote static standard-conv separability.",
    }


def cell_ref(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "p_label": cell["p_label"],
        "precision": cell["precision"],
        "batch": cell["batch"],
    }


def gain_span(values: list[float]) -> float:
    positives = [v for v in values if v and v > 0]
    if not positives:
        return 0.0
    return max(positives) / min(positives)


def batch_best_summary(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_precision: dict[str, dict[int, str]] = {}
    for comp in comparisons:
        tuned_rank = comp.get("tuned_rank") or []
        if tuned_rank:
            by_precision.setdefault(comp["precision"], {})[comp["batch"]] = tuned_rank[0]
    out = []
    for precision, values in sorted(by_precision.items()):
        unique = sorted(set(values.values()))
        out.append(
            {
                "precision": precision,
                "best_by_batch": values,
                "best_width_changes": len(unique) > 1,
            }
        )
    return out


def measured_text_summary(
    probe_id: str,
    flags: dict[str, bool],
    matrix_complete: bool,
) -> str:
    if not matrix_complete:
        return f"{probe_id} measurement is incomplete or failed; do not use as a pass."
    active = [name for name, value in flags.items() if value]
    if not active:
        return f"{probe_id} completed without S2 coupling-risk flags."
    return f"{probe_id} completed and raised S2 coupling-risk flags: {', '.join(active)}."


def build_report(queue: dict[str, Any], scan: dict[str, Any]) -> dict[str, Any]:
    results = []
    for probe in queue["probes"]:
        probe_id = probe["probe_id"]
        if probe_id in MEASURED_PROBES:
            result = measured_summary(scan, probe_id)
        else:
            result = dict(NON_MEASURED_STATUS[probe_id])
        results.append(
            {
                "probe_id": probe_id,
                "priority": probe.get("priority"),
                "readiness": probe.get("readiness"),
                "representative_models": probe.get("representative_models", []),
                "question": probe.get("question"),
                "stop_condition": probe.get("stop_condition"),
                "queue_pass_fail_criteria": probe.get("pass_fail_criteria", []),
                **result,
            }
        )

    measured = [r for r in results if r["scan_status"].startswith("MEASURED")]
    coupling = [r for r in results if r.get("verdict") == "COUPLING_SIGNAL_DETECTED"]
    blocked = [
        r
        for r in results
        if any(token in r["scan_status"] for token in ["BLOCKED", "UNRESOLVED", "BACKLOG"])
    ]
    active_gates = [
        r
        for r in results
        if not r["scan_status"].startswith("MEASURED")
        and r["probe_id"] != "codriving_resnet_completion_anchor"
    ]
    return {
        "schema": "stage1_s2_probe_completion_v1",
        "created": "2026-06-23",
        "source_queue": "results/stage1_model_predict/standard_conv_probe_queue_v1.json",
        "source_scan": "results/stage1_model_predict/s2_anchor_scan/stage1_s2_anchor_scan_v1_20260623_2110.json",
        "h800_scan": {
            "status": scan.get("status"),
            "tvm_version": scan.get("tvm_version"),
            "target": scan.get("target"),
            "trials": scan.get("trials"),
            "reps": scan.get("reps"),
            "batches": scan.get("batches"),
            "elapsed_s": scan.get("elapsed_s"),
            "cell_count": len(scan.get("cells", [])),
        },
        "completion_summary": {
            "total_probe_queue_items": len(results),
            "measured_anchor_items": len(measured),
            "measured_anchor_cells": sum(r.get("cell_count", 0) for r in measured),
            "measured_anchor_matrix_complete": len(measured) == len(MEASURED_PROBES)
            and all(r["matrix"].get("matrix_complete") is True for r in measured),
            "coupling_signal_probe_ids": [r["probe_id"] for r in coupling],
            "blocked_or_unresolved_probe_ids": [r["probe_id"] for r in blocked],
            "active_gate_probe_ids": [r["probe_id"] for r in active_gates],
            "s2_stop_goal_status": "QUEUE_SCANNED_ALL_PROBES_HAVE_STATUS",
            "scientific_conclusion": (
                "S2 queue scanning is complete, but the measured standard-conv anchors "
                "raise P x S / Q x S / batch x P risk signals. This supports a more "
                "conservative predictor, not a full-model separability claim."
            ),
        },
        "probe_results": results,
    }


def write_individual_artifacts(report: dict[str, Any], out_dir: Path) -> None:
    for result in report["probe_results"]:
        stem = result["probe_id"]
        (out_dir / f"{stem}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out_dir / f"{stem}.md").write_text(render_probe_md(result), encoding="utf-8")


def render_probe_md(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['probe_id']}",
        "",
        f"- scan_status: `{result['scan_status']}`",
        f"- verdict: `{result['verdict']}`",
        f"- evidence_kind: `{result['evidence_kind']}`",
        f"- readiness: `{result['readiness']}`",
        f"- representative_models: `{', '.join(result.get('representative_models', []))}`",
        "",
        "## Summary",
        "",
        result["summary"],
        "",
        "## Stop Condition",
        "",
        result.get("stop_condition") or "",
        "",
    ]
    if result["scan_status"].startswith("MEASURED"):
        lines.extend(
            [
                "## Measured Matrix",
                "",
                f"- cells: `{result['cell_count']}/{result['expected_cell_count']}`",
                f"- default_ok: `{result['matrix']['default_ok']}`",
                f"- own_tuned_ok: `{result['matrix']['own_tuned_ok']}`",
                f"- schedule_swap_ok: `{result['matrix']['schedule_swap_ok']}/{result['matrix']['expected_schedule_swap_count']}`",
                f"- gain_min/max: `{result['gain_min']:.3f}` / `{result['gain_max']:.3f}`",
                f"- schedule_swap_min/max: `{result['schedule_swap_min']:.3f}` / `{result['schedule_swap_max']:.3f}`",
                "",
                "## Comparisons",
                "",
                "| precision | batch | default rank | tuned rank | rank flip | gain min | gain max |",
                "|---|---:|---|---|---:|---:|---:|",
            ]
        )
        for comp in result["comparisons"]:
            gains = [g["gain"] for g in comp.get("schedule_gains", [])]
            lines.append(
                "| {precision} | {batch} | `{default}` | `{tuned}` | {flip} | {gmin:.3f} | {gmax:.3f} |".format(
                    precision=comp["precision"],
                    batch=comp["batch"],
                    default=" > ".join(comp["default_rank"]),
                    tuned=" > ".join(comp["tuned_rank"]),
                    flip=str(comp["default_vs_tuned_rank_flip"]).lower(),
                    gmin=min(gains),
                    gmax=max(gains),
                )
            )
        lines.extend(["", "## Batch Best Width", "", "| precision | best by batch | changed |", "|---|---|---:|"])
        for item in result["batch_best_width"]:
            best = ", ".join(f"b{k}: {v}" for k, v in sorted(item["best_by_batch"].items()))
            lines.append(f"| {item['precision']} | `{best}` | {str(item['best_width_changes']).lower()} |")
        lines.append("")
    lines.extend(["## Next Action", "", result["next_action"], ""])
    return "\n".join(lines)


def render_report_md(report: dict[str, Any]) -> str:
    summary = report["completion_summary"]
    lines = [
        "# Stage1 S2 Probe Completion v1",
        "",
        "## Status",
        "",
        f"- total_probe_queue_items: `{summary['total_probe_queue_items']}`",
        f"- measured_anchor_items: `{summary['measured_anchor_items']}`",
        f"- measured_anchor_cells: `{summary['measured_anchor_cells']}`",
        f"- measured_anchor_matrix_complete: `{str(summary['measured_anchor_matrix_complete']).lower()}`",
        f"- s2_stop_goal_status: `{summary['s2_stop_goal_status']}`",
        f"- active_gate_probe_ids: `{', '.join(summary['active_gate_probe_ids'])}`",
        "",
        summary["scientific_conclusion"],
        "",
        "## Probe Table",
        "",
        "| probe | scan status | verdict | evidence |",
        "|---|---|---|---|",
    ]
    for result in report["probe_results"]:
        lines.append(
            f"| `{result['probe_id']}` | `{result['scan_status']}` | `{result['verdict']}` | `{result['evidence_kind']}` |"
        )
    lines.extend(
        [
            "",
            "## Measured Anchor Signals",
            "",
            "| probe | cells | rank flips | gain min/max | swap min/max | batch flip |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in report["probe_results"]:
        if not result["scan_status"].startswith("MEASURED"):
            continue
        batch_flip = any(item["best_width_changes"] for item in result["batch_best_width"])
        lines.append(
            "| `{probe}` | {cells} | {flips} | {gmin:.3f}/{gmax:.3f} | {smin:.3f}/{smax:.3f} | {batch_flip} |".format(
                probe=result["probe_id"],
                cells=result["cell_count"],
                flips=result["rank_flip_count"],
                gmin=result["gain_min"],
                gmax=result["gain_max"],
                smin=result["schedule_swap_min"],
                smax=result["schedule_swap_max"],
                batch_flip=str(batch_flip).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- The two schedule anchors are measured H800 TVM/Relax MetaSchedule probes.",
            "- The other seven queue items are scanned as guardrails, existing-evidence bindings, or unresolved blockers.",
            "- A complete S2 queue scan is not a claim that every full model is separable.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue-json",
        type=Path,
        default=Path("results/stage1_model_predict/standard_conv_probe_queue_v1.json"),
    )
    parser.add_argument(
        "--scan-json",
        type=Path,
        default=Path(
            "results/stage1_model_predict/s2_anchor_scan/"
            "stage1_s2_anchor_scan_v1_20260623_2110.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/stage1_model_predict/s2_probe_results"),
    )
    args = parser.parse_args()

    queue = load_json(args.queue_json)
    scan = load_json(args.scan_json)
    report = build_report(queue, scan)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.out_dir / "stage1_s2_probe_completion_v1.json"
    report_md = args.out_dir / "stage1_s2_probe_completion_v1.md"
    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_md.write_text(render_report_md(report), encoding="utf-8")
    write_individual_artifacts(report, args.out_dir)
    print(report_json)
    print(report_md)


if __name__ == "__main__":
    main()
