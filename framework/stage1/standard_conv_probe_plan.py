#!/usr/bin/env python3
"""Materialize the S2 standard-conv anchor probe queue.

This script is intentionally local and static-only. It reads
standard_conv_census_v1.json and writes an executable experiment specification:
what each low-cost probe should test, which inputs are minimally required, what
artifacts should be produced, and which command skeletons a runner can fill in.

It does not import CUDA, TVM, OpenCOOD, or model code, and it explicitly forbids
full P x Q x S enumeration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DESIGN_DIR = ROOT / "multi_agent/methods/design/stage1-model-predict"
RESULTS_DIR = ROOT / "results/stage1_model_predict"
DEFAULT_CENSUS = RESULTS_DIR / "standard_conv_census_v1.json"
DEFAULT_OUT_JSON = RESULTS_DIR / "standard_conv_probe_queue_v1.json"
DEFAULT_OUT_MD = DESIGN_DIR / "standard_conv_probe_queue_v1.md"

NO_FULL_ENUMERATION = (
    "Full P x Q x S enumeration is prohibited for S2 anchor probes. "
    "Each probe may use only one representative regime, base plus one "
    "pruned/boundary P point, fp16/int8 Q points, and default/tuned or "
    "schedule-swap S checks."
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a JSON object")
    return data


def _priority_rank(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)


def _probe_templates() -> dict[str, dict[str, Any]]:
    return {
        "std_basebev_backbone_schedule_anchor": {
            "readiness": "MEASUREMENT_BACKLOG_RUNNER_REQUIRED",
            "next_stage_role": (
                "Not required for safe predictor v0. Keep full-model verdict low-confidence "
                "until a real dense-backbone anchor runner exists."
            ),
            "evidence_level": "static_only_pending_anchor_probe",
            "blocking_condition": (
                "No low-risk separability verdict for F-Cooper, AttFuse, "
                "V2X-ViT, or newly added HeterBaseline fusion variants until at least one representative "
                "base/pruned width pair has schedule-drift evidence."
            ),
            "minimal_inputs": [
                "standard_conv_census_v1.json recommended_probe_queue entry",
                "one BaseBEVBackbone candidate from F-Cooper, AttFuse, V2X-ViT, Where2comm, V2VNet, or DiscoNet",
                "base width and one pruned or boundary width from the census",
                "fp16 and int8 build labels",
                "default schedule label and tuned MetaSchedule label",
            ],
            "expected_artifacts": [
                "probe_specs/std_basebev_backbone_schedule_anchor.yaml",
                "probe_results/std_basebev_backbone_schedule_anchor.json",
                "probe_results/std_basebev_backbone_schedule_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_LOW_RISK_ANCHOR if tuned/default ordering is stable across P and Q",
                "PASS_LOW_RISK_ANCHOR if schedule gain variance stays within the declared noise band",
                "FAIL_COUPLED if P changes the best S choice or Q changes the best S choice",
                "UNKNOWN if shape fields are missing or latency is estimated rather than measured",
            ],
            "estimated_cost_class": "low_anchor_2P_x_2Q_x_2S_max_no_grid",
        },
        "attention_fusion_coverage_anchor": {
            "readiness": "BLOCKER_NOT_RUNNABLE_UNTIL_ATTENTION_TRACE_OR_EXPORT_EXISTS",
            "next_stage_role": (
                "Guardrail only for safe predictor v0: typed skipped-subgraph metadata must "
                "block full-model separability. Do not schedule an attention latency probe yet."
            ),
            "evidence_level": "uncovered_unknown_until_attention_integration",
            "blocking_condition": (
                "AttFuse, Where2comm, and V2X-ViT full-model separability remains blocked "
                "while attention/transformer fusion is skipped or unprofiled."
            ),
            "minimal_inputs": [
                "census skipped_subgraph_types for AttFuse, Where2comm, and V2X-ViT",
                "one skipped attention/fusion module identity",
                "typed skip reason showing attention/fusion is not integrated into trace/export",
                "explicit full-model verdict blocker flag",
            ],
            "expected_artifacts": [
                "probe_specs/attention_fusion_coverage_anchor.yaml",
                "probe_results/attention_fusion_coverage_anchor.json",
                "probe_results/attention_fusion_coverage_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_DENSE_ONLY if dense backbone is stable but skipped fusion remains unmeasured",
                "PASS_BLOCKER if attention/fusion is explicitly marked not runnable and blocks full-model verdict",
                "FAIL_OVERPROMOTION if full-model low-risk is emitted while attention/fusion is skipped",
                "UNKNOWN if skipped subgraph identity or blocker flag is missing",
            ],
            "estimated_cost_class": "no_measurement_in_v0_metadata_guardrail_only",
            "command_skeleton": [
                "# No attention latency probe exists in safe predictor v0.",
                "# Implement typed skip metadata tests instead:",
                "PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage1_manifest_predictor_fields.py",
            ],
        },
        "v2xvit_qgranularity_p_anchor": {
            "readiness": "EXISTING_EVIDENCE_BINDING",
            "next_stage_role": (
                "Bind existing C4 evidence into predictor gates; do not run a new AP/latency "
                "probe in the safe predictor v0 stop target."
            ),
            "evidence_level": "existing_measured_cell_needs_predictor_binding",
            "blocking_condition": (
                "V2X-ViT cannot receive a cheap full-model separability verdict "
                "until known Q-granularity x P evidence is attached to the predictor."
            ),
            "minimal_inputs": [
                "C4 Q-granularity x P result file or extracted AP/latency table",
                "one base P point and one pruned/boundary P point",
                "per-tensor and per-channel quantization labels",
                "AP noise floor and latency noise floor",
            ],
            "expected_artifacts": [
                "probe_specs/v2xvit_qgranularity_p_anchor.yaml",
                "probe_results/v2xvit_qgranularity_p_anchor.json",
                "probe_results/v2xvit_qgranularity_p_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_PAIR_REQUIRED if Q-granularity AP or latency rank changes across P",
                "PASS_LOW_RISK_ANCHOR only if both AP and latency ranks are stable within noise",
                "FAIL_STATIC_ONLY if only buildability or manifest metadata is available",
                "UNKNOWN if AP noise floor or quantization granularity metadata is missing",
            ],
            "estimated_cost_class": "reuse_existing_cell_or_low_2P_x_2Q_anchor",
            "command_skeleton": [
                "# Bind existing C4 evidence; do not launch a new AP probe in safe predictor v0.",
                "python scripts/stage1_predict_coupling.py --help",
            ],
        },
        "routing_fusion_coverage_anchor": {
            "readiness": "PARTIAL_EXISTING_EVIDENCE_BINDING",
            "next_stage_role": (
                "Bind existing C5 routing evidence for V2X-ViT and typed skip metadata for "
                "F-Cooper, AttFuse, Where2comm, V2VNet, and DiscoNet. Do not require a new fusion runner in v0."
            ),
            "evidence_level": "coverage_gate_pending_or_existing_c5_binding",
            "blocking_condition": (
                "Full-model prediction is blocked if routing/fusion operators alter "
                "the feasible Q/S action set or remain unbounded by profiling."
            ),
            "minimal_inputs": [
                "view_d_routing entries from the manifest",
                "operator blacklist or routing feasibility evidence",
                "one fusion/routing shape contract and latency bound",
                "dense-backbone anchor result, if available",
            ],
            "expected_artifacts": [
                "probe_specs/routing_fusion_coverage_anchor.yaml",
                "probe_results/routing_fusion_coverage_anchor.json",
                "probe_results/routing_fusion_coverage_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_COVERAGE_ONLY if routing feasibility is independent of P/Q/S and latency is bounded",
                "PASS_PAIR_REQUIRED if routing feasibility changes by Q or S",
                "FAIL_UNCOVERED if fusion/routing remains unprofiled or unbounded",
                "UNKNOWN if skipped subgraph identity or shape contract is missing",
            ],
            "estimated_cost_class": "reuse_existing_c5_or_single_fusion_routing_microprobe",
            "command_skeleton": [
                "# Bind existing C5 routing evidence and typed skip metadata where available.",
                "PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py",
            ],
        },
        "codriving_resnet_completion_anchor": {
            "readiness": "COMPLETED_EXISTING_MEASURED_BINDING",
            "next_stage_role": (
                "Already satisfied by measured-results binding; keep it as a regression/fixture, "
                "not as a next-stage experiment."
            ),
            "evidence_level": "measured_negative_anchor_with_estimated_latency_gap",
            "blocking_condition": (
                "CoDriving can keep measured-anchor wording only for the measured "
                "ResNet envelope; missing p25/p75 latency points block broader claims."
            ),
            "minimal_inputs": [
                "CoDriving backbone.s0/s1/s2 census candidates",
                "missing or estimated p25/p75 latency points from prior reports",
                "same H800 TVM protocol metadata used by existing coupling-map evidence",
            ],
            "expected_artifacts": [
                "probe_specs/codriving_resnet_completion_anchor.yaml",
                "probe_results/codriving_resnet_completion_anchor.json",
                "probe_results/codriving_resnet_completion_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_MEASURED_NEGATIVE if real anchor measurements preserve no argmin drift",
                "PASS_MEASURED_NEGATIVE if serial HV remains equivalent to joint HV",
                "FAIL_COUPLED if replacing estimates creates rank flips or schedule/Q drift",
                "UNKNOWN if new measurements use a different protocol or device envelope",
            ],
            "estimated_cost_class": "low_completion_remeasure_missing_anchor_points",
            "command_skeleton": [
                "# Existing measured-results binding is already present:",
                "python -m json.tool results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json >/tmp/standard_conv_anchor_measured_results_v1.check.json",
            ],
        },
        "standard_neck_deconv_anchor": {
            "readiness": "MEASUREMENT_BACKLOG_RUNNER_REQUIRED",
            "next_stage_role": (
                "Backlog measurement. Safe predictor v0 must keep neck/deconv low-confidence "
                "or dense-subgraph-only until a real runner exists."
            ),
            "evidence_level": "static_only_pending_anchor_probe",
            "blocking_condition": (
                "Neck/deconv standard-conv regimes cannot inherit CoDriving "
                "3x3 ResNet separability without their own S-axis anchor check."
            ),
            "minimal_inputs": [
                "one neck or deconv candidate from the census",
                "base width and one pruned or boundary width",
                "fp16/int8 labels",
                "default and tuned schedule labels",
                "schedule-swap pairing for Q/S checks",
            ],
            "expected_artifacts": [
                "probe_specs/standard_neck_deconv_anchor.yaml",
                "probe_results/standard_neck_deconv_anchor.json",
                "probe_results/standard_neck_deconv_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_LOW_RISK_ANCHOR if deconv/neck ordering is stable across P and Q",
                "FAIL_COUPLED if default-vs-tuned rank flips across P",
                "FAIL_COUPLED if fp16/int8 schedule-swap changes the best S choice",
                "UNKNOWN if ConvTranspose2d shape fields are missing",
            ],
            "estimated_cost_class": "low_anchor_2P_x_2Q_x_2S_max_no_grid",
        },
        "pyramid_mixed_p_hub_context_anchor": {
            "readiness": "EXISTING_EVIDENCE_BINDING",
            "next_stage_role": (
                "Bind existing Pyramid P-hub evidence and manifest adjacency as a model-level "
                "overpromotion guardrail."
            ),
            "evidence_level": "existing_measured_positive_p_hub_context",
            "blocking_condition": (
                "Local standard-conv groups in Pyramid cannot be promoted to "
                "model-level low-risk while grouped-conv IC_BN P-hub neighbors dominate."
            ),
            "minimal_inputs": [
                "C2/C3/C7 Pyramid coupling-map evidence",
                "manifest adjacency between grouped BEV encoder and local standard Conv2d groups",
                "fanout/head coupling metadata",
                "scope label distinguishing local dense subgraph from full model",
            ],
            "expected_artifacts": [
                "probe_specs/pyramid_mixed_p_hub_context_anchor.yaml",
                "probe_results/pyramid_mixed_p_hub_context_anchor.json",
                "probe_results/pyramid_mixed_p_hub_context_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_SCOPE_BLOCK if local standard-conv evidence is dense-subgraph-only",
                "PASS_P_HUB_CONTEXT if grouped-conv neighbor controls Q/S schedule choice",
                "FAIL_OVERPROMOTION if model-level low-risk is emitted from local standard Conv2d only",
                "UNKNOWN if adjacency or fanout metadata is missing",
            ],
            "estimated_cost_class": "reuse_existing_pyramid_cells_plus_static_adjacency",
            "command_skeleton": [
                "# Bind existing Pyramid P-hub evidence; no new latency probe in safe predictor v0.",
                "python -m json.tool results/coupling_map_matrix.json >/tmp/coupling_map_matrix.check.json",
            ],
        },
        "per_stage_q_ap_sensitivity_anchor": {
            "readiness": "BACKLOG_OR_EXISTING_EVIDENCE_BINDING",
            "next_stage_role": (
                "Safe predictor v0 should expose this as an AP-sensitive gate, but new AP "
                "experiments are outside the v0 stop target."
            ),
            "evidence_level": "ap_sensitive_q_gate_pending_or_existing_binding",
            "blocking_condition": (
                "Latency-only separability prediction is insufficient when pruning changes "
                "which stage or quantization granularity preserves AP."
            ),
            "minimal_inputs": [
                "one pruned P point and base P point",
                "per-stage quantization assignment or granularity labels",
                "AP metric with declared noise floor",
                "latency metric with declared noise floor",
            ],
            "expected_artifacts": [
                "probe_specs/per_stage_q_ap_sensitivity_anchor.yaml",
                "probe_results/per_stage_q_ap_sensitivity_anchor.json",
                "probe_results/per_stage_q_ap_sensitivity_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_PAIR_REQUIRED if the best per-stage Q choice changes after pruning",
                "PASS_LOW_RISK_ANCHOR only if AP and latency choices are stable within noise",
                "FAIL_LATENCY_ONLY if AP evidence is absent for an AP-sensitive model",
                "UNKNOWN if per-stage quantization metadata is missing",
            ],
            "estimated_cost_class": "low_2P_selected_stage_Q_AP_anchor_no_grid",
        },
        "maxfusion_coverage_anchor": {
            "readiness": "STATIC_PROOF_CANDIDATE",
            "next_stage_role": (
                "F-Cooper-only static proof candidate. Can mark MaxFusion as channel-preserving "
                "only if shape contract is present; otherwise keep full-model low-confidence."
            ),
            "evidence_level": "static_only_pending_anchor_probe",
            "blocking_condition": (
                "F-Cooper full-model separability remains blocked until MaxFusion "
                "is measured or statically proven channel-preserving and negligible."
            ),
            "minimal_inputs": [
                "F-Cooper skipped_subgraph_types from the census",
                "MaxFusion module identity and input/output shape contract",
                "one dense-backbone anchor result, if available",
            ],
            "expected_artifacts": [
                "probe_specs/maxfusion_coverage_anchor.yaml",
                "probe_results/maxfusion_coverage_anchor.json",
                "probe_results/maxfusion_coverage_anchor.md",
            ],
            "pass_fail_criteria": [
                "PASS_DENSE_ONLY if MaxFusion is unprofiled but dense backbone is stable",
                "PASS_FULL_SCOPE only if MaxFusion is negligible and shape-preserving",
                "FAIL_UNCOVERED if MaxFusion has measurable latency or rank impact",
                "UNKNOWN if MaxFusion cannot be bounded from available metadata",
            ],
            "estimated_cost_class": "lowest_static_proof_or_single_operator_check",
        },
    }


def _command_skeleton(probe_id: str, models: list[str]) -> list[str]:
    model_arg = ",".join(models) if models else "MODEL"
    return [
        (
            "python tools/run_s2_anchor_probe.py "
            f"--probe-id {probe_id} --models {model_arg} "
            "--p-points base,boundary --q-points fp16,int8 "
            "--s-points default,tuned --no-full-enumeration --dry-run"
        ),
        (
            "python tools/summarize_s2_anchor_probe.py "
            f"--probe-results probe_results/{probe_id}.json "
            f"--out-md probe_results/{probe_id}.md"
        ),
    ]


def _verdict_gates() -> dict[str, dict[str, Any]]:
    common_gate = (
        "groups=1 only removes the known grouped-conv IC_BN hard-format cliff; "
        "it is not direct evidence of P/Q/S separability."
    )
    return {
        "fcooper": {
            "gate": "LOW_CONFIDENCE_NEEDS_TARGETED_PROBE",
            "reason": common_gate,
            "required_before_full_model_separable": [
                "std_basebev_backbone_schedule_anchor",
                "maxfusion_coverage_anchor",
                "routing_fusion_coverage_anchor",
            ],
        },
        "attfuse": {
            "gate": "FUSION_UNCOVERED_UNKNOWN",
            "reason": common_gate,
            "required_before_full_model_separable": [
                "std_basebev_backbone_schedule_anchor",
                "attention_fusion_coverage_anchor",
                "routing_fusion_coverage_anchor",
            ],
        },
        "where2comm": {
            "gate": "FUSION_UNCOVERED_UNKNOWN",
            "reason": common_gate,
            "required_before_full_model_separable": [
                "std_basebev_backbone_schedule_anchor",
                "attention_fusion_coverage_anchor",
                "routing_fusion_coverage_anchor",
                "standard_neck_deconv_anchor",
            ],
        },
        "v2vnet": {
            "gate": "FUSION_UNCOVERED_UNKNOWN",
            "reason": common_gate,
            "required_before_full_model_separable": [
                "std_basebev_backbone_schedule_anchor",
                "routing_fusion_coverage_anchor",
                "standard_neck_deconv_anchor",
            ],
        },
        "disconet": {
            "gate": "FUSION_UNCOVERED_UNKNOWN",
            "reason": common_gate,
            "required_before_full_model_separable": [
                "std_basebev_backbone_schedule_anchor",
                "routing_fusion_coverage_anchor",
                "standard_neck_deconv_anchor",
            ],
        },
        "v2xvit": {
            "gate": "JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND",
            "reason": common_gate,
            "required_before_full_model_separable": [
                "std_basebev_backbone_schedule_anchor",
                "attention_fusion_coverage_anchor",
                "v2xvit_qgranularity_p_anchor",
                "routing_fusion_coverage_anchor",
                "per_stage_q_ap_sensitivity_anchor",
            ],
        },
        "codriving": {
            "gate": "MEASURED_ENVELOPE_ONLY",
            "reason": common_gate,
            "required_before_broader_standard_conv_claim": [
                "codriving_resnet_completion_anchor",
                "standard_neck_deconv_anchor",
            ],
        },
        "pyramid_camera": {
            "gate": "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION",
            "reason": common_gate,
            "required_before_broader_standard_conv_claim": [
                "pyramid_mixed_p_hub_context_anchor",
                "per_stage_q_ap_sensitivity_anchor",
            ],
        },
        "pyramid_lidar": {
            "gate": "P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION",
            "reason": common_gate,
            "required_before_broader_standard_conv_claim": [
                "pyramid_mixed_p_hub_context_anchor",
                "per_stage_q_ap_sensitivity_anchor",
            ],
        },
    }


def build_probe_plan(census: dict[str, Any]) -> dict[str, Any]:
    queue = census.get("recommended_probe_queue")
    if not isinstance(queue, list):
        raise ValueError("census JSON does not contain recommended_probe_queue")

    templates = _probe_templates()
    probes: list[dict[str, Any]] = []
    for item in sorted(queue, key=lambda value: (_priority_rank(str(value.get("priority"))), str(value.get("probe_id")))):
        probe_id = str(item.get("probe_id"))
        if probe_id not in templates:
            raise ValueError(f"no S2 template for probe_id={probe_id}")
        models = [str(model) for model in item.get("representative_models", [])]
        template = templates[probe_id]
        probes.append(
            {
                **item,
                **template,
                "enumeration_policy": NO_FULL_ENUMERATION,
                "command_skeleton": template.get("command_skeleton", _command_skeleton(probe_id, models)),
            }
        )

    return {
        "schema": "standard_conv_probe_queue_v1",
        "source_census_schema": census.get("schema"),
        "source_census_summary": census.get("summary", {}),
        "policy": {
            "runner_scope": "local_static_spec_generation_only",
            "forbidden": [
                "full_P_x_Q_x_S_enumeration",
                "CUDA_import_or_execution",
                "TVM_import_or_execution",
                "model_framework_import_or_execution",
            ],
            "no_full_enumeration": NO_FULL_ENUMERATION,
            "groups_1_verdict_boundary": (
                "groups=1 cannot directly classify any model as separable; "
                "it only lowers risk for the grouped-conv IC_BN hard cliff."
            ),
        },
        "verdict_gates": _verdict_gates(),
        "probes": probes,
    }


def _md_list(lines: list[str], values: list[str]) -> None:
    for value in values:
        lines.append(f"- {value}")


def write_markdown(plan: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Standard Conv Probe Queue v1")
    lines.append("")
    lines.append("This is a local S2 anchor-probe runner specification. It does not run CUDA, TVM, OpenCOOD, or full model construction.")
    lines.append("")
    lines.append("## Policy")
    lines.append("")
    lines.append(f"- {plan['policy']['no_full_enumeration']}")
    lines.append(f"- {plan['policy']['groups_1_verdict_boundary']}")
    lines.append("- The command skeletons are dry-run placeholders for future measurement runners.")
    lines.append("")
    lines.append("## Verdict Gates")
    lines.append("")
    lines.append("| model | gate | reason | required probes |")
    lines.append("|---|---|---|---|")
    for model, gate in plan["verdict_gates"].items():
        required = gate.get("required_before_full_model_separable") or gate.get("required_before_broader_standard_conv_claim") or []
        lines.append(
            "| {model} | {verdict} | {reason} | {required} |".format(
                model=model,
                verdict=gate["gate"],
                reason=gate["reason"],
                required="<br>".join(required),
            )
        )
    lines.append("")
    lines.append("## Probe Queue")
    lines.append("")
    lines.append("| priority | probe_id | readiness | evidence_level | representative_models | blocking_condition |")
    lines.append("|---|---|---|---|---|---|")
    for probe in plan["probes"]:
        lines.append(
            "| {priority} | {probe_id} | {readiness} | {evidence} | {models} | {blocking} |".format(
                priority=probe["priority"],
                probe_id=probe["probe_id"],
                readiness=probe["readiness"],
                evidence=probe["evidence_level"],
                models=", ".join(probe.get("representative_models", [])) or "-",
                blocking=probe["blocking_condition"],
            )
        )
    lines.append("")
    for probe in plan["probes"]:
        lines.append(f"## {probe['probe_id']}")
        lines.append("")
        lines.append(f"- priority: {probe['priority']}")
        lines.append(f"- readiness: {probe['readiness']}")
        lines.append(f"- next_stage_role: {probe['next_stage_role']}")
        lines.append(f"- evidence_level: {probe['evidence_level']}")
        lines.append(f"- estimated_cost_class: {probe['estimated_cost_class']}")
        lines.append(f"- blocking_condition: {probe['blocking_condition']}")
        lines.append(f"- question: {probe['question']}")
        lines.append(f"- enumeration_policy: {probe['enumeration_policy']}")
        lines.append("")
        lines.append("### Minimal Inputs")
        lines.append("")
        _md_list(lines, probe["minimal_inputs"])
        lines.append("")
        lines.append("### Expected Artifacts")
        lines.append("")
        _md_list(lines, probe["expected_artifacts"])
        lines.append("")
        lines.append("### Pass/Fail Criteria")
        lines.append("")
        _md_list(lines, probe["pass_fail_criteria"])
        lines.append("")
        lines.append("### Command Skeleton")
        lines.append("")
        lines.append("```bash")
        lines.extend(probe["command_skeleton"])
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the S2 standard-conv anchor probe queue.")
    parser.add_argument("--census-json", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args()

    census = _load_json(args.census_json)
    plan = build_probe_plan(census)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(plan, args.out_md)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
