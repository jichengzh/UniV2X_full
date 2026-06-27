# Standard Conv Probe Queue v1

This is a local S2 anchor-probe runner specification. It does not run CUDA, TVM, OpenCOOD, or full model construction.

## Policy

- Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.
- groups=1 cannot directly classify any model as separable; it only lowers risk for the grouped-conv IC_BN hard cliff.
- The command skeletons are dry-run placeholders for future measurement runners.

## Verdict Gates

| model | gate | reason | required probes |
|---|---|---|---|
| fcooper | LOW_CONFIDENCE_NEEDS_TARGETED_PROBE | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | std_basebev_backbone_schedule_anchor<br>maxfusion_coverage_anchor<br>routing_fusion_coverage_anchor |
| attfuse | FUSION_UNCOVERED_UNKNOWN | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | std_basebev_backbone_schedule_anchor<br>attention_fusion_coverage_anchor<br>routing_fusion_coverage_anchor |
| where2comm | FUSION_UNCOVERED_UNKNOWN | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | std_basebev_backbone_schedule_anchor<br>attention_fusion_coverage_anchor<br>routing_fusion_coverage_anchor<br>standard_neck_deconv_anchor |
| v2vnet | FUSION_UNCOVERED_UNKNOWN | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | std_basebev_backbone_schedule_anchor<br>routing_fusion_coverage_anchor<br>standard_neck_deconv_anchor |
| disconet | FUSION_UNCOVERED_UNKNOWN | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | std_basebev_backbone_schedule_anchor<br>routing_fusion_coverage_anchor<br>standard_neck_deconv_anchor |
| v2xvit | JOINT_OR_PAIR_SEARCH_REQUIRED_UNTIL_C4_C5_BOUND | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | std_basebev_backbone_schedule_anchor<br>attention_fusion_coverage_anchor<br>v2xvit_qgranularity_p_anchor<br>routing_fusion_coverage_anchor<br>per_stage_q_ap_sensitivity_anchor |
| codriving | MEASURED_ENVELOPE_ONLY | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | codriving_resnet_completion_anchor<br>standard_neck_deconv_anchor |
| pyramid_camera | P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | pyramid_mixed_p_hub_context_anchor<br>per_stage_q_ap_sensitivity_anchor |
| pyramid_lidar | P_HUB_CONTEXT_BLOCKS_MODEL_LEVEL_STANDARD_CONV_PROMOTION | groups=1 only removes the known grouped-conv IC_BN hard-format cliff; it is not direct evidence of P/Q/S separability. | pyramid_mixed_p_hub_context_anchor<br>per_stage_q_ap_sensitivity_anchor |

## Probe Queue

| priority | probe_id | readiness | evidence_level | representative_models | blocking_condition |
|---|---|---|---|---|---|
| high | attention_fusion_coverage_anchor | BLOCKER_NOT_RUNNABLE_UNTIL_ATTENTION_TRACE_OR_EXPORT_EXISTS | uncovered_unknown_until_attention_integration | attfuse, v2xvit, where2comm | AttFuse, Where2comm, and V2X-ViT full-model separability remains blocked while attention/transformer fusion is skipped or unprofiled. |
| high | routing_fusion_coverage_anchor | PARTIAL_EXISTING_EVIDENCE_BINDING | coverage_gate_pending_or_existing_c5_binding | attfuse, disconet, fcooper, v2vnet, v2xvit, where2comm | Full-model prediction is blocked if routing/fusion operators alter the feasible Q/S action set or remain unbounded by profiling. |
| high | std_basebev_backbone_schedule_anchor | MEASUREMENT_BACKLOG_RUNNER_REQUIRED | static_only_pending_anchor_probe | attfuse, disconet, fcooper, v2vnet, v2xvit, where2comm | No low-risk separability verdict for F-Cooper, AttFuse, V2X-ViT, or newly added HeterBaseline fusion variants until at least one representative base/pruned width pair has schedule-drift evidence. |
| high | v2xvit_qgranularity_p_anchor | EXISTING_EVIDENCE_BINDING | existing_measured_cell_needs_predictor_binding | v2xvit | V2X-ViT cannot receive a cheap full-model separability verdict until known Q-granularity x P evidence is attached to the predictor. |
| medium | codriving_resnet_completion_anchor | COMPLETED_EXISTING_MEASURED_BINDING | measured_negative_anchor_with_estimated_latency_gap | codriving | CoDriving can keep measured-anchor wording only for the measured ResNet envelope; missing p25/p75 latency points block broader claims. |
| medium | maxfusion_coverage_anchor | STATIC_PROOF_CANDIDATE | static_only_pending_anchor_probe | fcooper | F-Cooper full-model separability remains blocked until MaxFusion is measured or statically proven channel-preserving and negligible. |
| medium | per_stage_q_ap_sensitivity_anchor | BACKLOG_OR_EXISTING_EVIDENCE_BINDING | ap_sensitive_q_gate_pending_or_existing_binding | v2xvit, pyramid_camera, pyramid_lidar | Latency-only separability prediction is insufficient when pruning changes which stage or quantization granularity preserves AP. |
| medium | pyramid_mixed_p_hub_context_anchor | EXISTING_EVIDENCE_BINDING | existing_measured_positive_p_hub_context | pyramid_camera, pyramid_lidar | Local standard-conv groups in Pyramid cannot be promoted to model-level low-risk while grouped-conv IC_BN P-hub neighbors dominate. |
| medium | standard_neck_deconv_anchor | MEASUREMENT_BACKLOG_RUNNER_REQUIRED | static_only_pending_anchor_probe | attfuse, codriving, disconet, fcooper, pyramid_camera, pyramid_lidar, v2vnet, v2xvit, where2comm | Neck/deconv standard-conv regimes cannot inherit CoDriving 3x3 ResNet separability without their own S-axis anchor check. |

## attention_fusion_coverage_anchor

- priority: high
- readiness: BLOCKER_NOT_RUNNABLE_UNTIL_ATTENTION_TRACE_OR_EXPORT_EXISTS
- next_stage_role: Guardrail only for safe predictor v0: typed skipped-subgraph metadata must block full-model separability. Do not schedule an attention latency probe yet.
- evidence_level: uncovered_unknown_until_attention_integration
- estimated_cost_class: no_measurement_in_v0_metadata_guardrail_only
- blocking_condition: AttFuse, Where2comm, and V2X-ViT full-model separability remains blocked while attention/transformer fusion is skipped or unprofiled.
- question: Can dense-backbone separability be promoted to full-model separability when attention/transformer fusion is skipped?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- census skipped_subgraph_types for AttFuse, Where2comm, and V2X-ViT
- one skipped attention/fusion module identity
- typed skip reason showing attention/fusion is not integrated into trace/export
- explicit full-model verdict blocker flag

### Expected Artifacts

- probe_specs/attention_fusion_coverage_anchor.yaml
- probe_results/attention_fusion_coverage_anchor.json
- probe_results/attention_fusion_coverage_anchor.md

### Pass/Fail Criteria

- PASS_DENSE_ONLY if dense backbone is stable but skipped fusion remains unmeasured
- PASS_BLOCKER if attention/fusion is explicitly marked not runnable and blocks full-model verdict
- FAIL_OVERPROMOTION if full-model low-risk is emitted while attention/fusion is skipped
- UNKNOWN if skipped subgraph identity or blocker flag is missing

### Command Skeleton

```bash
# No attention latency probe exists in safe predictor v0.
# Implement typed skip metadata tests instead:
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage1_manifest_predictor_fields.py
```

## routing_fusion_coverage_anchor

- priority: high
- readiness: PARTIAL_EXISTING_EVIDENCE_BINDING
- next_stage_role: Bind existing C5 routing evidence for V2X-ViT and typed skip metadata for F-Cooper, AttFuse, Where2comm, V2VNet, and DiscoNet. Do not require a new fusion runner in v0.
- evidence_level: coverage_gate_pending_or_existing_c5_binding
- estimated_cost_class: reuse_existing_c5_or_single_fusion_routing_microprobe
- blocking_condition: Full-model prediction is blocked if routing/fusion operators alter the feasible Q/S action set or remain unbounded by profiling.
- question: Do skipped fusion/routing operators change the feasible Q/S action set or only add a constant uncovered cost?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- view_d_routing entries from the manifest
- operator blacklist or routing feasibility evidence
- one fusion/routing shape contract and latency bound
- dense-backbone anchor result, if available

### Expected Artifacts

- probe_specs/routing_fusion_coverage_anchor.yaml
- probe_results/routing_fusion_coverage_anchor.json
- probe_results/routing_fusion_coverage_anchor.md

### Pass/Fail Criteria

- PASS_COVERAGE_ONLY if routing feasibility is independent of P/Q/S and latency is bounded
- PASS_PAIR_REQUIRED if routing feasibility changes by Q or S
- FAIL_UNCOVERED if fusion/routing remains unprofiled or unbounded
- UNKNOWN if skipped subgraph identity or shape contract is missing

### Command Skeleton

```bash
# Bind existing C5 routing evidence and typed skip metadata where available.
PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_coupling_predictor_static.py
```

## std_basebev_backbone_schedule_anchor

- priority: high
- readiness: MEASUREMENT_BACKLOG_RUNNER_REQUIRED
- next_stage_role: Not required for safe predictor v0. Keep full-model verdict low-confidence until a real dense-backbone anchor runner exists.
- evidence_level: static_only_pending_anchor_probe
- estimated_cost_class: low_anchor_2P_x_2Q_x_2S_max_no_grid
- blocking_condition: No low-risk separability verdict for F-Cooper, AttFuse, V2X-ViT, or newly added HeterBaseline fusion variants until at least one representative base/pruned width pair has schedule-drift evidence.
- question: Does OpenCOOD BaseBEVBackbone standard Conv2d have schedule-gain drift across widths, or does it behave like CoDriving ResNet?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- standard_conv_census_v1.json recommended_probe_queue entry
- one BaseBEVBackbone candidate from F-Cooper, AttFuse, V2X-ViT, Where2comm, V2VNet, or DiscoNet
- base width and one pruned or boundary width from the census
- fp16 and int8 build labels
- default schedule label and tuned MetaSchedule label

### Expected Artifacts

- probe_specs/std_basebev_backbone_schedule_anchor.yaml
- probe_results/std_basebev_backbone_schedule_anchor.json
- probe_results/std_basebev_backbone_schedule_anchor.md

### Pass/Fail Criteria

- PASS_LOW_RISK_ANCHOR if tuned/default ordering is stable across P and Q
- PASS_LOW_RISK_ANCHOR if schedule gain variance stays within the declared noise band
- FAIL_COUPLED if P changes the best S choice or Q changes the best S choice
- UNKNOWN if shape fields are missing or latency is estimated rather than measured

### Command Skeleton

```bash
python tools/run_s2_anchor_probe.py --probe-id std_basebev_backbone_schedule_anchor --models attfuse,disconet,fcooper,v2vnet,v2xvit,where2comm --p-points base,boundary --q-points fp16,int8 --s-points default,tuned --no-full-enumeration --dry-run
python tools/summarize_s2_anchor_probe.py --probe-results probe_results/std_basebev_backbone_schedule_anchor.json --out-md probe_results/std_basebev_backbone_schedule_anchor.md
```

## v2xvit_qgranularity_p_anchor

- priority: high
- readiness: EXISTING_EVIDENCE_BINDING
- next_stage_role: Bind existing C4 evidence into predictor gates; do not run a new AP/latency probe in the safe predictor v0 stop target.
- evidence_level: existing_measured_cell_needs_predictor_binding
- estimated_cost_class: reuse_existing_cell_or_low_2P_x_2Q_anchor
- blocking_condition: V2X-ViT cannot receive a cheap full-model separability verdict until known Q-granularity x P evidence is attached to the predictor.
- question: Does the known V2X-ViT Q-granularity x P interaction block a cheap full-model separability prediction?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- C4 Q-granularity x P result file or extracted AP/latency table
- one base P point and one pruned/boundary P point
- per-tensor and per-channel quantization labels
- AP noise floor and latency noise floor

### Expected Artifacts

- probe_specs/v2xvit_qgranularity_p_anchor.yaml
- probe_results/v2xvit_qgranularity_p_anchor.json
- probe_results/v2xvit_qgranularity_p_anchor.md

### Pass/Fail Criteria

- PASS_PAIR_REQUIRED if Q-granularity AP or latency rank changes across P
- PASS_LOW_RISK_ANCHOR only if both AP and latency ranks are stable within noise
- FAIL_STATIC_ONLY if only buildability or manifest metadata is available
- UNKNOWN if AP noise floor or quantization granularity metadata is missing

### Command Skeleton

```bash
# Bind existing C4 evidence; do not launch a new AP probe in safe predictor v0.
python scripts/stage1_predict_coupling.py --help
```

## codriving_resnet_completion_anchor

- priority: medium
- readiness: COMPLETED_EXISTING_MEASURED_BINDING
- next_stage_role: Already satisfied by measured-results binding; keep it as a regression/fixture, not as a next-stage experiment.
- evidence_level: measured_negative_anchor_with_estimated_latency_gap
- estimated_cost_class: low_completion_remeasure_missing_anchor_points
- blocking_condition: CoDriving can keep measured-anchor wording only for the measured ResNet envelope; missing p25/p75 latency points block broader claims.
- question: Does CoDriving remain negative when estimated p25/p75 latencies are replaced by real anchor measurements?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- CoDriving backbone.s0/s1/s2 census candidates
- missing or estimated p25/p75 latency points from prior reports
- same H800 TVM protocol metadata used by existing coupling-map evidence

### Expected Artifacts

- probe_specs/codriving_resnet_completion_anchor.yaml
- probe_results/codriving_resnet_completion_anchor.json
- probe_results/codriving_resnet_completion_anchor.md

### Pass/Fail Criteria

- PASS_MEASURED_NEGATIVE if real anchor measurements preserve no argmin drift
- PASS_MEASURED_NEGATIVE if serial HV remains equivalent to joint HV
- FAIL_COUPLED if replacing estimates creates rank flips or schedule/Q drift
- UNKNOWN if new measurements use a different protocol or device envelope

### Command Skeleton

```bash
# Existing measured-results binding is already present:
python -m json.tool results/stage1_model_predict/standard_conv_anchor_measured_results_v1.json >/tmp/standard_conv_anchor_measured_results_v1.check.json
```

## maxfusion_coverage_anchor

- priority: medium
- readiness: STATIC_PROOF_CANDIDATE
- next_stage_role: F-Cooper-only static proof candidate. Can mark MaxFusion as channel-preserving only if shape contract is present; otherwise keep full-model low-confidence.
- evidence_level: static_only_pending_anchor_probe
- estimated_cost_class: lowest_static_proof_or_single_operator_check
- blocking_condition: F-Cooper full-model separability remains blocked until MaxFusion is measured or statically proven channel-preserving and negligible.
- question: Is F-Cooper MaxFusion truly non-blocking for full-model separability?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- F-Cooper skipped_subgraph_types from the census
- MaxFusion module identity and input/output shape contract
- one dense-backbone anchor result, if available

### Expected Artifacts

- probe_specs/maxfusion_coverage_anchor.yaml
- probe_results/maxfusion_coverage_anchor.json
- probe_results/maxfusion_coverage_anchor.md

### Pass/Fail Criteria

- PASS_DENSE_ONLY if MaxFusion is unprofiled but dense backbone is stable
- PASS_FULL_SCOPE only if MaxFusion is negligible and shape-preserving
- FAIL_UNCOVERED if MaxFusion has measurable latency or rank impact
- UNKNOWN if MaxFusion cannot be bounded from available metadata

### Command Skeleton

```bash
python tools/run_s2_anchor_probe.py --probe-id maxfusion_coverage_anchor --models fcooper --p-points base,boundary --q-points fp16,int8 --s-points default,tuned --no-full-enumeration --dry-run
python tools/summarize_s2_anchor_probe.py --probe-results probe_results/maxfusion_coverage_anchor.json --out-md probe_results/maxfusion_coverage_anchor.md
```

## per_stage_q_ap_sensitivity_anchor

- priority: medium
- readiness: BACKLOG_OR_EXISTING_EVIDENCE_BINDING
- next_stage_role: Safe predictor v0 should expose this as an AP-sensitive gate, but new AP experiments are outside the v0 stop target.
- evidence_level: ap_sensitive_q_gate_pending_or_existing_binding
- estimated_cost_class: low_2P_selected_stage_Q_AP_anchor_no_grid
- blocking_condition: Latency-only separability prediction is insufficient when pruning changes which stage or quantization granularity preserves AP.
- question: Does pruning change which stage should keep higher-precision or per-channel quantization for AP?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- one pruned P point and base P point
- per-stage quantization assignment or granularity labels
- AP metric with declared noise floor
- latency metric with declared noise floor

### Expected Artifacts

- probe_specs/per_stage_q_ap_sensitivity_anchor.yaml
- probe_results/per_stage_q_ap_sensitivity_anchor.json
- probe_results/per_stage_q_ap_sensitivity_anchor.md

### Pass/Fail Criteria

- PASS_PAIR_REQUIRED if the best per-stage Q choice changes after pruning
- PASS_LOW_RISK_ANCHOR only if AP and latency choices are stable within noise
- FAIL_LATENCY_ONLY if AP evidence is absent for an AP-sensitive model
- UNKNOWN if per-stage quantization metadata is missing

### Command Skeleton

```bash
python tools/run_s2_anchor_probe.py --probe-id per_stage_q_ap_sensitivity_anchor --models v2xvit,pyramid_camera,pyramid_lidar --p-points base,boundary --q-points fp16,int8 --s-points default,tuned --no-full-enumeration --dry-run
python tools/summarize_s2_anchor_probe.py --probe-results probe_results/per_stage_q_ap_sensitivity_anchor.json --out-md probe_results/per_stage_q_ap_sensitivity_anchor.md
```

## pyramid_mixed_p_hub_context_anchor

- priority: medium
- readiness: EXISTING_EVIDENCE_BINDING
- next_stage_role: Bind existing Pyramid P-hub evidence and manifest adjacency as a model-level overpromotion guardrail.
- evidence_level: existing_measured_positive_p_hub_context
- estimated_cost_class: reuse_existing_pyramid_cells_plus_static_adjacency
- blocking_condition: Local standard-conv groups in Pyramid cannot be promoted to model-level low-risk while grouped-conv IC_BN P-hub neighbors dominate.
- question: Does a local standard-conv group sit inside a model whose grouped-conv IC_BN P-hub dominates the architecture-level verdict?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- C2/C3/C7 Pyramid coupling-map evidence
- manifest adjacency between grouped BEV encoder and local standard Conv2d groups
- fanout/head coupling metadata
- scope label distinguishing local dense subgraph from full model

### Expected Artifacts

- probe_specs/pyramid_mixed_p_hub_context_anchor.yaml
- probe_results/pyramid_mixed_p_hub_context_anchor.json
- probe_results/pyramid_mixed_p_hub_context_anchor.md

### Pass/Fail Criteria

- PASS_SCOPE_BLOCK if local standard-conv evidence is dense-subgraph-only
- PASS_P_HUB_CONTEXT if grouped-conv neighbor controls Q/S schedule choice
- FAIL_OVERPROMOTION if model-level low-risk is emitted from local standard Conv2d only
- UNKNOWN if adjacency or fanout metadata is missing

### Command Skeleton

```bash
# Bind existing Pyramid P-hub evidence; no new latency probe in safe predictor v0.
python -m json.tool results/coupling_map_matrix.json >/tmp/coupling_map_matrix.check.json
```

## standard_neck_deconv_anchor

- priority: medium
- readiness: MEASUREMENT_BACKLOG_RUNNER_REQUIRED
- next_stage_role: Backlog measurement. Safe predictor v0 must keep neck/deconv low-confidence or dense-subgraph-only until a real runner exists.
- evidence_level: static_only_pending_anchor_probe
- estimated_cost_class: low_anchor_2P_x_2Q_x_2S_max_no_grid
- blocking_condition: Neck/deconv standard-conv regimes cannot inherit CoDriving 3x3 ResNet separability without their own S-axis anchor check.
- question: Do deconv/neck standard-conv regimes share CoDriving ResNet separability, or do they need separate S-axis treatment?
- enumeration_policy: Full P x Q x S enumeration is prohibited for S2 anchor probes. Each probe may use only one representative regime, base plus one pruned/boundary P point, fp16/int8 Q points, and default/tuned or schedule-swap S checks.

### Minimal Inputs

- one neck or deconv candidate from the census
- base width and one pruned or boundary width
- fp16/int8 labels
- default and tuned schedule labels
- schedule-swap pairing for Q/S checks

### Expected Artifacts

- probe_specs/standard_neck_deconv_anchor.yaml
- probe_results/standard_neck_deconv_anchor.json
- probe_results/standard_neck_deconv_anchor.md

### Pass/Fail Criteria

- PASS_LOW_RISK_ANCHOR if deconv/neck ordering is stable across P and Q
- FAIL_COUPLED if default-vs-tuned rank flips across P
- FAIL_COUPLED if fp16/int8 schedule-swap changes the best S choice
- UNKNOWN if ConvTranspose2d shape fields are missing

### Command Skeleton

```bash
python tools/run_s2_anchor_probe.py --probe-id standard_neck_deconv_anchor --models attfuse,codriving,disconet,fcooper,pyramid_camera,pyramid_lidar,v2vnet,v2xvit,where2comm --p-points base,boundary --q-points fp16,int8 --s-points default,tuned --no-full-enumeration --dry-run
python tools/summarize_s2_anchor_probe.py --probe-results probe_results/standard_neck_deconv_anchor.json --out-md probe_results/standard_neck_deconv_anchor.md
```
