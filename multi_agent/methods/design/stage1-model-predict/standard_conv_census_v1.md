# Standard Conv Coupling Census v1

This is a static-only census from existing Stage1 partition manifests. It is not a coupling verdict.

## Summary

- models: 9
- search_groups: 39
- standard_conv_candidates: 32
- high_priority_anchors: 20
- medium_priority_anchors: 11

## Counts By Model

- attfuse: 4
- codriving: 4
- disconet: 4
- fcooper: 4
- pyramid_camera: 2
- pyramid_lidar: 2
- v2vnet: 4
- v2xvit: 4
- where2comm: 4

## Counts By Family

- base_bev_backbone_conv: 18
- deconv_neck: 9
- neck_double_conv: 9
- resnet_basicblock_3x3: 5
- resnet_downsample_1x1_or_projection: 4

## Anchor Probe Candidates

| priority | model | search_group | bucket | widths | families | reasons |
|---|---|---|---|---|---|---|
| high | attfuse | backbone.s0 | backbone | 64 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | attfuse | backbone.s1 | backbone | 128 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | attfuse | backbone.s2 | backbone | 256 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | attfuse | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| high | disconet | backbone.s0 | backbone | 64 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | disconet | backbone.s1 | backbone | 128 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | disconet | backbone.s2 | backbone | 256 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | disconet | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| high | v2vnet | backbone.s0 | backbone | 64 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | v2vnet | backbone.s1 | backbone | 128 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | v2vnet | backbone.s2 | backbone | 256 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | v2vnet | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_unknown_fusion<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| high | v2xvit | backbone.s0 | backbone | 64 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | v2xvit | backbone.s1 | backbone | 128 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | v2xvit | backbone.s2 | backbone | 256 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | v2xvit | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | where2comm | backbone.s0 | backbone | 64 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | where2comm | backbone.s1 | backbone | 128 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | where2comm | backbone.s2 | backbone | 256 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| high | where2comm | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>full_model_has_skipped_fusion_or_attention<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| medium | codriving | backbone.s0 | backbone | 64 | resnet_basicblock_3x3<br>resnet_downsample_1x1_or_projection | skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| medium | codriving | backbone.s1 | backbone | 128 | resnet_basicblock_3x3<br>resnet_downsample_1x1_or_projection | skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| medium | codriving | backbone.s2 | backbone | 256 | resnet_basicblock_3x3<br>resnet_downsample_1x1_or_projection | skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| medium | codriving | neck | neck | 128 | deconv_neck<br>neck_double_conv | skipped_channel_preserving_fusion_needs_coverage_check<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| medium | fcooper | backbone.s0 | backbone | 64 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| medium | fcooper | backbone.s1 | backbone | 128 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| medium | fcooper | backbone.s2 | backbone | 256 | base_bev_backbone_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split |
| medium | fcooper | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | standard_conv_family_has_no_direct_three_arm_or_anchor_probe<br>skipped_channel_preserving_fusion_needs_coverage_check<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| medium | pyramid_camera | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| medium | pyramid_lidar | backbone.s0 | backbone | 64 | resnet_basicblock_3x3<br>resnet_downsample_1x1_or_projection | skipped_channel_preserving_fusion_needs_coverage_check<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |
| medium | pyramid_lidar | neck | neck | 128,256 | deconv_neck<br>neck_double_conv | skipped_channel_preserving_fusion_needs_coverage_check<br>deconv_or_neck_schedule_regime_not_covered_by_codriving_resnet<br>mixed_conv2d_convtranspose_group_requires_subgroup_split<br>heads_are_accuracy_sensitive_and_often_quantization_locked |

## Interpretation

- groups=1 removes the grouped-conv IC_BN hard-cliff mechanism, but does not prove P/Q/S separability.
- Existing manifests lack cin/cout/kernel/HW, so this census is a regime selector; Stage1 must add those fields before calibrated prediction.
- Run low-cost anchor probes only for high/medium candidates, then calibrate predictor verdicts.

## Recommended Low-Cost Probe Queue

| priority | probe_id | representative_models | question | minimal_measurement |
|---|---|---|---|---|
| high | std_basebev_backbone_schedule_anchor | attfuse, disconet, fcooper, v2vnet, v2xvit, where2comm | Does OpenCOOD BaseBEVBackbone standard Conv2d have schedule-gain drift across widths, or does it behave like CoDriving ResNet? | For one representative dense backbone, measure base and one pruned/boundary width under fp16/int8 default and tuned schedules. |
| high | attention_fusion_coverage_anchor | attfuse, v2xvit, where2comm | Can dense-backbone separability be promoted to full-model separability when attention/transformer fusion is skipped? | No attention measurement in safe predictor v0: attach typed skipped-subgraph metadata and keep full-model verdict blocked until attention trace/export exists. |
| high | v2xvit_qgranularity_p_anchor | v2xvit | Does the known V2X-ViT Q-granularity x P interaction block a cheap full-model separability prediction? | Bind C4-style per-channel/per-tensor AP and latency evidence to one base/pruned P pair; do not infer from buildability alone. |
| high | routing_fusion_coverage_anchor | attfuse, disconet, fcooper, v2vnet, v2xvit, where2comm | Do skipped fusion/routing operators change the feasible Q/S action set or only add a constant uncovered cost? | Bind existing C5-style routing feasibility/op-blacklist evidence where available; otherwise attach typed skip metadata and keep fusion/routing as a blocker. |
| medium | codriving_resnet_completion_anchor | codriving | Does CoDriving remain negative when estimated p25/p75 latencies are replaced by real anchor measurements? | Remeasure the missing/pruned CoDriving widths with the same H800 TVM protocol used for base/p50. |
| medium | standard_neck_deconv_anchor | attfuse, codriving, disconet, fcooper, pyramid_camera, pyramid_lidar, v2vnet, v2xvit, where2comm | Do deconv/neck standard-conv regimes share CoDriving ResNet separability, or do they need separate S-axis treatment? | Probe one neck/deconv representative with default vs tuned schedule and fp16/int8 schedule-swap. |
| medium | pyramid_mixed_p_hub_context_anchor | pyramid_camera, pyramid_lidar | Does a local standard-conv group sit inside a model whose grouped-conv IC_BN P-hub dominates the architecture-level verdict? | Use existing C2/C3/C7 Pyramid evidence plus manifest adjacency to mark local standard Conv2d as dense-subgraph-only. |
| medium | per_stage_q_ap_sensitivity_anchor | v2xvit, pyramid_camera, pyramid_lidar | Does pruning change which stage should keep higher-precision or per-channel quantization for AP? | One per-stage Q/AP sensitivity check after pruning, using existing AP-noise thresholds and no full P x Q grid. |
| medium | maxfusion_coverage_anchor | fcooper | Is F-Cooper MaxFusion truly non-blocking for full-model separability? | Measure or statically prove MaxFusion latency/shape preservation once; do not use it as a blanket standard-conv rule. |

## Required Manifest Upgrade

Every Conv2d/ConvTranspose2d group needs cin, cout, groups, kernel, stride, input HW, output HW, and fanout fields.
Without these fields, the predictor can only select probe candidates; it cannot make calibrated architecture-level claims.
