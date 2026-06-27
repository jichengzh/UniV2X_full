# Standard Conv Coupling Deepening Plan v1

## Purpose

This addendum corrects one weak assumption in the Stage1 predictor plan: `groups=1` must not be treated as proof of P/Q/S separability.

The existing evidence only supports a narrower statement:

> CoDriving-like standard 3x3 ResNet is measured negative under the tested P/Q/S settings. That does not automatically generalize to every standard Conv2d regime.

The predictor should therefore treat standard convolution as:

- low hard-format risk for the known grouped-conv IC_BN cliff,
- unknown schedule and quantization coupling risk until shape features or low-cost probes support a stronger claim.

## Evidence Boundary From Existing Reports

- `multi_agent/methods/design/coupling_map_v1.md`: the correct coupling criterion is argmin drift plus joint-vs-serial HV gap; separability is measured, not inferred from model labels.
- `results/coupling_map_matrix.json`: CoDriving is negative across C0c, C1cod, and C6, but this is still one standard-conv family.
- `results/b4_codriving_ablation_results.json`: the CoDriving negative result depends on a 3x3 ResNet regime with near-constant tuning ratio.
- `results/perstage_quant_AP_verdict_v2.md`: quantization sensitivity can change after pruning; buildability alone is not enough to rule out Q-related coupling.

## Phase S0: Evidence Envelope

Status: started.

Goal: replace broad wording such as "standard conv is separable" with an evidence envelope.

Deliverable:

- Existing CoDriving result should be recorded as `measured_negative_anchor_codriving_resnet`, not as a universal standard-conv rule.
- Predictor language should distinguish:
  - `standard_conv_hard_format_low_risk`
  - `standard_conv_unmeasured_schedule_risk`
  - `standard_conv_unmeasured_quantization_risk`

Acceptance:

- No architecture-level verdict may use `groups=1` alone.
- `PREDICTED_SEPARABLE_LOW_RISK` requires either measured evidence or anchor-probe evidence.

## Phase S1: Static Standard-Conv Census

Status: completed for current manifests.

Artifacts:

- `framework/stage1/standard_conv_census.py`
- `results/stage1_model_predict/standard_conv_census_v1.json`
- `multi_agent/methods/design/stage1-model-predict/standard_conv_census_v1.md`

Command:

```bash
python framework/stage1/standard_conv_census.py
```

Current census summary:

- 6 manifests scanned.
- 27 search groups inspected.
- 20 standard-conv candidates found.
- 8 high-priority anchors: AttFuse and V2X-ViT, because attention/transformer fusion is skipped.
- 11 medium-priority anchors: CoDriving, F-Cooper, and standard-conv neck/deconv regimes that need coverage or schedule checks.

Key limitation:

Current manifests do not expose `cin`, `cout`, `kernel`, `stride`, input HW, or output HW per Conv2d group. The census can select probe candidates, but it cannot calibrate coupling risk from shape mechanics yet.

Required Stage1 manifest upgrade:

- Add per-root-op structural fields:
  - `cin`
  - `cout`
  - `groups`
  - `kernel`
  - `stride`
  - `input_hw`
  - `output_hw`
  - `fanout_buckets`

## Phase S2: Low-Cost Standard-Conv Anchor Probes

Goal: detect whether standard Conv2d regimes have P/Q/S coupling without full enumeration.

Probe only one representative shape per regime:

| Regime | Models | Priority | Probe purpose |
|---|---|---:|---|
| BaseBEVBackbone standard conv | F-Cooper, AttFuse, V2X-ViT | high/medium | Check if OpenCOOD backbone behaves like CoDriving ResNet or has its own schedule drift |
| Attention/fusion-skipped standard conv | AttFuse, V2X-ViT | high | Prevent dense-backbone separability from being promoted to full-model separability |
| Channel-preserving / pooling fusion | CoDriving, F-Cooper | medium | Verify full-model coverage is actually low risk |
| Deconv / neck double conv | CoDriving, F-Cooper, AttFuse, V2X-ViT | medium | Check schedule regime not covered by CoDriving 3x3 ResNet evidence |
| CoDriving ResNet anchor | CoDriving | medium | Replace estimated p25/p75 latency with real anchor measurements if needed |

Each anchor probe should measure:

- P axis: base width plus one moderate/pruned width near a shape boundary.
- Q axis: fp16 and int8, preferably same-graph schedule-swap for Q/S.
- S axis: default dlight and tuned MetaSchedule.
- Optional H axis: batch 1 vs a throughput batch only for high-priority candidates.

Do not run full P x Q x S grids. The probe is only checking:

- default ordering vs tuned ordering,
- schedule gain variance across P,
- int8 own-tuned vs schedule-swapped latency,
- batch-induced rank flip.

## Phase S3: Quantization-Sensitivity Probe

Goal: catch standard-conv Q coupling that is invisible to int8 buildability.

Required checks:

- forced-all-INT8 vs auto/mixed INT8 under the same build path,
- per-tensor vs per-channel when available,
- stage-wise mixed precision for pruned models,
- AP delta or proxy sensitivity as a function of P.

Coupling signal:

- best Q choice changes as P changes,
- or Q choice changes the best S schedule at fixed P,
- or pruning makes a previously harmless forced-INT8 path lose AP beyond the noise floor.

This phase is motivated by `perstage_quant_AP_verdict_v2.md`: forced-all-INT8 and mixed precision diverged strongly after pruning.

## Phase S4: Mini Three-Arm Validation For High-Risk Shapes

Run only when S2/S3 marks a standard-conv regime high risk.

Mini arms:

- `A-joint`: P x Q x S small-budget search on selected anchors.
- `A-serial`: P -> Q -> S with the same budget.
- `A-noS`: P/Q with fixed default schedule.

Acceptance:

- JOINT if argmin drift exists and serial HV loses clearly.
- SERIAL if serial reaches joint HV and no rank flip survives noise checks.
- UNKNOWN if results rely on cast-chain artifacts, estimated latencies, or skipped subgraphs.

## Phase S5: Predictor Rule Update

Replace any implicit rule:

```text
groups=1 -> separable
```

with:

```text
groups=1 -> no grouped-conv IC_BN hard cliff; require shape/probe evidence before low-risk separability
```

Expected verdict behavior:

| Condition | Verdict |
|---|---|
| CoDriving measured evidence attached | `MEASURED_SEPARABLE` |
| CoDriving-like shape, no measured evidence | `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` or dense-subgraph-only low risk |
| F-Cooper MaxFusion with no anchor probe | `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` at full-model scope |
| AttFuse or V2X-ViT attention skipped | `FUSION_UNCOVERED_UNKNOWN` |
| Standard conv anchor shows schedule/Q drift | `Q_S_COUPLED` or `LOW_CONFIDENCE_NEEDS_TARGETED_PROBE` |

## Phase S6: Calibration And Reporting

Final deliverables:

- `results/stage1_model_predict/standard_conv_census_v1.json` becomes the static input to predictor calibration.
- Anchor probe results are stored as separate JSON files; they must not overwrite measured coupling-map evidence.
- The predictor report must include an evidence level:
  - `measured`
  - `anchor_probed`
  - `static_only`
  - `uncovered_unknown`

Success criterion:

The predictor can say "this standard-conv dense backbone is low risk" only when the evidence level justifies it, and it never says "this whole model is separable" while attention/fusion/custom subgraphs are skipped or unprofiled.
