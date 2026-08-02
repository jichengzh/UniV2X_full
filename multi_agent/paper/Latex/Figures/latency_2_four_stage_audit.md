# `latency_2.pdf` four-stage evidence audit

## Correct stage definition

The original figure has four latency categories:

1. Image processing for camera input / voxelization for LiDAR input.
2. Multi-scale BEV backbone.
3. LiDAR or camera feature fusion.
4. Detection, including the prediction heads and post-processing shown in the
   diagram.

The percentages printed in the original figure are 13.4%, 61.6%, 16.9%, and
8.1%, respectively. They sum to 100%.

## Pyramid: comparison with the available full-stage record

The original figure is the only local artifact that contains all four printed
percentages. The available raw Pyramid record is:

- `paper_learning/2. AAAI最终故事/data/p0_2c_dair_full_breakdown.json`
- DAIR-V2X validation split, 150 measured samples.
- NVIDIA GeForce RTX 4090.
- CUDA-event module timing with synchronized end-to-end wall time.

The following values can be recomputed directly from that JSON:

| Quantity | Calculation | Share of measured e2e |
|---|---:|---:|
| Input processing | `encoder / e2e` | 13.4198% |
| Outer backbone-like stages | `(backbone + aligner + pyramid) / e2e` | 60.4887% |
| Post-processing | `postproc / e2e` | 15.9717% |
| Shrink and prediction heads | `(shrink + heads) / e2e` | 8.1500% |

Therefore:

- 13.4% is directly reproduced by `encoder / e2e`.
- `(shrink + heads) / e2e` is numerically close to 8.1%, but it excludes
  post-processing and therefore does not reproduce the visual Detection
  boundary.
- 61.6% and 16.9% are not exactly reproduced by this seven-stage JSON.
- More importantly, the JSON's outer `pyramid` stage contains both multi-scale
  backbone and cooperative-fusion operations. It cannot be assigned wholly to
  either of the second and third visual categories without a separate,
  same-run internal breakdown.

The original four percentages must consequently be treated as the reported
Pyramid profile used to construct the WPS figure, not as four values derived
from `p0_2c_dair_full_breakdown.json`. The exact original calculation sheet or
same-run four-boundary profiler is not present in the repository.

## F-Cooper: stage records exist, but a residual remains

Source:

- `results/fcooper_postproc_recheck_20260728/fcooper_cuda_top1000_recheck.json`
- OPV2V, warm-up 20, measure 100.
- NVIDIA GeForce RTX 4090, PyTorch 2.0.1+cu118.
- CUDA rotated NMS with the same top-1000 candidate cap as the original CPU
  implementation.

Using the four visual boundaries gives:

| Stage | Direct aggregation | Latency (ms) | Share of measured e2e |
|---|---|---:|---:|
| Input processing / voxelization | `encoder` | 3.044861 | 14.0241% |
| Multi-scale BEV backbone | `backbone + shrinker_m1` | 10.340492 | 47.6265% |
| Feature fusion | `fusion_net` | 1.398678 | 6.4421% |
| Detection | `shrink_conv + heads + postproc` | 6.037457 | 27.8075% |
| Residual wall/framework time | `e2e - attributed stages` | 0.890132 | 4.0998% |

The four attributed stages total 20.821488 ms. The measured e2e mean is
21.711619 ms. The 0.890132 ms difference is retained as residual above.
Normalizing only the four attributed stages would yield 14.6236%, 49.6626%,
6.7175%, and 28.9963%, but those values are shares of attributed latency, not
end-to-end latency contributions, and are therefore not used for a cross-model
range.

The earlier `data/v2x_baseline_timing/fcooper_p0.json` used a temporary CUDA
wrapper that omitted the original top-1000 candidate cap. It measured 7.400981
ms of post-processing and 24.507849 ms e2e, but is not used for the corrected
four-stage comparison because its NMS boundary differs from both the CPU
baseline and the integrated Pyramid CUDA path. The implementation evidence is:

- `scripts/phase2/m4_9_v2x_baselines_timing.py` applies the temporary wrapper
  when invoked with `--opt p0`.
- `scripts/phase2/m4_8_cuda_kernel_replacements.py::nms_rotated_mmcv` sends
  every thresholded box directly to `mmcv.ops.nms_rotated` and has no top-k
  operation or `top` argument.
- The corrected remeasurement uses the integrated CUDA implementation with
  the original `top=1000` score cap and does not enable the temporary wrapper.

## CoDriving: complete four-stage evidence does not exist

Sources:

- `results/CD_A1_codriving_latency_4090.csv`
- `results/CD_A1b_fusion_internal_4090.csv`
- `multi_agent/model/codriving_structure_audit_v1.md`

The CoDriving record explicitly measures a post-scatter dense-BEV path and
states that pillar/VFE, scatter, NMS, and post-processing are not included.
It therefore cannot provide either:

- the first-stage input-processing / voxelization share, or
- the complete fourth-stage detection share.

Its dense-path backbone, fusion, and heads measurements must not be normalized
into a four-stage end-to-end distribution.

## Decision

The withdrawn `latency_2_range.pdf` used three attributed dense-forward
categories and ratio-mapped internal measurements from separate runs. It was
not equivalent to the original four-stage figure.

A three-model four-stage range is not reported until CoDriving is re-profiled
from real LiDAR input through post-processing under one consistent run and the
Pyramid four-stage calculation boundary is recovered or remeasured.
