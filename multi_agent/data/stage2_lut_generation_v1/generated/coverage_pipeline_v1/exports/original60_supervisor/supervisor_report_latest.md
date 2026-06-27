# Stage2 LUT Supervisor Report

- decision: CONDITIONAL_GO
- repeat_ratio: 0.000
- unique configs: 172
- latency/AP/energy cells: 116 / 0 / 56
- latency ready jobs: 0
- blocked jobs: 5
- active quarantined configs: 4

## Recommendations
- latency_queue_low: ask artifact-agent for next batch (latency_ready_jobs=0 < 10)
- ap_axis_lag: prioritize AP source/eval (ap_cells=0 < 0.40 * latency_cells=116)

## Latest Job Status Counts
- failed: 9
- preflight_blocked: 5
- succeeded: 114

## Top Missing Axes
- ap blocked: 60 (coverage:pyramid_lidar:w24x128x256:fp16, coverage:pyramid_lidar:w40x128x256:fp16, coverage:pyramid_lidar:w56x128x256:fp16, coverage:pyramid_lidar:w64x48x256:fp16, coverage:pyramid_lidar:w64x64x256:fp16)
- energy missing: 4 (coverage:pyramid_lidar:w64x128x96:fp16, coverage:pyramid_lidar:w56x48x256:fp16)
- latency missing: 4 (coverage:pyramid_lidar:w64x128x96:fp16, coverage:pyramid_lidar:w56x48x256:fp16)
