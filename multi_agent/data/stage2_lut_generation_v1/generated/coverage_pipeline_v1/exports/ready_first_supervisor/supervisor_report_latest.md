# Stage2 LUT Supervisor Report

- decision: CONDITIONAL_GO
- repeat_ratio: 0.000
- unique configs: 12
- latency/AP/energy cells: 12 / 0 / 0
- latency ready jobs: 0
- blocked jobs: 0
- active quarantined configs: 0

## Recommendations
- latency_queue_low: ask artifact-agent for next batch (latency_ready_jobs=0 < 1)
- energy_axis_lag: prioritize energy subset (energy_cells=0 < 0.40 * latency_cells=12)
- ap_axis_lag: prioritize AP source/eval (ap_cells=0 < 0.40 * latency_cells=12)

## Latest Job Status Counts
- failed: 2
- succeeded: 12

## Top Missing Axes
- ap blocked: 6 (coverage_ready:pyramid_lidar:w48x96x256:fp16, coverage_ready:pyramid_lidar:w48x32x128:fp16, coverage_ready:pyramid_lidar:w64x32x128:fp16, coverage_ready:pyramid_lidar:w48x64x192:fp16, coverage_ready:pyramid_lidar:w64x64x192:fp16)
