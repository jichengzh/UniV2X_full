# Stage2 LUT Supervisor Report

- decision: NO_GO
- repeat_ratio: 0.914
- unique configs: 113
- latency/AP/energy cells: 70 / 22 / 21
- latency ready jobs: 0
- blocked jobs: 0
- active quarantined configs: 0

## Recommendations
- repeat_ratio_high: stop repeat, generate new candidates (repeat_ratio=0.914 > 0.300)
- latency_queue_low: ask artifact-agent for next batch (latency_ready_jobs=0 < 10)
- energy_axis_lag: prioritize energy subset (energy_cells=21 < 0.40 * latency_cells=70)
- ap_axis_lag: prioritize AP source/eval (ap_cells=22 < 0.40 * latency_cells=70)

## Latest Job Status Counts

## Top Missing Axes
- ap blocked: 60 (coverage:pyramid_lidar:w24x128x256:fp16, coverage:pyramid_lidar:w40x128x256:fp16, coverage:pyramid_lidar:w56x128x256:fp16, coverage:pyramid_lidar:w64x48x256:fp16, coverage:pyramid_lidar:w64x64x256:fp16)
- energy blocked: 60 (coverage:pyramid_lidar:w24x128x256:fp16, coverage:pyramid_lidar:w40x128x256:fp16, coverage:pyramid_lidar:w56x128x256:fp16, coverage:pyramid_lidar:w64x48x256:fp16, coverage:pyramid_lidar:w64x64x256:fp16)
- latency missing: 60 (coverage:pyramid_lidar:w24x128x256:fp16, coverage:pyramid_lidar:w40x128x256:fp16, coverage:pyramid_lidar:w56x128x256:fp16, coverage:pyramid_lidar:w64x48x256:fp16, coverage:pyramid_lidar:w64x64x256:fp16)
