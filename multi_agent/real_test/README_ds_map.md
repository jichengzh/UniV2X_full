# CoDriving DS prediction map: DS = f(AP, latency)

Purpose: given a perception algorithm's `(vehicle AP50, tau_perc latency_ms)`, estimate its CoDriving closed-loop driving score (DS), and inspect the related safety cliff with a separate collision map.

## Current Artifacts

- `ds_ap_latency_all_measured.csv` - consolidated 65-cell measured table:
  `13 latency points x 5 AP points`, with DS, RC, vehicle-collision rate, timeout rate, and source.
- `ap_latency_metrics_by_source.csv` - by-source aggregation, preserving duplicate endpoint measurements.
- `ds_ap_latency_map_v2.png` - measured DS map, real axes `latency_ms x AP50`.
- `ap_tau_collision_map_v2.png` - measured vehicle collision episode-rate map.
- `ds_map_build_v2.py` - v2 plotting script and `predict_ds(ap50, latency_ms)` query function.

Legacy files are kept for traceability:

- `ds_ap_latency_map.png`, `ds_ap_latency_merged.csv`, `ds_map_build.py` - old 31-point map before plateau and Phase-A cliff-band completion.

## Query

```python
from ds_map_build_v2 import predict_ds

predict_ds(0.841, 600)  # -> 77.9
predict_ds(0.559, 700)  # -> 17.7
predict_ds(0.281, 750)  # -> 29.2
```

Interpolation is linear over the measured rectangle. Values outside the measured range are clipped to:

- AP50: `[0.281, 0.841]`
- latency: `[0, 800] ms`

## Measured DS Table

Honest DS: `score_composed`; `TIMEOUT_SKIP` or missing score is counted as `0`.

| AP50 \ latency ms | 0 | 100 | 200 | 300 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.841 | 85.3 | 91.7 | 84.7 | 68.1 | 67.2 | 64.2 | 73.3 | 81.6 | 77.9 | 28.0 | 28.2 | 30.2 | 26.6 |
| 0.783 | 68.5 | 77.8 | 83.3 | 67.8 | 66.9 | 63.1 | 65.1 | 61.9 | 65.9 | 29.4 | 28.6 | 25.7 | 37.2 |
| 0.559 | 56.4 | 72.2 | 75.0 | 68.6 | 68.5 | 63.4 | 56.0 | 54.9 | 76.8 | 27.5 | 17.7 | 22.6 | 23.1 |
| 0.360 | 63.8 | 75.0 | 69.4 | 58.3 | 54.6 | 48.1 | 64.3 | 58.6 | 58.5 | 26.9 | 31.3 | 29.9 | 27.5 |
| 0.281 | 66.7 | 75.0 | 60.0 | 62.5 | 62.5 | 72.9 | 82.6 | 85.5 | 81.3 | 28.7 | 27.9 | 29.2 | 25.6 |

## Key Reading

1. DS cliff is now measured across the full 5-AP grid: the missing `650/700/750ms x AP={0.783,0.559,0.281}` cells are no longer interpolated.
2. DS still shows latency as the dominant factor: 600ms is mostly high, 650ms and later are mostly low.
3. AP-dependent cliff movement should not be claimed from composed DS alone. The safety/collision claim belongs to `ap_tau_collision_map_v2.png`.
4. Collision/RC columns use the interaction-analysis convention: non-timeout episodes only, with `timeout_pct` reported separately. DS uses the honest convention and includes timeouts as zero.

## Sources

- Main grid: `latency={0,200,400,600,800} x drop={0,0.25,0.50,0.70,0.85}`, clean6 x N=3.
- Plateau completion: `latency={100,300,450,500,550} x all 5 AP`, clean6 x N=3.
- Interaction endpoints: `latency={600,650,700,750,800} x drop={0,0.70}`, clean6 x N=6.
- Phase A cliff-band fill: `latency={650,700,750} x drop={0.25,0.50,0.85}`, clean6 x N=6.

Raw H800 result pattern:

```text
/exdata/jichengzhi/V2Xverse_apknob/results/
  results_driving_grid_g{code}_r{route}_n{rep}/
  v2x_final/town05_short_collab/*/ego_vehicle_0/results.json
```

Drop to AP50 calibration:

| drop | AP50 |
|---:|---:|
| 0.00 | 0.841 |
| 0.25 | 0.783 |
| 0.50 | 0.559 |
| 0.70 | 0.360 |
| 0.85 | 0.281 |
