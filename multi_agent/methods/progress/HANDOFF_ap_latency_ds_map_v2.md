# HANDOFF: AP x latency DS map after Phase A/B (v2, 2026-06-24)

> This supersedes `HANDOFF_ap_latency_ds_map_v1.md` for the AP x latency map status. Phase A cliff-band AP completion and Phase B artifact regeneration are complete.

## 0. Current Status

1. **Phase A complete**: filled the cliff-band gaps `latency={650,700,750}ms x AP={0.783,0.559,0.281}`.
2. **Phase B complete**: regenerated the measured DS map and the separate collision/safety map from H800 raw results.
3. The AP x latency DS table is now a full measured rectangle: `13 latency points x 5 AP points = 65 cells`.
4. The old 31-point artifacts are retained for traceability but should not be used as the current map.

## 1. New Artifacts

Local files:

- `multi_agent/real_test/ds_ap_latency_all_measured.csv`
- `multi_agent/real_test/ap_latency_metrics_by_source.csv`
- `multi_agent/real_test/ds_ap_latency_map_v2.png`
- `multi_agent/real_test/ap_tau_collision_map_v2.png`
- `multi_agent/real_test/ds_map_build_v2.py`
- `multi_agent/real_test/README_ds_map.md`

Remote H800 log:

- `/exdata/jichengzhi/dsmap_cliff_phaseA.log`

Raw H800 result pattern:

```text
/exdata/jichengzhi/V2Xverse_apknob/results/
  results_driving_grid_g{code}_r{route}_n{rep}/
  v2x_final/town05_short_collab/*/ego_vehicle_0/results.json
```

## 2. Phase A Experiment

Scope:

- latency: `650, 700, 750ms`
- AP/drop:
  - drop 0.25 -> AP50 0.783
  - drop 0.50 -> AP50 0.559
  - drop 0.85 -> AP50 0.281
- routes: clean6 `[3,17,18,104,136,317]`
- N: 6 per route, i.e. 36 episodes/cell
- total: `9 x 36 = 324 episodes`
- GPUs used: H800 GPU1-5

Completion:

| Total | Completed | Failed | TIMEOUT_SKIP | Missing |
|---:|---:|---:|---:|---:|
| 324 | 174 | 141 | 9 | 0 |

Phase A DS:

| AP50 \ latency ms | 650 | 700 | 750 |
|---:|---:|---:|---:|
| 0.783 | 29.4 | 28.6 | 25.7 |
| 0.559 | 27.5 | 17.7 | 22.6 |
| 0.281 | 28.7 | 27.9 | 29.2 |

Phase A vehicle collision episode rate:

| AP50 \ latency ms | 650 | 700 | 750 |
|---:|---:|---:|---:|
| 0.783 | 27.8 | 40.0 | 30.6 |
| 0.559 | 45.7 | 53.1 | 54.3 |
| 0.281 | 38.9 | 34.3 | 42.9 |

## 3. Complete Measured DS Table

Honest DS: `score_composed`; `TIMEOUT_SKIP` or missing score is counted as `0`.

| AP50 \ latency ms | 0 | 100 | 200 | 300 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.841 | 85.3 | 91.7 | 84.7 | 68.1 | 67.2 | 64.2 | 73.3 | 81.6 | 77.9 | 28.0 | 28.2 | 30.2 | 26.6 |
| 0.783 | 68.5 | 77.8 | 83.3 | 67.8 | 66.9 | 63.1 | 65.1 | 61.9 | 65.9 | 29.4 | 28.6 | 25.7 | 37.2 |
| 0.559 | 56.4 | 72.2 | 75.0 | 68.6 | 68.5 | 63.4 | 56.0 | 54.9 | 76.8 | 27.5 | 17.7 | 22.6 | 23.1 |
| 0.360 | 63.8 | 75.0 | 69.4 | 58.3 | 54.6 | 48.1 | 64.3 | 58.6 | 58.5 | 26.9 | 31.3 | 29.9 | 27.5 |
| 0.281 | 66.7 | 75.0 | 60.0 | 62.5 | 62.5 | 72.9 | 82.6 | 85.5 | 81.3 | 28.7 | 27.9 | 29.2 | 25.6 |

## 4. Complete Collision Episode-Rate Table

This is the safety map metric. It uses the earlier interaction-analysis convention: non-timeout episodes only; timeout rate is separately recorded in `ds_ap_latency_all_measured.csv`.

| AP50 \ latency ms | 0 | 100 | 200 | 300 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.841 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 5.6 | 0.0 | 0.0 | 0.0 | 15.6 | 13.3 | 25.8 | 25.0 |
| 0.783 | 0.0 | 0.0 | 0.0 | 11.8 | 16.7 | 23.5 | 35.3 | 27.8 | 22.2 | 27.8 | 40.0 | 30.6 | 47.1 |
| 0.559 | 5.6 | 0.0 | 0.0 | 27.8 | 16.7 | 5.6 | 5.6 | 11.1 | 5.6 | 45.7 | 53.1 | 54.3 | 33.3 |
| 0.360 | 0.0 | 0.0 | 0.0 | 0.0 | 5.6 | 0.0 | 5.6 | 5.6 | 5.6 | 37.9 | 25.8 | 36.4 | 38.7 |
| 0.281 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 5.6 | 0.0 | 0.0 | 5.6 | 38.9 | 34.3 | 42.9 | 31.2 |

## 5. Main Reading

1. **DS map**: the DS cliff is confirmed across the full AP grid. 600ms is still mostly high; 650ms and later are mostly low.
2. **Collision map**: the safety/collision signal is stronger than composed DS for AP degradation. Mid/low AP has high collision rates in the cliff band, especially AP0.559 at 650-750ms.
3. **Do not mix the claims**:
   - Use `ds_ap_latency_map_v2.png` for overall closed-loop DS prediction.
   - Use `ap_tau_collision_map_v2.png` for safety/collision cliff discussion.
4. The exact cliff location remains bounded by the 50ms frame quantization: current data supports "between 600 and 650ms", not a sub-frame threshold.

## 6. Query Function

```python
from ds_map_build_v2 import predict_ds

predict_ds(0.841, 600)  # 77.9
predict_ds(0.559, 700)  # 17.7
predict_ds(0.281, 750)  # 29.2
```

Run:

```bash
/home/jichengzhi/miniconda3/envs/v2xverse/bin/python \
  /home/jichengzhi/V2X/multi_agent/real_test/ds_map_build_v2.py
```

## 7. Remaining Caveats

- The DS map remains CoDriving/Town05/clean6/full-traffic `_1` only.
- RSU latency, `tau_ego`, other towns/routes, and other models are not included.
- `ap_tau_collision_map_v2.png` is vehicle-collision based; pedestrian/layout infractions are not the plotted collision metric.
- `ds_map_build.py` and `ds_ap_latency_map.png` are old 31-point artifacts; use v2 unless explicitly comparing history.
