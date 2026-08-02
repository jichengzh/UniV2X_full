# Stage2 Pyramid 已有真实数据点清单 v1

日期: 2026-06-25

本文只整理已经落盘的 Pyramid 真实测量点, 不新增或修改原始数据。口径上分为三类:

1. **4090 TensorRT 协同核完整锚点**: AP + latency + engine size 已合表; 其中 4 个 engine 点还能与 E5 energy 对齐。
2. **H800 TVM latency-only LUT**: 已有真实 H800 TVM backbone/dense-core latency, 但不含 AP/energy, 不能当完整三轴点。
3. **4090 Q/backbone LUT 与 E5 energy pool**: 可作为补 LUT 和对齐验证的候选池, 但不能跨硬件混用 latency 轴。

## 1. 已有完整锚点: M4.9 Pyramid/DAIR collab2

主表: `results/m4_9_framework_pareto.csv`

AP 来源: `results/m4_8_hybrid_ap_dair_*.json`, DAIR val `n=1789`。

Latency/engine 来源: `results/m4_8_dair_*collab_trt*.json`, 4090 TensorRT collab2 engine, 输入 `(2,64,128,256)`。

Energy 对齐来源: `results/E5_collab2_energy.csv`, 4090 GPU1, NVML, idle-exclusive。E5 已有交叉核验: `results/E5_energy_verification.md`。

| anchor | prune | precision | lat p50 ms | AP30 | AP50 | AP70 | engine MB | energy mJ/frame | energy source label | 状态 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| A0_pytorch_fp32_baseline | 0% | FP32 | 5.713 | 0.8327 | 0.7907 | 0.6314 | n/a | n/a | n/a | AP+lat 真测; 无 TRT engine/energy |
| A1_trt_fp16_collab | 0% | FP16 | 1.261 | 0.8331 | 0.7909 | 0.6308 | 12.48 | 462.4518 | rtx4090_base_fp16 | AP+lat+size+energy 可对齐 |
| A2_trt_int8_collab | 0% | INT8 | 0.808 | 0.8333 | 0.7910 | 0.6243 | 8.59 | 261.0653 | rtx4090_base_int8 | AP+lat+size+energy 可对齐 |
| A3_pruned50_ft_fp16_collab | 50% | FP16 | 1.009 | 0.8184 | 0.7644 | 0.5641 | 7.66 | 433.5189 | rtx4090_sens_prune_50 | AP+lat+size+energy 可对齐 |
| A4_pruned50_ft_int8_collab | 50% | INT8 | 0.776 | 0.8049 | 0.7530 | 0.5542 | 5.27 | 287.0130 | rtx4090_pareto_p50_int8 | AP+lat+size+energy 可对齐 |

结论:

- 若“完整”定义为 `AP + latency + model/engine size`, 当前 Pyramid 有 **5 个** M4.9 锚点。
- 若“完整”定义为 `AP + latency + energy + size`, 当前可稳妥合并的是 **4 个** TensorRT engine 锚点: A1/A2/A3/A4。A0 是 PyTorch baseline, 没有 engine energy 行。
- 这些点是 **4090 TensorRT collab2 口径**, 不能直接并入 H800 TVM latency 轴。

## 2. 已有 H800 TVM latency-only 数据

### 2.1 旧 H800 TVM LUT

文件: `results/latency_lut_pyramid.json`

口径: H800 TVM, Pyramid backbone/dense-core width grid, 单位 us。

概况:

- `widths` 共 **19** 个真实 latency rows。
- 每行包含 `num_filters`, `label`, `default_us`, `tuned_us`。
- 来源字段: `gap1_grid_corrected + lut_results_grid (relaunch); -1 rows excluded; s2_128 added 2026-06-21 seed=42 fresh workdir`。

代表点:

| label | num_filters | default us | tuned us |
|---|---|---:|---:|
| p75 | [16,32,64] | 12689.07 | 483.63 |
| p50 | [32,64,128] | 26241.67 | 3010.51 |
| trap25 | [48,96,192] | 42355.27 | 21614.80 |
| s0_32 | [32,128,256] | 44665.72 | 5982.32 |
| s1_64 | [64,64,256] | 46445.27 | 5894.56 |

注意: 这是 latency-only, 不能输出 AP 或 energy claim。

### 2.2 新产品化 smoke row

文件:

- `results/stage2/pyramid_lidar/evidence_real_smoke_20260625_154907/latency/latency_lut_rows_v1.jsonl`
- `results/stage2/pyramid_lidar/evidence_real_smoke_20260625_154907/REAL_SMOKE_VERIFY.json`

口径:

- `schema=latency_lut_row_v1`
- `backend=h800_tvm`
- `measurement_status=measured`
- `optimized_scope=backbone_only`
- `latency_p50_us=56129.53`
- H800 host: `zs-nj-tap-gpu18`
- GPU: H800 GPU3, preflight idle

注意: 该 row 是产品化链路 smoke, AP/energy 目录为空, 没有写 placeholder row。

## 3. 4090 Q/backbone LUT 与 energy pool

### 3.1 Q/backbone LUT

文件: `results/latency_lut_pyramid_q.json`

口径: RTX 4090 TensorRT backbone-only, batch=2, FP16/INT8, 单位 ms p50。

概况:

- 10 个 width 已测。
- 每个 width 有 FP16/INT8 latency; 部分带 AP70 e2e delta。
- 文件内明确警告: 不能与 `results/latency_lut_pyramid.json` 的 H800 TVM latency 混成同一 Pareto latency 轴。

代表点:

| label | width | FP16 ms | INT8 ms | INT8 speedup |
|---|---|---:|---:|---:|
| base | [64,128,256] | 1.2661 | 0.8026 | 1.577 |
| pruned25/trap25 | [48,96,192] | 2.9193 | 2.7279 | 1.070 |
| pruned50 | [32,64,128] | 1.0086 | 0.7875 | 1.281 |
| pruned75 | [16,32,64] | 0.7786 | 0.6390 | 1.218 |

### 3.2 E5 energy pool

文件: `results/E5_collab2_energy.csv`

口径: RTX 4090 collab2 engine, NVML board energy, idle-exclusive。

概况:

- 28 个 collab2 engine energy rows。
- 能耗有两套独立测量: `energy_per_frame_mj` 与 `energy_per_frame_mj_counter`。
- `results/E5_energy_verification.md` 记录 28/28 双路径吻合, 最大偏差 < 0.9%。

该表可用于给 4090 TensorRT collab2 锚点补 energy, 但不能给 H800 TVM row 补 energy。

## 4. 当前可直接进入 Stage2 registry 的建议

1. A1/A2/A3/A4 可作为 **historical/measured_4090_trt_collab2** 一组完整历史锚点, 字段含 AP/latency/energy/size。
2. `latency_lut_pyramid.json` 与新 smoke row 可作为 **measured_h800_tvm latency-only**, 但需要统一转成 `latency_lut_row_v1` schema 后再进 registry。
3. H800 energy 仍缺真实 telemetry row; 在补齐前不允许输出 H800 energy claim。
4. DS 映射若使用 AP/latency 预测组合, 仍按既定规则标为 report-only, 不当作闭环 DS claim。

## 5. 相关计划文档位置

- 产品化总计划: `multi_agent/methods/progress/HANDOFF_stage2_lut_productization_plan_v1_zh.md`
- 真实 smoke 与大规模生成计划: `multi_agent/methods/progress/HANDOFF_stage2_lut_real_smoke_and_generation_plan_v1_zh.md`
- 执行计划原稿: `docs/superpowers/plans/2026-06-25-stage2-lut-productization.md`
- cost model v2 handoff: `multi_agent/methods/progress/HANDOFF_stage2_gap_cost_model_v2_zh.md`
