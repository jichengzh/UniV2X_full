# Stage2 dataset_v2 实测数据总表清单 v1

日期: 2026-06-25

源文件: `multi_agent/data/dataset_v2.csv`

本文以 `dataset_v2.csv` 为准, 重新整理已有 50+ 实测数据点。上一版 `HANDOFF_stage2_pyramid_real_points_inventory_v1_zh.md` 只按零散结果文件盘点, 不覆盖本总表。

## 1. 总体规模

| 项 | 数量 |
|---|---:|
| 数据行 | 66 |
| 字段列 | 93 |
| `is_real_measured=True` | 66 |
| `ap_valid=True` | 63 |
| 有 `lat_p50_ms` | 63 |
| 有 `energy_per_frame_mj` | 60 |
| 有 `engine_size_mb` | 51 |
| 有 `mATE/mASE/mAOE` | 30 |

模型/硬件分布:

| 维度 | 分布 |
|---|---|
| model_class | `pyramid_fusion`: 62; `v2x_vit`: 4 |
| hardware | `rtx4090`: 58; `orin_agx`: 8 |
| latency_kind | `body_subnet_collab2`: 41; `engine_board_energy`: 13; `body_subnet_collab2_orin`: 6; `NA_pending_trt_build`: 3; `dla_pipeline_e2e_single_frame`: 2; `forward_hook_pytorch_fp32`: 1 |
| regime | `front`: 32; `ablation_perstage_dominated`: 18; `ablation_guardrail_off`: 9; `ap_reference_pending_trt`: 4; `ablation_throughput_saturation`: 3 |

## 2. 按 dataset_src 分组

| dataset_src | n | 主要用途 | model | hardware | latency_kind | AP 有效 | energy 行 |
|---|---:|---|---|---|---|---:|---:|
| `complete_points_v1` | 7 | 4090 Pyramid 主前沿完整点 | pyramid | rtx4090 | body_subnet_collab2 | 7 | 7 |
| `perstage_AP_v2` | 22 | 分 stage mixed precision 消融 | pyramid | rtx4090 | body_subnet_collab2 | 22 | 22 |
| `E4_energy_v1` | 13 | 4090 board energy / batch 1-2 能耗锚 | pyramid | rtx4090 | engine_board_energy | 13 | 13 |
| `E6_orin_p03` | 6 | Orin body latency/energy 跨平台点 | pyramid | orin_agx | body_subnet_collab2_orin | 3 | 6 |
| `P0_1_p25_trap` | 2 | p25 trap / forced INT8 对照 | pyramid | rtx4090 | body_subnet_collab2 | 2 | 2 |
| `P0_2_136` | 2 | p50b2_136 宽度对照 | pyramid | rtx4090 | body_subnet_collab2 | 2 | 2 |
| `pathA_forced_int8` | 3 | 极端剪枝 forced INT8 guardrail | pyramid | rtx4090 | body_subnet_collab2 | 3 | 3 |
| `pathB_head_int8` | 2 | head-only INT8 guardrail | pyramid | rtx4090 | body_subnet_collab2 | 2 | 2 |
| `P12_collab2_throughput` | 3 | 多 request/batch 吞吐饱和消融 | pyramid | rtx4090 | body_subnet_collab2 | 3 | 3 |
| `E3_orin_dla_pipe_v1` | 2 | Orin DLA pipeline e2e single frame | pyramid | orin_agx | dla_pipeline_e2e_single_frame | 2 | 0 |
| `v2xvit_A1A2_ap` | 3 | V2X-ViT AP reference | v2x_vit | rtx4090 | NA_pending_trt_build | 3 | 0 |
| `v2xvit_dair_real_timing` | 1 | V2X-ViT PyTorch hook timing | v2x_vit | rtx4090 | forward_hook_pytorch_fp32 | 1 | 0 |

## 3. 主前沿可用点

`regime=front` 共 32 行。它们不是同一个 latency/energy 口径, 使用时必须按 `latency_kind` 和 `hardware` 分层。

### 3.1 4090 body_subnet_collab2 主前沿

这些行是当前最适合 Stage2 先接入 registry 的 `Pyramid / RTX4090 / body_subnet_collab2` 主前沿候选。

| config_id | label | planes | precision | lat ms | AP50 | AP70 | energy mJ | engine MB | source |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `pyr_64-128-256_FP16` | rtx4090_base_fp16 | 64/128/256 | FP16/FP16/FP16 | 1.2715 | 0.7910 | 0.6309 | 462.4518 | 12.651 | complete_points_v1 |
| `pyr_64-128-256_INT8` | rtx4090_base_int8 | 64/128/256 | INT8/INT8/INT8 | 0.8113 | 0.7905 | 0.6228 | 261.0653 | 8.642 | complete_points_v1 |
| `pyr_48-96-192_FP16` | rtx4090_sens_prune_25 | 48/96/192 | FP16/FP16/FP16 | 2.9051 | 0.7769 | 0.5905 | 1065.2781 | 13.495 | complete_points_v1 |
| `pyr_32-64-128_FP16` | rtx4090_sens_prune_50 | 32/64/128 | FP16/FP16/FP16 | 1.0138 | 0.7644 | 0.5641 | 433.5189 | 7.551 | complete_points_v1 |
| `pyr_16-32-64_FP16` | rtx4090_sens_prune_75 | 16/32/64 | FP16/FP16/FP16 | 0.7681 | 0.7567 | 0.5300 | 314.4557 | 5.907 | complete_points_v1 |
| `pyr_32-64-128_INT8` | rtx4090_pareto_p50_int8 | 32/64/128 | INT8/INT8/INT8 | 0.7956 | 0.7522 | 0.5542 | 287.0130 | 4.966 | complete_points_v1 |
| `pyr_16-32-64_III` | global_int8_automix | 16/32/64 | INT8/INT8/INT8 | 0.6124 | 0.7537 | 0.5236 | 202.6822 | 4.441 | perstage_AP_v2 |
| `pyr_48-96-192_III_auto` | p25_int8_automix | 48/96/192 | INT8/INT8/INT8 | 2.7331 | 0.7760 | 0.5841 | 877.5584 | 11.517 | P0_1_p25_trap |
| `pyr_32-64-136_FP16` | p50b2_136_fp16 | 32/64/136 | FP16/FP16/FP16 | 1.0825 | 0.7877 | 0.6425 | 399.1710 | 8.003 | P0_2_136 |
| `pyr_32-64-136_INT8` | p50b2_136_int8 | 32/64/136 | INT8/INT8/INT8 | 0.8077 | 0.7879 | 0.6389 | 267.4260 | 5.210 | P0_2_136 |

### 3.2 4090 engine_board_energy 点

`E4_energy_v1` 有 13 行, 适合做能耗/功率轴验证, 但它的 `latency_kind=engine_board_energy`, 不应和 `body_subnet_collab2` 直接混为同一 latency 前沿。

重点结论:

- base FP32/FP16/INT8 batch1: 2.2999 ms / 0.8202 ms / 0.4936 ms; energy 1015.457 / 291.1643 / 141.0203 mJ。
- p50 FP16/INT8 batch1: 0.7086 ms / 0.5161 ms; energy 265.7654 / 164.1475 mJ。
- p75 FP16/INT8 batch2: 0.7537 ms / 0.5908 ms; energy 279.7822 / 196.8765 mJ。

### 3.3 Orin 点

Orin 有 8 行:

- `E3_orin_dla_pipe_v1`: 2 行 FP16 DLA pipeline e2e, 有 AP reuse, 无 energy。
- `E6_orin_p03`: 6 行 body latency/energy, 其中 FP16 三行 AP 有效; INT8 三行 `ap_valid=False`, 不可用于 AP claim。

Orin 绝对 energy 不能和 4090 board energy 直接比较, 只能按本机 regime 内比较。

## 4. 消融/guardrail 子集

### 4.1 perstage mixed precision

`perstage_AP_v2` 共 22 行, 覆盖 base / p50 / p75 三个 width 档的 per-stage precision 组合。主要用途是证明 mixed precision 多数被支配, 不是主前沿默认候选。

分布:

- base 64/128/256: 7 行。
- p50 32/64/128: 7 行。
- p75 16/32/64: 8 行, 其中 `global_int8_automix` 标为 `front`。

### 4.2 极端/保护性消融

- `pathA_forced_int8`: 3 行, 包含 90%/95% 极端剪枝 forced INT8, 用于 guardrail。
- `pathB_head_int8`: 2 行, head-only INT8。
- `P12_collab2_throughput`: 3 行, request=2/4/8, 证明吞吐饱和行为; 不适合作单帧 latency 主轴。

## 5. V2X-ViT 行

V2X-ViT 共有 4 行:

| config_id | 用途 | lat ms | AP50 | AP70 | 状态 |
|---|---|---:|---:|---:|---|
| `v2xvit_64-128-256_FP32` | AP reference | n/a | 0.710326 | 0.521162 | TRT latency pending |
| `v2xvit_32-64-128_FP32` | AP reference | n/a | 0.716701 | 0.533597 | TRT latency pending |
| `v2xvit_64-32-64_FP32` | AP reference | n/a | 0.726099 | 0.544528 | TRT latency pending |
| `v2xvit_64-128-256_FP32_hook_timing` | PyTorch hook timing | 56.609 | 0.710326 | 0.521162 | forward_hook_pytorch_fp32 |

这些行不能直接与 Pyramid TensorRT/Orin 前沿混合; 只能作为 V2X-ViT 侧 AP/latency 参考。

## 6. 数据治理注意项

1. `config_id=pyr_16-32-64_III` 出现两次:
   - `global_int8_automix`, `lat=0.6124`, `energy=202.6822`, `regime=front`;
   - `c_all_int8_forced`, `lat=0.6963`, `energy=193.6273`, `regime=ablation_guardrail_off`。
   进入 registry 前必须重新生成唯一 `config_id` 或用 `(config_id, config_label, regime)` 复合键。

2. `pyr_16-32-64_INT8` 位于 `complete_points_v1`, 但 `engine_path=models/stage_a_cache/pruned75_fp16.engine`, `mean_power_w=417.8`, `energy_per_frame_mj=314.4557` 与 p75 FP16 行一致。该行 notes 写明 latency 来自 Q-LUT、AP 来自 q_int8_ap。若要做严格 energy claim, 优先用 `perstage_AP_v2/global_int8_automix` 或 `E4_energy_v1/T_prune75_INT8_b2`。

3. 三条 Orin INT8 行 `ap_valid=False`, 不允许输出 AP claim:
   - `pyr_64-128-256_INT8_orin_E6`
   - `pyr_32-64-128_INT8_orin_E6`
   - `pyr_16-32-64_INT8_orin_E6`

4. `latency_kind` 不能混用:
   - `body_subnet_collab2`: 4090 collab2 body subnet。
   - `engine_board_energy`: 4090 energy bench engine latency。
   - `body_subnet_collab2_orin`: Orin body subnet。
   - `dla_pipeline_e2e_single_frame`: Orin DLA pipeline e2e single frame。
   - `forward_hook_pytorch_fp32`: V2X-ViT PyTorch hook timing。

5. `is_real_measured=True` 不等于同一口径可合并。Stage2 cost model ingest 必须同时检查 `hardware`, `latency_kind`, `regime`, `ap_valid`, `ap_reuse_basis`, `source`。

## 7. 对 Stage2 LUT/registry 的接入建议

优先级:

1. 先把 `complete_points_v1` 中 AP/latency/energy/size 口径清楚的 rows 转为 historical 4090 TensorRT evidence。
2. 把 `perstage_AP_v2` 作为 ablation evidence, 不默认进入主前沿, 只在 Q/mixed precision 分析使用。
3. 把 `E4_energy_v1` 单独作为 energy validation/evidence pool, 不与 `body_subnet_collab2` latency 直接混成一个前沿。
4. Orin 行进入 cross-hardware evidence, 但 INT8 行只可作 latency/energy, 不可作 AP。
5. V2X-ViT 行保留在 AP reference / pending latency 分区, 等真实 TRT latency 后再升级。

与 H800 产品化 LUT 的关系:

- `dataset_v2.csv` 是 4090/Orin 历史实测总表, 可作为 Stage2 cost model 的历史锚点和校验基线。
- H800 TVM 新 LUT 仍应从 `generate_*_lut` 产物进入 registry, 不应把 dataset_v2 的 4090/Orin latency 改写成 H800 evidence。
