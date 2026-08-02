# 8_7_14 交接文档 — Phase1/E6 Pyramid-LiDAR RSU Pareto（★v3 框架驱动正解，取代 v1/v2 手搓）

> 日期 2026-07-05。**★本 v3 取代 v1/v2**：v1/v2 是我手搓锚点 + 独立 surrogate（偏离），用户拍板纠正：**Phase 1 的目标 = 靠框架 SMBO 闭环迭代搜索出 Pareto 前沿点**。本 v3 记录用既存框架 `scripts/stage2_smbo_loop_v1.py` 跑完三精度 SMBO 收敛 + 装配交付的正确路径。

## 0. 纠偏（接手先读）

- **错**：v1/v2 我手工建 8 对角锚点 + 跑 `phase1_surrogate_search_1029.py` 独立 surrogate。
- **对**：框架 **已存在** SMBO 闭环 `scripts/stage2_smbo_loop_v1.py`（"之前 3 次迭代"就是它）。Phase 1 = **正确运行它**：`select`（重训 surrogate→预测未测→feasibility gate→出 top-k 候选队列）→ top-k 真测（`framework/measure_config.py` H800 lat/energy）→ `feedback`（pred vs actual 校准+重训+Pareto advance）→ 迭代到收敛。
- H800 实验路径见 `7_7_14 §3.4`（本会话补入）。

## 1. 三精度 SMBO 全收敛（本会话产出）

LOOP_DIR = `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/smbo_loop/`

| 精度 | 轮次 | 收敛 lat-min | convergence 文件 |
|------|------|------|------|
| fp32 | round1-3 | 9.83ms | `convergence_fp32.json` converged=True |
| fp16 (WMMA) | round1-3 | **1.63ms** | `convergence_fp16.json` converged=True（round2/3 无 advance，MAPE 0.67→0.03） |
| int8_tc (MatmulInt8Tensorization) | round1-3 | **1.34ms** | `convergence_int8.json` converged=True（round2 advance 1.34，round3 无） |

- 每轮 select 出 K=4-6 候选，`measure_config.py --width --precision --gpu` 在 **H800 GPU3/4 串行独占**真测（复用 original60 tuned DB，~70s/点），feedback 重训 LightGBM cost model。
- **精度序正确**：int8_tc 1.34 < fp16 1.63 < fp32 9.83ms。

## 2. ★int8 口径修正（关键坑）

- 既存 int8 surrogate 训练表（`cost_model/train/original60_training_table_latest.json` 60 int8 行）**已是 int8_tc**（`measured_int8_tc`，张量化，公平）。`native_int8_full_onnx_latency_rows_v1.jsonl` 只是 3 宽度旧 artifact。
- **我 round1 误用 `--precision int8`（native SIMT，慢 2× 不公平）**；改 `--precision int8_tc`（`measure_config` 已有此路径，走 `measure_config_fp16(rewrite_dtype="int8")` + `--cast-fp16-source`，tensorcore_gate=True）后正确。验证：我 int8_tc 测值与表逐点完全一致（[24,64,128] 2.353=2.3531）。
- 结论：int8 轴不用重建语料，只需 round 候选按 int8_tc 测。见 [[project-precision-axis-double-unfairness]]。

## 3. 交付（全 3 轴真测）

- `results/phase1_pyramid_pareto.{json,csv}` + `results/figure/`+`multi_agent/figure/fig_phase1_pyramid_pareto.png`
- 装配脚本 `scripts/phase1/phase1_assemble_smbo_pareto.py`

**AP-real Pareto 前沿（8 点 = 4 对角锚 × {fp16, int8_tc}，全在前沿）**：

| width | 精度 | lat(ms) | energy(J) | AP70 |
|------|------|------|------|------|
| [16,32,64] | **int8_tc** | **1.343** | **0.383** | 0.5236 ← lat/E-min |
| [16,32,64] | fp16 | 1.625 | 0.483 | 0.5300 |
| [32,64,128] | int8_tc | 3.589 | 1.020 | 0.5542 |
| [32,64,128] | fp16 | 4.400 | 1.246 | 0.5641 |
| [48,96,192] | int8_tc | 5.050 | 1.430 | 0.5841 |
| [48,96,192] | fp16 | 5.329 | 1.571 | 0.5905 |
| [64,128,256] | int8_tc | 6.104 | 1.771 | 0.6228 |
| [64,128,256] | **fp16** | **6.523** | 1.925 | **0.6309** ← AP-max |

- **3 极值全 gold-real AP**：AP-max=[64,128,256]fp16、lat-min=energy-min=[16,32,64]int8_tc。
- **口径**：lat/energy = H800 TVM `measure_config`（input_hw=[128,256]，fp16 WMMA/int8_tc 张量化，gate=True，串行独占）；AP70 = gold `stage_a_ap_real`（DAIR val 1789）。

## 4. ★AP 轴已闭合（finetune verdict，本会话完成）

- **决定性 finetune（gold 协议，H800 GPU3/4）**：固定 w0=16 变 neck 测真 AP70：
  - [16,64,256]（中 neck）= **0.52**、[16,128,256]（大 neck）= **0.53**
  - vs gold [16,32,64]（小 neck）= 0.5300、[64,128,256]（满 w0=64）= 0.6309
- **verdict：w0 主导 AP、neck(w1,w2) 不携带 AP**。固定 w0=16 neck 从 [32,64]→[128,256]，AP70 恒 ≈0.52-0.53（= 小 neck 0.53）；AP 只随 w0 涨。→ **off-diagonal minimal-neck 点 AP 与 [16,32,64] 同但延迟 ~2× 高 → 被严格支配 → 对角线是真前沿**（45 个 SMBO off-diagonal 探索点全被支配确认）。json `offdiagonal_ap_finetune_verdict`。
- **闭合**：AP 轴现真验证——对角线前沿=真前沿，off-diagonal 前沿成员资格已定（支配）。收尾 finetune 已做，非待办。
- 坑：train_ddp 必须从 HEAL cwd 跑（dataset/my_dair_v2x 相对路径）；AP eval 用 PyTorch inference.py（避 TRT 保 TVM 纪律，大信号对 eval 引擎不敏感）。

## 4b. （历史）原 GAP 描述

- SMBO 环内 **AP 是非功能占位 proxy**（`ap_pred=ap_median-0.02×width_penalty`，且训练表 ap70=0 → pred_ap70≈0）。**前沿实际由 lat/energy 驱动**，AP 由 gold 补。
- **4 对角锚有真 AP，但 45 个 SMBO off-diagonal 探索点（如 minimal-neck [16,32,256]）只有真 lat/energy，AP 待 finetune**（json `smbo_explored_offdiagonal_ap_pending`）。
- 闭合工具：`scripts/stage2_h800_ap_finetune_smoke_runner.py`（H800，gold 协议：structural_prune L1 wpg4 g32 → train_ddp half epoches31 → DAIR val1789 AP eval）。若要证 off-diagonal 点是否真在前沿（minimal-neck AP 是否掉），finetune 这些点即可——这也经验闭合"minimal-neck vs 对角线"问题。

## 5. 复跑命令

```
# select round N（本地，重训 surrogate 出候选队列）
python scripts/stage2_smbo_loop_v1.py --step select --precision {fp16|int8|fp32} --round N
# 真测候选（H800，measure_config；int8 必用 int8_tc）
ssh H800: python framework/measure_config.py --width w0,w1,w2 --precision {fp16|int8_tc} --gpu {3|4}
# feedback（本地，组装 measured.json[{width,lat_tuned_ms,lat_default_ms,energy_j}] 后）
python scripts/stage2_smbo_loop_v1.py --step feedback --precision {..} --round N --measured-json PATH
# 收敛检查
python scripts/stage2_smbo_convergence.py {fp16|int8|fp32}
# 装配交付
python scripts/phase1/phase1_assemble_smbo_pareto.py
```

相关记忆 [[project-phase1-pyramid-pareto-v1]] [[feedback-framework-on-tvm-not-trt]] [[project-precision-axis-double-unfairness]] [[feedback-tvm-tune-apply-fresh-workdir]]。
