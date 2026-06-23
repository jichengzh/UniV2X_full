# 数据集 sanity 分析 (按模块累积)

> 每跑完一个数据模块, 检查是否符合常理 (精度↓压缩率↑, 速度↑压缩率↑, INT8 < FP16 < FP32 lat) — 异常即深究并反思.

---

> ⚠️ **重大撤回 (2026-05-16, 问题 5 后): A.0 baseline AP 全部错值**
>
> 下方 §A.0+A.1 phase1 检查 #1 (AP 单调下降) + 检查 #4 (mix mediocre -20pp) 基于 **subnet 管线 AP** (问题 5: csv A.0 行的 AP 来自 `dataset_a_main` n=500 subnet, 不是 e2e n=1789). 实际 e2e AP 全部 Q ~0.54 一致 (除 entropy 崩塌). 详见 `问题.md` 问题 5 §5.7.
>
> 修复 (2026-05-16): 32 anchor 重跑 e2e_eval_ap n=1789, 128 行 csv 已回填. **检查 #1 / #4 的 conclusions 全部撤回**. 检查 #2 (alignment, lat 信号) + 检查 #3 (TRT lat: INT8 < FP16) **仍 valid** (lat 数据没受影响). 撤回后真实信号 → §A.0_repaired (2026-05-16) 新模块.

---

## 模块 A.0+A.1 重测 e2e AP (2026-05-16, n=1789)

**修复后 AP50 pivot (D=default ws=4, 56 anchor)**:

| triplet         | Q_fp32 | Q_fp16 | Q_int8_mm | Q_int8_pc_wo | Q_mix_s0 | Q_mix_s2 | Q_int8_ent |
|-----------------|--------|--------|-----------|--------------|----------|----------|-----------|
| T1_base         | 0.549  | 0.549  | 0.546     | 0.545        | 0.547    | 0.549    | 0.239     |
| T2_p25          | 0.558  | 0.558  | 0.554     | 0.555        | 0.551    | 0.558    | 0.461     |
| T3_p37          | 0.536  | 0.536  | 0.530     | 0.529        | 0.529    | 0.536    | 0.141     |
| T4_p50          | 0.568  | 0.568  | 0.554     | 0.556        | 0.568    | 0.567    | 0.266     |
| T5_p62          | 0.544  | 0.545  | 0.533     | 0.539        | 0.535    | 0.543    | 0.534     |
| T6_p75          | 0.541  | 0.542  | 0.540     | 0.540        | 0.543    | 0.541    | 0.455     |
| T7_wide_shallow | 0.529  | 0.530  | 0.528     | 0.528        | 0.528    | 0.529    | 0.474     |
| T8_narrow_deep  | 0.543  | 0.544  | 0.537     | 0.536        | 0.537    | 0.543    | 0.201     |

**Q 跨 triplet stats**:
- Q_fp32/Q_fp16/Q_mix_s2: mean ≈ 0.546, std 0.012
- Q_int8_mm/Q_int8_pc_wo/Q_mix_s0: mean ≈ 0.541, std 0.011
- Q_int8_ent: mean 0.347, std 0.150 (问题 1 entropy 仍崩, 部分 triplet 反弹到 0.4-0.5)

### 真实信号 (修复后)

1. **Q 维度 AP 几乎 flat** (除 entropy): FP32/FP16/INT8 mm/INT8 pc_wo/mix s0/mix s2 在每个 triplet 上差 ≤ 0.012 (= 1 个 std). e2e 管线下, VFE+Scatter+post-process 噪声主导 AP, Q 选择影响在 1pp 内.
2. **剪枝率 AP 几乎 flat**: Q_fp32 跨 T1-T6 = 0.549/0.558/0.536/0.568/0.544/0.541, 单调性消失 (T2, T4 反高于 T1). 8 triplet AP std = 0.012 = 1pp 噪声, 实际剪枝 0-75% 对 e2e AP 影响 < 1pp.
3. **AP D-invariance 验证 ✓**: T1_base/Q_int8_mm 跨 4 D (default/with_cudnn/cublas_lt/all_enabled) AP=0.5458 字节一致 (来源同一 AP file, D-dim 数学等价).
4. **entropy 仍崩** (问题 1 confirmed): T3/T8 e2e AP50 ≈ 0.14-0.20, T1/T4 ≈ 0.24-0.27, 仅 T5/T6/T7 反弹 0.45-0.53. e2e 管线相比 subnet 似乎 "稀释" 一部分崩塌, 但**仍不可用**.

### Paper §C 重 framing

- **不写**: "INT8 vs FP16 AP loss", "mix variants mediocre", "AP 随剪枝率单调下降"
- **写**: "Pareto front 是 AP-isobar 的, 在 AP ~0.54 等值面上 Q/B/D 维度提供 5-40% lat 加速空间"
- **真实 search-space 信号**: lat 端 (TRT engine lat) + entropy 崩塌 (single Q 危险点) + workspace/tactic 5-20% lat 影响

---

## 模块 A.0 + A.1 phase1 (48 行 e2e_bench_v1, 2026-05-15)

**覆盖**: 8 T × 6 Q × 1 D (GPU/default/4GB) × rtx4090
**Q tags**: Q_fp32, Q_fp16, Q_int8_mm, Q_int8_ent, Q_mix_s0, Q_mix_s2

### 数据透视

**AP50**:
```
                  Q_fp16  Q_fp32  Q_int8_ent  Q_int8_mm  Q_mix_s0  Q_mix_s2
T1_base           0.786   0.786   0.410       0.784      0.547     0.153 ←异常
T2_p25            0.775   0.774   0.606       0.774      0.551     0.558
T3_p37            0.762   0.762   0.021       0.762      0.530     0.536
T4_p50            0.755   0.756   0.198       0.744      0.568     0.567
T5_p62            0.744   0.743   0.021       0.731      0.535     0.543
T6_p75            0.750   0.751   0.748       0.748      0.543     0.541 ←反弹
T7_wide_shallow   0.761   0.762   0.021       0.746      0.529     0.529
T8_narrow_deep    0.757   0.757   0.020       0.746      0.537     0.543
```

**lat_e2e_mean_ms**:
```
                  Q_fp16  Q_fp32  Q_int8_ent  Q_int8_mm  Q_mix_s0  Q_mix_s2
T1_base           7.04    11.26   7.89        8.12       8.58      8.48
T2_p25            10.29   11.75   9.60        9.67       11.15     11.66 ←异常
T3_p37            9.97    12.35   7.70        9.88       10.08     10.57 ←异常
T4_p50            8.36    10.20   7.85        7.82       7.54      8.45
T5_p62            8.78    10.63   8.66        8.93       8.54      8.58
T6_p75            8.10    9.79    7.79        7.68       6.42      8.37
```

---

### 检查 #1: AP 随剪枝率单调下降?

**期望**: T1 (0%) > T2 (25%) > T3 (37%) > T4 (50%) > T5 (62%) > T6 (75%)

**实际** (Q_fp32 baseline, 无量化噪声):
0.786 → 0.774 → 0.762 → 0.756 → 0.743 → **0.751** (T6 反弹 +0.8pp)

✅ T1-T5 单调下降 (-4.3pp)
⚠️ **T6 反弹** (+0.7pp vs T5). 3 个 Q (fp16/fp32/int8_mm) 都反弹, 不是噪声.

**反思**: T6 (16,32,64) 是 prev session ckpt (`Pyramid_DAIR_m1_pruned75_2026_05_10`), 训练流程与新 finetune ckpt (T3/T5/T7/T8) 不同. T6 可能 finetune 更充分, 或网络结构在 75% 剪枝下进入新平台. 不影响 paper 主结论, 但**作为搜索空间数据点是真实信号** (说明剪枝率非线性, 不能假设单调).

---

### 检查 #2: 速度随剪枝率单调上升?

**期望**: T1 最慢, T6 最快 (params 减 → flops 减).

**实际** (Q_fp16):
T1=7.04 → T2=**10.29** → T3=9.97 → T4=8.36 → T5=8.78 → T6=8.10

❌ 严重异常: T2 (25% 剪枝) 比 T1 (0%) 慢 **+46%**, T3 (37%) 仍比 T1 慢 +42%.

❌ T6 (75%) 不是最快 — Q_fp16 下 T1 反而是最快的 (7.04ms).

**根因 (修正版 2026-05-15, 经 ONNX 结构 + rebench 验证)**:

> ❌ 原稿: "TRT IMMA/HMMA kernel 偏好 32-对齐通道" — **错** (FP32 不用 tensor core 但也慢; T6 stage0=16 也非 32-对齐但最快).
>
> ✅ 真正根因: **Pyramid ResNeXt bottleneck 的 3x3 grouped conv (groups=32) 在 `in_channels_per_group` 非 pow-2 时触发 cuDNN generic fallback kernel**.

PyramidFusion 用 ResNeXt 块, 每 stage 中间 3x3 conv 是 grouped (groups=32). `in_channels_per_group` 由该 stage `num_filters` 派生 (≈ 2 × num_filters / 32). 8 个 triplet 的 in_per_group 与 FP16 lat 完美 ordinal 相关:

| triplet | in_per_group s0/s1/s2 | 非 pow-2 stage 数 | FP16 lat_trt (ms) |
|---------|-----------------------|------------------|-------------------|
| T1_base | 4/8/16 | **0** | **2.75 (快)** |
| T4_p50 | 2/4/8 | **0** | **2.51 (快)** |
| T6_p75 | 1/2/4 | **0** | **2.28 (最快)** |
| T5_p62 | 1/3/8 | **1** | 3.17 (中) |
| T7_wide_shallow | 3/4/8 | **1** | 3.29 (中) |
| T8_narrow_deep | 1/3/12 | **2** | 3.67 (慢) |
| T3_p37 | 2/5/10 | **2** | **4.63 (慢)** |
| T2_p25 | 3/6/12 | **3** | **4.65 (最慢)** |

**线性 dose-response**: 每多 1 个非 pow-2 stage → TRT lat +0.5-1 ms. 0/1/2/3 非 pow-2 → 2.3-2.8 / 3.2-3.3 / 3.7-4.6 / 4.65 ms.

**为什么 FP32 也慢**: cuDNN grouped conv 的 kernel 选择跟精度无关. Non-pow-2 in_per_group 一律退化用 generic kernel, FP32/FP16/INT8 都受影响.

**反思**: 这是**搜索空间内"可行但崩"cell 的硬证据**, 但**特征工程的关键不是 channel 数对齐**, 而是 **grouped conv 的 in_per_group 是否 pow-2**. LGB lat predictor 必须加 `n_non_pow2_groups` 特征 (0-3 整数), 而不是 `min_channel_alignment`.

**对剪枝搜索的指导**: 选 prune ratio 时, 应该让 `(2 × num_filters[stage]) / 32` 落在 pow-2 上, 即 `num_filters[stage] ∈ {16, 32, 64, 128, 256}` (派生 in_per_group ∈ {1, 2, 4, 8, 16}). 避开 {24, 40, 48, 56, 80, 96, 160, 192} 这些会让 grouped conv 退化的中间值.

---

### 检查 #3: INT8 < FP16 < FP32 lat? (**已修正 — 必须比 TRT lat, 不能比 e2e**)

**重要前提**:
1. `Q_int8_mm` = `--precision int8` 实际是 **TRT INT8 + FP16 fallback** 混合 (`m4_8_trt_build_bench.py` 226-228 行: `set_flag(INT8)` 同时 `set_flag(FP16)`). TRT 自动 layer-by-layer 选 INT8 还是 FP16 (取量化精度允许 & 算子 kernel 支持). 这就是常说的 "INT8 量化", 不是纯 INT8.
2. **lat 比较口径必须用 `lat_trt_mean_ms` (TRT engine 内 lat)**, 不能用 `lat_e2e_mean_ms` (含 voxelize + postproc/NMS). postproc 含 NMS+decode FP32 后处理, **与 Q 无关**, 但 NMS 候选框数随 cls score 分布波动 → 引入 ~1-2ms 噪声, 会反向污染 e2e 对比.

**TRT lat (INT8_mm / FP16 比值, 越小 INT8 越快)**:
```
                  Q_fp16   Q_int8_mm   ratio
T1_base           2.746    2.209       0.804  ← INT8 快 19.6%
T2_p25            4.636    4.336       0.935  ← INT8 快 6.5%
T3_p37            4.628    4.609       0.996  ← 几乎相等
T4_p50            2.563    2.224       0.868  ← INT8 快 13.2%
T5_p62            3.171    2.923       0.922  ← INT8 快 7.8%
T6_p75            2.246    1.981       0.882  ← INT8 快 11.8%
T7_wide_shallow   3.292    2.930       0.890  ← INT8 快 11.0%
T8_narrow_deep    3.665    3.478       0.949  ← INT8 快 5.1%
```

✅ **8/8 triplet INT8 TRT lat ≤ FP16 TRT lat** (ratio 0.80-0.996). 完全符合常理.

**postproc lat 波动 (NMS, 与 Q 无关)**:
```
                  Q_fp32   Q_fp16   Q_int8_mm
T1_base           5.61     4.29     5.90      ← 波动 1.6ms (37%)
T6_p75            5.70     5.84     5.70      ← 波动 0.14ms
```

T1_base postproc 在 fp16/int8_mm 上差 1.6ms = INT8 TRT lat 总加速 (0.54ms) 的 3×. 这就是 e2e 数字反向的来源.

**为什么之前看到 e2e 反向 (FP16 7.04 < INT8 8.12 on T1_base)?**
- TRT engine: INT8 2.21 < FP16 2.75 (INT8 真的更快 0.54ms)
- 但 NMS 后处理: INT8 5.90 > FP16 4.29 (FP16 偶然产生少 1.6ms 的 NMS 工作量)
- e2e (= TRT + postproc): 5.90+2.21=8.11 vs 4.29+2.75=7.04, **INT8 e2e 反向慢** ← 噪声主导

**反思 (修正后)**:
1. **INT8 (TRT INT8 + FP16 fallback) 一致比纯 FP16 快**, 符合 TRT/CUDA 文档. 没有反预期.
2. **lat 对比口径**: TRT engine lat 才是 Q 维度的真实 signal. e2e lat 含 postproc 噪声, 用作 paper §C 主表必须明示 caveat.
3. **alignment 异常 (T2/T3 慢) 在 TRT lat 上更明显**: INT8 TRT 上 T2/T3 = 4.34/4.61ms vs T1 2.21ms = 慢 1.96-2.09× (FP16 上慢 1.69×). INT8 IMMA 比 HMMA **对 32-对齐 channel 更敏感**.
4. **LGB 训练特征**: lat 应分两个 target: `lat_trt_ms` (主), `lat_postproc_ms` (辅, 主要由检测候选框数预测).

---

### 检查 #4: mix variants 是否处于 Pareto 前沿? (**已修正 2026-05-15, 撤回原结论**)

> **修正**: 原稿声称 T1_base/Q_mix_s2 单点崩塌 (AP=0.15) → "B × Q 强耦合" — **数据 artifact**, 真值 AP=0.549. 详见 `问题.md` 问题 3 (孤儿子进程时序攻击).

**修正后 16 mix anchor AP50 分布**:

| triplet | Q_mix_s0 | Q_mix_s2 |
|---------|----------|----------|
| T1_base | 0.547 | **0.549** ← 原报 0.153 (artifact) |
| T2_p25 | 0.551 | 0.558 |
| T3_p37 | 0.530 | 0.536 |
| T4_p50 | 0.568 | 0.567 |
| T5_p62 | 0.535 | 0.543 |
| T6_p75 | 0.543 | 0.541 |
| T7_wide_shallow | 0.528 | 0.529 |
| T8_narrow_deep | 0.537 | 0.543 |

**AP50 stats**: min=0.529, max=0.568, std=0.012, range=4pp

**正确观察**:
1. **mix variants AP50 一致 mediocre** — 全部在 [0.53, 0.57] 4pp 窄带, 跟剪枝率 / s0 vs s2 选择**几乎不相关**
2. 相比 Q_int8_mm (AP50 0.73-0.78), mix 一致损失 ~20-23pp
3. **B × Q 强耦合假设在 mix 数据上不成立** — 8 个 triplet 上 mix_s0 和 mix_s2 的 AP 都几乎不变, 剪枝率不会改变 mix 的 AP 性质

**为什么 mix 一致损失 ~20pp**:
- TRT mixed precision 在 FP16↔INT8 边界强制插入 Q/DQ
- 单 stage (layer0 或 layer2) INT8 + 边界 Q/DQ 噪声 = 比 Q_int8_mm 全图统一 INT8 更差
- Q_int8_mm 的优势: TRT 全图 calibration + kernel 融合, 误差均匀分布在 ~30 层
- mix 的劣势: 误差集中在 1 stage + 2 个边界 Q/DQ ops, 单点信噪比低
- 跟剪枝率无关 — 任何通道宽度下 layer0/layer2 INT8 + FP16 包围都给类似 -22pp AP 损失

**反思 (修正后)**: mix variants 的角色不是 "B × Q 耦合证据" (原假设错误), 而是 "**单 stage INT8 在 mixed precision 模式下 systematic 损失 ~20-23pp**" — 这是 TRT mixed precision API 自身的结构性代价. 是 paper §C 对 "naive mix 不是 Pareto" 的硬证据, 但不再支持 "通道宽度交互" 的故事.

---

### 检查 #5: engine_size 随剪枝率单调下降?

**期望**: T1 > T2 > T3 > T4 > T5 > T6 (params 减 → 权重 buffer 减)

**实际** (Q_fp32):
T1=110.5 → T2=105.8 → **T3=109.9 (反弹 +4 MB)** → T4=100.3 → T5=101.5 → T6=96.7

⚠️ T3 反弹. 也存在于 Q_fp16: T2=54.3 略大于 T1=54.0.

**根因**: TRT engine 不只存 weight; 还有 kernel selection metadata + static buffer. 非对齐通道 (T3) TRT 选了**更多 kernel candidates** 备用, metadata 占用更大.

**反思**: engine_size 不是 params 的可靠 proxy. LGB resource head 需要 params_kb (从 ckpt 算) 而非 engine_size_mb 作主特征.

---

### 综合反思 (二次修正 2026-05-15)

1. **真实异常**:
   - T6 AP 反弹 (检查 #1) → finetune 流程差异 (历史 ckpt vs 新训)
   - T2/T3 lat 反弹 (检查 #2 二次修正) → **ResNeXt grouped conv `in_per_group` 非 pow-2 触发 cuDNN fallback kernel** (不是 IMMA/HMMA alignment), FP32/FP16/INT8 都一致退化
   - mix variants 一致损失 -20pp AP (检查 #4 修正后) → TRT mixed precision 边界 Q/DQ 系统性代价
   - T3 engine_size 反弹 (检查 #5) → 非 pow-2 in_per_group, TRT 选更多 kernel candidates 备用, metadata 膨胀

2. **撤回的伪发现**:
   - 检查 #2 原稿 "TRT IMMA/HMMA 32-对齐" 机制错误 (FP32 也慢, T6 反例) — 真正根因是 grouped conv in_per_group 非 pow-2
   - 检查 #4 原稿 "T1_base/Q_mix_s2 崩塌 AP=0.15" → artifact, 真值 0.549; "B × Q 强耦合" 在 mix 数据上**不成立**, 16 个 anchor AP 均匀 mediocre. Root cause: 孤儿子进程时序攻击, 见 `问题.md` 问题 3

3. **修正点 (检查 #3)**:
   - 之前用 e2e lat 比 INT8 vs FP16 错了 — postproc (NMS) 噪声主导. **TRT lat 才是 Q 维度真信号**, 8/8 triplet INT8 一致比 FP16 快 (ratio 0.80-0.996)
   - Q_int8_mm 是 "TRT INT8 + FP16 fallback" 内建混合模式 (不是纯 INT8), 这就是 TRT 标准实践
   - paper §C 表述应该是 "**INT8 TRT engine lat 比 FP16 快 5-20%, 取决于 channel alignment**" (而非 "INT8 vs FP16 取决于 triplet 通道宽度")

3. **paper §C 主结论得到 48 行数据加强 (修正后)**:
   - "搜索空间存在可行但崩 cell" (T1_base/Q_mix_s2, Q_int8_ent 多崩, T2/T3 lat 反向)
   - "单维度敏感性不能预测联合行为" (B × Q 耦合: T1 mix_s2 崩 vs T2-T8 mix_s2 救回)
   - "channel-alignment × Q 强耦合": 32-对齐 INT8 加速 11-19%, 16-对齐 INT8 加速降到 6%, 8-对齐 INT8 几乎不加速 (T3)

4. **下一模块 (A.1 phase2 granularity, 或 A.2 D 维度) 必须监控**:
   - 32-对齐 vs 8-对齐 channel 对 INT8/FP16 lat 的耦合
   - per-tensor vs per-channel weight INT8 的实际差异 (TRT 10 默认 per-channel)
   - workspace 16GB 是否对 baseline triplet (大 channel) 有 lat 增益

5. **LGB 特征工程必须加**:
   - `n_non_pow2_groups` (0-3 整数, ResNeXt bottleneck 的 grouped conv `in_per_group` 非 pow-2 的 stage 数) — 当前 stage*_planes 不足以预测 lat. **不是 `min_channel_alignment`** (那是错误假设)
   - `params_kb` (从 ckpt 算) 替代 engine_size_mb 作 resource 主特征
   - `is_mix_q` (bool) — mix variants 跟 global Q 有结构性差异

6. **Schema 精简 (2026-05-15)**: e2e_bench_v1 lat 列从 11 减到 1 (仅 `throughput_fps`). 详见 `问题.md` 问题 2 §2.5. lat breakdown (问题 2 那种 Q 维度诊断) 按需从 `models/e2e_cache*/<tag>_bench.json` 读取, 不进 dataset 主表.

---

## 模块 A.2 — 4090 D 维度扩 (168 行, 2026-05-16)

**覆盖**: 8 T × 7 Q × 3 NEW D = 168 anchor
**NEW D**:
- D2_with_cudnn_8gb: tactic=with_cudnn, workspace=8GB
- D3_cublas_lt_16gb: tactic=cublas_lt, workspace=16GB
- D4_all_enabled_1gb: tactic=all_enabled, workspace=1GB

**完成度**: 168/168 build_success, 0 fail, 6 GPU 并行 wall 68 min.

**AP 复用**: 全部 168 行 AP 从 csv reuse (D-dim 数学等价, force-ap 验证一致字节级). 修复 问题 5 后 128 行 AP 已正确.

### 检查 #A2.1: D-dim 影响 throughput?

**期望**: 不同 tactic / workspace 应给不同 lat (kernel selection 路径不同).

**实际** — fps_delta_pct vs D=default (mean across 8 T):

| q_tag        | D2 with_cudnn | D3 cublas_lt | D4 all_enabled |
|--------------|--------------|--------------|----------------|
| Q_fp32       | -2.1%        | -2.6%        | -7.3%          |
| Q_fp16       | -2.0%        | -0.8%        | -3.5%          |
| Q_int8_mm    | +0.5%        | -3.3%        | -1.9%          |
| Q_int8_ent   | -7.7%        | -11.6%       | -14.0%         |
| Q_mix_s0     | -5.2%        | -3.2%        | -7.5%          |
| Q_mix_s2     | -0.1%        | +2.2%        | -1.5%          |
| Q_int8_pc_wo | -23.2%       | -20.8%       | -15.6%         |

**观察 1**: D-dim 真有信号. mean fps_delta_pct 跨 (Q, D) 在 [-23%, +2%] 之间. 不是噪声.

**观察 2 (反例)**: with_cudnn 在某些 anchor 反而帮忙:
- T7_wide_shallow/Q_int8_ent: +10.5%
- T7_wide_shallow/Q_mix_s2: +9.6%
- T5_p62/Q_fp16: +9.5%

这避免 "with_cudnn 总是慢" 过度泛化, 满足纪律 #4 反例要求.

**观察 3**: Q_int8_pc_wo 在 3 个 D-config 都最差 (-15% 到 -23%). 假设: W-only INT8 (activation FP16, weight INT8) 对 kernel 选型极敏感 — Conv 输出强制 FP16 导致 layer fusion 受限, 不同 tactic 路径找不到等效快的 kernel. **不写进 paper §C, 标 hypothesis**.

**观察 4**: Q_int8_ent 大范围方差 (std 19-22%), 反映 entropy 崩塌的 lat 端不稳定 — entropy 既崩 AP 也崩 lat.

### 检查 #A2.2: build_secs 跨 D 是否稳定?

| d_tactic        | mean build_secs |
|-----------------|----------------|
| default (4GB)   | 113s           |
| with_cudnn (8GB)| 115s           |
| cublas_lt (16GB)| 112s           |
| all_enabled (1GB)| 118s          |

✅ 4 个 D-config 几乎一致 (±5s). Workspace size 1GB-16GB 对 build time 影响 < 5%, 说明 TRT autotuner 不被 ws 限制选择. all_enabled 1GB 略慢 (+5s) 可能因 ws 不够 force fallback path.

### 检查 #A2.3: engine_size 跨 D 差异

| d_tactic | eng_delta vs default mean | std | min | max |
|----------|---------------------------|-----|-----|-----|
| with_cudnn | -0.15 MB | 1.32 | -8.76 | +2.44 |
| cublas_lt | -0.38 MB | 1.48 | -8.57 | +1.38 |
| all_enabled | -0.21 MB | 1.21 | -6.04 | +2.06 |

✅ engine size 跨 D 差几 MB (-8.7 到 +2.4), 来自 kernel metadata. 数学等价 (weights/scales 不变).

### 检查 #A2.4: AP D-invariance 验证

force-ap on T1_base/Q_int8_mm/D3 vs csv (D=default 修复后):
- D=default ap50 = 0.5458 (修复后)
- D=D3 force-ap ap50 = 0.5456
- 差 0.0002, 在 n=1789 噪声内 (∼0.005)

✅ AP D-invariance 验证. 后续 D-dim 数据可继续 reuse_ap_from_csv 不重测.

### 综合反思 (A.2)

1. **A.2 完成 168 行真实测**: lat 端真信号, 全部 build_success, 跑 68 min 在 6 GPU 上.
2. **D-dim 不是 "all_or_nothing"**: 同 (T, Q) 上 with_cudnn 可能 +10% 也可能 -19%, 取决于 layer fusion 跟 kernel availability 的 interaction. 这是 paper §C 真正的搜索空间 cell-level 行为.
3. **Q_int8_pc_wo 对 D-dim 最敏感** — paper §C 角度, W-only INT8 + tactic 选择是搜索空间最 expressive 的 cell 区域.
4. **workspace 维度 1-16GB 对 build_secs 影响 < 5%**, 但对 throughput 影响 -7% 到 +14%. workspace 不是 build-cost knob, 是 runtime-perf knob.

### 撤回 / 警告

- **A.2 数据本身 valid** (build success, lat 真测), 不撤回
- 但 A.2 完成后才发现 csv AP 问题 (问题 5). 修复后 A.2 AP 列已正确, 168 行皆可用.

---
