# V2X-ViT 全网结构审计 v1

> 编制: sw-optimizer (主笔) · hw-optimizer (INT8/TRT 列协助待填)
> 时间: 2026-06-05
> 目的: 用户审阅用 — 剪枝/量化计划制定前的全网结构与瓶颈分析
> 数据来源标注: 【真测】= CUDA Event / hook 实测 · 【文档】= FLOPs yaml · 【估算】= 类比/公式推导

---

## 一、全模块分解表

### 1.1 顶层模块

**基础信息**:
- 模型: `HeterModelBaseline` (HEAL V2X-ViT DAIR, epoch17 bestval)
- 总参数: **13,453,197** (13.45M)
- DAIR val 1789 samples, 每场景固定 2 agents (V2I)
- e2e 实测: **61.86ms mean** (PyTorch FP32, CUDA Event walltime, warmup=20, measure=100) 【复用历史真测 2026-05-14, scripts/phase2/m4_9_v2x_baselines_timing.py】

| # | 模块 | 网络类型 | 参数量 | 占比 | 计算时间(ms) | 口径 | 可剪枝? | 可量化 INT8? |
|---|---|---|---|---|---|---|---|---|
| 1 | **encoder_m1** (PointPillar VFE+Scatter) | Linear+BN (稀疏) | **768** | 0.01% | **2.90** | 【复用历史真测】CUDA Event | ⚠️ 几乎无意义(参数极少) | ✅ 低风险 |
| 2 | **backbone_m1** (BaseBEVBackbone) | Conv2d+BN (3-stage) | **6,577,408** | 48.9% | **2.99** | 【复用历史真测】CUDA Event | ✅ DepGraph L1，已验证 | ✅ 低风险 (同 Pyramid conv) |
| 3 | **shrinker_m1** (double_conv stride=2) | Conv2d+BN | **1,475,072** | 11.0% | **0.89** | 【复用历史真测】CUDA Event | ✅ 随 backbone 联动 | ✅ 低风险 |
| 4 | **fusion_net** (V2XTransformer 3层) | Transformer (HMSA+MSwin) | **5,394,809** | 40.1% | **27.39** | 【复用历史真测】CUDA Event (p50 p99=111ms) | ⚠️ attention head 剪枝可行但需专用工具 | ❌ 高风险(见§2) |
| 5 | **cls/reg/dir heads** | Conv2d 1×1 | **5,140** | 0.04% | **0.06** | 【复用历史真测】CUDA Event | ❌ 无意义 | ✅ 低风险 |
| 6 | **postproc.NMS** (Shapely Python) | CPU 循环 O(N²) | — | — | **23.75** | 【复用历史真测】CUDA Event | N/A (非权重) | N/A |

> ★ 【复用历史真测】= 2026-05-14 实测, `scripts/phase2/m4_9_v2x_baselines_timing.py`, warmup=20, measure=100, RTX4090, PyTorch FP32。数值与 `model/v2xvit/分段耗时实测_v1.md` 完全一致（同批次测量）。

**时间小计**:
- forward (encoder+backbone+shrinker+fusion+heads): 34.24ms (55.4% e2e)
- postproc (NMS+decode+corners+range): 27.05ms (43.7% e2e)
- **fusion_net 占 forward 的 80.0%**

---

### 1.2 fusion_net 内部子模块分解

**架构**: `V2XTransformer → V2XTEncoder` (depth=3 layers, num_blocks=1 per layer)
- 每层包含: `prior_feed(Linear)` + STTF + V2XFusionBlock[HMSA + MSwin] + FFN+PreNorm
- DAIR: L=2 agents, `spatial_correction_matrix = torch.eye(4)` (identity, STTF no-op, com_mask 常量全1)

**参数分布** (CPU torch.load 真算):

| 子模块 | 每层参数 | 3层合计 | 占 fusion_net | 占总参 |
|---|---|---|---|---|
| prior_feed (Linear 259→256) | 66,560 | 66,560 | 1.23% | 0.49% |
| STTF | **0** | 0 | 0% | 0% |
| HGTCavAttention (HMSA) | 591,872 | **1,775,616** | 32.91% | 13.20% |
| PyramidWindowAttention (MSwin 3 scale) | 1,051,091 | **3,153,273** | 58.45% | 23.44% |
| FFN+PreNorm | 131,584 | **394,752** | 7.32% | 2.93% |
| **fusion_net total** | — | **5,394,809** | **100%** | **40.10%** |
| (残差未列子行) | — | **4,608** | — | ★ 3层×3个PreNorm各含 `LayerNorm(256)` weight+bias = 512 params×9 = 4,608；归属 PreNorm wrapper，非 HMSA/MSwin/FFN 本身 |

**计算时间** (内部子模块 hook profiling, GPU1, fp32_pytorch, hook CUDA Event, warmup=10+measure=50, DAIR val 实测) 【真测·hook】:

> ★ 表头基准 **27.7ms** = 本次 hook-run 的 `fusion_net_total`（`results/v2xvit_fusion_profile.json` `fusion_net_total_ms=27.69`）。§1.1 的 **27.39ms** = 历史测量 module-level CUDA Event（独立测次）。两者均真实，Δ0.3ms = run-to-run 噪声 + hook 测量开销，无矛盾。

| 子模块 | mean_ms | pct_fusion(27.7ms) | pct_e2e(61.9ms) | 备注 |
|---|---|---|---|---|
| STTF | **2.122** | 7.7% | 3.4% | DAIR identity matrix → no-op 变换，仍有 reshape/permute 开销 |
| prior_feed (Linear 259→256) | —(太快) | — | — | 未捕获，推算 <0.05ms |
| **L0 HGTCavAttn (HMSA)** | **3.351** | 12.1% | 5.4% | L0 BEV 分辨率最高 |
| **L0 MSwin PyramidWinAttn** | **4.796** | **17.3%** | **7.8%** | ★ fusion 内单层最重 (50×176 窗口分割) |
| L0 FFN | 0.219 | 0.8% | 0.4% | — |
| L1 HGTCavAttn (HMSA) | **3.336** | 12.0% | 5.4% | — |
| L1 MSwin PyramidWinAttn | **3.279** | 11.8% | 5.3% | — |
| L1 FFN | 0.204 | 0.7% | 0.3% | — |
| L2 HGTCavAttn (HMSA) | **3.201** | 11.6% | 5.2% | — |
| L2 MSwin PyramidWinAttn | **3.210** | 11.6% | 5.2% | — |
| L2 FFN | 0.198 | 0.7% | 0.3% | — |
| **3× HMSA 合计** | **9.89** | **35.7%** | 16.0% | 派生 |
| **3× MSwin 合计** | **11.29** | **40.8%** | 18.3% | 派生 ← **fusion 内最重** |
| 3× FFN 合计 | 0.62 | 2.2% | 1.0% | 派生 |
| regroup/PreNorm/other | ≈3.47 | ≈12.5% | — | 差值=27.39(§1.1)−hook子模块合计23.92；含 Regroup/spatial_correction/PreNorm wrapper 等 |

**关键发现**:
- **MSwin PyramidWindowAttention 是 fusion 内最重子模块**（40.8%），L0 尤甚（4.8ms，50×176 BEV 窗口分割计算最大）
- HMSA 各层均匀（~3.2-3.4ms/层），MSwin L0 > L1≈L2（BEV 分辨率随层不变，L0 窗口数最多）
- FFN 极轻（0.2ms/层），不是优化目标
- STTF 2.1ms：虽是几何 no-op，reshape/permute 开销真实存在

---

### 1.3 剪枝/量化详细分析

#### Backbone_m1 (BaseBEVBackbone)

| 特征 | 内容 |
|---|---|
| 架构 | 3-stage Conv2d+BN, strides=[2,2,2], deblocks upsample=[1,2,4] |
| 原始 num_filters | [64, 128, 256] |
| grouped conv | **无** (纯标准 Conv2d, 比 Pyramid ResNeXt 更简单) |
| DepGraph 兼容 | ✅ 已验证 (A-2 实测) |
| A-2 剪枝结果 | p50 -74.8%: actual [32,64,128]; p75 -91.4%: actual [64,32,64] |
| p75 stage0 未剪原因 | DepGraph L1全局重要性+round_to=32+输入端依赖约束，自动保护 stage0 |
| INT8 风险 | **低**: 标准 BN+Conv, 无 attention, 同 Pyramid backbone 行为 |

#### fusion_net (V2XTransformer)

| 子模块 | 网络类型 | INT8 风险 | 风险依据 |
|---|---|---|---|
| STTF | warp_affine (grid_sample) | **低** | 纯几何变换，无统计量化难点；DAIR identity=no-op |
| prior_feed (Linear) | Linear 259→256 | **低** | 小线性层 |
| HGTCavAttention (HMSA) | Multi-head cross-attn, heads=8, dim_head=32 | **高** | QK^T softmax 对精度敏感；QuantV2X V2X-ViT INT8/INT8 AP30 57.4→40.0(-30%); INT4/INT8 57.4→29.9(-48%) |
| PyramidWindowAttn (MSwin) | Multi-scale window attn, 3 scales [4,8,16] | **高** | 同上 + 多尺度 reshape/unfold; split_attn 结构复杂 |
| FFN+PreNorm | Linear+LayerNorm | **中** | LayerNorm 量化有损; FFN Linear 本身低风险 |
| **fusion_net 整体** | Transformer | **❌ 高** | QuantV2X PTQ INT8 崩溃已实证 |

---

## 二、瓶颈结论

### 2.1 时间瓶颈

```
e2e 61.86ms 分解 (【真测】CUDA Event + hook profiling):
├── NMS (CPU Shapely)        23.75ms  38.4%  ← 瓶颈①: P0 CUDA NMS 可消除 (~1.6× e2e)
├── fusion_net (transformer) 27.39ms  44.3%  ← 瓶颈②: forward 主导 (占 body 80%)
│   ├── STTF                  2.12ms   7.7%  (no-op变换 reshape开销)
│   ├── 3× MSwin PyramidAttn 11.29ms  40.8%  ← fusion 内最重: 多尺度窗口 attention
│   │   └─ L0: 4.80ms / L1: 3.28ms / L2: 3.21ms (L0最贵,BEV窗口分割)
│   ├── 3× HMSA (HGTCavAttn)  9.89ms  35.7%  (跨 agent attention)
│   │   └─ L0: 3.35ms / L1: 3.34ms / L2: 3.20ms (各层均匀)
│   ├── 3× FFN                 0.62ms   2.2%  (轻量,非瓶颈)
│   └── regroup/PreNorm/other  3.47ms  12.5%
├── encoder_m1 (VFE)          2.90ms   4.7%
├── backbone_m1 (conv 3-stage) 2.99ms   4.8%  (DepGraph 已证可剪,AP无损)
├── shrinker_m1                0.89ms   1.4%
└── heads+decode+corners+range 3.84ms   6.2%
```

**P0 CUDA NMS 优先级**: 类比 F-Cooper/AttFuse 实测 3.7-4.0× e2e 加速（但 V2X-ViT NMS 仅 38% vs 77%，P0 单独仅 ~1.6×，**必须联合 fusion 优化才能再赢**）

### 2.2 参数瓶颈

| 排序 | 模块 | 参数量 | 剪枝ROI |
|---|---|---|---|
| 1 | backbone_m1 | 6.58M (48.9%) | **低** (A-2: 剪 -91% AP 无损，DAIR 过参数化) |
| 2 | fusion_net | 5.39M (40.1%) | **高潜力** (尚未测，transformer 剪枝需专用工具) |
| 3 | shrinker_m1 | 1.48M (11.0%) | 随 backbone 联动 |
| 4 | 其他 | <6K (0.05%) | 忽略 |

### 2.3 精度敏感点

| 维度 | 发现 | 来源 |
|---|---|---|
| backbone 剪枝 | **免费** (A-2: p50/p75 AP70 ≥ base; mAOE SNR<5×) | A-2 真测 |
| backbone INT8 | **近免费** (类比 Pyramid INT8 -0.5 AP30) | 预测(待 A-3 验证) |
| fusion transformer INT8 | **高危** (PTQ: AP30 57.4→40.0, -30%; INT4: →29.9, -48%) | QuantV2X 文献 (arXiv:2509.03704 Table1, DAIR-V2X, PTQ) |
| fusion transformer 剪枝 | **未测** (attention head pruning 需专用工具) | 待规划 |

**交叉结论**:
- "动哪里最值": **fusion_net** — 既占 44% 时间又有 INT8 崩溃风险，是 ROI 最高的优化目标
- backbone 是"免费"优化（参数省但时间收益 < 10%，精度无损）
- **联合优化价值**: backbone(conv) INT8 + fusion FP16 = 护栏 ON，可拿约 conv 部分加速同时保精度；fusion INT8 = 护栏 OFF，会崩，正是框架"事前规避"的论据

---

## 三、已知实验事实附录

### 3.1 A-1/A-2 实验数据（本项目真测）

| 配置 | AP70 | mAOE | mAOE CI | n_tp | epoch_used | 口径 |
|---|---|---|---|---|---|---|
| A-1 base (V2X-ViT FP32) | 0.5212 | 0.0656 | [0.0645,0.0668] | 24498 | bestval_at17 | fp32_pytorch, DAIR val 1789 |
| A-2 p50 backbone (actual [32,64,128]) | 0.5336 | 0.0665 | [0.0654,0.0677] | 24753 | bestval_at17_finetuned | fp32_pytorch, DAIR val 1789 |
| A-2 p75 backbone (actual [64,32,64]) | 0.5445 | 0.0630 | [0.0620,0.0641] | 24920 | bestval_at16_finetuned | fp32_pytorch, DAIR val 1789 |

**A-2 判据与结论**:
- p50 mAOE Δ=+0.0009, SNR=0.4× (半宽比) → **噪声内**
- p75 mAOE Δ=−0.0026, CI 不重叠 → **可分辨反向改善，但受训练预算混淆(+25ep FT vs base 零额外)**
- **净结论**: 剪枝+FT 后无精度退化信号; 过参数化定论的跨模型重复
- ⚠️ **预算混淆 caveat**: 干净归因需 iso-budget 对照 (base +25ep 续训)，是否补测待用户决策

### 3.2 Pyramid 对照（已知事实，用于 corroboration）

| 配置 | AP50 | 来源 |
|---|---|---|
| Pyramid base | 0.791 | stage_a_ap_real.parquet (DAIR val 1789) |
| Pyramid p75 (backbone剪-75%) | 0.757 | 同上 |
| Pyramid INT8 (base) | 0.790 (AP50) | 同上, Δ=-0.001 |

→ Pyramid backbone 剪枝有弱信号 (AP50 0.034 span); V2X-ViT backbone 剪枝反而超 base = 两者均过参数化，V2X-ViT 更甚

### 3.3 QuantV2X 外部文献数据（已原文核验）

来源: arXiv:2509.03704 Table 1, DAIR-V2X, PTQ (Post-Training Quantization)

| 模型 | Bits(W/A) | AP30 | AP50 |
|---|---|---|---|
| Pyramid Fusion | 32/32 | 75.1 | 68.2 |
| Pyramid Fusion | 8/8 (INT8) | 74.6 | 67.8 |
| Pyramid Fusion | 4/8 (INT4W) | 74.2 | 66.7 |
| **V2X-ViT** | **32/32** | **57.4** | **49.5** |
| **V2X-ViT** | **8/8 (INT8)** | **40.0** | **11.0** |
| **V2X-ViT** | **4/8 (INT4W)** | **29.9** | **8.8** |

→ Pyramid INT8 仅 -0.5 AP30; V2X-ViT INT8 崩溃 -17.4 AP30 (-30%); INT4W 更甚 -27.5 AP30 (-48%)

---

## 四、TRT/INT8 分析 (hw-optimizer 协助)

> **来源**: `multi_agent/model/v2xvit_trt_risk_v1.md` (v1.4, 2026-06-05; ★[2026-06-05 迁移] 原在 methods/design/)
> **口径**: 全部为 **静态代码分析 + 文档推导**, 标注【预研·static】。GPU 真测数字见 §4.2。
> **授权状态**: A-3 GPU 实验尚未获 team-lead 授权执行, 本节为 pre-flight 分析。

---

### 4.1 各模块 TRT 可 build 性与 INT8 风险

| # | 模块 | ONNX export 可行性 | TRT build 风险 | INT8 风险档 | 来源 |
|---|---|---|---|---|---|
| 1 | **encoder_m1** (PointPillar VFE+Scatter) | ❌ **不在 export scope** (spconv 稀疏算子, 不可 ONNX) | N/A | N/A | 【预研·static】 |
| 2 | **backbone_m1** (Conv2d+BN 3-stage) | ✅ 标准 conv, ONNX-friendly | **低**: 标准 conv kernel, 无特殊算子 | **🟢 低**: BN+Conv, 同 Pyramid backbone; 类比 Pyramid INT8 AP30 -0.5 pt | 【预研·static】 |
| 3 | **shrinker_m1** (stride=2 Conv2d+BN) | ✅ 同上 | **低** | **🟢 低**: 同 backbone | 【预研·static】 |
| 4 | **fusion_net** (V2XTransformer, 3层) | ⚠️ **可行但需特殊处理** (见 §4.2) | **中-高**: 多操作数 einsum + Python 循环展开 | **🔴 高**: QuantV2X PTQ INT8 -30% (见 §3.3); 需 mixed-precision | 【预研·static】 |
| 5 | **cls/reg/dir heads** | ✅ 1×1 Conv, 标准 | **低** | **🟢 低** | 【预研·static】 |

**推荐 export scope**: 只 export `V2XTransformer` (fusion_net 子模块), 输入 `(1, 2, 50, 176, 259)`, 不含 encoder/backbone/NMS。

---

### 4.2 fusion_net ONNX Export 关键风险与缓解

#### 风险清单 (按严重度)

| 风险编号 | 描述 | DAIR (L=2 固定) 下影响 | 缓解方案 |
|---|---|---|---|
| **R1** 🔴 | `HGTCavAttention.to_qkv/to_out()`: `self.q_linears[types[b,i]]` — 数据依赖 dispatch | **可缓解**: DAIR types=[0,1] 是常量, dummy input 固化路径即可正确 trace | export 时 `dummy_x[..., -1]` 设 vehicle=0/infra=1 |
| **R2** 🔴 | Python `for b,i,j` 循环展开 → ONNX 图膨胀 | **可接受**: L=2, B=1 → 最多 4 iter; 总节点 ~950-1,000 | 固定 B=1, L=2; 展开量小 |
| **R3** 🔴 | INT8 precision collapse: LayerNorm + QK 乘积 + masked_fill(-inf) | **高危**: QuantV2X 实证 AP30 57.4→40.0 (-30%); INT4 →29.9 (-48%) | mixed-precision: Q/K/LayerNorm 保 FP16, V+FFN 可 INT8 |
| **R4** 🟡 | 多操作数 `torch.einsum` (3 operands, 非标准下标) | TRT 展开为 pairwise MatMul, 可接受 | `opset_version=17` + `onnx-simplifier` |
| **R5** 🟡 | `F.grid_sample` in STTF + `use_roi_mask=true` 路径 | **★ STTF 是 no-op** (见下); com_mask 是常量全 1 | 预计算常量绕过 (见 §4.3) |
| **R6** 🟡 | `tensor_split` in `Regroup()` | **不在 export scope** (V2XViTFusion wrapper 层) | export V2XTransformer, 不 export wrapper |

#### R3 三档 INT8 缓解方案

| 方案 | 描述 | 预期精度 | 预期速度 |
|---|---|---|---|
| **方案 A** (全局 INT8 auto) | `--int8 --fp16`, TRT-auto 选层 | AP30 可能 -10~30% (参考 QuantV2X) | 最快: ~6–10 ms (估算) |
| **方案 B** (mixed-precision, 推荐) | Q/K/LayerNorm 保 FP16; V proj + FFN INT8 | AP30 跌幅估算 < 10% (待验证) | ~8–12 ms (估算) |
| **方案 C** (fusion 全 FP16) | backbone+shrinker INT8, fusion_net FP16 | AP 近无损 (backbone INT8 类比 Pyramid -0.5 pt) | ~18–22 ms (估算) |

> ⚠️ 上述 latency 均为估算, 非真测; 真测在 A-3 获授权后执行。

---

### 4.3 STTF no-op + com_mask 常量优化

> 来源: hw-optimizer 代码分析 + sw-optimizer 独立核验 PASS (2026-06-05)

**发现**:

| 组件 | 分析结论 | 依据 |
|---|---|---|
| `STTF` | **no-op**: `V2XViTFusion` 硬编码 `SCM = torch.eye(4)` → identity 2D affine → `warp_affine(x, I)` = x 原样 | `fusion_in_one.py L367` + `torch_transformation_utils.py` 调用链追踪 |
| `com_mask` | **全 1 常量**: identity SCM → `get_rotated_roi` 产出全 1 → `combine_roi_cav_mask` = `cav_mask=[1,1]` | `get_roi_and_cav_mask` 调用链 + `permute(0,3,4,2,1)` → shape `(1,50,176,1,2)` |

**ONNX 图简化收益**:

| 组件 | 完整 trace 节点 | 绕过后节点 | 省去 |
|---|---|---|---|
| STTF (grid_sample 链) | ~25 节点 | 0 | ~25 节点 |
| get_roi_and_cav_mask | ~15 节点 | 0 (常量替换) | ~15 节点 |
| **合计** | **~40 节点** | **~0 节点** | **~40 有效节点 (~60 含 ONNX 插入)** |

**总 ONNX 节点估算**: 完整 trace ~960-1,000 节点; 简化方案 ~900-960 节点。

**推荐 export 策略** (简化方案):
```python
# 简化 wrapper: 注入预计算常量, 跳过 STTF + com_mask 计算
dummy_com_mask = torch.ones(1, 50, 176, 1, 2).cuda()  # (B,H,W,1,L) ★ shape 已核验
# STTF: identity → 直接注入 x 原样 (跳过 warp_affine)
```

---

### 4.4 TRT Build 时间预算

| Build 类型 | 估算时间 | 建议时段 | 备注 |
|---|---|---|---|
| **TRT FP16** | **3–8 分钟** | 任意空闲时段 | ~950-1,000 节点, 中等规模 |
| **TRT INT8 auto** | **15–30 分钟** | **夜间空闲时段** | 含 100 帧 forward 校准 + tactic 搜索 |
| L=5 (V2V OPV2V 扩展) | ~20–60 分钟 | 夜间 | L² 循环展开效应 (~2,500-3,500 节点) |

> **前置条件** (宪章 §2 MUST-4): GPU util=0% / mem ≤ 50MiB; 锁频 `sudo nvidia-smi -lgc 2520`; ISS-024 epoch locking 必须 `bestval_at17`。

---

### 4.5 A-3 真测待填 (GPU 实验, 获授权后执行)

| 项 | 内容 | 预期结果 | 状态 |
|---|---|---|---|
| TRT FP16 build | engine 路径 / build log / engine size | latency ~9–14 ms (估算) | **待 A-3 授权** |
| TRT INT8 auto build | int8_layer_count / calib cache | latency ~6–10 ms (估算); AP 风险 | **待 A-3 授权** |
| TRT INT8 mixed-prec | per-layer 精度覆盖 log | latency ~8–12 ms (估算) | **待 A-3 授权** |
| collab 口径 latency | p50/p99 ms, 口径 `v2xvit_fusion_subnet` | 见上 | **待 A-3 授权** |
| energy 测量 | J/frame, perf/watt (nvidia-smi 50ms 采样) | 参照 E4 口径 | **待 A-3 授权** |
| fusion INT8 AP eval | AP30/AP50/AP70 vs A-1 base(锚: AP70=0.5212/AP30=0.7854), 方向对比 QuantV2X 57.4→40.0 (不强行对齐绝对值) | 预期 AP30 下跌 -10~30% | **待 A-3 授权** |
| backbone+shrinker INT8 AP eval | AP30/AP50/AP70 (护栏 ON), 锚同 A-1 | 预期近无损 (类比 Pyramid -0.5 pt) | **待 A-3 授权** |
| 校准集规格 | 100 帧 DAIR val, calib cache 路径 | `calibration/v2xvit_calib.bin` | **待 A-3 授权** |

---

*§四 填写: hw-optimizer (2026-06-05), 来源 v2xvit_trt_risk_v1.md v1.4 (static analysis)*
*§四真测数据: 待 A-3 GPU 实验获 team-lead 授权后填入*

---

*文档版本: v1.3 (F1-F4 小修: §1.1 历史真测标注/§1.2 两次测量注/残差脚注/footer更新)*
*§1.2 profiling 已填入(源: results/v2xvit_fusion_profile.json, PID 3459205@GPU1, 2026-06-05)*
*下次更新: A-3 GPU 实验获 team-lead 授权后填入 §4.5 真测数据*
