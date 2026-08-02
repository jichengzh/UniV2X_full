# 计划一:把 Transformer/注意力架构纳入加速框架 (v1, 2026-06-22)

## §0 接手须知 (新窗口冷启动 —— 直接从这里开始, 自包含)

**这份计划是可执行交接, 不需要前序对话上下文。** 先读 §0, 再按 §1 Phase T0 起跑。

**前置状态 (已就绪)**:
- V2X-ViT e2e 瓶颈**已实测**: fusion_net(transformer)216.7ms=**92.4%** / conv backbone 仅 2.4%
  (`results/v2xvit_e2e_breakdown.json`, memory `project-v2xvit-attention-bottleneck`)。这是本计划的触发与依据。
- stage1→stage2 bridge **已构造并 4/4 验收**(`framework/stage1_bridge.py` +
  `framework/stage1/*` + `framework/partitions/*.yaml`; memory `project-stage1-bridge-construction`):
  bridge 的 B1/B2/D + 耦合分数 `κ` 框架本计划要复用到注意力上。
- e2e profiler 模板: `scripts/phase2/profile_v2xvit_e2e_breakdown.py`(CUDA-event 逐模块)。
- V2X-ViT ckpt(真): `${V2X_HOME}/heal_research/checkpoints/baselines_hf/`
  `HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26`(config.yaml + net_epoch_bestval_at17.pth)。
- 注意力模块代码: `${V2X_HOME}/heal_research/HEAL/opencood/models/fuse_modules/transformer_fuse.py`
  (`TransformerFusion`=若干 `EncodeLayer`(MultiheadAtt+FFN); 现 stage1 adapter **skip 了它**)。

**本计划第一步 = Phase T0(go/no-go gate, 1 GPU session, 无需新模型)**:
扩 `profile_v2xvit_e2e_breakdown.py`: 在 `fusion_net` 内部逐子模块(HMSA self-attn /
MSwin window-attn / window partition·reverse / FFN Linear / softmax·LN)挂 CUDA-event hook,
把 216ms 拆到算子级, 标记后端(TVM/TRT)难算子。输出 `results/v2xvit_fusion_oplevel_breakdown.json`。
**gate 判据见 §1 Phase T0** (热点是 GEMM→三轴都做 / 是 window-partition eager 低效→先做 S 轴融合 /
编译后已不瓶颈→退回 conv 框架)。**T0+T1(后端可行性实测)= 整个计划生死 gate, 不假设后端支持。**

**环境 + 纪律 (硬约束)**:
- 真仓库 `${V2X_ROOT}`(**绝不用** `${V2X_HOME}/UniV2X` 断链空壳)。
- conda python `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python`; HEAL: `sys.path` 加
  `${V2X_HOME}/heal_research/HEAL` + `os.chdir(HEAL_ROOT)`(见 profiler 模板)。
- 若需 TVM 试编译注意力: H800 `ssh -p 30001 -o ConnectTimeout=45 -o
  StrictHostKeyChecking=accept-new ${V2X_REMOTE_USER}@<PRIVATE_HOST>`; python `${V2X_DATA_ROOT}/tvm310/bin/python`;
  **GPU 只 4/5/6**(`nvidia-smi` 先确认 idle)。
- 区分 eager vs 编译口径(216ms 是 eager, 编译后量级待复测); **不轻信自报**必复跑;
  **仅在用户明确要求时 commit**。

**与计划二的接口**: 先做计划二 P0-P1(便宜、ckpt 在手)产出瓶颈谱, 用它圈定本计划适用集
(fusion-瓶颈模型才需要本计划); 见 `plan_multimodel_probing_v1.md`。

---

> 触发: V2X-ViT e2e 瓶颈实测 = transformer 融合 **92.4%** / conv backbone 仅 2.4%
> (`results/v2xvit_e2e_breakdown.json`)。框架现状(剪 conv 通道 + int8 conv + conv 调度)
> 优化的是那 2.4%, 对 attention-瓶颈模型基本无效。本计划把注意力纳入优化范围。

---

## 0. 核心判断 (先确立, 决定整个计划的形状)

**好消息: Transformer 的三轴与框架抽象同构 —— 不必另起炉灶。** V2X-ViT 的
`TransformerFusion` = 若干 `EncodeLayer`(MultiheadAtt + FFN(Linear1/Linear2)), 标准
transformer 算子。它天然有与 prune×quant×schedule 对应的三轴:

| 框架轴 | conv 上 (现状) | transformer 上 (本计划) |
|---|---|---|
| **P 剪枝** | 通道宽度 | **embed_dim / # heads / depth(层数)** —— QKV/FFN 的 Linear 投影维度、注意力头数、EncodeLayer 层数 |
| **Q 量化** | conv int8 (dp4a) | **Linear/matmul int8** (QKV·FFN GEMM; softmax/LN 保高精) —— 同样有 embed_dim 对齐 → 可建性悬崖 |
| **S 调度** | conv tile/快核 | **attention kernel fusion** (flash-attn 式融合 QK^T·softmax·V; window partition/reverse 融合) |

⇒ **bridge / 耦合分数 / 三臂消融 全部可复用**, 只需让 stage1 的图扫描"认得"注意力模块、
吐出它的剪枝组(embed/head/depth)、量化单元(Linear)、调度余量。这是本计划的主轴。

**坏消息 (诚实风险):**
1. **后端能否吃下注意力算子** 是硬 gate。conv 走 TVM dlight 已通; attention 的
   window-partition/softmax/MHA 在 TVM/TRT 上能否拿到真加速未知 —— 可能要 flash-attn
   式自定义 kernel(类比 Orin DCNv4 插件)。**必须先实测探明, 不能假设。**
2. **eager 216ms 有多少是实现低效 vs 固有计算量** 未分离。若大头是 eager window
   partition/python 开销, 编译后端可能直接收掉一大块(那 P/Q 反而次要)。

---

---

## §0.5 实验结果 (2026-06-23 已完成 T0 + T1-S)

> ★ 新窗口读到这里: T0/T1-S 已完成, 直接从 T1-P/T1-Q 起跑。

### T0 结果 (V2XTransformer sub-module breakdown)
- V2XTransformer = 204ms = 98.2% of outer wrapper
- 子模块: STTF≈5.9ms / HMSA×3≈9.3ms / **MSwin(PyramidWindowAttention)×3≈214ms** / FFN×3≈2.9ms
- 错误假设: 最初以为 HMSA `to_qkv` Python loop 是瓶颈 → 实测仅 9ms, 非主因
- 真热点: **MSwin×3 = 214ms = 92% of V2XTransformer**
- 数据: `results/t1s_alt_v2.json`

### T1-S 结果 (S轴实测 + bug fix)

**根因分析** (`scripts/phase2/t1s_alt_mswin.py`):
- MSwin内部精确计时: `to_qkv`=0.16ms / `rearrange`=0.25ms / `einsum q@k^T`=0.10ms /
  `pos_embedding add`=**82.9ms** (这一步!) / `softmax`=0.14ms / `einsum attn@v`=0.69ms
- **Bug**: `BaseWindowAttention.__init__` 用 `self.relative_indices = ...` (plain attribute),
  `model.to('cuda')` 不移动它 → `relative_indices` 留在 CPU
- 每次 forward: `pos_emb[relative_indices[:,:,0], relative_indices[:,:,1]]` 触发隐式
  CPU→GPU H2D 传输 + CUDA stream 同步 = ~60ms per BWA call (ws=16 的 512KB relative_indices)
- 9 个 BWA × ~60ms ≈ 540ms (3 层 × ws=16; ws=4/8 因 relative_indices 小影响不大)

**Fix** (`HEAL/opencood/models/sub_modules/mswin.py`, 一行):
```diff
- self.relative_indices = get_relative_distances(window_size) + window_size - 1
+ self.register_buffer('relative_indices', get_relative_distances(window_size) + window_size - 1)
```
已 apply。

**实测结果** (同一 synthetic inputs 对比):
| 配置 | mean | p50 | min |
|------|------|-----|-----|
| Buggy (rel_idx on CPU) | 279.9ms | 254.5ms | 212.5ms |
| Fixed (rel_idx on CUDA) | **30.9ms** | 30.6ms | 17.3ms |
| **Speedup** | **9.1×** | | |

ws=16 BWA: 90ms → 1.32ms (68×); 9 BWA total: ~10ms (vs ~270ms buggy)
数据: `results/t1s_mswin_bugfix_v1.json`

**T1-S gate**: **VIABLE ✓** — 9.1× from proper device placement. Proceed to T2.

**fix 后的新瓶颈分布** (30.9ms total):
STTF≈6ms / HMSA×3≈9ms / MSwin×3≈10ms / FFN×3≈3ms / 其他≈3ms

**重要说明**: 这是 HEAL 代码库 pre-existing bug, 非框架新引入。框架 S轴的"调度实现质量"
包含此类正确性修复。固定后, V2X-ViT transformer 对框架而言是可优化的 32ms 组件(非不可分析的 280ms)。

---

## 1. 分阶段计划 (每阶段一个 gate, 实测优先)

### Phase T0 — 注意力算子级 breakdown (gate: 值不值得做)
**目标**: 把 V2X-ViT fusion 的 216ms 拆到算子级, 定位真热点 + 标记后端难算子。
- 扩 `scripts/phase2/profile_v2xvit_e2e_breakdown.py`: 在 `fusion_net` 内部逐子模块
  (HMSA self-attn / MSwin window-attn / window partition·reverse / FFN Linear / softmax/LN)
  挂 CUDA-event hook, 给出各占比。
- **gate 判据**: ① 若 fusion 热点是 GEMM(QKV/FFN/MHA matmul) → P/Q/S 三轴都有戏, 全速做;
  ② 若热点是 window partition/python/小 kernel(eager 低效) → 先做"编译/算子融合"(S 轴),
  P/Q 次要; ③ 若编译后 fusion 已不再是瓶颈 → 退回 conv 框架即可, 本计划降级。
- 产物: `results/v2xvit_fusion_oplevel_breakdown.json`。**无需新模型, 1 个 GPU session。**

### Phase T1 — 注意力优化轴定义 + 后端可行性实测 (gate: 后端能否真加速)
**目标**: 确定哪些轴框架抽象直接subsume、哪些需新增; 实测后端对注意力的真加速。
- **P 轴**: 形式化 transformer 剪枝组 —— (a) embed_dim(QKV/FFN Linear 的通道, 与 conv 通道
  同构, DepGraph 可追) (b) # heads(结构化, 整头剪除) (c) depth(整 EncodeLayer 剪除)。
  确认 DepGraph 能否追 Linear/MHA 依赖(QKV 投影共享 embed → 联合剪)。
- **Q 轴**: Linear int8 的 embed_dim 对齐 / 可建性 —— 复用 `int8_buildable_align` 概念
  (MHA 的 GEMM 也有 tensor-core K-dim 对齐悬崖)。**实测**: 把一个 EncodeLayer 的 Linear
  导 TVM/TRT int8, 测真加速 + 对齐悬崖(类比 q_int8_dp4a_pairs.csv 做 attention 版)。
- **S 轴 (最关键 gate)**: **实测** flash-attn 式融合 / TVM attention schedule 能否对
  V2X-ViT 的 MHA 拿到真加速。先试现成路径(torch SDPA/FlashAttention、TRT attention plugin、
  TVM MetaSchedule on MHA); 拿不到再评估自定义 kernel 成本。
- 产物: `results/attention_axis_feasibility_v1.{json,md}` + 一张"轴×后端×真加速"表。

### Phase T2 — stage1 扩到注意力 (扩 adapter + graph_scan)
**目标**: 让 manifest 自动吐出注意力的 B1/B2/D + 耦合分数(bridge 复用)。
- `framework/stage1/adapters.py`: 给 V2X-ViT adapter 增加 **fusion-trace 模式**(现状是
  skip fusion)。挑战: fusion forward 需 multi-agent 输入 + com_mask(`depgraph_v2xvit.py`
  注明"不可 trace") → 要么造 dummy multi-agent 输入、要么 trace 单 EncodeLayer 子图。
- `framework/stage1/graph_scan.py`: 识别 Linear/MHA → 产 transformer 剪枝组(embed/head/depth)
  + Linear 量化单元 + 注意力 `int8_buildable_align`(embed/head 对齐)。
- **复用点**: `stage1_bridge.py` 的 `KnobSpec.coupling_score` 直接算注意力旋钮的耦合
  (cliff: embed int8 对齐悬崖; schedule_headroom: attention 调优余量, 由 T1 实测定标)。
- 产物: V2X-ViT manifest 含 fusion 旋钮; `bridge` 自检显示 fusion 旋钮的耦合分数。

### Phase T3 — 注意力代价模型 (real-measured LUT + AP)
**目标**: 给搜索器喂注意力配置的真测延迟 LUT + AP, 不是估算。
- 选一组 (embed_dim, n_heads, depth, Linear_bits, attn_schedule) 配置, **逐配置真测**
  latency(编译后端) + finetune 后 AP(DAIR/OPV2V)。规模控制: 先稀疏网格定可行域。
- 产物: `results/latency_lut_v2xvit_fusion.json` + `ap_model_v2xvit_fusion.json`。

### Phase T4 — 接入搜索 + 端到端真加速验证
**目标**: 把注意力轴并入三臂搜索, 给出 V2X-ViT **e2e** 真加速(非 subnet)。
- 搜索器 `build_from_manifest` 已能读 fusion 旋钮 → 三臂消融在"conv+attention"联合空间跑。
- **e2e 口径**: 复用 `backbone_to_e2e_latency` 的 Amdahl 框架, 但 fusion 现在是可优化项
  (不再进固定开销) → e2e 加速应显著大于 conv-only(因为优化了 92% 的部分)。
- 产物: V2X-ViT e2e Pareto(prune×quant×schedule on conv+attention) + 与 conv-only 对比。

---

## 2. 与计划二的耦合 (重要)
计划二的 P0 瓶颈探测会判出哪些模型是 **fusion-瓶颈**(V2VNet fusion 28.6ms / V2X-ViT)
vs **conv-瓶颈**(F-Cooper 1.4ms / AttFuse 3.5ms)。**本计划只对 fusion-瓶颈模型必要**;
conv-瓶颈模型现状框架已够。⇒ 先做计划二 P0(便宜、快), 用它圈定本计划的适用集。

## 3. 落地次序建议
T0(1 GPU session, 立即可做) → T1 后端可行性(决定整个计划生死) → 若 T1 通过再 T2-T4。
**T0+T1 是 go/no-go gate**: 若后端拿不到注意力真加速且自定义 kernel 成本过高, 则本计划
转为"诚实限定 scope(框架适用于 conv-瓶颈协同感知)" + 把 attention 列为 future work。

## 4. 诚实边界
- 不假设后端支持; 一切"能加速 X×"必须 T1/T3 实测。
- eager 216ms 的绝对值在编译后会变, 瓶颈位置(fusion 92%)确定但量级待编译后复测。
- 框架抽象同构是设计判断, T2 trace 能否真打通(multi-agent 输入/com_mask)是工程风险。
