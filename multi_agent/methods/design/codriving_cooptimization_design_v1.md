# CoDriving 软硬件协同优化设计 v1 — 软件侧 (剪枝 × 量化)

> 编制: sw-optimizer
> 时间: 2026-06-14
> 数据来源: 结构审计 `multi_agent/model/codriving_structure_audit_v1.md` + 已测教训 `CLAUDE.md §〇` + 方法清单 `dims_pruning_v1.md` / `dims_quantization_v1.md`
> 标注说明: 【预研·static】= 代码静态分析/论证, 未跑实验; 【实测·Pyramid】= 团队 Pyramid/DAIR 真测迁移类比; 【TBD-等CD-A1】= 依赖 CD-A1 延迟剖面才能填; 【真测·CUDA Event @4090】= CD-A1/A1b 已测数据
>
> ★[2026-06-15 更新] CD-A1 + CD-A1b 真测已完成, 原 TBD-等CD-A1 项可填入真测值; 见各章节更新。
>
> ⚠️ 口径红线 (贯穿全文): CoDriving 指标 = Driving Score / waypoint L1, 严禁与 HEAL DAIR AP 混表比较; 本文所有"对比 Pyramid"只比设计空间结构, 不比绝对数值。

---

## §1 设计空间对比表 (CoDriving vs Pyramid, 逐维度论证)

> 核心问题: CoDriving 是否比 Pyramid 有更大、更非退化的协同优化设计空间?

### 1.1 逐维度比较

| 维度 | Pyramid/DAIR 已测结论 | CoDriving 结构特征 | 退化风险 | 论证理由 |
|------|----------------------|--------------------|----------|---------|
| **B1 剪枝 — backbone 大小** | backbone 3.758M (pyramid_backbone 68.9%), 可达区剪 90% 近免损 (AP50 0.74–0.75), Pareto 退化 | backbone.resnet **6.984M** (感知网络 85.4%), 比 Pyramid 大 **1.86×**; **车端 ego 上被调 2 次(主路2.81ms+fusion内3.04ms=5.85ms, 单卡BN=2口径)→ego 侧剪枝 2× 杠杆; 路测 RSU 仅调 1 次=1× 杠杆** 【真测·CUDA Event @4090】 | 仍有风险退化 【预研·static】 | 退化根因 = DAIR 任务过参数化, 非模型大小; CoDriving 换 CARLA 任务 + 闭环 Driving Score, 传导链不同(见 §1.2); 更大 backbone 不一定更难退化, 关键在指标传导; **⚠️[2026-06-16勘误] 系统级非 e2e 1.4×**: 该数是单卡BN=2合计, 部署时 ego/RSU 分设备并行取 max, 系统加速待 CD-A1c 分设备 BN=1 真测 |
| **B1 剪枝 — 指标传导** | AP50 高原 (span 0.034), AP70 span 0.101, 剪枝信号弱 | 指标 = Driving Score 闭环, latency↓ 直接传导 | **退化风险低** 【预研·static】 | 见 §1.2 详细论证 — Driving Score 含轨迹跟随精度 (waypoint L1), backbone 特征质量直接影响 planner 输入; 预期 剪枝→特征质量↓→waypoint 偏差↑→DS↓ 传导链更锐 |
| **B1 剪枝 — 硬件对齐约束** | ResNeXt groups=32 → INT8 需 %32 对齐; 极小 num_filters 触发 wpg 陷阱 | ResNet BasicBlock (groups=1) → 无 group 约束, 通道剪后 %8 或 %32 皆可 | **约束更松** 【预研·static】 | BasicBlock 残差结构: conv1 输出 = conv2 输入 = 残差 shortcut, DepGraph 依赖更简单; 无 groups 参数, 不触发 wpg=0 崩溃 |
| **B2 量化 — 感知backbone** | INT8 AP 近无损 (AP50 0.791→0.790); 全网 INT8 lat 0.825→0.530 ms (1.56×) | backbone.resnet INT8 风险低 【预研·static】 | 低退化风险 | 标准 ResNet Conv+BN, 与 Pyramid Conv+BN 同类; 类比 Pyramid INT8 经验可迁移 |
| **B2 量化 — fusion** | Pyramid 用 conv 多尺度加权融合 (有参数), INT8 低风险 | AttenFusion **0 可学参数** (纯 bmm+softmax), softmax 精度敏感 | **中等风险** 【预研·static】 | softmax 输入动态范围与 V2X-ViT HMSA 不同(无 QKV projection, N 更小); 但无学习参数 = 无量化误差累积, 风险比 V2X-ViT HMSA 低; 与 Pyramid fusion 不同类 |
| **B2 量化 — planner** | Pyramid 无 planner | planner MLP decoder (0.93M, Linear ×3) + Conv2d + **Conv3D** | MLP 低, Conv3D 中 | MLP INT8 友好 (纯 Linear); Conv3D temporal kernel (3,1,1) 需 temporal 校准样本 (见 §3.3) |
| **双网络独立搜索** | 单网络 | 感知 + planner **分离 ckpt** | **组合维度增大** | 两网络可各自独立剪枝/量化, 联合搜索空间 = 感知配置 × planner 配置; 感知剪枝后 planner 输入特征质量下降, 存在跨网络耦合 (见 §4) |
| **per-stage 混精** | 被 TRT-auto 支配, AP 有正向点但 Pareto 无用 | AttenFusion softmax 唯一中风险点(纯注意力仅 0.81ms, 8% e2e 【真测·CUDA Event @4090】); DCN 确认未启用(yaml 无 dcn 字段, self.dcn=False, DCNNet 死代码); Conv3D temporal 中风险(非 INT8 硬瓶颈) | 框架结论不变 | 同 Pyramid: TRT-auto 以延迟驱动选层, 不做 AP 分析; AttenFusion 实际计算量很小, 即使保 FP16 代价也极低; AttenFusion softmax 是否需保 FP16 = TBD-等CD-A3 INT8 build 实测 |

### 1.2 指标传导质变论证 (核心论点)

**Pyramid/DAIR 退化根因 (已实测)**:
- 任务: DAIR-V2X 3D 检测, 输出 AP50/AP70
- 退化路径: backbone 剪枝 90% → 特征质量↓ → 但 AP50 仅 -0.04 (span 0.034), 因为 DAIR 目标稀疏、anchor-based head 对特征质量不敏感、基线 AP 本来就不高 (0.791)
- 定论: 模型对 DAIR 严重过参数化 → 任何合理剪枝 ≈ 免费

**CoDriving/CARLA 传导链分析 【预研·static】**:
- 感知输出: BEV feature `[B, 128, 96, 288]` → 输入 planner
- planner 指标: waypoint L1 (轨迹精度) → 驾驶 Score (碰撞率、路线完成度)
- 传导链: backbone 通道数↓ → BEV 特征分辨率下降 → planner conv 接收更低质特征 → waypoint L1↑ → Driving Score↓
- 关键差异: Driving Score 是**闭环积分指标** (连续 T=5 帧时序 + 规划一体), 比单帧 AP 对特征质量更敏感; Conv3D temporal 融合层对多帧特征一致性有高要求

**结论 【预研·static】**: CoDriving 指标传导链更锐, 预期剪枝会触发真实 trade-off 曲线而非 AP 高原。但这是架构性推断, 必须 CD-B1 剪枝 + CARLA finetune 实测验证 (见 §2.4 finetune 成本说明)。

---

## §2 剪枝维度设计

### 2.1 剪枝目标 (按重要性排序)

| 优先级 | 目标模块 | 参数量 | 占感知% | DepGraph 可剪? | 说明 |
|--------|---------|--------|---------|---------------|------|
| P0 | backbone.resnet (ResNet 3-stage) | 6.984M | 85.4% | ✅ BasicBlock 结构更简单 | **主剪枝目标; 车端 ego 上延迟 2× 杠杆**: ego 调 2 次(主路 2.81ms + fusion 内重跑 3.04ms = 5.85ms, 单卡BN=2口径); **路测 RSU 仅调 1 次=1× 杠杆**。**⚠️[2026-06-16勘误] 系统级非 e2e 1.4×**(那是单卡BN=2合计, 部署 ego/RSU 分设备并行取 max); 部署级加速待 CD-A1c 分设备 BN=1 真测 【真测·CUDA Event @4090, `results/CD_A1b_fusion_internal_4090.csv`】 |
| P1 | backbone.deblocks (FPN neck) | 0.599M | 7.3% | ✅ 随 backbone 联动 | 与 backbone 通道耦合 |
| P1 | shrink_conv (384→128) | 0.590M | 7.2% | ✅ 输入随 deblocks | neck 终端 |
| P2 | planner Conv2d 系列 | ~0.584M | — | ✅ 独立网络, 独立剪 | 见 §2.3 |
| N/A | fusion_net (AttenFusion) | 0M | 0% | N/A | 无可学参数; 纯注意力(warp+bmm+softmax)仅 0.81ms(8% e2e) 【真测·CUDA Event @4090】; 不是独立优化目标 — fusion 内的主要开销是 backbone 重跑 |
| N/A | cls_head / reg_head | ~3.5K | ~0% | 无意义 | 太小 |

### 2.2 剪枝档位设计 (感知 backbone)

**ResNetBEVBackbone 架构约束**:
- BasicBlock: 每 block = conv1(3×3) + BN + ReLU + conv2(3×3) + BN + shortcut
- 无 group convolution → 无 wpg 陷阱
- 通道对齐: INT8 路径 → %32; FP16 路径 → %8 即可

**num_filters = [64, 128, 256] 基线 → 推荐档位** 【预研·static】:

| 档 | num_filters | 等比缩放比 | 估算参数(backbone.resnet) | 对比 Pyramid 类比档 | INT8 对齐 (%32) |
|----|------------|-----------|--------------------------|--------------------|--------------------|
| base | [64, 128, 256] | 1.0× | 6.984M | base (3.758M) | ✅ 64/128/256 全 %32 |
| p25 | [48, 96, 192] | 0.75× | ~3.93M (估算) | — (无直接类比) | ✅ 48/96/192 全 %32 |
| p50 | [32, 64, 128] | 0.5× | ~1.75M (估算) | 类比 Pyramid p50 | ✅ 32/64/128 全 %32 |
| p75 | [16, 32, 64] | 0.25× | ~0.44M (估算) | 类比 Pyramid cliff2_c | ✅ 16/32/64 全 %32 |
| aggressive | [8, 16, 32] | 0.125× | ~0.11M (估算) | 超出 Pyramid 实测范围 | ✅ 8/16/32 全 %32 |

> 参数估算基于 ResNet BasicBlock 公式 [k×k×C_in×C_out×N_blocks], 含 shortcut projection; 精确数值待 CD-A1 实例化真算。【预研·static】

**关键预期**: 与 Pyramid 不同, 预期 p75/aggressive 档会真正踩到精度悬崖 (Driving Score 明显下降), 形成非退化 Pareto 曲线。这是设计假设, 需实测验证。

### 2.3 planner 独立剪枝档位设计

**planner 剪枝特殊性**:
- Conv 系列 (conv1_1/1_2/2_1/2_2/3_1): 标准 2D conv, DepGraph 可剪
- Conv3D temporal (conv3d_1/3d_2): 时序融合, 剪通道需谨慎 (temporal 维度固定 k=3)
- MLP decoder (Linear 384→1025→512→20): INT8 量化更友好, 建议量化优先于剪枝 (见 §3)

**planner 剪枝档位** 【预研·static】:

| 档 | 目标模块 | 剪枝率 | 预期参数 | 风险 |
|----|---------|--------|---------|------|
| plan_none | 全 planner | 0 | 1.662M (base) | — |
| plan_conv25 | Conv2d 系列 (非 Conv3D) | 25% | ~1.50M (估算) | 低 |
| plan_conv50 | Conv2d 系列 (非 Conv3D) | 50% | ~1.18M (估算) | 中 |
| plan_all25 | 全部 Conv (含 Conv3D) | 25% | ~1.43M (估算) | 中 (temporal 维度) |

> MLP decoder (0.93M) 建议不剪而量化 (见 §3.2 planner 量化设计)。

### 2.4 悬崖假设与验证路径

**假设 (待验证)** 【预研·static】:
- 感知 backbone 剪枝 → BEV 特征质量↓ → planner waypoint L1↑ → Driving Score↓
- 预期悬崖区间: p25~p50 附近 (较 Pyramid 早得多, 因指标传导链更锐)
- 验证方法: p25/p50/p75 × CARLA finetune + Town5 闭环评估 Driving Score

**finetune 成本说明 (比 Pyramid 高得多)**:
- Pyramid DAIR finetune: HEAL 单卡, 用 DAIR 2-agent 真实数据, ~10-20h (1-2 epoch)
- CoDriving CARLA finetune: 需 CARLA 仿真环境 + V2Xverse 数据生成/加载, 成本估算 **2-5× Pyramid finetune** 【预研·static】; 具体时长 TBD-等CD-A1 (需确认 V2Xverse train 数据是否预生成或需在线仿真)
- planner finetune: 感知固定后 finetune planner, 或两网络联合 finetune (成本更高); 推荐分阶段: 先感知剪枝+单独 finetune, 再 planner adapt

### 2.5 剪枝准则选择

与 Pyramid 一致, 推荐优先级: **L1 (无梯度, 最稳) > FPGM > Taylor (需梯度)**

CoDriving 特别说明:
- Conv3D 层 L1 按 output channel 求范数 (跨 temporal dim 归约), 与 2D 同理
- planner decoder MLP 的 Linear 层: L1 按输出神经元归约 (结构化剪枝语义 = 裁神经元)

---

## §3 量化维度设计

### 3.1 感知 backbone 量化设计

**风险档: 低** 【预研·static, 类比 Pyramid 实测】

| 模块 | 推荐精度 | 校准器 | 说明 |
|------|---------|--------|------|
| backbone.resnet | INT8 | minmax | 标准 ResNet Conv+BN, 类比 Pyramid backbone INT8 近无损 |
| backbone.deblocks | INT8 | minmax | ConvTranspose2d, TRT 内建支持 |
| shrink_conv | INT8 | minmax | Conv2d 1×1, 低风险 |
| cls_head / reg_head | **FP16** | — | 检测头 logit 量化噪声 → score 失真 → DS↓; 沿用 Pyramid 敏感 substr 策略 |

**通道对齐约束**: INT8 路径 → 剪枝后通道数必须 %32 (TRT Tensor Core tile); CoDriving BasicBlock 无 groups 约束, %32 更容易满足 (Pyramid 的 groups=32 限制不存在)。

### 3.2 fusion_net (AttenFusion) 量化设计

**风险档: 中等** 【预研·static】

AttenFusion 的关键运算:
```
score = bmm(q, k^T) / sqrt(C)   # [H*W, N, N], N = agent 数 (通常 2-5)
attn = softmax(score, dim=-1)    # 精度敏感点
context = bmm(attn, v)           # [H*W, N, C]
```

与 V2X-ViT HMSA 的 INT8 风险对比:
- V2X-ViT HMSA: 有 QKV Linear projection (大矩阵量化误差) + multi-head (多头 softmax) → INT8 崩 (QuantV2X 须逐层混精)
- CoDriving AttenFusion: **无 QKV projection** (直接用 BEV feature 做 Q/K/V), N 很小 (2-5 agent), bmm 规模 `[H*W, N, N]` 远小于 V2X-ViT → **INT8 风险低于 V2X-ViT**

**推荐策略** 【预研·static】:
1. 首选 TRT-auto INT8: 让 TRT builder 自由决定 AttenFusion 层精度 (延迟驱动)
2. 若 Driving Score 掉点 >X% (阈值 TBD-等CD-A3 实测), 则 AttenFusion 保 FP16
3. 不推荐强制 per-stage 混精 (与 Pyramid 教训一致: TRT-auto 已包含这类收益)

**softmax 量化注意事项**: 若 TRT 选 INT8 for softmax, score 矩阵动态范围取决于 `sqrt(C)` 归一化后的值; C=64/128/256 时 `sqrt(C)=8/11.3/16`, 归一化后 score 范围比 NLP Transformer 更紧 (BEV 空间相关性局部) → 风险相对低。

### 3.3 planner 量化设计

**MLP decoder 量化 (0.93M, 55.9% planner 参数)** — INT8 友好:

| 子模块 | 推荐精度 | 理由 |
|--------|---------|------|
| decoder.layers (Linear 384→1025→512→20) | INT8 | 纯 FC, INT8 GEMM 效率最高; 输出 waypoint 20维, logit 非边界敏感 |
| conv3_1, conv2_1/2_2, conv1_1/1_2 | INT8 | 标准 Conv2d, 低风险 |
| conv3d_1, conv3d_2 (Conv3D temporal) | **FP16 优先** 【预研·static】 | temporal kernel (3,1,1): INT8 calibration 需含 temporal 多帧样本 (单帧 dummy 校准不充分); TRT 原生支持 Conv3d INT8 但 calibration 质量依赖样本多样性; 建议先 FP16, 实测后再降 INT8 |
| target_encoder (Linear 2→128) | INT8 | 输入维度极小, 量化噪声可接受 |

**planner 量化 INT8 收益估算** 【预研·static + 真测占比】:
- 主参数 MLP decoder (0.93M) 全 INT8: 预期 lat↓ 30-50% for planner 模块 (Linear 操作 INT8 GEMM 理论 2× vs FP16)
- planner 延迟占总 e2e 比例 **已测**: planner 1.50ms / 感知 e2e 10.17ms → planner 占联合总 ~12.9% 【真测·CUDA Event @4090, `results/CD_A1_codriving_latency_4090.csv`】; planner MLP decoder 自身仅 0.087ms(planner 的 5.8%)→ MLP decoder INT8 全局收益极微; planner 量化的实际收益更多来自 Conv2d 系列(含 conv3_1 0.295M)

### 3.4 量化校准器选择

沿用 Pyramid 已验证策略:
- **主路径: minmax** (避免 entropy 校准器; Pyramid 实测 entropy 触发 AP 崩塌, 根因 = 自训 ckpt 长尾激活)
- CoDriving CARLA 仿真数据激活分布待测, 但 minmax 作为保守主路径先行

**校准样本要求**: 感知校准 = BEV scatter 输出 `[N, 64, 192, 576]` (pillar_vfe+scatter 不 export); planner 校准 = `[B, T=5, ...]` 多帧序列 (必须含时序维度以覆盖 Conv3D 激活范围)

---

## §4 双网络联合搜索空间定义

### 4.1 配置空间结构

CoDriving 特有: 感知网络 (P) 和 planner 网络 (L) 分离 ckpt, 可各自独立搜索。

**感知网络配置维度 (P 轴)**:

| 维度 | 取值 | 实验点数 |
|------|------|---------|
| backbone 剪枝率 | {0, 0.25, 0.5, 0.75} | 4 |
| backbone 量化精度 | {FP16, INT8} | 2 |
| fusion 量化精度 | {FP16, INT8 (TRT-auto)} | 2 |

感知配置子空间大小: 4 × 2 × 2 = **16 点** (但 prune75 + fusion INT8 可能 Driving Score 崩, 需剪枝)

**planner 配置维度 (L 轴)**:

| 维度 | 取值 | 实验点数 |
|------|------|---------|
| planner Conv 剪枝率 | {0, 0.25, 0.5} | 3 |
| planner 量化精度 | {FP16, INT8 (excl. Conv3D)} | 2 |

planner 子空间大小: 3 × 2 = **6 点**

**联合笛卡尔积**: 16 × 6 = **96 理论配置点**

### 4.2 搜索空间约束 (合法性剪枝)

实际可行点远少于 96, 约束如下:

| 约束 | 描述 | 剪枝效果 |
|------|------|---------|
| C1 INT8 %32 对齐 | 剪枝后 num_filters 必须 %32 (INT8 配置时) | prune_rate × INT8 组合必须满足; BasicBlock 无 groups 约束, 放宽 |
| C2 感知→planner 特征传播 | 感知 shrink_conv 输出固定 128ch (planner 接口); 感知剪枝改变中间特征但接口维度固定 | 接口固定, 感知/planner 独立搜索合法 |
| C3 感知剪枝 finetune 先决 | 感知剪枝后 Driving Score 崩 (未 finetune) 不可接入 planner | 感知 finetune 是 planner 实验的前提; 不可跳过 |
| C4 planner 剪枝前提 | planner 使用感知输出特征; 感知大幅剪枝后的特征质量影响 planner 剪枝可行性 | 建议先固定感知档位再搜 planner |
| C5 Conv3D 校准 | planner Conv3D 使用 INT8 需多帧校准数据 | Conv3D INT8 需验证, 先标 FP16 |

**约束后有效空间估算**:
- 感知有效: ~12 点 (排除 prune75 未 finetune 直接上 INT8 的无效组合)
- planner 有效: ~5 点 (Conv3D INT8 暂不纳入)
- 联合有效: ~60 点 【预研·static】

### 4.3 跨网络耦合分析

**耦合点 1 — 感知特征质量 → planner waypoint 精度**:
- 感知 backbone 剪枝 → BEV 特征 `[B, 128, 96, 288]` 信息量↓
- planner 的 conv3d temporal 融合跨 T=5 帧 → 特征时序一致性要求; 感知剪枝可能破坏一致性
- 耦合效应: 感知 p50 剪枝 + planner plan_conv50 剪枝 的 Driving Score 可能 < 分别剪枝的下界 (负耦合); 需 (p50×plan_conv50) 联合实测确认

**耦合点 2 — 量化误差叠加 (类比 Pyramid 剪枝×INT8 对齐陷阱)**:
- Pyramid 陷阱: 通道剪到非 32 对齐 (如 90ch) → INT8 padding 90→96 → 同时抵消剪枝与量化收益
- CoDriving 对应: BasicBlock 无 groups 约束 → %32 更容易满足, 上述陷阱风险低 【预研·static】
- 新潜在陷阱: 感知 INT8 激活 → planner Conv3d 接收量化后特征 → 时序帧间量化噪声是否累积? 【预研·static, 需实测】

**耦合点 3 — 延迟分配与 Driving Score 的动态耦合**:
- e2e lat = t_perception + t_planner + t_communication
- 感知 INT8 lat 大幅降 → 若总 lat 降低到某阈值以下, 时延回灌对 Driving Score 的影响缩小 → 非线性改善
- 这是正向耦合 (latency 两段同时压缩 → 闭环收益超线性叠加)
- 【已填 CD-A1 真测】: 感知 e2e 10.17ms / planner 1.50ms, 感知占联合总 87.1%; 感知是主延迟段, 压感知 lat 是闭环收益的主要来源; planner 端 INT8 量化对全局延迟影响有限

### 4.4 联合搜索空间规模估算对比 (CoDriving vs Pyramid)

| 维度 | Pyramid (Pareto 退化) | CoDriving (预期) |
|------|-----------------------|-----------------|
| 软件配置有效点数 | ~8 (backbone 剪枝×量化, AP 退化) | **~60 点** 【预研·static】 |
| 是否有指标 trade-off | 否 (AP 高原, 无悬崖) | **预期有** (Driving Score 传导链更锐) |
| 双网络联合轴 | 单网络 | **感知 × planner 两轴** |
| planner 独立量化轴 | N/A | MLP INT8 友好独立轴 |
| 耦合陷阱 (负向) | 剪到非32对齐 + INT8 padding | 感知质量↓ × planner 精度↓ 叠加; Conv3D 量化校准 |

---

## §5 依赖实验数据的 TBD 项

> ★[2026-06-15 更新] CD-A1 + CD-A1b 已完成, TBD-1 已填; 其余待后续实验。

| TBD 项 | 依赖内容 | 影响 | 状态 |
|--------|---------|------|------|
| **TBD-1** 感知/planner 延迟占比 | CD-A1: 感知 e2e lat 模块分解 | 决定哪个网络是延迟瓶颈, 优先压哪段 | ✅ **已填(单卡BN=2口径)**: 感知 10.17ms(87.1%) / planner 1.50ms(12.9%); backbone.resnet 被调 2 次 = 5.85ms(57.5% e2e) = 首要优化目标; planner MLP decoder 0.087ms 可忽略 【真测·CUDA Event @4090】。**⚠️[2026-06-16] 此为仿真单卡 BN=2(ego+RSU 混测)口径, ≠ 分布式部署延迟; 部署须分设备 BN=1 → 见新增 TBD-8(CD-A1c)** |
| **TBD-8(CD-A1c)** 分设备部署延迟 | RSU(backbone×1, BN=1) vs ego(主路+fusion, BN=1) 分设备真测 | 给部署级 RSU/ego 关键路径延迟 + 正确的剪枝/INT8 系统杠杆 | ❌ 待跑。系统延迟 ≈ max(RSU+通信, ego)+planner; 「2× 杠杆」仅 ego 侧成立 |
| **TBD-1b** fusion 内部分解 | CD-A1b: fusion_net 内计算来源 | 澄清 fusion 延迟是 attention 还是 backbone 重跑 | ✅ **已填**: backbone重跑3.04ms(65%) / 纯注意力0.81ms(16%) / deblock0.32ms; 主因是 backbone 重跑非 attention `results/CD_A1b_fusion_internal_4090.csv` |
| **TBD-2** AttenFusion 层 INT8 实际影响 | CD-A3: TRT INT8 build + Driving Score 实测 | 决定 AttenFusion 是否需保 FP16 | ❌ 等 CD-A3 |
| **TBD-3** Conv3D temporal 校准方案 | CD-A3 含 temporal 多帧校准数据设计 | planner Conv3D INT8 是否可行 | ❌ 等 CD-A3 |
| **TBD-4** 感知剪枝 Driving Score 悬崖位置 | CD-B1 × CARLA finetune × Town5 eval | 确认 p25/p50/p75 哪档是悬崖点 | ❌ 等 CD-B1 |
| **TBD-5** planner 剪枝对 DS 独立影响 | 感知 finetune 后 planner Conv50 消融实验 | planner 剪枝是否有独立贡献 | ❌ 等 CD-B1 之后 |
| **TBD-6** 联合量化耦合: 感知 INT8 → planner Conv3D 输入噪声 | CD-A3 感知 INT8 engine 接 planner 真测 DS | 量化误差跨网络叠加是否显著 | ❌ 等 CD-A3 |
| **TBD-7** 感知 e2e ONNX export scope 确认 | CD-A2: pillar_vfe + scatter 后的 BEV tensor 作为 export 起点 | 影响 calibration 接口设计 | 进行中 (CD-A2) |

---

## §6 [占位] 硬件侧章节 (待 hw-optimizer 补充)

> 以下章节标题由 hw-optimizer 补充。软件侧设计与硬件侧的接口约定:
> - 软件输出: 感知 ONNX (`backbone+fusion+heads`) + planner ONNX (全网), 含剪枝后通道数 manifest
> - 硬件接收: ONNX → TRT FP16/INT8 engine build; 输出 lat/throughput/energy 真测
> - 通道对齐接口: 软件侧保证 INT8 配置下所有 Conv 通道数 %32 (BasicBlock 无 groups 约束, 满足容易)

```
§7 TRT Build 策略 (hw-optimizer)
§8 4090 GPU 延迟/吞吐/能耗真测
§9 Orin AGX 部署方案 (TBD-等Orin访问)
§10 Pareto 前沿构建 (联合 sw+hw 数据)
```

---

*文档版本: v1.1 (2026-06-15, doc-curator — CD-A1+A1b 真测整合)*
*变更: §1.1 补 backbone 2× 杠杆真测数据 + DCN 确认; §2.1 优先级表补 fusion N/A 行真测说明; §3.3 planner MLP decoder 收益重新评估; §4.3 填入感知/planner 占比; §5 TBD-1/1b 标完成; 纠正 fusion>backbone 误导性旧表述*
*下次更新触发条件: CD-A2 TRT FP16 结果(填 §6/硬件侧); CD-B1 剪枝 finetune 结果(更新 §2.2 悬崖假设验证状态, TBD-4)*

*[2026-06-16 口径纠错] 撤回「系统级 e2e 1.4× / backbone 2× 杠杆」的系统级解读: 5.85ms/57.5%/2× 是 CARLA 单卡 BN=2(ego+RSU 混测)口径; 真实分布式部署 RSU 调 backbone 1 次、ego 2 次, 两设备并行系统延迟取 max。「2× 杠杆」仅车端 ego 成立; 部署级系统加速待 CD-A1c 分设备 BN=1 真测。*
