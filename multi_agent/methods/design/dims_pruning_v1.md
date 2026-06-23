# 剪枝可配置维度清单 v1.0

> 交付物 2 (Configurable Pruning Engineer). 配套工具: `tools/configurable/prune_config.py`.
> 参考 DepGraph (Torch-Pruning, CVPR'23) + UniV2X 已有 `prune_univ2x.py` / `structural_prune_pyramid.py`.
> 与共享契约 `framework/config_schema.py` (只读) 字段映射见 §5.
> 数字来源标注: [实测]=本工具 smoke 跑通 / [契约]=config_schema 定义 / [MEMORY]=项目记忆 / [论文]=survey.

---

## 1. 维度总览 (可配置搜索空间)

| 维度 | config_schema 字段 | 类型 | 取值域 | 粒度 | 约束/说明 |
|------|--------------------|------|--------|------|-----------|
| D1 剪枝率 | `prune_rate[module]` | float | `[0.0, 1.0)` | per-module | 剪掉比例; 受 §3 每模块上限 |
| D2 剪枝对象 | `prune_object` | str | `channel / head / 2:4 / element / none` [契约] | **全局** (不混用) | structural=channel/head; 2:4=半结构化; element=非结构化(不上 TRT) |
| D3 重要性准则 | `prune_criterion[module]` | str | `L1 / Taylor / FPGM / Wanda / none` [契约] | per-module | Taylor 需梯度; Wanda 需激活校准 |
| D4 通道对齐 round_to | (派生, 见 §5) | int | `8 / 32` | per-module | INT8→32 [MEMORY]; FP16/FP32→8 |
| D5 剪枝粒度 | (建议补充, 见 §5) | str | `local / global / isomorphic` | 全局 | local 可预测; global 跨层重分配 |
| D6 ResNeXt wpg | (建议补充) | int | `4 / 16` | per-stage | 极小 planes 需 wpg=16 防崩 [MEMORY] |

> D2 全局粒度是 `config_schema` 的设计 (`prune_object: str` 单值, 注释 "channel/head/2:4 三选一,因为不会混用").

---

## 2. 每维度取值语义

### D1 prune_rate
- 语义: **剪掉的比例**. `0.5` = 砍一半通道. (注意 `config_schema.from_prune_json` 把 1.2 的 `ffn_mid_ratio`=保留比例 转成 `1-kept`.)
- Pyramid (`model` 模块): 全局 rate 映射到 `num_filters` 各 stage. `0.5` → `[64,128,256]→[32,64,128]` [实测].
- UniV2X: per-module rate 分发到 FFN / attn_proj / head (见 `prune_univ2x._build_ratio_dict`).

### D2 prune_object
| 取值 | 含义 | 是否 structural rebuild | 上 TRT 加速 |
|------|------|------------------------|-------------|
| `channel` | 结构化通道剪枝 (默认主路径) | 是 (真减通道) | 是 |
| `head` | 注意力头剪枝 (per-head, num_heads dict) | 是 | 是 |
| `2:4` | 半结构化稀疏 (每 4 元素留 2) | 否 (mask, 但 Ampere+ 硬件加速) | 仅 sparse tensor core |
| `element` | 非结构化逐元素 | 否 (mask) | **否** (无加速, 已是教训) |
| `none` | 不剪 | — | — |

> **主路径 = channel** (用户原话: mask 测 latency 无意义). `2:4` / `element` 列出但本工具不产出真 rebuild.

### D3 prune_criterion (重要性准则)
| 取值 | 内部名 | 依赖 | 实现位置 |
|------|--------|------|----------|
| `L1` | `l1_norm` | 仅权重 (默认, 最稳) | handwritten `l1_along_dim` / tp.MagnitudeImportance(p=1) |
| `Taylor` | `taylor` | **需梯度** (forward+backward 若干 batch) | tp.GroupTaylorImportance / prune_univ2x grad_collector |
| `FPGM` | `fpgm` | 仅权重 (几何中位数) | tp.FPGMImportance / prune_direct cdist |
| `Wanda` | `wanda` | **需激活校准** | (契约保留, 当前未实现 rebuild) |

### D4 round_to (通道对齐)
- `INT8` 模块 → `32` [MEMORY: INT8 路径剪后通道数必须 %32==0, TRT Tensor Core tile 对齐].
- `FP16/FP32` 模块 → `8` (与 `prune_univ2x` default 一致).
- 由本工具从 `config.q_bits[module]` 自动推断 (量化师写入).

### D5 剪枝粒度 local/global/isomorphic
- `local`: 每层独立按 rate 剪 — 可预测, Pyramid 默认.
- `global`: 跨层按全局重要性重分配 — DepGraph 实测 Pyramid 唯一能"local-fail-但-global-通"的路径 (见 §6), 但不可控.
- `isomorphic`: tp 1.6 支持, 同构层组统一剪.

### D6 ResNeXt width_per_group (wpg)
- HEAL Bottleneck 宽度: `width = int(planes * wpg / 64) * groups` [MEMORY].
- `g=32 + wpg=4` 时 `planes<16` → `width=0` 崩溃; 剪到极小须 `wpg→16`.
- 本工具 `resolve_pyramid_num_filters` 自动检测并建议提升 wpg.

---

## 3. 每子模块"最大可剪率边界"

> 受 分组卷积 / 精度 / 跨层依赖 限制. 越界本工具仅 WARN, 不强制截断.

| 模块 | config 模块名 | 建议上限 | 限制来源 |
|------|--------------|---------|---------|
| Pyramid 整体 | `model` | **0.75** | num_filters `[64,128,256]→[16,32,64]` (g32+wpg=16); 已有 pruned25/50/75 ckpt [MEMORY 目录] |
| backbone (UniV2X) | `backbone` | 0.6 | ResNet/ResNeXt 分组约束 |
| encoder | `encoder` | 0.5 | FFN/attn_proj; attn 头不可低于 num_heads 整除 |
| decoder | `decoder` | 0.5 | 同上 + 跨 layer 残差 |
| heads | `heads` | 0.3 | 检测头中间维度小, 剪多直接掉点 |
| v2x_comm | `v2x_comm` | 0.3 | agent_query 通信带宽敏感 |

> Pyramid `model` 的硬下限: 受 `round_to=32` floor, 任何 stage 剪后 `num_filters>=32`.
> 即 `prune_rate` 实际有效上限对 stage0(64ch) 是 ~0.5 (再大也 floor 到 32), stage2(256ch) 可到 0.875.
> 全局 rate>0.75 时低 stage 无法再降, 高 stage 仍可降 → rate 语义在 Pyramid 是"近似目标" [实测 T2].

> **[2026-06-02] 已测区间 (0–75%) 全部落在 AP 高原 (缓降, AP50 span 仅 0.034 / AP70 0.101), 尚无真测点踩到精度悬崖**。一条有说服力的 Pareto trade-off 曲线必须含拐点 → 建议补 **80 / 87 / 93%**(触 `round_to=32` floor + ResNeXt wpg→16 防崩)真测以锚定崖位置。见 §8。

---

## 4. 约束检查清单 (工具内置, fail-fast)

工具 `prune_config.py` 在 `resolve_*` 阶段强制检查:

1. **prune_rate ∈ [0,1)** — 越界抛 `PruneConstraintError`.
2. **prune_object / criterion 合法** — 对照 `PRUNE_OBJECT_VALUES` / `PRUNE_CRITERION_VALUES` [契约].
3. **grouped-conv %32 对齐** — Pyramid num_filters 全部 round 到 32 倍数 [实测 all %32 = True].
4. **ResNeXt wpg 陷阱** — `width=int(p*wpg/64)*g` 崩为 0 时自动建议 wpg→16, 仍不可行则抛错 [MEMORY].
5. **跨层依赖一致性** — 复用 `structural_prune_pyramid.transfer_weights`: stage conv3 输出 / downsample / deblocks 输入 / 下一 stage conv1 输入 统一用 `stage_keeps[i]` 索引 (DepGraph 依赖图实测覆盖同样的耦合, 见 §6).
6. **INT8 round_to=32** — `q_bits[module]=="INT8"` 的模块强制对齐 [实测 T1 round_to=32].

---

## 5. 与 config_schema 字段映射

| 维度 | config_schema 现有字段 | 状态 |
|------|------------------------|------|
| D1 prune_rate | `prune_rate: dict[str,float]` | ✅ 直接消费 |
| D2 prune_object | `prune_object: str` (全局) | ✅ 直接消费 |
| D3 criterion | `prune_criterion: dict[str,str]` | ✅ 直接消费 (L1/Taylor/FPGM/Wanda/none) |
| D4 round_to | — (从 `q_bits` 推断) | ✅ 工具内 INT8→32 推断, 无需新字段 |
| D5 粒度 local/global | — | ⚠️ **建议补充** `prune_granularity: str = "local"` (当前工具默认 local) |
| D6 wpg | — | ⚠️ **建议补充** `prune_resnext_wpg: dict[str,int]` (当前工具自动建议, 不入 Config) |

> **建议补充字段** (留给 framework owner 决策, 本工具不改契约):
> - `prune_granularity: str` ∈ `{local, global, isomorphic}`, default `local`.
> - `prune_resnext_wpg: dict[str,int]`, 仅 ResNeXt 类模块用, default 推断.
> 在补充前, 工具用 `arch=` 参数 + 内部默认 (local / wpg 自动) 兜底.

---

## 6. DepGraph 可用性实测结论 (交付物 4)

> 实测: `tools/configurable/smoke_prune_config.py::test_depgraph_feasibility` [实测 T4], torch 2.0.1 + torch_pruning 1.6.0.

| 项 | 结论 |
|----|------|
| **build_dependency trace** | ✅ **通过**. DepGraph 完整 trace Pyramid 子模块 (PyramidSubnet): 正确捕获 stage 内 3 个 Bottleneck 的 conv3 输出残差耦合 + downsample + deblocks 输入 + 跨 stage conv1 输入 + 跨 stage 的 AddBackward/ReluBackward 算子. 依赖图正确. |
| **local 剪枝 step + forward** | ❌ **失败**. `pruning_ratio=0.5, round_to=32, global_pruning=False` 剪枝后 forward 报 `RuntimeError: size 32 must match 64 at dim 1` — ResNeXt grouped conv2 (groups=32) 与残差 identity 通道数被剪得不一致. tp 1.6 对 "grouped-conv 输出 + 残差 add" 的 local 剪枝量算不一致. |
| **global 剪枝 step + forward** | ⚠️ 偶然通过 (本 ckpt). global 跨层重分配恰好避开冲突, 但不可控/不可复现. |

**决议 (旧)**: Pyramid 默认走手写 rebuild (`structural_prune_pyramid.transfer_weights`), DepGraph 当时回退。

> **[2026-06-02 更新] DepGraph 全网迁移已成功 + 收敛实测确认全网剪枝是可用维度**:
> `tools/configurable/depgraph_pyramid.py` 的 `PyramidFullTraceNet` wrapper 在完整 HeterPyramidCollab 上 trace+剪+forward 通 (根因: 旧 subnet 的 method-call 入口让 DepGraph 看不到 stage0 残差 identity 来源; 全网 wrapper 修复)。**deblocks/shrink 实测可剪**, 推翻 §1 "只 backbone 可剪"。
> **收敛 finetune (ep→32/36) + DAIR val 1789 真测** (`results/P1_wholenet_prune_real.csv`):
> backbone_only_p50(2.58M)/ wn_light(2.20M)/ wn_p50(1.86M)/ **wn_aggr(1.32M, -74.7%)**, lat_fp32_p50 4.40→3.18ms 单调降。
>
> **★[2026-06-02 勘误 — AP50 "全平 0.75" 是假象]**: 该 CSV 的 AP50 列只保留 2 位小数, 把真实信号抹平; 且 wn 五行**共享同一 backbone=[32,64,128], 行间只动 neck(deblocks/shrink)**。金标准 `data/stage_a_ap_real.parquet`(4 位真值)显示 **backbone 剪枝 AP 单调降**: AP50 0.791→0.777→0.764→0.757(base→p75, span 0.034), AP70 0.631→0.530(span 0.101), 信号 ≈30× 噪声。⇒ **AP 由 backbone 通道主导, neck 几乎不携带 AP 信号(代价 <0.01 AP50)** —— 这才是 wn 行"看着平"的真因, 不是"剪枝无损"。"wn_aggr 严格支配 backbone_only" 仅在 (lat, AP50@2dp) 成立, 是 **Pareto 退化症状**(0–75% 全在 AP 高原缓降, **未触精度悬崖**), 需补 80/87/93% 找崖。Pareto AP 轴用 **AP70**(信号最强)。昨夜非单调 0.73/0.75/0.76 是欠拟合噪声, 收敛后消失。图 `fig5_wholenet_prune.png`。

UniV2X 主干: forward 签名复杂 (BEV query + img_metas + 多模态), DepGraph 已知 trace 不通 (见 `prune_univ2x.prune_model` 注释), 沿用 `prune_direct` 手写局部依赖剪枝.

---

## 7. 接口 (给协作者)

- **量化师** (`tools/configurable/quant_config.py`): 读本工具 manifest 的 `num_filters_new` / `out_ckpt` 做 INT8 校准; 本工具读其 `q_bits[module]=="INT8"` 决定 round_to=32. manifest 字段 `constraint_check.all_num_filters_mod32` 保证 INT8 对齐.
- **硬件师** (`scripts/phase1/m4_8_trt_build_bench.py`): 读 manifest `out_ckpt` → ONNX export (`tools/export_onnx_pyramid.py`) → TRT build. 小模型通道数已 %32, 可直接 INT8 build.
- **数据生成师**: manifest 的 `params_total_new` / `params_backbone_reduction_pct` / `num_filters_new` 作为 LGB 特征 (剪枝维度数值化).

---

## 8. Pareto 指标设计 (剪枝维度, [2026-06-02])

> 由剪枝专家分析后确立。回应"剪枝在 AP 方向是否有信号 / 是否需加模型大小作 Pareto 轴"。

### 8.1 现状诊断
- **2D (latency, AP) 在安全区 (0–75%) 近退化**: AP 缓降 (AP50 span 0.034 / AP70 0.101), latency 随剪枝单调降 → 剪得越狠越"支配", trade-off 太平缓, 搜索区分度低。`wn_aggr 严格支配 backbone_only` 正是退化症状。
- **AP 真有信号** (≈30× 噪声), 只是**没踩到悬崖**, 曲线缺拐点。

> **[2026-06-02 与框架决策对齐]**: 框架层已定 Pareto = 预测全 5 指标 + regime 条件目标, 默认主前沿 **(AP, latency, energy)**, **model_size 默认降为 SLA 约束(显存预算), 仅 Orin/FPGA regime 升为目标轴**(详见 `background/00_研究目标与实验档案_v1.md` Pareto 决策)。下文剪枝维度的 3D 提议据此并入: **AP 轴用 AP70**(信号最强)、补能耗后剪枝点落到统一 (AP, latency, energy) 前沿、model_size 作约束(边缘 regime 才升目标)。

### 8.2 推荐: 3D Pareto `(latency, AP70, model_size)` + 补悬崖
1. **AP 轴用 AP70**(span 0.101, 是 AP50 的 3×, 区分度最高)。
2. **必须补 AP 悬崖真测点** (出路 a, P0): 把缓降曲线延伸到崖 (80/87/93%), 才有真 trade-off。
3. **model_size / params 作第三轴** (出路 b): 服务**边缘部署** —— Orin 显存 / FPGA BRAM 容量(硬闸门, 与 latency 正交)/ 能耗 / OTA 加载带宽。
   - 在 **4090-only** 分析里 size 降为约束/标注列(24GB 充裕, size 几乎无约束)。
   - 在 **Orin/FPGA** 部署叙事里 size 升为正式 Pareto 轴(引入它后 (size, AP) 平面的退化支配关系消失, 恢复真权衡)。
   - 注意 size 与 latency 强相关但非共线(params↓ 通常 lat↓), 作独立轴有部分信息冗余 → 需 Orin 显存/功耗真测 (P2) 坐实其独立价值。

### 8.3 待补真测点 (优先级)
| 优先级 | 内容 | 配置 | 状态 |
|---|---|---|---|
| **P0** 找悬崖 | prune85/90/95 → flat ckpt → 收敛 finetune(48ep)→ DAIR val 1789 真测 | [10,20,40]/[6,13,26]/[4,6,13], wpg=16 | ✅ **已完成, 见 §8.4** |
| **P1** 补三轴配对 | wn_p50/wn_aggr 的 `lat_fp16_p50` | 已有 ckpt | ✅ 已完成 (wn_p50=3.75ms / wn_aggr=3.30ms fp16) |
| **P2** 验 size 非冗余 | Orin 上测 1–2 剪枝点显存 + 功耗 | 远程 Orin | 待 Orin |

### 8.4 ★ P0 找悬崖结果 (2026-06-02, DAIR val 1789 真测, `results/ap_cliff_converged.json`)

| 档 | num_filters | wpg | subnet_params | AP30 | AP50 | AP70 |
|---|---|---|---|---|---|---|
| prune85 | [10,20,40] | 16 | 2.10M | 0.79 | 0.74 | 0.59 |
| prune90 | [6,13,26] | 16 | 1.74M | 0.79 | **0.75** | 0.58 |
| prune95 | [4,6,13] | 16 | 1.56M | 0.78 | 0.74 | 0.57 |

**结论 1 — 没找到悬崖**: 即便 nominal [4,6,13], AP50 仍稳在 0.74、AP70 0.57, **plateau 一直延伸没崩**。
**结论 2 — 根因: AP 由 bottleneck 宽度(wpg)主导, 不是 num_filters**: 为避 width=0 强制的 wpg=16 把 bottleneck 撑住, 导致这三档**实际 params(1.56–2.10M)反而 > wn_aggr(1.32M @ AP50 0.75)**。所以"[4,6,13]"是**假激进**——真正决定 AP 的是 wpg/bottleneck 宽度。这**精化了**§6 "AP 由 backbone 通道主导"为"由 bottleneck 宽度主导"。
**⚠️ comparability caveat**: 这三档 wpg=16, 与 stage_a 金标准(wpg=4)**不在同一架构族, 不能直接拼 stage_a 曲线**, 只能按 params 轴比。
**结论 3 — 要真踩悬崖需把 params 压到 1.3M 以下**(wpg=16 这条路压不下去); 下一步应**降 wpg 或直接搜更小 params**, 而非降 num_filters。
**数据修复记录**: `eval_cliff_converged.py` 解析器找 `eval_intermediate.yaml`(未生成), AP 实际在 inference.py stdout, 已从 log 回填 json。

### 8.5 ★★ 第二轮找悬崖 (降 wpg=4, 2026-06-02, `results/ap_cliff2_converged.json`) — 定论: 无悬崖

把 wpg 降到 **4**(最窄 bottleneck), 直接压**可剪的 pyramid_backbone(pb)**到极小:

| 档 | num_filters | wpg | 可剪 pb | AP30 | AP50 | AP70 |
|---|---|---|---|---|---|---|
| cliff2_a | [32,64,128] | 4 | 1.10M | 0.79 | 0.75 | 0.60 |
| cliff2_b | [24,48,96] | 4 | 0.68M | 0.78 | 0.74 | 0.59 |
| **cliff2_c** | **[16,32,64]** | **4** | **0.36M** | 0.79 | **0.75** | **0.60** |

**★定论 — 可达范围内无精度悬崖**: 即便 pb 压到 **0.36M(比 base 3.76M 砍 ~90%)+ wpg=4**, AP50 仍 **0.75** / AP70 **0.60**, 三档差异在 finetune 噪声(±0.01)内, 全在同一 plateau。**§8.4 的"wpg 主导 AP"假设也被推翻**——wpg=4 照样不崩。
**根因 [2026-06-02 第三轮研究订正 — 我之前归因错了]**: AP 下限(~0.75/0.60)**不是"被固定主干 encoder_m1/backbone_m1 撑底", 而是 DAIR-V2X 任务 + Pyramid head 的固有 AP 上限** —— 金标准实测 **base 全模型 AP50 本来就只有 0.791 / AP70 0.631**, cliff 模型(0.74–0.75 / 0.57–0.60)离 base 仅差 ~0.04 → **剪枝从头到尾几乎无损**。参数实测分布: pyramid_backbone **68.9%** / shrink_conv **26.8%** / backbone_m1 4.1% / encoder_m1 **0.02%**(DAIR 下仅一层 `Linear(10→64)`, 非重型 sparse VFE) / heads 0.1%。**encoder_m1/backbone_m1 既非不可剪(backbone_m1 是标准 ResNet 完全可结构化剪枝, 只是占 4.1% 不值; encoder_m1 无参可剪)也非 AP 主因**(占 4.1% 不可能驱动 AP)。**没悬崖的真因 = 模型对 DAIR 严重过参数化**(砍占 69% 的主干 90%+ 仅掉 4 点 AP)。真正可搜的 AP-trade-off 在量化 INT8 边界(`pruned50 int8` 已比 fp16 掉 1.2 点)/ 激进剪枝×int8×欠 finetune 叠加 / 跨更难任务或模型, 不在 pyramid_backbone 剪枝率。
**★对框架的含义**: 剪 pyramid_backbone 几乎"免费"(参数/延迟大降、AP≈0 损失)→ **剪枝维度内 (params/lat, AP) Pareto 退化**(剪到底即最优, 无可搜 trade-off)。与"per-stage 混精被 TRT-auto 支配"同类: **框架应识别这类退化维度、直接取极值, 而非枚举搜索**。真正可搜的 AP-trade-off 在量化(INT8 边界)/ 剪固定主干 / 跨模型, 不在 pyramid_backbone 剪枝率。

### 8.6 ★ V2X-ViT 跨模型重复验证 (2026-06-05, supervisor PASS)

> 来源: HANDOFF `HANDOFF_model_reselection_and_pyramid_audits_v1.md` §1.3; supervisor 核验 PASS。工具: `tools/configurable/depgraph_v2xvit.py`(BaseBEVBackbone)。

V2X-ViT backbone(BaseBEVBackbone) L1 剪枝 × finetune 25ep × DAIR val 1789 全集真测:

| 档 | actual_filters | backbone_params_reduced | AP70 (eval) | mAOE |
|---|---|---|---|---|
| base (epoch17, 零额外训练) | [64,128,256] | — | 0.5212 | 0.0656 |
| p50 (finetune 25ep, best@ep17) | [32,64,128] | -74.8% | 0.5336 | 0.0665 |
| **p75** (finetune 25ep, best@ep16) | **[64,32,64]*** | **-91.4%** | **0.5445** | **0.0630** |

*p75 actual_filters 非单调(L1+round_to=32 把 stage0 留满); **禁止拼"剪枝率→AP"单调曲线**, 入库用实测结构。

**结论(supervisor 修正版措辞)**:
- **"剪枝+FT 无退化信号"** — SNR avg-half 0.8×/2.3× 均 <5×。
- **禁说"剪枝提升"**: base 零额外训练 vs pruned +25ep = **非等预算对比**; p75 mAOE CI 与 base 不重叠 = "可分辨反向改善"(ISS-032), 非"噪声内"。
- **跨模型含义**: 独立架构(V2X-ViT, 3层 transformer + BaseBEVBackbone) 在剪枝 -91.4% 仍无精度悬崖, **复现 Pyramid 过参数化定论** —— 对 DAIR-V2X 任务, **两个模型(Pyramid / V2X-ViT)均**存在 AP-backbone 剪枝 Pareto 退化现象(当前 n=2)。
- **证据文件**: `results/v2xvit_baseline_a1.json` · `results/v2xvit_a2_p50.json` · `results/v2xvit_a2_p75.json` · `output/a2_prune/` · `output/a2_finetune/`。

> 纪律: **AP = 全模型 eval(DAIR val 1789, 已 finetune)**; 剪枝对象仅 backbone(fusion/transformer 未剪)。
