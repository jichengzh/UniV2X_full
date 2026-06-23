# 量化可配置维度清单 v1.0

> **★[2026-06-04 Phase M 补充定论, ISS-020/024/029 终核 PASS]** 换指标实测确认: **INT8 量化在 Pyramid/DAIR 上连几何误差指标(mATE/mASE/mAOE)都无信号** — ΔmAOE 非单调变号, p25/p50/p75 SNR=**0.58/0.05/0.82×**(噪声级, 3/4 档 CI 重叠+非单调; base 档 Δ+0.0030 CI 可分辨但非单调+量级仅5%, 不构成轴信号, ISS-020 §4)→ "量化近无损"是**真结论而非 mAP 指标盲区**; 量化的精度 trade-off 在本模型上不可提取, 须换模型(V2X-ViT, Task#8)。详见 `pareto_definition_v1.md §七.2`(原 `ap_activation_strategy_v1.md` 已并入该节, 过程版存 `multi_agent/archive/`)。
>
> 角色: 可配置量化工程师 (Configurable Quantization Engineer) 交付物 2
> 参考: QuantV2X (Wei et al., ICCV'25) PTQ 路线 = MinMax-init + per-channel weight + AdaRound + block-reconstruction
> 工具: `tools/configurable/quant_config.py` (消费 `framework/config_schema.py` 的 Config)
> 实测依据: `data/问题.md §问题1/4/5/7` + `methods/量化和剪枝方法.md §三`
> 数据标注: [实测] = 本项目 e2e/subnet 真测; [论文] = 文献结论; [TRT约束] = NVIDIA 硬件/API 约束; [估算] = 推断

---

## 一、维度总览 (摘要表)

| # | 维度 | Config 字段 | 取值域 | 约束 | 状态 |
|---|------|------------|--------|------|------|
| Q0 | 位宽 | `q_bits[module]` | FP32 / FP16 / INT8 | 敏感模块 INT8→FP16 | ✅ 实装 (TRT BuilderFlag) |
| Q1 | 分模块混合精度 | `q_bits` (per-module dict) | 任意 INT8/FP16 组合 | — | ✅ 实装 (`--mixed-fp16-substr`) |
| Q1' | per-stage 混合精度 | (扩展, 见 §5) | {FP16,INT8}³ | reformat 开销大 | ✅ 实装 (substr 匹配) |
| Q2 | 量化粒度 | `q_granularity[module]` | per-tensor / per-channel / none | activation 只能 per-tensor | ✅ weight per-channel 默认 |
| Q3 | 量化对象 | `q_object[module]` | W+A / W-only / none | W-only 是全局 flag | ✅ W+A; 🟡 W-only |
| Q4 | 校准器 | `q_calibrator[module]` | minmax / percentile_99_99 / entropy / none | **entropy 禁用** | ✅ minmax 主路径 |
| Q5 | AdaRound 开关 | (建议新增, 见 §6) | on / off | 仅 PyTorch 侧 | ⚠️ 仅 reference, 不进 TRT |
| Q6 | 校准样本数 | (build 参数, 非 Config) | 50-200 | — | ✅ (`--calib-data`) |

> **核心能力**: Q0+Q1 组合 = **分模块混合精度** (用户明确要求). 同一引擎内某些模块 INT8、某些 FP16, 通过 TRT `PREFER_PRECISION_CONSTRAINTS` + 逐层 substr 匹配实现.

---

## 二、各维度详解 (取值域 + 约束)

### Q0 — 位宽 `q_bits[module]`

- **取值域**: `FP32` / `FP16` / `INT8` (`config_schema.Q_BITS_VALUES`)
- **实现**: TRT `IBuilderConfig` flag (`BuilderFlag.FP16` / `BuilderFlag.INT8`); FP32 默认 fallback
- **约束**:
  - 敏感模块 (cls/reg head, attention/MSDA, grid_sample fusion) 请求 INT8 会被工具改写为 FP16 [论文 + 实测 问题1]
  - INT4 / 更低位宽: 不支持 (TRT GPU path 无 INT4 IMMA, 已剔除)
- **代码**: `m4_8_trt_build_bench.py:238-260`, `build_trt_int8_univ2x.py`

### Q1 — 分模块混合精度 `q_bits` (per-module dict)

- **取值域**: 任意 per-module {FP16, INT8} 组合 (FP32 也可混)
- **实现机制**: `quant_config.resolve_quant_plan()` 把 FP16 模块的 layer-name 子串收集为 `--mixed-fp16-substr`, 其余层 INT8; TRT 用 `PREFER_PRECISION_CONSTRAINTS` (软约束) 逐层匹配
- **约束**: m4_8 是**全局 calibrator + 全局 w_only flag** — 多模块若请求不同 calibrator/object, 工具统一为主路径 minmax 并发 warning
- **实测最终结论 (2026-06-01, 见 §2.5)**: per-stage 强制混精在 AP 轴对 forced-all-INT8 有正向点 (剪枝档保 stage1 FP16 +0.022~0.025), **但 lat-AP 平面上被 TRT-auto 支配, 不产生 Pareto 新点**。旧 v1 "无价值/负样本" 曾因两个 bug 误判, 已修正; 详见 §2.5 + `results/perstage_quant_AP_verdict_v2.md`

### Q2 — 量化粒度 `q_granularity[module]`

- **取值域**: `per-tensor` / `per-channel` / `none` (`config_schema.Q_GRAN_VALUES`)
- **约束** [TRT约束]:
  - **weight**: per-channel (TRT INT8 默认, 推荐) 或 per-tensor 均可
  - **activation**: **只能 per-tensor** — GPU GEMM / Orin DLA 不支持 per-channel activation. 工具强制改写并 warn
  - per-group: TRT 不支持, 已剔除 (v1.5 §0.2)
- **工具行为**: INT8 时若未指定, weight 默认 per-channel; activation 恒 `per-tensor`

### Q3 — 量化对象 `q_object[module]`

- **取值域**: `W+A` (权重+激活 INT8) / `W-only` (仅权重 INT8, 激活 FP16) / `none`
- **实现**:
  - W+A: TRT INT8 默认行为 ✅ 完全实装
  - W-only: `m4_8 --w-only` flag (`layer.set_output_type(0, FP16)`) — 🟡 **全局** flag, 非 per-module
- **约束 / 限制**: per-module 不同 W-only/W+A 在当前 m4_8 路径下不可分离 (全局 flag). 工具检测到混合请求会 warn, 建议走 `inject_qdq_from_config.py` per-layer Q/DQ 路径 (见 §未完成项)
- **实测**: W-only (Q_int8_pc_wo) e2e AP 0.541 vs W+A (Q_int8_mm) 0.540, 差 0.001 在噪声内 [实测 问题5.7]

### Q4 — 校准器 `q_calibrator[module]`

- **取值域**: `minmax` / `percentile_99_99` / `entropy` / `none` (`config_schema.Q_CALIBRATOR_VALUES`)
- **约束 (硬性)**:
  - **`entropy` 禁用** [实测 问题1]: 自训 ckpt 长尾激活 (shrink_conv p99 = 2.5-3.5 vs 官方 1.2-1.4) 触发 INT8 AP 崩塌 (0.02-0.41). 工具自动回退 minmax
  - `minmax`: ✅ 主路径 (`trt.IInt8MinMaxCalibrator`), 全部 INT8 anchor / Pareto / LGB 数据
  - `percentile_99_99`: ⏸ 备选 (手工 `tensor.dynamic_range`, `build_percentile_e2e_engine.py`), calibrator 消融用
- **工具行为**: INT8 模块未指定校准器 → fallback minmax (config_schema 约定)

---

## 二.5 [2026-06-01 更新] TRT 自动精度的判定依据 + per-stage 混精最终结论

> 本节解释一个关键机制(回答"TRT 凭什么判断敏感层"), 并据 clean-GPU latency 实测给出 per-stage 强制混精的最终裁决。证据: `dataset_v2.csv` / `perstage_quant_pareto_verdict_v2.md`。

### (1) TRT 逐层精度是**延迟驱动, 不是精度驱动** (常见误解纠正)
开 INT8 flag (implicit/calibrator 模式) 时, TRT builder 对**每一层**:
1. 枚举该层 INT8/FP16/FP32 的可用 tactic;
2. 连同**精度边界处的 reformat (量化/反量化) 开销**一起计时;
3. 选**整个 engine 总延迟最小**的逐层精度组合。
硬约束只有: (a) 该 op 有无 INT8 kernel; (b) 该 tensor 有无校准动态范围。
**TRT 全程不测 AP**。`calibrator` 只提供量化 scale (动态范围), **不决定哪层 INT8 vs FP16** —— 那是 builder 的 tactic 选择。

### (2) "敏感层被保 FP16" 是副作用, 不是 TRT 主动判断
TRT 为**速度**把"INT8 不划算的层"(层小 / reformat 开销 > INT8 省的算力 / 无 INT8 kernel)留 FP16; 而这些层**恰好高度重合于精度敏感层** → 精度被顺带保住, 是 correlation, 不是 TRT 做了敏感度分析。

### (3) 实测铁证 (prune50p, body_subnet_collab2)
| 配置 | lat (ms) | ap50 |
|---|---|---|
| auto-int8 (TRT 自由选层) | **0.796** | 0.7522 |
| forced-all-int8 (强按每层 INT8) | 0.907 | **0.7280** |
| c3 最佳手工混精 (保 stage1 FP16) | 1.017 | 0.7526 |
| auto-fp16 | 1.014 | **0.7644** |

forced-all-int8 **又慢又低** → 印证: 强压 INT8 到不划算的层 = 多插 reformat (慢) + 多量化误差 (AP 低)。auto 留这些层 FP16 → 同时更快更准。

### (4) per-stage 强制混精最终裁决
- **AP 轴**: 对 forced-all-int8 (稻草人基线 0.728) 有正向 (+0.024); 但对真实可用的 auto-int8 (0.752) 仅 +0.0004 (噪声内) —— 因为 **auto 已免费拿到那 +0.024**。
- **Pareto (lat-AP)**: 3 个 triplet 的非支配前沿都只有 {auto-int8, auto-fp16}, **无任何手工 per-stage 混精在前沿** (auto-fp16 比 c3 又快又准, 严格支配)。
- **结论**: `per-stage 强制混精` 在 4090/Pyramid 上 **AP 有意义但 Pareto 无用**; **框架应直接用 TRT-auto 混精, 而非枚举强制 per-stage**。
- **例外/未来价值**: 真·精度驱动的逐层混精 (需显式跑逐层 ΔAP 敏感度 + `PREFER_PRECISION_CONSTRAINTS` 手工指定) 只在 **INT8 掉点大的 transformer/MSDA 模型** (如 V2X-ViT) 才值得; Pyramid INT8 近无损, 收益空间小。

---

## 三、每个子模块的可量化边界

| 模块 | layer 子串 | INT8-able? | 边界依据 |
|------|-----------|-----------|---------|
| backbone (Conv/ResNeXt) | resnet, conv, layer, backbone | ✅ INT8-able | 纯 Conv, TRT IMMA 融合, 实测 INT8 AP 不掉 [实测 问题5.7] |
| encoder (VFE/Scatter/BEV) | encoder, vfe, scatter | ✅ INT8-able | Conv-heavy; 但 VFE+Scatter INT8 引入 ~0.12 e2e AP 损失 [实测 问题1.5.x] |
| shrink_conv / deblock | shrink, deblock | ✅ INT8-able | 1×1 / 上采样 Conv |
| decoder (transformer/attention) | decoder, attn, msda, transformer | ❌ **FP16-only** | attention/MSDA 是自定义 plugin, TRT 无 INT8 实现, 已默认 FP16 [ADR-004] |
| heads (cls/reg/dir) | cls, reg, dir, head, bbox | ❌ **FP16-only** | logit 量化噪声 → score 边界失真 → AP 崩 [论文 + 实测 问题1] |
| v2x_comm (grid_sample fusion) | v2x, comm, fusion, grid_sample | ❌ **FP16-only** | 跨 agent warp grid_sample 高动态范围, plugin 无 INT8 |
| Pyramid "model" (合并模块) | resnet+conv+shrink+deblock+cls+reg+dir | ✅ INT8-able (模块级) | 99.9% params 是 Conv; cls/reg 子层在 build 时由全局敏感 substr 保 FP16 |

- **FP32-must**: 当前无强制 FP32 模块 (FP16 已是最低 baseline 精度). 仅 debug/数值对照时全 FP32.
- **敏感层 substr** (工具 `SENSITIVE_FP16_SUBSTR`): `cls, reg, dir, msda, attn, attention, grid_sample`

---

## 四、与 `framework/config_schema.py` 的字段映射

| Config 字段 | 类型 | 本工具消费方式 |
|------------|------|--------------|
| `q_bits` | `dict[str,str]` | per-module 位宽 → 推断 trt_precision (int8/fp16/mixed) |
| `q_granularity` | `dict[str,str]` | weight 粒度; activation 强制改 per-tensor |
| `q_object` | `dict[str,str]` | W+A / W-only → 全局 `--w-only` flag |
| `q_calibrator` | `dict[str,str]` | entropy→minmax 回退; 全局统一校准器 |
| `config_id` / `source` | `Optional[str]` | manifest 元数据追踪 |

- **完全复用现有字段, 无需改 schema** (Q0-Q4 全部已在 schema 中定义).

### 建议 schema 补充 (不直接改 schema, 仅记录于此)

| 建议字段 | 用途 | 现状 workaround |
|---------|------|----------------|
| `q_adaround: dict[str,bool]` | Q5 AdaRound 开关 (per-module) | 当前 AdaRound 仅 PyTorch reference (`adaptive_rounding.py`), 不进 TRT engine |
| `q_per_stage_bits: dict[str,tuple]` | Q1' per-stage {s0,s1,s2} 位宽 | 当前用 layer substr (stage0/1/2) 间接表达 |
| `q_object` per-module 真正生效 | W-only/W+A 逐模块分离 | 当前 m4_8 全局 flag; 需走 `inject_qdq_from_config.py` per-layer Q/DQ |

---

## 五、接口契约

### 5.1 与硬件师 (TRT build) 的接口 — **我产出什么**

1. **manifest JSON** (`<engine>.manifest.json`): 记录每模块 `effective_bits / granularity_w / granularity_a / q_object / calibrator / layer_substr / forced_fp16 / notes`, 以及全局 `trt_precision / mixed_fp16_substr / w_only / calibrator / int8_modules / fp16_modules`
2. **TRT build 命令** (`QuantPlan.to_build_command()`): 直接可执行的 `m4_8_trt_build_bench.py` argv, 含 `--precision / --calibrator / --mixed-fp16-substr / --w-only`
3. 硬件师只需 `subprocess.run(cmd)` 或拷贝到 shell, 喂 FP32 ONNX + calib numpy, 拿 INT8/混精 engine

### 5.2 与数据生成师 (枚举搜索空间) 的接口 — **可枚举维度**

- **可枚举**: Q0 位宽 (3 值) × Q2 weight 粒度 (2 值) × Q3 对象 (2 值) × Q4 校准器 (2 值 minmax/percentile, entropy 排除) × Q1 per-module 组合
- **约束自动剪枝**: 工具的 `resolve_quant_plan()` 会把非法组合 (entropy / 敏感层 INT8 / activation per-channel) 归一化, 数据生成师枚举时**无需手工排除**, 直接用 manifest 的 `effective_*` 字段作为真实标签
- **去重**: 多个请求经约束后可能落到同一 effective plan, 数据生成师应按 manifest 去重再 build

---

## 六、未完成项 + 后续工作

| 项 | 状态 | 后续 |
|----|------|------|
| 真 TRT build 验证 | ❌ 未实测 | smoke test 仅验证 plan/manifest 接口; 需硬件师用真 Pyramid ONNX + calib 跑 `to_build_command()` 确认 mixed engine build 成功 + 测 lat/AP |
| per-module W-only/W+A 分离 | 🟡 全局 flag | 需走 `inject_qdq_from_config.py` per-layer Q/DQ 路径; 但实测 ONNX Q/DQ 破坏残差融合 (问题4), 需 TRT C++ API set weight range |
| Q5 AdaRound 接入 TRT | ⚠️ 仅 PyTorch | AdaRound 当前只在 PyTorch 侧做 amota reference; QuantV2X 的 block-reconstruction 未移植到 TRT engine 路径 |
| Q1' per-stage 自动 substr | 🟡 部分 | stage0/1/2 substr 已定义, 但需按真实 Pyramid layer name 校准 (HEAL ResNeXt 命名) |
| 敏感层 substr 精度 | ⚠️ 需校准 | `MODULE_LAYER_SUBSTR` / `SENSITIVE_FP16_SUBSTR` 是基于命名约定, 真 ONNX 上需 dump layer names 核对命中率 |

---

## 七、交付物文件

- `tools/configurable/quant_config.py` — 配置驱动混合精度量化工具 (消费 Config → manifest + build 命令)
- `tools/configurable/test_quant_config_smoke.py` — smoke test (14 断言, 全 PASS)
- `paper_learning/2. AAAI最终故事/methods/dims_quantization_v1.md` — 本文档
