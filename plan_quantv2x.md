# UniV2X 量化加速 — Multi-Agent 协作计划

> **PM**：主 Claude 会话（Sonnet 4.6）  
> **状态看板**：`docs/quant/PROGRESS.md`（每次 session 开始必读）  
> **Agent 简报库**：`docs/quant/AGENT_BRIEFS.md`（每次派遣 agent 前查阅）  
> **架构决策记录**：`docs/quant/DECISIONS.md`

---

## 1. 研究目标

将 QuantV2X 的 **PTQ + AdaRound** 量化技术应用到 UniV2X，替换其基于 TRT ModelOpt 工具包的量化方案。

**约束**（硬性，不可违反）：
1. 不修改 `tools/test_trt.py` 的主体逻辑（4 个 hook 接口不变）
2. 不修改任何 `*_trt*.py` 模型文件
3. 不修改任何 config 文件
4. AMOTA 相比 FP32 TRT baseline（0.370）下降 ≤ 0.010

**当前 baseline**（end-to-end, 168 samples, V2X-Seq-SPD cooperative val）：
| 配置 | AMOTA | AMOTP | mAP |
|------|-------|-------|-----|
| PyTorch FP32 | 0.338 | 1.474 | 0.0727 |
| Hook-A+B+C+D TRT FP32 | 0.370 | 1.446 | 0.0760 |
| **目标：INT8 TRT** | **≥ 0.360** | — | — |

**核心技术约束**：
- MSDAPlugin（多尺度可变形注意力）是自定义 CUDA kernel，TRT **无法**对其内部做 INT8 calibration
- 与 MSDAPlugin 直接相连的 Linear（`sampling_offsets`、`attention_weights`）**不量化**
- LayerNorm、Softmax、坐标变换（`inverse_sigmoid`）**不量化**
- 采用混合精度：可量化层 INT8，plugin 层强制 FP16

---

## 2. Agent 角色分配

| 代号 | Agent 类型 | 项目角色 | 主要职责 | 上下文预算 |
|------|-----------|---------|---------|-----------|
| **PM** | 主 Claude 会话 | 项目经理 / 编排者 | 读取 PROGRESS.md、派遣 agent、更新状态 | 全量 |
| **ARCH** | `everything-claude-code:architect` | 系统架构师 | 接口设计、ADR、量化边界决策 | 中（< 8k tokens 输入） |
| **EXPLORE** | `Explore` | 代码库探索员 | 读取源码（节省 PM 上下文） | 小（focused search） |
| **PLAN** | `everything-claude-code:planner` | 冲刺规划师 | 将每个阶段分解为具体任务 | 小 |
| **TDD** | `everything-claude-code:tdd-guide` | 测试工程师 | 先写测试（RED → GREEN） | 中 |
| **BUILD** | `everything-claude-code:pytorch-build-resolver` | 构建工程师 | 修复 PyTorch/CUDA/TRT 错误（on-call） | 小 |
| **PY-REV** | `everything-claude-code:python-reviewer` | Python 审查员 | 实现后代码质量审查 | 中 |
| **CODE-REV** | `superpowers:code-reviewer` | 阶段门审查员 | 对照计划做阶段完成度审查 | 中 |
| **DOC** | `everything-claude-code:doc-updater` | 文档工程师 | 每阶段完成后更新文档 | 小 |

---

## 3. 上下文窗口管理策略

### 3.1 PM 会话保护规则
```
SESSION START 协议（每次新 session 必须执行）：
  1. 读取 docs/quant/PROGRESS.md → 恢复当前状态
  2. 读取 docs/quant/DECISIONS.md → 了解已做的架构决策
  3. 不直接读取大型源码文件 → 派遣 EXPLORE agent 代劳
  4. 当前 session 上下文 > 60% 时 → 将中间结论写入文件再继续
```

### 3.2 Agent 上下文隔离原则
```
每个 Agent 调用遵循：
  输入：docs/quant/AGENT_BRIEFS.md 中的对应简报（< 3k tokens）
       + 需要读取的具体文件路径列表
  输出：写入 docs/quant/outputs/<phase>/<task>.md
  
PM 不直接读取 Agent 的实现代码 → 只读取输出的 .md 摘要文件
```

### 3.3 阶段间状态传递
```
阶段 N 完成 → CODE-REV 审查 → PM 更新 PROGRESS.md → 阶段 N+1 的 agent 读取 PROGRESS.md
所有关键决策写入 DECISIONS.md（永久有效）
所有产出物写入 docs/quant/outputs/（持久化）
```

---

## 4. 工作流架构

```
┌─────────────────────────────────────────────────────────────────┐
│  SESSION START:  PM reads PROGRESS.md + DECISIONS.md           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │  Phase A: 量化基础设施移植（串行）           │
         │  派遣：PLAN → TDD → (实现) → PY-REV        │
         │  产出：quant/ 包 + smoke test 通过          │
         └─────────────────────┬─────────────────────┘
                               │ CODE-REV 门控
         ┌─────────────────────▼─────────────────────┐
         │  Phase B: QuantBlock 实现（3路并行）         │
         │  B1 ──── ARCH brief → TDD → 实现 → PY-REV  │
         │  B2 ──── ARCH brief → TDD → 实现 → PY-REV  │
         │  B3 ──── ARCH brief → TDD → 实现 → PY-REV  │
         │  产出：quant_bevformer/fusion/downstream.py  │
         └─────────────────────┬─────────────────────┘
                               │ CODE-REV 门控
         ┌─────────────────────▼─────────────────────┐
         │  Phase C: PTQ 校准流水线（串行）             │
         │  C1: dump_univ2x_calibration.py             │
         │  C2: calibrate_univ2x.py                    │
         │  派遣：PLAN → TDD → 实现 → PY-REV           │
         └─────────────────────┬─────────────────────┘
                               │ CODE-REV 门控
         ┌─────────────────────▼─────────────────────┐
         │  Phase D: INT8 TRT 引擎构建（串行）          │
         │  D1: build_trt_int8_univ2x.py               │
         │  on-call: BUILD（遇错时触发）               │
         │  产出：3 个 *_int8.trt 引擎文件             │
         └─────────────────────┬─────────────────────┘
                               │
         ┌─────────────────────▼─────────────────────┐
         │  Phase E: 端到端验证（2路并行）              │
         │  E1: 模块级精度验证（cosine 指标）           │
         │  E2: 端到端 AMOTA 验证                      │
         │  产出：实验报告 → result.log 更新            │
         └─────────────────────┬─────────────────────┘
                               │ DOC 更新文档
                        PROJECT COMPLETE
```

---

## 5. 阶段详细说明

### Phase A：量化基础设施移植

**目标**：将 QuantV2X 架构无关组件复制/改写到 UniV2X，通过 smoke test。

**负责 Agent 序列**：
```
Step 1: PLAN  → 输出 docs/quant/outputs/phase_A/sprint.md
Step 2: TDD   → 输出 docs/quant/outputs/phase_A/tests.py
Step 3: PM    → 执行实现（直接 Write/Edit 工具）
Step 4: PY-REV → 输出 docs/quant/outputs/phase_A/review.md
Step 5: CODE-REV → 阶段门，通过后更新 PROGRESS.md
```

**新建文件**：
```
projects/mmdet3d_plugin/univ2x/quant/
├── __init__.py
├── quant_layer.py       ← 直接复制自 QuantV2X（UniformAffineQuantizer + QuantModule）
├── adaptive_rounding.py ← 直接复制自 QuantV2X（AdaRoundQuantizer）
├── fold_bn.py           ← 复制 + 移除 spconv 依赖块
├── quant_model.py       ← 复制 + 移除 opencood_specials 引用
├── data_utils.py        ← 复制 + 适配 mmcv DataLoader 格式
└── layer_recon.py       ← 直接复制自 QuantV2X
```

**验收标准**：
```python
# docs/quant/outputs/phase_A/smoke_test.py
from projects.mmdet3d_plugin.univ2x.quant import (
    QuantModule, UniformAffineQuantizer, AdaRoundQuantizer
)
# 所有导入成功 + W8A8 forward pass 无报错
```

---

### Phase B：UniV2X 特定 QuantBlock（3路并行）

**负责 Agent 序列**（三路同时派遣）：
```
B1/B2/B3 各自：
  Step 1: ARCH  → 读取 DECISIONS.md + 对应源码 → 输出接口规范
  Step 2: TDD   → 读取 ARCH 输出 → 输出单元测试
  Step 3: PM    → 执行实现
  Step 4: PY-REV → 代码审查
合并后：CODE-REV 对 3 个文件同时审查
```

**B1 产出**：`projects/mmdet3d_plugin/univ2x/quant/quant_bevformer.py`
- `QuantBEVFormerLayer`：量化 FFN + value_proj/output_proj，跳过 sampling_offsets/attention_weights

**B2 产出**：`projects/mmdet3d_plugin/univ2x/quant/quant_fusion.py`
- `QuantAgentFusionMLPs`：量化 4 个 MLP Linear，跳过坐标变换
- `QuantLaneFusionMLPs`：同上结构

**B3 产出**：`projects/mmdet3d_plugin/univ2x/quant/quant_downstream.py`
- `QuantMotionMLP`：量化预测 MLP，跳过 MSDA
- `QuantOccConvs`：量化 Conv2d/ConvTranspose2d
- `QuantPlanningFC`：量化 FC 层

**关键量化边界决策**（写入 DECISIONS.md）：

| 层 | 决策 | 原因 |
|----|------|------|
| `sampling_offsets` Linear | ❌ FP16 | 输出直接进 MSDAPlugin，INT8 量化误差导致 attention 崩溃 |
| `attention_weights` Linear | ❌ FP16 | 同上 |
| FFN Linear（GELU 后） | ✅ INT8 | 激活分布集中（后 GELU 通常为正值） |
| `value_proj` / `output_proj` | ✅ INT8 | 不直接输入 plugin，可量化 |
| `cross_agent_align` MLP | ✅ INT8 | 纯 MLP，分布稳定 |

---

### Phase C：PTQ 校准流水线

**负责 Agent 序列**：
```
Step 1: PLAN  → 拆分 C1/C2 子任务，输出 sprint.md
Step 2: TDD   → 写 C1/C2 测试
Step 3: PM    → 实现 C1（dump_univ2x_calibration.py）
Step 4: PY-REV → 审查 C1
Step 5: PM    → 实现 C2（calibrate_univ2x.py）
Step 6: PY-REV → 审查 C2
Step 7: CODE-REV → 阶段门
```

**C1 产出**：`tools/dump_univ2x_calibration.py`
- 拦截 3 个 TRT 子图（BEV encoder、V2X heads、downstream）的输入
- 保存为 `.npz`：`calibration/{bev_encoder,heads_v2x,downstream}/sample_NNN.npz`
- 支持 `--num-cali-batches 32`

**C2 产出**：`tools/calibrate_univ2x.py`
- 关键超参（transformer 优化版）：
  ```python
  wq_params = {'n_bits': 8, 'channel_wise': True, 'scale_method': 'mse'}
  aq_params = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'entropy',
               'leaf_param': True, 'prob': 0.5}
  adaround_cfg = {'iters': 5000, 'weight': 0.01, 'T': 7.0, 'warmup': 0.2}
  ```
- 层处理顺序：FFN Linear → proj Linear → Fusion MLP → MotionHead → OccHead → PlanningHead

---

### Phase D：INT8 TRT 引擎构建

**负责 Agent 序列**：
```
Step 1: PM    → 实现 build_trt_int8_univ2x.py（基于 QuantV2X DataCalibrator）
Step 2: BUILD → on-call（遇 TRT/CUDA 错误时触发）
Step 3: PY-REV → 代码审查
Step 4: PM    → 构建 3 个 INT8 引擎（运行脚本）
Step 5: CODE-REV → 验证引擎文件大小合理
```

**D1 产出**：`tools/build_trt_int8_univ2x.py`
- 关键设计：
  ```python
  config.set_flag(trt.BuilderFlag.INT8)
  config.set_flag(trt.BuilderFlag.FP16)  # 混合精度
  # MSDAPlugin 节点强制 FP16
  for i in range(network.num_layers):
      layer = network.get_layer(i)
      if 'MSDAPlugin' in layer.name:
          layer.precision = trt.DataType.HALF
  ```
- pycuda 依赖：方案 1（推荐，`pip install pycuda`），build 时用，inference 不用

**构建命令**（产出引擎）：
```bash
# 引擎 1: BEV 编码器 INT8
python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --calibration-dir calibration/bev_encoder/ \
    --out trt_engines/univ2x_ego_bev_encoder_int8.trt

# 引擎 2: V2X 检测头 INT8 (N_PAD=1101)
python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_heads_v2x_1101.onnx \
    --calibration-dir calibration/heads_v2x/ \
    --out trt_engines/univ2x_ego_heads_v2x_1101_int8.trt

# 引擎 3: 下游头 INT8
python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_downstream.onnx \
    --calibration-dir calibration/downstream/ \
    --out trt_engines/univ2x_ego_downstream_int8.trt
```

---

### Phase E：端到端验证

**2路并行**（PM 同时派遣）：

**E1 路（模块级，EXPLORE agent 辅助）**：
```bash
python tools/validate_downstream_trt.py \
    --engine trt_engines/univ2x_ego_bev_encoder_int8.trt \
    --engine-fp32 trt_engines/univ2x_ego_bev_encoder_200_1cam.trt \
    --checkpoint ckpts/univ2x_coop_e2e_stg2.pth --model ego
# 目标：cosine > 0.998
```

**E2 路（端到端，test_trt.py 不改）**：
```bash
python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego   trt_engines/univ2x_ego_bev_encoder_int8.trt \
    --heads-engine-ego trt_engines/univ2x_ego_heads_v2x_1101_int8.trt \
    --downstream-engine trt_engines/univ2x_ego_downstream_int8.trt \
    --use-lane-trt --use-agent-trt --eval bbox
# 目标：AMOTA ≥ 0.360
```

**E 完成后**：
```
DOC agent → 更新 docs/TRT_EVAL.md + result.log
PM → 更新 PROGRESS.md 为 COMPLETED
```

---

## 6. 验收标准总表

| 阶段 | 验收指标 | 目标值 | 测量方法 |
|------|---------|--------|---------|
| A | smoke test 通过 | 全部 import + forward 无错 | `python -c "from ... import ..."` |
| B1 | QuantBEVFormerLayer 单测 | cosine(quant, fp32) > 0.999 (W8 only) | 单元测试 |
| B2 | QuantFusionMLPs 单测 | cosine > 0.9999 (W8 only) | 单元测试 |
| B3 | QuantDownstream 单测 | cosine > 0.999 (W8 only) | 单元测试 |
| C | 校准数据完整性 | 32 sample × 3 子图 npz 文件存在 | `ls calibration/*/sample_*.npz \| wc -l` ≥ 96 |
| D | BEV encoder INT8 精度 | cosine(INT8,FP32) > 0.998 | validate_downstream_trt.py |
| D | 检测头 INT8 精度 | cosine_bbox > 0.994 | validate_downstream_trt.py |
| D | 下游头 INT8 精度 | cosine_traj > 0.997 | validate_downstream_trt.py |
| E | 端到端 AMOTA | ≥ 0.360 | test_trt.py + dataset.evaluate() |
| E | 推理延迟 | 比 FP32 TRT 降低 ≥ 20% | torch.cuda.Event 计时 |
| E | GPU 内存峰值 | 比 FP32 TRT 降低 ≥ 15% | nvidia-smi |

---

## 7. 风险矩阵与降级策略

| 风险 | 概率 | 影响 | 降级策略 |
|------|------|------|---------|
| MSDAPlugin INT8 边界崩溃 | 中 | 高 | BEV encoder 整体保持 FP16，只量化检测头+下游头 |
| Transformer W8A8 精度崩溃 | 中 | 高 | 回退到 W8A16（仅权重 INT8，激活 FP16） |
| N_PAD=1101 零填充污染加剧 | 中 | 中 | V2X 检测头保持 FP16，其余 INT8 |
| pycuda 安装失败 | 低 | 中 | 方案 2：torch tensor data_ptr() 传指针 |
| 校准数据不足（分布覆盖差） | 低 | 中 | 增加 batch 至 64，混合训练/验证集数据 |

**降级优先级**（精度无法达标时依次尝试）：
```
W8A8 全模型 INT8
  → W8A16 全模型（仅权重量化）
  → W8A8 仅下游头
  → W8A8 仅下游头 + W8A16 BEV encoder
```

---

## 8. 快速启动路径（3 天 MVP）

如需最快看到 INT8 结果：

```
Day 1:  阶段 A（基础设施移植）+ 阶段 B3（QuantDownstreamHeads）
Day 2:  阶段 C（仅 downstream 校准）+ 阶段 D（仅构建 downstream INT8 引擎）
Day 3:  阶段 E 验证下游头 INT8

预期：下游头 INT8 对 AMOTA 影响 < 0.003（下游头非检测关键路径）
      推理延迟 Motion+Occ+Planning 部分降低 ~30%
```

---

## 9. 文件清单

### 新建（共 13 个）

| 文件 | 来源 | Phase |
|------|------|-------|
| `projects/.../quant/__init__.py` | 新写 | A |
| `projects/.../quant/quant_layer.py` | 复制自 QuantV2X | A |
| `projects/.../quant/adaptive_rounding.py` | 复制自 QuantV2X | A |
| `projects/.../quant/fold_bn.py` | 复制 + 去 spconv | A |
| `projects/.../quant/quant_model.py` | 改写自 QuantV2X | A |
| `projects/.../quant/data_utils.py` | 改写自 QuantV2X | A |
| `projects/.../quant/layer_recon.py` | 复制自 QuantV2X | A |
| `projects/.../quant/quant_bevformer.py` | 新写 | B1 |
| `projects/.../quant/quant_fusion.py` | 新写 | B2 |
| `projects/.../quant/quant_downstream.py` | 新写 | B3 |
| `tools/dump_univ2x_calibration.py` | 新写 | C |
| `tools/calibrate_univ2x.py` | 新写 | C |
| `tools/build_trt_int8_univ2x.py` | 改写自 QuantV2X | D |

### 修改（共 1 个）

| 文件 | 改动 | Phase |
|------|------|-------|
| `tools/validate_downstream_trt.py` | 增加 `--engine-fp32` 双引擎对比模式 | E |

### 不修改（硬性约束）

`test_trt.py`、`export_onnx_univ2x.py`、`build_trt_downstream.py`、所有 `*_trt*.py`、所有 config

---

## 10. 参考资料

| 资料 | 路径 |
|------|------|
| QuantV2X 量化核心 | `/home/jichengzhi/QuantV2X/opencood/quant/` |
| UniV2X TRT 文档 | `docs/TRT_EVAL.md` |
| 端到端测试结果 | `result.log` |
| Agent 简报库 | `docs/quant/AGENT_BRIEFS.md` |
| 状态看板 | `docs/quant/PROGRESS.md` |
| 架构决策记录 | `docs/quant/DECISIONS.md` |
| AdaRound 论文 | https://arxiv.org/abs/2004.10568 |
