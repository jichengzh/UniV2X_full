# UniV2X AdaRound 优化 — Multi-Agent 协作计划

> **PM**：主 Claude 会话（Sonnet 4.6）  
> **状态看板**：`docs/quant/adaround/PROGRESS.md` ← **每次 session 开始必读**  
> **Agent 简报库**：`docs/quant/adaround/AGENT_BRIEFS.md`  
> **架构决策**：`docs/quant/adaround/DECISIONS.md`  
> **Agent 输出**：`docs/quant/adaround/outputs/<phase>/`

---

## 0. 背景与目标

### 当前状态

| 配置 | AMOTA | 引擎大小 |
|------|-------|---------|
| FP16 TRT 全链路 baseline | 0.370 | 296 MB |
| Vanilla PTQ INT8 (v5, 50样本) | 0.364 | 43 MB (BEV) |
| **AdaRound INT8 目标** | **≥ 0.370** | ≤ 43 MB |

### AdaRound 原理（一句话）

AdaRound 在标准 PTQ scale 校准完成后，对每一个可量化层的权重进行**自适应舍入策略优化**（而非固定四舍五入），通过最小化输出重建误差学习每个权重的最佳舍入方向，通常带来 +0.010~0.020 AMOTA 提升。

### 可复用的已有产出

```
calibration/bev_encoder_calib_inputs.pkl   ← 50帧 temporal 校准数据（~4 GB）
trt_engines/univ2x_ego_bev_encoder_int8.trt ← vanilla PTQ 引擎（对照基准）
tools/build_trt_int8_univ2x.py             ← 引擎构建（直接复用）
tools/validate_quant_bev.py                ← 模块级验证（直接复用）
projects/.../quant/adaptive_rounding.py    ← AdaRoundQuantizer（已移植）
projects/.../quant/layer_recon.py          ← 层重建核心（已移植）
projects/.../quant/quant_bevformer.py      ← QuantBlock 定义（已实现）
```

### 硬性约束（继承自 plan_quantv2x.md）

1. **不修改** `tools/test_trt.py`（4个 hook 接口不变）
2. **不修改** 任何 `*_trt*.py` 模型文件
3. **不修改** 任何 config 文件
4. **不修改** `tools/calibrate_univ2x.py`（保留 vanilla PTQ fallback）

---

## 1. Agent 角色分配

| 代号 | Agent 类型 | 角色 | 上下文预算 |
|------|-----------|------|-----------|
| **PM** | 主 Claude 会话 | 编排 + 实现 | 全量（保护 > 60% 阈值） |
| **EXPLORE** | `Explore` | API 摘要（读源码，节省 PM 上下文） | < 6k tokens 输入 |
| **ARCH** | `everything-claude-code:architect` | 集成方案设计 | < 5k tokens 输入 |
| **PY-REV** | `everything-claude-code:python-reviewer` | 实现后审查 | < 6k tokens 输入 |
| **BUILD** | `everything-claude-code:pytorch-build-resolver` | 错误修复（on-call） | 按需，最小 |

> **未使用** TDD、CODE-REV、DOC agents：  
> AdaRound 是对现有基础设施的扩展，已有 validate_quant_bev.py 作为验收脚本，  
> 额外 TDD 开销超过收益；DOC 更新由 PM 直接执行。

---

## 2. 上下文窗口管理策略

```
SESSION START 协议（每次新 session 必须执行，< 5 次工具调用）：
  1. Read docs/quant/adaround/PROGRESS.md          ← 恢复状态
  2. Read docs/quant/adaround/DECISIONS.md          ← 了解决策
  3. 按 Kanban 中 "🔄 进行中" 的任务 ID 继续
  4. 不直接读取大型源码 → 派遣 EXPLORE 代劳

AGENT 隔离原则：
  输入 = AGENT_BRIEFS.md 对应简报（< 3k tokens）+ 指定文件路径
  输出 = docs/quant/adaround/outputs/<phase>/<task>.md
  PM 只读 .md 输出，不读 agent 处理过的源码

上下文 > 60% 时：
  写入中间结论到 outputs/ 目录，再继续下一步
```

---

## 3. 工作流

```
SESSION START: PM reads PROGRESS.md + DECISIONS.md
                          │
        ┌─────────────────▼──────────────────┐
        │  Phase A: API 分析 + 架构决策（串行） │
        │  A-1: EXPLORE → api_summary.md      │  ~10 min
        │  A-2: ARCH   → arch_decision.md     │  ~15 min
        └─────────────────┬──────────────────┘
                          │ PM 读 arch_decision.md（< 1k tokens）
        ┌─────────────────▼──────────────────┐
        │  Phase B: 实现 calibrate_adaround   │
        │  B-1: PM 实现（参考 arch_decision）  │  ~30 min
        │  B-2: PY-REV → review.md            │  ~10 min
        │  B-3: PM 修复 review 中的 CRITICAL  │  ~10 min
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  Phase C: 运行 AdaRound 校准         │
        │  C-1: PM 执行脚本（~30 min GPU 运行）│
        │  C-2: PM 重建 INT8 引擎（~10 min）   │
        │  on-call: BUILD（遇错时触发）        │
        └─────────────────┬──────────────────┘
                          │
        ┌─────────────────▼──────────────────┐
        │  Phase D: 验证 + 文档更新            │
        │  D-1: 模块级 cosine 验证             │
        │  D-2: 端到端 AMOTA 验证              │
        │  D-3: PM 更新 quant_result.md       │
        │  D-4: PM 更新 PROGRESS.md           │
        └─────────────────────────────────────┘
                   PROJECT COMPLETE
```

---

## 4. 阶段详细说明

### Phase A：API 分析 + 架构决策

**目的**：在不消耗 PM 大量上下文的前提下，搞清楚 `layer_recon.py` 和 `adaptive_rounding.py` 的调用接口，并设计 `calibrate_univ2x_adaround.py` 的结构。

#### A-1：EXPLORE agent（参考 BRIEF-A1）

```
派遣命令（PM 执行）：
  subagent_type: Explore
  prompt: 见 docs/quant/adaround/AGENT_BRIEFS.md 中的 BRIEF-A1
  run_in_background: false（需要输出才能继续）
```

**验收**：`docs/quant/adaround/outputs/phase_A/api_summary.md` 存在且包含：
- AdaRoundQuantizer 构造参数
- layer_reconstruction() 签名
- quant_bevformer.py 中 skip 层列表

#### A-2：ARCH agent（参考 BRIEF-A2）

```
派遣命令（PM 执行）：
  subagent_type: everything-claude-code:architect
  prompt: 见 docs/quant/adaround/AGENT_BRIEFS.md 中的 BRIEF-A2
  run_in_background: false
```

**验收**：`docs/quant/adaround/outputs/phase_A/arch_decision.md` 包含：
- calibrate_univ2x_adaround.py 伪代码
- layer_reconstruction() 调用示例
- ADR-004 的具体答案
- 已知风险 ≤ 3 条

**A-2 完成后**：PM 读取 arch_decision.md（< 1k tokens），更新 DECISIONS.md 中的 ADR-004。

---

### Phase B：实现 calibrate_univ2x_adaround.py

**B-1：PM 直接实现**

参考 arch_decision.md 伪代码，新建 `tools/calibrate_univ2x_adaround.py`。

**核心流程（对应伪代码）**：

```python
# Step 1: 加载模型 + 构建 QuantModel
model = build_model(cfg, checkpoint)
register_bevformer_specials()          # 从 quant_bevformer.py
qmodel = QuantModel(model.pts_bbox_head.bev_encoder, ...)

# Step 2: Activation scale calibration（复用现有逻辑）
qmodel.set_quant_state(weight_quant=False, act_quant=True)
with torch.no_grad():
    for data in calib_loader:          # calib_inputs.pkl 前32帧
        qmodel(data['feats'], ...)

# Step 3: Weight scale calibration（MSE，channel-wise）
qmodel.set_quant_state(weight_quant=True, act_quant=False)
# 触发 init_delta_from_weights()

# Step 4: AdaRound 层重建（核心新增）
qmodel.set_quant_state(weight_quant=True, act_quant=False)
for name, module in qmodel.named_modules():
    if isinstance(module, QuantBEVFormerLayer):  # 逐 block 重建
        layer_reconstruction(module, ...)         # 来自 layer_recon.py

# Step 5: 保存 AdaRound 权重
torch.save(qmodel.state_dict(), 'calibration/quant_encoder_adaround.pth')
```

**运行命令（写入 PROGRESS.md 的 C-1 节）**：
```bash
conda run -n UniV2X_2.0 python tools/calibrate_univ2x_adaround.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --out calibration/quant_encoder_adaround.pth \
    --iters 5000 --warmup 0.2 --lam 0.01
```

**B-2：PY-REV agent（参考 BRIEF-B2）**

```
派遣命令（PM 执行）：
  subagent_type: everything-claude-code:python-reviewer
  prompt: 见 docs/quant/adaround/AGENT_BRIEFS.md 中的 BRIEF-B2
  run_in_background: false
```

**验收**：`docs/quant/adaround/outputs/phase_B/review.md` 存在；PM 修复所有 CRITICAL 级问题。

---

### Phase C：运行校准 + 构建引擎

**C-1：运行 AdaRound 校准**

```bash
# 预期耗时：30~60 min（每层 5000 iters × ~40 层）
# 监控：watch -n 5 nvidia-smi
conda run -n UniV2X_2.0 python tools/calibrate_univ2x_adaround.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --out calibration/quant_encoder_adaround.pth \
    --iters 5000 --warmup 0.2 --lam 0.01
```

**on-call BUILD**：若出现 CUDA OOM，派遣 BUILD agent（提供完整 traceback）。  
降级策略：`--iters 2000` 或 `--batch-size 4`。

**C-2：重建 INT8 引擎**

```bash
# 复用现有引擎构建脚本，传入 AdaRound 权重
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --out trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt \
    --target bev_encoder \
    --plugin plugins/build/libuniv2x_plugins.so \
    --quant-weights calibration/quant_encoder_adaround.pth
```

---

### Phase D：验证 + 文档更新

**D-1：模块级 cosine 验证**

```bash
conda run -n UniV2X_2.0 python tools/validate_quant_bev.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --quant-weights calibration/quant_encoder_adaround.pth \
    --n-samples 10
# 目标：cosine > 0.997（高于 vanilla PTQ 的 0.9947）
```

结果写入 `docs/quant/adaround/outputs/phase_D/cosine_result.md`。

**D-2：端到端 AMOTA 验证**

```bash
conda run -n UniV2X_2.0 python tools/test_trt.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --bev-engine-ego trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt \
    --plugin plugins/build/libuniv2x_plugins.so \
    --use-lane-trt --use-agent-trt \
    --eval bbox
# 目标：AMOTA ≥ 0.370（持平 FP16 baseline）
```

**D-3：文档更新（PM 直接执行）**

- 在 `docs/quant/quant_result.md` 精度表新增 AdaRound 行
- 更新 `result.log`（新增 Phase AdaRound 节）
- 更新 `docs/quant/adaround/PROGRESS.md` 为 `COMPLETE`

---

## 5. 验收标准

| 阶段 | 指标 | 目标 |
|------|------|------|
| A-1 | api_summary.md 存在 | ✅ |
| A-2 | arch_decision.md 含伪代码 + ADR-004 答案 | ✅ |
| B-1 | calibrate_univ2x_adaround.py 存在 | ✅ |
| B-2 | review.md 无 CRITICAL 问题（或已修复） | ✅ |
| C-1 | quant_encoder_adaround.pth 存在，> 10 MB | ✅ |
| C-2 | adaround_int8.trt 存在，大小 40~50 MB | ✅ |
| D-1 | cosine(AdaRound, FP32) > 0.997 | ✅ |
| D-2 | **AMOTA ≥ 0.370** | ✅ |

---

## 6. 降级策略

若 D-2 AMOTA < 0.370（AdaRound 未超越 vanilla PTQ）：

```
尝试 1: 增加 iters 至 10000（重新运行 C-1）
尝试 2: 改为 block-wise 重建（BEVFormerEncoderTRT 整体，而非逐 layer）
尝试 3: 混合策略 — AdaRound 仅用于 FFN，value/output_proj 保留 vanilla PTQ
最终降级: 保留 vanilla PTQ 引擎（AMOTA 0.364），记录 AdaRound 未带来额外收益
```

---

## 7. 关键文件索引

| 文件 | 用途 | 是否存在 |
|------|------|:-------:|
| `docs/quant/adaround/PROGRESS.md` | 状态看板（session开始必读） | ✅ |
| `docs/quant/adaround/DECISIONS.md` | 架构决策记录 | ✅ |
| `docs/quant/adaround/AGENT_BRIEFS.md` | Agent 派遣简报 | ✅ |
| `docs/quant/adaround/outputs/phase_A/api_summary.md` | EXPLORE 输出 | ⬜ |
| `docs/quant/adaround/outputs/phase_A/arch_decision.md` | ARCH 输出 | ⬜ |
| `docs/quant/adaround/outputs/phase_B/review.md` | PY-REV 输出 | ⬜ |
| `docs/quant/adaround/outputs/phase_D/cosine_result.md` | 模块级验证结果 | ⬜ |
| `tools/calibrate_univ2x_adaround.py` | 新建（核心实现） | ⬜ |
| `calibration/quant_encoder_adaround.pth` | AdaRound 权重 | ⬜ |
| `trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt` | AdaRound 引擎 | ⬜ |
| `tools/calibrate_univ2x.py` | Vanilla PTQ（不修改，作为对照） | ✅ |
| `tools/build_trt_int8_univ2x.py` | 引擎构建（复用） | ✅ |
| `tools/validate_quant_bev.py` | 模块级验证（复用） | ✅ |

---

## 8. 快速启动路径（单 session）

```
Step 1: Read docs/quant/adaround/PROGRESS.md + DECISIONS.md   (< 2 min)
Step 2: 派遣 EXPLORE (A-1) — background=false                 (~ 5 min)
Step 3: 读 api_summary.md，派遣 ARCH (A-2) — background=false (~ 5 min)
Step 4: 读 arch_decision.md，PM 实现 B-1                       (~ 20 min)
Step 5: 派遣 PY-REV (B-2) — background=false                  (~ 5 min)
Step 6: 修复 CRITICAL，运行 C-1（后台 GPU 任务）               (~ 30~60 min)
Step 7: 运行 C-2 构建引擎，运行 D-1/D-2 验证                  (~ 15 min)
Step 8: 更新文档                                               (~ 5 min)
```

---

*创建于：2026-04-02*  
*参考：plan_quantv2x.md、docs/quant/quant_result.md*
