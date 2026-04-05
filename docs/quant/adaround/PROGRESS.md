# AdaRound 优化 — 状态看板

> **每次 session 开始必读此文件**
> 最后更新：2026-04-04
> 当前负责人：PM（主 Claude 会话）

---

## 当前阶段

```
[ Phase A ] ── [ Phase B ] ── [ Phase C ] ── [ Phase D ]
  ✅ 完成       ✅ 完成       ✅ 完成       ✅ 完成（结论：失败）
```

**当前状态**：`EXPERIMENT_CONCLUDED` — 全部阶段完成。AdaRound fake-quant ONNX 路径因双重量化问题导致 AMOTA=0.190，结论：需改用 Q/DQ ONNX 节点正确部署。

---

## 已有基础（Phase E 遗留产出，可直接复用）

| 产出物 | 路径 | 说明 |
|-------|------|------|
| 50帧校准数据 | `calibration/bev_encoder_calib_inputs.pkl` | ~4 GB，含temporal prev_bev |
| 现有INT8引擎 | `trt_engines/univ2x_ego_bev_encoder_int8.trt` | 43 MB，vanilla PTQ，AMOTA 0.364 |
| 引擎构建脚本 | `tools/build_trt_int8_univ2x.py` | 可直接用，只需换权重 |
| 验证脚本 | `tools/validate_quant_bev.py` | 可直接用 |
| 端到端测试 | `tools/test_trt.py` | 不改动 |
| AdaRound核心 | `projects/.../quant/adaptive_rounding.py` | 已移植 |
| 层重建核心 | `projects/.../quant/layer_recon.py` | 已移植 |
| QuantBEVFormer | `projects/.../quant/quant_bevformer.py` | 已实现 |

---

## Kanban 看板

### ⬜ 待开始

| ID | 任务 | 负责 Agent | 依赖 | 输出文件 |
|----|------|-----------|------|---------|
| ~~A-1~~ | ~~读取API~~ | ~~EXPLORE~~ | — | `outputs/phase_A/api_summary.md` ✅ |
| ~~A-2~~ | ~~架构决策~~ | ~~ARCH~~ | A-1 | `outputs/phase_A/arch_decision.md` ✅ |
| ~~B-1~~ | ~~实现 export_onnx_adaround.py~~ | ~~PM~~ | A-2 | `tools/export_onnx_adaround.py` ✅ |
| ~~B-2~~ | ~~PY-REV 审查~~ | ~~PY-REV~~ | B-1 | `outputs/phase_B/review.md` ✅ |
### ✅ 全部完成

| ID | 任务 | 完成时间 | 产出 | 结果 |
|----|------|---------|------|------|
| A-1 | 读取 layer_recon+adaptive_rounding API | 2026-04-02 | `outputs/phase_A/api_summary.md` | ✅ |
| A-2 | 架构决策：AdaRound集成方案 | 2026-04-02 | `outputs/phase_A/arch_decision.md` | ✅ |
| B-1 | 实现 export_onnx_adaround.py | 2026-04-02 | `tools/export_onnx_adaround.py` | ✅ |
| B-2 | PY-REV 审查（CRITICAL×2 已修复） | 2026-04-02 | `outputs/phase_B/review.md` | ✅ |
| C-1 | AdaRound 校准（10 样本，500 iter/层） | 2026-04-04 | `calibration/quant_encoder_adaround.pth`（31 MB，36 层） | ✅ |
| C-2 | 导出 AdaRound ONNX（W_fq 内嵌） | 2026-04-04 | `onnx/univ2x_ego_bev_encoder_adaround.onnx`（104.2 MB） | ✅ |
| C-3 | 构建 INT8 TRT 引擎 | 2026-04-04 | `trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt`（43.7 MB） | ✅ |
| D-1 | 模块级 cosine 验证 | 2026-04-04 | 平均 cosine = 0.819 | ❌ < 0.99 |
| D-2 | 端到端 AMOTA 验证 | 2026-04-04 | **AMOTA = 0.190** | ❌ < 0.370 |
| D-3 | 更新 quant_result.md | 2026-04-04 | `docs/quant/quant_result.md` 第六节 | ✅ |

---

## 精度目标与实际结果

| 配置 | AMOTA | 说明 |
|------|-------|------|
| FP16 TRT baseline | 0.370 | 目标基准 |
| Vanilla PTQ INT8 (Hook-A) | **0.381** | 已超越 FP16 ✅ |
| Vanilla PTQ INT8 (v5, 旧结果) | 0.364 | 旧校准结果 |
| **AdaRound INT8（本次实验）** | **0.190** | ❌ 双重量化导致严重下降 |

---

## 失败根因

**双重权重量化（Double Weight Quantization）**：

AdaRound fake-quant ONNX 路径将 W_fq（已在 INT8 网格上的 FP32 值）内嵌到 ONNX 中。TRT 构建 INT8 引擎时从 W_fq 重新推导权重 scale（scale_trt ≠ scale_ada），对 W_fq 再次量化，覆盖了 AdaRound 的舍入决策。

**正确方案**：在 ONNX 中插入显式 `QuantizeLinear + DequantizeLinear` 节点（Q/DQ 风格），指定 scale_ada，TRT 会使用这些 scale 直接调度 INT8 内核而不再重新推导。

---

## Session 日志

| 日期 | 完成任务 | 下次继续 |
|------|---------|---------|
| 2026-04-02 | 创建AdaRound计划文档体系 | 执行Phase A（从A-1开始） |
| 2026-04-02 | Phase A+B 全部完成：EXPLORE→ARCH→实现→PY-REV | 执行 Phase C：运行 `calibrate_univ2x.py --adaround` |
| 2026-04-03~04 | Phase C+D 全部完成，修复7个阻塞bug，验证完成 | 实验结论：双重量化失败，改用 Q/DQ ONNX 方案 |
