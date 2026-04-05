# 架构决策记录（Architecture Decision Records）

> 每个已确认的技术决策记录在此。实施 agent 必须遵守。

---

## ADR-001：MSDAPlugin 相邻 Linear 不量化

**状态**：已确认 ✅  
**日期**：2026-04-01  
**决策者**：PM

**决策**：`sampling_offsets` 和 `attention_weights` 两个 Linear 层不量化（保持 FP16）。

**原因**：
- 这两个 Linear 的输出直接作为 MSDAPlugin 的输入（采样偏移量和注意力权重）
- TRT INT8 calibration 无法跨越 custom plugin 边界做融合
- 在 custom plugin 输入端引入 INT8 量化误差，会导致采样位置偏移，造成 attention 精度显著下降
- 实验（参考 QuantV2X 中类似 MSDA 场景）表明此类边界量化导致 cosine < 0.95

**影响范围**：
- `quant_bevformer.py`：SCA 的 `sampling_offsets_proj`、`attention_weights_proj` 保持 `nn.Linear`
- `quant_downstream.py`：MotionHead 内 MSDA 相关 Linear 同样保持

---

## ADR-002：激活量化使用 entropy 方法

**状态**：已确认 ✅  
**日期**：2026-04-01  
**决策者**：PM

**决策**：激活量化 (`aq_params`) 使用 `scale_method='entropy'` 而非 `minmax`。

**原因**：
- UniV2X 的 Transformer 激活值分布呈长尾（attention score 可达 ±50，FFN 输出集中在 ±5）
- `minmax` 会被极端值拉大量化范围，有效精度仅 6-7 bit
- `entropy`（KL 散度最小化）自动找到最优截断点，适合长尾分布

**超参**：`{'n_bits': 8, 'scale_method': 'entropy', 'channel_wise': False, 'leaf_param': True}`

---

## ADR-003：pycuda 方案选择

**状态**：待确认 ⚠️（Phase D 开始前确认）  
**候选方案**：

**方案 1（推荐）**：引入 pycuda
- 优点：直接复用 QuantV2X 的 DataCalibrator 实现，风险低
- 缺点：新增依赖
- 安装：`conda run -n UniV2X_2.0 pip install pycuda`

**方案 2**：无 pycuda，用 torch tensor data_ptr()
- 优点：无新依赖
- 缺点：需要自行管理 GPU 内存生命周期，实现复杂

**决定原则**：Phase D-1 开始时，先尝试方案 1。若安装失败（CUDA 版本冲突），切换方案 2。

---

## ADR-004：TRT 混合精度策略

**状态**：已确认 ✅  
**日期**：2026-04-01

**决策**：构建 INT8 引擎时同时开启 FP16 flag，并对 MSDAPlugin 显式强制 FP16。

```python
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if any(p in layer.name for p in ['MSDAPlugin', 'RotatePlugin', 'InversePlugin']):
        layer.precision = trt.DataType.HALF
        layer.set_output_type(0, trt.DataType.HALF)
```

**原因**：
- INT8 + FP16 允许 TRT 对普通 Linear/Conv 用 INT8，对 custom plugin 自动回退 FP16
- 不设 FP16 时，plugin 层可能回退 FP32，反而更慢

---

## ADR-005：量化粒度选择

**状态**：已确认 ✅  
**日期**：2026-04-01

| 量化对象 | 权重粒度 | 激活粒度 | 原因 |
|---------|---------|---------|------|
| Linear (FFN, proj) | per-channel (channel_wise=True) | per-tensor | 权重 per-channel 精度更高；激活 per-tensor 避免 transformer 动态范围过于复杂 |
| Conv2d (OccHead) | per-channel | per-tensor | 与 Linear 保持一致 |
| 最后 3 个 Linear (cls/reg head) | 8-bit per-channel | 不量化激活 | 参考 QuantV2X `disable_network_output_quantization` 策略 |

---

## ADR-006：降级触发条件

**状态**：已确认 ✅  
**日期**：2026-04-01

如果 Phase E 端到端验证 AMOTA < 0.360（低于目标），按以下顺序降级：

```
Level 0（目标）：全模型 W8A8 INT8
  → AMOTA < 0.360 →
Level 1：W8A16（仅权重 INT8，激活 FP16）
  → AMOTA < 0.360 →
Level 2：仅下游头 INT8，BEV encoder + 检测头 FP16
  → AMOTA < 0.360 →
Level 3：仅下游头 W8A16（最保守，几乎不影响精度）
```

---

## ADR-007：不修改现有文件的硬性约束

**状态**：已确认 ✅（高优先级，不可违反）

以下文件任何 agent 都**不得修改**：
- `tools/test_trt.py`
- `tools/export_onnx_univ2x.py`
- `tools/build_trt_downstream.py`
- 所有 `projects/configs_e2e_univ2x/*.py`
- 所有 `*_trt*.py` 模型变体文件

违反此约束的任何 agent 输出将被 CODE-REV 拒绝。
