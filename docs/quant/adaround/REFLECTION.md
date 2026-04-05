# AdaRound 量化实验反思文档

**日期**：2026-04-04  
**状态**：实验结论总结，指导后续 Q/DQ 正确实现

---

## 一、回答核心问题

### 1.1 Q/DQ 是 TRT Modelopt 库专属方案吗？

**不是。** Q/DQ（`QuantizeLinear + DequantizeLinear`）是 **ONNX opset 13+ 的标准算子**，与任何特定库无关。TRT 原生支持识别和融合 Q/DQ 节点为真正的 INT8 内核。

NVIDIA ModelOpt 只是众多能输出 Q/DQ ONNX 的工具之一，同类工具还有：
- PyTorch `torch.ao.quantization`（FX 图模式 QAT 导出）
- `torch.fake_quantize_per_channel_affine` + 自定义 ONNX symbolic
- 手工 ONNX 图编辑（ONNX Python API）

本项目缺失的就是**最后一项**：把 AdaRound 校准得到的 `delta` 和 `zero_point` 写入 ONNX Q/DQ 节点的这一步。

### 1.2 Q/DQ 能真正加速计算吗？

**可以。** 当 TRT 在 ONNX 中识别到标准 Q/DQ+Gemm 或 Q/DQ+Conv 融合模式时：

```
Weights → QuantizeLinear(scale_w) → DequantizeLinear(scale_w) ─┐
                                                                 ├─► INT8 Gemm
Input   → QuantizeLinear(scale_a) → DequantizeLinear(scale_a) ─┘
```

TRT 将整个组合融合为一个 INT8 GEMM 内核，实现真正的 INT8 计算加速（RTX 4090 上 INT8 Tensor Core 吞吐量约为 FP16 的 2 倍）。Q/DQ 节点本身在推理时被融合消除，不产生额外开销。

### 1.3 直接 Q/DQ 校准 vs AdaRound + Q/DQ，哪种更好？

**AdaRound + Q/DQ 更好**，但二者并不是对立关系：

| 方面 | 直接校准 + Q/DQ (Vanilla PTQ) | AdaRound + Q/DQ |
|------|:----------------------------:|:---------------:|
| 权重舍入决策 | 最邻近舍入（round-to-nearest） | 任务感知最优舍入（最小化重建误差） |
| 校准耗时 | 快（1-2分钟） | 慢（约 2 小时，36 层 × 500 iter） |
| 精度（典型值） | 基准 | 高 +0.5%~2% accuracy |
| TRT 加速 | ✅ 真 INT8 | ✅ 真 INT8（需正确 Q/DQ 节点） |
| 实现难度 | 低（TRT 内部 Calibrator） | 中（需手工插入 Q/DQ 节点） |

**结论**：Q/DQ 是编码量化 scale 的**机制**，AdaRound 是获取**更优 scale** 的**方法**。两者完全互补，正确做法是 AdaRound 提供 scale → Q/DQ 将其编码进 ONNX → TRT 执行真 INT8。

### 1.4 能否保留 AdaRound 而舍弃 Q/DQ？

**不能**，原因如下：

TRT 构建 INT8 引擎的方法只有两种：
1. **IInt8Calibrator**：TRT 内部从校准数据重新推导所有 scale（Vanilla PTQ，不能指定外部 scale）
2. **Q/DQ ONNX 节点**：ONNX 中显式写入 scale，TRT 直接使用（可指定 AdaRound 的 scale）

若不用 Q/DQ，则只能用 IInt8Calibrator，此时 TRT 从 ONNX 的权重值（可以是 FP32 也可以是 W_fq）重新推导自己的 scale，覆盖 AdaRound 的舍入决策 → AdaRound 失效。

---

## 二、当前实现的根本错误

### 2.1 错误路径（已实施，导致 AMOTA=0.190）

```
AdaRound 校准
  alpha[i] 优化完成  →  delta_ada[i] = m.weight_quantizer.uaq.delta

save_quantized_weight()
  W_fq[i] = (round_hard(W[i] / delta_ada[i]) - zp) * delta_ada[i]   ← 第1次量化
  m.weight.data = W_fq[i]

ONNX 导出（BEVEncoderWrapper 前向）
  ONNX 节点 Gemm.weight = W_fq[i]   ← 已在 INT8 网格上的 FP32 值

TRT INT8 引擎构建（IInt8EntropyCalibrator2）
  scale_trt[i] = derive_from(W_fq[i])   ← TRT 从 W_fq 重新推导 scale
  ≠ delta_ada[i]（因为 W_fq 的分布已被截断，TRT 的 minmax/entropy 估计偏差）
  W_qq[i] = quant(W_fq[i], scale_trt)   ← 第2次量化 ← 覆盖了 AdaRound 舍入决策
```

**量化误差 = 第1次误差 + 第2次误差**（通常远大于单次量化误差）

### 2.2 正确路径（应实施）

```
AdaRound 校准
  alpha[i] 优化完成  →  delta_ada[i], zero_point_ada[i]

ONNX 导出（保留原始 FP32 权重）
  正常导出 BEVEncoderWrapper，不预先量化权重

ONNX 后处理：为每个 QuantModule 对应的 Gemm 插入 Q/DQ 节点
  for each Gemm in ONNX:
      → 在 weight 输入前插入：
         QuantizeLinear(W_fp32, scale=delta_ada, zero_point=zp_ada, axis=0)
         DequantizeLinear(W_int8, scale=delta_ada, zero_point=zp_ada, axis=0)
      → 在 input 前插入（若需要 A8）：
         QuantizeLinear(input, scale=delta_act, zero_point=zp_act)
         DequantizeLinear(Q_int8, scale=delta_act, zero_point=zp_act)

TRT INT8 引擎构建（Q/DQ 模式，无需 IInt8Calibrator）
  TRT 识别 Q/DQ+Gemm 融合模式
  直接使用 delta_ada 作为 INT8 内核 scale
  W 被量化一次，scale = delta_ada   ← 第1次也是唯一一次量化
```

### 2.3 `UniformAffineQuantizer` 为何无法直接 ONNX 导出为 Q/DQ 节点

```python
# 当前实现（forward 展开为算术 ops）
x_int = round_ste(x / self.delta) + self.zero_point   # Div + Round + Add
x_quant = torch.clamp(x_int, 0, self.n_levels - 1)    # Clip
x_dequant = (x_quant - self.zero_point) * self.delta   # Sub + Mul
```

ONNX 导出时以上代码变为 5 个独立的算术节点（Div, Round, Add, Clip, Sub, Mul），TRT **不会**将此序列识别为 INT8 量化模式。

若使用 `torch.fake_quantize_per_channel_affine`（PyTorch 内置假量化），注册对应 ONNX symbolic 后可导出为标准 `QuantizeLinear + DequantizeLinear` 节点。

---

## 三、问题清单与修复建议

### 问题 1：双重权重量化（导致 AMOTA 0.190）

| 属性 | 内容 |
|------|------|
| **严重级别** | CRITICAL |
| **影响** | AMOTA 0.381 → 0.190（-50%） |
| **原因** | fake-quant ONNX + TRT INT8 重新推导 scale → scale 不匹配 → 两次量化 |
| **修复方案** | 保留 FP32 权重，在 ONNX 后处理阶段插入 Q/DQ 节点（见第四节实现方案） |
| **文件** | `tools/export_onnx_adaround.py` |

### 问题 2：`UniformAffineQuantizer` 不能导出标准 Q/DQ

| 属性 | 内容 |
|------|------|
| **严重级别** | HIGH |
| **影响** | 所有通过 fake-quant 导出 → TRT 的路径均无法保留量化 scale |
| **原因** | 自定义算术前向函数，ONNX trace 不产生标准 QuantizeLinear 节点 |
| **修复方案 A**（推荐） | ONNX 后处理：解析图中 Gemm/Conv 节点，手工插入 Q/DQ 节点 |
| **修复方案 B** | 修改 forward 使用 `torch.fake_quantize_per_channel_affine` + 注册 ONNX symbolic |
| **文件** | `projects/.../quant/quant_layer.py` |

### 问题 3：AdaRound 校准样本数不足（10 样本 vs 推荐 1024）

| 属性 | 内容 |
|------|------|
| **严重级别** | MEDIUM |
| **影响** | alpha 参数未充分优化，rounding 决策质量低 |
| **原因** | OOM（50 样本 × 160 MB BEV embedding = ~8 GB）+ 时间限制（10 样本已需 ~2h） |
| **修复方案** | 分批 keep_gpu=False 可支持 50 样本；使用梯度累积减少内存；探索 beam search rounding |
| **期望改善** | 论文中 1024 样本在检测任务上比 10 样本提升约 0.3% AP |
| **文件** | `tools/calibrate_univ2x.py`，`--adaround-n-samples` 参数 |

### 问题 4：校准数据帧间依赖（prev_bev 序列性）

| 属性 | 内容 |
|------|------|
| **严重级别** | MEDIUM |
| **影响** | TSA 层的激活分布与真实推理不一致 |
| **原因** | 10 帧校准数据没有保证时序连续性（帧间 prev_bev 不一致） |
| **修复方案** | 收集连续时序片段；BEVEncoderCalibModel 应在帧间传递真实 bev_embed |
| **文件** | `tools/calibrate_univ2x.py::BEVEncoderCalibModel` |

### 问题 5：`validate_adaround_bev.py` 模型加载 Bug（MultiAgent 加载错误）

| 属性 | 内容 |
|------|------|
| **严重级别** | LOW（已修复） |
| **影响** | 验证时 cosine=-0.09（两个随机权重模型对比）误判为 AdaRound 本身失败 |
| **原因** | `load_checkpoint(MultiAgent, ckpt, revise_keys=strip_ego_prefix)` 导致 key 不匹配，模型载入随机权重 |
| **修复** | 使用 `build_model_from_cfg(cfg, 'model_ego_agent', ckpt_path)` 直接加载 ego 模型（已修复） |
| **文件** | `tools/validate_adaround_bev.py` |

### 问题 6：`weight_quantizer.delta` 未保存到 state_dict

| 属性 | 内容 |
|------|------|
| **严重级别** | LOW（已修复） |
| **影响** | 加载 AdaRound checkpoint 后 delta=1.0 → W_fq 在 [0,255] 范围内 → NaN |
| **原因** | `delta` 是普通 float 属性，非 `nn.Parameter` 或 `Buffer`，`state_dict()` 不保存 |
| **修复** | 加载前先调用 `set_weight_quantize_params(qmodel)` 重新初始化 delta（已修复） |
| **文件** | `tools/export_onnx_adaround.py`, `tools/validate_adaround_bev.py` |

---

## 四、正确的 AdaRound + TRT 实现方案（推荐路径）

### 4.1 总体流程

```
[现有] AdaRound 校准（calibrate_univ2x.py --adaround）
         ↓  产出：quant_encoder_adaround.pth
         ↓        含 36 × alpha（rounding 参数）
         ↓        含 36 × act_quantizer.delta（激活 scale）

[待实现] export_onnx_adaround_qdq.py
         Step 1: 构建 FP32 BEVEncoderWrapper（正常导出，不量化权重）
         Step 2: 加载 AdaRound checkpoint，提取每层的 delta_w, zp_w, delta_a, zp_a
         Step 3: 导出 FP32 ONNX（同 export_onnx_univ2x.py，含 MSDAPlugin 补丁）
         Step 4: 后处理 ONNX：为每个 Gemm 节点插入权重+激活 Q/DQ 节点对
         Step 5: 保存 QDQ ONNX（无需 IInt8Calibrator）

[已有] build_trt_int8_univ2x.py（小改：Q/DQ 模式不传 calibrator）
         trt_config.set_flag(trt.BuilderFlag.INT8)
         # 不设置 calibrator → TRT 从 Q/DQ 节点读取 scale
         构建 INT8 引擎
```

### 4.2 Q/DQ 节点插入伪代码

```python
def insert_qdq_for_adaround(onnx_model, qdq_scales):
    """
    qdq_scales: dict mapping Gemm node name → {
        'weight_scale': delta_w (np.ndarray, per-channel),
        'weight_zp':    zp_w    (np.ndarray, per-channel, uint8),
        'act_scale':    delta_a (float, per-tensor),
        'act_zp':       zp_a    (int, per-tensor),
    }
    """
    graph = onnx_model.graph
    for node in graph.node:
        if node.op_type != 'Gemm':
            continue
        layer_name = node.name
        if layer_name not in qdq_scales:
            continue

        scales = qdq_scales[layer_name]

        # --- 权重 Q/DQ（per-channel，axis=0）---
        w_scale_init = make_initializer(scales['weight_scale'])
        w_zp_init    = make_initializer(scales['weight_zp'])
        w_q_out      = f'{layer_name}_weight_quant'
        w_dq_out     = f'{layer_name}_weight_dequant'
        # 原始 Gemm weight 输入 → QuantizeLinear → DequantizeLinear → Gemm
        insert_node('QuantizeLinear',
                    inputs=[node.input[1], w_scale_init, w_zp_init],
                    outputs=[w_q_out], attrs={'axis': 0})
        insert_node('DequantizeLinear',
                    inputs=[w_q_out, w_scale_init, w_zp_init],
                    outputs=[w_dq_out], attrs={'axis': 0})
        node.input[1] = w_dq_out   # 替换 Gemm 的权重输入

        # --- 激活 Q/DQ（per-tensor）---
        a_scale_init = make_initializer(np.float32(scales['act_scale']))
        a_zp_init    = make_initializer(np.uint8(scales['act_zp']))
        a_q_out      = f'{layer_name}_act_quant'
        a_dq_out     = f'{layer_name}_act_dequant'
        insert_node('QuantizeLinear',
                    inputs=[node.input[0], a_scale_init, a_zp_init],
                    outputs=[a_q_out])
        insert_node('DequantizeLinear',
                    inputs=[a_q_out, a_scale_init, a_zp_init],
                    outputs=[a_dq_out])
        node.input[0] = a_dq_out   # 替换 Gemm 的 input 输入

    return onnx_model
```

### 4.3 从 AdaRound checkpoint 提取 scale 的关键代码

```python
# 加载 AdaRound 校准结果
ckpt = torch.load('calibration/quant_encoder_adaround.pth')
qmodel = QuantModel(encoder, wqp, aqp)
set_weight_quantize_params(qmodel)          # 初始化 delta
for m in qmodel.modules():                  # 包裹 AdaRoundQuantizer
    if isinstance(m, QuantModule):
        m.weight_quantizer = AdaRoundQuantizer(m.weight_quantizer, ...)
qmodel.load_state_dict(ckpt['state_dict'], strict=False)
qmodel.set_quant_state(weight_quant=True, act_quant=True)
for m in qmodel.modules():
    if isinstance(m, QuantModule):
        m.weight_quantizer.soft_targets = False  # hard rounding

# 提取每层的 scale（对应 ONNX 中的 Gemm 节点）
qdq_scales = {}
for name, m in qmodel.named_modules():
    if isinstance(m, QuantModule):
        # 权重 scale：AdaRoundQuantizer → uaq.delta
        delta_w = m.weight_quantizer.uaq.delta   # Tensor[out_ch]
        zp_w    = m.weight_quantizer.uaq.zero_point
        # 激活 scale：act_quantizer.delta（nn.Parameter after calibration）
        delta_a = float(m.act_quantizer.delta)
        zp_a    = int(m.act_quantizer.zero_point)
        qdq_scales[name] = {
            'weight_scale': delta_w.cpu().numpy(),
            'weight_zp':    zp_w.cpu().numpy().astype(np.uint8),
            'act_scale':    delta_a,
            'act_zp':       zp_a,
        }
```

---

## 五、两种方案对比总结

| 比较维度 | 当前方案（fake-quant ONNX + TRT） | 正确方案（AdaRound + Q/DQ ONNX） |
|---------|:---------------------------------:|:--------------------------------:|
| 权重量化次数 | **2次**（AdaRound + TRT重新量化） | **1次**（TRT直接用AdaRound scale） |
| AdaRound舍入决策是否保留 | ❌ 被TRT覆盖 | ✅ 完整保留 |
| 真 INT8 加速 | ✅（但精度极差） | ✅（精度正常） |
| TRT 构建方式 | IInt8Calibrator（外部校准） | Q/DQ ONNX（无需Calibrator） |
| 预期 AMOTA | 0.190（双重量化损坏） | 预期 ≥ 0.381（≥ Vanilla PTQ） |
| 实现复杂度 | 低（已实现） | 中（需 ~200 行 ONNX 图编辑代码） |
| 已有参考实现 | — | ONNX Python API + onnx-simplifier |

---

## 六、后续行动建议

**优先级 P0（阻塞性 bug，不解决 AdaRound 无意义）**

- [ ] 实现 `tools/export_onnx_adaround_qdq.py`：提取 AdaRound scale → 插入 Q/DQ 节点 → 导出 Q/DQ ONNX
- [ ] 修改 `tools/build_trt_int8_univ2x.py`：支持 Q/DQ 模式（不传 calibrator，仅设 INT8 flag）

**优先级 P1（精度改善）**

- [ ] 增加 AdaRound 校准样本至 50 帧（keep_gpu=False 已支持）
- [ ] 确保校准数据时序连续（prev_bev 帧间传递）

**优先级 P2（可选）**

- [ ] 修改 `UniformAffineQuantizer.forward()` 使用 `torch.fake_quantize_per_channel_affine`，注册 ONNX symbolic，实现自动 Q/DQ 导出（更干净但改动大）

---

*作者：Claude Code，2026-04-04*
