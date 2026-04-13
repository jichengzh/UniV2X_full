# QuantV2X AdaRound + TRT 研究报告与 UniV2X 新方案规划

> 生成时间：2026-04-05  
> 作者：Claude（基于对 QuantV2X 源码的全面阅读和分析）

---

## 一、QuantV2X 的完整量化体系解析

### 1.1 两条独立管线，不是一条整合管线

阅读 QuantV2X 源码后，最核心的发现是：**AdaRound 和 TRT INT8 在 QuantV2X 中是两条平行管线，不存在"AdaRound 为 TRT 提供 scale"的直接整合**。

| 管线 | 文件 | 目的 | 产物 |
|------|------|------|------|
| AdaRound (PyTorch) | `inference_quant.py` | 测量 W8A8 PyTorch 量化精度 | 量化误差指标，不是 TRT 引擎 |
| TRT INT8 (隐式校准) | `build_trt_int8.py` + `inference_onnx_dump_calibration.py` | 部署加速 | `.plan` TRT 引擎 |

这是许多研究代码库的常见模式：AdaRound = 论文精度验证工具；TRT = 实际部署工具。两者在 QuantV2X 中**各自独立运行**，没有数据通路。

---

### 1.2 AdaRound 管线（inference_quant.py）详解

```
FP32 checkpoint
    ↓
QuantModel(model)                    # 用 QuantModule 替换 nn.Linear
    ↓
set_weight_quantize_params()         # MinMax 初始化 weight scale
    ↓
recon_model()                        # 分发到各重建函数
    ├── layer_reconstruction()       # 单层 AdaRound（layer_recon.py）
    ├── block_reconstruction()       # 块级别 AdaRound（block_recon.py）
    └── v2xvit_reconstruction()      # V2X 融合模块专用（v2xvit_recon.py）
    ↓
PyTorch W8A8 推理（eval）
    ↓
打印精度指标
```

**layer_reconstruction() 关键逻辑**（layer_recon.py:36-149）：

```python
# 1. 缓存量化输入和 FP 输出
cached_inps = get_init(model, layer, cali_data, ...)
cached_outs, ... = get_dc_fp_init(fp_model, fp_layer, cali_data, ...)
set_act_quantize_params(layer, cached_inps)

# 2. 替换 weight quantizer 为 AdaRound
layer.weight_quantizer = AdaRoundQuantizer(
    uaq=layer.weight_quantizer,
    round_mode='learned_hard_sigmoid',
    weight_tensor=layer.org_weight.data
)
layer.weight_quantizer.soft_targets = True

# 3. Adam 迭代优化 alpha（rounding policy）和 act delta
for i in range(iters=20000):
    out_drop = layer(drop_inp)
    err = loss_func(out_drop, cur_out, ...)  # MSE + round_loss
    err.backward()
    w_opt.step()  # 更新 alpha
    a_opt.step()  # 更新 act_quantizer.delta

# 4. 固化 rounding（soft → hard）
layer.weight_quantizer.soft_targets = False
```

**结论**：优化完成后，`layer.org_weight` 仍是 FP32 原始权重，`layer.weight_quantizer.alpha` 是学到的 rounding 策略。这些参数**只在 PyTorch 中有效**，不能直接传给 ONNX 或 TRT。

---

### 1.3 TRT INT8 管线（build_trt_int8.py + inference_onnx_dump_calibration.py）详解

```
FP32 checkpoint → export_onnx() → model.onnx
                                        ↓
inference_onnx_dump_calibration.py:
    1. 运行 ONNX 推理（onnxruntime），收集 N 帧的中间激活 tensor
    2. 将每层输入保存为 calibration/*.npz
                                        ↓
build_trt_int8.py:
    class DataCalibrator(trt.IInt8EntropyCalibrator2):
        def get_batch(self):
            return next(iter_npz_dir(calibration_dir))  # 读 NPZ 文件
    
    builder.int8_mode = True
    builder.int8_calibrator = DataCalibrator(...)
    engine_bytes = builder.build_cuda_engine(network)
    → 保存 .plan 文件
```

**TRT 隐式量化**：TRT 读取 calibration 数据，自动对每层计算 activation scale（min/max entropy），weight scale 也由 TRT 内部对称 per-channel 方式计算。**不存在 Q/DQ 节点**，不使用任何 AdaRound 结果。

---

### 1.4 UniformAffineQuantizer 的设计约束

QuantV2X 的 `UniformAffineQuantizer`（quant_layer.py）：

```python
class UniformAffineQuantizer(nn.Module):
    def __init__(self, ..., symmetric=False, ...):
        # symmetric=True raises NotImplementedError
        # 默认是 asymmetric UINT8
        self.zero_point = None  # 动态计算，≠ 0

    def forward(self, x):
        # ... 非对称量化，zero_point ∈ [0, 255]
```

**这是导致我们 Q/DQ 方案失败的根本原因**：QuantV2X/UniV2X 的量化器设计就是非对称 UINT8，而 TRT GPU 要求对称 INT8（zero_point = 0）。两者在设计上不兼容。

---

## 二、我们 Q/DQ 方案的失败复盘

### 2.1 遇到的四个问题

#### 问题 1：ONNX 节点模式检测错误（24 vs 36）
- **现象**：`find_quant_matmul_nodes()` 只找到 24 个 MatMul 节点，而 QuantModule 有 36 个
- **根因**：`nn.Linear` 导出为 `Initializer → Transpose → MatMul`（标准 Linear），但 QuantModule 的 `org_weight` 是普通 `torch.Tensor`（非 `nn.Parameter`），导出为 `Constant → Transpose → MatMul`。最初的代码只匹配 Initializer 模式，漏掉了 Constant 模式
- **修复**：改为检测 `Constant → Transpose → MatMul` 模式，同时从 Constant 节点属性中提取权重值

#### 问题 2：ONNX 拓扑排序违反
- **现象**：`onnx.checker` 报告 "input X of node QL is not output of any previous nodes"
- **根因**：将所有 Q/DQ 节点 prepend 到图头部，但 Constant 节点在图的中间；Q/DQ 的输入尚未定义
- **修复**：在图遍历时，紧跟对应 Constant 节点之后插入 Q/DQ 对

#### 问题 3：非零 zero_point 被 TRT 拒绝
- **现象**：`Assertion failed: shiftIsAllZeros(zeroPoint): Non-zero zero point is not supported`
- **根因**：AdaRound 使用非对称量化，zero_point ≠ 0；TRT GPU 不支持 GPU 上的非对称整数
- **修复**：改为从 W_fq 计算对称 per-channel scale（`max|W_fq|/127`），zero_point = 0

#### 问题 4（核心失败）：权重 Q/DQ 导致 AMOTA 0.137（远低于 FP16 的 0.353）
- **现象**：TRT 构建成功（40.8 MB），但 AMOTA 从 0.353 跌至 0.137
- **根因**：
  1. TRT 看到任意 Q/DQ 节点 → 进入**显式量化模式**
  2. 显式量化模式下，`IInt8Calibrator2` **被完全忽略**（TRT 日志打印 "Calibrator won't be used"）
  3. 有 Q/DQ 的权重层：INT8 精度（但 scale 来自 W_fq 的对称近似，不是 AdaRound 真正优化后的值）
  4. 没有 Q/DQ 的激活层：**退化为 FP16**（无 scale，无 INT8）
  5. 净效果：比纯 FP16 还差（权重额外量化噪声 + 激活无法与权重的 INT8 配合）

### 2.2 根本性反思

**我犯的最大错误**：在还没有搞清楚 TRT 隐式/显式量化模式区别时，就直接实现 Q/DQ 插入。正确的调研顺序应该是：

1. 先阅读 TRT 文档，理解隐式 vs 显式量化的全局影响
2. 先读 QuantV2X 代码，理解参考实现的设计决策
3. 再决定方案

**另一个错误**：以为"只给权重加 Q/DQ，激活走 calibrator"是可行的。实际上这在 TRT 中是不可能的——Q/DQ 的存在是全局性的，开关没有粒度控制。

**最应该先问的问题**：QuantV2X 为什么不将 AdaRound 的 alpha 转化为 Q/DQ 节点？答案是：因为 QuantV2X 设计者知道非对称 UINT8 和 TRT 不兼容，所以两条管线彻底分开。

---

## 三、基于 QuantV2X 分析的 UniV2X 新规划

### 3.1 正确理解 AdaRound 的价值边界

AdaRound 在 QuantV2X 中的真实定位：
- **有价值**：在 PyTorch W8A8 推理中比 round-to-nearest 精度更高（文章核心贡献）
- **无价值**：对 TRT 部署没有直接贡献（TRT 有自己的隐式 PTQ calibration）

对 UniV2X 的启示：
- **如果目标是 TRT INT8 加速**：AdaRound 是无关路径，正确做法是 TRT 隐式 PTQ calibration
- **如果目标是评估 INT8 量化误差上界**：AdaRound 是有价值的 baseline，在 PyTorch 中运行

### 3.2 新方案 A：TRT 隐式 INT8 Calibration（推荐，短期）

**原理**：完全绕开 AdaRound，直接用 TRT 的 entropy calibration 构建 INT8 引擎。

**步骤**：
```
Step 1: 收集 calibration 数据
    - 运行 BEV 编码器，保存 N 帧的 bev_embed 输入特征（NPZ 格式）
    - 推荐 N=512-1024 帧（与 QuantV2X 一致）
    - 存储路径：calibration/bev_encoder_inputs/

Step 2: 实现 DataCalibrator
    class BEVCalibratorINT8(trt.IInt8EntropyCalibrator2):
        def get_batch(self):
            return load_next_npz(self.calibration_dir)
        
        def get_batch_size(self):
            return 1  # BEV encoder always batch=1

Step 3: 构建 INT8 TRT 引擎
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = BEVCalibratorINT8(...)
    # 不插入任何 Q/DQ 节点（保持隐式模式）
    engine_bytes = builder.build_serialized_network(network, config)

Step 4: 验证精度
    - BEV encoder: cosine sim vs FP16 TRT
    - 端到端 AMOTA vs PyTorch baseline
```

**预期效果**：
- 构建时间：~5-10 分钟（有 calibration 数据）
- 推理速度：比 FP16 快约 1.5-2x（INT8 算力翻倍，但带宽受限时收益有限）
- 精度：与当前 FP16 TRT（AMOTA=0.381）的差异 < 0.01

**实现难点**：
1. BEV 编码器有 MSDAPlugin（自定义算子）：需要在 calibration 时正确注册插件
2. 需要确保 calibration 数据的多样性（各种场景类型）

---

### 3.3 新方案 B：对称 AdaRound + 显式 Q/DQ（研究性，中期）

如果确实需要 AdaRound 的精度优势进入 TRT，需要彻底重写量化器：

**前提**：重写 `UniformAffineQuantizer` 支持对称 INT8

```python
class SymmetricInt8Quantizer(nn.Module):
    """Per-channel symmetric INT8 quantizer（对称，zero_point=0）"""
    def __init__(self, n_bits=8, channel_dim=0):
        self.n_bits = n_bits
        self.channel_dim = channel_dim
        # scale: per-channel, abs max / 127
        self.register_buffer('scale', None)
    
    def forward(self, x):
        # W: [out, in, ...]
        max_val = x.abs().amax(dim=[d for d in range(x.dim()) if d != self.channel_dim])
        scale = (max_val / 127.0).clamp(min=1e-8)
        x_int = (x / scale.view(-1, *([1]*(x.dim()-1)))).round().clamp(-128, 127)
        return x_int * scale.view(-1, *([1]*(x.dim()-1)))
```

**Q/DQ 插入**：激活也需要 Q/DQ（不能只有权重）

```python
def insert_weight_and_act_qdq(onnx_model, weight_scales, act_scales):
    """
    在每个量化 Linear 前后插入：
    - 权重：Constant → QuantizeLinear → DequantizeLinear → Transpose → MatMul
    - 激活：... → QuantizeLinear → DequantizeLinear → MatMul
    """
    # 这才是 TRT 显式量化模式的正确用法
    # weight Q/DQ: per-channel (axis=0)
    # act Q/DQ: per-tensor（TRT 要求 per-tensor activation scale）
```

**工作量估计**：2-3周
- 重写 quant_layer.py（对称量化器）
- 修改 AdaRound 的 alpha 更新（适配对称量化）
- 修改 Q/DQ 插入逻辑（同时处理激活）
- 重新收集 calibration 数据（激活 scale）
- 验证精度

**预期效果**：AdaRound + 显式 INT8 的精度应高于隐式 calibration，但实现复杂度高 5-10x。

---

### 3.4 新方案 C：当前最优配置（已验证，维持现状）

根据 `result_log.md` 中记录的实验结果（最优配置见 §15）：

| 配置 | AMOTA | AMOTP |
|------|-------|-------|
| PyTorch baseline | 0.338 | 1.474 |
| Hook A (BEV TRT FP16) | 0.381 | 1.450 |
| Hooks A+B+C+D (全 TRT FP16) | 0.370 | 1.446 |
| 目标 INT8（未实现） | ~0.360? | ? |

当前 FP16 TRT 已经达到 AMOTA=0.381（超越 PyTorch baseline 4.3个点），INT8 进一步提速但精度可能略降。

**短期推荐**：维持 FP16 TRT 现状，用方案 A（隐式 INT8 calibration）做一轮实验，若 AMOTA > 0.370 则值得部署。

---

## 四、QuantV2X 架构与 UniV2X 的差异对比

| 维度 | QuantV2X | UniV2X |
|------|----------|--------|
| 模型类型 | 点云 voxel-based（SECOND backbone） | 相机 BEV (BEVFormer + Transformer) |
| 主要算子 | 3D 稀疏卷积（SparseConv） | MSDeformableAttention（自定义 CUDA） |
| TRT 难点 | 稀疏卷积无官方 TRT 支持 | MSDAPlugin 需要 C++ 自定义层 |
| 量化目标 | Fusion module 压缩（V2X 通信带宽） | BEV encoder 加速（推理延迟） |
| AdaRound 粒度 | Layer + Block + V2X fusion block | （未适配，当前实验性） |
| TRT calibration | NPZ dump from ONNXRT | 待实现（可复用 QuantV2X 设计） |

**关键差异**：QuantV2X 的 AdaRound 主要针对 V2X 融合 block（`v2xvit_reconstruction`），而非全模型。UniV2X 若要做 AdaRound，最有价值的也是融合模块（AgentQueryFusion / LaneQueryFusion），而非 BEV 编码器。

---

## 五、行动计划（按优先级）

### 优先级 1（立即可做，1-2天）：TRT INT8 隐式 calibration

```bash
# Step 1: 收集 BEV encoder calibration 数据
python tools/dump_bev_calibration.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    --checkpoint ckpts/univ2x_coop_e2e_stg2.pth \
    --output-dir calibration/bev_encoder/ \
    --num-frames 512

# Step 2: 构建 INT8 BEV encoder TRT 引擎
python tools/build_trt_int8_bev.py \
    --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --calibration-dir calibration/bev_encoder/ \
    --output trt_engines/univ2x_ego_bev_encoder_200_int8.trt

# Step 3: 验证精度
python tools/test_trt.py \
    --use-bev-trt trt_engines/univ2x_ego_bev_encoder_200_int8.trt \
    ...
```

**需要新建的文件**：
- `tools/dump_bev_calibration.py`：运行 BEV encoder，保存输入 NPZ
- `tools/build_trt_int8_bev.py`：DataCalibrator + INT8 engine build

### 优先级 2（可选，2-3周）：对称 AdaRound 重写

仅在方案 A 精度不足时考虑，且需要大量工作。

### 优先级 3（文档）：更新 docs/quant/PROGRESS.md

记录 AdaRound Q/DQ 实验的最终结论。

---

## 六、总结

1. **QuantV2X 的 AdaRound 与 TRT 是完全分离的两条管线**，没有整合。AdaRound 用于 PyTorch 精度评估，TRT 用独立的隐式 calibration。

2. **我们的 Q/DQ 实验失败的根本原因**是低估了 TRT 显式量化模式的全局影响，以及 QuantV2X 量化器的非对称 UINT8 设计与 TRT 对称 INT8 的不兼容性。

3. **正确的 TRT INT8 路径**是 QuantV2X 的 DataCalibrator 方案：收集 calibration 数据 → 隐式 INT8 build，不涉及 Q/DQ 节点，不涉及 AdaRound。

4. **AdaRound 的价值**在于 PyTorch W8A8 推理质量评估，以及（在重写为对称量化后）作为 TRT 显式量化的 scale 来源。后者需要 2-3 周额外工作。

5. **当前最优配置**（FP16 TRT，AMOTA=0.381）已是高质量基准，INT8 是进一步优化，而非 AdaRound 的必要前提。
