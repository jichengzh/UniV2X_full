# QuantV2X 深度研究报告

> 生成时间：2026-04-05  
> 基于对 QuantV2X 全部量化相关源码的完整阅读与分析

---

## 一、项目整体架构

QuantV2X 是一个面向 V2X 协同感知的量化框架，支持多种传感器模态（LiDAR点云/相机）、多种融合策略（early/late/intermediate）和多种骨干网络（SECOND、PointPillar、LSS等）。其核心量化目标是：**在保证检测精度的前提下压缩 V2X 通信的 BEV 特征，降低通信带宽**。

---

## 二、完整量化体系（三层结构）

### 2.1 量化层（quant_layer.py）

#### QuantModule（针对 nn.Conv2d / nn.Linear）

```
FP32 Conv/Linear
    → QuantModule
        ├── weight_quantizer: UniformAffineQuantizer（W8，非对称，per-channel）
        ├── act_quantizer:    UniformAffineQuantizer（A8，非对称，per-tensor）
        ├── org_weight:       原始 FP32 权重（plain tensor，非 Parameter）
        ├── norm_function:    BatchNorm（BN folding 后融合进来）
        └── activation_function: ReLU 等
```

#### QuantSpconvModule（针对 3D 稀疏卷积）

这是 QuantV2X 的关键特殊设计，专门处理 spconv 库中的稀疏卷积层：

```python
class QuantSpconvModule(nn.Module):
    """
    支持: SubMConv3d, SparseConv3d, SparseInverseConv3d（spconv 库）
    """
    def forward(self, input: SparseConvTensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.org_weight)  # W8 量化
        else:
            weight = self.org_weight
        
        # 临时覆盖 spconv 模块的权重，再做稀疏卷积前向
        self.spconv_module.weight = nn.Parameter(weight)
        out = self.spconv_module(input)   # 稀疏卷积（保持稀疏格式）
        
        # BN + ReLU + 激活量化
        out = out.replace_feature(self.norm_function(out.features))
        out = out.replace_feature(self.activation_function(out.features))
        if self.use_act_quant:
            out = out.replace_feature(self.act_quantizer(out.features))  # A8
        return out
```

**设计意义**：spconv 的稀疏卷积在 PyTorch 侧可以做 W8A8 fake-quantization，但其稀疏格式（SparseConvTensor）原生不被 TRT 支持。这是 QuantV2X 与 UniV2X 最大的技术差异之一。

#### UniformAffineQuantizer（非对称 UINT8）

```python
# 非对称量化：zero_point ∈ [0, 255]，不一定为 0
x_int = torch.clamp(torch.round(x / self.delta) + self.zero_point, 0, self.n_levels-1)
x_float_q = (x_int - self.zero_point) * self.delta
```

**限制**：TRT GPU 要求对称 INT8（zero_point = 0），此设计与 TRT 直接不兼容，这是为何 AdaRound 结果无法直接导入 TRT Q/DQ 节点的根本原因。

---

### 2.2 量化块（quant_block.py）

QuantV2X 针对各种网络模块构建了专用量化块：

| 量化块 | 对应原始模块 | 特殊处理 |
|--------|------------|---------|
| `QuantBasicBlock` | ResNet BasicBlock | 残差分支量化 |
| `QuantVoxelBackBone8x` | SECOND 3D稀疏骨干 | 用 `QuantSpconvModule` 替换所有 SubMConv3d/SparseConv3d |
| `QuantSECOND` | SECOND 编码器 | 包含 vfe + map_to_bev + 量化稀疏骨干 |
| `QuantPyramidFusion` | 融合模块 | V2X 通信压缩的核心目标 |
| `QuantV2XViTFusion` | V2X-ViT Transformer 融合 | 注意力权重量化 |
| `QuantCamEncode_Resnet101` | LSS 相机编码器 | BN folding + 量化 |

关键代码（QuantVoxelBackBone8x）：

```python
def quantize_sparse_seq(sparse_seq):
    for i, layer in enumerate(sparse_seq):
        if isinstance(layer, (SubMConv3d, SparseConv3d)):
            quant_layer = QuantSpconvModule(layer, weight_quant_params, act_quant_params)
        elif isinstance(layer, nn.BatchNorm1d):
            quant_seq[-1].norm_function = layer   # BN folded into QuantSpconvModule
        elif isinstance(layer, nn.ReLU):
            quant_seq[-1].activation_function = layer
```

---

### 2.3 量化模型（quant_model.py）

`QuantModel` 递归遍历原始模型，将所有层替换为对应的量化版本：

```python
qt_model = QuantModel(model=trained_model, weight_quant_params=wq_params, act_quant_params=aq_params)
# → nn.Conv2d / nn.Linear → QuantModule
# → SubMConv3d / SparseConv3d → QuantSpconvModule（via QuantVoxelBackBone8x）
# → PyramidFusion → QuantPyramidFusion
# → V2XViTFusion → QuantV2XViTFusion
```

---

## 三、AdaRound 量化流程（PyTorch W8A8 路径）

### 3.1 完整流程

```
FP32 checkpoint
    ↓
QuantModel(FP32)         # 替换所有层为 QuantModule / QuantSpconvModule
    ↓
set_weight_quantize_params(qt_model)   # MinMax 初始化 weight scale（per-channel）
    ↓
recon_model(qt_model, fp_model)        # AdaRound 逐层/块优化
    ├── layer_reconstruction()         # 单个 QuantModule
    ├── block_reconstruction()         # ResNet block 等
    ├── second_reconstruction()        # SECOND 3D 稀疏骨干 ← QuantVoxelBackBone8x
    ├── pyramid_reconstruction()       # PyramidFusion 融合块
    ├── lss_reconstruction()           # LSS 相机编码器
    └── v2xvit_reconstruction()        # V2X-ViT 融合块（注：代码中此行被注释掉）
    ↓
qt_model.set_quant_state(True, True)   # 开启 W+A 量化
    ↓
PyTorch W8A8 推理（qt_model）          # 最终量化模型推理
```

### 3.2 AdaRound 核心逻辑

```python
# 替换 weight_quantizer 为 AdaRoundQuantizer
layer.weight_quantizer = AdaRoundQuantizer(
    uaq=layer.weight_quantizer,
    round_mode='learned_hard_sigmoid',
    weight_tensor=layer.org_weight.data
)
# alpha: nn.Parameter，形状 = weight 形状，初始化使得 sigmoid(alpha) ≈ fractional part
# soft_targets=True → 软取整（训练期）
# soft_targets=False → 硬取整：floor(x/delta) + (alpha >= 0).float()

# 20000 次 Adam 迭代优化 alpha（rounding policy）+ act delta（LSQ）
# 损失函数 = MSE(quant_out, fp_out) + round_loss（regularization）
```

**关键**：AdaRound 的 `alpha` 只在 PyTorch forward 中生效，无法直接导出到 ONNX 或 TRT。

### 3.3 稀疏卷积的 AdaRound（second_reconstruction）

QuantVoxelBackBone8x 中的 QuantSpconvModule 同样参与 AdaRound：
- `org_weight`：原始 FP32 稀疏卷积核
- `weight_quantizer`：先被初始化为 UniformAffineQuantizer，再被替换为 AdaRoundQuantizer
- AdaRound 优化在 PyTorch 稀疏格式（SparseConvTensor）下正常进行

---

## 四、TRT 部署路径（隐式 INT8 Calibration）

### 4.1 完整流程

```
FP32 ONNX（来自原始 FP32 模型，非量化模型）
    ↓
inference_onnx_dump_calibration.py
    → 用 ONNX Runtime 运行真实数据
    → 保存每帧输入 tensor → calibration/*.npz
    ↓
build_trt_int8.py
    DataCalibrator(trt.IInt8EntropyCalibrator2)
        → 读取 NPZ，逐批喂给 TRT
        → TRT 自动计算每层 activation scale（entropy 方法）
        → TRT 自动计算 weight scale（对称 per-channel）
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # fallback
    → 输出 INT8 TRT 引擎（.plan 文件）
```

### 4.2 ONNX 导出中的稀疏卷积问题

**核心难题**：spconv 的 `SubMConv3d`、`SparseConv3d` 使用稀疏 COO 格式（indices + features），TRT 原生不支持此格式。

**QuantV2X 的处理方式**：论文明确说明：
> "Since native TensorRT does not support quantization for certain network modules, we implement custom CUDA kernels and integrate them as TensorRT plug-ins to ensure compatibility and accurate latency profiling."

**代码库现状**：公开的 QuantV2X 仓库**不包含**这些 TRT plugin 源码（未找到任何 `.cu` plugin 文件或 TRT plugin 目录）。可能的情况：
1. 插件代码尚未开源（论文代码与开源代码不同步）
2. ONNX 导出时将稀疏卷积 densify 为标准 Conv3d，TRT 正常处理
3. 使用了第三方稀疏卷积 TRT plugin（如 CenterPoint-TRT 等方案）

**ONNX 导出路径推测**（基于 inference_onnx_dump_calibration.py 使用 ONNXRT 运行）：
- spconv 的 ONNX 导出会将稀疏操作展开（densify）为标准的密集张量操作
- 导出的 ONNX 中不存在稀疏格式，TRT 可以处理
- "custom CUDA kernels" 可能针对性能优化，而非基础功能支持

---

## 五、两条路径的关系总结

```
                    FP32 Checkpoint
                          │
          ┌───────────────┴───────────────┐
          │                               │
    [路径 A: AdaRound]             [路径 B: TRT 部署]
          │                               │
    QuantModel(FP32)              ONNX Export (FP32)
    set_weight_quantize_params    （稀疏卷积 densify）
    recon_model: AdaRound               │
    ├── QuantModule (Conv2d)      ONNX Runtime 收集
    ├── QuantSpconvModule (spconv) calibration NPZ
    └── QuantV2XViTFusion               │
          │                       build_trt_int8.py
    W8A8 PyTorch 推理              DataCalibrator
    （精度测量，论文主要贡献）      INT8 TRT Engine
                                  （延迟测量，工程部署）
          │                               │
          └───────── 完全独立 ─────────────┘
                  结果不共享，数据不互通
```

**AdaRound 的结果（alpha 参数）不传入 TRT 路径。**  
**TRT 的 INT8 scale 由 DataCalibrator entropy 方法独立计算。**

---

## 六、对 UniV2X 的启示

### 6.1 相似点

| 维度 | QuantV2X | UniV2X |
|------|----------|--------|
| TRT 路径 | FP32 ONNX → DataCalibrator → INT8 | FP32 ONNX → （待实现） → INT8 |
| 自定义算子 | spconv TRT plugin（未开源） | MSDAPlugin（已实现，开源） |
| AdaRound 使用 | PyTorch W8A8 推理 | 当前也在 PyTorch 侧 |

### 6.2 关键差异

| 维度 | QuantV2X | UniV2X |
|------|----------|--------|
| 不支持的算子 | 3D 稀疏卷积（spconv） | MSDeformableAttention（CUDA） |
| 插件状态 | 论文提及，代码未开源 | MSDAPlugin 已实现并验证 ✅ |
| TRT INT8 状态 | 已实现（DataCalibrator） | **待实现** |
| 量化器设计 | 非对称 UINT8 | 同样非对称 UINT8（复制自 QuantV2X） |

### 6.3 UniV2X 下一步行动

根据 QuantV2X 的完整设计，UniV2X 的 TRT INT8 正确路径是：

**Step 1**：实现 `dump_bev_calibration.py`
- 运行 BEV encoder（已有 FP32 ONNX），喂真实 V2X-Seq-SPD 数据
- 保存每帧输入（img_feats + prev_bev + can_bus + lidar2img）为 NPZ 格式
- 推荐 512 帧以上

**Step 2**：实现 `build_trt_int8_bev.py`
- 仿照 QuantV2X 的 DataCalibrator 设计
- 关键：需要在 calibration 时注册 MSDAPlugin（plugins/build/libuniv2x_plugins.so）
- 使用 `IInt8EntropyCalibrator2` + `BuilderFlag.INT8`

**Step 3**：端到端验证
- 比较 INT8 TRT vs FP16 TRT 的 AMOTA 差异
- 目标：AMOTA 不低于 FP16 基准（0.381）的 0.01

---

## 七、论文中提及的 Custom CUDA Kernels 的准确理解

### 7.1 论文原话
> "we implement custom CUDA kernels and integrate them as TensorRT plug-ins to ensure compatibility and accurate latency profiling"

### 7.2 现有证据

| 证据 | 内容 |
|------|------|
| 代码库搜索 | 无 `.cu` plugin 文件，无 TRT plugin 目录 |
| `QuantSpconvModule` | 只在 PyTorch 侧处理稀疏卷积量化 |
| `build_trt_int8.py` | 标准 DataCalibrator，无 plugin 注册 |
| ONNX Runtime 使用 | inference_onnx_dump_calibration.py 用 ONNXRT 运行模型 |

### 7.3 结论

论文描述的 custom CUDA kernels 属于以下之一：
1. **未开源的工程实现**：论文中完成了 TRT plugin，但公开代码未包含
2. **ONNX densify 方案**：稀疏卷积在 ONNX 导出时展开为密集操作，TRT 原生支持；"custom kernels" 指性能优化层面
3. **第三方 plugin**：复用了现有的 spconv-TRT plugin 方案

**对 UniV2X 的借鉴**：UniV2X 已有完整的 MSDAPlugin（C++/CUDA，开源），在 TRT plugin 工程化方面比 QuantV2X 更透明、更完备。

---

## 八、完整结论

1. **AdaRound 是 QuantV2X 的核心量化方法**，用于 PyTorch W8A8 推理，覆盖 Dense Conv（QuantModule）和 3D 稀疏卷积（QuantSpconvModule），是论文精度结果的来源。

2. **TRT 部署使用标准隐式 PTQ**（DataCalibrator + entropy calibration），与 AdaRound 完全独立，AdaRound 学到的 alpha 不传给 TRT。

3. **论文提及的 custom CUDA kernels** 针对 spconv 的 TRT 适配，在公开代码库中不存在，可能未开源或采用了 densify 导出方案。

4. **UniV2X 相比 QuantV2X 的优势**：MSDAPlugin 已完整实现并开源，TRT plugin 工程化更完备；只需补充 DataCalibrator 路径即可完成 INT8 TRT 引擎构建。

5. **下一步最高优先级**：实现 UniV2X BEV encoder 的 INT8 DataCalibrator 路径（仿照 QuantV2X 的 build_trt_int8.py），无需涉及 AdaRound 或 Q/DQ 节点。
