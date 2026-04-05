# Phase A-1 API 摘要

> 由 EXPLORE agent 生成，2026-04-02  
> 来源文件：adaptive_rounding.py / layer_recon.py / quant_bevformer.py（前200行）/ calibrate_univ2x.py（前150行）

---

## adaptive_rounding.py

**AdaRoundQuantizer 构造参数**：
- `uaq: UniformAffineQuantizer` — 量化参数初始化源（复制 n_bits、sym、delta、zero_point、n_levels）
- `weight_tensor: torch.Tensor` — 用于初始化 alpha 参数（学习舍入策略）
- `round_mode: str` — 默认 `'learned_hard_sigmoid'`

**注意**：无 `init_delta_from_weights()` 函数，等价初始化通过构造时 `self.init_alpha(weight_tensor)` 完成。

**soft/hard 切换方式**：
- 训练期间：`layer.weight_quantizer.soft_targets = True`
- 重建完成后：`layer.weight_quantizer.soft_targets = False`（切换为硬舍入）

---

## layer_recon.py

**`layer_reconstruction()` 完整签名**：
```python
def layer_reconstruction(
    model: QuantModel,
    fp_model: QuantModel,
    layer: QuantModule,        # 量化版 layer
    fp_layer: QuantModule,     # FP32 版对应 layer
    cali_data: list,           # 校准数据 list（DataLoader 迭代批次）
    batch_size: int = 1,
    iters: int = 20000,
    weight: float = 0.001,     # 轮舍正则项权重
    opt_mode: str = 'mse',
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    p: float = 2.0,
    lr: float = 4e-5,
    input_prob: float = 1.0,
    keep_gpu: bool = True,
    lamb_r: float = 0.2,
    T: float = 7.0,
    bn_lr: float = 1e-3,
    lamb_c: float = 0.02
)
```

**⚠️ 关键限制**：
- `cali_data` 为 `list`（DataLoader 迭代批次），**不是原始 tensor 列表**
- **只支持层级（QuantModule）重建**，不支持 block 级重建
- forward 调用直接 `layer(x)`，无需 wrapper
- 需要同时传入 `fp_model` / `fp_layer`（FP32 对照模型用于计算重建 loss）

**LossFunction 主要参数**（参考 AdaRound 论文）：
- `weight`：轮舍正则项权重（λ）
- `b_range`：温度调度范围 [b_start, b_end]，控制正则项从软到硬的退火
- `warmup`：预热比例（前 warmup 比例 iters 只优化重建 loss，不加正则）
- `T`：KL 散度温度

---

## quant_bevformer.py（前200行）

**已注册 QuantBlock 类**（8个映射，4原始+4 TRT变体）：
1. `QuantMSDA3D` ← MSDeformableAttention3D + TRT 变体
2. `QuantSCA` ← SpatialCrossAttention + TRT 变体
3. `QuantTSA` ← TemporalSelfAttention + TRT 变体
4. `QuantCustomMSDA` ← CustomMSDeformableAttention + TRT 变体

**Skip 层（FP16，不参与 AdaRound）**：
- `sampling_offsets`
- `attention_weights`
（ADR-001：直接输入 MSDAPlugin，INT8 误差导致 BEV 特征失效）

**register 接口**：
```python
register_bevformer_specials() -> None   # 无参数，调用一次即可
```

---

## calibrate_univ2x.py（前150行）

**ArgParse 参数列表**：
`config`, `checkpoint`, `--cali-data`, `--out`, `--model`, `--scale-method`,
`--n-bits-w`, `--n-bits-a`, `--adaround`（store_true）, `--adaround-iters`,
`--batch-size`, `--cfg-options`

> **重要发现**：`calibrate_univ2x.py` **已经支持 `--adaround` flag**！
> 可能已经有 AdaRound 实现框架，需要确认是否已完整实现还是占位。

**数据加载**：
```python
with open(args.cali_data, 'rb') as f:
    cali_data = pickle.load(f)   # list，可索引
```

**scale calibration 触发**：通过 `set_weight_quantize_params()` 函数（150行后）。

---

## ⚠️ 关键架构发现（供 ARCH agent 使用）

**`build_trt_int8_univ2x.py` 完全不使用 `calibrate_univ2x.py` 的输出！**
- TRT 引擎构建只读 FP32 ONNX + calibration pkl（TRT 自己计算 INT8 scales）
- AdaRound 权重（.pth）目前对 TRT 引擎无任何影响
- 要让 AdaRound 改善 TRT INT8，必须先将 AdaRound 舍入写入 ONNX（"假量化导出"）

**正确的 AdaRound TRT 流水线**：
```
calibrate_univ2x.py --adaround  →  quant_encoder_adaround.pth
                                         ↓
export_onnx_adaround.py           →  univ2x_ego_bev_encoder_adaround.onnx
(加载 adaround.pth，对每个 weight 做 dequant(quant_adaround(w))，保持 FP32 精度格式)
                                         ↓
build_trt_int8_univ2x.py         →  INT8 engine（TRT 发现权重已在 INT8 格网上，保留 AdaRound 决策）
```

**`calibrate_univ2x.py` 已有 `--adaround` 实现**（完整，可直接用）：
```python
# 已实现的流程（lines 182-209）
fp_model = deepcopy(qmodel)
for name, layer in modules:
    layer_reconstruction(qmodel, fp_model, layer, fp_layer, cali_data, iters=args.adaround_iters)
torch.save({'state_dict': qmodel.state_dict(), 'adaround': True, ...}, args.out)
```

**缺失的环节**（需要新建）：`tools/export_onnx_adaround.py` 或直接 patch 现有 ONNX 中的权重。

## 关键结论（供 ARCH agent 使用）

1. `layer_reconstruction()` 需要 FP32 对照模型（`fp_model` + `fp_layer`），必须同时维护量化和 FP32 两份模型
2. 只支持 QuantModule 级重建，BEVFormer 中可重建的候选层为 QuantMSDA3D/QuantSCA/QuantTSA 内部的 `value_proj`、`output_proj`、FFN Linear 等 QuantModule
3. **`calibrate_univ2x.py` 已有 `--adaround` flag** — 需要读取 150 行后的代码确认是否已实现
4. `cali_data` 格式为 pickle list，与 layer_recon 的 `cali_data: list` 参数格式兼容
