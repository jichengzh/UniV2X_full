# Phase A-2 架构决策

> 由 ARCH agent 生成，2026-04-02  
> 决策者：ARCH + PM 确认

---

## 1. 方案推荐：方案 B（重新 Export ONNX）

**推荐原因**：方案 A（ONNX initializer patch）中 PyTorch `named_parameters()` 路径与 ONNX trace 自动生成的 initializer 名称之间没有稳定映射，QuantModel BN folding 后层级路径进一步变化。方案 B 复用已验证的 export 路径，无名称映射风险，新代码约 80 行。

**新增文件**：`tools/export_onnx_adaround.py`（无需修改现有脚本）

---

## 2. 核心函数设计

### apply_adaround_weights()

```python
def apply_adaround_weights(qmodel: QuantModel, adaround_ckpt_path: str) -> None:
    """Bake AdaRound fake-quantized weights (W_fq) into QuantModule.org_weight."""
    ckpt = torch.load(adaround_ckpt_path, map_location='cpu')
    qmodel.load_state_dict(ckpt['state_dict'])
    
    # Step A: weight_quant=True, soft_targets=False → dequant(quant_adaround(W)) → W_fq
    qmodel.set_quant_state(weight_quant=True, act_quant=False)
    for m in qmodel.modules():
        if hasattr(m, 'weight_quantizer') and hasattr(m, 'soft_targets'):
            m.weight_quantizer.soft_targets = False  # 确保用硬舍入
    
    # Step B: save_quantized_weight → module.weight.data = W_fq
    save_quantized_weight(qmodel)
    
    # Step C: 让 forward 使用 W_fq（QuantModule.forward 实际读的是 org_weight）
    for m in qmodel.modules():
        if isinstance(m, QuantModule):
            m.org_weight.copy_(m.weight.data)
    
    # Step D: 关闭量化器，ONNX export 时走纯 FP32 linear(input, W_fq)
    qmodel.set_quant_state(weight_quant=False, act_quant=False)
```

### 完整流程伪代码（export_onnx_adaround.py）

```python
# 1. 加载模型（同 calibrate_univ2x.py）
model = load_model(cfg, ckpt)
encoder = get_bev_encoder(model, args.model)

# 2. 构建 QuantModel（参数必须与 calibration 时一致，从 ckpt 读取）
ckpt_data = torch.load(args.adaround_ckpt)
weight_quant_params = ckpt_data['weight_quant_params']
act_quant_params    = ckpt_data['act_quant_params']
register_bevformer_specials()
qmodel = QuantModel(encoder, weight_quant_params, act_quant_params, is_fusing=True)

# 3. 注入 AdaRound 权重 → W_fq 写入 org_weight
apply_adaround_weights(qmodel, args.adaround_ckpt)

# 4. 取出内部 encoder（已有 W_fq，量化器已关闭）
fq_encoder = qmodel.model  # org_weight = W_fq，forward = 纯 FP32

# 5. ONNX export（复用 BEVEncoderWrapper + _patch_and_verify_onnx）
fq_encoder.cuda().eval()
wrapper = BEVEncoderWrapper(fq_encoder)
dummy = make_dummy_inputs(...)
torch.onnx.export(wrapper, dummy, args.out, ...)
_patch_and_verify_onnx(args.out)
```

---

## 3. ADR-004 确认答案

1. **layer_reconstruction() 支持级别**：仅 QuantModule 级，不支持 Block 级。`calibrate_univ2x.py` 已按此正确实现（逐 QuantModule 重建）。

2. **cali_data 格式兼容性**：`bev_encoder_calib_inputs.pkl` 为 list，与 `layer_recon.py` 的 `cali_data: list` 参数格式直接兼容。

3. **temporal prev_bev**：AdaRound 重建时的 `cali_data` 是 `save_inp_oup_data()` 预缓存的 QuantModule 局部输入，采集时已经过完整的模型 forward（含 TSA）。不需要在 layer_reconstruction 内额外处理 prev_bev。

---

## 4. 风险清单

| 风险 | 缓解措施 |
|------|---------|
| QuantModel 构造参数与 calibration 不一致（导致 load_state_dict 失败） | 将 `weight_quant_params` / `act_quant_params` 存入 .pth checkpoint，export 时读取而不硬编码 |
| BN folding 与原始 ONNX 的 Patch 4 冲突 | `is_fusing=True` 已在 QuantModel 构造时折叠 BN，export 时无 BN 模块，Patch 4 自然无效。需验证 BN folding 前后 cosine > 0.9999 |
| `org_weight.copy_()` 后 shape 不匹配（channel_wise scale 导致 weight reshape） | `weight_quantizer(weight)` 返回与 weight 同 shape 的 tensor，`org_weight` 与 `weight` 同 shape，copy_ 安全 |

---

## 5. 完整流水线命令

```bash
# Step 1: AdaRound 校准（calibrate_univ2x.py 已实现，直接使用）
conda run -n UniV2X_2.0 python tools/calibrate_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --out calibration/quant_encoder_adaround.pth \
    --adaround --adaround-iters 5000

# Step 2: 导出含 AdaRound 权重的 ONNX（新脚本）
conda run -n UniV2X_2.0 python tools/export_onnx_adaround.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego --bev-size 200 --num-cam 1 --img-h 1088 --img-w 1920 \
    --adaround-ckpt calibration/quant_encoder_adaround.pth \
    --out onnx/univ2x_ego_bev_encoder_adaround.onnx

# Step 3: 构建 TRT INT8 引擎（现有脚本，不改动）
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_bev_encoder_adaround.onnx \
    --out trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt \
    --target bev_encoder \
    --plugin plugins/build/libuniv2x_plugins.so \
    --cali-data calibration/bev_encoder_calib_inputs.pkl
```
