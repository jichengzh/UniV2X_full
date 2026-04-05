# Phase C 执行指南

> 状态：待执行（需要 GPU 运行时）

---

## C-1：运行 AdaRound 校准

**脚本**：`tools/calibrate_univ2x.py`（已有 `--adaround` 实现，无需修改）

```bash
conda run -n UniV2X_2.0 python tools/calibrate_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --cali-data calibration/bev_encoder_calib_inputs.pkl \
    --out calibration/quant_encoder_adaround.pth \
    --scale-method mse \
    --adaround \
    --adaround-iters 5000
```

**预期输出**：
```
  Found N QuantModules to reconstruct.
  [1/N] AdaRound: pts_bbox_head.transformer.encoder.layers.0...
  ...
  AdaRound complete.
  Saved quantized weights → calibration/quant_encoder_adaround.pth
  File size: ~XX MB
```

**预期耗时**：20~40 分钟（每层 5000 iters × ~30-40 层）  
**监控**：`watch -n 5 nvidia-smi`

**降级选项**（若 OOM）：`--adaround-iters 2000`

---

## C-2：导出含 AdaRound 权重的 ONNX

**脚本**：`tools/export_onnx_adaround.py`（Phase B-1 新建）

```bash
conda run -n UniV2X_2.0 python tools/export_onnx_adaround.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego \
    --bev-size 200 --num-cam 1 --img-h 1088 --img-w 1920 \
    --adaround-ckpt calibration/quant_encoder_adaround.pth \
    --out onnx/univ2x_ego_bev_encoder_adaround.onnx
```

**预期输出**：
```
  N QuantModules updated with W_fq.
  Exporting AdaRound ONNX → onnx/univ2x_ego_bev_encoder_adaround.onnx
  ONNX saved. Patching INT64→INT32 ...
  ✓ AdaRound ONNX saved: onnx/univ2x_ego_bev_encoder_adaround.onnx (XXX MB)
```

**验收**：ONNX 文件存在，大小接近原 ONNX（`onnx/univ2x_ego_bev_encoder_200_1cam.onnx` ≈ 105 MB）

---

## C-3：构建 AdaRound INT8 TRT 引擎

**脚本**：`tools/build_trt_int8_univ2x.py`（复用，不改动）

```bash
conda run -n UniV2X_2.0 python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_bev_encoder_adaround.onnx \
    --out trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt \
    --target bev_encoder \
    --plugin plugins/build/libuniv2x_plugins.so \
    --cali-data calibration/bev_encoder_calib_inputs.pkl
```

**预期输出**：
- 引擎大小：~40~45 MB（接近 vanilla PTQ 的 43 MB）
- 无 INT8 calibration cache 时需要完整 50 帧数据，耗时约 5~10 分钟

---

## 产出物

| 文件 | 预期大小 | 验收条件 |
|------|---------|---------|
| `calibration/quant_encoder_adaround.pth` | > 10 MB | 文件存在，内含 `adaround=True` |
| `onnx/univ2x_ego_bev_encoder_adaround.onnx` | ~100 MB | 文件存在，可被 onnx.load() |
| `trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt` | 40~45 MB | 文件存在 |
