# Phase B2: TRT 显式模式 + 端到端验证

> 覆盖: Q/DQ ONNX -> TRT engine 构建 + PyTorch-TRT 一致性验证
> 修改文件: tools/build_trt_int8_univ2x.py
> 依赖: Phase B1 (Q/DQ ONNX)
> 预计工作量: 1.5 天
> 状态: 待开始

---

## 1. 任务清单

- [ ] Task 1: build_trt_int8_univ2x.py 增加 --mode explicit
- [ ] Task 2: 单配置构建测试 (全 INT8 Q/DQ ONNX -> TRT)
- [ ] Task 3: 混合精度构建测试 (INT4+INT8+FP16 共存)
- [ ] Task 4: PyTorch-TRT 一致性验证 (3-5 个 quant_config)
- [ ] Task 5: 性能测量 (latency, memory, model size)
- [ ] 反思文档

---

## 2. TRT 显式模式构建

### 2.1 与隐式模式的区别

```
隐式模式 (当前):
  config.set_flag(INT8)
  config.int8_calibrator = calibrator
  -> TRT 用 Calibrator 推导所有层的 scale
  -> 用户无法控制逐层精度

显式模式 (本阶段):
  config.set_flag(INT8)
  # 不设置 Calibrator
  -> TRT 从 ONNX 中的 Q/DQ 节点读取 scale
  -> 无 Q/DQ 的层自动走 FP16
  -> 用户完全控制逐层精度
```

### 2.2 实现

```python
# 新增命令行: --mode explicit --onnx xxx_qdq.onnx
# 关键区别: 不设置 calibrator, 不调用 _force_msda_fp16
```

---

## 3. 验证矩阵

| 验证项 | 标准 |
|--------|------|
| 构建成功 | Q/DQ ONNX 通过 onnx.checker; TRT engine 构建无错误 |
| 精度一致 | \|AMOTA_trt - AMOTA_pytorch\| < 0.01 (3-5 个 config) |
| 位宽模拟 | INT4 层: TRT vs PyTorch 余弦 > 0.99 |
| 粒度正确 | per-channel: scale dim = out_channels |
| 混合精度 | INT4+INT8+FP16 三者共存构建+推理正确 |
| W-only | 仅权重量化时不退出 explicit 模式 |

---

## 4. 完成标志

Phase B2 完成 = 整个 1.1 可配置量化完成:
- Part A: 给定任意 quant_config.json -> 3-5min 得到 PyTorch AMOTA
- Part B: 将该 config 部署到 TRT -> AMOTA 偏差 < 0.01
- D1-D9 全部可搜索、可评估、可部署

---

## 5. 执行记录

- [x] Task 1: build_trt_int8_univ2x.py 增加 --mode explicit + build_explicit_int8_engine()
- [x] Task 2: 单配置构建测试通过 — 38.2 MB engine (vs 隐式 43 MB, FP16 75 MB)
- [ ] Task 3: 混合精度构建测试 (需 GPU)
- [ ] Task 4: PyTorch-TRT 一致性验证 (需 GPU)
- [ ] Task 5: 性能测量 (需 GPU)

### 代码变更
- `build_trt_int8_univ2x.py`: +83 行
  - 新增 `--mode {implicit, explicit}` 参数
  - 新增 `build_explicit_int8_engine()` 函数
  - main() 按 mode 分支到不同构建路径

---

## 6. 反思

### 完成时间
2026-04-12, 代码部分完成。GPU 验证待执行。

### 端到端工作流 (供验证时参考)

```bash
# Step 1: PyTorch 侧校准, 产出 quant_config.json (含 scale)
python tools/quick_eval_quant.py \
    --config ... --checkpoint ... \
    --quant-config quant_configs/test_config.json \
    --export-scales

# Step 2: 注入 Q/DQ 节点
python tools/inject_qdq_from_config.py \
    --input-onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --quant-config quant_configs/test_config.json \
    --output onnx/univ2x_ego_bev_encoder_qdq.onnx

# Step 3: 构建 TRT engine (显式模式)
python tools/build_trt_int8_univ2x.py \
    --onnx onnx/univ2x_ego_bev_encoder_qdq.onnx \
    --out trt_engines/univ2x_ego_bev_encoder_explicit_int8.trt \
    --target bev_encoder \
    --mode explicit

# Step 4: 端到端评估
python tools/test_trt.py \
    --use-bev-trt trt_engines/univ2x_ego_bev_encoder_explicit_int8.trt \
    ...
```
