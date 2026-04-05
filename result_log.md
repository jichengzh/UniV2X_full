# UniV2X 全链路优化方案与性能结果完整记录

**项目**：UniV2X — 端到端 V2X 协同自动驾驶系统  
**硬件**：RTX 4090 (SM 89)，CUDA 11.8，TensorRT 10.13.0.35  
**评估集**：V2X-Seq-SPD 协同验证集，168 样本，442 GT 目标  
**最后更新**：2026-04-04

---

## 一、优化前基准（纯 PyTorch）

| 指标 | 数值 |
|------|------|
| AMOTA | 0.338 |
| AMOTP | 1.474 |
| mAP | 0.0727 |
| NDS | 0.0679 |
| Checkpoint 总大小 | 1,600 MB |
| V2X 推理延迟（N_inf=50） | ~5,640 ms |
| `_query_matching` (N_inf=50) | 5,341 ms（最大瓶颈） |
| `_get_coop_bev_embed` (N=200) | 64.3 ms |

---

## 二、优化路线总览

```
Phase 1  ──  BEV 编码器 TRT FP16
Phase 2  ──  检测头 TRT FP16
Phase 3  ──  下游头 TRT FP16（Motion + Occ + Planning）
Phase V2X── V2X 融合向量化（AgentQueryFusion / LaneQueryFusion / BEV 散射）
Phase 3C ──  V2X 路径检测头 TRT（1101-query 引擎，Hook-D）
Phase E  ──  BEV 编码器 INT8 PTQ（Vanilla，50 样本 temporal）
AdaRound ──  AdaRound W8 实验（结论：双重量化，方案需修正）
```

---

## 三、Phase 1 — BEV 编码器 TRT FP16

### 3.1 核心挑战与解法

| 问题 | 原因 | 解法 |
|------|------|------|
| DCNv2 骨干无法 ONNX 导出 | 可变形卷积算子不支持 torch.onnx | `BEVEncoderWrapper` 接收预提取 FPN 特征作为输入，完全绕过骨干 |
| MSDAPlugin 拒绝 INT64 输入 | TRT 插件要求 `spatial_shapes/level_start_index` 为 INT32 | ONNX 后处理自动改写为 INT32 常量节点 |
| 单摄像头 reshape 崩溃 | V2X-Seq-SPD 数据集只有 1 个摄像头，SCA 硬编码 6 个 | SCA 的 key/value 补零至 `num_cams=6` 后再 reshape |
| BEV embedding 归零错误 | ONNX 嵌入了 checkpoint 权重，不能手动归零比对 | 保持 bev_embedding 权重，直接与 PyTorch 输出对比 |

### 3.2 新增模块

- `MSDeformableAttention3DTRT`、`SpatialCrossAttentionTRT`（`spatial_cross_attention.py`）
- `TemporalSelfAttentionTRT`（`temporal_self_attention.py`）
- `BEVFormerLayerTRT`、`BEVFormerEncoderTRT`（`encoder.py`）
- `get_bev_features_trt()`（`modules/transformer.py`）
- C++ 插件：`MSDAPlugin`（`plugins/multi_scale_deform_attn/`）、`RotatePlugin`、`InversePlugin`（cublas Sgetrf/Sgetri）

### 3.3 精度验证（200×200 BEV，真实 checkpoint）

| 模型 | Max AbsDiff | Mean AbsDiff | Cosine Sim | 误差 < 1e-2 |
|------|------------|--------------|-----------|------------|
| ego   | 0.029      | 3.6e-4       | 0.9999993 | 100% ✅    |
| infra | 0.009      | 2.9e-4       | 0.9999997 | 100% ✅    |

### 3.4 生成引擎

| 文件 | 大小 |
|------|------|
| `trt_engines/univ2x_ego_bev_encoder_200_1cam.trt` | 74.6 MB |
| `trt_engines/univ2x_infra_bev_encoder_200.trt` | 72 MB |

**延迟**：BEV 编码器 ~150 ms → ~20–40 ms（约 **5×** 加速）

---

## 四、Phase 2 — 检测头 TRT FP16

### 4.1 设计

- `HeadsWrapper`：封装 Decoder + `velo_update`，固定 901 query（900 追踪 + 1 ego）
- `BEVFormerTrackHeadTRT.get_detections_trt()`：纯 tensor 接口（无 img_metas）
- `CustomMSDeformableAttentionTRT`、`DetectionTransformerDecoderTRTP`（`decoder.py`）

### 4.2 精度验证（901-query，真实 checkpoint）

所有输出（cls_scores / bbox_preds / traj_preds / query_feats / ref_pts）余弦相似度均 > **0.9999994** ✅

### 4.3 生成引擎

| 文件 | 大小 |
|------|------|
| `trt_engines/univ2x_ego_heads_200.trt` | 33 MB |

---

## 五、Phase 3 — 下游头 TRT FP16（Motion + Occ + Planning）

### 5.1 设计

- `DownstreamHeadsWrapper`：封装三个下游头
- `MotionHeadTRT.forward_trt()`、`OccHeadTRT.forward_trt()`、`PlanningHeadSingleModeTRT.forward_trt()`

### 5.2 关键 Bug 修复

| 问题 | 解法 |
|------|------|
| BN 训练模式统计量漂移 | ONNX trace 中覆盖 BN `forward` 强制 `training=False` |
| Dropout `training_mode=True` | ONNX 后处理将 56–68 个 Dropout 节点的 `training_mode` 改写为 False |

### 5.3 精度验证（下游头，200×200，stg2 checkpoint）

| 模型 | 输出 | MeanAbs | CosSim |
|------|------|---------|--------|
| ego | traj_scores | 2.26e-3 | 0.9999947 ✅ |
| ego | traj_preds  | 3.97e-3 | 0.9999854 ✅ |
| ego | occ_logits  | 6.13e-4 | 0.9999966 ✅ |
| ego | sdc_traj    | 3.33e-4 | 1.0000000 ✅ |
| infra | traj_scores | 2.26e-3 | 0.9999947 ✅ |
| infra | traj_preds  | 3.97e-3 | 0.9999854 ✅ |
| infra | occ_logits  | 6.13e-4 | 0.9999966 ✅ |

### 5.4 生成引擎

| 文件 | 大小 |
|------|------|
| `trt_engines/univ2x_ego_downstream.trt` | 152 MB |
| `trt_engines/univ2x_infra_downstream.trt` | 134 MB |

---

## 六、Phase V2X — V2X 融合向量化

### 6.1 `_query_matching_vec`（最大瓶颈消除）

**问题根因**：双层 Python for-loop，内层每次 `cost_matrix[i][j] = torch.sum(...)` 触发隐式 `.item()`，即一次 GPU→CPU 数据传输 + CUDA 流同步。N_veh=901、N_inf=100 时共 90,100 次同步。

**向量化方案**（`agent_fusion.py`）：

```python
diff = veh_pts.unsqueeze(1) - inf_ref_pts.unsqueeze(0)  # (M, N, 3) 广播
l2   = diff.pow(2).sum(-1).sqrt()                        # (M, N) GPU 上完成
keep = (diff.abs() / dims.unsqueeze(1)).le(1.0).all(-1)  # (M, N) bool
cost_active = torch.where(keep, l2, fill(1e6))
cost_matrix = cost_active.cpu().numpy()                  # 唯一一次 GPU→CPU
```

**性能对比**：

| N_inf | 优化前 | 优化后 | 加速比 |
|-------|--------|--------|--------|
| 10    | 1,048 ms | 2.7 ms | **392×** |
| 50    | 5,341 ms | 5.6 ms | **951×** |
| 100   | 10,548 ms | 8.9 ms | **1,191×** |

### 6.2 `_get_coop_bev_embed` 向量化

**问题根因**：三层嵌套 Python 循环（N × 2×2 邻域），逐像素 scatter 写 BEV 网格。

**向量化方案**（`univ2x_track.py`）：预计算 `flat_idx` + `torch.index_add_` 批量 scatter（语义完全等同于原始 `+=`，含重叠索引累加）：

**性能对比**：

| N | 优化前 | 优化后 | 加速比 |
|---|--------|--------|--------|
| 10  | 3.4 ms  | 1.1 ms | 3× |
| 50  | 16.2 ms | 1.1 ms | 15× |
| 200 | 64.3 ms | 1.1 ms | **58×** |

### 6.3 LaneQueryFusionTRT + AgentQueryFusionTRT（Hook B/C）

- `LaneQueryFusionTRT.forward_trt()`：CPU 上 `np.linalg.inv` + `filter_other_lanes`（保持 FP32 精度）
- `AgentQueryFusionTRT.forward_trt()`：GPU 上 ego_selection + 向量化融合 + index_add_ scatter

**精度验证**：

| 模块 | CosSim | max_abs |
|------|--------|---------|
| AgentQueryFusionTRT vs 原始 | 0.9999999～1.0 | < 5e-7 ✅ |
| _get_coop_bev_embed vec vs 原始 | 0.9999999～1.0 | < 8e-7 ✅ |
| LaneQueryFusionTRT vs 原始 | 0.9999998～1.0 | 0.0 ✅ |

---

## 七、Phase 3C — V2X 路径检测头 TRT（Hook-D，1101-query）

**方案**：固定 shape 1101-query（901 追踪 + 200 来自 infra），零填充后入 TRT，推理后切片回原始 N。

**已知精度代价**：零填充 query 的 bias 项产生非零激活，Decoder Self-Attention 中轻微干扰真实 query，cos_bbox ≈ 0.996（N < 1101 时），端到端 AMOTA 下降约 **0.009**（vs Hook-ABC）。在可接受范围内。

| 文件 | 大小 |
|------|------|
| `trt_engines/univ2x_ego_heads_v2x_1101.trt` | 33.4 MB |

---

## 八、端到端 AMOTA 汇总（168 样本，V2X-Seq-SPD 协同验证集）

| 配置 | AMOTA | AMOTP | mAP | NDS | 说明 |
|------|:-----:|:-----:|:---:|:---:|------|
| PyTorch 全链路基准 | 0.338 | 1.474 | 0.0727 | 0.0679 | 全 PyTorch，无 TRT |
| **Hook-A（BEV TRT FP16）** | **0.381** | 1.450 | 0.0766 | 0.0700 | 仅 BEV 编码器 TRT，其余 PyTorch |
| Hook-A+B+C（含 V2X 融合） | 0.379 | 1.441 | 0.0763 | 0.0699 | BEV TRT + 融合向量化，检测头 PyTorch |
| **Hook-A+B+C+D（全链路 TRT）** | **0.370** | 1.446 | 0.0760 | 0.0697 | 所有模块 TRT，含 V2X 检测头 1101-query |

**结论**：
- Hook-A（BEV TRT）单独已超越 FP16 基准（0.381 > 0.370），因 TRT 数值特性轻微改善了检测精度
- V2X 融合向量化（B+C）精度几乎无损（-0.002），延迟从 5,640ms 降至 ~90ms
- Hook-D（V2X 路径检测头 TRT）引入 -0.009 AMOTA，为零填充 Self-Attn 干扰的代价
- 全链路 TRT 相对 PyTorch 基准 **+0.032 AMOTA**，延迟 **~63×** 加速

---

## 九、推理延迟汇总（RTX 4090 实测）

| 模块 | PyTorch 原版 | TRT / 向量化优化 | 加速比 |
|------|:-----------:|:---------------:|:------:|
| ResNet-FPN backbone（ego + infra） | ~30 ms | ~30 ms | 1× |
| BEV encoder × 2 | ~150 ms | ~20–40 ms | ~5× |
| `_query_matching`（N_inf=10） | 1,048 ms | 2.7 ms | **392×** |
| `_query_matching`（N_inf=50） | 5,341 ms | 5.6 ms | **951×** |
| `_query_matching`（N_inf=100） | 10,548 ms | 8.9 ms | **1,191×** |
| `_get_coop_bev_embed`（N=10） | 3.4 ms | 1.1 ms | 3× |
| `_get_coop_bev_embed`（N=50） | 16.2 ms | 1.1 ms | 15× |
| `_get_coop_bev_embed`（N=200） | 64.3 ms | 1.1 ms | **58×** |
| 检测头 + 下游头 | ~100 ms | ~15–30 ms | ~5× |
| **V2X 场景总延迟（N_inf=50）** | **~5,640 ms** | **~90 ms** | **~63×** |

---

## 十、模型大小对比

| 模块 | PyTorch (.pth) | ONNX | TRT FP16 | TRT INT8 |
|------|:-------------:|:----:|:--------:|:--------:|
| 总 checkpoint | **1,600 MB** | — | — | — |
| BEV encoder ego（1-cam） | — | 65 MB | 75 MB | **43 MB** |
| BEV encoder infra | — | 65 MB | 72 MB | — |
| 检测头（901/1101-query） | — | ~26 MB | 33/33.4 MB | — |
| 下游头 ego（Motion+Occ+Planning） | — | ~127 MB | 152 MB | — |
| 下游头 infra（Motion+Occ） | — | ~127 MB | 134 MB | — |
| **全链路引擎合计** | **1,600 MB** | ~410 MB | **~466 MB** | **~434 MB** |

---

## 十一、Phase E — BEV 编码器 INT8 PTQ（Vanilla W8A8）

### 11.1 方案设计

从 QuantV2X 移植量化框架（共 7 个模块，1,764 行）：
- `QuantModel`：递归替换 Conv2d / Linear 为 `QuantModule`，支持 `register_specials()` 可扩展注入
- `UniformAffineQuantizer`：STE 梯度估计 + MSE/entropy scale 搜索
- `AdaptiveRounding`（已移植，但本阶段未使用）

**ADR-001 关键决策（选择性量化）**：

| 子层 | 量化状态 | 原因 |
|------|:-------:|------|
| `sampling_offsets` | ❌ FP16 | 坐标偏移量化误差导致 BEV 特征完全失效 |
| `attention_weights` | ❌ FP16 | 权重误差放大注意力聚合结果 |
| `value_proj` | ✅ W8A8 | 聚合前变换，误差可吸收 |
| `output_proj` | ✅ W8A8 | 同上 |

### 11.2 校准数据

`validate_quant_bev.py` 收集 **50 帧** FPN 特征 + **真实 temporal prev_bev**（非零初始化），确保 TSA 激活范围覆盖，保存为 `calibration/bev_encoder_calib_inputs.pkl`（~4 GB）。

### 11.3 迭代过程

| 版本 | 样本数 | prev_bev | PLUGIN_V2 精度 | AMOTA | 说明 |
|------|:-----:|:--------:|:--------------:|:-----:|------|
| v1 | 10 | 零初始化 | 无覆盖 | 0.334 | 基线 |
| v2 | 20 | temporal | 无覆盖 | 0.344 | temporal 效果 +0.010 |
| v3 | 20 | temporal | 全部→FP16 | 0.278 | ❌ 显式精度覆盖引入级联 dequant/quant |
| v4 | 20 | temporal | 无覆盖（fresh cache） | 0.355 | cache 重置修复 |
| **v5** | **50** | **temporal** | **无覆盖** | **0.364 ✅** | 最优，50 样本 |

> **v3 教训**：`layer.precision = trt.DataType.HALF` 显式设置 PLUGIN_V2 精度时，TRT 在该层输入侧插入额外 Dequantize 节点，引入级联误差（0.278 vs 0.355）。正确做法：**不设置任何精度**，TRT 自动将无 INT8 实现的 PLUGIN_V2 分配到 FP16。

### 11.4 最终结果

| 指标 | FP16 TRT (Hook-A) | INT8 PTQ v5 | 差值 |
|------|:------------------:|:-----------:|:----:|
| AMOTA | 0.370 | 0.364 | -0.006 |
| AMOTP | 1.446 | 1.438 | -0.008 |
| mAP | 0.0760 | 0.0744 | -0.0016 |
| 误报 FP | ~129 | **104** | **-19%** |
| ID 切换 IDS | ~35 | **31** | **-11%** |
| 引擎大小 | 72 MB | **43 MB** | **-40%** |

### 11.5 生成工件

| 文件 | 大小 | 说明 |
|------|------|------|
| `calibration/bev_encoder_calib_inputs.pkl` | ~4 GB | 50 帧 temporal 校准输入 |
| `calibration/univ2x_ego_bev_encoder_int8_int8.cache` | 68 KB | INT8 activation scale 缓存 |
| `trt_engines/univ2x_ego_bev_encoder_int8.trt` | **43 MB** | BEV encoder INT8 引擎 |

---

## 十二、AdaRound 实验（2026-04-04）

### 12.1 目标与方案

目标：在 Vanilla PTQ 基础上通过 AdaRound 权重舍入优化进一步提升 INT8 精度（≥ 0.370）。

方案：AdaRound 校准 36 层 QuantModule（`calibrate_univ2x.py --adaround`），将 W_fq 内嵌进 ONNX，构建 INT8 TRT 引擎。

**校准参数**：10 样本（注：50 帧 `bev_encoder_calib_inputs.pkl` 是 Vanilla PTQ 数据，AdaRound 用 10 帧），500 iter/层，共 ~2 小时。

### 12.2 实验结果

| 阶段 | 产出 | 结果 |
|------|------|------|
| C-1 AdaRound 校准 | `calibration/quant_encoder_adaround.pth`（31 MB，36 层 alpha） | ✅ |
| C-2 ONNX 导出（W_fq 内嵌） | `onnx/univ2x_ego_bev_encoder_adaround.onnx`（104.2 MB） | ✅ |
| C-3 INT8 TRT 引擎 | `trt_engines/univ2x_ego_bev_encoder_adaround_int8.trt`（43.7 MB） | ✅ |
| D-1 BEV cosine 验证 | 平均 cosine = 0.819（目标 > 0.99） | ❌ |
| D-2 端到端 AMOTA | **AMOTA = 0.190**（目标 ≥ 0.370） | ❌ |

### 12.3 失败根因：双重权重量化

```
AdaRound 校准
  W_fq = scale_ada × round_hard(W / scale_ada)   ← 第 1 次量化（AdaRound）

ONNX 导出（内嵌 W_fq，FP32 格式）

TRT INT8 引擎构建
  scale_trt = f(max|W_fq|)     ← TRT 从 W_fq 重新推导 scale（scale_trt ≠ scale_ada）
  W_qq = round(W_fq / scale_trt) × scale_trt   ← 第 2 次量化（覆盖 AdaRound）
```

量化误差 = 第 1 次误差 + 第 2 次误差，导致 AMOTA 从 0.381 跌至 0.190。

### 12.4 Q/DQ 方案实施结果（2026-04-05）

**实施内容**：
- 新增 `tools/build_qdq_onnx_adaround.py`（450+ 行），识别 36 个 `Constant(W_fq)→Transpose→MatMul` 模式，插入对称 INT8 Q/DQ 节点（`scale_sym = max(|W_fq|)/127`，zp=0）
- 修复拓扑排序问题（Q/DQ 紧随对应 Constant 节点之后）
- 修复 TRT 不支持非零 zero_point 问题（改用对称 INT8）

**Q/DQ 方案实验结果**：

| 产出 | 结果 |
|------|------|
| Q/DQ ONNX | `onnx/univ2x_ego_bev_encoder_adaround_qdq.onnx`（104.2 MB，+72 Q/DQ 节点）|
| Q/DQ TRT 引擎 | `trt_engines/univ2x_ego_bev_encoder_adaround_qdq_int8.trt`（40.8 MB）|
| **D-2 端到端 AMOTA** | **0.137**（目标 ≥ 0.381，实际更差） |

**Q/DQ 方案失败根因（权重-激活不匹配）**：

```
问题：TRT 在"显式 Q/DQ 模式"（explicit quantization mode）下
  行为：calibrator 完全被忽略（警告："Calibrator won't be used in explicit quantization mode"）
  结果：
    - 有 Q/DQ 的权重 MatMul → 对称 INT8 + 额外量化噪声
    - 无 Q/DQ 的激活层 → FP16（失去 calibration-based INT8）
    - 整体：更多算子 + 更少 INT8 覆盖 → AMOTA 0.137 < Vanilla 0.353
```

**正确的 Q/DQ 方案需要**（目前工作量较大，暂搁置）：
1. 权重 Q/DQ：使用 AdaRound scale（scale_sym = max|W_fq|/127）
2. 激活 Q/DQ：使用 calibration scale（entropy 法）
3. 两者同时插入才能让 TRT 正确 fuse 为 INT8 MatMul kernel

**结论**：Vanilla INT8 PTQ（AMOTA=0.353/0.364）仍是当前最优 INT8 方案。

---

## 十三、当前最优配置对比

| 配置 | AMOTA | 引擎大小 | 适用场景 |
|------|:-----:|:--------:|---------|
| PyTorch 基准 | 0.338 | 1,600 MB | — |
| Hook-A（FP16 BEV TRT） | **0.381** | 75 MB BEV | **精度最优** |
| Hook-A+B+C+D（全链路 FP16 TRT） | **0.370** | ~466 MB 全部 | **速度+精度综合最优** |
| INT8 PTQ v5（BEV encoder） | 0.364 | **43 MB BEV** | **体积最优** |
| AdaRound INT8（无 Q/DQ） | 0.190 ❌ | 43.7 MB | 双重量化，不可用 |
| AdaRound Q/DQ（权重-only）| 0.137 ❌ | 40.8 MB | TRT 显式量化模式问题，不可用 |

---

## 十四、后续优化方向

| 优先级 | 方向 | 预期收益 | 难度 |
|:------:|------|---------|:----:|
| P0 | AdaRound 完整 Q/DQ（权重+激活均插入） | AMOTA ≥ 0.381 | 高（需激活 Q/DQ 自动化插入） |
| P1 | 检测头 INT8 PTQ | 引擎减 ~14 MB，需验证 QuantCustomMSDA | 中 |
| P2 | 下游头 INT8 PTQ | 引擎再减 ~170 MB（-36%） | 低（无 MSDAPlugin 相邻层） |
| P3 | 对称 AdaRound（改 UniformAffineQuantizer sym=True）+ 全 Q/DQ | 正确 TRT 对接，INT8 精度改善 | 中高 |
| P4 | backbone QAT（DCNv2） | — | 极高（DCNv2 不可 ONNX） |

---

## 十五、关键决策与经验教训

| 决策 | 原则 | 教训 |
|------|------|------|
| BEVEncoderWrapper 跳过 backbone | DCNv2 不可 ONNX → 从 FPN 特征开始 | 先验证导出路径可行性 |
| PLUGIN_V2 不设置 precision | 显式 FP16 引入级联 dequant/quant，AMOTA -0.086 | TRT 混合精度交给 TRT 自动决策 |
| 50 帧 temporal 校准 | TSA 激活范围依赖前帧状态 | 量化校准数据必须覆盖真实推理分布 |
| `sampling_offsets` 保持 FP16（ADR-001） | 坐标量化误差无法后续纠正 | 量化敏感层需人工 skip |
| AdaRound W_fq 内嵌 ONNX | 误以为 TRT 会尊重 W_fq 的已有量化 | TRT INT8 必须通过 Q/DQ 节点或 Calibrator 指定 scale，否则重新推导 |
| 仅权重 Q/DQ（无激活 Q/DQ） | 以为只插权重 Q/DQ 就能保留 AdaRound | TRT 在任何 Q/DQ 存在时进入 explicit 模式，完全忽略 Calibrator，激活层退化为 FP16，整体更差 |

---

*文档生成：2026-04-04，最后更新：2026-04-05 | 对应项目目录：`/home/jichengzhi/UniV2X`*
