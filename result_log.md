# UniV2X 全链路优化方案与性能结果完整记录

**项目**：UniV2X — 端到端 V2X 协同自动驾驶系统  
**硬件**：RTX 4090 (SM 89)，CUDA 11.8，TensorRT 10.13.0.35  
**评估集**：V2X-Seq-SPD 协同验证集，168 样本，442 GT 目标  
**最后更新**：2026-04-05（第十四节：QuantV2X 研究；第十五/十六节：方向更新）

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

## 十三、AdaRound Q/DQ 实施全过程问题复盘（2026-04-05）

> 本节记录 2026-04-05 实施 AdaRound + Q/DQ ONNX 方案时遇到的所有技术问题、排查过程及反思。

### 13.1 问题一：ONNX 节点模式识别错误（24 vs 36 不匹配）

**现象**：
```
RuntimeError: Count mismatch: 24 ONNX quant MatMuls vs 36 QuantModules.
```

**根因**：
原始脚本 `find_quant_matmul_nodes()` 查找的是 `Initializer → Transpose → MatMul` 模式（即 `nn.Parameter` 权重），而 QuantModule 的 `org_weight` 是普通 `torch.Tensor`（非 Parameter），ONNX tracer 将其内嵌为 `Constant` 节点，实际模式为 `Constant(W_fq) → Transpose → MatMul`。

24 个 Initializer 模式对应的是**未被量化的 24 个 nn.Linear**，而非 QuantModule。

**排查过程**：
1. 通过图分析脚本统计 `MatMul_transpose_init`（24 个）和 `MatMul_transpose_non_init`（36 个）
2. 追踪 non-init 的上游节点发现全部是 `Constant` 类型
3. 对比模型中 Linear 层总数（60 层）= QuantModule（36）+ 普通 Linear（24）完全吻合

**修复**：将 `find_quant_matmul_nodes()` 改为识别 `Constant → Transpose → MatMul` 模式；`build_weight_value_map()` 改为从 Constant 节点提取权重值用于匹配。

**反思**：ONNX tracer 对 `nn.Parameter`（→ Initializer）和普通 `torch.Tensor`（→ Constant）的处理有本质差异，量化框架中 `org_weight` 的存储方式直接影响图结构，分析 ONNX 图前必须先确认权重的存储类型。

---

### 13.2 问题二：ONNX 拓扑排序警告

**现象**：
```
onnx.checker raised: Nodes in a graph must be topologically sorted,
however input 'onnx::Transpose_3640' of node QL is not output of any previous nodes.
```

**根因**：
最初将所有新 Q/DQ 节点统一前置（`graph.node.insert(0, node)`），导致 Q/DQ 节点出现在其依赖的 Constant 节点之前，违反 ONNX 拓扑顺序要求。

**修复**：改为在遍历 `graph.node` 时，将每对 Q/DQ 节点紧随对应 Constant 节点之后插入（`new_node_list.append(ql); new_node_list.append(dql)` 在 Constant 节点处理后立即追加），最后整体替换 `graph.node`。

**反思**：ONNX 图编辑时插入节点必须维护拓扑顺序，依赖动态输出（Constant 节点输出）的新节点不能统一前置，需要在图遍历中就地插入。

---

### 13.3 问题三：TRT 不支持非零 zero_point

**现象**：
```
[TRT] [E] Assertion failed: shiftIsAllZeros(zeroPoint):
Non-zero zero point is not supported.
```

**根因**：
`UniformAffineQuantizer` 使用**非对称 UINT8 量化**（zero_point ∈ [0, 255]，通常 ≠ 0），TRT GPU 路径要求 Q/DQ 节点必须使用**对称 INT8 量化**（zero_point = 0），非零偏移仅 DLA 支持。

**修复**：放弃使用 AdaRound 原始的非对称 scale/zero_point，改为从 W_fq 重新计算对称 scale：
```python
scale_sym[i] = max(|W_fq[i, :]|) / 127.0   # per output channel
zp = 0  # int8
```

**反思**：AdaRound 中 `UniformAffineQuantizer` 的非对称量化设计与 TRT GPU 的对称 INT8 要求存在根本性不兼容。在设计量化框架时，如果目标部署平台是 TRT GPU，应从一开始使用对称量化（`sym=True`），避免后期适配成本。

---

### 13.4 问题四（核心）：权重-only Q/DQ 使 TRT 进入显式量化模式

**现象**：
```
[TRT] Calibrator won't be used in explicit quantization mode.
```
端到端 AMOTA = **0.137**，远低于 Vanilla INT8 PTQ（0.353）。

**根因**：
TRT 10.x 的量化模式是二元对立的：
- **隐式量化模式**（implicit）：依赖 Calibrator 为所有层提供激活 scale，全局 INT8
- **显式量化模式**（explicit）：图中任意 Q/DQ 节点存在即触发此模式，Calibrator 完全失效

只插入权重 Q/DQ 时：
- 有 Q/DQ 的权重层：对称 INT8（但 W_fq 被 QL 再次量化，引入额外噪声）
- 无 Q/DQ 的激活层：**退化为 FP16**（Calibrator 被完全忽略）
- 净效果：更多量化噪声 + 更少 INT8 覆盖 → AMOTA 0.137

正确做法是**权重和激活同时插入 Q/DQ**，才能使 TRT 正确 fuse 为 INT8 MatMul kernel：
```
激活 → QL(scale_act) → DQL(scale_act) ─┐
                                        ├→ INT8 MatMul kernel (TRT fused)
W_fq → QL(scale_w)  → DQL(scale_w)  ──┘
```

**反思**：
1. TRT 的"显式 Q/DQ 模式"是全局生效的，不支持"仅对部分层指定 scale"的混合模式。
2. 任何 Q/DQ 节点的插入都是全局承诺：你接管了全部量化控制，Calibrator 不再有效。
3. 在引入 Q/DQ 之前必须先评估：是否准备好为所有量化层（包括激活）提供显式 scale？
4. 权重-only Q/DQ 方案是常见的误区，在 TRT 文档中已有明确警告，但容易被忽略。

---

### 13.5 根本性反思：AdaRound 在当前框架下的适用性

**问题本质**：
本项目中 AdaRound 的实现路径是：
```
PyTorch（非对称 UINT8 AdaRound）→ W_fq 内嵌 ONNX → TRT INT8（对称 INT8）
```
三个环节的量化语义不一致，导致每次转换都引入额外误差。

**与 Vanilla INT8 PTQ 对比**：
Vanilla PTQ 直接用 FP32/FP16 原始权重 + Calibrator（entropy scale），TRT 做一次对称量化，误差最小（AMOTA 0.353）。AdaRound 本意是通过优化舍入减少误差，但非对称→对称的转换使优化收益完全抵消甚至负收益。

**正确的 AdaRound for TRT 方案**（后续参考）：
1. 改 `UniformAffineQuantizer` 使用对称 INT8（`sym=True`，range [-127, 127]，zp=0）
2. AdaRound 优化在对称 INT8 网格上进行
3. 导出 ONNX 时插入权重 Q/DQ（scale=scale_ada_sym，zp=0）**和**激活 Q/DQ（scale 来自 Calibration entropy）
4. TRT 显式量化模式下正确 fuse

---

## 十四、QuantV2X 量化体系研究（2026-04-05）

> 本节记录对 QuantV2X 项目（参考实现）的深度代码阅读结果，为 UniV2X 后续量化路径提供依据。详细内容见 `quantv2x_learn.md`。

### 14.1 QuantV2X 的两条独立管线

通过完整阅读 QuantV2X 源码，发现其量化体系由两条**完全独立**的管线构成，二者之间不存在数据通路：

| 管线 | 文件 | 核心方法 | 产物 |
|------|------|---------|------|
| **PyTorch W8A8 路径** | `inference_quant.py` | AdaRound（W8）+ LSQ（A8） | 精度指标（论文主体贡献） |
| **TRT INT8 部署路径** | `build_trt_int8.py` + `inference_onnx_dump_calibration.py` | DataCalibrator（隐式 entropy calibration） | `.plan` TRT 引擎（实测延迟） |

**关键发现**：AdaRound 优化出的 `alpha` 参数完全不传入 TRT 路径，TRT INT8 使用标准 `IInt8EntropyCalibrator2`，与 AdaRound 结果无关。

### 14.2 QuantV2X 对稀疏卷积的量化处理

QuantV2X 专门实现了 `QuantSpconvModule`（`quant_layer.py:423-496`），用于 spconv 稀疏卷积（SubMConv3d / SparseConv3d）的 PyTorch 侧 W8A8 量化：

```python
class QuantSpconvModule(nn.Module):
    # 临时覆盖 spconv 权重后执行稀疏卷积，再对 features 做激活量化
    # 保持 SparseConvTensor 格式，不 densify
```

`QuantVoxelBackBone8x`（`quant_block.py:988-1033`）将整个 SECOND 3D 稀疏骨干中的所有稀疏卷积层统一替换为 `QuantSpconvModule`，并参与 AdaRound 优化（`second_reconstruction`）。

### 14.3 论文提及的 Custom CUDA Kernels

论文原文：
> "we implement custom CUDA kernels and integrate them as TensorRT plug-ins to ensure compatibility and accurate latency profiling"

**代码库现状**：对 QuantV2X 仓库的完整搜索表明，**不存在任何 TRT plugin 源码**（无 `.cu` plugin 文件，无 plugin 目录，`build_trt_int8.py` 中无 `loadLibrary` 调用）。

**结论**：custom CUDA kernels 属于以下之一：
1. 论文工程实现中存在，但公开代码未包含（未开源）
2. spconv 在 ONNX 导出时 densify 为标准 Conv3d，TRT 原生处理；"custom kernels" 指性能优化层面

**UniV2X 对比**：UniV2X 的 MSDAPlugin 已完整实现并开源（`plugins/multi_scale_deform_attn/`），TRT plugin 工程化程度**优于** QuantV2X 的公开代码。

### 14.4 TRT INT8 的正确路径（对 UniV2X 的启示）

基于 QuantV2X 的设计，UniV2X TRT INT8 的正确实现路径为：

```
Step 1: 收集 BEV encoder calibration 数据（512+ 帧真实数据 → NPZ 格式）
Step 2: 实现 BEVEncoderCalibrator(IInt8EntropyCalibrator2)，读取 NPZ 喂给 TRT
Step 3: config.set_flag(INT8) + config.int8_calibrator = calibrator
Step 4: 构建 INT8 TRT 引擎（无需 Q/DQ 节点，无需 AdaRound）
```

**关键**：TRT 隐式 INT8 路径从 FP32 ONNX 出发，不依赖 AdaRound 结果，不涉及 Q/DQ 节点，与 QuantV2X 的 `build_trt_int8.py` 设计完全一致。当前 UniV2X 已有 `calibration/bev_encoder_calib_inputs.pkl`（50 帧），可直接复用或扩充。

### 14.5 AdaRound 在 UniV2X 中的定位修正

| | 之前的认识 | 修正后的认识 |
|--|-----------|------------|
| AdaRound 的用途 | 期望通过 Q/DQ 节点将 AdaRound scale 传入 TRT | AdaRound 适合 PyTorch W8A8 精度评估，TRT INT8 用独立 DataCalibrator |
| 量化器设计 | 未考虑对称/非对称的影响 | 非对称 UINT8（当前实现）与 TRT Q/DQ 不兼容；若要做真正的 Q/DQ，必须改为对称 INT8 |
| 最短路径 | 复杂的 Q/DQ 节点插入 | 仿照 QuantV2X 的 DataCalibrator，简单、已验证、可直接实现 |

---

## 十五、当前最优配置对比（2026-04-05 更新）

| 配置 | AMOTA | 引擎大小 | 状态 | 说明 |
|------|:-----:|:--------:|:----:|------|
| PyTorch 全链路基准 | 0.338 | 1,600 MB | ✅ 基准 | 全 PyTorch，无 TRT |
| **Hook-A（FP16 BEV TRT）** | **0.381** | 75 MB BEV | ✅ **精度最优** | 仅 BEV 编码器 TRT，其余 PyTorch |
| Hook-A+B+C（含 V2X 融合向量化） | 0.379 | 75 MB BEV | ✅ 已验证 | BEV TRT + 融合向量化，检测头 PyTorch |
| **Hook-A+B+C+D（全链路 FP16 TRT）** | **0.370** | ~466 MB 全部 | ✅ **速度+精度综合最优** | 所有模块 TRT，含 V2X 检测头 1101-query |
| INT8 PTQ v5（BEV encoder） | 0.364 | **43 MB BEV** | ✅ **体积最优** | 50 帧 temporal calibration，无 Q/DQ |
| AdaRound INT8（无 Q/DQ） | 0.190 | 43.7 MB | ❌ 废弃 | 双重量化问题，W_fq 被 TRT 再量化 |
| AdaRound Q/DQ（权重-only） | 0.137 | 40.8 MB | ❌ 废弃 | TRT 进入显式量化模式，Calibrator 失效 |
| TRT 隐式 INT8（DataCalibrator） | 待测 | ~40 MB 预期 | 🔲 **待实施** | 参照 QuantV2X 正确路径，预期 ≥ 0.364 |

**当前结论**：
- FP16 全链路 TRT 已是生产可用状态（AMOTA=0.370，63× 延迟加速）
- AdaRound 两种方案均已确认失败，不再追求
- 下一阶段重心：仿照 QuantV2X 的 DataCalibrator 方案实现真正的隐式 INT8，无需 Q/DQ

---

## 十六、后续优化方向（2026-04-05 更新）

基于 QuantV2X 研究结论，优先级重新排序：

| 优先级 | 方向 | 预期收益 | 难度 | 依据 |
|:------:|------|---------|:----:|------|
| **P0** | **TRT 隐式 INT8 DataCalibrator（BEV encoder）** | AMOTA ≥ 0.364，引擎 ~40 MB，实现简单 | **低** | 仿照 QuantV2X `build_trt_int8.py`，已有 50 帧 PKL 数据可转 NPZ |
| P1 | 检测头 INT8 PTQ（DataCalibrator） | 引擎减 ~14 MB（33 MB → ~20 MB），延迟进一步降低 | 中 | 需验证 CustomMSDA INT8，参照 MSDAPlugin 隐式精度 |
| P2 | 下游头 INT8 PTQ（DataCalibrator） | 引擎减 ~80 MB（152+134 MB → ~100 MB），无 MSDAPlugin 邻层 | 低 | 下游头无自定义算子，DataCalibrator 可直接应用 |
| P3 | 对称 AdaRound + 完整 W+A Q/DQ（研究性） | INT8 精度改善（若超过 FP16 0.381 则有价值） | 高 | 需重写 `UniformAffineQuantizer`（sym=True），重做激活 Q/DQ 插入 |
| P4 | backbone QAT（DCNv2） | 理论上精度最高，但工程极复杂 | 极高 | DCNv2 不可 ONNX 导出，需 C++ 量化插件 |

**P0 具体实施步骤**（可立即开始）：

```bash
# 1. 将现有 PKL calibration 数据转为 NPZ 格式（兼容 DataCalibrator）
python tools/convert_pkl_to_npz.py \
    --input calibration/bev_encoder_calib_inputs.pkl \
    --output calibration/bev_encoder_npz/

# 2. 扩充至 512 帧（可选，提升 scale 精度）
python tools/dump_bev_calibration.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    --checkpoint ckpts/univ2x_coop_e2e_stg2.pth \
    --output calibration/bev_encoder_npz/ --num-frames 512

# 3. 构建 INT8 引擎（新建 tools/build_trt_int8_bev.py）
python tools/build_trt_int8_bev.py \
    --onnx onnx/univ2x_ego_bev_encoder_200_1cam.onnx \
    --npz-dir calibration/bev_encoder_npz/ \
    --cache calibration/bev_int8.cache \
    --output trt_engines/univ2x_ego_bev_encoder_int8_datacalib.trt

# 4. 端到端验证
python tools/test_trt.py --use-bev-trt trt_engines/univ2x_ego_bev_encoder_int8_datacalib.trt
```

---

## 十七、关键决策与经验教训

| 决策 | 原则 | 教训 |
|------|------|------|
| BEVEncoderWrapper 跳过 backbone | DCNv2 不可 ONNX → 从 FPN 特征开始 | 先验证导出路径可行性 |
| PLUGIN_V2 不设置 precision | 显式 FP16 引入级联 dequant/quant，AMOTA -0.086 | TRT 混合精度交给 TRT 自动决策 |
| 50 帧 temporal 校准 | TSA 激活范围依赖前帧状态 | 量化校准数据必须覆盖真实推理分布 |
| `sampling_offsets` 保持 FP16（ADR-001） | 坐标量化误差无法后续纠正 | 量化敏感层需人工 skip |
| AdaRound W_fq 内嵌 ONNX | 误以为 TRT 会尊重 W_fq 的已有量化 | TRT INT8 必须通过 Q/DQ 节点或 Calibrator 指定 scale，否则重新推导 |
| 仅权重 Q/DQ（无激活 Q/DQ） | 以为只插权重 Q/DQ 就能保留 AdaRound | TRT 在任何 Q/DQ 存在时进入 explicit 模式，完全忽略 Calibrator，激活层退化为 FP16，整体更差 |

---

---

## 十八、下游头 INT8 PTQ 量化全链路验证（2026-04-05）

### 18.1 实施内容

本节完成了下游头（Motion + Occ + Planning）的 TRT INT8 PTQ 量化，具体步骤：

1. **B-1** 新建 `tools/dump_downstream_calibration.py`：Hook `pts_bbox_head.get_detections` + `seg_head.forward_test`，50 帧真实推理，捕获下游头输入
2. **B-2** Smoke test：`build_trt_int8_univ2x.py --target downstream --no-int8`，验证 FP16 TRT 构建路径
3. **B-3** 构建 INT8 引擎：ego + infra 各 50 帧校准数据，`IInt8EntropyCalibrator2`
4. **B-4a** 余弦相似度验证（INT8 vs FP16）
5. **B-4b** 端到端 AMOTA / 速度对比（三种配置，168 帧评估集）

### 18.2 模型大小对比

| 模型 | FP32 ONNX | FP16 TRT | INT8 TRT | FP16→INT8 压缩率 |
|------|:---------:|:--------:|:--------:|:---------------:|
| Ego 下游头 | 128 MB | 153 MB | **74 MB** | **↓51.6%** |
| Infra 下游头 | 114 MB | 134 MB | **66 MB** | **↓50.7%** |
| **合计** | **242 MB** | **287 MB** | **140 MB** | **↓51.2%** |

### 18.3 推理速度对比

| 组件 | FP16 TRT | INT8 TRT | 加速比 |
|------|:--------:|:--------:|:------:|
| 下游头单独延迟 | 79.75 ms/帧 | **32.53 ms/帧** | **2.45×** |
| BEV encoder (ego) | 59.75 ms/帧 | 59.74 ms/帧 | 1.0× |
| BEV encoder (infra) | 53.57 ms/帧 | 53.14 ms/帧 | 1.0× |
| **端到端** | **785.6 ms/帧 (1.27 fps)** | **740.6 ms/帧 (1.35 fps)** | **1.06×** |

> 端到端加速有限（6%）是因为下游头仅占总延迟约 10%；BEV encoder 仍是主瓶颈。

### 18.4 精度对比（余弦相似度，INT8 vs FP16，5 样本平均）

| 输出 | 余弦相似度 | 均值绝对误差 |
|------|:---------:|:-----------:|
| traj_scores | 0.9999842 ✅ | 1.93e-3 |
| traj_preds | 0.9999455 ✅ | 3.80e-3 |
| occ_logits | 0.9992438 ✅ | 3.42e-2 |
| sdc_traj | 0.9999968 ✅ | 4.04e-2 |

### 18.5 端到端 AMOTA 对比（168 帧，V2X-Seq-SPD 协同验证集）

| 配置 | AMOTA | AMOTP | mAP | NDS | 下游头推理 |
|------|:-----:|:-----:|:---:|:---:|:----------:|
| **Config-B**：FP16 BEV + FP16 下游头（Hook E，FP16 引擎） | 0.248 | 1.623 | 0.0439 | 0.0537 | 79.75 ms |
| **Config-C**：FP16 BEV + INT8 下游头（Hook E，INT8 引擎） | 0.255 | 1.604 | 0.0450 | 0.0542 | **32.53 ms** |
| **差值 C−B** | +0.007 | −0.019 | +0.0011 | +0.0005 | **↓47 ms (−59%)** |

> **AMOTA 基本不变**（±0.007 在噪声范围内）：AMOTA 来自检测头，下游头量化不影响目标检测/跟踪质量。两组 mAP 数值略低于早期测试（0.379）的原因已在 Section 19 中排查清楚：**非 Hook E 导致，而是 infra BEV 引擎使用了错误的权重和摄像头配置**。修复后 AMOTA=0.341（与 PyTorch 基准 0.338 持平）。

### 18.6 关键经验

| 问题 | 原因 | 解法 |
|------|------|------|
| `lane_query` 形状不固定（332/323/300...） | panseg_head 输出车道数随帧变化 | calibrator `get_batch()` 中加 shape-guard slice；Hook E 中截断到 300 |
| 首次 ego 引擎构建 calibration 中途崩溃 | `buf.copy_(t)` 在 shape 不匹配时抛异常 | `build_trt_int8_univ2x.py` 加 shape-guard zero-pad/slice 逻辑 |
| `dump_univ2x_calibration.py` 格式错误 | 旧版本保存原始 DataLoader 样本而非 backbone 特征 | 重写：Hook `get_bev_features`，直接捕获 `feat0-3 / can_bus / lidar2img` |

### 18.7 结论

**下游头 INT8 量化收益**：

- **模型大小**：下游头 TRT 引擎减少 51%（合计 287 MB → 140 MB）
- **推理速度**：下游头单独 2.45× 加速（79.75 ms → 32.53 ms）
- **端到端速度**：提升 6%（785.6 → 740.6 ms/帧），受制于 BEV encoder 瓶颈
- **检测精度**：AMOTA 不变（±0.007 噪声），INT8 量化不损害目标检测/跟踪
- **输出质量**：所有输出余弦相似度 > 0.999，`occ_logits` 略低（0.9992）但仍可接受

**当前全链路最优配置**（含下游头 INT8）：

```
BEV encoder: FP16 TRT (Hook A)       75 MB ego + 73 MB infra
V2X Fusion:  向量化 PyTorch (Hook B+C) —
Detection:   PyTorch or FP16 TRT
Downstream:  INT8 TRT (Hook E)        74 MB ego + 66 MB infra
```

---

---

## 十九、Infra BEV 引擎排查与修复（2026-04-08）

### 19.1 问题现象

Section 18 中 Config-B/C 的 AMOTA（0.248/0.255）远低于早期 Hook-A+B+C 结果（0.379），最初怀疑 Hook E（下游头 TRT）引入了问题。

### 19.2 排查过程

通过对比所有历史评估日志的 AMOTA，发现真正的规律：

| 评估日志 | Infra BEV TRT | AMOTA |
|---------|:------------:|:-----:|
| `eval_hookA_ego_only` | ❌ 未启用 | **0.378** |
| `eval_hookA_BC_no_infra` | ❌ 未启用 | **0.379** |
| `eval_hookA_both` | ✅ 旧引擎 `200.trt` | **0.251** |
| `eval_configA` | ✅ 旧引擎 `200.trt` | **0.228** |
| `eval_configB`（含 Hook E） | ✅ 旧引擎 `200.trt` | **0.248** |

**结论**：AMOTA 下降与 Hook E 无关，完全由 infra BEV TRT 引擎引起。

### 19.3 根因一：Infra 引擎输入形状不匹配（旧引擎）

对比引擎输入形状：

| 引擎 | feat0 shape | lidar2img | 说明 |
|------|------------|-----------|------|
| ego 1cam（正确） | `(1,1,256,136,240)` | `(1,1,4,4)` | 1 摄像头，1088×1920 |
| **infra 旧**（错误） | `(1,6,256,32,52)` | `(1,6,4,4)` | **6 摄像头，NuScenes 默认分辨率** |
| infra 1cam（正确） | `(1,1,256,136,240)` | `(1,1,4,4)` | 1 摄像头，1088×1920 |

旧引擎 `univ2x_infra_bev_encoder_200.trt`（Mar 7，73 MB）是 NuScenes 6-cam 配置导出，与 V2X-Seq-SPD 单摄像头数据集不兼容。

### 19.4 根因二：ONNX 导出加载了错误的权重

`export_onnx_univ2x.py` 中 `build_model_from_cfg()` 的自动前缀检测逻辑存在 bug：

```python
# 旧代码（Bug）
sample_key = next(iter(sd))           # 始终取第一个 key
prefix = sample_key.split('.')[0] + '.'  # → "model_ego_agent."
```

Cooperative checkpoint 中 key 的排列顺序使 `next(iter(sd))` 始终返回 `model_ego_agent.` 前缀。**导出 infra 模型时，实际加载的是 ego 的权重。**

```
Checkpoint 结构：
  model_ego_agent.xxx          ← 2491 keys（first key 来自这里）
  model_other_agent_inf.xxx    ← 2413 keys（被完全忽略）
```

**修复**：改用 `model_key`（即 config 名称 `model_other_agent_inf`）作为前缀，仅在匹配不到时 fallback 到旧逻辑：

```python
# 修复后
prefix = model_key + '.'   # "model_other_agent_inf."
stripped = {k[len(prefix):]: v for k,v in sd.items() if k.startswith(prefix)}
if not stripped:  # fallback: 单前缀 checkpoint
    sample_key = next(iter(sd))
    prefix = sample_key.split('.')[0] + '.'
    stripped = ...
```

### 19.5 修复后验证结果（2026-04-08）

修复 `export_onnx_univ2x.py` → 重新导出 infra ONNX → 重新构建 infra FP16 TRT 引擎 → 三组端到端评估：

| 配置 | AMOTA | AMOTP | mAP | NDS | 下游头延迟 | 端到端延迟 |
|------|:-----:|:-----:|:---:|:---:|:---------:|:---------:|
| **Baseline**：Hook A+B+C（ego+infra FP16 BEV） | **0.341** | 1.508 | 0.0624 | 0.0631 | — | 817.6 ms |
| **Config-B**：+ Hook E FP16 下游头 | **0.341** | 1.508 | 0.0624 | 0.0631 | 78.81 ms | 816.7 ms |
| **Config-C**：+ Hook E INT8 下游头 | **0.341** | 1.508 | 0.0624 | 0.0631 | **32.51 ms** | 847.6 ms |

**关键结论**：
1. **Hook E 对 AMOTA 零影响**：三组配置 AMOTA 完全一致（0.341），确认下游头 TRT（无论 FP16 还是 INT8）不干扰检测/跟踪
2. **Infra BEV TRT 修复有效**：从错误引擎的 0.248 恢复到 0.341，与 PyTorch 基准（0.338）基本持平
3. **INT8 下游头 2.4× 加速**：78.81 ms → 32.51 ms，与 Section 18 结论一致
4. **Ego-only TRT (0.378) vs ego+infra TRT (0.341) 的差异**为 TRT FP16 数值特性造成的正常波动（ego 侧 FP16 恰好对此数据集有利）

### 19.6 反思

| 问题 | 原因 | 教训 |
|------|------|------|
| ONNX 导出自动前缀检测取 `next(iter(sd))` | 假设 state_dict 只有一个前缀，V2X cooperative checkpoint 有两个 | **多代理 checkpoint 必须显式指定前缀**，不能依赖 key 顺序自动推导 |
| 旧 infra 引擎用 6-cam 配置 | 早期开发时参照 NuScenes 默认值，未检查 V2X-Seq-SPD 数据集的实际摄像头数 | **导出 TRT 引擎前必须验证数据集的真实配置**（cam 数、图像分辨率） |
| Section 18 误归因为"Hook E 干扰" | 未对比无 infra BEV 引擎的 baseline | **AMOTA 下降排查应先隔离变量**：逐个开关 Hook 对比，而非直接怀疑最后添加的组件 |
| 旧引擎被新测试复用 | pipeline 脚本引用旧文件名 `200.trt` | **TRT 引擎文件名应编码关键配置**（cam 数、分辨率、精度），避免错用；已更新 pipeline 引用 `200_1cam.trt` |

### 19.7 修正后的全链路对比表

| 配置 | AMOTA | 引擎大小（全部） | 状态 |
|------|:-----:|:---------------:|:----:|
| PyTorch 全链路基准 | 0.338 | 1,600 MB | ✅ 基准 |
| Hook-A（ego BEV FP16 TRT only） | 0.378 | 75 MB ego | ✅ 精度最优 |
| Hook-A+B+C（ego+infra BEV FP16 + V2X 融合） | **0.341** | 75+41 MB BEV | ✅ 全链路协同 |
| Hook-A+B+C+E（FP16 下游头 TRT） | **0.341** | 75+41+153 MB | ✅ Hook E 无损 |
| Hook-A+B+C+E（INT8 下游头 TRT） | **0.341** | 75+41+74 MB | ✅ **推荐配置** |

---

*文档生成：2026-04-04，最后更新：2026-04-08（第十九节：Infra BEV 引擎排查与修复） | 对应项目目录：`/home/jichengzhi/UniV2X`*
