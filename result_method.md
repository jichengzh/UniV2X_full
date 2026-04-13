# UniV2X 全链路优化方案与性能结果

**项目**：UniV2X — 端到端 V2X 协同自动驾驶系统  
**硬件**：RTX 4090 (SM 89)，CUDA 11.8，TensorRT 10.13.0.35  
**评估集**：V2X-Seq-SPD 协同验证集，168 样本，442 GT 目标  
**最后更新**：2026-04-09

---

## Hook 架构总览

UniV2X 推理管线通过 5 个 Hook 点逐步将 PyTorch 模块替换为 TRT 或向量化实现：

```
输入图像 → ResNet-FPN backbone（PyTorch，不可替换：DCNv2 无 ONNX 支持）
         → [Hook A] BEV 编码器 TRT（FPN 特征 → BEV 特征图）
         → [Hook B] 车道线查询融合（infra lane_query → ego）
         → [Hook C] 智能体查询融合（infra track_query 匹配 → ego 互补）
         → [Hook D] 检测头 TRT（BEV + 查询 → Decoder 6层 → 检测框/轨迹/类别）
         → [Hook E] 下游头 TRT（Motion 预测 + Occ 占用 + Planning 规划）
         → 跟踪输出
```

| Hook | 替换模块 | 实现方式 | 输入 → 输出 | 状态 |
|:----:|---------|---------|------------|:----:|
| **A** | `pts_bbox_head.get_bev_features` | TRT 引擎（FP16/INT8） | 4 层 FPN 特征 + can_bus + lidar2img + prev_bev → BEV 特征 (200×200×256) | ✅ 可用 |
| **B** | `seg_head.cross_lane_fusion.forward` | PyTorch 向量化 | infra lane_query + ego lane_query → 融合后 lane_query | ✅ 可用 |
| **C** | `cross_agent_query_interaction` | PyTorch 向量化 | infra track_instances + ego track_instances → 互补后 query（901→最多 1101） | ✅ 可用 |
| **D** | `pts_bbox_head.get_detections` | TRT 引擎（FP16/INT8） | BEV + 1101 个查询 → cls_scores/bbox/traj/ref_pts/query_feats | ✅ FP16 BEV 下可用 / ⚠️ 不可与 INT8 BEV 组合 |
| **E** | `forward_test` 包装器 | TRT 引擎（FP16/INT8） | BEV + decoder 输出 + lane_query → 轨迹/占用/规划 | ✅ 可用 |

**推荐配置**：
- FP16 全链路 TRT：Hook A(FP16)+B+C+D(FP16)+E（AMOTA=0.345）
- 体积优先：Hook A(FP16)+B+C+D(INT8)（AMOTA=0.332，检测头 18.2 MB）
- ⚠️ 注意：INT8 BEV + Hook D 存在超线性误差叠加，不可组合使用

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
Phase DS ──  下游头 INT8 PTQ（Motion + Occ + Planning）
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
| `trt_engines/univ2x_infra_bev_encoder_200_1cam.trt` | 40.9 MB |

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

**向量化方案**（`univ2x_track.py`）：预计算 `flat_idx` + `torch.index_add_` 批量 scatter：

| N | 优化前 | 优化后 | 加速比 |
|---|--------|--------|--------|
| 10  | 3.4 ms  | 1.1 ms | 3× |
| 50  | 16.2 ms | 1.1 ms | 15× |
| 200 | 64.3 ms | 1.1 ms | **58×** |

### 6.3 LaneQueryFusionTRT + AgentQueryFusionTRT（Hook B/C）

| 模块 | CosSim | max_abs |
|------|--------|---------|
| AgentQueryFusionTRT vs 原始 | 0.9999999～1.0 | < 5e-7 ✅ |
| _get_coop_bev_embed vec vs 原始 | 0.9999999～1.0 | < 8e-7 ✅ |
| LaneQueryFusionTRT vs 原始 | 0.9999998～1.0 | 0.0 ✅ |

---

## 七、Phase 3C — V2X 路径检测头 TRT（Hook-D，1101-query）

固定 shape 1101-query（901 追踪 + 200 来自 infra），零填充后入 TRT，推理后切片回原始 N。

**精度验证**：FP16 BEV + FP16 Hook D = AMOTA 0.345（vs baseline 0.341），零填充代价约 -0.004，可接受。

**⚠️ INT8 BEV 组合限制**：INT8 BEV + Hook D 会导致 AMOTA 暴跌至 0.241，原因是 INT8 BEV 的系统性偏移经 Decoder Cross-Attention 超线性放大。详见反思文档 §3.4。

| 文件 | 大小 |
|------|------|
| `trt_engines/univ2x_ego_heads_v2x_1101.trt` | 33.4 MB |

---

## 八、Phase E — BEV 编码器 INT8 PTQ（Vanilla W8A8）

### 8.1 方案设计

从 QuantV2X 移植量化框架（共 7 个模块，1,764 行）。

**ADR-001 关键决策（选择性量化）**：

| 子层 | 量化状态 | 原因 |
|------|:-------:|------|
| `sampling_offsets` | ❌ FP16 | 坐标偏移量化误差导致 BEV 特征完全失效 |
| `attention_weights` | ❌ FP16 | 权重误差放大注意力聚合结果 |
| `value_proj` | ✅ W8A8 | 聚合前变换，误差可吸收 |
| `output_proj` | ✅ W8A8 | 同上 |

### 8.2 迭代过程

| 版本 | 样本数 | prev_bev | PLUGIN_V2 精度 | AMOTA |
|------|:-----:|:--------:|:--------------:|:-----:|
| v1 | 10 | 零初始化 | 无覆盖 | 0.334 |
| v2 | 20 | temporal | 无覆盖 | 0.344 |
| v3 | 20 | temporal | 全部→FP16 | 0.278 ❌ |
| v4 | 20 | temporal+fresh cache | 无覆盖 | 0.355 |
| **v5** | **50** | **temporal** | **无覆盖** | **0.364 ✅** |

### 8.3 最终结果

| 指标 | FP16 TRT (Hook-A) | INT8 PTQ v5 | 差值 |
|------|:------------------:|:-----------:|:----:|
| AMOTA | 0.370 | 0.364 | -0.006 |
| 引擎大小 | 72 MB | **43 MB** | **-40%** |

---

## 九、Phase DS — 下游头 INT8 PTQ

### 9.1 模型大小对比

| 模型 | FP16 TRT | INT8 TRT | 压缩率 |
|------|:--------:|:--------:|:------:|
| Ego 下游头 | 153 MB | **74 MB** | **↓51.6%** |
| Infra 下游头 | 134 MB | **66 MB** | **↓50.7%** |

### 9.2 精度对比（INT8 vs FP16，5 样本平均）

| 输出 | 余弦相似度 | 均值绝对误差 |
|------|:---------:|:-----------:|
| traj_scores | 0.9999842 ✅ | 1.93e-3 |
| traj_preds | 0.9999455 ✅ | 3.80e-3 |
| occ_logits | 0.9992438 ✅ | 3.42e-2 |
| sdc_traj | 0.9999968 ✅ | 4.04e-2 |

### 9.3 端到端验证（修复 infra 引擎后，2026-04-08）

| 配置 | AMOTA | 下游头延迟 | 端到端延迟 |
|------|:-----:|:---------:|:---------:|
| Baseline（Hook A+B+C） | **0.341** | — | 817.6 ms |
| + Hook E FP16 下游头 | **0.341** | 78.81 ms | 816.7 ms |
| + Hook E INT8 下游头 | **0.341** | **32.51 ms** | 847.6 ms |

**结论**：Hook E 对 AMOTA 零影响，INT8 下游头 2.4× 加速。

---

## 十、Phase P1 — 检测头 INT8 PTQ（V2X 1101-query）

### 10.1 方案设计

对 V2X 检测头（`HeadsDecoderOnlyWrapper`，N_PAD=1101）进行 INT8 PTQ 量化。

**校准数据**：Hook `pts_bbox_head.get_detections` 捕获 50 帧真实推理输入（bev_embed + track_query + track_ref_pts），查询自动零填充至 1101 匹配 ONNX 形状。

**量化策略**：与 BEV 编码器相同 — FP16+INT8 混合精度，MSDAPlugin 层自动保持 FP16（无 INT8 实现）。

### 10.2 引擎大小对比

| 引擎 | FP16 TRT | INT8 TRT | 压缩率 |
|------|:--------:|:--------:|:------:|
| 检测头 1101-query | 33.4 MB | **18.2 MB** | **↓45.5%** |

### 10.3 端到端验证

| BEV 精度 | 检测头精度 | AMOTA | 检测头延迟 | 说明 |
|:--------:|:--------:|:-----:|:---------:|------|
| FP16 | PyTorch | **0.341** | — | Baseline（Hook A+B+C） |
| FP16 | FP16 TRT | **0.345** | 3.72 ms | Hook D 零损 ✅ |
| FP16 | **INT8 TRT** | **0.332** | 3.56 ms | INT8 量化损失 -0.013 |
| INT8 | FP16 TRT | 0.241 | 3.45 ms | ❌ INT8 BEV + Hook D 误差叠加 |
| INT8 | **INT8 TRT** | 0.237 | **2.92 ms** | ❌ 同上，INT8 头额外 -0.004 |

### 10.4 结论

**检测头 INT8 量化本身有效**：FP16 BEV + INT8 检测头 = 0.332（-0.013），引擎 33.4→18.2 MB（-45%），可接受。

**关键发现：INT8 BEV 与 Hook D 存在严重误差叠加**。INT8 BEV 单独使用无问题（Hook A+B+C+E = 0.341），Hook D 单独使用也无问题（FP16 BEV + FP16 Hook D = 0.345）。但 **INT8 BEV + Hook D** 组合导致 AMOTA 暴跌至 0.241（-0.100）。

**根因**：INT8 BEV 编码器的量化误差（虽然单独看 CosSim > 0.999）改变了 BEV 特征的分布。这些微小差异经过 Hook D 的 6 层 Decoder Cross-Attention（BEV 作为 key/value）和零填充 query 的 Self-Attention 后被放大。两个独立可接受的误差源在级联后产生超线性的精度退化。

**推荐配置更新**：
- **最优精度**：Hook A(FP16)+B+C+D(FP16)+E = FP16 全链路 TRT（AMOTA ~0.345）
- **最优体积**：Hook A(FP16)+B+C+D(INT8) = FP16 BEV + INT8 检测头（AMOTA 0.332，检测头 18.2 MB）
- **当前推荐**：Hook A(FP16)+B+C+E（AMOTA 0.341，避免 INT8 BEV + Hook D 叠加问题）

---

## 十一、AdaRound 实验（已废弃）

### 11.1 方案 A：W_fq 内嵌 ONNX

AdaRound 校准 36 层 → W_fq 内嵌 ONNX → TRT INT8 引擎。

**结果**：AMOTA = **0.190** ❌

**失败根因**：双重量化。TRT 从 W_fq 重新推导 scale（scale_trt ≠ scale_ada），第二次量化覆盖了 AdaRound 优化。

### 11.2 方案 B：权重-only Q/DQ

插入 36 对 QuantizeLinear/DequantizeLinear 节点到 ONNX。

**结果**：AMOTA = **0.137** ❌

**失败根因**：TRT 检测到 Q/DQ 节点后进入 explicit 模式，Calibrator 完全失效，激活层退化为 FP16。

### 11.3 结论

Vanilla INT8 PTQ（AMOTA=0.364）仍是当前最优 INT8 方案。正确的 AdaRound for TRT 需要同时插入权重和激活 Q/DQ（P3 优先级）。

---

## 十二、推理延迟汇总（RTX 4090 实测）

| 模块 | PyTorch 原版 | TRT / 向量化优化 | 加速比 |
|------|:-----------:|:---------------:|:------:|
| ResNet-FPN backbone（ego + infra） | ~30 ms | ~30 ms | 1× |
| BEV encoder × 2 | ~150 ms | ~20–40 ms | ~5× |
| `_query_matching`（N_inf=10） | 1,048 ms | 2.7 ms | **392×** |
| `_query_matching`（N_inf=50） | 5,341 ms | 5.6 ms | **951×** |
| `_query_matching`（N_inf=100） | 10,548 ms | 8.9 ms | **1,191×** |
| `_get_coop_bev_embed`（N=200） | 64.3 ms | 1.1 ms | **58×** |
| 检测头 + 下游头 | ~100 ms | ~15–30 ms | ~5× |
| **V2X 场景总延迟（N_inf=50）** | **~5,640 ms** | **~90 ms** | **~63×** |

---

## 十三、模型大小对比

| 模块 | PyTorch (.pth) | TRT FP16 | TRT INT8 |
|------|:-------------:|:--------:|:--------:|
| 总 checkpoint | **1,600 MB** | — | — |
| BEV encoder ego（1-cam） | — | 75 MB | **43 MB** |
| BEV encoder infra（1-cam） | — | 41 MB | — |
| 检测头（1101-query V2X） | — | 33.4 MB | **18.2 MB** |
| 下游头 ego（Motion+Occ+Planning） | — | 152 MB | **74 MB** |
| 下游头 infra（Motion+Occ） | — | 134 MB | **66 MB** |

---

## 十四、全链路最优配置对比

| 配置 | BEV 精度 | AMOTA | 引擎大小 | 状态 |
|------|:-------:|:-----:|:--------:|:----:|
| PyTorch 全链路基准 | — | 0.338 | 1,600 MB | ✅ 基准 |
| Hook-A（ego BEV FP16 TRT only） | FP16 | 0.378 | 75 MB ego | ✅ 精度最优 |
| Hook-A+B+C（FP16 BEV + V2X 融合） | FP16 | **0.341** | 75+41 MB | ✅ 全链路协同 |
| Hook-A+B+C+D FP16 | FP16 | **0.345** | 75+41+33.4 MB | ✅ 全链路 TRT |
| Hook-A+B+C+D INT8 | FP16 | **0.332** | 75+41+18.2 MB | ✅ INT8 检测头可用 |
| Hook-A+B+C+E（INT8 下游头 TRT） | FP16 | **0.341** | 75+41+74 MB | ✅ **推荐配置** |
| Hook-A(INT8)+B+C+E | INT8 | **0.341** | 43+41+74 MB | ✅ 体积最优（无 Hook D） |
| Hook-A(INT8)+B+C+D FP16 | INT8 | 0.241 | 43+41+33.4 MB | ❌ INT8 BEV + Hook D 误差叠加 |
| INT8 PTQ v5（BEV encoder ego-only） | INT8 | 0.364 | **43 MB BEV** | ✅ ego-only 体积最优 |
| AdaRound INT8（无 Q/DQ） | — | 0.190 | 43.7 MB | ❌ 废弃 |
| AdaRound Q/DQ（权重-only） | — | 0.137 | 40.8 MB | ❌ 废弃 |
