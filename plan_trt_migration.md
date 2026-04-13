# UniV2X-TRT 迁移计划

本文档分析 [uniad-trt](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt) 针对 UniAD 实施的 TensorRT 加速方案，梳理 UniAD 与 UniV2X 的同构关系，并给出详细的迁移计划。

---

## 一、UniAD-TRT 方案核心组成

UniAD-TRT 的 TensorRT 部署方案包含以下五个层次：

```
① 自定义 TRT 插件（C++ 算子）
    MultiScaleDeformableAttnTRT  —— 可变形注意力
    InverseTRT                   —— 矩阵求逆
    RotateTRT                    —— 空间旋转

② Python 侧 ONNX 导出适配（模块 *TRT 变体）
    BEVFormerEncoderTRT / BEVFormerLayerTRT
    SpatialCrossAttentionTRT / TemporalSelfAttentionTRT
    PerceptionTransformerUniADTRT
    BEVFormerTrackHeadTRT
    DetectionTransformerDecoderTRTP
    OccHeadTRT
    MotionHeadTRT / MotionTransformerDecoderTRT
    UniADTrackTRT / UniADTRT（顶层）

③ 关键算子替换（Python 工具函数）
    custom_torch_atan2_trt       —— 替换 torch.atan2（不可导出 ONNX）
    denormalize_bbox_trt         —— 替换含 if 分支的 bbox 反归一化
    线性化控制流 / 消除 Python 动态分支

④ C++ 推理应用
    推理引擎封装（enqueueV2 / enqueueV3）
    CUDA 预/后处理 kernel
    帧间状态管理（prev_track_instances 0~13）

⑤ 显式量化
    ModelOpt INT8/FP16 校准流程（TensorRT Model Optimizer）
```

---

## 二、UniAD 与 UniV2X 的同构模块对比

| UniAD 模块 | UniV2X 对应模块 | 代码路径（UniV2X） | 同构程度 |
|---|---|---|---|
| `BEVFormerEncoder` | `BEVFormerEncoder` | `modules/encoder.py:27` | **完全相同** |
| `BEVFormerLayer` | `BEVFormerLayer` | `modules/encoder.py:239` | **完全相同** |
| `SpatialCrossAttention` | `SpatialCrossAttention` | `modules/spatial_cross_attention.py:31` | **完全相同** |
| `MSDeformableAttention3D` | `MSDeformableAttention3D` | `modules/spatial_cross_attention.py:178` | **完全相同** |
| `TemporalSelfAttention` | `TemporalSelfAttention` | `modules/temporal_self_attention.py:25` | **完全相同** |
| `DetectionTransformerDecoder` | `DetectionTransformerDecoder` | `modules/decoder.py:53` | **完全相同** |
| `CustomMSDeformableAttention` | `CustomMSDeformableAttention` | `modules/decoder.py:133` | **完全相同** |
| `BEVFormerTrackHead` | `BEVFormerTrackHead` | `dense_heads/track_head.py:21` | **完全相同** |
| `MotionHead` | `MotionHead` | `dense_heads/motion_head.py:16` | **完全相同** |
| `OccHead` | `OccHead` | `dense_heads/occ_head.py:17` | **完全相同** |
| `PlanningHead` | `PlanningHead` | `dense_heads/planning_head.py` | **完全相同** |
| `PanSegHead` | `PanSegHead` | `dense_heads/panseg_head.py` | **完全相同** |
| `UniADTrack` | `UniV2XTrack` | `detectors/univ2x_track.py:33` | 高度相似，入口/状态管理有差异 |
| `UniAD` | `UniV2X` | `detectors/univ2x_e2e.py:19` | 高度相似，多了 V2X 分支 |
| — | `MultiAgent` | `detectors/multi_agent.py:11` | **UniV2X 独有** |
| — | `AgentQueryFusion` | `fusion_modules/agent_fusion.py:15` | **UniV2X 独有** |
| — | `LaneQueryFusion` | `fusion_modules/lane_fusion.py:12` | **UniV2X 独有** |

UniV2X 的感知骨干（BEVFormer 编码器 + 解码器 + 所有 dense heads）与 UniAD **代码完全一致**，这意味着 UniAD-TRT 超过 80% 的 ONNX 适配工作可直接搬运，无需重新开发。

---

## 三、可直接迁移的优化方案（无需改动）

### 3.1 三个 TRT 插件（直接复用）

这三个插件的 C++ 代码、CMakeLists、`symbolic()` 注册接口均可从 `uniad-trt/inference_app/` 直接迁移，无需任何修改：

| 插件 | 用途 | 涉及的 UniV2X 模块 |
|---|---|---|
| `MultiScaleDeformableAttnTRT` | BEV 编码器中的可变形注意力（SCA + TSA 内核） | `SpatialCrossAttention`, `TemporalSelfAttention`, `MotionTransformerDecoder` |
| `InverseTRT` | 坐标变换中的矩阵求逆（TSA 时序对齐） | `TemporalSelfAttention` |
| `RotateTRT` | BEV 特征的空间旋转（BEV warping） | `TemporalSelfAttention` |

### 3.2 ONNX 导出适配模块（直接迁移，约覆盖 80% 的工作量）

来源：`uniad-trt/patch/uniad-onnx-export.patch`

| 迁移内容 | 目标文件（UniV2X） |
|---|---|
| `BEVFormerEncoderTRT` / `BEVFormerLayerTRT` | `projects/mmdet3d_plugin/univ2x/modules/encoder.py` |
| `SpatialCrossAttentionTRT`（含 `MSDAPlugin`） | `projects/mmdet3d_plugin/univ2x/modules/spatial_cross_attention.py` |
| `TemporalSelfAttentionTRT` | `projects/mmdet3d_plugin/univ2x/modules/temporal_self_attention.py` |
| `DetectionTransformerDecoderTRTP` | `projects/mmdet3d_plugin/univ2x/modules/decoder.py` |
| `BEVFormerTrackHeadTRT` | `projects/mmdet3d_plugin/univ2x/dense_heads/track_head.py` |
| `MotionHeadTRT` + `MotionTransformerDecoderTRT` | `projects/mmdet3d_plugin/univ2x/dense_heads/motion_head.py` |
| `OccHeadTRT` | `projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py` |
| `custom_torch_atan2_trt` / `denormalize_bbox_trt` | `projects/mmdet3d_plugin/core/bbox/util.py` |
| ONNX export 脚本框架 | `tools/export_onnx_univ2x.py`（基于 `uniad-trt/tools/export_onnx.py` 适配） |

### 3.3 C++ 推理应用框架（直接复用）

`uniad-trt/inference_app/enqueueV3/` 下的以下文件可直接复用，仅需修改 `uniad.hpp` 中的 `KernelInput`/`KernelOutput` 结构体以增加 V2X 独有状态字段：

- `tensor.cu` / `tensorrt.cpp` — 引擎封装
- `pre_process.cu` — 图像预处理 CUDA kernel
- `visualize.cu` — 可视化输出

### 3.4 显式量化流程（直接复用）

ModelOpt INT8/EQ 量化流程无模型结构依赖，`documents/explicit_quantization.md` 中的脚本可直接复用，仅需替换输入形状配置（BEV size、img size、track_instances 数量）。

---

## 四、需要适配修改的内容

### 4.1 顶层探测器 TRT 变体（中等工作量）

**`UniADTrackTRT` → `UniV2XTrackTRT`**：
- UniAD 使用 `cfg.model` 单模型；UniV2X 使用 `cfg.model_ego_agent` + `cfg.model_other_agent_inf` 双模型
- 需拆分推理流：① 先运行基础设施模型得到 inf queries；② 再运行自车模型加入 V2X 融合

**`UniADTRT` → `UniV2XTRT`**：
- 在 Stage 2 的 forward 中插入 `LaneQueryFusion` 和 `AgentQueryFusion` 的 TRT 兼容分支
- 输入/输出字典需扩充 V2X 相关字段

### 4.2 帧间状态扩充

UniAD ONNX 管理 14 个 track 状态张量（`prev_track_instances0~13`）。UniV2X 需额外增加：

| 新增状态字段 | 用途 |
|---|---|
| `prev_inf_bev` | 基础设施侧上一帧 BEV 特征 `[H*W, 1, C]` |
| `prev_ego2other_rt` | 上一帧 V2X 标定矩阵 `[4, 4]` |
| `inf_track_instances0~N` | 基础设施侧 track 状态（结构同自车侧） |

---

## 五、V2X 独有的新增工作（需从头实现）

### 5.1 `AgentQueryFusion` TRT 适配（核心难点）

**问题**：`_query_matching()` 调用 `scipy.optimize.linear_sum_assignment`（匈牙利算法），完全不可导出为 ONNX 静态图。

**推荐方案：CPU-GPU 分离（方案 A）**

- 将匈牙利匹配作为 ONNX 图外的 CPU 预处理步骤，输出 `matched_veh_idx` / `matched_inf_idx` 作为 ONNX 图的额外输入张量
- 只将匹配后的特征融合（MLP 部分）导出为 ONNX 子图
- C++ 推理侧实现轻量匈牙利算法（可用 `lapjv` 库，N ≤ 200，单次 < 1 ms）

**备选方案：Sinkhorn 可微匹配（方案 B）**

- 将 `linear_sum_assignment` 替换为可微的 Sinkhorn-Knopp 迭代（约 20 轮），整体可 ONNX 导出
- 需对融合模块重新微调，工程量较大，适合后续精度优化阶段

### 5.2 `LaneQueryFusion` TRT 适配（较易）

`LaneQueryFusion.forward()` 中无 `scipy` 调用，但存在以下 ONNX 不友好操作：

| 原始操作 | TRT 替换方案 |
|---|---|
| Python `for` 循环坐标变换 | batch 矩阵乘法向量化（`torch.bmm`） |
| `np.linalg.inv` | 复用现有 `InverseTRT` 插件 |
| `torch.where` 动态过滤 | 改为 mask 乘法（静态图兼容） |

### 5.3 双模型推理管线（C++ 侧新增）

UniAD 为单引擎单帧推理，UniV2X 需要双引擎有序调度：

```
Frame t:
  ┌─────────────────────────────────────────────────────┐
  │  [基础设施侧 TRT 引擎]                               │
  │  输入: img_inf, prev_inf_bev, prev_inf_track_state  │
  │  输出: inf_track_queries, inf_lane_queries           │
  └────────────────────┬────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────┐
  │  [CPU 预处理]                                        │
  │  · 匈牙利匹配 (lapjv)                               │
  │  · ego2other 坐标变换                               │
  │  输出: matched_indices, transformed_queries          │
  └────────────────────┬────────────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────────────┐
  │  [自车侧 TRT 引擎]                                   │
  │  输入: img_ego, prev_ego_bev, prev_ego_track_state,  │
  │        fused_queries, matched_indices                │
  │  输出: planning, tracking, occ, seg                  │
  └─────────────────────────────────────────────────────┘
```

基础设施引擎推理可与自车侧数据预处理在不同 CUDA stream 上并行，减少端到端延迟。

---

## 六、迁移计划

### Phase 1：基础设施复用（约 1 周）

- [ ] 将 3 个 TRT 插件（MSDA、Inverse、Rotate）整合进 UniV2X 构建系统（CMakeLists 适配）
- [ ] 将 `BEVFormerEncoderTRT`、`BEVFormerLayerTRT`、`SpatialCrossAttentionTRT`、`TemporalSelfAttentionTRT` 复制到 UniV2X 对应文件
- [ ] 将 `custom_torch_atan2_trt`、`denormalize_bbox_trt` 工具函数复制到 `core/bbox/util.py`
- [ ] 验证：单帧 ego 模型 forward 能否 export 到 ONNX（不含 V2X 融合，仅 BEVFormer 骨干部分）

**交付物**：BEVFormer 骨干部分可成功导出为有效 ONNX，3 个插件编译通过

### Phase 2：单智能体端到端 ONNX 导出（约 2 周）

- [ ] 迁移 `BEVFormerTrackHeadTRT`、`DetectionTransformerDecoderTRTP`
- [ ] 迁移 `OccHeadTRT`、`MotionHeadTRT`（含 `MotionTransformerDecoderTRT`）
- [ ] 对 `PlanningHead` 做 TRT 兼容适配（消除 `casadi` 依赖的后处理，移至 ONNX 图外）
- [ ] 实现 `UniV2XTrackTRT`（基于 `UniADTrackTRT`，适配 `model_ego_agent` 路径）
- [ ] 完成 Stage 1（Perception Only）ego 单智能体 ONNX 导出 + TRT engine 构建
- [ ] 精度验证：TRT FP32 输出 vs PyTorch 输出，tracking AMOTA 差异 < 0.005

**交付物**：可运行的 ego 单智能体 TRT engine（Stage 1），精度达标

### Phase 3：V2X 融合模块适配（约 2 周）

- [ ] 实现 `LaneQueryFusionTRT`（向量化坐标变换，消除 Python for 循环，复用 `InverseTRT`）
- [ ] 实现 `AgentQueryFusionTRT`（采用方案 A：ONNX 图仅含 MLP 融合部分）
- [ ] 在 C++ 推理侧集成 `lapjv` 库，实现 CPU 侧匈牙利匹配
- [ ] 导出完整 Stage 2 ONNX（含 V2X 融合分支）
- [ ] 精度验证：V2X 协同 Planning Col. 与 PyTorch 基准差异 < 0.005

**交付物**：含 V2X 融合的完整 Stage 2 ONNX + TRT engine

### Phase 4：双引擎部署与 C++ 推理应用（约 1~2 周）

- [ ] 构建基础设施侧 TRT engine（inf model → ONNX → engine）
- [ ] 改造 C++ 推理应用为双引擎调度器（基于 `enqueueV3` 框架扩展）
- [ ] 实现 V2X 标定坐标变换的 CUDA kernel（在 `pre_process.cu` 中扩充）
- [ ] 实现双 CUDA stream 并行调度（基础设施引擎 + 自车预处理并行）
- [ ] 端到端系统验证：TRT 输出 vs PyTorch 输出，误差满足精度要求

**交付物**：完整 C++ 双引擎推理应用，支持 V2X 协同感知规划

### Phase 5：显式量化（可选，约 1 周）

- [ ] 使用 ModelOpt 对 ego 侧引擎做 INT8 EQ 校准（参照 `documents/explicit_quantization.md`）
- [ ] 验证 INT8 精度降落（参考基准：planning L2 增加 < 0.01 m，Col 变化 < 0.01）
- [ ] 评估基础设施侧引擎是否也适合 INT8（精度要求相对宽松）

**交付物**：INT8 量化 TRT engine，性能/精度报告

---

## 七、预期收益参考

以 UniAD-tiny 在 NVIDIA Drive Orin 上的实测数据为参照，UniV2X 规模类似（BEV 50×50，图像 ~400×256），预估收益如下：

| 部署方式 | 精度损失 | 推理延迟估算（单帧，ego 侧） |
|---|---|---|
| PyTorch FP32 | 基准 | ~600~900 ms |
| TRT FP32 | 可忽略（planning MSE < 1e-6） | ~50~80 ms |
| TRT FP16 | 极小（Col 变化 < 0.01） | ~35~55 ms |
| TRT INT8(EQ)+FP16 | 小（L2 增加 < 0.01 m） | ~25~40 ms |

V2X 管线的额外开销：
- CPU 侧匈牙利匹配：N ≤ 200，< 1 ms，可忽略
- 基础设施侧引擎推理：与自车侧可通过双 CUDA stream 部分并行，额外延迟 < 20 ms

---

## 八、关键风险与对策

| 风险 | 严重程度 | 对策 |
|---|---|---|
| `AgentQueryFusion` 中动态目标数量导致 ONNX 形状不确定 | 高 | 对 track 实例数设置固定上限（如 200），使用 mask 填充 |
| `casadi` 规划后处理无法 ONNX 导出 | 中 | 将碰撞优化后处理移至 ONNX 图外，作为 C++ 后处理步骤 |
| 基础设施侧与自车侧 BEV 坐标系对齐误差 | 中 | 在 Phase 3 验证中增加坐标变换单元测试 |
| INT8 量化对 V2X 融合 MLP 精度影响 | 低 | 对融合层保持 FP16，其余层 INT8 |
