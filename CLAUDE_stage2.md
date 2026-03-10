# Phase 2：单智能体端到端 ONNX 导出实现计划

> 对应 `plan.md` 第六章 Phase 2，目标：在 Phase 1 BEV encoder TRT 的基础上，将完整的 ego 单智能体感知链路（TrackHead + OccHead + MotionHead）导出为 TRT 可编译的 ONNX，精度验收：TRT FP32 vs PyTorch tracking AMOTA 差异 < 0.005。
>
> **Phase 1 已交付物**（均已验证）：
> - `onnx/univ2x_ego_bev_encoder_200.onnx` / `trt_engines/univ2x_ego_bev_encoder_200.trt`
> - `onnx/univ2x_infra_bev_encoder_200.onnx` / `trt_engines/univ2x_infra_bev_encoder_200.trt`
> - `plugins/build/libuniv2x_plugins.so`（MSDAPlugin / InversePlugin / RotatePlugin 均验证通过）
>
> **Phase 2 推理流水线（三段式）**：
> ```
> [Stage A] 已完成
>   img (bs, num_cam, C, H, W)
>     → Backbone + FPN (PyTorch, DCNv2 不导出)
>     → img_feats (4 FPN levels)
>
> [Stage B] Phase 1 交付
>   img_feats + can_bus + lidar2img + prev_bev
>     → BEV Encoder TRT engine
>     → bev_embed (bev_h*bev_w, bs, C)
>
> [Stage C] Phase 2 目标
>   bev_embed + 固定大小 track 状态张量
>     → Heads TRT engine
>     → 新 track 状态 + cls/box/occ/motion 输出
> ```
>
> **核心约束**：ego 侧和基础设施侧各一套独立 engine；is_cooperation=False（单智能体）；Stage 1 不含 PlanningHead（casadi 依赖移至图外）。

---

## 一、Phase 2 的关键技术挑战

Phase 1 的模块是"静态计算图"（固定形状的张量变换），Phase 2 的挑战集中在三类**动态性**：

### 1.1 Track 实例的动态数量

原始推理代码用 `Instances` 对象（类似 detectron2）管理每帧激活的 track 数量，数量随帧变化：

```python
# univ2x_track.py — 不可 ONNX 导出
active_mask = track_instances.scores > self.score_thresh
active_instances = track_instances[active_mask]      # 动态 shape
```

**解决方案**：固定上限 N=num_query（默认 901，含 ego query），所有帧均处理全部 N 个 queries，
在 ONNX 图**外部**（C++ 或 Python 验证脚本）用 `score_thresh` 做过滤。`Instances` 类完全移出 ONNX 图。

### 1.2 `CustomMSDeformableAttention` 缺少 MSDAPlugin 符号路径

`DetectionTransformerDecoder` 的每一层使用 `CustomMSDeformableAttention`，其 `forward()` 调用
`MultiScaleDeformableAttnFunction_fp32`（原生 CUDA，无 ONNX symbolic），会导致导出失败。

**解决方案**：新增 `CustomMSDeformableAttentionTRT` 变体，将 CUDA 调用替换为 `MSDAPlugin()`，
新增 `DetectionTransformerDecoderTRTP` 使用该变体。

### 1.3 `_get_coop_bev_embed` 含 Python for 循环

`UniV2XTrack._get_coop_bev_embed()` 中用 Python `for i in range(N)` 按 track 位置更新 BEV：

```python
for idx in range(act_track_instances.ref_pts.shape[0]):
    w = int(locs[idx, 0])   # Python int 转换 → 不可追踪
    h = int(locs[idx, 1])
    bev_embed[hh * self.bev_w + ww, ...] += ...  # 动态索引
```

**解决方案**：Phase 2 导出 `is_cooperation=False`（单智能体）路径，该函数**完全不进入计算图**。
Phase 3 再处理 V2X 融合部分。

### 1.4 `QueryInteractionModule` 含复杂 Python 控制流

`QueryInteractionModule`（track 的激活/冻结/删除逻辑）依赖 Python 条件判断和 `Instances` 索引，
无法导出 ONNX。

**解决方案**：将 `QueryInteractionModule` 整体移至 ONNX 图**外**（C++ 或 Python 状态管理层），
ONNX 图只导出"给定固定 N 个 queries → 输出 N 个 queries 的更新特征和预测结果"的纯张量计算。

---

## 二、Phase 2 任务分解

```
T1: CustomMSDeformableAttentionTRT + DetectionTransformerDecoderTRTP
    → decoder.py 末尾追加

T2: get_detections_trt() 方法 + BEVFormerTrackHeadTRT 类
    → track_head.py 末尾追加

T3: MotionTransformerDecoderTRT + MotionHeadTRT
    → motion_head.py 及其 plugin 目录

T4: OccHeadTRT
    → occ_head.py 末尾追加

T5: UniV2XTrackTRT（顶层推理包装器，仅含纯张量前向）
    → univ2x_track.py 末尾追加

T6: 扩展 ONNX 导出脚本，实现 Stage C（Heads）导出
    → tools/export_onnx_univ2x.py

T7: 新增 TRT 推理配置
    → projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py

T8: 精度验证脚本
    → tools/validate_trt_heads.py
```

---

## 三、详细实现步骤

### T1：CustomMSDeformableAttentionTRT + DetectionTransformerDecoderTRTP

**文件**：`projects/mmdet3d_plugin/univ2x/modules/decoder.py`

#### T1.1 新增 `CustomMSDeformableAttentionTRT`

在 `CustomMSDeformableAttention` 类末尾之后追加，继承原类，仅重写 `forward()` 中的 CUDA 调用部分。
核心改动：将 `MultiScaleDeformableAttnFunction_fp32(...) / _fp16(...)` 替换为
`MSDAPlugin(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)`。

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `CustomMSDeformableAttentionTRTP` 类（约第 2800 行起），
直接复制后将导入路径从 `uniad.functions` 改为 `univ2x.functions`，类名去掉末尾的 `P`。

注意事项：
- `spatial_shapes` 和 `level_start_index` 仍为 int64 在 Python 侧，ONNX post-processing（已有的 INT64→INT32 补丁）会自动处理
- decoder 中只有 1 个 feature level（BEV 自注意力），`spatial_shapes = [[bev_h, bev_w]]`，
  `level_start_index = [0]`，均为常量，会被 ONNX constant folding 捕获

mmcv Registry 注册：
```python
@ATTENTION.register_module()
class CustomMSDeformableAttentionTRT(CustomMSDeformableAttention):
    ...
```

#### T1.2 新增 `DetectionTransformerDecoderTRTP`

继承 `DetectionTransformerDecoder`，重写 `forward()` 使其调用每层的
`CustomMSDeformableAttentionTRT` 而非原版。

实现策略：在 `forward()` 中，将 `self.layers` 里每个 `CustomMSDeformableAttention`
动态替换为对应的 `CustomMSDeformableAttentionTRT`（通过 `type()` 检查 + in-place 替换），
或在构建时通过 config 指定 `type="CustomMSDeformableAttentionTRT"`。

建议用 config 方式（更干净）：`DetectionTransformerDecoderTRTP` 不改变 `forward()` 逻辑，
仅作为 registry 标识，实际通过 `univ2x_coop_e2e_track_trt_p2.py` 中的 config 将
子层 `type` 改为 `"CustomMSDeformableAttentionTRT"`。

```python
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class DetectionTransformerDecoderTRTP(DetectionTransformerDecoder):
    """Registry alias; actual TRT variant achieved via config (sub-layer type change)."""
    pass
```

#### T1.3 更新 `modules/__init__.py`

```python
from .decoder import (DetectionTransformerDecoder, CustomMSDeformableAttention,
                      CustomMSDeformableAttentionTRT, DetectionTransformerDecoderTRTP)
```

---

### T2：BEVFormerTrackHeadTRT（含 `get_detections_trt`）

**文件**：`projects/mmdet3d_plugin/univ2x/dense_heads/track_head.py`

#### T2.1 问题分析

`BEVFormerTrackHead.get_detections()` 当前签名：

```python
def get_detections(self, bev_embed, object_query_embeds=None,
                   ref_points=None, img_metas=None):
```

其中 `img_metas` 是 Python dict，不可 ONNX 导出。返回值是 Python dict，也不可直接输出。
`object_query_embeds` 是 `nn.Embedding` 权重（常量）。

输出中 `outs['last_ref_points']` 需要传回下一帧，是帧间状态之一。

#### T2.2 新增 `get_detections_trt` 方法

```python
def get_detections_trt(self, bev_embed, query, ref_pts):
    """TRT-compatible detection forward.

    Args:
        bev_embed (Tensor): (bev_h*bev_w, bs, embed_dims) — 来自 BEV encoder
        query     (Tensor): (num_query, bs, embed_dims*2)  — track query (pos + feat 拼接)
        ref_pts   (Tensor): (bs, num_query, 3)             — 上一帧参考点（raw，非 sigmoid）

    Returns:
        all_cls_scores    (Tensor): (num_dec_layers, bs, num_query, num_classes)
        all_bbox_preds    (Tensor): (num_dec_layers, bs, num_query, code_size)
        all_past_traj     (Tensor): (num_dec_layers, bs, num_query, past+fut, 2)
        last_ref_points   (Tensor): (bs, num_query, 3)  — 传下一帧
        query_feats       (Tensor): (num_dec_layers, bs, num_query, embed_dims)
    """
```

- 将 `img_metas` 相关逻辑（仅用于 `bev_h`/`bev_w` 这类常量）全部移除或改为 `self.bev_h` / `self.bev_w`
- `object_query_embeds` 从 `self.query_embedding.weight` 直接取，不作为外部输入
- 将 `transformer.get_states_and_refs()` 改为调用 `transformer.get_states_and_refs_trt()`
  （需在 `transformer.py` 中同步新增 TRT 变体，主要去掉 `img_metas` 参数传递）

#### T2.3 新增 `BEVFormerTrackHeadTRT` 类

继承 `BEVFormerTrackHead`，覆盖 `get_detections` 为调用 `get_detections_trt`，
保留 `get_bev_features_trt` 不变（Phase 1 已完成）。

注册 Registry：
```python
@HEADS.register_module()
class BEVFormerTrackHeadTRT(BEVFormerTrackHead):
    def get_detections(self, bev_embed, query, ref_pts, img_metas=None):
        return self.get_detections_trt(bev_embed, query, ref_pts)
```

#### T2.4 `transformer.py` 补充 `get_states_and_refs_trt`

在 `PerceptionTransformer` 中新增 `get_states_and_refs_trt` 方法，
与 `get_states_and_refs` 逻辑完全相同，但：
- 移除 `img_metas` 参数
- `spatial_shapes` 和 `level_start_index` 用确定性常量构建（不依赖运行时 Python dict）
- decoder 调用的 `forward()` 中确保走 `CustomMSDeformableAttentionTRT` 路径（通过 config 控制）

---

### T3：MotionTransformerDecoderTRT + MotionHeadTRT

**文件**：`projects/mmdet3d_plugin/univ2x/dense_heads/motion_head.py`
及 `dense_heads/motion_head_plugin/` 子目录

#### T3.1 检查 MotionHead 内部结构

在开始编码前，先读取以下文件确认其 MSDA 使用情况：
- `dense_heads/motion_head_plugin/base_motion_head.py`
- `dense_heads/motion_head_plugin/` 下的 transformer 相关文件

关注点：
1. `transformerlayers` 配置中是否包含 `CustomMSDeformableAttention` 或 `MSDeformableAttention3D`
2. 是否有 Python for 循环或动态 shape 操作
3. `nonlinear_smoother` 是否可 ONNX 导出（怀疑含 scipy/casadi 依赖）

#### T3.2 `MotionTransformerDecoderTRT`

若 MotionHead 的 transformer decoder 使用 `CustomMSDeformableAttention`（概率高）：
新增 `MotionTransformerDecoderTRT` 继承原 decoder，将子层替换为 `CustomMSDeformableAttentionTRT`。

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `MotionTransformerDecoderTRT`。

#### T3.3 `MotionHeadTRT`

关键点：
- `use_nonlinear_optimizer=False` 时，`nonlinear_smoother` 不被调用；Phase 2 导出时强制设为 False
- `anchor_coordinate_transform` 是否含不可 ONNX 的操作（需检查 `functional.py`）
- `bivariate_gaussian_activation` / `norm_points` / `pos2posemb2d` 均为纯张量操作，可直接导出

注意：若 `nonlinear_smoother` 含 casadi/scipy，与 PlanningHead 一样移至 ONNX 图外。

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `MotionHeadTRT`，适配类名前缀。

---

### T4：OccHeadTRT

**文件**：`projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py`

#### T4.1 检查 OccHead 内部结构

在编码前，先读取以下文件：
- `dense_heads/occ_head_plugin/` 下所有文件
- 特别检查 `transformer_decoder` 的类型（`OccHead.__init__` 中 `build_transformer_layer_sequence` 的 config）

`OccHead` 结构预测有三类组件：
1. `BevFeatureSlicer` — 纯张量切片/插值，可 ONNX 导出
2. `CVT_Decoder` / `SimpleConv2d` / `Bottleneck` — 标准 Conv2D，可导出
3. `transformer_decoder` — 需检查是否用 `CustomMSDeformableAttention`

#### T4.2 `OccHeadTRT`

若 transformer_decoder 含 `CustomMSDeformableAttention`：
新增 `OccHeadTRT` 继承 `OccHead`，通过 config 将 decoder 子层替换为 TRT 变体。

注意 `predict_instance_segmentation_and_trajectories` 函数：
- 若含 Python 控制流（如动态阈值过滤、非 torch 操作），需提取为单独 TRT 变体
- 若仅含 softmax/argmax 等标准算子，可直接导出

来源：`uniad-trt/patch/uniad-onnx-export.patch` 中 `OccHeadTRT`。

---

### T5：UniV2XTrackTRT（顶层推理包装器）

**文件**：`projects/mmdet3d_plugin/univ2x/detectors/univ2x_track.py`

#### T5.1 TRT 推理的帧间状态设计

参照 UniAD-TRT，将 `Instances` 对象展平为 14 个固定大小张量作为 ONNX 的输入/输出：

| 张量名 | shape | 说明 |
|---|---|---|
| `track_query` | (num_query, embed_dims*2) | query 位置 + 特征拼接 |
| `track_ref_pts` | (num_query, 3) | 参考点（inverse sigmoid 空间） |
| `track_scores` | (num_query,) | 上一帧分类分数 |
| `track_obj_idxes` | (num_query,) int | track ID（-1 = 未激活） |
| `track_disappear_time` | (num_query,) int | 消失帧数（用于删除判断） |
| `track_pred_boxes` | (num_query, 10) | 上一帧预测框 |
| `track_pred_logits` | (num_query, num_classes) | 上一帧预测 logits |
| `track_output_embedding` | (num_query, embed_dims) | decoder 输出特征 |
| `track_mem_bank` | (num_query, mem_bank_len, embed_dims) | 历史记忆特征 |
| `track_mem_padding_mask` | (num_query, mem_bank_len) bool | 记忆 padding mask |
| `track_iou` | (num_query,) | 上一帧 IOU |
| `track_track_scores` | (num_query,) | track 分数（用于过滤） |
| `track_save_period` | (num_query,) | 持久化计数器 |
| `track_past_traj` | (num_query, past+fut, 2) | 历史+预测轨迹 |

> 注：UniAD-TRT 使用 0~13 编号，UniV2X 与 UniAD `Instances` 字段相同，直接对应。

#### T5.2 ONNX 图范围（Stage C）

`UniV2XTrackTRT.forward_trt()` 的输入/输出（纯张量，无 Python dict）：

**输入**（对应 ONNX input_names）：
```
bev_embed        (bev_h*bev_w, bs, C)     — 来自 Stage B
track_query      (num_query, C*2)          — 帧间状态
track_ref_pts    (num_query, 3)
... (其余 12 个 track 状态张量)
l2g_r1           (3, 3)                   — 坐标变换（上一帧 local→global 旋转）
l2g_t1           (3,)                     — 坐标变换（上一帧 local→global 平移）
l2g_r2           (3, 3)                   — 当前帧
l2g_t2           (3,)
time_delta       scalar                   — 帧间时间差
```

**输出**（对应 ONNX output_names）：
```
new_track_query       (num_query, C*2)
new_track_ref_pts     (num_query, 3)
new_track_scores      (num_query,)
... (其余更新后的 track 状态)
all_cls_scores        (num_dec_layers, num_query, num_classes)
all_bbox_preds        (num_dec_layers, num_query, code_size)
occ_output            (视 OccHead 输出而定)
motion_output         (视 MotionHead 输出而定)
```

#### T5.3 `velo_update` 的 TRT 兼容性

`velo_update()` 中 `torch.linalg.inv(l2g_r2)` 已通过 Phase 1 的 `register_inverse_symbolic()` 注册。
无需额外修改，导出时会自动生成 `InversePlugin` 节点。

#### T5.4 需排除在 ONNX 图外的逻辑

以下逻辑在 C++ 推理框架（Phase 4）或 Python 验证脚本中实现，**不进入 ONNX 图**：
- `QueryInteractionModule.forward()`（track 激活/冻结决策）
- `RuntimeTrackerBase`（score_thresh 过滤）
- `MemoryBank.update()`（记忆更新）
- `_generate_empty_tracks()`（初帧初始化）
- `_get_coop_bev_embed()`（V2X 融合，Phase 3 处理）

> 这些逻辑均在 ONNX 图前后用 PyTorch 或 C++ 执行，作为"状态管理层"。

#### T5.5 注册

```python
@DETECTORS.register_module()
class UniV2XTrackTRT(UniV2XTrack):
    def forward_trt(self, bev_embed, *track_states, l2g_r1, l2g_t1,
                    l2g_r2, l2g_t2, time_delta):
        ...
```

---

### T6：扩展 ONNX 导出脚本

**文件**：`tools/export_onnx_univ2x.py`

在现有脚本基础上新增 `--heads-only` 模式（Stage C 导出）：

#### T6.1 新增 `HeadsWrapper`

```python
class HeadsWrapper(nn.Module):
    """Stage C: bev_embed + track 状态 → 新状态 + 预测输出"""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector  # UniV2XTrackTRT 实例

    def forward(self, bev_embed,
                track_query, track_ref_pts, track_scores,
                track_obj_idxes, track_disappear_time,
                track_pred_boxes, track_pred_logits,
                track_output_embedding, track_mem_bank,
                track_mem_padding_mask, track_iou,
                track_track_scores, track_save_period, track_past_traj,
                l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta):
        return self.detector.forward_trt(
            bev_embed,
            track_query, track_ref_pts, track_scores,
            track_obj_idxes, track_disappear_time,
            track_pred_boxes, track_pred_logits,
            track_output_embedding, track_mem_bank,
            track_mem_padding_mask, track_iou,
            track_track_scores, track_save_period, track_past_traj,
            l2g_r1=l2g_r1, l2g_t1=l2g_t1,
            l2g_r2=l2g_r2, l2g_t2=l2g_t2,
            time_delta=time_delta,
        )
```

#### T6.2 新增命令行参数

```python
parser.add_argument('--heads-only', action='store_true',
                    help='Phase 2: export BEV heads (Stage C) only')
parser.add_argument('--num-query', type=int, default=901,
                    help='Fixed track query count (including ego query)')
parser.add_argument('--mem-bank-len', type=int, default=4)
```

#### T6.3 Step A（图结构验证）

随机权重 + 50×50 BEV：

```bash
python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    --model ego \
    --random-weights \
    --heads-only \
    --bev-size 50 \
    --out onnx/univ2x_ego_heads_50_rand.onnx
```

#### T6.4 Step B（精度验收）

真实 checkpoint + 200×200 BEV：

```bash
python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    ckpts/univ2x_coop_e2e_stg1.pth \
    --model ego \
    --heads-only \
    --bev-size 200 \
    --out onnx/univ2x_ego_heads_200.onnx
```

#### T6.5 ONNX 后处理扩展

在现有 INT64→INT32 MSDAPlugin 补丁的基础上，检查 `DetectionTransformerDecoderTRTP`
内部的 `CustomMSDeformableAttentionTRT` 节点是否也产生 INT64 常量（相同的 `spatial_shapes` 问题），
若有则扩展 post-processing 逻辑覆盖这些节点。

---

### T7：TRT 配置文件

**文件**：`projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py`

在 Phase 1 的 `univ2x_coop_e2e_track_trt_p.py` 基础上，对 ego 和 infra 模型的 head 部分
追加 TRT 变体替换：

```python
_base_ = './univ2x_coop_e2e_track_trt_p.py'  # 继承 Phase 1 config（BEV encoder 已是 TRT 变体）

# 对 model_ego_agent 和 model_other_agent_inf 同步做以下替换：

# TrackHead → TRT 变体
pts_bbox_head=dict(
    type='BEVFormerTrackHeadTRT',
    transformer=dict(
        ...
        decoder=dict(
            type='DetectionTransformerDecoderTRTP',
            transformerlayers=dict(
                attn_cfgs=[
                    dict(type='CustomMSDeformableAttentionTRT', ...),
                ]
            )
        )
    )
)

# MotionHead → TRT 变体
motion_head=dict(
    type='MotionHeadTRT',
    use_nonlinear_optimizer=False,  # 强制关闭 casadi 依赖
    transformerlayers=dict(
        type='MotionTransformerDecoderTRT',
        ...
    )
)

# OccHead → TRT 变体（若内部 decoder 含 MSDA）
occ_head=dict(
    type='OccHeadTRT',
    ...
)
```

---

### T8：精度验证脚本

**文件**：`tools/validate_trt_heads.py`

#### T8.1 单帧前向对比

```python
"""
验证 Stage C TRT engine vs PyTorch 输出。

步骤：
1. 加载真实 checkpoint，构建 PyTorch 模型
2. 用相同的随机种子构造 dummy bev_embed + dummy track 状态
3. 分别用 PyTorch forward_trt() 和 TRT engine 做推理
4. 对比关键输出的 max abs diff 和 cosine similarity
"""
```

#### T8.2 验收指标

| 指标 | 阈值 | 说明 |
|---|---|---|
| `all_cls_scores` max abs diff | < 1e-3 | 分类 logits |
| `all_bbox_preds` max abs diff | < 5e-3 | 回归预测框 |
| `new_track_ref_pts` max abs diff | < 1e-3 | 参考点更新 |
| cosine similarity（所有输出） | > 0.9999 | 整体方向一致性 |

注：最终精度验收需在 nuScenes val set 上运行完整 tracking pipeline，测 AMOTA 差异 < 0.005。
单帧 dummy 对比只用于 ONNX/TRT 编译正确性验证。

---

## 四、执行顺序与依赖关系

```
Step 1: 阅读 motion_head_plugin/ 和 occ_head_plugin/ 下的所有文件
        → 确认 MotionHead / OccHead 内部是否用 CustomMSDeformableAttention
        → 明确哪些函数不可 ONNX 导出（nonlinear_smoother 等）

Step 2: T1（decoder TRT 变体）
        → 最基础依赖，T2/T3 都需要

Step 3: T2（TrackHead TRT）
        → 依赖 T1

Step 4: T3（MotionHead TRT）
        → 依赖 T1，与 T2 并行

Step 5: T4（OccHead TRT）
        → 与 T2/T3 并行，但需先完成 Step 1 的阅读

Step 6: T5（UniV2XTrackTRT）
        → 依赖 T2 T3 T4

Step 7: T7（config 文件）
        → 依赖 T1-T5 所有新类注册完成

Step 8: T6（导出脚本 + Step A 图结构验证）
        → 依赖 T7

Step 9: Step B（精度验收）+ T8（验证脚本）
        → 依赖 Step 8 ONNX 正确生成
```

---

## 五、预期遇到的 Bug 及应对方案

### Bug 1：`MotionHead.nonlinear_smoother` 含 casadi

**现象**：导出时 `import casadi` 或调用 `scipy.optimize` 报错
**应对**：在 `MotionHeadTRT.__init__` 中强制 `self.use_nonlinear_optimizer = False`，
并在导出脚本中设置 `--cfg-options model_ego_agent.motion_head.use_nonlinear_optimizer=False`

### Bug 2：`OccHead.predict_instance_segmentation_and_trajectories` 含 argmax + np

**现象**：导出时遇到 numpy 操作或非 torch 操作
**应对**：将该函数的后处理部分（阈值过滤、实例分配）移至 ONNX 图外；ONNX 图只输出 raw logits

### Bug 3：`DetectionTransformerDecoder` 的 `reference_points` 更新中 `torch.zeros_like` 被追踪为动态形状

**现象**：ONNX export 时出现动态形状警告或 checker 错误
**应对**：用显式 `torch.zeros(bs, num_query, 3, device=...)` 替代 `zeros_like`

### Bug 4：`CustomMSDeformableAttentionTRT` 中 `spatial_shapes` INT64 问题

**现象**：TRT 构建 engine 时 `MSDAPlugin` inputs[1] 类型错误
**应对**：扩展现有 ONNX post-processing，覆盖 decoder 里的 `MSDAPlugin` 节点（同 BEV encoder 的修复方案）

### Bug 5：`velo_update` 中 `torch.linalg.inv` 未被 InversePlugin 捕获

**现象**：ONNX 中出现原生 `Inverse` op 而非 `InversePlugin` 节点
**应对**：在导出脚本头部确认 `register_inverse_symbolic()` 在 `torch.onnx.export` 之前被调用
（Phase 1 已验证，沿用相同调用顺序）

### Bug 6：`track_obj_idxes` / `track_disappear_time` 是 int 类型张量

**现象**：ONNX export 时 INT64 索引张量在 TRT 中出现类型不支持错误
**应对**：在 TRT 推理时将这两个张量作为 INT32，若 ONNX 导出为 INT64 则在 post-processing 中修正

---

## 六、验收标准

| 验收项 | Step | 通过标准 |
|---|---|---|
| T1：decoder TRT 变体类导入 + Registry 注册 | — | import 不报错，mmcv Registry 有对应 key |
| T2：TrackHead TRT 前向（随机权重） | — | `get_detections_trt()` 输出 shape 与原版一致 |
| T3：MotionHead TRT 前向（随机权重） | — | 无 casadi/scipy 依赖，前向不报错 |
| T4：OccHead TRT 前向（随机权重） | — | 前向不报错，输出 shape 正确 |
| T5：UniV2XTrackTRT 单帧前向（随机权重） | — | `forward_trt()` 输出 5 类张量 shape 正确 |
| ONNX 图结构（ego + infra） | **A** | 含 MSDAPlugin（decoder 内）/ InversePlugin 节点 |
| TRT engine 编译（ego + infra） | **A** | 无不支持层，engine 生成成功 |
| ONNX checker（ego + infra） | **B** | `onnx.checker.check_model` 通过 |
| TRT FP32 单帧精度（ego + infra） | **B** | `all_cls_scores` max diff < 1e-3，`all_bbox_preds` max diff < 5e-3 |
| 完整 tracking AMOTA | **B** | TRT FP32 vs PyTorch AMOTA 差异 < 0.005 |

---

## 七、文件变更清单

| 文件路径 | 操作 | 核心内容 |
|---|---|---|
| `projects/mmdet3d_plugin/univ2x/modules/decoder.py` | 末尾追加 | `CustomMSDeformableAttentionTRT`、`DetectionTransformerDecoderTRTP` |
| `projects/mmdet3d_plugin/univ2x/modules/transformer.py` | 末尾追加 | `get_states_and_refs_trt()` 方法 |
| `projects/mmdet3d_plugin/univ2x/modules/__init__.py` | 修改 | 导出新增 TRT 类 |
| `projects/mmdet3d_plugin/univ2x/dense_heads/track_head.py` | 末尾追加 | `get_detections_trt()`、`BEVFormerTrackHeadTRT` |
| `projects/mmdet3d_plugin/univ2x/dense_heads/motion_head.py` | 末尾追加 | `MotionHeadTRT` |
| `projects/mmdet3d_plugin/univ2x/dense_heads/motion_head_plugin/` | 追加/修改 | `MotionTransformerDecoderTRT`（待确认路径） |
| `projects/mmdet3d_plugin/univ2x/dense_heads/occ_head.py` | 末尾追加 | `OccHeadTRT` |
| `projects/mmdet3d_plugin/univ2x/detectors/univ2x_track.py` | 末尾追加 | `UniV2XTrackTRT`，`forward_trt()` |
| `projects/mmdet3d_plugin/univ2x/detectors/__init__.py` | 修改 | 导出 `UniV2XTrackTRT` |
| `projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py` | 新建 | Phase 2 TRT 配置（继承 p.py，追加 head TRT 类型） |
| `tools/export_onnx_univ2x.py` | 修改 | 新增 `--heads-only` 模式、`HeadsWrapper`、dummy input 生成 |
| `tools/validate_trt_heads.py` | 新建 | Stage C TRT engine 精度验证脚本 |

---

## 八、注意事项

1. **先读后改**：T3（MotionHead）和 T4（OccHead）开始前必须先读 `motion_head_plugin/` 和
   `occ_head_plugin/` 下的所有文件，不能假设内部结构与 UniAD 完全一致。
   UniV2X 在这些 head 上可能有独立修改。

2. **不破坏训练路径**：所有 TRT 变体类均通过继承 + 重写实现，不修改原类的任何方法。
   训练代码（`forward_train`、`_forward_single_frame_train`）完全不受影响。

3. **帧间状态接口与 Phase 4 C++ 对齐**：T5 设计的 14 个张量名称和 shape 需与
   Phase 4 C++ 推理应用（`enqueueV3` 框架）中的 `KernelInput`/`KernelOutput` 结构对应。
   定义时写清楚 shape 和 dtype 注释，方便 Phase 4 编写绑定代码。

4. **BEV encoder ONNX 与 Heads ONNX 的接口对齐**：Stage B（BEV encoder）输出的 `bev_embed`
   shape 为 `(bev_h*bev_w, bs, embed_dims)`，Stage C 输入的 `bev_embed` 必须相同格式，
   **不可在中间插入任何 reshape**（避免引入不必要的 permute 节点影响精度调试）。

5. **PlanningHead 不在 Phase 2 范围内**：`UniV2X` 顶层模型有 `planning_head`，
   但 Phase 2 目标是 Stage 1（Perception Only），导出脚本应通过 config 将 `planning_head=None`
   或使用 `UniV2XTrackTRT`（继承 `UniV2XTrack`，不含 `planning_head`）。

6. **OccHead 的 `receptive_field` 时序维度**：`OccHead` 使用 `receptive_field=3` 个历史帧的
   BEV 特征拼接输入，Phase 2 导出时需决定：①将历史 BEV 帧作为额外输入张量（推荐，与 track 状态类似），
   或②在 ONNX 图外拼接后输入。推荐方案①，清晰且可端到端验证。

7. **INT64 补丁覆盖范围**：现有的 ONNX post-processing 只扫描 `MSDAPlugin` 节点。
   Phase 2 引入新的 `MSDAPlugin` 节点（来自 `CustomMSDeformableAttentionTRT`），
   扫描逻辑不变，但需验证这些新节点也被正确修补。
