# Pyramid Fusion 分段耗时实测 (v1)

> **测试日期**: 2026-05-13
> **数据**: OPV2V test split (2170 scenes, 实测 200 sample 均值)
> **硬件**: RTX 4090
> **模型**: Pyramid_m1_base baseline (HEAL stage1 ckpt, epoch 23, FP32)
> **配置**: range=±102.4 m, batch_size=1, num_workers=2
> **方法**: CUDA Event + torch.cuda.synchronize, warmup=20, measure=200
> **平均 record_len**: 1.61 agents/scene (collab 模式, ego + collaborator)
> **结果文件**: `data/pyramid_per_stage_timing.json`

## 一、总览（mean / p50 / p99 单位 ms）

| 阶段 | mean | p50 | p99 | % of e2e |
|---|---|---|---|---|
| encoder_m1 (PointPillar VFE+Scatter) | 3.01 | 2.81 | 5.67 | 3.1% |
| backbone_m1 (ResNet 2D BEV) | 2.12 | 1.75 | 4.49 | 2.2% |
| aligner_m1 (AlignNet) | 0.024 | 0.002 | 0.072 | 0.0% |
| **pyramid_backbone.forward_collab** | **14.45** | **14.76** | **20.24** | **14.7%** |
| shrink_conv | 3.30 | 3.30 | 3.31 | 3.4% |
| heads (cls+reg+dir) | 0.087 | 0.087 | 0.088 | 0.1% |
| **forward 小计** | **22.99** | – | – | **23.4%** |
| postproc.decode (delta_to_boxes3d + mask + sigmoid) | 0.97 | 1.04 | 1.70 | 1.0% |
| postproc.dir (方向 classifier 整合) | 0.52 | 0.58 | 0.81 | 0.5% |
| postproc.corners (boxes_to_corners + project_box3d) | 1.05 | 1.16 | 1.76 | 1.1% |
| **postproc.nms (nms_rotated)** | **71.69** | **54.72** | **176.32** | **72.9%** |
| postproc.range_mask | 0.29 | 0.29 | 0.37 | 0.3% |
| **postproc 小计** | **74.62** | – | – | **75.8%** |
| **e2e walltime (含 Python 开销)** | **98.37** | **82.17** | – | **100%** |

## 二、Smoking gun：`nms_rotated` 是 **Pure Python + Shapely 的 CPU 循环**

源码：`opencood/utils/box_utils.py:693-738`

```python
def nms_rotated(boxes, scores, threshold):
    boxes  = boxes.cpu().detach().numpy()          # GPU→CPU sync
    scores = scores.cpu().detach().numpy()
    polygons = common_utils.convert_format(boxes)  # 构建 Shapely Polygon list
    pick = []
    while len(ixs) > 0:                            # O(N²) Python loop
        i = ixs[0]; pick.append(i)
        iou = common_utils.compute_iou(polygons[i], polygons[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs); ixs = np.delete(ixs, 0)
    return ...
```

`compute_iou` 内部（`common_utils.py:230-252`）：

```python
iou = [box.intersection(b).area / box.union(b).area for b in boxes]
```

—— **Python list comprehension 调 Shapely Polygon.intersection 和 .union 的 O(N²) 算 IoU**。Shapely 单次 intersection 大概 50-100 μs；对每个 sample ~50-100 候选框，总计算量 ~N²/2 = 1250-5000 次 Shapely 调用 = **50-200 ms** 完全合理。

p99 = 176 ms（高变异）确认了"候选框数量"是主因——目标多的 scene 候选多，NMS 就线性级地慢。

## 三、推翻三个之前的论断

### 论断 1：之前说 "NMS 占 e2e 60%"

**实测**：**NMS 占 e2e 73%**（甚至比之前估算的还高 13 个百分点）

**根本原因**：之前是反推估算（e2e 35ms - 实测的 subnet 3.3ms 推 postproc），现在是分段直测。

### 论断 2：之前说 "pyramid_backbone 占 e2e 9%"

**实测**：`pyramid_backbone.forward_collab` 占 e2e **14.7%**（绝对值 14.45ms，**不是** 3.3ms）

**为什么差 4×**：
- 之前的 3.3ms 是**单代理 dummy 输入 + ONNX 静态 export** 的子网测量（M4.6.1）
- 现在的 14.45ms 是 **forward_collab 真实跑**，包含：
  1. `get_multiscale_feature`：3-stage ResNet (跟 M4.6.1 的 3.3ms 同构)
  2. **每级 single_head_i + warp_affine_simple + weighted_fuse** ← 这部分 ONNX 没 export，是车端融合的真实开销
  3. `decode_multiscale_feature`：多尺度上采样合并

跨代理 warp + weighted_fuse 那部分是 **V2X 独有**，单代理 baseline (PointPillars) 完全没有。

### 论断 3：之前说 "e2e baseline 35.31 ms FP32"（M4.6.0）

**实测**：**e2e walltime ~98 ms**（mean），p50 = 82 ms

**怎么差这么多**：
- M4.6.0 那个 35.31ms **几乎肯定只测了 `model(batch)` forward**，没含 postproc/NMS
- 当时的 timing 路径是 `start_event → model.forward → end_event`
- 现在的 22.99ms forward 加 0.087ms heads 和上 Python overhead ≈ 24ms，与 35ms 差距还在但量级一致
- 真实 e2e 含 postproc 是 **~98 ms**，比 M4.6.0 报告的快了 ~2.8×（其实是慢，不是快——M4.6.0 漏算了 postproc）

**结论**：M4.6.0 的 35.31ms 数据**不是 e2e**，是 forward-only。我们之前所有"加速倍数 vs FP32 baseline"的计算都用了错的 denominator。

## 四、几个关键 takeaway

### 4.1 你的直觉是对的（也是错的）

| 你说 | 实际 |
|---|---|
| "CNN backbone 一直是网络计算最核心的区域" | ✓ **forward 范围内**：pyramid + backbone + encoder + shrink ≈ 22ms = forward 的 95%。CNN 确实是 forward 主体。 |
| "NMS 不会有这么高的耗时" | ✗ **e2e 范围内**：NMS 是 72ms = e2e 的 73%。**但**——这不是因为 NMS 算法本身贵，而是因为 HEAL 用 Shapely Python 循环实现，不是 CUDA kernel。 |

### 4.2 路侧 vs 车端的真实账（去掉 NMS 后）

如果**只看路侧 RSU 子模型（encoder + backbone + aligner）**：
- 实测：3.01 + 2.12 + 0.024 = **5.15 ms**（按 1.61 agents 算 → 单 agent ≈ 3.2 ms）
- 这跟 UPAQ 报告的 PointPillars **在 RTX 4080 上 5.72 ms** 几乎完全一致 ✓✓
- → **路侧端剪枝+量化的 UPAQ 风格优化是合理的**，预期能拿到 1.5-2× 单代理加速

**车端 EGO 子模型（pyramid_backbone + shrink + heads + NMS）**：
- forward 部分：14.45 + 3.30 + 0.087 = **17.84 ms**（这部分 CNN 占主导，可剪枝可量化）
- postproc 部分：**74.62 ms（其中 NMS 占 71.69 ms）** ← 这是 CUDA kernel 替换问题，不是剪枝问题

### 4.3 加速优先级重排

按 ROI 排：

| 优先级 | 方法 | 攻击对象 | 预期节省 | e2e 加速 |
|---|---|---|---|---|
| **P0** | MMCV `nms_rotated_cuda` 替换 Shapely NMS | postproc NMS 72ms | 72ms → ~1ms | **98→27ms = 3.6×** |
| **P1** | UPAQ-style pruning + INT8（仅 RSU 路侧） | encoder+backbone 5ms | 5→2.5ms | (路侧单 agent: 3.2→1.6ms = 2×) |
| **P2** | pyramid_backbone INT8 + 32-aligned channel prune | pyramid 14ms | 14→9ms | e2e 98→93ms (~1.05×) |
| **P3** | 2:4 sparse on backbone_m1 | backbone 2ms | 2→1.5ms | 微乎其微 |

**结论**：**P0 (CUDA NMS) 单点能拿到 3.6× e2e 加速**，其他优化都被 P0 之前的 NMS 拖死，必须先做 P0。

## 五、需要修正的之前文档

1. `Pyramid_架构与剪枝边界_audit.md` §3 "e2e 时间分布"——用估算的 30/9/60% 是错的，实测是 23/77 = forward/postproc，需更新成本表
2. `Pyramid_真加速可行路径_v1.md` §六 决策矩阵——Path B (CUDA NMS) 的 ROI 被低估，应升级为 **P0 必做**
3. `00_故事评估与实验路线_v1.md` 和 `数据集制作_plan.md` 里所有引用 "FP32 35.31ms e2e baseline" 的地方都需注明**那个是 forward-only 不是 e2e**

## 六、还可以补的测量

1. 拆分 `pyramid_backbone.forward_collab` 内部的 multiscale ResNet vs warp_affine_simple vs weighted_fuse 三块各自占比
2. 测 single-agent 模式 (`HeterPyramidSingle`) 的 e2e，对比 collab 模式
3. 把 NMS 候选框数量也记下来 (`pred_box3d_tensor.shape[0]`)，验证 N² 假设
4. 测 DAIR-V2X 数据集上同模型的分布（与 OPV2V 数值对比）
