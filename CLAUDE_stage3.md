# Phase 3：V2X 融合模块 TRT 适配实现计划

> 对应 `plan.md` 第六章 Phase 3，目标：将 UniV2X 独有的 V2X 协同融合模块
> （`AgentQueryFusion` + `LaneQueryFusion`）适配为 TRT 可导出形式，完成含
> V2X 融合分支的完整 Stage 2 ONNX 导出，并通过精度验收。

---

## 当前进度总结（Phase 1 + Phase 2 已完成）

### Phase 1 — BEV 编码器 TRT（✅ 已完成）

| 交付物 | 文件 | 精度 |
|--------|------|------|
| C++ 插件 | `plugins/build/libuniv2x_plugins.so` | — |
| ego BEV encoder | `trt_engines/univ2x_ego_bev_encoder_200.trt` (72 MB) | 余弦相似度 0.9999993 ✅ |
| infra BEV encoder | `trt_engines/univ2x_infra_bev_encoder_200.trt` (72 MB) | 余弦相似度 0.9999997 ✅ |

主要工作：MSDAPlugin / RotatePlugin / InversePlugin 三个 C++ TRT 插件；
`BEVFormerEncoderTRT` / `SpatialCrossAttentionTRT` / `TemporalSelfAttentionTRT` Python TRT 模块变体；
`BEVEncoderWrapper`（跳过含 DCNv2 的骨干网络）；ONNX 后处理修补 INT64→INT32。

关键 Bug 修复：MSDAPlugin 拒绝 INT64 输入；`inversePlugin.cpp` cudaMalloc 类型转换；
cooperative checkpoint `model_ego_agent.` 前缀剥离。

### Phase 2 — 检测头 + 下游头 TRT（✅ 已完成）

#### 第一部分：检测头（CLAUDE_stage2.md）

| 交付物 | 文件 | 精度 |
|--------|------|------|
| ego 检测头 | `trt_engines/univ2x_ego_heads_200.trt` (33 MB) | 余弦相似度 > 0.9999994 ✅ |

主要工作：`CustomMSDeformableAttentionTRT`；`DetectionTransformerDecoderTRTP`；
`BEVFormerTrackHeadTRT.get_detections_trt()`；`HeadsWrapper`（bev_embed + track 状态 → 新状态 + 预测）；
`--heads-only` 导出模式。

#### 第二部分：下游头 Motion + Occ + Planning（本阶段新完成）

| 交付物 | 文件 | 精度 |
|--------|------|------|
| ego 下游头 | `trt_engines/univ2x_ego_downstream.trt` (152 MB) | 余弦相似度 > 0.9999854 ✅ |
| infra 下游头 | `trt_engines/univ2x_infra_downstream.trt` (134 MB) | 余弦相似度 > 0.9999854 ✅ |

主要工作：`DownstreamHeadsWrapper`（MotionHeadTRTP → OccHeadTRTP → PlanningHeadSingleModeTRT）；
`tools/build_trt_downstream.py`；`tools/validate_downstream_trt.py`（Step A/B 两阶段验证）；
`onnx_compatible_attention()` 四项补丁（SDPA / unflatten / Dropout / **BN 冻结**）；
ONNX 后处理修补 Dropout training_mode。

关键 Bug 修复（最难排查）：`TrainingMode.TRAINING` 导致 BN 在追踪期间原地更新
`running_mean` / `running_var`，使 ONNX 嵌入的统计量与 checkpoint 不一致。
**修复方案**：在 `onnx_compatible_attention` 中逐实例覆盖 BN.forward 为
`F.batch_norm(..., training=False)`，强制 eval 模式计算，BN 节点自动以
`training_mode=0` 导出，无需 ONNX 后处理。

---

## Phase 3 目标

在 Phase 2 单智能体基础上，完整接入 V2X 协同分支：

```
[基础设施侧 TRT 引擎]
  img_feats_inf → BEV encoder → bev_inf
  bev_inf → 检测头 → inf_track_queries, inf_lane_queries

         ↓
[CPU 预处理层（图外）]
  · 匈牙利匹配（lapjv / scipy，N ≤ 200，< 1ms）
  · ego2other 坐标变换
  输出: matched_veh_idx, matched_inf_idx, transformed_queries

         ↓
[自车侧 TRT 引擎]
  img_feats_ego + bev_ego + ego 状态
  + fused_agent_queries（来自 AgentQueryFusion TRT 子图）
  + fused_lane_queries（来自 LaneQueryFusion TRT 子图）
  → 完整 Stage 2 输出（planning + tracking + occ）
```

---

## 一、Phase 3 的关键技术挑战

### 1.1 `AgentQueryFusion._query_matching()` 含匈牙利算法

**问题**：`scipy.optimize.linear_sum_assignment` 是 Python 标量操作，完全不可 ONNX 导出。

**解决方案（方案 A：CPU-GPU 分离，推荐）**：
- 将匈牙利匹配作为 ONNX 图**外部** CPU 预处理步骤
- 输出 `matched_veh_idx`（形状 `[M]`）和 `matched_inf_idx`（形状 `[M]`）作为 ONNX 图的额外输入
- ONNX 图只导出匹配后的特征融合（MLP 部分）
- Python 验证脚本用 `scipy`，C++ 推理用 `lapjv`（N ≤ 200，< 1ms）

### 1.2 `LaneQueryFusion` 含 Python for 循环坐标变换

**问题**：
```python
for i in range(lane_query.shape[1]):   # 不可 ONNX 追踪
    transformed[i] = rot @ lane_query[i] + trans
```
以及 `np.linalg.inv` 调用。

**解决方案**：
- for 循环 → 批矩阵乘法向量化（`torch.bmm`）
- `np.linalg.inv` → 复用已有 `InversePlugin`
- `torch.where` 动态过滤 → mask 乘法（静态图兼容）

### 1.3 V2X 坐标系对齐的动态性

infra 侧与 ego 侧的标定矩阵 `ego2other_rt` 每帧变化，且 infra track 数量动态变化。

**解决方案**：同 track 状态处理策略，固定上限 N_inf（如 200），padding + mask。

---

## 二、Phase 3 任务分解

```
T1: 阅读并分析融合模块代码
    → fusion_modules/agent_fusion.py
    → fusion_modules/lane_fusion.py
    → 确认动态操作位置和可导出边界

T2: LaneQueryFusionTRT
    → 向量化坐标变换，消除 Python for 循环
    → 复用 InversePlugin

T3: AgentQueryFusionTRT（方案 A）
    → ONNX 图含 MLP 融合部分
    → 匹配索引作为额外输入

T4: UniV2XTRT 顶层融合包装器
    → 接入 infra queries → AgentQueryFusion → 融合后 ego forward

T5: 扩展 ONNX 导出脚本（--v2x 模式）
    → export_onnx_univ2x.py 新增 V2XWrapper

T6: 更新 TRT 配置
    → univ2x_coop_e2e_track_trt_p3.py

T7: 精度验证脚本
    → tools/validate_v2x_trt.py
    → Step A（图结构 + 随机权重）
    → Step B（真实 checkpoint 精度对比）
```

---

## 三、详细实现步骤

### T1：阅读并分析融合模块

**必须在开始 T2/T3 之前完整阅读以下文件**：

```
projects/mmdet3d_plugin/univ2x/
├── fusion_modules/
│   ├── __init__.py
│   ├── agent_fusion.py        # AgentQueryFusion 核心
│   └── lane_fusion.py         # LaneQueryFusion 核心
├── detectors/
│   ├── univ2x_e2e.py          # UniV2X.forward_test() V2X 分支入口
│   └── univ2x_track.py        # UniV2XTrack._get_coop_bev_embed()
```

需要确认的关键问题：

| 问题 | 影响 |
|------|------|
| `AgentQueryFusion._query_matching()` 的代价矩阵构造方式 | 决定 CPU 侧需要计算哪些量 |
| `LaneQueryFusion` 的输入 tensor 类型和 shape | 决定 ONNX 输入接口设计 |
| `_get_coop_bev_embed()` 的完整逻辑 | 决定是否需要额外的 BEV 融合 TRT 子图 |
| V2X 分支是否使用了 `casadi` / `scipy` 以外的不可导出操作 | 决定额外需要处理的 Bug |

---

### T2：`LaneQueryFusionTRT`

**文件**：`projects/mmdet3d_plugin/univ2x/fusion_modules/lane_fusion.py`

#### T2.1 分析原始 `LaneQueryFusion.forward()`

预期结构（需 T1 阅读后确认）：
```python
def forward(self, ego_lane_query, inf_lane_query, ego2other_rt):
    # 1. 用 ego2other_rt 变换坐标
    # 2. 计算相似度矩阵
    # 3. 融合特征
    ...
```

#### T2.2 新增 `LaneQueryFusionTRT`

替换策略：

| 原始操作 | TRT 替换方案 |
|----------|-------------|
| `for i in range(N): transformed[i] = ...` | `torch.bmm(ego2other_r, lane_q) + ego2other_t` |
| `np.linalg.inv(ego2other_rt)` | `InversePlugin(ego2other_rt)` |
| `torch.where(mask, a, b)` 动态过滤 | `mask.float() * a + (1 - mask.float()) * b` |

```python
@FUSION_MODULES.register_module()
class LaneQueryFusionTRT(LaneQueryFusion):
    def forward_trt(self, ego_lane_query, inf_lane_query,
                    ego2other_r, ego2other_t):
        """TRT-compatible lane query fusion.

        Args:
            ego_lane_query  (Tensor): (1, M, C)  — ego 侧车道 query
            inf_lane_query  (Tensor): (1, M, C)  — infra 侧车道 query
            ego2other_r     (Tensor): (3, 3)     — 旋转矩阵（拆开以避免求逆）
            ego2other_t     (Tensor): (3,)       — 平移向量
        Returns:
            fused_lane_query (Tensor): (1, M, C)
        """
        ...
```

注意：将 `ego2other_rt` 拆分为旋转 `ego2other_r` 和平移 `ego2other_t` 作为独立输入，
避免在 ONNX 图内做矩阵分解。

---

### T3：`AgentQueryFusionTRT`（方案 A：CPU-GPU 分离）

**文件**：`projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py`

#### T3.1 ONNX 图边界设计

```
ONNX 图外（CPU 侧）：
  cost_matrix = compute_cost(ego_queries, inf_queries)   ← 纯矩阵运算，可在 GPU 或 CPU
  matched_veh_idx, matched_inf_idx = lapjv(cost_matrix)  ← CPU, < 1ms

ONNX 图内（TRT）：
  输入: ego_queries (N, C), inf_queries (N_inf, C),
        matched_veh_idx (M,) int32, matched_inf_idx (M,) int32
  输出: fused_queries (N, C)
```

#### T3.2 新增 `AgentQueryFusionTRT`

```python
@FUSION_MODULES.register_module()
class AgentQueryFusionTRT(AgentQueryFusion):
    def forward_trt(self, ego_queries, inf_queries,
                    matched_veh_idx, matched_inf_idx):
        """TRT-compatible agent query fusion.

        匈牙利匹配在 ONNX 图外完成，此处只做特征融合（MLP）。

        Args:
            ego_queries      (Tensor): (N, C)   — 自车 track queries
            inf_queries      (Tensor): (N_inf, C)— infra track queries
            matched_veh_idx  (Tensor): (M,) int32— 自车侧匹配下标（-1 = 无匹配）
            matched_inf_idx  (Tensor): (M,) int32— infra 侧匹配下标
        Returns:
            fused_queries    (Tensor): (N, C)   — 融合后 ego queries
        """
        # 用 scatter/gather（固定 shape）而非动态 mask 索引
        # 将 matched 特征通过 MLP 融合，unmatched 保持原 ego_queries
        ...
```

关键约束：
- 使用固定上限 N=901（ego queries）和 N_inf=200（infra queries，需确认实际上限）
- 匹配下标以 `int32` 传入（TRT 支持 INT32 gather，不支持 INT64）
- 无匹配位置（`matched_veh_idx == -1`）用 mask 乘法处理

#### T3.3 匈牙利匹配辅助函数（ONNX 图外）

在 Python 验证脚本 + C++ 推理应用中实现，不进入 ONNX：

```python
def compute_matching(ego_queries, inf_queries, cost_fn):
    """图外执行，供 Python 验证和 C++ 参考实现。"""
    from scipy.optimize import linear_sum_assignment
    cost = cost_fn(ego_queries, inf_queries)         # (N, N_inf)
    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return torch.tensor(row_ind, dtype=torch.int32), \
           torch.tensor(col_ind, dtype=torch.int32)
```

---

### T4：`UniV2XTRT` 顶层融合包装器

**文件**：`projects/mmdet3d_plugin/univ2x/detectors/univ2x_e2e.py`
（或新文件 `univ2x_v2x_trt.py`，视代码耦合度决定）

#### T4.1 完整 Stage 2 推理流

```
forward_v2x_trt(
    # 当前帧输入
    bev_ego,              # (H*W, 1, C)  — Stage B 输出
    bev_inf,              # (H*W, 1, C)  — infra Stage B 输出
    # 帧间状态（ego 侧，同 Phase 2）
    track_query, track_ref_pts, ...（14 个）
    # 帧间状态（infra 侧，新增）
    inf_track_query, inf_track_ref_pts, ...（N 个）
    # V2X 坐标变换
    ego2other_r, ego2other_t,         # (3,3), (3,)
    # 匹配索引（图外预计算）
    matched_veh_idx, matched_inf_idx,  # (M,) int32
    # 时间信息
    time_delta,
):
    # 1. 基础设施侧检测（无 V2X 分支）
    inf_cls, inf_box, ..., inf_lane_q = self.infra_model.forward_trt(bev_inf, ...)

    # 2. V2X 融合
    fused_agent_q = self.agent_fusion.forward_trt(
        ego_track_q, inf_track_q, matched_veh_idx, matched_inf_idx)
    fused_lane_q  = self.lane_fusion.forward_trt(
        ego_lane_q, inf_lane_q, ego2other_r, ego2other_t)

    # 3. 自车侧检测（含融合 queries）
    ego_out = self.ego_model.forward_trt(bev_ego, fused_agent_q, fused_lane_q, ...)
    return ego_out
```

注意：实际是否需要 infra 侧检测进入同一 ONNX 图，取决于 infra 状态更新逻辑。
建议先保持两个独立 TRT 引擎（infra engine + ego engine），融合模块作为两者之间的 CPU/GPU 中间层。

---

### T5：扩展 ONNX 导出脚本

**文件**：`tools/export_onnx_univ2x.py`

新增 `--v2x` 导出模式：

```bash
# Step A — 随机权重图结构验证
python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p3.py \
    --model ego \
    --random-weights \
    --v2x \
    --bev-size 50 \
    --out onnx/univ2x_ego_v2x_50_rand.onnx

# Step B — 真实 checkpoint 精度验收
python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p3.py \
    ckpts/univ2x_coop_e2e_stg2.pth \
    --model ego \
    --v2x \
    --bev-size 200 \
    --out onnx/univ2x_ego_v2x.onnx
```

#### T5.1 新增 `V2XWrapper`

```python
class V2XWrapper(nn.Module):
    """Stage 2 V2X 融合包装器：ego track 状态 + infra queries + 匹配索引 → 融合后 ego 输出"""

    def __init__(self, ego_model, lane_fusion, agent_fusion, pc_range):
        super().__init__()
        self.ego_detector   = ego_model
        self.lane_fusion    = lane_fusion   # LaneQueryFusionTRT
        self.agent_fusion   = agent_fusion  # AgentQueryFusionTRT

    def forward(self, bev_ego,
                # ego track 状态（14 个张量）
                track_query, track_ref_pts, ...,
                # infra queries（来自 infra engine 输出）
                inf_track_queries, inf_lane_queries,
                # 匹配索引（图外预计算）
                matched_veh_idx, matched_inf_idx,
                # V2X 坐标变换
                ego2other_r, ego2other_t,
                # 时间
                l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta):
        ...
```

#### T5.2 ONNX 后处理更新

在 `_patch_and_verify_onnx()` 中检查 V2X 融合引入的新节点：
- `AgentQueryFusionTRT` 内的 Gather 操作是否产生 INT64 索引常量
- `LaneQueryFusionTRT` 内是否有新的 MSDAPlugin 节点（若融合用到 MSDA）

---

### T6：TRT 配置文件

**文件**：`projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p3.py`

```python
_base_ = './univ2x_coop_e2e_track_trt_p2.py'   # 继承 Phase 2 配置

# 替换融合模块为 TRT 变体
model_ego_agent = dict(
    ...
    # 若 agent_fusion / lane_fusion 在 ego model config 内
    agent_fusion=dict(type='AgentQueryFusionTRT', ...),
    lane_fusion=dict(type='LaneQueryFusionTRT',   ...),
)
```

---

### T7：精度验证脚本

**文件**：`tools/validate_v2x_trt.py`

```
验证流程：
1. 加载 stg2 checkpoint，构建 PyTorch 模型（含 V2X 融合分支）
2. 构造 dummy 输入（固定随机种子）：
   - bev_ego, bev_inf（200×200）
   - ego track 状态（14 个张量）
   - infra queries（N_inf=200, C=256）
   - 匹配索引（M=50 对）
   - ego2other_r/t（随机旋转+平移）
3. 分别用 PyTorch forward_v2x_trt() 和 TRT engine 推理
4. 对比关键输出 max abs diff + cosine similarity
```

验收指标（参照 Phase 2，适当放宽以容纳 V2X 融合引入的累积误差）：

| 输出 | 阈值 |
|------|------|
| 融合后 traj_scores cosine | > 0.999 |
| 融合后 occ_logits mean abs diff | < 5e-3 |
| fused_lane_query cosine | > 0.999 |
| fused_agent_query cosine | > 0.999 |

---

## 四、执行顺序与依赖关系

```
Step 0: 阅读 fusion_modules/ 下所有文件（T1）
        → 必须在动手写代码前完成

Step 1: LaneQueryFusionTRT（T2）
        → 无外部依赖，可独立实现
        → 验证：随机输入前向不报错，输出 shape 正确

Step 2: AgentQueryFusionTRT（T3）
        → 无外部依赖，与 T2 并行
        → 验证：给定固定匹配索引，MLP 融合输出 shape 正确

Step 3: UniV2XTRT 顶层包装器（T4）
        → 依赖 T2 + T3

Step 4: 扩展导出脚本 + Step A 图结构验证（T5）
        → 依赖 T3 + T4

Step 5: TRT 配置文件（T6）
        → 依赖 T2 + T3

Step 6: Step B 精度验收 + 验证脚本（T7）
        → 依赖 Step 4 ONNX 正确生成
```

---

## 五、预期遇到的 Bug 及应对方案

### Bug 1：`AgentQueryFusion` 代价矩阵计算含 softmax / topk

**现象**：代价矩阵构造中用到 `F.softmax` 或 `torch.topk`，这些在静态图中可以导出，
但如果使用了动态 K 值则不行。

**应对**：将代价矩阵计算整体移至图外（与匈牙利匹配同层），只把融合 MLP 导入 ONNX。

### Bug 2：Gather 操作中 `matched_veh_idx` 是 INT64

**现象**：TRT 对 Gather 节点的索引要求 INT32，若 ONNX 中索引是 INT64 则构建 engine 失败。

**应对**：在导出脚本中将匹配索引转为 `torch.int32`，并在 `_patch_and_verify_onnx()` 中
扫描所有 Gather 节点的索引输入，INT64 常量一律转 INT32。

### Bug 3：`LaneQueryFusionTRT` 中 `bmm` 广播维度不匹配

**现象**：`ego2other_r` 是 `(3,3)` 而 lane_query 是 `(1, M, C)`，
直接 bmm 时维度不对齐。

**应对**：将 `ego2other_r` unsqueeze 后 expand，或用 `einsum` 明确指定维度。

### Bug 4：`_get_coop_bev_embed()` 含 Python int 索引

**现象**：BEV 特征的空间更新逻辑如下，不可 ONNX 追踪：
```python
w = int(locs[idx, 0]);  h = int(locs[idx, 1])
bev_embed[h * self.bev_w + w, ...] += feature
```

**应对**：将该逻辑用 `scatter_add` 向量化替换，或完全移至 ONNX 图外，
以 BEV embed 更新作为一个独立的 CPU/CUDA 步骤（推荐后者，更简单）。

### Bug 5：infra 侧 ONNX 与 ego 侧接口不匹配

**现象**：infra 模型无 `command` 输入，`sdc_traj` 输出不存在，
直接拼合时 V2XWrapper 输入列表对不上。

**应对**：infra 引擎保持独立（已有 `univ2x_infra_downstream.trt`），
V2XWrapper 只封装 ego 侧的 V2X 融合部分，infra 引擎输出通过 Python/C++ 传入。

---

## 六、文件变更清单

| 文件路径 | 操作 | 核心内容 |
|----------|------|----------|
| `projects/mmdet3d_plugin/univ2x/fusion_modules/lane_fusion.py` | 末尾追加 | `LaneQueryFusionTRT`（向量化坐标变换，无 Python for 循环） |
| `projects/mmdet3d_plugin/univ2x/fusion_modules/agent_fusion.py` | 末尾追加 | `AgentQueryFusionTRT`（MLP 融合部分，匹配索引作为输入） |
| `projects/mmdet3d_plugin/univ2x/fusion_modules/__init__.py` | 修改 | 导出新增 TRT 类 |
| `projects/mmdet3d_plugin/univ2x/detectors/univ2x_e2e.py` | 末尾追加 | `UniV2XTRT.forward_v2x_trt()` |
| `projects/mmdet3d_plugin/univ2x/detectors/__init__.py` | 修改 | 导出 `UniV2XTRT` |
| `projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p3.py` | 新建 | Phase 3 TRT 配置 |
| `tools/export_onnx_univ2x.py` | 修改 | 新增 `--v2x` 模式、`V2XWrapper`、匹配索引 dummy 输入 |
| `tools/validate_v2x_trt.py` | 新建 | V2X TRT engine 精度验证脚本（Step A/B 两阶段） |

---

## 七、验收标准

| 验收项 | Step | 通过标准 |
|--------|------|----------|
| `LaneQueryFusionTRT` 前向（随机权重） | — | 无 Python for 循环，输出 shape 与原版一致 |
| `AgentQueryFusionTRT` 前向（固定匹配索引） | — | MLP 融合输出 shape 正确，无 scipy 调用 |
| ONNX 图结构（ego V2X） | **A** | 含 MSDAPlugin / InversePlugin，无 ATen 算子，无动态控制流 |
| TRT engine 编译（ego V2X） | **A** | 无不支持层，engine 生成成功 |
| TRT FP32 精度（ego V2X） | **B** | fused_agent/lane_query cosine > 0.999，occ_logits mean diff < 5e-3 |
| V2X 协同 Planning 精度 | **B**（最终验收）| TRT FP32 vs PyTorch：Planning Col 差异 < 0.005 |

---

## 八、后续 Phase 4 展望

Phase 3 完成后，下一步（plan.md Phase 4）为 **C++ 双引擎推理应用**：

- 基于 `uniad-trt/inference_app/enqueueV3/` 框架改造
- 双 CUDA stream 并行：infra 引擎推理 ‖ ego 侧图像预处理
- 集成 CPU 侧 `lapjv` 匈牙利匹配（替换 Python 验证脚本中的 scipy）
- 实现 V2X 坐标变换 CUDA kernel（`pre_process.cu` 扩充）
- 帧间状态管理（KernelInput / KernelOutput 结构体扩充 V2X 字段）
- 端到端系统测试：TRT C++ 推理结果 vs PyTorch 基准

Phase 5（可选）：ModelOpt INT8/FP16 量化，目标延迟 35~55 ms/帧（ego 侧）。
