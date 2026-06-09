# V2X-ViT ONNX → TRT 风险预研报告 (hw-optimizer, v1)

> 编制: 2026-06-04 · hw-optimizer (新生实例)
> 状态: **纯读研究, 不动 GPU** (待 sw 交付工件 + team-lead 点头后执行 A-3)
> 对应任务: Task#5 (V2X-ViT 实验 A-3 部分)
> 授权状态: **A-3 尚未获 team-lead 授权, 本文仅为 pre-flight 分析**

---

## 〇、信息来源与核验说明

本报告所有结论来自**静态代码阅读 + 已有实测文档**, 无任何 GPU 操作:
- `HEAL/opencood/models/sub_modules/v2xvit_basic.py` — 主干架构 (直接读)
- `HEAL/opencood/models/sub_modules/hmsa.py` — HGT attention (直接读)
- `HEAL/opencood/models/sub_modules/mswin.py` — Pyramid Window Attention (直接读)
- `HEAL/opencood/models/sub_modules/base_transformer.py` — PreNorm / CavAttention / FFN (直接读)
- `HEAL/opencood/models/sub_modules/split_attn.py` — SplitAttn (直接读)
- `HEAL/opencood/models/fuse_modules/fusion_in_one.py` — V2XViTFusion wrapper (直接读)
- `HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_v2xvit.yaml` — 超参 (直接读)
- `model/v2xvit/分段耗时实测_v1.md` — 分段 latency 基线 (项目已有实测)
- `HEAL/checkpoints/baselines_hf/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/` — ckpt 确认存在, 参数计数通过 CPU 加载验证

> **[2026-06-04 sw-optimizer 原文核验完成 — ISS-022/028 归档]**
> 原项目文档 "75.1→29.9" 是**跨模型混淆**，已被 sw 回查 arXiv:2509.03704 Table 1 (DAIR-V2X, PTQ) 更正：
>
> | 模型 | 量化档 | AP30 FP32 | AP30 量化后 | 跌幅 |
> |---|---|---|---|---|
> | **V2X-ViT** | **INT8/INT8** | **57.4** | **40.0** | **-17.4 pt (-30%)** |
> | V2X-ViT | INT4/INT8 | 57.4 | 29.9 | -27.5 pt (-48%) |
> | Pyramid Fusion | INT8/INT8 | 75.1 | 74.6 | -0.5 pt (近免损) |
>
> **旧引用错误来源**: 75.1 = Pyramid FP32，29.9 = V2X-ViT INT4/INT8，跨模型行混读。
> **正确论文引用**: "QuantV2X PTQ (arXiv:2509.03704) 表明，V2X-ViT INT8/INT8 量化导致 AP30
> 57.4→40.0 (-30%)，Pyramid Fusion 仅 -0.5 pt；INT4/INT8 更激进达 57.4→29.9 (-48%)。"
> **R3 结论方向不变**: INT8 transformer 仍高风险，-30% 已足够支持 mixed-precision 方案；
> -48% 进一步说明 INT4 完全不可用。

---

## 一、模型架构速查

### 1.1 整体结构 (DAIR-V2X, LiDAR-only)

```
输入: 2 agents (固定 L=2, DAIR V2I: 1车+1路侧)
  ↓
encoder_m1 (PointPillar VFE + Scatter)     — 2.90 ms FP32 [真测]
  ↓
backbone_m1 (BaseBEVBackbone 3-5-8层)      — 2.99 ms FP32 [真测]
  ↓
shrinker_m1 (stride=2 DownsampleConv)      — 0.89 ms FP32 [真测]
  → 输出: (sum_cav, 256, 50, 176), 即 (2, 256, 50, 176)
  ↓
V2XViTFusion.forward()                     — 27.39 ms mean / 24.02 ms p50 [真测 FP32]
  → Regroup: (2, 256, 50, 176) → (B=1, L=2, 256, 50, 176)
  → permute → (B=1, L=2, H=50, W=176, C=256) + prior_encoding(3) → C=259
  → V2XTransformer.forward(x, mask, spatial_correction_matrix)
      → STTF: warp_affine(grid_sample) + mask 计算
      → 3× V2XFusionBlock:
          → HGTCavAttention (HMSA, 8 heads, dim_head=32)
          → PyramidWindowAttention (MSwin, 3 scales: ws=[4,8,16])
          → FeedForward (Linear-GELU-Linear)
  → 输出 (B=1, H=50, W=176, C=256) [ego agent]
  ↓
heads (cls+reg+dir, tiny)                  — 0.06 ms [真测]
  ↓
NMS (Shapely Python, CPU)                  — 23.75 ms [真测]
```

### 1.2 关键参数

| 项目 | 数值 | 来源 |
|---|---|---|
| 全模型参数 | **13.46M** | CPU 加载真算 |
| V2XTransformer 参数 | **5.39M** | CPU 加载真算 |
| depth (encoder 层数) | 3 | yaml |
| HMSA heads / dim_head | 8 / 32 | yaml |
| MSwin window_size | [4, 8, 16] | yaml |
| MSwin heads | [16, 8, 4] | yaml |
| fuse_method | split_attn | yaml |
| use_RTE | **false** | yaml (RTE 路径不激活) |
| use_hetero | **true** | yaml → HGTCavAttention 生效 |
| BEV feature size | 50 × 176 | stride-2 shrinker |
| L (agent 数) | **2 (DAIR 固定)** | 实测文档 |
| types (DAIR 固定) | **[0=vehicle, 1=infra]** | DAIR V2I 协议 |
| **AP70 基线** | **0.521** | sw A-0 核验 (epoch17 eval yaml; 口径: DAIR-V2X val) |
| **ckpt 格式** | **flat state_dict (ISS-005 安全)** | sw A-0 核验 |

---

## 二、ONNX Export 风险清单 (按严重性排序)

### 🔴 风险 R1: 数据依赖的 Python 动态 dispatch — `HGTCavAttention`  [CRITICAL]

**代码位置**: `hmsa.py::HGTCavAttention.to_qkv() / to_out()`

```python
# to_qkv: 每个 agent 按 types[b,i] 选不同的 Linear
for b in range(x.shape[0]):          # batch 维
    for i in range(x.shape[-2]):     # agent 维
        q_list.append(
            self.q_linears[types[b, i]](x[b,:,:,i,:].unsqueeze(2)))
        # types[b, i] 是运行时 tensor 值 → Python list 索引
```

**风险详述**: `types[b, i]` 是从 `prior_encoding` 中读出的 INT tensor 值，用来从 Python ModuleList
`self.q_linears[0]` / `self.q_linears[1]` 中选不同的 `nn.Linear`。这是**数据依赖的控制流**
(data-dependent control flow), `torch.onnx.export()` 在 trace 模式下会按实际执行路径固化:
- **DAIR 推理时 types 是常量 `[0, 1]`** (vehicle=0, infra=1, 固定) → trace 路径固定 → 可导出
- 但必须保证 export 时的 dummy input 的 `types` 与实际推理完全一致 (否则 trace 路径错误)
- 若 `types` 在不同帧变化 (同质性 DAIR 不会), 则 ONNX 会静默走错路径

**缓解方案**:
1. Export 时固定 `prior_encoding` 中的 types 通道为 `[0, 1]` 常量
2. 在 ONNX export wrapper 中预先展开: 避免 Python for loop, 改写为 `torch.where` + 矩阵运算
3. 验证: export 后 ONNX runtime 输出 vs PyTorch 输出最大绝对误差 < 1e-4

---

### 🔴 风险 R2: Python for 循环 — `HGTCavAttention` 展开图爆炸风险  [CRITICAL]

**代码位置**: `hmsa.py` 多处

```python
# to_qkv: B × L 次循环
# get_hetero_edge_weights: B × L × L 次循环 (L=2 → 4次; L>5 → 25次+)
# to_out: B × L 次循环
```

**风险详述**: `torch.onnx.export(trace)` 会将 Python 循环**展开**成 ONNX 节点。
- DAIR L=2, 推理 B=1: 展开量小 (最多 4×4=16 iter), ONNX 图可接受
- 若 B>1 或 L>2: 图节点数线性增长, TRT build 时间和 engine size 倍增
- `RTE` 路径 (`use_RTE=false`): **DAIR yaml 关闭, 不激活**, 无需考虑

**缓解方案**:
1. 固定 B=1, L=2 export (DAIR 推理 batch=1)
2. 展开后验证节点数: `onnx.shape_inference + netron 可视化` 确认无异常重复
3. 可选: 重写 `to_qkv/to_out` 为矩阵化实现 (B=1, L=2 情况下可手动展开+stack)

---

### 🔴 风险 R3: INT8 量化 precision collapse — attention + LayerNorm  [CRITICAL for INT8]

**代码位置**: `hmsa.py::att_map.masked_fill(-inf)` + `base_transformer.py::PreNorm(LayerNorm)`

**风险详述**:
1. **`masked_fill(-inf)`**: 无效 agent 的 attention score 填 -inf → softmax → 0。INT8 量化
   activation range 时, -inf 会导致 per-tensor MinMax calibration 失效 (range 变 inf → 所有
   activation 缩放到 0)。需使用 `EntropyCalibrator2` 或对 masked 区域截断处理。
2. **`LayerNorm` 在 INT8**: 标准化操作 (mean/var) 在 INT8 精度下损失严重。TRT 对 LayerNorm
   有 FP16 fallback, 但 forced-INT8 build 时会导致精度崩溃。
3. **多头 attention QK 乘积 INT8**: Q×K^T 的精度对 attention distribution 极敏感,
   INT8 会引入 outlier magnification。这是 QuantV2X (arXiv:2509.03704) 实证的根因:
   **V2X-ViT INT8/INT8 PTQ: AP30 57.4→40.0 (-30%)**；更激进的 INT4/INT8: 57.4→29.9 (-48%)。
   对比 Pyramid Fusion INT8/INT8 仅 -0.5 pt，差异说明 transformer attention 对量化极敏感。
   (旧引用 "75.1→29.9" 是跨模型混淆，已由 sw 原文核验更正，详见 §〇。)

**缓解方案 (mixed-precision INT8)**:
1. LayerNorm 层保 FP16 (TRT `setLayerPrecision` 或 per-layer 精度覆盖)
2. Q/K projection (`q_linears`, `k_linears`) 及 attention score 计算保 FP16
3. V projection + FFN Linear 可量化 INT8 (这部分是主要计算量)
4. 使用 `--fp16` baseline 作为 INT8 parent engine (TRT 的 mixed precision 路径)
5. 校准数据: DAIR val 前 100 帧的 fusion_net 输入 `(1, 2, 50, 176, 259)` 张量

---

### 🟡 风险 R4: 多操作数 `torch.einsum` 与非标准下标  [MEDIUM]

**代码位置**: `hmsa.py::HGTCavAttention.forward()`

```python
# 3操作数 einsum, 7维输出
att_map = torch.einsum(
    'b m h w i p, b m i j p q, bm h w j q -> b m h w i j',
    [q, w_att, k]) * self.scale

v_msg = torch.einsum(
    'b m i j p c, b m h w j p -> b m h w i j c',
    w_msg, v)
```

**风险详述**:
- TRT 10.x ONNX 解析时会将多操作数 einsum 展开为 `MatMul + Transpose + Reshape` 链
- `bm` (无空格) 下标在第一个 einsum 中 — 需确认 ONNX opset 支持此语法 (通常需 opset ≥ 12)
- `w_att` 是 learnable parameter (`relation_att` [4,8,32,32]), 在 export 时会被 fold 为常量
- 展开后可能产生多余 Reshape 导致 TRT tactic 选择次优

**缓解方案**:
1. `torch.onnx.export(opset_version=17)` (最新稳定 opset)
2. ONNX 导出后用 `onnx-simplifier` 消除冗余 Reshape/Transpose
3. 若 TRT 报 unsupported node, 在 einsum 前加 `torch.onnx.export` 的 `custom_opsets` 或手动
   替换为等价 bmm 链

---

### 🟡 风险 R5: `F.grid_sample` / `F.affine_grid` + `use_roi_mask=true` 路径  [MEDIUM - 有简化方案]

**代码位置**: `torch_transformation_utils.py::warp_affine()` + `get_roi_and_cav_mask()`

```python
# V2XTEncoder.forward() — use_roi_mask=true 路径 (DAIR yaml)
com_mask = get_roi_and_cav_mask(x.shape, mask, spatial_correction_matrix,
                                self.discrete_ratio, self.downsample_rate)

# get_roi_and_cav_mask 调用链:
# get_discretized_transformation_matrix → get_transformation_matrix
# → get_rotated_roi(warp_affine(all-ones, T, nearest)) → combine_roi_and_cav_mask
```

**关键简化发现 (sw A-0 + hw 独立核验)**:
`V2XViTFusion.forward()` 硬编码 `spatial_correction_matrix = torch.eye(4).expand(B, L, 4, 4)`
(identity 变换)。经调用链追踪:
- `get_discretized_transformation_matrix(I)` → 2D 平移 = 0 / 旋转 = 单位阵
- `get_rotated_roi(warp_affine(ones, T_identity))` → **全 1 mask** (没有旋转/平移偏移)
- `combine_roi_and_cav_mask` → `com_mask = cav_mask`(全 1, 2 agents 均有效)

⇒ **DAIR 推理时 `com_mask` 是常量全 1 张量** `(B=1, H=50, W=176, 1, L=2)` = `(1,50,176,1,2)`。
(shape 推导: `combine_roi_and_cav_mask` 输出 `(B,L,C,H,W)` → `permute(0,3,4,2,1)` → `(B,H,W,C,L)` = `(B,H,W,1,L)`; sw-optimizer 独立核验 PASS)

**原始风险**:
- `use_roi_mask=true` 路径含 `warp_affine` (affine_grid + grid_sample) + `torch.inverse` in STTF
- ONNX opset 16+ 支持 `grid_sample`; 但 `torch.inverse` 和矩阵变换链会产生多个 MatMul 节点
- `src.half() if grid.dtype==torch.half else src` — trace 固化一条路径

**★ 额外发现: STTF 也是 no-op (sw-optimizer 独立核验 PASS)**:

```python
# v2xvit_basic.py::STTF.forward():
# spatial_correction_matrix = I (identity, V2XViTFusion 硬编码)
# → dist_correction_matrix = identity 2D affine [[1,0,0],[0,1,0]]
# → T = get_transformation_matrix(identity) = identity affine
# → warp_affine(cav_features[:, 1:, ...], identity) = cav_features[:, 1:, ...] 原样
# ⇒ STTF 整块输出 x 与输入 x 完全相同 (所有 agent features 不变)
```

**STTF + com_mask 合计可消除约 30–50 个 ONNX 中间算子节点**
(STTF: ~25 节点 + get_roi_and_cav_mask: ~15 节点)

**缓解方案 (三档可选)**:
1. **推荐: 完整绕过方案** — 修改版 export wrapper 直接跳过 STTF 和 get_roi_and_cav_mask,
   注入预计算常量:
   - `com_mask = torch.ones(1, 50, 176, 1, 2)` (`(B,H,W,1,L)` ← 正确 shape)
   - STTF 输出 = 输入 x 原样 (identity warp)
   - 省约 30–50 ONNX 节点, 总图规模从 ~1,000 降至 ~950 以内
2. **完整 trace** — 不修改模型, 以 FP32 mode export, ONNX simplifier 可能自动 fold 常量;
   验证 STTF 输出数值与 PyTorch 一致 (identity 变换 maxdiff ≈ 0)
3. **hybrid**: trace 后用 onnx-simplifier + constant folding 自动消除, 效果接近方案 1

---

### 🟡 风险 R6: `torch.tensor_split` in `Regroup()`  [MEDIUM - wrapper 层]

**代码位置**: `fusion_in_one.py::V2XViTFusion.forward()` → `fuse_utils.Regroup()`

```python
def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x
```

**风险详述**:
- `torch.tensor_split` 的分割点是 `cum_sum_len[:-1].cpu()` (动态值) → 数据依赖 split
- ONNX export 可能将此展开或报错
- **缓解思路**: V2XViTFusion wrapper 不需要 export; 只 export `V2XTransformer.forward()`
  (即 Regroup 之后的部分, 输入已是 `(B, L, H, W, C)`)
- 这是 **export scope 的关键设计决策** (见 §三)

---

### 🟢 风险 R7: `einops.rearrange` + `nn.Dropout`  [LOW - 无风险]

- `einops.rearrange` → 标准 ONNX Reshape/Transpose, TRT 完全支持
- `nn.Dropout(p=0.3)` — 推理模式 (`model.eval()`) 下等价 Identity, ONNX export 时自动消除
- `nn.GELU` — TRT 10.x 原生支持
- `nn.LayerNorm` — TRT FP16 模式下支持; INT8 下需 FP16 fallback (见 R3)

---

### 🟢 风险 R8: Sparse op (encoder_m1) 不在 export scope  [不适用]

- PointPillar VFE 含 spconv 稀疏算子 → 不可 ONNX export
- **export scope = `V2XTransformer` 子模块**, 不含 encoder_m1 / backbone_m1
- 这与 Pyramid 的 `pyramid_backbone` subnet export 策略一致 (CLAUDE.md Phase A.1)

---

## 三、ONNX Export 策略与 Export Scope 设计

### 3.0 ⚠️ Types 常量假设的数据集范围限制 [team-lead 补充, 2026-06-04]

本方案能够工作的前提: **DAIR-V2X 是固定 V2I 形态** (1 辆车 + 1 路侧, 永远 `types=[0,1]`)。
这使得 `HGTCavAttention` 中的 data-dependent dispatch 在推理时是**常量**。

若未来实验扩展到以下场景, 该假设**失效**, 需重新评估 export 策略:
- **V2V (Vehicle-to-Vehicle) 数据集**: 所有 agent types = 0 (全车辆), 但 types 值可能变
- **OPV2V (3+ agents)**: L > 2, for 循环展开量更大, 图规模 ×L²
- **混合异构多 agent**: types 可能在帧间变化 (不同 vehicle 类型)

**此 caveat 必须在论文 §通用性 / limitations 段落中明确声明**: A-3 测量结论仅对 DAIR V2I (L=2, types 固定) 有效; 通用多 agent 部署需改写 `to_qkv` 为向量化实现。

---

### 3.1 推荐 Export Unit

**只 export `V2XTransformer`** (即 `V2XViTFusion.fusion_net`), 不含 Regroup/V2XViTFusion wrapper:

```python
# 推荐 export wrapper
class V2XTransformerExportWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model

    def forward(self, x, mask, spatial_correction_matrix):
        # x: (B=1, L=2, H=50, W=176, C=259) = feature + prior_encoding
        # mask: (B=1, L=2) — 原始 agent 有效性 mask (V2XTEncoder 内部计算 com_mask)
        #   若使用简化方案(§R5 缓解1): 在修改版 wrapper 中直接注入 com_mask=(B,H,W,1,L)=(1,50,176,1,2)
        # spatial_correction_matrix: (B=1, L=2, 4, 4)
        output = self.model(x, mask, spatial_correction_matrix)
        return output  # (B=1, H=50, W=176, C=256)
```

**固定 dummy input**:
```python
dummy_x    = torch.zeros(1, 2, 50, 176, 259)  # (B, L, H, W, C+3)
dummy_mask = torch.ones(1, 2)                  # (B, L) — agent 有效性 mask
dummy_scm  = torch.eye(4).unsqueeze(0).expand(1, 2, 4, 4)  # identity (STTF no-op)
```

> ⚠️ `prior_encoding` 的最后 1 维是 `types` (vehicle=0 / infra=1)。
> 必须设置 `dummy_x[..., -1] = torch.tensor([[[[[0.0]]]], [[[[1.0]]]]])`
> 保证 HGTCavAttention 的 dispatch 路径与 DAIR 实际推理一致。

### 3.2 Export 命令草案

```python
# 环境: conda env UniV2X_2.0, 在 HEAL repo root
import sys
sys.path.insert(0, '/home/jichengzhi/heal_research/HEAL')
import torch
from opencood.models.sub_modules.v2xvit_basic import V2XTransformer
import yaml, torch.onnx

cfg = yaml.safe_load(open('opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_v2xvit.yaml'))
vit_args = cfg['model']['args']['v2xvit']['transformer']['encoder']
model = V2XTransformer({'encoder': vit_args}).eval().cuda()

# 加载 fusion_net 权重 (从全模型 ckpt 中提取)
# ★ ISS-024: 必须用 bestval_at17, 不可换 epoch
ckpt_path = 'checkpoints/baselines_hf/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26/net_epoch_bestval_at17.pth'
full_ckpt = torch.load(ckpt_path, map_location='cpu')
# flat state_dict (ISS-005 安全格式, sw A-0 确认)
fusion_sd = {k.replace('fusion_net.fusion_net.', ''): v
             for k, v in full_ckpt.items() if k.startswith('fusion_net.fusion_net.')}
model.load_state_dict(fusion_sd, strict=True)  # strict=True 验证完整

# dummy input (types=[0,1] 固化为 DAIR 常量)
B, L, H, W, C = 1, 2, 50, 176, 259
dummy_x = torch.zeros(B, L, H, W, C).cuda()
dummy_x[0, 0, :, :, -1] = 0.0   # vehicle type=0
dummy_x[0, 1, :, :, -1] = 1.0   # infra type=1
# 简化方案: 预计算常量 com_mask + STTF no-op (R5 §三 缓解方案1 + sw独立核验PASS)
# com_mask shape = (B, H, W, 1, L) ← sw-optimizer 核验更正(旧错误: (B,H,W,L,1))
# permute 推导: combine_roi_cav_mask→(B,L,C,H,W)→permute(0,3,4,2,1)→(B,H,W,C,L)=(B,H,W,1,L)
dummy_com_mask = torch.ones(B, H, W, 1, L).cuda()   # (1,50,176,1,2) ★ 注意最后两维
dummy_scm  = torch.eye(4).unsqueeze(0).expand(B, L, 4, 4).cuda()
# STTF no-op: SCM=I → warp_affine(x[:,1:,...], identity) = x[:,1:,...] 原样; STTF 整块可跳过
# 优化 export: 修改版 wrapper 可完全绕过 STTF + get_roi_and_cav_mask, 省约 30-50 算子节点

torch.onnx.export(
    model,
    (dummy_x, dummy_mask, dummy_scm),
    'models/v2xvit_fusion_fp32.onnx',
    input_names=['x', 'mask', 'spatial_correction_matrix'],
    output_names=['fused_feature'],
    opset_version=17,
    do_constant_folding=True,
    verbose=False
)
# 验证: onnxruntime inference 数值与 PyTorch 对比 (maxdiff < 1e-3)
```

> ⚠️ 实际 V2XTEncoder.forward 的 `com_mask` 需特别处理 (use_roi_mask 控制)。
> 建议先测 PyTorch trace 能否通过, 再调整 mask 维度。

---

## 四、A-3 测量计划草案 (TRT FP16/INT8 build + latency/energy)

> **授权前提**: A-3 必须等 sw-optimizer 交付 A-0/A-1 工件 + team-lead 显式授权后执行。
> 以下为计划草案, **当前不动 GPU**。

### 4.1 前置条件核查

执行前必须确认:
- [ ] `nvidia-smi`: 目标 GPU util=0% / mem ≤ 50MiB (宪章 §2 MUST-4)
- [ ] 使用 GPU 编号明确指定 `CUDA_VISIBLE_DEVICES=<空闲卡>`
- [ ] GPU clock 锁频: `sudo nvidia-smi -lgc 2520` (4090 base clock)
- [ ] V2X-ViT fusion_net ONNX 已 export 且数值验证通过 (maxdiff < 1e-3)
- [ ] **ISS-024 epoch locking (sw A-0 强调)**: 必须使用 `net_epoch_bestval_at17.pth`，
  不可用其他 epoch checkpoint。A-3 build 的 TRT engine 与 A-1 AP eval 必须同一 epoch，
  保证 hybrid eval (TRT latency + PyTorch AP) 的 ckpt 一致性。

### 4.2 TRT FP16 Build

```bash
# 步骤 1: 检查 ONNX
python3 -c "import onnx; m=onnx.load('models/v2xvit_fusion_fp32.onnx'); onnx.checker.check_model(m); print('OK')"

# 步骤 2: TRT FP16 engine build (4090 TRT 10.x)
CUDA_VISIBLE_DEVICES=<空闲卡> \
trtexec --onnx=models/v2xvit_fusion_fp32.onnx \
        --saveEngine=models/v2xvit_fusion_fp16.engine \
        --fp16 \
        --memPoolSize=workspace:4096MiB \
        --avgRuns=200 --warmUp=200 --duration=10 \
        --exportLayerInfo=results/v2xvit_fp16_layer_profile.json \
        2>&1 | tee results/v2xvit_trt_fp16_build.log

# 记录: engine size (MB), mean/p50/p99 latency
```

### 4.3 校准数据准备 (INT8 前置)

```python
# 从 DAIR val 前 100 帧采集 V2XTransformer 输入
# 需修改 HEAL inference pipeline, 在 fusion_net 前 hook 采集 x, mask, scm
# 输出: calibration/v2xvit_calib_inputs.bin (100帧 × 3 tensor)
# 校准器: IInt8MinMaxCalibrator 或 EntropyCalibrator2 (推荐 Entropy, 避免 -inf 污染)
```

### 4.4 TRT INT8 Build (Mixed Precision 方案)

```bash
# 方案 A: TRT-auto INT8 (全局, 高风险)
CUDA_VISIBLE_DEVICES=<空闲卡> \
trtexec --onnx=models/v2xvit_fusion_fp32.onnx \
        --saveEngine=models/v2xvit_fusion_int8_auto.engine \
        --int8 --fp16 \
        --calib=calibration/v2xvit_calib.bin \
        --memPoolSize=workspace:4096MiB \
        --avgRuns=200 --warmUp=200 --duration=10 \
        2>&1 | tee results/v2xvit_trt_int8_auto_build.log

# 方案 B: 手工 per-layer 混精 (如 A 精度崩溃)
# 通过 Python TRT API: engine inspector 定位 LayerNorm + QK attention 层, 强制 FP16
# 参考: scripts/phase1/m4_8_trt_build_bench.py 的 per-layer precision 覆盖逻辑
```

### 4.5 Latency 测量口径 (collab2 类比)

V2X-ViT 无直接等价的 `body_subnet_collab2` 口径 (Pyramid 双模型协同), 定义新口径:

| 口径名 | 测量范围 | 类比 | 备注 |
|---|---|---|---|
| `v2xvit_fusion_subnet` | `V2XTransformer.forward()` alone | Pyramid `body_subnet_collab2` 中的 fusion 部分 | ONNX export 的 scope |
| `v2xvit_body` | encoder×2 + backbone×2 + shrinker×2 + fusion | Pyramid `body_subnet_collab2` 全程 | 需 PyTorch CUDA-Event 补测 |
| `v2xvit_e2e` | 含 NMS | Pyramid `e2e` | NMS 需 CUDA NMS 替换后再测 |

> 论文/数据集收录时使用 `v2xvit_fusion_subnet` 作主要 TRT latency 口径,
> 并明确注释与 Pyramid `body_subnet_collab2` 的**不可直接横比**关系。

### 4.6 Energy 测量方案

参照 `results/E4_energy_4090.csv` 口径:

```bash
# 在 GPU 推理循环运行时 (200 warmup + 200 measure):
nvidia-smi --query-gpu=power.draw \
           --format=csv,noheader,nounits \
           -l 1 > results/v2xvit_power_fp16.txt
# J/frame = mean_power(W) × lat_p50(s)
# perf/watt = 1/lat_p50 / mean_power
```

### 4.7 预期结果参考 (估算, 非真测)

| 精度档 | latency 估算 | vs FP32 baseline (27.4ms) | 备注 |
|---|---|---|---|
| PyTorch FP32 | 27.39 ms (p50=24.02) | 1.00× | 真测 baseline |
| TRT FP16 | **~9–14 ms** (估算) | **~2–3×** | 估算; 真测待执行 |
| TRT INT8 auto | **~6–10 ms** (估算) | **~3–4×** | **AP 崩溃风险高**; 真测待执行 |
| TRT INT8 mixed | **~8–12 ms** (估算) | **~2.5–3×** | LayerNorm/QK FP16; 真测待执行 |

> ⚠️ 以上估算基于 Pyramid 同等规模 transformer 加速比推断, **非真测, 必须标"估算"**。
> 真测后替换此表。

---

## 五、风险优先级汇总

| 风险 | 严重度 | 影响阶段 | 缓解难度 | 优先处理 |
|---|---|---|---|---|
| R1: data-dependent dispatch (types) | 🔴 CRITICAL | ONNX export | 中 (DAIR 常量可 fix) | ★ 首解 |
| R2: Python for 循环展开 | 🔴 CRITICAL | ONNX export | 低 (L=2 可接受) | ★ 同步处理 |
| R3: INT8 precision collapse | 🔴 CRITICAL | INT8 engine | 高 (需 mixed-prec) | ★ 决定 INT8 价值 |
| R4: 多操作数 einsum | 🟡 MEDIUM | ONNX/TRT build | 低 (simplifier 可处理) | 观察 |
| R5: grid_sample 静态 shape | 🟡 MEDIUM | ONNX export | 低 (H/W 固定) | 验证即可 |
| R6: tensor_split (Regroup) | 🟡 MEDIUM | export scope | 低 (不 export wrapper) | 设计规避 |
| R7: einops/Dropout/GELU | 🟢 LOW | 无 | 无 | 无需处理 |
| R8: sparse encoder op | 🟢 无风险 | export scope | — | 设计规避 |

---

## 六、与 Pyramid TRT 经验对比 (硬件维度视角)

| 维度 | Pyramid (已完成) | V2X-ViT (待做) |
|---|---|---|
| 主要算子 | Dense Conv (grouped conv) | Transformer (attention + einsum) |
| INT8 风险 | kernel-cliff (通道对齐, ISS-014) | precision collapse (QK + LayerNorm) |
| DLA 适配 | FP16 部分成功 (8/12); INT8 全失败 (0/12) | **未测; 推测 DLA attention support 更差** |
| export scope | `pyramid_backbone` subnet | `V2XTransformer` fusion subnet |
| 动态 shape | 无 (H/W/C 固定) | 同样固定 (L=2 DAIR) |
| ONNX 特殊算子 | 无 | einsum + dynamic dispatch (data-dep) |

---

## 七、产出说明

- **本文件**: `multi_agent/model/v2xvit_trt_risk_v1.md` (预研报告, 纯读产物; ★[2026-06-05 迁移] 原在 methods/design/)
- **尚无**: 任何 ONNX 文件 / engine 文件 / latency CSV (未授权执行 A-3)
- **下一步**: team-lead 授权 A-3 → 执行 §三 export + §四 TRT build → 结果落 `results/A3_v2xvit_trt_*.csv`

---

---

## 八、R2 循环展开 ONNX 节点数量级估算 [team-lead 补充任务, 纯静态分析]

> **目的**: 为 A-3 TRT build 时间预算提供参考。B=1, L=2 (DAIR 固定)。
> **方法**: 逐模块手工追踪 Python for 循环展开后的 ONNX 节点数 (MatMul/Add/Reshape/Transpose/
> Gather/Cat 等), 保守估算 × 1.5 作为 ONNX runtime 实际节点数 (含 ONNX 自动插入的 Cast/Shape 节点)。

### 8.1 HGTCavAttention (×3 encoder 层)

| 循环 | 展开迭代数 (B=1, L=2) | 主要 ONNX 节点 |
|---|---|---|
| `to_qkv`: for b×i | 2 次 | 2 agent × (q+k+v Linear各2节点) = 12 MatMul/Add + 多个 Unsqueeze/Cat |
| `get_hetero_edge_weights`: for b×i×j | 4 次 | 4×(w_att+w_msg) Gather + Cat = ~16 Gather + ~8 Cat |
| `to_out`: for b×i | 2 次 | 2 × (a_linear: 2节点 + Cat) = ~6 节点 |
| einsum×3 (att_map/v_msg/out) | — | 每个 ~4–6 MatMul/Transpose = ~15 节点 |
| rearrange×3 (q,k,v) | — | ~9 Reshape+Transpose |
| masked_fill(-inf) + softmax | — | ~4 节点 |
| 其他 (permute/LayerNorm/残差) | — | ~8 节点 |
| **HGTCavAttention 小计** | | **~78–90 节点** |

### 8.2 PyramidWindowAttention (×3 encoder 层, 含 3 个 window scale)

| 子模块 | ONNX 节点 |
|---|---|
| 每个 BaseWindowAttention (×3 scale): to_qkv + chunk + 2×rearrange + 2×einsum + pos_embed + softmax + to_out | ~22–28 节点/scale |
| 3 scale 合计 | ~66–84 节点 |
| SplitAttn (fc1+LayerNorm+ReLU+fc2+RadixSoftmax+加权) | ~18–22 节点 |
| **PyramidWindowAttention 小计** | **~84–106 节点** |

### 8.3 FeedForward + PreNorm + 残差 (每层)

| 组件 | ONNX 节点 |
|---|---|
| PreNorm (LayerNorm + pass-through) | ~4 节点 |
| FeedForward (Linear + GELU + Dropout=Identity + Linear) | ~6 节点 |
| 残差加法 | ~2 节点 |
| **FeedForward+PreNorm 小计** | **~12 节点** |

### 8.4 每 encoder 层 (V2XFusionBlock + FFN)

```
PreNorm(HGTCavAttention) + residual + PreNorm(PyramidWindowAttention) + residual + FFN
≈ (4+80+2) + (4+95+2) + 12 = 约 199 节点/层
```

### 8.5 STTF + prior_feed (含 no-op 简化分析)

| 组件 | 完整 trace 节点 | 简化方案节点 | 备注 |
|---|---|---|---|
| get_discretized_transformation_matrix | ~8 | 0 | STTF no-op: 全部可跳过 |
| get_transformation_matrix + warp_affine(affine_grid+grid_sample) | ~18 | 0 | identity → 原样输出 |
| get_roi_and_cav_mask (com_mask 计算) | ~12 | 0 | 替换为常量 ones(1,50,176,1,2) |
| prior_feed Linear (259→256) | ~3 | ~3 | 必须保留 |
| **STTF 小计** | **~41 节点** | **~3 节点** | **简化省 ~38 节点** |

> **★ [sw-optimizer 独立核验 PASS]**: SCM=I → STTF 是 no-op (feature 原样), com_mask 是全1常量。
> 两者合计可消除约 **30–40 个有效算子节点** (加上 ONNX 自动插入约 ×1.5 → 约 45–60 实际节点)。

### 8.6 全 V2XTransformer 估算汇总 (B=1, L=2)

```
【完整 trace 方案】
STTF (~41) + prior_feed (~3) + 3层 × (~199节点/层) + 最终输出索引 (~2)
= 41 + 3 + 597 + 2 = 约 643 "功能节点"
加 ONNX 自动插入 (~1.5×) → 约 960–1,000 节点

【推荐: 绕过 STTF + com_mask 简化方案】
STTF简化 (~3) + prior_feed (~3) + 3层 × (~199节点/层) + 输出 (~2)
= 3 + 3 + 597 + 2 = 约 605 "功能节点"
加 ONNX 自动插入 (~1.5×) → 约 900–960 节点 (省 ~40–50 节点)
```

### 8.7 结论与 TRT Build 时间预算

| 对比对象 | 估算 ONNX 节点数 | 参考 TRT Build 时间 |
|---|---|---|
| Pyramid backbone (pyramid_backbone, conv-dense) | ~250–350 节点 | ~30–60 秒 (实测) |
| **V2XTransformer 完整 trace (B=1, L=2)** | **~960–1,000 节点** | **估算 ~3–8 分钟** |
| **V2XTransformer 简化方案 (绕过STTF+com_mask)** | **~900–960 节点** | **估算 ~3–7 分钟** |
| 大型 ViT (ViT-L/16) | ~1,500–3,000 节点 | ~10–30 分钟 (文献参考) |

**关键结论**:
1. **节点规模可接受**: ~900–1,000 节点对 TRT 10.x 属中等规模, build 不会超时
2. **STTF+com_mask 简化收益有限** (约减少 5% 节点), 主要收益是 ONNX 图更干净, grid_sample 算子消除
3. **INT8 calibration 会显著延长 build 时间** (100 次 forward pass 校准 + tactic 搜索): 预估 **15–30 分钟**
4. **Python 循环展开不是 build 瓶颈** (L=2 展开量小); 真正的 build 时间主要由 tactic 搜索决定
5. **如果 L 扩大到 5 (V2V OPV2V)**: 节点数 ~2,500–3,500 (L² 效应), build 时间可能 20–60 分钟

> **使用方式**: A-3 TRT FP16 build 时设 `--duration=10 --avgRuns=200` 足够;
> INT8 build 预留 30 分钟时间窗, 建议在空闲夜间提交。

---

*报告更新: 2026-06-04 (v1.4) — sw 核验整合: ① com_mask shape 订正 (1,50,176,2,1)→**(1,50,176,1,2)** (permute推导,sw PASS); ② STTF no-op 发现(SCM=I→warp_affine=identity→feature原样); 两者合计可省 ~30-50 ONNX 节点; 更新 R5 三档缓解方案 + §8.5 双方案节点估算。**预研报告定稿(v1.4, 待 A-3 授权)。***
*报告更新: 2026-06-04 (v1.3) — QuantV2X 数字更正(sw 原文核验): 旧"75.1→29.9"跨模型混淆→正确 V2X-ViT INT8/INT8 57.4→40.0(-30%) / INT4/INT8 57.4→29.9(-48%)，来源 arXiv:2509.03704 Table 1 DAIR-V2X PTQ；更正 §〇 + R3。*
*报告更新: 2026-06-04 (v1.2) — sw A-0 产出整合: ap70=0.521+flat ckpt(§1.2) + ISS-024 epoch locking(§4.1+export代码) + use_roi_mask=true→com_mask常量简化(R5+export) + QuantV2X核验注更新(§〇)*
*报告更新: 2026-06-04 (v1.1) — team-lead 三点补充已整合: types caveat(§3.0) + QuantV2X核验注(§〇) + R2 节点估算(§八)*
*报告初版: 2026-06-04 (v1.0) — hw-optimizer 新生实例*
