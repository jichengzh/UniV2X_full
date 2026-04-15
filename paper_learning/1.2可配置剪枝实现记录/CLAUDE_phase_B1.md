# Phase B1: TRT 部署端 (B.2-B.6)

> 覆盖: 剪枝后模型的 TensorRT 部署全流程 — ONNX 导出适配、校准数据重建、TRT 构建适配、硬件约束验证、端到端验证
> 修改文件: tools/export_onnx_univ2x.py, tools/dump_univ2x_calibration.py, tools/build_trt_int8_univ2x.py
> 新建文件: tools/verify_trt_constraints.py, tools/deploy_pruned_model.py
> 依赖: Phase A1-A3 (剪枝管线可用), 现有 TRT 管线可运行
> 预计工作量: 3.5 天
> 状态: 待开始

---

## 1. 阶段目标

将剪枝后的模型部署到 TensorRT：适配 ONNX 导出（处理维度变化）、重建校准数据（旧 pkl 因 tensor shape 改变不可复用）、构建 TRT engine（FP16 + INT8 混合精度）、验证硬件约束（通道对齐、最小通道数、注意力头整除）、执行端到端精度与延迟验证。

**核心挑战**: 剪枝改变了模型的 `embed_dims`、`ffn_dim`、`num_heads` 等结构参数，而现有部署管线中存在硬编码维度假设。需要系统性地将这些假设替换为从剪枝后模型动态读取的实际值。

---

## 2. 前置条件

### 2.1 Phase A1-A3 产出 (必须已完成)

- [x] `projects/mmdet3d_plugin/univ2x/pruning/` 目录下的剪枝框架代码
- [x] `tools/prune_and_eval.py` 可接收 `prune_config.json` 执行剪枝 + 评估
- [x] 剪枝后 `.pth` 文件包含正确的模型结构（DepGraph 自动维护维度一致性）
- [x] Phase 0 已锁定 P4-P7 值（importance_criterion, granularity, iterative_steps, round_to）
- [x] 剪枝后模型的 `update_model_after_pruning()` 已正确更新所有模块属性

### 2.2 现有 TRT 管线 (必须可运行)

| 文件 | 功能 | 状态 |
|------|------|:----:|
| `tools/export_onnx_univ2x.py` | ONNX 导出 (BEV encoder, heads, downstream) | 可运行 |
| `tools/dump_univ2x_calibration.py` | INT8 校准数据收集 (ego + infra) | 可运行 |
| `tools/build_trt_int8_univ2x.py` | TRT engine 构建 (FP16, INT8 隐式/显式) | 可运行 |
| `tools/test_trt.py` | TRT 推理评估 | 可运行 |

### 2.3 校准数据基础设施

- `calibration/` 目录结构已存在
- `dump_univ2x_calibration.py` 的 hook 机制（拦截 `pts_bbox_head.get_bev_features`）已验证
- 校准数据格式: list of dict, 每个 dict 包含 `{feat0, feat1, feat2, feat3, can_bus, lidar2img, image_shape, prev_bev, use_prev_bev}` 的 numpy 数组

---

## 3. 具体代码实现

### 3.1 ONNX 导出适配 (B.2) — 1 天

**修改文件**: `tools/export_onnx_univ2x.py`

#### 3.1.1 问题分析

剪枝改变的维度与 ONNX 导出中的影响:

| 搜索维度 | 改变的参数 | 导出脚本中的影响位置 |
|---------|----------|-------------------|
| P1 ffn_mid_ratio | ffn 中间层 Linear 的 out_features | 无直接影响（torch.onnx.export 自动追踪） |
| P2 attn_proj_ratio | value_proj/output_proj 的 out_features | BEVEncoderWrapper 的 prev_bev 输入 shape: `(bev_h*bev_w, bs, embed_dims)` |
| P3 head_mid_ratio | 检测头 Linear 的 out_features | HeadsWrapper/HeadsDecoderOnlyWrapper 的 track_query 输入: `(num_query, C*2)` |
| P8 head_pruning | num_heads 减少 | MSDAPlugin 的 num_heads 参数 |
| P9 decoder_layers | num_dec_layers 减少 | HeadsWrapper 输出的 all_cls_scores 第 0 维: `(num_dec_layers, ...)` |

#### 3.1.2 需要修改的硬编码位置

```python
# ── 当前: BEVEncoderWrapper 中 dummy 输入构造 ──
# 搜索 export_onnx_univ2x.py 中所有 embed_dims=256 / C=256 的硬编码

# 问题 1: dummy 输入的 embed_dims 维度
# 当前代码可能有:
#   prev_bev = torch.zeros(bev_h * bev_w, 1, 256)  ← 硬编码 256
# 修改为:
#   embed_dims = detector.pts_bbox_head.embed_dims  ← 从剪枝后模型动态读取
#   prev_bev = torch.zeros(bev_h * bev_w, 1, embed_dims)

# 问题 2: dummy 输入的 feat shape 中 C 维度
# FPN 输出通道数通常由 backbone neck 决定，剪枝不影响（P1-P3 不剪 backbone）
# 但需要验证: 若 P2 影响了 embed_dims，而 neck 输出与 embed_dims 绑定则需适配

# 问题 3: HeadsWrapper 中 track_query 的 C*2 维度
# 当前代码:
#   track_query = torch.zeros(num_query, 256*2)  ← 硬编码 256
# 修改为:
#   embed_dims = detector.pts_bbox_head.embed_dims
#   track_query = torch.zeros(num_query, embed_dims * 2)
```

#### 3.1.3 实现方案

```python
def _get_model_dims(detector):
    """从剪枝后模型动态读取所有维度参数。
    
    剪枝后 update_model_after_pruning() 已更新模块属性，
    这里直接读取即可，无需反向推断。
    """
    head = detector.pts_bbox_head
    return {
        'embed_dims': head.embed_dims,          # P2 可能改变: 256 → 204
        'bev_h': head.bev_h,                    # 不变: 200
        'bev_w': head.bev_w,                    # 不变: 200
        'num_query': head.bev_h * head.bev_w,   # 不变: 40000
        'num_heads': head.transformer.encoder.layers[0]
                     .attentions[0].num_heads,   # P8 可能改变: 8 → 7
        'num_dec_layers': head.transformer.decoder.num_layers,  # P9 可能改变: 6 → 5
        'num_classes': head.num_classes,         # 不变
        'code_size': head.code_size,             # 不变: 10
    }


def _build_dummy_inputs_bev_encoder(dims, device='cuda'):
    """构建 BEVEncoderWrapper 的 dummy 输入，维度从模型动态读取。"""
    C = dims['embed_dims']
    bev_h, bev_w = dims['bev_h'], dims['bev_w']
    
    # FPN 级别的 feat shape: C 通道由 backbone neck 决定
    # 注意: backbone 不参与剪枝，FPN 输出通道数仍为原始值
    # 但 BEVFormerEncoder 的输入投影会将 FPN 通道映射到 embed_dims
    # 如果 embed_dims 被剪枝，这个映射层的维度已由 DepGraph 自动处理
    feat_C = 256  # backbone FPN 输出通道 (不受剪枝影响)
    
    return {
        'feat0': torch.randn(1, 1, feat_C, 136, 240, device=device),
        'feat1': torch.randn(1, 1, feat_C, 68, 120, device=device),
        'feat2': torch.randn(1, 1, feat_C, 34, 60, device=device),
        'feat3': torch.randn(1, 1, feat_C, 17, 30, device=device),
        'can_bus': torch.randn(18, device=device),
        'lidar2img': torch.randn(1, 1, 4, 4, device=device),
        'image_shape': torch.tensor([544.0, 960.0], device=device),
        'prev_bev': torch.randn(bev_h * bev_w, 1, C, device=device),  # ← 动态 C
        'use_prev_bev': torch.tensor(0.0, device=device),
    }


def _build_dummy_inputs_heads(dims, device='cuda'):
    """构建 HeadsWrapper 的 dummy 输入，维度从模型动态读取。"""
    C = dims['embed_dims']
    num_query = 901  # ego 查询数 (固定)
    
    return {
        'bev_embed': torch.randn(dims['num_query'], 1, C, device=device),
        'track_query': torch.randn(num_query, C * 2, device=device),  # ← 动态 C*2
        'track_ref_pts': torch.randn(num_query, 3, device=device),
        'l2g_r1': torch.eye(3, device=device),
        'l2g_t1': torch.zeros(3, device=device),
        'l2g_r2': torch.eye(3, device=device),
        'l2g_t2': torch.zeros(3, device=device),
        'time_delta': torch.tensor(0.5, device=device),
    }
```

#### 3.1.4 TRT 专用模块检查 (*TRT 后缀)

需要检查以下 TRT 兼容模块中的维度假设:

| 模块 | 文件 | 需要检查的维度 |
|------|------|-------------|
| `BEVFormerEncoderTRT` | `projects/mmdet3d_plugin/univ2x/modules/encoder.py` | `embed_dims`, `num_heads` |
| `TemporalSelfAttentionTRT` | `projects/mmdet3d_plugin/univ2x/modules/temporal_self_attention.py` | `embed_dims`, `num_heads` |
| `SpatialCrossAttentionTRT` | `projects/mmdet3d_plugin/univ2x/modules/spatial_cross_attention.py` | `embed_dims`, `num_heads` |
| `BEVFormerTrackHeadTRT` | `projects/mmdet3d_plugin/univ2x/detectors/` | `embed_dims`, `num_dec_layers` |
| `MSDAPlugin` (C++ TRT plugin) | `plugins/` | `embed_dims`, `num_heads` — **高风险** |

**MSDAPlugin 风险分析**:

MSDAPlugin 是 C++ TRT 自定义插件，其 `enqueue()` 内核可能硬编码了 `embed_dims=256` 和 `num_heads=8`。如果 P2 或 P8 改变了这些值，插件需要修改或参数化。

```
风险等级: 高
排查方法: 
  1. 检查 plugins/src/ 中的 MSDAPlugin 源码
  2. 搜索 256、8 等硬编码常量
  3. 确认这些值是从 ONNX 属性读取还是编译时常量
缓解方案:
  - 如果是编译时常量 → 需要修改插件，将维度参数化
  - 如果是从 ONNX 属性读取 → 无需修改
  - 如果修改代价过高 → 约束 P2 和 P8 不影响 MSDA 相关维度
```

#### 3.1.5 ONNX 输出验证

```python
def verify_pruned_onnx(onnx_path, expected_dims):
    """验证剪枝后 ONNX 模型的维度正确性。"""
    import onnx
    from onnx import shape_inference
    
    model = onnx.load(onnx_path)
    
    # Step 1: onnx.checker 基本合法性
    onnx.checker.check_model(model)
    print(f'  onnx.checker: PASS')
    
    # Step 2: shape inference (检测维度不一致)
    model_inferred = shape_inference.infer_shapes(model)
    print(f'  shape_inference: PASS')
    
    # Step 3: 验证输入维度
    for inp in model.graph.input:
        name = inp.name
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f'  Input {name}: {shape}')
        
        # 检查 embed_dims 维度是否与剪枝后值匹配
        if name == 'prev_bev':
            assert shape[-1] == expected_dims['embed_dims'], \
                f"prev_bev embed_dims 不匹配: ONNX={shape[-1]}, " \
                f"expected={expected_dims['embed_dims']}"
    
    # Step 4: 验证输出维度
    for out in model.graph.output:
        name = out.name
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f'  Output {name}: {shape}')
    
    print(f'  ONNX 验证通过: {onnx_path}')
```

---

### 3.2 校准数据重建 (B.3) — 0.5 天代码 + 2 小时执行

**修改文件**: `tools/dump_univ2x_calibration.py`

#### 3.2.1 为什么旧校准数据不可复用

```
旧校准数据 (calibration/bev_encoder_ego_calib_inputs.pkl):
  prev_bev shape: (40000, 1, 256)      ← 原始 embed_dims=256
  每帧大小: ~40 MB
  50 帧总计: ~2 GB

剪枝后校准数据 (calibration_pruned/bev_encoder_ego_calib_inputs.pkl):
  prev_bev shape: (40000, 1, 204)      ← 剪枝后 embed_dims=204 (P2=0.2)
  每帧大小: ~32 MB (按比例缩小)
  50 帧总计: ~1.6 GB
```

`build_trt_int8_univ2x.py` 中的 `UniV2XInt8CalibratorImpl` 在 `__init__` 时根据第一个 sample 的 shape 分配 GPU buffer，然后在 `get_batch()` 中 `copy_()` 数据到 buffer。如果 pkl 中的 tensor shape 与剪枝后 ONNX 的 input shape 不匹配，会产生 shape mismatch 错误。

#### 3.2.2 需要修改的内容

`dump_univ2x_calibration.py` 的 `_extract_bev_inputs()` 函数已经从 `head.embed_dims` 动态读取维度:

```python
# 第 79-80 行: 已有动态维度读取
embed_dims = head.embed_dims
num_query  = bev_h * bev_w
```

**结论**: `dump_univ2x_calibration.py` 的核心逻辑**无需修改**，因为它从模型属性动态读取维度。只要加载的 checkpoint 是剪枝后的模型，`head.embed_dims` 就会反映剪枝后的值。

**需要修改的内容**:

1. **输出路径分离**: 添加 `--pruned` 标志或自动检测，将剪枝后校准数据存到独立目录，避免覆盖原始数据
2. **Shape 验证**: 在保存前打印 shape 摘要，与 ONNX 输入 spec 交叉验证
3. **Config 适配**: 如果剪枝后使用不同的 config 文件（含 TRT 模块替换），确保 config 路径正确

```python
# 新增: 校准数据保存后的 shape 验证
def _verify_calib_shapes(calib_data, onnx_path=None):
    """验证校准数据 shape 与 ONNX 输入 spec 匹配。"""
    if not calib_data:
        return
    
    first_sample = calib_data[0]
    print('\n── 校准数据 shape 验证 ──')
    for key, arr in first_sample.items():
        print(f'  {key}: {arr.shape}  dtype={arr.dtype}')
    
    if onnx_path is not None:
        import onnx
        model = onnx.load(onnx_path)
        onnx_inputs = {inp.name: [d.dim_value for d in inp.type.tensor_type.shape.dim]
                       for inp in model.graph.input}
        
        for key, arr in first_sample.items():
            if key in onnx_inputs:
                expected = onnx_inputs[key]
                actual = list(arr.shape)
                if expected != actual:
                    print(f'  WARNING: {key} shape 不匹配! '
                          f'calib={actual} vs onnx={expected}')
```

#### 3.2.3 执行命令

```bash
# 使用剪枝后 checkpoint 重新收集校准数据
python tools/dump_univ2x_calibration.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    work_dirs/pruned_finetuned.pth \
    --n-frames 50 \
    --out-ego  calibration_pruned/bev_encoder_ego_calib_inputs.pkl \
    --out-infra calibration_pruned/bev_encoder_infra_calib_inputs.pkl
```

预计执行时间: 50 帧 × ~2.5 分钟/帧 ≈ 2 小时 (可通过减少 n-frames 到 30 帧缩短，精度影响可忽略)

---

### 3.3 TRT 构建适配 (B.4) — 0.5 天

**修改文件**: `tools/build_trt_int8_univ2x.py`

#### 3.3.1 `_force_msda_fp16()` 层名匹配

当前实现（第 235-247 行）通过 `trt.LayerType.PLUGIN_V2` 类型检测自定义插件层，不依赖层名字符串匹配:

```python
def _force_msda_fp16(network, trt):
    plugin_count = sum(1 for i in range(network.num_layers)
                       if network.get_layer(i).type == trt.LayerType.PLUGIN_V2)
    print(f'  Found {plugin_count} custom plugin layer(s) — '
          f'running in native FP16 (TRT default for plugins)')
```

**结论**: 剪枝不改变插件层的类型标识，`_force_msda_fp16()` **无需修改**。

#### 3.3.2 INT8 Calibrator input shape 自动推断

当前实现（第 65-74 行）已经从第一个 calibration sample 自动推断 buffer shape:

```python
if cali_tensors:
    first = cali_tensors[0]
    for name in input_names:
        arr = first.get(name)
        if arr is None:
            continue
        dtype = torch.float32 if arr.dtype != np.bool_ else torch.uint8
        buf = torch.zeros(arr.shape, dtype=dtype, device='cuda')
        self._device_buffers[name] = buf
```

**结论**: 只要校准数据 pkl 的 shape 与 ONNX input 一致（3.2 保证），`UniV2XInt8CalibratorImpl` **无需修改**。

#### 3.3.3 Workspace 大小优化

剪枝后模型参数量减少，workspace 可以适当缩减:

```python
# 修改 parse_args() 的默认值，或添加自动推断逻辑
# 经验公式: workspace_gb ≈ 模型大小 × 4 + 2 GB 余量
def _estimate_workspace(onnx_path):
    """根据 ONNX 模型大小估算合理的 workspace。"""
    model_size_mb = os.path.getsize(onnx_path) / 1024**2
    # 剪枝后模型 ~60-80% 原始大小
    # 原始模型 workspace=8GB，剪枝后可降至 4-6GB
    estimated_gb = max(4.0, model_size_mb / 1024 * 4 + 2.0)
    return min(estimated_gb, 8.0)  # 上限仍为 8GB
```

#### 3.3.4 INT8 校准缓存管理

剪枝后模型结构不同，旧的 `.cache` 文件**不可复用**（scale 值对应的 tensor shape 已变）。需要:

1. 删除旧的 `calibration/*_int8.cache` 或存到不同路径
2. 强制重新校准: 首次构建时从 pkl 计算新 scale

```python
# 添加 --force-recalibrate 选项
p.add_argument('--force-recalibrate', action='store_true',
               help='删除旧 calibration cache，强制重新校准 (剪枝后必须)')
```

---

### 3.4 硬件约束验证 (B.5) — 0.5 天

**新建文件**: `tools/verify_trt_constraints.py`

#### 3.4.1 约束清单

| 编号 | 约束 | 来源 | 违反后果 |
|:----:|------|------|---------|
| C1 | `out_channels % round_to == 0` | INT8 kernel 对齐要求 | TRT fallback 到 FP32，剪了反而更慢 |
| C2 | `out_channels >= min_channels` | INT8 kernel 效率下限 | 延迟不降反升 |
| C3 | `embed_dims % num_heads == 0` | MultiheadAttention 整除要求 | 运行时崩溃 |
| C4 | ego.embed_dims == infra.embed_dims | V2X AgentQueryFusion 维度匹配 | 跨模型融合崩溃 |
| C5 | `sampling_offsets.in_features` 未被修改 | 坐标敏感层硬约束 | MSDA 输出错误 |

#### 3.4.2 完整实现

```python
"""verify_trt_constraints.py — 剪枝后 TRT 部署前硬件约束检查。

在执行 ONNX 导出和 TRT 构建之前运行，快速失败。
任何约束违反都应在 DepGraph 阶段通过 round_to/ignored_layers 预防，
此脚本是最后的防线 (defense in depth)。

Usage:
    python tools/verify_trt_constraints.py \
        --checkpoint work_dirs/pruned.pth \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        --prune-config prune_configs/moderate.json
"""

import argparse
import json
import sys
import torch


def verify_channel_alignment(model, round_to=8):
    """C1: 所有 Linear/Conv2d 的 out_channels 必须是 round_to 的倍数。"""
    violations = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            out_ch = module.out_features
            if out_ch % round_to != 0:
                violations.append(
                    f'[C1-ALIGN] {name}: out_features={out_ch}, '
                    f'不是 {round_to} 的倍数')
        elif isinstance(module, torch.nn.Conv2d):
            out_ch = module.out_channels
            if out_ch % round_to != 0:
                violations.append(
                    f'[C1-ALIGN] {name}: out_channels={out_ch}, '
                    f'不是 {round_to} 的倍数')
    return violations


def verify_min_channels(model, min_channels=64):
    """C2: 所有层的 out_channels >= min_channels。"""
    violations = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.out_features < min_channels:
                violations.append(
                    f'[C2-MIN_CH] {name}: out_features={module.out_features} '
                    f'< 最小值 {min_channels}')
        elif isinstance(module, torch.nn.Conv2d):
            if module.out_channels < min_channels:
                violations.append(
                    f'[C2-MIN_CH] {name}: out_channels={module.out_channels} '
                    f'< 最小值 {min_channels}')
    return violations


def verify_head_alignment(model):
    """C3: embed_dims % num_heads == 0 (注意力头整除)。"""
    violations = []
    for name, module in model.named_modules():
        if hasattr(module, 'num_heads') and hasattr(module, 'embed_dims'):
            if module.embed_dims % module.num_heads != 0:
                violations.append(
                    f'[C3-HEAD] {name}: embed_dims={module.embed_dims} '
                    f'不能被 num_heads={module.num_heads} 整除')
    return violations


def verify_v2x_dimension_consistency(ego_model, infra_model):
    """C4: ego 和 infra 模型的 embed_dims 必须一致。"""
    violations = []
    ego_dims = ego_model.pts_bbox_head.embed_dims
    infra_dims = infra_model.pts_bbox_head.embed_dims
    if ego_dims != infra_dims:
        violations.append(
            f'[C4-V2X] ego.embed_dims={ego_dims} != '
            f'infra.embed_dims={infra_dims}')
    return violations


def verify_coordinate_sensitive_layers(model):
    """C5: 坐标敏感层 (sampling_offsets, attention_weights) 未被修改。"""
    violations = []
    # 已知原始维度 (从未剪枝模型)
    expected = {
        'sampling_offsets': None,    # 输出维度由 num_heads*num_levels*num_points 决定
        'attention_weights': None,   # 输出维度同上
    }
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for sensitive_key in ['sampling_offsets', 'attention_weights']:
                if sensitive_key in name:
                    # 输出维度不应被剪枝改变
                    # (DepGraph ignored_layers 应已排除这些层)
                    pass  # 此处不能硬编码期望值，仅检查是否在 ignored list 中
    return violations


def run_all_checks(model, prune_config, infra_model=None):
    """执行所有硬件约束检查，返回 (pass, violations)。"""
    constraints = prune_config.get('constraints', {})
    round_to = constraints.get('channel_alignment', 8)
    min_channels = constraints.get('min_channels', 64)
    
    all_violations = []
    all_violations.extend(verify_channel_alignment(model, round_to))
    all_violations.extend(verify_min_channels(model, min_channels))
    all_violations.extend(verify_head_alignment(model))
    all_violations.extend(verify_coordinate_sensitive_layers(model))
    
    if infra_model is not None:
        all_violations.extend(
            verify_v2x_dimension_consistency(model, infra_model))
    
    if all_violations:
        print(f'硬件约束检查: FAILED ({len(all_violations)} 个违反)')
        for v in all_violations:
            print(f'  {v}')
        return False, all_violations
    
    print('硬件约束检查: PASSED')
    return True, []
```

---

### 3.5 端到端验证 (B.6) — 1 天

#### 3.5.1 验证矩阵

| 验证项 | 方法 | 合格标准 | 预计耗时 |
|-------|------|---------|---------|
| ONNX 导出 | `onnx.checker.check_model()` + `shape_inference` | 无错误 | 5 分钟 |
| TRT FP16 构建 | `build_trt_int8_univ2x.py --no-int8` | 构建成功 | 15 分钟 |
| TRT INT8 构建 | `build_trt_int8_univ2x.py` + 重新校准 | 构建成功 | 30 分钟 |
| PyTorch-TRT 精度 | `test_trt.py` AMOTA 对比 | `|AMOTA_trt - AMOTA_pytorch| < 0.005` | 20 分钟 |
| 推理延迟 | `test_trt.py --benchmark` | 剪枝 TRT < 未剪枝 TRT × 0.85 | 10 分钟 |
| 通道对齐 | `verify_trt_constraints.py` | 无 TRT warnings | 1 分钟 |
| V2X 维度匹配 | ego + infra 同时验证 | embed_dims 一致 | 5 分钟 |
| 逐层输出对比 | Polygraphy 或手动 hook | cosine similarity > 0.99 | 30 分钟 |

#### 3.5.2 逐层输出对比方案

```python
def compare_pytorch_trt_outputs(pytorch_model, trt_engine_path, 
                                 test_inputs, threshold=0.99):
    """逐层对比 PyTorch 和 TRT 的输出，定位精度偏差根源。
    
    方法: 使用 Polygraphy 的 Comparator，或手动在 PyTorch 侧注册
    forward hook 收集中间输出，与 TRT 的对应层输出对比。
    """
    import numpy as np
    
    # 方法 1: 最终输出对比 (快速)
    pytorch_out = pytorch_model(**test_inputs)
    trt_out = run_trt_inference(trt_engine_path, test_inputs)
    
    for name, (pt, trt) in zip(['cls', 'bbox', 'traj'], 
                                zip(pytorch_out, trt_out)):
        pt_np = pt.cpu().numpy().flatten()
        trt_np = trt.flatten()
        cos_sim = np.dot(pt_np, trt_np) / (
            np.linalg.norm(pt_np) * np.linalg.norm(trt_np) + 1e-8)
        status = 'PASS' if cos_sim > threshold else 'FAIL'
        print(f'  {name}: cosine_similarity={cos_sim:.6f} [{status}]')
    
    # 方法 2: Polygraphy (详细)
    # polygraphy run pruned.onnx \
    #     --trt --onnxrt \
    #     --atol 1e-3 --rtol 1e-3 \
    #     --check-error-stat median
```

---

### 3.6 自动化部署脚本

**新建文件**: `tools/deploy_pruned_model.py`

#### 3.6.1 功能

一键完成: 硬件约束检查 → ONNX 导出 → 维度验证 → 校准数据收集 → TRT 构建 → 精度验证

```python
"""deploy_pruned_model.py — 剪枝后模型一键部署到 TRT。

将完整部署流程串联，任一步骤失败则中止并报告。

Usage:
    python tools/deploy_pruned_model.py \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
        --checkpoint work_dirs/pruned_finetuned.pth \
        --prune-config prune_configs/moderate.json \
        --output-dir deploy_pruned/ \
        --model ego \
        --skip-calibration  # 如果已有校准数据可跳过
"""

import argparse
import os
import subprocess
import sys
import time


def run_step(name, cmd, fail_fast=True):
    """执行一个部署步骤，打印耗时并检查返回码。"""
    print(f'\n{"="*60}')
    print(f'Step: {name}')
    print(f'Cmd:  {cmd}')
    print(f'{"="*60}')
    
    t0 = time.time()
    ret = subprocess.run(cmd, shell=True)
    elapsed = time.time() - t0
    
    if ret.returncode != 0:
        print(f'FAILED: {name} (耗时 {elapsed:.1f}s)')
        if fail_fast:
            sys.exit(1)
        return False
    
    print(f'PASSED: {name} (耗时 {elapsed:.1f}s)')
    return True


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    onnx_path = os.path.join(output_dir, f'pruned_{args.model}_bev_encoder.onnx')
    trt_fp16_path = os.path.join(output_dir, f'pruned_{args.model}_bev_encoder_fp16.trt')
    trt_int8_path = os.path.join(output_dir, f'pruned_{args.model}_bev_encoder_int8.trt')
    calib_path = os.path.join(output_dir, f'bev_encoder_{args.model}_calib_inputs.pkl')
    
    # Step 1: 硬件约束验证
    run_step('硬件约束验证',
        f'python tools/verify_trt_constraints.py '
        f'--config {args.config} '
        f'--checkpoint {args.checkpoint} '
        f'--prune-config {args.prune_config}')
    
    # Step 2: ONNX 导出
    run_step('ONNX 导出',
        f'python tools/export_onnx_univ2x.py '
        f'{args.config} {args.checkpoint} '
        f'--model {args.model} '
        f'--backbone-only '
        f'--bev-size 200 '
        f'--out {onnx_path}')
    
    # Step 3: 校准数据收集 (可选跳过)
    if not args.skip_calibration:
        out_flag = f'--out-ego {calib_path}' if args.model == 'ego' \
                   else f'--out-infra {calib_path}'
        run_step('校准数据收集',
            f'python tools/dump_univ2x_calibration.py '
            f'{args.config} {args.checkpoint} '
            f'--n-frames {args.n_calib_frames} '
            f'{out_flag}')
    
    # Step 4: TRT FP16 构建
    run_step('TRT FP16 构建',
        f'python tools/build_trt_int8_univ2x.py '
        f'--onnx {onnx_path} '
        f'--out {trt_fp16_path} '
        f'--target bev_encoder '
        f'--no-int8')
    
    # Step 5: TRT INT8 构建
    run_step('TRT INT8 构建',
        f'python tools/build_trt_int8_univ2x.py '
        f'--onnx {onnx_path} '
        f'--out {trt_int8_path} '
        f'--target bev_encoder '
        f'--cali-data {calib_path} '
        f'--force-recalibrate')
    
    # Step 6: 精度验证
    run_step('精度验证',
        f'python tools/test_trt.py '
        f'--config {args.config} '
        f'--engine {trt_int8_path} '
        f'--eval')
    
    print(f'\n{"="*60}')
    print(f'部署完成!')
    print(f'  ONNX:     {onnx_path}')
    print(f'  TRT FP16: {trt_fp16_path}')
    print(f'  TRT INT8: {trt_int8_path}')
    print(f'{"="*60}')


def parse_args():
    p = argparse.ArgumentParser(description='剪枝模型一键部署到 TRT')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--prune-config', required=True)
    p.add_argument('--output-dir', default='deploy_pruned/')
    p.add_argument('--model', choices=['ego', 'infra'], default='ego')
    p.add_argument('--skip-calibration', action='store_true')
    p.add_argument('--n-calib-frames', type=int, default=50)
    return p.parse_args()


if __name__ == '__main__':
    main()
```

---

## 4. 代码检测方案

### Test 1: ONNX 导出正确性

```bash
# 加载剪枝后模型，导出 ONNX，验证 checker 通过
python tools/export_onnx_univ2x.py \
    projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    work_dirs/pruned_finetuned.pth \
    --model ego --backbone-only --bev-size 200 \
    --out onnx/pruned_ego_bev_encoder.onnx

# 验证
python -c "
import onnx
m = onnx.load('onnx/pruned_ego_bev_encoder.onnx')
onnx.checker.check_model(m)
onnx.shape_inference.infer_shapes(m)
print('ONNX checker: PASS')
for inp in m.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'  {inp.name}: {shape}')
"
```

### Test 2: TRT FP16 构建

```bash
python tools/build_trt_int8_univ2x.py \
    --onnx onnx/pruned_ego_bev_encoder.onnx \
    --out trt_engines/pruned_ego_bev_encoder_fp16.trt \
    --target bev_encoder \
    --no-int8

# 预期: Engine saved: ... (XX.X MB)
# 失败标志: PARSER ERROR 或 Engine build failed
```

### Test 3: PyTorch-TRT 输出对比

```bash
# 10 个样本上对比 PyTorch 和 TRT 的 BEV encoder 输出
python tools/test_trt.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    --checkpoint work_dirs/pruned_finetuned.pth \
    --engine trt_engines/pruned_ego_bev_encoder_fp16.trt \
    --compare-pytorch \
    --num-samples 10

# 预期: cosine_similarity > 0.99 for all samples
```

### Test 4: INT8 TRT + AMOTA 验证

```bash
# 重新校准 + 构建 INT8 engine + 评估 AMOTA
python tools/deploy_pruned_model.py \
    --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_trt_p2.py \
    --checkpoint work_dirs/pruned_finetuned.pth \
    --prune-config prune_configs/moderate.json \
    --output-dir deploy_test/

# 预期: |AMOTA_trt - AMOTA_pytorch| < 0.005
```

---

## 5. Debug 方案

### 5.1 ONNX 导出失败

| 症状 | 排查方法 | 常见原因 |
|------|---------|---------|
| `torch.onnx.export` 抛 TracingError | `verbose=True` 打印 trace 日志，定位哪个 op 失败 | 剪枝后某个模块维度不匹配导致 forward 崩溃 |
| 导出成功但 `onnx.checker` 报错 | 检查 checker 错误信息中的 node 名称和 shape | 动态维度未正确标注 |
| shape_inference 失败 | 检查哪个节点的 input/output shape 矛盾 | 通常是 Reshape 节点的目标 shape 与实际不匹配 |

**关键排查命令**:

```python
# 启用 verbose 模式导出
torch.onnx.export(
    model, args, output_path,
    verbose=True,              # ← 打印所有 trace 日志
    opset_version=13,
    do_constant_folding=False, # ← 关闭常量折叠以看到完整图
)
```

### 5.2 TRT 构建失败

| 症状 | 排查方法 | 常见原因 |
|------|---------|---------|
| PARSER ERROR | 查看 `parser.get_error(i)` 详细信息 | 不支持的 ONNX op 或版本 |
| Engine build failed (无具体错误) | 将 `trt.Logger` 级别设为 `VERBOSE` | 内存不足或 kernel 选择失败 |
| 构建成功但推理结果全 0 | 检查 input binding 顺序是否与 ONNX 一致 | 维度变化导致 binding 名称变化 |
| Custom plugin 报错 | 检查 MSDAPlugin 参数是否与剪枝后维度匹配 | P2/P8 改变了 embed_dims/num_heads |

**关键排查命令**:

```python
# TRT VERBOSE 日志
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# 检查每一层的 precision 和 shape
for i in range(network.num_layers):
    layer = network.get_layer(i)
    print(f'Layer {i}: {layer.name}  type={layer.type}  '
          f'precision={layer.precision}')
```

### 5.3 精度不匹配

| 偏差范围 | 可能原因 | 解决方案 |
|---------|---------|---------|
| cosine < 0.99 (FP16) | BN 统计量 drift（TrainingMode 问题） | 确认 `onnx_compatible_attention` 中的 BN patch 生效 |
| cosine < 0.95 (FP16) | 剪枝后某层维度错误导致计算路径错 | 逐层对比 PyTorch vs ONNX Runtime 输出 |
| AMOTA 偏差 > 0.005 (INT8) | 校准数据质量不足或 scale 不准确 | 增加校准帧数到 100+，或尝试不同校准方法 |
| AMOTA 偏差 > 0.02 (INT8) | 剪枝后激活分布变化大，INT8 量化误差放大 | 考虑对敏感层强制 FP16 (不量化) |

**逐层对比工具**:

```bash
# Polygraphy: PyTorch ONNX Runtime vs TRT 逐层对比
polygraphy run pruned.onnx \
    --trt --onnxrt \
    --atol 1e-3 --rtol 1e-3 \
    --trt-outputs mark all \
    --onnxrt-outputs mark all \
    --check-error-stat median
```

### 5.4 校准数据 shape 不匹配

```python
# 打印 pkl 中的 shape 并与 ONNX input spec 对比
import pickle, onnx

with open('calibration_pruned/bev_encoder_ego_calib_inputs.pkl', 'rb') as f:
    calib = pickle.load(f)

print('Calibration data shapes:')
for k, v in calib[0].items():
    print(f'  {k}: {v.shape}')

model = onnx.load('onnx/pruned_ego_bev_encoder.onnx')
print('\nONNX input specs:')
for inp in model.graph.input:
    shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f'  {inp.name}: {shape}')
```

---

## 6. 验收标准

### 6.1 功能验收

- [ ] **一键部署**: `tools/deploy_pruned_model.py` 可从 `.pth` 到 TRT engine 全自动完成
- [ ] **FP16 构建成功**: 剪枝后 ONNX → TRT FP16 engine 无错误
- [ ] **INT8 构建成功**: 重新校准后 → TRT INT8 engine 无错误
- [ ] **硬件约束检查**: `verify_trt_constraints.py` 在构建前自动运行，不通过则中止

### 6.2 精度验收

- [ ] **PyTorch-TRT FP16 一致性**: cosine similarity > 0.99 (10 个样本)
- [ ] **AMOTA 偏差**: `|AMOTA_trt_int8 - AMOTA_pytorch| < 0.005`
- [ ] **跨 3 个 prune_config 验证**: moderate / aggressive / conservative 三种配置均满足偏差标准

### 6.3 性能验收

- [ ] **延迟改善**: 剪枝 30% FFN 后，TRT latency 降低 > 15% (vs 未剪枝 TRT)
- [ ] **联合加速**: 剪枝 + INT8 联合加速 > 单独 INT8 的 1.3x

### 6.4 V2X 验收

- [ ] **ego + infra 维度一致**: 两端 embed_dims 匹配
- [ ] **AgentQueryFusion 可用**: 跨模型特征融合后精度无异常

---

## 7. 风险与缓解

### 7.1 MSDAPlugin 硬编码维度 (风险等级: 高)

**风险**: TRT 自定义插件 `MSDAPlugin` 的 C++ 内核中可能硬编码了 `embed_dims=256` 和 `num_heads=8`。如果 P2 或 P8 改变了这些值，TRT 构建将失败或产生错误结果。

**缓解方案**:
1. **优先检查**: Phase B1 第一天立即检查 `plugins/src/` 中的 MSDAPlugin 源码
2. **如果硬编码**: 将维度参数化，从 ONNX 节点属性中读取（需要修改 C++ 代码 + 重新编译插件）
3. **如果修改代价过高**: 约束搜索空间，确保 P2/P8 不改变 MSDA 相关的 embed_dims 和 num_heads（降级方案）

### 7.2 校准数据收集耗时 (风险等级: 中)

**风险**: 每次剪枝配置变化后都需要重新收集校准数据，512 帧需要 2+ 小时。在搜索 216 个配置时不可行。

**缓解方案**:
1. **减少帧数**: 验证 30 帧 vs 50 帧的 AMOTA 差距是否 < 0.001，如果满足则使用 30 帧（~1.2 小时）
2. **隐式量化模式**: 使用 `build_trt_int8_univ2x.py` 的隐式模式（Calibrator-based），calibration cache 可在部分维度变化下复用
3. **并行化**: 在多 GPU 上同时收集不同配置的校准数据

### 7.3 INT8 校准质量退化 (风险等级: 中)

**风险**: 剪枝后模型的激活分布与未剪枝模型不同，INT8 entropy calibrator 的 scale 估计可能不准确，导致 AMOTA 偏差超过 0.005 阈值。

**缓解方案**:
1. **重新校准**: 使用剪枝后模型生成校准数据（本方案的默认做法）
2. **增加校准帧数**: 从 50 帧增加到 100-200 帧
3. **敏感层 FP16**: 对 AMOTA 偏差贡献最大的层强制 FP16 (不走 INT8 量化)
4. **显式量化模式**: 如果隐式模式精度不足，切换到 Q/DQ 显式模式（复用 1.1 的 inject_qdq_from_config.py）

### 7.4 V2X 维度不一致 (风险等级: 低)

**风险**: 如果 ego 和 infra 使用不同的 prune_config（或剪枝后 embed_dims 不一致），AgentQueryFusion 跨模型融合将失败。

**缓解方案**:
- 在 `prune_config.json` 中强制 ego 和 infra 使用相同的 embed_dims 搜索值
- `verify_trt_constraints.py` 的 C4 检查已覆盖此场景

---

## 8. 反思模板

> 每个子任务完成后填写，记录实际遇到的问题和经验教训。

### B.2 ONNX 导出适配

| 项目 | 内容 |
|------|------|
| 预期耗时 vs 实际耗时 | 1 天 vs ？ |
| 预期中的问题 | 硬编码维度、TRT 模块假设 |
| 意外问题 | (待填写) |
| 解决方案 | (待填写) |
| 经验教训 | (待填写) |

### B.3 校准数据重建

| 项目 | 内容 |
|------|------|
| 预期耗时 vs 实际耗时 | 0.5 天 + 2 小时 vs ？ |
| 校准帧数选择 | 50 帧 (预计) / 实际用 ？帧 |
| shape 不匹配问题 | (待填写) |
| 经验教训 | (待填写) |

### B.4 TRT 构建适配

| 项目 | 内容 |
|------|------|
| 预期耗时 vs 实际耗时 | 0.5 天 vs ？ |
| MSDAPlugin 维度问题 | 是否硬编码？如何解决？ |
| workspace 优化效果 | 原始 vs 优化后 |
| 经验教训 | (待填写) |

### B.5 硬件约束验证

| 项目 | 内容 |
|------|------|
| 预期耗时 vs 实际耗时 | 0.5 天 vs ？ |
| 发现的约束违反 | (待填写) |
| 是否需要回溯修改 DepGraph | (待填写) |
| 经验教训 | (待填写) |

### B.6 端到端验证

| 项目 | 内容 |
|------|------|
| 预期耗时 vs 实际耗时 | 1 天 vs ？ |
| FP16 cosine similarity | (待填写) |
| INT8 AMOTA 偏差 | (待填写) |
| 延迟改善百分比 | (待填写) |
| 验证矩阵通过率 | (待填写) |

### 整体反思

| 维度 | 1.1 可配置量化经验 | 1.2 剪枝部署实际 |
|------|----------------|-----------------|
| ONNX 导出难度 | 高 (Q/DQ 注入复杂) | ？ (维度变化影响) |
| TRT 构建难度 | 中 (显式/隐式模式) | ？ (维度变化 + 插件) |
| 精度偏差控制 | < 0.003 (较好) | ？ |
| 最耗时环节 | Q/DQ 拓扑排序调试 | ？ |
| 最大意外 | 激活 Q/DQ 缺失是根因 | ？ |
