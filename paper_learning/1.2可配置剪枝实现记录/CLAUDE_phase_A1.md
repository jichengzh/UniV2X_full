# Phase A1: DepGraph Foundation (自定义剪枝器 + 梯度收集)

**覆盖任务**: A.2.1 (自定义剪枝器注册) + A.2.4 (梯度收集器)
**预计工时**: 2.5 天
**日期**: 2026-04-13

---

## 1. 阶段目标

使 UniV2X 中 4 个自定义 CUDA 注意力模块与 Torch-Pruning 的 DepGraph 依赖追踪兼容:

- **MSDeformableAttention3D** (`spatial_cross_attention.py:185-405`): Encoder 中的多尺度可变形注意力, `output_proj=None` (由外层 SCA 提供)
- **SpatialCrossAttention** (`spatial_cross_attention.py:31-181`): Encoder 中的空间交叉注意力, 内含 `deformable_attention` (MSDeformableAttention3D 实例)
- **TemporalSelfAttention** (`temporal_self_attention.py:25-269`): Encoder 中的时序自注意力, 有 `embed_dims*num_bev_queue` 的 concat 特殊逻辑
- **CustomMSDeformableAttention** (`decoder.py:133-345`): Decoder 中的多尺度可变形注意力, 有独立的 `output_proj`

同时实现梯度收集器, 为 Taylor/Hessian 重要性评估提供梯度数据.

**核心约束**: `sampling_offsets` 和 `attention_weights` 的输出维度由结构参数 (`num_heads*num_levels*num_points*2` 等) 决定, 绝不可被剪枝; 只有输入维度跟踪 `embed_dims`, 需要随剪枝传播.

---

## 2. 前置条件

| 前置条件 | 验证方式 |
|----------|----------|
| `torch-pruning>=1.4.0` 已安装 | `python -c "import torch_pruning; print(torch_pruning.__version__)"` |
| UniV2X 模型 checkpoint 可加载 | `torch.load('ckpts/univ2x_xxx.pth')` 成功 |
| 理解 DepGraph 的 `BasePruningFunc` 接口 | 阅读 `torch_pruning/pruner/function.py` 源码 |
| CUDA 编译环境可用 (mmcv CUDA ops) | `python -c "from mmcv.ops import multi_scale_deform_attn; print('OK')"` |
| mmdet3d 及 UniV2X 代码可正常 import | `python -c "from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import MSDeformableAttention3D"` |

---

## 3. 具体代码实现

### 3.1 custom_pruners.py

**文件路径**: `projects/mmdet3d_plugin/univ2x/pruning/custom_pruners.py`

```python
"""
UniV2X 自定义剪枝器 — 为 Torch-Pruning DepGraph 提供依赖追踪规则.

核心设计原则:
1. sampling_offsets / attention_weights 的 **输出维度不可变** (由 num_heads*num_levels*num_points 决定)
2. 只有输入维度跟踪 embed_dims, 需要随 pruning group 传播
3. value_proj / output_proj 的两个维度都跟踪 embed_dims
4. TemporalSelfAttention 的 sampling_offsets/attention_weights 输入维度是
   embed_dims*num_bev_queue (concat), 需要特殊的 double-index 映射
"""

from typing import Dict, List, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn

import torch_pruning as tp
from torch_pruning.pruner.function import BasePruningFunc

from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import (
    MSDeformableAttention3D,
    SpatialCrossAttention,
)
from projects.mmdet3d_plugin.univ2x.modules.temporal_self_attention import (
    TemporalSelfAttention,
)
from projects.mmdet3d_plugin.univ2x.modules.decoder import (
    CustomMSDeformableAttention,
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _expand_idxs_for_concat(idxs: List[int], embed_dims: int,
                            num_repeats: int) -> List[int]:
    """将 embed_dims 上的剪枝索引扩展为 concat 后的索引.

    TemporalSelfAttention 中 query = cat([value[:bs], query], dim=-1),
    生成 embed_dims * num_bev_queue 维度的输入.
    若对 embed_dims 维度剪枝索引为 [2, 5], embed_dims=256, num_repeats=2,
    则返回 [2, 5, 258, 261] (即 [i, i+256] for each i in idxs).

    Args:
        idxs: 在单个 embed_dims 空间中的剪枝通道索引.
        embed_dims: 原始嵌入维度.
        num_repeats: concat 重复次数 (num_bev_queue).

    Returns:
        扩展后的索引列表, 长度为 len(idxs) * num_repeats.
    """
    expanded = []
    for r in range(num_repeats):
        offset = r * embed_dims
        expanded.extend([idx + offset for idx in idxs])
    return sorted(expanded)


# ---------------------------------------------------------------------------
# MSDeformableAttention3D 剪枝器 (Encoder 内部可变形注意力)
# ---------------------------------------------------------------------------

class MSDeformableAttention3DPruner(BasePruningFunc):
    """处理 MSDeformableAttention3D (spatial_cross_attention.py:185-405).

    层结构:
        - sampling_offsets:  Linear(embed_dims, num_heads*num_levels*num_points*2) [L251-252]
        - attention_weights: Linear(embed_dims, num_heads*num_levels*num_points)  [L253-254]
        - value_proj:        Linear(embed_dims, embed_dims)                       [L255]
        - output_proj:       None (由外层 SpatialCrossAttention 提供)             [L227]

    剪枝策略:
        - sampling_offsets:  输出维度固定不动, 输入维度跟踪 embed_dims
        - attention_weights: 输出维度固定不动, 输入维度跟踪 embed_dims
        - value_proj:        输入/输出都跟踪 embed_dims
        - 不处理 output_proj (为 None)
    """

    TARGET_MODULE = MSDeformableAttention3D

    def check(self, layer: MSDeformableAttention3D, idxs: List[int],
              to_output: bool, **kwargs) -> List[int]:
        return idxs

    def prune_in_channels(self, layer: MSDeformableAttention3D,
                          idxs: Sequence[int]) -> nn.Module:
        """剪枝 embed_dims 输入通道.

        影响:
        - sampling_offsets.weight[:, idxs] (输入维度)
        - attention_weights.weight[:, idxs] (输入维度)
        - value_proj.weight[:, idxs] (输入维度)
        """
        idxs_tensor = torch.tensor(idxs, dtype=torch.long,
                                   device=layer.sampling_offsets.weight.device)

        # sampling_offsets: 只剪输入维度
        _prune_linear_in_channels(layer.sampling_offsets, idxs_tensor)
        # attention_weights: 只剪输入维度
        _prune_linear_in_channels(layer.attention_weights, idxs_tensor)
        # value_proj: 剪输入维度
        _prune_linear_in_channels(layer.value_proj, idxs_tensor)

        # 更新 embed_dims 属性
        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def prune_out_channels(self, layer: MSDeformableAttention3D,
                           idxs: Sequence[int]) -> nn.Module:
        """剪枝 embed_dims 输出通道.

        影响:
        - value_proj.weight[idxs, :] 和 value_proj.bias[idxs] (输出维度)
        - sampling_offsets 和 attention_weights 输出维度不动
        - output_proj 为 None, 不处理
        """
        idxs_tensor = torch.tensor(idxs, dtype=torch.long,
                                   device=layer.value_proj.weight.device)

        # value_proj: 剪输出维度
        _prune_linear_out_channels(layer.value_proj, idxs_tensor)

        # 注意: 不剪 sampling_offsets / attention_weights 的输出维度
        # 因为输出维度 = num_heads * num_levels * num_points * {2,1}, 是固定的

        # 更新 embed_dims 属性
        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def get_in_channel_groups(self, layer: MSDeformableAttention3D) -> int:
        return 1

    def get_out_channel_groups(self, layer: MSDeformableAttention3D) -> int:
        return 1

    def get_in_channels(self, layer: MSDeformableAttention3D) -> int:
        return layer.embed_dims

    def get_out_channels(self, layer: MSDeformableAttention3D) -> int:
        return layer.embed_dims


# ---------------------------------------------------------------------------
# SpatialCrossAttention 剪枝器
# ---------------------------------------------------------------------------

class SpatialCrossAttentionPruner(BasePruningFunc):
    """处理 SpatialCrossAttention (spatial_cross_attention.py:31-181).

    层结构:
        - output_proj:           Linear(embed_dims, embed_dims)  [L66]
        - deformable_attention:  MSDeformableAttention3D 实例     [L63]

    剪枝策略:
        - output_proj: 输入/输出都跟踪 embed_dims
        - deformable_attention: 由 MSDeformableAttention3DPruner 处理,
          DepGraph 会自动通过依赖关系传播到内部
    """

    TARGET_MODULE = SpatialCrossAttention

    def check(self, layer: SpatialCrossAttention, idxs: List[int],
              to_output: bool, **kwargs) -> List[int]:
        return idxs

    def prune_in_channels(self, layer: SpatialCrossAttention,
                          idxs: Sequence[int]) -> nn.Module:
        idxs_tensor = torch.tensor(idxs, dtype=torch.long,
                                   device=layer.output_proj.weight.device)
        _prune_linear_in_channels(layer.output_proj, idxs_tensor)
        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def prune_out_channels(self, layer: SpatialCrossAttention,
                           idxs: Sequence[int]) -> nn.Module:
        idxs_tensor = torch.tensor(idxs, dtype=torch.long,
                                   device=layer.output_proj.weight.device)
        _prune_linear_out_channels(layer.output_proj, idxs_tensor)
        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def get_in_channel_groups(self, layer: SpatialCrossAttention) -> int:
        return 1

    def get_out_channel_groups(self, layer: SpatialCrossAttention) -> int:
        return 1

    def get_in_channels(self, layer: SpatialCrossAttention) -> int:
        return layer.embed_dims

    def get_out_channels(self, layer: SpatialCrossAttention) -> int:
        return layer.embed_dims


# ---------------------------------------------------------------------------
# TemporalSelfAttention 剪枝器
# ---------------------------------------------------------------------------

class TemporalSelfAttentionPruner(BasePruningFunc):
    """处理 TemporalSelfAttention (temporal_self_attention.py:25-269).

    层结构:
        - sampling_offsets:  Linear(embed_dims*num_bev_queue,
                                    num_bev_queue*num_heads*num_levels*num_points*2) [L98-99]
        - attention_weights: Linear(embed_dims*num_bev_queue,
                                    num_bev_queue*num_heads*num_levels*num_points)   [L100-101]
        - value_proj:        Linear(embed_dims, embed_dims)                          [L102]
        - output_proj:       Linear(embed_dims, embed_dims)                          [L103]

    关键挑战 (embed_dims*2 concat):
        在 forward() L194: `query = torch.cat([value[:bs], query], -1)`
        这使得 sampling_offsets / attention_weights 的输入维度 = embed_dims * num_bev_queue.
        当剪枝 embed_dims 中的索引 idxs 时, 需要在 concat 空间中映射为:
        [idxs, idxs + embed_dims] (即每个 bev_queue 段都要对应剪枝).

    剪枝策略:
        - sampling_offsets:  输出维度固定, 输入维度用 double-index 映射
        - attention_weights: 同上
        - value_proj:        输入/输出都跟踪 embed_dims
        - output_proj:       输入/输出都跟踪 embed_dims
    """

    TARGET_MODULE = TemporalSelfAttention

    def check(self, layer: TemporalSelfAttention, idxs: List[int],
              to_output: bool, **kwargs) -> List[int]:
        return idxs

    def prune_in_channels(self, layer: TemporalSelfAttention,
                          idxs: Sequence[int]) -> nn.Module:
        """剪枝 embed_dims 输入通道.

        sampling_offsets/attention_weights 输入维度 = embed_dims * num_bev_queue,
        需要用 _expand_idxs_for_concat 扩展索引.
        """
        device = layer.value_proj.weight.device
        idxs_list = list(idxs)
        idxs_tensor = torch.tensor(idxs_list, dtype=torch.long, device=device)

        # sampling_offsets: 输入维度 = embed_dims * num_bev_queue
        expanded_idxs = _expand_idxs_for_concat(
            idxs_list, layer.embed_dims, layer.num_bev_queue)
        expanded_tensor = torch.tensor(expanded_idxs, dtype=torch.long,
                                       device=device)
        _prune_linear_in_channels(layer.sampling_offsets, expanded_tensor)

        # attention_weights: 同样的 concat 结构
        _prune_linear_in_channels(layer.attention_weights, expanded_tensor)

        # value_proj: 标准 embed_dims
        _prune_linear_in_channels(layer.value_proj, idxs_tensor)

        # output_proj: 标准 embed_dims 输入
        _prune_linear_in_channels(layer.output_proj, idxs_tensor)

        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def prune_out_channels(self, layer: TemporalSelfAttention,
                           idxs: Sequence[int]) -> nn.Module:
        """剪枝 embed_dims 输出通道.

        sampling_offsets/attention_weights 输出维度固定, 不处理.
        """
        device = layer.value_proj.weight.device
        idxs_tensor = torch.tensor(list(idxs), dtype=torch.long, device=device)

        # value_proj: 输出维度
        _prune_linear_out_channels(layer.value_proj, idxs_tensor)

        # output_proj: 输出维度
        _prune_linear_out_channels(layer.output_proj, idxs_tensor)

        # sampling_offsets / attention_weights 输出维度固定, 不剪

        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def get_in_channel_groups(self, layer: TemporalSelfAttention) -> int:
        return 1

    def get_out_channel_groups(self, layer: TemporalSelfAttention) -> int:
        return 1

    def get_in_channels(self, layer: TemporalSelfAttention) -> int:
        return layer.embed_dims

    def get_out_channels(self, layer: TemporalSelfAttention) -> int:
        return layer.embed_dims


# ---------------------------------------------------------------------------
# CustomMSDeformableAttention 剪枝器 (Decoder)
# ---------------------------------------------------------------------------

class CustomMSDeformableAttentionPruner(BasePruningFunc):
    """处理 CustomMSDeformableAttention (decoder.py:133-345).

    层结构 (与 MSDeformableAttention3D 类似, 但有 output_proj):
        - sampling_offsets:  Linear(embed_dims, num_heads*num_levels*num_points*2) [L201-202]
        - attention_weights: Linear(embed_dims, num_heads*num_levels*num_points)   [L203-204]
        - value_proj:        Linear(embed_dims, embed_dims)                        [L205]
        - output_proj:       Linear(embed_dims, embed_dims)                        [L206]

    剪枝策略:
        - sampling_offsets:  输出维度固定, 输入维度跟踪 embed_dims
        - attention_weights: 输出维度固定, 输入维度跟踪 embed_dims
        - value_proj:        输入/输出都跟踪 embed_dims
        - output_proj:       输入/输出都跟踪 embed_dims
    """

    TARGET_MODULE = CustomMSDeformableAttention

    def check(self, layer: CustomMSDeformableAttention, idxs: List[int],
              to_output: bool, **kwargs) -> List[int]:
        return idxs

    def prune_in_channels(self, layer: CustomMSDeformableAttention,
                          idxs: Sequence[int]) -> nn.Module:
        device = layer.value_proj.weight.device
        idxs_tensor = torch.tensor(list(idxs), dtype=torch.long, device=device)

        _prune_linear_in_channels(layer.sampling_offsets, idxs_tensor)
        _prune_linear_in_channels(layer.attention_weights, idxs_tensor)
        _prune_linear_in_channels(layer.value_proj, idxs_tensor)
        _prune_linear_in_channels(layer.output_proj, idxs_tensor)

        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def prune_out_channels(self, layer: CustomMSDeformableAttention,
                           idxs: Sequence[int]) -> nn.Module:
        device = layer.value_proj.weight.device
        idxs_tensor = torch.tensor(list(idxs), dtype=torch.long, device=device)

        # value_proj: 输出维度
        _prune_linear_out_channels(layer.value_proj, idxs_tensor)
        # output_proj: 输出维度
        _prune_linear_out_channels(layer.output_proj, idxs_tensor)
        # sampling_offsets / attention_weights 输出维度固定不动

        layer.embed_dims = layer.embed_dims - len(idxs)
        return layer

    def get_in_channel_groups(self, layer: CustomMSDeformableAttention) -> int:
        return 1

    def get_out_channel_groups(self, layer: CustomMSDeformableAttention) -> int:
        return 1

    def get_in_channels(self, layer: CustomMSDeformableAttention) -> int:
        return layer.embed_dims

    def get_out_channels(self, layer: CustomMSDeformableAttention) -> int:
        return layer.embed_dims


# ---------------------------------------------------------------------------
# Linear 层剪枝辅助函数
# ---------------------------------------------------------------------------

def _prune_linear_in_channels(linear: nn.Linear,
                              idxs: torch.Tensor) -> None:
    """就地剪除 Linear 层的输入通道 (weight 的第 1 维).

    Args:
        linear: 要剪枝的 Linear 层.
        idxs: 要移除的输入通道索引 (1-D LongTensor).
    """
    keep = torch.ones(linear.in_features, dtype=torch.bool,
                      device=linear.weight.device)
    keep[idxs] = False
    linear.weight = nn.Parameter(linear.weight[:, keep])
    linear.in_features = linear.weight.shape[1]


def _prune_linear_out_channels(linear: nn.Linear,
                               idxs: torch.Tensor) -> None:
    """就地剪除 Linear 层的输出通道 (weight 的第 0 维 + bias).

    Args:
        linear: 要剪枝的 Linear 层.
        idxs: 要移除的输出通道索引 (1-D LongTensor).
    """
    keep = torch.ones(linear.out_features, dtype=torch.bool,
                      device=linear.weight.device)
    keep[idxs] = False
    linear.weight = nn.Parameter(linear.weight[keep])
    if linear.bias is not None:
        linear.bias = nn.Parameter(linear.bias[keep])
    linear.out_features = linear.weight.shape[0]


# ---------------------------------------------------------------------------
# 注册接口
# ---------------------------------------------------------------------------

def register_univ2x_pruners() -> Dict[Type[nn.Module], BasePruningFunc]:
    """返回 UniV2X 自定义模块到剪枝器的映射字典.

    用于传入 tp.pruner.MetaPruner 的 customized_pruners 参数:

        pruner = tp.pruner.MetaPruner(
            model, example_inputs,
            importance=...,
            customized_pruners=register_univ2x_pruners(),
            ...
        )

    Returns:
        Dict[Type[nn.Module], BasePruningFunc]: 模块类 -> 剪枝器实例 的映射.
    """
    return {
        MSDeformableAttention3D: MSDeformableAttention3DPruner(),
        SpatialCrossAttention: SpatialCrossAttentionPruner(),
        TemporalSelfAttention: TemporalSelfAttentionPruner(),
        CustomMSDeformableAttention: CustomMSDeformableAttentionPruner(),
    }
```

---

### 3.2 grad_collector.py

**文件路径**: `projects/mmdet3d_plugin/univ2x/pruning/grad_collector.py`

```python
"""
梯度收集器 — 为 Taylor/Hessian 重要性评估提供梯度数据.

UniV2X 基于 mmdet3d 框架, forward 返回 loss dict (而非单一标量 loss).
此模块封装梯度收集逻辑, 兼容 mmdet3d 的 train_step 接口.
"""

import logging
from typing import Any, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def collect_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    num_batches: int = 32,
    target_loss_keys: Optional[List[str]] = None,
    grad_clip_value: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """对模型做 num_batches 次前向+反向传播, 累积所有可剪枝参数的梯度.

    Args:
        model: UniV2X 模型 (需要已加载权重).
        dataloader: 训练数据加载器 (mmdet3d 格式, 每个 batch 是 dict).
        num_batches: 累积梯度的 batch 数量. 默认 32.
        target_loss_keys: 用于反向传播的 loss 键名列表.
            若为 None, 则对 loss dict 中所有以 'loss' 开头的键求和.
        grad_clip_value: 梯度裁剪阈值, 防止 NaN. 默认 1.0.
        device: 目标设备. 若为 None 则自动检测.

    Returns:
        Dict[str, torch.Tensor]: 参数名 -> 累积梯度 的映射.
            只包含 requires_grad=True 且梯度非 None 的参数.

    Raises:
        RuntimeError: 如果所有 batch 都产生了 NaN 梯度.
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    model.zero_grad()

    # 确保所有参数的 grad 被清零
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    nan_batch_count = 0
    processed_batches = 0

    for batch_idx, data_batch in enumerate(dataloader):
        if processed_batches >= num_batches:
            break

        try:
            # mmdet3d 的 train_step 接口
            # data_batch 是 dict, 包含 img, img_metas 等
            if hasattr(model, 'train_step'):
                # train_step 需要 optimizer 参数, 但我们只需要梯度
                # 用 forward + parse_losses 替代
                losses = model(**data_batch)
            elif hasattr(model, 'forward_train'):
                losses = model.forward_train(**data_batch)
            else:
                losses = model(**data_batch)

            # 解析 loss dict
            total_loss = _parse_losses(losses, target_loss_keys)

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(
                    f"Batch {batch_idx}: loss 为 NaN/Inf, 跳过")
                nan_batch_count += 1
                model.zero_grad()
                continue

            # 反向传播 (累积梯度, 不 zero_grad)
            total_loss.backward()

            # 梯度裁剪: 逐参数检查并裁剪
            _clip_and_check_gradients(model, grad_clip_value)

            processed_batches += 1
            logger.info(
                f"Batch {batch_idx}/{num_batches}: "
                f"loss={total_loss.item():.4f}")

        except RuntimeError as e:
            logger.warning(f"Batch {batch_idx} 出错: {e}, 跳过")
            model.zero_grad()
            continue

    if processed_batches == 0:
        raise RuntimeError(
            f"所有 {nan_batch_count} 个 batch 都产生了 NaN 梯度, "
            "请检查模型权重和数据.")

    # 收集梯度
    grad_dict: Dict[str, torch.Tensor] = {}
    zero_grad_params: List[str] = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # 平均梯度
            avg_grad = param.grad.data / processed_batches
            if torch.all(avg_grad == 0):
                zero_grad_params.append(name)
            grad_dict[name] = avg_grad.clone()

    if zero_grad_params:
        logger.warning(
            f"以下 {len(zero_grad_params)} 个参数的梯度为全零: "
            f"{zero_grad_params[:10]}...")

    logger.info(
        f"梯度收集完成: {processed_batches} batches, "
        f"{len(grad_dict)} 个参数有梯度, "
        f"{len(zero_grad_params)} 个全零梯度")

    return grad_dict


def _parse_losses(
    losses: Dict[str, torch.Tensor],
    target_keys: Optional[List[str]] = None,
) -> torch.Tensor:
    """将 loss dict 转换为单一标量 loss.

    Args:
        losses: 模型返回的 loss 字典.
        target_keys: 指定的 loss 键名. 若为 None, 使用所有以 'loss' 开头的键.

    Returns:
        聚合后的标量 loss.
    """
    if target_keys is not None:
        selected = {k: v for k, v in losses.items() if k in target_keys}
    else:
        selected = {k: v for k, v in losses.items()
                    if 'loss' in k.lower()}

    if not selected:
        raise ValueError(
            f"在 loss dict 中找不到匹配的 loss 键. "
            f"可用键: {list(losses.keys())}")

    total = sum(selected.values())
    return total


def _clip_and_check_gradients(model: nn.Module,
                              clip_value: float) -> None:
    """逐参数检查梯度是否为 NaN, 并应用梯度裁剪.

    NaN 梯度会被置零, 非 NaN 梯度被裁剪到 [-clip_value, clip_value].

    Args:
        model: 要检查的模型.
        clip_value: 梯度裁剪阈值.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                logger.warning(f"参数 {name} 梯度包含 NaN, 置零")
                param.grad.data.zero_()
            else:
                param.grad.data.clamp_(-clip_value, clip_value)
```

---

### 3.3 __init__.py

**文件路径**: `projects/mmdet3d_plugin/univ2x/pruning/__init__.py`

```python
"""UniV2X 剪枝模块 — 自定义剪枝器与梯度收集."""

from .custom_pruners import (
    MSDeformableAttention3DPruner,
    SpatialCrossAttentionPruner,
    TemporalSelfAttentionPruner,
    CustomMSDeformableAttentionPruner,
    register_univ2x_pruners,
)
from .grad_collector import collect_gradients

__all__ = [
    'MSDeformableAttention3DPruner',
    'SpatialCrossAttentionPruner',
    'TemporalSelfAttentionPruner',
    'CustomMSDeformableAttentionPruner',
    'register_univ2x_pruners',
    'collect_gradients',
]
```

---

## 4. 代码检测方案

### 4.1 单元测试 1: DepGraph 构建

**目标**: 验证 DepGraph 能在包含 MSDeformableAttention3D 的模型上成功构建依赖图.

```python
"""tests/test_pruning/test_depgraph_build.py"""
import pytest
import torch
import torch.nn as nn
import torch_pruning as tp

from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import (
    MSDeformableAttention3D,
)
from projects.mmdet3d_plugin.univ2x.pruning import register_univ2x_pruners


class MinimalModelWithMSDA3D(nn.Module):
    """包含 MSDeformableAttention3D 的最小模型, 用于测试 DepGraph."""

    def __init__(self, embed_dims: int = 256, num_heads: int = 8,
                 num_levels: int = 4, num_points: int = 8):
        super().__init__()
        self.input_proj = nn.Linear(embed_dims, embed_dims)
        self.msda = MSDeformableAttention3D(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        # MSDeformableAttention3D 的 output_proj=None, 需要外部提供
        self.msda.output_proj = nn.Linear(embed_dims, embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points

    def forward(self, x):
        # 简化的前向: 只用 value_proj + output_proj + ffn
        # 绕过 CUDA deformable attention (不需要实际计算)
        out = self.input_proj(x)
        out = self.msda.value_proj(out)
        out = self.msda.output_proj(out)
        out = self.ffn(out)
        return out


def test_depgraph_build_succeeds():
    """DepGraph.build_dependency() 在自定义 pruner 注册后应成功."""
    model = MinimalModelWithMSDA3D(embed_dims=256)
    example_input = torch.randn(1, 100, 256)

    DG = tp.DependencyGraph()
    DG.build_dependency(
        model,
        example_inputs=example_input,
        customized_pruners=register_univ2x_pruners(),
    )

    # 如果没有抛异常, 说明依赖图构建成功
    assert DG is not None
```

### 4.2 单元测试 2: 剪枝组排除 sampling_offsets

**目标**: 验证 `get_pruning_group()` 对 `value_proj` 不会包含 `sampling_offsets`.

```python
def test_pruning_group_excludes_sampling_offsets():
    """value_proj 的剪枝组不应包含 sampling_offsets/attention_weights."""
    model = MinimalModelWithMSDA3D(embed_dims=256)
    example_input = torch.randn(1, 100, 256)

    DG = tp.DependencyGraph()
    DG.build_dependency(
        model,
        example_inputs=example_input,
        customized_pruners=register_univ2x_pruners(),
    )

    # 获取 value_proj 的剪枝组
    group = DG.get_pruning_group(
        model.msda.value_proj,
        tp.prune_linear_out_channels,
        idxs=[0, 1, 2, 3],
    )

    # 检查组中不包含 sampling_offsets 和 attention_weights 的输出维度剪枝
    group_str = str(group)
    # sampling_offsets 的输出维度不应出现在组中
    for dep, _ in group:
        target_module = dep.target.module
        if target_module is model.msda.sampling_offsets:
            # 如果出现, 应该只是输入维度的剪枝, 不是输出维度
            assert dep.handler.__name__ != 'prune_out_channels', \
                "sampling_offsets 的输出维度不应被剪枝!"
```

### 4.3 单元测试 3: FFN 层剪枝维度变化

```python
def test_ffn_pruning_dimensions():
    """剪枝 FFN 后, 模型的中间维度应正确变化."""
    model = MinimalModelWithMSDA3D(embed_dims=256)
    example_input = torch.randn(1, 100, 256)

    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs=example_input,
        importance=tp.importance.MagnitudeImportance(),
        pruning_ratio=0.25,
        customized_pruners=register_univ2x_pruners(),
    )

    pruner.step()

    # 验证 FFN 第一层输入维度 = 剪枝后的 embed_dims
    new_embed_dims = model.input_proj.out_features
    assert new_embed_dims == 192  # 256 * 0.75 = 192
    assert model.ffn[0].in_features == new_embed_dims
    assert model.ffn[2].out_features == new_embed_dims

    # 验证模型仍可前向传播
    new_input = torch.randn(1, 100, new_embed_dims)
    output = model(new_input)
    assert output.shape == (1, 100, new_embed_dims)
```

### 4.4 单元测试 4: 梯度收集非零

```python
def test_gradient_collection_nonzero():
    """collect_gradients 应产生非零梯度."""
    from projects.mmdet3d_plugin.univ2x.pruning import collect_gradients

    # 用简单模型模拟
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # 构造简单 dataloader
    dataset = torch.utils.data.TensorDataset(
        torch.randn(64, 64),
        torch.randint(0, 10, (64,)),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

    # 需要包装为返回 loss dict 的模型
    class WrappedModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x, labels):
            logits = self.backbone(x)
            loss = self.criterion(logits, labels)
            return {'loss_cls': loss}

    wrapped = WrappedModel(model)

    # 调整 dataloader 格式为 dict
    class DictDataLoader:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for x, y in self.loader:
                yield {'x': x, 'labels': y}

    grad_dict = collect_gradients(
        wrapped,
        DictDataLoader(dataloader),
        num_batches=4,
    )

    # 至少应有一些非零梯度
    nonzero_count = sum(
        1 for g in grad_dict.values()
        if torch.any(g != 0)
    )
    assert nonzero_count > 0, "所有参数梯度都为零!"
```

### 4.5 TSA double-index 映射测试

```python
def test_tsa_expand_idxs_for_concat():
    """验证 _expand_idxs_for_concat 的索引扩展逻辑."""
    from projects.mmdet3d_plugin.univ2x.pruning.custom_pruners import (
        _expand_idxs_for_concat,
    )

    # embed_dims=8, num_bev_queue=2, 剪枝索引 [1, 3]
    result = _expand_idxs_for_concat([1, 3], embed_dims=8, num_repeats=2)
    expected = [1, 3, 9, 11]  # [1, 3, 1+8, 3+8]
    assert result == expected, f"Expected {expected}, got {result}"

    # 边界情况: 空索引
    result_empty = _expand_idxs_for_concat([], embed_dims=256, num_repeats=2)
    assert result_empty == []

    # num_repeats=1 时应等同原索引
    result_single = _expand_idxs_for_concat([0, 5], embed_dims=256, num_repeats=1)
    assert result_single == [0, 5]
```

---

## 5. Debug 方案

### 5.1 DepGraph 追踪失败 (CUDA 自定义算子)

**症状**: `DG.build_dependency()` 抛出 `RuntimeError` 或静默跳过 CUDA ops.

**诊断步骤**:

1. **启用 verbose 模式**:
   ```python
   DG = tp.DependencyGraph()
   DG.build_dependency(
       model, example_inputs=example_input,
       customized_pruners=register_univ2x_pruners(),
       verbose=True,  # 打印追踪过程
   )
   ```

2. **检查哪些模块被跳过**:
   ```python
   # 列出 DepGraph 中已识别的模块
   for module_name, module in model.named_modules():
       if module in DG._module_to_node:
           print(f"[OK] {module_name}: {type(module).__name__}")
       else:
           print(f"[SKIP] {module_name}: {type(module).__name__}")
   ```

3. **CPU 降级测试**: 将模型移到 CPU, 使 `MultiScaleDeformableAttnFunction.apply` 走 PyTorch 纯 Python 路径 (`multi_scale_deformable_attn_pytorch`), 确认问题是否来自 CUDA op:
   ```python
   model_cpu = model.cpu()
   example_input_cpu = example_input.cpu()
   DG.build_dependency(model_cpu, example_inputs=example_input_cpu, ...)
   ```

4. **绕过方案**: 如果 CUDA op 完全阻塞追踪, 在简化的 `forward` 中用纯 PyTorch 代替:
   ```python
   # 临时 monkey-patch forward, 仅用于 DepGraph 构建
   original_forward = MSDeformableAttention3D.forward
   MSDeformableAttention3D.forward = _simplified_forward_for_tracing
   DG.build_dependency(...)
   MSDeformableAttention3D.forward = original_forward
   ```

### 5.2 剪枝后维度不匹配

**症状**: 剪枝后 `model(input)` 抛 `RuntimeError: size mismatch`.

**诊断步骤**:

```python
def verify_dimensions_after_pruning(model: nn.Module) -> List[str]:
    """遍历所有 Linear/Conv 层, 检查相邻层维度是否一致."""
    issues = []
    prev_out = None
    prev_name = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if prev_out is not None and module.in_features != prev_out:
                issues.append(
                    f"维度不匹配: {prev_name}.out={prev_out} -> "
                    f"{name}.in={module.in_features}")
            prev_out = module.out_features
            prev_name = name

    return issues

# 使用
issues = verify_dimensions_after_pruning(model)
for issue in issues:
    print(f"[ERROR] {issue}")
```

### 5.3 TSA concat 维度错误

**症状**: `TemporalSelfAttention.forward()` 在 `sampling_offsets(query)` 处报维度错误.

**诊断**:

```python
def check_tsa_consistency(tsa: TemporalSelfAttention) -> None:
    """检查 TSA 的 embed_dims * num_bev_queue 一致性."""
    expected_in = tsa.embed_dims * tsa.num_bev_queue

    actual_so = tsa.sampling_offsets.in_features
    actual_aw = tsa.attention_weights.in_features

    assert actual_so == expected_in, \
        f"sampling_offsets.in_features={actual_so}, " \
        f"expected={expected_in} (embed_dims={tsa.embed_dims} * " \
        f"num_bev_queue={tsa.num_bev_queue})"

    assert actual_aw == expected_in, \
        f"attention_weights.in_features={actual_aw}, " \
        f"expected={expected_in}"

    print(f"[OK] TSA 一致性检查通过: embed_dims={tsa.embed_dims}, "
          f"concat_dim={expected_in}")
```

**关键不变量**: 剪枝后必须满足:
- `sampling_offsets.in_features == embed_dims * num_bev_queue`
- `attention_weights.in_features == embed_dims * num_bev_queue`
- `value_proj.in_features == embed_dims`
- `value_proj.out_features == embed_dims`
- `output_proj.in_features == embed_dims`
- `output_proj.out_features == embed_dims`

### 5.4 梯度累积产生 NaN

**症状**: `collect_gradients` 返回的梯度包含 NaN.

**诊断步骤**:

1. **启用异常检测**:
   ```python
   torch.autograd.set_detect_anomaly(True)
   ```

2. **逐层检查**:
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           if torch.any(torch.isnan(param.grad)):
               print(f"[NaN] {name}: grad shape={param.grad.shape}")
           elif torch.any(torch.isinf(param.grad)):
               print(f"[Inf] {name}: grad shape={param.grad.shape}")
   ```

3. **降低 grad_clip_value**: 从 1.0 降到 0.1, 观察是否有改善.

4. **减少 num_batches**: 先用 1 个 batch 测试, 确认单次梯度正常后再累积.

5. **FP32 强制**: 确保模型在 FP32 下运行:
   ```python
   model = model.float()
   ```

---

## 6. 验收标准

| # | 验收项 | 验证方式 | 状态 |
|---|--------|----------|------|
| 1 | `DG.build_dependency()` 在包含 4 种自定义模块的模型上成功, 无报错 | 运行测试 4.1 | [ ] |
| 2 | `get_pruning_group()` 对 FFN/value_proj/output_proj 返回有效组 | 运行测试 4.2 + 手动检查 | [ ] |
| 3 | `sampling_offsets` 和 `attention_weights` 的**输出维度**永远不出现在剪枝组中 | 运行测试 4.2, 检查 group 中无对应条目 | [ ] |
| 4 | TSA 的 `_expand_idxs_for_concat` 正确生成 double-index | 运行测试 4.5 | [ ] |
| 5 | FFN 剪枝后维度正确变化, 模型可正常前向传播 | 运行测试 4.3 | [ ] |
| 6 | `collect_gradients` 产生非零 `.grad` | 运行测试 4.4 | [ ] |
| 7 | 剪枝后 TSA 的 `embed_dims*num_bev_queue` 一致性保持 | 运行 5.3 的 `check_tsa_consistency` | [ ] |
| 8 | 所有单元测试通过 | `pytest tests/test_pruning/ -v` | [ ] |

---

## 7. 风险与缓解

| # | 风险 | 影响 | 概率 | 缓解措施 |
|---|------|------|------|----------|
| 1 | CUDA 自定义算子 (`MultiScaleDeformableAttnFunction`) 完全阻塞 DepGraph 追踪 | 高: 依赖图无法构建 | 中 | 使用 `customized_pruners` + `ignored_layers` 参数绕过; 为追踪阶段提供简化的 `forward` (仅走 Linear 路径, 跳过 CUDA op) |
| 2 | TSA 的 `embed_dims*2` concat 在剪枝后维度不一致 | 高: 模型推理崩溃 | 中 | `TemporalSelfAttentionPruner` 中使用 `_expand_idxs_for_concat` 做显式双索引映射; 剪枝后立即运行 `check_tsa_consistency` 验证 |
| 3 | DepGraph 无法自动追踪 `MSDeformableAttention3D.output_proj = None` 的情况 | 中: 依赖链断裂 | 高 | `MSDeformableAttention3DPruner` 不处理 `output_proj`; 由外层 `SpatialCrossAttentionPruner` 的 `output_proj` 承接依赖 |
| 4 | mmdet3d 的 `train_step` 接口与标准 `forward` 不兼容 | 低: 梯度收集失败 | 低 | `collect_gradients` 中按优先级尝试 `train_step` -> `forward_train` -> `forward`; 用 `_parse_losses` 统一处理 loss dict |
| 5 | 大模型梯度累积导致显存溢出 | 中: OOM | 中 | 每个 batch 后 `del data_batch`; 减少 `num_batches`; 使用 `torch.cuda.empty_cache()` |
| 6 | `num_heads` 变化导致 CUDA deformable attention 内部 `dim_per_head` 不对齐 | 高: 运行时崩溃 | 低 | 本阶段只剪 `embed_dims` (通道数), 不剪 `num_heads`; `dim_per_head = embed_dims // num_heads` 在剪枝后需要保证整除, 在 `check()` 方法中验证 |
| 7 | Torch-Pruning 版本 API 变动 (1.4.x -> 1.5.x) | 低: 接口不兼容 | 低 | 在 requirements 中锁定 `torch-pruning>=1.4.0,<2.0` |

---

## 附录: 模块维度速查表

### MSDeformableAttention3D (Encoder, spatial_cross_attention.py:185-405)

| 子层 | 输入维度 | 输出维度 | 随 embed_dims 剪枝? |
|------|----------|----------|---------------------|
| `sampling_offsets` | embed_dims (256) | num_heads * num_levels * num_points * 2 (512) | 输入: 是, 输出: **否** |
| `attention_weights` | embed_dims (256) | num_heads * num_levels * num_points (256) | 输入: 是, 输出: **否** |
| `value_proj` | embed_dims (256) | embed_dims (256) | 输入: 是, 输出: 是 |
| `output_proj` | None | None | N/A (由外层 SCA 提供) |

### SpatialCrossAttention (Encoder, spatial_cross_attention.py:31-181)

| 子层 | 输入维度 | 输出维度 | 随 embed_dims 剪枝? |
|------|----------|----------|---------------------|
| `output_proj` | embed_dims (256) | embed_dims (256) | 输入: 是, 输出: 是 |
| `deformable_attention` | (MSDeformableAttention3D) | - | 由对应 pruner 处理 |

### TemporalSelfAttention (Encoder, temporal_self_attention.py:25-269)

| 子层 | 输入维度 | 输出维度 | 随 embed_dims 剪枝? |
|------|----------|----------|---------------------|
| `sampling_offsets` | embed_dims * num_bev_queue (512) | num_bev_queue * num_heads * num_levels * num_points * 2 (512) | 输入: 是 (double-index), 输出: **否** |
| `attention_weights` | embed_dims * num_bev_queue (512) | num_bev_queue * num_heads * num_levels * num_points (256) | 输入: 是 (double-index), 输出: **否** |
| `value_proj` | embed_dims (256) | embed_dims (256) | 输入: 是, 输出: 是 |
| `output_proj` | embed_dims (256) | embed_dims (256) | 输入: 是, 输出: 是 |

### CustomMSDeformableAttention (Decoder, decoder.py:133-345)

| 子层 | 输入维度 | 输出维度 | 随 embed_dims 剪枝? |
|------|----------|----------|---------------------|
| `sampling_offsets` | embed_dims (256) | num_heads * num_levels * num_points * 2 (256) | 输入: 是, 输出: **否** |
| `attention_weights` | embed_dims (256) | num_heads * num_levels * num_points (128) | 输入: 是, 输出: **否** |
| `value_proj` | embed_dims (256) | embed_dims (256) | 输入: 是, 输出: 是 |
| `output_proj` | embed_dims (256) | embed_dims (256) | 输入: 是, 输出: 是 |
