"""
自定义剪枝器：为 UniV2X 的自定义 CUDA 算子模块注册 DepGraph 兼容的剪枝逻辑。

核心问题：UniV2X 使用 MultiScaleDeformableAttnFunction (CUDA 自定义算子)，
DepGraph 的 torch.fx 追踪无法穿透这些算子。需要手动告诉 DepGraph 每个模块的
输入/输出通道依赖关系。

关键约束：
- sampling_offsets 的输出维度 = num_heads * num_levels * num_points * 2，与 embed_dims 无关，不可剪枝输出
- attention_weights 的输出维度 = num_heads * num_levels * num_points，同上
- 但当上游 embed_dims 被剪枝时，sampling_offsets/attention_weights 的输入维度需要同步调整
- value_proj/output_proj 的输入输出维度均与 embed_dims 绑定，可剪枝
"""
from typing import Sequence

import torch.nn as nn
import torch_pruning as tp
from torch_pruning.pruner.function import (
    prune_linear_in_channels,
    prune_linear_out_channels,
)


class MSDeformableAttention3DPruner(tp.BasePruningFunc):
    """处理 MSDeformableAttention3D 模块 (spatial_cross_attention.py L185-405)。

    结构：
      - sampling_offsets:  Linear(embed_dims, num_heads*num_levels*num_points*2)  -- 输出维度固定
      - attention_weights: Linear(embed_dims, num_heads*num_levels*num_points)    -- 输出维度固定
      - value_proj:        Linear(embed_dims, embed_dims)                         -- 可剪枝
      - output_proj:       None (由外部 SpatialCrossAttention 持有)
    """

    TARGET_MODULES = ()  # 动态注册，不在这里硬编码

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        # 输出通道 = value_proj 的输出维度
        prune_linear_out_channels(layer.value_proj, idxs)
        # sampling_offsets/attention_weights 的输出维度与 embed_dims 无关，不动
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        # 输入通道 = embed_dims，影响 value_proj/sampling_offsets/attention_weights 的输入
        prune_linear_in_channels(layer.value_proj, idxs)
        prune_linear_in_channels(layer.sampling_offsets, idxs)
        prune_linear_in_channels(layer.attention_weights, idxs)
        return layer

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.value_proj.out_features

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.value_proj.in_features


class SpatialCrossAttentionPruner(tp.BasePruningFunc):
    """处理 SpatialCrossAttention 模块 (spatial_cross_attention.py L31-181)。

    结构：
      - deformable_attention: MSDeformableAttention3D (内部子模块，有自己的 pruner)
      - output_proj: Linear(embed_dims, embed_dims) -- 可剪枝
      - dropout: Dropout
    """

    TARGET_MODULES = ()

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        prune_linear_out_channels(layer.output_proj, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        prune_linear_in_channels(layer.output_proj, idxs)
        return layer

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.output_proj.out_features

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.output_proj.in_features


class TemporalSelfAttentionPruner(tp.BasePruningFunc):
    """处理 TemporalSelfAttention 模块 (temporal_self_attention.py L25-269)。

    结构：
      - sampling_offsets:  Linear(embed_dims*num_bev_queue, ...)  -- 输入维度是 embed_dims*2
      - attention_weights: Linear(embed_dims*num_bev_queue, ...)  -- 同上
      - value_proj:        Linear(embed_dims, embed_dims)
      - output_proj:       Linear(embed_dims, embed_dims)

    关键点：TSA 在 forward 中做 concat (L194):
      query = torch.cat([value[:bs], query], -1)  # 维度变为 embed_dims*2
    因此 sampling_offsets/attention_weights 的输入维度 = embed_dims * num_bev_queue。
    当 embed_dims 被剪枝 idxs 时，需要将 idxs 映射到双倍维度：
      [idx, idx + embed_dims] for each idx
    """

    TARGET_MODULES = ()

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        prune_linear_out_channels(layer.output_proj, idxs)
        prune_linear_out_channels(layer.value_proj, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        # value_proj 和 output_proj 的输入维度 = embed_dims
        prune_linear_in_channels(layer.value_proj, idxs)
        prune_linear_in_channels(layer.output_proj, idxs)

        # sampling_offsets/attention_weights 的输入维度 = embed_dims * num_bev_queue
        # 需要将 embed_dims 上的 idxs 扩展到 concat 后的维度
        expanded_idxs = _expand_idxs_for_concat(
            idxs, layer.embed_dims, layer.num_bev_queue
        )
        prune_linear_in_channels(layer.sampling_offsets, expanded_idxs)
        prune_linear_in_channels(layer.attention_weights, expanded_idxs)
        return layer

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.output_proj.out_features

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.value_proj.in_features


class CustomMSDeformableAttentionPruner(tp.BasePruningFunc):
    """处理 decoder 的 CustomMSDeformableAttention (decoder.py L133-345)。

    结构与 MSDeformableAttention3D 相同，但有自己的 output_proj:
      - sampling_offsets:  Linear(embed_dims, num_heads*num_levels*num_points*2)
      - attention_weights: Linear(embed_dims, num_heads*num_levels*num_points)
      - value_proj:        Linear(embed_dims, embed_dims)
      - output_proj:       Linear(embed_dims, embed_dims)  -- 区别：这里有 output_proj
    """

    TARGET_MODULES = ()

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        prune_linear_out_channels(layer.value_proj, idxs)
        prune_linear_out_channels(layer.output_proj, idxs)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        prune_linear_in_channels(layer.value_proj, idxs)
        prune_linear_in_channels(layer.output_proj, idxs)
        prune_linear_in_channels(layer.sampling_offsets, idxs)
        prune_linear_in_channels(layer.attention_weights, idxs)
        return layer

    def get_out_channels(self, layer: nn.Module) -> int:
        return layer.output_proj.out_features

    def get_in_channels(self, layer: nn.Module) -> int:
        return layer.value_proj.in_features


def _expand_idxs_for_concat(
    idxs: Sequence[int],
    embed_dims: int,
    num_bev_queue: int,
) -> list:
    """将 embed_dims 上的剪枝索引扩展到 concat(prev_bev, current_bev) 后的维度。

    TSA 的 forward 中做了 torch.cat([value[:bs], query], -1)，
    使得 sampling_offsets/attention_weights 的输入维度 = embed_dims * num_bev_queue。
    例如 embed_dims=256, num_bev_queue=2 时，输入维度=512：
      [prev_bev_dim0..255, current_bev_dim0..255]

    如果要剪掉 embed_dims 中的 idx，则需要同时剪掉每个 queue 对应的位置：
      [idx, idx + embed_dims, idx + 2*embed_dims, ...]

    Args:
        idxs: 在 embed_dims 维度上要剪枝的索引
        embed_dims: 当前的 embed_dims（剪枝前的值）
        num_bev_queue: BEV queue 数量（通常为 2）
    """
    expanded = []
    for idx in idxs:
        for q in range(num_bev_queue):
            expanded.append(idx + q * embed_dims)
    return sorted(expanded)


def register_univ2x_pruners() -> dict:
    """注册 UniV2X 特有模块的自定义剪枝器。

    返回 {module_class: pruner_instance} 字典，
    供 DependencyGraph.build_dependency(customized_pruners=...) 使用。
    """
    from ..modules.decoder import CustomMSDeformableAttention
    from ..modules.spatial_cross_attention import (
        MSDeformableAttention3D,
        SpatialCrossAttention,
    )
    from ..modules.temporal_self_attention import TemporalSelfAttention

    return {
        MSDeformableAttention3D: MSDeformableAttention3DPruner(),
        SpatialCrossAttention: SpatialCrossAttentionPruner(),
        TemporalSelfAttention: TemporalSelfAttentionPruner(),
        CustomMSDeformableAttention: CustomMSDeformableAttentionPruner(),
    }
