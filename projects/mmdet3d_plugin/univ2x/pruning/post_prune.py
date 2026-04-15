"""
剪枝后状态更新与一致性验证。

Torch-Pruning 修改了权重矩阵的实际形状，但模块内部的属性
(embed_dims, num_heads, head_dim, feedforward_channels, normalized_shape)
不会自动更新。本模块负责遍历所有模块，将这些属性与实际权重对齐。
"""
import logging
from dataclasses import dataclass
from typing import List

import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# 公共 API
# ============================================================================

def update_model_after_pruning(model: nn.Module) -> None:
    """遍历模型所有模块，更新内部属性以匹配剪枝后的权重维度。"""
    # 延迟导入避免循环依赖
    from projects.mmdet3d_plugin.univ2x.modules.temporal_self_attention import (
        TemporalSelfAttention,
    )
    from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import (
        MSDeformableAttention3D,
        SpatialCrossAttention,
    )
    from projects.mmdet3d_plugin.univ2x.modules.decoder import (
        CustomMSDeformableAttention,
    )
    from projects.mmdet3d_plugin.univ2x.modules.encoder import (
        BEVFormerEncoder,
    )

    update_count = 0

    for name, module in model.named_modules():

        # --- TemporalSelfAttention ---
        if isinstance(module, TemporalSelfAttention):
            old_dims = module.embed_dims
            new_dims = module.value_proj.out_features
            if old_dims != new_dims:
                module.embed_dims = new_dims
                # 从 sampling_offsets 输出维度反推 num_heads
                so_out = module.sampling_offsets.out_features
                nq = module.num_bev_queue
                nl = module.num_levels
                np_ = module.num_points
                inferred_heads = so_out // (nq * nl * np_ * 2)
                if inferred_heads != module.num_heads:
                    module.num_heads = inferred_heads
                module.head_dim = new_dims // module.num_heads
                update_count += 1
                logger.info("[TSA] %s: embed_dims %d->%d, heads=%d, head_dim=%d",
                            name, old_dims, new_dims, module.num_heads, module.head_dim)

        # --- SpatialCrossAttention ---
        elif isinstance(module, SpatialCrossAttention):
            old_dims = module.embed_dims
            new_dims = module.output_proj.out_features
            if old_dims != new_dims:
                module.embed_dims = new_dims
                update_count += 1
                logger.info("[SCA] %s: embed_dims %d->%d", name, old_dims, new_dims)

        # --- MSDeformableAttention3D / CustomMSDeformableAttention ---
        elif isinstance(module, (MSDeformableAttention3D, CustomMSDeformableAttention)):
            old_dims = module.embed_dims
            new_dims = module.value_proj.out_features
            if old_dims != new_dims:
                module.embed_dims = new_dims
                so_out = module.sampling_offsets.out_features
                nl = module.num_levels
                np_ = module.num_points
                inferred_heads = so_out // (nl * np_ * 2)
                if inferred_heads != module.num_heads:
                    module.num_heads = inferred_heads
                module.head_dim = new_dims // module.num_heads
                update_count += 1
                logger.info("[MSDA] %s: embed_dims %d->%d, heads=%d, head_dim=%d",
                            name, old_dims, new_dims, module.num_heads, module.head_dim)

        # --- BEVFormerEncoder ---
        elif isinstance(module, BEVFormerEncoder):
            if len(module.layers) > 0:
                first_layer = module.layers[0]
                if hasattr(first_layer, "attentions") and len(first_layer.attentions) > 0:
                    tsa = first_layer.attentions[0]
                    if hasattr(tsa, "value_proj"):
                        new_dims = tsa.value_proj.out_features
                        if hasattr(module, "embed_dims") and module.embed_dims != new_dims:
                            old_dims = module.embed_dims
                            module.embed_dims = new_dims
                            update_count += 1
                            logger.info("[Encoder] %s: embed_dims %d->%d",
                                        name, old_dims, new_dims)

        # --- LayerNorm ---
        elif isinstance(module, nn.LayerNorm):
            actual_shape = tuple(module.weight.shape)
            if module.normalized_shape != actual_shape:
                old_shape = module.normalized_shape
                module.normalized_shape = actual_shape
                update_count += 1
                logger.info("[LN] %s: normalized_shape %s->%s",
                            name, old_shape, actual_shape)

    # --- FFN feedforward_channels 更新 ---
    for name, module in model.named_modules():
        if hasattr(module, "ffns"):
            for ffn_idx, ffn in enumerate(module.ffns):
                if hasattr(ffn, "layers") and len(ffn.layers) > 0:
                    first_layer = ffn.layers[0]
                    if hasattr(first_layer, "__getitem__"):
                        linear = first_layer[0]
                        if isinstance(linear, nn.Linear):
                            new_fc = linear.out_features
                            if (hasattr(ffn, "feedforward_channels")
                                    and ffn.feedforward_channels != new_fc):
                                old_fc = ffn.feedforward_channels
                                ffn.feedforward_channels = new_fc
                                update_count += 1
                                logger.info("[FFN] %s.ffns.%d: feedforward_channels %d->%d",
                                            name, ffn_idx, old_fc, new_fc)

    # --- BEVFormerTrackHead embed_dims ---
    try:
        from projects.mmdet3d_plugin.univ2x.dense_heads.track_head import BEVFormerTrackHead
        for name, module in model.named_modules():
            if isinstance(module, BEVFormerTrackHead):
                if hasattr(module, "cls_branches") and len(module.cls_branches) > 0:
                    first_linear = module.cls_branches[0][0]
                    if isinstance(first_linear, nn.Linear):
                        new_dims = first_linear.in_features
                        if module.embed_dims != new_dims:
                            old_dims = module.embed_dims
                            module.embed_dims = new_dims
                            update_count += 1
                            logger.info("[Head] %s: embed_dims %d->%d",
                                        name, old_dims, new_dims)
    except ImportError:
        pass

    logger.info("状态更新完成: 共更新 %d 个模块", update_count)


# ============================================================================
# 一致性验证
# ============================================================================

@dataclass
class DimViolation:
    """维度不一致的报告项。"""
    module_name: str
    attribute: str
    expected: int
    actual: int
    description: str


def verify_model_consistency(model: nn.Module) -> List[DimViolation]:
    """遍历所有模块，检查内部属性与实际权重维度是否一致。

    Returns:
        违规列表，空列表表示通过验证。
    """
    from projects.mmdet3d_plugin.univ2x.modules.temporal_self_attention import (
        TemporalSelfAttention,
    )
    from projects.mmdet3d_plugin.univ2x.modules.spatial_cross_attention import (
        MSDeformableAttention3D,
        SpatialCrossAttention,
    )
    from projects.mmdet3d_plugin.univ2x.modules.decoder import (
        CustomMSDeformableAttention,
    )

    violations = []

    for name, module in model.named_modules():

        # --- 注意力模块: embed_dims 一致性 ---
        if isinstance(module, (TemporalSelfAttention, MSDeformableAttention3D,
                                CustomMSDeformableAttention)):
            actual_dims = module.value_proj.out_features
            if module.embed_dims != actual_dims:
                violations.append(DimViolation(
                    module_name=name, attribute="embed_dims",
                    expected=actual_dims, actual=module.embed_dims,
                    description=f"embed_dims({module.embed_dims}) != value_proj.out({actual_dims})",
                ))

            # embed_dims 可被 num_heads 整除
            if module.embed_dims % module.num_heads != 0:
                violations.append(DimViolation(
                    module_name=name, attribute="num_heads",
                    expected=0, actual=module.embed_dims % module.num_heads,
                    description=f"embed_dims({module.embed_dims}) % num_heads({module.num_heads}) != 0",
                ))

        # --- SpatialCrossAttention ---
        elif isinstance(module, SpatialCrossAttention):
            actual_dims = module.output_proj.out_features
            if module.embed_dims != actual_dims:
                violations.append(DimViolation(
                    module_name=name, attribute="embed_dims",
                    expected=actual_dims, actual=module.embed_dims,
                    description=f"embed_dims({module.embed_dims}) != output_proj.out({actual_dims})",
                ))

        # --- LayerNorm ---
        elif isinstance(module, nn.LayerNorm):
            actual_shape = tuple(module.weight.shape)
            if module.normalized_shape != actual_shape:
                violations.append(DimViolation(
                    module_name=name, attribute="normalized_shape",
                    expected=actual_shape[0], actual=module.normalized_shape[0],
                    description=f"normalized_shape {module.normalized_shape} != weight.shape {actual_shape}",
                ))

    # --- 解码器分支长度一致性 ---
    if hasattr(model, "pts_bbox_head"):
        head = model.pts_bbox_head
        decoder = head.transformer.decoder
        n_layers = len(decoder.layers)
        for attr in ("cls_branches", "reg_branches", "past_traj_reg_branches"):
            if hasattr(head, attr):
                n_branches = len(getattr(head, attr))
                if n_branches != n_layers:
                    violations.append(DimViolation(
                        module_name=f"pts_bbox_head.{attr}",
                        attribute="length",
                        expected=n_layers, actual=n_branches,
                        description=f"{attr} 长度({n_branches}) != decoder.layers({n_layers})",
                    ))

    # --- FFN 残差连接一致性 ---
    for name, module in model.named_modules():
        if hasattr(module, "ffns"):
            for ffn_idx, ffn in enumerate(module.ffns):
                if hasattr(ffn, "layers") and len(ffn.layers) >= 2:
                    # 最后一个 Linear 的 out_features 应等于第一个 Linear 的 in_features
                    try:
                        first_linear = ffn.layers[0][0]
                        last_linear = ffn.layers[-1]
                        if isinstance(last_linear, nn.Sequential):
                            last_linear = last_linear[0]
                        if isinstance(first_linear, nn.Linear) and isinstance(last_linear, nn.Linear):
                            if first_linear.in_features != last_linear.out_features:
                                violations.append(DimViolation(
                                    module_name=f"{name}.ffns.{ffn_idx}",
                                    attribute="residual",
                                    expected=first_linear.in_features,
                                    actual=last_linear.out_features,
                                    description=(
                                        f"FFN input({first_linear.in_features}) != "
                                        f"output({last_linear.out_features}), 残差连接将失败"
                                    ),
                                ))
                    except (IndexError, AttributeError):
                        pass

    if violations:
        logger.warning("一致性验证发现 %d 个问题:", len(violations))
        for v in violations:
            logger.warning("  [%s] %s: %s", v.module_name, v.attribute, v.description)
    else:
        logger.info("一致性验证通过")

    return violations
