# Phase A2: 剪枝执行引擎（Pruning Execution Engine）

> 任务范围：A.2.2 + A.2.3 + A.2.5
> 预估工时：3 天
> 前置依赖：Phase A1 完成
> 涉及文件：`prune_univ2x.py`, `post_prune.py`
> 关键参考：`实施计划_1.2_可配置剪枝.md` 中的模块映射关系

---

## 1. 阶段目标

构建剪枝执行引擎，包含三个核心能力：

1. **剪枝主入口** (`prune_univ2x.py`)：接收 `prune_config` 配置字典，通过 DepGraph 执行结构化剪枝。支持搜索维度 P1-P3（通道剪枝）、P8（注意力头剪枝），以及锁定维度 P4-P7 的参数化控制。
2. **解码器层剪枝** (P9)：直接删除 `DetectionTransformerDecoder` 中的 Transformer 层，并同步更新 `cls_branches`、`reg_branches`、`past_traj_reg_branches` 的长度。
3. **剪枝后状态更新** (`post_prune.py`)：遍历模型所有模块，将 `embed_dims`、`num_heads`、`head_dim`、`feedforward_channels`、`normalized_shape` 等内部属性与剪枝后的实际权重维度对齐，并提供一致性验证函数。

---

## 2. 前置条件

- **Phase A1 已完成**：
  - `custom_pruners.py` 中的 `MSDeformableAttention3DPruner`、`SpatialCrossAttentionPruner`、`TemporalSelfAttentionPruner` 已实现并测试通过
  - `grad_collector.py` 的梯度收集器可正常工作（Taylor 重要性评估所需）
  - 自定义剪枝器已通过 `register_univ2x_pruners()` 注册到 DepGraph

- **DepGraph 可追踪 UniV2X 模型**：
  - 自定义 CUDA 算子（`MultiScaleDeformableAttnFunction`）通过自定义剪枝器绕过
  - `SpatialCrossAttention` 中的 `nonzero()` 动态索引通过自定义剪枝器绕过
  - `example_inputs` 构造方案已验证（需要 `mlvl_feats`, `img_metas` 等输入）

- **依赖库**：
  - `torch_pruning >= 1.3.0`
  - PyTorch >= 1.10

---

## 3. 具体代码实现

### 3.1 prune_univ2x.py

**文件路径**: `projects/mmdet3d_plugin/univ2x/pruning/prune_univ2x.py`

```python
"""
剪枝执行引擎主入口。

接收 prune_config 字典，通过 Torch-Pruning DepGraph 框架执行结构化剪枝。
支持维度：P1 (FFN中间维度), P2 (注意力投影通道), P3 (检测头中间维度),
         P4-P7 (锁定维度), P8 (注意力头剪枝), P9 (解码器层数剪枝)
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch_pruning as tp

from .custom_pruners import register_univ2x_pruners

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# prune_config 示例结构 (供调用方参考)
# ---------------------------------------------------------------------------
# prune_cfg = {
#     # --- 搜索维度 ---
#     "ffn_mid_ratio": 0.5,          # P1: FFN 中间维度保留比例
#     "attn_proj_ratio": 0.1,        # P2: 注意力投影通道剪枝比例
#     "head_mid_ratio": 0.7,         # P3: 检测头中间维度保留比例
#     "head_pruning_ratio": 0.0,     # P8: 注意力头剪枝比例
#     "decoder_num_layers": 6,       # P9: 解码器保留层数
#
#     # --- 锁定维度 ---
#     "importance_criterion": "taylor",  # P4
#     "pruning_granularity": "local",    # P5: global / local / isomorphic
#     "iterative_steps": 5,              # P6
#     "round_to": 8,                     # P7: INT8 硬约束
#
#     # --- 约束 ---
#     "ignored_layer_keywords": [        # 不参与剪枝的层名关键词
#         "sampling_offsets",
#         "attention_weights",
#     ],
#     "ignored_output_keywords": [       # 输出层不剪枝
#         "cls_branches.*.6",            # 分类最终输出层 (最后一个 Linear)
#         "reg_branches.*.2",            # 回归最终输出层
#         "past_traj_reg_branches.*.2",
#     ],
# }


# ============================================================================
# 公共 API
# ============================================================================

def prune_model(
    model: nn.Module,
    example_inputs: Dict[str, Any],
    prune_cfg: Dict[str, Any],
) -> nn.Module:
    """剪枝主入口：按 prune_cfg 执行完整剪枝流程。

    执行顺序：
      1. P9 解码器层剪枝（先做，因为它不涉及通道维度，
         且会改变 cls_branches 等长度）
      2. P1-P3, P8 通道/头剪枝（通过 DepGraph）
      3. 状态更新（由 post_prune.py 负责）

    Args:
        model: 待剪枝的 UniV2X 模型 (已加载权重)
        example_inputs: DepGraph 追踪所需的示例输入
        prune_cfg: 剪枝配置字典

    Returns:
        剪枝后的模型（原地修改，同时返回引用）
    """
    # Step 1: P9 解码器层剪枝
    target_num_layers = prune_cfg.get("decoder_num_layers", 6)
    current_num_layers = len(model.pts_bbox_head.transformer.decoder.layers)
    if target_num_layers < current_num_layers:
        logger.info(
            "P9: 解码器层剪枝 %d -> %d", current_num_layers, target_num_layers
        )
        prune_decoder_layers(model, target_num_layers)

    # Step 2: 通道/头剪枝 (P1-P3, P8)
    has_channel_pruning = (
        prune_cfg.get("ffn_mid_ratio", 1.0) < 1.0
        or prune_cfg.get("attn_proj_ratio", 0.0) > 0.0
        or prune_cfg.get("head_mid_ratio", 1.0) < 1.0
        or prune_cfg.get("head_pruning_ratio", 0.0) > 0.0
    )

    if has_channel_pruning:
        logger.info("P1-P3/P8: 执行通道/头剪枝")
        pruner = build_pruner(model, example_inputs, prune_cfg)
        pruner.step()
        logger.info("通道/头剪枝完成")

    # Step 3: 状态更新由调用方在外部调用 post_prune.update_model_after_pruning()
    # 这里不自动调用，保持职责单一

    return model


# ============================================================================
# DepGraph 构建
# ============================================================================

def build_dependency_graph(
    model: nn.Module,
    example_inputs: Dict[str, Any],
    prune_cfg: Dict[str, Any],
) -> tp.DependencyGraph:
    """构建 DepGraph 依赖图。

    注册自定义剪枝器、收集忽略层和 unwrapped 参数，
    然后调用 tp.DependencyGraph.build_dependency()。

    Args:
        model: UniV2X 模型
        example_inputs: 追踪用的示例输入字典
        prune_cfg: 剪枝配置

    Returns:
        构建完成的 DependencyGraph 实例
    """
    customized_pruners = register_univ2x_pruners()
    ignored_layers = _collect_ignored_layers(model, prune_cfg)
    unwrapped_parameters = _collect_unwrapped_params(model)

    DG = tp.DependencyGraph()
    DG.build_dependency(
        model,
        example_inputs=example_inputs,
        customized_pruners=customized_pruners,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
    )
    return DG


# ============================================================================
# Pruner 构建
# ============================================================================

def build_pruner(
    model: nn.Module,
    example_inputs: Dict[str, Any],
    prune_cfg: Dict[str, Any],
) -> tp.pruner.BasePruner:
    """构建 BasePruner 实例。

    锁定维度 (P4-P7) 作为 Pruner 构造参数传入，
    搜索维度 (P1-P3, P8) 通过 ratio_dict / head_pruning_ratio 控制。

    Args:
        model: UniV2X 模型
        example_inputs: 追踪用的示例输入
        prune_cfg: 剪枝配置

    Returns:
        配置完成的 BasePruner 实例
    """
    customized_pruners = register_univ2x_pruners()
    ignored_layers = _collect_ignored_layers(model, prune_cfg)
    unwrapped_parameters = _collect_unwrapped_params(model)

    # P4: 重要性评估准则
    importance = _select_importance(
        prune_cfg.get("importance_criterion", "taylor")
    )

    # P1-P3: 构建 per-module 剪枝比例字典
    # 注意：ratio_dict 中值为"剪掉的比例"，不是"保留的比例"
    ratio_dict = _build_ratio_dict(model, prune_cfg)

    # P8: 注意力头剪枝
    head_pruning_ratio = prune_cfg.get("head_pruning_ratio", 0.0)
    num_heads_dict = {}
    if head_pruning_ratio > 0.0:
        num_heads_dict = _collect_num_heads(model)

    # P5: 剪枝粒度
    granularity = prune_cfg.get("pruning_granularity", "local")
    # 映射到 Torch-Pruning 的 global_pruning 参数
    global_pruning = (granularity == "global")
    isomorphic = (granularity == "isomorphic")

    # P6: 迭代步数
    iterative_steps = prune_cfg.get("iterative_steps", 5)

    # P7: 通道对齐
    round_to = prune_cfg.get("round_to", 8)

    pruner = tp.pruner.MetaPruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio_dict=ratio_dict,
        global_pruning=global_pruning,
        isomorphic=isomorphic,
        iterative_steps=iterative_steps,
        round_to=round_to,
        customized_pruners=customized_pruners,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        head_pruning_ratio=head_pruning_ratio if head_pruning_ratio > 0 else None,
        num_heads=num_heads_dict if num_heads_dict else None,
    )

    return pruner


# ============================================================================
# 重要性评估准则选择 (P4)
# ============================================================================

def _select_importance(criterion: str) -> tp.importance.Importance:
    """将字符串准则名映射到 tp.importance 类实例。

    Args:
        criterion: 准则名称，支持 "l1_norm", "taylor", "bn_scale", "fpgm"

    Returns:
        对应的 Importance 实例

    Raises:
        ValueError: 不支持的准则名称
    """
    criterion_map = {
        "l1_norm": tp.importance.MagnitudeImportance(p=1),
        "taylor": tp.importance.TaylorImportance(),
        "bn_scale": tp.importance.BNScaleImportance(),
        "fpgm": tp.importance.FPGMImportance(p=2),
    }

    if criterion not in criterion_map:
        raise ValueError(
            f"不支持的重要性准则: {criterion}，"
            f"可选值: {list(criterion_map.keys())}"
        )

    return criterion_map[criterion]


# ============================================================================
# 剪枝比例字典构建 (P1-P3)
# ============================================================================

def _build_ratio_dict(
    model: nn.Module,
    prune_cfg: Dict[str, Any],
) -> Dict[nn.Module, float]:
    """为模型中的每个可剪枝模块构建剪枝比例字典。

    根据模块名分类 (FFN / 注意力投影 / 检测头)，
    映射到对应的剪枝比例 P1 / P2 / P3。

    注意：ratio 值表示"剪掉的比例"。
      - P1 ffn_mid_ratio 是"保留比例"，需要转换：prune_ratio = 1 - ffn_mid_ratio
      - P2 attn_proj_ratio 已经是"剪掉的比例"
      - P3 head_mid_ratio 是"保留比例"，需要转换：prune_ratio = 1 - head_mid_ratio

    Args:
        model: UniV2X 模型
        prune_cfg: 剪枝配置

    Returns:
        {module: pruning_ratio} 字典
    """
    ffn_prune_ratio = 1.0 - prune_cfg.get("ffn_mid_ratio", 1.0)
    attn_prune_ratio = prune_cfg.get("attn_proj_ratio", 0.0)
    head_prune_ratio = 1.0 - prune_cfg.get("head_mid_ratio", 1.0)

    ratio_dict = {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if _is_ffn_layer(name):
            if ffn_prune_ratio > 0:
                ratio_dict[module] = ffn_prune_ratio
        elif _is_attn_proj(name):
            if attn_prune_ratio > 0:
                ratio_dict[module] = attn_prune_ratio
        elif _is_head_layer(name):
            if head_prune_ratio > 0:
                ratio_dict[module] = head_prune_ratio

    logger.info(
        "ratio_dict 构建完成: FFN=%d 层 (ratio=%.2f), "
        "attn_proj=%d 层 (ratio=%.2f), head=%d 层 (ratio=%.2f)",
        sum(1 for n, _ in model.named_modules() if _is_ffn_layer(n)),
        ffn_prune_ratio,
        sum(1 for n, _ in model.named_modules() if _is_attn_proj(n)),
        attn_prune_ratio,
        sum(1 for n, _ in model.named_modules() if _is_head_layer(n)),
        head_prune_ratio,
    )

    return ratio_dict


# ============================================================================
# 模块名分类器
# ============================================================================

def _is_ffn_layer(name: str) -> bool:
    """判断模块名是否属于 FFN 层。

    FFN 模块位于 BEVFormerEncoder 和 DetectionTransformerDecoder 的每一层中。
    FFN 的 Linear 层名称模式：
      - encoder.layers.*.ffns.*.layers.*.0  (FFN 中的 Linear 子层)
      - decoder.layers.*.ffns.*.layers.*.0

    注意：FFN 的最后一层 Linear 输出维度应与输入 embed_dims 相同（残差连接），
    只有中间层的维度是 feedforward_channels。但 DepGraph 会自动处理依赖关系，
    我们只需标记 FFN 中间层的 Linear。

    典型路径:
      pts_bbox_head.transformer.encoder.layers.0.ffns.0.layers.0.0  (Linear: 256->512)
      pts_bbox_head.transformer.encoder.layers.0.ffns.0.layers.1    (Linear: 512->256)
    """
    # 匹配 ffns 下的 Linear 层（仅中间层，即 layers.0.0）
    # layers.1 是投影回原维度的层，由 DepGraph 自动处理
    if "ffns" in name and "layers.0.0" in name:
        return True
    return False


def _is_attn_proj(name: str) -> bool:
    """判断模块名是否属于注意力投影层 (value_proj / output_proj)。

    P2 控制的层：
      - TemporalSelfAttention.value_proj, output_proj
      - SpatialCrossAttention.output_proj
      - MSDeformableAttention3D.value_proj, output_proj
      - CustomMSDeformableAttention.value_proj, output_proj

    排除 sampling_offsets 和 attention_weights（硬约束不剪枝）。

    典型路径:
      pts_bbox_head.transformer.encoder.layers.0.attentions.0.value_proj
      pts_bbox_head.transformer.encoder.layers.0.attentions.0.output_proj
      pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.value_proj
    """
    if "value_proj" in name or "output_proj" in name:
        # 排除非注意力模块中的同名层
        if "ffns" not in name and "branches" not in name:
            return True
    return False


def _is_head_layer(name: str) -> bool:
    """判断模块名是否属于检测头中间层。

    P3 控制的层：
      - cls_branches.*.0, cls_branches.*.3  (中间 Linear 层)
      - reg_branches.*.0  (中间 Linear 层)
      - past_traj_reg_branches.*.0

    排除最终输出层：
      - cls_branches.*.6   -> 输出 num_classes
      - reg_branches.*.2   -> 输出 code_size (10)
      - past_traj_reg_branches.*.2 -> 输出 (past+fut)*2

    典型路径:
      pts_bbox_head.cls_branches.0.0    (Linear: 256->256, 中间层)
      pts_bbox_head.cls_branches.0.3    (Linear: 256->256, 中间层)
      pts_bbox_head.cls_branches.0.6    (Linear: 256->10, 输出层 -> 不剪枝)
      pts_bbox_head.reg_branches.0.0    (Linear: 256->256, 中间层)
      pts_bbox_head.reg_branches.0.2    (Linear: 256->10, 输出层 -> 不剪枝)
    """
    # cls_branches 中间层
    if "cls_branches" in name:
        # 排除最终输出层 (Sequential 中最后一个 Linear)
        # cls_branch 结构: [Linear, LN, ReLU] * num_reg_fcs + [Linear]
        # 最后的 Linear 索引 = num_reg_fcs * 3 (默认 num_reg_fcs=2, 索引=6)
        parts = name.split(".")
        try:
            layer_idx = int(parts[-1])
            # 中间 Linear 层的索引: 0, 3 (每组 Linear+LN+ReLU 占 3 个位置)
            if layer_idx % 3 == 0 and "." + str(layer_idx) != "." + parts[-1]:
                return True
            # 更可靠的判断：检查是否为最后一层
            return layer_idx != 6  # 默认 num_reg_fcs=2, 最后 Linear 在索引 6
        except (ValueError, IndexError):
            pass
        return False

    # reg_branches 中间层
    if "reg_branches" in name and "past_traj" not in name:
        parts = name.split(".")
        try:
            layer_idx = int(parts[-1])
            return layer_idx != 2  # 最后 Linear 在索引 2 (num_reg_fcs=2)
        except (ValueError, IndexError):
            pass
        return False

    # past_traj_reg_branches 中间层
    if "past_traj_reg_branches" in name:
        parts = name.split(".")
        try:
            layer_idx = int(parts[-1])
            return layer_idx != 2
        except (ValueError, IndexError):
            pass
        return False

    return False


# ============================================================================
# 忽略层收集
# ============================================================================

def _collect_ignored_layers(
    model: nn.Module,
    prune_cfg: Dict[str, Any],
) -> List[nn.Module]:
    """收集不参与剪枝的层。

    包含三类：
    1. 坐标敏感层：sampling_offsets, attention_weights
       (它们的输出维度由 num_heads * num_levels * num_points 决定，
        与 embed_dims 无关，剪枝会破坏几何语义)
    2. 输出层：cls_branches 最终分类层、reg_branches 最终回归层
       (改变输出维度会破坏任务语义)
    3. 用户通过 ignored_layer_keywords 额外指定的层

    Args:
        model: UniV2X 模型
        prune_cfg: 剪枝配置

    Returns:
        不参与剪枝的 nn.Module 列表
    """
    ignored = []

    # 默认关键词 + 用户自定义关键词
    default_keywords = ["sampling_offsets", "attention_weights"]
    user_keywords = prune_cfg.get("ignored_layer_keywords", [])
    all_keywords = default_keywords + user_keywords

    # 输出层关键词
    output_keywords = prune_cfg.get("ignored_output_keywords", [])

    for name, module in model.named_modules():
        # 坐标敏感层
        if any(kw in name for kw in all_keywords):
            ignored.append(module)
            continue

        # 输出层检测
        if _is_output_layer(name):
            ignored.append(module)
            continue

        # 用户自定义输出层
        if any(_match_pattern(name, pat) for pat in output_keywords):
            ignored.append(module)

    logger.info("收集到 %d 个忽略层", len(ignored))
    return ignored


def _is_output_layer(name: str) -> bool:
    """判断是否为输出层（不可剪枝）。

    输出层定义：
      - cls_branches.X.6  (X=0..N, 6=最终分类 Linear 索引)
      - reg_branches.X.2  (最终回归 Linear)
      - past_traj_reg_branches.X.2 (最终轨迹回归 Linear)
      - reference_points (3D 坐标投影层)
    """
    if "cls_branches" in name:
        parts = name.split(".")
        try:
            if int(parts[-1]) == 6:
                return True
        except (ValueError, IndexError):
            pass

    if ("reg_branches" in name or "past_traj_reg_branches" in name):
        parts = name.split(".")
        try:
            if int(parts[-1]) == 2:
                return True
        except (ValueError, IndexError):
            pass

    if name.endswith("reference_points"):
        return True

    return False


def _match_pattern(name: str, pattern: str) -> bool:
    """简单通配符匹配，支持 * 作为单级通配符。"""
    import fnmatch
    return fnmatch.fnmatch(name, pattern)


# ============================================================================
# 注意力头数收集 (P8)
# ============================================================================

def _collect_num_heads(model: nn.Module) -> Dict[nn.Module, int]:
    """收集所有注意力模块的 num_heads 属性。

    Torch-Pruning 的头剪枝需要知道每个注意力模块有多少个头，
    以便按头为单位进行剪枝。

    支持的模块类型：
      - TemporalSelfAttention: num_heads
      - MSDeformableAttention3D: num_heads
      - CustomMSDeformableAttention: num_heads
      - nn.MultiheadAttention: num_heads

    Args:
        model: UniV2X 模型

    Returns:
        {module: num_heads} 字典
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

    head_types = (
        TemporalSelfAttention,
        MSDeformableAttention3D,
        CustomMSDeformableAttention,
        nn.MultiheadAttention,
    )

    num_heads_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, head_types):
            num_heads = getattr(module, "num_heads", None)
            if num_heads is not None:
                num_heads_dict[module] = num_heads
                logger.debug("num_heads[%s] = %d", name, num_heads)

    logger.info("收集到 %d 个注意力模块的 num_heads", len(num_heads_dict))
    return num_heads_dict


# ============================================================================
# Unwrapped 参数收集
# ============================================================================

def _collect_unwrapped_params(
    model: nn.Module,
) -> List[Tuple[nn.Parameter, int]]:
    """收集非 Module 包裹的独立参数。

    这些参数直接参与计算但不属于任何子 Module，
    DepGraph 需要显式告知它们存在和对应的剪枝维度。

    当前需要收集的参数：
      - bev_embedding.weight:  (bev_h*bev_w, embed_dims)
        剪枝维度 = 1（embed_dims 维度）
      - query_embedding.weight: (num_query+1, embed_dims*2)
        剪枝维度 = 1（embed_dims 维度，但注意是 *2）

    Args:
        model: UniV2X 模型

    Returns:
        [(parameter, pruning_dim)] 列表
    """
    unwrapped = []

    for name, param in model.named_parameters():
        if "bev_embedding" in name and "weight" in name:
            # bev_embedding: (bev_h*bev_w, embed_dims), 剪枝第1维
            unwrapped.append((param, 1))
            logger.debug("unwrapped: %s shape=%s dim=1", name, param.shape)

        elif "query_embedding" in name and "weight" in name:
            # query_embedding: (num_query+1, embed_dims*2), 剪枝第1维
            # 注意：这个参数的维度是 embed_dims*2，
            # 前半部分是 query，后半部分是 pos_encoding
            # 两者都需要随 embed_dims 同步剪枝
            unwrapped.append((param, 1))
            logger.debug("unwrapped: %s shape=%s dim=1", name, param.shape)

    logger.info("收集到 %d 个 unwrapped 参数", len(unwrapped))
    return unwrapped


# ============================================================================
# P9: 解码器层剪枝
# ============================================================================

def prune_decoder_layers(
    model: nn.Module,
    target_num_layers: int,
) -> None:
    """P9 实现：删除解码器层并同步分支。

    策略：保留最后 N 层（靠后的层对精度贡献更大）。

    同步更新的组件：
      - decoder.layers: ModuleList, 原始长度 = num_layers
      - cls_branches: ModuleList, 原始长度 = num_layers (with_box_refine)
      - reg_branches: ModuleList, 原始长度 = num_layers
      - past_traj_reg_branches: ModuleList, 原始长度 = num_layers

    关键索引关系 (track_head.py lines 110-124):
      num_pred = decoder.num_layers  (当 as_two_stage=False 时)
      cls_branches = [clone(fc_cls)] * num_pred
      reg_branches = [clone(reg_branch)] * num_pred

    在 get_detections (track_head.py lines 218-228) 中的使用：
      for lvl in range(hs.shape[0]):    # hs.shape[0] == num_layers
          outputs_class = cls_branches[lvl](hs[lvl])
          tmp = reg_branches[lvl](hs[lvl])
          outputs_past_traj = past_traj_reg_branches[lvl](hs[lvl])

    因此 cls_branches 长度必须 == decoder.layers 长度。

    DetectionTransformerDecoder.forward (decoder.py lines 93-129):
      for lid, layer in enumerate(self.layers):
          ...
          if reg_branches is not None:
              tmp = reg_branches[lid](output)  # lid 索引 reg_branches

    因此 reg_branches 传入 decoder 时也需要与 layers 长度匹配。

    策略细节：
      原始: layers = [L0, L1, L2, L3, L4, L5], branches = [B0, B1, B2, B3, B4, B5]
      保留最后4层: layers = [L2, L3, L4, L5], branches = [B2, B3, B4, B5]
      理由: 最后几层已经学到了最精细的特征表示，前面的层主要做粗筛

    Args:
        model: UniV2X 模型
        target_num_layers: 保留的层数

    Raises:
        ValueError: target_num_layers >= 当前层数或 <= 0
    """
    head = model.pts_bbox_head
    decoder = head.transformer.decoder

    current_num_layers = len(decoder.layers)
    if target_num_layers >= current_num_layers:
        logger.warning(
            "target_num_layers=%d >= current=%d, 跳过层剪枝",
            target_num_layers, current_num_layers,
        )
        return

    if target_num_layers <= 0:
        raise ValueError(
            f"target_num_layers 必须 > 0，当前值: {target_num_layers}"
        )

    # 计算要删除的层数和起始索引
    num_to_remove = current_num_layers - target_num_layers
    # 保留最后 target_num_layers 层，即删除前 num_to_remove 层
    keep_start = num_to_remove

    logger.info(
        "解码器层剪枝: 删除前 %d 层 (L0..L%d), 保留 L%d..L%d",
        num_to_remove, num_to_remove - 1,
        keep_start, current_num_layers - 1,
    )

    # --- 裁剪 decoder.layers ---
    new_layers = nn.ModuleList(
        list(decoder.layers)[keep_start:]
    )
    decoder.layers = new_layers
    decoder.num_layers = target_num_layers

    # --- 同步裁剪 cls_branches ---
    if hasattr(head, "cls_branches"):
        new_cls = nn.ModuleList(
            list(head.cls_branches)[keep_start:]
        )
        head.cls_branches = new_cls
        logger.info("cls_branches: %d -> %d", current_num_layers, len(new_cls))

    # --- 同步裁剪 reg_branches ---
    if hasattr(head, "reg_branches"):
        new_reg = nn.ModuleList(
            list(head.reg_branches)[keep_start:]
        )
        head.reg_branches = new_reg
        logger.info("reg_branches: %d -> %d", current_num_layers, len(new_reg))

    # --- 同步裁剪 past_traj_reg_branches ---
    if hasattr(head, "past_traj_reg_branches"):
        new_traj = nn.ModuleList(
            list(head.past_traj_reg_branches)[keep_start:]
        )
        head.past_traj_reg_branches = new_traj
        logger.info(
            "past_traj_reg_branches: %d -> %d",
            current_num_layers, len(new_traj),
        )

    # --- 验证一致性 ---
    final_layers = len(decoder.layers)
    final_cls = len(head.cls_branches) if hasattr(head, "cls_branches") else -1
    final_reg = len(head.reg_branches) if hasattr(head, "reg_branches") else -1
    final_traj = (
        len(head.past_traj_reg_branches)
        if hasattr(head, "past_traj_reg_branches")
        else -1
    )

    assert final_layers == target_num_layers, (
        f"decoder.layers 长度不匹配: {final_layers} != {target_num_layers}"
    )
    assert final_cls == target_num_layers, (
        f"cls_branches 长度不匹配: {final_cls} != {target_num_layers}"
    )
    assert final_reg == target_num_layers, (
        f"reg_branches 长度不匹配: {final_reg} != {target_num_layers}"
    )
    assert final_traj == target_num_layers, (
        f"past_traj_reg_branches 长度不匹配: {final_traj} != {target_num_layers}"
    )

    logger.info(
        "P9 解码器层剪枝完成: %d -> %d 层, 所有分支已同步",
        current_num_layers, target_num_layers,
    )
```

---

### 3.2 post_prune.py

**文件路径**: `projects/mmdet3d_plugin/univ2x/pruning/post_prune.py`

```python
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
    """遍历模型所有模块，更新内部属性以匹配剪枝后的权重维度。

    更新策略按模块类型分类：

    TemporalSelfAttention:
      - embed_dims <- value_proj.out_features
      - head_dim <- embed_dims // num_heads  (若头数未变)
      - num_bev_queue 不变 (固定值 2)

    SpatialCrossAttention:
      - embed_dims <- output_proj.out_features

    MSDeformableAttention3D / CustomMSDeformableAttention:
      - embed_dims <- value_proj.out_features
      - num_heads: 若执行了头剪枝 (P8), 从 sampling_offsets 输出维度反推
      - head_dim <- embed_dims // num_heads

    BEVFormerEncoder:
      - embed_dims <- 从第一个子层推断

    FFN 模块 (mmcv BaseTransformerLayer 内部):
      - feedforward_channels <- ffn.layers.0.0.out_features

    LayerNorm:
      - normalized_shape <- 从前一层推断，或直接设为 (embed_dims,)

    BEVFormerTrackHead:
      - embed_dims <- 从 cls_branches[0][0] 的 in_features 推断

    Args:
        model: 剪枝后的 UniV2X 模型 (原地修改)
    """
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
    from projects.mmdet3d_plugin.univ2x.dense_heads.track_head import (
        BEVFormerTrackHead,
    )

    update_count = 0

    for name, module in model.named_modules():

        # --- TemporalSelfAttention ---
        if isinstance(module, TemporalSelfAttention):
            old_dims = module.embed_dims
            new_dims = module.value_proj.out_features
            if old_dims != new_dims:
                module.embed_dims = new_dims
                # num_heads 可能因头剪枝而改变
                # 从 sampling_offsets 输出维度反推:
                # out = num_bev_queue * num_heads * num_levels * num_points * 2
                so_out = module.sampling_offsets.out_features
                nq = module.num_bev_queue
                nl = module.num_levels
                np_ = module.num_points
                inferred_heads = so_out // (nq * nl * np_ * 2)
                if inferred_heads != module.num_heads:
                    module.num_heads = inferred_heads
                module.head_dim = new_dims // module.num_heads
                update_count += 1
                logger.info(
                    "[TSA] %s: embed_dims %d->%d, num_heads=%d, head_dim=%d",
                    name, old_dims, new_dims,
                    module.num_heads, module.head_dim,
                )

        # --- SpatialCrossAttention ---
        elif isinstance(module, SpatialCrossAttention):
            old_dims = module.embed_dims
            new_dims = module.output_proj.out_features
            if old_dims != new_dims:
                module.embed_dims = new_dims
                update_count += 1
                logger.info(
                    "[SCA] %s: embed_dims %d->%d", name, old_dims, new_dims
                )

        # --- MSDeformableAttention3D / CustomMSDeformableAttention ---
        elif isinstance(module, (MSDeformableAttention3D,
                                  CustomMSDeformableAttention)):
            old_dims = module.embed_dims
            new_dims = module.value_proj.out_features
            if old_dims != new_dims:
                module.embed_dims = new_dims
                # 从 sampling_offsets 反推 num_heads
                so_out = module.sampling_offsets.out_features
                nl = module.num_levels
                np_ = module.num_points
                inferred_heads = so_out // (nl * np_ * 2)
                if inferred_heads != module.num_heads:
                    module.num_heads = inferred_heads
                module.head_dim = new_dims // module.num_heads
                update_count += 1
                logger.info(
                    "[MSDA] %s: embed_dims %d->%d, num_heads=%d, head_dim=%d",
                    name, old_dims, new_dims,
                    module.num_heads, module.head_dim,
                )

        # --- BEVFormerEncoder ---
        elif isinstance(module, BEVFormerEncoder):
            # embed_dims 从第一层的 TSA.value_proj 推断
            if len(module.layers) > 0:
                first_layer = module.layers[0]
                # 在 MyCustomBaseTransformerLayer 中，
                # attentions[0] 通常是 TemporalSelfAttention
                if hasattr(first_layer, "attentions") and len(first_layer.attentions) > 0:
                    tsa = first_layer.attentions[0]
                    if hasattr(tsa, "value_proj"):
                        new_dims = tsa.value_proj.out_features
                        if hasattr(module, "embed_dims") and module.embed_dims != new_dims:
                            old_dims = module.embed_dims
                            module.embed_dims = new_dims
                            update_count += 1
                            logger.info(
                                "[Encoder] %s: embed_dims %d->%d",
                                name, old_dims, new_dims,
                            )

        # --- BEVFormerTrackHead ---
        elif isinstance(module, BEVFormerTrackHead):
            # embed_dims 从 cls_branches[0] 的第一个 Linear 推断
            if hasattr(module, "cls_branches") and len(module.cls_branches) > 0:
                first_branch = module.cls_branches[0]
                # cls_branch 是 Sequential: [Linear, LN, ReLU, Linear, LN, ReLU, Linear]
                first_linear = first_branch[0]
                if isinstance(first_linear, nn.Linear):
                    new_dims = first_linear.in_features
                    if module.embed_dims != new_dims:
                        old_dims = module.embed_dims
                        module.embed_dims = new_dims
                        update_count += 1
                        logger.info(
                            "[Head] %s: embed_dims %d->%d",
                            name, old_dims, new_dims,
                        )

        # --- LayerNorm ---
        elif isinstance(module, nn.LayerNorm):
            actual_shape = module.weight.shape
            if module.normalized_shape != tuple(actual_shape):
                old_shape = module.normalized_shape
                module.normalized_shape = tuple(actual_shape)
                update_count += 1
                logger.info(
                    "[LN] %s: normalized_shape %s->%s",
                    name, old_shape, tuple(actual_shape),
                )

    # --- FFN feedforward_channels 更新 ---
    # FFN 通常在 MyCustomBaseTransformerLayer 中，需要遍历更新
    for name, module in model.named_modules():
        if hasattr(module, "ffns"):
            for ffn_idx, ffn in enumerate(module.ffns):
                if hasattr(ffn, "layers") and len(ffn.layers) > 0:
                    # ffn.layers[0] 是 Sequential(Linear, act)
                    # ffn.layers[0][0] 是中间维度的 Linear
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
                                logger.info(
                                    "[FFN] %s.ffns.%d: "
                                    "feedforward_channels %d->%d",
                                    name, ffn_idx, old_fc, new_fc,
                                )

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

    检查项：
    1. 注意力模块: embed_dims == value_proj.out_features
    2. 注意力模块: head_dim == embed_dims // num_heads
    3. 注意力模块: embed_dims % num_heads == 0
    4. LayerNorm: normalized_shape == weight.shape
    5. FFN: feedforward_channels == layers[0][0].out_features
    6. 检测头: len(cls_branches) == len(decoder.layers)
    7. 检测头: len(reg_branches) == len(decoder.layers)
    8. FFN 残差连接: 输出 Linear 的 out_features == 输入 Linear 的 in_features
    9. BEV embedding: weight.shape[1] 应与 embed_dims 一致

    Args:
        model: 剪枝并更新后的 UniV2X 模型

    Returns:
        违规列表，空列表表示通过验证
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

        # --- 注意力模块检查 ---
        if isinstance(module, (TemporalSelfAttention, MSDeformableAttention3D,
                                CustomMSDeformableAttention)):
            # 检查 embed_dims 与 value_proj 一致
            actual_dims = module.value_proj.out_features
            if module.embed_dims != actual_dims:
                violations.append(DimViolation(
                    module_name=name,
                    attribute="embed_dims",
                    expected=actual_dims,
                    actual=module.embed_dims,
                    description=(
                        f"embed_dims ({module.embed_dims}) != "
                        f"value_proj.out_features ({actual_dims})"
                    ),
                ))

            # 检查 embed_dims 可被 num_heads 整除
            if module.embed_dims % module.num_heads != 0:
                violations.append(DimViolation(
                    module_name=name,
                    attribute="num_heads",
                    expected=0,
                    actual=module.embed_dims % module.num_heads,
                    description=(
                        f"embed_dims ({module.embed_dims}) 不能被 "
                        f"num_heads ({module.num_heads}) 整除"
                    ),
                ))

            # 检查 head_dim
            if hasattr(module, "head_dim"):
                expected_hd = module.embed_dims // module.num_heads
                if module.head_dim != expected_hd:
                    violations.append(DimViolation(
                        module_name=name,
                        attribute="head_dim",
                        expected=expected_hd,
                        actual=module.head_dim,
                        description=(
                            f"head_dim ({module.head_dim}) != "
                            f"embed_dims//num_heads ({expected_hd})"
                        ),
                    ))

        # --- SCA embed_dims 检查 ---
        if isinstance(module, SpatialCrossAttention):
            actual_dims = module.output_proj.out_features
            if module.embed_dims != actual_dims:
                violations.append(DimViolation(
                    module_name=name,
                    attribute="embed_dims",
                    expected=actual_dims,
                    actual=module.embed_dims,
                    description=(
                        f"embed_dims ({module.embed_dims}) != "
                        f"output_proj.out_features ({actual_dims})"
                    ),
                ))

        # --- LayerNorm 检查 ---
        if isinstance(module, nn.LayerNorm):
            actual_shape = tuple(module.weight.shape)
            if module.normalized_shape != actual_shape:
                violations.append(DimViolation(
                    module_name=name,
                    attribute="normalized_shape",
                    expected=actual_shape[0] if actual_shape else -1,
                    actual=(module.normalized_shape[0]
                            if module.normalized_shape else -1),
                    description=(
                        f"normalized_shape {module.normalized_shape} != "
                        f"weight.shape {actual_shape}"
                    ),
                ))

    # --- 分支长度检查 ---
    head = None
    decoder = None
    for name, module in model.named_modules():
        if hasattr(module, "cls_branches") and hasattr(module, "transformer"):
            head = module
        if hasattr(module, "layers") and name.endswith("decoder"):
            decoder = module

    if head is not None and decoder is not None:
        num_layers = len(decoder.layers)
        if hasattr(head, "cls_branches"):
            cls_len = len(head.cls_branches)
            if cls_len != num_layers:
                violations.append(DimViolation(
                    module_name="cls_branches",
                    attribute="length",
                    expected=num_layers,
                    actual=cls_len,
                    description=(
                        f"cls_branches 长度 ({cls_len}) != "
                        f"decoder.layers 长度 ({num_layers})"
                    ),
                ))

        if hasattr(head, "reg_branches"):
            reg_len = len(head.reg_branches)
            if reg_len != num_layers:
                violations.append(DimViolation(
                    module_name="reg_branches",
                    attribute="length",
                    expected=num_layers,
                    actual=reg_len,
                    description=(
                        f"reg_branches 长度 ({reg_len}) != "
                        f"decoder.layers 长度 ({num_layers})"
                    ),
                ))

    # --- FFN 残差连接检查 ---
    for name, module in model.named_modules():
        if hasattr(module, "ffns"):
            for ffn_idx, ffn in enumerate(module.ffns):
                if hasattr(ffn, "layers") and len(ffn.layers) >= 2:
                    # layers[0][0] = 扩展 Linear (in->mid)
                    # layers[1] = 投影 Linear (mid->out)
                    try:
                        expand_linear = ffn.layers[0][0]
                        project_linear = ffn.layers[1]
                        if (isinstance(expand_linear, nn.Linear)
                                and isinstance(project_linear, nn.Linear)):
                            if expand_linear.in_features != project_linear.out_features:
                                violations.append(DimViolation(
                                    module_name=f"{name}.ffns.{ffn_idx}",
                                    attribute="residual_dims",
                                    expected=expand_linear.in_features,
                                    actual=project_linear.out_features,
                                    description=(
                                        f"FFN 残差维度不匹配: "
                                        f"in={expand_linear.in_features}, "
                                        f"out={project_linear.out_features}"
                                    ),
                                ))
                    except (IndexError, TypeError):
                        pass

    # --- BEV embedding 维度检查 ---
    for name, param in model.named_parameters():
        if "bev_embedding" in name and "weight" in name:
            bev_dim = param.shape[1]
            # 找到对应的 head 的 embed_dims
            if head is not None and bev_dim != head.embed_dims:
                violations.append(DimViolation(
                    module_name=name,
                    attribute="embed_dims",
                    expected=head.embed_dims,
                    actual=bev_dim,
                    description=(
                        f"bev_embedding 维度 ({bev_dim}) != "
                        f"head.embed_dims ({head.embed_dims})"
                    ),
                ))

    if violations:
        logger.warning("发现 %d 个维度不一致:", len(violations))
        for v in violations:
            logger.warning("  %s.%s: %s", v.module_name, v.attribute, v.description)
    else:
        logger.info("模型一致性验证通过: 无违规项")

    return violations
```

---

## 4. 代码检测方案

### Test 1: FFN 50% 剪枝验证

```python
"""
测试 P1: 将 FFN 中间维度从 512 剪枝到 256 (50% 保留比例)。
验证所有 FFN 中间 Linear 层的 out_features == 256。
"""
import torch
from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import prune_model
from projects.mmdet3d_plugin.univ2x.pruning.post_prune import (
    update_model_after_pruning,
    verify_model_consistency,
)


def test_ffn_pruning(model, example_inputs):
    """验证 FFN 中间维度被正确剪枝。"""
    prune_cfg = {
        "ffn_mid_ratio": 0.5,         # P1: 保留 50%
        "attn_proj_ratio": 0.0,       # P2: 不剪枝
        "head_mid_ratio": 1.0,        # P3: 不剪枝
        "head_pruning_ratio": 0.0,    # P8: 不剪枝
        "decoder_num_layers": 6,      # P9: 不剪枝
        "importance_criterion": "l1_norm",
        "iterative_steps": 1,
        "round_to": 8,
    }

    prune_model(model, example_inputs, prune_cfg)

    # 验证 FFN 中间维度
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "ffns" in name and "layers.0.0" in name:
            # 原始 feedforward_channels = 512, 50% -> 256
            # 因为 round_to=8, 实际值应为 256 (已对齐)
            assert module.out_features <= 256 + 8, (
                f"{name}: out_features={module.out_features}, "
                f"expected <= 264 (256 + round_to tolerance)"
            )
            assert module.out_features >= 248, (
                f"{name}: out_features={module.out_features}, "
                f"expected >= 248 (256 - round_to tolerance)"
            )
            print(f"PASS: {name} out_features={module.out_features}")

    print("Test 1 PASSED: FFN 剪枝验证通过")
```

### Test 2: 解码器层剪枝验证

```python
"""
测试 P9: 将解码器从 6 层剪枝到 4 层。
验证 decoder.layers, cls_branches, reg_branches, past_traj_reg_branches 长度一致。
"""


def test_decoder_layer_pruning(model):
    """验证解码器层被正确删除，分支同步更新。"""
    from projects.mmdet3d_plugin.univ2x.pruning.prune_univ2x import (
        prune_decoder_layers,
    )

    head = model.pts_bbox_head
    decoder = head.transformer.decoder

    # 验证初始状态
    assert len(decoder.layers) == 6, f"初始层数应为 6, 实际 {len(decoder.layers)}"
    assert len(head.cls_branches) == 6
    assert len(head.reg_branches) == 6
    assert len(head.past_traj_reg_branches) == 6

    # 执行层剪枝
    prune_decoder_layers(model, target_num_layers=4)

    # 验证剪枝结果
    assert len(decoder.layers) == 4, (
        f"decoder.layers 应为 4, 实际 {len(decoder.layers)}"
    )
    assert decoder.num_layers == 4, (
        f"decoder.num_layers 应为 4, 实际 {decoder.num_layers}"
    )
    assert len(head.cls_branches) == 4, (
        f"cls_branches 应为 4, 实际 {len(head.cls_branches)}"
    )
    assert len(head.reg_branches) == 4, (
        f"reg_branches 应为 4, 实际 {len(head.reg_branches)}"
    )
    assert len(head.past_traj_reg_branches) == 4, (
        f"past_traj_reg_branches 应为 4, "
        f"实际 {len(head.past_traj_reg_branches)}"
    )

    print("Test 2 PASSED: 解码器层剪枝验证通过")
```

### Test 3: 剪枝后 forward 通过性验证

```python
"""
测试: 剪枝 + 状态更新后，model.forward() 不产生维度不匹配错误。
"""


def test_forward_after_pruning(model, example_inputs):
    """验证剪枝后模型可正常前向传播。"""
    prune_cfg = {
        "ffn_mid_ratio": 0.7,
        "attn_proj_ratio": 0.1,
        "head_mid_ratio": 0.7,
        "head_pruning_ratio": 0.0,
        "decoder_num_layers": 5,
        "importance_criterion": "l1_norm",
        "iterative_steps": 1,
        "round_to": 8,
    }

    prune_model(model, example_inputs, prune_cfg)
    update_model_after_pruning(model)

    # 前向传播测试
    model.eval()
    with torch.no_grad():
        try:
            output = model(**example_inputs)
            print(f"Test 3 PASSED: forward 成功, 输出类型={type(output)}")
        except RuntimeError as e:
            print(f"Test 3 FAILED: forward 失败, 错误={e}")
            raise
```

### Test 4: 一致性验证

```python
"""
测试: verify_model_consistency 应在正确更新后返回空违规列表。
"""


def test_consistency_after_update(model, example_inputs):
    """验证状态更新后模型一致性。"""
    prune_cfg = {
        "ffn_mid_ratio": 0.5,
        "attn_proj_ratio": 0.1,
        "head_mid_ratio": 0.7,
        "decoder_num_layers": 5,
        "importance_criterion": "l1_norm",
        "iterative_steps": 1,
        "round_to": 8,
    }

    prune_model(model, example_inputs, prune_cfg)
    update_model_after_pruning(model)

    violations = verify_model_consistency(model)

    if violations:
        print("Test 4 FAILED: 发现以下违规:")
        for v in violations:
            print(f"  {v.module_name}.{v.attribute}: {v.description}")
        raise AssertionError(f"发现 {len(violations)} 个违规")
    else:
        print("Test 4 PASSED: 一致性验证通过, 无违规项")
```

---

## 5. Debug 方案

### 5.1 维度不匹配 (forward 中 RuntimeError: mat1 and mat2 shapes cannot be multiplied)

**Hook 维度追踪器**：在每个模块的 forward 前后记录 tensor 形状，定位不匹配发生的具体位置。

```python
def attach_shape_hooks(model):
    """为所有模块注册 forward hook，记录输入/输出 tensor 形状。"""
    hooks = []
    shape_log = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            input_shapes = []
            for inp in input:
                if isinstance(inp, torch.Tensor):
                    input_shapes.append(inp.shape)
                elif inp is None:
                    input_shapes.append(None)
            output_shapes = []
            if isinstance(output, torch.Tensor):
                output_shapes.append(output.shape)
            elif isinstance(output, (tuple, list)):
                for o in output:
                    if isinstance(o, torch.Tensor):
                        output_shapes.append(o.shape)
            shape_log[name] = {
                "input": input_shapes,
                "output": output_shapes,
            }
        return hook_fn

    for name, module in model.named_modules():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    return hooks, shape_log
```

**使用方法**：
1. 剪枝后先不运行 `update_model_after_pruning`
2. `attach_shape_hooks(model)`
3. 运行 `model.forward()`，捕获异常
4. 检查 `shape_log`，找到最后一个成功记录的模块，下一个模块就是崩溃点
5. 对比该模块的输入 shape 与其 weight shape

### 5.2 LayerNorm shape 不匹配

**症状**：`RuntimeError: Given normalized_shape=[256], expected input with shape [*, 256], but got input of size [*, 200]`

**定位方法**：

```python
def check_layernorm_shapes(model):
    """检查所有 LayerNorm 的 normalized_shape 与实际 weight 是否一致。"""
    mismatches = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            if module.normalized_shape != tuple(module.weight.shape):
                mismatches.append({
                    "name": name,
                    "normalized_shape": module.normalized_shape,
                    "weight_shape": tuple(module.weight.shape),
                })
    return mismatches
```

**修复**：确保 `update_model_after_pruning` 覆盖了所有 LayerNorm 实例。

### 5.3 解码器分支长度不匹配

**症状**：`IndexError: list index out of range` 在 `decoder.forward()` 中。

**诊断断言**：

```python
def assert_branch_consistency(model):
    """断言 decoder.layers 与所有 branches 长度一致。"""
    head = model.pts_bbox_head
    decoder = head.transformer.decoder
    n = len(decoder.layers)

    assert len(head.cls_branches) == n, (
        f"cls_branches={len(head.cls_branches)} != layers={n}"
    )
    assert len(head.reg_branches) == n, (
        f"reg_branches={len(head.reg_branches)} != layers={n}"
    )
    assert len(head.past_traj_reg_branches) == n, (
        f"past_traj_reg_branches={len(head.past_traj_reg_branches)} != layers={n}"
    )
    assert decoder.num_layers == n, (
        f"decoder.num_layers={decoder.num_layers} != len(layers)={n}"
    )
```

### 5.4 FFN 残差连接维度错误

**症状**：`RuntimeError: The size of tensor a (200) must match the size of tensor b (256)` 在残差 add 操作中。

**原因**：FFN 的输出 Linear 的 `out_features` 必须等于输入 Linear 的 `in_features`（即 `embed_dims`），否则残差连接 `output + identity` 维度不匹配。

**诊断**：

```python
def check_ffn_residual_dims(model):
    """检查 FFN 的输入/输出维度是否匹配（残差连接要求）。"""
    issues = []
    for name, module in model.named_modules():
        if hasattr(module, "ffns"):
            for idx, ffn in enumerate(module.ffns):
                if hasattr(ffn, "layers") and len(ffn.layers) >= 2:
                    try:
                        in_dim = ffn.layers[0][0].in_features
                        out_dim = ffn.layers[1].out_features
                        if in_dim != out_dim:
                            issues.append(
                                f"{name}.ffns.{idx}: "
                                f"in={in_dim}, out={out_dim}"
                            )
                    except (IndexError, AttributeError):
                        pass
    return issues
```

**修复**：Torch-Pruning 的 DepGraph 应该会自动保证 FFN 的输入输出维度一致（它们在同一个依赖组中）。如果出现不一致，检查 FFN 的 Linear 是否都被正确纳入 DepGraph 追踪。

---

## 6. 验收标准

| 编号 | 标准 | 验证方法 |
|:----:|------|---------|
| 1 | `prune_univ2x.py` 能剪枝任意 P1-P3, P8, P9 组合 | 运行 Test 1-4，覆盖单维度和多维度组合 |
| 2 | `post_prune.py` 正确更新所有内部属性 | `verify_model_consistency()` 返回空列表 |
| 3 | 剪枝 + 更新后 `model.forward()` 成功 | Test 3 无 RuntimeError |
| 4 | 解码器层剪枝后分支长度一致 | `len(cls_branches) == len(decoder.layers)` |
| 5 | 剪枝后模型可序列化 | `torch.save(model.state_dict(), path)` 后 `torch.load(path)` 成功 |
| 6 | P1=0.5 时 FFN 中间 Linear 的 out_features 约为 256 | Test 1 断言通过 |
| 7 | P9=4 时 decoder.layers 长度为 4 | Test 2 断言通过 |
| 8 | 所有 LayerNorm 的 normalized_shape 与 weight.shape 一致 | `verify_model_consistency` 中的 LN 检查 |
| 9 | BEV embedding 维度与 embed_dims 一致 | `verify_model_consistency` 中的 BEV 检查 |

---

## 7. 风险与缓解

### 风险 1: cls_branches 索引 off-by-one

**描述**：`track_head.py` 中 `_init_layers()` (lines 110-111) 的 `num_pred` 计算逻辑：
```python
num_pred = (self.transformer.decoder.num_layers + 1) if \
    self.as_two_stage else self.transformer.decoder.num_layers
```
当 `as_two_stage=False`（UniV2X 默认）时，`num_pred == num_layers`。
`get_detections()` 中使用 `for lvl in range(hs.shape[0])` 遍历，其中 `hs.shape[0] == num_layers`。

**风险点**：若错误地将 `cls_branches` 长度设为 `num_layers + 1`（误以为需要额外的 proposal 分支），则最后一个分支永远不会被使用，浪费参数且可能在其他代码路径中引发索引错误。

**缓解**：
- 在 `prune_decoder_layers()` 末尾加入严格断言：`assert len(cls_branches) == len(decoder.layers)`
- 不支持 `as_two_stage=True` 的分支（UniV2X 未使用）

### 风险 2: FFN 残差连接维度不匹配

**描述**：FFN 的结构是 `output = Linear2(act(Linear1(x))) + x`。如果 `Linear2.out_features != x.shape[-1]`（即 `embed_dims`），残差 add 会维度不匹配。

**风险场景**：如果 DepGraph 没有正确识别 `Linear1` 和 `Linear2` 属于同一个 FFN 模块，可能只剪枝了中间维度但没有同步调整投影维度。

**缓解**：
- FFN 的两个 Linear 层通过 DepGraph 的自动追踪应该在同一个依赖组中
- `verify_model_consistency()` 显式检查 FFN 的输入/输出维度
- Debug 方案 5.4 提供了诊断工具

### 风险 3: BEV embedding 维度未更新

**描述**：`bev_embedding` 是 `nn.Embedding(bev_h*bev_w, embed_dims)`。如果通道剪枝修改了 `embed_dims` 但没有同步剪枝 `bev_embedding.weight`，则 `query_pos` 维度不匹配。

**风险表现**：`bev_queries = self.bev_embedding.weight.to(dtype)` 返回 `(bev_h*bev_w, old_embed_dims)`，后续与 `(bev_h*bev_w, new_embed_dims)` 的 BEV 特征相加时维度不匹配。

**缓解**：
- 在 `_collect_unwrapped_params()` 中注册 `bev_embedding.weight` 作为 unwrapped parameter，确保 DepGraph 追踪到它
- `verify_model_consistency()` 检查 `bev_embedding.weight.shape[1] == head.embed_dims`

### 风险 4: query_embedding 维度为 embed_dims*2

**描述**：`query_embedding = nn.Embedding(num_query+1, embed_dims*2)`。前半是 query content，后半是 positional encoding。剪枝 `embed_dims` 时，需要同时剪枝两个半区的对应通道。

**缓解**：
- 作为 unwrapped parameter 注册，剪枝维度 = 1
- 需要在自定义剪枝器中处理 `*2` 的映射关系（A1 阶段应已解决）
- 若 DepGraph 无法自动处理，fallback 方案是在 `update_model_after_pruning()` 中手动重建 `query_embedding`

### 风险 5: head_dim 不是 2 的幂

**描述**：`MSDeformableAttention3D` 中有检查 `dim_per_head` 是否为 2 的幂（CUDA 效率要求）。剪枝后 `embed_dims` 改变可能导致 `head_dim` 不再是 2 的幂。

**缓解**：
- `round_to=8` (P7) 约束通道对齐到 8 的倍数
- 默认 `num_heads=8`，`embed_dims` 对齐到 8 后，`head_dim` 也对齐到 1
- 这只是 warning 不是 error，CUDA kernel 仍可工作但效率稍低
