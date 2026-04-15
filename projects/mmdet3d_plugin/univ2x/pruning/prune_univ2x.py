"""
剪枝执行引擎主入口。

接收 prune_config 配置，通过 Torch-Pruning DepGraph 框架执行结构化剪枝。
搜索维度：P1 (FFN中间维度), P2 (注意力投影通道), P3 (检测头中间维度), P8 (注意力头剪枝), P9 (解码器层数)
锁定维度：P4 (importance_criterion), P5 (pruning_granularity), P6 (iterative_steps), P7 (round_to)
"""
import fnmatch
import logging
from typing import Any, Dict, List, Tuple

import torch.nn as nn
import torch_pruning as tp

from .custom_pruners import register_univ2x_pruners

logger = logging.getLogger(__name__)


# ============================================================================
# 公共 API
# ============================================================================

def prune_model(
    model: nn.Module,
    example_inputs,
    prune_cfg: Dict[str, Any],
) -> nn.Module:
    """剪枝主入口：按 prune_cfg 执行完整剪枝流程。

    执行顺序：
      1. P9 解码器层剪枝（先做，不涉及通道维度）
      2. P1-P3, P8 通道/头剪枝

    剪枝策略：
      - 若 example_inputs is None: 使用直接剪枝（bypass DepGraph tracing）
        适用于 forward 签名复杂无法 trace 的模型（如 UniV2X）
      - 若 example_inputs 提供: 使用 DepGraph 全自动追踪

    剪枝后需要调用 post_prune.update_model_after_pruning() 更新模块属性。
    """
    # Step 1: P9 解码器层剪枝
    target_num_layers = prune_cfg.get("decoder_num_layers", 6)
    decoder = model.pts_bbox_head.transformer.decoder
    current_num_layers = len(decoder.layers)
    if target_num_layers < current_num_layers:
        logger.info("P9: 解码器层剪枝 %d -> %d", current_num_layers, target_num_layers)
        prune_decoder_layers(model, target_num_layers)

    # Step 2: 通道/头剪枝 (P1-P3, P8)
    has_channel_pruning = (
        prune_cfg.get("ffn_mid_ratio", 1.0) < 1.0
        or prune_cfg.get("attn_proj_ratio", 0.0) > 0.0
        or prune_cfg.get("head_mid_ratio", 1.0) < 1.0
        or prune_cfg.get("head_pruning_ratio", 0.0) > 0.0
        or any(prune_cfg.get("per_layer_override", {}).values())
    )

    if has_channel_pruning:
        if example_inputs is None:
            logger.info("P1-P3/P8: 执行直接剪枝 (bypass DepGraph tracing)")
            prune_direct(model, prune_cfg)
        else:
            logger.info("P1-P3/P8: 执行 DepGraph 剪枝")
            pruner = build_pruner(model, example_inputs, prune_cfg)
            pruner.step()
        logger.info("通道/头剪枝完成")

    return model


# ============================================================================
# 直接剪枝（bypass DepGraph tracing）
# ============================================================================

def prune_direct(model: nn.Module, prune_cfg: Dict[str, Any]) -> None:
    """直接剪枝实现，不依赖 DepGraph 对整个模型的 forward trace。

    剪枝策略：
      1. FFN: Linear(d->ffn_d) 输出 + Linear(ffn_d->d) 输入 同步剪枝（局部依赖）
      2. value_proj/output_proj: 通过自定义剪枝器，保证坐标敏感层不受影响
      3. head_mid: cls_branches/reg_branches 中间 Linear 剪枝

    注：不使用 DepGraph 的代价是失去了跨模块依赖追踪，但 UniV2X 剪枝目标
    全部是局部结构 (FFN/attn_proj/head)，残差连接保证 in=out，无需跨模块。
    """
    import torch

    ffn_prune_ratio = 1.0 - prune_cfg.get("ffn_mid_ratio", 1.0)
    attn_prune_ratio = prune_cfg.get("attn_proj_ratio", 0.0)
    head_prune_ratio = 1.0 - prune_cfg.get("head_mid_ratio", 1.0)
    per_layer = prune_cfg.get("per_layer_override", {})
    round_to = prune_cfg.get("round_to", 8)
    criterion = prune_cfg.get("importance_criterion", "l1_norm")

    from torch_pruning.pruner.function import (
        prune_linear_in_channels,
        prune_linear_out_channels,
    )

    def _pick_channels_by_importance(linear: nn.Linear, num_to_prune: int):
        """根据重要性准则选择要剪掉的输出通道索引"""
        if num_to_prune <= 0:
            return []
        weight = linear.weight.data  # (out, in)
        if criterion == "l1_norm":
            scores = weight.abs().sum(dim=1)
        elif criterion == "l2_norm":
            scores = (weight ** 2).sum(dim=1).sqrt()
        elif criterion == "fpgm":
            # geometric median: sum of distances to other filters
            scores = torch.cdist(weight.unsqueeze(0), weight.unsqueeze(0)).squeeze(0).sum(dim=1)
            scores = -scores  # lower distance = more redundant
        elif criterion == "taylor":
            if linear.weight.grad is not None:
                scores = (weight * linear.weight.grad).abs().sum(dim=1)
            else:
                logger.warning("  Taylor 缺少梯度, 回退到 L1")
                scores = weight.abs().sum(dim=1)
        else:
            scores = weight.abs().sum(dim=1)  # fallback

        # 按重要性升序，剪掉分数最低的 num_to_prune 个
        _, idxs = torch.sort(scores, descending=False)
        return idxs[:num_to_prune].tolist()

    def _align(n: int, round_to: int) -> int:
        """对齐到 round_to 倍数"""
        return (n // round_to) * round_to

    # ── P1: FFN 中间层剪枝 ──
    ffn_pairs = _collect_ffn_pairs(model)
    logger.info("找到 %d 对 FFN Linear", len(ffn_pairs))

    for name, first_linear, second_linear in ffn_pairs:
        # 判断剪枝比例 (per_layer_override 优先)
        ratio = ffn_prune_ratio
        for prefix, override in per_layer.items():
            if prefix in name and "ffn_mid_ratio" in override:
                ratio = 1.0 - override["ffn_mid_ratio"]
                break
        if ratio <= 0:
            continue

        orig_out = first_linear.out_features
        target_keep = _align(int(orig_out * (1.0 - ratio)), round_to)
        num_prune = orig_out - target_keep
        if num_prune <= 0:
            continue

        idxs = _pick_channels_by_importance(first_linear, num_prune)
        # 同步剪枝: first.out 和 second.in
        prune_linear_out_channels(first_linear, idxs)
        prune_linear_in_channels(second_linear, idxs)
        logger.debug("  FFN %s: %d -> %d (prune %d)", name, orig_out, target_keep, num_prune)

    # ── P2: value_proj/output_proj 剪枝 (使用自定义剪枝器保证一致性) ──
    if attn_prune_ratio > 0:
        from .custom_pruners import register_univ2x_pruners
        pruners = register_univ2x_pruners()

        for name, module in model.named_modules():
            pruner_cls = type(module)
            if pruner_cls not in pruners:
                continue

            # 获取当前 embed_dims
            try:
                current = pruners[pruner_cls].get_out_channels(module)
            except Exception:
                continue

            target_keep = _align(int(current * (1.0 - attn_prune_ratio)), round_to)
            num_prune = current - target_keep
            if num_prune <= 0:
                continue

            # 按 value_proj 权重选择索引（代表 embed_dims 重要性）
            vp = getattr(module, "value_proj", None)
            if vp is None:
                continue
            idxs = _pick_channels_by_importance(vp, num_prune)

            # 应用 out + in 剪枝
            pruners[pruner_cls].prune_out_channels(module, idxs)
            pruners[pruner_cls].prune_in_channels(module, idxs)
            logger.debug("  Attn %s: %d -> %d", name, current, target_keep)

    # ── P3: 检测头中间层剪枝 ──
    if head_prune_ratio > 0:
        head_pairs = _collect_head_pairs(model)
        for name, first_linear, second_linear in head_pairs:
            orig_out = first_linear.out_features
            target_keep = _align(int(orig_out * (1.0 - head_prune_ratio)), round_to)
            num_prune = orig_out - target_keep
            if num_prune <= 0:
                continue
            idxs = _pick_channels_by_importance(first_linear, num_prune)
            prune_linear_out_channels(first_linear, idxs)
            prune_linear_in_channels(second_linear, idxs)
            logger.debug("  Head %s: %d -> %d", name, orig_out, target_keep)


def _collect_ffn_pairs(model: nn.Module):
    """收集所有 FFN 的 (first_linear, second_linear) 对。

    支持两种常见 FFN/MLP 结构:

    1. mmcv FFN (pts_bbox_head + seg_head.transformer):
       module.ffns[i].layers[0] = Sequential(Linear(d, ffn_d), activation, [dropout])
       module.ffns[i].layers[1] = Linear(ffn_d, d)

    2. mask_head MLP (seg_head.stuff_mask_head + things_mask_head 的 blocks.N.mlp):
       block.mlp.fc1 = Linear(d, ffn_d)
       block.mlp.fc2 = Linear(ffn_d, d)
    """
    pairs = []

    # --- 模式 1: mmcv FFN (module.ffns) ---
    for name, module in model.named_modules():
        if not hasattr(module, "ffns"):
            continue
        for ffn_idx, ffn in enumerate(module.ffns):
            if not hasattr(ffn, "layers") or len(ffn.layers) < 2:
                continue
            try:
                first = ffn.layers[0][0]  # Sequential[0] = Linear
                second = ffn.layers[1]     # Linear(ffn_d, d)
                if not isinstance(second, nn.Linear):
                    for sub in ffn.layers[1:]:
                        if isinstance(sub, nn.Linear):
                            second = sub
                            break
                if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                    pair_name = f"{name}.ffns.{ffn_idx}"
                    pairs.append((pair_name, first, second))
            except (IndexError, AttributeError):
                continue

    # --- 模式 2: mask_head MLP (block.mlp.fc1/fc2) ---
    # 关键签名: 存在名为 'mlp' 的子模块, 且其有 fc1/fc2 子属性
    for name, module in model.named_modules():
        if not hasattr(module, "mlp"):
            continue
        mlp = module.mlp
        if hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            first = mlp.fc1
            second = mlp.fc2
            if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                # 只有当 fc1.out == fc2.in 时才是合法的 MLP 对
                if first.out_features == second.in_features:
                    pair_name = f"{name}.mlp"
                    pairs.append((pair_name, first, second))

    return pairs


def _collect_head_pairs(model: nn.Module):
    """收集检测头中间层的 (first, second) 对。

    cls_branches[i] = Sequential(Linear, LN, ReLU, Linear, LN, ReLU, Linear_out)
      中间对: (layers[0]=Linear, layers[3]=Linear), (layers[3]=Linear, layers[6]=Linear_out)
    但 layers[6] 是输出层不剪枝, 所以只剪 (0->3) 这一对
    """
    pairs = []
    for name, module in model.named_modules():
        # cls_branches: Sequential with mid layers at indices 0, 3 (output at 6)
        if name.endswith("cls_branches"):
            for branch_idx, branch in enumerate(module):
                if not isinstance(branch, nn.Sequential):
                    continue
                # 仅剪 0 -> 3 这一对 (6 是输出, 不剪)
                try:
                    first = branch[0]
                    second = branch[3]
                    if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                        pair_name = f"{name}.{branch_idx}"
                        pairs.append((pair_name, first, second))
                except (IndexError, TypeError):
                    continue
        # reg_branches: Sequential(Linear, ReLU, Linear_out)
        # 中间维度由第一个 Linear 的 out 决定, 输出层在 layers[2]
        # 这里 mid Linear (index 0) 的 out 要和下一个 Linear (index 2) 的 in 同步
        elif name.endswith("reg_branches") or name.endswith("past_traj_reg_branches"):
            for branch_idx, branch in enumerate(module):
                if not isinstance(branch, nn.Sequential):
                    continue
                try:
                    first = branch[0]
                    second = branch[2]  # 跳过 ReLU
                    if isinstance(first, nn.Linear) and isinstance(second, nn.Linear):
                        pair_name = f"{name}.{branch_idx}"
                        pairs.append((pair_name, first, second))
                except (IndexError, TypeError):
                    continue
    return pairs


# ============================================================================
# Pruner 构建
# ============================================================================

def build_pruner(
    model: nn.Module,
    example_inputs: Dict[str, Any],
    prune_cfg: Dict[str, Any],
) -> tp.pruner.MetaPruner:
    """构建 MetaPruner 实例。

    锁定维度 (P4-P7) 作为 Pruner 构造参数传入，
    搜索维度 (P1-P3, P8) 通过 ratio_dict / head_pruning_ratio 控制。
    """
    customized_pruners = register_univ2x_pruners()
    ignored_layers = _collect_ignored_layers(model, prune_cfg)
    unwrapped_parameters = _collect_unwrapped_params(model)

    # P4: 重要性评估准则（锁定维度）
    importance = _select_importance(
        prune_cfg.get("importance_criterion", "taylor")
    )

    # P1-P3: 构建 per-module 剪枝比例字典
    ratio_dict = _build_ratio_dict(model, prune_cfg)

    # P8: 注意力头剪枝
    head_pruning_ratio = prune_cfg.get("head_pruning_ratio", 0.0)
    num_heads_dict = {}
    if head_pruning_ratio > 0.0:
        num_heads_dict = _collect_num_heads(model)

    # P5: 剪枝粒度（锁定维度）
    granularity = prune_cfg.get("pruning_granularity", "local")
    global_pruning = (granularity == "global")
    isomorphic = (granularity == "isomorphic")

    # P6: 迭代步数（锁定维度）
    iterative_steps = prune_cfg.get("iterative_steps", 5)

    # P7: 通道对齐（锁定维度，INT8 硬约束）
    round_to = prune_cfg.get("round_to", 8)

    # num_heads 和 head_pruning_ratio 必须成对出现
    # torch_pruning 要求 num_heads 为 dict (即使空), 不能为 None
    kwargs = dict(
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
    )
    if head_pruning_ratio > 0 and num_heads_dict:
        kwargs["head_pruning_ratio"] = head_pruning_ratio
        kwargs["num_heads"] = num_heads_dict

    pruner = tp.pruner.MetaPruner(**kwargs)

    return pruner


# ============================================================================
# 重要性评估准则选择 (P4)
# ============================================================================

def _select_importance(criterion: str):
    """将字符串准则名映射到 tp.importance 类实例。"""
    criterion_map = {
        "l1_norm": tp.importance.MagnitudeImportance(p=1),
        "l2_norm": tp.importance.MagnitudeImportance(p=2),
        "taylor": tp.importance.GroupTaylorImportance(),
        "bn_scale": tp.importance.BNScaleImportance(),
        "fpgm": tp.importance.FPGMImportance(p=2),
        "hessian": tp.importance.GroupHessianImportance(),
        "random": tp.importance.RandomImportance(),
    }
    if criterion not in criterion_map:
        raise ValueError(
            f"不支持的重要性准则: {criterion}，可选: {list(criterion_map.keys())}"
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

    注意：ratio 值表示"剪掉的比例"。
      - P1 ffn_mid_ratio 是"保留比例"，转换：prune_ratio = 1 - ffn_mid_ratio
      - P2 attn_proj_ratio 已经是"剪掉的比例"
      - P3 head_mid_ratio 是"保留比例"，转换：prune_ratio = 1 - head_mid_ratio

    支持 per_layer_override：对特定层应用不同的剪枝比例。
      例如 {"encoder.layers.0.ffns": {"ffn_mid_ratio": 0.6}} 表示
      仅对 encoder.layers.0 的 FFN 层应用 40% 剪枝。
    """
    ffn_prune_ratio = 1.0 - prune_cfg.get("ffn_mid_ratio", 1.0)
    attn_prune_ratio = prune_cfg.get("attn_proj_ratio", 0.0)
    head_prune_ratio = 1.0 - prune_cfg.get("head_mid_ratio", 1.0)

    # per-layer override: {layer_name_prefix: {ffn_mid_ratio: ...}}
    per_layer_override = prune_cfg.get("per_layer_override", {})

    ratio_dict = {}
    ffn_count, attn_count, head_count = 0, 0, 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if _is_ffn_layer(name):
            # 检查是否有 per-layer override
            layer_ratio = ffn_prune_ratio
            for prefix, override in per_layer_override.items():
                if prefix in name and "ffn_mid_ratio" in override:
                    layer_ratio = 1.0 - override["ffn_mid_ratio"]
                    break
            if layer_ratio > 0:
                ratio_dict[module] = layer_ratio
                ffn_count += 1
        elif _is_attn_proj(name) and attn_prune_ratio > 0:
            ratio_dict[module] = attn_prune_ratio
            attn_count += 1
        elif _is_head_layer(name) and head_prune_ratio > 0:
            ratio_dict[module] = head_prune_ratio
            head_count += 1

    logger.info(
        "ratio_dict: FFN=%d层(ratio=%.2f), attn_proj=%d层(ratio=%.2f), head=%d层(ratio=%.2f)",
        ffn_count, ffn_prune_ratio, attn_count, attn_prune_ratio, head_count, head_prune_ratio,
    )
    return ratio_dict


# ============================================================================
# 模块名分类器
# ============================================================================

def _is_ffn_layer(name: str) -> bool:
    """判断是否为 FFN 中间层 (Linear: embed_dims -> feedforward_channels)。

    路径模式: *.ffns.*.layers.0.0 (中间层的 Linear)
    """
    return "ffns" in name and "layers.0.0" in name


def _is_attn_proj(name: str) -> bool:
    """判断是否为注意力投影层 (value_proj / output_proj)。

    排除 FFN 和 head 中可能的同名层。
    """
    if "value_proj" in name or "output_proj" in name:
        if "ffns" not in name and "branches" not in name:
            return True
    return False


def _is_head_layer(name: str) -> bool:
    """判断是否为检测头中间层（非最终输出层）。

    cls_branches: [Linear,LN,ReLU] × num_reg_fcs + [Linear(输出)]
      中间层索引: 0, 3 (默认 num_reg_fcs=2)
      输出层索引: 6

    reg_branches / past_traj_reg_branches: [Linear,ReLU] × num_reg_fcs + [Linear(输出)]
      中间层索引: 0
      输出层索引: 2
    """
    if "cls_branches" in name:
        parts = name.split(".")
        try:
            idx = int(parts[-1])
            return idx != 6  # 6 是最终输出层
        except (ValueError, IndexError):
            return False

    if "reg_branches" in name:  # 同时匹配 reg_branches 和 past_traj_reg_branches
        parts = name.split(".")
        try:
            idx = int(parts[-1])
            return idx != 2  # 2 是最终输出层
        except (ValueError, IndexError):
            return False

    return False


# ============================================================================
# 忽略层收集
# ============================================================================

def _collect_ignored_layers(
    model: nn.Module,
    prune_cfg: Dict[str, Any],
) -> List[nn.Module]:
    """收集不参与剪枝的层：坐标敏感层 + 输出层 + 用户指定层。"""
    ignored = []
    keywords = ["sampling_offsets", "attention_weights"]
    keywords += prune_cfg.get("ignored_layer_keywords", [])

    for name, module in model.named_modules():
        if any(kw in name for kw in keywords):
            ignored.append(module)
        elif _is_output_layer(name):
            ignored.append(module)

    logger.info("收集到 %d 个忽略层", len(ignored))
    return ignored


def _is_output_layer(name: str) -> bool:
    """判断是否为输出层（不可剪枝）。"""
    if "cls_branches" in name:
        parts = name.split(".")
        try:
            return int(parts[-1]) == 6
        except (ValueError, IndexError):
            pass

    if "reg_branches" in name:
        parts = name.split(".")
        try:
            return int(parts[-1]) == 2
        except (ValueError, IndexError):
            pass

    if name.endswith("reference_points"):
        return True

    return False


# ============================================================================
# 注意力头数收集 (P8)
# ============================================================================

def _collect_num_heads(model: nn.Module) -> Dict[nn.Module, int]:
    """收集所有注意力模块的 num_heads 属性。"""
    num_heads_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, "num_heads") and hasattr(module, "embed_dims"):
            num_heads_dict[module] = module.num_heads
    logger.info("收集到 %d 个注意力模块的 num_heads", len(num_heads_dict))
    return num_heads_dict


# ============================================================================
# Unwrapped 参数收集
# ============================================================================

def _collect_unwrapped_params(model: nn.Module) -> List[Tuple[nn.Parameter, int]]:
    """收集位置编码等非 Module 包裹的独立参数。"""
    unwrapped = []
    for name, param in model.named_parameters():
        if "bev_embedding" in name and "weight" in name:
            unwrapped.append((param, 1))  # (bev_h*bev_w, embed_dims), 剪枝 dim=1
        elif "query_embedding" in name and "weight" in name:
            unwrapped.append((param, 1))  # (num_query+1, embed_dims*2), 剪枝 dim=1
    logger.info("收集到 %d 个 unwrapped 参数", len(unwrapped))
    return unwrapped


# ============================================================================
# P9: 解码器层剪枝
# ============================================================================

def prune_decoder_layers(model: nn.Module, target_num_layers: int) -> None:
    """P9：删除解码器层并同步 cls_branches/reg_branches。

    策略：保留最后 N 层（靠后的层对精度贡献更大）。
    """
    head = model.pts_bbox_head
    decoder = head.transformer.decoder
    current_num_layers = len(decoder.layers)

    if target_num_layers >= current_num_layers:
        logger.warning("target=%d >= current=%d, 跳过", target_num_layers, current_num_layers)
        return
    if target_num_layers <= 0:
        raise ValueError(f"target_num_layers 必须 > 0，当前: {target_num_layers}")

    keep_start = current_num_layers - target_num_layers
    logger.info("解码器层剪枝: 删除前 %d 层, 保留 L%d..L%d",
                keep_start, keep_start, current_num_layers - 1)

    # 裁剪 decoder.layers
    decoder.layers = nn.ModuleList(list(decoder.layers)[keep_start:])
    decoder.num_layers = target_num_layers

    # 同步裁剪各分支
    for attr in ("cls_branches", "reg_branches", "past_traj_reg_branches"):
        if hasattr(head, attr):
            branches = getattr(head, attr)
            new_branches = nn.ModuleList(list(branches)[keep_start:])
            setattr(head, attr, new_branches)
            logger.info("%s: %d -> %d", attr, len(branches), len(new_branches))

    # 验证一致性
    n = len(decoder.layers)
    for attr in ("cls_branches", "reg_branches", "past_traj_reg_branches"):
        if hasattr(head, attr):
            assert len(getattr(head, attr)) == n, \
                f"{attr} 长度 {len(getattr(head, attr))} != decoder.layers 长度 {n}"

    logger.info("P9 完成: %d -> %d 层", current_num_layers, target_num_layers)


# ============================================================================
# 统一配置入口 (prune_config.json -> 完整剪枝流程)
# ============================================================================

def apply_prune_config(model, prune_config, dataloader=None):
    """从 prune_config.json 解析配置并执行完整剪枝流程。

    编排步骤：
      1. 解析 locked + search 维度
      2. 收集梯度（如果使用 Taylor/Hessian）
      3. 构建示例输入
      4. 执行剪枝（P9 层数 + P1-P3/P8 通道）
      5. 更新模块内部状态
      6. 一致性验证

    Args:
        model: 已加载权重的 UniV2X 模型
        prune_config: 从 prune_config.json 加载的配置字典
        dataloader: 训练数据加载器（Taylor/Hessian 需要）

    Returns:
        剪枝后的模型
    """
    from .post_prune import update_model_after_pruning, verify_model_consistency
    from .grad_collector import collect_gradients
    import torch

    locked = prune_config.get("locked", {})
    enc = prune_config.get("encoder", {})
    dec = prune_config.get("decoder", {})
    heads = prune_config.get("heads", {})

    # Step 1: 收集梯度（锁定维度 P4 决定是否需要）
    criterion = locked.get("importance_criterion", "taylor")
    if criterion in ("taylor", "hessian") and dataloader is not None:
        logger.info("收集梯度 (criterion=%s)...", criterion)
        collect_gradients(model, dataloader, num_batches=32)

    # Step 2: 构建统一的 prune_cfg（合并 locked + search 维度）
    prune_cfg = {
        # 锁定维度
        "importance_criterion": criterion,
        "pruning_granularity": locked.get("pruning_granularity", "local"),
        "iterative_steps": locked.get("iterative_steps", 5),
        "round_to": locked.get("round_to", 8),
        # 搜索维度
        "ffn_mid_ratio": enc.get("ffn_mid_ratio", 1.0),
        "attn_proj_ratio": enc.get("attn_proj_ratio", 0.0),
        "head_mid_ratio": heads.get("head_mid_ratio", 1.0),
        "head_pruning_ratio": enc.get("head_pruning_ratio", 0.0),
        "decoder_num_layers": dec.get("num_layers", 6),
        # 约束
        "ignored_layer_keywords": prune_config.get("constraints", {}).get(
            "skip_layers", ["sampling_offsets", "attention_weights"]
        ),
    }

    # Step 3: 执行剪枝
    # 对 UniV2X 这类复杂 forward 签名的模型, 使用直接剪枝模式 (bypass tracing)
    # 若需要 DepGraph 全自动追踪, 可在 prune_config 中设置 "use_depgraph": True
    if prune_cfg.get("use_depgraph", False):
        logger.info("构建示例输入...")
        example_inputs = _build_example_inputs(model)
    else:
        example_inputs = None

    # Step 4: 执行剪枝
    model = prune_model(model, example_inputs, prune_cfg)

    # Step 5: 更新模块内部状态
    logger.info("更新模块内部状态...")
    update_model_after_pruning(model)

    # Step 6: 一致性验证
    violations = verify_model_consistency(model)
    if violations:
        logger.warning("一致性验证发现 %d 个问题，请检查", len(violations))

    return model


def _build_example_inputs(model):
    """构建 DepGraph 追踪所需的示例输入。

    UniV2X 模型的 forward 需要复杂的输入结构，但 DepGraph 追踪
    只需要 tensor 的形状正确即可（不需要语义正确的值）。
    """
    import torch

    device = next(model.parameters()).device

    # 从模型配置中推断维度
    embed_dims = 256
    bev_h, bev_w = 100, 100
    num_query = 900

    for name, module in model.named_modules():
        if hasattr(module, "embed_dims"):
            embed_dims = module.embed_dims
            break

    # BEV query: 编码器的主要输入
    bev_query = torch.randn(1, bev_h * bev_w, embed_dims, device=device)

    return bev_query
