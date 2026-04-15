"""
梯度收集器：为 Taylor/Hessian 重要性评估收集梯度信息。

使用方法：
    在调用 DepGraph 剪枝前，先用少量校准数据计算梯度。
    DepGraph 的 GroupTaylorImportance 会读取 param.grad 来评估通道重要性。

已知问题 (2026-04-15)：
    在 UniV2X MultiAgent wrapper 上通过 MMDataParallel.train_step 收集梯度会
    因为嵌套 DataContainer scatter 不完整而失败 (indices device mismatch)。
    这会阻塞 Phase B.2 的 Taylor 对比实验，但不影响 L1/FPGM 准则和所有
    不需要梯度的实验。暂时降级为 L1 基线，待需要 Taylor 时再修。
    修复方向：手动 scatter ego_agent_data / other_agent_data_dict 中的
    DataContainer 到 GPU。参考 mmcv/parallel/scatter_gather.py 的实现。
"""
import logging

import torch

logger = logging.getLogger(__name__)


def collect_gradients(
    model: torch.nn.Module,
    dataloader,
    num_batches: int = 32,
    device=None,
) -> None:
    """用少量数据计算梯度，供 Taylor/Hessian 重要性评估使用。

    梯度会累积在各参数的 .grad 属性上（不清零），供后续
    tp.importance.GroupTaylorImportance 读取。

    Args:
        model: UniV2X 模型（需要支持 train_step 或 forward + loss）
        dataloader: mmdet3d 格式的数据加载器
        num_batches: 使用多少个 batch 来累积梯度（默认 32）
        device: 计算设备，None 时使用模型当前设备
    """
    from mmcv.parallel import MMDataParallel

    model.train()
    model.requires_grad_(True)

    # 清零已有梯度
    model.zero_grad()

    # mmdet3d 的 dataloader 返回 DataContainer 包装的 batch，
    # 需要通过 MMDataParallel 自动解包。如果模型已经是 MMDataParallel 则直接用。
    if isinstance(model, MMDataParallel):
        wrapped = model
        inner_model = model.module
    else:
        # 放到 GPU 并包装
        device_id = next(model.parameters()).device.index or 0
        wrapped = MMDataParallel(model.cuda(device_id), device_ids=[device_id])
        inner_model = model

    collected = 0
    for batch_idx, data in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        try:
            # MMDataParallel.train_step 会自动展开 DataContainer
            losses = wrapped.train_step(data, optimizer=None)
            if isinstance(losses, dict):
                # Runner 风格返回 {"loss": ..., "log_vars": ...}
                if "loss" in losses and isinstance(losses["loss"], torch.Tensor):
                    loss = losses["loss"]
                else:
                    loss = _extract_loss(losses)
            else:
                loss = losses

            if loss is not None and loss.requires_grad:
                loss.backward()
                collected += 1
            else:
                logger.warning(
                    f"Batch {batch_idx} produced no grad-required loss"
                )

        except Exception as e:
            logger.warning(
                f"Batch {batch_idx} gradient collection failed: {e}"
            )
            continue

    logger.info(
        f"Gradient collection: {collected}/{num_batches} batches, "
        f"grad available on {_count_grad_params(inner_model)} parameters"
    )

    # 切回 eval 模式，但保留 .grad
    inner_model.eval()


def _extract_loss(loss_dict: dict):
    """从 mmdet3d 的 loss dict 中提取总 loss。"""
    losses = []
    for key, value in loss_dict.items():
        if 'loss' in key and isinstance(value, torch.Tensor) and value.requires_grad:
            losses.append(value)

    if not losses:
        return None

    return sum(losses)


def _count_grad_params(model: torch.nn.Module) -> int:
    """统计有梯度的参数数量。"""
    return sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
