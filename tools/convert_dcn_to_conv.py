"""
DCN → 普通 Conv2d 权重转换工具。

目的: 将 UniV2X 中 img_backbone 的 ModulatedDeformConv2dPack 替换为普通 Conv2d,
      以便解锁 backbone 的剪枝和量化能力。

原理: DCN 的核心是一个普通的 Conv2d 权重 + offset/mask 分支。
      核心权重的 shape 与普通 Conv2d 完全一致, 可直接迁移。
      offset/mask 分支会被丢弃 (精度损失来源)。

用法:
    python tools/convert_dcn_to_conv.py \
        --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
        --input-ckpt ckpts/univ2x_coop_e2e_stg2.pth \
        --output-ckpt ckpts/univ2x_coop_e2e_stg2_no_dcn.pth \
        --output-config projects/configs_e2e_univ2x/univ2x_coop_e2e_track_no_dcn.py
"""
import argparse
import copy
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dcn_convert")


def parse_args():
    parser = argparse.ArgumentParser(description="DCN → Conv2d 权重转换")
    parser.add_argument("--config", required=True, help="mmdet3d 配置文件")
    parser.add_argument("--input-ckpt", required=True, help="原始 checkpoint (含 DCN)")
    parser.add_argument("--output-ckpt", required=True, help="转换后 checkpoint")
    parser.add_argument("--output-config", default=None,
                        help="可选: 保存修改后的 config (替换 dcn=None)")
    parser.add_argument("--verify", action="store_true",
                        help="转换后加载验证, 确认模型可正常 forward")
    return parser.parse_args()


def convert_dcn_in_model(model):
    """就地替换模型中所有 ModulatedDeformConv2dPack 为 nn.Conv2d。

    返回:
        (converted_count, total_params_removed)
    """
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack

    converted = 0
    removed_params = 0

    # 遍历所有 parent 模块, 替换其子模块中的 DCN
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child in list(parent_module.named_children()):
            if isinstance(child, ModulatedDeformConv2dPack):
                # 创建等效 Conv2d
                has_bias = child.bias is not None
                new_conv = nn.Conv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=has_bias,
                )

                # 迁移核心权重
                new_conv.weight.data = child.weight.data.clone()
                if has_bias:
                    new_conv.bias.data = child.bias.data.clone()

                # 统计丢弃的参数 (offset + mask 分支)
                offset_params = sum(p.numel() for p in child.conv_offset.parameters())
                removed_params += offset_params

                # 替换
                setattr(parent_module, child_name, new_conv)
                converted += 1

                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                logger.debug("  替换: %s (offset+mask 参数 %d 已丢弃)", full_name, offset_params)

    return converted, removed_params


def convert_state_dict_only(state_dict):
    """仅对 state_dict 进行转换: 删除 conv_offset.* 相关的 key。

    适用于保存转换后的 checkpoint 时, state_dict 中已经不需要 conv_offset 参数。
    """
    new_state_dict = {}
    dropped = 0
    for key, value in state_dict.items():
        if ".conv_offset." in key:
            dropped += 1
            continue
        new_state_dict[key] = value
    return new_state_dict, dropped


def write_non_dcn_config(input_path, output_path):
    """复制 config 并移除 DCN 配置"""
    with open(input_path, "r") as f:
        content = f.read()

    # 把所有 dcn=dict(...) 改成 dcn=None (ego + other_agent 都要)
    # 保留尾部的逗号 (确保 dict 项之间分隔正确)
    import re
    pattern = r"dcn\s*=\s*dict\s*\([^)]*\)(\s*,\s*#[^\n]*)?,?"
    replaced = re.sub(pattern, "dcn=None,", content)  # 全部替换, 并强制带逗号

    # 所有 stage_with_dcn 改为全 False
    replaced = re.sub(
        r"stage_with_dcn\s*=\s*\([^)]+\)",
        "stage_with_dcn=(False, False, False, False)",
        replaced,
    )

    with open(output_path, "w") as f:
        f.write(replaced)
    logger.info("已保存 non-DCN config: %s", output_path)


def main():
    args = parse_args()

    # 1. 加载原始模型
    logger.info("加载原始模型 + checkpoint")
    import mmdet3d  # 触发 registry 注册
    from tools.pruning_sensitivity_analysis import load_model_fresh, get_prune_target

    model, cfg = load_model_fresh(args.config, args.input_ckpt)

    # 2. 统计转换前 (整个 MultiAgent, 包括 ego + other_agent)
    params_before = sum(p.numel() for p in model.parameters())
    dcn_count_before = sum(
        1 for m in model.modules()
        if type(m).__name__ == "ModulatedDeformConv2dPack"
    )
    logger.info("转换前: %d 个 DCN (含所有 agent), 总参数 %.2fM",
                dcn_count_before, params_before / 1e6)

    # 3. 执行转换 (作用于整个 model, 含 ego + other_agent)
    logger.info("开始替换 DCN → Conv2d (所有 agent) ...")
    converted, removed = convert_dcn_in_model(model)
    logger.info("已替换 %d 个 DCN", converted)
    logger.info("丢弃参数 (offset+mask): %.2fM", removed / 1e6)

    params_after = sum(p.numel() for p in model.parameters())
    logger.info("转换后: 总参数 %.2fM (减少 %.2fM, -%.1f%%)",
                params_after / 1e6,
                (params_before - params_after) / 1e6,
                (params_before - params_after) / params_before * 100)

    # 4. 验证模型可 forward (可选)
    if args.verify:
        logger.info("验证 backbone 前向传播...")
        ego = get_prune_target(model)
        ego.eval()
        dummy = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            try:
                out = ego.img_backbone(dummy)
                logger.info("  ego backbone forward OK, 输出 shape: %s",
                            [o.shape for o in out] if isinstance(out, (list, tuple)) else out.shape)
            except Exception as e:
                logger.error("  forward 失败: %s", e)
                raise

    # 5. 保存转换后 checkpoint
    logger.info("保存转换后 checkpoint...")
    state_dict = model.state_dict()
    clean_sd, dropped = convert_state_dict_only(state_dict)
    logger.info("state_dict 清理: 丢弃 %d 个 conv_offset.* 键", dropped)

    save_dict = {
        "state_dict": clean_sd,
        "meta": {
            "source_ckpt": args.input_ckpt,
            "dcn_converted": converted,
            "params_before": params_before,
            "params_after": params_after,
            "conversion_method": "zero_retrain",
        },
    }

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save(save_dict, args.output_ckpt)
    size_mb = os.path.getsize(args.output_ckpt) / 1e6
    logger.info("已保存: %s (%.1f MB)", args.output_ckpt, size_mb)

    # 6. 保存对应 config (可选)
    if args.output_config:
        write_non_dcn_config(args.config, args.output_config)

    logger.info("=" * 50)
    logger.info("DCN 转换完成!")
    logger.info("  源 checkpoint: %s", args.input_ckpt)
    logger.info("  输出 checkpoint: %s", args.output_ckpt)
    if args.output_config:
        logger.info("  输出 config: %s", args.output_config)
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
