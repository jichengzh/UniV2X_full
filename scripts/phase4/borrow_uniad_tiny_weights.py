"""阶段 4.2 — 从 UniAD-tiny 借 R50 backbone + FPN neck 权重作 UniV2X-tiny 初始化

依据(阶段 2 验证):
  - UniAD-tiny 的 R50 backbone 在 SPD 图像上 100% 产生合理特征
  - 借用可节省 ~50% 训练时间(vs 从 ImageNet 重训)

借用范围:
  - img_backbone.*  (R50 backbone, 无 DCN)         — 100% 借
  - img_neck.*      (FPN)                          — 100% 借
  - 其他模块 (pts_bbox_head, motion_head, ...)     — UniV2X 重新随机初始化训练

key 转换:
  uniad_tiny_b2d.pth                          UniV2X-tiny init ckpt
    img_backbone.conv1.weight        ──→     model_ego_agent.img_backbone.conv1.weight  (车端)
                                     ──→     model_other_agent_inf.img_backbone.conv1.weight (路侧)
    img_neck.lateral_convs.0.conv...  ──→     model_ego_agent.img_neck.... (同上)

    其他 UniV2X 独有模块 (cross_agent_query_interaction 等) 不在借用范围,留空(训练时随机初始化)

输出:
  ckpts/univ2x_tiny_init.pth — 初始化 ckpt,可作为训练起点 (load_from)
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def borrow_weights(uniad_path: str, output_path: str, also_borrow_pretrain_r50: bool = False) -> dict:
    """主流程: 加载 UniAD-tiny → 抽 backbone+neck → 复制到车端+路侧端 → 保存."""

    print(f"[load] UniAD-tiny ckpt: {uniad_path}")
    uniad_ck = torch.load(uniad_path, map_location="cpu", weights_only=False)
    uniad_sd = uniad_ck.get("state_dict", uniad_ck)
    print(f"  total keys: {len(uniad_sd)}")

    # 1. 抽 img_backbone.* + img_neck.*
    new_sd: OrderedDict = OrderedDict()
    n_backbone = 0
    n_neck = 0

    for key, val in uniad_sd.items():
        if key.startswith("img_backbone."):
            inner = key  # 'img_backbone.conv1.weight'
            # 复制到车端
            new_sd[f"model_ego_agent.{inner}"] = val.clone()
            # 复制到路侧端(对称结构,两边 backbone 用同一份初始化是合理的)
            new_sd[f"model_other_agent_inf.{inner}"] = val.clone()
            n_backbone += 1
        elif key.startswith("img_neck."):
            new_sd[f"model_ego_agent.{key}"] = val.clone()
            new_sd[f"model_other_agent_inf.{key}"] = val.clone()
            n_neck += 1

    print(f"[borrow]")
    print(f"  img_backbone keys borrowed: {n_backbone} (复制到 ego + inf, 共 {n_backbone*2} 条)")
    print(f"  img_neck     keys borrowed: {n_neck}    (复制到 ego + inf, 共 {n_neck*2} 条)")
    print(f"  total new keys in init ckpt: {len(new_sd)}")

    # 2. 验证 shape 合理性 (sanity)
    sample_keys = [
        "model_ego_agent.img_backbone.conv1.weight",
        "model_ego_agent.img_backbone.bn1.weight",
        "model_ego_agent.img_neck.lateral_convs.0.conv.weight",
    ]
    print(f"\n[sanity] 关键张量 shape 检查:")
    for k in sample_keys:
        if k in new_sd:
            print(f"  {k}: {tuple(new_sd[k].shape)}")
        else:
            print(f"  {k}: MISSING ⚠️")

    # 3. 保存
    out = {
        "state_dict": new_sd,
        "meta": {
            "from": str(uniad_path),
            "borrowed_modules": ["img_backbone", "img_neck"],
            "borrow_strategy": "copy_to_both_ego_and_inf",
            "missing_modules": "pts_bbox_head, motion_head, occ_head, seg_head, cross_agent_query_interaction, ... (UniV2X 训练时随机初始化)",
            "n_total_keys": len(new_sd),
        }
    }
    torch.save(out, output_path)
    print(f"\n[save] {output_path}")
    print(f"  size: {Path(output_path).stat().st_size / 1e6:.1f} MB")

    return out["meta"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniad-tiny", default="/home/jichengzhi/Bench2DriveZoo_trb/ckpts/uniad_tiny_b2d.pth")
    parser.add_argument("--output", default="/home/jichengzhi/UniV2X/ckpts/univ2x_tiny_init.pth")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("阶段 4.2 — 借 UniAD-tiny 权重作 UniV2X-tiny 初始化")
    print("=" * 70)

    meta = borrow_weights(args.uniad_tiny, args.output)

    print()
    print("=" * 70)
    print("使用方法")
    print("=" * 70)
    print(f"  在 UniV2X-tiny config 里设置:")
    print(f"    load_from = '{args.output}'")
    print(f"  或训练命令:")
    print(f"    python tools/train.py PROJECT_CONFIG --resume-from {args.output}")
    print()
    print("[note] 此 ckpt 只含 backbone+neck (覆盖率 ~30% 总参数),其他模块随机初始化")
    print("       期望 stage1 训练 3-5 epoch 收敛(vs 从 ImageNet 重训需要 6-12 epoch)")


if __name__ == "__main__":
    main()
