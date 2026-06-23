"""阶段 4.3 — UniV2X-tiny config + 借权重 sanity 验证

验证:
  1. tiny config 能被 mmcv 正确加载
  2. UniV2X 模型能用 tiny config build 出来
  3. univ2x_tiny_init.pth 权重能正确加载到模型
  4. 模型参数总量合理(预期 ~150M, 比 base ~200M 小)
  5. backbone + neck 的权重确实来自借用(随机抽几个 tensor 比对)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def main() -> None:
    print("=" * 70)
    print("阶段 4.3 — UniV2X-tiny sanity 验证")
    print("=" * 70)

    # 1. config 加载
    from mmcv import Config
    cfg = Config.fromfile(str(ROOT / "projects/configs_e2e_univ2x/univ2x_tiny_e2e_track.py"))
    print("\n[1] config 加载: OK")
    print(f"    veh backbone depth: {cfg.model_ego_agent.img_backbone.depth}")
    print(f"    inf backbone depth: {cfg.model_other_agent_inf.img_backbone.depth}")
    print(f"    bev_h/w: {cfg.bev_h_}/{cfg.bev_w_}")
    print(f"    encoder num_layers: {cfg.model_ego_agent.pts_bbox_head.transformer.encoder.num_layers}")
    print(f"    load_from: {cfg.load_from}")

    # 2. 验证 init ckpt 文件
    init_path = Path(cfg.load_from)
    if not init_path.exists():
        print(f"\n[2] init ckpt MISSING: {init_path}")
        return
    init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
    init_sd = init_ck["state_dict"]
    print(f"\n[2] init ckpt 加载: OK ({len(init_sd)} keys, {init_path.stat().st_size/1e6:.1f}MB)")
    # 抽样几个 key
    for k in ["model_ego_agent.img_backbone.conv1.weight",
              "model_ego_agent.img_neck.lateral_convs.0.conv.weight",
              "model_other_agent_inf.img_backbone.conv1.weight"]:
        if k in init_sd:
            print(f"    ✓ {k}: shape={tuple(init_sd[k].shape)}")
        else:
            print(f"    ✗ {k}: MISSING")

    # 3. 尝试用 mmcv build 模型(可能因 plugin / Custom UniV2X 类不可访问而失败,先看看)
    print("\n[3] 尝试 build UniV2X 模型(用 mmdet build_detector)...")
    try:
        # 加载 plugin
        if hasattr(cfg, "plugin") and cfg.plugin:
            import importlib
            plugin_dir = cfg.plugin_dir if hasattr(cfg, "plugin_dir") else "projects/mmdet3d_plugin/"
            plugin_dir = plugin_dir.replace("/", ".").rstrip(".")
            try:
                importlib.import_module(plugin_dir)
                print(f"    plugin '{plugin_dir}' loaded")
            except Exception as e:
                print(f"    plugin import WARN: {e}")

        # 用 mmdet3d 的 build_model
        try:
            from mmdet3d.models import build_model
            model = build_model(cfg.model_ego_agent)  # 单 agent 先试
            print(f"    ego_agent build: OK")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    n_params: {n_params:,}  ({n_params/1e6:.1f}M)")
        except KeyError as e:
            # UniV2X 可能不在 detectors registry, 用 build_detector
            from mmdet.models import build_detector
            model = build_detector(cfg.model_ego_agent)
            print(f"    ego_agent build via build_detector: OK")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    n_params: {n_params:,}  ({n_params/1e6:.1f}M)")
    except Exception as e:
        print(f"    [WARN] build 失败: {type(e).__name__}: {str(e)[:200]}")
        print(f"    (这可能是 plugin 注册问题,不影响 config 本身合理性)")
        return

    # 4. 加载 init 权重到模型
    print("\n[4] 加载 init 权重到模型...")
    # init ckpt 用 model_ego_agent 前缀, 我们 build 的就是 ego_agent, 所以要去掉前缀
    ego_only_sd = {
        k.replace("model_ego_agent.", ""): v
        for k, v in init_sd.items()
        if k.startswith("model_ego_agent.")
    }
    miss, unexp = model.load_state_dict(ego_only_sd, strict=False)
    print(f"    missing keys: {len(miss)} (这是预期的 — 我们只借了 backbone+neck)")
    print(f"    unexpected keys: {len(unexp)} (应该 0)")
    if len(miss) > 0:
        # 检查 missing 的都是非 backbone+neck 模块
        non_borrow = [k for k in miss if not (k.startswith("img_backbone") or k.startswith("img_neck"))]
        backbone_neck_miss = [k for k in miss if k.startswith("img_backbone") or k.startswith("img_neck")]
        print(f"    missing 中:")
        print(f"      backbone+neck (不该 missing): {len(backbone_neck_miss)}")
        print(f"      其他模块 (预期 missing): {len(non_borrow)}")
        if backbone_neck_miss:
            print(f"      backbone+neck missing 样本: {backbone_neck_miss[:3]}")

    # 5. 验证 backbone 权重确实来自借用 (vs 随机)
    print("\n[5] 验证 backbone 权重是借来的而非随机:")
    bb_conv1 = model.img_backbone.conv1.weight
    init_conv1 = init_sd["model_ego_agent.img_backbone.conv1.weight"]
    is_match = torch.allclose(bb_conv1, init_conv1, atol=1e-6)
    print(f"    img_backbone.conv1.weight 是否等于 init ckpt: {is_match}")
    if is_match:
        print(f"    ✓ 借权重成功,backbone 不是随机初始化的")
    else:
        print(f"    ✗ 借权重失败! backbone 跟 init 不一致")

    # 6. 模型规模比较
    print("\n[6] 参数规模对比")
    # UniV2X-base (R101+DCN): 100.9M (来自 UniAD-tiny 我们看过) — 但 UniV2X-base 应该更多
    base_pth = "/home/jichengzhi/UniV2X/univ2x_coop_e2e_stg2_old_mode_inference_only.pth"
    if Path(base_pth).exists():
        base_ck = torch.load(base_pth, map_location="cpu", weights_only=False)
        base_sd = base_ck.get("state_dict", base_ck)
        n_base = sum(v.numel() for v in base_sd.values() if hasattr(v, "numel"))
        print(f"    UniV2X-base (R101+DCN): {n_base:,} ({n_base/1e6:.1f}M)")
        print(f"    UniV2X-tiny (R50, ours): {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"    缩减比例: {(1 - n_params/n_base)*100:+.1f}%")

    print("\n" + "=" * 70)
    print("[sanity] 阶段 4.3 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
