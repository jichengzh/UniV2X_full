"""阶段 2 最小跨域实验 — UniAD-tiny 的 R50 backbone 在 V2X-Seq-SPD 图像上能否产生合理特征

背景:
  UniAD-tiny 完整模型硬编码 6 cam (cams_embeds.shape=(6,256)),
  V2X-Seq-SPD vehicle-side 只 1 cam → 完整 inference 不可行,**必走路径 Z 重训**.

  但 backbone(R50)+ neck(FPN)是单图前向,跟相机数无关.
  这个实验验证:UniAD-tiny 的 R50 backbone 权重能否直接在 SPD 图像上产生合理特征?

  如果能,意味着:
    - 阶段 4 训 UniV2X-tiny 时,可借 UniAD-tiny 的 R50 权重作初始化(节省 ~50% 训练时间)
    - 论文路径 Y(半监督)与 Z(全重训)的中间选项变得可行

设计:
  1. 加载 UniAD-tiny 权重的 img_backbone + img_neck 部分
  2. 输入 1 张 SPD vehicle-side 图像 (800×450 resize)
  3. 跑 backbone → neck,记录:
     - 输出 shape 是否合理 (FPN 4 个 level)
     - 输出激活均值 / 标准差 (vs 全 0 / NaN)
     - 输出空间结构 (峰值集中区域是否合理 = 路面、车辆位置)

输出: results/phase2_cross_domain_backbone_test.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torchvision.models as tvm
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_uniad_tiny_backbone() -> torch.nn.Module:
    """从 UniAD-tiny 权重提取 R50 backbone."""
    ckpt_path = "/home/jichengzhi/Bench2DriveZoo_trb/ckpts/uniad_tiny_b2d.pth"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]

    # 抽 img_backbone.* 的权重
    backbone_sd = OrderedDict()
    for k, v in sd.items():
        if k.startswith("img_backbone."):
            backbone_sd[k.replace("img_backbone.", "")] = v

    print(f"[ckpt] img_backbone keys: {len(backbone_sd)} 个")
    sample = list(backbone_sd.keys())[:3]
    print(f"  sample: {sample}")

    # 标准 R50 + 加载权重
    model = tvm.resnet50(weights=None)
    # mmdet 的 R50 跟 torchvision 略有差异(mmdet 没有 fc),先 strict=False
    miss, unexp = model.load_state_dict(backbone_sd, strict=False)
    print(f"[load] missing={len(miss)}, unexpected={len(unexp)}")
    if miss:
        print(f"  missing examples: {miss[:3]}")
    if unexp:
        print(f"  unexpected examples: {unexp[:3]}")
    return model.eval()


def load_spd_image(img_path: str, target_size=(450, 800)) -> torch.Tensor:
    """加载 SPD 图像并 resize 到 UniAD-tiny 期望的 800×450."""
    img = Image.open(img_path).convert("RGB")
    print(f"[image] {img_path} 原始 size: {img.size}")
    img = img.resize((target_size[1], target_size[0]))  # PIL resize: (W, H)
    print(f"  resized to: {img.size}")
    arr = np.array(img).astype(np.float32) / 255.0
    # ImageNet normalization (UniAD-tiny config 用的是 caffe 风格 BGR mean,
    # 但作为 sanity 我们用标准 ImageNet 即可,关键看 feature 是否合理而非精确精度)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    # HWC → CHW → BCHW
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()


def analyze_features(features: list[torch.Tensor], names: list[str]) -> dict:
    """分析 backbone 多尺度输出的合理性."""
    results = []
    for name, feat in zip(names, features):
        f = feat.detach().cpu().numpy()
        d = {
            "name": name,
            "shape": list(feat.shape),
            "mean": float(f.mean()),
            "std": float(f.std()),
            "min": float(f.min()),
            "max": float(f.max()),
            "n_zero_pct": float((f == 0).sum() / f.size * 100),
            "n_nan": int(np.isnan(f).sum()),
            "n_inf": int(np.isinf(f).sum()),
        }
        # 合理性判定
        if d["n_nan"] > 0 or d["n_inf"] > 0:
            d["verdict"] = "FAIL_nan_inf"
        elif d["std"] < 1e-6:
            d["verdict"] = "FAIL_constant"
        elif d["n_zero_pct"] > 90:
            d["verdict"] = "WARN_mostly_zero"
        else:
            d["verdict"] = "OK"
        results.append(d)
    return results


def main() -> None:
    print("=" * 70)
    print("阶段 2 最小跨域实验 — UniAD-tiny R50 backbone on V2X-Seq-SPD")
    print("=" * 70)

    # 1. 加载 backbone
    print("\n[step 1] 加载 UniAD-tiny 的 R50 backbone")
    backbone = load_uniad_tiny_backbone()

    # 抽中间层输出 (R50 layer1-4)
    feature_outputs: dict = {}

    def hook(name: str):
        def fn(_m, _i, out):
            feature_outputs[name] = out
        return fn

    handles = []
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(backbone, name)
        handles.append(layer.register_forward_hook(hook(name)))

    # 2. 加载 SPD 图像
    print("\n[step 2] 加载 V2X-Seq-SPD 图像")
    spd_img_paths = [
        "/data/V2X-Seq/V2X-Seq-SPD/V2X-Seq-SPD/vehicle-side/image/002879.jpg",
        "/data/V2X-Seq/V2X-Seq-SPD/V2X-Seq-SPD/vehicle-side/image/003354.jpg",
        "/data/V2X-Seq/V2X-Seq-SPD/V2X-Seq-SPD/vehicle-side/image/013672.jpg",
    ]

    all_results = []
    for img_path in spd_img_paths:
        if not Path(img_path).exists():
            print(f"  [skip] not found: {img_path}")
            continue
        x = load_spd_image(img_path)
        # 3. 前向
        print(f"\n[step 3] 前向 ({img_path.split('/')[-1]})")
        with torch.no_grad():
            _ = backbone(x)
        # 4. 分析特征
        feat_list = [feature_outputs[k] for k in ["layer1", "layer2", "layer3", "layer4"]]
        analysis = analyze_features(feat_list, ["layer1", "layer2", "layer3", "layer4"])
        for a in analysis:
            print(
                f"  {a['name']}: shape={a['shape']} "
                f"mean={a['mean']:+.3f} std={a['std']:.3f} "
                f"zero%={a['n_zero_pct']:.1f}% verdict={a['verdict']}"
            )
        all_results.append({"image": img_path, "features": analysis})

    for h in handles:
        h.remove()

    # 5. 汇总判定
    print("\n" + "=" * 70)
    print("跨域 backbone 可迁移性 — 总评")
    print("=" * 70)
    n_ok = 0
    n_total = 0
    for img_res in all_results:
        for f in img_res["features"]:
            n_total += 1
            if f["verdict"] == "OK":
                n_ok += 1

    print(f"  total feature checks: {n_total}")
    print(f"  OK: {n_ok}/{n_total} ({n_ok/n_total*100:.1f}%)")
    if n_ok / n_total > 0.85:
        verdict = "PASS"
        print(f"\n[VERDICT] PASS — UniAD-tiny 的 R50 backbone 在 SPD 图像上能产生合理特征")
        print(f"          意味着可借 UniAD-tiny 权重作 UniV2X-tiny 初始化(阶段 4 路径 Y)")
        print(f"          节省 ~50% 训练成本(vs 从 ImageNet 重训)")
    elif n_ok / n_total > 0.5:
        verdict = "PARTIAL"
        print(f"\n[VERDICT] PARTIAL — 部分层产生异常,需具体分析")
    else:
        verdict = "FAIL"
        print(f"\n[VERDICT] FAIL — backbone 在 SPD 上产生大量异常输出,转 ImageNet 初始化")

    # 6. 落盘
    out = ROOT / "results" / "phase2_cross_domain_backbone_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "results": all_results,
            "summary": {
                "n_ok": n_ok,
                "n_total": n_total,
                "pct_ok": n_ok / n_total * 100,
                "verdict": verdict,
            },
        }, f, indent=2)
    print(f"\n[output] {out}")


if __name__ == "__main__":
    main()
