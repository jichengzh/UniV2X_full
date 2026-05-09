"""M4.5 post-process: HEAL inference 输出 → baseline_4090.parquet 的 amota 字段.

输入:
  /tmp/m4_5_heal_inference.log — HEAL inference 输出, 应含 'The Average Precision' 行

提取 AP30/AP50/AP70 → 写:
  results/m4_5_pyramid_baseline_eval.txt  — AP 报告 + 时间戳
  data/baseline_4090.parquet 中 source='m4_3_pyramid_baseline' 行的 amota 字段
    (用 AP50 作 amota anchor, 是 OPV2V Pyramid 评估的标准指标)

注意:
  - HEAL inference 的 "Average Precision @ 0.30/0.50/0.70" 对应 AP30/50/70
  - 我们用 AP50 作为 amota proxy (cross-model anchor)
  - baseline_4090.parquet 已有 8 行 pyramid_fusion (M4.3+M4.5_6 latency-only); 这里只填 baseline (FP32+FP16) 两行的 amota
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG = Path("/tmp/m4_5_heal_inference.log")
BASELINE = REPO_ROOT / "data/baseline_4090.parquet"
OUT_REPORT = REPO_ROOT / "results/m4_5_pyramid_baseline_eval.txt"


def parse_ap(log_path: Path) -> dict[str, float]:
    """从 HEAL inference log 提取 AP30/50/70.

    HEAL inference.py 输出格式:
      The Average Precision at IoU 0.3 is xxx
      The Average Precision at IoU 0.5 is xxx
      The Average Precision at IoU 0.7 is xxx
    """
    if not log_path.exists():
        raise FileNotFoundError(f"HEAL inference log 不存在: {log_path}")
    text = log_path.read_text()

    out = {}
    for iou_str, key in [("0.3", "AP30"), ("0.5", "AP50"), ("0.7", "AP70")]:
        m = re.search(
            rf"Average Precision\s*(?:@|at)\s*IoU\s*{re.escape(iou_str)}\s*[:\s]+(\d+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if m:
            out[key] = float(m.group(1))
        else:
            # fallback 模式: 找形如 "AP@0.30 = 0.834"
            m2 = re.search(rf"AP@?{re.escape(iou_str)}\s*[:=]\s*(\d+\.?\d*)", text, re.IGNORECASE)
            if m2:
                out[key] = float(m2.group(1))

    if not out:
        raise ValueError(f"未能从 log 提取 AP, log 末尾:\n{text[-2000:]}")
    return out


def write_report(ap: dict[str, float]) -> None:
    """写 results/m4_5_pyramid_baseline_eval.txt."""
    OUT_REPORT.parent.mkdir(exist_ok=True)
    with OUT_REPORT.open("w") as f:
        f.write(f"# M4.5 Pyramid Fusion baseline AP 评估\n\n")
        f.write(f"**日期**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: HEAL Pyramid_m1_base (LiDAR-only, 3.76M params)\n")
        f.write(f"**ckpt**: stage1/Pyramid_m1_base_2023_08_14_04_28_12/net_epoch_bestval_at23.pth\n")
        f.write(f"**数据集**: OPV2V test split (~21GB from HF gqk/opv2v)\n")
        f.write(f"**evaluation 框架**: HEAL `opencood/tools/inference.py --fusion_method intermediate`\n\n")
        f.write(f"## AP 结果\n\n")
        for k, v in sorted(ap.items()):
            f.write(f"- **{k}**: {v:.4f}\n")
        f.write(f"\n## 用途\n\n")
        f.write(f"- **AP50** ({ap.get('AP50', 'N/A')}) → baseline_4090.parquet 的 amota 字段 anchor\n")
        f.write(f"- 三模型 spectrum: pyramid_fusion = AP50, univ2x_full = nuScenes mAP, uniad_tiny_variant = nuScenes amota\n")
        f.write(f"- 用作 Phase 2.5 LGB 重训时的 cross-model anchor\n")
    print(f"✅ Wrote {OUT_REPORT}")


def update_baseline(ap50: float) -> None:
    """更新 baseline_4090.parquet 的 pyramid_baseline 行 amota 字段."""
    df = pd.read_parquet(BASELINE)
    # 找 pyramid baseline 行 (M4.3 输出 + M4.5_6 latency)
    mask = df["source"].isin(["m4_3_pyramid_baseline", "m4_5_pyramid_configs"]) & \
           df["config_id"].astype(str).str.contains("baseline", na=False)
    n_match = mask.sum()
    print(f"找到 {n_match} 行匹配 pyramid baseline (source=m4_3_pyramid_baseline or m4_5 configs baseline)")

    if n_match == 0:
        # fallback: 任何 model_class=pyramid_fusion 的行
        mask = df["model_class"] == "pyramid_fusion"
        n_match = mask.sum()
        print(f"  fallback 用 model_class=pyramid_fusion: {n_match} 行")

    df.loc[mask, "amota"] = ap50
    df.to_parquet(BASELINE, index=False)
    df.to_csv(BASELINE.with_suffix(".csv"), index=False)
    print(f"✅ 更新 {BASELINE}: pyramid 行 amota={ap50:.4f}")


def main() -> None:
    print("=" * 60)
    print("M4.5 post-process — Pyramid baseline AP → baseline_4090.parquet")
    print("=" * 60)

    ap = parse_ap(LOG)
    print(f"\n=== 解析 AP ===")
    for k, v in sorted(ap.items()):
        print(f"  {k}: {v:.4f}")

    write_report(ap)
    if "AP50" in ap:
        update_baseline(ap["AP50"])
    else:
        print(f"⚠️  AP50 未提取到, 跳过 baseline_4090 更新")
        sys.exit(1)

    print(f"\n✅ M4.5 post-process 完成")
    print(f"   报告: {OUT_REPORT}")
    print(f"   baseline 已更新")


if __name__ == "__main__":
    main()
