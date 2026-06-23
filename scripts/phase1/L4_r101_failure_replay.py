"""阶段 1 L4 — R101 失败实验在 R50 上重跑对照

对应李星峰 2026-04-25 报告里 4 个失败现象,对照看 R50 + 无 DCN 下哪些"复活":

  现象 ①: fastnas 第 3/4 stage 敏感度几乎全 0
           → R101+DCN 下因关 DCN 后第 3/4 stage 权重随机初始化,敏感度退化
           → R50 用 ImageNet 预训权重,敏感度分布应正常
           对照点: 测各层敏感度分布,看是否有"全 0"层

  现象 ②: 95% 和 90% 保留率搜出相同子网
           → 因敏感度全 0 退化为最小通道
           → R50 下不同剪枝率应能区分
           对照点: 跑 0.05 / 0.10 / 0.25 三档剪枝,看子网是否真的不同

  现象 ③: 跨层梯度约束 vs 实际拐点
           → 经验"channels mod 32 == 0"是否真有 latency 拐点?
           → "K_dim >= 64" 在 R50 上成立吗?
           对照点: 测 channel = 28/32/36/64,K_dim = 32/48/63/64/72 的 latency 跳变

  现象 ④: e2e latency 噪声淹没收益(P95 比 mean 大很多)
           → track_base_update 等动态控制流导致
           → R50 标准 conv 上,P95-P50 的差距应该小很多
           对照点: 跑 100 次同一前向,看 P50/P95/P99 spread

输出:
  results/phase1_L4_r101_failure_replay.json (全 4 现象统一报告)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.measure_latency import LatencyBenchmark


# ============================================================
# 现象 ① — 各层敏感度分布
# ============================================================

def phenomenon_1_sensitivity_distribution(
    pretrain_path: str,
    device: str = "cuda",
    input_size: tuple = (1, 3, 800, 450),
) -> dict:
    """计算 R50 各层的剪枝敏感度(用 L1 范数变化作 proxy).

    R101+DCN 下,关 DCN 后第 3/4 stage 权重是随机初始化,各 conv 的 L1 几乎全 0.
    R50 ImageNet 预训权重应该有合理的 L1 分布(尾部不会全 0).
    """
    print("\n" + "─" * 60)
    print("现象 ①: 各层剪枝敏感度分布")
    print("─" * 60)

    model = tvm.resnet50(weights=None)
    sd = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    layer_l1: list[dict] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        w = mod.weight.detach()
        # 按输出通道算 L1 (这是剪枝重要性的常见 proxy)
        per_channel_l1 = w.abs().sum(dim=(1, 2, 3))
        l1_min = float(per_channel_l1.min())
        l1_max = float(per_channel_l1.max())
        l1_mean = float(per_channel_l1.mean())
        l1_std = float(per_channel_l1.std())
        # 归类:全 0?接近 0?正常?
        if l1_max < 1e-8:
            cat = "all_zero"        # ⚠️ 退化层
        elif l1_min / l1_max < 0.001:
            cat = "highly_skewed"   # ⚠️ 部分通道几乎为 0
        else:
            cat = "normal"
        layer_l1.append({
            "name": name,
            "out_channels": w.shape[0],
            "l1_min": l1_min,
            "l1_max": l1_max,
            "l1_mean": l1_mean,
            "l1_std": l1_std,
            "category": cat,
        })

    cnt_normal = sum(1 for x in layer_l1 if x["category"] == "normal")
    cnt_skewed = sum(1 for x in layer_l1 if x["category"] == "highly_skewed")
    cnt_zero = sum(1 for x in layer_l1 if x["category"] == "all_zero")

    print(f"  total convs: {len(layer_l1)}")
    print(f"  normal:        {cnt_normal} ({cnt_normal/len(layer_l1)*100:.1f}%)")
    print(f"  highly_skewed: {cnt_skewed} ({cnt_skewed/len(layer_l1)*100:.1f}%)")
    print(f"  all_zero:      {cnt_zero} ({cnt_zero/len(layer_l1)*100:.1f}%)  ← R101 下大量出现")

    revived = cnt_zero == 0  # 没有全 0 层 = 现象 ① 在 R50 上"复活"
    print(f"  → R101 失败现象 ①: {'复活 ✓' if revived else '仍然死 ✗'}")
    return {
        "phenomenon": "1_sensitivity_distribution",
        "n_convs": len(layer_l1),
        "n_normal": cnt_normal,
        "n_highly_skewed": cnt_skewed,
        "n_all_zero": cnt_zero,
        "revived": revived,
        "layers_top5_skewed": sorted(layer_l1, key=lambda x: x["l1_min"] / max(x["l1_max"], 1e-12))[:5],
    }


# ============================================================
# 现象 ② — 不同剪枝率搜出的子网是否真的不同
# ============================================================

def phenomenon_2_subnet_uniqueness(
    pretrain_path: str,
    ratios: list[float] = [0.05, 0.10, 0.25],
    device: str = "cuda",
    input_size: tuple = (1, 3, 800, 450),
) -> dict:
    """跑不同剪枝率,看每层留下的通道数是否真的不同."""
    print("\n" + "─" * 60)
    print("现象 ②: 不同剪枝率搜出的子网是否真的不同")
    print("─" * 60)

    import torch_pruning as tp
    dummy = torch.randn(*input_size).to(device)

    sub_signatures: list[tuple] = []  # (ratio, [layer1.out_ch, layer2.out_ch, ...])
    for r in ratios:
        m = tvm.resnet50(weights=None)
        m.load_state_dict(torch.load(pretrain_path, map_location="cpu", weights_only=False))
        m = m.to(device).eval()
        ignored = [x for x in m.modules() if isinstance(x, nn.Linear)]
        pruner = tp.pruner.MetaPruner(
            m, dummy,
            importance=tp.importance.MagnitudeImportance(p=1),
            pruning_ratio=r,
            ignored_layers=ignored,
            global_pruning=True,
        )
        pruner.step()
        # 收集每个 conv 的 out_channels
        sig = tuple(
            x.weight.shape[0]
            for x in m.modules()
            if isinstance(x, nn.Conv2d)
        )
        sub_signatures.append((r, sig))
        print(f"  ratio {r:.2f}: signature = {sig[:5]}...{sig[-3:]} (len={len(sig)})")

    # 对比唯一性
    unique_count = len(set(s for _, s in sub_signatures))
    print(f"  → unique subnets: {unique_count}/{len(ratios)}")
    revived = unique_count == len(ratios)
    print(f"  → R101 失败现象 ②: {'复活 ✓' if revived else '仍然死 ✗(子网相同 / 退化)'}")

    return {
        "phenomenon": "2_subnet_uniqueness",
        "ratios": ratios,
        "n_unique_subnets": unique_count,
        "revived": revived,
        "signatures": [list(s) for _, s in sub_signatures],
    }


# ============================================================
# 现象 ③ — 通道对齐拐点(channels mod 32)
# ============================================================

def phenomenon_3_channel_alignment_cliff(
    device: str = "cuda",
    n_warmup: int = 10,
    n_runs: int = 30,
) -> dict:
    """测一组合成 conv 在 channel ∈ {28, 32, 36, 64, 96, 128} 上的 latency 跳变."""
    print("\n" + "─" * 60)
    print("现象 ③: channel 对齐 32 拐点 + K_dim ≥ 64 拐点")
    print("─" * 60)

    bench = LatencyBenchmark(n_warmup=n_warmup, n_runs=n_runs, device=device, verbose=False)

    # channel 拐点测试
    test_channels = [28, 32, 36, 64, 88, 96, 128]
    channel_results = []
    in_ch = 64
    for c in test_channels:
        layer = nn.Conv2d(in_ch, c, kernel_size=3, padding=1).to(device)
        dummy = torch.randn(1, in_ch, 200, 200).to(device)
        m = bench.run(layer, dummy)
        channel_results.append({"out_channels": c, "p50_ms": m.p50_ms, "p95_ms": m.p95_ms})
        print(f"  conv64→{c:3d}: P50={m.p50_ms:.3f}ms P95={m.p95_ms:.3f}ms")

    # 检查拐点是否存在(28 → 32 跳变)
    p_28 = next(x["p50_ms"] for x in channel_results if x["out_channels"] == 28)
    p_32 = next(x["p50_ms"] for x in channel_results if x["out_channels"] == 32)
    p_36 = next(x["p50_ms"] for x in channel_results if x["out_channels"] == 36)
    cliff_28_32 = (p_28 - p_32) / max(p_28, 1e-6) * 100  # 28 比 32 慢多少%
    has_cliff = abs(cliff_28_32) > 5
    print(f"  cliff 28→32: P50 改善 {cliff_28_32:+.1f}% {'(有拐点)' if has_cliff else '(无明显拐点)'}")

    # K_dim 拐点测试 (matmul / Linear)
    test_kdims = [32, 48, 63, 64, 72, 128]
    kdim_results = []
    for k in test_kdims:
        layer = nn.Linear(k, k).to(device)
        dummy = torch.randn(1, 128, k).to(device)
        m = bench.run(layer, dummy)
        kdim_results.append({"k_dim": k, "p50_ms": m.p50_ms, "p95_ms": m.p95_ms})
        print(f"  linear k={k:3d}: P50={m.p50_ms:.3f}ms P95={m.p95_ms:.3f}ms")

    p_63 = next(x["p50_ms"] for x in kdim_results if x["k_dim"] == 63)
    p_64 = next(x["p50_ms"] for x in kdim_results if x["k_dim"] == 64)
    cliff_63_64 = (p_63 - p_64) / max(p_63, 1e-6) * 100
    has_kdim_cliff = abs(cliff_63_64) > 5
    print(f"  cliff K_dim 63→64: P50 改善 {cliff_63_64:+.1f}% {'(有拐点)' if has_kdim_cliff else '(无明显拐点)'}")

    revived = has_cliff or has_kdim_cliff
    print(f"  → R101 失败现象 ③: {'复活 ✓ (经验拐点真实存在)' if revived else '仍然不清 (拐点不显著,可能 4090 GPU 太富裕看不出来)'}")

    return {
        "phenomenon": "3_alignment_cliffs",
        "channel_alignment": channel_results,
        "k_dim_alignment": kdim_results,
        "cliff_28_32_pct": cliff_28_32,
        "cliff_63_64_pct": cliff_63_64,
        "revived": revived,
    }


# ============================================================
# 现象 ④ — latency 测量稳定性 (P95 - P50 的 spread)
# ============================================================

def phenomenon_4_latency_stability(
    pretrain_path: str,
    device: str = "cuda",
    input_size: tuple = (1, 3, 800, 450),
    n_runs: int = 100,
) -> dict:
    """跑 100 次同一前向,看 P95 - P50 的 spread."""
    print("\n" + "─" * 60)
    print("现象 ④: latency 测量稳定性 (跑 100 次,看 spread)")
    print("─" * 60)

    bench = LatencyBenchmark(n_warmup=20, n_runs=n_runs, device=device, verbose=False)
    model = tvm.resnet50(weights=None)
    sd = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    dummy = torch.randn(*input_size).to(device)
    m = bench.run(model, dummy)
    p95_p50_spread = (m.p95_ms - m.p50_ms) / m.p50_ms * 100  # spread 占 P50 多少%
    p99_p50_spread = (m.p99_ms - m.p50_ms) / m.p50_ms * 100

    print(f"  100 次推理统计:")
    print(f"    P50  = {m.p50_ms:.3f}ms")
    print(f"    P95  = {m.p95_ms:.3f}ms (P95-P50 spread = {p95_p50_spread:+.1f}%)")
    print(f"    P99  = {m.p99_ms:.3f}ms (P99-P50 spread = {p99_p50_spread:+.1f}%)")
    print(f"    std  = {m.std_ms:.3f}ms ({m.std_ms/m.p50_ms*100:.1f}% of P50)")

    # 李星峰报告 P95 比 mean 多 6ms,即约 ~5% spread
    # R50 上如果 P95-P50 < 10% 算稳定(测量可信)
    stable = abs(p95_p50_spread) < 10
    revived = stable
    print(f"  → R101 失败现象 ④: {'复活 ✓ (测量稳定)' if stable else '仍然死 ✗ (噪声大)'}")

    return {
        "phenomenon": "4_latency_stability",
        "n_runs": n_runs,
        "p50_ms": m.p50_ms,
        "p95_ms": m.p95_ms,
        "p99_ms": m.p99_ms,
        "std_ms": m.std_ms,
        "p95_p50_spread_pct": p95_p50_spread,
        "p99_p50_spread_pct": p99_p50_spread,
        "stable": stable,
        "revived": revived,
    }


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default="/home/jichengzhi/Bench2DriveZoo_trb/ckpts/resnet50-19c8e357.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 800, 450])
    parser.add_argument("--phenomena", default="1234", help="选哪几个现象,如 '12' / '1234'")
    parser.add_argument("--output", default="results/phase1_L4_r101_failure_replay.json")
    args = parser.parse_args()

    print("=" * 70)
    print("阶段 1 L4 — R101 失败实验在 R50 上重跑对照")
    print("=" * 70)
    print(f"  pretrain: {args.pretrain}")
    print(f"  phenomena: {args.phenomena}")

    results: dict = {}
    if "1" in args.phenomena:
        results["p1"] = phenomenon_1_sensitivity_distribution(
            args.pretrain, args.device, tuple(args.input_size)
        )
    if "2" in args.phenomena:
        results["p2"] = phenomenon_2_subnet_uniqueness(
            args.pretrain, [0.05, 0.10, 0.25], args.device, tuple(args.input_size)
        )
    if "3" in args.phenomena:
        results["p3"] = phenomenon_3_channel_alignment_cliff(args.device)
    if "4" in args.phenomena:
        results["p4"] = phenomenon_4_latency_stability(
            args.pretrain, args.device, tuple(args.input_size)
        )

    # 汇总
    print("\n" + "=" * 70)
    print("L4 汇总 — R101 4 个失败现象在 R50 上的复活情况")
    print("=" * 70)
    revived_count = 0
    for k, v in results.items():
        revived = v.get("revived", False)
        revived_count += int(revived)
        print(f"  {k}: {'复活 ✓' if revived else '仍然死 ✗'} — {v['phenomenon']}")

    print(f"\n[L4 总分]: {revived_count}/{len(results)} 现象复活")
    if revived_count >= 2:
        print(f"[L4 PASS] ≥ 2 现象复活,论文图 3 数据成立")
    else:
        print(f"[L4 PARTIAL] 仅 {revived_count} 现象复活,需进一步分析")

    # 落盘
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[output] {out}")


if __name__ == "__main__":
    main()
