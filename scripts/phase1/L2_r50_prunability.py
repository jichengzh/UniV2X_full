"""阶段 1 L2 — R50 全网络剪枝可行性实验

科学问题:
  R101+DCN 下因 DCN 不可剪 + neck 输入通道约束,实际只有 conv1 可剪,可剪参数 12.5%
  R50 + 无 DCN 是不是真的能让所有 conv 层都进入剪枝搜索空间?

实验设计:
  1. 加载标准 R50 (ImageNet pretrain)
  2. 用 Torch-Pruning DepGraph 在剪枝率 {10%, 25%, 50%, 70%} 上扫描
  3. 每档记录:
     - 可剪 conv 层数 / 总 conv 层数
     - 实际可剪参数 / 总参数 (这是核心指标)
     - 剪枝后模型仍能正常前向 (sanity)
     - 剪枝后 latency 加速比(配合 measure_latency)
  4. 对照 R101+DCN 的 12.5% 上限

验收标准 (D1.8):
  R50 全网可剪参数比例 > 50% → 论文级证据(对比 R101 的 12.5% 提升 4 倍)

输出:
  results/phase1_L2_r50_prunability.csv
  results/phase1_L2_r50_prunability.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.measure_latency import LatencyBenchmark


@dataclass
class PruneResult:
    target_ratio: float
    n_total_convs: int
    n_prunable_convs: int
    pct_prunable_convs: float
    params_before: int
    params_after: int
    params_pruned: int
    pct_params_pruned: float
    flops_before_g: float
    flops_after_g: float
    pct_flops_pruned: float
    forward_ok: bool
    latency_p50_before_ms: float = 0.0
    latency_p50_after_ms: float = 0.0
    speedup: float = 1.0


# ============================================================
# R50 模型加载
# ============================================================

def load_resnet50(pretrain_path: str | None = None) -> nn.Module:
    """加载 R50 (可选: 加载下载好的 pretrain)."""
    model = tvm.resnet50(weights=None)
    if pretrain_path and Path(pretrain_path).exists():
        sd = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        # torchvision 标准格式直接加载
        try:
            model.load_state_dict(sd)
            print(f"  [pretrain loaded] {pretrain_path}")
        except Exception as e:
            print(f"  [pretrain WARN] {e}")
    return model


# ============================================================
# 剪枝核心(用 Torch-Pruning DepGraph)
# ============================================================

def prune_resnet50(
    model: nn.Module,
    dummy_input: torch.Tensor,
    target_ratio: float,
    importance: str = "L1",
) -> tuple[nn.Module, PruneResult]:
    """对 R50 做全网剪枝.

    使用 Torch-Pruning 的 MetaPruner + DepGraph,自动追踪通道依赖.
    R50 没有 DCN/MSDeformAttn 等不可追踪算子,理论上能全网剪.

    Args:
        target_ratio: 0.1 / 0.25 / 0.5 / 0.7
        importance: 'L1' / 'Random' / 'Taylor'

    Returns:
        (pruned_model, PruneResult)
    """
    import torch_pruning as tp

    # 重要性度量
    if importance == "L1":
        imp = tp.importance.MagnitudeImportance(p=1)
    elif importance == "Random":
        imp = tp.importance.RandomImportance()
    elif importance == "Taylor":
        imp = tp.importance.TaylorImportance()
    else:
        raise ValueError(f"unknown importance: {importance}")

    # 剪枝前统计
    params_before = sum(p.numel() for p in model.parameters())
    flops_before, _ = tp.utils.count_ops_and_params(model, dummy_input)

    # 计 conv 层
    convs_total = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

    # 配置 MetaPruner — 全网剪枝,排除最后的 fc(分类头)
    ignored_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    pruner = tp.pruner.MetaPruner(
        model,
        dummy_input,
        importance=imp,
        pruning_ratio=target_ratio,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )
    # 看看 DepGraph 实际识别的"可剪 conv"集合
    prunable_convs: list[nn.Module] = []
    for group in pruner.DG.get_all_groups():
        for dep, idxs in group:
            mod = dep.target.module
            if isinstance(mod, nn.Conv2d) and mod not in prunable_convs:
                prunable_convs.append(mod)

    # 执行剪枝
    pruner.step()

    # 剪枝后统计
    params_after = sum(p.numel() for p in model.parameters())
    flops_after, _ = tp.utils.count_ops_and_params(model, dummy_input)

    # 验证前向通畅
    forward_ok = True
    try:
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
    except Exception as e:
        print(f"  [forward FAIL] {e}")
        forward_ok = False

    res = PruneResult(
        target_ratio=target_ratio,
        n_total_convs=len(convs_total),
        n_prunable_convs=len(prunable_convs),
        pct_prunable_convs=len(prunable_convs) / max(len(convs_total), 1) * 100,
        params_before=params_before,
        params_after=params_after,
        params_pruned=params_before - params_after,
        pct_params_pruned=(params_before - params_after) / params_before * 100,
        flops_before_g=flops_before / 1e9,
        flops_after_g=flops_after / 1e9,
        pct_flops_pruned=(flops_before - flops_after) / flops_before * 100,
        forward_ok=forward_ok,
    )
    return model, res


# ============================================================
# 主流程
# ============================================================

def run_l2_experiment(
    pretrain_path: str | None,
    target_ratios: list[float],
    importance: str = "L1",
    measure_latency: bool = True,
    input_size: tuple = (1, 3, 800, 450),
    device: str = "cuda",
) -> list[PruneResult]:
    """L2 实验主流程."""

    print("=" * 70)
    print("阶段 1 L2 — R50 全网络剪枝可行性")
    print("=" * 70)
    print(f"  pretrain:    {pretrain_path}")
    print(f"  target_ratios: {target_ratios}")
    print(f"  importance:  {importance}")
    print(f"  input_size:  {input_size}")
    print(f"  device:      {device}")
    print()

    dummy = torch.randn(*input_size).to(device)

    # baseline latency (未剪枝)
    base_latency_p50 = 0.0
    if measure_latency:
        print("[baseline] 测 R50 未剪枝 latency...")
        base_model = load_resnet50(pretrain_path).to(device).eval()
        bench = LatencyBenchmark(n_warmup=10, n_runs=20, device=device, verbose=False)
        m = bench.run(base_model, dummy)
        base_latency_p50 = m.p50_ms
        print(f"  baseline P50 = {base_latency_p50:.2f}ms")
        del base_model
        torch.cuda.empty_cache()

    # 扫描剪枝率
    results: list[PruneResult] = []
    for r in target_ratios:
        print(f"\n[prune] target_ratio = {r:.2f}")
        model = load_resnet50(pretrain_path).to(device).eval()
        try:
            pruned_model, res = prune_resnet50(model, dummy, target_ratio=r, importance=importance)
        except Exception as e:
            print(f"  [FAIL] {type(e).__name__}: {e}")
            res = PruneResult(
                target_ratio=r, n_total_convs=0, n_prunable_convs=0, pct_prunable_convs=0,
                params_before=0, params_after=0, params_pruned=0, pct_params_pruned=0,
                flops_before_g=0, flops_after_g=0, pct_flops_pruned=0, forward_ok=False,
            )
            results.append(res)
            continue

        # 剪后 latency
        if measure_latency and res.forward_ok:
            bench = LatencyBenchmark(n_warmup=10, n_runs=20, device=device, verbose=False)
            m = bench.run(pruned_model, dummy)
            res.latency_p50_after_ms = m.p50_ms
            res.latency_p50_before_ms = base_latency_p50
            res.speedup = base_latency_p50 / m.p50_ms if m.p50_ms > 0 else 1.0

        results.append(res)
        print(
            f"  → 可剪 conv: {res.n_prunable_convs}/{res.n_total_convs} "
            f"({res.pct_prunable_convs:.1f}%)"
        )
        print(
            f"  → 参数: {res.params_before/1e6:.1f}M → {res.params_after/1e6:.1f}M "
            f"(剪掉 {res.pct_params_pruned:.1f}%)"
        )
        print(
            f"  → FLOPs: {res.flops_before_g:.2f}G → {res.flops_after_g:.2f}G "
            f"(降 {res.pct_flops_pruned:.1f}%)"
        )
        if measure_latency and res.forward_ok:
            print(
                f"  → latency: {base_latency_p50:.2f}ms → {res.latency_p50_after_ms:.2f}ms "
                f"(加速 {res.speedup:.2f}×)"
            )

        del model, pruned_model
        torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", default="/home/jichengzhi/Bench2DriveZoo_trb/ckpts/resnet50-19c8e357.pth")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.7])
    parser.add_argument("--importance", default="L1", choices=["L1", "Random", "Taylor"])
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 800, 450])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-prefix", default="results/phase1_L2_r50_prunability")
    args = parser.parse_args()

    results = run_l2_experiment(
        pretrain_path=args.pretrain,
        target_ratios=args.ratios,
        importance=args.importance,
        measure_latency=not args.no_latency,
        input_size=tuple(args.input_size),
        device=args.device,
    )

    # 落盘
    out_csv = ROOT / f"{args.output_prefix}.csv"
    out_json = ROOT / f"{args.output_prefix}.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(out_csv, index=False)

    with open(out_json, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # 摘要
    print()
    print("=" * 70)
    print("L2 实验摘要 (对照 R101+DCN 的 12.5% 上限)")
    print("=" * 70)
    print(f"{'ratio':<10} {'pct_prunable':<15} {'pct_params_cut':<18} {'speedup':<10} {'verdict'}")
    for r in results:
        v = "PASS" if r.pct_params_pruned > 50 else ("partial" if r.pct_params_pruned > 12.5 else "FAIL")
        print(
            f"{r.target_ratio:<10.2f} {r.pct_prunable_convs:<15.1f} "
            f"{r.pct_params_pruned:<18.1f} {r.speedup:<10.2f} {v}"
        )

    # 验收
    max_pct_pruned = max((r.pct_params_pruned for r in results), default=0)
    print()
    if max_pct_pruned > 50:
        print(f"[L2 PASS] R50 最大可剪 {max_pct_pruned:.1f}% > 50% (D1.8 验收线)")
        print(f"          相比 R101+DCN 的 12.5% 上限,提升 {max_pct_pruned/12.5:.1f}×")
    elif max_pct_pruned > 25:
        print(f"[L2 PARTIAL] 最大 {max_pct_pruned:.1f}% (25%-50%),需扩大剪枝率扫描或调 importance")
    else:
        print(f"[L2 FAIL] 最大 {max_pct_pruned:.1f}% < 25%, R50 也救不了 → 弃用 tiny 路径")

    print(f"\n[output]")
    print(f"  {out_csv}")
    print(f"  {out_json}")


if __name__ == "__main__":
    main()
