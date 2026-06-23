"""全网剪枝 vs backbone-only 的 Pareto 对比图 (收敛后真测)。
读 results/P1_wholenet_prune_real.csv -> multi_agent/figure/data/fig5_wholenet_prune.png"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jichengzhi/V2X")
OUT = ROOT / "multi_agent/figure/data"
d = pd.read_csv(ROOT / "results/P1_wholenet_prune_real.csv")
# dense DAIR base AP50 ≈ 0.79 (stage_a base, 非 CSV 里 OPV2V 的 0.96)
d.loc[d.variant == "dense_baseline", "ap50"] = 0.79

COLOR = {"dense_baseline": "#7f7f7f", "backbone_only_p50": "#d62728",
         "wholenet_light": "#1f77b4", "wholenet_p50": "#1f77b4", "wholenet_aggr": "#2ca02c"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (xcol, xlabel) in zip(axes, [("lat_fp32_p50_ms", "latency FP32 p50 (ms, raw-PyTorch)"),
                                      ("subnet_params", "subnet params")]):
    for _, r in d.iterrows():
        x = r[xcol] / 1e6 if xcol == "subnet_params" else r[xcol]
        ax.scatter(x, r.ap50, s=90, c=COLOR.get(r.variant, "k"), edgecolors="k", zorder=3)
        lbl = r.variant.replace("_p50", "").replace("wholenet_", "wn_").replace("_baseline", "")
        ax.annotate(f"{lbl}\n{r.subnet_params/1e6:.2f}M", (x, r.ap50), fontsize=7,
                    xytext=(4, -4), textcoords="offset points")
    ax.set_xlabel(xlabel + ("  (M)" if xcol == "subnet_params" else ""))
    ax.set_ylabel("AP@0.5 (DAIR val 1789, converged)")
    ax.axhline(0.75, color="gray", ls=":", lw=1)
    ax.grid(alpha=0.3)
fig.suptitle("Fig5  Whole-net pruning (deblocks/shrink) — AP50 flat ~0.75 across pruning;\n"
             "wn_aggr (1.32M) dominates backbone-only (2.58M): same AP, -49% params, -28% latency",
             fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig5_wholenet_prune.png", dpi=130, bbox_inches="tight")
print("saved", OUT / "fig5_wholenet_prune.png")
