"""生成 dataset_v2 的描述性统计图 -> multi_agent/figure/data/*.png
可复跑: 数据更新(重跑 build_dataset_v2.py)后再跑本脚本即可刷新图。
标签用英文避免 matplotlib 中文缺字。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jichengzhi/V2X")
OUT = ROOT / "multi_agent/figure/data"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "multi_agent/data/dataset_v2.csv")

QMODE_COLOR = {"auto": "#2ca02c", "uniform_forced": "#d62728",
               "per_stage_mixed": "#1f77b4", "uniform": "#7f7f7f"}

def pareto_front(pts):  # pts: list[(label, lat, ap)] ; min lat, max ap
    nd = []
    for a in pts:
        dom = any((b[1] <= a[1] and b[2] >= a[2]) and (b[1] < a[1] or b[2] > a[2])
                  for b in pts if b is not a)
        if not dom:
            nd.append(a)
    return sorted(nd, key=lambda x: x[1])

# ---------- Fig 1: Pareto lat-AP (collab2, all 33 points unified) ----------
# 任务 A 后 33 点全部 body_subnet_collab2 同口径 -> 按 prune_rate 分面板,
# 两源 (complete_points uniform + perstage mixed/auto/forced) 合并同图比较。
c2 = df[df.latency_kind == "body_subnet_collab2"].copy()
trips = ["T_baseline", "T_prune50p", "T_prune75"]  # used by Fig3 below
PANELS = [(0.0, "prune 0% (base)"), (0.25, "prune 25%"),
          (0.5, "prune 50%"), (0.75, "prune 75%")]
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for ax, (pr, title) in zip(axes, PANELS):
    sub = c2[c2.prune_rate == pr]
    for _, r in sub.iterrows():
        ax.scatter(r.lat_p50_ms, r.ap50, c=QMODE_COLOR.get(r.q_mode, "k"), s=60,
                   edgecolors="k", linewidths=0.4, zorder=3)
        lbl = (str(r.config_label).replace("global_", "").replace("_automix", "-auto")
               .replace("rtx4090_", ""))
        ax.annotate(lbl, (r.lat_p50_ms, r.ap50), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    front = pareto_front([(r.config_label, r.lat_p50_ms, r.ap50) for _, r in sub.iterrows()])
    if len(front) >= 2:
        ax.plot([f[1] for f in front], [f[2] for f in front], "k--", lw=1, zorder=2,
                label="Pareto front")
    ax.set_title(f"{title}  (n={len(sub)})")
    ax.set_xlabel("lat_p50 (ms)  [collab2 body]")
    ax.set_ylabel("AP@0.5")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
handles = [plt.Line2D([0], [0], marker="o", ls="", c=c, label=k, mec="k")
           for k, c in QMODE_COLOR.items()]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.04))
fig.suptitle("Fig1  Pareto (latency vs AP), all 33 collab2 points unified — "
             "uniform (complete_points) + per-stage/auto/forced (perstage) same latency_kind",
             y=-0.02, fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig1_pareto_latAP_collab2.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- Fig 2: AP vs prune_rate (uniform-precision anchors, fp16 vs int8) ----------
# complete_points 的 6 个 uniform anchor (现 collab2 latency 口径), AP 沿用真测。
b1 = df[df.dataset_src == "complete_points_v1"].copy()
b1["prec"] = b1.stage0_prec  # uniform precision
fig, ax = plt.subplots(figsize=(6, 4.5))
for prec, mk, col in [("FP16", "o", "#1f77b4"), ("INT8", "s", "#d62728")]:
    s = b1[b1.prec == prec].sort_values("prune_rate")
    ax.plot(s.prune_rate, s.ap50, marker=mk, color=col, label=prec)
ax.set_xlabel("prune_rate")
ax.set_ylabel("AP@0.5 (DAIR val 1789)")
ax.set_title("Fig2  AP vs prune_rate (uniform-precision anchors)\nINT8 near-lossless vs FP16")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig2_ap_vs_prune.png", dpi=130)
plt.close(fig)

# ---------- Fig 3: delta_ap50_vs_baseline for per-stage mixed (the stage1-FP16 finding) ----------
ps = df[df.q_mode == "per_stage_mixed"].dropna(subset=["delta_ap50_vs_baseline"])
fig, ax = plt.subplots(figsize=(9, 4.5))
order = sorted(ps.config_label.unique())
x = range(len(order))
width = 0.27
for i, t in enumerate(trips):
    vals = [float(ps[(ps.triplet == t) & (ps.config_label == c)]["delta_ap50_vs_baseline"].iloc[0])
            if not ps[(ps.triplet == t) & (ps.config_label == c)].empty else 0 for c in order]
    ax.bar([xi + (i - 1) * width for xi in x], vals, width, label=t)
ax.axhline(0, color="k", lw=0.8)
ax.axhline(0.0013, color="gray", ls=":", lw=1, label="noise scale (+0.0013)")
ax.set_xticks(list(x)); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("ΔAP@0.5 vs forced-all-INT8")
ax.set_title("Fig3  per-stage mixed ΔAP vs forced-INT8 — c3/c5/c7 (keep stage1 FP16) are positive on pruned")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "fig3_delta_ap_perstage.png", dpi=130)
plt.close(fig)

# ---------- Fig 4: data inventory ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, title in [(axes[0], "latency_kind", "by latency_kind"),
                       (axes[1], "q_mode", "by q_mode"),
                       (axes[2], "dataset_src", "by dataset_src")]:
    vc = df[col].value_counts()
    ax.bar(range(len(vc)), vc.values, color="#4c72b0")
    ax.set_xticks(range(len(vc))); ax.set_xticklabels(vc.index, rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(vc.values):
        ax.text(i, v + 0.2, str(v), ha="center", fontsize=8)
    ax.set_title(title); ax.set_ylabel("# rows")
fig.suptitle(f"Fig4  dataset_v2 inventory (total {len(df)} complete points, all lat+AP)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig4_inventory.png", dpi=130)
plt.close(fig)

print("saved figures to", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name)
