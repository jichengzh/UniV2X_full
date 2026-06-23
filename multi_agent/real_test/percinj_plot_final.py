"""最终 18 点折线图: 感知延迟 τ_perc → V2X 驾驶退化 (诚实口径)
双面板: 上=诚实DS(带95%CI), 下=碰撞率+灾难率; 标注 平台/过渡/硬悬崖/第二平台/续降 五区段。
数据直接读 percinj_curve_full.csv。渲染: /data/jichengzhi_v2x/envs/v2xverse/bin/python
"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/data/jichengzhi_v2x/percinj_out/percinj_curve_full.csv"
OUT = "/data/jichengzhi_v2x/percinj_out"

tau, cat, ds, ci, col = [], [], [], [], []
with open(CSV) as f:
    for row in csv.DictReader(f):
        tau.append(float(row["tau_perc_ms"]))
        cat.append(float(row["catastrophe_rate_pct"]))
        ds.append(float(row["honest_meanDS"]))
        ci.append(float(row["ci95"]))
        col.append(float(row["completed_collision_rate"]) * 100)  # → %

# 三色风险分区
regions = [
    (0, 600,    "#a5d6a7", "SAFE  (0-600ms)\nDS plateau, gentle decline"),
    (600, 800,  "#fff176", "WARNING  (600-800ms)\nhard cliff, DS -> ~66"),
    (800, 1000, "#ef9a9a", "DANGER  (800-1000ms)\ncollapse, catastrophe -> 30%"),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2]})

for x0, x1, c, _ in regions:
    ax1.axvspan(x0, x1, color=c, alpha=0.55, zorder=0)
    ax2.axvspan(x0, x1, color=c, alpha=0.55, zorder=0)

# 上: 诚实 DS
ax1.errorbar(tau, ds, yerr=ci, fmt="o-", color="tab:blue", lw=2.2, ms=7, capsize=4, zorder=3, label="Honest Driving Score (timeout=0)")
ax1.set_ylabel("Honest Driving Score\n(timeout=DS0, mean +/- 95%CI)", fontsize=11)
ax1.set_ylim(40, 102)
ax1.grid(alpha=0.3, zorder=1)
for x, y in zip(tau, ds):
    ax1.annotate("%.0f" % y, (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7.5, color="tab:blue")
ax1.axvline(600, color="red", lw=1.5, ls="--", zorder=4)
ax1.text(600, 100, " critical threshold ~600ms", color="red", fontsize=8, va="top", ha="left")
ax1.set_title("Perception latency -> V2X driving degradation (HONEST: timeout=DS0)\n"
              "H800 full-traffic, 35 tp0-clean routes x N=3 (105 runs/point), 18 latency points", fontsize=12)
ax1.legend(loc="upper right", fontsize=9)
# 区段标签
for x0, x1, c, lab in regions:
    ax1.text((x0 + x1) / 2, 44, lab, ha="center", va="bottom", fontsize=7.5, color="#444")

# 下: 碰撞率 + 灾难率
ax2.plot(tau, col, "s-", color="tab:red", lw=2, ms=6, zorder=3, label="Collision rate (completed routes) %")
ax2.plot(tau, cat, "^--", color="darkviolet", lw=1.8, ms=6, zorder=3, label="Catastrophe (timeout) rate %")
ax2.set_ylabel("rate (%)", fontsize=11)
ax2.set_xlabel(r"perception latency $\tau_{perc}$ (ms)", fontsize=12)
ax2.set_ylim(0, 65)
ax2.grid(alpha=0.3, zorder=1)
ax2.axvline(600, color="red", lw=1.5, ls="--", zorder=4)
ax2.legend(loc="upper left", fontsize=9)
ax2.set_xticks(tau); ax2.tick_params(axis="x", labelsize=8, rotation=45)

fig.tight_layout()
fig.savefig(OUT + "/percinj_curve_final.png", dpi=145)
plt.close()
print("最终折线图 -> percinj_curve_final.png (%d 点)" % len(tau))
