"""对比流程图: 原始 V2Xverse 仿真架构 vs 本文 latency-aware 架构。
突出 τ_perc(ego 推理延迟)注入点 = ego 原始传感器输入与感知模型之间。
英文标签(matplotlib CJK 字体不保证); 中文解释在 experiments_simulation_zh_v1.md。
渲染: matplotlib(本地或 H800 v2xverse env)。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fc="#e3f2fd", ec="#1565c0", fs=9, lw=1.4, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#333", lw=1.6, ls="-", label=None, lcol=None, ldy=0.28):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=lw, ls=ls, zorder=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + ldy, label, ha="center", va="bottom",
                fontsize=7.8, color=lcol or color, style="italic", zorder=4)

fig, ax = plt.subplots(figsize=(14, 8.4))
ax.set_xlim(0, 17); ax.set_ylim(0, 11.5); ax.axis("off")

# ---- 通用 ego 管线 x 坐标 ----
xs = {"sensor": 1.7, "perc": 5.6, "fuse": 8.3, "plan": 10.8, "ctrl": 13.2, "carla": 15.6}
W, H = 2.0, 1.05

# ===================== 上: 原始架构 (baseline) =====================
yT = 9.0
ax.text(0.2, 11.0, "(a) Original V2Xverse closed-loop  —  ego perception real-time, tau_ego = 0",
        fontsize=12, fontweight="bold", color="#37474f")
box(ax, xs["sensor"], yT, W, H, "Ego sensors\n(LiDAR + pose)\nframe k", fc="#eceff1", ec="#546e7a")
box(ax, xs["perc"], yT, W, H, "Perception\n(CenterPoint)")
box(ax, xs["fuse"], yT, W, H, "Multi-scale\nBEV Fusion")
box(ax, xs["plan"], yT, W, H, "CoDriving\nPlanner")
box(ax, xs["ctrl"], yT, W, H, "V2X Controller\n(PID + L1)")
box(ax, xs["carla"], yT, W, H, "CARLA\nactuation", fc="#eceff1", ec="#546e7a")
arrow(ax, xs["sensor"] + W / 2, yT, xs["perc"] - W / 2, yT, label="real-time", lcol="#2e7d32")
arrow(ax, xs["perc"] + W / 2, yT, xs["fuse"] - W / 2, yT)
arrow(ax, xs["fuse"] + W / 2, yT, xs["plan"] - W / 2, yT)
arrow(ax, xs["plan"] + W / 2, yT, xs["ctrl"] - W / 2, yT)
arrow(ax, xs["ctrl"] + W / 2, yT, xs["carla"] - W / 2, yT)
# RSU 分支 (上)
yTr = yT - 1.85
box(ax, xs["perc"], yTr, 2.2, 0.95, "RSU sensors\n+ perception", fc="#f1f8e9", ec="#558b2f", fs=8.5)
box(ax, xs["perc"] + 2.7, yTr, 2.0, 0.95, "tau_RSU delay\n+ ZOH", fc="#fff3e0", ec="#ef6c00", fs=8.5)
arrow(ax, xs["perc"] + 1.1, yTr, xs["perc"] + 2.7 - 1.0, yTr)
arrow(ax, xs["perc"] + 2.7 + 1.0, yTr, xs["fuse"], yT - H / 2, label="comm latency", lcol="#ef6c00", ldy=0.05)

# 分隔线
ax.axhline(6.2, color="#bbb", lw=1, ls=":")

# ===================== 下: 本文架构 (ours) =====================
yB = 3.2
ax.text(0.2, 5.7, "(b) Ours: latency-aware  —  ego compute latency injected as perception information-age (tau_perc)",
        fontsize=12, fontweight="bold", color="#b71c1c")
box(ax, xs["sensor"], yB, W, H, "Ego sensors\n(LiDAR + pose)\nframe k", fc="#eceff1", ec="#546e7a")
# ★ 注入框 (红, 在 sensor 与 perception 之间)
xinj = (xs["sensor"] + xs["perc"]) / 2
box(ax, xinj, yB + 1.45, 3.0, 1.25,
    "* tau_perc INJECTION *\nframe bank + ZOH\n=> use frame  k - Delta_perc\nDelta_perc = ceil(tau_perc / 50ms)",
    fc="#ffebee", ec="#c62828", fs=8.2, lw=2.2, bold=True)
box(ax, xs["perc"], yB, W, H, "Perception\n(CenterPoint)")
box(ax, xs["fuse"], yB, W, H, "Multi-scale\nBEV Fusion")
box(ax, xs["plan"], yB, W, H, "CoDriving\nPlanner")
box(ax, xs["ctrl"], yB, W, H, "V2X Controller\n(PID + L1)")
box(ax, xs["carla"], yB, W, H, "CARLA\nactuation", fc="#eceff1", ec="#546e7a")
# ego sensor -> 注入框 -> perception
arrow(ax, xs["sensor"] + W / 2, yB + 0.2, xinj - 1.0, yB + 1.45 - 0.55, color="#c62828")
arrow(ax, xinj + 1.0, yB + 1.45 - 0.55, xs["perc"] - W / 2, yB + 0.2, color="#c62828",
      label="stale frame -> cognitive lag", lcol="#c62828", ldy=-0.75)
arrow(ax, xs["perc"] + W / 2, yB, xs["fuse"] - W / 2, yB)
arrow(ax, xs["fuse"] + W / 2, yB, xs["plan"] - W / 2, yB)
arrow(ax, xs["plan"] + W / 2, yB, xs["ctrl"] - W / 2, yB)
arrow(ax, xs["ctrl"] + W / 2, yB, xs["carla"] - W / 2, yB)
# RSU 分支 (下) - 不变
yBr = yB - 1.85
box(ax, xs["perc"], yBr, 2.2, 0.95, "RSU sensors\n+ perception", fc="#f1f8e9", ec="#558b2f", fs=8.5)
box(ax, xs["perc"] + 2.7, yBr, 2.0, 0.95, "tau_RSU delay\n+ ZOH", fc="#fff3e0", ec="#ef6c00", fs=8.5)
arrow(ax, xs["perc"] + 1.1, yBr, xs["perc"] + 2.7 - 1.0, yBr)
arrow(ax, xs["perc"] + 2.7 + 1.0, yBr, xs["fuse"], yB - H / 2, label="comm latency (unchanged)", lcol="#ef6c00", ldy=0.05)

# 注脚
ax.text(0.2, 0.3,
        "Key difference: the ego self-perception input is delayed by Delta_perc frames (ZOH on a frame bank) BEFORE the perception model, "
        "so the\nwhole perception->fusion->planning->control chain reasons on a stale world snapshot. CARLA runs at 20 Hz (50 ms/frame); "
        "tau_RSU (collaborative comm latency) is unchanged from the baseline.",
        fontsize=8.3, color="#444", va="bottom")

fig.tight_layout()
fig.savefig("/tmp/sim_arch_compare.png", dpi=150, bbox_inches="tight")
print("saved /tmp/sim_arch_compare.png")
