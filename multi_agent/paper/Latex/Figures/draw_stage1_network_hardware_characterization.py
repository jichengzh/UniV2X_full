#!/usr/bin/env python3
"""Draw the Stage-1 network--hardware characterization framework.

All geometry is defined in normalized axes coordinates so the vector outputs
remain editable and stable at the final two-column manuscript size.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path(__file__).resolve().parent
OUT_STEM = OUT_DIR / "fig_stage1_network_hardware_characterization_zh"
CJK_FONT_PATH = "/usr/share/fonts/truetype/arphic/uming.ttc"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Droid Sans Fallback", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    }
)


COLORS = {
    "ink": "#243447",
    "muted": "#667384",
    "line": "#AAB4C0",
    "neutral": "#F5F7F9",
    "neutral_2": "#E9EEF3",
    "hardware_fill": "#E4F0FA",
    "hardware_edge": "#4C78A8",
    "network_fill": "#EEE9F7",
    "network_edge": "#7A6AAE",
    "couple_fill": "#E3F2EE",
    "couple_edge": "#3E8C7A",
    "integrate_fill": "#F8E9D7",
    "integrate_edge": "#C47A32",
    "profile_fill": "#EDF3F7",
    "validate_fill": "#EAF2E8",
    "manifest_fill": "#2F526F",
    "white": "#FFFFFF",
}


def box(ax, x, y, w, h, *, fc, ec, lw=1.0, radius=0.012, z=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax,
    x,
    y,
    text,
    *,
    size=7,
    color=None,
    weight="normal",
    ha="center",
    va="center",
    z=4,
    linespacing=1.15,
):
    font = font_manager.FontProperties(fname=CJK_FONT_PATH, size=size, weight=weight)
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontproperties=font,
        color=color or COLORS["ink"],
        linespacing=linespacing,
        zorder=z,
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.15,
    style="-|>",
    scale=9,
    connection="arc3,rad=0",
    z=2,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=lw,
        color=color or COLORS["muted"],
        connectionstyle=connection,
        transform=ax.transAxes,
        clip_on=False,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def chip(ax, x, y, w, h, title, body, *, fc, ec):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=0.75, radius=0.008, z=2)
    label(ax, x + 0.012, y + h * 0.68, title, size=6.4, weight="bold", ha="left")
    label(
        ax,
        x + 0.012,
        y + h * 0.31,
        body,
        size=5.25,
        color=COLORS["muted"],
        ha="left",
        linespacing=1.05,
    )


def draw_hardware_scan(ax):
    x, y, w, h = 0.03, 0.625, 0.45, 0.305
    box(ax, x, y, w, h, fc=COLORS["hardware_fill"], ec=COLORS["hardware_edge"], lw=1.25)
    box(ax, x, y + h - 0.045, w, 0.045, fc=COLORS["hardware_edge"], ec=COLORS["hardware_edge"], radius=0.010)
    label(ax, x + 0.018, y + h - 0.0225, "A", size=8.5, color=COLORS["white"], weight="bold", ha="left")
    label(ax, x + 0.055, y + h - 0.0225, "硬件能力扫描", size=8, color=COLORS["white"], weight="bold", ha="left")

    box(ax, x + 0.055, y + 0.205, w - 0.11, 0.035, fc=COLORS["white"], ec=COLORS["hardware_edge"], lw=0.75, radius=0.017)
    label(ax, x + w / 2, y + 0.2225, "目标加速器  +  capability YAML", size=6.2, weight="bold")

    chip(ax, x + 0.025, y + 0.128, 0.19, 0.060, r"$\mathcal{I}$  计算 IP", "支持精度 · 算子白名单", fc=COLORS["white"], ec="#A7C5DF")
    chip(ax, x + 0.235, y + 0.128, 0.19, 0.060, r"$\mathbf{a}$  通道对齐", "INT8/FP16 · 强制模式", fc=COLORS["white"], ec="#A7C5DF")
    chip(ax, x + 0.025, y + 0.058, 0.19, 0.060, r"$\mathcal{Q}$  量化能力", "位宽 · 粒度 · 对称性", fc=COLORS["white"], ec="#A7C5DF")
    chip(ax, x + 0.235, y + 0.058, 0.19, 0.060, r"$\mathcal{T}$  工具链", "后端与版本锁定", fc=COLORS["white"], ec="#A7C5DF")

    box(ax, x + 0.092, y + 0.009, w - 0.184, 0.034, fc="#D4E7F6", ec=COLORS["hardware_edge"], lw=0.85, radius=0.015)
    label(ax, x + w / 2, y + 0.026, r"$\mathcal{H}=\langle\mathcal{I},\mathbf{a},\mathcal{Q},\mathcal{T}\rangle$", size=7.0, weight="bold")
    return (x + w / 2, y)


def draw_network_scan(ax):
    x, y, w, h = 0.52, 0.625, 0.45, 0.305
    box(ax, x, y, w, h, fc=COLORS["network_fill"], ec=COLORS["network_edge"], lw=1.25)
    box(ax, x, y + h - 0.045, w, 0.045, fc=COLORS["network_edge"], ec=COLORS["network_edge"], radius=0.010)
    label(ax, x + 0.018, y + h - 0.0225, "B", size=8.5, color=COLORS["white"], weight="bold", ha="left")
    label(ax, x + 0.055, y + h - 0.0225, "计算图依赖分区", size=8, color=COLORS["white"], weight="bold", ha="left")

    box(ax, x + 0.095, y + 0.205, w - 0.19, 0.035, fc=COLORS["white"], ec=COLORS["network_edge"], lw=0.75, radius=0.017)
    label(ax, x + w / 2, y + 0.2225, "待加速网络  +  checkpoint", size=6.2, weight="bold")

    flow_y, flow_h = y + 0.145, 0.043
    flow_specs = [
        (x + 0.018, 0.115, "轻量 trace\n适配器"),
        (x + 0.167, 0.122, "稠密通道\n耦合核心"),
        (x + 0.323, 0.108, "DepGraph\n依赖分析"),
    ]
    for fx, fw, text in flow_specs:
        box(ax, fx, flow_y, fw, flow_h, fc=COLORS["white"], ec="#B9ADD5", lw=0.75, radius=0.008)
        label(ax, fx + fw / 2, flow_y + flow_h / 2, text, size=5.35, weight="bold", linespacing=1.0)
    arrow(ax, (x + 0.135, flow_y + flow_h / 2), (x + 0.164, flow_y + flow_h / 2), color=COLORS["network_edge"], scale=7)
    arrow(ax, (x + 0.291, flow_y + flow_h / 2), (x + 0.320, flow_y + flow_h / 2), color=COLORS["network_edge"], scale=7)

    label(
        ax,
        x + w / 2,
        y + 0.126,
        "分区外：稀疏/数据依赖编码；绕过：通道保持融合 neck",
        size=4.9,
        color=COLORS["muted"],
    )

    views_y, views_h = y + 0.056, 0.052
    view_specs = [
        (x + 0.020, "B1  剪枝组", r"$G=|\mathrm{B1}|$"),
        (x + 0.165, "B2  量化单元", "完整 B1 组并集"),
        (x + 0.310, "D  路由标注", "逐算子可达性"),
    ]
    for vx, title, body in view_specs:
        box(ax, vx, views_y, 0.125, views_h, fc=COLORS["white"], ec="#B9ADD5", lw=0.75, radius=0.008)
        label(ax, vx + 0.0625, views_y + 0.034, title, size=5.4, weight="bold")
        label(ax, vx + 0.0625, views_y + 0.014, body, size=4.75, color=COLORS["muted"])

    box(ax, x + 0.095, y + 0.009, w - 0.19, 0.034, fc="#E0D8F0", ec=COLORS["network_edge"], lw=0.85, radius=0.015)
    label(ax, x + w / 2, y + 0.026, r"$\mathcal{G}=\langle\mathrm{B1},\mathrm{B2},\mathrm{D}\rangle$", size=7.0, weight="bold")
    return (x + w / 2, y)


def draw_coupling(ax, left_anchor, right_anchor):
    x, y, w, h = 0.09, 0.468, 0.82, 0.135
    arrow(ax, left_anchor, (x + 0.24, y + h), color=COLORS["hardware_edge"], lw=1.2, scale=9)
    arrow(ax, right_anchor, (x + 0.58, y + h), color=COLORS["network_edge"], lw=1.2, scale=9)
    box(ax, x, y, w, h, fc=COLORS["couple_fill"], ec=COLORS["couple_edge"], lw=1.2)
    label(ax, x + w / 2, y + h - 0.025, "硬件–网络约束耦合", size=7.7, weight="bold", color="#275F54")

    col_w = 0.245
    col_x = [x + 0.025, x + 0.288, x + 0.551]
    specs = [
        ("A → B", "对齐 → 剪枝宽度\n量化能力 → 位宽/粒度"),
        ("D 与 B2 双向传播", "DLA → INT8 / per-tensor\nper-channel → 禁用 DLA"),
        ("B → A", "算子集合 → 扫描范围\n路由标注 → 连续分段"),
    ]
    for cx, (head, body) in zip(col_x, specs):
        box(ax, cx, y + 0.018, col_w, 0.065, fc=COLORS["white"], ec="#A9D0C7", lw=0.7, radius=0.007)
        label(ax, cx + 0.012, y + 0.062, head, size=5.8, weight="bold", color=COLORS["couple_edge"], ha="left")
        label(ax, cx + 0.012, y + 0.034, body, size=5.0, color=COLORS["muted"], ha="left", linespacing=1.05)
    return (x + w / 2, y)


def draw_integration(ax, anchor):
    x, y, w, h = 0.09, 0.265, 0.82, 0.155
    arrow(ax, anchor, (x + w / 2, y + h), color=COLORS["couple_edge"], lw=1.25, scale=9)
    box(ax, x, y, w, h, fc=COLORS["integrate_fill"], ec=COLORS["integrate_edge"], lw=1.2)
    label(ax, x + 0.025, y + h - 0.026, "整合层：结构真相 → 搜索旋钮", size=7.7, weight="bold", color="#80501F", ha="left")

    rows = [
        (y + 0.092, "B1", r"$G=|\mathrm{B1}|$ 个剪枝组", r"$K\ll G$ 个 stage 级旋钮"),
        (y + 0.059, "B2", "完整耦合组并集", "合法位宽 / 粒度"),
        (y + 0.026, "D", "逐算子路由标注", "2–4 个连续路由段"),
    ]
    for ry, tag, left, right in rows:
        box(ax, x + 0.025, ry - 0.012, 0.044, 0.025, fc=COLORS["white"], ec=COLORS["integrate_edge"], lw=0.65, radius=0.010)
        label(ax, x + 0.047, ry, tag, size=5.5, weight="bold", color=COLORS["integrate_edge"])
        label(ax, x + 0.085, ry, left, size=5.25, ha="left")
        arrow(ax, (x + 0.285, ry), (x + 0.345, ry), color=COLORS["integrate_edge"], lw=0.85, scale=7)
        label(ax, x + 0.365, ry, right, size=5.25, weight="bold", ha="left")

    box(ax, x + 0.615, y + 0.027, 0.175, 0.080, fc=COLORS["white"], ec=COLORS["integrate_edge"], lw=0.85, radius=0.009)
    label(ax, x + 0.7025, y + 0.084, "剪枝空间压缩", size=5.7, weight="bold", color=COLORS["integrate_edge"])
    label(ax, x + 0.7025, y + 0.055, r"$6^{43}\approx10^{33}$", size=6.4, weight="bold")
    arrow(ax, (x + 0.677, y + 0.039), (x + 0.728, y + 0.039), color=COLORS["integrate_edge"], lw=0.9, scale=7)
    label(ax, x + 0.759, y + 0.039, r"$6^{5}\approx10^{4}$", size=6.4, weight="bold")
    return (x + w / 2, y)


def draw_outputs(ax, anchor):
    profile = (0.045, 0.055, 0.245, 0.145)
    validate = (0.355, 0.055, 0.215, 0.145)
    manifest = (0.635, 0.045, 0.33, 0.165)

    arrow(ax, anchor, (profile[0] + profile[2] / 2, profile[1] + profile[3]), color=COLORS["integrate_edge"], lw=1.15, scale=9, connection="arc3,rad=0.10")

    box(ax, *profile, fc=COLORS["profile_fill"], ec="#6F8FA6", lw=1.0)
    label(ax, profile[0] + profile[2] / 2, profile[1] + profile[3] - 0.028, r"基线延迟刻画  $\lambda$", size=6.7, weight="bold", color="#45677F")
    label(ax, profile[0] + profile[2] / 2, profile[1] + 0.072, "CUDA-Event 逐算子计时\n按统一层名聚合至搜索旋钮", size=5.2, linespacing=1.12)
    box(ax, profile[0] + 0.027, profile[1] + 0.014, profile[2] - 0.054, 0.027, fc=COLORS["white"], ec="#AFC2D0", lw=0.65, radius=0.012)
    label(ax, profile[0] + profile[2] / 2, profile[1] + 0.0275, "描述性输入先验 · 非合法性约束", size=4.7, color=COLORS["muted"])

    arrow(ax, (profile[0] + profile[2], profile[1] + profile[3] / 2), (validate[0], validate[1] + validate[3] / 2), color=COLORS["muted"], lw=1.05, scale=8)
    box(ax, *validate, fc=COLORS["validate_fill"], ec="#6E9271", lw=1.0)
    label(ax, validate[0] + validate[2] / 2, validate[1] + validate[3] - 0.030, r"试剪校验  $v$", size=6.7, weight="bold", color="#4F7553")
    label(ax, validate[0] + validate[2] / 2, validate[1] + 0.077, r"$r=0.5$ 物理通道剪枝", size=5.4, weight="bold")
    label(ax, validate[0] + validate[2] / 2, validate[1] + 0.047, "+ 一次前向", size=5.2)
    box(ax, validate[0] + 0.050, validate[1] + 0.012, validate[2] - 0.100, 0.025, fc=COLORS["white"], ec="#B8CDBA", lw=0.65, radius=0.011)
    label(ax, validate[0] + validate[2] / 2, validate[1] + 0.0245, "fail-fast", size=4.9, color=COLORS["muted"], weight="bold")

    arrow(ax, (validate[0] + validate[2], validate[1] + validate[3] / 2), (manifest[0], manifest[1] + manifest[3] / 2), color=COLORS["muted"], lw=1.05, scale=8)
    box(ax, *manifest, fc=COLORS["manifest_fill"], ec=COLORS["manifest_fill"], lw=1.1)
    label(ax, manifest[0] + manifest[2] / 2, manifest[1] + manifest[3] - 0.032, "分区清单  partition manifest", size=6.8, color=COLORS["white"], weight="bold")
    label(ax, manifest[0] + manifest[2] / 2, manifest[1] + 0.091, r"$\mathcal{M}=\langle\mathcal{H},\mathcal{G},\Theta,\lambda,v\rangle$", size=7.2, color=COLORS["white"], weight="bold")
    label(ax, manifest[0] + manifest[2] / 2, manifest[1] + 0.055, r"$\Theta$: 低维合法搜索空间", size=5.2, color="#DCE8F0")
    box(ax, manifest[0] + 0.078, manifest[1] + 0.012, manifest[2] - 0.156, 0.027, fc=COLORS["white"], ec=COLORS["white"], lw=0.5, radius=0.012)
    label(ax, manifest[0] + manifest[2] / 2, manifest[1] + 0.0255, "阶段二联合搜索输入", size=5.2, color=COLORS["manifest_fill"], weight="bold")


def build_figure():
    fig = plt.figure(figsize=(7.2, 4.65), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    label(ax, 0.5, 0.979, "阶段一：面向部署的网络–硬件协同刻画", size=10.0, weight="bold")
    label(ax, 0.5, 0.947, "并行扫描定义合法配置域，整合与校验生成阶段二可直接消费的低维搜索清单", size=5.7, color=COLORS["muted"])

    hardware_anchor = draw_hardware_scan(ax)
    network_anchor = draw_network_scan(ax)
    coupling_anchor = draw_coupling(ax, hardware_anchor, network_anchor)
    integration_anchor = draw_integration(ax, coupling_anchor)
    draw_outputs(ax, integration_anchor)
    return fig


def save_figure(fig):
    fig.savefig(OUT_STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUT_STEM.with_suffix(".tiff"), dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    figure = build_figure()
    save_figure(figure)
    plt.close(figure)
    print(f"Saved figure bundle to {OUT_STEM.parent}")
