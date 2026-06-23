# -*- coding: utf-8 -*-
"""
fig_latency_blindspot_v1.png
对比"真实世界时间线" vs "V2Xverse 现有仿真机制", 说明现有机制为何无法表现时延影响。
配套文档: real_test/latency_aware_simulation_design_v1.md
代码依据(已核实):
  - 同步模式挂起等推理: CARLA sync mode + leaderboard 主循环
  - 控制零阶保持: pnp_agent_e2e.py:448-450 (return self.infer.prev_control)
  - comm_latency 全局常数帧移: pnp_infer_action_e2e.py:494-516
"""
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 中文字体 (matplotlib 3.5 无逐字形回退, 需单一字体同时覆盖拉丁+CJK)
font_manager.fontManager.addfont('/usr/share/fonts/truetype/arphic/uming.ttc')
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

C_RSU, C_COMM, C_COMPUTE, C_CTRL_NEW, C_CTRL_OLD = '#8e7cc3', '#6fa8dc', '#f6b26b', '#93c47d', '#cccccc'
LANE_RSU, LANE_CMP, LANE_CTL = 3.0, 2.0, 1.0
BH = 0.5  # bar height

fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True,
                         gridspec_kw={'hspace': 0.42})


def setup_ax(ax, title):
    ax.set_xlim(-360, 620)
    ax.set_ylim(0.4, 3.9)
    ax.set_yticks([LANE_RSU + BH/2, LANE_CMP + BH/2, LANE_CTL + BH/2])
    ax.set_yticklabels(['RSU 协作数据', '自车 感知+规划', '控制生效'], fontsize=11)
    ax.set_title(title, fontsize=13, loc='left', pad=10)
    ax.axvline(0, color='k', lw=0.6, ls=':', alpha=0.5)
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)


# ================= Panel A: 真实世界 =================
ax = axes[0]
setup_ax(ax, '(A) 真实世界: 时延占据物理时间, 世界持续运动')

# RSU: 编码 + 传输
ax.broken_barh([(-150, 50)], (LANE_RSU, BH), color=C_RSU, alpha=0.9)
ax.broken_barh([(-100, 100)], (LANE_RSU, BH), color=C_COMM, alpha=0.9)
ax.text(-125, LANE_RSU + BH + 0.13, 'RSU编码\n50ms', ha='center', fontsize=8.5)
ax.text(-50, LANE_RSU + BH + 0.13, '传输 τ_comm\n100ms', ha='center', fontsize=8.5)
ax.annotate('', xy=(0, LANE_CMP + BH), xytext=(0, LANE_RSU),
            arrowprops=dict(arrowstyle='->', color=C_COMM, lw=1.4))

# 自车: 传感器采样 + 计算
ax.plot([0], [LANE_CMP + BH/2], marker='o', ms=7, color='k')
ax.text(8, LANE_CMP + BH + 0.13, '采样 t=0\n(障碍物已出现)', fontsize=8.5)
ax.broken_barh([(0, 300)], (LANE_CMP, BH), color=C_COMPUTE, alpha=0.95)
ax.text(150, LANE_CMP + BH/2, 'τ_compute = 300ms\n(期间车辆/障碍物都在动)',
        ha='center', va='center', fontsize=9)

# 控制: 旧计划跟踪 → 新计划生效
ax.broken_barh([(-300, 300)], (LANE_CTL, BH), color=C_CTRL_OLD)
ax.text(-150, LANE_CTL + BH/2, '底层控制器持续跟踪上一周期计划', ha='center', va='center', fontsize=8.5)
ax.broken_barh([(300, 300)], (LANE_CTL, BH), color=C_CTRL_NEW)
ax.text(450, LANE_CTL + BH/2, '基于 t=0 数据的新计划', ha='center', va='center', fontsize=9)
ax.annotate('', xy=(300, LANE_CTL + BH), xytext=(300, LANE_CMP),
            arrowprops=dict(arrowstyle='->', color='#e06666', lw=1.6))

# 反应延迟标注
ax.annotate('', xy=(300, 0.62), xytext=(0, 0.62),
            arrowprops=dict(arrowstyle='<->', color='#cc0000', lw=1.5))
ax.text(150, 0.44, '看到 → 反应 = 300ms', ha='center', color='#cc0000', fontsize=10, weight='bold')
ax.annotate('', xy=(300, 3.82), xytext=(-150, 3.82),
            arrowprops=dict(arrowstyle='<->', color='#674ea7', lw=1.3))
ax.text(75, 3.88, 'RSU 信息年龄 = 450ms (编码+传输+自车计算 链式相加)', ha='center',
        color='#674ea7', fontsize=9)

# ================= Panel B: V2Xverse 现状 =================
ax = axes[1]
setup_ax(ax, '(B) V2Xverse 现状: 同步模式把时延"折叠"为零')

# ④ comm_latency: 仅取旧帧
ax.plot([-300], [LANE_RSU + BH/2], marker='s', ms=8, color=C_RSU)
ax.annotate('', xy=(-8, LANE_RSU + BH/2), xytext=(-292, LANE_RSU + BH/2),
            arrowprops=dict(arrowstyle='->', color=C_RSU, lw=1.4, ls='--'))
ax.text(-150, LANE_RSU + BH + 0.13,
        '④ comm_latency(可选,官方配置均未启用→默认0延迟):\n启用时=常数帧移套给全部协作方; 无带宽/抖动/丢包', ha='center', fontsize=8.5)
ax.annotate('', xy=(0, LANE_CMP + BH), xytext=(-4, LANE_RSU),
            arrowprops=dict(arrowstyle='->', color=C_RSU, lw=1.2))

# ① 世界挂起, 推理不占仿真时间
ax.plot([0], [LANE_CMP + BH/2], marker='*', ms=18, color='#e06666', zorder=5)
ax.text(12, LANE_CMP + BH/2, '① CARLA 同步模式: 世界挂起等推理\n    仿真内 τ_compute ≡ 0 (模型快慢无差别)',
        fontsize=9, va='center')

# ② 控制瞬时生效
ax.annotate('', xy=(0, LANE_CTL + BH), xytext=(0, LANE_CMP),
            arrowprops=dict(arrowstyle='->', color='#e06666', lw=1.8))
ax.text(-12, (LANE_CMP + LANE_CTL + BH)/2, '② 瞬时生效', fontsize=9, ha='right', color='#cc0000')

# ③ skip_frames 间指令冻结
for k, (start, lbl) in enumerate([(0, '指令A'), (200, '指令B'), (400, '指令C')]):
    ax.broken_barh([(start, 200)], (LANE_CTL, BH), color=C_CTRL_NEW if k == 0 else C_CTRL_OLD,
                   edgecolor='white')
    ax.text(start + 100, LANE_CTL + BH/2, f'{lbl} 冻结 200ms', ha='center', va='center', fontsize=8.5)
    ax.plot([start], [LANE_CMP + BH/2], marker='*', ms=12, color='#e06666', zorder=5)
ax.text(300, LANE_CTL - 0.32,
        '③ skip_frames=4: 非推理帧 return prev_control — 油门/转向角原样冻结 (pnp_agent_e2e.py:448-450)',
        ha='center', fontsize=8.8, color='#444444')

ax.annotate('', xy=(0, 0.62), xytext=(-1, 0.62),
            arrowprops=dict(arrowstyle='-', color='#cc0000', lw=0))
ax.text(0, 0.44, '看到 → 反应 = 0ms,  与模型快慢无关', ha='left', color='#cc0000',
        fontsize=10, weight='bold')

axes[1].set_xlabel('仿真时间 (ms),  CARLA 20Hz → 1帧 = 50ms', fontsize=11)

fig.suptitle('为什么现有 V2Xverse 仿真机制无法表现时延对驾驶的影响', fontsize=15, y=0.985)
fig.text(0.5, 0.005,
         '盲区: ① 计算耗时被同步模式吞掉(不占仿真时间)  ② 控制基于 t 帧数据在 t 帧瞬时生效  '
         '③ 延迟期间指令零阶保持而非轨迹跟踪  ④ 通信延迟只是全局常数帧移\n'
         '→ 结果: 无论模型推理 50ms 还是 500ms, 闭环驾驶成绩完全相同; '
         '改造方案见 real_test/latency_aware_simulation_design_v1.md',
         ha='center', fontsize=9.5, color='#333333')

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
out = '/home/jichengzhi/V2X/multi_agent/figure/fig_latency_blindspot_v1.png'
fig.savefig(out, dpi=160)
print('saved:', out)
