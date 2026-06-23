# I-1 L1 轨迹跟踪控制器设计 (τ_ego 硬前提)

> 作者: sim-integrator, 2026-06-09
> 依据: `sim_ego_rsu_latency_strategy_v1.md §3 I-1`
> 状态: **待 team-lead 审核** — 审核通过后开始写码
> 代码落点: V2Xverse `feature/l1-trajectory-tracker` 分支

---

## §1 代码路径审计结果 (逐文件实测)

### 1.1 控制调用链

```
每 CARLA tick (20Hz = 50ms):
  autonomous_agent.__call__()                 [autoagents/autonomous_agent.py:103-118]
    timestamp = GameTime.get_time()           # CARLA 仿真时间 (秒), 每 tick +0.05s
    input_data = sensor_interface.get_data()
    control = self.run_step(input_data, timestamp)
    → PnP_Agent.run_step(input_data, timestamp) [pnp_agent_e2e.py:436-489]
```

### 1.2 PnP_Agent.run_step 现状 (L0)

```python
# pnp_agent_e2e.py 456-458
if self.step % skip_frames != 0 and self.step > skip_frames:
    return self.infer.prev_control          # ★L0: 冻结上一帧指令, 无 tick() 调用
                                            #  → target_point 不更新, 弯道偏转失真

# 推理帧 (每 4 tick 一次, 200ms):
ego_data = [self.tick(input_data, vehicle_num) ...]  # 更新 pose + target_point
control_all = self.infer.get_action_from_list_inter(..., timestamp=timestamp)
```

### 1.3 planner 输出格式 (实测)

| 字段 | 形状 | 坐标系 | 来源 |
|------|------|--------|------|
| `pred_waypoints` | (10, 2) | **ego-local** | `planning_model(planning_input)['future_waypoints']` |
| `waypoints[i][0]` | float | 横向 (右=+) | BEV 列方向 |
| `waypoints[i][1]` | float | 纵向 | BEV 行方向 |
| 时间间隔 | — | **0.2s/wp** | 从 `desired_speed = spacing × 5` 推导 (5 = 1/0.2s) |

> ★ 坐标系实现时必须跑 round-trip sanity check (§5.1 验收条件 V0)。

### 1.4 V2X_Controller.run_step 现状

```python
# v2x_controller.py 134-141
aim_wp = (waypoints[-2] + waypoints[-1]) / 2.0      # ← 永远跟 wp[-1/-2], 与 plan 年龄无关
theta_tg = np.arctan2(aim[0], aim[1]+1e-7)         # aim = target_point (全局路由)
weight = 1
angle = angle_wp * (1-weight) + angle_tg * weight   # weight=1 → 转向 100% 由 target_point 驱动
desired_speed = np.mean(np.diff(displacement)[:3]) * max_speed  # 前 3 wp 间距
```

**关键结论**: 转向由 `target_point` (全局路由) 驱动, **预测 waypoints 只影响 desired_speed**。
L1 修改点:
1. 每 tick 更新 `target_point` (当前最新, 不冻结)
2. desired_speed 改用**时间索引**后的 waypoint 间距 (计划变旧时自动减速)

---

## §2 L1 设计

### 2.1 核心思路

```
帧 k (推理帧): ego_pose_k + pred_waypoints_local_k
  → 锚定到世界系: waypoints_world = ego_local_to_world(wp_local, pose_k)
  → 存入 active_plan{waypoints_world, t_anchor=timestamp_k, pose_anchor=pose_k}

帧 k+j (非推理帧, j=1,2,3):
  → tick() 获取最新 pose_now, target_point_now
  → world → current ego-local: wp_now = world_to_ego_local(waypoints_world, pose_now)
  → 时间索引: dt = t_now − t_anchor; t_idx = dt / 0.2
  → 插值参考点: wp_ref = interp(wp_now, t_idx)
  → V2X_Controller.run_step({speed, waypoints=wp_now, target=target_point_now, aim_override=wp_ref})
  → 返回控制指令 (非冻结, 每 tick 刷新)
```

### 2.2 坐标变换 (ego-local ↔ world)

ego-local → world:
```python
# measurements['x'] = pos[1], measurements['y'] = -pos[0] (pnp_agent_e2e.py:374-375)
# theta = compass (radians)
# 与 transform_2d_points / x_to_world 一致的旋转约定 (pnp_infer_action_e2e.py:255-272)
R = [[cos(theta), -sin(theta)],
     [sin(theta),  cos(theta)]]
wp_world = (R @ wp_local.T).T + [ego_x, ego_y]
```

world → ego-local:
```python
wp_rel = wp_world - [ego_x_now, ego_y_now]
wp_local_now = (R_now.T @ wp_rel.T).T
```

> 实现后跑 round-trip: `wp_local_rt = world_to_local(local_to_world(wp, p), p)`, 要求 `|wp - wp_local_rt| < 1e-4` ✓

### 2.3 时间索引插值

```python
DT_PER_WP = 0.2   # 秒/waypoint (从 desired_speed = spacing×5 推导)

def time_index_ref(wp_local: ndarray, dt: float) -> ndarray:
    """dt: 自 t_anchor 起经过的秒数. 返回 ego-local 参考点 (2,)."""
    t_idx = dt / DT_PER_WP        # 浮点索引
    t_idx = min(t_idx, 9.0)       # 夹断到计划末端
    lo = int(t_idx); frac = t_idx - lo
    if lo < 9:
        return wp_local[lo] * (1 - frac) + wp_local[lo+1] * frac
    return wp_local[9]
```

### 2.4 desired_speed 修改

当前: `np.mean(np.diff(displacement)[:3]) * max_speed` (永远用前 3 点)
L1: `np.mean(np.diff(displacement)[lo:lo+3]) * max_speed` (从时间索引当前位置起前 3 点)
→ plan 变旧时速度自然收敛到计划尾端的慢速区域, 不会一直按计划初速狂冲

### 2.5 转向: V2X_Controller 的 `aim_override`

`V2X_Controller` `weight=1` → 转向已纯由 `target_point` 驱动。L1 的关键不是改转向算法, 而是:
- **每 tick 都调 tick() → target_point 永远是当前最新路由点** (L0 冻结帧不调 tick() 是 bug 来源)
- `aim_wp` (只影响 `angle_wp`, weight=0 后无效) 可保留原有逻辑; 或传入 `wp_ref` 作为备用

> 决策: V1 实现先保持 `weight=1` 不变, L1 只修 speed + 解冻 `tick()` 调用。如 baseline 验收异常再考虑调 `weight`。

---

## §3 文件改动清单

### 3.1 新文件: `simulation/leaderboard/team_code/closedloop/l1_trajectory_controller.py`

```
class L1PlanStore:            # 存储当前有效 plan (world-frame)
  .update(wp_ego, x, y, θ, t_anchor)
  .get_local_ref(x_now, y_now, θ_now, t_now) → (wp_local_now, dt)

def ego_local_to_world(...)   # ~10 行
def world_to_ego_local(...)   # ~10 行
def time_index_ref(...)       # ~10 行
def l1_desired_speed(...)     # ~5 行 (时间索引版)
```

预计 **~60 行**

### 3.2 `pnp_infer_action_e2e.py` 修改

位置: `generate_action_from_model_output()` line ~940, 在 `pred_waypoints` 计算后

```python
# ★I-1 L1: 锚定 waypoints 到世界系
if self._l1_plan_store is not None:
    m = car_data_raw[0]['measurements']
    self._l1_plan_store.update(
        pred_waypoints,          # ego-local (10, 2)
        m['x'], m['y'], m['theta'],
        timestamp                # 来自 get_action_from_list_inter 参数
    )
```

预计 **~20 行** (含初始化 `_l1_plan_store`)

### 3.3 `pnp_agent_e2e.py` 修改

```python
# 每 tick 都调 tick() (而非只在推理帧)
ego_data_tick = [self.tick(input_data, v) for v in range(self.ego_vehicles_num)]

if self.step % skip_frames != 0 and self.step > skip_frames:
    # ★I-1 L1: 非推理帧用 L1 controller (替代 L0 冻结)
    if self.config['simulation'].get('l1_control', False) \
            and self.infer._l1_plan_store is not None:
        return self.infer.run_l1_step(ego_data_tick, timestamp)
    return self.infer.prev_control  # L0 fallback (config 未开启 or 还没有计划)
```

`run_l1_step()` 加到 `PnP_infer` (~30 行): 调 `_l1_plan_store.get_local_ref()`, 构 `route_info`, 调 `V2X_Controller.run_step()`。

预计 **~40 行**

### 3.4 Config 新增字段

```yaml
simulation:
    l1_control: true    # 开启 L1 轨迹跟踪 (默认 false → L0 兼容)
```

### 3.5 改动摘要

| 文件 | 改动性质 | 估计行数 |
|------|---------|---------|
| `closedloop/l1_trajectory_controller.py` | 新增 | ~60 |
| `pnp_infer_action_e2e.py` | 修改 (2 处) | ~20 |
| `pnp_agent_e2e.py` | 修改 (1 处) | ~40 |
| config yaml | 新增字段 | ~5 |
| **合计** | | **~125 行** |

---

## §4 与现有代码的兼容性

- `l1_control: false` (默认) → 完全走原来 L0 路径, 零行为变化, 不影响已有 probe 结果
- 在新 config 中加 `l1_control: true` 才触发 L1 逻辑
- `V2X_Controller.run_step()` 接口不变 (只改 `route_info` 内容)
- 已有 latency_inject / trace_replay / disable_rsu 全部正交, 不冲突

---

## §5 验收条件

### V0 (纯代码, 无 GPU)
```bash
cd /home/jichengzhi/V2Xverse
python -c "
from simulation.leaderboard.team_code.closedloop.l1_trajectory_controller import *
import numpy as np
wp = np.random.randn(10, 2)
pose = (10.0, 20.0, 0.5)
wp_w = ego_local_to_world(wp, *pose)
wp_rt = world_to_ego_local(wp_w, *pose)
assert np.allclose(wp, wp_rt, atol=1e-5), f'round-trip fail: max_err={np.max(np.abs(wp-wp_rt))}'
print('V0 PASS: round-trip ok')
"
```

### V1 (需 GPU 分卡冒烟, 宣告后执行)
- 跑 r146 × {d0ms, d500ms, norsu} with `l1_control: true`
- 验收: L1 + τ_ego=0 时 DS ≥ 90 (L0 同条件 DS=100, 允许控制机制切换带来小幅变化)
- 特别关注: r146 d500ms DS 是否仍 < r146 d0ms (保留敏感信号)

---

## §6 开问题 (供 team-lead 决策)

| # | 问题 | 建议 |
|---|------|------|
| Q1 | `weight=1` 不变还是改为 `weight=0.5` 让预测 waypoints 参与转向? | 建议 v1 保持 `weight=1` 先验收 baseline, 需要时再调 |
| Q2 | 非推理帧也调 `tick()` → BirdView producer 每 tick 都跑 (~5ms): 可接受? | 可接受, 后续可缓存; 先跑通优先 |
| Q3 | `l1_control` config flag 是在现有 config yaml 加字段, 还是新建 `pnp_config_codriving_lat_d0_l1.yaml`? | 建议现有 yaml 加字段, 向后兼容 |
| Q4 | 若 `_l1_plan_store` 为 None (计划尚未计算, 仿真开头几帧): fall back L0 还是 gentle-brake? | 建议 L0 fallback (prev_control), 简单可靠 |
