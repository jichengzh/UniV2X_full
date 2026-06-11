# L1 控制器架构现存问题拆解 (供 review, 决定如何修)

> 作者: team-lead, 2026-06-10。背景: I-2 τ_ego sweep 在 r146 平 (te0/rep=60撞车, te108/136/219=65撞墙), 机制已确认生效 (Δ_ego 真施加) 但 DS 无单调安全信号。根因经代码审计定位到 L1 控制器架构。本文逐条拆解, 供用户 review 后决定修法。
> 代码: V2Xverse `feature/l1-trajectory-tracker`, `closedloop/l1_trajectory_controller.py` + `pnp_infer_action_e2e.py:run_l1_step()`。

---

## §0 L1 是干什么的 (定位)

CoDriving 推理 = **5Hz** (skip_frames=4, 每 4 个 CARLA tick=200ms 才推理一次)。CARLA 是 **20Hz** (50ms/tick)。中间 3 个 tick 没有新推理结果, 需要一个控制器让车**平滑前进**。这就是 L1:
- **推理帧 (每 4 tick)**: 感知 → CoDriving planner → `pred_waypoints` (10点, ego-local), 走 `self.controller` 出控制。
- **非推理帧 (3/4 tick)**: `run_l1_step()` 用上一次推理锚定的 plan 出控制 (替代 L0 的"冻结上一条指令")。

L1 的物理意义也是 **τ_ego 注入的载体**: 计划变旧 → 控制应反映旧计划 (沿旧轨迹推进, 对新障碍反应迟)。

---

## §1 当前 V2 L1 的实际控制流 (实测代码)

```
推理帧 k:
  pred_waypoints (10,2 ego-local)  ← planner 输出, 内含避障+车道+转弯形状
  L1PlanStore.update():
    - waypoints 锚定到世界系 (供 L1 帧 world→local 回放)
    - ★ 冻结一个标量 desired_speed = mean(diff(‖wp‖)[:3]) × max_speed
  self.controller.run_step({waypoints=pred_waypoints, target=target_point})  → 出控制

非推理帧 k+1,k+2,k+3:  run_l1_step()
  steering ← V2X_Controller(weight=1) ⇒ 100% 由 fresh 全局 target_point 驱动
  speed    ← 合成直线 waypoints 编码 frozen desired_speed
  (★ 锚定的 world-frame 轨迹 + time_index 机制存在但未被 steering 使用)
```

---

## §2 五个架构问题 (逐条)

### ★问题1 (核心硬伤): 转向完全无视 planner 的避障波形

`V2X_Controller.run_step()` 里 `weight=1` ⇒ `angle = angle_wp×(1-weight) + angle_tg×weight = angle_tg`。
- **转向 = 100% 朝全局路由点 (`target_point`, 距车 10m) 走** = 纯路由跟随。
- planner 辛苦算出的 `pred_waypoints` (编码了避障、车道保持、转弯收弯) **除了被压成一个标量速度外, 转向上被完全丢弃**。

**后果**:
- **(A) 撞墙**: 弯道处全局 target_point 在前方 10m, 车直接切弯 → 撞 layout (这就是 DS=65 的 layout 碰撞, V1 和 te108+ 都是它)。
- **(B) τ_ego 平**: τ_ego 延迟的是 plan, 但转向根本不用 plan → τ_ego 动不了转向 → 撞墙类失败对 τ_ego 免疫 → 曲线平。

> 这一条同时解释了 **V1 撞墙** 和 **I-2 τ_ego 无信号** —— 同一个根。

### 问题2: desired_speed 是冻结标量, 不是速度剖面

`update()` 只冻结**一个**速度值 (前 3 个 waypoint 间距), 整个 200ms 推理间隔内不变。
- planner 的 waypoints 实际编码了**速度剖面** (可能计划前方减速避障)。压成单标量 → 丢失减速意图。
- 对 τ_ego: 延迟一个缓变标量速度效果微弱; 真正安全相关的信号 (为障碍刹车) 在剖面形状里, 已丢失。

### 问题3: 时间索引轨迹跟踪机制造好了却没用

`l1_trajectory_controller.py` 有 `time_index_ref()` + `get_local_ref()` + `l1_desired_speed()` —— 这套是"按经过时间索引锚定世界系轨迹"的**物理正确 L1**。
- 但 `run_l1_step()` 转向**没用它们**, 用的是合成 waypoints + 全局 target_point。
- 即"正确的 L1"建好了但被绕过。V2 退回"全局路由转向+冻结速度"是因为前几轮直接轨迹跟踪有不稳定 (速度尖峰/旋转 bug), 求稳但和 plan 解耦了。

### 问题4: 双控制器拆分 (Fix-D) 是绕过, 不是修复

Fix-D 给推理帧/L1帧建了独立 PID 实例, 避免 rate-mismatch 腐蚀 (stop_steps 4× 累积 → 弯道 forced_forward)。
- 修了 PID 状态腐蚀, 但没碰"转向走全局路由"的根本问题。
- 且推理帧/L1帧用**不同控制器** → 帧边界可能不连续。

### 问题5: 推理帧控制没被 τ_ego 延迟 (只 L1 帧延迟了)

I-2 只延迟了 L1PlanStore (作用于 L1 帧)。推理帧 (每 4 tick) 仍用 fresh pred_waypoints 立即出控制。
- 1/4 帧绕过 τ_ego → 稀释效果。
- 物理正确的 τ_ego 应延迟**感知输入到 planner**, 让推理帧的 waypoints 本身就是旧的。

---

## §3 根因总结

V2 L1 做了一个 **稳定性 vs 保真度** 的权衡: 为避开前几轮轨迹跟踪的不稳定 (速度尖峰、旋转 bug、PID rate mismatch), 它把**转向和 plan 解耦** (走全局路由) + **速度压成标量**。换来了稳定 (能跑完 route), 但:
- 切断了 感知/规划 → 转向 的联系 → 持续撞墙 (planner 避障被无视)。
- 让 τ_ego 无法作用于安全关键的转向路径 → τ_ego 曲线平。

---

## §4 修复需要解决什么 (不预设具体方案, 供 review)

一个真正的修复需同时满足:
1. **转向跟 planner 避障轨迹** (时间索引跟踪锚定轨迹), 而非只跟全局路由 → 避障生效 + τ_ego 能影响转向。
2. **速度跟轨迹的速度剖面** (时间索引), 而非冻结标量。
3. **整条推理链 (感知→规划) 被 τ_ego 延迟**, 让 planner 对旧世界做规划 → 晚避障 → τ_ego→安全 信号。
4. **但要避开 V2 当初退缩的不稳定**: 旋转 bug 已修 (§V0-6); 速度尖峰需用"锚定时刻冻结"或正确剖面处理; PID rate mismatch 需隔离控制器**或**改无状态转向律。

### 候选修法方向 (待用户拍板, 本文不预选)
- **方向A — 无状态纯追踪 (pure-pursuit) 跟踪锚定轨迹**: 每 tick 从当前 pose 对锚定的 planner 轨迹做纯追踪求转向 (无 PID 积分/微分状态)。天然: 用避障波形 + τ_ego 敏感 + 无 PID 状态腐蚀。≈ 用户之前 declined 的"无状态转向", 但现在有 τ_ego 无信号的实证支撑其必要性。
- **方向B — 启用已造好的 time_index 跟踪 + 调 weight**: 让 `run_l1_step` 真用 `get_local_ref`/`time_index_ref`, `V2X_Controller` weight 从 1 调到混合, 转向部分跟 planner 轨迹。改动小但仍带 PID 状态。
- **方向C — 深化 τ_ego 到感知输入**: 在 §问题5 基础上, 把 ego 自感知 (喂 planner 的 occupancy/box) 也延迟 Δ_ego。与 A/B 正交, 可叠加。

> **多路判别 (T1) 结果会影响优先级**: 若扩多路后聚合 DS 仍平 → 强证据指向"必须修 A/B (架构)"; 若交互路线 (S3/S4) 上 te219<te0 出现 → 说明问题主要在"r146 失败模式是撞墙(转向几何)", 换交互场景即可, 架构修可缓。
