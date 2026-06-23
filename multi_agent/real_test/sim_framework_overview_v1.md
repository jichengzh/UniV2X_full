# V2Xverse 闭环测试框架总览 (v1, 2026-06-11)

> 用途: 一页看清当前闭环仿真测试框架 —— 运行架构 / 覆盖场景 / 评价指标 / 路线清单(按场景分类)。
> 全部基于 H800 实测 + 主控独立复核。诚实区分 **真执行** / **机制已通待布点**。
> 关联: 迁移分析 `sim_bench2drive_scenario_migration_v1.md`, 架构 `sim_h800_ipc_split_v1.md`, 进度 `PROGRESS_h800_sim_smoke_v1.md`。

---

## 一、运行架构 (H800 双进程 IPC)
- **进程 A**(py3.7): CARLA 0.9.10 + leaderboard + pnp_agent(传感/控制 I/O)。**进程 B**(torch2/py3.10): codriving 感知整脑(感知+BEV+planner+控制) on GPU。A↔B 经 zmq+msgpack_numpy。
- **算力隔离**: CARLA 一张卡 / 感知 process B 另一张卡 / 互不抢。IPC 实测 **4600+ 帧 @ ~130ms 稳定**。
- 感知=CenterPoint+多尺度注意力融合; planner=CoDriving WaypointPlanner(MotionNet); 控制=V2X_Controller(PID)+L1 轨迹保持。

## 二、覆盖的场景

> 场景类资产: V2Xverse `srunner/scenarios/` 共 **18 个可用行为类**; `NUMBER_CLASS_TRANSLATION` 现接 **Scenario1-17**(本会话新接 12-17)。**两个场景定义文件**: `town05_all_scenarios.json`(大, 含 S1/3/4/7/8/9/10) 与 `town05_all_scenarios_2.json`(eval 默认, 含 S1/3/4 + 本会话新加 S11-17)。

### 真执行 (行为已实测验证) — 9 类
| Scenario | 行为类 | 行为 | V2X 价值 | 验证 |
|---|---|---|---|---|
| S1 | ControlLoss | 低摩擦/扰动, ego 恢复 | 低 | 104/105 路线 |
| S3 | DynamicObjectCrossing | 行人/骑车横穿(常从遮挡处) | **高(遮挡)** | 104/105 路线 |
| S4 | VehicleTurningRoute | 路口横穿车 | **高(路口)** | 80/105 路线 |
| S7/8/9 | SignalJunctionCrossingRoute | 信号路口让行(左/右/对向) | **高** | 在 all_scenarios.json |
| S10 | NoSignalJunctionCrossingRoute | 无信号路口协商 | **高** | 在 all_scenarios.json |
| S11 | ConstructionSetupCrossing | 施工锥/警示牌阻挡, 绕行 | 中 | ✅本会话: ego 撞锥, RC=100 |
| S13 | ManeuverOppositeDirection | 对向车逼 ego 借道 | 中 | ✅本会话: 对向车堵 ego 480s |

### 机制已通 / 待几何匹配布点 (wired+触发匹配, 但 r146 几何不符→FreeRide/no-op) — 5 类
| Scenario | 行为类 | 卡在哪 | 激活条件 |
|---|---|---|---|
| S12 | OtherLeadingVehicle | r146 单车道 | 需多车道路线 |
| S14 | OppositeVehicleRunningRedLight | r146 无红绿灯 | 需信号路口触发点 |
| S15 | SignalizedJunctionLeftTurn | 同上 | 需信号路口 |
| S16 | SignalizedJunctionRightTurn | 同上 | 需信号路口 |
| S17 | CutIn | r146 无右车道 | 需多车道路线 |

> 结论: **激活机制完全打通**(6 类全注册+触发匹配+实例化确认, 零行为代码)。瓶颈=**把触发点放到几何匹配的路线/路段**(找 Town05 红绿灯+多车道段), 是 authoring 工作非代码问题。

## 三、评价指标
| 指标 | 状态 | 定义 / 实测 |
|---|---|---|
| **Driving Score (DS)** | ✅ 已有 | `RC × ∏违规惩罚`; 违规乘子 撞行人.50/撞车.60/撞静物.65/红灯.70/stop.80 |
| **Route Completion (RC)** | ✅ 已有 | 完成里程% |
| **Success Rate** | ✅ 可算 | 完赛 且 无(除min_speed外)违规 |
| **Efficiency 效率** | ✅ 本会话新增 | ego速度/邻车均速, 每帧采样滤>1000%; **r146 实测 53.86%** |
| **Comfort 舒适性** | ✅ 本会话新增 | 20帧段×6运动学量阈值(lon/lat acc, jerk, yaw); **r146 实测 0.034** |
| **Multi-Ability** | ⚠️ 框架就绪待映射 | 5 能力(Merging/Overtaking/EmergencyBrake/GiveWay/TrafficSign)分项成功率; 需定 scenario→ability 映射 |

> 实现: per-frame logger(`pnp_agent_e2e.py`, `METRIC_LOG=1` 门控) + 离线 `bench2drive_metrics.py`(B2D 常数逐字)。**Comfort 对 τ_ego 敏感**(jerk 失败主因=IPC 时延致指令不连续)→ 可能在 DS 平时仍出 τ_ego 信号。

## 四、路线清单 (evaluation_routes)
- **总数: 105 条, 全部 Town05**(`town05_short_r{0..N}.xml`)。另有 `additional_routes`(Town01-06)/`training_routes`/`42routes`/`official` 未计入标准 eval。
- **按场景分类**(RouteParser 2m/10° 匹配 `town05_all_scenarios_2.json`; 每条路线可叠加多场景):

| 场景类 | 含该场景的路线数 (/105) |
|---|---|
| ControlLoss (S1) | 104 |
| DynamicObjectCrossing (S3) | 104 |
| VehicleTurningRoute (S4) | 80 |
| ConstructionSetupCrossing (S11) | 5 |
| OtherLeadingVehicle (S12) | 5 |
| ManeuverOppositeDirection (S13) | 5 |
| OppositeVehicleRunningRedLight (S14) | 5 |
| SignalizedJunctionLeftTurn (S15) | 5 |
| SignalizedJunctionRightTurn (S16) | 5 |
| CutIn (S17) | 5 |
| 无场景(纯自由驾驶) | 1 条 (r112) |

- 典型路线 = **ControlLoss + DynamicObjectCrossing 近乎必含**, 视几何叠加 VehicleTurning / junction 类。
- 本会话新增的 7 类各落在 **5 条**经过 r146 走廊(x≈34.5, y∈[-110,-165])的路线上 —— 说明 **town-wide 触发点会自动套用到所有经过该处的路线**(布点一次, 多路线受益)。
- ⚠️ 标准 eval 用 `_2.json` → 不含 S7-10 junction-crossing(那在大 `town05_all_scenarios.json`); 跨 town 多样性需启用 `additional_routes`。

## 五、当前能力边界 / 下一步
1. **真执行场景 9 类**(S1/3/4/7/8/9/10/11/13), 覆盖 control-loss / 行人横穿 / 路口 / 施工 / 对向借道。
2. **待激活 5 类**(S12/14/15/16/17): 需在 Town05 找红绿灯+多车道段布触发点 (找 `map.get_traffic_lights()` 附近 waypoint)。
3. **指标**: DS/RC/Efficiency/Comfort 可用; Multi-Ability 待定能力映射。
4. **路线多样性**: 105 全 Town05; 跨 town 需启用 additional_routes(Town01-06)。
5. **V2X 叙事**: 遮挡/路口类(S3/4/7/8/9/10/14)是 RSU 补盲价值最强处 — 优先在这些上做协同 vs 无协同对比。
