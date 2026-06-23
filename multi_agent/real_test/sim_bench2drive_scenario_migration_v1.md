# Bench2Drive 交通场景迁移到 V2Xverse(CARLA 0.9.10) — 可行性与方案 (v1, 2026-06-11)

> 问题: 把 Bench2Drive(B2D) 丰富的交通场景"迁移"到当前 V2Xverse 闭环仿真(CARLA 0.9.10 + 老 Leaderboard 1.0)。
> 方法: 2 个并行 agent 调研 (B2D 外部仓库/论文 + V2Xverse 本地场景系统), 结论均带出处/文件路径。
> 一句话: **不能直接搬文件; 能且应该搬的是"场景分类学+对抗行为语义+参数", 而非坐标和传感器数据。且 V2Xverse 已自带 23 个场景类却只接了 7 个 → 最大的低成本收益是先"激活休眠场景"。**

---

## 一、两边系统对照 (事实)

| 维度 | Bench2Drive | V2Xverse (本项目) |
|------|-------------|-------------------|
| CARLA / Leaderboard | **0.9.15 / LB2.0**(硬绑, 依赖 Large-Map 流式引擎) | **0.9.10 / LB1.0**(已锁死, 不能升, 见 handoff §5.1) |
| 场景数 | **44 类**(LB2.0 的 39 个 NHTSA pre-crash + 5 自加), 220 routes(每类×5) | **7 类已接**(Scenario1/3/4/7/8/9/10), 但 **srunner 里有 23 个场景类** |
| 场景定义方式 | 嵌在 route XML 内 `<scenarios><scenario type= ><trigger_point/></scenario>` + 参数子元素 | route XML(纯 waypoints) **+ 独立** `townXX_all_scenarios.json`(scenario_type + trigger transform 列表) |
| 触发匹配 | route-percentage / trigger_point | trigger 与 route waypoint 距离<**2.0m** 且 yaw<**10°** 匹配(`route_parser.py`) |
| 城镇 | **Town12/13/15 为主**(0.9.15-only 大图) + 少量 01-10 | Town01-07 + Town10HD(0.9.10 全有) |
| 行为实现 | LB2.0 `scenario_runner` 场景类(假定 0.9.15 API/blueprint/actor-flow) | LB1.0 `srunner/scenarios/*.py`(0.9.10 atomic behaviors) |
| 选择/混比 | 路线内固定 | `scenario_parameter*.yaml` 按 proportion 混比(NUMBER_CLASS_TRANSLATION + ScenarioClassRegistry) |
| ego | 单车 | **多 ego + RSU 协作**(V2X 特有) |

## 二、两个硬阻塞 (为什么不能直接搬)

1. **城镇不存在(主阻塞)**: B2D 220 routes 绝大多数在 **Town12/13**(0.9.15 Large-Map, 0.9.10 没有这张图)。其绝对坐标(如 x=-497,y=3672,z=364, z≈364m 是大图分块)在 0.9.10 **无对应地图** → 无法回放, 必须**重锚定**到 0.9.10 的 Town01-07/10HD。
2. **场景类 API 代差**: B2D 的 LB2.0 场景类假定 0.9.15 API。**非 drop-in**, 需用 0.9.10 atomic behaviors 重写。其中一批是 0.9.15/大图专属, 0.9.10 **不可行**: `EnterActorFlow`/`MergerIntoSlowTraffic`/`HighwayCutIn`(依赖 LB2.0 background actor-flow + 高速大图)、`*TwoWays`(对向车流)、`Vanilla*` junction、`ParkingExit`(依赖大图停车位几何)。

**可迁移的价值** = 场景**语义/分类学/参数**(什么场景、什么对抗行为、什么难度), **不是** ① 传感器数据(渲染锁定 0.9.15 Town12/13, 跨 sim 无用) ② route 坐标(城镇锁定)。

## 三、★关键发现: V2Xverse 已自带大量休眠场景类

`simulation/scenario_runner/srunner/scenarios/` 已实现 **23 个类**, 但 `NUMBER_CLASS_TRANSLATION`(`route_scenario.py:61-72`)只接了 7 个。**已实现但未激活**的包括 (正好覆盖 B2D 一大片可移植类型):
- `CutIn`(→B2D StaticCutIn/ParkingCutIn/HighwayCutIn 的城市版)
- `ChangeLane`(→B2D LaneChange)
- `ConstructionSetupCrossing`(→B2D ConstructionObstacle 单车道版)
- `ManeuverOppositeDirection`(→B2D 对向接管/借道)
- `OtherLeadingVehicle`(→B2D 前车制动/慢车)
- `opposite_vehicle_taking_priority`(→B2D OppositeVehicleTakingPriority)
- `object_crash_two_vehicle` / `signalized_junction_{left,right}_turn`(→B2D 各路口左右转)
- `FollowLeadingVehicle(WithObstacle)`、`StationaryObjectCrossing`

⇒ **B2D 可移植类型的多数, 在 V2Xverse 里已有现成行为代码**, 只是没接进场景分类 + 没在 JSON 里布触发点。**最便宜的"迁移"= 把这些休眠类接上 + 布点 + YAML 启用**, 不写新行为。

## 四、推荐方案 (按性价比排序)

### Phase 0 — Inventory & gap map (0.5 day)
把 B2D 44 类逐一归入三桶: **(A) V2Xverse 已实现只需激活** / **(B) 可移植但缺类需新写**(ParkedObstacle、StaticCutIn 的纯静态版、Accident=静态障碍、HazardAtSideLane —— 都是"静态 actor + 变道判据"模式, 各 ~50-100 LOC) / **(C) 0.9.10 不可行**(highway actor-flow / 大图 / TwoWays —— 明确放弃并记录)。

### Phase 1 — 激活休眠场景 (★最高杠杆, 1-2 day)
- 把 §三 的已实现类接进 `NUMBER_CLASS_TRANSLATION` / `ScenarioClassRegistry`(给 ScenarioN 编号或走 parameterized 路径)。
- 在 Town03/04/05/06/10 的合适路口/路段**作者式布触发点**(写进 `townXX_all_scenarios.json`, transform=trigger pose), 复用已有 2m/10° 匹配。
- 用 `scenario_parameter*.yaml` 的 proportion 把新类混进来(已支持按比例混)。
- **不动 B2D 文件**, 借的是它的"类型清单 + 难度参数"。

### Phase 2 — 补可移植但缺的类 (B 桶, 2-3 day)
用 0.9.10 atomic behaviors 写少量新 `srunner/scenarios/` 类(ParkedObstacle/静态 Accident/StaticCutIn/HazardAtSideLane), 注册同上。

### Phase 3 — (可选) B2D route 子集自动转换工具 (1-2 day)
写解析器: `bench2drive220.xml` → 中性表 `(type, town, trigger_pose, params, waypoints)` → 只留 **Town01-07/10 + 可移植类型** → 重排成 V2Xverse route XML + scenarios JSON。产量有限(B2D 在共享城镇的 route 少), 但对该子集可自动化。Town12/13 部分只能人工重锚定(Phase 1 的布点即是)。

### 明确跳过
0.9.15-only 场景(highway/actor-flow/TwoWays/大图) + Town12/13/15 + B2D 渲染数据集。

## 五、与本项目目标的契合 (研究叙事加分)
- **V2X 价值最强的恰是 B2D 的遮挡类场景**: 路口横穿(Scenario7/8/9/10)、转弯遇横穿行人(Scenario4)、BlockedIntersection、对向占道 —— 这些 **ego 视野被遮挡、RSU 能补盲** 的场景正是协同感知的卖点。**优先移植遮挡/路口类**, 让"V2X 协同 → 难场景驾驶分提升"的因果在闭环可证。
- 与 τ_ego/τ_RSU 时延实验联动: 难场景(探针已在找 r146/r160)+ 多样场景库 = 时延×场景的二维 sweep 有了素材, 解决 `sim_ego_rsu_latency_strategy_v1.md` §6 "场景太易则信号测不出"的前提。
- 多 ego + RSU: 新场景布点须落在 `_cal_multi_routes` 能生成多车轨迹的路段; 触发点设计要考虑协作车/RSU 的相对位置。

## 六、诚实 caveat
1. "44 类全移植"不现实; 现实目标 = **把 V2Xverse 从 7 类活跃场景扩到 ~15-20 类**(已实现激活 + 少量新写), 覆盖 B2D 可移植子集, 重点遮挡/路口类。
2. Phase 1 的"布触发点"是**人工/半自动作者工作**(在 0.9.10 城镇挑路口、定 trigger pose), 不是跑个脚本就出 220 routes。
3. 每个新接的场景类要在 0.9.10 + 多 ego + RSU 下**实跑冒烟验证**(场景能触发、车能完赛或合理失败), 不能只接代码不验。
4. B2D 的 difficulty/weather 多样性(23 weather)可借鉴 → V2Xverse route XML 也支持 `<weather>` 元素(短 route 当前未用), 可低成本加天气多样性。

## 七、建议下一步 (待用户拍板)
- **Phase 1 激活 1-2 个休眠类**(如 CutIn/ConstructionSetupCrossing)在 Town05 布点冒烟 —— 一周内能让闭环场景多样性显著提升, 且全在已验证的 H800 闭环 smoke 之上增量。
- Phase 0 三桶对照表见 §八。

---

## 八、Phase 0 对照表 — B2D 类型 → V2Xverse 三桶 (★)

> 两边均已核实: B2D 取**实际用于 220 routes 的 21 个 type**(`bench2drive220.xml` 去重; 比 44 类全集更贴实战)。V2Xverse 取 `srunner/scenarios/` **18 个可用行为类**(已核类名), 状态分 **活**(已接 NUMBER_CLASS_TRANSLATION 且 town05 JSON 有触发点)/**接无点**(已接但缺触发点, 只需布点)/**休眠**(类已实现但未接 translation)。

### 桶 A — 已实现, 接上+布点即用 (12/21, 最高杠杆, 不写新行为)
| B2D type | NHTSA | V2Xverse 对应类 | 状态 | 工作量 |
|----------|-------|----------------|------|--------|
| VehicleTurningRoute | 路口 | VehicleTurningRoute | **活(S4)** | 已在用 |
| VanillaSignalizedTurnEncounterRedLight | 路口 | SignalJunctionCrossingRoute | **活(S7-9)** | 现有信号路口已覆盖红灯合规 |
| VanillaNonSignalizedTurn | 路口 | NoSignalJunctionCrossingRoute | **活(S10)** | 已覆盖 |
| ConstructionObstacle (base) | 障碍 | ConstructionSetupCrossing | 休眠 | 接 translation + 布点 |
| SignalizedJunctionRightTurn | 路口 | SignalizedJunctionRightTurn | 休眠 | 同名类, 接+布点 |
| OppositeVehicleRunningRedLight | 路口 | OppositeVehicleRunningRedLight | 休眠 | 同名类, 接+布点 |
| OppositeVehicleTakingPriority | 路口 | opposite_vehicle_taking_priority | 休眠 | 文件已存在, 接+布点 |
| NonSignalizedJunctionLeftTurn | 路口 | NoSignalJunctionCrossingRoute / SignalizedJunctionLeftTurn | 休眠/活 | 无信号左转, 参数适配 |
| NonSignalizedJunctionRightTurn | 路口 | NoSignalJunctionCrossingRoute / SignalizedJunctionRightTurn | 休眠/活 | 右转适配 |
| ParkingCutIn | 停车 | CutIn | 休眠 | adversary 起点设路边, 加速切入 |
| VehicleTurningRoutePedestrian | 路口 | DynamicObjectCrossing(S3,活) + VehicleTurning | 改 | 转弯路径布行人横穿 |
| VanillaNonSignalizedTurnEncounterStopsign | 路口 | NoSignalJunctionCrossingRoute + stop-sign 判据 | 改 | 0.9.10 有 stop sign, 加合规判据 |

### 桶 B — 可移植但缺类, 需新写小类 (4/21, 各 ~50-100 LOC, "静态 actor + 变道/等待判据"模式)
| B2D type | NHTSA | 最近的 V2Xverse 类 | 新写要点 |
|----------|-------|-------------------|---------|
| ParkedObstacle | 障碍 | StationaryObjectCrossing(可改) | 路边停车阻塞单车道 + 绕行判据 |
| Accident (base) | 障碍 | — | 静态撞车残骸阻塞 + 绕行 |
| HazardAtSideLane (base) | 障碍 | — | 慢速两轮车在车道边 + 安全超车判据 |
| BlockedIntersection | 路口 | — | 静止车堵路口出口 + 等待(不抢) |

### 桶 C — 0.9.10 不可行, 放弃 (5/21, LB2.0 actor-flow / 大图 / 对向车流机制专属)
| B2D type | NHTSA | 不可行原因 |
|----------|-------|-----------|
| AccidentTwoWays / ConstructionObstacleTwoWays / *TwoWays | 障碍 | TwoWays 对向车流机制 LB2.0 新增, 0.9.10 无 |
| EnterActorFlow (+V2/Interurban) | 路口 | 依赖 LB2.0 background actor-flow |
| MergerIntoSlowTraffic (+V2) | 高速 | 高速大图(Town12/13) + actor-flow |
| HighwayCutIn | 高速 | 高速大图; 城市切入由桶A的 CutIn 替代 |
| ParkingExit | 停车 | 依赖大图并排停车位几何 |

### 顺手白拿 — V2Xverse 现成但 B2D-220 未用的类 (额外多样性)
ControlLoss(活 S1) · DynamicObjectCrossing(活 S3) · FollowLeadingVehicle(接无点 S2) · OtherLeadingVehicle(接无点 S5) · ManeuverOppositeDirection(接无点 S6) · ChangeLane(休眠) · SignalizedJunctionLeftTurn(休眠) · VehicleTurningLeft/Right(休眠) · FollowLeadingVehicleWithObstacle(休眠) · StationaryObjectCrossing(休眠)

### 结论 (量化工作量)
- **桶A 12 类**: 多数只需 ① 接进 `NUMBER_CLASS_TRANSLATION`/registry ② 在 Town03/04/05/06/10 路口布 trigger 点(JSON) ③ YAML 启用 —— **零新行为代码**。
- **桶B 4 类**: 各 ~50-100 LOC 新 `srunner/scenarios/` 类。
- **桶C 5 类**: 明确放弃。
- **净效果**: V2Xverse 活跃场景从 **~7 类 → ~18-20 类**(≈3×), 全部 0.9.10 原生 + 多 ego/RSU 兼容。
- **V2X 优先级**: 桶A 的**路口/遮挡类**(VehicleTurning* / NonSignalized junctions / OppositeVehicle* / BlockedIntersection) 是 RSU 补盲价值最强处 → 先做这些, 让"协同→难场景驾驶分"可证。

---

## 九、B2D 评价指标迁移 (★ 已核 B2D 源码逐条定义)

> 源: `Thinklab-SJTU/Bench2Drive`(LB2.0/0.9.15) + NeurIPS24 paper(arXiv 2406.03877v3)。B2D 指标 = LB2.0 composed score + 2 个论文新增(Efficiency/Comfort)。下列**常数即载重值, 迁移时逐字照抄**。

### 9.1 关键结论 (一句话)
- **DS/RC/SR/Multi-Ability**: V2Xverse(LB1.0)**已产出 DS/RC/惩罚**(smoke 实测 score_composed=11.95)。差异仅在惩罚系数 dict + 场景分类, 要 B2D 可比就对齐系数。
- **Efficiency + Comfort**: V2Xverse **没有**, 但 **纯离线后处理、与 CARLA 版本无关、强可迁移** —— 这才是值得迁的两个新指标。

### 9.2 逐指标定义 + 迁移要点
**① DS / RC / 违规惩罚**(`statistics_manager.py`): DS=max(RC×∏penalty, 0); RC=完成里程%。固定惩罚乘子 `PENALTY_VALUE_DICT`: 撞行人0.50/撞车0.60/撞静物0.65/闯红灯0.70/违stop0.80/场景超时0.70/不让急救车0.70; 出车道按 `(1−pct/100)`; **min_speed 记录但不计分**。→ 0.9.10 可复现(RC+碰撞+信号几何), 但**违规事件探测器需按 0.9.10 场景重实现**; 惩罚 dict 照抄。

**② Success Rate**(`ability_benchmark.py`): 二值 = status∈{Completed,Perfect} **且** 除 min_speed 外所有违规列表为空(无 DS 阈值)。SR=成功 route/总 route。→ 同 ① 可迁。

**③ Efficiency**(`efficiency_smoothness_benchmark.py` + paper): `speed_pct = ego_speed / mean(附近车辆速度)`, 每完成 ~5% route(≈20 检查点)采一次, **滤掉 >1000%**(防除以极小值), route 内平均 → 最终=各 route 平均(**无 min_speed 记录的 route 跳过**)。→ 强可迁: 逐 checkpoint 记 ego 速度 + 半径内邻车速度算比值。**迁移建议: 每 5% 都记比值并全平均(比原版"无违规就跳过"更干净)**。

**④ Comfort/Smoothness**(`compute_comfort_metric`): 把逐帧序列切 **20 帧段**(不足 20 丢弃); 段内 **Savitzky-Golay 平滑(window=7, polyorder=2, dt=0.1s)**, jerk 用 savgol deriv=1; 检 6 个量是否全程在界:
| 量 | 界 |
|---|---|
| 纵向加速度 lon_acc=a·forward | [−4.05, 2.40] m/s² |
| 横向加速度 lat_acc=a·right | [−4.89, 4.89] m/s² |
| jerk 幅值(d‖a‖) | [−8.37, 8.37] m/s³ |
| 纵向 jerk(d lon_acc) | [−4.13, 4.13] m/s³ |
| yaw 加速度(savgol yaw rate) | [−1.93, 1.93] rad/s² |
| yaw rate(angular_velocity.z, 解缠) | [−0.95, 0.95] rad/s |
段内 6 量全在界=舒适; route comfort=舒适段/总段; 最终=route 平均。**仅用 ego 运动学**(accel xy + angular_velocity.z + forward/right 向量, 20Hz), 无邻车、无 0.9.15 API → **最干净可移植**。0.9.10 全有(`get_acceleration/get_angular_velocity/get_forward_vector/get_right_vector`); 须 **20Hz 记录**(dt=0.1 保 savgol 常数有效), 异率则调 time_interval。

**⑤ Multi-Ability**(`ability_benchmark.py`): 5 能力(Merging/Overtaking/Emergency_Brake/Give_Way/Traffic_Signs), 每 scenario type 映射到能力, 分项=成功/计数, 报 5 项均值。前 4 项=SR 按 type 分组; **Traffic_Signs 特殊**(纯无违规太易): 需 ①RC/100>junction_completion(过了路口, junction 位置由 GlobalRoutePlanner 实时算) ②无 stop 违规 ③无红灯违规。→ 可迁但需**自定义 0.9.10 场景→能力映射**; Traffic_Signs 的过路口判据可复用(GlobalRoutePlanner 0.9.10 有)。与 §八 场景扩充耦合(场景越全分项越有意义)。

### 9.3 落地方案 (最小工程)
1. **加 per-frame logger**: 在 `pnp_agent_e2e.run_step`(或 tick)每帧记 `metric_info.json` 行 = {acceleration(3D), angular_velocity(3D), forward_vector, right_vector, location, rotation} @20Hz + (Efficiency 用)每 5% checkpoint 的 ego 速度 + 半径内邻车速度。CARLA 0.9.10 全可取。
2. **加离线 metrics 脚本**: 抄 B2D `efficiency_smoothness_benchmark.py` 的 Comfort(savgol+6 阈值+20帧段) + Efficiency(速度比+5%采样+1000%滤), 直接套常数。
3. **DS/RC/SR 已有**(V2Xverse statistics_manager); 若要 B2D 可比, 对齐 `PENALTY_VALUE_DICT`。
4. **Multi-Ability**: 待 §八 场景扩充后, 定本项目 scenario→ability 映射再算。

### 9.4 对本项目的价值
- Efficiency/Comfort 给"V2X 协同 / 加速 → 驾驶质量"开**多维度**(不止安全 DS): 通行效率 + 平顺性。
- **Comfort 对 τ_ego 敏感**(时延↑→控制抖→jerk↑→comfort↓): 可能在 **DS 信号平**(`l1_architecture_problems_v1.md` 瓶颈)时, **comfort/efficiency 仍出 τ_ego 单调信号** → 潜在突破口。**建议补 τ_ego sweep 时一并记 comfort/efficiency**。
