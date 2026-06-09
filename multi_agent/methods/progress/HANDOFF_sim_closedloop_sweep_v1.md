# 交接: V2Xverse 闭环时延 sweep 实验 (新窗口 team-lead 按此执行)

> 写于 2026-06-09 (上一窗口主动收口, 用户准备新开窗口继续)。
> 背景: 本窗口从零搭起 **V2Xverse CARLA 闭环仿真**, 完成 Sim-A(环境+冒烟)/Sim-B(时延感知+算力隔离改造) 并 supervisor 级自核, **Sim-C 时延 sweep 正在 GPU0 跑 pilot**。用户当前主诉求 = **尽快用 GPU 0/1/2 跑完时延 sweep, 把闭环指标(驾驶分/碰撞率)入 dataset_v2**, 并要求 **supervisor 监督仿真 + data-orchestrator 随时接收结果入库**。
> ⚠️ 本窗口只拉了 `sim-integrator` 一个 agent(Agent 后台 subagent), 没拉全队 — 这是用户点名的疏漏。**新窗口第一件事 = 用 TeamCreate 拉全队**(见 §〇)。

---

## 〇、接手第 1 件事(按序)

1. **确认仿真进程是否还活着**(OS 级, 独立于窗口):
   ```bash
   nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | head -3
   ps aux | grep -E "world-port|closedloop_sweep|eval_driving_e2e" | grep -v grep | grep jichengzhi
   ls -d /home/jichengzhi/V2X/results/results_driving_lat_* 2>/dev/null   # 已完成几档
   ```
   - 上一窗口收口时: **GPU0 在跑 r0 pilot(CARLA world-port=40000), GPU1/2 空闲未扇出**(单 agent 串行, 扇出指令没轮到)。若进程已死/窗口关导致中断, 按 §二重起。

2. **用 TeamCreate 重拉团队**(旧 team `sw-hw-cooptim` 若有残留先 `pgrep -af sw-hw-cooptim` 清):
   - **sim-integrator**(本窗口新建, 角色定义 `.claude/agents/sim-integrator.md`): 接管/继续 Sim-C sweep。
   - **supervisor**: 监督仿真 — 30-45min/轮巡检(PID+log mtime+nvidia-smi), 死亡/完成即核验, 对 sim 自报"已跑/已出数"复跑核验(读 results.json + audit csv + git diff)。
   - **data-orchestrator**: 随时接收 sim-integrator 的 sweep 结果, 议定闭环指标 schema 并入 `dataset_v2`(Sim-D)。
   - (sw-optimizer / hw-optimizer 本任务非必需, 按需拉; hw 可在 P2 Orin 真隔离阶段接入。)
   - **用 TeamCreate 而非 Agent subagent** 的理由: 本窗口 sim-integrator 报告它环境里 **没有 SendMessage/TaskUpdate**(subagent 限制), 导致它无法把结果直接推给 data-orchestrator、无法回执。用户要的"data agent 随时接收 sim 结果"**必须靠 team 内消息机制** ⇒ 必须 TeamCreate。

3. **三 agent 首任务**(§三 有完整 spawn prompt):
   - sim-integrator: 扇出 GPU1/2 跑剩余 5 route(见 §二拓扑), pilot 完成先报 r0 信号。
   - supervisor: 立即起后台巡检, 第一轮先核 pilot 是否真在推进。
   - data-orchestrator: 先与 sim-integrator 议定 §四 的闭环指标 schema, 待首批结果就绪即入库。

---

## 一、本窗口已完成 (team-lead 已逐项盘面核验)

### 1.1 Sim-A 环境构建 + 冒烟 ✅ (Task #1 completed)
- conda env **`v2xverse`**(Python 3.7.16) + CARLA **0.9.10.1** + CoDriving 官方 ckpt 全装齐, `import torch/carla/spconv/opencood` 全通。
- **sm89 风险实证排除**: 4090 上 torch 1.10.1+cu113 无 sm89 cubin 但走 PTX-JIT 前向兼容(首次算子 1.6s 暖机, 稳态正常), **保官方栈 0 偏差**。
- 主要偏差 **D-3**: spconv 1.2.1 源码 build → **spconv-cu113 2.3.6 wheel**(opencood 自带 v2 兼容分支, voxel 输出已真验证)。其余偏差(下载手段/setuptools/numpy)无功能风险。详见设计文档 §1.5。
- **冒烟**(1 route town05_short r0, GPU2): status=Completed, DS 12.5/route 100%/3 次行人碰撞 — **只证链路通, 不入库**。
- ckpt(★现役): `checkpoints/codriving/perception/net_epoch_bestval_at16.pth`(加载 missing 2/unexpected 0) + `checkpoints/codriving/planner/codriving_planner.ckpt`。

### 1.2 Sim-B 三项改造 ✅ (Task #2 completed, 全 commit 到 V2Xverse `sim-closedloop` 分支)
- **时延感知**(C-3): 把 V2Xverse 原生静态 `comm_latency` 升级为 **动态实测时延驱动 + 零阶保持(ZOH)**。Δ=ceil(L_ms/50)(CARLA 20Hz=50ms/帧), 封顶 MAX_DELAY_FRAMES=10。ZOH 在**融合前、只作用 RSU/CAV 路**(ego 本车感知每帧实时)。6 字段审计 log(含 `latency_ms_source` + `zoh_age_frames`)。
- **算力隔离**(C-1): 推理拆到独立进程, CUDA-Event 只计 forward(口径纯净), 返回 `(perception_result, latency_ms)` 二元组。
- **C-4 双对照实测**: d0(Δ=0 实时 173帧)/d108(Δ=3 滞后+前3帧ZOH), 审计 log 帧对齐正确, `ceil(108/50)=3` 折算对, **单测 13 项全过**。两档都 100 分(易 route 机制验证, 非时延影响测量)。
- commit 链: `68196ae`(spconv兼容) → `9abe0dc`(C-3 aligner) → `de3b509`(C-1) → `a3bacec`(C-2/C-3接线)。

### 1.3 Sim-C 离线准备 ✅ (Task #3 准备阶段, commit 到 `sim-closedloop`)
- **C-5 无路侧对照开关**(`fa4e8e6`): `simulation.disable_rsu` flag(关 spawn_rsu + rsu_data=[], 退化 ego-only)。配置 `agent_config/pnp_config_codriving_norsu.yaml`。离线自检 `closedloop/test_disable_rsu.py` 全过。
  - ★为何不用"无限大 inject_ms"冒充无路侧: 无限延迟=有 RSU 但永远过时(仍融合陈旧 BEV), 关 RSU=根本没有 RSU(融合分支关闭)。用户要的是后者。
- **C-6 sweep 驱动 + 聚合脚本**(`45bf4e0`): `scripts/closedloop_sweep_v1.sh` + `scripts/closedloop_aggregate_v1.py`。dry-run(bash -n / py_compile / config 注入)全过; 聚合脚本对真实 d0/d108 干跑 delta_mean=0.0/3.0 精确匹配。
- **C-7 6-route 定稿**(`addfdbb` + V2X `6c00563`): 按 Town05 路口几何 + 触发密度选 6 条(见 §二)。
- **C-8 P1 单卡模式**(`ab53d68` + `ab26bfb`): `SIM_SINGLE_CARD=1` 绕过双卡 abort + 打 warning + 结果标 `isolation=single_card_shared`; L=0 自我覆盖 bug 已修。

---

## 二、Sim-C sweep — 正在跑 / 待扇出 (★核心进行项)

### 2.1 实验设计要点(完整见设计文档 §4)
- **核心假设**: 路侧 e2e 推理 < 200ms 时 V2X 对驾驶安全有可证提升, 超某临界时延后收益崩塌。产出 = "驾驶分 vs 时延"曲线 + 拐点定位。
- **8 档(加密找拐点)**: 注入档 ms = {0, 50, 108(最优链), 150, 219(FP32 baseline), 300, 500} + 第8档"无路侧"(disable_rsu)。对应 Δ帧 = {0,1,3,3,5,6,10}。低端密(打 200ms 阈值), 高端疏。
- **★4090 注入档(用户 2026-06-09 追加, 设计文档 §4.A 末)**: 现有 8 档物理依据是 **Orin 实测**(边缘设备)。追加 **4090 档**(datacenter 级 RSU 对照): 4090 e2e ~27-35ms(M4.6.0, OPV2V/autocast caveat)**< 50ms 一帧 ⇒ Δ=1 帧**(亚帧延迟, "近乎无感")。与 Orin(Δ=3-5)成"强力 RSU vs 边缘 RSU"对照。**帧对齐上等价 50ms 档**, 但作 4090-出处标注点有价值。**待办**: 让 hw-optimizer 用 4090 TRT 引擎测一个 DAIR 同口径(pre-body+body+NMS)的干净 4090 e2e, 替换 OPV2V autocast 值; 无论口径 Δ=1 稳。`latency_ms_source=measured_on_4090_*`。
- **6 route**(对抗性, 含行人横穿/车辆切入/遮挡): `r0`(冒烟已验) / `r146`(最密路口) / `r28`(直道触发密度最高) / `r160`(最长221m+最高密度) / `r141`(90°转弯逼近路口) / `r135`(最急转向盲区)。
- **统计(诚实)**: 种子写死(CARLA/Traffic=2000)⇒ 确定性, **不做同-route 重复**(假0方差), 改 **6-route 间聚合**拿 mean±std; 仅 1 个组合重跑 1 次作确定性自检。

### 2.2 ★时延口径 — 关键诚实标注 (论文/入库必带)
- **时延是注入档(`latency_ms_source=injected_from_E7`), 不是现场实测**: 模型用 V2Xverse 原生 CoDriving 感知(保 CARLA 数据 AP 有效), 时延数字来自团队 Orin E7 真测(FP32 131/FP16 48/INT8+剪枝 20ms; e2e 合成 219→108ms)。
  - 原因: 团队加速 Pyramid(HEAL/DAIR 训练)与 V2Xverse CoDriving(CARLA 数据训练)**架构+数据集双重不匹配, 不能换权重**(硬塞=跨数据集脏推理, 项目纪律禁止, ISS-012 同类)。
- **单卡共享(`isolation=single_card_shared`)对 sweep 有效性无影响**: CARLA 同步模式下感知结果确定、控制逐位一致、Δ 由注入值算而非墙钟 ⇒ driving score 与分卡相同, 仅墙钟更慢。算力隔离保护的是"现场实测时延纯净度", 注入档不需要它。
- **P2(真隔离, 后续)**: CARLA 在 4090 / 推理在 Orin 边缘设备, 补一个**现场实测真实边缘时延**锚点 + 真物理隔离, 标 `separate_gpu` / `measured_on_orin`。是用户"一张卡纯做推理"设想的真正落地形态。需把 CoDriving 移植 Orin + 网络 IPC(工程量大, 未启)。

### 2.3 三卡并行拓扑(用户已放行 GPU 0/1/2, "尽快开始实测")
单卡注入档实例彼此独立 ⇒ 3 卡并行 3 实例, 墙钟 ~9h → ~3h。端口/TM 端口已隔开。

| GPU | 端口(TM) | route | 起跑命令 |
|-----|---------|-------|---------|
| 0 | 40000(40005) | **r0**(pilot, 上窗口已起) | `SIM_SINGLE_CARD=1 bash scripts/closedloop_sweep_v1.sh 0 0 40000`(ROUTES 临时只 r0) |
| 1 | 41000(41005) | **r146, r160** | 同上, `... 1 1 41000`, ROUTES 覆盖 r146 r160 |
| 2 | 42000(42005) | **r28, r141, r135** | 同上, `... 2 2 42000`, ROUTES 覆盖 r28 r141 r135 |

- 临时改 ROUTES **别动脚本 C-7 定义**; 加 `ROUTES_OVERRIDE` 环境变量读取(小改顺手 commit)更干净。
- **单卡模式 = `SIM_SINGLE_CARD=1`**(否则脚本对 render==infer 直接 abort)。
- 起跑前各 GPU `nvidia-smi` 自查空闲(脚本 gpu_guard 内建 util>5%/mem>50MiB 即 abort), 发**宣告**(授权引用+三卡快照+三命令+ETA)给 team-lead+supervisor。
- 结果落 `results/results_driving_lat_d{ms}/.../r{rid}_repeat0/ego_vehicle_0/results.json` + 配对 audit csv; 全跑完聚合 → `results/closedloop_sweep_v1.csv`。

### 2.3.1 ★r0 PILOT 已完成 (2026-06-09, GPU0 单卡) — 结果 + 研判
> 结果落 **`/home/jichengzhi/V2Xverse/results/`**(脚本从 V2Xverse repo 跑, 相对 results/ 在那边, **不是 V2X/results/**): `results_driving_lat_{d0,d50,d108,d150,d219,d300,d500,norsu}/` + 聚合 `closedloop_sweep_v1.csv`。team-lead 已读 raw json + CSV 核验。
> 新增 commit: `ab53d68`(C-8 单卡)`ab26bfb`(C-8.1 config L=0 自清空 bug 修)`0b2a2fd`(C-9 并行安全多实例, per-PORT 隔离 — 新窗口 3 卡并行直接用)。

- **机制层 ✅ 全验证**: delta_mean 精确 = ceil(L/50)(d0/50/108/150/219/300/500 → Δ 0/1/3/3/5/6/10), zoh_held_frac 随时延单调升。注入/帧对齐/ZOH 全对, **非 bug**。
- **信号层 ❌ 命中 §4.F 证伪条件 1**: **r0 八档 DS 全 = 100.0 / 碰撞全 0 / route 全 100**, 连 **norsu(关路侧)也 = 100**。⇒ r0 对时延、甚至对"有无路侧"完全不敏感。
- **单卡墙钟**: 新跑单档 ~5.6min(dur_system ~335s)+ 启停 ~0.5min ≈ **6.3min/档**; 全新 8 档单 route ≈ **~50min**(单卡)。剩余 5 route 单卡串行 ~4.2h / 3 卡并行 ~1.4h。
- **⚠️ team-lead 标注的待查疑点(比"r0 太易"更深一层)**: Sim-A **原版冒烟** r0 撞 3 次行人(DS 12.5), 而改造版(C-4 起)全程 0 碰撞/DS 100。两解释: (善意)driving config 不同; (需警惕)**改造后 ego 靠自车感知避开一切 ⇒ 路侧感知可能未接入控制决策**。`norsu=100` 必须分清是"场景太易路侧冗余"还是"路侧根本不影响驾驶"——**后者则换多难场景都测不出 V2X 价值**。
- **★下一步(team-lead 研判, 优先级排序)**:
  1. **[P0 诊断, 先做] 验证"路侧感知是否真接入 ego 控制"**: 不是堆场景, 而是先确认 RSU/CAV 融合结果是否实际改变 control 输出。最小验证: 同一难场景跑 norsu vs d0, 若 DS/轨迹**完全一致** ⇒ 路侧未接入控制(架构问题), 换场景无用, 须查 CoDriving 的 planner 是否消费融合 BEV。顺带查清 smoke(12.5)vs 改造(100)差异根因。
  2. **[P0 信号探针] 跑 r146 + r160**(密路口/221m 长程, 含遮挡): 若它们出现 DS 随时延下降 ⇒ r0 确属"太易", 信号在难场景出现, 继续全量。
  3. **[P1 条件触发] 若 r146/r160 仍全平 + 诊断1 显示路侧已接入控制**: 触发 §4.F 场景加难(DynamicObjectCrossing proportion↑ / pedestrian_amount 60→120 / CRAZY_LEVEL↑ / 换 long route 强遮挡)。
  - **不要直接无脑铺剩余 5 route 全量** —— pilot 已证现成场景可能全平, 先诊断+探针, 再决定全量或加难。

### 2.4 汇报节奏(给 sim-integrator 的硬要求)
- ①**r0 pilot 完成 → 立即单独报 r0 八档信号**(0ms vs 500ms DS 差多少 = §4.F 证伪条件1 早期检测), 别等其他 route。
- ②**全 6×8=48 跑完 → 报完整聚合表**(DS/碰撞率分类/zoh_age 均值 vs 时延) + 单卡墙钟实测 + 三卡并行加速比 + 异常。
- ③**信号判定**: r0 有信号 → 继续/已在跑全量; r0 无信号(0 vs 500 差距 < route间std)→ 触发 §4.F 换难场景(调高 scenario_parameter 的 DynamicObjectCrossing proportion + pedestrian 60→120 + CRAZY_LEVEL, 或换 long route)。

---

## 三、团队 spawn prompt (新窗口 TeamCreate 后逐个派)

> 三 agent 角色定义已在 `.claude/agents/`(sim-integrator/supervisor/data-orchestrator)。每个 spawn prompt 首行必含 **MUST-6 + 启动回执协议**(实验启动消息逐字引用 team-lead 授权原文; 验收≠授权)。

### sim-integrator(接管 Sim-C)
"你是 sim-integrator(续上窗口: 已完成 Sim-A/B, Sim-C 离线准备全 commit 到 V2Xverse sim-closedloop 分支, pilot 在 GPU0 跑)。先读 `.claude/agents/sim-integrator.md` + 设计文档 `multi_agent/real_test/sim_test_design_v1.md` §3-4(整合后主文档; 旧 sim_closedloop_arch 已归档 archive/) + 本交接 §二。**team-lead 授权**(用户原话): '好了现在你可以使用GPU0,1,2 尽快开始实测实验'。任务: 按 §2.3 拓扑三卡并行扇出跑完 6route×8档, 按 §2.4 节奏汇报。起跑发宣告。单卡模式 SIM_SINGLE_CARD=1, 口径标 injected_from_E7/single_card_shared。结果就绪即推给 data-orchestrator。"

### supervisor(监督仿真)
"你是 supervisor。先读 `.claude/agents/supervisor.md` + 团队宪章 `multi_agent/methods/progress/team_charter_v1.md`(§4 后台巡检官职责)+ 本交接。任务: ①立即起后台巡检(有实验在跑 30-45min/轮: PID+log mtime+nvidia-smi, 死亡/完成/log停滞>20min 即预警+核验)②对 sim-integrator 一切'已跑/已出数'自报**复跑核验**(读 results.json + audit csv + git diff, 历史抓过 dry-run 冒充实测)③核验时延口径标注是否诚实(injected_from_E7 不能写成现场实测; single_card_shared 不能漏)④问题入 `issues_log_v1.md`。第一轮先核 pilot(GPU0 port40000)是否真在推进。"

### data-orchestrator(接收结果入库 = Sim-D)
"你是 data-orchestrator。先读 `.claude/agents/data-orchestrator.md` + `multi_agent/data/schema_v2.md` + 本交接 §四。任务: ①先与 sim-integrator 议定闭环指标入 dataset_v2 的 schema(新增列, **不擅改主表现有列**)②sweep 首批结果就绪即按 (route,arm) 聚合入 `multi_agent/data/dataset_v2.{csv,parquet}` + 更新 schema_v2.md ③每行带口径元数据(injected_from_E7/single_card_shared/确定性单次)④数据合理性检查(DS∈[0,100]/碰撞率非负/zoh_age 随时延单调)发现异常打回 sim-integrator。Phase H 冻结数据不碰。"

---

## 四、Sim-D 闭环指标入库 (data-orchestrator 主理, Task #4 pending)

- **数据源**: `results/closedloop_sweep_v1.csv`(sweep 聚合输出)。
- **建议新增列**(与 dataset_v2 现有 5 指标 {AP,latency,throughput,energy,model_size} 并列, 议定后定): `closedloop_driving_score`, `collision_rate_pedestrian/vehicle/layout`, `route_completion`, `zoh_age_frames_mean`, `latency_inject_ms`, `delta_frames`。
- **元数据列**: `latency_ms_source`(injected_from_E7), `isolation`(single_card_shared), `sim_route_set`(6route id), `n_repeat`(=1 确定性)。
- **schema 纪律**: 闭环指标与现有感知指标(AP/AMOTA)**不同任务口径, 绝不混入同列/同曲线**; 新增列须 schema_v2.md 显式登记。

---

## 五、纪律 / caveat (本窗口适用, 全部沿用宪章)

1. **真测就是真测**: 闭环指标必来自真实 CARLA 闭环运行; 拿不到说"未实测"。sim 自报 supervisor 必复跑核验。
2. **GPU 空闲才跑**: util>5%/mem>50MiB 即 abort(脚本 gpu_guard 内建); GPU 常被 wuyuegao 占, 跑前必查; 用完清理并复核释放。
3. **宣告协议**: 任何 GPU 实验启动前发宣告(授权原文+设备快照+ETA), 确认送达(宣告本质是送达不是书写)。
4. **时延口径三标注**(§2.2): injected_from_E7 / single_card_shared / 确定性单次 — 论文与入库一字不可漏。
5. **teardown 挂死坑**(Sim-A §2.5 B-3): route 跑完 CARLA teardown 挂死、顶层 results.json 不 flush — **权威结果在 `ego_vehicle_0/results.json`**; 每 route 后 kill -9 + pkill + sleep 5(脚本已内建)。
6. **subagent 唤醒漏消息 bug**(历史 3 次): 派单后 10min 内要回执, 超时查岗(查岗=二次唤醒); 别攒多任务在一条消息连发。
7. Phase H 冻结数据(H1/H2)不入库不出图不引用。

---

## 六、关键路径速查

| 资源 | 路径 |
|------|------|
| **真仓库** | `/home/jichengzhi/V2X`(UniV2X 是断裂符号链接空壳, 不读不写) |
| V2Xverse repo | `/home/jichengzhi/V2Xverse`(分支 **sim-closedloop**, 不污染 main) |
| conda env | `/home/jichengzhi/miniconda3/envs/v2xverse`(Python 3.7) |
| 设计文档(主) | `multi_agent/real_test/sim_test_design_v1.md`(整合后唯一权威: §2 感知测试/§3 算力分离+时延注入/§4 初始验证实验/§5 入库)。★旧 `methods/design/sim_closedloop_arch_v1.md` 已整合并归档到 `multi_agent/archive/` |
| sweep 脚本 | `V2Xverse/scripts/closedloop_sweep_v1.sh` + `closedloop_aggregate_v1.py` |
| 改造代码 | `V2Xverse/simulation/leaderboard/team_code/closedloop/`(infer_server/latency_frame_align/test_*) + `pnp_infer_action_e2e.py` + `pnp_agent_e2e.py` |
| 结果落点 | `V2X/results/results_driving_lat_d*/` + `closedloop_sweep_v1.csv` + `sweep_failures.log` |
| 数据主表 | `multi_agent/data/dataset_v2.{csv,parquet}` + `schema_v2.md` |
| Orin(P2 用) | `172.16.62.222` 用户 jichengzhi(密码每会话向用户确认), TRT 8.5.2.2 |

## 七、用户桌上待决 / 后续

- **P2 Orin 真隔离**: 用户设想"一张卡纯做推理" = CARLA 在 4090 / 推理在 Orin。补现场实测边缘时延锚点。等 sweep 出曲线后 + Orin 时间到位再立项。
- **更多 GPU**: 用户说"协调出更多卡会告诉你" — 若得 6 卡可 2 实例/route 或并行翻倍。
- **闭环故事对照量级**: CoDriving 论文 +62.49%DS / −53.50%collision(协同 vs 单车) — 我们 sweep 的"无路侧档"vs"0ms档"应复现这个量级方向。
- **TaskList**: #1 Sim-A done / #2 Sim-B done / #3 Sim-C in_progress(sweep 跑中)/ #4 Sim-D pending(data-orchestrator 接)。
