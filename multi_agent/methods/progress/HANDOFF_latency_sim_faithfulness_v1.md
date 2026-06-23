# 交接: 时延感知闭环仿真 — 时延危害随 ego 速度放大 (开放问题已闭合) (HANDOFF_latency_sim_faithfulness_v1)

> 写于 2026-06-18。承接 [[HANDOFF_tau_perc_curve_v1]] + memory [[project-tau-perc-signal]]。
> ★★[2026-06-18 开放问题已闭合] 原疑问"RSU/感知时延实测影响小, 仿真是否真实模拟危害?"已由**提速验证实验**回答: **时延危害随 ego 速度强烈放大 → 注入机制忠实, 此前低速场景系统性低估了高动态下的时延危害**(保真度缺口在场景动态覆盖, 非注入 bug)。详见 §2.★闭合 + §6 执行记录。图 `figure/data/percinj_fastslow_compare.png`。

---

## §0 一句话: 推理时延插入流程
首先 AV 传感器在第 k 帧采集感知数据, 之后对该帧整体输入插入 Δ=⌈τ/50ms⌉ 帧固定时延(帧库缓存 + 零阶保持 ZOH), 在这 Δ 帧时延持续过程中车辆在仿真世界中按已有决策持续前进, 而感知→融合→规划→控制全链路只能基于第 k−Δ 帧的旧世界快照决策, 直到 Δ 帧后第 k 帧数据才生效 ——「车在走, 脑中世界停在 τ ms 前」。

## §1 已 settle 的结论(别重测)

### A. 单 ego 感知时延 (τ_perc, RSU=0) — 18 点诚实曲线
- 注入点 = ego 自感知输入 `car_data_raw[0]`(LiDAR+pose), 帧库+ZOH+深拷贝, Δ=⌈τ/50⌉ cap=20帧。代码 `pnp_infer_action_e2e.py` 704–718 行。
- 诚实口径(TIMEOUT_SKIP/无score 记 DS=0), 35/47 路 tp0-clean matched。
- **三区: ① tp0-500 噪声平台(DS~87-98, 灾难≈0); ② 硬悬崖 600→650(DS 82→68, 碰撞 0.19→0.38 翻倍)= 临界阈值; ③ 650-800 第二平台(~67) → 800-1000 续降(DS→48, 灾难30%)**。图 `figure/data/percinj_curve_full.png` + `percinj_curve_final.png`(三色分区)。
- 关键修正: 灾难率有 ~5-8% 随机 ambient 噪声地板(N=9 验证); 延迟轴物理量化到 50ms(CARLA 20Hz, 无亚帧)。

### B. ego×RSU 耦合 (τ_RSU=τ_ego+τ_comm, β=1) — RSU 边际效应≈0
- 物理: ego/RSU 同款加速网络→计算时延一起降(β=1); RSU 多通信 τ_comm。两旋钮独立: `tau_perc_ms`(ego) + `latency_inject_ms`(RSU, RsuLatencyAligner)。代码互不重叠(ego=i0, RSU=i>0+rsu_data)。
- **comm=50ms 固定, 扫 τ_ego∈{0,200,...,1000}(10点) vs ego-only(18点), 24路严格matched: 两条线基本重合, ΔDS 均值≈+0.3 range±5.5 全在CI内, 无系统方向**。图 `figure/data/percinj_c50_compare.png`。
- ⇒ **τ_ego 主导驾驶分; +50ms RSU 时延的边际效应不可分辨**。之前 5 点的"-8 一致下降"被 10 点严格 matched 推翻(路集+N=3 伪信号)。
- caveat: 仅 comm=50ms(1帧); 大 comm(100/200/400, 16点网格停在 percinj_grid 639/2256)未结论。

## §2 ★开放问题: 为什么 RSU/感知时延影响小? → ★已闭合(提速验证)

### ★★[2026-06-18 闭合 — 提速验证实验定论]
用户质疑"低速实测时延几乎无效, 仿真是否真实模拟危害"。**提 ego 速度上限(max_speed 5→8 m/s, L1控制器 desired_speed=mean(diff(wp))×max_speed, max_speed实为waypoint时间步倒数1/0.2=5; 提它=沿同轨迹跑更快=每帧位移变大)后重测 τ_perc 曲线, 与低速版同 29 条交集路对比(诚实DS, 超时=0)**:

| τ_perc | 提速DS(spd8,Eff~150%) | 低速DS(spd5,Eff~90%) | gap(低-提速) |
|---|---|---|---|
| 0 | 99.5 | 99.4 | -0.1 |
| 100 | 94.0 | 96.0 | +2.0 |
| 200 | 91.5 | 94.1 | +2.6 |
| 300 | 83.6 | 95.5 | +11.9 |
| 400 | 82.7 | 95.0 | +12.3 |
| 500 | 60.3(cat10%) | 89.8 | +29.5 |
| 600 | 64.0 | 84.9 | +20.9 |
| 800 | 55.9 | 69.7 | +13.8 |

- **ΔDS(0→600): 提速 -35.5 vs 低速 -14.5 (~2.4×)**。提速臂悬崖从低速的 ~600ms **前移到 ~400-500ms** 且更陡(τ500 崩 60.3+10%真灾难, 低速同点还89.8); gap τ500 达峰+29.5; τ800 收窄(低速臂自身悬崖追上)。
- **结论**: **时延危害随 ego 速度强烈放大 → 注入机制忠实(代码层 D2/D2b 已证陈旧确实改感知); 此前低速场景(ego≈邻车半速, 每帧位移小)系统性低估高动态时延危害 = 场景动态覆盖缺口, 非注入 bug**。这直接回答了用户的保真度质疑: **仿真是对的, 只是温和场景没把危害放出来**。
- 物理: 同 τ_ms = 同 Δ=⌈τ/50⌉ 帧陈旧(与速度无关), 但高速下同帧数 = 更多米空间陈旧 → 更伤前向在途目标决策。
- 数据/产物: `results_driving_percinjfast_tpfast{0,100,200,300,400,500,600,800}_r*_n*`(各141, N=3); 图+csv `figure/data/percinj_fastslow_compare.{png,csv}`; 脚本 `real_test/percinj_fastslow_final.py`。配置 `pnp_config_codriving_tpfast{τ}_l1.yaml`(max_speed=8)。
- caveat: 交集29路是两臂 τ=0 都干净的子集(去路集难度confound, 必须用交集口径——直接比会被"高速下只有易路线干净"反向误导, 见§6执行记录); 高τ档(500/800)灾难率噪声~5-10%。

---

### [历史] 原开放问题(RSU 时延小)与诊断链 — 已被上面闭合涵盖
用户质疑: 原理上 RSU 陈旧必伤融合, 实测却几乎无效 → 仿真是否真实模拟了 RSU 陈旧危害?

### 候选根因(按优先级排查)
1. **★最大嫌疑: 通信门控退化**。迁移文档 `sim_h800_ipc_split_v1.md` 记: codriving ckpt 有 2 个 missing key `fusion_net.naive_communication.gaussian_filter.{w,b}` (4090 也缺, strict=False 随机初始化)。**若这层在推理路径上 → 通信门控被随机权重破坏 → RSU 融合可能退化/被忽略 → RSU 时延自然无效**。**第一步必查: 这层是否在 forward 路径? 是否真把 RSU 特征融进 BEV?**
2. **RSU 在 47 路上是否在场/有贡献**: 若多数 route 没 spawn RSU 或 RSU 视野与 ego 高度冗余 → 删/延迟 RSU 都无感。查 fusion 输入 batch 里 RSU 分支占比 + 融合后 BEV 对 RSU 的敏感度。
3. **低速场景下陈旧≈不陈旧(保真度核心)**: Town05 低速, 1-2 帧(50-100ms)世界位移极小 → ZOH 取的旧 RSU 帧与当前帧几乎一样 → 融合输出几乎不变。**真实高速/路口快速横穿时, 100ms 陈旧 = 数米误差 = 大危害; 本仿真场景可能系统性低估 RSU 陈旧危害**。
4. ZOH 实现是否真生效: 审计 `bsave/*/meta/latency_align_audit.csv` 已确认 delta=2 应用(rsu100→delta2), 机制在; 但"应用了延迟"≠"延迟改变了融合输出"。

### ★[2026-06-18 调查进展 — 已查的]
- **嫌疑1(通信门控随机退化)已排除**: `comm_modules/codriving.py` Communication 里 `gaussian_filter` 是**死代码**(`if False: # self.smooth`, 第 80 行从不执行) 且 `init_gaussian_filter()` 在 __init__ 确定性初始化 → missing key `gaussian_filter.{w,b}` **无害**, 非根因。
- **RSU 确实在场且被融合**: `center_point_codriving.py` forward 用 `self.fusion_net`(CoDriving attention, `fuse_modules/codriving_attn.py`)把多 agent(record_len 含 ego+RSU)特征融成 `fused_feature`, cls/reg 头用 fused 出检测(另有 ego-only single 分支但主检测是 fused)。RSU 特征**有路径**。
- **RSU 配置(te0_l1.yaml)**: RSU 在 ego **前方 12m**(`rsu_distance:12`)、每 **5 帧 respawn**(`change_rsu_frame:5`)、`ego_num:1`、`collaborative_perception`。
- **★浮现主因(待 D2 量化)**: RSU 仅前方 12m + Town05 低速 → **RSU 视野与 ego 自身前向高度冗余**; 且低速 1-2 帧位移极小 → ZOH 陈旧 RSU ≈ 当前 → 融合输出几乎不变 → 驾驶分不敏感。**非 bug, 而是"低速+前向冗余 RSU"场景特性系统性低估 RSU 陈旧危害(= 用户担心的保真度缺口)**。
- **工具就绪**: `payload_capture/corridor_payload_step*.pkl` + `b_test_payload_replay.py` 可做 D2。

### ★★[2026-06-18 D2/D2b 诊断定论 — 根因找到]
单帧 payload 复现(`b_test_payload_replay.py`, GPU2, t2lib):
- **D2 (有RSU vs 去RSU, step1044)**: 有RSU **5 检测** vs 去RSU **3 检测**, fused_feature md5 不同 → **RSU 对感知有实质贡献**(去掉丢 2 框含最高分近距目标 score0.82)。**冗余假设证伪**。
- **D2b (当前RSU vs 600ms陈旧RSU, ego@1056)**: 当前 **7 检测** vs 陈旧 **6 检测**。**陈旧确实改变感知**(非 no-op), 但**丢的框是 ego 后方(cx=-3.69)+ 外侧(cy=-10.4)**, 而**前向在途检测(cx=33.6/21.3/10.7)基本保留**(位置微移)。
- **★根因(三段证据综合)**: RSU 时延 **确实传导到感知层**(仿真没坏), 但 ① 低速 Town05 前向走廊场景下, 陈旧主要扰动**非驾驶关键区**(后方/外侧)的检测, ② 决定避障/驾驶分的**前向在途目标**因低速位移小而被保留 → 感知扰动**不转化为驾驶决策变化** → 驾驶分无效应。
- **回答用户"仿真是否真实模拟 RSU 陈旧危害"**: **部分** —— 感知层传导是真的, 但**当前低速前向场景套件系统性低估了驾驶层危害**。真实**高速/快速横穿/遮挡路口**(前向目标位置强依赖新鲜 RSU)下陈旧会丢前向关键目标 → 才有驾驶危害。**这是场景覆盖缺口, 非注入 bug**。
- **下一步建议**: 在 **遮挡(S3)/路口横穿(S4/S14)/提速** 场景重测 RSU 时延(RSU 补盲价值最强、前向目标快速变化处); 或量化高速位移下陈旧 RSU 的前向目标位置误差。工具/payload 已就绪(`d2_rsu_diff.log`/`d2b_stale_diff.log`)。

### ★[2026-06-18 用户反思 — 低 ego 速度是时延无效的物理底因]
- **Efficiency 指标(Bench2Drive 迁移, ego速度/邻车均速)= 53.86%(r146)** → **ego 只跑到环境车流约一半速度**。这闭合因果链: ego 慢 → 每帧位移小 → 陈旧帧(50-600ms)世界≈当前 → 时延扰动不到驾驶关键前向目标 → 驾驶分无变化(正是 D2b "600ms 陈旧下前向在途目标仍保留"的根因)。**用户判断"低 ego 速度=感知/时延没效果"成立, 有数字支撑(ego≈54%环境车速)**。caveat: 53.86% 是 r146 单路线单次, 指示性强非全量。
- ⇒ **暴露 RSU/感知时延驾驶危害的两条路: ① 选 RSU 补盲价值最强且前向目标快速变化的场景; ② 提高 ego 速度(让位移在陈旧窗口内变大)**。
- **场景可用性(均 V2Xverse srunner 原生)**: S3 遮挡横穿(DynamicObjectCrossing)✅真执行 104/105路线、S4 路口横穿(VehicleTurningRoute)✅真执行 80/105路线 —— **直接可用**; S14 闯红灯对向 ⚠️机制已通待在Town05信号路口布触发点(authoring); **"提速"非场景 = 改 config/planner 的 ego 目标速度**。详见 `sim_framework_overview_v1.md` §二/§四。

### 建议诊断步骤(高杠杆优先)
- **D1 [最高杠杆]**: 读 `opencood/.../fusion` + `naive_communication`/`gaussian_filter` 代码, 确认是否在 PnP_infer forward 路径; grep ckpt 实际 load 的 key vs 模型 forward 用的 module。若门控随机初始化且在路径上 → 找到根因。
- **D2**: 用已存的 corridor payload pickle(`PAYLOAD_CAPTURE_DIR` 抓的) + `b_test_payload_replay.py`, 对同一帧跑 **RSU delta=0 vs delta=20** 的融合, diff 检测框/BEV occupancy。若几乎不变 → RSU 对融合贡献小(根因2/3)。
- **D3**: 抓含强 RSU 价值场景(遮挡 S3 / 路口横穿 S4/S14)的 route, 单独看这些 route 上 RSU 时延是否有效(若全局被低 RSU 价值 route 稀释)。
- **D4 [保真度]**: 测 Town05 这些 route 的 ego/邻车速度 → 算 1-2 帧位移 → 评估陈旧 RSU 的位置误差量级。若 < 检测分辨率 → 证实"低速场景低估陈旧危害"。

## §3 基础设施(全部已建, H800 /data/jichengzhi_v2x/)
- sweep 框架 `tau_curve_fulltraffic.py`(env: CFG_TAG/TAG_PREFIX/GPULIST/SLOTS_PER_GPU/TAUS/NREP/ROUTE_FILE/PORT_BASE/BPORT_BASE/SCEN_SUFFIX/RUN_TIMEOUT)。
- 注入代码 `.../team_code/pnp_infer_action_e2e.py`(ego τ_perc 704-718; RSU 在 724-750 RsuLatencyAligner)。RSU cap `latency_frame_align.py` DEFAULT_MAX_DELAY_FRAMES=22。
- 47 路线库 `tau_route_lib_tp0clean47.txt`。诚实绘图 `percinj_plot_honest.py`/`percinj_plot_final.py`/`percinj_c50_compare.py`/`percinj_grid_heatmap.py`(本地副本在 `multi_agent/real_test/`)。
- 渲染 env `/data/jichengzhi_v2x/envs/v2xverse/bin/python`(matplotlib 3.5.3, 英文标签)。结果 `V2Xverse/results/results_driving_{TAG_PREFIX}_{CFG}{tau}_r{R}_n{N}/.../results.json` 读 `_checkpoint.global_record.scores.score_composed`。

## §4 关键坑(血泪, 详见 [[project-tau-perc-signal]])
① H800 ssh 间歇 255(pkill 大量 CARLA 致节点 stall, 服务端实际执行); 只读命令正常; 重启用纯 launch 模式不混 kill+sleep; kill 独立 ssh + 只读 pgrep 验证; D 态孤儿 pkill -9 需等 GPU 操作完成。② 诚实口径: 超时记 DS=0 非排除。③ 灾难率 ~5-8% 噪声地板, 单点 N=3 不可分辨, 需方向性/加 N。④ 延迟 50ms 帧量化, 160-190ms 全=4帧=tp200。⑤ 多主控并行不同 tau 值结果目录不冲突, 但 bport 硬编码冲突→BPORT_BASE env; 12 slot cport 排到 5600 撞 bport(用 BPORT_BASE=6000)。⑥ te0 base 本就含 latency_inject_ms:0(RSU 对齐器一直 ON 在 0ms), 改值用 sed 替换非新增。⑦ 用户 GPU 约定: 避开其训练卡(本会话先 GPU3-5, 后 0-5, 最后 2-5 避 0/1)。

## §5 现状一句话
单 ego 时延曲线(悬崖 600-650)已 settle; ego×RSU 耦合 +50ms RSU 边际效应≈0; **保真度开放问题已闭合(§2.★): 提速验证实验(max_speed 5→8, Eff 90%→150%)证明时延危害随 ego 速度强烈放大(同29交集路 ΔDS0→600 提速-35.5 vs 低速-14.5, 悬崖前移600→400-500ms), 反证注入机制忠实, 此前低速场景系统性低估高动态危害=场景动态覆盖缺口非bug**。图 `figure/data/percinj_fastslow_compare.png`。**§6 阶段1+2 已完成**。剩余可选: §6 阶段3(S4路口横穿+τ_RSU)。

## §6 ★下一步计划(可执行)

> ★★[2026-06-18 阶段1+2 已执行完成, 结论见 §2.★闭合]
> - **阶段1 ✓**: 定位 ego 速度上限 = `control.max_speed`(yaml, codriving=5), L1控制器 `desired_speed=mean(diff(wp_disp)[:3])×max_speed`(max_speed=1/DT_PER_WP, DT=0.2s)。建 `pnp_config_codriving_tpfast{τ}_l1.yaml`(max_speed=8)。配对验证(5路, τ=0, METRIC_LOG=1): mean ego速度 3.03→5.13 m/s(+69%), Eff 89.5%→150%, baseline DS 不崩(84.7→90.0)。
> - **阶段2 ✓**: 提速版 τ_perc 曲线 8 点(0/100/200/300/400/500/600/800)× 47路 × N=3, TAG_PREFIX=percinjfast CFG_TAG=tpfast。两轮跑完(round1=0/200/400/600/800 GPU2,3; round2=100/300/500 GPU2,3,4,5)。交集口径(tpfast0-clean ∩ tp0-clean=29路)对比低速18点 → **时延危害随速度强烈放大, 悬崖前移**(见 §2 表)。
> - ★执行教训(写给接手者): 早期非交集预览曾出**反向**伪结论(提速 ΔDS 比低速小), 因 tpfast0-clean 只27条=高速下只有易路线 τ=0 干净 → 路集难度 confound。**必须用两臂 τ=0 都干净的交集路集**才公平。诚实DS依 N 与灾难地板, 高τ档灾难~5-10%噪声。
> - 复用框架支持断点续跑(`already_done()` 按结果文件存在跳过), 可中途加卡: 杀master→清CARLA/process_b→改GPULIST重启, 自动跳已完成。

### [历史/可选] 原阶段计划(阶段3 仍可做)

> 目标: 验证"提高动态后 RSU/感知时延危害会显现", 反证注入机制是对的、此前只是场景太温和。**主杠杆 = 提 ego 速度**(S3 遮挡已在 104/105 路线含, 但被低速掩盖)。

### 阶段 1 — 定位并提高 ego 速度上限 (P0, 半天)
1. 找速度上限来源: `grep -rn "target_speed\|max_speed\|speed_limit\|desired_speed\|MAX_SPEED" V2Xverse/simulation/leaderboard/team_code/`(疑在 V2X_Controller PID / WaypointPlanner / pnp_config yaml)。
2. 提高一档(如 ×1.5 或解除保守限速), **先 τ=0 跑 3-5 条路线确认 ego 真跑快了**(Efficiency 从 53.86% 往 ≥80% 走)且 baseline DS 不崩(提速本身别把车开飞)。落 `multi_agent/real_test/sim_egospeed_probe_v1.md`。

### 阶段 2 — 提速后重测 τ_perc 时延曲线 (P0, ~4-6h)
3. 用提速 config 重跑 τ_perc 关键档(复用框架, 新 CFG_TAG 如 `tpfast`):
   ```bash
   cd /data/jichengzhi_v2x
   CFG_TAG=tpfast TAG_PREFIX=percinj GPULIST=<空闲卡> SLOTS_PER_GPU=2 SCEN_SUFFIX=_1 \
     ROUTE_FILE=/data/jichengzhi_v2x/tau_route_lib_tp0clean47.txt \
     TAUS=0,200,400,600,800 NREP=3 RUN_TIMEOUT=720 \
     nohup setsid python3 tau_curve_fulltraffic.py > percinj_fast.log 2>&1 < /dev/null & disown
   ```
   (先生成 `pnp_config_codriving_tpfast{τ}_l1.yaml` = tp 配置 + 提速参数)
4. 诚实口径出曲线, **与低速版 18 点对比**: 若提速后同 τ 驾驶分掉得更多 → **时延危害随速度放大 = 注入忠实, 此前低速掩盖** ✓。脚本仿 `percinj_c50_compare.py`。

### 阶段 3 — (可选) S4 路口横穿 + RSU 时延 (P1)
5. 筛含 S4(VehicleTurningRoute, 80/105)的路线做子集, 提速下扫 τ_RSU, 看快速横穿车的协作补盲是否对 RSU 陈旧敏感。

### 判据 / 预期产出
- **成功信号**: 提速后 τ_perc(或 τ_RSU)曲线出现低速版没有的额外下降 → 写入论文"时延危害随场景动态放大, 低速保真但低估高动态危害"。
- **若仍无效**: 升级排查 planner 对感知的依赖(检测变了但 waypoint 不变?)—— 用 D2/D2b 同法在提速帧上 diff `bbox_preds` → `waypoints`。

## §7 ★接手者动作规范(纪律, 违反=返工)
1. **不轻信自报**: 凡"已跑/已修/已 build", 必复跑 / 读文件 / 看日志核验(本会话多次靠核验抓到问题)。注入是否生效查审计(`bsave/*/meta/latency_align_audit.csv`)或 B 日志打印, 别假设。
2. **诚实口径恒定**: 闭环 DS 统计, TIMEOUT_SKIP/无 score 记 **DS=0**(灾难性失败), 不剔除; tp0-clean 配对; 不同 N 的点不可直接比(诚实 DS 依赖 N, 灾难地板 ~5-8%)。
3. **区分真测/估算/声明**, 区分延迟口径(τ_perc ego自感知 / τ_RSU 协作路 / τ_ego plan), 标 caveat。
4. **GPU 礼仪**: 跑前 `nvidia-smi` 确认目标卡空闲(util 0/mem≤50MiB); **避开用户训练卡**(本会话约定动态变化, 跑前问/看); 仿真 latency 测量须卡隔离(CARLA 与 perception 分卡)。
5. **H800 ssh 纪律**: 间歇 255 是节点 stall 非失败(pkill 大量 CARLA 时); 只读命令(echo/pgrep/grep/cat)正常; **重启用纯 launch 模式**(不把 kill+sleep+launch 挤一条 ssh); kill 用独立 ssh + 只读 pgrep 验证; 用 `-o ServerAliveInterval=5`。
6. **重启清理**: pkill tau_curve/CARLA/process_b/leaderboard → 确认 CARLA=0 + 端口空(`nc -z 127.0.0.1 PORT`) → 再 launch; 改 config 用 sed **替换**非新增(te0 base 已含 latency_inject_ms:0)。
7. **改了就记**: 每个核验过的新结论 **整合(非追加)** 进本文档对应段 + 更新 memory [[project-tau-perc-signal]]; 落盘脚本本地副本 `multi_agent/real_test/`。
8. **长跑监控**: 后台 sweep 用 ScheduleWakeup 30min 轮询(查主控存活/Progress 推进/ERROR=0/GPU 隔离); 崩 route 有 RUN_TIMEOUT 兜底不用管, 盯最终 matched 路数(≥40 可信)。
