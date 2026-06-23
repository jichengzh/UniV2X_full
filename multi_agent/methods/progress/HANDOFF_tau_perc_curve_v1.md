# 交接: 感知延迟(τ_perc)→V2X驾驶退化曲线 — 补点验证 (HANDOFF_tau_perc_curve_v1)

> 写于 2026-06-15。承接 τ_ego 长探索:ego-plan 延迟注入两次 N=3 验证均 null,改为**感知链路注入**后拿到单调信号。
> ★当前任务:补 τ_perc=100/300/400/1000 加密曲线,确认 3 点单调**不是样本少/噪声假象**,再下最终结论。
> 接手先读 §1 现状、§2 待办命令、§5 已知坑。

---

## §0 一句话
感知延迟注入(延迟 ego 自感知输入 car_data_raw[0])在 H800 full-traffic 上拿到 **tp0/tp200/tp500 三点单调退化曲线**(碰撞率 0.04→0.15→0.25,DS 98.1→91.6→82.6,47 路 matched N=3);但只有 3 点,需补 tp100/300/400/1000 验证单调性、排除噪声,才能定论。

## §1 现状(已完成 + 已settled)

### 已完成实验(数据在 H800)
- **tp0/tp200/tp500 × 55 路 te0-clean × N=3 全跑完**(percinj_main5,06:51 完成,376/376)。
- 最终 47 路严格 matched(tp0-clean ∩ 三档都有效):

  | τ_perc | 碰撞率(ped+veh) | 平均DS ±95%CI | n_run |
  |---|---|---|---|
  | 0 | 0.04 | 98.1 ±1.7 | 141 |
  | 200 | 0.15 | 91.6 ±3.6 | 141 |
  | 500 | 0.25 | 82.6 ±5.2 | 141 |
- 图/CSV(3 点版):`multi_agent/figure/data/percinj_curve.{png,csv}`。

### 整个 τ_ego 探索脉络(4 实验对比,别重复踩)
| 实验 | 注入点 | 配置 | 结果 |
|---|---|---|---|
| `_1notraffic` ego-plan(tau_ego) | 轨迹延迟 | 删ambient | **null**(删了信号源) |
| full-traffic ego-plan N=3 matched | 轨迹延迟 | _1 | **null** Δ=+1.5 |
| 4090 n=1 ego-plan | 轨迹延迟 | _1 | 假象退化(n=1噪声,被N=3证伪) |
| **感知注入(tau_perc) N=3** | **感知延迟** | _1 | **单调信号 ✅** |

### 核心机理结论(已验证)
延迟危害本质=**对世界认知滞后**(感知延迟:还没看见障碍→晚反应→撞),不是计划滞后(ego-plan:看见了只晚执行→控制器/轨迹冗余吸收→无害)。

## §2 ★待办:补点命令(直接可跑)

**H800 接入**:`sshpass -p '12345678' ssh -p 30001 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215`(密码每会话确认),repo `/data/jichengzhi_v2x/V2Xverse`。

**补 tp100/300/400/1000(断点续跑,已完成的 tp0/200/500 自动 skip):**
```bash
cd /data/jichengzhi_v2x
# 先确认无残留: ps -eo cmd|grep -E "^python3 tau_curve_fulltraffi[c]"|wc -l 应=0
# GPULIST 按当前空闲卡改(避开他人训练); 当前约定用 3,4,5
CFG_TAG=tp TAG_PREFIX=percinj GPULIST=3,4,5 SLOTS_PER_GPU=2 SCEN_SUFFIX=_1 \
  ROUTE_FILE=/data/jichengzhi_v2x/tau_route_lib_fullclean_v2.txt \
  TAUS=0,100,200,300,400,500,1000 NREP=3 RUN_TIMEOUT=720 \
  nohup setsid python3 tau_curve_fulltraffic.py > percinj_main6.log 2>&1 < /dev/null & disown
# ssh 会超时返回(124/255)正常, 进程已起; 单独查:
# ps -eo cmd|grep -E "^python3 tau_curve_fulltraffi[c]"|wc -l  (应=1)
# grep "done=" percinj_main6.log  (done=已完成数, pending=新档)
```
- 规模:55 路 × 4 新档 × N3 = **660 新 run**(崩 route 有 TIMEOUT_SKIP 标记会 skip)。3 卡 6 slot ~10-11h;卡多更快。
- **tp1000 的 Δ_perc=20 帧**(cap 已从 10 提到 20,见 §5.7);tp0/200/500 不重跑。

**进度/分析:**
```bash
python3 /data/jichengzhi_v2x/snap_perc.py   # 全档 matched 碰撞率/DS±CI (未跑档自动跳过)
```
**出图(所有目标档跑完后):**
```bash
/data/jichengzhi_v2x/envs/v2xverse/bin/python /data/jichengzhi_v2x/percinj_plot.py
# 产出 /data/jichengzhi_v2x/percinj_out/percinj_curve.{png,csv}
scp -P 30001 ...:/data/jichengzhi_v2x/percinj_out/percinj_curve.* multi_agent/figure/data/
```

**判据(回答"是否噪声"):** 7 点(0/100/200/300/400/500/1000)若碰撞率/DS 仍单调、相邻点差 > CI → 真信号;若中间点乱跳超出 CI → 样本不足需加 N 或更多 route。

## §3 拉起的 agent / 工作流
本阶段可由**主控直接驱动**(实验是机械 sweep + 定时快照,不必多 agent):起 sweep → 后台 watcher 盯 COMPLETE → 出图。若要并行可拉 sim-executor 跑、supervisor 核验。纪律:SAVE_PATH/TMPDIR 在 /data;不轻信自报,复跑核验;1 卡组隔离。

## §4 基础设施清单(文件位置)
- **sweep 框架**:H800 `/data/jichengzhi_v2x/tau_curve_fulltraffic.py`(本地副本 `multi_agent/real_test/tau_curve_fulltraffic.py`)。env:CFG_TAG(te/tp)/TAG_PREFIX/GPULIST/SLOTS_PER_GPU/SCEN_SUFFIX/TAUS/NREP/RUN_TIMEOUT/ROUTE_FILE/PORT_BASE。含**单run超时保护**(§5.1)。
- **感知注入代码**:H800 `.../team_code/pnp_infer_action_e2e.py`(原文件备份 `.bak_preperc`);本地副本 `multi_agent/real_test/pnp_infer_action_e2e.percinj.py`。改动 2 处:init(527-530 读 tau_perc_ms+建 bank)、get_action(708-717 延迟 car_data_raw[0]+ZOH+深拷贝)。`tau_perc_ms=0` 默认零影响(向后兼容)。
- **config**:`.../agent_config/pnp_config_codriving_tp{0,100,200,300,400,500,1000}_l1.yaml`(tau_ego_ms=0 不延迟plan + tau_perc_ms=X 延迟感知;从 te0_l1.yaml sed 生成)。
- **route 库**:H800 `/data/jichengzhi_v2x/tau_route_lib_fullclean_v2.txt`(**55 条**,已剔崩的 r26/r30);57 条原始本地 `multi_agent/real_test/tau_route_lib_fullclean_v1.txt`。来源=4090 mr_te0 n=1 筛 te0-clean。
- **分析/出图**:H800 `snap_perc.py`(matched 统计)、`percinj_plot.py`(双轴图,v2xverse env 渲染:`/data/jichengzhi_v2x/envs/v2xverse/bin/python`,matplotlib 3.5.3)。
- **结果数据**:H800 `.../results/results_driving_percinj_tp{TAU}_r{R}_n{N}/.../ego_vehicle_0/results.json`,读 `_checkpoint.global_record.scores.score_composed`。

## §5 ★已知坑(必读,血泪)
1. **崩 route + 超时**:H800 full-traffic 下 **~25% run 偶发 route error 崩**(`'Location' object is not iterable`/`route_length`,ambient actor 生成/销毁竞态,**非代码 bug**——tp0 不激活注入也崩 14 次)。框架 **RUN_TIMEOUT=720s 超时保护**:>12min 无 results→kill+写 `status=TIMEOUT_SKIP` 标记→断点续跑永久 skip,不卡死。崩的 eval **不退出**所以必须靠超时。
2. **双主控误判**:`pgrep -fc tau_curve` 会匹配 bash wrapper 显示 2;**真主控数**用 `ps -eo cmd|grep -E "^python3 tau_curve_fulltraffi[c]"|wc -l`。
3. **ssh 启动超时(124/255)是正常的**:nohup setsid 后台命令 ssh 会 hang 被 timeout 杀,但**进程已起**,单独查确认,别重复启动(会叠多个主控)。
4. **重启必须彻底清理 + 确认端口空**:`pkill -9 -f tau_curve_fulltraffi[c]` → `pkill -9 -f leaderboard_evaluato[r]` → `pkill -9 -f process_b_serve[r]` → `pkill -9 -f CarlaUE[4]` → sleep 6 → 确认 `pgrep -fc CarlaUE[4]`=0 **且端口 3000-4000 空**(`nc -z 127.0.0.1 3000`)。否则旧 CARLA 占端口→新 eval 连错→route error 卡死(踩过)。
5. **CARLA 进程数 = slot×2**(CarlaUE4.sh launch + CarlaUE4-Linux-Shipping 两进程),6 slot=12 CARLA 正常,不是残留。
6. **深拷贝 car_data_raw[0]**:感知注入必须深拷贝(preprocess 原地翻转点云 `lidar[:,1]*=-1` + ZOH 重复取用会污染 bank)。已验证 OK。
7. **cap=20 帧**:`delta_p=min(...,20)`=1000ms 上限(原为 10/500ms,改过才能测 tp1000)。RsuLatencyAligner 的 DEFAULT_MAX_DELAY_FRAMES=10 是 RSU 分支,与 ego 感知注入无关。
8. **必须用 `_1`(full-traffic, SCEN_SUFFIX=_1)**:保留 60 车+60 人 ambient=信号源。`_1notraffic` 会抹平信号(已验证 null)。
9. **N=3 + 严格 matched + tp0-clean 二次确认 缺一不可**:n=1 有假象(4090 退化是假象);tp0-clean=tp0 时 DS=100/0碰撞 在≥2/3 reps;matched 只统计**所有已测档都有数据**的路线(崩 route 被 skip 后才不污染)。
10. **GPU**:GPULIST env 控制;避开他人训练卡(本会话约定不碰 GPU0/1/2)。DS 验证是同步仿真,DS 与算力无关,可共卡(SKIP 不需要,但别 OOM)。
11. **percinj_plot.py 全档出图**:目标档全跑完再出图(line 27 common 交集逻辑,若某档完全无数据 common 会空;snap_perc.py 已加 `if T[t]` 守卫,plot 没有——补点跑完即可)。

## §6 一句话给接手者
信号已现(3 点单调),但点少不足定论。补 tp100/300/400/1000(命令见 §2,一条龙)→ snap_perc 看 7 点是否仍单调 → percinj_plot 出图。崩 route 有超时兜底不用管,关注最终 matched 路数(≥40 可信)。若 7 点单调且相邻>CI = "感知延迟单调伤驾驶"坐实,配 ego-plan 的 null 形成"认知滞后伤驾驶/计划滞后鲁棒"的完整对比故事。
