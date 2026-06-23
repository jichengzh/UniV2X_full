# HANDOFF: AP×τ 闭环二维网格 + GPU 环境 blocker (v1, 2026-06-21)

> 自足交接。承接 [[HANDOFF_ap_knob_closedloop_v1]](AP 旋钮机制/验证/隔离副本已完成) 与 [[project-tau-perc-signal]](延迟轴)。
> 本阶段目标: 跑 **DS(τ_perc, AP) 5×5 二维网格** = 闭环"价值函数"(把离线 (latency,AP) Pareto 翻译成驾驶分)。
> **当前 blocked**: H800 **整机** GPU 渲染子系统 wedge, CARLA 在所有 GPU(含从未跑过 TVM 的 5/6/7)都崩, 需 node reboot。

---

## 0. 一句话现状
AP 旋钮机制 + 离线 knob→AP 标定 + 闭环有效性验证**已全部完成并落盘**(见 v1 handoff)。
正在跑 **AP×τ 5×5 网格**的 Step A(τ=0 行定 AP 崖口),**已 96/120 有效结果**;
但 **CARLA 在 H800 所有 GPU 上崩(整机渲染层问题), 实验暂停**, 需 node reboot 后用加固的 harness 续跑。

## 1. ★ 当前 blocker: 整机 GPU 渲染子系统 wedge (必读, 决定能否续跑)
**★[2026-06-21 晚 诊断修正 — 旧"TVM 退化特定卡"假说已被推翻]**
- **现象**: 单个 idle CARLA 在**任意 GPU(0-7 全试过, 含从未跑过 TVM 的 5/6/7)**上 ~60s 崩 (`GameThread timed out waiting for RenderThread after 60.00 secs` → SIGSEGV signal 11)。
- **系统化排查 (逐一排除, 见本会话)**:
  1. **非 GPU 特定/非 TVM 退化**: GPU5/6/7 从没跑过 TVM, 单起照样同样的 RenderThread 超时崩 → 推翻旧 handoff "TVM 把 GPU0/1/3 图形状态搞退化、GPU2 已恢复"的假说(那是基于不完整观测的误判)。
  2. **非并发抢占**: 单个 CARLA 独占一卡也崩(不是"同卡 2 个 init 竞争")。
  3. **非 CPU 饱和**: `nproc=224` 核, load avg ~20 = 仅 ~10% 利用率, ~200 核空闲。(注: 别被 4 个 ~420%CPU 的 python 唬住——那是别的用户 lixingfeng 的 `eval_pruned_lidar_pyramid.py`, 在 224 核上无关痛痒。)
  4. **非内存**: 可用 1951 GB。
  5. **非渲染后端单点**: `-opengl`(EGL 离屏)60s RenderThread 超时崩; 换 `-vulkan` 路径**进程更早静默死(无端口)**——两条 RHI 路径都废 ⇒ 整机 GL/Vulkan 渲染栈都 wedge。
  6. **无 X/DISPLAY 依赖**(headless EGL, 早上还能跑出 96/120, 说明不是缺 X)。驱动 535.54.03 GSP firmware, dmesg 无权限看 Xid。
- **结论**: H800 **node 级 GPU 渲染/驱动状态 wedge**(疑似重度 CUDA 负载把 GSP/显示引擎拖坏, 但**与具体哪张卡跑过无关, 是全机**)。**早上 96/120 能跑 → 现在全崩**, 是这之后整机渲染栈坏掉。
- **★★[2026-06-21 晚 修复成功 — 真正的恢复手段 = per-GPU `nvidia-smi --gpu-reset`]**: 用户 `sudo nvidia-smi --gpu-reset -i 5/6/7` **逐卡 reset 成功**, reset 后 CARLA 在 GPU7 立即 PORT-UP(显存 7→2587MiB, 渲染上下文正常创建)→ **gpu-reset 清掉了渲染 wedge, 无需 reboot/无需迁 4090**。诊断"render context wedge"是对的, 修复就是 reset。
- **★关于"gpu-reset 不可用"的纠正**: 旧 handoff 写的 "Fabric Manager 持卡→gpu-reset 必失败" **是误判**。真相: **per-GPU reset 只在该卡有 compute 进程持着时失败**(报 "In use by another client" 指的是那个进程, 不是 FM)。**卡上无 compute 进程时, `sudo nvidia-smi --gpu-reset -i N` 能成功**(本次 5/6/7 都成了; GPU4 因有 jichengzhi python 持着才 reset 不了)。⇒ **恢复流程 = ①先 kill/挪走该卡上的 compute 进程 ②`sudo nvidia-smi --gpu-reset -i N`(需 sudo, 用户执行) ③重测单 CARLA 确认 PORT-UP**。
- **验证旁证**: 同一条 `-opengl -RenderOffScreen` 命令 + 同 CARLA, 在**本地 4090(bm-2s55he)上能正常 PORT-UP**(GPU6 显存 1648→2381MiB), 只有 H800 wedge → 坐实是 H800 节点 GPU 状态问题, reset 即解。
- **4090 fallback(备用, 本次未走)**: 本地 `/home/jichengzhi/V2Xverse` 有完整 CARLA+CoDriving 权重+`v2xverse`/`t2val` env, CARLA 渲染实测正常; 但 closedloop 是**旧快照**(只有 `infer_server.py`, 无 `process_b_server.py`/AP 注入/t2lib), 迁移需从 H800 传 isolated 副本+t2lib, 且 8 卡中仅 GPU6 空。仅当 H800 reset 也救不了时才考虑。
- **重测命令**(确认某 GPU 是否恢复, 在 GPU 空闲时):
  ```
  cd /data/jichengzhi_v2x/carla
  setsid env CUDA_VISIBLE_DEVICES=<g> ./CarlaUE4.sh -prefer-nvidia -opengl -RenderOffScreen \
    -world-port=7700 -nosound -quality-level=Low > /tmp/carla_test_g<g>.log 2>&1 < /dev/null &
  # 等 ~100s; grep "re-raising signal" 该log=还崩; nc -z 127.0.0.1 7700=恢复
  ```
  **教训**: ssh 经常 stall 截断 inline 启动命令 → CARLA 没真启动但看不出来。**启动 CARLA 必须用服务器端脚本 + 起后立即核验 log 非空 + 进程在**。

## 2. 实验设计: DS(τ_perc, AP) 5×5 网格
- **两个正交注入轴**(机制见 v1 handoff §1):τ_perc=感知**输入**端延迟(认知滞后, Δ=⌈τ/50ms⌉帧);AP=感知**输出**端丢框(featmask=1 一致退化, 忠实)。
- **网格刻度**:τ ∈ {0,200,400,600,800} ms × drop ∈ {0,0.25,0.5,0.7,0.85}(featmask=1)。
- **drop→veh AP50 标定**(离线200帧): 0→0.84 / 0.25→0.78 / 0.5→0.56 / 0.75→0.31 / 0.9→0.27(0.7≈0.36, 0.85≈0.28 插值)。
- **路集**: 8 条 tp0-clean (`/exdata/jichengzhi/grid_routes.txt` = 3,17,18,104,136,312,317,324), N=3, 诚实DS(超时→0), full-traffic `_1`。
- **config 编码**: code = τ×100 + drop_pct; 文件 `pnp_config_codriving_g{code}_l1.yaml`(副本 agent_config, 25 个已生成, simulation 段含 tau_perc_ms/ap_drop/ap_featmask)。
- **执行顺序(用户定)**: 先 Step A(τ=0 行=code 0/25/50/70/85 定 AP 崖口)→ 看刻度合适否 → 再放剩余 4 行(480 run 量级)。

## 3. ★[2026-06-22] Step A 完成 120/120 + 结果(非单调,重要)
gpu-reset 修复后在专属 GPU 3/4/5 续跑, **Step A 全 120/120 有效**。τ=0 行 DS vs drop(clean 交集 6 路 [3,17,18,104,136,317] × N=3 = 18 reps/格):
| drop | veh_AP50 | honest DS | 灾难超时% |
|---|---|---|---|
| 0.00 | 0.84 | 85.3 | 0 |
| 0.25 | 0.78 | 68.5 | 0 |
| 0.50 | 0.56 | **56.4(谷底)** | 0 |
| 0.70 | 0.36 | 63.8 ↑ | 0 |
| 0.85 | 0.28 | 66.7 ↑ | 11 |
**★非单调,不是 bug = 两个真实机制**(看每路原始 DS, 分析见本会话):①**N=3 方差大**(如 r17@drop50=`100,1,25`);②**高 drop 出现 DS=50 地板**(drop70/85 大量路掉到正好 50)= 检测框丢多→planner **过度保守**→车半路堵住/不敢动→RC~50% 但**不撞**(灾难少)→均值从谷底回升。⇒ **AP→DS 因果是真的**(clean drop0 DS85 vs 退化 56-67), 但 composed-DS 口径**非干净单调崖口**, 退化主表现是**过度保守(堵住)非撞车**。**正式分析建议拆 RC/碰撞率子指标**(过度保守应在 RC 上更单调)。
r312/r324 被 clean 交集排除(连 drop0 都频繁 900s 超时=固有难/不可靠路)。
结果路径 `/exdata/jichengzhi/V2Xverse_apknob/results/results_driving_grid_g{code}_r{R}_n{N}/v2x_final/town05_short_collab/r{R}_repeat0/ego_vehicle_0/results.json`。

## 3c. ★★[2026-06-22] 完整 5×5 网格 480/480 完成 — 核心结论(论文级)
DS(τ,AP) honest 热图(clean 6 路 × N=3, `grid_out/grid_ap_tau_heatmap.png`, 本地副本 `multi_agent/real_test/ap_tau_heatmap.png`, CSV `ap_tau_grid_ds.csv`):
| τ\drop | AP.84 | AP.78 | AP.56 | AP.36 | AP.28 |
|---|---|---|---|---|---|
| 0 | 85.3 | 68.5 | 56.4 | 63.8 | 66.7 |
| 200 | 84.7 | 83.3 | 75.0 | 69.4 | 60.0 |
| 400 | 67.2 | 66.9 | 68.5 | 54.6 | 62.5 |
| 600 | 77.4 | 65.9 | 76.8 | 59.2 | 81.3 |
| **800** | **26** | **37** | **23** | **28** | **26**(全含灾难超时) |

**子指标(`/tmp/grid_submetric.py`)拆解机制**:
- **RC(路程完成%)**: τ=0-600 ~85-100%; **τ=800 暴跌 52-72%**(车堵住跑不完)。
- **车辆碰撞/局**: τ=0-600 **恒 ~0.00-0.03(与 AP drop 无关!)**; **τ=800 才 0.08-0.19**(出现碰撞, drop0.7 最高 0.19)。

**★★核心结论(协同加速闭环落点)**:
1. **AP 退化不引发碰撞** — AP 0.84→0.28 全程碰撞≈0(τ≤600)。丢检测框 → planner **过度保守(堵车)非鲁莽(撞车)**, 模型对丢弱检测鲁棒。
2. **AP 的 DS 损失来自 RC/罚分(过度保守), 非碰撞**。
3. **延迟是危险主导轴** — 仅 τ=800 同时引发碰撞+跑不完 → DS 崩(23-37)。延迟悬崖在 600-800ms 间。
4. **论文含义**: 闭环里**可激进压缩感知(剪枝/量化↓AP)而驾驶代价小, 但延迟必须压在悬崖下**。优化预算砸降延迟; AP 损失"便宜"。
**caveat**: N=3 × 6 路, 噪声大; AP 轴在 τ≤600 内非单调(噪声+过度保守地板, 见 §3); τ=800 碰撞的 AP 梯度(0.08→0.19)suggestive 但在噪声内。**若要强化结论建议提 N 或扩路集, 并以碰撞率/RC 为主轴(比 composed-DS 干净)**。

## 3b. ★[2026-06-22] 完整 5×5 网格 τ>0 四行 (已完成, 历史记录)
用户拍板"直接铺完整 5×5"。τ>0 四行(τ=200/400/600/800 × drop 0/.25/.5/.7/.85 = 20 codes)在 **clean 6 路**(避 r312/r324 超时浪费)× N=3 = **360 run**, 专属 GPU 3/4/5(SLOTS_PER_GPU=1), 约 8h 过夜跑。harness pid 见 grid_full.log; 启动脚本 `/tmp/launch_grid_full.sh`(TAUS=20 codes, ROUTE_FILE=grid_routes_clean6.txt, PORT_BASE=4600 BPORT_BASE=6500)。完成判据=grid_g* 有 status 数达 **480**(120 Step A + 360)。完成后: 全 τ 跑 `grid_analyze.py`(不设 ONLY_TAUS)→ `grid_out/grid_ap_tau_heatmap.png`。
**★GPU 纪律(本会话血泪)**: TVM(tvm310, s2_tvm/, 动态迁移抢卡)与 CARLA 网格**必须分卡**, 否则 TVM workers 迁到 CARLA 卡上撞死 harness(本会话死 2 次)。用户已把 GPU 3/4/5 专门空出给 CARLA。

## 4. harness bug 修复记录 + 待加固 (★续跑前必看)
harness = `/exdata/jichengzhi/tau_curve_apknob.py`(VXDIR/CARLA_L 已指副本+noadapter脚本)。
**已修**:
1. **eval-died 误判丢数据**(原 bug): harness 把 bash-wrapper PID 退出当 "job 完成" → 提前 free slot → 同端口起新 job 杀掉旧 job 仍在跑的 procB/CARLA → 留无 status 半截结果(~28% 损耗)。**已去掉该误判分支**(slot 只在 result-file 或 RUN_TIMEOUT 结束)。
2. **CARLA `-graphicsadapter=$GPU` 在 GPU1 段错误**: 已改用无 graphicsadapter 的启动脚本 `/exdata/jichengzhi/h800_carla_launch_noadapter.sh`(靠 CUDA_VISIBLE_DEVICES 隔离), harness CARLA_L 已指它。
3. **同卡并发 CARLA init 竞争**: 2 slot/GPU 时同卡第 2 个 CARLA init segfault → 已降 SLOTS_PER_GPU=1(代价: 慢一半)。
4. **★★[2026-06-22] `epid` NameError = harness 反复死亡的真根因(修了)**: check_done 超时分支第 159 行用了未定义的 `epid`(应 `slot_epid[slot]`)——修 eval-died 误判(已修#1)时删了 `epid=slot_epid[slot]` 定义却留了引用。**每次某 slot 超时 → check_done 抛 NameError → 整 harness 崩**。本会话前 3 次 harness 死亡(归因"TVM 撞卡")真相 = TVM 撞卡致超时、此 bug 把超时变致命崩溃。**已 `sed s/\bepid\b/slot_epid[slot]/g` 修复**(备份 `tau_curve_apknob.py.bak_epidfix`, 语法验过), 修后 harness 遇超时只跳过该 slot 继续。教训: 删代码留悬空引用; 且超时路径平时不走、一走就崩, 测试覆盖不到。

**待加固(非阻塞, 当前修复版已能跑)**:
1. **CARLA 偶发崩(~25%)的快速 requeue**: 现去掉 eval-died 分支后, CARLA 一崩 job 死等 900s。应在 check_done 加**进程组存活检测**: `os.killpg(slot_epid[slot], 0)`(slot_epid 因 start_new_session 即 pgid; killpg 检查整组=wrapper+python 都死才判失败, 避开原误判)+ no-result + 过 grace(120s)→ requeue(加 module 级 `_retry` 列表, main 循环排空进 jobs+total, 带 retry 上限防死循环)。这样崩了快速重排不等 900s。
2. **同卡 CARLA 启动错峰**: 想回到 2/GPU 吞吐, 需在 start_slot 间加 stagger(同 GPU 两个 CARLA 启动间隔 ≥30s), 避免 init 竞争。

## 5. 环境恢复后的续跑步骤
1. **确认 GPU 0/1/3 恢复**(§1 重测命令, 单 CARLA 稳 100s+)。
2.(可选, 先加固 harness §4 待办 1, 让大网格抗 CARLA 偶发崩。)
3. **续 Step A 补缺口**(服务器端脚本避 ssh 截断):
   ```
   cd /exdata/jichengzhi
   # 先删无status半截目录, 再 resume
   CFG_TAG=g TAUS=0,25,50,70,85 NREP=3 ROUTE_FILE=/exdata/jichengzhi/grid_routes.txt \
     TAG_PREFIX=grid SCEN_SUFFIX=_1 GPULIST=0,1,2,3 SLOTS_PER_GPU=1 \
     PORT_BASE=4600 BPORT_BASE=6500 RUN_TIMEOUT=900 \
     setsid nohup python3 /exdata/jichengzhi/tau_curve_apknob.py > /exdata/jichengzhi/grid_stepA.log 2>&1 </dev/null &
   ```
4. **Step A 分析**(AP 崖口 1D): 用 `/tmp/grid_analyze.py`(本地副本待存到 real_test) ONLY_TAUS=0 → 看 DS vs drop 崖口位置, 确认/调整 drop 刻度。
5. **放大网格**: TAUS=按需把 τ=200/400/600/800 行的 code 加进去(如 TAUS=20000,20025,...,80085), 或分批。建议加固 harness 后用 2/GPU+错峰。
6. **网格热图**: `grid_analyze.py`(全 τ)→ `grid_out/grid_ap_tau_heatmap.png` + iso-DS 等高线。

## 6. 关键资源
- **H800**: `sshpass -p '12345678' ssh -p 30001 jichengzhi@222.95.84.215`(密码每会话确认; ssh 易 stall, 长任务用 setsid+服务器脚本)。
- **隔离副本**(所有 AP 工作, 主框架零污染): `/exdata/jichengzhi/V2Xverse_apknob`。
- **harness**: `/exdata/jichengzhi/tau_curve_apknob.py`; **CARLA 启动**: `/exdata/jichengzhi/h800_carla_launch_noadapter.sh`。
- **路集**: `/exdata/jichengzhi/grid_routes.txt`。
- **离线 AP 标定**: `V2Xverse_apknob/opencood/tools/inference_apcalib.py`(env APCALIB_TESTDIR=/exdata/jichengzhi/v2xverse_dataset/ AP_DROP=<d> APCALIB_NMAX=200)。
- **分析脚本**(本地 repo): `multi_agent/real_test/ap_knob_*`(ds_analysis/make_configs/calibration_curve/injection.patch) + 待存 `grid_analyze.py`。
- **配套 memory**: [[project-ap-knob-closedloop]] [[project-tau-perc-signal]] [[project-real-repo-is-v2x]]。
- **GPU 纪律**: 用户授权 GPU 0,1,2,3; 但**别和 TVM/重型 CUDA 任务抢同批卡**(本次教训); GPU4 常有他人任务。

## 7. 教训 (本会话血泪)
1. **CARLA 整机渲染 wedge 的诊断纪律**: "某些卡崩→换别的卡跑"是错的——必须先单卡逐一排除(GPU特定?并发?CPU?内存?渲染后端?)才能定性。本次换 GPU5/6/7(从没跑过 TVM)照样崩, 才确认是**整机渲染栈坏**而非单卡退化。NVSwitch 机 gpu-reset 永远无效(Fabric Manager 持卡), 整机 wedge **只能 node reboot**(共享节点需协调)。教训: 别用"局部退化/空闲自愈"的乐观假说掩盖整机故障。
2. **ssh stall 频繁截断 inline 命令** → kill/launch 看似执行实则没跑。补救: 服务器端脚本 + 起后立即核验(进程在/log非空/GPU mem变), `pkill -f` 不可靠时用 explicit PID kill。
3. **harness "COMPLETE N/N" ≠ N 个有效结果**(eval-died 计入完成但无结果)→ 必数"有 status 的结果数", 缺的 resume 补。
4. **进程计数被 pgrep 自匹配污染**(命令行含 "CarlaUE4" 等字串)→ 用 `ps -ef|grep X|grep -v grep|wc -l` 或 explicit PID。
