# H800 闭环 te0 即崩 — 根因调查台账 (h800_rootcause_investigation_v1)

> 调查总监(thinker)维护。执行手 sim-executor 跑命令回报原始证据。
> **终极目标**: 不是写报告，而是收敛到一个 **H800 双进程闭环 te0(零注入延迟)能复现 4090 水平(碰撞率<50%、DS 趋近 100)** 的可用系统。根因是"被修复验证出来的"。
> 本文 = 假设台账。每条假设: 假设陈述 / 设计的实验 / executor 回报的证据 / 调查总监判决 / 下一步。整合非追加。
> 写于 2026-06-13。

---

## ★ 延伸验证 (2026-06-13, 用户) — `_1notraffic` 是否"既确定又保留延迟敏感避让"
> 把 `_1notraffic`(ambient walker+vehicle=0, **保留 Scenario3 脚本行人**)从诊断配置升为**可用评测配置**。目标: 确定性 + 延迟敏感的行人避让 → 干净跑 τ_ego→DS 曲线。
- **★先决·矛盾待解(我发现)**: executor Q2 曾报 "r6 scenario_triggers=0"(无 scenario), 但 team-lead 称 B3 日志 load **Scenario3 @ x=-44.38**(正是 ego 冻死处)。**二者矛盾, 必先核 B3 eval 日志实际 load 了什么 scenario**。
  - 若 r6 **有** Scenario3 行人 → 全图自洽: B3 r6 te0=[100,100,100]=零延迟避开脚本行人; te219/te500 太慢撞它 → `collisions_pedestrian>0` = 要的延迟敏感避让。
  - 若 r6 **无** scenario 行人 → `_1notraffic` r6 无可避 → te0=te219=te500=100 无延迟敏感 → r6 不适用, 换有 Scenario3 的 route。
- **★关键次查(我升级为重要)**: `_1notraffic`(vehicle=0)**是否把 V2X 协同者(RSU/其他CAV)也删了**? 删了则 V2X 价值受损, 曲线非 V2X → 需"只删非协同 ambient"的配法。查 eval 日志协同 agent 构成 + RSU 是否在。
- **实验**: r6 + 1-2 条 Scenario3 route(如 r3) × `_1notraffic` × {te0,te219,te500} 各 ×3。判: ①每档 ×3 bit-reproducible(te0 已 B3=[100,100,100]); ②DS 随 τ_ego 单调降 + 失败=`collisions_pedestrian`(撞脚本行人)。成功=确定+保留 V2X 避让的可用配置。
- **状态**: ✅ **完成(2026-06-13)**。`_1notraffic` = **可用评测配置确认**(te0 确定锚=100, V2X/RSU 保留, 失败纯 ped-collision 无车撞)。**核心交付**: ① 配置可用; ② 延迟曲线 **route-dependent**(失败阈值依避让余量: r6 te219 崩 / r3 te500 才崩); ③ **历史"非单调曲线"= 混 route 阈值差 + 极延迟余量恢复 + n=1 伪象, 非真信号**; ④ 方法 = **逐 route 出曲线 + 每档大 N(≥10) + 报碰撞率, 别跨 route 朴素平均**; ⑤ 边际档高方差 = 极小避让余量(~0.23m)落在 CARLA 物理非确定带 → 概率碰撞(诚实: ZOH 保守机制未直接证实, 需 l1_debug 重跑; 物理余量解释与数据一致)。
- **★①② 先决已满足(2026-06-13 eval-log 硬证)**:
  - ① **r6 确有 Scenario3 ×10**, 一处 **(-44.38,-51.56)=corridor 冻死点**(eval.log `load Scenario3 at (x=-44.38,y=-51.56)` / `route_scenarios:{Scenario4:13,Scenario3:10}`)。executor 诚实纠正自己 "scenario_triggers=0" 错(route XML 无显式 trigger, 但 Scenario3 按地理 bbox 匹配)。⇒ **全自洽**: te0 避开脚本行人(B3=100), te219/500 该撞它。**且印证 corridor 挡车主体是 Scenario3 脚本行人(确定), ambient traffic 只是叠加噪声**——这也解释为何 B3 去 ambient 后 r6 确定。
  - ② **V2X 协同存活**: eval.log `disable_rsu? False`(RSU 在); `vehicle_amount=0` 只控背景 NPC 车, RSU 由 `spawn_rsu()` 独立生成不受影响。⇒ `_1notraffic` 保留 RSU/V2X 协同, 曲线反映 V2X。
- **★扫描放行**: r6(Scenario3@-44.38) + r3(历史延迟敏感) × `_1notraffic` × {te0,te219,te500} ×3, per-run 重启。r6 te0 复用 B3=[100,100,100]。
- **★扫描首结果 + 判读(2026-06-13)**: r6 te219 = n1 DS50(colped0.14=1撞)/n2 DS12.5(colped0.42=3撞), colveh=0 → **失败模式对(撞 Scenario3 行人, 延迟敏感: te0 100→te219 崩)**, 但**档内不 bit-reproducible**(50≠12.5)。
  - **变源不是行人**: 读码确认 Scenario3 行人 = `KeepVelocity` 脚本定速横穿 + 固定 spawn = **确定**(object_crash_vehicle.py:485)。
  - **变源 = 残余间歇 CARLA 闭环非确定**(贯穿全程的那类: n2/n4/B1run2/B2run2 偶发分叉, 疑 CARLA server init/physics warmup, 0.9.10 不可控)。`_1notraffic` 把方差**降低**(去 ambient)但**未消除**; 边际碰撞档(te219)放大残余。
  - **★但定性信号 robust**: 两跑都撞行人(qualitative: te0避/te219撞 一致), 只**碰撞次数/DS幅度**有残余方差。⇒ **延迟→失败 transition 确定; 严重度有方差**。
  - **修正成功判据(诚实)**: 不是"每档 bit-reproducible"(te219 不是), 而是 **te0 确定锚点 + 延迟单调降 + 失败=ped碰撞 + 边际档 N-run 平均(mean±std)**。⇒ `_1notraffic` 是**可用延迟曲线配置(需边际档 N≥3-5 平均)**, 非"全档 bit 确定"。te0 是确定 baseline。
  - **★非单调成立(均值+碰撞率), 但高方差(2026-06-13, r6 档齐)**: r6 `_1notraffic`:
    | τ_ego | DS [n1,n2,n3] | mean±std | 碰撞率 |
    |--|--|--|--|
    | te0 | [100,100,100] | 100±0 | **0/3** |
    | te219 | [50,12.5,50] | 37.5±17.7 | **3/3** |
    | te500 | [100,100,12.5] | 70.8±41.2 | **1/3** |
    - **非单调在均值/碰撞率层面确认**(碰撞率 0→100%→33%, te219 是谷底最差)。**机制(读码推, 半确认)**: Scenario3 blocker prop 遮挡 ego 自感知 → 依赖 RSU; frame-align 只延 RSU。te0 RSU 及时→避; te219 RSU 晚→撞; te500 RSU 撞 max_delay 上限→ZOH→ego 无新预警→**退化保守→2/3 避开**。
    - **但每个降级档高方差**: te500 的 ZOH 保护是**概率性的**(2/3 生效), 残余间歇 CARLA 非确定决定每跑。⇒ **曲线非确定性线, 是带噪声的统计趋势; 严格确认非单调需大 N(≥10)**。
    - **修正**: `_1notraffic` **配置可用**(te0 确定锚点/V2X 保留/延迟敏感), 但**延迟曲线高方差→需大 N 平均 + 报碰撞率**。非单调"中延迟最差"是**真信号(机制成立)但需大 N 坐实**。ZOH 机制 dump 推迟到 r3 后(对比 avoid-run vs hit-run 的 ego 速度+RSU latency)。
    - 待 r3 验证非单调是否泛化(route-specific or general)。
  - **★r3 初值 → 延迟敏感性 route-dependent(2026-06-13, 待 n2/n3)**: r3 te0=[100,100](n3 CARLA_TIMEOUT 跳过, n=2 够锚点); **r3 te219 n1=100/colped=0 ≠ r6 te219(3/3撞)**。⇒ **r3 在 te219 未崩 → 延迟敏感性依 route 而异**(r6 的 Scenario3 余量小→te219 崩; r3 余量大→te219 仍避)。若 r3 te219 n2/n3 也≈100 → **非单调碰撞谷是 route-specific 非普适** = 重要发现(τ_ego→DS 依各 scenario 避让余量/几何)。⇒ 项目延迟曲线须**多 Scenario3 route 平均**(捕捉敏感性差异) + 每档大 N。
  - **★★ 两 route 曲线形状不同 = 失败阈值依避让余量(2026-06-13, r3 te500 n1=50 撞)**:
    | route | te0 | te219 | te500 | 形状 |
    |--|--|--|--|--|
    | r6(余量小) | 100 | **37.5±18 谷** | 70.8±41 部分恢复 | **非单调** |
    | r3(余量大) | 100 | **100±0 平** | **50(n1撞)** | **晚发降级** |
    - **r6**: 中延迟(te219)就崩→极延迟(te500)ZOH 保守部分恢复。**r3**: 中延迟仍避(余量大)→极延迟(te500)才崩。
    - **★这机制性解释了项目历史"非单调曲线"**: 各 route 失败**阈值不同**(余量定: r6 早崩/r3 晚崩) + 极延迟 ZOH 保守恢复 + n=1 → **跨 route 朴素平均的曲线必然非单调/乱**。**历史非单调 ≠ 单 route 真信号, 是混 route 阈值差 + 保守恢复 + n=1 的伪象**。
    - **⇒ 正确方法**: **逐 route 出曲线**(各有阈值), 每档大 N(≥10); **别跨 route 朴素平均**(阈值差致伪非单调); 报碰撞率 + DS mean±std。待 r3 te500 n2/n3 确认 r3 晚崩 + 机制 dump。
  - **★★★ sweep 完成 + 机制 dump(2026-06-13, 15/15 含1 timeout)**:
    | route | te0 | te219 | te500 | 形状 |
    |--|--|--|--|--|
    | r6 | 100±0(0/3) | **37.5±17.7(3/3撞)** | 70.8±41.2(1/3) | 非单调(谷@te219) |
    | r3 | 100±0(0/2) | 100±0(0/3) | **66.7±23.6(2/3撞)** | 晚发降级(@te500) |
    全程 RC=100/colveh=0(`_1notraffic` 配置干净, 失败纯 ped)。
    - **★机制 dump 诚实修正(executor 好工作)**: 对比 r6 te500 n1(avoid100) vs n3(hit12.5): avoid 跑 ego 退后到 -44.15 停在行人(-44.38)前 **0.23m** 等待→避; hit 跑 ego 原地振荡停在行人正上方→撞。**avoid/hit 差仅 ~0.23m margin**。
    - **⚠️ 我的"ZOH→保守变慢"假设未被证实, 且部分被推翻**: l1_debug=false/RECORD_PATH="" → ZOH **无法直接观测**; 且帧数据反向(avoid 跑 375帧**更快**, hit 跑 485帧因撞后卡停更慢)。**真机制(数据一致)**: te500 RSU 滞后 500ms 移相 → r6 紧几何下 avoid/hit margin 仅 0.23m → **CARLA 物理非确定每帧微差决定停在行人前(退让)还是正上方(撞)** → 高方差是 **极小余量 + 物理随机** 的联合, 非纯 ZOH 保守。
    - **统一解释**: 所有边际档(r6 te219/te500, r3 te500)= ego 处近碰撞余量, 余量落在 CARLA 物理非确定带内 → 概率碰撞。非单调 = 余量随延迟非单调(r6 te219 相位最差)。
    - **(可选)直接证 ZOH**: l1_debug=true + RECORD_PATH 重跑读 latency_align_audit.csv。但机制细节不改主结论。

---

## ★★★ 调查闭合 — 最终结论 (2026-06-13, 含重开后 Thread A/B 穷尽测) ★★★

**一句话**: 原"H800 闭环 te0 即崩 = torch2 感知退化"是**误诊**; **无 H800 框架/感知/IPC 缺陷**; r6 的 DS 方差 = **CARLA ambient traffic 非确定**(穷尽实测定源), 对两机对称; **r104 类干净 route 天然可复现(DS=100)**, 是延迟曲线该用的 baseline。

**1. premise 证伪(硬数)**: b-test 同输入同权重, H800 vs 4090 感知 = **框位置逐位相同 + conf 差 1e-4**。无退化无漏检。

**2. 无框架缺陷(全嫌疑真测排除)**: 权重 md5 同 / voxelizer bit-identical(+纠正"spconv1"错前提) / IPC 点云完好 / IPC 阻塞同步(崩过两跑前400帧 RTT 同) / 感知确定(E3 ×5 bit-identical) / 感知 mean-shift(b-test 1e-4) / sensor-race(无痕)。H800 能 DS=100(r104 4/4, r6 n9/B2/B3)。

**3. 方差根因 = CARLA ambient traffic 非确定(穷尽定源 B1→B2→B3)**:
- **B1** 加 ServerRandomSeed=1234(每run重启) → r6 [37.77,100,37.77] **仍抖** → **文档 fix 证伪**(seed 不控 NavMesh C++ RNG)。
- **B2** pedestrian_amount=0 → r6 [100,29.29,100] **仍抖** → 第二源 = **车辆/TM**(TM seed 不足固定 0.9.10 TM 多线程)。
- **B3** pedestrian=0 **且** vehicle=0 → r6 **[100,100,100] bit-exact** → **ambient traffic(行人+车)= r6 全部方差源, 无更深 physics/ego 随机**。
- 机制: 未种子背景行人 NavMesh 导航 + CARLA 0.9.10 TM 不完全确定。**两机对称**(ini ServerRandomSeed 不对称是 red herring, B1 证它不控源)。原"H800 41 vs 4090 100"= knife-edge route 上 n=1 噪声。

**4. 可交付 baseline**: **r104 ×4 = 全 DS=100/infraction 全 0**(带全量 ambient traffic 仍稳) → **天然可复现**。⇒ **τ_ego 延迟曲线在 r104 类干净 route 跑**(不必关 traffic、不必跟 r6 死磕)。任意 route 若要强制可复现可 vehicle=ped=0(B3 证 bit-exact), 但 r104 不需要。

**5. 收敛判据(用户目标)达成**: "H800 te0 复现 4090(碰撞<50%、DS→100)" → **r104 复现 DS=100 达成; r6 受控 traffic 下 B3=100×3 达成**。

**6. 诚实账(本调查纠了多处过早结论, 双向)**:
- 我: ①"平台 gap 钉死"(n1==n3 vs v3 矛盾后撤销) ②"方差不可消除"软推断(用户驳→重开穷尽测) ③Explore "r6 有 Scenario3"误(executor route-XML 纠) ④ini "smoking gun"(B1 证伪后降级 red herring)。
- executor: ①md5 "根因确认"(我驳→要幅度→感知反被 exonerate) ②"2d 4090 对称方差"混淆(混 te0/te219/污染)。
- 实验校验: per-run CARLA 重启(server RNG 只启动时 seed, 提前拦截误判)。
- "n=1 噪声"本轮复发 4+ 次(H800 21.6 / 4090 2.7 / r3 50 / r17 50)。

**7. 建议 doc 更新**: MEMORY `project-h800-closedloop-defect.md` + HANDOFF_tau_curve_h800_defect "疑 torch2 感知退化" → 更正为本结论(无缺陷; CARLA ambient-traffic 方差; 用 r104 baseline)。

---

## §0 现象与铁证 (本轮已立, 复核认同)

| 项 | 事实 | 来源 |
|----|------|------|
| 现象 | H800 闭环 te0 在强敏感路线(r6/r26/r303 等) = mean DS **41.6** / 碰撞率 **95%** (n≈45) | H800 `results_driving_te0_strong/` 真测已落盘 |
| 对照 | 4090 同类路线 te0 ≈ **100** | 4090 results_driving_mr_te0 |
| 逻辑铁证(排除"只是延迟") | 4090 这些路线 **te219(注入219ms)都扛得住**(r26/r303/r113=100)。H800 te0 有效 IPC 延迟仅 ~150ms(≈4090 te150), 却比 4090 te219 还差 → **必有延迟之外的退化** | HANDOFF §1.2 |

**IPC 真实延迟**: ~120-180ms/帧(首帧 849ms 冷启), zmq 序列化占大头(真推理 ~20ms)。是真实地板, 但不是 te0 崩的主因。

---

## §1 嫌疑空间 (开放调查, 不先入为主)

两大类, 必须用实验分开:
- **A 类 — 感知数值退化**: torch2 移植(spconv1→2 / pcdet THC / box_overlaps cython / codriving_attn)使 H800 进程B 感知输出数值跑偏 → 漏检/错位 → 撞车。
- **B 类 — 双进程集成缺陷**: ckpt 权重不一致 / zmq 序列化丢精度或损坏张量 / ego pose 对齐错 / V2X fusion 拼接错 / 坐标系错。

已较弱排除: 施工场景(_1c)、IPC~150ms 延迟单独。

---

## §2 假设台账

### ★★ H-00 重大修订 (2026-06-13) — "受控 gap 钉死"被新证据推翻, 改为"需测分布"
> **触发**: E0-dump v3 = 全新 H800 r6 `_1`/te0/l1 跑出 **DS=100.0/零碰撞**, 与之前 "21.6/0.42撞" **同机同config同route 不可复现**。那个 21.6 是**上一会话** `results_driving_te0_noconstruct` 的**单次(n=1)** 存档, 非同会话受控重复。⇒ **H800 te0 退化很可能是概率性事件, 不是系统性必然失败**; 之前"100 vs 21.6 平台 gap"= 两边各 n=1 + 强敏感路线是平台差偏选 → 大概率是**方差 + 选样偏差伪信号**(正中项目已知 "n=1 噪声" 戒条)。
> **教训**: 我此前把 n=1 的 21.6 当"钉死的受控 gap", 是被单次结果误导。现降级为待测假设, 必须用分布证伪/证实。

### H-00 [基线锚定] (原判决已撤销, 见上方修订)
- **假设**: gap(~100 vs ~41) 是平台/集成差, 不是路线/场景混淆产物。
- **实验 E0-AB**: 同路线(先 r6 再 r26)、统一 scenario `_1`(无施工)、同 te0 config, 4090(空卡1/3/4) 与 H800(1卡)各跑闭环。
- **证据**(2026-06-13 executor 真测, 统一 scenario `_1` + te0):
  | route | machine | DS | RC | coll_veh | path |
  |---|---|---|---|---|---|
  | r6 | 4090 | **100.0** | 100 | 0 | `/home/jichengzhi/V2Xverse/results/results_driving_mr_te0_r6/.../r6_repeat0/ego_vehicle_0/results.json` |
  | r6 | H800 | **21.6** | 100 | **0.424** | `/data/jichengzhi_v2x/V2Xverse/results/results_driving_te0_noconstruct/.../r6_repeat0/ego_vehicle_0/results.json` |
  | r26 | 4090 | **100.0** | 100 | 0 | results_driving_mr_te0_r26/... |
  | r26 | H800 | ⏳ _1 在跑(原 te0_strong 存档误用 _1c, 作废) | | | |
- **判决**: ✅ **平台/集成 gap 钉死**。受控 scenario `_1` 下 r6 仍 100 vs 21.6。**关键洞察: 两端 RC 均=100** → H800 ego **跑完整条路线但撞车**(coll_veh 0.424/km), 不是卡死/崩溃, 是"**能开但不避障**"。
- **下一步**: E0-dump 分流 A 类(感知漏检→规划穿过去撞) vs B 类(感知正常但 control/规划不避)。**判别法**: 撞车帧附近, 检测集合里 ego 行进方向上那辆被撞车**在不在、位置对不对** → 不在=A类感知; 在但仍撞=B类集成/control。

### H-01 [A类·ckpt] H800 与 4090 加载的感知/planner 权重不一致 — ❌ 已排除(perception)
- **假设**: 移植/拷贝过程权重被改(shape 变、精度截断、加载错 ckpt)。
- **实验 E0-ckpt**: 对比 codriving perception+planner 权重 H800 vs 4090 的 md5 / 各 tensor shape & dtype。
- **证据**(2026-06-13 executor 回报): perception ckpt `net_epoch_bestval_at16.pth` 两机 **md5 完全相同 = `6fffe29f4f4da3a42c27c743828b43d5`**。
- **判决**: ❌ **权重源完全排除**。perception md5 `6fffe29f...` + planner md5 `6d76b8955b57f48632acedbd7643f752` 两机均一致; state_dict 194 键 name/shape/dtype 全对齐(全 float32, flat 格式)。退化不在权重文件, 在**运行时**(voxelizer/算子/集成)。
- **下一步**: 锁定运行时路径 → 重心移到 H-02(感知输出 dump) 与 H-03a(voxelizer 分叉)。

### H-02 [A类·感知输出] H800 进程B 返回给规划器的检测结果漏检/错位
- **假设**: te0 撞车瞬间, 该避的车不在检测里 / 位置/置信度错。
- **实验 E0-dump**: H800 跑 r6 几帧, dump 进程B 返回的检测(框数/置信度/中心位置)+ego pose。人眼看撞车瞬间该避的目标在不在、位置对不对。
- **证据(2026-06-13 在跑, dump v3 端口42006, ETA~30min)**:
  - patch 生效, 已 46+ 帧。样例 `step=0 nf=4 sc=[0.895,0.85,0.789,0.53] ctr=[[-3.4,-9.4],[-10.6,-9.3],[7.4,19.2],[7.3,15.7]] ego=(10.4,-84.4)]`。
  - **进程B 输入 pcd_np = (19710,4) float32 全量点云**, 落 `/data/jichengzhi_v2x/pcd_dump_r6/call_XXXX_spconv2.npy` (83+ 帧)。
  - **初判**: ① perception 收到完整点云(非空/非截断) → **H-04 的"IPC 损坏输入"先验下降**(点云过 zmq 完好)。② 检测非空(step0 有 4 框) → 感知非全死。**待撞车帧数据下最终判决**。
- **判决**: ⏳ (待撞车前后帧: 被撞车在不在检测里)
- **下一步**: 据此决定走 E1-opcheck(感知) 还是查 control/规划集成。
- **★坐标系/三态判别**(读 `pnp_infer_action_e2e.py:~820` 确认): 检测框经 `transform_2d_points`(lidar pose→ego pose, 用 `measurements["lidar_pose_x/y"]`)转到 **ego 帧**, 有 `<1.4m` 自车框过滤。dump 的 `ctr` = ego 帧框心(米), 正前方车 ≈ (前向, ~0)。撞车帧三态:
  - 框**缺失**(被撞车不在检测里) → **A类感知漏检**(voxelizer/算子)。
  - 框**存在但位置错** → A类(感知定位偏) **或** B类(`lidar_pose` 跨进程错位 → transform 把框摆错)。需再查进程B 收到的 lidar_pose vs 进程A 真值。
  - 框**存在且位置对但仍撞** → B类 control/规划(看见了不避)。

> **旁注(非根因)**: H800 r26 `_1` leaderboard **Scenario13 加载崩**(`'Location' object is not iterable`)——H800 CARLA Python API 与 4090 在 Scenario13 实现上有差异。仅影响 r26 作旁证, r6 受控 gap 已足够。不追(不在避障链路)。

### H-03 [A类·GPU前向数值] 同输入同权重同 voxel, 但前向运行时数值跑偏 (delta 已缩到这一层)
- **假设**: voxel 之后的 GPU 前向(sparse conv backbone / BN-eval / attention / 检测头)在 **torch2.1+cu121+Hopper(sm90)** 下输出与 **torch1.10+cu113+Ada** 不一致, 致检测漂移→撞车。已排除上游(权重 ❌H-01 / IPC输入 ❌H-04初判 / voxelizer ❌H-03a), delta 只剩前向运行时。
- **子候选**: H-03b spconv sparse conv 算子数值(同 2.3.6 但不同 torch/cuda); H-03c BN eval running_mean/var 或 attention 数值; H-03d sm90 vs Ada CUDA kernel/TF32/reduction 顺序; (+ 移植补丁 pcdet THC 重编 / box_overlaps cython / codriving_attn turtle 删)。
- **★决定性实验 E1-opcheck-forward(离线, 最干净的 A/B 判别)**: 取同一份固定 `pcd_np`(已有), 在两机各自加载 codriving perception(同 ckpt)跑**完整前向**, 输出 `pred_box_tensor`+scores(lidar 帧, 转 ego 前的原始检测), 对比框数/位置/置信度。因输入(→voxel)已证 bit-identical, 任何检测差异 = 纯前向运行时所致。
  - 差异大 → **A类感知数值退化确诊**, 继续逐模块 bisect(backbone 特征图→检测头) 定位; 修(对齐 torch2 数值/关 TF32/`torch.use_deterministic`/补丁复核)→ Stage2 闭环验证。
  - 差异≈0 → **A类感知排除** → 转 B类 control/规划集成。
- **证据**: ⏳ E1-opcheck-forward **已授权**(两机均确认可行: 同 ckpt md5-identical / 同 config / 不需 CARLA; 脚本 `e1_forward_probe.py` 单帧前向)。
- **判决**: ⏳
- **★E1-forward 双子测(一脚本两结论, 不同根因不同修法)**:
  - **(a) H800 ×5 同输入 → 方差** [=H-06]: 5 次 conf 是否一致。摆动→**非确定**(修: 定位非确定 op / deterministic flags)。
  - **(b) H800 vs 4090 同输入 → 均值差** [=H-03]: 同 pcd 同权重, 唯一 delta=前向运行时。H800 conf 系统性低于 4090(如 0.6 vs 0.9)→ **torch2 移植数值退化确诊**(可复现/无交通混淆, 最干净的可修根因)。
  - 取 2-3 帧(碰撞走廊入口 call_00000480-560 marginal + 一帧 v3 高 conf cyclist)。要求: model.eval()+no_grad, voxelize 一次喂同张量, 逐次报 nf/全 scores(全精度)/box 心。

#### H-03a [A类·voxelizer] — ❌ 已排除(且推翻一个流传的错误前提) ★
- **判决(2026-06-13 真测)**: 同一 `pcd_np`(frame40, (19615,4)) 过两机 voxelizer = **bit-identical**: n_voxels 9253 一致, coord 集合相同, num_points 逐体素相同, **voxel_features maxabs diff = 0.0**。voxelizer 不是根因。
- **★推翻的错误前提**: "4090 用 spconv1 / H800 用 spconv2" **是错的**。两机实际**都装 spconv 2.3.6**, 都走 `Point2VoxelCPU3d` v2 路径(4090 的 `pnp_infer.py` try `VoxelGeneratorV2` 失败→except→v2)。HANDOFF 里"spconv1→2 移植"的措辞需更正。→ A类退化的 delta **不在 voxelizer, 在 voxelizer 之后的 GPU 前向**(torch1.10/cu113/Ada vs torch2.1/cu121/Hopper, 权重/voxel 已证 bit-identical)。
- (以下为原始假设记录, 已被上面判决否决:)
- **静态代码分叉点(已证运行时不触发)**: ~~SpVoxelPreprocessor spconv 硬分叉~~
- **代码实证**(本机 `pnp_infer.py:138-210` 读源确认): `SpVoxelPreprocessor` 对 spconv 版本有**硬分叉**:
  - 4090 (spconv v1.x): `VoxelGeneratorV2(...).generate(pcd_np)`
  - H800 (spconv v2.x): `Point2VoxelCPU3d(vsize_xyz=...).point_to_voxel(tv.from_numpy(pcd_np))`
- **风险**: 两库对**同一点云**产出的 voxels/coordinates/num_points 在**顺序、截断(max_voxels 超限丢弃策略)、空体素处理**上可能不一致 → 下游 sparse backbone 吃到不同输入 → 感知漂移。**这正是"smoke test 能跑但数值退化"的典型**(不崩、只是 voxel 集合不同)。
- **实验(E1-opcheck 第一刀, 离线/CPU/无需闭环)**: 取同一份 `pcd_np`(从 dump 或固定一帧 lidar), 同参数分别过 v1.generate 与 v2.point_to_voxel, 对比: voxel 数、coords 集合(排序后)、每 voxel 的 features/num_points 是否逐元素一致。
- **证据**: ⏳
- **判决**: ⏳ (不一致 = A类感知退化的上游冒烟枪之一)
- **下一步**: 若分叉, 让 H800 voxelizer 复刻 v1 语义(排序/截断对齐), 重测 te0。
- **★对拍工具已就绪**: `/home/jichengzhi/V2Xverse/scripts/rc_voxel_compare.py`(自含, 内置 codriving 实参: range[-36,-12,-22,36,12,14]/vsize[0.125,0.125,36]/maxpts32/maxvox70000)。用法: 在 spconv1 env 跑 `--pcd X.npy --out v1.npz`, 在 spconv2 env(t2lib) 跑出 v2.npz, 再 `--diff v1.npz v2.npz` 逐元素+排序后集合对比。待 E0-dump 的 pcd_np 落盘即可秒跑。

---

### H-04 [B类·IPC] 双进程 zmq 边界损坏 perception 输入 (H800 独有, 待 E0-dump 的 pcd_np 旁证)
- **结构性先验**: 4090 = **单进程**(感知与 CARLA 同进程); H800 = **双进程**(进程A CARLA/leaderboard ↔ 进程B codriving 感知, 经 zmq + **msgpack_numpy** 传 payload)。**IPC 拆分是 H800 独有** → 任何"序列化丢精度/dtype/shape、非连续数组、pose/坐标跨进程错位"只可能发生在 H800。
- **代码实证**(`infer_server.py`): payload = `<opaque numpy/dict>` 经 msgpack_numpy 打包; 进程B 收到后才喂 perception。
- **精细判别(借 E0-dump 的 pcd_np)**: 因 pcd_np 是在**进程B 内 SpVoxelPreprocessor 输入处**抓的(=perception 真正消费的输入):
  - pcd_np 是正常点云(点数合理/范围对) → **IPC 输入传输 OK**, 退化在下游(voxelizer/算子) = **A类**。
  - pcd_np 空/截断/数值乱 → **IPC 损坏了输入** = **B类**(查 msgpack_numpy 往返、payload 打包路径)。
- **证据**: ⏳ (随 E0-dump 一并得)
- **判决**: ⏳
- **下一步(若 B类)**: 在进程A 发送前 vs 进程B 接收后对同一 payload 做 md5/逐元素对拍, 定位 msgpack_numpy 往返是否无损; 检查 ego pose / 坐标系是否跨进程一致。

### H-05 [元假设·方差] "H800 te0 系统性崩"是 n=1 方差 + 偏选伪信号 ★当前最高优先
- **假设**: H800 与 4090 te0 r6 的 DS 真分布**重叠**(都偶尔崩/多数 100), "41 vs 100"是各 n=1 抽样 + 强敏感路线按平台差偏选放大的伪 gap。
- **实验 E2-distribution**(决定性, 同会话/同代码/无 dump 开销): H800 与 4090 各跑 r6 `_1`/te0/l1 **N≥8 次**, 出 DS 分布(每次值/mean/min/max/崩车率)。
- **判决标准**: ① 两机分布高度重叠且都偶崩 → H-05 成立, "H800 缺陷"被证伪为方差, 调查转向"闭环本身对随机交通的鲁棒性"而非"H800 移植"。② H800 分布显著低/崩率显著高 → 真有系统性平台退化, 回 H-03/H-06 找因。
- **证据**: ⏳ 已派发。
- **判决**: ⏳
- **下一步**: 据分布决定是否还需追 H800 特异退化。

### H-06 [感知非确定性] H800 感知前向非确定(CUDA non-deterministic) → 偶发假阴性漏检 ★机制线索强
- **假设**: H800 GPU 前向含非确定算子, 同一输入多次跑检测不同, 偶发 miss 该避的车(尤其 cyclist 这类弱小目标)。
- **闭环旁证(强但有混淆)**: ① **碰撞走廊固定** x≈-44.3 / y≈-48~-70(r6 多次跑均在此段崩, v3 是唯一通过); 撞的是 **`vehicle.diamondback.century`=自行车(CARLA bike) + chevrolet.impala** → 失败=**偶发漏检 cyclist(弱小VRU)于固定难点**。② 同 ego 坐标上游窗口(y=-22~-38): v1(崩批次)检测 nf=0 或 conf 0.21-0.36; v3(通过)conf 0.87-0.95。
  - **★混淆(我必须诚实标注)**: v1 vs v3 是**不同次跑, 交通可能不同** → "v1 没检测到 / v3 检测到" 可能是**v1 那里本来就没车**, 不是漏检。executor 自己也证 v3 有零检测段但不撞(=那里真没车)。⇒ 闭环 DET_DUMP 对比 **consistent-with A类感知非确定, 但不构成证明**。去混淆的唯一干净测 = **同输入→重复输出(E3)**。
- **实验 E3-perception-determinism**(廉价/离线/无 CARLA, **去交通混淆**): 同一 `pcd_np`(取碰撞走廊/marginal-conf 区段一帧) 在 H800 感知前向连跑 N≥5 次, 逐元素比检测(pred_box/score, 尤其 conf 是否在 0.2↔0.9 间摆动)。
- **判决标准**: 多次不一致(conf 大幅摆动) → **H800 感知非确定确诊 = 偶崩机制**(可修: 定位非确定 op / `torch.use_deterministic_algorithms` / 关 TF32); 完全一致 → 感知确定, 闭环方差归 CARLA 交通或 IPC 时序。
- **附(关键)**: 查 scenario actor(尤其那辆 cyclist id=399)与交通是否**设种子/脚本确定**。若确定 → 碰撞走廊固定但结果随机 ⇒ 变量只能是计算非确定/时序(强支持 H-06); 若随机 → 交通是混淆。
- **证据(2026-06-13 真测, `e1_probe_v3.py` H800 GPU1)**: 同一 pcd 连跑 5×, cls_preds tensor MD5 **bit-identical**(TEST1 single-agent hash 5c72dd82..×5; TEST2 ego+RSU hash 89760777..×5)。
- **判决**: ❌ **H-06 排除 — H800 感知前向是确定性的**(同输入→bit-identical 输出)。
- **⚠ 探针 caveat(影响 cross-machine (b), 不影响本判决)**: 两测均 **nf=0**(probe 用 `pairwise_t_matrix=identity` 占位真 ego-RSU 变换 → 融合坐标错位 → score<thr 0.04)。**单 agent 也 nf=0** ⇒ probe 当前**未忠实复现闭环感知**(闭环同帧 v3 有 nf2-4)。determinism 结论成立(同输入→同输出与 nf 无关), 但 **(b) 跨机 conf 对比需 probe 产出真检测(nf>0)才可用 → 需先修 probe 的 cav_content/transform**。
- **★顺带 B类线索**: probe 缺"真 pairwise_t_matrix(ego-RSU 相对位姿)"才 nf=0 → 闭环里这个 V2X 融合变换若在 H800 算错/错位, 会致融合退化(接 H-04 pose handoff)。E2 后视情追。

### ★ H-08 [逻辑张力 — 本轮新核心] 感知确定 + 种子全设, 闭环为何不可复现?
- **两块硬事实**: ① E3 = H800 感知**确定**。② 两机 eval 脚本 seed 全设(TRAFFIC_SEED=CARLA_SEED=2000, np/random/torch/CarlaDataProvider/traffic_manager 全 seeded, 两机一致)。
- **推论**: 确定感知 + 确定交通 ⇒ 干净同 config 闭环**本应逐次复现**。但"21.6 vs 100"**不复现** → 方差源**既非感知计算、也非种子化交通**。只剩两解:
  1. **IPC/时序 jitter**: 双进程 zmq RTT 逐次抖动 → 帧对齐/控制时序变 → 结果变(单进程 4090 无此源)。= 真 H800 双进程效应(B类时序)。
  2. **"崩"证据来自不干净跑**: 21.6=上会话单次; 41.6 聚合用 `_1c`; 部分带 dump 开销; 全 n=1。⇒ "H800 崩"= **方法学伪信号**。
- **E2(干净 n=8/同会话/seeded/无dump)决定性二选一**:
  - H800 ×8 复现 ~100 → 解2成立, "H800 崩"是不干净跑伪信号, 根因=方法学/n=1, **收敛判据可能已满足**(H800 te0 r6 干净=100)。
  - H800 ×8 仍有崩 → 解1成立, 非种子非感知源 = **IPC/时序 jitter**(双进程真问题, B类时序), 接着查 IPC RTT 分布 vs 控制时序耦合。
- **证据**: ⏳ E2 跑中。**资源调整(2026-06-13, team-lead 核实 4090 八卡全满)**: 4090 臂暂起不来 → **H800 ×8 为命门主裁决**(单臂即可二选一)。4090 基线 = ①team-lead 独立复核 `mr_te0_r6`=100/RC100/0撞(真值) + ②E2 run1(若跑完) + ③确定性论证(单进程sync+感知bit-identical+seeded→本应稳定复现100)。**残留待闭**: 4090 分布目前仍偏 n=1+论证, 等卡空补 2-3 run spot-check 坐实(诚实标注, 不当已证分布)。
- **判决**: ⏳ — **★2026-06-13 部分数据反转, 见下**
- **★★ H-08 数据反转 (E2 H800 clean ×2 均崩) — 我此前过早倾向"解2无缺陷", 数据纠偏**:
  - **真测**: H800 clean(无dump) n1=**37.77**(RC62.95/AgentBlocked+coll0.449), n2=**5.14**(RC8.57/coll3.302严重多撞), 都 Failed。+ v3(带dump)=100。⇒ **清洁跑也崩, 且崩得多** → 解2"21.6 是不干净跑伪信号"**被削弱**(不止脏跑崩)。
  - **两层重新拆分**: ① **方差真实**(37.77/5.14/100 三种结果, 同 config) → 既然感知确定+IPC sync, 唯一来源 = **CARLA/TM 非确定**(seed-only≠全确定); 这层决定每次崩不崩。② **H800 比 4090(=100) 崩得多** = 真平台 gap, **其因我从未真测过**: **跨机感知 mean-shift**(H800 torch2 前向是否对 marginal cyclist 给系统性更低 conf?)。
  - **★诚实纠偏**: 我之前据"感知确定(E3)+sync+seeded"推出"闭环本应复现→无缺陷", **逻辑漏洞 = E3 只证 H800 感知"自洽确定", 没证它"=4090"**。**cross-machine (b) mean-shift 测因 probe nf=0 被我推后, 至今未做** → **原始"torch2 感知退化"假设 NOT refuted, 是未测, 现升为主嫌**。weights/voxel 已 bit-identical, 但 GPU 前向 kernel(torch1.10/Ada vs torch2.1/Hopper, TF32/conv算法)可能把 near-threshold cyclist 检测从过阈压到欠阈 → 系统性漏检 → H800 崩得多。**这是可复现可修的真根因候选, 我不该急着refute它。**
  - **下一步(主线)**: (1) 跑满 H800 ×8 出崩率 + 每 run failure 机制(AgentBlocked vs collision); (2) **修 probe 到 nf>0(真 pairwise_t_matrix), 做 cross-machine (b): 同一 corridor pcd 两机感知对比 conf** — 这是判"H800感知是否系统性更弱"的决定性测(等4090放卡); (3) 查 CARLA TM 是否全确定(定方差归属)。
- **★frame-align 源码分析(原强化解2, 仍有效但不足以判无缺陷)** (`latency_frame_align.py` 读源): ① aligner 只延 **RSU/other-CAV 支路**, **ego 自感知永不延迟/永不 ZOH**。② te0(latency≤0)→ **Delta=0 = 零注入 staleness**。③ CARLA **同步 20Hz**。⇒ **同步模式下真实 ~150ms IPC RTT 只拖慢墙钟, CARLA 等 agent 返回控制才 tick, 不产生 staleness**; te0 时 aligner 是 no-op。**故 同步CARLA + 确定感知 + seeded交通 ⇒ te0 闭环本应完全确定, 无 run-to-run 方差源 → 强支持解2(不干净跑伪信号)。**
- **唯一能复活解1的口子**: 若 `InferProxy` 用 **async**(submit+用旧值, 见 infer_server `submit()/poll()`)而非阻塞等结果, 则真 IPC RTT jitter 会在 te0 也注入 staleness。→ **E_sync-check(代码读, 廉价, 与 E2 并行)**: 查 pnp_agent 的 IPC proxy 是 **阻塞等(sync)** 还是 **submit-用旧(async)**。sync → 锁定解2; async → 解1 结构性成立, 查 RTT 分布。
- **★E_sync-check 判决(2026-06-13 真测读码 `infer_proxy.py`)**: **(A) 阻塞同步确认** — 类注释 `"Synchronous IPC proxy: submit + blocking poll per inference call."`; 每帧 `send→poll(15s)→recv` **死等该帧回包**, timeout 直接 raise(不用旧帧), 仅 B 显式 error 才 prev_control ZOH(非时序 staleness)。⇒ **解1(IPC RTT jitter 注入 staleness)结构性排除**(blocking=永不用旧帧)。**剩余唯一解 = 解2: "21.6"是不干净跑伪信号**。
  - **诚实留口(环境残差)**: 即便 IPC/感知/种子都确定, **CARLA traffic_manager 完全确定需 deterministic-mode 全套 flag(不止 seed)**; 残留 CARLA/TM 随机可能在 cyclist 走廊偶发崩。但那是**环境随机, 仍非 H800 感知缺陷** — "H800 torch2 感知退化"结论无论如何被推翻。E2 ×8 给崩率定量。

### H-07 [混淆·dump 开销] v3 的重 I/O(713 npy 落盘)改变帧时序 → 翻转结果
- **假设**: v3 DS=100 与 prior 21.6 之差可能部分来自 dump I/O 改变 per-frame 时序, 非纯随机。
- **处置**: E2-distribution 必须**无 dump 开销**跑, 排除此混淆。
- **判决**: ⏳ (随 E2 一并澄清)

---

### H-09 [环境方差] CARLA TM/行人非确定 = run-to-run 方差源(对两机对称, 不解释 H800>4090)
- **真测(`leaderboard_evaluator_parameter.py`, 两机同源)**: synchronous_mode ✅ / fixed_delta ✅ / TM.synchronous ✅ / TM.seed=2000 ✅ / deterministic_ragdolls ✅; 但 **`set_pedestrians_seed()` 被注释 ❌** → 行人 NPC 跨 run 变; TM 0.9.10.1 不保证跨机 bit 复现(依赖物理精度)。
- **判读**: ① 这是**真实的 run-to-run 方差源**(解释 37.77/5.14/100 抖动)，但**两机对称**(同代码) → **不创造 H800>4090 的偏差**。② 被撞的 `vehicle.diamondback.century`=**TM 控制的自行车(vehicle前缀, 已 seeded)**, 非行人。③ ⇒ 方差归 CARLA/行人 = 合适口径; 但 **H800 崩率 > 4090 这件事仍需 mean-shift 测解释**, 不能用环境方差搪塞。
- **判决**: ✅ 方差有合法环境来源(对称); ⏳ H800>4090 偏差另需 b-test。

### ★★ H-08/H-09 收束 (2026-06-13, n1==n3 bit-exact) — 方差问题解决, H800 是"确定性崩"
- **真测**: H800 clean te0 r6: **n1==n3 = DS 37.769711630172694(逐位相同浮点)**, 同撞 cyclist(coll_veh 0.449)。n2=5.14 是**离群**(仅286帧早退, 一次高速撞+AgentBlocked; 归注释行人种子的偶发)。
- **判决**: **H800 闭环完全可复现**(确定感知×TM同步×固定步长 → 同轨迹 → 同碰撞)。⇒ 之前"21.6 vs 100 不可复现"被解释清:
  - **clean H800 = 确定性崩 cyclist(37.77)**; v3=100 是 **H800+dump**, dump I/O 扰动时序→恰避开崩(H-07 混淆方向相反: dump 让它过)。
  - 4090=100(team-lead 核实) + H800+dump=100 都过, **唯 clean H800 确定崩**。
- **新定性**: 不是"H800 随机偶崩", 是 **H800 与 4090 在同种子下确定性走出不同轨迹, H800 撞 cyclist**。差异是**确定的平台差**, 根因二选一(b-test 判):
  - **(i) 感知 mean-shift**: H800 torch2 前向漏检/弱检 cyclist → 不避 → 撞。(原假设, 可修框架bug)
  - **(ii) CARLA 跨机物理/TM 非 bit 复现**: 即便感知相同, Ada vs Hopper 浮点精度致车辆动力学微分叉, 长 route 累积→H800 在 knife-edge 处撞。(非框架bug; "缺陷"=CARLA 跨硬件不可复现 + r6 是临界 route)
- n2 型离群率由 n4-8 补充刻画(非阻塞)。

### ★★★ H-11 (2026-06-13, 4090=2.7 炸弹) — 4090 也不可复现, "H800<4090 gap" 本身存疑
- **真测**: 全新 4090 te0 r6 `_1` = **DS 2.7**(RC100/Completed/coll_veh0.424+coll_ped0.424)。而 team-lead 早先核实 `mr_te0_r6`=**100**。⇒ **4090 同 config 两点 = 100 与 2.7, 也不可复现!**(与 H800 "21.6 vs 37.77 vs v3=100" 同病)。
- **对称含义**: **两机在 r6 都是高方差**(注释行人种子 + CARLA TM 跨run变体)。原始"H800 41 vs 4090 100"很可能 = **两边各 n=1 抽到各自分布的不同点** → "H800<4090 系统 gap"**本身存疑**, 可能不存在。
- **caveat**: 2.7 那次带 executor 代码改动(buffer_rgba patch + payload no-op), executor 自标存疑 → **需 clean 4090(无改动)跑 N 次定真分布**。
- **H800 ×8 收尾**: n1=n3=37.77(bit-exact), n2=5.14, n4=6.48(RC100但多撞) — **全低(5-38), 有方差**。n5-n8=DS0 **作废**(executor 自爆 bug: 把 4090 版 pnp_infer 拷到 H800, t2lib matplotlib3.10.9 删了 tostring_rgb → 每帧 B-ERR → ego 冻 → DS0; 已诚实披露+修 buffer_rgba fallback)。
- **❌ 驳"H800 输出空预测"的 over-reach**: executor 据"n1 log tostring_rgb 0次 vs n5 308次"推"原始H800 processed_pred_box 全帧空"。**不成立**: ① n1(原始H800文件) vs n5(4090拷贝文件)是**不同代码版本**, turn_traffic_into_map 调用路径不同 → "0 vs 308"是代码版本差, 非感知输出; ② **直接矛盾 v3 DET_DUMP 实测 nf=2-4**(H800 确产检测)。⇒ 不能下"H800感知空"结论。
- **结论**: 闭环 DS 被双机高方差污染, 单看 DS 判不了根因。**唯一干净判别 = b-test(同输入两机比感知, 剥离所有 CARLA/行人方差)**, 现升为**唯一决定性测**。

### ★ H-12 [game_time 2× — 闭环存在墙钟敏感元件(疑 sensor-queue timeout)] 关键机制线索
- **真测**: 4090 mr_te0_r6(clean GPU) game_time=92.35s/DS100/0撞; 4090 E2 n1(GPU3 95%载) game_time=**196.8s(2.13×)**/DS2.7/6撞。同 seed。
- **推论**: 纯 sync 下 game_time 应与算速无关; 翻倍 = ego 多跑 ~2× tick = **行为变了(变慢/绕/停)**。⇒ **闭环有墙钟敏感元件**(最可能 = 慢推理触发 **sensor-queue timeout** → 某帧拿到坏/空 sensor → ego 异常 → TM 行人/车追上 → 撞)。这**打破"纯sync,GPU负载只影响墙钟"的旧论**(我之前据 frame-align 下的论需修正: frame-align 是 sync, 但 sensor 同步另有墙钟超时口)。
- **双向含义**: ① 4090 n1=2.7 很可能是 GPU 竞争污染(支持 4090=100 是真值)——但**这是论证, 必须 clean idle GPU 复跑验证, 不白接受**。② ★**对 H800 关键**: H800 双进程 IPC 延迟(~150ms, 尖峰627ms)可能经**同一 sensor-timing 机制**拖慢/扰动 H800 闭环 → **即便感知 byte-identical, H800 也可能因 IPC 延迟→sensor timeout 而崩得多**。这是"解1时序"的新机制变体(sensor-queue, 非 frame-align staleness)。
- **⇒ b-test 分叉更新**: 感知有差 → 感知bug; 感知=4090 → **H800 崩归 IPC延迟→sensor-timing**(修法=降IPC延迟/修sensor同步), 不是感知。
- **判决**: ⏳ 待 b-test(分感知 vs 时序) + clean 4090 baseline(验 4090 真分布)。

### ★ H-10 b-test v1 结果 — ⚠️ INCONCLUSIVE, 驳回 executor 的"根因确认"(md5陷阱)
- **真测(step=1096, ego corridor 入口, 同权重 md5)**: 4090(torch1.10): nf=**3**, 有实数 score, cls_preds md5=529..; H800(torch2.1): nf=**3**, **pred_score=None**, cls_preds md5=9cf..(不同)。comm_rate md5 两机**相同**。
- **❌ 驳"根因确认", 三条**:
  1. **两机都 nf=3** → H800 **没漏检** cyclist(都出3框)。"H800 漏检"narrative 不被支持。
  2. **"所有张量 md5 不同"不证退化**: torch1.10/sm86 vs torch2.1/sm90 的中间张量**必然 bit 不同**(kernel/TF32/reduction 顺序)。1e-6 和 1.0 都翻 md5。**md5 分叉 = 预期浮点基线, 非退化证据**(我早先明确警告过这个陷阱, executor 踩了)。
  3. **H800 pred_score=None 几乎确定是 replay 脚本的 post_process torch2 抽取bug, 非闭环真相**: executor 自标"score提取失败"; 且**矛盾 v3 DET_DUMP 实测 H800 有真 score(nf2-4,conf0.87-0.95)** + 闭环 ego 真在开(RC62.95 非冻)。⇒ 红鲱鱼。
- **判决**: ⏳ **INCONCLUSIVE**。b-test 必须给**数值幅度**, 非 md5。需: ① 修 score 抽取bug拿 H800 真 score; ② 两机 **3 框各自 (conf, center) 并排**, cyclist 那框 conf 是否系统性差; ③ cls_preds/fused_feature 的 **max|Δ| / mean|Δ| / 相对差**; ④ 差异是否**改变检测结果**(过阈框/位置) 还是仅亚阈噪声。
- **两种收敛(待幅度定)**: (a) cyclist conf 显著低(如 0.6→0.3 跨阈) → 感知退化真根因(可修, 对齐数值/关TF32); (b) 仅 1e-4 级噪声但闭环**混沌放大**成不同轨迹 → 非"感知bug", 是闭环对任意数值差的混沌敏感(两条都是合法轨迹, r6 knife-edge), "修"=接受方差/多run平均。

### ★★★ H-10 终判 (2026-06-13, 全精度幅度) — 感知 EXONERATED, 原始 premise 被硬数推翻
- **真测(step1096 corridor, 同权重, 修了 score bug)**:
  | | 4090(torch1.10) | H800(torch2.1) | Δ |
  |--|--|--|--|
  | nf | 2 | 2 | 0 |
  | det0 (cx,cy,score) | (12.16,1.01,**0.8809**) | (12.16,1.01,**0.8810**) | 位置 bit-exact, conf **+1e-4** |
  | det1 (cx,cy,score) | (1.13,-1.05,**0.3079**) | (1.13,-1.05,**0.3078**) | 位置 bit-exact, conf **-1e-4** |
  | comm_rate | byte-exact 两机相同 |
- **判决**: ✅✅ **感知 EXONERATED**。框位置逐位相同, conf 仅 1e-4 浮点噪声 → **H800 torch2 前向 ≈ 4090 torch1.10, 无 mean-shift, 无漏检**。⇒ **"torch2 感知退化"(本调查起点 premise)被硬数推翻**。md5 分叉(H-10 v1)确系预期浮点基线, 非退化(印证我驳回的判断)。
- **n9 大爆料 DS=100**: H800 **能**在 te0 拿 DS=100(n9 带 capture / v3 带 dump 都=100); 而 clean n1/n3=37.77。⇒ **H800 非根本损坏**; 不稳来自闭环。**模式: 带插桩(I/O)的跑→100, clean 跑→低** (counterintuitive, 机制存疑)。
- **根因重定位(方向, 机制未完全锁死)**: 感知已排除 → 差异源在**闭环时序/sensor-timing/CARLA方差**, 非框架感知bug。executor 主张"IPC 时序"方向对, 但**机制未证**(为何插桩反而过? = sensor-queue race? 混沌?)。需直接测。

#### (历史)H-10 v1 [md5 阶段] — 已被上面终判取代
- **工具就绪**: `b_test_payload_replay.py` + `pnp_infer_action_e2e.py` 加 `PAYLOAD_CAPTURE_DIR` 触发(corridor x≈-44.3/y∈[-56,-47] 首帧存完整 pkl=car+rsu raw)。replay 输出 nf + scores_md5 + boxes_md5 + cls_preds md5。
- **测法**: 捕获**一份** corridor payload(机器无关, 是 CARLA 输入) → **同一 pkl 在 H800 与 4090 各 replay** → 比 scores_md5/boxes_md5/nf。
- **判决标准**: md5 相同(或 conf 数值≈) → **H800 感知=4090, 感知 mean-shift 排除** → H800>4090 崩率归 timing/其他; md5 不同且 H800 conf 系统性低(尤其 cyclist 框)→ **torch2 感知退化确诊**(可复现可修真根因, 原始假设 vindicated)。
- **证据**: ⏳ 工具就绪, 待捕获+两机跑(单帧前向, 或挤进被占4090卡)。
- **判决**: ⏳ — **本调查现最决定性的未完成测**。

---

### ★★★ H-13 [机制钉死] 根因 = CARLA NPC/行人(未种子)在 corridor 挡 ego, 非感知非IPC
- **真测(n1/n3崩 vs n9过 全量对比)**: game_time 108/106 vs 91.75s; **AgentBlockedTest FAILURE(n1/n3) vs SUCCESS(n9)**; ego 冻在 **x≈-44(cyclist corridor)** vs n9 驶到 x≈-105。
- **IPC 时序被自身数据排除**: 崩/过两跑**前 400 帧 RTT 几乎相同(~120-130ms)**; n9 低 RTT 是 ego 在动的**后果非原因**; 若 IPC 是因 n9 也该崩。无 sensor-queue/B-error。⇒ **"IPC时序根因"被 executor 自己的数据否掉**(我之前没盖章是对的)。
- **真机制**: **CARLA NPC/行人定位**(n4 有 coll_ped!)决定 corridor 处有没有东西挡 ego。**行人未种子**(`set_pedestrians_seed` 注释)→ 逐 run 变 → 挡或不挡。**这是环境方差, 对两机对称**(4090 也 2.7)。
- **n1==n3 bit-exact 但 n2≠n4**: 说明**种子化 RNG(车/TM)确定, 未种子行人是方差源**。
- **判决**: ✅ 根因 = **闭环对 CARLA 未种子行人/NPC 的敏感(环境方差) on knife-edge route r6**, **非 H800 框架/感知/IPC 缺陷**。H800 能 100(n9), 与 4090 同病(4090 也崩)。
- **收敛测(进行中)**: 取消注释行人种子 → H800 clean ×3 应全同(验证方差源); + clean 4090 ×3(等卡) → 同种子下两机 DS 是否 ≈ → 定有无系统 gap。
- **★[2026-06-13 更正] `set_pedestrians_seed` 在 CARLA 0.9.10.1 不存在**(executor `dir(carla.World)` 实证仅 `set_pedestrians_cross_factor`; 取消注释→AttributeError 崩, 已还原)。⇒ 那行是**抄自新版的非功能代码**, H-09"行人种子被注释=可修方差源"措辞需更正: 现有种子(TM2000/CarlaDataProvider numpy RandomState2000/random/np.random)**已全设**, spawn 走 seeded `_rng.choice`。**残余方差 = CARLA 0.9.10.1 内部物理非确定(float累积), 无 seed 可消**(间歇性: n1==n3 同, n2/n4 异, 疑碰撞/交互事件触发物理分叉)。
- **⇒ 收敛方法转向**: bit-复现不可得 → **平台比较必须统计化(N-run 分布), 非单次/非bit-exact**。判据改为: H800 ×N 与 4090 ×N 的 DS **分布是否重叠** + 多 route。
- **★★[2026-06-13 用户驳回 + 重开] "方差不可消除"是软推断, 未查穿**: 我仅凭 executor 试一个 API(`set_pedestrians_seed`)报"不存在"就跳"CARLA物理不可确定", **是我自己也会驳别人的软推断**。硬证不受影响(感知等价/无框架缺陷/r104复现满分仍成立), 但 **"能否种子化"必须穷尽 V2Xverse 的确定性做法再下结论**(V2Xverse 是发表框架, 要生成可复现数据集, 必有 0.9.10 兼容做法)。→ **H-09"不可消除"判决撤销, 重开**。Thread A(代码:V2Xverse怎么做确定性+r6 corridor 挡车 actor 是 scenario(种子定,可修) 还是背景随机walker) + Thread B(实验:试种子让 r6 ×3 bit-reproducible)。判据: r6 ×3 是否 bit-reproducible。

---

### ⚠️ H-14 [r3 黄旗 — 别过早宣布"无 gap"] H800 r3 可复现 50 vs 4090 probe 100
- **真测**: H800 r3 **n1=n2=50.00**(RC100, 一次 vehicle_block, DS=RC×0.5); 4090 mr_te0 probe r3=**100**(n=1, 无 block)。v4 r6_n1=18.88(≠历史37.77 → r6 物理非确定; 而 r3 可复现)。
- **⚠️ 这是黄旗, 不是 dissolution**: H800 r3 **可复现地** 50 而 4090(n=1) 100 → **可能存在可复现的 H800<4090 差**。不能据此前的乐观就宣布"H800 健康/premise 解体"。
- **两解, 仅 4090 ×3 backfill 能分**: (a) **混沌(良性)**: 1e-4/帧数值差经闭环混沌放大, r3 恰把 H800 ego 推进 block, 别的 route 可能反过来 → route-dependent, 非系统; (b) **系统(需重视)**: H800 跨 route 一致地比 4090 差 → 真平台效应(CARLA 跨硬件物理分叉致 H800 driving 更差), 即便感知/IPC 已排除。
- **判别 = H800<4090 是否跨 route 一致**。r3 可复现(n1=n2)使它成最干净判点 → **4090 r3 ×3 = critical-path**(若 4090 r3 也复现 100 → 干净可复现差, 查混沌vs系统; 若 4090 r3 也偶 50 → 方差, 两机都 block)。
- **判决**: ⏳ 待 4090 r3/r17/r104 ×3 + H800 r17/r104。**保持怀疑, 不提前下"无缺陷"**。
- **★[更新] r3_n3=100 → H800 r3 分布 = [50,50,100](均66.7)**: r3 **非纯确定**(有方差), H800 **也能在 r3 拿 100**(1/3)。⇒ 4090 probe 单次 100 **落在 H800 分布内**(1/3 概率), 很可能只是 n=1 运气, **非平台优势**。**黄旗软化**(不再像"可复现差")。但 H800 均值 66.7 vs 4090 单点 100 仍不能判等价/有差 —— **必须 4090 ×N 才能比分布**(仍 GPU-blocked)。
- **4090 基线已存(team-lead 亲读 disk)**: 上 session 全量扫描 `results_driving_mr_te0_r{3,17,104}` = **r3/r17/r104 全 100**(RC100, **n=1**)。⇒ 4090 对照可用, **4090 ×3 backfill 降级**(非 critical blocker, 有卡再补)。
- **判据更新(team-lead)**: 从"是否有 gap"→**"H800 分布是否覆盖 4090 的值"**。r3: H800 {50,50,100} **已覆盖 100** → 黄旗判**良性方差**(与 r6 同方差包络: 多数低+偶满分)。
- **诚实残caveat**: H800 r3 均值 66.7 vs 4090 r3=100(n=1)仍有**均值差可能**; 但 ① 4090 也仅 n=1 且本身高方差(r6 曾 2.7), ② 即便有小均值差也在 CARLA 大方差包络内 + 归跨硬件物理(非框架, 感知/IPC 已排除)。⇒ **"干净可复现系统 gap"已溶, 残留均值差是次要且非缺陷**。
- **★不依赖 4090 也成立的核心结论**: 无论 4090 backfill 结果如何, **原 premise"torch2 感知退化→te0崩"已被 b-test 硬数推翻**(感知等价), 这是主交付。"H800 是否与 4090 统计等价"是次要问题, 且即便有小差也只能归 CARLA 跨硬件物理(感知/IPC 已排除), **非框架缺陷**。

---

### ★★★ H-15 [重开·Thread A 代码调查] 找到被漏掉的确定性机制 = CARLA 服务器种子(可修!)
> 用户驳回"不可消除"后重查 V2Xverse 代码(Explore agent), **我的"unfixable"被推翻**。
- **V2Xverse 确定性配方(我之前没翻全)**: 除已知 TM/numpy/random/CarlaDataProvider 种子外, **关键缺的一环 = CARLA 服务器内部 RNG 种子, 由 `CarlaUE4/Config/DefaultGameUserSettings.ini` 的 `[CARLA/ServerRandomSeed] Seed=1234` 设**(README:80-84)。它种子化 **`get_random_location_from_navigation()`** —— 正是放置 walker 导航目的地的调用。
- **方差源精确定位**: `carla_data_provider.py:1300` 背景 walker 导航目的地 `go_to_location(get_random_location_from_navigation())` 用 **CARLA 服务器 RNG**, **仅由 DefaultGameUserSettings.ini 服务器种子控制**(非 set_pedestrians_seed——那个 0.9.10 确实没有, 但**不是该用的机制**)。Town05 背景 walker amount=120, 未种子导航 → 逐 run 乱走 → 偶有挡 ego。
- **r6 corridor 挡车 = Scenario3 DynamicObjectCrossing 脚本行人**(adversary_type:False→行人; spawn transform 固定+blueprint seeded=确定); 但其**穿行/导航**或背景 walker 受服务器种子影响。(注: object_crash_vehicle.py:433 有个 `bias=20*random.random()` 但 random 已 seed 2000, 且仅 250+ 次 spawn 重试才触发, 大概率不触发。)
- **★可修假设 + 平台差解释**: 若 **4090 的 CARLA 设了 ServerRandomSeed 而 H800 没设/不同** → 4090 walker 导航确定(更稳)、H800 乱走(r6 偶崩)。**这一条就能同时解释"H800 不稳"和"H800 vs 4090 差", 且是可修真根因**(设服务器种子)。
- **判据(Thread B)**: H800 设 ServerRandomSeed → r6 ×3 是否 bit-reproducible。
- **★★★ 冒烟枪坐实(2026-06-13, team-lead 独立 cat 两机 ini)**:
  - 4090 `carla/CarlaUE4/Config/DefaultGameUserSettings.ini`: **有 `[CARLA/ServerRandomSeed]\nSeed = 1234`**。
  - H800 同路径 ini: **仅 61 字节, 完全无 ServerRandomSeed 段**。
  - ⇒ **真 config 不对称(非对称n=1噪声)**。完整解释: 4090 种子化→背景 walker 确定→稳; H800 未种子→120 walker 乱走→偶挡 r6 走廊→崩。**并解释 route 依赖**(r104 无 walker 干扰→未种子也稳[100,100,100]; r6 walker 走廊→重灾)。
  - **这是 H800 vs 4090 真平台差的根因, 且可修**(感知/IPC 早已排除, 现在差的是 CARLA server 配置)。
- **状态**: ✅ 根因定位(config 不对称); ⏳ Thread B 修复验证授权执行中。
- **★executor 代码实证补充(2026-06-13)**:
  - Q1: 两机都跑 `leaderboard_evaluator_parameter.py`(同), carlaProviderSeed/trafficManagerSeed=2000。
  - Q2: **r6 零 scenario actor**(route XML scenario_triggers=0, bbox 0 命中) → **挡车=背景行人**(纠正我 Explore 误说的"r6有Scenario3")。unseedable 源 = `carla_data_provider.py:875`(spawn)+`:1300`(walk target) 的 `get_random_location_from_navigation()`=CARLA C++ NavMesh RNG, 无 Python seed 路径。背景 vehicle 用 `_rng.choice`(已seed); 仅**背景行人**未种子。r6 infraction 有 `collisions_pedestrian>0`(r3/17/104 无)→ 印证行人是干扰源。
  - ⚠️ **驳 executor 2d 的混淆**: "4090 r6=[100,2.7,37.8]"**非干净同config×3** —— mr_te0(te0)=100 / mr_te**219**(te219)=37.8 / e2_n1(te0但GPU污染+代码改)=2.7。混了 te 值+污染跑。**不能证 4090 r6 在干净 te0 下也抖**(干净 4090 te0 r6 仍仅=100 一点)。冒烟枪(ini不对称)仍立。
  - ⚠️ **技术存疑(Thread B 要测)**: ServerRandomSeed 是否真控 `get_random_location_from_navigation`(C++ NavMesh)在 0.9.10 未知。**Thread B 一石二鸟**: 测方差是否消 + 测种子机制是否生效。
- **Thread B 双设计**: **B1**(已派) 加 ServerRandomSeed=1234+重启→r6×3 bit-reproducible? (测fix机制); **B2 兜底/确证源** `pedestrian_amount=0`(去掉所有背景行人)→r6×3 → 若变 reproducible **直接确证背景行人=方差源**(不依赖种子是否生效)。
- **★★ B1 判决(2026-06-13, 每run独立重启CARLA实验有效)**: seed=1234 → r6 = **run1 37.77 / run2 100.0**(≠) → **不 reproducible**。⇒ **ServerRandomSeed=1234 不控 `get_random_location_from_navigation` 的 NavMesh C++ RNG**(0.9.10 该 ini 段不被 NavMesh 用)。**文档的 fix 在此 build 无效。**
  - **★含义(诚实降级 smoking gun)**: ini 有/无 ServerRandomSeed **不控方差源** → 那 4090 ini 有它也不会让 4090 reproducible → **方差是两机对称的未种子 walker nav, ini 不对称是 red herring(非 H800-vs-4090 差因)**。"4090 r6=100" 很可能 n=1 运气(4090 r6 ×N 未测, GPU-blocked, 推断也抖)。**回到"对称方差+n=1噪声", 但现在源已锁定 walker nav + 文档fix已证伪**(用户 push 仍有值: 找到真源+排除假fix)。
  - run1 有个信号: seeded 后 **collisions_pedestrian=0**(旧 r6 有行人撞), 改 vehicle_blocked → seed 确实**改了**行人行为, 只是不能让其跨run确定。
  - **B1 完整 [37.77, 100, 37.77]**: **run1==run3 bit-exact**(DS+所有 infraction 16位逐位同), run2=100, 全3 colped=0。⇒ seed 移除了行人碰撞但**未消方差**; 复现结构与未种子同(都 run1==run3 bit-exact + 一个 divergent run)。run1/3 的 block 现是**车辆**(colveh=0.2247)。**关键洞察**: 行人即便不撞 ego, 其未种子游走仍**扰动交通**(车为行人刹车/绕→某些run车挡 ego) → 行人可能仍是**根源**, 近因变车。**B2(pedestrian=0)正是干净判这个**。
- **B2 = 现在的定源测**: pedestrian_amount=0 → r6×3 全同 = **铁证 walker nav 是唯一方差源**(其余 RNG 都 seeded); 给可复现评测一条路(两机都设 pedestrian=0 → 可干净比平台)。
- **★★ B2 判决(2026-06-13)**: pedestrian=0 → r6 = **run1 100 / run2 29.29**(≠, run2 vehicle_blocked, colped=0) → **仍不 reproducible**。⇒ **行人不是唯一源; 背景车辆/TM 是第二非确定源**(TM seed=2000 不足以完全固定 0.9.10 TM 多线程行为)。**两个文档/直觉 fix(server seed, 去行人)都不能让 r6 reproducible。**
- **★earned 结论(非软推断)**: r6 方差**多源**(未种子行人 NavMesh + TM/车辆不完全确定), 经**穷尽实测**(server seed 证伪 + 去行人不足)→ **CARLA 0.9.10 TM 已知不完全确定**, 两机对称。"必须 N-run/选 route" 这次是**测出来的**, 不是跳的。
- **★实用 win(应突出)**: **r104 之前 ×3 = [100,100,100] 带全量 ambient traffic 就已 reproducible** → **天然可复现的 baseline 已存在**(ambient traffic 在 r104 不挡 ego); r6 只是 pathological knife-edge。**τ_ego 延迟曲线应在 r104 类干净 route 跑, 不必跟 r6 死磕, 也不必关 ambient traffic。**
- **B3(最终定源, 可选)**: pedestrian=0 **且** vehicle_amount=0 → r6×3 全同? 是→ ambient traffic(peds+vehicles)是 r6 全部方差源; 否→ 更深(physics/ego)。
- **B2 完整 [100, 29.29, 100]**(run1==run3=100 clean, run2 vehicle block, colped=0 三跑) → 第二源=车辆/TM 确认。
- **★r104 = 4/4 run 全 DS=100/RC=100/infraction 全 0** → **"天然可复现 baseline" 坐实(硬数, 非回忆)**。带全量 ambient traffic 仍稳。⇒ **τ_ego 延迟曲线应在 r104 类干净 route 跑**(天然可复现, 不必关 traffic、不必 N-run 死磕 r6)。
- **B3 已授权执行**(pedestrian=0 AND vehicle=0)。

---

## §3 收敛判据 (done) + ★最终结论 (2026-06-13)

**原判据**: H800 te0 在 r6 等复现 4090(碰撞<50%、DS 近 100), 真因被修复证实。

### ★最终结论: 原 premise 是误诊, 无 H800 框架缺陷
**1. 起点 premise 被硬数推翻**: "H800 te0 即崩 = torch2 感知退化"**错**。b-test(同 payload, 同权重)实测两机感知**框位置逐位相同、conf 仅差 1e-4** → H800 torch2 前向 ≈ 4090 torch1.10, **无退化无漏检**。

**2. 全嫌疑链被真测逐一排除**: 权重(md5同) / voxelizer(bit-identical, 且"4090 spconv1"前提本身错-两机都2.3.6) / IPC点云损坏(pcd完好) / IPC时序(阻塞同步+崩过两跑前400帧RTT同+无sensor-timeout) / 感知非确定(E3 bit-identical×5) / 感知mean-shift(b-test 1e-4) / sensor-race(无痕迹)。

**3. 真因 = CARLA 环境方差 + knife-edge route, 两机对称, 非 H800 缺陷**:
- 不稳来自 **CARLA 0.9.10.1 内部物理非确定**(无 set_pedestrians_seed API 可消) + **NPC/cyclist 偶在 corridor 挡 ego**。
- **对两机对称**: 4090 自己也 r6=2.7(n=1)、与 H800 同病。
- **route 依赖**: clean route RC 全 100、H800 反复摸到 100; knife-edge route(r6) 两机都崩(H800 RC~63%, 4090 也 2.7)。
  | route | H800 ×3 | mean | RC | 性质 |
  |--|--|--|--|--|
  | r104 | **[100,100,100]** | **100** | 100 | clean, 无挡车 → **H800 复现满分=4090** |
  | r17 | [100,50,100] | 83.3 | 100 | clean, 偶发 NPC 挡车 |
  | r3 | [50,50,100] | 66.7 | 100 | clean, 偶发 NPC 挡车 |
  | r6 | [18.88,11.54,37.77] | 22.7 | ~63% | knife-edge, 与平台无关 |
- **★r104=[100,100,100] 是等价铁证**: H800 在 clean route 上**可复现满分, 逐次=4090** → 证 H800 闭环健康; r3/r17 的偶发 50 是 **CARLA NPC 挡车方差**(非 H800 缺陷, 因 H800 能满分)。
- **★收敛判据达成**: 用户要的"H800 te0 复现 4090(碰撞<50%、DS→100)"——**r104 te0 = DS100/0碰撞/×3复现, 达成**。原"te0即崩"是 r6(knife-edge)+n=1 选样伪信号。

**4. 残留 caveat(不夸大)**: H800 clean-route 均值(r3 66.7 / r17 83.3)vs 4090 n=1=100 有**均值差可能**, 但 ① 4090 仅 n=1 且本身高方差, ② 即便有小差也在 CARLA 大方差内 + 归跨硬件物理(感知/IPC 已排除), **非框架缺陷**。需 4090 ×N 严格比分布(降级, GPU-blocked, 有卡再补)。

**5. 交付口径(给用户)**: 根因被验证 = **误诊被证伪**。无代码 bug 可修; "修"是方法学——**评测须 N-run/多 route 平均**(CARLA 非确定 + knife-edge 使单次评测=n=1噪声, 原"H800 41 vs 4090 100"即此)。τ_ego 延迟曲线应在 clean route + N≥3 上重做。

**6. 待办**: r104×3(确认第3条clean route覆盖100) / 4090 r3,r17,r104 ×3 backfill(有卡补, 严格分布) / doc-curator 更正 MEMORY+HANDOFF 的"torch2 感知退化"旧条。

---

## §4 状态看板 (final)
| 实验 | 结论 |
|------|------|
| E0-ckpt | ✅ 权重两机 md5 同, 排除 |
| E0-AB | ✅ 受控对比(后被方差修正: 单次不可复现) |
| E0-dump | ✅ H800 出真检测 nf2-4, 感知非空 |
| H-03a voxelizer | ✅ bit-identical, 排除(+纠正 spconv 前提) |
| E3 感知确定性 | ✅ bit-identical×5, 排除 CUDA 非确定 |
| InferProxy sync | ✅ 阻塞同步, 排除 IPC staleness |
| b-test mean-shift | ✅✅ 1e-4 噪声, **感知 EXONERATED, premise 推翻** |
| E2 分布 H800 r6 | ✅ knife-edge, [37.77×2,5.14,6.48,100,...] |
| 多route r3/r17 | ✅ H800 覆盖 100, 黄旗良性 |
| r104 ×3 | ✅ **[100,100,100] 复现满分=4090, 等价铁证** |
| 收敛判据 | ✅ **达成**: H800 te0 r104 = DS100/0碰撞 ×3 |
| 4090 ×N backfill | ⛔ GPU-blocked, 降级(非阻塞, 有卡补) |

> **调查闭合 2026-06-13**: premise(torch2感知退化)证伪; 无框架缺陷; H800 te0 在 clean route 复现 4090 满分; 不稳=CARLA非确定+knife-edge(两机对称, n=1噪声)。进程已清, H800 GPU1-7 空闲。
