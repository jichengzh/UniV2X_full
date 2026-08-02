# 交接: τ_ego 延迟曲线 + H800 闭环缺陷诊断 (HANDOFF_tau_curve_h800_defect_v1)

> 写于 2026-06-13 00:35。供新窗口接手。本轮从"画延迟→驾驶分曲线"做到"发现 H800 闭环本身有缺陷"。
> ★接手第一件事: 读本文 §1 当前定论 + §2 必须先确诊的下一步。别再急着画曲线——曲线的前提(H800 te0 能复现 4090 的 ~100)还没满足。

---

## ★★★ [2026-06-13 勘误 — 本文核心 premise 已被硬数证伪, 接手必读]

> **本文写于早期调查阶段, 核心 premise "H800 te0 即崩 = torch2 感知移植退化" 已被后续穷尽实测证伪。本文保留历史可追溯性, 但 §0/§1.3/§2/§7 的相关结论已失效。**
>
> **新定论(完整台账: `multi_agent/real_test/h800_rootcause_investigation_v1.md` §3 最终结论)**:
> 1. **b-test 硬数证伪感知退化**: 同 payload 同权重, 4090(torch1.10) vs H800(torch2.1) 检测框位逐位相同、conf 仅差 6e-5。无退化、无漏检。H800 torch2 感知 ≈ 4090 torch1.10。
> 2. **真因 = CARLA ambient 背景 traffic 非确定(穷尽定源 B1→B2→B3, 两机对称)**: B3 pedestrian=0 且 vehicle=0 → r6 [100,100,100] bit-exact; ambient traffic 是 r6 全部方差源。
> 3. **"H800 41 vs 4090 100" 是偏选 route(r6 knife-edge) + n=1 噪声的伪信号**, 与 H800 框架/感知/IPC 无关。
> 4. **H800 在 clean route 复现满分**: r104 × 4 = DS100/0碰撞(带全量 ambient traffic), 收敛判据达成。
> 5. **可用评测配置**: ① r104 类 clean route 天然可复现(无需关 traffic); ② `_1notraffic` config(ambient vehicle=ped=0, 保留 Scenario3 脚本行人 + RSU) = 任意 route 可复现确定锚点, te0 确定=100, 失败纯 ped-collision; ③ 延迟曲线须逐 route 出(失败阈值依 scenario 避让余量), 每档 N≥3 平均, 别跨 route 朴素平均。
> 6. **附带勘误**: "4090 spconv1 vs H800 spconv2 移植"是错的——两机都 spconv 2.3.6。
>
> **⇒ 曲线实验无需等"修 H800 感知", H800 已健康; 用正确评测配置(r104 类/`_1notraffic`)直接跑 τ_ego 延迟曲线即可。详见 `multi_agent/real_test/exp_tau_ego_curve_design_v1.md §8`。**

---

## §0 一句话现状

~~目标是画一张 **τ_ego(推理延迟) → driving score** 的曲线证明延迟伤害 V2X 场景驾驶。但发现 **H800 闭环仿真在零延迟(te0)就把路线跑崩(DS~41/95%碰撞)，而 4090 同路线 te0≈100**。所以当前所有 H800 延迟数据都是"坏基线+延迟"，曲线非单调且无效。**用户选 B 方案 = 修 H800 让它 te0 能复现 ~100，再做延迟实验。** 根因诊断已收敛到"**H800 torch2 感知移植很可能数值退化**"，待最后确诊。~~

★[2026-06-13 勘误] 上文 premise 已证伪。**真实现状**: H800 闭环无框架缺陷; "te0 即崩"是 knife-edge route(r6) + n=1 + ambient traffic 非确定的伪信号。H800 在 clean route(r104 等)复现 DS=100。曲线非单调的根因是"平台混淆 + 路线偏选 + n=1 噪声"(见 `tau_curve_rootcause_v1.md`), 不是感知退化。详见顶部勘误段。

---

## §1 已查清的定论 (按可信度)

### 1.1 延迟曲线非单调的根因 (详见 `tau_curve_rootcause_v1.md`)
四重实验设计缺陷叠加, 把真实延迟信号埋没:
1. **平台混淆(主因)**: te0/te219 在 **4090** 跑, te50-500 在 **H800** 跑。曲线"te0(85)→te50(35)悬崖"是 4090→H800 平台跳变, 不是延迟。铁证: 4090 上 te0=te219=100(延迟+219ms 零影响)。
2. **碰撞饱和(地板)**: H800 上这批强敏感路线 te50 即 ~98% 碰撞, DS 钉地板无动态范围。
3. **筛选污染**: 54 条"强敏感"路线用 `delta = te0(4090) − te500(H800)` 选 = 平台差, 不是延迟敏感度。
4. **n=1 噪声**: 单次跑, 饱和区掷骰子, 放大非单调。

### 1.2 ★更深的发现: H800 闭环零延迟即崩 (本轮核心)
- **H800 te0(τ_ego=0 注入) = mean DS 41.6 / 碰撞率 95%** (n=39, 仍在跑向 54)。**4090 te0 这些路线 ≈ 100。**
- **逻辑铁证(排除"只是延迟")**: 4090 在这些路线 **te219(219ms注入)都扛得住**(r26/r303/r113=100)。若 H800 只是 IPC ~150ms 延迟, te0 应 ≈ 4090 te150 ≈ 85-100。但实测只 ~41/95%撞 —— **比 4090 的 219ms 还差** → 必有延迟之外的退化。

### 1.3 三个嫌疑的判决

~~旧表 (该表的感知退化结论已被证伪, 保留可追溯)~~:

| 嫌疑 | 旧判决(本文写作时) | ★[2026-06-13 更正] |
|------|------|------|
| 施工场景(`_1c` scenario 变体) | ❌ 不是主因 | ❌ 仍成立 |
| IPC ~150ms 延迟(单独) | ❌ 解释不了量级 | ❌ 仍成立(阻塞同步, IPC staleness 排除) |
| **H800 torch2 感知移植退化** | ~~✅ **最可能主因(待确诊)**~~ | ★★ **❌ 被 b-test 硬数推翻**: 两机感知框位逐位相同、conf 仅差 6e-5, 无退化 |
| **★真因(本文写时未知)** | — | ✅ **CARLA ambient traffic 非确定(B3 定源)**: ambient vehicle=ped=0 → r6 bit-exact, 两机对称 |

### 1.4 两个真实测到的物理量
- **IPC 每帧真实延迟 ~120-180ms**(process A 日志 `[InferProxy] latency_ms=`, 首帧849ms冷启)。zmq 序列化传输开销占大头(真推理应~20ms)。这是真实的延迟地板, 但**不是 te0 崩的主因**(见1.2)。
- **scenario `_1c` vs `_1` 差异**: `_1c`(H800全程用)是"construction"变体, 比 4090 的 `_1` 多了 Scenario11(ConstructionSetupCrossing 施工障碍)+ 改了 Scenario3 行人配比。**做对比实验必须统一 scenario 参数**。

---

## §2 ★下一步: 必须先确诊"感知退化 vs 延迟/control" (不用4090)

**廉价确诊**: 在 H800 跑感知, dump 几帧检测输出(框数/置信度/位置), 人眼看 ego 到底"看没看见"该避的车/人。
- 检测稀疏/错位/漏检 → **感知移植坏了(确诊)** → 修 torch2 移植。
- 检测正常但反应晚 → 才是延迟/control 问题。

若确诊感知退化, 逐个核对移植补丁对**同一输入**的输出 vs 4090 torch1.10, 定位哪个算子数值跑偏:
- spconv1→2 import(opencood 4文件)
- pcdet_utils THC 删/重编 sm90
- box_overlaps cython 重编
- codriving_attn 删 turtle
- (补丁清单见 `PROGRESS_h800_sim_smoke_v1.md`)

修好让 H800 te0 ≈ 100 后, 才能用 §4 框架重跑干净的延迟曲线。

---

## §3 当前运行中的任务 (接手时先查状态)

- **H800 te0/te219 基线**: master `pgrep -f "baseline_sweep_6slo[t]"`, 进度 `grep Progress /data/jichengzhi_v2x/baseline_sweep_progress.log|tail -1`(截至交接 42/108, 仅te0段, te219未开始)。**价值已不大**(只是封口"全H800扁平~41线"); 可让它跑完存档, 也可杀掉省卡(`kill $(pgrep -f "baseline_sweep_6slo[t]")` + 清 CARLA/B)。完成后 `python3 /data/jichengzhi_v2x/peek_te0.py` 看全54条 te0 mean。
- **两个 agent**: 仿真监督 a8b781001539bc2d1 / 数据分析 a9c2556e90c6a7e5f —— 数据分析已交付(根因报告), 仿真监督在盯基线。清上下文后这些 agent 会失联, 不影响 H800 上已 detach 的进程。
- **noconstruct 隔离测试**: 已结束清理(master有完成检测bug, 卡死已杀, GPU6已释放)。r6 结果已取得(见1.3)。

---

## §4 H800 闭环操作手册 (复现/重跑必读)

> 完整版 `REPRODUCE_h800_closedloop_sim_v1.md`。核心要点:

### 连接
`ssh -p 30001 -o StrictHostKeyChecking=accept-new ${V2X_REMOTE_USER}@<PRIVATE_HOST>` (密码每会话确认; repo=`/data/jichengzhi_v2x/V2Xverse`)

### 双进程架构 (绕开 Hopper sm90 对老 cu113 不兼容)
- 进程A: CARLA客户端+leaderboard, py3.7 `/data/jichengzhi_v2x/envs/v2xverse/bin` (cu113, 纯CPU)
- 进程B: codriving感知整脑, py3.10 `t2lib` (torch2.1/cu121, sm90原生, GPU推理)
- A↔B 经 zmq IPC (USE_INFER_SERVER=1)。tau_ego 在进程A的 run_l1_step 注入(L1PlanStore延迟激活)。

### 6槽并行框架 (已跑通311条, 可复用)
- 脚本: `/data/jichengzhi_v2x/strong_sweep_6slot.py` (B-first启动顺序版, 已修TypeError) / `baseline_sweep_6slot.py`(TAUS=[0,219]变体)。
- 启动: `cd /data/jichengzhi_v2x; TMPDIR=/data/jichengzhi_v2x/tmp setsid nohup python3 -u <脚本> >run.log 2>&1 </dev/null &`
- 收集: `collect_strong_v2.py`(DS=score_composed, infractions是per-km float三类>0即collided, Eff/Comfort跑bench2drive_metrics.py)。

### ★血泪坑 (全踩过, 别再踩)
1. **`/`(含/tmp)曾100%满** → B server SAVE_PATH 写`/tmp`失败崩→"not ready"。**SAVE_PATH/TMPDIR 必须在`/data`**(脚本已改)。曾误判为"GPU饿死", 实为/tmp满。
2. **pkill self-kill SSH**: `pkill -f X` 会匹配自己命令行里的X→杀掉自己SSH(exit255)。**杀进程一律用bracket trick**: `pkill -9 -f "CarlaUE4-Linux-Shippin[g]"` / `"process_b_serve[r]"` / `"python3 -u baseline_sweep_6slo[t]"`, 且命令里别出现无bracket同名串。
3. **2-per-card失败**: 一张卡2个CARLA→100%空转饿死B加载。**1卡1组(1CARLA+1B)**才稳(每卡~35%util有余量)。
4. **结果路径**: DS在 `<resultdir>/r{N}_repeat0/ego_vehicle_0/results.json` 的 `_checkpoint.global_record.scores.score_composed`。**顶层 `r{N}_repeat0/results.json` 的 global_record 是空的, 别读错**(noconstruct master 就栽在这: 检查顶层路径→永远检测不到完成→死循环)。
5. **scenario统一**: 对比实验必须同scenario参数。`_1c`有施工, `_1`无施工。eval第6参选后缀。
6. **detach**: master必 `setsid nohup ... </dev/null &`, 否则agent turn结束被SIGHUP杀。
7. **长任务慢**: 每route重载B模型(~40s)+CARLA启动, 6槽~0.4-0.8 route/min, 311条跑了~3.5h。

---

## §5 数据与文件清单 (全绝对路径)

### 本机 ${V2X_ROOT}/multi_agent/real_test/
- `tau_curve_rootcause_v1.md` — 非单调四重根因报告(数据分析agent产)
- `tau_curve_sensitive_routes_v1.csv` — 54强敏感路线(★选样被平台差污染, 慎用)
- `data_strong_sweep_te50-400_H800.csv` — H800 te50-400 × 54路 (DS/RC/collided/Eff/Comfort)
- `data_te500_H800.csv` — H800 te500 × 105路
- `route_scenario_classification_v1.csv` — 105路场景类型/强度(无偏选库依据)
- `exp_tau_ego_curve_design_v1.md` — 实验设计(§1有"中强度=黄金区"结论)
- `l1_fix_proposal_purepursuit_v1.md` / `l1_architecture_problems_v1.md` — L1控制器(本轮未深入)

### 4090 (本机) te0/te219 真值
`${V2X_ROOT}verse/results/results_driving_mr_te{0,219}_r{N}/v2x_final/town05_short_collab/r{N}_repeat0/ego_vehicle_0/results.json`

### H800 /data/jichengzhi_v2x/
- `V2Xverse/results/results_driving_te{0,50,100,150,200,300,400,500}_strong/...` — H800强敏感sweep结果
- `peek_te0.py` / `collect_strong_v2.py` — 收集脚本
- `strong_sweep_6slot.py` / `baseline_sweep_6slot.py` — 6槽跑批框架
- `*_progress.log` / `*_run_main.log` — 进度/主日志

### 关键数据点速查 (54强敏感路线)
| τ_ego | 平台 | mean DS | 碰撞率 |
|-------|------|---------|--------|
| te0 | **4090** | 84.65 | 27.8% |
| te0 | **H800** | **41.6** | **95%** ← 缺陷在此 |
| te50-500 | H800 | 34-44(扁平) | 93-100% |
| te219 | 4090 | 74.61 | 42.6% |
| te500 | H800 | 33.91 | 100% |

---

## §6 路线图

1. **[P0] 确诊感知退化**(§2, 不用4090): dump H800 检测输出看质量。
2. **[P0] 若感知坏→修torch2移植**: 逐补丁对比4090输出, 定位数值跑偏的算子。
3. **[P1] 修好后验证**: H800 te0 应≈100(<50%碰撞)。
4. **[P1] 重跑干净延迟曲线**: 同平台(H800修好后)+无偏选库(中强度39条,避饱和)+统一scenario(_1)+n≥3重复, 才能出有效单调曲线。
5. **[备选] 若H800修不动**: 回4090做延迟实验(单进程te0=100干净基线), 但4090当前满。

---

## §7 给接手者一句话

~~别被"非单调曲线"带偏去调参/换库——**真问题是 H800 双进程闭环在零延迟就跑崩(te0 DS41/95%撞 vs 4090 100), 根子最可能在 torch2 感知移植的数值退化**。先按 §2 确诊感知质量, 这是 B 方案(修H800)的命门。曲线是果, H800 闭环正确性是因。~~

★[2026-06-13 勘误] **上文已过时, 勿参考。** H800 闭环无感知退化、无框架缺陷; "te0 崩"是 knife-edge route + ambient traffic 非确定 + n=1 噪声(两机对称)。**无需"修 H800 感知"**。直接使用 `_1notraffic` config 或 r104 类 clean route 即可做干净的 τ_ego 延迟曲线。详见顶部 ★★★ 勘误段。

---

## §8 ★2026-06-13 最新: E2分布实验(H800×8) + matplotlib bug + cross-machine b-test方案

### 8.1 H800×8 E2实验结果 (r6_1, te0_l1)

**脚本**: `/data/jichengzhi_v2x/h800_e2_r6_n8.sh`
**结果路径**: `/data/jichengzhi_v2x/logs_e2_r6/SUMMARY.txt`

| run | DS | RC | coll_veh | 备注 |
|-----|----|----|----------|------|
| n1 | **37.769711630172694** | 62.95% | 0.449 | bit-exact, CARLA确定性 |
| n2 | 5.14 | 8.57% | 3.302 | CARLA TM非确定性早终止(n2与n1/n3不同) |
| n3 | **37.769711630172694** | 62.95% | 0.449 | n1 bit-exact复现 |
| n4 | 6.48 | **100%** | 0.566+ped0.141 | CARLA TM变体(900+帧,完整路线) |
| n5-n8 | 0.0 | 0% | 0 | **★无效 — 我引入的matplotlib bug(见8.2)** |

**结论**: H800在标准CARLA流下确定性崩cyclist走廊(n1==n3 bit-exact DS=37.77)。n4是CARLA TM随机变体。

### 8.2 ★我引入的 Bug — matplotlib 3.10.9 在 t2lib PYTHONPATH

**根因**: 我把4090版 `pnp_infer_action_e2e.py` 复制到H800以加 payload capture 代码, 没注意 H800 process_b 的 `PYTHONPATH=/data/jichengzhi_v2x/t2lib:...` 里 t2lib 包含 **matplotlib 3.10.9**, 该版本移除了 `tostring_rgb()`。

**症状**: 每帧 `B-ERR: AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'` → process_b 返回 `ok=False` → ego完全不动 → DS=0, RC=0。

**关键旁证(诊断价值极高)**:
- n1(原始H800代码) `grep tostring_rgb process_b.log → 0次`
- n5(4090代码) `grep tostring_rgb process_b.log → 308次`

**→ 原始H800代码从不调用 `turn_traffic_into_map`, 即 H800 的 `processed_pred_box` 在所有帧都是空的(无检测)。这与 torch2 感知退化假设高度一致: H800模型输出空预测框 → occ_map全空 → ego看不到cyclist → 碰撞。**

**修复** (已应用到 `/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard/team_code/pnp_infer_action_e2e.py` 和本机4090版):
```python
# turn_traffic_into_map 里 line ~308:
if hasattr(fig.canvas, 'tostring_rgb'):
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
else:
    data = np.asarray(fig.canvas.buffer_rgba())[..., :3]
```

### 8.3 Cross-machine b-test 方案 (payload capture)

**目的**: 证明"H800感知真的看不见cyclist" vs "H800感知正常但别的原因导致碰撞"。

**工具已就绪**:
- `pnp_infer_action_e2e.py` 里的 payload capture 代码: 设 `PAYLOAD_CAPTURE_DIR=/data/jichengzhi_v2x/payload_capture` 则在闭环时自动保存corridor帧(x≈-44.3, y∈[-56,-47])的完整 `extra_source` 为 pkl。
- H800 `h800_e2_r6_n8.sh` 已有 `PAYLOAD_CAPTURE_DIR` 环境变量设置。
- `b_test_payload_replay.py`: 单帧前向, 输出 `nf=` / `scores_md5` / per-box置信度。

**执行**: H800跑一轮r6_1 → corridor pkl保存 → scp到4090 → 两机分别跑 `b_test_payload_replay.py`:
```bash
# H800:
PAYLOAD_CAPTURE_DIR=/data/jichengzhi_v2x/payload_capture \
  bash /data/jichengzhi_v2x/h800_e2_r6_n8.sh  # 1轮即可

# b-test on H800:
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/data/jichengzhi_v2x/t2lib:/data/jichengzhi_v2x/V2Xverse:/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard \
  python3 .../b_test_payload_replay.py --pkl corridor_payload_*.pkl

# b-test on 4090 (same pkl):
CUDA_VISIBLE_DEVICES=<空闲卡> \
  python3 .../b_test_payload_replay.py --pkl corridor_payload_*.pkl
```

**判读**: H800 nf < 4090 nf (尤其cyclist框) → torch2感知退化确诊。

### 8.4 4090 n1 参考数据

DS=2.7, RC=100%, coll_veh=coll_ped=0.424 (r6_1, te0, single-card)。
比 §5 表里4090 te0 mean DS=84.65 低很多 — r6_1可能是特别难的路线(有cyclist+pedestrian双场景)。
4090 loop 按 rc-supervisor 指示在 run1 后 kill(保护wuyuegao的 GPU3)。
