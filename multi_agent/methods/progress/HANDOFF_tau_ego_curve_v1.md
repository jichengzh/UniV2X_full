# 交接: τ_ego 驾驶分曲线实验 (HANDOFF_tau_ego_curve_v1)

> 写于 2026-06-11。用于新窗口接手"延迟→V2X 场景驾驶分退化曲线"实验。
> 配套: `multi_agent/methods/design/exp_tau_ego_curve_design_v1.md`(实验设计) + `l1_fix_proposal_purepursuit_v1.md`(L1 修法) + `l1_architecture_problems_v1.md`(L1 问题)。
> ★接手第一件事: 读本文 §1 当前进度 + §5 教训, 别重复踩端口/CSV 的坑。

---

## §0 目标 (一句话)

在 ~60-70 条按场景类型选的 V2X 路线库上, 扫 τ_ego ∈ {0,50,...,500}ms, 出一张 **延迟 vs 驾驶分(+碰撞率)** 折线图 + 机理解释。多 agent: runner 跑 / supervisor 监督反思 / analyst 出图。

---

## §1 当前进度 (截至 2026-06-11)

### 已完成 ✅
- **I-2 τ_ego 注入机制**: V2Xverse `feature/l1-trajectory-tracker` commit **f63e92b**(已 push myfork)。`L1PlanStore` 延迟激活缓冲, config `simulation.tau_ego_ms` → Δ_ego=ceil(τ/50)。τ_ego=0 字节级等价 baseline。V0-10 单测全过。
- **全量 te219 扫描**: 105 条全扫完(τ_ego=219ms)。结果在 `results/results_driving_mr_te219_r*/`。
- **★场景分类(105 条全, 真相源, 不依赖碰撞)**: 从运行日志 `route_scenarios: {...}` 解析每条路线实例化的场景类型+数量。存 `multi_agent/real_test/route_scenario_classification_v1.csv`。
  - **全 105 条都是 V2X 场景**(含 Scenario3 行人 和/或 Scenario4 车辆; 无纯控制丢失)。
  - **类型**: 纯行人 24 / 纯车辆 13 / 混合 67。
  - **★强度(用户选定按此分类)**: 高(≥25)38 条 / 中(8-24)39 条 / 低(<8)28 条(r112=0 场景排除)。
- **★强度 vs 延迟敏感性交叉验证(te0/te219, 26/105 配对)**: 高 56.4→47.4(Δ-9) / 中 86.0→65.0(**Δ-21**) / 低 100→87.5(Δ-12.5)。三档都 te219<te0(方向对); **中强度信号最干净**(高强度 baseline 已低=太难无退化空间, 强度≈难度)。详见 `exp_tau_ego_curve_design_v1.md §1`。
- **配置**: `pnp_config_codriving_te{0,108,136,219}_l1.yaml` 已建(agent_config/)。

### 进行中 🔄
- **te0 全量补跑**(目标: 105 条全有 te0 对照, firm up 强度×延迟结论): 已有 te0 26 条 + te0fill 批 28 条(GPU1/2 port45000/45100) + 剩余 51 条(GPU3/7 port46000/46100)。log: `results/te_confirm/te0{fill,r}_g*_*.log`。完成后所有强度档都有完整 te0 vs te219 配对。

### 待做 (新窗口) ⏸
1. **[待用户拍板] 库方案(按强度)**: 选项 A 单档(中强度 39 条信号最干净) / B 三档各跑出对比曲线 / C 全 105 加权。定后从 csv 筛 route 列表 = 库, 冻结。
2. **补延迟档 config**: 生成 te50/150/200/250/300/350/400/450/500(照 te108 模板, 见 §3)。
3. **拉起多 agent**(§4) 跑 P1 pilot → supervisor 反思 → P2 full。
4. **出图 + 解释**。
5. **(并行/可选) L1 纯追踪修法**: 用户待 review `l1_fix_proposal_purepursuit_v1.md`。决定后实施, 让转向类避障也 τ_ego 敏感 + 消除撞墙基线。

---

## §2 关键资产 (路径)

- **repo**: `/home/jichengzhi/V2Xverse` (branch `feature/l1-trajectory-tracker`); 主仓 `/home/jichengzhi/V2X`。
- **跑实验脚本**: `scripts/te_multiroute_sc.sh <gpu> <port> <tau> "<route_ids>"` — 单卡扫一组路线一个 τ_ego 档。已修 unbound-var bug。
- **conda**: `/home/jichengzhi/miniconda3/envs/v2xverse/bin/python` (py3.7)。
- **CARLA**: `external_paths/carla_root/CarlaUE4.sh` (0.9.10.1)。
- **eval**: `scripts/eval_driving_e2e.sh <route> <port> <tag> <repeat> <agent> <scen>`; agent name `codriving_te{TAU}_l1` → config `pnp_config_codriving_te{TAU}_l1.yaml`。
- **路线**: `simulation/leaderboard/data/evaluation_routes/town05_short_r{0..331}.xml` (105 条)。
- **结果**: `results/results_driving_{tag}/.../ego_vehicle_0/results.json` ← ★权威。
- **L1 代码**: `simulation/leaderboard/team_code/closedloop/l1_trajectory_controller.py` + `pnp_infer_action_e2e.py`(run_l1_step, L1PlanStore wiring, tau_ego_ms 读取)。
- **te219 全量结果**: `results/results_driving_mr_te219_r*/`。

---

## §3 怎么补延迟档 config (新窗口先做)

```bash
cd /home/jichengzhi/V2Xverse/simulation/leaderboard/team_code/agent_config
for TAU in 50 150 200 250 300 350 400 450 500; do
  sed "s/tau_ego_ms: 108/tau_ego_ms: ${TAU}/" pnp_config_codriving_te108_l1.yaml \
    > pnp_config_codriving_te${TAU}_l1.yaml
done
# 校验: grep tau_ego_ms pnp_config_codriving_te*_l1.yaml
```

---

## §4 多 Agent 启动 (新窗口)

参 `exp_design §4`。建议:
```
TeamCreate → 3 agent:
  exp-runner (sim-integrator): 跑 (route×τ_ego) 矩阵, 管 CARLA/GPU/端口/重试, 落 results.json
  exp-supervisor (supervisor): 聚合 mean DS+碰撞率+CI, 验单调性, 不符则诊断(机制/噪声/库/转向免疫/端口), 落反思日志
  exp-analyst (可 main 兼): 出折线图 + 机理解释
```
- runner 跑前必 `nvidia-smi` 确认卡空(共享服务器, 他人占用浮动); `setsid nohup` 起; **端口宽间隔 ≥100**。
- supervisor 对 runner 自报必复核 results.json 实存。
- **main 不轻信 agent 自报"已完成/已修"**, 复跑/读文件核验(项目纪律)。

---

## §5 ★教训 (本轮血泪, 别再踩)

1. **CARLA TrafficManager 端口冲突**: `eval_driving_e2e.sh` 里 `TM_PORT=PORT+5`。多实例并行时, 一个实例的 world-port 不能等于另一个的 TM 端口(=PORT+5)。**端口必须宽间隔 ≥10(建议 ≥100)**。撞了报 `rpc::rpc_error during call in function version`, 整批 route 静默 ERR(像首轮 g3/g6 全废)。
2. **CSV 被 truncate**: `te_multiroute_sc.sh` 启动时 `echo header > summary.csv` 会清空同名文件。**真相源永远用 results.json 实存**, 别信 summary CSV(尤其多批复用同 g 号时)。
3. **静默崩**: 脚本 `set -u` 下任何未绑定变量 → 第一条 route 崩 → trap teardown 杀 CARLA → 整批没。**放后台后立即查"是否越过第一条 route 真出结果"**, 别盲信进程数。
4. **共享 GPU**: latency 不准但 DS/RC 准(DS 验证用 `SKIP_GPU_GUARD=1`)。跑前 nvidia-smi 看真实空卡(util+mem)。
5. **单路 n=1 噪声大**: "恰好 1 碰撞"洗牌使 DS ±噪声; 单路看不出 τ_ego 单调。必须大库聚合 + n 重复。
6. **failure 是 per-km float 非 list**: 解析碰撞用 `float(v)` 不是 `len(v)`。
7. **H800 不可用于此实验**: H800(sm90) 上老栈 torch1.10/cu113 第一个 CUDA 算子就卡死(见 `HANDOFF_new_server_deploy_v1.md §7.4`)。**本实验只能在旧 4090 跑**。

---

## §6 已知科学结论 (背景, 别重新发现)

- **τ_ego 敏感信号来自行人横穿(走速度/刹车路径)**: 延迟→晚刹车→撞行人。I-2 已支持, 现有 L1 即可显现(r104/r136/r3/r317 = te0 100→te219 50)。
- **车辆交互(S4)噪声大**(双向洗牌), 不如行人干净。
- **撞墙(layout)对 τ_ego 免疫**(L1 转向走全局路由, 非感知驱动) → 这类失败不响应延迟。纯追踪修法可解(让转向跟规划轨迹)。
- **必须 te0 对照**: "te219 撞了"不等于"τ_ego 敏感"; 要"te0 不撞→te219 撞"才是延迟造成。23 行人路线里部分(r21/320/330)te0 也撞=route 难, 非干净信号。

---

## §7 接手即可执行的第一步

```bash
# 1. 看 te0 确认批跑完没
cd /home/jichengzhi/V2Xverse
for r in 3 15 16 17 18 312 317 318 320 324 327; do
  j="results/results_driving_mr_te0_r${r}/v2x_final/town05_short_collab/r${r}_repeat0/ego_vehicle_0/results.json"
  [ -f "$j" ] && python3 -c "import json;d=json.load(open('$j'))['_checkpoint']['global_record'];print('r${r} te0 DS=',d['scores']['score_composed'])"
done
# 2. 补延迟档 config (§3)
# 3. 建库 scenario_library_v1.txt (§ design §1)
# 4. TeamCreate 拉起 runner+supervisor 跑 P1 pilot
```
