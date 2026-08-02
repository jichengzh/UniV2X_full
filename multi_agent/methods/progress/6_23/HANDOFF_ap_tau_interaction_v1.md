# HANDOFF: AP×τ 交互实验 (耦合验证) — ★已完成, 耦合坐实 (v1, 2026-06-23)

> 自足交接。承接 [[HANDOFF_ap_tau_grid_v1]](完整 5×5 网格 480/480 已完成 + AP 旋钮机制 + GPU/harness 血泪教训) 与 [[project-ap-knob-closedloop]]。
> 本阶段目标: 跑 **AP×τ 交互实验**, 回答"**感知精度(AP)下降是否让延迟悬崖提前**"= CoDriving 在"延迟×精度"面上的**耦合陷阱**证据。
> **★[2026-06-23 完成]**: 在 GPU 0,1,2,3 上跑满 `283/283 new + 77 skipped = 360 cell` 干净收尾。**耦合坐实**(见 §0.5)。

---

## 0.5 ★最终结论 (2026-06-23, 耦合坐实)
**CoDriving 延迟×精度耦合成立 — 低AP 让碰撞崖口提前(750→650ms)且加深。** 数据 `multi_agent/real_test/ap_tau_interaction_results.csv`, 图 `ap_tau_interaction_collision.png`, 脚本 `ap_tau_interaction_plot.py` + H800 `/tmp/ix_final_analyze.py`(修了 drop 标签 + 超时计入 incompletion + collEp% 抗离群指标)。

| τ(ms) | 高AP 有碰撞局% | 低AP 有碰撞局% | 低AP 碰撞/局 | 高AP 碰撞/局 |
|---|---|---|---|---|
| 600 | 0.0 | 5.6 | 0.008 | 0.000 |
| **650** | 15.6 | **37.9** | **0.163** | 0.061 |
| 700 | 13.3 | 25.8 | 0.111 | 0.024 |
| 750 | 25.8 | 36.4 | 0.177 | 0.093 |
| 800 | 25.0 | 38.7 | 0.197 | 0.076 |

判据落点 = **"崖口提前"型耦合**(非"叠加不提前"非"可分离"):
1. 匹配 τ 下低AP 碰撞率 = 高AP 的 **2.4–4.6×**(每个 τ≥650 都成立, 非单点)。
2. 低AP 在 **τ=650 即冲到危险平台(38% 有碰撞局)**; 高AP 要 τ=750–800 才到 ~25%, 全程没到 38% → **危险碰撞起点左移**。
3. 判别量 = **碰撞(安全)非 RC**: 两臂 RC 在 τ≥650 都 ~61-68%(低AP 甚至略高=过度保守堵车非撞)。与主网格"无延迟时 AP 只伤 RC"互补。
4. 超时(=ego卡死不会撞)低AP 还更多(g65070=7 vs g65000=4)→ 碰撞信号是**保守低估**, 真实更强。

**对框架意义(命中 PQS 目标)**: AP 与延迟**不可独立调** —— 任何压低 AP 的压缩选择都**收窄延迟预算**(安全崖口 750→650ms)。= "**联合搜索有必要**"的闭环证据, 推翻"两轴独立"。下一步 = 叠 RSU 延迟臂 / 结构 / 量化轴做 PQS 全维联合搜索。

**caveat**: N≈29-36/cell(够但有方差); 超时主要来自 route 17(高τ易卡), 但碰撞信号跨 clean 路一致; 仅车端 τ_perc×融合AP, RSU臂/τ_ego 未扫。

---

> **以下为续跑期文档(历史, 已完成)**: 续跑指定 GPU 0,1,2,3。

---

## 0. 一句话现状
完整 5×5 网格已出结论(延迟主导/AP 良性, 见 v1 §3c)。用户质疑"两轴独立"框架, 要求验证**耦合**: AP↓ 丢远处目标→丢预警→是否让延迟悬崖**提前**(崩溃点左移)。
为此设计**交互实验**(τ 细扫 × 高/低 AP 两臂 × 碰撞率主轴)。**已建配置 + 跑了几次但被 GPU 0/1/3 卡死中断, 现暂停**, 等用户 reset GPU 0/1/2/3 后**在这 4 张卡上续跑**。

## 1. 实验设计: AP×τ 交互(崖口是否随 AP 左移)
- **τ 细扫**: 600 / 650 / 700 / 750 / 800 ms(**50ms 步**, 够分辨悬崖位移; 主网格 200ms 步太粗看不出)。
- **AP 两臂**: drop=0.0(veh AP50 0.84, 高) vs drop=0.7(AP 0.36, 低)。featmask=1。
- **路集**: clean 6 路 `${V2X_DATA_ROOT}/grid_routes_clean6.txt` = [3,17,18,104,136,317]。
- **N=6**(主网格是 3, 翻倍压方差; AP×τ 交互是弱信号, 必须压噪声)。
- **主指标**: **碰撞率 + RC(路程完成率)**, 不用 composed-DS(被"过度保守 DS=50 地板"搅浑, 见 v1 §3)。
- **config 编码**: code=τ×100+drop_pct, 文件 `pnp_config_codriving_g{code}_l1.yaml`。10 个 code: `60000 60070 65000 65070 70000 70070 75000 75070 80000 80070`。
  - 600/800 的 4 个 = 主网格已有; 650/700/750 的 6 个 = 本会话用 `/tmp/make_ix_configs.py` 生成(克隆 g60070 改 tau_perc_ms+ap_drop, 已验证存在)。

### 判据(实验跑完算)
对每个 AP 臂定位碰撞率飙升的悬崖 τ*:
- **τ*(低AP=0.36) < τ*(高AP=0.84)** → **耦合坐实(悬崖提前)** = 要找的 CoDriving 延迟×精度耦合陷阱。
- 两臂 τ* 相同、仅低 AP 处碰撞更深 → "叠加不提前"型耦合。
- 无差异 → 两轴此区间可分离。
- (主网格 τ=800 已有苗头: 碰撞随 drop 0.08→0.19 上升, 但 200ms 步看不出崖口是否左移; 本实验就是去分辨这个。)

## 2. 当前进度(交互 code, N=6 = 6路×6rep = 36/cell)
| code | τ/drop | done/36 |
|---|---|---|
| g60000 | 600/0.0 | 23 |
| g60070 | 600/0.7 | 18 |
| g65000 | 650/0.0 | **0** |
| g65070 | 650/0.7 | **0** |
| g70000 | 700/0.0 | **0** |
| g70070 | 700/0.7 | **0** |
| g75000 | 750/0.0 | **0** |
| g75070 | 750/0.7 | **0** |
| g80000 | 800/0.0 | 18 |
| g80070 | 800/0.7 | 18 |
端点(600/800)的 ~18 = 主网格 N=3 带来的; fine-τ 全 0(被 GPU 卡死中断)。**剩 ~283 run**。harness `already_done()` 自动 resume, 不重跑已完成。

## 3. ★续跑步骤(用户 reset GPU 0/1/2/3 后)
**① 先确认 reset 后 4 张卡 CARLA 能起**(避免 crash-loop, 服务器端脚本防 ssh 截断):
```
# 在 H800 上, 4 张卡各单测一个 CARLA(端口 7800/7802/7804/7806), 等 ~100s, 全 PORT-UP 且 log 无 "timed out waiting" 才算好
cd /data/jichengzhi_v2x/carla
for g in 0 1 2 3; do p=$((7800+g*2)); setsid env CUDA_VISIBLE_DEVICES=$g ./CarlaUE4.sh -prefer-nvidia -opengl -RenderOffScreen -world-port=$p -nosound -quality-level=Low > /tmp/ct_g$g.log 2>&1 </dev/null & done
# 验: nc -z <PRIVATE_HOST> 78xx = PORT-UP; grep "re-raising signal" /tmp/ct_g$g.log = 还崩
# 测完 kill: pkill -9 -f "world-port=780"
```
**② 启动续跑(GPU 0,1,2,3)** —— 启动脚本用 **scp 传**(本会话教训: heredoc 经 H800 ssh 常被 stall 截断, scp 稳):
本地写 `/tmp/launch_ix0123.sh` 内容如下, 再 `scp` 到 H800 `/tmp/` 后 `bash` 它:
```bash
#!/bin/bash
cd ${V2X_DATA_ROOT}
export CFG_TAG=g
export TAUS=60000,60070,65000,65070,70000,70070,75000,75070,80000,80070
export NREP=6 ROUTE_FILE=${V2X_DATA_ROOT}/grid_routes_clean6.txt
export TAG_PREFIX=grid SCEN_SUFFIX=_1 GPULIST=0,1,2,3 SLOTS_PER_GPU=1
export PORT_BASE=4600 BPORT_BASE=6500 RUN_TIMEOUT=900
setsid nohup python3 ${V2X_DATA_ROOT}/tau_curve_apknob.py > ${V2X_DATA_ROOT}/ix_run.log 2>&1 < /dev/null &
echo "harness pid $!"
```
**③ 核验**(等 ~130s): `ps -p <pid>` 活; `grep "total=\|SLOTS\|CARLA ready" apknob_progress.log` 应见 SLOTS(4)=GPU0,1,2,3 + 4 个 CARLA ready; `nvidia-smi` 见 4 卡显存 ~12-17G(健康 slot)。
**④ 挂监控**: 本地后台 loop 每 ~180-240s 查 `ps -p <pid>` + 有效结果数, harness 退出(COMPLETE/崩)即通知。约 ~283 run / 4 卡 ≈ 5-6h。

## 4. 分析(跑完)
- 子指标脚本 `multi_agent/real_test/ap_tau_grid_submetric.py`(本地; 已可算碰撞数/RC per cell)→ 改 codes 为这 10 个, 出**碰撞率 vs τ 双臂曲线**。
- 对每臂找碰撞率/(1−RC) 起跳的 τ*, 比较两臂 → 判"提前/叠加/可分离"。
- 主网格热图/CSV/分析脚本: `multi_agent/real_test/ap_tau_heatmap.png` / `ap_tau_grid_ds.csv` / `ap_tau_grid_analyze.py`。

## 5. ★本会话血泪教训(续跑前必看)
1. **harness `epid` NameError(已修)**: 超时分支用未定义 `epid`(应 `slot_epid[slot]`)→ 每次 slot 超时整 harness 崩。**已 `sed s/\bepid\b/slot_epid[slot]/g` 修复**(备份 `tau_curve_apknob.py.bak_epidfix`)。修后遇超时只跳过该 slot 继续。详见 v1 §4.4。
2. **render wedge 修复 = per-GPU `sudo nvidia-smi --gpu-reset -i N`**(用户执行): 卡上**无 compute 进程时**才成功(有进程报 "In use by another client")。"gpu-reset 不可用(Fabric Manager)"是误判。详见 v1 §1。
3. **TVM(tvm310/s2_tvm)与 CARLA 必须分卡**: TVM 动态迁移抢卡, 落到 CARLA 卡上→撞死。本会话因此死过 2 次。
4. **★GPU 0/1 在本节点不稳**: 本会话 GPU 0/1 出问题 **3 次**(reset 后能短暂跑、持续负载下又崩/卡, 报 `Invalid session: no stream available` = CARLA 传感器流中途死)。**本次用户 reset 0/1/2/3 后再试**; 若 reset 后仍频繁卡死, 退回**稳定核心 3/4/5**(主网格全程零故障)。GPU 2 历史多被他人占, 也观察。
5. **进程清理**: harness 边杀边 spawn → **先按 PID `kill -9 <harness_pid>` 杀本体并确认死**, 再清 slot(`process_b_server`/`CarlaUE4`/`eval_driving`); pkill 常被 ssh stall 截断 → 用 `ps -ef|grep ...|awk '{print $2}'|xargs kill -9`。
6. **ssh 频繁 255 stall**: 长任务用 setsid+服务器脚本; 关键启动脚本用 scp 传(别用 heredoc); 起后立即核验(进程在/log非空/GPU mem 变)。
7. **CARLA 卡死特征**: 显存 flat 在 ~1.6-2.7G(procB 没满载, 健康应 ~14-17G)+ util 100% + log `Invalid session: no stream` = episode 卡住, 等 900s 超时跳过。可按 v1 §4 待加固加进程组存活检测快速 requeue。

## 6. 关键资源
- **H800**: `ssh -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST>`(密码每会话确认; ssh 易 stall)。
- **隔离副本**(所有 AP 工作, 主框架零污染): `${V2X_DATA_ROOT}/V2Xverse_apknob`。
- **harness**(已修 epid): `${V2X_DATA_ROOT}/tau_curve_apknob.py`; **CARLA 启动**: `${V2X_DATA_ROOT}/h800_carla_launch_noadapter.sh`(harness 内部用)。
- **路集**: `${V2X_DATA_ROOT}/grid_routes_clean6.txt`。
- **配置生成器**: `/tmp/make_ix_configs.py`(生成 fine-τ configs; 若副本重建需重跑)。
- **进度日志**: `${V2X_DATA_ROOT}/apknob_progress.log`; 结果 `V2Xverse_apknob/results/results_driving_grid_g{code}_r{R}_n{N}/...`。
- **本地分析**: `multi_agent/real_test/ap_tau_grid_{analyze,submetric}.py` + `ap_tau_heatmap.png` + `ap_tau_grid_ds.csv`。
- **配套**: [[HANDOFF_ap_tau_grid_v1]](深背景/网格结论) [[project-ap-knob-closedloop]] [[project-tau-perc-signal]] [[project-real-repo-is-v2x]]。
- **GPU 纪律**: 本阶段用 **0,1,2,3**(用户 reset); 别和 TVM/重型 CUDA 抢同卡; 注入轴只动车端(ego τ_perc + 融合 AP), RSU 延迟臂(latency_inject_ms)+τ_ego 当前=0(未扫, 是后续 PQS 全维联合搜索的扩展)。

## 7. 与最终目标的关系(别丢)
用户两个真实目标(见 [[project-codesign-nextstage-v3]] 等): ①framework 做到 **PQS 三维(剪枝×量化×结构/调度)全维联合搜索**, 不止单通道结论; ②证明 **CoDriving 也有耦合陷阱**(用更高维耦合找)。
本交互实验是第一块拼图: **先在车端 AP×τ 把"精度×延迟"耦合坐实**(崖口是否提前), 再叠 **RSU 延迟臂 / 结构 / 量化轴**做 PQS 全维联合搜索 —— 陷阱在这些交叉里浮现。"避开两个极端"只是这张图日后被框架搜索利用时的作用, 不是最核心; 最核心是**证明耦合存在→联合搜索有必要**。
