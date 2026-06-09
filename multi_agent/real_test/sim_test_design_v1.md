# V2Xverse 闭环仿真测试 — 设计与初始验证 (sim_test_design_v1)

> 整合: team-lead (Claude), 2026-06-09。
> **本文是闭环仿真测试部分的唯一权威设计文档**, 由 `sim_closedloop_arch_v1.md`(实装架构+实测记录) + `latency_aware_simulation_design_v1.md`(理想设计愿景) 整合而来; 两份源文档已归档到 `multi_agent/archive/`。
> 配套(同目录 real_test/): `closedloop_v2x_validation_survey_v1.md`(平台/方法选型调研) · `v2xverse_framework_analysis_v1.md`(框架全景调研) · `edge_latency_budget_v1.md`(Orin 200ms 时延预算)。时延档真测源 = `methods/design/orin_pyramid_baseline_report_v1.md`(E7/E8 ground truth)。
>
> **诚实纪律(贯穿全文)**: ① 每个数字标口径(真测 / 估算 / 注入档 / 设计稿); ② **严格区分「已实现并验证」「已设计未实现」「待跑」** —— 实装比理想愿景简化, 不假装愿景已落地; ③ 闭环指标与感知 AP/AMOTA 不同任务口径, 绝不混表混曲线。

---

## §0 当前进展速览 (截至 2026-06-09)

| 阶段 | 内容 | 状态 | 关键结论 |
|------|------|------|---------|
| **Sim-A** | 环境构建 + 冒烟 | ✅ 完成 | v2xverse env(py3.7)+CARLA 0.9.10.1+CoDriving ckpt 全装齐; 冒烟 r0 链路通(DS 12.5/3 行人碰撞, **只证通不入库**) |
| **Sim-B** | 时延感知 + 算力隔离改造 | ✅ 完成并 commit | 动态时延→帧折算 Δ=ceil(L/50)+ZOH(融合前只 RSU 路); 推理拆独立进程 CUDA-Event 计时; C-4 双对照 d0/d108 帧对齐正确, 单测 13 项过 |
| **Sim-C** | 时延 sweep | ⚠️ 部分 | r0 pilot 8 档全跑完但**信号全平**(DS 全 100, norsu 也 100)。**P0 诊断已定论: 路侧融合真接入 ego control(架构无 bug), 全平真因=r0 太易**(§3.4)。r146/r160 探针起跑即被用户叫停 |
| **Sim-D** | 闭环指标入库 | ⏸ 仅草案 | schema DRAFT 落盘(cl_ 前缀, 独立 regime), `dataset_v2` 零行(等信号有效再入, §4) |

commit 链(V2Xverse `sim-closedloop` 分支): `68196ae`(spconv 兼容)→`9abe0dc`(C-3 aligner)→`de3b509`(C-1 推理进程)→`a3bacec`(C-2/C-3 接线)→`fa4e8e6`(C-5 disable_rsu)→`45bf4e0`(C-6 sweep 驱动)→`addfdbb`(C-7 6-route)→`ab53d68`/`ab26bfb`(C-8 单卡模式)→`0b2a2fd`(C-9 并行安全多实例)。

---

## §1 测试目标与核心假设

- **核心假设(待验证, 来自 closedloop survey §1)**: **路侧 e2e 推理 < 200ms 时, V2X 对驾驶安全(碰撞率↓/驾驶分↑)有可证明提升; 超过某临界时延后收益崩塌。** 产出 = "驾驶分 vs 时延"完整曲线 + 拐点定位。
- 假设分两个子命题: **A** = 路侧 e2e 可达 <200ms(由 `edge_latency_budget` + `orin_pyramid_baseline` 真测回答, 已基本证实: Orin INT8 p75 e2e ~108ms ✅); **B** = V2X 对闭环驾驶有提升且随时延衰减(**只能闭环回答, 即本仿真测试的任务**)。
- 对照量级: CoDriving 论文(arXiv:2404.09496)报协同 vs 单车 **DS +62.49% / 碰撞率 −53.50%** —— 我们 sweep 的"无路侧档 vs 0ms 档"应复现这个量级方向。

本测试拆成两个互锁的设计问题: **如何把感知接进闭环并保证有效(§2)** + **如何把算力/时延分离出来可控注入(§3)**, 再用一串**初始实验逐项证明设计成立(§4)**。

---

## §2 感知仿真测试如何设计

### 2.1 测试链路与数据流(实查 V2Xverse 代码确认)

闭环驾驶 agent(`pnp_agent_e2e.py`)用 **实时 CARLA 传感器**(RGB/LiDAR 在仿真里 spawn), **不依赖离线 45GB 数据集**。每帧数据流:

```
CARLA 实时传感器(20Hz, ego + RSU/CAV)
  → 感知 perception_model (CoDriving center_point)
  → intermediate fusion (record_len = ego + RSU/CAV, RSU 真进融合)
  → planner (MotionNet) 输出 waypoints
  → PID 控制 → CARLA actuator
```

| 既有事实 | 值 | 来源(实查) | 设计意义 |
|---------|----|-----------|---------|
| CARLA 步长 | sync @ 20Hz = **50ms/帧** | `leaderboard_evaluator_parameter.py` L128/273 | 时延→帧折算的基本量子 |
| 感知推理节拍 | 每 `skip_frames=4` 帧 = **200ms/次** | `pnp_config_codriving_5_10.yaml`; agent L448 | 时延注入作用在"结果何时可用", 非每帧 |
| 感知→规划→控制 | 单进程内串行 `get_action_from_list_inter` | agent L472-476; infer L505-565 | 时延注入点在融合**前**(delay raw RSU) |
| 感知缓冲设施 | `perception_memory_bank`/`pre_raw_data_bank`/`prev_control` 已存在 | infer L466/480/477 | ZOH 可复用 `prev_*` 模式 |

### 2.2 平台与模型选型 — 为何用原生 CoDriving 而非团队 Pyramid (★诚实报告)

团队加速 Pyramid(HEAL m1, DAIR 训练)与 V2Xverse CoDriving(CARLA 数据训练)**架构 + 数据集双重不匹配**:

| 维度 | 团队 Pyramid | V2Xverse CoDriving | 兼容 |
|------|-------------|--------------------|------|
| 模型类 | `HeterPyramidCollab` | `center_point_codriving` | ❌ |
| 训练数据 | DAIR-V2X(真实路测) | CARLA 仿真 | ❌ 跨数据集 |
| 检测头/voxel range | Pyramid 多尺度 / DAIR range | center-point multiclass / voxel 0.125 [-12,-36,-22,12,12,14] | ❌ |
| 真测资产 | E7: Orin body FP32 131/FP16 48/INT8 p75 20ms | — | latency 可用 |

- **结论(诚实)**: 不能直接 load 权重替换。硬塞 = 跨数据集脏推理, 项目纪律明令禁止(CLAUDE.md ISS-012 同类)。
- **⇒ 集成形态 = 时延档注入(latency-injection form)**: **模型仍用原生 CoDriving**(保 CARLA 数据上 AP 有效); **用团队 E7 真测时延档驱动 §3.4 的延迟帧数 Δ**。这正是回答"路侧推理 <200ms 是否提升驾驶安全"的**最小充分形态** —— 不需要团队模型真在 CARLA 跑, 只需真测时延档驱动延迟, 看驾驶分随时延变化。
- **future work(非必须)**: 若要"团队模型真在 CARLA 跑", 须在 CARLA 数据上重训 Pyramid(大工程, 超当前 scope)。

### 2.3 环境构建关键点 + 与官方栈的偏差(★supervisor 核验用, 详见附录 A)

- **sm89 风险已实证排除**: 4090 上 torch 1.10.1+cu113 无 sm89 cubin 但走 PTX-JIT 前向兼容(首次算子 1.58s 一次性暖机, 稳态 0.21ms/iter, GPU vs CPU maxdiff 9.2e-4)。**保官方栈 0 偏差, 不升级 torch。**
- 主要偏差 **D-3**: spconv 1.2.1 源码 build → **spconv-cu113 2.3.6 wheel**(opencood 自带 v2 兼容分支 `Point2VoxelCPU3d`, voxel 输出已真验证)。CoDriving 用 PointPillar 密集 BEV backbone, 当前路径不依赖稀疏 3D 卷积, 风险低。
- ckpt 加载: **missing 2 / unexpected 0**(0 unexpected = 架构匹配)。其余偏差(下载手段/setuptools/numpy)无功能风险。完整偏差表 D-1~D-5 见附录 A。

---

## §3 算力分离仿真测试如何设计

### 3.1 为什么必须算力分离

感知融合 = ego 本车感知 ⊕ RSU/CAV 感知。要测"路侧推理时延对驾驶的影响", 必须拿到**纯净的推理时延**。若 CARLA 渲染与推理同卡, **渲染负载会污染推理时延测量**(冒烟单卡 duration 660s 即此)。⇒ 算力分离 = 把推理拆到独立进程/GPU, 用 CUDA-Event 只计 forward。

### 3.2 进程拓扑 (★实现形态 — 本机分卡)

```
进程 A: CARLA server (GPU_render)         进程 B: leaderboard+agent 主控
  - 世界/渲染/物理 @20Hz sync     ◄─RPC─►   - run_step 每帧调; ego 感知/融合/规划/控制
  - 唯一占 GPU_render                        - ★感知推理 NOT 在本进程跑
                                              │ IPC (zmq REQ/REP + msgpack_numpy)
                                              ▼
                                    进程 C: 推理服务 (GPU_infer)
                                      - 加载 CoDriving 感知
                                      - CUDA-Event 只计 forward
                                      - 返回 (det 结果, latency_ms)
```

- **算力隔离落实**: A `CUDA_VISIBLE_DEVICES=<render>` 仅渲染; C `CUDA_VISIBLE_DEVICES=<infer>` 仅推理; 物理分卡 ⇒ 推理时延不被渲染污染。
- **IPC 决策**: py3.7 无 `multiprocessing.shared_memory`(3.8+ 才有), 故 shm 主方案不可用, **采用设计预案的回退: zmq REQ/REP + msgpack_numpy**(实测可用)。序列化开销(~1-2MB 点云)计入 IPC 往返、与纯 forward 时延分开标注; **CUDA-Event 仍只计 forward, 口径纯净**。
- **二元组契约**: 进程 C 每次返回 `(perception_result, latency_ms)`。Sim-C 模式下 `latency_ms` 可被配置为"注入档"(team E7 数字)替代实测, 口径明确标 `injected_from_E7` 而非 `measured_in_carla`。
- **Orin 版(延伸目标, 见 §3.7 HIL)**: 进程 C 换成 Orin AGX 推理服务, IPC 走网络 socket, 返回 Orin 真实端侧时延。本机分卡先打通。

### 3.3 统一时延模型 — 一条时间轴, 两个注入点 (★设计原理, from 理想设计愿景)

> 来自 `latency_aware_simulation_design` 的核心论证。**实装(§3.4)采用其简化形态**, 但原理决定了为何这样简化是对的。

通信延迟 τ_comm 与计算延迟 τ_compute 作用在管线**不同位置**, **不能合成一个标量**:

```
RSU/协作车: 传感器 → 编码(τ_collab_compute) → 发送 ─┐ τ_comm(传输)
自车:       传感器(t) ───────────────────────────→ 融合+规划(τ_ego_compute) → 控制生效(+τ_act)
```

- **τ_comm 效果**: 协作特征与自车特征**时间错位** → 空间错位/动态目标鬼影/漏检。**不推迟自车出控制**。
- **τ_compute 效果**: 基于 t 时刻数据算的计划要到 t+τ_compute 才可用 → **控制反应滞后**。
- 正确统一量 = **逐路径信息年龄(per-path information age)**: `age_collab = τ_collab_compute + τ_comm + τ_wait + τ_ego_compute + τ_act`; `age_ego = τ_ego_compute + τ_act`。两条路径年龄不同, 必须分注。CARLA 50ms/帧下全量化为帧数。

### 3.4 时延→帧折算 + 零阶保持 (★已实现, Sim-B C-3)

实装把原生静态 `comm_latency`(配置写死常数 + 进程内同步直调)升级为**动态实测/注入时延驱动 + ZOH 跨进程异步**:

- **折算**: `Δ = min(ceil(inject_ms / 50), MAX_DELAY_FRAMES=10)`。**ceil(向上取整)**: 推理没算完的帧绝不提前可用(安全侧)。例: L=20→Δ=1; L=108→Δ=3; L=219→Δ=5; L=500→Δ=10(封顶, 对齐 CoBEVFlow 上界)。
- **多帧延迟语义**: frame-id keyed buffer **排队不丢帧**; ego 融合取"已可用且最新"的 RSU 结果(满足 `源帧id+Δ ≤ k` 中源帧 id 最大者)。
- **ZOH 位置 = 融合前、只作用 RSU/CAV 路**: 当 `step-Δ` 的 RSU 结果不可用时用 `last_known_rsu` 顶替; **ego 本车感知每帧实时, 不经 ZOH**。放融合后会冻住 ego 当前感知 → 违反"ego 实时", 故必须融合前。
- **非阻塞提交(关键不变量)**: CARLA sync 模式下 B 的 wallclock 阻塞不进 game time, 故 game-time 延迟**唯一来源 = Δ**(帧数×50ms), 与 B 是否阻塞无关。但异步提交让 C 推理与 B tick 重叠, 省大量墙钟。⇒ B 提交后立即返回, 绝不同步 join C。
- **6 字段审计 log**(`meta/latency_align_audit.csv`, 供 supervisor 核验 + Sim-C 因果分析):

| 字段 | 含义 |
|------|------|
| `step` | 当前融合帧号 |
| `Δ` | 该帧 RSU 结果延迟帧数 = ceil(L/50) |
| `used_rsu_frame_id` | 实际用的 RSU 结果来自哪个源帧(后推对齐核验) |
| `is_zoh_held` | 本帧 RSU 是否走零阶保持 |
| `latency_ms_source` | `measured_in_carla` / `injected_from_E7` — 口径标注 |
| `zoh_age_frames` | 当前 RSU 结果距今几帧 = step−used_rsu_frame_id−Δ ≥0 — **因果链中介变量** |

### 3.5 延迟期间车辆控制保持策略 (★三档 — L0 已实现, L1/L2 设计待实现)

真实自动驾驶栈规划-控制分频(规划 ~10Hz 有延迟, 控制器 50-100Hz 跟踪最近轨迹)。三档(也是消融对象):

| 档 | 策略 | 状态 | 说明 |
|----|------|------|------|
| **L0** 指令零阶保持 | 非推理帧冻结上一条 throttle/steer/brake | ✅ **当前实装 = L0** | 延迟越大失真越大(弯道固定转向 6 帧/300ms 显著偏离)。**保留只为量化"仿真方法本身"的误差** |
| **L1** 轨迹跟踪保持 | waypoints 锚定世界系+时间戳; PID 每 tick 跟踪"最近计划"; 按 (t_now−t_anchor) 时间索引插值取参考点 | 🔶 **设计推荐, 未实现** | 对应真实系统: 车沿旧轨迹推进, 对新障碍反应滞后 age_ego, 但不失控。需 planner 输出转世界系+t_anchor |
| **L2** 安全看门狗 | plan age 超阈(500ms)降速; 超更大阈(1s)缓刹保持车道 | 🔶 **可选叠加, 未实现** | 没它则大延迟尾部是"无意义失控"而非"有意义降级" |

- **实验价值**: L0 vs L1 的 DS 差 = "仿真保真度修正量"; L1 vs L2 = fallback 策略研究。
- **⚠️ 当前局限**: 实装是 L0(感知陈旧 + 控制零阶保持), pilot 全平部分原因可能是 L0 下 planner 对感知延迟不够敏感(§4.7 证伪条件 2)。**L1 是提升时延敏感度的关键升级项**, 列为后续。

### 3.6 计算延迟三种来源模式

| 模式 | 做法 | 状态 | 适用 |
|------|------|------|------|
| **HIL Orin 在环** | 仿真机序列化输入发 Orin, Orin 跑 TRT engine 回传 {结果, τ_infer} | 🔶 未实现(P2) | 最高保真: 真硬件/真模型产物/真延迟分布; 同时解决"仿真与推理抢本机 GPU"。是用户"一张卡纯做推理"设想的真正落地形态 |
| **trace 回放 / 延迟档注入** | 查 {模型×精度×平台} 静态表(edge_latency_budget)或回放 HIL trace; 推理本机跑 | ✅ **当前用此**(injected_from_E7) | 严格可复现、不依赖 Orin 在线 |
| 在线计时 | `torch.cuda.Event` 测每帧耗时 | ✅ 用于 C-1 进程纯 forward | 测仿真机, 受渲染争抢污染则不可迁移; 分卡后纯净 |

### 3.7 单卡共享模式 caveat (★Sim-C 实跑用)

资源紧张时 `SIM_SINGLE_CARD=1` 绕过双卡 abort, 结果标 `isolation=single_card_shared`:
- **对 sweep 有效性无影响**: CARLA sync 模式下感知结果确定、控制逐位一致、Δ 由注入值算而非墙钟 ⇒ driving score 与分卡相同, 仅墙钟更慢。算力隔离保护的是"现场实测时延纯净度", **注入档不需要它**。
- 论文/入库三标注一字不漏: `injected_from_E7` / `single_card_shared` / 确定性单次。

---

### 3.8 ★RSU 时延感知回灌 — 定稿实现方案 (用户 2026-06-09 拍板)

> 用户原话要求: "记录每次路测设施的推理时延, 将时延信号提供给车端, 用于确定当前路测推理结果后推多少帧应用; 无路测信息时车辆继续用上一帧路测感知结果。skip 多少帧不该手动设定, 而是延时感知机制决定。"
> **拍板**: ① 时延源 = **trace 回放(A2)**, 用**原生 CoDriving 在 4090 采集**逐次推理时延, **算力分离**(采集设备 ≠ 仿真设备); ② **只做 RSU 真实逐次时延**, 车端 τ_ego **暂定 = 0**(推理无时延, 不动 skip_frames / 不做 ego 调度器, 留作第二步)。

**澄清两个被混淆的时延(见 §3.3/§3.5)**: `skip_frames=4` 是 **τ_ego 车端节拍**(本步 τ_ego=0, 暂不碰); 本方案改的是 **τ_RSU 路测时延 → 后推帧数 Δ**。两者正交。

#### 数据流(本步落地形态 — 用户 2026-06-09 定: 真实点云分卡采集)
```
[采集一次] 分卡跑一次 r0 短 route: CARLA(GPU_render) ∥ CoDriving 推理进程C(GPU_infer=4090)
           进程C 每次 forward → CUDA-Event 真实逐次 τ(真点云、真分卡、随场景体素数真变化)
           → rsu_latency_trace.csv (call_idx, τ_ms, 排除前 20 warmup)
           ↓
[仿真时] 每次 RSU 推理(每 skip_frames 触发):
   从 trace 取下一个 τ → Δ = ceil(τ/50)   ← 时延信号逐次驱动后推帧数(非手设)
   结果标"自帧 (源帧+Δ) 起可用" → frame-keyed buffer (②已建)
   车端帧 k 融合: 取 (源帧+Δ)≤k 最新 RSU; 无新→ZOH 沿用上一份(③已建), zoh_age++
   ego 本车感知当前帧实时(τ_ego=0)
```

#### 实现清单(★只补时延源 ①, 复用已建 ②③④)
| 步 | 改动 | 卡 | 状态 |
|----|------|----|------|
| **R-1 trace 采集** | **分卡跑一次 r0 route**: CARLA(GPU_render) ∥ CoDriving 进程C(GPU_infer=4090), 记录进程C 真实逐次 forward τ → `rsu_latency_trace.csv`(call_idx, τ_ms, 排除前 20 warmup)。**真点云·真分卡·τ 随场景真变化**(否决固定 N 合成)。⚠️ R-1 实测出真实 τ 分布; CoDriving@4090 快, τ 多半 <50ms ⇒ Δ≈1(真实低延迟形态), 高延迟区间由常数注入 sweep(Orin 档)覆盖 | CARLA + 4090 分卡 | 🔶 待做 |
| **R-2 config 扩展** | agent config `simulation.latency_inject_ms`(标量) → 增 `simulation.latency_trace: <path>` 选项; 二者互斥 | — | 🔶 ~10 LOC |
| **R-3 trace 消费** | C-3 aligner 每次 RSU 推理从 trace 取下一个 τ 驱动 Δ(复用已有 Δ/ZOH/audit); trace 耗尽则循环或保持末值(标注) | — | 🔶 ~30 LOC |
| **R-4 验证** | 同 C-4: 跑一 route, 审计 log 显示 Δ **逐次变化**(非恒定)且 = ceil(trace_τ/50); zoh_age 随 τ 波动 | 分卡 | 🔶 待做 |
| 口径标注 | `latency_ms_source = trace_replay_codriving_4090`(区别于注入档 injected_from_E7 与实测 measured_in_carla) | — | — |

#### ★R-1~R-4 实测完成 (2026-06-09, team-lead 独立核验通过)
- **R-1 trace 采集** ✅: 分卡 r0(CARLA GPU3 ∥ CoDriving 推理 GPU4), 真点云逐次 CUDA-Event(forward+postproc, excl voxelize/IPC) → `results/rsu_latency_trace.csv`(153 条, 排除 20 warmup)。**τ 分布: min 47.1 / median 65.4 / p95 82.7 / max 122.3 ms**。**Δ=ceil(τ/50) 真三档逐次变化: {Δ1:5, Δ2:146, Δ3:2}**(含 post_process 让 τ>50ms, 落 50-150ms 滞后区, 非恒定=1)。team-lead 读 csv 重算逐位吻合。
- **R-2/R-3** ✅: config `latency_trace` 选项(与 `latency_inject_ms` 互斥) + `LatencyTraceReader` 逐次驱动 Δ; 单测 18/18; commit `9535874`/`88ac0b4`/`5a5959d`(sim-closedloop 分支)。
- **R-4 闭环验证** ✅: r0 分卡 trace 回放, audit log `delta` 列 **{1:5, 2:158, 3:2}** 复现 trace ceil(τ/50)(Δ2 +12 = 165 calls>153 trace 循环 1 次, 重用早期 Δ2 条目, 已 log `cycling`); `zoh_age_frames` **{0:2,1:2,2:156,3:5}** 随 τ 波动非恒定; `latency_ms_source` 全 `trace_replay_codriving_4090` 无污染; 初始 3 帧 ZOH(首个 RSU 到达前用 last_known)。team-lead 读 audit csv 独立核验吻合。
- **⇒ 用户要的 RSU 时延感知回灌达成**: 记录每次真实推理时延 → 逐次驱动后推帧数 Δ → 无新 RSU 时 ZOH 沿用上一帧, 全链真实逐次、非手设常数。**车端 τ_ego=0(本步未碰 skip_frames), L1 控制/ego 调度器留作第二步。**

**与常数注入档的关系**: 常数注入档(§4.6 sweep)仍保留作**可控对照**(扫不同时延量级找拐点); trace 回放是**真实逐次形态**(单一真实时延分布下看驾驶)。两者口径不同, 入库分列。

**第二步(暂不做, 记账)**: 车端 τ_ego≠0 + 时延驱动的 ego 推理触发(LatencyScheduler 取代 skip_frames=4), 详见 §3.5 L1 控制 + 归档 `latency_aware_simulation_design §3.1`。

---

## §4 需要哪些初始实验证明设计有效

> 实验阶梯: 链路通(§4.2) → 时延注入+帧对齐正确(§4.3) → 路侧真接入控制(§4.4) → 机制全验+场景敏感度(§4.5) → 信号 sweep(§4.6)。每级是下一级的前置闸门。

### 4.1 实验矩阵总览

| # | 实验 | 证明什么 | 状态 |
|---|------|---------|------|
| E-smoke | 冒烟 1 route | 闭环链路端到端打通 | ✅ §4.2 |
| E-C4 | d0/d108 双对照 | 时延注入生效 + 帧对齐正确 | ✅ §4.3 |
| E-P0 | norsu vs d0 逐帧 control diff | 路侧融合真接入 ego control(非架构 bug) | ✅ §4.4 |
| E-pilot | r0 八档 sweep | 机制全验 + 场景敏感度探测 | ✅(部分)§4.5 |
| E-probe | r146/r160 八档 | 难场景是否显现时延信号 | 🔶 待跑 §4.6 |
| E-full | 6route×8档全量 | 驾驶分 vs 时延曲线 + 拐点 | 🔶 条件触发 §4.6 |

### 4.2 [✅ 已验] 冒烟 — 闭环链路打通 (Sim-A)

route town05_short r0: **status=Completed, RouteCompletion=100%**, 闭环链路(CARLA→实时传感器→CoDriving 感知+规划→PID→driving score)端到端打通。DS=12.5(penalty 0.125 = 3 次同一行人簇碰撞 ×0.5³), duration_system 660s(含 PTX-JIT 暖机+单卡推理)。**仅证链路通, 不代表方法性能, 不入库不对照 CoDriving 基准。**

### 4.3 [✅ 已验] C-4 双对照 — 时延注入 + 帧对齐正确 (Sim-B)

- d0(Δ=0 实时 173 帧) vs d108(Δ=3 滞后+前 3 帧 ZOH): 审计 log 帧对齐正确, `ceil(108/50)=3` 折算对, **单测 13 项全过**。
- 验证了折算自洽: game-time 影响只由 Δ 决定(§3.3 不变量); 注入旋钮 = agent config `simulation.latency_inject_ms` 单字段。

### 4.4 [✅ 已验 — ★关键定论] P0 诊断: 路侧融合真接入 ego control

> pilot r0 八档 DS 全 100、norsu(关路侧)也 100, 一度怀疑"②路侧感知根本没接入控制(架构 bug)"。两路独立证据定论为 **GO(架构无 bug)**:

**证据一(supervisor 读逐帧 control 产物 `image/.../ego_vehicle_0/NNNN.json`)**: d0(有 RSU) vs norsu(无 RSU) 166 共同帧中 **157 帧不同**, max|Δsteer|=0.48, **max|Δthrottle|=0.75, max|Δbrake|=1.0**(决策级满量程, 非浮点抖动)。相同帧 = 连续前缀 {0..32}, **从帧 36(warmup 后 RSU 介入处)起发散** —— 一箭双雕: ① warmup 前逐位一致 = **确定性自检通过**(排除 run-to-run 随机); ② 发散起点 = **RSU 融合确实改 ego control**。

**证据二(sim-integrator 追代码链路)**: `rsu_data → C-3 aligner → extra_source['rsu_data'] → perception_dataloader → intermediate_fusion(record_len=ego+RSU) → planner → control`。d0 audit 确认 `used_rsu_frame_id` 每步更新(0,1,2,3,4,8,12…)、`is_zoh_held=False`; norsu 时 `rsu_data=[]`→`record_len=1`→fusion 分支彻底关闭(audit_frames=0, duration 219s 比有 RSU 档少 ~40%, 省掉 RSU 推理)。

**⇒ 定论**: "架构 bug"被高置信反驳, **路侧 BEV 真进 control 决策**。DS 全平真因 = **r0(70m 直道)场景太易**(有/无 RSU 两套不同 control 都能零碰满分)。**残余 caveat(诚实)**: 本 diff 只证"RSU 改 control", 未证"RSU 改善安全"(易 route 上两套都安全 = 正常); 难场景探针才检验 RSU 的不同 control 是否带来**可测 DS 提升**。

### 4.5 [✅ 部分] pilot r0 八档 — 机制全验 + 暴露 r0 太易

- **机制层 ✅ 全验**: 八档 delta_mean 精确 = ceil(L/50)(d0/50/108/150/219/300/500/norsu → Δ 0/1/3/3/5/6/10/n.a.), `zoh_held_frac` 随时延单调升。注入/帧对齐/ZOH 全对。
- **信号层 ❌ 命中证伪条件 1**: 八档 DS 全 = 100 / 碰撞全 0 / norsu 也 = 100。⇒ r0 对时延、甚至对"有无路侧"完全不敏感。结合 §4.4 定论 = **r0 场景太易, 非架构问题**。
- **单卡墙钟**: 单档 ~6.3min(dur_system ~335s + 启停); 全 8 档单 route ~50min。

### 4.6 [🔶 待跑] 信号探针 + 全量 sweep 设计

> **执行优先级(team-lead 研判, 防无脑铺量)**: 先探针 r146/r160 找信号 → 有信号才铺全量; 无信号则按证伪条件换难场景。

#### 时延档设计 (8 档, 加密找拐点; 全用 `latency_inject_ms` 注入)

| # | 档名 | inject_ms | Δ 帧 | 滞后 | 物理依据 |
|---|------|-----------|------|------|---------|
| 1 | 理想 | 0(不设) | 0 | 0 | 完美同步上界 |
| 2 | 50ms | 50 | 1 | 50ms | 1 帧最小可分辨档 |
| 3 | 最优链 | 108 | 3 | 150ms | E7 INT8 p75 RSU+V2X 合成(108ms) |
| 4 | 150ms | 150 | 3 | 150ms | 折算自洽校验点(与档 3 同 Δ, 验"档值≠Δ 时是否同结果") |
| 5 | baseline | 219 | 5 | 250ms | FP32 零优化 e2e 合成 |
| 6 | 300ms | 300 | 6 | 300ms | CoBEVFlow 测试中点 |
| 7 | 500ms | 500 | 10 | 500ms | CoBEVFlow 上界 = MAX 封顶 |
| 8 | 无路侧 | —(关 RSU) | n.a. | n.a. | ego-only 下界(disable_rsu, **非无限延迟**: 关 spawn_rsu+rsu_data=[], fusion 分支关闭 ≠ "有 RSU 但永远过时") |

- **★4090 注入档(数据中心级 RSU 对照, 用户追加)**: 4090 e2e ~27-35ms **< 50ms(一帧) ⇒ Δ 恒=1**(亚帧延迟, 帧对齐上等价档 2)。作"强力 RSU vs 边缘 RSU"对照锚定曲线低端。`latency_ms_source=measured_on_4090_*`。⚠️ 现有 35.31/26.70ms 是 OPV2V autocast 口径, 待 hw-optimizer 用 4090 TRT 引擎补 DAIR 同口径(pre-body+body+NMS); 无论口径 Δ=1 稳。

#### 6 route 集合(对抗性, junction 几何筛选, C-7 定稿)

| rid | 长(m) | njunc | 转向° | 触发密度 | 入选理由 |
|-----|------|-------|------|---------|---------|
| r0 | 70.7 | 1 | 0 | 132 | 冒烟已验, 确认对抗基线(实测触发行人横穿) |
| r146 | 94.6 | 3 | 88 | 372 | 全网最密路口+转弯进路口=遮挡最重 |
| r28 | 70.7 | 2 | 0 | 608 | 同长直道触发密度最高(北象限) |
| r160 | 221.8 | 3 | 0 | 768 | 最长+最高触发密度(西象限) |
| r141 | 123.0 | 2 | 90 | 612 | 90° 转弯逼近路口=视线遮挡 |
| r135 | 120.2 | 2 | 102 | 412 | 最急转向(盲区横穿风险最高) |

⚠️ junction 中心是 OpenDRIVE 几何均值的**启发式近似**(非 CARLA `is_junction` 真值), 足够做相对排序选差异化 route, "是否真跨路口"待首跑 audit 复核。

#### 墙钟 + 统计
- 首轮 6 route × 8 档 × **1 次**(确定性, 见下) = 48 次; 分卡+异步 ~8min/route; **~7.2h 单 GPU 对**; 2 对并行减半 ~3.6h。
- **确定性 → 不做同 route 重复**: 种子写死(CARLA/Traffic=2000), 同 (route,档) 重复 = 假 0 方差。统计量来源 = **6-route 间方差**(每档报均值±route 间 std)。**仅对 1 个组合重跑 1 次作确定性自检**(证伪闸门, 非统计重复); 若不一致回退每组 3 次。

### 4.7 [✅ 预先写死] 预期曲线与证伪条件 (★防确认偏误)

**预期形态**: 低端(0–108ms)平台(RSU 预警仍及时, DS≈平) → 中端(108–250ms)缓降→陡降(**拐点假设落此, 接近 200ms 阈值**) → 高端(300–500ms)触底趋近无路侧。无路侧 = ego-only 下界(若高时延档 DS<无路侧 = "陈旧 RSU 比没有更糟", 有价值的反直觉发现非 bug)。

**证伪条件(出现即承认假设/场景有问题)**:
1. **0ms 与 500ms DS 无显著差异**(< route 间 std) ⇒ route 集对时延不敏感。**应对: 换难 route 或调高 scenario_parameter 的 DynamicObjectCrossing proportion + pedestrian 60→120 + CRAZY_LEVEL**。← **pilot r0 已命中此条**。
2. **zoh_age 随档单调升但 DS 全平** ⇒ 陈旧度增大但不传导到驾驶(planner 对感知延迟不敏感, 或场景太易, 或 L0 控制不敏感)。**应对: 同 1 加难 + 考虑上 L1 控制(§3.5)**。
3. **DS 随档非单调** ⇒ 确定性破裂(4.6 自检该抓到)或单 route 噪声 ⇒ 回退多次均值。
4. **无路侧 DS ≥ 低时延档** ⇒ RSU 在本场景无净增益(与 CoDriving +62.49% 矛盾)⇒ 排查 RSU spawn/融合(audit `used_rsu_frame_id` 是否真更新)。← §4.4 已排除此架构隐患。

---

## §5 闭环指标入库 (Sim-D) — schema + GO 门

> 数据源 `results/closedloop_sweep_v1.csv`。**闭环指标与感知 AP/AMOTA 不同任务口径, 绝不混入同列/同曲线。**

- **隔离设计(DRAFT 已落 `data/schema_v2.md`)**: `cl_` 前缀性能列(`cl_driving_score`/`cl_collision_{ped,veh,layout}`/`cl_route_completion`/`cl_zoh_age_mean`/`cl_delta_frames`/`cl_zoh_held_frac`…) + 元数据列(`latency_inject_ms`/`latency_ms_source=injected_from_E7`/`isolation=single_card_shared`/`sim_route_set`/`n_repeat=1`/`rsu_enabled`)。schema 字段: `model_class=codriving_v2xverse`(严禁与 pyramid_fusion 混)/`regime=sim_closedloop`/`latency_kind=injected_sim_arm`/`lat_p50_ms=NaN`(注入值≠测量, 不填主 latency 轴)/`ap_valid=False`。
- **norsu 行**: `latency_inject_ms`+`cl_delta_frames` 均填 **NaN**(无路侧分支 ≠ 0ms 有路侧, 语义不同, 不可同一时延轴连续绘制)。
- **质检门槛**: DS∈[0,100] / 碰撞率非负 / route_completion∈[0,100] / status=Completed / delta=ceil(inject/50) / zoh_held_frac per-route 单调 / norsu 字段 NaN。
- **入库 GO 门(team-lead 把)**: 两硬条件 — ① P0 诊断确认路侧接入 control(✅ §4.4 已达) + ② 至少 1 route 出现 DS 随时延可分辨区分(🔶 待探针)。GO 前 pilot 8 行**不落主表**, 仅 schema 草案。

---

## 附录

### A 环境构建偏差表 (D-1~D-5, supervisor 核验)

| 编号 | 偏差 | 官方 | 实际 | 风险 |
|------|------|------|------|------|
| D-1 | torch 大包安装 | 裸 conda install | wget -c 续传到缓存后 conda install | 无(同一官方包) |
| D-2 | torchaudio | 离线 | 在线装补 ffmpeg 依赖 | 无 |
| D-3 | **spconv** | 1.2.1 源码 build | **2.3.6 wheel**(v2 兼容分支) | 低(voxel 已验; 若用稀疏 3D 卷积需复验) |
| D-4 | setuptools | 装完升级 | 留 41.2.0 | 无(删 distutils-precedence.pth 避告警) |
| D-5 | numpy/opencv | reqs 交叉 | numpy 1.18.3 + opencv 4.2.0.34 | 待观察(import 链未报错) |

### B 实测踩坑清单 (B-1~B-3)

| 编号 | 卡点 | 修复 |
|------|------|------|
| B-1 | spconv v1 API 硬 import | 移植 opencood spconv-2.x 兼容分支到 `pnp_infer*.py` |
| B-2 | scenario 配置名拼接错 | 第 6 参数传 `_1`(带下划线)不是 `1` |
| B-3 | **route 跑完 CARLA teardown 挂死** | 顶层 results.json 不 flush; **权威结果在 `ego_vehicle_0/results.json`**; 每 route 后 `kill -9 CarlaUE4-Linux-Shipping/CarlaUE4.sh` + `pkill -9 leaderboard_evaluator` + sleep 5 |

### C 复现命令 (冒烟, 实测通过版)

```bash
source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh && conda activate v2xverse
cd /home/jichengzhi/V2Xverse
# CARLA server (空闲卡; 启动前 nvidia-smi 确认 util 0%/mem≤50MiB)
CUDA_VISIBLE_DEVICES=<gpu> ./external_paths/carla_root/CarlaUE4.sh --world-port=40000 -prefer-nvidia -opengl &
# 1 route 闭环冒烟 (route_id=0, port=40000, tag=smoke, repeat=0, agent=codriving_5_10, scenario=_1)
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/eval_driving_e2e.sh 0 40000 smoke 0 codriving_5_10 _1
# 跑完手动 teardown (B-3)
kill -9 <CarlaUE4-pids>; pkill -9 -f leaderboard_evaluator_parameter
```

sweep 驱动: `scripts/closedloop_sweep_v1.sh <render_gpu> <infer_gpu> <carla_port>`(单卡模式 `SIM_SINGLE_CARD=1`; 每 route 一次 CARLA 启停避 B-3; 失败重试 1 次记 `sweep_failures.log`; 聚合 → `results/closedloop_sweep_v1.csv`)。

### D 实现计划 / commit 链 (C-1~C-9, V2Xverse sim-closedloop 分支)

C-1 推理服务进程(CUDA-Event 计时+zmq) · C-2 agent 侧 IPC client(异步提交不 join) · C-3 动态时延折算+ZOH+6 字段审计 · C-4 分卡冒烟双对照 · C-5 disable_rsu 无路侧开关 · C-6 sweep 驱动+聚合 · C-7 6-route 定稿 · C-8 单卡模式+isolation 标注 · C-9 并行安全多实例(per-PORT 隔离)。**理想愿景未实装项(future work)**: L1 轨迹跟踪控制器 · per-link 链路模型(τ_comm 抖动/丢包/传输量, ~120 LOC) · HIL Orin 在环(hil_client+server, ~250 LOC) · L2 看门狗。详见归档 `latency_aware_simulation_design_v1.md` §三/§五(~600 LOC 完整改造清单)。

### E 时延档真测来源

注入档全部来自团队 Orin E7 真测(`orin_pyramid_baseline_report_v1.md` §3/§4, E7/E8 ground truth): body TRT FP32 131.33 / FP16 47.99 / INT8 p75 20.02ms; e2e 混合链合成 FP32 219 / FP16 136 / **INT8 p75 108ms ✅<200ms**。200ms 预算可达性分解见 `edge_latency_budget_v1.md`(本目录)。

---

## §6 待决 / 后续 (用户桌上)

1. **P2 Orin 真隔离(HIL §3.6)**: CARLA 在 4090 / 推理在 Orin, 补现场实测真实边缘时延锚点 + 真物理隔离(标 `separate_gpu`/`measured_on_orin`)。用户"一张卡纯做推理"设想的真正落地形态。需 CoDriving 移植 Orin + 网络 IPC(工程量大, 未启)。
2. **L1 控制器升级**: 若探针仍现时延不敏感, L0→L1 轨迹跟踪是提升时延敏感度的关键(§3.5)。
3. **场景加难**: 证伪条件 1/2 命中则调 scenario_parameter(DynamicObjectCrossing proportion↑/pedestrian 60→120/CRAZY_LEVEL↑)或换 long route。
4. **更多 GPU**: 得 6 卡可 2 实例/route 并行翻倍。
