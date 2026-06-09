# 时延感知闭环仿真 — 设计方案 v1

> 编制: team-lead (Claude), 2026-06-06
> 承接: [v2xverse_framework_analysis_v1.md](./v2xverse_framework_analysis_v1.md) §五(a)(b) 迁移并大幅扩展
> 讨论结论: 统一的是**时间轴**而非单一延迟标量; 通信延迟作用于融合输入新鲜度, 计算延迟作用于控制生效时间
> 关联: edge_latency_budget_v1.md (Orin 实测延迟表是本方案 τ_compute 的数据来源)

---

## 〇、现状与问题（代码核实）

> 配图: [../figure/fig_latency_blindspot_v1.png](../figure/fig_latency_blindspot_v1.png)（生成脚本 `figure/make_latency_blindspot_v1.py`）—— 真实世界时间线 vs V2Xverse 现状的对比, 四个盲区①②③④见图注

V2Xverse 现有机制（均已核实到行号）:

| 机制 | 位置 | 问题 |
|------|------|------|
| `comm_latency`（帧）可选机制: 从 data bank 取 N 帧前的协作方数据 | `pnp_infer_action_e2e.py:494-516` | **默认未启用**: 全仓库无任何配置/脚本设置该字段 → 官方所有实验 = 0 通信延迟; 启用时也是单常数套给全部协作方（其他车+RSU 一视同仁, ego 保持最新 :510-513）, 无抖动/丢包、与传输量无关 |
| 协作方**原始传感器数据**直接拼进 ego 感知 batch, 融合网络（含 RSU 一路的编码器）在 ego 一次 forward 跑完 | `pnp_infer_action_e2e.py:518-537` | "协作方本地编码耗时 τ_collab_compute" 在仿真中**概念上不存在**——等于 RSU 免费瞬传原始点云、ego 免费替所有人计算 |
| `skip_frames: 4`, 每 4 帧推理一次 | `pnp_agent_e2e.py:448` | 隐含假设感知+规划**零耗时**（CARLA 同步模式挂起等推理） |
| 非推理帧 `return self.infer.prev_control` | `pnp_agent_e2e.py:448-450` | **控制指令零阶保持**: 油门/转向角原样冻结, 弯道上保持 200ms+ 固定转向会系统性偏离 |

→ 现状下"时延"只会让感知输入变旧，**车辆的反应速度、执行行为完全不受影响**，无法回答"<200ms 假设"这类问题。

---

## 一、统一时延模型: 一条时间轴, 两个注入点

### 1.1 为什么不能合成一个标量加到控制延迟上

通信延迟和计算延迟作用在管线**不同位置**:

```
RSU/协作车:  传感器 → 编码器(τ_collab_compute) → 发送 ──┐
                                                        │ τ_comm (传输)
自车:        传感器(t) ────────────────────────────────→ 融合+规划 (τ_ego_compute) → 控制生效(+τ_act)
                                  ▲
                          协作特征已是 t − τ_collab_compute − τ_comm 时刻的
```

- **τ_comm 的效果**: 融合时协作方特征与自车特征**时间错位** → 空间错位（双方都在动）、动态目标鬼影、远端目标漏检。它不推迟自车出控制指令。
- **τ_ego_compute 的效果**: 基于 t 时刻数据算出的计划, 要到 t + τ_ego_compute 才可用 → **控制反应滞后**。
- 若把 τ_comm 直接加进控制延迟: 把"感知质量下降"错记成"反应变慢", 双重失真。

### 1.2 正确的统一量: 逐路径信息年龄 (per-path information age)

每条信息在**控制生效瞬间**的年龄:

```
age_ego    = τ_ego_compute + τ_act
age_collab = τ_collab_compute + τ_comm + τ_wait + τ_ego_compute + τ_act
             (τ_wait: 到达后等待自车下一个推理周期的对齐时间, 0 ~ 推理周期)
```

- 协作路径确实是**链式相加**的（这一点上"统一延迟"的直觉是对的）;
- 但两条路径年龄不同, 必须分别注入: τ_comm 进 data bank（决定融合用哪帧协作数据）, τ_ego_compute 进调度器（决定计划何时可用）。
- CARLA 20Hz (50ms/帧) 下全部量化为帧数。

---

## 二、延迟期间车辆靠什么控制 — 三档控制保持策略

这是逼真度的核心。真实自动驾驶栈是**规划-控制分频**的: 规划器 ~10Hz 且有延迟, 底层控制器 50-100Hz 持续跟踪最近一份轨迹。据此设计三档（同时也是消融实验对象）:

### L0 — 指令零阶保持（现状, 作为 baseline 保留）
非推理帧冻结上一条 throttle/steer/brake。延迟越大失真越大: 固定转向角在弯道上 6 帧（300ms）就会显著偏离车道。**保留它只为了量化"仿真方法本身"带来的误差**。

### L1 — 轨迹跟踪保持（推荐默认, 对应真实系统）
1. 规划输出的 waypoints 锚定**世界坐标系 + 数据时间戳** `{waypoints_world[10], t_anchor, pose_anchor}`（现 planner 输出自车系, 需用 `pose_anchor` 转换一次）;
2. **PID 控制器每个 CARLA tick 都运行**, 跟踪"最近一份可用计划";
3. **按时间索引取参考点**: 当前时刻 t_now, 计划锚点 t_anchor, 控制器在 waypoint 序列上插值取 (t_now − t_anchor) 对应的目标点, 而不是永远跟踪 waypoint[0]。计划变旧时车辆沿旧轨迹继续推进——这正是真实世界延迟下的行为: 车不会失控, 但**对新出现的障碍物反应滞后了 age_ego 那么久**。

### L2 — 安全看门狗（可选叠加, 对应真实系统 fallback）
计划年龄超过阈值（如 500ms）→ 降速跟踪; 超过更大阈值（如 1s, 模拟通信/计算长时中断）→ 缓刹车保持车道。没有 L2, 大延迟扫描实验的尾部结果是"无意义的失控"而非"有意义的降级"。

**实验价值**: L0 vs L1 的 driving score 差即"仿真保真度修正量"; L1 vs L2 可研究 fallback 策略本身。

---

## 三、实现方案

### 3.1 总体架构: LatencyScheduler（异步作业模型叠加在 CARLA 同步 tick 上）

```
每个 CARLA tick (50ms):
  1. 传感器入 bank（自车 + 各协作方, 协作方按 per-link 延迟模型打时间戳）
  2. if 无在跑作业 and 到达推理触发帧:
        快照输入{自车数据(t), 各协作方数据(t − τ_link_i)}
        创建作业 job{submit=t, ready = t + ceil(τ_compute/50ms)}
  3. if 存在 job.ready ≤ t:  active_plan ← job 结果 {waypoints_world, t_anchor}
  4. 每 tick 运行 L1 控制器: 按 (t − t_anchor) 时间索引跟踪 active_plan
  5. L2 看门狗: 检查 plan age, 必要时降级
```

要点:
- **延迟同时影响年龄和频率**: 并发=1 时, 计划更新率 = 1/max(τ_compute, 触发间隔)。τ_compute=300ms 意味着不仅反应慢 300ms, 计划还只能 3.3Hz 更新。可选并发>1 模拟流水线化的多进程栈（感知规划重叠执行）。
- 替代现有 `skip_frames` 固定逻辑: 推理触发由"上一作业完成"驱动, 而非固定取模。

### 3.2 通信延迟: per-link 链路模型

在现有 data bank (`pnp_infer_action_e2e.py:494-516`) 上扩展, 改为 per-agent 索引:

```
τ_link_i = τ_base + bytes_i / BW + Jitter      Jitter ~ Gamma(k, θ)
P(drop)  = p_loss                              丢包 → 该协作方本帧不参与融合
```

- `bytes_i` 可直接计算: intermediate fusion 特征图 C×H×W×2 bytes (fp16); Where2comm 按通信掩码后的稀疏量计; late fusion 按检测框数 ×~32B。**由此不同方法的通信量差异自动转化为延迟差异**——Where2comm vs V2X-ViT 在同一带宽下产生不同 τ_comm。
- 带宽参考: C-V2X PC5 典型 10–27 Mbps; 参数全部进 yaml。
- 训练侧对齐: dataset 已有 `time_delay` 字段 (`v2xverse_basedataset.py:40-51`), 离线训练注入同分布延迟做 delay-robust 训练, 避免"训练零延迟、测试有延迟"的分布漂移。
- 随机性必须可复现: 链路 RNG 用独立 seed 进 config。

### 3.3 计算延迟来源: 三种模式（HIL 录制 → trace 回放 的生产关系）

| 模式 | 做法 | 适用 |
|------|------|------|
| **HIL Orin 在环（推荐, 最高保真）** | 每次推理触发, 仿真机把输入序列化发给 Orin, Orin 跑 TRT engine 并回传 {结果, τ_infer}; 调度器据实测 τ 决定生效帧 | 真实硬件、真实模型产物(TRT/INT8)、真实延迟分布; 同时解决"仿真与推理抢本机 GPU 算力"问题 |
| **trace 回放 / 延迟表注入** | 回放 HIL 录制的逐帧 τ 轨迹, 或查 {模型×精度×平台} 静态表（见 edge_latency_budget_v1.md）; 推理在本机跑 | 可严格复现、不依赖 Orin 在线; 是 HIL 的离线复现形态 |
| 在线计时 | `torch.cuda.Event` 测仿真机每帧耗时 | 仅快速摸底; 测的是仿真机且受 CARLA 渲染争抢污染, 结果不可迁移 |

**HIL 模式设计要点**:

```
仿真机(只跑 CARLA)                          Orin(只跑推理, nvpmodel+jetson_clocks 锁频)
帧 t: 序列化输入 ──上传──→ t_start=本地钟 → TRT 推理 → t_end=本地钟
  (同步模式, 仿真时间冻结, 阻塞等待)
  收到 {结果, τ_infer=t_end−t_start} → ready = t + ceil(τ_infer/50ms)
  帧 t..ready−1: L1 跟踪旧计划;  帧 ready: 新计划生效
```

1. **τ_compute 取 Orin 本地时间戳的纯推理段**, 不取网络往返——上传/回传是实验室局域网产物, 车上不存在; 真实链路延迟由 3.2 link_model 单独建模。阻塞等待不污染仿真时间(同步模式冻结), 只拖慢墙钟(一帧点云 1-2MB, 千兆网 ~20ms/次)。
2. **吞吐同时扣住**: Orin 推理期间为"忙", 下次触发不得早于 ready(并发=1), 避免出现"每帧都推理但每次都延迟"的物理上不可能时间线。
3. **每帧 τ 写入 trace 文件**: HIL 跑一次 → 录得真实延迟分布(含抖动) → 后续用 trace 回放模式离线复现, 不必每次挂 Orin。
4. **RSU 编码耗时**: v1 沿用整批推理(τ_infer=全管线); v2 把 RSU 编码器单独计时(或部署第二台设备), 使 age_collab 链中的 τ_collab_compute 首次可测(现有仿真中该项概念上不存在, 见 §〇)。
5. 通信实现: Orin 端起 ZeroMQ/gRPC 服务收 batch、回结果+时戳, ~200 LOC; 多 ego 时串行排队, 墙钟变慢但仿真时间正确。

### 3.4 帧率量化误差

CARLA 20Hz → 延迟分辨率 50ms。若需区分 80ms vs 120ms 这类差异, 选项: (i) 接受 ±25ms 量化误差并在结论中声明; (ii) 提高仿真频率到 40Hz（`fixed_delta_seconds=0.025`, 仿真耗时翻倍）。建议 v1 用 (i)。

### 3.5 延迟补偿方法（成为可评测对象, 非本方案必做项）

本框架建好后, 以下补偿方法可作为实验变量接入:
- **陈旧特征位姿补偿**: 协作特征按双方相对位姿变化做 BEV warp（HEAL/V2X-ViT 论文均有此鲁棒性实验）;
- **delay embedding**: V2X-ViT 自带 delay-aware positional encoding, 现框架下首次可在闭环中验证;
- **预测补偿**: planner 侧用占用图时序外推抵消 age_ego。

---

## 四、评测设计

### 4.1 扫描实验矩阵

| 轴 | 扫描点 | 固定项 |
|----|--------|--------|
| 通信延迟 | τ_comm ∈ {0, 100, 200, 300, 500} ms | τ_compute = Orin 实测值 |
| 计算延迟 | τ_compute ∈ {50(理想), Orin-INT8, Orin-FP16, 300} ms | τ_comm = 100ms |
| 联合 | (τ_comm, τ_compute) 网格 | — |
| 控制策略 | L0 / L1 / L1+L2 | 任一延迟点 |
| 协同方法 | codriving / v2xvit / where2comm / late / single | 同一延迟配置 |

### 4.2 指标
- 闭环: driving score / 碰撞率 / 路线完成率 vs 延迟曲线（核心产出: **"V2X 增益随延迟衰减曲线"**——V2X 相对 single 的增益在多大延迟下归零, 直接回答 "<200ms 假设"）;
- 离线对照: 同延迟分布下的 sAP（流式感知指标）, 验证离线-闭环相关性;
- 过程量: 每 route 记录 plan age 分布、丢包率、L2 触发次数。

---

## 五、文件改动清单

| 文件 | 改动 | 估计 |
|------|------|------|
| `team_code/latency/latency_scheduler.py` | **新建**: 作业队列、active_plan 管理、L2 看门狗 | ~200 LOC |
| `team_code/latency/link_model.py` | **新建**: per-link 延迟/抖动/丢包采样, 传输量计算 | ~120 LOC |
| `team_code/latency/compute_latency_table.yaml` | **新建**: {模型×精度×平台} → 延迟(ms), 来源标 Orin 实测 | 数据文件 |
| `team_code/latency/hil_client.py` + Orin 端 `hil_server.py` | **新建**: ZeroMQ/gRPC 收发 batch、Orin 本地时戳、τ trace 录制 | ~250 LOC |
| `pnp_agent_e2e.py` | run_step 改造: 每 tick 调控制器; 推理触发交给 scheduler（替代 :448-450 的 prev_control 保持） | ~60 LOC 改 |
| `pnp_infer_action_e2e.py` | 拆分"生成计划"与"生成控制"; data bank 改 per-agent 索引 (:494-516) | ~100 LOC 改 |
| `V2X_Controller`（control 模块） | 支持时间索引取参考点（waypoint 序列插值） | ~40 LOC 改 |
| planner 输出 | waypoints 转世界系并附 t_anchor | ~20 LOC 改 |
| `agent_config/pnp_config_*.yaml` | 新 schema: `latency: {comm: {...}, compute: {...}, control_hold: L1, watchdog: {...}, seed}` | 配置 |
| `statistics_manager.py` | 增加 plan-age / 丢包 / L2 触发统计列 | ~40 LOC |

合计 ~600 LOC, 其中 scheduler 与控制器改造是关键路径。建议实施顺序: ① per-link 通信模型（独立可测）→ ② L1 控制器分频改造（不带延迟先验证与现状等价）→ ③ scheduler 注入计算延迟 → ④ L2 看门狗 → ⑤ 扫描实验。

---

## 六、与能耗感知的关系

能耗记账（energy ledger）方案保留在 [v2xverse_framework_analysis_v1.md](./v2xverse_framework_analysis_v1.md) §五(c)。两者共享传输量计算（`link_model.py` 的 bytes_i 同时供延迟换算与通信能耗 `E = bytes × J/bit` 使用），实现时 link_model 应暴露统一的 per-frame 传输量接口。
