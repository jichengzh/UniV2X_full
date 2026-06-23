# V2Xverse 闭环仿真集成架构 (sim_closedloop_arch_v1)

> 编制: sim-integrator, 2026-06-07
> 任务: Task #1 [Sim-A] 环境构建与冒烟 (本阶段)
> 状态: §1 环境构建 = 已完成真实搭建 ✅ (torch/carla/spconv/checkpoint 全部 import 级验证通过); §2 冒烟 = 待 team-lead 授权后启动 CARLA server (GPU 实验, 宣告协议); §3+ B 阶段架构 = TODO
> 纪律: 本文只记录环境层真实搭建结果 + 偏差列表; 未真测的数字一律标"未测/估算"; CARLA 闭环指标在拿到真实 leaderboard json 前不入本文。

---

## §1 环境构建记录 (4090 host, 已真实搭建并逐步验证)

### 1.1 目标与授权

- **team-lead 授权原文 (逐字)**: "现在可以帮我重新构建一个agent 主要负责仿真测试环境得构建,主要基于/home/jichengzhi/V2Xverse 项目构建仿真项目" —— 覆盖范围: conda env 新建 + CARLA 0.9.10.1 下载安装 + 冒烟测试 (1 route)。latency sweep (Task #3) 不在本授权内。
- repo: `/home/jichengzhi/V2Xverse` (官方 clone, branch 改动落新分支 `sim-closedloop`, 不碰上游 main)。
- conda env: `v2xverse` (`/home/jichengzhi/miniconda3/envs/v2xverse`)。

### 1.2 环境构建步骤与结果 (全部真实执行)

| 步骤 | 官方要求 (README) | 实际执行 | 结果 |
|------|------------------|---------|------|
| env 建立 | python 3.7 + cmake 3.22.1 | conda create | Python 3.7.16 ✅ |
| torch 栈 | torch 1.10.1 + tv 0.11.2 + ta 0.10.1 + cudatoolkit 11.3 | conda install (大包 wget -c 续传后命中缓存) | torch 1.10.1 / cuda 11.3 / tv 0.11.2 ✅ |
| opencood deps | `pip install -r opencood/requirements.txt` | 同 | ✅ (open3d 0.17, matplotlib 3.5.3, h5py 等) |
| simulation deps | `pip install -r simulation/requirements.txt` | 同 (含 setuptools==41 用于 easy_install) | ✅ (numpy 降到 1.18.3, opencv 4.2.0.34, py-trees 0.8.3) |
| CARLA 0.9.10.1 | `setup_carla.sh` (wget tar) | 手动 `wget -c` 两个 tar (CDN 对大文件不稳, 用续传) + tar 解压 | CARLA 3.96GB + AdditionalMaps 1.82GB 解压完整 ✅ |
| carla egg | `easy_install carla-0.9.10-py3.7-...egg` (setuptools==41) | 同 | `import carla` OK ✅ |
| 随机种子固定 | DefaultGameUserSettings.ini 加 ServerRandomSeed Seed=1234 | 已追加 (README L80-83) | ✅ |
| spconv | 官方要求 1.2.1 (源码 cmake build) | **改用 spconv-cu113 2.3.6 wheel** (见偏差 D-3) | voxel gen 真验证 OK ✅ |
| opencood setup | `python setup.py develop` + bbx IOU cuda 编译 | 同 | OpenCOOD 0.1.0 develop ✅; box_overlaps.so 编译 OK ✅ |
| pypcd | clone klintan/pypcd + python-lzf + setup.py install | 同 | ✅ |
| efficientnet | `pip install efficientnet_pytorch==0.7.0` | 同 | ✅ |
| CoDriving checkpoint | HF `gjliu/v2xverse` → `checkpoints/codriving/{perception,planner}` | wget -c HF resolve URL | perception 32MB (net_epoch_bestval_at16.pth) + planner 20MB (codriving_planner.ckpt) + config.yaml, 全部 zip 校验通过 ✅ |

### 1.3 sm89 风险验证 (关键 — 已实证, 官方栈可用)

任务预警: 4090 是 sm89, torch 1.10+cu113 不含 sm89 cubin, 可能 "no kernel image"。

**实测结果 (`/tmp/sm89_check.py`, CUDA_VISIBLE_DEVICES=5)**:
- `torch.cuda.is_available()` = **True**; device = "NVIDIA GeForce RTX 4090" / capability **sm_89**。
- torch build arch_list = `['sm_37','sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','compute_37']` —— **无 sm_89 cubin, 但含 `compute_37` PTX**, 走 PTX JIT 前向兼容路径。
- **conv forward 实测通过**: 首次 conv (含一次性 PTX JIT 编译) = **1.58s**; 编译后稳态 = **0.21 ms/iter**; GPU vs CPU 数值 maxdiff 9.2e-4 (float32 容差内) = **PASS**。
- **结论**: 官方 torch 1.10.1+cu113 在 4090 上**功能正常**, 无 "no kernel image"。代价 = 每进程首次 GPU 算子 ~1.6s 一次性 JIT 暖机 (可接受)。**因此不升级 torch, 保持官方栈 0 偏差** (升级方案未启用)。

### 1.4 codriving 感知模型加载验证 (集成层真测)

`opencood.tools.train_utils.create_model(cfg)` + `load_state_dict(ckpt, strict=False)`:
- model core_method = `center_point_codriving`, 类 = `centerpointcodriving` 成功实例化。
- ckpt 加载: **missing 2 / unexpected 0** (0 unexpected = ckpt 架构匹配; 2 missing 为 buffer/aux 键, 非权重缺失)。⇒ 官方 checkpoint 与代码链路对齐 ✅。
- 感知预处理 = `SpVoxelPreprocessor` (走 spconv 2.x `Point2VoxelCPU3d` 兼容分支, 已验证 voxel 生成)。

### 1.5 与官方栈的偏差列表 (★诚实记录, supervisor 核验用)

| 编号 | 偏差点 | 官方 | 实际 | 原因 | 风险评估 |
|------|--------|------|------|------|---------|
| **D-1** | torch 大包安装方式 | 裸 `conda install` | `wget -c` 续传拉 conda 包到 pkgs 缓存后 `conda install` 命中缓存 | CDN 对 1.2GB 包反复 IncompleteRead 断连 | 无 — 最终装的是同一官方包 (torch 1.10.1, cuda 11.3), 仅下载手段不同 |
| **D-2** | torchaudio 离线安装失败 | — | 放弃 `--offline`, 改在线装 (复用缓存的 torch 大包, 仅补下 harfbuzz/libass/libglib 小依赖) | `--offline` 拒绝拉未缓存的 torchaudio ffmpeg 依赖 | 无 — torch/torchvision 是同一官方版本; torchaudio 仅其 ffmpeg 链补全 |
| **D-3** | **spconv 版本** | **1.2.1 (源码 build)** | **spconv-cu113 2.3.6 (prebuilt wheel)** | 1.2.1 在现代 CUDA/sm89 上源码编译成本高且易失败; opencood 的 `SpVoxelPreprocessor` 自带 v2.x 兼容分支 (`Point2VoxelCPU3d`) | **低** — 仅影响 voxel 生成 API; 已真测 voxel 输出形状正确。⚠️ 若后续用到 `sparse_backbone_3d` (3D 稀疏卷积) 需复验 v2 API 兼容; codriving 用 PointPillar 密集 BEV backbone, 当前路径不依赖稀疏 3D 卷积 |
| **D-4** | setuptools 终态 | 41.2.0 (README 建议装完 egg 后升级) | 41.2.0 (egg 已装完, 未再升级) | easy_install 需 41; 升级留待后续如遇 distutils_hack 报错再处理 | 无 — 已删除 setuptools 65 残留的 `distutils-precedence.pth` 避免启动告警 |
| **D-5** | numpy/opencv 版本拉扯 | opencood reqs 与 simulation reqs 有版本交叉 | 终态: numpy 1.18.3 + opencv-python 4.2.0.34 (sim reqs 覆盖) + opencv-python-headless 4.13 (opencood) | 两份 requirements 顺序安装, 后者覆盖前者 | **待观察** — numpy 1.18.3 较旧, 若冒烟时遇 numpy API 报错需调和; 当前 import 链未报错 |

### 1.6 设备与磁盘快照 (构建期间)

- GPU (nvidia-smi): 构建期间 GPU 0 被他人占用 ~5% util / 524MiB; **GPU 1-7 空闲** (mem ≤5MiB, util 0%)。sm89 验证用 GPU 5 (空闲)。
- 磁盘 `/home`: 构建前 242G → CARLA 下载+解压后 **206G 可用** (94% 已用)。CARLA 占用 ~12GB (tar 5.78GB + 解压 ~6GB; tar 可后续删以回收)。

---

## §2 冒烟测试 (1 route) — 已完成 ✅ [GPU 实验, team-lead 已授权并确认送达]

> 授权: team-lead 回执 (2026-06-07) 逐字引用用户授权链 "现在可以帮我重新构建一个agent 主要负责仿真测试环境得构建" + Sim-A 任务定义内含冒烟 = 对冒烟启动的显式授权与宣告确认送达。
> 链路验证: route town05_short_r0 repeat0 **status=Completed, RouteCompletion=100%** —— 闭环链路 (CARLA → 实时传感器 → CoDriving 感知+规划 → PID 控制 → driving score) 端到端打通。

### 2.1 冒烟链路 (自包含, 已验证)

closed-loop 驾驶 agent (`pnp_agent_e2e.py`) 用**实时 CARLA 传感器** (RGB/LiDAR 在仿真里 spawn), **不依赖离线 45GB 数据集** (`DATA_ROOT` 仅作结果输出回退路径)。冒烟自包含: 启 CARLA server → leaderboard 跑 route r0 → agent 用实时传感器 + 已下载 checkpoint 驱动。

### 2.2 复现命令 (从 conda activate 到 leaderboard, 完整可粘贴 — 实测通过版)

```bash
# 1. 激活环境
source /home/jichengzhi/miniconda3/etc/profile.d/conda.sh
conda activate v2xverse
cd /home/jichengzhi/V2Xverse

# 2. 启 CARLA server (渲染绑定一张空闲卡; 启动前 nvidia-smi 确认目标卡 util 0%/mem≤50MiB)
#    ★GPU 选择按当时空闲情况; 本次实测用 GPU 2 (GPU 0-3 空闲, 4-7 被 wuyuegao 占)
CUDA_VISIBLE_DEVICES=2 ./external_paths/carla_root/CarlaUE4.sh \
    --world-port=40000 -prefer-nvidia -opengl &
#    等 ~30-60s server 起 (首次加载 Town05 较慢; GPU mem 升到 ~1GB 即就绪)

# 3. 跑 1 条最短 route (r0) 闭环驾驶冒烟
#    参数: route_id=0, carla_port=40000, method_tag=smoke, repeat=0,
#          agent_config=codriving_5_10, scenario_config=_1 (★必须带下划线前缀)
CUDA_VISIBLE_DEVICES=2 bash scripts/eval_driving_e2e.sh 0 40000 smoke 0 codriving_5_10 _1

# 4. 结果 json 落点 (★权威 = ego_vehicle_0 子文件; 顶层 results.json 因 teardown 挂死未 flush)
#    results/results_driving_smoke/v2x_final/town05_short_collab/r0_repeat0/ego_vehicle_0/results.json

# 5. 跑完 CARLA teardown 会挂死 (见 §2.5 B-3), 手动清理:
#    kill -9 <CarlaUE4-Linux-Shipping pid> <CarlaUE4.sh pid>; pkill -9 -f leaderboard_evaluator_parameter
```

### 2.3 冒烟结果 (★只证链路通, 数字不入 dataset_v2)

| 指标 | 值 |
|------|-----|
| status | **Completed** |
| Driving Score (composed) | **12.5** |
| Route Completion | **100.0 %** |
| Infraction Penalty | 0.125 |
| Collisions (pedestrian) | **3 次** (t=33.9/35.6/36.9s, 同一行人簇) |
| Collisions (vehicle/layout) | 0 / 0 |
| Red light / Stop / Off-road / Route-dev | 全 0 |
| route_length | 70.71 m |
| duration_game | 56.7 s |
| duration_system (wallclock) | 660.4 s (含 PTX-JIT 暖机 + 单卡推理) |

> ⚠️ 数字解读: DS=12.5 偏低源于 penalty 0.125 (3 次行人碰撞, 每次 ×0.5 ⇒ 0.5³=0.125)。这是**原生 CoDriving 未改造 + 单卡 GPU (CARLA 渲染与推理同卡致推理慢) + 只跑 1 route 不调参**的冒烟结果, **仅证闭环链路打通, 不代表方法性能, 不入库不对照 CoDriving 论文基准**。

### 2.4 GPU 落位证据 (算力隔离纪律)

- **CARLA server (pid 1404597, CarlaUE4-Linux-Shipping)** 落 **GPU 2** (UUID a48f...c48, 1329MiB) — nvidia-smi `--query-compute-apps` 实证。
- **agent 推理**: 本次单卡冒烟与 CARLA 同在 GPU 2 (任务允许 "本任务内 CARLA 单卡即可")。⚠️ **B/C 阶段必须分卡** (CARLA 渲染负载会污染推理时延测量)。
- GPU 4-7 = wuyuegao 训练占用 (util 82-96%), 全程未碰; GPU 0/1/3 保持空闲 (留作后续推理隔离)。
- 启动时刻快照 (2026-06-07 15:45:10): GPU 0-3 = 5MiB/0%, GPU 4-7 = 2-2.8GB/90-96%。

### 2.5 阻塞项 / 实测踩坑清单 (本次冒烟修复记录)

| 编号 | 卡点 | 报错原文 | 修复 |
|------|------|---------|------|
| **B-1** | spconv v1 API 硬 import (agent 推理路径 2 文件) | `cannot import name 'VoxelGenerator' from 'spconv.utils'` | 移植 opencood `sp_voxel_preprocessor.py` 的 spconv-2.x 兼容分支 (`Point2VoxelCPU3d`+`point_to_voxel`+`.numpy()`) 到 `pnp_infer.py` 与 `pnp_infer_action_e2e.py` (落 sim-closedloop 分支)。spconv 偏差 D-3 的连带修补 |
| **B-2** | scenario 配置文件名拼接错 | `No such file: scenario_parameter1.yaml` | 实际文件名带下划线 `scenario_parameter_1.yaml`; `eval_driving_e2e.sh` 用 `scenario_parameter$6.yaml` 拼接, 故第 6 参数须传 `_1` 不是 `1` |
| **B-3** | route 跑完 CARLA teardown 挂死 (非致命) | `ERROR: failed to destroy actor NNN: not found` 刷屏后进程 do_wait 卡死, 顶层 results.json `progress=[0,1]` 不 flush | 已知 V2Xverse/leaderboard teardown 缺陷; **权威结果在 `ego_vehicle_0/results.json` (status=Completed, 已正常写盘)**; 顶层文件忽略。跑完手动 kill CARLA+evaluator (README 亦要求及时 kill carla) |

### 2.6 冒烟产物路径

- 权威结果 (archived): `/home/jichengzhi/V2X/results/closedloop_smoke_r0_repeat0_ego.json`
- 过滤后 leaderboard log (去 ALSA/destroy 噪声, 101 行): `/home/jichengzhi/V2X/results/closedloop_smoke_r0_leaderboard.log`
- 原始 (V2Xverse 内, gitignored): `results/results_driving_smoke/v2x_final/town05_short_collab/r0_repeat0/ego_vehicle_0/results.json`

---

## §3 B 阶段架构设计 (★Sim-B 设计稿, 待 team-lead 过目后再写代码)

> 授权: team-lead Sim-B 授权 (2026-06-07) 逐字引用用户原话 "他的主要职责是将我们的加速模型集成到仿真框架中, 第二点是优化现有仿真流程实现两点要求: 1.时延感知... 2.算力隔离..."。
> 本章是 **设计稿**, 不含已写代码。三项决策 (IPC / 时延→帧折算 / 零阶保持位置) 在 §3.3 逐条论证。**本章经 team-lead 过目确认后才进入实现 (§3.6 实现计划)**。

### 3.0 关键既有事实 (本章设计的硬约束, 实查 V2Xverse 代码确认)

| 事实 | 值 | 来源 (实查) | 对设计的意义 |
|------|----|------------|------------|
| **CARLA 仿真步长** | sync 模式 @ **20Hz → fixed_delta = 50ms/帧** | `leaderboard_evaluator_parameter.py` L128 `frame_rate=20.0`, L273-274 | 时延→帧折算的基本量子 = 50ms; 这是"后推帧数"的换算分母 |
| **感知推理节拍** | 每 `skip_frames=4` 帧跑一次 = **每 200ms** | `pnp_config_codriving_5_10.yaml` `skip_frames:4`; agent L448 跳帧逻辑 | 推理本身 200ms 跑一次; 时延注入作用在"推理结果何时可用"而非每帧 |
| **★原生已有时延机制 `comm_latency`** | 静态帧数延迟, 从 `pre_raw_data_bank[step - latency_step]` 取过去帧数据 | `pnp_infer_action_e2e.py` L522-543 | **不需要从零造轮子**; 本设计 = 把这个静态 frame-count 升级为 **动态 per-frame 实测时延驱动 + 零阶保持** |
| **时延注入的语义范围** | 原生 `comm_latency` 延迟 **RSU + 其他 CAV (`i>0`) 数据, ego 自身 (`i==0`) 保持当前帧** | L538-541 `if i>0: car_data_raw[i]=raw_data_used[...]; rsu_data_raw=raw_data_used['rsu_data']` | **完全契合用户要求**: "无路测推理信息时车辆用上一帧感知" = 延迟 RSU/协同数据, ego 本车感知不延迟 |
| **感知→规划数据流** | `run_step` → `infer.get_action_from_list_inter(car_data, rsu_data, step, ts)` 内做 感知(perception_model) + 融合 + 规划(planning MotionNet) → control | agent L472-476; infer L505-565 | 单进程内串行; 时延注入点在融合**前** (delay raw RSU data 进融合) |
| **感知缓冲基础设施** | `perception_memory_bank` / `pre_raw_data_bank` / `prev_control` 已存在 | infer L466/480/477 | 零阶保持可复用 `prev_*` 模式 |

### 3.1 进程拓扑 (算力隔离 — 本机分卡方案, 首选)

```
┌─────────────────────────────────────┐         ┌──────────────────────────────────────┐
│  进程 A: CARLA server  (GPU 2)        │         │  进程 B: leaderboard+agent (主进程)     │
│  - 仿真世界/渲染/物理 @ 20Hz sync      │◄──RPC──►│  - run_step 每帧调 (CARLA python API)   │
│  - 唯一占 GPU 2 (渲染)                 │ port    │  - ego 本车感知/融合/规划/控制           │
└─────────────────────────────────────┘ 40000   │  - ★感知推理 NOT 在本进程 GPU 跑          │
                                                 └────────────┬─────────────────────────┘
                                                              │ IPC (§3.3-①)
                                                              │ 送: RSU/CAV 原始点云+pose+frame_id
                                                              │ 收: (感知结果, per-frame 真实时延 ms)
                                                 ┌────────────▼─────────────────────────┐
                                                 │  进程 C: 推理服务 (GPU 0/1/3 之一)        │
                                                 │  - 加载感知模型 (CoDriving 原生 / 团队档)  │
                                                 │  - 每请求: CUDA-Event 计时 forward       │
                                                 │  - 返回 (det结果, 真实 latency_ms)        │
                                                 └──────────────────────────────────────┘
```

- **算力隔离落实**: 进程 A `CUDA_VISIBLE_DEVICES=2` (仅渲染); 进程 C `CUDA_VISIBLE_DEVICES=0` (仅推理); 进程 B 主控不占 GPU (或仅轻量后处理)。**CARLA 与推理物理分卡 ⇒ 推理时延测量不被渲染负载污染** (这是用户算力隔离要求的本质, 也是 Sim-A §2.4 标注的 B 阶段必改项)。
- **Orin 版 (延伸目标, 不阻塞主线)**: 进程 C 换成 Orin AGX (172.16.62.222) 上的推理服务, IPC 走网络 socket; 返回 Orin 真实端侧时延。本机分卡先打通, Orin 版作为"边缘设备"形态的扩展验证。

### 3.2 时延注入点与帧对齐时序 (★生产者视角, 非阻塞提交)

> ★[team-lead 评审修正①] 改写为生产者视角, 消除消费者视角的循环感 (旧图在 k-Δ "发起"而 Δ 又依赖事后才知的 L)。**核心: B 非阻塞提交 — game-time 延迟唯一来源 = Δ。**

```
帧 j (生产者侧, B 提交点):
  B 把帧 j 的 RSU 点云 + frame_id=j 提交进程 C, ★立即返回继续 tick (不阻塞等 C)
        │
        ▼ (异步, B 已在跑帧 j+1, j+2, ...)
进程 C: 对帧 j 的 RSU 点云推理, CUDA-Event 实测 L_ms
        ⇒ Δ = ceil(L_ms / 50ms)   [见 §3.3-②]
        结果挂到 frame-id buffer, 标记 "自帧 (j+Δ) 起可用"
        │
        ▼
帧 k (消费者侧, k ≥ j+Δ, ego 融合点):
  ego 融合取 buffer 中"已可用且最新"的 RSU 结果:
    = 满足 (源帧 id + 其 Δ) ≤ k 的结果里源帧 id 最大者
  若该结果距上次更新已过数帧 (k 与其可用帧之间无新结果) ⇒ 零阶保持 (沿用), 记 zoh_age
  ego 本车感知始终用当前帧 k (不延迟, 不经 ZOH)
```

- **非阻塞提交 (关键不变量)**: CARLA sync 模式下, **B 的 wallclock 阻塞不进 game time** —— 即便 B 同步等 C 算完, 仿真世界时钟也不前进, 故 game-time 延迟**唯一来源 = Δ** (帧数 × 50ms), 与 B 是否阻塞无关。但**同步阻塞会白白拖慢实验墙钟** (C 算 219ms × 每条 route 上千帧 = 墙钟爆炸); **异步提交让 C 的推理与 B 的 tick 重叠, sweep 阶段省大量墙钟**。⇒ B 提交后必须立即返回, 绝不同步 join 进程 C。
- 与原生 `comm_latency` 的差别: 原生 `latency_step` 是**配置写死的常数 + 进程内同步直调**; 本设计 `Δ = ceil(L/50)` 由**进程 C 实测/注入档动态算出 + 跨进程异步**。语义从"通信延迟"换成"**路侧推理计算延迟**" — 正是用户要的"记录每次推理计算时延, 后推对应帧数"。

### 3.3 三项重点设计决策 (★team-lead 要求逐条论证)

#### ① IPC 选型 → **决策: 本地 shared-memory + 轻量 socket 信令 (顺 V2Xverse 单进程结构最小改)**

- **现状**: 冒烟里 V2Xverse 感知是**进程内直调** (`get_action_from_list_inter` 内 `self.perception_model(...)`), 无任何跨进程结构。算力隔离要求把推理拆到另一进程/GPU, 必须引入 IPC。
- **候选对比**:
  | 方案 | 延迟开销 | 实现成本 | 与原结构契合 | 时延测量纯净度 |
  |------|---------|---------|------------|--------------|
  | **shm (点云 numpy) + socket 信令** | 最低 (零拷贝大 tensor) | 中 | 高 (Python multiprocessing.shared_memory) | ✅ IPC 开销与推理时延可分离计 |
  | zmq REQ/REP | 中 (序列化点云 ~1-2MB/帧) | 低 | 中 | ✅ 但序列化耗时混入 |
  | socket pickle | 高 (大 tensor pickle 慢) | 低 | 低 | ❌ |
- **决策依据**: 路侧点云每帧 ~1-2MB, shm 零拷贝避免序列化; socket 只传 (frame_id, shm 句柄, 控制信令) 小消息。**关键**: 进程 C 用 CUDA-Event 只计 **forward 本身** (不含 IPC 往返), 返回的 `L_ms` 是纯推理时延; IPC 往返另计 (供 §5 口径标注 "含/不含 IPC")。⇒ 满足纪律红线"每个数字标口径"。
- **★[实现期确认 — 采用回退方案] py3.7 无 `multiprocessing.shared_memory` (3.8+ 才有, 实测 `ModuleNotFoundError`)**, 故 shm 主方案不可用。**实际采用设计预案的回退: zmq REQ/REP + `msgpack_numpy`** (pyzmq 26.2.1 + msgpack_numpy 0.4.8, py3.7 实测可用)。序列化开销 (~1-2MB 点云) 计入 IPC 往返、与纯 forward 时延分开标注。这是与原设计的偏差但在 §3.3-① 预案内, 非临时起意。CUDA-Event 仍只计 forward, 口径纯净不变。

#### ② 时延→帧数折算规则 → **决策: Δ = ceil(L_ms / 50ms); 多帧延迟用 frame-id keyed buffer (排队不丢帧); 上限封顶**

- **步长**: 50ms/帧 (§3.0)。**取整: ceil (向上取整)** — 推理没算完的那一帧绝不能提前可用, 宁可多等一帧 (安全侧)。例: L=20ms → Δ=1 帧 (50ms 后可用); L=108ms → Δ=3 帧 (150ms); L=219ms → Δ=5 帧 (250ms)。
- **L > 一帧间隔 (多帧延迟) 的语义**:
  - **排队不丢帧 (首选)**: 沿用原生 `pre_raw_data_bank` 的 frame-id keyed 设计, 每个推理结果带其**发起帧 id**; ego 融合时取 `step - Δ` 对应 id 的结果。若该 id 不存在 (推理还没回来) → 零阶保持上一可用结果 (§3.3-③)。
  - **不做时间维丢帧**: 因为推理 200ms 一次 (skip_frames=4) 而 Δ 通常 1-5 帧, 推理节拍 ≥ 延迟跨度时不会堆积; 若未来 L 极大致堆积, 加 buffer 上限 `MAX_DELAY_FRAMES` (默认封顶 = 推理周期对应帧数), 超限丢最旧 (与原生 L534 `pop(sorted_keys[0])` 一致)。
- **封顶**: `Δ = min(ceil(L/50), MAX_DELAY_FRAMES)`, 防病态时延把 RSU 数据推到远古帧。MAX 默认 = 10 帧 (500ms, 对齐 CoBEVFlow 测试上界, 见 closedloop survey §3.4)。

#### ③ 零阶保持 (ZOH) 实现位置 → **决策: 融合前, 在 RSU/CAV raw-data 注入层做 (ego 本车数据不经 ZOH)**

- **位置论证**: 用户要求"无路测推理信息时车辆继续使用上一帧的感知信息"。感知融合 = ego 本车感知 ⊕ RSU/CAV 感知。ZOH 只该作用在**延迟的那一路 (RSU/CAV)**, ego 本车感知每帧实时。
- **⇒ ZOH 放在融合前的 raw-data 选择层** (即原生 L538-541 的位置): 当 `step-Δ` 的 RSU 结果不可用时, 用 `last_known_rsu_data` 顶替 (而非原生的 `print('latency data not found!')` 跳过)。这样融合算子拿到的永远是"当前 ego + 最近可用 RSU", 语义干净且可审计。
- **不放在融合后**: 若在融合后做 ZOH (保持整张融合 BEV), 会把 ego 本车的当前感知也冻住 → 违反"ego 本车实时"。**故必须融合前、且只对 RSU 路。**
- **可审计**: 每帧记录 per-frame log, 字段 (★[team-lead 评审修正②] 6 字段):
  | 字段 | 含义 |
  |------|------|
  | `step` | 当前融合帧号 (game-time 帧) |
  | `Δ` | 该帧 RSU 结果的延迟帧数 = ceil(L/50) |
  | `used_rsu_frame_id` | 实际用的 RSU 结果来自哪个源帧 (后推对齐核验) |
  | `is_zoh_held` | 本帧 RSU 是否走零阶保持 (无新结果沿用旧的) |
  | `latency_ms_source` | `measured_in_carla` (进程C实测) / `injected_from_E7` (时延档注入) — 口径标注, 纪律红线 |
  | `zoh_age_frames` | 当前用的 RSU 结果距今几帧 (= step − used_rsu_frame_id − Δ, ≥0) — **Sim-C 因果链中间变量: 时延↑→感知陈旧度↑→驾驶分↓; 不记 sweep 时就丢** |
  供 supervisor 核验帧对齐逻辑 + Sim-C 因果分析。

### 3.4 ★加速模型接入: 资产盘点 + 架构不匹配的诚实报告

> 用户原话要求 #3 (team-lead 转述): "若 ckpt 架构不匹配...如实报告不匹配, 集成方案改为'时延档注入'...别硬塞跨数据集模型冒充集成"。

**盘点结论 (实查 `models/` + `results/E7` + 两边 config)**:

| 维度 | 团队加速 Pyramid 资产 | V2Xverse CoDriving 感知 | 是否兼容 |
|------|---------------------|------------------------|---------|
| 模型类 | `HeterPyramidCollab` (HEAL m1) | `center_point_codriving` | ❌ 不同架构 |
| 训练数据 | **DAIR-V2X (真实路测)** | **CARLA 仿真数据** | ❌ 跨数据集 |
| 检测头 | Pyramid 多尺度加权融合 | center-point multiclass (anchor=3) | ❌ 不同 |
| voxel/range | DAIR m1 range | voxel_size 0.125, range [-12,-36,-22,12,12,14] | ❌ 不同 |
| 真测资产 | E7: body FP32 131ms / FP16 48ms / p75+INT8 20ms (Orin) | — | latency 可用 |

- **结论 (诚实)**: 团队 Pyramid ckpt 与 V2Xverse CoDriving 感知**架构 + 数据集双重不匹配**, **不能直接 load 权重替换**。硬塞 = 跨数据集脏推理, 已被项目纪律明令禁止 (CLAUDE.md ISS-012 同类教训)。
- **⇒ 集成方案 = 时延档注入 (latency-injection form)**: **模型仍用 V2Xverse 原生 CoDriving 感知** (保证 CARLA 数据上 AP 有效); **用团队 E7 真测时延数字驱动 §3.2 的延迟帧数 Δ**。即: 进程 C 跑原生 CoDriving 感知拿驾驶有效性, 但把"路侧推理时延"按团队真测的优化档 (FP32 131ms / FP16 48ms / p75+INT8 20ms / e2e 合成 219ms→108ms) 注入。
- **这正是 latency sweep (Sim-C) 需要的最小充分形态**: 回答用户核心假设"路侧推理 <200ms 是否提升驾驶安全" —— 不需要团队模型真在 CARLA 里跑, 只需要团队**真测的时延档**驱动延迟, 看驾驶分/碰撞率随时延档的变化。
- **可选增强 (非必须, 不阻塞)**: 若日后要"团队模型真在 CARLA 跑", 须在 CARLA 数据上重训 Pyramid (大工程, 超 Sim-B scope), 列为 future work。

### 3.5 算力隔离 — 推理进程返回二元组契约

进程 C 每次推理返回 `(perception_result, latency_ms)`:
- `perception_result`: 检测框/BEV 特征 (供 ego 融合)。
- `latency_ms`: 进程 C 内 **CUDA-Event 实测** forward 时延 (口径 = 推理进程内 GPU compute, 不含 IPC)。**这是注入 §3.2 折算 Δ 的真实时延源**。
- 时延档模式 (Sim-C 用): 进程 C 可被配置为"用注入档时延" (team E7 数字) 替代实测时延, 实现 latency sweep 的可控档位; 此时 `latency_ms` = 注入档值, 明确标注口径 "injected_from_E7" 而非 "measured_in_carla"。

### 3.6 实现计划 (★team-lead 过目本设计后才动手, 小步 commit)

1. **C-1** 推理服务进程骨架 (进程 C): 加载 CoDriving 感知 + CUDA-Event 计时 + IPC server (shm/zmq)。GPU 0。**[GPU 实验, 跑前宣告]**
2. **C-2** agent 侧 IPC client: 把 `get_action_from_list_inter` 内 `self.perception_model(...)` 直调改为 IPC 请求进程 C。**验收标准 (★评审修正①): 提交后立即返回, 不同步 join 进程 C** (异步提交, B 的 tick 不被推理阻塞; 用 future/句柄轮询取结果)。
3. **C-3** 动态时延→帧折算 + ZOH: 扩展原生 `comm_latency` 逻辑 (L522-543) 为 §3.2/§3.3 的动态版; 加 per-frame 审计 log (6 字段, 含 `latency_ms_source` + `zoh_age_frames`)。
4. **C-4** 分卡冒烟 (1 route): CARLA GPU 2 + 推理 GPU 0, 验证链路通 + 时延注入生效 + 帧对齐 log 正确。**验收交付: per-frame 审计 log 前 20 行 + 同一 route Δ=0 vs Δ=注入档 的 used_rsu_frame_id 差异证据。[GPU 实验, 跑前宣告]**
5. 全程落 `sim-closedloop` 分支, 每步独立 commit。

---

## §4 latency sweep 闭环实验设计 (Sim-C, Task #3) — ★设计稿 (不跑 GPU, 待 team-lead 过目)

> 授权状态: Sim-B 已 team-lead 验收 PASS (C-1~C-4 全 commit, C-4 双对照 d0/d108 审计 log 正确)。**本 §4 = 实验设计稿, 不动 GPU** —— team-lead 尚未授权运行 Task #3, 且当前 8 卡被他人训练占满。本章交 team-lead 过目 + 排队后才执行。
> 核心假设 (待验证, 来自闭环 survey §一): **路侧 e2e 推理 < 200ms 时, V2X 对驾驶安全 (碰撞率↓/驾驶分↑) 有可证明提升; 超过某临界时延后收益崩塌**。Sim-C 的产出 = "驾驶分 vs 时延"完整曲线 + 拐点定位。
> 设计纪律: 每个数字标口径 (真测/估算/注入档); 时延档来源 = 团队 E7 真测 (§3.4, edge_latency_budget_v1); 注入机制 = §3.2 动态折算 + §3.3 ZOH (已在 C-3 实现, agent config `simulation.latency_inject_ms` 驱动)。

### 4.0 既有事实 (实查 V2Xverse, 本设计硬约束)

| 事实 | 值 | 来源 (实查) | 对设计的意义 |
|------|----|------------|------------|
| **时延注入旋钮** | agent config `simulation.latency_inject_ms` (float, ms) | `pnp_infer_action_e2e.py` L490-497 | sweep 唯一可控量 = 改这一个 yaml 字段; C-4 d0=不设(Δ=0)/d108=设108(Δ=3) 已验证 |
| **帧→ms 量子** | 50ms/帧, Δ=ceil(L/50), 封顶 MAX_DELAY_FRAMES=10 | §3.0 + §3.3-② | 决定每个 ms 档对应 Δ 帧; 见 4.A 表 |
| **审计 log** | 每帧 6 字段 CSV `meta/latency_align_audit.csv` (含 `zoh_age_frames`) | C-3 实现; d108 实测 171 行 | 因果链中介 zoh_age 已落盘, 可聚合 |
| **结果 json schema** | `ego_vehicle_0/results.json` → `_checkpoint/global_record/{scores,infractions}` | 实查 d0/d108 ego json | DS/碰撞分类/完成率聚合源 (4.D) |
| **种子固定** | `eval_driving_e2e.sh` 写死 `CARLA_SEED=2000 TRAFFIC_SEED=2000`; evaluator L285-293/560-563 每 route 重置 np/random/torch/CARLA-provider/traffic-manager 种子 | 实查脚本+evaluator | **同 (route,档) 重复 → 交通流/物理确定性, 见 4.C 论证** |
| **RSU spawn 点** | `spawn_rsu()` @ `pnp_agent_e2e.py` L403-425, 每 `change_rsu_frame=5` 帧重生; rsu_data 为空时 L950 `if len(rsu_data_raw)>0` 安全跳过 | 实查 agent | **"无路侧"= 关 spawn_rsu, rsu_data=[], 不是无限延迟 (见 4.A-③)** |
| **场景触发机制** | 13176 个/类 trigger point 密铺全图 (Scenario1/3/4); 实际触发哪个子类型由 `scenario_parameter_$N.yaml` 的 proportion 控制 | 实查 `town05_all_scenarios_2.json` + `scenario_parameter_1.yaml` | 时延敏感场景 = `DynamicObjectCrossing`(行人横穿)/`CutIn`(车辆切入), 由 parameter yaml 配比 (见 4.B) |
| **背景对抗度** | `scenario_parameter_1.yaml` Background: `pedestrian_amount:60, CRAZY_LEVEL:3, CRAZY_PROPORTION:50` | 实查 | 背景行人+激进交通流 = 动态障碍突现源, 时延伤害的载体 |

### 4.A 时延档设计 (★加密版, 找拐点) — 逐条论证

#### 完整档位表 (4 基础 + 4 加密 = 8 档, 全用 `latency_inject_ms` 注入)

| # | 档名 | inject_ms | Δ=ceil(L/50) 帧 | game-time 滞后 | 物理依据 (E7/survey) |
|---|------|-----------|----------------|--------------|---------------------|
| 1 | **理想** | 0 (不设字段) | 0 | 0ms | 完美同步上界; C-4 d0 已验证 |
| 2 | 50ms | 50 | 1 | 50ms | **1 帧最小可分辨档** (Δ=1, 滞后量子底) |
| 3 | **最优链** | 108 | 3 | 150ms | E7: INT8 p75 RSU ~54ms + 典型 V2X 35ms ≈ 89ms→保守合成档 108; C-4 d108 已验证 |
| 4 | 150ms | 150 | 3 | 150ms | 拐点候选区间 (Δ 与 108 同为 3 帧, 校验"档值≠Δ时是否同结果"——验证折算单调性) |
| 5 | **baseline** | 219 | 5 | 250ms | FP32 零优化 e2e 合成 (orin baseline 219ms 口径, edge_budget §4) |
| 6 | 300ms | 300 | 6 | 300ms | CoBEVFlow 测试中点 (survey §3.4 测 100/300/500) |
| 7 | 500ms | 500 | 10 | 500ms (=MAX 封顶) | **CoBEVFlow 测试上界**; 恰 = MAX_DELAY_FRAMES×50ms, 封顶边界 |
| 8 | **无路侧** | — (关 RSU) | n/a | n/a | ego-only 下界对照 (见 ③) |

#### ★4090 注入档 (用户 2026-06-09 追加 — 数据中心级 RSU 对照)
现有注入档 1-8 的物理依据是 **Orin AGX 实测**(边缘设备, E7)。追加 **4090 档**代表"路侧单元配数据中心级 GPU"情形:

| 档名 | inject_ms | Δ=ceil(L/50) 帧 | game-time 滞后 | 来源/口径 (★诚实标注) |
|------|-----------|----------------|--------------|---------------------|
| **4090_fp32** | 35.31 | **1** | 50ms | M4.6.0 4090 e2e, **OPV2V 口径 + autocast**(非 DAIR 同口径, 见 caveat) |
| **4090_fp16** | 26.70 | **1** | 50ms | M4.6.0 4090 e2e FP16 autocast, 同上 caveat |

- **关键结论**: 4090 e2e ~27-35ms **< 50ms(一个 CARLA 帧)⇒ Δ 恒 = 1 帧**(最小非零延迟)。即数据中心级 RSU 在 20Hz 下对车辆"近乎无感延迟", 与 Orin 边缘(Δ=3-5)形成"强力 RSU vs 边缘 RSU"对照, 锚定时延曲线低端。
- **⇒ 4090 档在帧对齐上等价于档 2(50ms, Δ=1)**: 不产生新 Δ 值, 但作为**带 4090 出处的标注数据点**有价值(证"datacenter RSU = 亚帧延迟")。`latency_ms_source` 标 `measured_on_4090_M4.6.0_OPV2V_autocast`。
- **⚠️ caveat + 待办**: 35.31/26.70 是 OPV2V 口径 autocast, 与 Orin 的 DAIR 合成口径(pre-body+body+NMS)**不完全可比**。要干净的 DAIR 同口径 4090 e2e, 应让 **hw-optimizer 用 4090 TRT 引擎实测**(team 有引擎; 合成 = 4090 pre-body + 4090 TRT body + 4090 NMS)。但无论口径, **4090 e2e < 50ms ⇒ Δ=1 结论稳**。

#### ① 每档 Δ 折算 (50ms/帧, ceil)
Δ = ceil(inject_ms/50)，封顶 min(Δ, 10)。档 4(150ms,Δ=3) 与档 3(108ms,Δ=3) 折算同帧数 = **有意设计的折算自洽校验点**: 若两者驾驶分一致, 证明"game-time 影响只由 Δ 决定、与 ms 原值无关"(§3.2 不变量); 若不一致 = ZOH/审计逻辑有 bug, 反过来当回归检验。

#### ② 为什么这 8 点能定位拐点
- **0–500ms 全覆盖** CoBEVFlow 实测上界 (survey §3.4), 与文献延迟轴可对齐。
- **低端密 (0/50/108/150)**: 假设拐点在"接近 200ms 阈值"附近 (核心假设的 200ms), 故 100–250ms 区间放 3 个点 (108/150/219) 加密, 提高拐点分辨率。
- **高端疏 (300/500)**: 若已崩塌则只需确认"崩塌后是否继续恶化 or 触底", 2 点够。
- **50ms=Δ1 最小可分辨**: 比 50ms 更小的 ms 仍 Δ=1 (folding 到同帧), 故 50ms 是注入能产生非零滞后的下限, 无需测 <50ms 的中间值 (会与 0 或 50 重合)。
- **500ms=MAX 封顶**: 再大 inject 也被 §3.3-② 封到 Δ=10, 与 500ms 同结果, 故 500 是有意义上界。

#### ③ "无路侧单车"的代码层实现 (★不是无限延迟)
- **做法**: agent config 加 `simulation.disable_rsu: true` (新增小 flag), 在 `pnp_agent_e2e.py` `run_step` 里 gate 掉 `self.spawn_rsu()` 调用 + 强制 `rsu_data=[]`。下游 `get_action_from_list_inter` 的 L950 `if len(rsu_data_raw)>0` 已能优雅处理空 RSU (融合退化为 ego-only 感知)。
- **为何不用"无限大 inject_ms"**: 无限延迟会让 ZOH 永远沿用一个远古帧 (或封顶帧), 语义是"有 RSU 但永远过时", ≠ "根本没有 RSU"。前者仍在融合一张陈旧 BEV, 后者融合分支彻底关闭。**用户要的"无路侧单车" = 后者**, 必须在数据生成层关 RSU, 不在延迟层造假。
- **C-3 改动量**: 仅加 1 个 config flag + run_step 里 1 个 if 分支; 不碰 ZOH/aligner。列为 **C-5 (无路侧对照开关)**, 实现前同样需 team-lead 过目。

#### ④ 拐点附近自适应插档策略 (初跑后按需)
- **触发条件**: 若初跑发现相邻两档 DS 落差 > 全程跨度的 40% (如 150ms→219ms 骤降), 判定拐点落在该区间。
- **插档动作**: 在骤降区间二分插 1–2 档 (如 150/219 之间插 185ms, Δ=4)。Δ=4 是 0–500 区间唯一未被基础档覆盖的帧数 (基础档 Δ∈{0,1,3,5,6,10}, 缺 2/4/7/8/9), 故插档优先选能产生新 Δ 的 ms 值 (Δ=2→[51,100], Δ=4→[151,200])。
- **上限**: 自适应插档总数 ≤ 3 档, 防无限细分。**插档是初跑结果驱动的第二轮**, 不在首轮总量内; 首轮固定 8 档。

### 4.B route 集合与场景 — 逐条论证 + 总量估算

#### 现有 route 盘点 (实查)
- `evaluation_routes/`: **105 条** `town05_short_r{N}.xml`, 每条 = 2 waypoint 短程 (r0 实测 route_length 70.71m)。
- `validation_routes/routes_town05_short.xml`: 多 route 合集 (r0–r9+); 与 eval 单文件 r0 起终点一致。
- 场景由 `scenario_parameter_$N.yaml` 配比, **route 本身不含场景**, 场景靠 trigger point 密铺 + parameter proportion 激活。

#### 时延敏感场景识别 (核心: 时延伤害=动态障碍突现时 RSU 早预警的价值)
冒烟 r0 用 `scenario_parameter_1` 已含: `DynamicObjectCrossing`(行人横穿, proportion>0, adversary=pedestrian) + `CutIn`(车辆切入) + 背景 `pedestrian_amount:60 CRAZY_LEVEL:3`。冒烟 3 次行人碰撞 (t=33.9/35.6/36.9s 同一行人簇) = **正是行人横穿场景被触发的证据** —— 说明 r0+param_1 已是时延敏感场景, 时延↑会让"RSU 早看到横穿行人"的预警提前量丢失。

#### route 子集选择 (★已落地 — 离线筛选完成, D1=方案A)
**筛选方法 (实做, 纯离线读文件, 不跑 GPU)**: 从 105 条 `town05_short_r*.xml` 提取全部 waypoint; 从 `carla/.../Town05.xodr` 解析 **21 个 junction 几何中心** (OpenDRIVE 内部连接路参考线均值, CARLA 坐标 y 翻转); 对每条 route 计算 ① route 多段长度, ② 路径到最近 junction 的距离 dmin, ③ 路径 25m 内 junction 数 njunc, ④ 首末 yaw 转向角; 再叠加 `town05_all_scenarios_2.json` 触发点密度核验 (Scenario1=CutIn 族 / Scenario3·4=DynamicObjectCrossing 族, 13176/类 密铺, 故近路触发数 ∝ 路口密度+长度 → 印证 junction 是正确判别量)。评分 = njunc×3 + 贴路口奖励 + 长度带 [40,160]m 奖励 + 转向奖励。

**入选 6 条 (含强制 r0) + 理由**:

| rid | 长度(m) | njunc(≤25m) | 转向° | 触发密度(/类) | 入选理由 |
|-----|--------|------------|------|--------------|---------|
| **r0** | 70.7 | 1 | 0 (直道) | 132 | ★强制: 冒烟已验, 实测触发 3 次行人横穿碰撞 = 确认对抗性基线 |
| **r146** | 94.6 | **3** | 88 (转弯) | 372 | 全网最密路口几何 (3 junction 贯穿) + 转弯进路口 = 遮挡最重, RSU 高位价值最大 |
| **r28** | 70.7 | 2 | 0 (直道) | **608** | 同长直道里触发密度最高; 北象限 (与 r0 不同区域, 增场景多样性) |
| **r160** | 221.8 | **3** | 0 (直道) | **768** | 最长 (曝光最多动态障碍) + 全网最高触发密度; 西象限 |
| **r141** | 123.0 | 2 | 90 (转弯) | 612 | 90° 转弯逼近路口 = 视线遮挡型; 最西象限 |
| **r135** | 120.2 | 2 | **102** (急转) | 412 | 最急转向 (盲区横穿风险最高); 中央象限 |

**几何多样性核验**: 6 条起点两两间距全 >20m, 横跨 Town05 四象限 ⇒ route 间 std 有意义 (反映场景多样性, 非同点冗余)。**长短搭配** (70.7–221.8m): 含 2 条直道 (r0/r28) / 1 条长直道 (r160) / 3 条转弯 (r146/r141/r135), 覆盖"直行突现 vs 转弯盲区"两类时延伤害形态。
- ⚠️ **诚实标注**: junction 中心是 OpenDRIVE 连接路几何均值的**启发式近似** (非 CARLA 运行时 `is_junction` API 真值); dmin/njunc 为路径-中心点最短距, 未含车道宽度。足够做相对排序选差异化 route, 但"是否真跨路口"待首跑 audit + 4.F 证伪条件 1/2 数据驱动复核。
- **r160 为 221.8m** (>160 带): 有意纳入以最大化动态障碍曝光; 墙钟略超 8min 估计, 仍在 ~9min 单 route 上限内。
- 落地: `scripts/closedloop_sweep_v1.sh` 的 `ROUTES=(0 146 28 160 141 135)`。
- **为何 6 条不是 1 条**: 单 route 确定性 (4.C) 下无重复方差, 必须靠**多 route 聚合**拿统计量; 6 条 × 8 档 = 48 组, 足以画曲线 + 算 route 间 std。
- **为何不全 105 条**: 墙钟爆炸 (见下); 6 条是"够画曲线 + 够算 std"的最小集。

#### 总实验量 + 墙钟预算
- **首轮**: 6 route × 8 档 × **1 次** (确定性, 见 4.C) = **48 次 route 跑**。
- **单 route 墙钟**: C-4 实测 d0/d108 单 route `duration_system` ≈ 356–660s (含 PTX-JIT 暖机 + 单卡推理)。**分卡 + 异步提交后** (§3.2): 推理与 tick 重叠, 估单 route ~5–8 min。取保守 **8 min/route**。
- **首轮总墙钟**: 48 × 8 min = **~6.4 h** (纯计算; 不含 CARLA 启停开销, 见 4.E)。加每 route CARLA 启停 ~1 min × 48 = +48 min ⇒ **~7.2 h 墙钟**。
- **自适应插档 (条件第二轮)**: ≤3 档 × 6 route × 8 min ≈ +2.4 h。
- **无路侧对照 (C-5)**: 已含在 8 档的档 8, 6 route 共用首轮预算, 不额外加。
- **总预算**: 首轮 **~7.2 h**, 含插档上限 **~9.6 h**。单 GPU 对 (CARLA + 推理 2 卡) 串行跑; 若给 2 对 GPU 并行减半至 ~3.6–4.8 h。

### 4.C 重复与统计 — ★诚实论证 (不为 error bar 硬跑)

#### 确定性核查 (实查证据)
- `eval_driving_e2e.sh` 写死 `CARLA_SEED=2000 TRAFFIC_SEED=2000` (非每次随机)。
- `leaderboard_evaluator_parameter.py` L285-293 在 load world 后 `CarlaDataProvider.set_random_seed(2000)` + np/random/torch/cuda 全 seed=2000; L560-563 每 route 重入再 seed; L293 `traffic_manager.set_random_device_seed(2000)`。
- ⇒ **交通流生成 + 行人 spawn + 物理 = 同种子确定性**。同 (route, 档) 重复跑, 交通流/障碍物轨迹应逐帧一致。
- ⚠️ **唯一残余非确定性**: CARLA 物理引擎在 sync 模式下基本确定, 但浮点/异步 actor 销毁 (§2.5 B-3 teardown) 可能引入极小扰动; **推理本身确定** (固定 ckpt + torch seed)。

#### 统计方案 (★决策: 不做同-route 重复, 改多-route 聚合)
- **同 (route,档) 只跑 1 次**: 种子确定性下重复跑结果应一致, error bar 会是"假 0 方差", 硬跑纯属浪费墙钟 (反 CLAUDE.md "不为有 error bar 硬跑无意义重复")。
- **统计量来源 = route 间方差**: 对每个时延档, 聚合 6 条 route 的 DS → 报 **均值 ± route 间 std**。曲线 y 轴用档均值, error bar = route 间 std。这才是有意义的方差 (反映场景多样性, 非种子噪声)。
- **确定性自检 (廉价保险)**: 首轮**仅对 1 个 (route,档) 组合** (如 r0,108ms) 重跑 1 次, 比对 DS + 审计 log 逐字一致 ⇒ 证实确定性假设。若不一致 (出现非确定性), 则回退到"每组 3 次取均值"方案 (墙钟 ×3, 需重新申请预算)。**这 1 次自检是设计的证伪闸门, 不是统计重复。**

### 4.D 指标 — 聚合方法 (实查 json/log schema)

| 指标 | 类型 | 来源字段 (实查) | 聚合 |
|------|------|----------------|------|
| **driving_score** | 主 | `ego_vehicle_0/results.json` → `_checkpoint/global_record/scores/score_composed` | 每档 6-route 均值±std |
| route_completion | 主 | 同上 `scores/score_route` | 每档均值 |
| infraction_penalty | 辅 | 同上 `scores/score_penalty` | 每档均值 |
| **collision_rate (分类)** | 主 | `global_record/infractions/{collisions_pedestrian, collisions_vehicle, collisions_layout}` (per-km 计数) | 分行人/车辆/设施三轴, 每档求和或均值 |
| 其他 infraction | 辅 | `infractions/{red_light, stop_infraction, outside_route_lanes, route_dev, vehicle_blocked, route_timeout}` | 监控异常 |
| duration_game/system | 元 | `records[0]/meta/duration_{game,system}` | 墙钟核算 + 异常检测 |
| **zoh_age_frames (均值)** | ★中介 | `meta/latency_align_audit.csv` 第 6 列 | 每 route 跑出 → 全帧均值; 跨 route 再均值 |
| delta / is_zoh_held 占比 | 中介辅 | audit csv 第 2/4 列 | 验证注入档 Δ 与 4.A 表一致 (帧对齐核验) |

#### 因果链分析 (用户要的"中间变量")
**时延档(inject_ms) → zoh_age_frames均值↑ → 感知陈旧度↑ → DS↓ / 碰撞率↑**。
- 散点: x=inject_ms, y1=DS (主曲线), y2=zoh_age 均值 (中介验证)。
- 若 DS 随档下降但 zoh_age 不随档上升 ⇒ 折算/ZOH 逻辑异常 (回归检验)。
- 若 zoh_age 随档线性上升但 DS 全平 ⇒ 该 route 集对时延不敏感 (证伪条件, 见 4.F)。
- 聚合脚本: 从每个 `results_driving_lat_*/.../ego_vehicle_0/results.json` 抽 scores+infractions, 从配对的 `image/.../meta/latency_align_audit.csv` 抽 zoh_age 均值, 按 (route,档) join 落 `results/closedloop_sweep_v1.csv`。

### 4.E 执行编排 — sweep 驱动 (★骨架写出但不执行)

#### 编排决策
- **每 (route,档) 一次 CARLA 启停**: 不复用 server 跨 route。理由: §2.5 B-3 teardown 挂死缺陷 —— route 跑完 CARLA actor 销毁卡死, 复用 server 跑多 route 会让前一 route 的挂死污染后一 route。**单 route 跑完即 kill CARLA + evaluator** (C-4 已验证此 teardown 流程), 干净隔离。
- **分卡**: CARLA `CUDA_VISIBLE_DEVICES=<render_gpu>`; 推理服务 (C-1 进程) `CUDA_VISIBLE_DEVICES=<infer_gpu>`, 两卡物理隔离 (§3.1 算力隔离)。跑前 `nvidia-smi` 确认两卡 util 0%/mem≤50MiB。
- **失败重试**: 单 (route,档) 失败 (status≠Completed 或 json 缺失) → 重试 1 次; 二次失败记入 `sweep_failures.log` 跳过, 不阻塞 sweep。
- **落盘命名**: `RESULT_ROOT=results/results_driving_lat_d{inject_ms}` (d0/d108 已是此规范); 聚合表 `results/closedloop_sweep_v1.csv`; 审计 log 随 image 目录。
- **teardown 挂死规避**: 每 route 后显式 `kill -9 <CarlaUE4-Linux-Shipping> <CarlaUE4.sh>; pkill -9 -f leaderboard_evaluator_parameter` (§2.2 步骤 5), sleep 5s 让端口释放再起下一 route。

#### sweep 驱动脚本骨架 (★写出, 不执行)
```bash
#!/bin/bash
# scripts/closedloop_sweep_v1.sh — Sim-C latency sweep driver (设计稿, 待授权)
# 用法: bash closedloop_sweep_v1.sh <render_gpu> <infer_gpu> <carla_port>
set -u
RENDER_GPU=${1:-2}; INFER_GPU=${2:-0}; PORT=${3:-40000}
ROUTES=(0 5 13 21 28 100)          # 6 条 (待 4.B prep 按路口几何确定)
INJECT_MS=(0 50 108 150 219 300 500)  # 7 注入档; 第 8 档"无路侧"用 disable_rsu 单列
AGENT_BASE=codriving_5_10
SWEEP_CSV=results/closedloop_sweep_v1.csv

for rid in "${ROUTES[@]}"; do
  for L in "${INJECT_MS[@]}"; do
    TAG="lat_d${L}"
    # 0. nvidia-smi 守卫: 两卡必须空闲 (util 0% / mem<=50MiB), 否则 abort
    # 1. 写临时 agent config: 注入 simulation.latency_inject_ms=$L (L=0 时不设字段)
    # 2. 起推理服务 (C-1) 于 INFER_GPU (后台)
    # 3. 起 CARLA 于 RENDER_GPU (后台), 等 ~30-60s 就绪
    CUDA_VISIBLE_DEVICES=$RENDER_GPU ./external_paths/carla_root/CarlaUE4.sh \
        --world-port=$PORT -prefer-nvidia -opengl &
    # 4. 跑 leaderboard (1 次, 确定性): repeat=0
    CUDA_VISIBLE_DEVICES=$RENDER_GPU bash scripts/eval_driving_e2e.sh \
        "$rid" "$PORT" "$TAG" 0 "$AGENT_BASE" _1
    # 5. 校验 ego results.json status=Completed; 失败重试 1 次, 二次失败记 sweep_failures.log
    # 6. teardown: kill -9 CarlaUE4-Linux-Shipping / CarlaUE4.sh; pkill evaluator; sleep 5
    # 7. 聚合: 抽 scores+infractions + audit zoh_age → append SWEEP_CSV (rid,L,delta,DS,...)
  done
done
# 第 8 档"无路侧"对照 (C-5 disable_rsu=true), 同 6 route 各跑 1 次
```
> 骨架是**伪步骤注释 + 真命令混排**, 真正落地前 team-lead 过目 + 实现 4.A-③ 的 disable_rsu flag (C-5) + 临时 config 注入逻辑。

### 4.F 预期曲线与证伪条件 (★防确认偏误, 预先写死)

#### 预期形态 (基于 CoDriving +62.49%DS / -53.50%碰撞 量级 + survey)
- **低端 (0–108ms) 平台**: DS 接近"无路侧 (档8)"之上的协同收益高位, 随时延缓降。E7 显示 INT8 p75 e2e ~89ms < 200ms, 此区间 RSU 预警仍及时, 假设 DS ≈ 平。
- **中端 (108–250ms) 缓降→陡降**: 接近核心假设 200ms 阈值, 预警提前量被 ZOH 陈旧化吃掉, DS 开始明显掉、行人碰撞率回升。**拐点假设落在此区间** (4.A 加密点正打这里)。
- **高端 (300–500ms) 触底/趋平**: 陈旧 RSU 数据已无用, DS 趋近"无路侧"水平。CoBEVFlow 测到 500ms 仍跑 ⇒ 不会崩到 0, 但应明显低于低端。
- **无路侧 (档8)**: ego-only 下界, DS 应 ≤ 所有有 RSU 的档 (若高时延档 DS < 无路侧, 说明陈旧 RSU 比没有更糟 —— 是有价值的反直觉发现, 不是 bug)。

#### 证伪条件 (出现即说明假设/场景有问题, 预先承认)
1. **0ms 与 500ms DS 无显著差异** (跨度 < route 间 std) ⇒ **该 route 集对时延不敏感** —— 说明 r0+param_1 场景里 RSU 预警价值太低 (短 route / 行人簇太少 / 无遮挡)。**应对: 换更难 route (含强遮挡/密集横穿) 或调高 `scenario_parameter` 的 DynamicObjectCrossing proportion + pedestrian_amount + CRAZY_LEVEL**。
2. **zoh_age 随档单调升但 DS 全平** ⇒ 感知陈旧度确实增大但不传导到驾驶 —— 可能 planner 对感知延迟不敏感 (用了里程计/规则兜底), 或场景太易 (冒烟 r0 机制验证 100 分即此类)。**应对: 同 1, 加难度。**
3. **DS 随档非单调 (忽高忽低)** ⇒ 确定性假设破裂 (4.C 自检该提前抓到) 或单 route 噪声主导 ⇒ 回退每组多次取均值。
4. **无路侧 DS ≥ 低时延档 DS** ⇒ RSU 协同在本场景无净增益 (与 CoDriving +62.49% 矛盾) ⇒ 排查 RSU spawn/融合是否真生效 (审计 used_rsu_frame_id 是否真在更新)。

> 证伪条件 1/2 直接回答 team-lead 决策点 (下方): 若初跑命中 1 或 2, 即证"现有 route 不够对抗", 需升级场景。

---

### ★给 team-lead 的决策点 (单列, 过目时拍板)

> **决策点 D1 — 现有 route 是否够对抗性 / 是否需更难 route?**
>
> **现状**: 冒烟 r0 + `scenario_parameter_1` 已含行人横穿 (DynamicObjectCrossing) + 车辆切入 (CutIn) + 背景 60 行人/CRAZY_LEVEL3, 且冒烟实测触发 3 次行人碰撞 = 场景确有动态障碍。**但 r0 是"最易 route"** (团队定性: 70m 短程, 机制验证非时延测量), 单 route 100 分。
>
> **我的建议 (待 team-lead 裁定)**:
> - **方案 A (先跑现有, 推荐)**: 首轮直接用 6 条 short route + param_1, 跑 8 档。**靠 4.F 证伪条件 1/2 自动检测对抗性是否足够** —— 若 0ms vs 500ms 拉不开差距, 即数据驱动地证明"需换难场景", 再升级。优点: 不预先猜难度, 用真实验数据说话; 缺点: 可能首轮白跑 (但仅 ~7h)。
> - **方案 B (预先加难)**: 首轮就调高 `scenario_parameter` 的 DynamicObjectCrossing proportion + pedestrian_amount(60→120, param 文件已有 120 档) + CRAZY_LEVEL, 或换 `additional_routes` 的 long route (town03/04/06_long, 更多路口遮挡)。优点: 一次到位; 缺点: 预设难度可能过/欠调, 偏离冒烟已验证的 r0+param_1 基线。
>
> **倾向 A** (符合 CLAUDE.md "用真测说话, 别预先假设")。请 team-lead 确认走 A 还是 B, 以及 6 条 route 具体选哪些 (我可按"含路口/遮挡几何"从 105 条筛一份候选给你审)。
>
> **决策点 D2 — 4.A-③ 无路侧对照 (C-5 disable_rsu flag) 是否现在实现?** 需加 1 个 config flag + run_step 1 个 if 分支, 小改动, 但属新代码, 按纪律需授权。
>
> **决策点 D3 — GPU 资源**: 8 档 × 6 route 串行 ~7.2h (1 对 GPU)。当前 8 卡被 wuyuegao 占满, 需 team-lead 协调空出 2 卡 (CARLA + 推理分卡), 或 4 卡 (2 对并行减半至 ~3.6h)。

## §5 闭环指标入库 (D 阶段) — TODO

> 新增闭环指标列须先与 data-orchestrator 议定 schema, 不擅自改主表列。
> [TODO]
