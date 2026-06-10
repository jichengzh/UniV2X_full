# 新服务器部署交接 (HANDOFF_new_server_deploy_v1)

> 写于 2026-06-10。目的: 在**全新服务器**上从零配好环境 + 跑通现有代码(尤其 V2Xverse CARLA 闭环仿真 + 我们写的 RSU 时延回灌 + L1 控制器)。
> **★给接手 AI 的硬指令**: 本文 §2 列了**不能上传 git 的资产**(CARLA/数据集/ckpt/conda 环境)。**遇到这些"不能上传"标记项, 必须先向用户确认如何获取/传输, 不要假设它们已存在、也不要擅自下载几十 GB。** 见 §2 每项的 `[需用户确认]` 标记。

---

## §0 ★最重要先读 — 什么没在云端 (否则新服务器拿不到代码)

| 内容 | 在云端吗? | 处置 |
|------|----------|------|
| **V2X 主仓** (`hw-deploy-d-space` 分支) | ✅ 已 push 到 `git@github.com:jichengzh/UniV2X_full.git` | 可直接 clone。**但有 1132 个未提交改动**(多为既有论文/模型工作)未上云, 见 §1.1 |
| **V2Xverse 仿真仓 — 我们写的全部闭环/L1 代码** | ✅ **已 push 到用户 fork `git@github.com:jichengzh/V2Xverse.git`** (2026-06-10) | `sim-closedloop`(29c) + `feature/l1-trajectory-tracker`(33c, L1 全部工作) 已上云。新服务器从这个 fork clone, 见 §1.2。 |
| **CARLA 0.9.10.1** (16GB) | ❌ 不能进 git | §2-A `[需用户确认]` |
| **CoDriving / HEAL checkpoints** | ❌ 不能进 git | §2-B/C `[需用户确认]` |
| **DAIR/OPV2V 数据集** (71GB) | ❌ 不能进 git | §2-D `[需用户确认]` |
| **conda 环境** (v2xverse / UniV2X_2.0) | ❌ 不能进 git | §3 重建 |

---

## §1 两个代码仓库

### 1.1 V2X 主仓 (论文/框架/multi_agent 文档)
- 路径(旧服务器): `/home/jichengzhi/V2X`
- remote: `git@github.com:jichengzh/UniV2X_full.git` (用户自己的 fork, 可 push)
- 现役分支: **`hw-deploy-d-space`** (已与云端同步, 0 未 push commit)
- **clone**: `git clone -b hw-deploy-d-space git@github.com:jichengzh/UniV2X_full.git`
- ⚠️ **1132 个未提交改动**(`git status`): 多为既有 `paper_learning/`、`models/`、`framework/` 工作 + 本次 `multi_agent/` 文档。**这些不在云端**。若新服务器需要它们 → `[需用户确认]` 是否要我先 commit+push 这批(量大、含多类文件, 需用户决定哪些要上云)。
- 本次会话已 push 的: `07821e5`(闭环仿真文档整合) — 在云端 ✅。

### 1.2 ★V2Xverse 仿真仓 (闭环仿真 + 我们写的 L1/时延回灌) — 已上云
- 路径(旧服务器): `/home/jichengzhi/V2Xverse`
- remote:
  - `origin` = `https://github.com/CollaborativePerception/V2Xverse.git` (官方上游, 只读)
  - **`myfork` = `git@github.com:jichengzh/V2Xverse.git` (用户 fork, 可读写, 我们的代码在这)**
- 工作分支(**已 push 到 myfork, 2026-06-10**):
  - **`feature/l1-trajectory-tracker`** (33 commit, ★最新, 含 L1 控制器全部工作) — HEAD `dff86d5`
  - `sim-closedloop` (29 commit, 时延回灌 + 探针 + 闭环改造基础)
- **新服务器 clone**:
  ```bash
  git clone -b feature/l1-trajectory-tracker git@github.com:jichengzh/V2Xverse.git
  # (sim-closedloop 分支同 fork 可 fetch; feature 分支是从 sim-closedloop 长出来的, 含全部工作)
  ```
- ⚠️ 旧服务器还有 `?? multi_agent/`(stray, 属于 V2X 不属于 V2Xverse, **未 commit 进 V2Xverse, 也不该**)。L1 代码已全部 commit 到 `dff86d5` 并 push。

---

## §2 ★不能上传 git 的资产 (接手 AI: 每项遇到必向用户确认)

> 这些是运行时硬依赖, 体积大/是二进制/是数据, **绝不能进 git**。新服务器需另行获取。**接手 AI 看到下面 `[需用户确认]`, 必须先问用户怎么拿到, 不要假设已存在、不要擅自下几十 GB。**

### A. CARLA 0.9.10.1 (16GB) `[需用户确认]`
- 旧路径: `/home/jichengzhi/V2Xverse/carla/` (CarlaUE4 二进制 + AdditionalMaps)
- V2Xverse 通过软链接用它: `external_paths/carla_root -> V2Xverse/carla/`
- 获取: 官方 `setup_carla.sh`(wget 两个 tar: CARLA_0.9.10.1.tar.gz 3.96GB + AdditionalMaps 1.82GB) 或从旧服务器 scp。
- **确认点**: 新服务器是 scp 旧的 / 还是重新 wget? CDN 对大文件不稳, 旧服务器是 `wget -c` 续传装的。
- 装完需: `easy_install carla-0.9.10-py3.7-linux-x86_64.egg`(setuptools==41) 让 `import carla` 通; `DefaultGameUserSettings.ini` 加 `ServerRandomSeed Seed=1234`。

### B. CoDriving checkpoints (51MB) `[需用户确认]`
- 旧路径: `/home/jichengzhi/V2Xverse/checkpoints/codriving/`
  - `perception/net_epoch_bestval_at16.pth` (32MB, ★闭环现役感知)
  - `planner/codriving_planner.ckpt` (20MB, ★规划) + `config.yaml`
- 获取: HF `gjliu/v2xverse` 或从旧服务器 scp。**确认点**: HF 下 or scp?

### C. HEAL Pyramid checkpoints `[需用户确认]`
- 旧路径: `/home/jichengzhi/heal_research/checkpoints/stage1/`
  - `Pyramid_DAIR_m1_base_2023_08_14_11_42_29/` (★DAIR 金标准, 用于硬件加速/AP 实验, 闭环仿真**不直接用**)
  - `Pyramid_DAIR_m1_base_sparse24_2026_06_03/`
- **闭环仿真(V2Xverse)用不到 HEAL ckpt**(闭环用 CoDriving)。只有硬件加速/AP 实验需要。**确认点**: 新服务器是否要跑硬件/AP 实验? 若只跑闭环则不必传。

### D. 数据集 `[需用户确认]`
- `/home/jichengzhi/heal_research/dataset/OPV2V_orig` (71GB) — HEAL 训练/AP eval 用。
- `/home/jichengzhi/V2Xverse/dataset/` (空, 4KB) — ★**闭环仿真不需要离线数据集**(用 CARLA 实时传感器), 空目录即可。
- **确认点**: 新服务器只跑闭环仿真 → **不需要 71GB 数据集**(省事); 要跑 HEAL 训练/AP 才需要 → 确认是否传。

---

## §3 环境配置 (conda, 两个环境)

### 3.1 `v2xverse` (Python 3.7.16) — ★闭环仿真主环境
旧服务器 `/home/jichengzhi/miniconda3/envs/v2xverse`。完整重建(按 V2Xverse README + 本项目实测偏差):
```bash
conda create -n v2xverse python=3.7 -y && conda activate v2xverse
# torch 栈 (官方版, 4090 上走 PTX-JIT 前向兼容, 首次算子 ~1.6s 暖机, 功能正常)
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install -r opencood/requirements.txt        # open3d 0.17, matplotlib 3.5.3, h5py...
pip install -r simulation/requirements.txt       # numpy 降到 1.18.3, opencv 4.2.0.34, py-trees 0.8.3 (含 setuptools==41 装 egg)
# ★关键偏差 (本项目实测, 别用官方 spconv 1.2.1 源码 build):
pip install spconv-cu113==2.3.6                  # opencood 自带 v2.x 兼容分支, voxel 已验证
python setup.py develop                           # opencood 0.1.0 + box_overlaps.so 编译
# pypcd (klintan/pypcd + python-lzf), efficientnet_pytorch==0.7.0
easy_install <carla egg>                          # 见 §2-A
```
- ★实测偏差(见 V2X 仓 `multi_agent/archive/sim_closedloop_arch_v1.md §1.5` D-1~D-5): spconv 用 2.3.6 wheel(非 1.2.1 源码); setuptools 留 41.2.0 + 删 `distutils-precedence.pth`; numpy 1.18.3。
- sm89(4090): torch 1.10 无 sm89 cubin 但走 PTX-JIT, **功能正常不用升级**。若新服务器是别的卡(如 A100 sm80)有官方 cubin, 更顺。

### 3.2 `UniV2X_2.0` (Python 3.9.25) — V2X 主仓/纯数学测试用
- 旧服务器 `/home/jichengzhi/miniconda3/envs/UniV2X_2.0`。
- L1 的 V0 纯数学测试(无 CARLA)用它跑: `python simulation/leaderboard/team_code/closedloop/test_l1_trajectory_controller.py`。
- **确认点**: 这个环境的完整 requirements 没在本文; 若新服务器要重建, `[需用户确认]` 是否需要(纯数学测试其实任何带 numpy 的 py3 都能跑)。

---

## §4 如何在新服务器跑现有代码

> 前提: §1.2 拿到 V2Xverse 代码 + §2 资产就位 + §3 环境建好。所有路径把 `/home/jichengzhi/` 换成新服务器实际路径。

### 4.1 V0 纯数学测试 (无 GPU, 先验代码完整)
```bash
cd <V2Xverse>; conda activate v2xverse   # 或任意 numpy 环境
python simulation/leaderboard/team_code/closedloop/test_l1_trajectory_controller.py
# 期望: V0-1~V0-9 全 PASS (round-trip/约定/frozen-speed/synthetic/PID隔离)
```

### 4.2 CARLA 闭环冒烟 (需 GPU + CARLA)
```bash
cd <V2Xverse>; conda activate v2xverse
# 起 CARLA (空闲卡; 启动前 nvidia-smi 确认 util 0%/mem<=50MiB)
CUDA_VISIBLE_DEVICES=<gpu> ./external_paths/carla_root/CarlaUE4.sh --world-port=40000 -prefer-nvidia -opengl -RenderOffScreen &
# 等 CARLA 就绪 (冷启动可能 >90s, 用 nc -z 127.0.0.1 40000 探测; 别用 90s 死超时)
# 跑 1 route 闭环 (route0, port40000, tag=smoke, repeat0, agent=codriving_5_10, scenario=_1)
CUDA_VISIBLE_DEVICES=<gpu> bash scripts/eval_driving_e2e.sh 0 40000 smoke 0 codriving_5_10 _1
# 权威结果: results/results_driving_smoke/.../r0_repeat0/ego_vehicle_0/results.json
# 跑完手动 kill (teardown 挂死坑): kill -9 <CarlaUE4 pids>; pkill -9 -f leaderboard_evaluator_parameter
```

### 4.3 L1 控制器验证 (V1, 需 GPU)
```bash
# 单卡: bash scripts/v1_l1_smoke_r146_singlecard.sh <gpu> <port>
# 分卡: bash scripts/v1_l1_smoke_r146.sh <render_gpu> <infer_gpu> <port>
# 用 setsid nohup ... & 起 (否则随 shell 结束被 SIGHUP 杀)
# 结果直接读 results/results_driving_v1_l1_{d0,d500,norsu}/.../results.json
```

### 4.4 ★CARLA 运行已知坑 (本会话血泪, 脚本已修, 新脚本继承)
1. **冷启动慢**: 首次 CARLA 启动可能 >90s, 用 `nc -z` 轮询 + 360s 超时(别 90s)。
2. **孤儿进程占端口**: 超时杀 CARLA 只杀 wrapper, `CarlaUE4-Linux-Shipping` 二进制变孤儿占端口 → 后续启动撞。每次启动前 `pkill -9 -f "CarlaUE4.*world-port=$PORT"` 清孤儿。
3. **teardown 挂死**: route 跑完 CARLA actor 销毁卡死, 顶层 results.json 不 flush; **权威在 `ego_vehicle_0/results.json`**; 每 route 后 kill -9。
4. **infraction 是 per-km float 不是 list**: 读 results.json 解析碰撞用 `float(v)` 别 `len(v)`。
5. **长后台任务必须 `setsid nohup`**: 否则编排脚本随 agent/shell idle 被 SIGHUP 杀, 只剩孤儿单跑不自续。

---

## §5 当前工作状态 (接手时的进度)

- **闭环仿真框架**: Sim-A(环境+冒烟)/Sim-B(时延感知回灌+算力隔离)已完成。详见 V2X 仓 `multi_agent/real_test/sim_test_design_v1.md`(整合主文档) + `sim_latency_fidelity_reflection_v1.md`(诚实反思) + `sim_ego_rsu_latency_strategy_v1.md`(τ_ego 策略)。
- **RSU 时延回灌**: trace 回放真实逐次 τ 已打通(R-1~R-4 完成)。
- **难场景探针**: r146/r160 × {d0,d500,norsu} 跑完, 诚实结论 = "时延伤驾驶是脆弱单点存在性证明, 非鲁棒; RSU 无正向增益"。
- **★item4 I-1 L1 控制器(τ_ego 前提)**: 经多轮修复(旋转符号 / V2 frozen-speed / Fix-D PID 隔离), V0-1~V0-9 全过。**最后一步 V1 重验(Fix-D)未跑** —— 旧服务器上 staged 待 GPU+用户 GO。验收: d0_l1 DS≥90 无 layout 碰 / norsu RC=100 / d500≤d0。代码全在 `feature/l1-trajectory-tracker` HEAD `dff86d5`。
- 下一步(L1 过后): I-2 τ_ego 注入 → 加速档 sweep → "加速→驾驶分"曲线(项目卖点)。详见 `sim_ego_rsu_latency_strategy_v1.md`。

---

## §6 接手 AI 的行为约定 (重申)
1. **遇到 §2 `[需用户确认]` 项, 先问用户怎么获取/传输, 别假设存在、别擅自下几十 GB。**
2. **V2Xverse 代码不在云端**(§1.2): 接手第一件事确认用户选了哪种传输方案, 拿到代码再动。
3. GPU 实验前 `nvidia-smi` 确认空闲(util≤5%/mem≤50MiB); 用 `setsid nohup`; 跑完清理。
4. 区分两个 conda 环境(v2xverse py3.7 跑 CARLA / 任意 numpy 跑 V0 纯数学)。
