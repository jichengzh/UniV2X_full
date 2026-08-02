# 交接文档 — H800 torch2 迁移完成 + 仿真迁移待启动 (v1, 2026-06-10)

> 写给**新会话冷启动接手**。读完这份应能：知道 H800 上已搭好什么、怎么用、下一步仿真迁移的核心难题在哪。
> 本次会话成果：在 8×H800 服务器上把 HEAL/Pyramid 感知栈从 torch1.10 迁到 torch2，**finetune 能力已端到端验证通过**。
> 下一步（本文重点）：**V2Xverse 闭环仿真迁移到 H800**。

---

## 〇、一句话现状
- ✅ H800 (Hopper sm90) 上 **torch2 感知栈 + Pyramid finetune 已跑通验证**（forward+backward+step×3，loss 正常，权重 0 缺失载入）。
- ⬜ **仿真（V2Xverse 闭环）尚未迁移** —— 卡在"py3.7 的 CARLA 客户端 ↔ py3.10+torch2 的 sm90 感知"无法同进程共存，需要架构决策（见 §五）。

---

## 一、服务器接入 + 致命坑（先读，全是踩过的）

| 项 | 值 |
|---|---|
| SSH | `ssh ${V2X_REMOTE_USER}@<PRIVATE_HOST> -p 30001` 密码 `12345678`（每次会话确认） |
| GPU | 8×H800 80GB (sm90/Hopper)。**GPU 5/6/7 空闲可用；0–4 被 `wangziqi` 训练占用，勿碰**。永远 `CUDA_VISIBLE_DEVICES=5`(或6/7) |
| 工作根目录 | `/data/jichengzhi_v2x/`（data 盘）。**`/data` 只剩 ~124G 且 99% 满、多人共享，传大文件前先 `df -h /data`** |
| **内网隔离** | H800 **无外网**，pip/conda/apt 直连全失败。**必须走代理 `http://<PRIVATE_HOST>:7890`**（Clash；`17890` 是坏的别用）。`pip3 install --proxy http://<PRIVATE_HOST>:7890 ...` |
| 家目录 | `/` 98% 满，**不要装到 home**，一律 `--target=/data/jichengzhi_v2x/t2lib` |
| 工具 | 系统 `python3`=3.10、`pip3`；`gcc/g++ 11`、`nvcc 12.2`(`/usr/local/cuda`)。**`pigz` 没装，解压用 `gzip`/`tar -xzf`** |
| **连接易断** | 长前台 ssh 命令会被远端掐断（~分钟级），跑出来的进程随之死。**所有长任务必须 `nohup ... > log 2>&1 &` detached + 轮询日志**，绝不前台跑 |

---

## 二、已搭好的 torch2 环境（怎么用）

**环境 = pip `--target` 目录**（非 venv，因系统缺 python3.10-venv）：`/data/jichengzhi_v2x/t2lib`

跑任何 HEAL 代码的固定前缀：
```bash
cd /data/jichengzhi_v2x/HEAL
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/jichengzhi_v2x/t2lib:/data/jichengzhi_v2x/HEAL python3 ...
```

关键包版本（已验证 sm90 可跑）：
```
torch 2.1.2+cu121   torchvision 0.16.2+cu121   numpy 1.26.4(★必须<2)
spconv 2.3.6(spconv-cu120)   cumm 0.4.11   open3d 0.19.0   h5py 3.16.0
python-lzf 0.2.6   pypcd(已打补丁)   timm/efficientnet_pytorch/seaborn/pyquaternion
icecream/einops/easydict/numba/opencv-python-headless/matplotlib/scikit-image/tensorboardX/cython/shapely/pandas/scipy
```
> ★**numpy 陷阱**：open3d/scikit 会把 numpy 拉到 2.x，与 torch2.1 ABI 不兼容。装完任何新包后若报 `_ARRAY_API not found`，执行
> `pip3 install --proxy http://<PRIVATE_HOST>:7890 --target=/data/jichengzhi_v2x/t2lib --upgrade --force-reinstall --no-deps "numpy==1.26.4"`
> ★**torchvision 陷阱**：装 timm 等会拉太新的 torchvision（报 `torch.library has no attribute register_fake`）。固定用 `torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121`

---

## 三、HEAL 代码迁移补丁（已做，列清以便复现/排查）

代码在 `/data/jichengzhi_v2x/HEAL`（从本地 `${V2X_HOME}/heal_research/HEAL` rsync，排除了 logs/.git/*.pth）。

1. **pcdet_utils CUDA 算子（删 THC 死代码 + 重编）**：10 个 pointnet2 `.cpp` 删了 `#include <THC/THC.h>` 和 `extern THCState *state;`（torch2 已删 THC）。重编命令：
   ```bash
   cd /data/jichengzhi_v2x/HEAL
   CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH PYTHONPATH=/data/jichengzhi_v2x/t2lib \
   TORCH_CUDA_ARCH_LIST="9.0" CUDA_VISIBLE_DEVICES=5 python3 opencood/pcdet_utils/setup.py build_ext --inplace
   ```
   产物：iou3d_nms / roiaware_pool3d / pointnet2_batch / pointnet2_stack 的 `.cpython-310...so`（已验证 sm90 真执行）。
2. **spconv 1.x→2.x import 补丁**（4 文件）：`opencood/utils/spconv_utils.py`、`models/sub_modules/{matcher_v2,matcher_v3,sparse_backbone_3d}.py` —— `import spconv`→`import spconv.pytorch as spconv`；`from spconv.modules`→`from spconv.pytorch.modules`。（`SparseConvTensor` 签名/属性兼容；`sp_voxel_preprocessor.py` 已自带 2.x 分支无需改。）
3. **box_overlaps cython 重编**（py3.10）：`cd HEAL && PYTHONPATH=t2lib python3 opencood/utils/setup.py build_ext --inplace`
4. **pypcd 补丁**：`t2lib/pypcd/pypcd.py` 把 `import lzf` 包成 try/except（但 lzf 已装，正常走）。
5. **modality_assign 软链**：`HEAL/opencood/logs/heter_modality_assign/opv2v_4modality.json` → `HEAL/opencood/modality_assign/opv2v_4modality.json`（logs 目录传时被排除，手动补的软链）。

---

## 四、数据 + checkpoint（感知侧）

- **DAIR-V2X-C 原始数据**（48G）：`/data/jichengzhi_v2x/dair/cooperative-vehicle-infrastructure/`（从本地 `/data/DAIR-V2X/...` rsync，跳过了重复的压缩包）。
- **HEAL dataset 接入**：`/data/jichengzhi_v2x/HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/` = `train.json`(4811) + `val.json`(1789，即"DAIR val 1789") + 软链 `cooperative/infrastructure-side/vehicle-side` → 上面那 48G。
- **checkpoint**：`/data/jichengzhi_v2x/ckpt/Pyramid_DAIR_m1_base/{config.yaml, net_epoch_bestval_at23.pth}`，载入 **0 missing / 0 unexpected**。
- ⚠️ **AP 可信度警告（务必处理再报 AP）**：`cooperative/label_world_backup` 和 `vehicle-side/label/lidar_backup` 当前是**软链到原始标签**（`label_world`/`lidar`）的 **front≈all 近似**，不是 HEAL 训练用的补全标注。**要拿能写论文的 AP，必须配正确的补全标注**：本地 `${V2X_HOME}/heal_research/dataset/my_dair_v2x_supplement/DAIR-V2X-C_Complemented_Anno/`（231M，未传），里面 `new_labels/{cooperative_label,vehicle-side_label}/label_world/` 才是真 backup 标签源。

**finetune 验证脚本**：`/data/jichengzhi_v2x/finetune_test.py`（建数据集→载权重→forward→loss→backward→step×3）。最近一次输出：`[STEP 0/1/2] ... loss=0.27/0.78/0.78  === FINETUNE_OK ===`。

---

## 五、★下一步主任务：V2Xverse 闭环仿真迁移到 H800

### 5.1 已验证的事实
- **CARLA 0.9.10 无头渲染在 H800 上能跑**（已实测）：`/data/jichengzhi_v2x/carla/CarlaUE4.sh`，命令
  `CUDA_VISIBLE_DEVICES=6 ./CarlaUE4.sh -prefer-nvidia -opengl -RenderOffScreen -graphicsadapter=6 -world-port=3000 -nosound -quality-level=Low`
  → RPC 端口 3000 监听，渲染落指定空闲 GPU。（H800 缺 Vulkan loader，但 0.9.10 走 OpenGL/EGL 路径 OK。）
- **CARLA 升级 0.9.15 路线已否决**：0.9.15 预编译只有 cp37、py3.8 要源码编 CARLA（内网不可行），且 V2Xverse vendored 的 leaderboard/scenario_runner 绑死 0.9.10 API。**结论：留在 0.9.10**。

### 5.2 核心矛盾（必须先解决的架构问题）
- V2Xverse 的 CARLA python 客户端**只支持 py3.7**（egg=`carla-0.9.10-py3.7`）。
- 但 **H800(sm90) 上跑 GPU 感知必须 torch2 / py3.10**（torch1.10 在 sm90 上 conv 直接崩，本会话实测确认）。
- → **py3.7(CARLA) 和 py3.10+torch2(感知) 不能同进程**。

### 5.3 推荐架构（待新会话落地）
**双进程 + IPC 解耦**：
- **进程 A（py3.7，CPU）**：跑 CARLA 客户端 + leaderboard/scenario_runner + 车辆控制 I/O。用已传的 v2xverse py3.7 环境 `/data/jichengzhi_v2x/envs/v2xverse`（注意：它的 torch1.10 **只能 CPU 用**，GPU 跑不了 sm90）。
- **进程 B（py3.10+torch2，GPU）**：跑 Pyramid 感知推理（用 §二的 t2lib 环境）。
- **A↔B 通信**：传感器数据→检测结果，走共享内存 / socket / ZMQ。
- 关键设计点：**codriving 规划器若也用 GPU torch，要一并放进程 B**；进程 A 尽量只留 CARLA I/O。需先读 V2Xverse agent 代码确认 perception/planning 的耦合度。

### 5.4 相关文件（仿真侧，都在本地 `${V2X_ROOT}verse`，尚未传 H800）
- 闭环入口/脚本：`scripts/closedloop_sweep_v1.sh`、`scripts/eval_driving_e2e.sh`、`simulation/leaderboard/scripts/eval_pnp_short_collab.sh`
- agent 代码：`simulation/leaderboard/team_code/`（perception + planner + PID 控制，是要拆的核心）
- agent 配置：`simulation/leaderboard/team_code/agent_config/pnp_config_*.yaml`（含 `perception_model_dir` / `planner_model_checkpoint`）
- vendored harness：`simulation/leaderboard/`、`simulation/scenario_runner/`（0.9.10 API）
- 本会话的仿真设计/时延策略文档：`multi_agent/real_test/sim_test_design_v1.md`、`sim_ego_rsu_latency_strategy_v1.md`、`v2xverse_framework_analysis_v1.md`

### 5.5 新会话建议的推进顺序
1. 把 V2Xverse repo（排除大数据）传 H800，在 py3.7 env 下先跑通 **纯 CARLA 客户端连通**（连上 §5.1 起的 server，能 tick/拿传感器）。
2. 读 `team_code` agent，定 perception/planning 与 CARLA-client 的拆分边界。
3. 实现 A↔B IPC（先用最简单的本机 socket 传 BEV/点云→检测框）。
4. 接 §二的 torch2 Pyramid 感知到进程 B。
5. 跑通单条 route 闭环 smoke → 再上时延感知回灌（见 `sim_ego_rsu_latency_strategy_v1.md`）。
6. ⚠️ 算力隔离：CARLA 渲染用一张卡（如 GPU6），感知推理用另一张（GPU5），避免抢占污染时延测量。

---

## 六、命令速查（cheat-sheet）

```bash
# 连服务器
ssh -p 30001 -o StrictHostKeyChecking=accept-new ${V2X_REMOTE_USER}@<PRIVATE_HOST>
# 经代理装包（永远 --target，永远 --proxy）
/usr/bin/pip3 install --proxy http://<PRIVATE_HOST>:7890 --target=/data/jichengzhi_v2x/t2lib <pkg>
# 跑 HEAL（torch2）
cd /data/jichengzhi_v2x/HEAL && CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/jichengzhi_v2x/t2lib:/data/jichengzhi_v2x/HEAL python3 <script>
# 长任务（必须 detached + 轮询）
... nohup python3 -u xxx.py > /data/jichengzhi_v2x/xxx.log 2>&1 &
# 本地→H800 传文件
rsync -e "ssh -p 30001 -o StrictHostKeyChecking=accept-new" -a <src> ${V2X_REMOTE_USER}@<PRIVATE_HOST>:/data/jichengzhi_v2x/<dst>
# 起 CARLA(无头, GPU6)
cd /data/jichengzhi_v2x/carla && CUDA_VISIBLE_DEVICES=6 ./CarlaUE4.sh -prefer-nvidia -opengl -RenderOffScreen -graphicsadapter=6 -world-port=3000 -nosound -quality-level=Low
```

## 七、未决/提醒
- `/data` 仅剩 ~124G 且共享，传大数据前先看空间。
- 真实 finetune（剪枝+1-2 epoch）尚未跑（只验证了能跑）；跑前先配正确补全标注（§四警告）以保 AP 可信。
- v2xverse py3.7 env 的 torch1.10 在 sm90 上**仅 CPU**可用——仿真进程 A 的感知/规划如需 GPU 必走 torch2 进程 B。
