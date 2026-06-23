# HEAL(PyramidFusion) → V2Xverse 闭环移植设计 v1

> 编制: 2026-06-12, team-lead (Claude)
> 目标: 把 HEAL 的 PyramidFusion 协同感知集成进 V2Xverse CARLA 闭环, 在 V2Xverse 数据上重训, 最终产出"时延×驾驶分"曲线。
> 状态: 设计已与用户确认; 基础设施(网络/docker/数据)已就绪, 进入实施。

---

## 一、目标与范围

**完整可用闭环(properly)**: 移植 Pyramid 模型代码 → 适配 V2Xverse multiclass → **在 V2Xverse CARLA 数据上重训** → 接进 pnp 闭环 → 出"时延×驾驶分"曲线。

**为什么必须重训**: DAIR/OPV2V 训出的 Pyramid 权重与 CARLA 仿真存在域间隔(点云分布/类别/传感器布局全不同), 直接用于闭环检测几乎全废 → ego 瞎开。所以移植绕不开"在 V2Xverse 数据上重训"。

---

## 二、已验证的基础设施 (2026-06-12 实测)

### 2.1 训练机 = H800 + Docker (非宿主 env)
- **宿主 `v2xverse_env`(torch1.10/py3.7) 在 H800 Hopper(sm90) 上不可用**: 老栈第一个 CUDA 算子 200s 卡死(handoff §7.4)。
- **解法 = Docker**: 本地已有镜像 `pytorch:2.7.1-cuda11.8-cudnn9-devel`(py3.11)。实测容器内 `--gpus all`: cap(9,0) 可见, matmul 0.30s / conv2d 0.21s **无卡死**。
- **依赖实测可装**: `spconv-cu118`(2.3.8) + opencood deps + **`opencood + create_model` 在 torch2.7 下导入 OK**。
- 固化镜像: `v2xverse-train:torch27`(Dockerfile 在 `/data/jichengzhi_v2x/docker_build/`)。

### 2.2 网络: H800 经本地梯子端口转发 `127.0.0.1:7897` 有外网
- 容器内用 `--network host` + `http_proxy=http://127.0.0.1:7897` 即可 pip/下载。
- 速率 ~0.5MB/s 单流, 多并发聚合 ~1.7MB/s。

### 2.3 数据: V2Xverse 数据集 163GB
- HF `gjliu/V2Xverse` 实际仅 **weather-0, 96 routes, 163GB**, 按 route 打包 zip(不能按模态筛)。
- 下载到 H800 `/data`(1.8T 空闲), **HF_HOME 必须设到 /data**(默认 `~/.cache` 在 root 盘 132G 会撑爆)。
- 闭环 eval 用实时 CARLA, 不需要离线数据集; 离线数据只用于训练感知。

### 2.4 代码工作流
- **4090 `/home/jichengzhi/V2Xverse` 是 git 正源**(myfork, 含 L1/τ_ego 提交)。H800 副本非 git。
- 流程: 在 4090 git 仓编辑+提交 → 同步改动到 H800 `/data/jichengzhi_v2x/V2Xverse` → docker 挂载训练。

---

## 三、移植 6 阶段

| 阶段 | 内容 | 关键文件 |
|---|---|---|
| **P1 模型搬运** | HEAL `heter_pyramid_collab.py`/`heter_pyramid_single.py`/`fuse_modules/pyramid_fuse.py`/`loss/point_pillar_pyramid_loss.py` 拷进 V2Xverse opencood, 按 importlib 命名约定注册(类名=文件名去下划线) | `opencood/models/`, `opencood/loss/` |
| **P2 multiclass 适配** | HEAL 单类检测头 → V2Xverse 多类头(车+RSU 多类别), 对齐 `VoxelPostprocessor`/loss 类别维度 | 模型头 + loss |
| **P3 配置+数据适配** | 新建 `pyramid_multiclass_config.yaml`(仿 `codriving_multiclass_config.yaml`), 适配 V2XVERSEBaseDataset collate | `hypes_yaml/v2xverse/` |
| **P4 训练** | docker 内 H800 单卡(GPU6 空)train, AP 对标 codriving baseline | `opencood/tools/train.py` |
| **P5 闭环接入** | `pnp_infer_action_e2e.py` 感知插件换 Pyramid, 确认 fused feature 喂得进 planner BEV memory bank(128 通道约定) | `team_code/pnp_infer_action_e2e.py` + `pnp_config_pyramid_*.yaml` |
| **P6 验证出曲线** | 闭环跑 DS/RC, 叠加 τ_ego 时延注入 → 出"时延×驾驶分"曲线 | 复用现有 sweep 框架 |

**先 P1→P4 拿到能用的 Pyramid 感知(AP 达标), 再 P5→P6 接闭环。**

---

## 四、关键适配点与风险

1. **multiclass**: HEAL 是单类(车), V2Xverse 是多类(含 RSU/行人等)。检测头通道数、loss 类别维度、postprocessor anchor 都要对齐。**最大代码工作量**。
2. **collate 格式**: HEAL OpenCOOD 与 V2Xverse OpenCOOD 同源但 dataset collate 有差异(`record_len`/`pairwise_t_matrix` 约定需核对)。
3. **torch2 运行时 API**: import + create_model 已过, 但训练 forward/backward 可能有零星 torch1.10→2.7 deprecation 需小修。
4. **闭环 feature 约定**: planner memory bank 期望 128 通道 BEV feature; Pyramid 输出需对齐(改动小, 见框架分析 §四.3)。
5. **AP 达标线**: 目标对标 V2Xverse codriving baseline; 具体阈值待 P4 首轮训练后定。

---

## 五、当前进度 (2026-06-12)
- [x] 基础设施验证(docker/网络/数据/env)
- [~] Docker 镜像 `v2xverse-train:torch27` 构建中(后台)
- [~] 163GB 数据集下载中(后台, /data, ~12-24h)
- [ ] P1 模型搬运 — 待数据/镜像就绪后启动
- [ ] P2-P6

---

## 附: 关键路径
| 资源 | 路径 |
|---|---|
| HEAL Pyramid 源 | `/home/jichengzhi/heal_research/HEAL/opencood/models/heter_pyramid_*.py` |
| V2Xverse git 正源 | 4090 `/home/jichengzhi/V2Xverse` |
| V2Xverse 训练副本 | H800 `/data/jichengzhi_v2x/V2Xverse` |
| Docker build | H800 `/data/jichengzhi_v2x/docker_build/Dockerfile` |
| 数据集 | H800 `/data/jichengzhi_v2x/hf_cache`(HF_HOME) |
| 闭环推理入口 | `simulation/leaderboard/team_code/pnp_infer_action_e2e.py` |
