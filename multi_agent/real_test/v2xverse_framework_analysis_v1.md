# V2Xverse 框架分析 — 算法支持 / 整合机制 / HEAL 移植 / 时延-能耗感知改造 v1

> 编制: team-lead (Claude), 2026-06-06
> 调研对象: /home/jichengzhi/V2Xverse (TPAMI2025, CoDriving 官方实现)
> 方法: 3 个并行 Explore agent 全库扫描, 所有结论带文件路径可核
> 注意: 区分"README 声称支持"与"代码实际存在", 下文均以代码为准

---

## 一、支持的车路协同（协同感知）算法

代码实际支持的比 README 声称的多：

| 方法 | 状态 | 关键文件 |
|------|------|---------|
| Single（无协同基线） | ✅ 完整 | `opencood/models/point_pillar.py` / `center_point.py` |
| Early Fusion（点云级） | ✅ 完整 | `hypes_yaml/v2xverse/early_fusion_multiclass_config.yaml` |
| Late Fusion（检测框级） | ✅ 完整 | `fuse_modules/max_fuse.py` |
| F-Cooper [SEC2019] | ✅ 完整 | `fuse_modules/f_cooper_fuse.py` |
| V2VNet [ECCV2020] | ✅ 完整 | `fuse_modules/v2v_fuse.py` |
| V2X-ViT [ECCV2022] | ✅ 完整 | `fuse_modules/v2xvit_basic.py` |
| DiscoNet | ✅ 完整（README 未勾选但代码已有） | `models/point_pillar_disconet.py` |
| Where2comm / When2comm | ✅ 完整（README 未列出） | `comm_modules/where2comm.py` |
| CoDriving（本项目自研, 多尺度注意力融合） | ✅ 完整 | `models/center_point_codriving.py` + `fuse_modules/codriving_attn.py` |
| HEAL [ICLR2024] | ⚠️ 只有少量 OPV2V 配置残留, **无模型代码** | `hypes_yaml/opv2v/HEAL/`（仅 yaml） |

## 二、支持的自动驾驶算法

- **CoDriving（核心, 自研）**: `codriving/models/planning_end2end.py` 的 `WaypointPlanner_e2e`（MotionNet 骨干, BEV occupancy → 未来 waypoints），配合 `simulation/leaderboard/team_code/pnp_agent_e2e.py` 闭环驾驶。✅ 完整。
- **SOTA 单车 e2e 方法**: Transfuser / Interfuser / TCP / LAV / WOR 的 agent 框架都在 `simulation/leaderboard/team_code/*_agent.py`，但 ⚠️ 模型代码依赖作者内部路径（hardcode 的 `sys.path.append('/GPFS/data/...')`），**开箱不可用**，需自行接入官方权重和代码。

## 三、自动驾驶与车路协同的整合机制

整合发生在闭环驾驶的 **PnP Agent**（`pnp_agent_e2e.py`），架构是"**两段式、可插拔**"：

```
CARLA 传感器 (多车 + RSU)
   │ tick(): RGB×4 + LiDAR + GPS/IMU      pnp_agent_e2e.py:301-399
   ▼
协同感知 (opencood 模型, 可换 fusion 方法)   pnp_infer_action_e2e.py:518-550
   │ 多 agent 数据 collate → intermediate fusion 推理 → 3D 检测框
   ▼
检测框 → BEV occupancy map (192×96)        pnp_infer_action_e2e.py:580-622
   │ + 5 帧时序 memory bank
   ▼
WaypointPlanner (codriving 模型)           → 未来 10 个 waypoints
   ▼
V2X_Controller (PID) → throttle/brake/steer
```

关键设计：

1. **配置即切换**: `team_code/agent_config/pnp_config_*.yaml` 里只改 `perception_model_dir` 指向不同 fusion 方法的 checkpoint（early/late/fcooper/v2xvit/codriving），planner 不变 → 协同感知方法是即插即用的"插件"。
2. **分开训练, 推理时组合**: 感知用 `opencood/tools/train.py` 训，规划用 `codriving/tools/train_end2end.py` 训（感知特征冻结），**没有联合端到端训练**。
3. **通信是内存共享 + 可选帧级延迟（默认零延迟）**: 车与 RSU 间没有真实网络栈，协作方**原始传感器数据**直接拼进 ego 的感知 batch（融合网络含 RSU 编码器都在 ego 一次 forward 跑完）；唯一的通信建模是可选的 `comm_latency` 字段（`pnp_infer_action_e2e.py:494-516`，单常数套给全部协作方、ego 保持最新），但**全仓库 18 个官方配置均未设置该字段 → 默认 0 通信延迟**。无带宽、丢包、抖动建模，协作方本地编码耗时不存在。
4. **RSU 仿真**: `simulation/leaderboard/leaderboard/sensors/fixed_sensors.py` 的 `RoadSideUnit` 类（lines 783-1135），部署在车前方 12m 路侧 7.5m 高处，带 LiDAR + 多相机。

## 四、HEAL 移植评估

**结论: 中等复杂度，约 1800 LOC / 10 个文件，2-3 周（熟悉两库的人）。** 两边同源（HEAL 本身基于 OpenCOOD），且 V2Xverse 已内置异构基础设施。

### 已有可复用的基础

| 组件 | 位置 |
|------|------|
| 异构 dataset（m1/m2/m3 多模态 agent 分配） | `intermediate_heter_fusion_dataset.py` + `utils/heter_utils.py` |
| 7 种特征对齐网络（SCAligner/CBAM/FANet 等） | `models/sub_modules/feature_alignnet.py` |
| 两阶段训练钩子（冻结全部、只解冻新 modality 的 aligner） | `heter_model_sharedhead.py` 的 `model_train_init()`，`train.py:109` 已调用 |
| 非严格 checkpoint 加载（支持部分加载） | `train_utils.py`（`strict=False`） |

### 缺失的核心工作量

| 任务 | 工作量 |
|------|--------|
| **Pyramid Fusion backbone**（HEAL 核心: 多尺度金字塔融合 + 深度监督）→ 新建 `pyramid_fusion_backbone.py` + `pyramid_modules.py` + `fuse_modules/pyramid_fusion.py` | ~1200 LOC（主要工作） |
| encoder 改为输出多尺度中间特征（而非只输出最终层） | ~150 LOC |
| 两阶段训练打通（stage1 训 collaboration base → stage2 冻结只训新 agent aligner），YAML 加 `training_stage`/`stage2_added_modality` | ~100 LOC |
| loss 支持金字塔各层级监督（per-scale foreground supervision） | ~150 LOC |
| 新配置 `heal_multiclass_config.yaml` + 适配 V2Xverse 数据集（多类别、车+RSU） | ~200 LOC |

### 实操注意点

1. **最省力路径**: 直接从 HEAL 官方仓库 (github.com/yifanlu0227/HEAL) 移植 `pyramid_fuse.py` 等文件——两边 OpenCOOD 结构高度相似（importlib 动态加载、`record_len`/`pairwise_t_matrix` 约定一致）。主要适配点是 V2Xverse 的**多类别检测头**（multiclass）和 **V2XVERSEBaseDataset** 的 collate 格式。
2. **模型注册机制**: V2Xverse 用 importlib 动态加载（`train_utils.py:142-175`），新模型必须是独立 py 文件且类名 = 文件名去下划线，无装饰器注册。
3. **闭环接入**: 要在闭环驾驶里用 HEAL，需在 `pnp_infer_action_e2e.py` 确认 HEAL 输出的 fused feature 能喂给 planner 的 memory bank（128 通道 BEV feature 约定）——改动小。

## 五、时延感知、能耗感知的仿真改造方案

现状: 通信只有固定帧延迟，**计算延迟和能耗完全没有建模**。

### (a)(b) 时延感知 → 已迁移至独立设计文档

通信时延（链路级模型）与计算时延（推理耗时折算进闭环）的完整设计已整合进闭环仿真测试主文档
**[sim_test_design_v1.md](./sim_test_design_v1.md)** §3（理想愿景 + 实装形态; 完整 ~600 LOC 改造清单见归档 `../archive/latency_aware_simulation_design_v1.md`），要点:

- 统一**时间轴**而非单一延迟标量: τ_comm 注入融合输入新鲜度（per-link data bank），τ_compute 注入控制生效时间（LatencyScheduler 作业模型），统一量为逐路径信息年龄;
- 延迟期间控制保持三档策略: L0 指令零阶保持（现状 `pnp_agent_e2e.py:448-450`）/ L1 轨迹跟踪保持（规划-控制分频 + 时间索引取点, 推荐）/ L2 安全看门狗;
- 评测核心产出: "V2X 增益随延迟衰减曲线"，直接回答 "<200ms 假设"。

### (c) 能耗感知: 新增 energy ledger（新模块, 逻辑简单, 难点在功率标定）

完全空白，建议新建 `simulation/leaderboard/team_code/energy_meter.py`：

- **计算能耗**: 每次推理记 `E = P_avg × t_infer`，P 取目标平台实测功率（Orin 上用 tegrastats/INA3221 标定离线功率表，按模型/分辨率/精度 FP16/INT8 查表），或退化为 FLOPs × J/FLOP 估算。
- **通信能耗**: `E = 传输字节数 × J/bit`（按 C-V2X 射频能耗参数）。Where2comm 这类带通信压缩的方法能在能耗维度与 V2X-ViT 拉开差距。
- **输出**: 在 leaderboard 结果统计（`leaderboard/utils/statistics_manager.py` 一类）加 per-route 能耗/平均时延列，形成 "driving score vs 能耗 vs 时延" 的 Pareto 评测。

### 优先级建议

| 改造 | 改动量 | 价值 |
|------|--------|------|
| (a)(b) 时延感知（详见 sim_test_design_v1.md §3） | 中 (~600 LOC) | delay-robustness 实验 + 与 Orin 真测延迟表联动, 闭环验证 "<200ms 假设" |
| (c) 能耗 ledger | 中（标定是难点） | 形成三维 Pareto 评测, 论文叙事差异化 |

---

## 附: 调研中的关键文件索引

| 组件 | 路径 |
|------|------|
| 闭环 agent 入口 | `simulation/leaderboard/team_code/pnp_agent_e2e.py` (setup: 89-161) |
| 核心推理管线 | `simulation/leaderboard/team_code/pnp_infer_action_e2e.py` (get_action_from_list_inter: 477-657) |
| 通信延迟实现 | `pnp_infer_action_e2e.py:494-516` |
| RSU 仿真 | `simulation/leaderboard/leaderboard/sensors/fixed_sensors.py:783-1135` |
| 感知训练入口 | `opencood/tools/train.py` |
| 规划训练入口 | `codriving/tools/train_end2end.py` |
| fusion 方法切换配置 | `simulation/leaderboard/team_code/agent_config/pnp_config_*.yaml` |
| 异构 agent 支持 | `opencood/data_utils/datasets/intermediate_fusion_dataset.py`, `models/heter_model_sharedhead.py` |
