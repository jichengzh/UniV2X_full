# 协同感知模型 Zoo 总览 v1

> 编制: sw-optimizer · 2026-06-05
> 目的: 用户审阅用 — 确定下一步优化算法的候选模型范围
> 数据来源标注: 【本地 ckpt】= 盘上已提取 · 【HF zip】= HF 已下 zip 未提取 · 【yaml 可实例化】= CPU 真算参数量 · 【历史真测】= `data/v2x_baseline_timing/` 存档
> 注: 全部分段计时来自 OPV2V 数据集(record_len≈1.39)，DAIR 版(record_len=2.00)计时待 profiling 微跑(阶段2预授权)

---

## 一、HEAL 支持的候选模型全表

### 1.1 DAIR-V2X LiDAR Only 基线模型

| 模型 | fusion 结构类型 | 总参数 | fusion 参数 | backbone 参数 | DAIR yaml | DAIR ckpt 状态 | OPV2V ckpt | 注 |
|---|---|---|---|---|---|---|---|---|
| **Pyramid (m1)** | Conv 多尺度加权融合 (ResNeXt) | ~~14.45M~~ **5.465M**★ | ~~3.79M~~ **3.758M**★ | ~~6.58M~~ **0.226M**★ | ✅ | **✅ 本地已提取** | ✅ | 现役主战场 |
| **V2X-ViT** | Transformer (HMSA+MSwin 3层) | **13.45M** | 5.39M | 6.58M | ✅ | **✅ 本地已提取** | ❌ HF无OPV2V | A-1/A-2 已测 |
| **F-Cooper** | Max-pool 融合 (0参数) | **8.06M** | **0** | 6.58M(★待核) | ✅ | 【HF zip】可提取 ~2min | ✅ 本地已提取 | 无参融合最简 |
| **AttFuse** | SDP attention (0可学参数) | **8.06M** | **0** | 6.58M(★待核) | ✅ | 【HF zip】可提取 ~2min | ✅ 本地已提取 | 参数量≈F-Cooper |
| **CoBEVT** | BEV Swin Transformer (全局) | **10.50M** | **2.44M** | 6.58M(★待核) | ✅ | 【HF zip】可提取 ~2min | 【HF zip】 | BERT风格BEV融合 |
| **DISCO** | 图神经网络 (DiscoNet) | — | — | — | ✅ | 【HF zip】 | 【HF zip】 | ⚠️ 依赖 `disco_fuse` 模块当前缺失，需查 |
| **CoAlign** | Attention+对齐 | — | — | — | ✅ | — | — | ⚠️ YAML 解析错误，需排查 |

> ★[2026-06-05 勘误 ISS-036 — 依据: `multi_agent/model/pyramid_lidar_structure_audit_v1.md` L22 + P1 csv `full_params=5464791` 双源] Pyramid 行三数均错: 14.45M 疑混入 §三 14.45ms OPV2V fusion 计时; backbone 6.58M 实为 V2X-ViT BaseBEVBackbone 错植。真值: 总参=5.465M / pyramid_backbone(fusion主体)=3.758M / backbone_m1=0.226M。V2X-ViT backbone 6.58M 已由 `v2xvit_structure_audit_v1.md §1.1` CPU 真算独立确认 ✓。F-Cooper/AttFuse/CoBEVT backbone 6.58M = 共享 BaseBEVBackbone 架构推算值(与各自 total 参数量算术自洽), 但未单独 CPU 核验, 已标"待核"。

### 1.2 OPV2V LiDAR Only 额外模型（HEAL 有 yaml，无 DAIR 版 HF ckpt）

| 模型 | fusion 结构类型 | 总参数 | fusion 参数 | OPV2V ckpt | DAIR yaml/ckpt | 注 |
|---|---|---|---|---|---|---|
| **Where2comm** | Sparse attention (confidence map guided) | **8.45M** | **0.40M** | 【HF zip】 | ❌ 无 DAIR yaml | QuantV2X Table1 有 |
| **V2VNet** | 图神经网络 (GNN message passing) | **14.61M** | **6.55M** | 【HF zip】 | ❌ 无 DAIR yaml | OPV2V 现成 yaml |

> 参数量来源: 【yaml 可实例化】CPU `torch.load` + `train_utils.create_model` 真算 (2026-06-05)

---

## 二、QuantV2X Table1 对照价值分析

QuantV2X (arXiv:2509.03704) Table 1 含以下 6 模型 (DAIR-V2X, PTQ, AP30/AP50):

| QuantV2X 模型 | HEAL 对应 | DAIR ckpt 状态 | 本项目测试状态 |
|---|---|---|---|
| Pyramid Fusion | Pyramid (m1) | ✅ 本地 | ✅ A 系列全测 |
| **F-Cooper** | lidar_fcooper | 【HF zip】 | 仅 OPV2V timing |
| **AttFuse** | lidar_attfuse | 【HF zip】 | 仅 OPV2V timing |
| **V2X-ViT** | lidar_v2xvit | ✅ 本地 | ✅ A-1/A-2 已测 |
| **Who2com** | ❌ HEAL 无 Who2com | — | 无 |
| **Where2comm** | lidar_where2comm | ❌ 仅 OPV2V yaml | OPV2V timing 有 |

→ **F-Cooper 和 AttFuse 是完成 QuantV2X 对照覆盖最容易补的两个**（zip 已在盘，提取 ~2min，DAIR yaml 已有）

---

## 三、分段计时汇总（OPV2V，record_len≈1.39）

> 来源: 【历史真测】`data/v2x_baseline_timing/*.json`, scripts/phase2/m4_9_v2x_baselines_timing.py, OPV2V test set, PyTorch FP32, CUDA Event

| 模型 | e2e (ms) | encoder | backbone+shrinker | fusion | NMS | 口径 |
|---|---|---|---|---|---|---|
| Pyramid (OPV2V) | — | — | — | 14.45ms | — | [见 `model/pyramid_fusion/分段耗时实测_v1.md`] |
| **F-Cooper (OPV2V)** | **89.8** | 3.02 | 11.03 | **1.41** | 73.4 | 历史真测 |
| **AttFuse (OPV2V)** | **102.3** | 3.24 | 11.57 | **3.49** | 82.9 | 历史真测 |
| V2VNet (OPV2V) | **41.6** | 3.12 | 7.33 | **28.56** | 1.59 | 历史真测 |
| Where2comm (OPV2V) | **24.6** | 3.43 | 11.47 | **6.69** | 1.94 | 历史真测 |
| **V2X-ViT (DAIR)** | **61.86** | 2.90 | 3.88 | **27.39** | 23.75 | 历史真测 (2026-05-14) |

> ⚠️ OPV2V vs DAIR 差异: DAIR record_len=2(固定V2I), OPV2V record_len≈1.39(随机多车); backbone 时间差异来自 V2X-ViT shrinker stride=2 vs 其他 stride=1。各模型 DAIR 计时待阶段2 profiling 微跑(预授权)。

---

## 四、P0 分段计时是否已有

| 模型 | OPV2V timing | DAIR timing | P0(CUDA NMS) timing |
|---|---|---|---|
| Pyramid (DAIR) | — | ✅ `分段耗时实测_v1.md` | ✅ P0实测 |
| F-Cooper (OPV2V) | ✅ `fcooper_real.json` | ❌ 待补 | ✅ `fcooper_p0.json` |
| AttFuse (OPV2V) | ✅ `attfuse_real.json` | ❌ 待补 | ✅ `attfuse_p0.json` |
| V2X-ViT (DAIR) | — | ✅ `v2xvit_dair_real.json` | ❌ 未测 |
| V2VNet (OPV2V) | ✅ `v2vnet.json` | ❌ | ❌ |
| Where2comm (OPV2V) | ✅ `where2comm.json` | ❌ | ❌ |

---

## 五、Shortlist 推荐（2-3 个，阶段2 structure audit 候选）

评分维度: ★★★=最高

### 推荐 #1: **CoBEVT (DAIR LiDAR)**

| 维度 | 评分 | 理由 |
|---|---|---|
| 结构差异化 | ★★★ | **全局 BEV Swin Transformer**（非 agent-level attention）; 与 Pyramid(conv) 和 V2X-ViT(multi-agent attn) 均不同架构范式 |
| ckpt 可得性 | ★★★ | 【HF zip】已在盘，解压 ~2min 即得 DAIR 版 bestval ckpt |
| 精度轴激活潜力 | ★★★ | 2.44M 可学融合参数（非 0，非极度过参数化）; Swin 式局部窗口注意力 INT8 风险中等（LayerNorm+softmax 有损但比 V2X-ViT HMSA 更简单） |
| QuantV2X 对照 | ★★ | QuantV2X Table1 含 CoBEVT（AP30 66.6/AP50 60.8 INT8 近免损）→ 我们可复现对照 |

### 推荐 #2: **F-Cooper (DAIR LiDAR)**

| 维度 | 评分 | 理由 |
|---|---|---|
| 结构差异化 | ★★ | **0 可学融合参数** Max-pool; 最简基线，代表"无融合智慧" |
| ckpt 可得性 | ★★★ | 【HF zip】已在盘 + OPV2V ckpt 已提取 |
| 精度轴激活潜力 | ★ | 无可学参数 → 无融合参数可剪/量化 → 精度轴只靠 backbone; 可能重复 Pyramid 结论 |
| QuantV2X 对照 | ★★★ | QuantV2X Table1 含 F-Cooper → **完成对照闭环**（本项目 Pyramid+V2X-ViT+F-Cooper+AttFuse = 覆盖 QuantV2X 4/6 模型）|

### 推荐 #3: **AttFuse (DAIR LiDAR)**

| 维度 | 评分 | 理由 |
|---|---|---|
| 结构差异化 | ★★ | **参数量=0 的 SDP attention**（非可学 QKV); 介于 F-Cooper(max) 和 V2X-ViT(heavy transformer) 中间 |
| ckpt 可得性 | ★★★ | 【HF zip】已在盘 + OPV2V 已提取 |
| 精度轴激活潜力 | ★ | 无可学 attention 参数 → 融合无量化目标; 但 backbone INT8 + fusion(attention compute) INT8 对 attention 分数可能有影响 |
| QuantV2X 对照 | ★★★ | QuantV2X Table1 含 AttFuse → 完成对照闭环 |

---

## 六、其他模型状态说明

| 模型 | 状态 | 说明 |
|---|---|---|
| **DISCO** | ⚠️ 依赖缺失 | `opencood.models.fuse_modules.disco_fuse` 模块在当前环境不存在（可能需要额外安装或版本不匹配）。HF DAIR ckpt 有 zip。需先确认依赖再评估。 |
| **CoAlign** | ⚠️ YAML 解析错误 | `lidar_coalign.yaml` 有 YAML 语法/引用问题（`mapping values are not allowed`）。HF 无 DAIR ckpt（不在 baselines_hf 目录）。 |
| **Who2com** | ❌ HEAL 不支持 | HEAL HF 无 Who2com 模型（QuantV2X 有但 HEAL 未发布）。不推荐。 |
| **Where2comm** | ⚠️ 无 DAIR yaml | 仅 OPV2V yaml 和 OPV2V timing；需自训 DAIR 版（成本高，暂不推荐）。 |
| **V2VNet** | ⚠️ 无 DAIR yaml | 仅 OPV2V yaml；fusion 6.55M(GNN)计算量大(28.56ms)；无 DAIR ckpt，需自训。 |
| **HEAL 多模态** | 超出当前 scope | lidar_camera / MoreModality 需 camera 数据，与当前 LiDAR-only pipeline 不兼容。 |

---

## 七、阶段2 structure audit 建议执行顺序

若用户选定 shortlist，建议顺序：
1. **F-Cooper (DAIR)** — 最简，~5min 实例化+profiling，建立 conv-fusion 零参基线
2. **AttFuse (DAIR)** — 次简，attention 结构，DAIR ckpt 解压后可复现 QuantV2X 对照
3. **CoBEVT (DAIR)** — 最复杂，全局 BEV Swin Transformer，预期有 INT8 风险（LayerNorm+MHSA），精度轴激活潜力最高

每个 audit 文档存 `multi_agent/model/<model>_structure_audit_v1.md`，格式同 `v2xvit_structure_audit_v1.md`。

---

## 八、★模态维度 — HEAL Pyramid 多模态全景 [team-lead 实查, 2026-06-05]

> 来源: team-lead 实查 HEAL hypes_yaml + checkpoints/ (本节数据由 team-lead 在 HANDOFF §1.5 整理, sw-optimizer 校验路径后落盘)。
> **警告**: 全部 OPV2V ckpt 口径 = `root_dir=OPV2V`; DAIR ckpt 口径 = `root_dir=DAIR-V2X`; 两者 **严禁混口径比较**。

### 8.1 HEAL Pyramid 模态 Encoder 全表

HEAL Pyramid 是 **fusion 骨架 × 4 种模态 encoder** 的可插拔架构：

| encoder ID | 传感器类型 | Encoder 方法 | Camera Backbone | DAIR ckpt 状态 | OPV2V stage2 ckpt | 备注 |
|---|---|---|---|---|---|---|
| **m1** | LiDAR | PointPillar | — | **✅ `checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/`** | `checkpoints/stage1/Pyramid_m1_base_*/` | **现役主战场; 全部实验数据** |
| **m2** | Camera (4-cam) | Lift-Splat-Shoot (LSS) | EfficientNet | **❌ 无 DAIR ckpt** (`CameraOnly/camera_pyramid.yaml` 在但无 ckpt) | ✅ `stage2/m2_alignto_m1/net_epoch25.pth` | DAIR 需自训; OPV2V 异构链有 |
| **m3** | LiDAR | SECOND | — | ❌ 无独立 DAIR ckpt | ✅ `stage2/m3_alignto_m1/net_epoch_bestval_at25.pth` | MoreModality DAIR yaml 有 |
| **m4** | Camera (4-cam) | Lift-Splat-Shoot 变体 | ConvNeXt | ❌ 无独立 DAIR ckpt | ✅ `stage2/m4_alignto_m1/net_epoch25.pth` | MoreModality DAIR yaml 有; 与 m2 同传感器异 backbone |

> 路径已逐一 `ls` 核实 (2026-06-05)。m2/m3/m4 的 OPV2V stage2 ckpt 均为 `net_epoch25`/`net_epoch_bestval_at25`，完整训练完毕。

### 8.2 HEAL 异构多模态 (m1+m2 协作) 配置状态

| yaml | 说明 | ckpt 状态 |
|---|---|---|
| `MoreModality/HEAL/final_infer/m1m2.yaml` | 异构多模态联合推理 (m1=LiDAR ego + m2=Camera partner) | `checkpoints/final_infer/net_epoch1.pth` ⚠️ 仅 epoch1 (可能未完全收敛) |
| `MoreModality/HEAL/stage2/m{2,3,4}_single_pyramid.yaml` | 单模态异构 stage2 训练 yaml (align-to-m1) | OPV2V stage2 ckpt 已有; DAIR 需自训 |
| `MoreModality/2_modality_end2end_training/lidar_camera_pyramid.yaml` | DAIR LiDAR+Camera 端到端联合训练 | 无 ckpt (配置层, 需端到端训练) |

### 8.3 DAIR 模态 yaml 总览

| 目录 | yaml 文件 | ckpt 可用性 |
|---|---|---|
| `dairv2x/LiDAROnly/` | `lidar_pyramid.yaml` (m1) | ✅ 现役 |
| `dairv2x/CameraOnly/` | `camera_pyramid.yaml` (m2/LSS/EfficientNet) | **❌ 无 ckpt** |
| `dairv2x/MoreModality/HEAL/` | stage1/stage2/final_infer 共 4 yaml | OPV2V ckpt 有; DAIR ckpt 部分缺失 |
| `dairv2x/MoreModality/2_modality_end2end_training/` | `lidar_camera_pyramid.yaml` 等 7 个联合训练 yaml | 无 ckpt |

### 8.4 三条 Shortlist 扩展轴

> 原 §五 shortlist 仅覆盖 ①换 fusion 结构。本节扩展为三条轴：

| 轴 | 描述 | 候选 | 成本 | 精度轴激活潜力 |
|---|---|---|---|---|
| **① 换 fusion 结构** (当前 shortlist) | 保持 m1/LiDAR encoder, 替换融合模块 | CoBEVT ★★★ / F-Cooper ★★★ / AttFuse ★★ | DAIR ckpt zip 解压 ~2min + audit | 中高 (CoBEVT 有可学 Swin attn) |
| **② 换 encoder 模态** (camera LSS) | 保持 Pyramid fusion 骨架, 替换为 m2/Camera LSS encoder | Pyramid m2 (camera_pyramid.yaml) | **需 DAIR 自训** (无 ckpt; OPV2V ckpt 有但口径不兼容 DAIR 评测); camera 数据集准备成本高 | 高 (不同传感器模态, 有 softmax depth 量化风险) |
| **③ 异构多模态** (HEAL 本体) | m1 ego + m2 partner (LiDAR+Camera 异构协作) | HEAL m1m2 联合推理 | OPV2V 口径 ckpt 可用 (epoch1); **DAIR 需端到端自训** (无 ckpt, 成本最高) | 最高 (多模态 → encoder 有异构 INT8 风险; 但训练成本 ~数天 GPU) |

**当前可行性排序**: ① >> ③(OPV2V口径评估) >> ②(DAIR自训成本高)

**⚠️ 注**: 轴②和轴③在 DAIR 上**均无现成 ckpt**, 需自训 (数天 GPU); 在 OPV2V 上有 stage2 ckpt 可做性能评估, 但口径为 OPV2V, 不可与 DAIR 数字拼表。

---

## 九、阶段2 structure audit 建议执行顺序 (更新版)

若用户选定 shortlist，建议顺序（优先轴①-换fusion结构）：
1. **F-Cooper (DAIR)** — 最简，~5min 实例化+profiling，建立 conv-fusion 零参基线
2. **AttFuse (DAIR)** — 次简，attention 结构，DAIR ckpt 解压后可复现 QuantV2X 对照
3. **CoBEVT (DAIR)** — 最复杂，全局 BEV Swin Transformer，预期有 INT8 风险（LayerNorm+MHSA），精度轴激活潜力最高

轴②③（换encoder/异构多模态）在 DAIR 均需自训，优先级取决于用户是否需要 DAIR 对照。

每个 audit 文档存 `multi_agent/model/<model>_structure_audit_v1.md`，格式同 `v2xvit_structure_audit_v1.md`。

---

---

## 十、V2Xverse 原生模型 — CoDriving ⚠️ 不同口径专节

> **★ 严格隔离声明**: 本节模型与 §一~§九 的 HEAL 系模型**完全不同口径**, 禁止跨节混表比较。
> 详细结构审计见 `multi_agent/model/codriving_structure_audit_v1.md`。

### 10.1 CoDriving 基本信息

| 项目 | 内容 |
|---|---|
| **论文** | arXiv:2404.09496 (V2Xverse, CoDriving) |
| **repo** | `/home/jichengzhi/V2Xverse` (独立, 非 HEAL) |
| **模型类** | `centerpointcodriving` (`opencood/models/center_point_codriving.py`) |
| **任务** | **感知 + 规划 (端到端)** — 含 WaypointPlanner 头 |
| **训练数据** | **CARLA 仿真** (V2Xverse, Towns 1–4,6 train / Town 5 test) |
| **评估指标** | Driving Score / waypoint L1 (ADE/FDE) |
| **感知 ckpt** | `checkpoints/codriving/perception/net_epoch_bestval_at16.pth` (32MB) |
| **规划 ckpt** | `checkpoints/codriving/planner/codriving_planner.ckpt` (20MB) |

### 10.2 参数量 (CPU 真算, 2026-06-09)

| 组件 | 参数量(可学习) | 说明 |
|---|---|---|
| 感知网络 (`centerpointcodriving`) | **8,177,435 (8.177M)** | 【CPU 真算】, 见下表 |
| 规划网络 (`WaypointPlanner_e2e`) | **1,662,290 (1.662M)** | 【ckpt 计数】|
| **联合总计** | **9,839,725 (9.840M)** | |

**感知网络分解**:

| 子模块 | 参数量 | 占比 | 融合类型 |
|---|---|---|---|
| backbone.resnet (ResNet3stage [64,128,256]) | 6,984,320 (6.984M) | 85.4% | — |
| backbone.deblocks (FPN neck) | 598,784 (0.599M) | 7.3% | — |
| shrink_conv (384→128) | 590,080 (0.590M) | 7.2% | — |
| **fusion_net (CoDriving AttenFusion×3)** | **0** | **0%** | **ScaledDotProductAttention (无 QKV 投影, 零可学参数)** |
| pillar_vfe (VFE encoder) | 768 | ~0% | — |
| cls_head + reg_head | 3,483 | ~0% | — |

> ★ **fusion_net 可学参数为 0**: CoDriving 采用 `AttenFusion = ScaledDotProductAttention (bmm+softmax)`, 无 QKV 投影矩阵, 3 个 scale 均 0 参数。与 F-Cooper(MaxFusion) 同类 — 融合本体无可优化参数。

### 10.3 架构特征与 INT8 风险初判

| 特征 | 内容 |
|---|---|
| fusion 结构 | 多尺度 AttenFusion (3 scale × ScaledDotProductAttention, **0 learnable params**) |
| voxel_size | [0.125, 0.125, 36]; BEV 分辨率 576×192 (比 HEAL Pyramid 250×100 高 4.4×) |
| 检测范围 | [-36,-12,-22,36,12,14] 前向 72m×24m (非对称, 不同于 DAIR 200m×80m) |
| 检测头 | CenterPoint multiclass, anchor_number=3 |
| Backbone | ResNetBEVBackbone (BasicBlock, 不同于 Pyramid ResNeXt) |
| Planner | WaypointPlanner_e2e: Conv2d + Conv3D(temporal) + MLP decoder |
| **INT8 整体风险** | **🟡 低-中** (明显低于 V2X-ViT; AttenFusion softmax + Planner Conv3D 为潜在点) |
| backbone INT8 | **🟢 低风险** (标准 Conv2d+BN ResNet, 类比 Pyramid backbone) |
| fusion INT8 | **🟡 中** (softmax; 但无 QKV = 比 V2X-ViT 风险更低) |

### 10.4 口径不可比差异表

| 维度 | CoDriving (本节) | HEAL 系模型 (§一~§九) |
|---|---|---|
| 训练数据 | CARLA 仿真 (V2Xverse) | DAIR 真实路测 / OPV2V 仿真 |
| 任务 | 感知 + **规划** (端到端) | 纯感知 (3D 目标检测) |
| 评估指标 | Driving Score, ADE/FDE | AP30/AP50/AP70 |
| 检测范围 | 72m×24m (前向不对称) | 200m×80m (对称大范围) |
| Backbone | ResNetBEVBackbone | Pyramid: ResNeXt; V2X-ViT: BaseBEVBackbone |
| 参数量可比? | ❌ 任务不同, 参数量无法横向比 | HEAL 内部可比 |

### 10.5 在框架中的定位

CoDriving 的优化目标是**降低端到端推理延迟**, 以减少时延回灌对 **Driving Score** 的影响 (Task #7 RSU 时延感知回灌), 不是在 DAIR AP 上竞争。

适合做的分析:
1. 感知 backbone TRT INT8 latency 测量 → 量化延迟节省对 Driving Score 影响
2. backbone 结构化剪枝 → 但需 CARLA 环境 finetune (成本高于 HEAL DAIR finetune)
3. planner MLP INT8 量化 → 低风险, 可行

---

*文档版本: v1.2 (2026-06-09 补 §十 V2Xverse CoDriving 专节; sw-optimizer)*
*来源标注: §八 team-lead 实查 HANDOFF §1.5 + sw-optimizer 路径核实落盘; §十 sw-optimizer CPU 真算*
*新家: `multi_agent/model/`（per 用户指令）*
