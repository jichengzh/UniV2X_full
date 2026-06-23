# CoDriving 全网结构审计 v1

> 编制: sw-optimizer
> 时间: 2026-06-09
> 目的: 用户审阅用 — CoDriving model zoo 加入 + 剪枝/量化计划制定前的全网结构与瓶颈分析
> 数据来源标注: 【CPU 真算】= `centerpointcodriving` 实例化 + `model.parameters()` 真算 · 【ckpt 计数】= `state_dict` tensor 求和(含 BN buffers) · 【yaml 推算】= 从 config.yaml 公式推导 · 【static】= 代码静态分析

> ⚠️ **口径警告 — 严禁与 HEAL DAIR/OPV2V 混表**:
> CoDriving 训练于 **CARLA 仿真数据** (V2Xverse Towns 1–4,6/7/8/10, Town 5 测试集), 含 **planner 头**(感知+规划一体), 评估指标为 **Driving Score / waypoint L1**, 与 HEAL 系 Pyramid/V2X-ViT 的 DAIR/OPV2V 真实路测 AP50/AP70 **完全不同口径**。任何 AP vs Driving Score 交叉比较均无意义, 见 §三.1。

---

## 一、全模块分解表

### 1.1 顶层模块 (感知网络 `centerpointcodriving`)

**基础信息**:
- 模型类: `centerpointcodriving` (`opencood/models/center_point_codriving.py`)
- Checkpoint: `checkpoints/codriving/perception/net_epoch_bestval_at16.pth` (32MB)
- 总可学习参数: **8,177,435** (8.177M) 【CPU 真算】
- state_dict 全量(含 BN buffers): 8,187,194 (8.187M) 【ckpt 计数】; 差值 9,759 = BN running_mean/var/num_batches_tracked

| # | 模块 | 网络类型 | 参数量(可学习) | 占比 | 可剪枝? | 可量化 INT8? |
|---|---|---|---|---|---|---|
| 1 | **pillar_vfe** (PillarVFE) | Linear+BN (稀疏) | **768** | 0.01% | ⚠️ 几乎无意义 | ❌ 不在 ONNX export scope (spconv 稀疏算子) |
| 2 | **scatter** (PointPillarScatter) | scatter op (无权重) | **0** | 0% | N/A | ❌ 不在 ONNX export scope |
| 3 | **backbone.resnet** (ResNetBEVBackbone 3-stage) | ResNet Conv2d+BN, layer_nums=[3,4,5] | **6,984,320** | 85.4% | ✅ DepGraph L1, 同 HEAL 模型 | ✅ 低风险 (标准 Conv+BN) |
| 4 | **backbone.deblocks** (FPN upsample neck) | ConvTranspose2d+BN, 3× upsample | **598,784** | 7.3% | ✅ 随 backbone 联动 | ✅ 低风险 |
| 5 | **shrink_conv** (DownsampleConv 384→128) | Conv2d 3×3, stride=1 | **590,080** | 7.2% | ✅ 随 backbone | ✅ 低风险 |
| 6 | **fusion_net** (CoDriving, AttenFusion×3) | ScaledDotProductAttention (无 QKV 投影) | **0** | 0% | N/A (无可学参数) | ⚠️ 中风险: softmax 精度敏感 (见 §2.2) |
| 7 | **cls_head** (Conv2d 128→3) | Conv2d 1×1, anchor=3 | **387** | 0.005% | ❌ 无意义 | ✅ 低风险 |
| 8 | **reg_head** (Conv2d 128→24) | Conv2d 1×1, 3×8=24 | **3,096** | 0.04% | ❌ 无意义 | ✅ 低风险 |

> 【CPU 真算】脚本: `/home/jichengzhi/V2Xverse` 下运行 `centerpointcodriving(args)`, `model.parameters()` 求和, Python 3.7, v2xverse conda env

**关键发现: fusion_net 可学参数为 0**
- `CoDriving` fusion 类使用 `AttenFusion(num_filters[i])` = `ScaledDotProductAttention(dim)`;
- `ScaledDotProductAttention` 内只有 `self.sqrt_dim = np.sqrt(dim)`, **无任何 nn.Parameter**;
- 3 个 scale 的 `AttenFusion(64/128/256)` 均为 0 参数 (纯 bmm+softmax 运算);
- 融合本体不贡献 AP 轴可优化空间, **剪枝/量化目标在 backbone+neck**。

---

### 1.2 backbone 内部分解

**架构**: `ResNetBEVBackbone` with `ResNetModified(BasicBlock, layer_nums=[3,4,5], strides=[2,2,2], num_filters=[64,128,256], inplanes=64)`

**输入/输出 BEV 尺寸** 【yaml 推算】:
- scatter 输出: `[N, 64, 192, 576]` (voxel_size=0.125, range x=72m/y=24m → 576×192)
- resnet stage-0 (64ch, stride=2): `[N, 64, 96, 288]`
- resnet stage-1 (128ch, stride=2): `[N, 128, 48, 144]`
- resnet stage-2 (256ch, stride=2): `[N, 256, 24, 72]`
- deblock-0 (upsample_stride=1, 64→128): `[N, 128, 96, 288]`
- deblock-1 (upsample_stride=2, 128→128): `[N, 128, 96, 288]`
- deblock-2 (upsample_stride=4, 256→128): `[N, 128, 96, 288]`
- FPN concat: `[N, 384, 96, 288]`
- shrink_conv (384→128): `[N, 128, 96, 288]`

| 子模块 | 参数量 | 占比(总) |
|---|---|---|
| backbone.resnet (ResNetModified) | 6,984,320 | 85.4% |
| backbone.deblocks (ConvTranspose2d FPN neck, 3×) | 598,784 | 7.3% |
| **backbone 合计** | **7,583,104** | **92.7%** |

> backbone 主导参数量; 相比 HEAL Pyramid backbone_m1(0.226M BaseBEVBackbone) 与 ResNetBEVBackbone(6.984M), 差异 30×, 两者**不可类比**。

---

### 1.3 fusion_net 内部 (CoDriving multi-scale AttenFusion)

**多尺度融合结构** (multi_scale=True, 3 levels):

| Level | 作用特征 (resnet 各 stage 输出) | AttenFusion dim | 参数量 |
|---|---|---|---|
| 0 | feats[0]: `[N, 64, 96, 288]` (stage-0) | AttenFusion(64) | 0 |
| 1 | feats[1]: `[N, 128, 48, 144]` (stage-1) | AttenFusion(128) | 0 |
| 2 | feats[2]: `[N, 256, 24, 72]` (stage-2) | AttenFusion(256) | 0 |

**融合流程**: 每 level → warp_affine 对齐各 agent 特征到 ego → AttenFusion(ego+others) → deblock upsample → cat(3×128ch) → shrink 384→128

**AttenFusion 运算** (无可学参数):
```
x: [N, C, H, W] → reshape [H*W, N, C]
→ ScaledDotProductAttention(q=x, k=x, v=x):
    score = bmm(q, k^T) / sqrt(C)   # [H*W, N, N]
    attn = softmax(score, dim=-1)    # [H*W, N, N]
    context = bmm(attn, v)           # [H*W, N, C]
→ take ego [0] → [1, C, H, W]
```

---

### 1.4 Planner 模块 (WaypointPlanner_e2e)

**基础信息**:
- 模型类: `WaypointPlanner_e2e` (`codriving/models/planning_end2end.py`)
- Checkpoint: `checkpoints/codriving/planner/codriving_planner.ckpt` (20MB)
- 总参数: **1,662,290** (1.662M) 【ckpt 计数 `model_state_dict`】
- 说明: planner 单独训练/存储, 与感知网络分离 ckpt

**输入**: occupancy map `[B, T=5, C=6, H=192, W=96]` + BEV feature `[B, T=5, 128, 192, 96]` + target point `(2D)`
**输出**: 预测 waypoints (10 points × 2D = 20 维)

| 子模块 | 参数量 | 占比(planner) | 说明 |
|---|---|---|---|
| **decoder.layers** (MLP 384→20, hidden=[1025,512]) | **930,197** | **55.9%** | 最重模块; Linear 384→1025→512→20 |
| conv3_1 (Conv2d 128→256, stride=2) | **294,912** | 17.7% | |
| conv2_2 (Conv2d 128→128) | 147,456 | 8.9% | |
| conv2_1 (Conv2d 64→128, stride=2) | 73,728 | 4.4% | |
| conv3d_2 (Conv3D 128→128, k=(3,1,1)) | 49,280 | 3.0% | temporal 3D conv (T维) |
| conv_pre_1_f (Conv2d 128→32) | 36,864 | 2.2% | BEV feature path |
| conv1_2 (Conv2d 64→64) | 36,864 | 2.2% | |
| conv_pre_1_f2 (Conv2d 64→32) | 18,432 | 1.1% | |
| conv1_1 (Conv2d 32→64, stride=2) | 18,432 | 1.1% | |
| conv3d_1 (Conv3D 64→64, k=(3,1,1)) | 12,352 | 0.7% | temporal 3D conv (T维) |
| target_encoder (MLP 2→128, hid=[16,64]) | 9,456 | 0.6% | 目标点编码 |
| 其他 BN layers | ~34,317 | 2.1% | |

**总计 (感知 + 规划)**:
- 感知: 8,177,435 (8.177M)
- 规划: 1,662,290 (1.662M)
- **联合总计: 9,839,725 (9.840M)** 【CPU 真算 + ckpt 计数】

---

## 二、时间瓶颈

### 2.0 CD-A1 推理延迟真测结果 (@4090, PyTorch FP32)

> 全部为 **【真测·CUDA Event @4090】**, GPU 6 (util=0%, mem=5MiB 完全空闲确认后执行), warmup=50, measure=200, FP32.
> 数据文件: `results/CD_A1_codriving_latency_4090.csv`

**感知网络延迟剖面** (post-scatter BEV tensor 输入 `[N=2, 64, 192, 576]`, max_cav=2, batch=1)

| 子模块 | mean_ms | p50_ms | p99_ms | 参数占比 | 备注 |
|---|---|---|---|---|---|
| backbone.resnet (3-stage ResNet) | **2.807** | 2.756 | 3.699 | 85.4% | 独立 CUDA Event |
| backbone.deblocks (FPN neck) | ≈0.965 | ≈0.745 | — | 7.3% | 差值法 (backbone_full − resnet), 近似 |
| backbone_full (resnet+deblocks+concat) | **3.772** | 3.502 | 4.859 | 92.7% | 独立 CUDA Event |
| shrink_conv (384→128 Conv2d) | **1.270** | 1.269 | 1.283 | 7.2% | 独立 CUDA Event |
| fusion_net (multi-scale AttenFusion ×3) | **4.865** | 4.739 | 6.533 | 0.0%参数 | 独立 CUDA Event; 内部再次调用 backbone.resnet |
| cls+reg heads (Conv2d 1×1) | 0.082 | 0.081 | 0.104 | ~0% | 独立 CUDA Event |
| **感知 e2e** | **10.166** | **10.031** | **12.006** | — | post-scatter→heads, 不含pillar/scatter/NMS |

> ⚠️ 口径: pillar_vfe + scatter (稀疏算子) **未测** — 无法用 dense dummy tensor 替代, 需配合真实 LiDAR 点云才能计时。NMS/后处理亦未含。

**规划网络延迟剖面** (occ `[1,5,6,192,96]` + feat `[1,5,128,192,96]` + target `[1,2]`, B=1, T=5)

| 子模块 | mean_ms | p50_ms | 参数占比 | 备注 |
|---|---|---|---|---|
| conv_pre (occ路径 Conv2d) | 0.152 | 0.149 | — | |
| conv_pre_f (feat路径 Conv2d) | 0.235 | 0.234 | — | |
| conv3d_1 (时序 Conv3D, kernel=(3,1,1)) | 0.095 | 0.092 | 0.7% | temporal T维 |
| conv3d_2 (时序 Conv3D, kernel=(3,1,1)) | 0.137 | 0.136 | 3.0% | temporal T维 |
| decoder MLP (384→1025→512→20) | 0.087 | 0.081 | 55.9% | 纯 Linear |
| **Planner e2e** | **1.504** | **1.460** | **2.125** | 含全部子模块 |

**延迟瓶颈诊断**:

> ★[2026-06-15 勘误 — CD-A1b 真测纠正] ~~fusion_net 是最大瓶颈~~ → 错误。fusion 4.87ms 的大部分来自 backbone.resnet 在 fusion 内部的**第二次完整重跑**(见下 CD-A1b 补充)。

- **★ backbone.resnet 被调用 2 次**: 主路(forward 入口) 2.81ms + fusion_net 内部(`codriving_attn.py:264` `feats=backbone.resnet(x)`, 对原始`[2,64,192,576]`跑全3 scale) 3.04ms = **合计 5.85ms = e2e 的 57.5%**。这才是真正的计算瓶颈。
- **⚠️★[2026-06-16 口径勘误 — 单卡 BN=2 ≠ 分布式部署]**: 上述 5.85ms/57.5% 是 **CARLA 仿真单 GPU 整体 forward(BN=2 = ego+RSU 两 agent 批在一张卡)** 口径,**把两个物理分离设备的计算混测了**。真实分布式部署: **路测 RSU 只调 backbone 1 次**(跑自身点云→抽特征→发出),**车端 ego 才调 2 次**(主路 + fusion 内多尺度重跑)。两设备**并行**, 系统延迟 ≈ `max(RSU路径+通信, ego路径) + planner`, **不是一张卡做 2×**。⇒ 「backbone 被调 2 次」只在 **ego 设备**成立; RSU 1×。部署级真延迟须**分设备 BN=1 重测**(CD-A1c, 待跑), 当前 BN=2 数字不可当部署延迟。
- **fusion_net 内部分解** 【真测·CUDA Event @4090, `results/CD_A1b_fusion_internal_4090.csv`】: backbone重跑 3.04ms (fusion 内 65%) / 纯注意力(warp+bmm+softmax) 0.81ms (8% e2e) / deblock 0.32ms。纯注意力仅 8%, 单独优化 softmax/warp 几乎无全局收益。
- **优化杠杆(按设备区分, 勿当系统级)**: backbone 剪枝/INT8 在 **车端 ego** 同时省主路+fusion 内重跑(ego 上是 2× 杠杆), 在 **路测 RSU** 仅省单次 backbone(1× 杠杆)。**系统级不是 e2e 1.4×** —— 之前算出的「e2e 节省 ~2.9ms / 1.4×」是**单卡 BN=2 口径的合计**, 部署时两设备并行、系统延迟取 max, 该数字**撤回**。fusion 注意力(0.81ms)仍非主要优化方向, 但「2× 杠杆」务必标注**仅 ego 设备**。部署级系统加速待 CD-A1c 分设备 BN=1 真测后才能给。
- backbone.resnet 独立计时 2.81ms (e2e ~28%), 参数主体 85.4%; fusion_net 计时 4.87ms 但其中 3.04ms 是 backbone 重跑(非注意力开销)。
- **planner 仅 1.50ms**, 占联合总时间 (~11.7ms) 的 ~13%, 远非瓶颈。
- 感知 e2e (10.17ms) vs HEAL Pyramid 单网(body_subnet ~25ms): CoDriving **更轻** (体量 8.2M vs Pyramid 多模块), 但注意 CoDriving 含 multi-scale AttenFusion 额外 backbone 重跑代价。

**⚠️ 未测项 (真实 e2e 额外延迟)**:
1. `pillar_vfe + scatter` (稀疏前端): 需真实点云, 估计 ~3-8ms (参考 HEAL Pyramid VFE 经验值, **未测**)
2. NMS / 后处理: 参考 HEAL CUDA NMS ~3ms (**未测**)
3. 占据图生成 (感知→planner 的 occupancy): **未测**

---

## 三、关键结论 (原 §二)

### 3.1 参数瓶颈

```
感知网络 8.177M 分解:
├── backbone.resnet (ResNet 3-stage)  6.984M  85.4%  ← 参数主体, DepGraph 可剪
├── backbone.deblocks (FPN neck)      0.599M   7.3%
├── shrink_conv (384→128)             0.590M   7.2%
├── fusion_net (AttenFusion ×3)       0.000M   0.0%  ← 无可学参数 ★
├── cls_head                          0.000M   ~0%
├── reg_head                          0.000M   ~0%
└── pillar_vfe                        0.001M   ~0%

规划网络 1.662M 分解:
├── decoder MLP                       0.930M  55.9%  ← 最重
├── conv3_1                           0.295M  17.7%
└── 其他 Conv2d+Conv3D+BN+target_enc  0.437M  26.3%
```

**结论**:
- 感知部分剪枝目标 = backbone.resnet (85.4%); 与 V2X-ViT backbone 类似过参数化特征
- fusion 无可学参数 → **融合维度无剪枝目标**, 与 HEAL AttFuse/F-Cooper 同类 (零融合参数)
- planner MLP decoder 是 INT8 量化友好目标 (纯 Linear)
- ★[2026-06-15 勘误] ~~时间瓶颈 = fusion_net (4.87ms, ~48% e2e), 而非 backbone~~ → **真正瓶颈 = backbone.resnet 被调 2 次 (5.85ms, 57.5% e2e, 单卡 BN=2 口径)**; fusion 4.87ms 中 65% 是 backbone 第二次重跑, 纯注意力(warp+bmm+softmax)仅 0.81ms。剪枝 backbone = 同时优化主路+fusion, 不该割裂两者。DCN 确认未启用(yaml 无 dcn 字段, `self.dcn=False`, DCNNet 是死代码)。
  - **⚠️[2026-06-16 口径勘误]**: 「被调 2 次/2× 杠杆」仅在**车端 ego 单设备**成立; 分布式部署中**路测 RSU 只调 1 次**, 两设备并行系统延迟取 max, **不是系统级 2×**。5.85ms/57.5% 是单卡 BN=2 仿真口径, 非部署延迟。详见 §2.0「优化杠杆(按设备区分)」+ 待 CD-A1c 分设备真测。

### 3.2 INT8 / TRT 风险初判

> 全部为 **静态代码分析**, 标注【预研·static】; 未进行 GPU 实测。

| 子模块 | ONNX export | TRT build 风险 | INT8 风险 | 依据 |
|---|---|---|---|---|
| **pillar_vfe + scatter** | ❌ 稀疏算子不可 export | N/A | N/A | 与 HEAL Pyramid encoder 同类 |
| **backbone.resnet** | ✅ 标准 Conv2d+BN | 低 | **🟢 低** | 标准 ResNet, 类比 Pyramid/V2X-ViT backbone INT8 近免损 |
| **backbone.deblocks** | ✅ ConvTranspose2d | 低 | **🟢 低** | 转置卷积, TRT 内建支持 |
| **shrink_conv** | ✅ Conv2d | 低 | **🟢 低** | 标准 |
| **fusion_net (AttenFusion)** | ✅ bmm+softmax | 低-中 | **🟡 中** | softmax 精度敏感; **但无 QKV projection**(不同于 V2X-ViT HMSA); bmm(Q,K^T) 数值范围与 V2X-ViT 类似但规模更小; 参考 QuantV2X V2X-ViT INT8 崩溃场景, CoDriving AttenFusion 风险更低 |
| **cls/reg head** | ✅ Conv2d 1×1 | 低 | **🟢 低** | |
| **Planner Conv2d 系列** | ✅ | 低 | **🟢 低** | 标准 conv |
| **Planner Conv3D** | ✅ (TRT 支持 Conv3d) | 低-中 | **🟡 中** | temporal kernel (3,1,1), TRT 原生支持但 INT8 calibration 需 temporal 样本 |
| **Planner MLP decoder** | ✅ Linear | 低 | **🟢 低** | 纯 FC, INT8 量化友好 |

**整体 INT8 风险档**: **低-中** (明显低于 V2X-ViT; AttenFusion softmax 是唯一潜在风险点)

**推荐 export scope** (类比 V2X-ViT 经验):
- 感知: export `backbone` + `shrink_conv` + `fusion_net` + `cls_head/reg_head`, 输入 `scatter` 输出 BEV tensor
- 规划: export `WaypointPlanner_e2e` 整体 (占 1.66M, 无稀疏算子)
- 不 export: `pillar_vfe` + `scatter` (稀疏算子)

---

## 四、与 HEAL 系模型的口径差异 (禁止混表)

### 4.1 根本差异总结

| 维度 | CoDriving | HEAL Pyramid / V2X-ViT |
|---|---|---|
| **训练数据** | CARLA 仿真 (V2Xverse Towns 1-10) | DAIR-V2X 真实路测 / OPV2V 仿真 |
| **任务** | **感知 + 规划 (端到端)** | 纯感知 (3D 目标检测) |
| **指标** | Driving Score, waypoint L1 (ADE/FDE) | AP30/AP50/AP70 |
| **协作场景** | CARLA V2X: ego + RSU + 最多 max_cav=5 辆 | DAIR: V+I(2 agent固定); OPV2V: ~1.39 车 |
| **检测范围** | [-36,-12,-22,36,12,14] (72m×24m×36m, 前向) | DAIR/OPV2V: [-100,-40,-3,100,40,1] 等 |
| **voxel_size** | [0.125, 0.125, 36] (高分辨率, z 大 bin) | Pyramid: [0.4,0.4,4]; V2X-ViT: 同 Pyramid |
| **BEV grid** | 576×192 → 96×288 (after backbone) | Pyramid: 250×100; V2X-ViT: 176×50 |
| **Backbone** | ResNetBEVBackbone (BasicBlock) | Pyramid: ResNeXt; V2X-ViT: BaseBEVBackbone |
| **Fusion** | AttenFusion (0 learnable 参数, 纯 attention) | Pyramid: Conv 多尺度加权; V2X-ViT: Transformer |
| **检测头** | CenterPoint multiclass (anchor_number=3) | HEAL: anchor-based SSD-like |
| **Planner** | ✅ WaypointPlanner_e2e (1.66M) | ❌ 无规划头 |
| **是否属于 HEAL** | ❌ V2Xverse 独立 repo | ✅ HEAL repo |

### 4.2 框架定位

- CoDriving 在本项目的定位 = **闭环仿真感知链路的部署模型** (Task #7 RSU 时延感知回灌); 它的优化目标是 **降低推理延迟以减少时延回灌对 Driving Score 的影响**, 不是在 DAIR AP 上竞争
- 适合的量化/剪枝分析角度: latency reduction (端到端 Driving Score 影响) > AP

---

## 五、已知实验事实附录

> 截至 2026-06-09, **CoDriving 尚无本项目真测数据**; 以下为声明值来源于 arXiv:2404.09496 (V2Xverse 论文)。

### 5.1 论文发布性能数据 (【声明值·arXiv】)

| 配置 | Driving Score | 来源 | 口径 |
|---|---|---|---|
| CoDriving (no noise, Town 5) | ~0.40+ | arXiv:2404.09496 Table2 (估读) | CARLA 0.9.11 仿真, Town 5, V2Xverse benchmark |

> **⚠️ 上述数字为估读文献值, 未本项目自测; 标注【声明值·arXiv】, 不可用于论文对比 (需自测复现)。**

### 5.2 A-系列实验状态

| 实验 | 内容 | 状态 |
|---|---|---|
| CD-A1 基线推理延迟 | CoDriving 感知 e2e latency @4090, PyTorch FP32 | ✅ 完成: 感知e2e=10.17ms, planner=1.50ms (见 §二.2.0); CSV: `results/CD_A1_codriving_latency_4090.csv` |
| CD-A1b fusion 内部分解 | fusion_net 内部计算来源剖析 @4090 | ✅ 完成(单卡BN=2口径): backbone重跑3.04ms(65%)/纯注意力0.81ms/deblock0.32ms; backbone被调2次共5.85ms(57.5% e2e)**仅车端ego, RSU 1次**; CSV: `results/CD_A1b_fusion_internal_4090.csv` |
| CD-A1c 分设备部署延迟 | RSU(BN=1×1) vs ego(BN=1, 主路+fusion) 分设备真测 | ❌ 待跑 — 纠正 BN=2 单卡口径, 给部署级关键路径延迟 |
| CD-A2 TRT FP16 build | backbone+fusion+heads ONNX→TRT FP16 | 进行中 (结果待回填) |
| CD-A3 TRT INT8 | 含 calibration, Driving Score 影响 | ❌ 未授权 |
| CD-B1 backbone 剪枝 | ResNet [64,128,256]→[32,64,128] DepGraph L1 | ❌ 未授权 (需 CARLA finetune) |

---

*文档版本: v1.1 (2026-06-15, doc-curator — CD-A1+A1b 真测整合)*
*参数量来源: CPU 真算 `/home/jichengzhi/V2Xverse`, v2xverse conda env (Python 3.7), `centerpointcodriving` + `ckpt` state_dict*
*变更: §2.0 延迟瓶颈诊断纠正 backbone 2× 杠杆 + 补 CD-A1b fusion 内部分解; §3.1 结论勘误 fusion>backbone 旧定论; §5.2 CD-A1b 状态填完成 + DCN 确认*

*[2026-06-16 口径纠错] 撤回「系统级 e2e 1.4× / backbone 2× 杠杆」的系统级解读: 5.85ms/57.5%/2× 是 CARLA 单卡 BN=2(ego+RSU 混测)口径; 真实分布式部署 RSU 调 backbone 1 次、ego 2 次, 两设备并行系统延迟取 max。「2× 杠杆」仅车端 ego 成立; 部署级系统加速待 CD-A1c 分设备 BN=1 真测。*
