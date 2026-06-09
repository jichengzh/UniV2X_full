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

## 二、关键结论

### 2.1 参数瓶颈

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

### 2.2 INT8 / TRT 风险初判

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

## 三、与 HEAL 系模型的口径差异 (禁止混表)

### 3.1 根本差异总结

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

### 3.2 框架定位

- CoDriving 在本项目的定位 = **闭环仿真感知链路的部署模型** (Task #7 RSU 时延感知回灌); 它的优化目标是 **降低推理延迟以减少时延回灌对 Driving Score 的影响**, 不是在 DAIR AP 上竞争
- 适合的量化/剪枝分析角度: latency reduction (端到端 Driving Score 影响) > AP

---

## 四、已知实验事实附录

> 截至 2026-06-09, **CoDriving 尚无本项目真测数据**; 以下为声明值来源于 arXiv:2404.09496 (V2Xverse 论文)。

### 4.1 论文发布性能数据 (【声明值·arXiv】)

| 配置 | Driving Score | 来源 | 口径 |
|---|---|---|---|
| CoDriving (no noise, Town 5) | ~0.40+ | arXiv:2404.09496 Table2 (估读) | CARLA 0.9.11 仿真, Town 5, V2Xverse benchmark |

> **⚠️ 上述数字为估读文献值, 未本项目自测; 标注【声明值·arXiv】, 不可用于论文对比 (需自测复现)。**

### 4.2 A-系列实验状态 (待授权)

| 实验 | 内容 | 状态 |
|---|---|---|
| CD-A1 基线推理延迟 | CoDriving 感知 e2e latency @4090, PyTorch FP32 | ❌ 未测 (Task #7 pending) |
| CD-A2 TRT FP16 build | backbone+fusion+heads ONNX→TRT FP16 | ❌ 未授权 |
| CD-A3 TRT INT8 | 含 calibration, Driving Score 影响 | ❌ 未授权 |
| CD-B1 backbone 剪枝 | ResNet [64,128,256]→[32,64,128] DepGraph L1 | ❌ 未授权 (需 CARLA finetune) |

---

*文档版本: v1.0 (2026-06-09, sw-optimizer)*
*参数量来源: CPU 真算 `/home/jichengzhi/V2Xverse`, v2xverse conda env (Python 3.7), `centerpointcodriving` + `ckpt` state_dict*
*下次更新: CD-A1 真测推理延迟获取后填入 §二 时间瓶颈*
