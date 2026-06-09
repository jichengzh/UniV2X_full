# Pyramid m2 (Camera/LSS) 结构审计 v1.0

> 作者: sw-optimizer | 日期: 2026-06-05
> 授权: HANDOFF §二 2.3 预授权「profiling 微跑(随机权重+dummy 4-cam 合法, 口径标 "fp32_pytorch hook-level random-weight")【此微跑同样已预授权, 引用本句】」

---

## ⚠️ 顶部声明 — 无 DAIR ckpt, AP 不可测

> **DAIR AP 状态: ❌ 不可测 — 无任何 DAIR camera ckpt。**
>
> 现有 ckpt 仅有 OPV2V 版 (`stage2/m2_alignto_m1/net_epoch25.pth`), 在 DAIR val set 上的 AP 完全未知。
> 若要获得 DAIR AP 需从零自训练 (参见 §三.3 训练成本估算)。
> 本审计中所有 AP 列均标注 **`无实证/类比级`**。

---

## §零 基础信息

| 项目 | 值 |
|------|-----|
| 模型变体 | PyramidFusion m2 (Camera, DAIR yaml) |
| Encoder | LiftSplatShoot (LSS) + EfficientNet-B0 |
| 总参数量 | **20.166M** |
| DAIR ckpt | ❌ 不存在 → AP 不可测 |
| OPV2V ckpt | `stage2/m2_alignto_m1/net_epoch25.pth` |
| DAIR yaml | `opencood/hypes_yaml/dairv2x/CameraOnly/camera_pyramid.yaml` |
| 对比: m1 (LiDAR) | 5.465M — 本审计见 `pyramid_lidar_structure_audit_v1.md` |

> 参数量来源: CPU 直接实例化 + `model.named_children()` 求和，已核验 ∑分模块 = total。

---

## §一 全模块分解

### §1.1 参数量分布

| 子模块 | 参数量 | 占比 | 备注 |
|--------|--------|------|------|
| `encoder_m2` (LiftSplatShoot) | **14,661,446** | **72.70%** | Camera BEV encoder |
| `backbone_m2` (ResNetBEVBackbone) | 267,136 | 1.32% | BEV stride-2 Conv |
| `aligner_m2` | 0 | 0.00% | Identity placeholder |
| `pyramid_backbone` (ResNeXt) | 3,757,635 | 18.63% | **与 m1 共享架构** (见 §1.4) |
| `shrink_conv` (DownsampleConv) | 1,475,072 | 7.31% | **与 m1 共享架构** |
| `cls_head` | 514 | 0.00% | 1×1 Conv |
| `reg_head` | 3,598 | 0.02% | 1×1 Conv |
| `dir_head` | 1,028 | 0.01% | 1×1 Conv |
| **合计** | **20,166,429** | **100%** | |

### §1.2 encoder_m2 (LiftSplatShoot) 内部分解

LSS 由两大计算阶段组成: **camencode** (图像特征提取) + **voxel_pooling** (3D→BEV 投影)。

| 子模块 | 参数量 | 占总量 | 说明 |
|--------|--------|--------|------|
| `.camencode.trunk` (EfficientNet-B0) | 5,288,548 | 26.22% | ImageNet 预训练骨干 |
| `.camencode.up1` (FPN Up decoder 1) | 4,352,000 | 21.58% | 特征上采样 |
| `.camencode.up2` (FPN Up decoder 2) | 4,904,960 | 24.32% | 特征上采样 |
| `.camencode.depth_head` (Conv2d) | 50,274 | 0.25% | 深度预测分类头 → softmax |
| `.camencode.image_head` (Conv2d) | 65,664 | 0.33% | 图像特征头 |
| `voxel_pooling` | 0 | 0.00% | scatter_add 无权重 |
| **encoder_m2 合计** | **14,661,446** | **72.70%** | |

### §1.3 LSS 数据流 (DAIR yaml 配置)

```
输入: [B, N, 3, 288, 512]   (B=2 agents, N=4 cameras, H=288, W=512)
  │
  ├── get_geometry(rots, trans, intrins, post_rots, post_trans)
  │     → geom: [B, N, D, fH, fW, 3]  (3D 点云几何投影)
  │
  ├── get_cam_feats(x)
  │     x → [B*N, 3, 288, 512]  reshape
  │     → camencode.forward([B*N, 3, 288, 512])
  │         trunk (EfficientNet-B0) → backbone features
  │         up1 + up2 (FPN) → [B*N, C_img, fH, fW]
  │         depth_head → depth logits → softmax [B*N, D=98, fH, fW]
  │         image_head → image feats [B*N, C_img, fH, fW]
  │         outer product: depth × image → [B*N, C_img, D, fH, fW]
  │     → reshape [B, N, C_img, D, fH, fW] → permute [B,N,D,fH,fW,C_img]
  │
  └── voxel_pooling(geom, x)   ← **计算瓶颈: 132.65ms / 87% 全体**
        scatter_add + cumsum 展平 → BEV [B, C_img=128, H=256, W=512]
  │
输出: BEV [B=2, 128, 256, 512]
  │
backbone_m2 → [B, 64, 128, 256]
pyramid_backbone → [1, 384, 128, 256]   (同 m1 形状)
shrink_conv → [1, 256, 128, 256]
heads → cls/reg/dir
```

**配置参数 (DAIR yaml)**:
- `xbound: [-102.4, 102.4, 0.4]` → BEV W = 512
- `ybound: [-51.2, 51.2, 0.4]` → BEV H = 256
- `ddiscr: [2, 100, 98]` → D = 98 深度分箱
- `final_dim: [288, 512]` → 相机图像分辨率
- `Ncams = 4`, `ego_modality = m2`

### §1.4 与 m1 (LiDAR) 共享的子模块

`pyramid_backbone` 和 `shrink_conv` 在 m1/m2 中**结构完全相同**, 差别仅在 backbone 输出形状:
- m1 backbone_m1 输出: `[2, 64, 64, 128]` (因 m1 PointPillar BEV = [2,64,128,256])
- m2 backbone_m2 输出: `[2, 64, 128, 256]` (因 m2 LSS BEV = [2,128,256,512], bb stride-2 后)

pyramid_backbone 接收两路 64-ch 特征进行跨 agent 融合 — 形状一致。

---

## §二 子模块计时审计

**口径**: fp32_pytorch CUDA-Event, DAIR dummy input (record_len=2, 4-cam), random weights, GPU0, warmup=10, measure=50 iterations

**源文件**: `results/pyramid_m2_submodule_profile.json`

### §2.1 各阶段延迟

| 子模块 | 延迟(ms) | 占 full_body | 备注 |
|--------|----------|-------------|------|
| `encoder_m2` (LSS full) | **143.72** | **94.4%** | get_geometry + camencode + voxel_pooling |
| ┣ `.camencode` (trunk+FPN+heads) | 11.07 | 7.3% | [8,3,288,512] → 特征+深度 |
| ┃ ┗ `.trunk` (EfficientNet-B0) | 5.71 | 3.8% | 图像骨干 |
| ┃ ┗ `.FPN+heads` (derived) | 5.35 | 3.5% | up1+up2+depth_head+image_head |
| ┗ `.voxel_pooling+geometry` (derived) | **132.65** | **87.1%** | ⚠️ 绝对瓶颈 |
| `backbone_m2` | 1.24 | 0.8% | BEV stride-2 |
| `pyramid_backbone` | 5.54 | 3.6% | 共享 ResNeXt |
| `shrink_conv` | 1.39 | 0.9% | 共享 DownsampleConv |
| `heads` | 0.07 | 0.0% | 1×1 Conv |
| **full body chain** | **152.23** | 100% | enc+bb+pb+sc+heads |

> 对比: m1 (LiDAR) full_body = **5.69 ms** (不含 encoder_m1, 估 2-4ms); m2 (Camera) full_body = **152.23 ms** (含 encoder_m2 143.72ms)
> ⚠️ **scope 不对等**: m2 的 143.72ms encoder 含在 full_body 内, m1 的 VFE encoder 未测/未含。对等比较: **encoder_m2 单独 143.72ms ≈ m1 full_body 25×**; 含 encoder 对等估算: m2/m1 ≈ 152/(5.69+3) = **~17-27× 区间** (依 encoder_m1 估算值)。

### §2.2 voxel_pooling 为何主导延迟

voxel_pooling 的计算量正比于 `B × N × D × fH × fW`:
- B=2, N=4, D=98, fH=72 (288/4), fW=128 (512/4) → 2×4×98×72×128 = **7,225,344 ≈ 7.2M** 个 3D 采样点
- `scatter_add` + `cumsort` 是 gather/scatter 类操作, 在 GPU 上内存访问模式不规则 (非对齐 scatter)
- BEV 分辨率 256×512 远大于 m1 的 128×256 → 投影代价高

> **注**: 此瓶颈可用 `cumsum_trick` 优化 (BEVFusion/Fast-BEV 方案); 本项目未做此优化。

---

## §三 DAIR 数据体系

### §3.1 AP 状态

| 状态 | 值 |
|------|-----|
| DAIR base AP50 | ❌ **不可测** — 无 DAIR ckpt |
| DAIR base AP70 | ❌ 不可测 |
| OPV2V ckpt AP50 | 未测 (有 ckpt, 仅 OPV2V test 有效) |

所有 AP 维度均无实证数据。

### §3.2 与 m1 的相对 AP 预期 (仅文献/定性)

- 在协同感知任务中, Camera-only (LSS) 一般比 LiDAR 低 5-15 AP 点 (3D 深度精度不足)
- HEAL paper 在 OPV2V 上: m1 (LiDAR) > m2 (Camera), 差距约 8-12% AP50
- DAIR 场景更复杂 (ego=车载, infra=路侧), camera 协同效果更不确定
- **以上均为类比推断, 不可作为实验数据引用**

### §3.3 训练成本估算 (若需 DAIR AP)

| 项目 | 估算 |
|------|------|
| DAIR train set | ~5000 帧 (V2I 协同) |
| 预计训练时间 (4090×1) | ~20-30h (40 epochs, HEAL camera pipeline) |
| 预计 AP50 @ OPV2V 可参考 | ~0.60-0.68 (类比 HEAL paper 结果) |
| DAIR AP50 预期 | **未知** — 需自训练后真测 |

---

## §四 剪枝可行性评估

| 维度 | 状态 | 备注 |
|------|------|------|
| pyramid_backbone 剪枝 | ✅ 架构可行 (同 m1) | 但无 DAIR ckpt → 无实证 AP |
| encoder_m2 (EfficientNet) 剪枝 | ⚠️ 需结构化重构 | EfficientNet 通道依赖复杂 |
| backbone_m2 剪枝 | ✅ 可行 (简单 Conv) | 参数少 (267K), 剪枝收益有限 |
| **实证状态** | **全部 `无实证/类比级`** | 无 DAIR ckpt → 无法 finetune 验证 |
| DepGraph 支持 | 类比 m1 可行 | 未验证 camera encoder 的 dependency |

**典型陷阱 (类比 m1 ISS-014)**:
- EfficientNet 中的 inverted residual block 有宽度约束 (depthwise conv)
- depth_head 输出 D=98 channels (非 32 对齐通道几何) → 有 kernel-selection cliff 风险 (类比 ISS-014; padding 假设已证伪, 真机制 = TRT grouped-conv kernel 选择断崖)

---

## §五 量化可行性评估

| 量化方案 | 风险 | AP 预测 | 备注 |
|----------|------|---------|------|
| FP16 | 低 | 类比 m1 近无损 | 标准方案 |
| INT8 MinMax (W+A) | **高** | **可能显著下降** | ⚠️ depth_head softmax 敏感 |
| INT8 Entropy | 极高 | 可能崩塌 | 同 m1 calibration 崩塌问题 |
| INT8 W-only | 中 | 较安全 | 激活量化跳过 |

**⚠️ 关键风险: depth_head softmax 量化**

depth_head 输出经 softmax(D=98 bins) 得到深度概率分布。softmax 对 FP precision 高度敏感:
- 类比 Transformer attention softmax: INT8 量化导致 softmax 数值 collapse (集中/扁平) → 深度预测退化 → BEV 特征错位
- QuantV2X 对 V2X-ViT **整模型 PTQ INT8 实测** AP 从 57.4 → 40.0 (arXiv:2509.03704 Table 1; ISS-031; 归因 attention softmax 是分析性结论, 非逐层实测; 类比参考, 非本模型实测)
- 建议: 若做 INT8, 优先**保 depth_head FP16 / 跳过 softmax 层量化**

| 实证状态 | 全部 `无实证/类比级` | — |

---

## §六 硬件部署预期

| 场景 | 预期延迟 | 说明 |
|------|---------|------|
| 4090 FP16 TRT (预测) | ~60-80 ms | 主瓶颈 voxel_pooling GPU scatter 未必受益于 TRT |
| 4090 INT8 TRT (预测) | ~40-60 ms | depth softmax 风险高 |
| Orin FP16 (预测) | **~300-500 ms** | voxel_pooling scatter 在 Orin 极慢 |
| Orin INT8 (预测) | 参考 FP16 | scatter 不受量化影响 |

> **所有预测均为 "估算" 级 — 无 TRT build 实测**; voxel_pooling 的 GPU scatter 实际性能高度依赖 BEV 分辨率和 batch size。

**m1 vs m2 部署对比** (⚠️ scope 不对等: m1 body 不含 encoder_m1, m2 body 含 encoder_m2):
- m1 LiDAR: 5.69ms body (fp32 pytorch, 不含 VFE) → **实时可用** (>170 FPS body)
- m2 Camera: 152.23ms body (fp32 pytorch, 含 encoder 143.72ms) → **非实时** (~6.6 FPS body, voxel_pooling 瓶颈)
- 若要 Camera 达实时需要: BEV 分辨率降低 / cumsum-trick voxel_pooling / 异步管道

---

## §七 与 m1 (LiDAR) 结构对比

| 维度 | m1 (LiDAR/PointPillar) | m2 (Camera/LSS) |
|------|------------------------|-----------------|
| Encoder | PointPillar VFE (sparse) | LiftSplatShoot (dense) |
| Encoder params | 0.768K (0.01%) | **14.661M (72.70%)** |
| Encoder latency | 未测, 估 2-4ms (VFE sparse, 不含 TRT scope) | **143.72ms** |
| BEV 输入 | [B, 64, 128, 256] | [B, 128, 256, 512] |
| backbone stride | stride-2 → [64, 64, 128] | stride-2 → [64, 128, 256] |
| pyramid_backbone | 共享 (3.758M) | 共享 (3.758M) |
| shrink_conv | 共享 (1.475M) | 共享 (1.475M) |
| Full body (fp32) | **5.69ms** | **152.23ms** |
| DAIR AP50 | ✅ 0.791 (gold std) | ❌ 不可测 |
| INT8 AP风险 | 低 (MinMax) | **高** (depth softmax) |

---

## §八 附录: 实验数据文件

| 文件 | 描述 |
|------|------|
| `results/pyramid_m2_submodule_profile.json` | 本次 profiling 原始 JSON |
| `results/pyramid_m1_submodule_profile.json` | m1 profiling 对比基准 |
| `multi_agent/model/pyramid_lidar_structure_audit_v1.md` | m1 审计 (含 DAIR 实证数据) |
| `multi_agent/model/model_zoo_survey_v1.md` §八 | 多模态全景 (m1/m2/m3/m4) |

---

*文档版本: v1.0 | 最后更新: 2026-06-05*
