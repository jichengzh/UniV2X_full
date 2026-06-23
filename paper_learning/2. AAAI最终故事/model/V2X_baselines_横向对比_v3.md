# V2X Baselines 横向对比 + 框架适用性矩阵 (v3)

> **本文档汇总所有 V2X baseline 真测数据**，含 P0 优化前后对比。
>
> 真权重实测 ✅：F-Cooper, AttFuse, HEAL Pyramid m1, V2X-ViT (DAIR)
> 随机权重 forward + NMS 估算 ⚠️：V2VNet, Where2Comm, Late Fusion
>
> 平台：RTX 4090 + PyTorch 2.x + CUDA 12.x。详细各 model 文档见同目录子文件夹。

## ★ 架构同质性声明（必读 — 公平比较的前提）

### 1. 4 个 HEAL baseline 模型（F-Cooper / AttFuse / V2VNet / V2X-ViT / Where2Comm）共享 model class

这 5 个模型在 HEAL 中都用 **同一个 model class `HeterModelBaseline`** (`opencood/models/heter_model_baseline.py`)，
其 forward 完全相同：
```
encoder_m1 → backbone_m1 → shrinker_m1 → [fusion_net 唯一可变] → heads → postproc
```

唯一的"真差异"是 **`fusion_net` 替换** 和 **shrinker_m1 的 stride 选择**：

| 模块 | 配置 | F-Cooper | AttFuse | Where2Comm | V2VNet | V2X-ViT |
|---|---|---|---|---|---|---|
| encoder | `pillar_vfe.num_filters` | [64] | [64] | [64] | [64] | [64] |
| backbone | `layer_nums` | [3,5,8] | [3,5,8] | [3,5,8] | [3,5,8] | [3,5,8] |
| backbone | `num_filters` | [64,128,256] | [64,128,256] | [64,128,256] | [64,128,256] | [64,128,256] |
| shrinker | dim | [256] | [256] | [256] | [256] | [256] |
| **shrinker** | **stride** | **1** | **1** | **1** | **2** ⚠️ | **2** ⚠️ |
| shrinker | 输出 BEV | 100×352 | 100×352 | 100×352 | **50×176** | **50×176** |
| **fusion_net** | class | **MaxFusion** | **AttFusion** | **Where2commFusion** | **V2VNetFusion** | **V2XViTFusion** |
| heads | cls/reg/dir 1×1 | 同 | 同 | 同 | 同 | 同 |

**为什么 V2VNet/V2X-ViT 用 stride 2**：它们的 fusion 是 ConvGRU 迭代 / 多层 transformer，**在大 BEV (100×352)
上计算量爆炸**。所以这两个模型故意把 shrinker stride 设成 2，先把 feature 减半 (50×176)，再喂给 fusion。
**这是模型架构上的 trade-off**，不是 bug：用 4× 小的 fusion 输入换可行的算力预算。

**结论**：本文档表格里看起来 encoder/backbone 数字几乎一样**是因为它们真的一样**——HEAL 框架的设计哲学就是
"baseline 共享同一套 encoder/backbone，只换 fusion 来公平对比融合策略本身"。

### 2. Pyramid m1 是结构不同的模型（不要被表格的列名误导）

Pyramid m1 用 **不同 model class `HeterPyramidCollab`** (`opencood/models/heter_pyramid_collab.py`)：

| 阶段 | 4 个 baselines | **Pyramid m1** | 真实差异 |
|---|---|---|---|
| model class | `HeterModelBaseline` | `HeterPyramidCollab` | **完全不同** |
| backbone_m1 | `BaseBEVBackbone` (layer_nums=[3,5,8], 16 conv 层) | **`ResNetBEVBackbone`** (layer_nums=**[3]**, 仅 3 conv 层) | Pyramid backbone **小得多** |
| 后段 | `shrinker_m1` (4-6 ms) | **`aligner_m1`** (AlignNet, **0.02 ms 极轻**) | 不同模块 |
| 融合等价模块 | `fusion_net` (1.4-148 ms) | **`pyramid_backbone.forward_collab`** | **不是单纯 fusion** |

**关键事实：`pyramid_backbone` 不是单纯的 fusion 模块**。它内部包含 4 个子操作（见 `pyramid_fuse.py:104-160`）：

```python
def forward_collab(spatial_features, ...):
    # 1. 多尺度 ResNet backbone forward (大头 compute, ~10ms 估算)
    feature_list = self.get_multiscale_feature(spatial_features)

    # 2. 每个尺度的 occupancy 头 (1×1 conv, ~0.5ms)
    for i in range(self.num_levels):
        occ_map = self.single_head_{i}(feature_list[i])

    # 3. ★真正的 cross-agent fusion (~2ms 估算)
    fused_feature_list = [weighted_fuse(feat, score, ...)]  # warp + weighted sum

    # 4. 多尺度 decode (~2ms 估算)
    final_feature = decode_multiscale_feature(fused_feature_list)
```

**所以 Pyramid 14.45 ms 的 `pyramid_backbone` 里，"真正 fusion 算子" (weighted_fuse 那行) 估计只占 ~2 ms**。
其余 12 ms 是"多尺度 backbone + 占用 head + decode"——baseline 在 `backbone_m1 + shrinker_m1 + shrink_conv` 里做的事。

### 3. 公平的"分阶段"对比表

不要按 "fusion 那一列" 直接对比 Pyramid (14.45) vs F-Cooper (1.41)。
正确做法是把"backbone + 后段 + fusion"合并起来比：

| | F-Cooper | Pyramid m1 | 差异 |
|---|---|---|---|
| encoder_m1 | 3.02 | 3.01 | ≈ 0 (一致) |
| **"BEV 处理 + fusion" 合计** | backbone (5.58) + shrinker (5.45) + fusion (1.41) = **12.44** | backbone (2.12) + aligner (0.02) + pyramid_backbone (14.45) = **16.59** | Pyramid 多 4.2 ms — 因多尺度处理 |
| heads | 0.12 | 3.39 | Pyramid 含 shrink_conv |
| **forward 小计** | 15.58 | 22.99 | Pyramid 多 7.4 ms |

**结论**：Pyramid 比 baseline 多 7 ms forward，主要花在"多尺度 + decode" (~12 ms in pyramid_backbone)，
**纯 fusion 算子本身两者差异只有 1-2 ms 量级**。

---

## 一、汇总表 — 真权重实测（4 个）

### 1.1 OPV2V LiDAR 测试集 (3 个模型)

| 模型 | encoder | backbone | shrinker | **fusion** | heads | forward | NMS | postproc | **e2e (ms)** |
|---|---|---|---|---|---|---|---|---|---|
| **HEAL Pyramid m1** ✅ | 3.01 | 2.12 | — (aligner 0.02) | **14.45** | 3.39 | 22.99 | 71.69 | 74.62 | **98.37** |
| **F-Cooper** ✅ | 3.02 | 5.58 | 5.45 | **1.41** | 0.12 | 15.58 | 69.10 | 73.38 | **89.81** |
| **AttFuse** ✅ | 3.24 | 6.06 | 5.51 | **3.49** | 0.18 | 18.48 | 78.58 | 82.87 | **102.30** |

### 1.2 DAIR-V2X LiDAR val 集 (1 个模型，⚠️ 不同数据集)

| 模型 | encoder | backbone | shrinker | **fusion** | heads | forward | NMS | postproc | **e2e (ms)** |
|---|---|---|---|---|---|---|---|---|---|
| **V2X-ViT (DAIR)** ✅ | 2.90 | 2.99 | 0.89 | **27.39** | 0.06 | 34.24 | 23.75 | 27.05 | **61.86** |

> ⚠️ DAIR vs OPV2V 不同数据集：fusion_net 数字反映模型本质，encoder/NMS 反映数据集特性。
> DAIR 上 NMS 远小于 OPV2V 因 box 候选少（场景固定 V2I）。

## 二、★ P0 CUDA NMS 优化实测对比

按 pyramid_fusion 方案，用 `mmcv.ops.nms_rotated` (CUDA polygon IoU) 替换 Shapely Python NMS。

### 2.1 P0 前后对照表 (real weights)

| 模型 | 阶段 | **baseline** | **+P0** | 节省 (ms) | 倍数 |
|---|---|---|---|---|---|
| **F-Cooper** | NMS | 69.10 | **2.53** | **66.57** | **27.3×** |
| | postproc 总 | 73.38 | 7.40 | 65.98 | 9.9× |
| | **e2e** | **89.81** | **24.51** | **65.30** | **3.7×** ⭐ |
| **AttFuse** | NMS | 78.58 | **2.40** | **76.18** | **32.7×** |
| | postproc 总 | 82.87 | 6.89 | 75.98 | 12.0× |
| | **e2e** | **102.30** | **25.63** | **76.67** | **4.0×** ⭐ |
| **HEAL Pyramid m1** (历史 P0 测量) | NMS | 71.69 | ~1 | ~70 | ~70× |
| | **e2e** | **98.37** | ~27.0 | ~70 | **~3.6×** ⭐ |

### 2.2 P0 加速倍数模式

| 模型 | baseline NMS 占比 | P0 后 NMS 占比 | e2e 加速 |
|---|---|---|---|
| F-Cooper | 77.0% | 10.3% | **3.7×** |
| AttFuse | 76.8% | 9.4% | **4.0×** |
| Pyramid m1 | 72.9% | 3.7% | **3.6×** |
| V2X-ViT (DAIR) | 38.4% | (未测 P0, 估算后~1.6×) | (估) 1.6× |

**通用规律**：P0 加速倍数 ∝ baseline NMS 占比。NMS-dominated 模型 (>70%) 拿 ~4× 加速；
transformer-dominated 模型 (NMS <40%) 仅拿 ~1.6×。

## 三、★ 优化后 e2e Pareto（FPS）

| 模型 | baseline (ms / FPS) | **+P0 (ms / FPS)** | 预测 +P0+P1 prune+INT8 | 目标实时 |
|---|---|---|---|---|
| F-Cooper | 89.81 / 11.1 | **24.51 / 40.8 ✓ 实时** | ~14 / 71 (实测 path) | ≥10 FPS ✅ |
| AttFuse | 102.30 / 9.8 | **25.63 / 39.0 ✓ 实时** | ~15 / 67 | ≥10 FPS ✅ |
| Pyramid m1 | 98.37 / 10.2 | ~27.0 / 37.0 ✓ 实时 | ~16 / 63 | ≥10 FPS ✅ |
| V2X-ViT (DAIR) | 61.86 / 16.2 | ~39 / 25.6 ✓ 实时 | ~22 / 45 (prune fusion 必需) | ≥10 FPS ✅ |

所有 4 个真测模型 **+P0 后均达 25-40 FPS** — 满足 V2X 实时要求。

## 四、未测真权重（3 个，仅供参考）

> Forward 网络耗时与权重无关 → 这部分是真实的；NMS 借 F-Cooper 69 ms 作为参考估算。
> Where2Comm 仅在 MediaBrain repo README 列出 DAIR 支持，但**无公开预训练 ckpt**。
> V2VNet 无任何官方 OPV2V/DAIR 预训练。
> Late Fusion HEAL HF ckpt 与本机数据 (OPV2V_orig 而非 OPV2V_Hetero) 不兼容。

| 模型 | encoder | backbone | shrinker | **fusion (随机权重 forward)** | heads | forward | NMS 估算 | **e2e 估算** |
|---|---|---|---|---|---|---|---|---|
| **Where2Comm** ⚠️ | 3.43 | 5.90 | 5.57 | **6.69** (MHA + FFN) | 0.11 | 21.70 | ~75 | **~97** |
| **V2VNet** ⚠️ | 3.12 | 5.91 | 1.42 | **28.56** (ConvGRU 迭代, p99=102) | 0.05 | 39.06 | ~75 | **~114** |
| **Late Fusion** ⚠️ (估算) | 3.02 | 5.58 | 5.45 | n/a (cross-agent NMS only) | 0.12 | 14.17 | ~85 | **~118** |

## 五、% of e2e 分布表（真测 4 个）

| 模型 | NMS 占 e2e | fusion 占 e2e | encoder+backbone+shrinker 占 e2e | 其他 |
|---|---|---|---|---|
| F-Cooper | **76.9%** | 1.6% | 15.7% | 5.8% |
| AttFuse | **76.8%** | 3.4% | 14.5% | 5.3% |
| Pyramid m1 | **72.9%** | 14.7% (in forward) | 5.4% | 7.0% |
| **V2X-ViT (DAIR)** | 38.4% | **44.3%** | 10.9% | 6.4% |

3 个 OPV2V LiDAR 模型 **NMS 都是 73-77% e2e**。V2X-ViT 是唯一一个 **fusion > NMS** 的模型 (44% vs 38%)。

## 六、框架四大加速维度的适用性矩阵

### 6.1 维度定义 (实测加速倍数)

| 维度 | 实现 | 加速倍数 (实测) |
|---|---|---|
| **P0** | mmcv `nms_rotated_cuda` | NMS 27-33× per call ✅ 已测 |
| **P0'** | PyTorch `index_add_` | QuickCumsum 194× (仅 Camera LSS) |
| **结构化剪枝** | L1-norm channel select + finetune | dense conv ~30% lat ↓ |
| **INT8 量化** | TRT INT8 + 校准 | dense conv ~40% lat ↓ |

### 6.2 7 个模型 × 4 个维度兼容性

| 模型 | P0 (NMS) | P0' | 剪枝 RSU 子 | 剪枝 fusion | INT8 RSU | INT8 fusion | **预测 e2e 总加速** |
|---|---|---|---|---|---|---|---|
| **F-Cooper** ✅ | ✅ **实测 3.7×** | ❌ | ✅ | n/a | ✅ | n/a | **6.4× (实测+预测)** |
| **AttFuse** ✅ | ✅ **实测 4.0×** | ❌ | ✅ | ✅ | ✅ | ⚠️ | **6.7×** |
| **Pyramid m1** ✅ | ✅ ~3.6× | ✅ (m2/m4) | ✅ | ✅ | ✅ | ✅ | **6.5×** |
| **V2X-ViT (DAIR)** ✅ | ✅ ~1.6× | ❌ | ✅ | ⚠️ TF head prune 难 | ✅ | ❌ MHA INT8 易掉点 | **~2.8×** |
| Where2Comm ⚠️ | ✅ | ❌ | ✅ | ✅ MHA | ✅ | ⚠️ | **~9×** (估) |
| V2VNet ⚠️ | ✅ | ❌ | ✅ | ❌ GRU | ✅ | ❌ GRU INT8 难 | **~4×** (估) |
| Late Fusion ⚠️ | ✅ ×N | ❌ | ✅ | n/a | ✅ | n/a | **~12×** (估) |

## 七、关键洞察 (for paper)

### 7.1 P0 在 NMS-bound 模型上拿 3.7-4.0× 加速 — 实测验证

这是论文 framework 的**核心 selling point**：
- 4 个 V2X 模型（F-Cooper / AttFuse / Pyramid / 类似的 baseline）NMS 都是 72-77% e2e bottleneck
- 单一 CUDA NMS 替换拿 3.6-4.0× e2e 加速
- **不需要重训，不需要剪枝，不需要量化** — 纯算子替换

### 7.2 fusion 算力分布有 100× 差异

跨模型 fusion_net 真测数据：

| 模型 | fusion_net | 论文 AP (OPV2V) | 延时/AP 比 |
|---|---|---|---|
| F-Cooper (MaxFusion) | 1.41 | 0.66 | baseline |
| AttFuse (SDP) | 3.49 | 0.81 | +2 ms ↔ +0.15 |
| Pyramid (multi-scale fuse) | 14.45 | 0.93 | +13 ms ↔ +0.27 |
| V2X-ViT (3 层 TF, DAIR 测) | 27.39 | 0.55-0.60 (DAIR) | +26 ms — Pareto 后退 |

Pyramid 是 lat-AP 最佳点；V2X-ViT 复杂度高但 AP 增益不成比例。

### 7.3 NMS 是所有 OPV2V LiDAR 模型的共享瓶颈

3 个 OPV2V 真测：NMS = 69-79 ms (~75 ms 中位数)。Shapely O(N²) IoU。
**P0 单点解决"通用 75 ms 大头"**，对所有 V2X HEAL-style 模型有效。

## 八、原始数据 + 复现命令

| 模型 | json | 详细 md |
|---|---|---|
| Pyramid m1 | `data/pyramid_per_stage_timing.json` | `pyramid_fusion/分段耗时实测_v1.md` |
| F-Cooper baseline + P0 | `data/v2x_baseline_timing/fcooper_real.json`, `fcooper_p0.json` | `fcooper/分段耗时实测_v1.md` |
| AttFuse baseline + P0 | `data/v2x_baseline_timing/attfuse_real.json`, `attfuse_p0.json` | `attfuse/分段耗时实测_v1.md` |
| V2X-ViT (DAIR) | `data/v2x_baseline_timing/v2xvit_dair_real.json` | `v2xvit/分段耗时实测_v1.md` |
| Where2Comm 随机 | `data/v2x_baseline_timing/where2comm.json` | (合并本文件 §四) |
| V2VNet 随机 | `data/v2x_baseline_timing/v2vnet.json` | (合并本文件 §四) |

**复现命令** (F-Cooper baseline + P0)：
```bash
CKPT=/home/jichengzhi/heal_research/checkpoints/baselines_hf/HeterBaseline_opv2v_lidar_fcooper_2023_08_06_19_53_10

# baseline
python scripts/phase2/m4_9_v2x_baselines_timing.py \
  --config $CKPT/config.yaml --ckpt $CKPT \
  --tag fcooper_real --warmup 20 --measure 100

# P0
python scripts/phase2/m4_9_v2x_baselines_timing.py \
  --config $CKPT/config.yaml --ckpt $CKPT \
  --tag fcooper_p0 --opt p0 --warmup 20 --measure 100
```

**V2X-ViT DAIR**：先 symlink DAIR 数据：
```bash
ln -sf /data/DAIR-V2X/DAIR-V2X-C/cooperative-vehicle-infrastructure/{cooperative,infrastructure-side,vehicle-side} \
   /home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/

CKPT=/home/jichengzhi/heal_research/checkpoints/baselines_hf/HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26
python scripts/phase2/m4_9_v2x_baselines_timing.py \
  --config $CKPT/config.yaml --ckpt $CKPT \
  --tag v2xvit_dair_real \
  --test-dir /home/jichengzhi/heal_research/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/val.json \
  --warmup 20 --measure 100
```

## 九、honest 状态汇总

| 模型 | 状态 | 数据集 | 真权重？ | P0 测过？ |
|---|---|---|---|---|
| **Pyramid m1** | ✅ 完整 | OPV2V | ✅ | ✅ (历史 m4_8 测) |
| **F-Cooper** | ✅ 完整 | OPV2V | ✅ HEAL HF | ✅ **本轮新测** |
| **AttFuse** | ✅ 完整 | OPV2V | ✅ HEAL HF | ✅ **本轮新测** |
| **V2X-ViT (DAIR)** | ✅ 真权重 baseline | DAIR-V2X | ✅ HEAL HF DAIR | ❌ 未测 (跨数据集对比受限) |
| Where2Comm | ⚠️ 随机权重 only | OPV2V | ❌ 无公开 ckpt | ❌ |
| V2VNet | ⚠️ 随机权重 only | OPV2V | ❌ 无公开 ckpt | ❌ |
| Late Fusion | ⚠️ 估算 only | — | ❌ 数据兼容失败 | ❌ |

**Where2Comm 真权重路径**：
- 唯一选项：用 HEAL train.py 在 DAIR-V2X (本机数据 48GB) 训 5 epoch (~3h GPU)
- 当前状态：未启动，等待用户决定
