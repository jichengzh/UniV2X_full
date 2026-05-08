# M4 / M5 Audit 报告 v1

> **日期**: 2026-05-08
> **任务**: M5.1 (univ2x_tiny ONNX audit) + M4.1 (HEAL Pyramid Fusion 调研)
> **目的**: 为 §4.3 / §4.4 工作分解的后续子任务提供决策依据

---

## 一、M5.1 — univ2x_tiny ONNX Audit (4090 端)

### 1.1 文件清单

`onnx/univ2x_tiny_*.onnx` 共 **29 个文件**，按命名分组：

| 组 | configs | 用途 |
|---|---|---|
| **baseline** | `univ2x_tiny_baseline_bev_200`, `univ2x_tiny_ego_bev_200` | FP32 PyTorch baseline |
| **A 系列** | `A1` ... `A6` (6 个) | 剪枝 baseline (encoder FFN 不同剪枝率) |
| **C 系列** | `C1`, `C2` | plan_b_active 联合搜索点 |
| **D 系列** | `D1` ... `D5` (5 个) | D 空间维度变体 |
| **E 系列** | `E1` ... `E12` (12 个) | encoder 模块级剪枝/量化对照 |
| **P 系列** | `P1_60` | 60% 剪枝 |
| **heads** | `univ2x_tiny_ego_heads_200` | 单独 heads 模块 |

### 1.2 Plugin 含量检查 (关键 audit 结果)

| ONNX | Plugin/Op |
|---|---|
| **28 × `*_bev_200.onnx`** | `MSDAPlugin + RotatePlugin` (BEVFormer 自定义算子) |
| **`univ2x_tiny_ego_heads_200.onnx`** ❌ | `InversePlugin + PythonOp` (PyTorch autograd 直接 export，跨平台死) |

### 1.3 audit 结论 (对 M5 主线影响)

✅ **28 个 BEV ONNX 在 4090 端 PyTorch / mmcv 可用** — 用于 4090 端 PyTorch CUDA Event timing
❌ **不能直接 scp 给 Orin trtexec build** — 同 M2 Plan A 的 plugin name mismatch + 接口不兼容问题
✅ **走 M2 拟合的 f 函数** — 4090 PyTorch FP32 latency 经 `f_fp16: y = 0.099 + 0.948x` 估算到 Orin TRT FP16 latency

### 1.4 M5 主线策略调整

**原 M5.1-5.6 7 个子任务保持不变**，但实施细节调整为：
- **M5.2**: 在 4090 端用 **univ2x-tiny PyTorch 模型直接 timing** (而非 ONNX 跨平台)
- **M5.4**: 用 M2 f 函数估算 Orin AGX latency (确认无需在 Orin trtexec build)
- **M5.5**: 整合 `data/phase4/stage5_v4_dspace.csv` 23 configs (含 4090 PyTorch FP32 latency 488-562ms range) 标 `source='uniad_tiny_variant'`
- **M5.6**: framework 跑 50 候选 Pareto，输出 `results/phase2_pareto_uniad_tiny.csv`

---

## 二、M4.1 — HEAL Pyramid Fusion 调研

### 2.1 Repo 结构

```
~/heal_research/HEAL/
├── opencood/                          # 核心代码
│   ├── models/
│   │   ├── heter_pyramid_collab.py    ★ 主入口 (异构协同 Pyramid)
│   │   ├── heter_pyramid_single.py    ★ 单模态 Pyramid (M4 优先目标)
│   │   ├── fuse_modules/
│   │   │   └── pyramid_fuse.py        ★ PyramidFusion class (ResNetBEVBackbone 子类)
│   │   └── sub_modules/
│   │       ├── base_bev_backbone_resnet.py  ★ ResNetBEVBackbone (CNN backbone)
│   │       ├── feature_alignnet.py    AlignNet (跨模态对齐)
│   │       ├── downsample_conv.py     DownsampleConv
│   │       └── naive_compress.py
│   ├── data_utils/                    # 数据加载
│   ├── hypes_yaml/                    # 配置文件
│   └── tools/
└── requirements.txt
```

### 2.2 模块结构

`HeterPyramidCollab` (`heter_pyramid_collab.py`) 主要模块：

```
HeterPyramidCollab(nn.Module):
├── encoder_{modality}           # 各模态 encoder (LiDAR PointPillars / Camera Lift-Splat)
├── backbone_{modality}          # ResNetBEVBackbone (CNN backbone)
├── aligner_{modality}           # AlignNet (per-modality 对齐)
├── pyramid_backbone (PyramidFusion)  ★ 主 fusion 模块 (ResNetBEVBackbone 子类)
├── shrink_conv (DownsampleConv)      # optional shrink
└── cls_head / reg_head / dir_head    # 共享检测头
```

### 2.3 映射到 v1.5 Config schema (5 模块)

| v1.5 Config 模块 | HEAL Pyramid Fusion 对应 | 备注 |
|---|---|---|
| `backbone` | `encoder_{m}` + `backbone_{m}` (per modality) | 多模态 encoder + ResNet 主干 |
| `encoder` | `ResNetBEVBackbone` 内部 conv blocks | BEV encoder, 与 UniV2X 同 layer |
| `decoder` | `pyramid_backbone` (PyramidFusion) | Pyramid 多尺度 fusion |
| `heads` | `cls_head` / `reg_head` / `dir_head` | 检测头 (与 UniV2X heads 类似) |
| `v2x_comm` | `aligner_{m}` + multi-modality fusion | 跨 agent / 跨 modality 通信 |

### 2.4 与 UniV2X 的关键差异

| 维度 | UniV2X | Pyramid Fusion (HEAL) |
|---|---|---|
| 主架构 | ResNet101 + DCNv2 + Transformer decoder | **全 ResNet (CNN-only, 无 DCN)** |
| BEV 编码 | BEVFormer-style attention | Pyramid 多尺度 conv |
| 多 agent fusion | V2X cross-attention | Pyramid weighted CNN fusion |
| Plugin 依赖 | MSDAPlugin + RotatePlugin + DCNv2 | **零自定义 plugin** ✅ |
| 剪枝可行性 | backbone 死 (DCN fused), encoder/decoder 部分可剪 | **全模块可剪 (标准 conv)** ✅ |
| Orin 部署 | plugin 不兼容 (M2 验证) | **直接 trtexec build** ✅ |
| DL4AGX 资产 | dcnv4-trt (闸门 1+1.5 通过) | SparsityINT8 现成 ✅ |

### 2.5 数据集与训练

- **支持数据集**: OPV2V / V2XSet / V2X-Sim 2.0 / DAIR-V2X-C
- **预训练权重**: HEAL repo 提供，需从 OpenReview / GitHub release 下载
- **训练命令**: `opencood/tools/train.py` (单卡) 或 `train_ddp.py` (多卡)

### 2.6 M4 主线策略可行性确认

✅ **完全可行**, 估时 5-7 day 合理：
1. M4.1 ✅ 完成 (本报告)
2. M4.2 (HEAL 环境 setup) — 0.5 day, 用 4090 + UniV2X_2.0 conda env 兼容
3. M4.3 (PyTorch baseline 验证) — 0.5 day, 加载预训练权重跑 OPV2V
4. M4.4 (Config 适配器) — 1 day, 按 §2.3 映射写 `framework/adapters/pyramid_fusion.py`
5. M4.5 (5-10 configs B1+B2) — 1-2 day, 最关键的 backbone 剪枝实验
6. M4.6 (M2 f 函数估算 Orin) — 0.25 day, 直接套用
7. M4.7 (50 候选 Pareto) — 0.5 day, 复用 A7 流程

**M4 输出**: `results/phase2_pareto_pyramid.csv` (Pyramid Fusion 50 候选 4090 + Orin 双平台 Pareto)

---

## 三、综合结论 — 三模型 spectrum 立得住

```
UniV2X (DCN 反例, 不可剪 backbone)
  ↓ baseline_4090.parquet plan_b_active 18 + 1.1/1.2 各类 ~30 行
  ↓
univ2x-tiny variant (R50 + 50×50 BEV, 部分可剪)  ← M5 主线
  ↓ 28 个 *_bev_200.onnx + stage5_v4_dspace 23 configs amota+latency
  ↓
Pyramid Fusion (HEAL, 全 CNN, 完全可剪)  ← M4 主线
  ↓ 待 M4.5 跑出 5-10 configs
```

这是论文 contribution C4 "框架对多种网络架构的自适应能力" 的天然 dataset。

**下一步**:
- M5.2 启动 (univ2x-tiny PyTorch CUDA Event timing, 0.5 day)
- M4.2 启动 (HEAL 环境 setup, 0.5 day)
- 二者可并行（不同 venv）

---

## 修订历史

| 版本 | 日期 | 内容 |
|---|---|---|
| **v1** | **2026-05-08** | 初版 — M5.1 + M4.1 完成；29 个 univ2x_tiny ONNX audit；HEAL Pyramid 模块结构 + 映射到 v1.5 Config schema |
