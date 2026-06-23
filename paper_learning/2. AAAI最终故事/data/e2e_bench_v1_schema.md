# `e2e_bench_v1.csv` 指标含义参考

> **Schema 版本**: v1.3 (2026-05-15) — lat 列精简至单 `throughput_fps` (见 §六 + `问题.md` 问题 2)
> **测量对象**: HEAL Pyramid Fusion (DAIR-V2X-C, m1 LiDAR collab, N=2 agents)
> **测量平台**: RTX 4090
> **数据集**: DAIR-V2X-C cooperative-vehicle-infrastructure val 1789 frames (latency: 80 measure + 10 warmup; AP: 1789 full DAIR val)
> **当前规模**: 48 行 × 31 列 (32 baseline + 16 mix variants, 2026-05-15)
> **生成脚本**: `scripts/phase2/e2e_bench_v1_orchestrator.py` (build+lat) + `scripts/phase2/a1_run_one_anchor.py` (A.1 phase1 mix) + `scripts/phase2/augment_e2e_bench_v1.py` (schema 派生列)
> **关联文档**:
> - 搜索空间标准: `paper_learning/2. AAAI最终故事/搜索空间一览.md`
> - 测量链路设计: `tools/export_onnx_pyramid_e2e.py` + `scripts/phase2/e2e_bench_pyramid.py`
> - 性能拆解参照: `paper_learning/2. AAAI最终故事/model/pyramid_fusion/分段耗时实测_v1.md`
> - 整体计划: `paper_learning/2. AAAI最终故事/data/数据集制作_plan.md`

---

## 〇、测量链路 (lat_e2e 由什么组成)

```
            [CUDA Event start ──────────────────────────────────┐
                                                                │
   PyTorch wrapper (~3 ms):                                     │
     voxel_features / voxel_num_points / voxel_coords           │ ← lat_e2e_ms 含
        zero-padding 到 MAX_VOX=32000                            │
     从 pairwise_t_matrix 解出 t_ego (N=2, 2, 3)                │
     ↓                                                          │
   TRT engine (~2-7 ms):                                        │ ← lat_trt_ms 单独测量
     VFE (PFN linear+BN+ReLU+ReduceMax)                         │
     Scatter (ScatterND, padding mask 路由 sentinel)             │
     backbone_m1 (ResNet 2D stride-2)                           │
     aligner_m1 (Identity)                                       │
     pyramid_backbone.forward_collab (N=2)                       │
       (multi-scale ResNeXt + single_head + warp_affine +       │
        weighted_fuse + decode_multiscale)                       │
     shrink_conv (384→256)                                       │
     cls_head / reg_head / dir_head (1×1 conv)                  │
     ↓                                                          │
   PyTorch postproc (~3-6 ms):                                  │ ← lat_postproc_ms 单独测量
     delta_to_boxes3d                                            │
     方向 classifier                                              │
     boxes_to_corners_3d + project_box3d                         │
     mmcv CUDA NMS (rotated, P0 优化版)                          │
     range mask                                                  │
            [CUDA Event end ──────────────────────────────────┐
                                                              ▼
                                                          lat_e2e_ms
```

**不在计时范围里的**:
- 数据加载 / voxelization (HEAL SpVoxelPreprocessor 由 DataLoader workers 在 CPU 上做)
- I/O / Python 控制流外部开销

---

## 一、坐标列 — B 剪枝维度

| 列名 | 类型 | 含义 | 当前 v1 取值 | 搜索空间全集 (DAIR Orin) |
|------|------|------|-------------|--------------------------|
| **triplet** | str | 模型剪枝标签 (一个 ckpt 对应一个 triplet) | T1_base/T2_p25/T3_p37/T4_p50/T5_p62/T6_p75/T7_wide_shallow/T8_narrow_deep | — |
| **stage0_planes** | int | ResNeXt backbone stage 0 通道数 | 16, 24, 32, 40, 48, 64 | 4090: {16,24,32,40,64} \| Orin: {16,24,32,40,48,56,64} |
| **stage1_planes** | int | stage 1 通道数 | 32, 48, 56, 64, 80, 96, 128 | 4090: {16,24,32,40,64,72,128} \| Orin: ~11 取值 |
| **stage2_planes** | int | stage 2 通道数 | 64, 128, 160, 192, 256 | 4090: {16,24,32,40,64,72,128,136,256} \| Orin: ~15 取值 |
| **prune_object** | str | 剪枝对象类型 | "none" (T1_base) / "channel" (T2-T8) | {channel, 2:4, none} |
| **sparse_mask** | str | 2:4 sparsity mask 配置 (when `prune_object=2:4`) | 全部 "dense" (v1 未覆盖 2:4) | {dense, mask_s0_only, ..., 2:4_all} |

### triplet 命名约定

| triplet | (s0, s1, s2) | 剪枝率 vs baseline | 备注 |
|---------|--------------|-------------------|------|
| T1_base | (64, 128, 256) | 0% (prune_object=none) | HEAL 官方 stage1 baseline ckpt |
| T2_p25 | (48, 96, 192) | 25% | 标准 25% 剪枝 ckpt |
| T3_p37 | (40, 80, 160) | 37% | 我们自训, ft_040_080_160 |
| T4_p50 | (32, 64, 128) | 50% | 标准 50% 剪枝 ckpt |
| T5_p62 | (24, 56, 128) | 62% | 我们自训, 非标准 ratio |
| T6_p75 | (16, 32, 64) | 75% | 标准 75% 剪枝 ckpt |
| T7_wide_shallow | (48, 64, 128) | ~40% | 我们自训, stage0 保, stage1/2 减 |
| T8_narrow_deep | (24, 48, 192) | ~30% | 我们自训, stage2 保, stage0/1 减 |

---

## 二、坐标列 — Q 量化维度

| 列名 | 类型 | 含义 | 当前 v1 取值 | 搜索空间全集 |
|------|------|------|-------------|--------------|
| **q_tag** | str | 量化配置短标签 (与 class_a 对齐) | Q_fp32/Q_fp16/Q_int8_mm/Q_int8_ent | + Q_mix_s0/s1/s2/s01/s02/s12 (per-stage) |
| **prec_flag** | str | TRT build precision flag (传给 Python API) | fp32/fp16/int8 | 同左 |
| **q_bits** | str | 全局量化位宽 (搜索空间一览 §三 Q1 字段) | FP32/FP16/INT8 | {FP32, FP16, INT8} |
| **q_bits_per_stage** | str | per-stage Q1' 三元组 (搜索空间 Q 总 17 中的 8 个 per-stage) | 全部 NaN (v1 仅全局 Q) | (s0,s1,s2) ∈ {FP16,INT8}³ |
| **q_granularity** | str | 量化粒度 (搜索空间 §三 Q2) | none (FP32) / per-tensor (FP16, INT8) | {per-tensor, per-channel, none} |
| **q_object** | str | 量化目标 (搜索空间 §三 Q3) | none (FP32) / W+A (FP16, INT8) | {W-only, W+A, none} |

### q_tag → calibration / runtime 对照

| q_tag | TRT build flag | 校准 calibrator | 校准 cache | 备注 |
|-------|---------------|-----------------|-----------|------|
| Q_fp32 | `--fp32` | (无) | — | 单 FP32 — 不量化 |
| Q_fp16 | `--fp16` | (无) | — | Tensor Core HMMA, 无校准 |
| Q_int8_mm | `--int8 --calibrator minmax` | `IInt8MinMaxCalibrator` | `T{n}_{tag}_Q_int8_mm_calib.cache` | 100 sample DAIR test |
| Q_int8_ent | `--int8 --calibrator entropy` | `IInt8EntropyCalibrator2` | `T{n}_{tag}_Q_int8_ent_calib.cache` | 同上 |

> **⚠️ 实证: entropy calibrator 在 6/8 个 triplet 触发 AP 崩塌** (AP50 0.02-0.4 vs minmax 的 0.74-0.79). 详见 §九.

---

## 三、坐标列 — D 部署维度

| 列名 | 类型 | 含义 (搜索空间一览 §三 D) | 当前 v1 取值 | 4090 全集 | Orin 全集 |
|------|------|--------------------------|-------------|-----------|-----------|
| **d_scheme** | str | IP 路由 + 流水线 一体化 | 全部 "GPU" (4090 无 DLA) | {GPU} (1 个) | {GPU, DLA0, DLA1, dual_A, dual_B, triple_C} (6 个) |
| **d_tactic** | str | TRT tactic_sources | 全部 "default" (未传 --tactic-sources) | {default, with_cudnn, cublas_lt, all_enabled, edge_only} (5 个) | (TRT 8.5) {default, with_cudnn, cublas_lt, all_enabled} (4 个) |
| **d_workspace_gb** | int | TRT workspace size (GB) | 全部 4 (--workspace-mb 4096) | {1, 2, 4, 8, 16} | {1, 2, 4, 8} |

> v1 仅覆盖 D 空间的 1 个 cell (GPU + default tactic + 4GB workspace). 4090 完整 D 子空间 = 1 × 5 × 5 = **25 cells**, Orin = 6 × 4 × 4 = **96 cells**.

---

## 四、硬件列

| 列名 | 类型 | 含义 | 当前 v1 取值 | 搜索空间全集 |
|------|------|------|-------------|--------------|
| **hardware** | str | 硬件 capability YAML 标签 | 全部 "rtx4090" | {rtx4090, orin_agx_64gb} |
| **device** | str | GPU 名称 (TRT runtime 报告) | "NVIDIA GeForce RTX 4090" | — |

---

## 五、采样规模

| 列名 | 类型 | 含义 | 取值 |
|------|------|------|------|
| **max_voxels** | int | TRT engine 静态 padding 上限 | 32000 (固定; DAIR test p99=24884, 留 28% 余量) |
| **n_collected** | int | 真实进入 latency 统计的样本数 | 80 (warmup 后采样的 measure 计数) |
| **n_skipped** | int | 跳过的样本数 (voxel > MAX_VOX 或 record_len != 2) | 14 (~5%) |
| **real_voxels_mean** | float | n_collected 样本 voxel 数 mean | DAIR 典型 ~23000 |
| **real_voxels_p99** | float | p99 | ~24200, 远小于 max_voxels |

---

## 六、性能 — 延迟 (单列, schema v1.3 精简后)

> **精简决议 (2026-05-15, 详见 `问题.md` 问题 2)**: 原 11 个 lat 列 (lat_e2e_{mean,p50,p99} + lat_trt_{mean,p50,p99} + lat_postproc_{mean,p50} + speedup_{e2e,trt}) 强相关 → LGB 特征归因混乱 + paper 表表头臃肿. 数据表只保留 `throughput_fps` 作 Pareto 唯一 lat 指标. lat breakdown 按需从 `models/e2e_cache*/<triplet>_<Q>_bench.json` 读取.

### 6.1 `throughput_fps` — 端到端吞吐 (唯一 lat 指标)

| 列 | 公式 | 含义 |
|----|------|------|
| `throughput_fps` | `1000 / lat_e2e_mean_ms` | 端到端吞吐 (单 IP 单实例), 论文 §C Pareto 主指标 |

**校准参照** (T1_base+Q_fp32 baseline): throughput ≈ 88.8 fps (lat_e2e ≈ 11.26 ms).

### 6.2 按需诊断: 从 bench JSON 读 lat 三段 breakdown

每个 anchor 对应 `models/e2e_cache*/<triplet>_<Q>_bench.json` 含完整 lat breakdown:

```json
{
  "lat_e2e_ms":      {"mean": ..., "p50": ..., "p99": ...},  // 端到端 wall-clock
  "lat_trt_ms":      {"mean": ..., "p50": ..., "p99": ...},  // TRT engine forward (Q/B/D 真信号)
  "lat_postproc_ms": {"mean": ..., "p50": ...},              // NMS+decode (FP32, 跟 Q 无关)
  ...
}
```

**何时读 bench JSON 而非数据表**:
- **Q 维度对比 (INT8 vs FP16)** → `lat_trt_ms.mean` (e2e 含 NMS 噪声会反向, 见 `问题.md` 问题 2)
- **B 维度对比 (channel alignment)** → `lat_trt_ms.mean` (alignment 异常在 TRT lat 上更明显)
- **SLA 长尾 / 工程报告** → `lat_e2e_ms.p99` (用户实际体验)
- **NMS workload 异常排查** → `lat_postproc_ms.mean` (postproc 异常多半因 score_threshold 候选数飙升)

---

## 八、性能 — AP 指标

> **v1.2 (2026-05-15) 数据源更新**: AP 已切换到 e2e engine 实测 (方案 1 主路径决议落地). 不再用 subnet AP.

| 列名 | 类型 | 含义 | 当前数据源 |
|------|------|------|----------|
| **ap30** | float | DAIR-V2X val 500 sample sweep, IoU 阈值 0.30 | INT8 minmax: `results/e2e_ap_minmax_v1/*.json`; 其他 prec: 待补 |
| **ap50** | float | 同上, IoU 0.50 (主指标) | 同上 |
| **ap70** | float | 同上, IoU 0.70 (长尾) | 同上 |

**v1.2 切换原因 (2026-05-15)**:
- 旧版 schema 写 "AP 跟 TRT engine ONNX 范围无关 (因为 cls/reg/dir 在 forward 最末端)" — **被 e2e 实测推翻**
- 8 个 INT8 minmax engine 在 e2e 上实测 AP50 = 0.62-0.65, 而 subnet (`class_a_pyramid_full`) 报 0.73-0.79 — **系统性高估 0.10-0.16**
- 物理解读: TRT INT8 build 会量化 VFE+Scatter, 这部分量化噪声 subnet 测量看不到
- 详见 `data/问题.md §1.5.2`

**旧版数据保留**: `data/_by_class/class_a_pyramid_full.parquet` 仍存在, 但**论文 / Pareto 表 / LGB 训练数据**统一改用 `results/e2e_ap_minmax_v1/*.json`. 跨 D 配置 AP 仍近似不变 (workspace/tactic 不影响数值精度).

---

## 九、资源 (TRT 构建期)

| 列名 | 类型 | 含义 |
|------|------|------|
| **engine_size_mb** | float | TRT engine 序列化文件大小 (MB) |
| **build_secs** | float | TRT engine build 耗时 (秒, 含校准; cached engine 是 NaN) |

**engine_size_mb 解读** (T1_base):
| prec | size | vs FP32 ratio | 解读 |
|------|------|---------------|------|
| FP32 | 110.5 MB | 1.00× | 全 FP32 权重 |
| FP16 | 54.0 MB | 0.49× | 权重全 FP16 (~2× 缩) |
| INT8 minmax | 49.8 MB | 0.45× | 部分 INT8 (Conv 类), 部分 FP16 fallback |
| INT8 entropy | 49.6 MB | 0.45× | 同 minmax |

INT8 vs FP16 只缩 ~8% — 因为只 ~58% 的层真的量化, 其余 ~42% 在 Myelin foreign node 里仍是 FP16 (TRT INT8 不支持 GridSample/Sigmoid/Softmax/ScatterND).

---

## 十、元数据

| 列名 | 类型 | 含义 |
|------|------|------|
| **build_success** | bool | engine build + bench 是否成功 |
| **fail_reason** | str | 失败时记 |
| **ts** | str (ISO 8601) | anchor 写入时间戳 |

---

## 十一、读数据时该注意的事 (gotchas)

| 陷阱 | 解释 |
|------|------|
| **Q_int8_ent 6/8 个 triplet AP 崩塌** | entropy calibrator 把 cls 分支量化 scale 选错, 导致 sigmoid 输出整体偏低/偏高 → AP50 ~0.02-0.41 (vs minmax ~0.75); 仅 T2_p25 (0.61) 和 T6_p75 (0.75) 没崩。**搜索器应避开 entropy 校准** (或对 cls 头单独 fallback FP16) |
| `lat_e2e` ≠ `lat_trt + lat_postproc` 之和 | 差 0.1-0.5 ms 是 PyTorch padding wrapper 开销, 不是 bug |
| INT8 e2e 有时比 FP16 慢 (T1_base) | NMS 工作量噪声 (~±2 ms) 盖过引擎 INT8 gain (~0.2-0.6 ms). 用 `lat_trt_mean_ms` 比, INT8 始终 ≤ FP16 |
| `lat_trt` INT8 speedup 只有 1.0-1.3× over FP16 | ~42% 层不可量化, 走 FP16 Myelin foreign node. 引擎 inspector 已验证 |
| 同一 triplet 不同 prec 的 `real_voxels_mean` 一致 | 数据集固定, voxel 数跟 prec 无关 |
| AP 列跨 D 取自 D_ws4, 同 (triplet, q_tag) 多 D 看 class_a_pyramid_full | AP 跟 workspace 几乎独立, 取 D_ws4 代表 |
| `q_bits_per_stage` 全 NaN | v1 仅含全局 Q (Q_fp32/Q_fp16/Q_int8_mm/Q_int8_ent), 没含 Q_mix_s0 等 6 种 per-stage |
| `d_scheme` / `d_tactic` / `d_workspace_gb` 全相同 | v1 仅占 4090 D 子空间 1/25 cell (GPU + default + 4GB) |
| `prune_object` T1_base="none", T2-T8="channel" | 严格按搜索空间一览 §三 C1/C2 约束: planes=baseline 即 prune_object=none |
| `hardware` 全 "rtx4090" | Orin 跨硬件部分见 plan §〇.6.3 Class A Orin |

---

## 十二、当前数据范围 (v1)

✅ 已含:
- 32 anchor = 8 triplet × 4 prec (FP32/FP16/INT8_minmax/INT8_entropy) on RTX 4090
- 全 41 列 schema 对齐搜索空间一览 §三
- AP30/50/70 (500 sample DAIR val, 跨 D 取 D_ws4)
- Speedup_e2e/trt vs T1_base+Q_fp32
- Throughput_fps

❌ 暂未含 (待 v2/v3 扩展):
- D 维度变化: d_tactic ≠ default, d_workspace_gb ≠ 4, DLA 路由 — 需要 23-95 个新 anchor
- Q1' per-stage mixed: Q_mix_s0/s1/s2/s01/s02/s12 (6 种) × 8 triplet = 48 anchor
- 2:4 sparse: prune_object="2:4" + per-stage 2:4 mask
- Orin 跨硬件: 同 triplet/q_tag 但 hardware="orin_agx_64gb"
- 资源详细: peak_gpu_mem_mb / params_kb (plan §3.3)

---

## 十三、修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1 | 2026-05-14 | 初版 — 32 anchor × 25 cols. TRT engine 含 VFE+Scatter+backbone+aligner+pyramid+shrink+heads |
| v1.1 | 2026-05-14 | 扩 schema 到 41 cols — 加 B 维度 (prune_object/sparse_mask), Q 维度 (q_bits/q_bits_per_stage/q_granularity/q_object), D 维度 (d_scheme/d_tactic/d_workspace_gb), hardware, AP30/50/70 (从 class_a_pyramid_full 拉), speedup_e2e/trt_vs_baseline, throughput_fps. 显式对齐搜索空间一览 §三 |
