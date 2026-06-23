# AP 崩溃边界探索 plan v2 — 架构改造推到 capacity 真下限

> **状态**: ✅ **整 plan 完成 (2026-05-22)** — 路径 A' (B×Q 协同崩溃) + 路径 C 触发. 详见 [`phase3_finding.md`](./phase3_finding.md).
>
> **背景**: Plan v1 (`AP不敏感解决方案_plan.md`) 已早停 — Phase 1 实测发现在 prune ∈ [14%, 89%] + FT ∈ [4, 15] + groups=32 + FP32 搜索空间内无 AP 崩溃点. 但用户正确指出: 物理上必然有崩溃点, 当前没找到只因搜索空间太窄. T22 (16,16,16) 是 groups=32 约束硬下限, **真容量底在 groups=8 / planes < 16 区域**.
>
> **目标**: 通过架构改造 (ResNeXt groups: 32 → 8) 打破当前 plane=16 硬下限, push 到极端 (4,4,4) ~97.5% prune 配置, **找到真 AP 崩溃边界** 或证明 Pyramid Fusion 任务即使 g8+(4,4,4) 也不崩 (后者本身是更强论文级发现).
>
> **范围 (用户 2026-05-21 确认)**:
> - ✅ 包含 groups=8 架构改造
> - ❌ **跳过 INT4** (TRT 10.13 不支持, 工程 blocker)
> - ✅ **包含 INT8 + 极端 prune 组合崩溃测试** (Phase 3b)
> - ✅ 接受任意 FT ∈ [4, 25] (单 seed)
> - ✅ 接受重压缩 + 重 collapse — 只要"压缩到极端"是 collapse 理由
>
> **关联文档**:
> - 前置 plan v1: `AP不敏感解决方案_plan.md` (已 ✅ Phase 1 早停)
> - 问题清单: `paper_learning/2. AAAI最终故事/data/问题.md` §7.7 (Phase 1 实测)
> - 数据集: `paper_learning/2. AAAI最终故事/data/数据集制作_plan.md` v1.4

---

## Phase 0 — 工具改造 (~30 min, 单 GPU)

### 0.1 Patch HEAL ResNeXt: groups 参数化

**文件**: `/home/jichengzhi/heal_research/HEAL/opencood/models/sub_modules/resblock.py:229`
- 现状: `groups=32` 硬编码
- 改: 从 hypes_yaml 读 `resnext_groups` (默认 32 保持向后兼容)
- 同时 patch: `/home/jichengzhi/heal_research/HEAL/opencood/models/fuse_modules/pyramid_fuse.py:78`

### 0.2 Patch `tools/structural_prune_pyramid.py`

- 现状: 假设 groups=32, planes/16 必须 pow-2 (in_per_group constraint)
- 改: 加 `--groups N` CLI 参数, 重新计算约束 (in_per_group = 2×planes/N 必须 pow-2 整数)
- 验证表 (groups=8): planes ∈ {4, 8, 16, 32, 64, 128} 都可行

### 0.3 准备 baseline_g8 hypes_yaml

**新文件**: `/home/jichengzhi/heal_research/HEAL/opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pyramid_dair_v2x_basedair_g8.yaml`
- 基于 `lidar_pyramid_dair_v2x_basedair.yaml`
- 加 `resnext_groups: 8` 到 fusion_backbone config
- 加 `width_per_group: 16` (默认 4 会让 `int(p*4/64)*g` 在 p<16 退化为 0; wpg=16 让 p∈{4,8,16} 都有非零 width)
- num_filters [64, 128, 256] 不变
- epoches: 30 (单次完整训练)
- 实测 baseline 容量: 7.55M total params (pyramid_backbone 5.85M, 比 g32 baseline ~3.79M 更大)

### Phase 0 成功信号

✅ Phase 0 完成:
1. resblock.py + pyramid_fuse.py 已 patch, 默认行为不变 (groups=32 时旧行为)
2. structural_prune_pyramid.py 支持 `--groups N`, 测试用 groups=8 + planes=8 能 prune 成功
3. baseline_g8 hypes_yaml 文件存在且加载无错
4. Smoke test: 用 groups=8 hypes 初始化 model + 前向 100 sample 无 NaN

❌ Phase 0 失败:
- 找不到所有 hardcoded groups=32 位置 (可能还有其他模块)
- 改后旧 baseline (groups=32) 加载失败 (向后兼容破坏)

**失败时回退**: 改用 group restructuring 方案 — 不改 HEAL 源码, 直接在结构剪枝时把 32 个 groups 合并为 8 个 (avg 4 channels per new group).

---

## Phase 1 — 训练 baseline_g8 from scratch (~6-8h DDP 4 GPU)

### 1.1 训练

```bash
cd /home/jichengzhi/heal_research/HEAL
python -m torch.distributed.launch \
    --nproc_per_node=4 --use_env --master_port=29500 \
    opencood/tools/train_ddp.py \
    --hypes_yaml opencood/hypes_yaml/dairv2x/LiDAROnly/lidar_pyramid_dair_v2x_basedair_g8.yaml \
    --model_dir /home/jichengzhi/heal_research/checkpoints/Pyramid_DAIR_m1_base_g8_2026_05_22 \
    --half
```

- 30 epoch DDP train (类比官方 baseline 训练 schedule)
- AMP (--half), batch_size 同官方
- Eval 每 epoch, 保留 bestval ckpt

### 1.2 Phase 1 成功信号

✅ Phase 1 完成:
1. 30 epoch 训完, ckpt 保存
2. `net_epoch_bestval_at*.pth` 存在
3. **验证 AP**: 用 e2e_eval_ap.py n=1789 跑 baseline_g8 bestval ckpt
   - **AP50 ≥ 0.50** (baseline_g32 是 0.549, 允许 -5pp 内): ✅ 架构改造成功, 进 Phase 2
   - AP50 ∈ [0.40, 0.50]: ⚠️ 架构损失 5-10pp, 仍可用作 reference (但 Phase 2 比较要扣这个 offset)
   - **AP50 < 0.40**: ❌ Phase 1 早停 — "groups=8 改太多导致 baseline 都学不会", 写报告

### 1.3 早停路径

如果 baseline_g8 AP < 0.40:
- 不是 plan 失败, 是 architecture-level finding
- 论文 §C 可 claim: "Pyramid Fusion 任务 ResNeXt groups 是 capacity 关键因素, groups<32 直接破坏 representation"
- 写 `phase1_finding_g8_baseline_failed.md`, 跳过 Phase 2/3

---

## Phase 2 — 极端 prune in baseline_g8 (~3-4h on 3 GPU 并行)

### 2.1 设计 3 个极端 triplet

groups=8 下可行 plane 配置 (in_per_group=2×p/8 须 pow-2):
- p=4 → in_per_group=1 ✅
- p=8 → in_per_group=2 ✅
- p=16 → in_per_group=4 ✅
- p=32 → in_per_group=8 ✅
- p=64 → in_per_group=16 ✅

3 个极端 triplet (vs baseline 64/128/256):

| Tag | planes | 总 prune% | backbone prune% | 对比 T22 (g32) |
|---|---|---|---|---|
| T_g8_p87 | (8, 8, 8) | ~87% | ~99% | 跟 T22 同尺寸但 g=8 |
| T_g8_p93 | (8, 4, 4) | ~93% | ~99.5% | 介于中间 |
| T_g8_p97 | (4, 4, 4) | ~97.5% | ~99.8% | **极端**: 比 T22 backbone 小 4× |

### 2.2 训练流程 (每 triplet, ~1h on 1 GPU)

```bash
# 1. Structural prune from baseline_g8
python tools/structural_prune_pyramid.py \
    --orig-dir /home/jichengzhi/heal_research/checkpoints/Pyramid_DAIR_m1_base_g8_2026_05_22 \
    --out-dir models/dataset_a_cache/ft_T_g8_pXX_raw \
    --num-filters-new {planes} \
    --groups 8

# 2. Patch config: save_freq=1, eval_freq=1, epoches=38 (baseline_epoch + 15)
sed -i ...

# 3. Train 15 epoch finetune
python -m torch.distributed.launch ... train_ddp.py \
    --hypes_yaml {ft_dir}/config.yaml --model_dir {ft_dir} --half
```

3 个 triplet 并行 (3 GPU). 单次 finetune ~1h.

### 2.3 Phase 2 成功信号

✅ Phase 2 完成:
1. 3 个 raw pruned ckpt 已生成 (T_g8_p87/p93/p97)
2. 每个 raw 跑 AP eval 一次, **应崩** (AP ~ 0, 类似之前 T22_raw)
3. 每个 triplet 完成 15 epoch finetune, 保存 epoch24-38 共 15 个 ckpt

❌ Phase 2 失败:
- structural_prune 因 groups=8 + planes=4 边界 case 报错
- finetune 不收敛 (loss 不下降)

---

## Phase 3 — AP eval × FT × Q 三维 corner cases (~1.5h on 6 GPU 并行)

### 3.1 实验设计 (3a + 3b 两段)

**3a: B × FT 维度 (FP32 only)** — 测纯 prune+finetune 的崩溃边界

每个极端 triplet 测 3 个 FT 档 (FP32):
- FT=4 (临界 sweet spot)
- FT=8 (中等)
- FT=15 (long finetune)

3 triplet × 3 FT + 3 raw (FT=0) = **12 个 AP eval**

**3b: B × Q 组合崩溃 (FT=8 固定)** — 测 prune+quant 协同崩溃

对 3 个极端 triplet, 用 FT=8 ckpt (3a 中等档), 测 4 个 Q 变体:
- Q_fp32 (3a 已测, reuse 不重测)
- Q_fp16 (新)
- Q_int8_mm (新, 期待 INT8 + 极端 prune 是否触发新崩溃)
- Q_int8_pc_wo (新, per-channel W-only, 之前对 T22 友好)
- Q_int8_ent (新, 已知崩 calibrator, 跟极端 prune 协同效应?)

3 triplet × 4 新 Q = **12 个 AP eval**

**3a + 3b 总计 = 24 个 fresh AP eval**, 6 GPU 并行 ~1h.

外加 baseline_g8 自身的 AP (Phase 1 已测) 当 reference.

### 3.1.1 为什么加 Q 维度?

之前 csv 数据显示 Q 在 finetuned 模型 (FT≥25) 上几乎无影响 — 因为 backbone 在 finetune 后 weight 分布"干净", INT8 量化误差被吸收.

但**极端 prune 配合 INT8 可能触发新模式**:
- (4,4,4) backbone params 仅 ~70K, **每个 weight 都至关重要**, INT8 量化误差比例可能大
- Q_int8_ent 在 T22_p89 已 AP=0.458 (小崩), 在 (4,4,4) 可能完全崩 (AP < 0.1)
- Q_int8_mm 在大模型上 robust, 但 (4,4,4) 也许暴露 calibration scale 选择敏感

预期: 至少 1 个 (triplet, Q) 组合 AP < 0.30 (触发崩溃) — 这是论文 §C 的硬数据点.

### 3.2 Phase 3 成功信号

✅ Phase 3 完成 (终止条件 — 满足任一即整个 plan 成功):

**成功路径 A — 找到 B 维度纯崩溃点 (Phase 3a)**:
- 至少 1 个 (triplet, FT, FP32) 配置 AP < 0.30
- 且崩溃 anchor 是极端配置 (T_g8_p97 或 p93)
- 论文 §C: "找到真崩溃边界, AP collapse 在 planes ≤ 4 + g=8 + FP32 出现"

**成功路径 A' — 找到 B×Q 协同崩溃点 (Phase 3b)**:
- 至少 1 个 (triplet, FT=8, Q≠fp32) 配置 AP < 0.30 (而对应 FP32 AP > 0.45)
- 论文 §C: "**极端 prune + INT8 量化协同崩溃** — 单独不崩但组合崩, 反直觉发现"
- 这是**比 A 更有价值的发现** (展示 framework 必须感知组合崩溃, 单维 predictor 不够)

**成功路径 B — 全部不崩 (更强 over-param 证据)**:
- 全部 24 个 anchor (含 INT8 极端) AP > 0.45
- 即 (4,4,4) + INT8_minmax + FT=8 都不崩
- 论文 §C: "Pyramid Fusion + DAIR-V2X 容量+量化双重 over-parameterized, framework 可激进搜索"

**成功路径 C — 部分 spread**:
- AP 跨 24 anchor std > 0.05 (远大于 plan v1 实测的 0.011)
- B 或 Q 或两者交互对 AP 有可量化影响
- 可用于 LGB predictor 训练 (有信号)

❌ Phase 3 失败:
- TRT engine build fail for groups=8 ONNX (不太可能, 但需 verify)
- 所有 24 anchor AP 接近 baseline (-2pp) — 仍是 success path B

---

## 整体 /goal 终止条件

✅ **整体 plan 完成判定** (任一路径):

1. **完整路径 (A/B/C 任一)**: Phase 0 ✓ + Phase 1 ✓ + Phase 2 ✓ + Phase 3 ✓, 写 `phase3_finding.md` 报告
2. **早停 (Phase 1 baseline_g8 失败)**: 报告"架构改造路径不可行"
3. **早停 (Phase 0 失败)**: 找不到所有 hardcoded groups, 写 `phase0_blocked.md`

❌ **整体 plan 失败** (需人工介入):
- Phase 1 训 baseline_g8 反复 OOM/不收敛
- Phase 2 任意 triplet finetune 反复 fail
- 任意 phase 超过预算 2× (Phase 0 > 1h, Phase 1 > 16h, Phase 2 > 8h, Phase 3 > 1h)

---

## 时间预算汇总

| Phase | 工作 | 预算 wall | GPU 占用 |
|---|---|---|---|
| 0 | Patch HEAL + structural_prune, smoke test | ~30 min | 1 GPU |
| 1 | baseline_g8 30 epoch DDP train + AP eval | ~6-8h | 4 GPU DDP |
| 2 | 3 极端 triplet × 15 epoch FT | ~3-4h | 3 GPU 并行 |
| 3a | 12 AP eval B×FT (FP32) | ~30 min | 6 GPU 并行 |
| 3b | 12 AP eval B×Q (FT=8 固定, 4 Q 变体) — INT8 build calib ~3 min each | ~1h | 6 GPU 并行 |
| **总计** | | **~11-14h** | 4-6 GPU |

---

## 关键脚本/路径清单

| 文件 | 状态 |
|---|---|
| HEAL `resblock.py` groups patch | 🔲 Phase 0 待改 |
| HEAL `pyramid_fuse.py` groups patch | 🔲 Phase 0 待改 |
| `tools/structural_prune_pyramid.py` --groups CLI | 🔲 Phase 0 待改 |
| `lidar_pyramid_dair_v2x_basedair_g8.yaml` | 🔲 Phase 0 待写 |
| `scripts/phase2/a10_baseline_g8_train.py` | 🔲 Phase 1 dispatcher |
| `scripts/phase2/a10_extreme_prune_dispatcher.py` | 🔲 Phase 2 dispatcher (3 triplet 并行) |
| `scripts/phase2/a10_eval_3a_b_ft_fp32.py` | 🔲 Phase 3a dispatcher (12 anchor, FP32) |
| `scripts/phase2/a10_eval_3b_b_q_combo.py` | 🔲 Phase 3b dispatcher (12 anchor, B×Q, FT=8 固定) |
| HEAL `train_ddp.py` save freq | ✅ 已 patch (v1 用过, 仍生效) |

---

## 风险与回退策略

| 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|
| HEAL groups=32 硬编码处还有未找到的 | 中 | Phase 0 阻塞 | 用 group restructuring (不改源码, 训练时合并 groups) |
| baseline_g8 训练 AP 严重低于 baseline_g32 | 中 | Phase 1 早停, 但仍是发现 | 写报告: groups=8 破坏了 Pyramid 表达能力 |
| (4,4,4) raw → finetune 不收敛 (太小) | 低 | Phase 2 部分失败 | 跳过 (4,4,4), 用 (8,4,4) 作最极端 |
| 所有配置都不崩 (路径 B) | 中 | 任务真的 over-parameterized | **这是想要的发现**, 写完整报告 |
| DDP 4 GPU 不可用 (其他用户占) | 中 | Phase 1 延期 | 降 batch_size, 用 2 GPU DDP, 时间翻倍 |

---

## 验证清单 (执行中检查点)

- [ ] Phase 0.1: resblock.py + pyramid_fuse.py patched, smoke test pass
- [ ] Phase 0.2: structural_prune_pyramid.py --groups CLI 加完, groups=8+plane=4 测试 OK
- [ ] Phase 0.3: baseline_g8 hypes_yaml 写完, HEAL train 能 load
- [ ] Phase 1: baseline_g8 训 30 epoch 完成, bestval ckpt 存在
- [ ] Phase 1: baseline_g8 AP eval ≥ 0.40 (Phase 2 进 vs 早停)
- [ ] Phase 2: 3 个极端 raw ckpt 已 prune
- [ ] Phase 2: 3 个 triplet 完成 15 epoch finetune
- [ ] Phase 3a: 12 个 B×FT (FP32) AP eval 完成
- [ ] Phase 3b: 12 个 B×Q (FT=8, 4 Q 变体) AP eval 完成
- [ ] Phase 3: `phase3_finding.md` 写出, 标 path A/A'/B/C
- [ ] 更新 `paper_learning/2. AAAI最终故事/data/问题.md` §7.9 加 g8 + Q 组合实验结果
- [ ] 更新 `数据集制作_plan.md` 至 v1.5 (含 g8 + extreme Q 数据)
