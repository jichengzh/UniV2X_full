# 1.2 可配置剪枝 — 进度追踪

> 最后更新: **2026-04-16**
> 当前阶段: **Phase B 全部完成**，准备进入 Phase C（剪枝 × 量化联合优化）
> 相关文档：
> - [实验结果汇总](实验结果汇总_2026-04-16.md)（主表，含全部 AMOTA/mAP/参数/耗时）
> - [搜索空间_联合优化](搜索空间_联合优化_2026-04-16.md)（联合视角下重新定义）
> - [实验反思与调整](实验反思与调整_2026-04-14.md)（§13 新增 Phase B 完成后的 4 大认知升级）
> - [完成报告](1.2_可配置剪枝_完成报告.md)（§3.9-3.10 详细结果）
> - [实施计划 v2.1](实施计划_1.2_可配置剪枝_2.md)

---

## ⭐ 核心成就（2026-04-15 ~ 2026-04-16）

**🏆 FFN 60% + queue=2 + 3 epoch 微调 → AMOTA 0.3354 > baseline 0.3298**

这是 Phase B 的里程碑结果：证明 v2.1 假设（零微调崩溃线是假的），首次让剪枝 60% 的 UniV2X 超越未剪枝 baseline。

---

## ⭐ 已在真实 checkpoint 上完成的实验汇总（累计 ~30 次）

> 数据集：V2X-Seq-SPD val (168 samples, 442 GT)；模型：`ckpts/univ2x_coop_e2e_stg2.pth`

### 零微调阶段（2026-04-14，10 次实验）

| 实验 | 工具 | AMOTA | 备注 |
|---|---|---:|---|
| DCN baseline | `test.py` | **0.3298** | 无剪枝基线 |
| DCN 零重训替换 | `convert_dcn_to_conv.py` | 0.1039 | -68%，失败对照 |
| P1 FFN 20% | `run_ffn_sweep.sh` | 0.3055 | — |
| P1 FFN 30% | `test_with_pruning.py` | 0.3189 | -3.3% |
| P1 FFN 40% | `run_ffn_sweep.sh` | 0.2973 | -9.9% |
| P1 FFN 50% | `run_ffn_sweep.sh` | 0.2668 | -19.1% |
| P1 FFN 60% | `run_ffn_sweep.sh` | 0.2366 | -28.3% |
| P1 + seg_MLP 30% | `test_with_pruning.py` | 0.3189 | 与 FFN 30% 相同 |
| Latency baseline/pruned | `benchmark_latency.py` | — | bev_encoder -14.9%, e2e -5.0% |

### Phase B 微调阶段（2026-04-15 ~ 2026-04-16，11 次实验）

**P1 FFN × queue_length 3×2 ablation + 扩展**：

| Ratio | queue=1 AMOTA | queue=2 AMOTA | 参数省 |
|---|---:|---:|---:|
| 30% | **🏆 0.3356** | 0.3306 | -3.55% |
| 50% | 0.2904 | **0.3129** | -5.74% |
| **60%** | 0.2826 | **🏆 0.3354** | **-6.96%** |
| 70% | — | 0.3102 | -8.08% |
| 80% | — | 0.3102 | -9.29% |

**P9 decoder layers × q=2**：

| Layers | AMOTA | 参数省 | 状态 |
|---|---:|---:|---|
| 6 (base) | 0.3298 | 0% | — |
| 5 | 0.3095 | -1.08% | 可用 |
| 4 | 0.2151 | -2.16% | 断崖 |
| 3 | 0.0694 | -3.25% | 崩溃 |

### 累计已验证

- ✅ P1 FFN：6 个压缩率点（含 zero + q=1 + q=2）
- ✅ P9 decoder：4 个层数点
- ✅ queue_length：2 个值，证明与压缩率交互
- ✅ 微调管线（train-modules + skip-aux-heads + queue-length）：完整闭环
- ✅ seg_head MLP 免费午餐
- ✅ Latency benchmark

### 未做/不做的实验（以及原因）

- ❌ **Taylor 准则对比**：grad_collector 在 MultiAgent 上 scatter 失败，且微调预期抹平差距，**降级为 Phase C 可选项**
- ❌ **P2 attn_proj / P8 head_pruning**：级联实现复杂 + ROI 低，**退出搜索空间**
- ❌ **P3 head_mid 单独**：ROI 低（<2M），并入 Phase C 联合配置
- ❌ **Phase 0-B 逐层敏感度**：被全局扫描替代，**不做**
- ❌ **>6 epoch 微调**：收益边际递减，**预留 Phase C.3 备用**
- 🔜 **剪枝 × 量化联合**：Phase C 首要任务

---

## 阶段总览

| 阶段 | 名称 | 工作量 | 状态 | 依赖 | 产出物 |
|:----:|------|:-----:|:----:|:----:|--------|
| A1 | DepGraph 基础设施 | 2.5 天 | `已完成` | 无 | custom_pruners.py, grad_collector.py |
| A2 | 剪枝执行引擎 | 3 天 | `已完成` | A1 | prune_univ2x.py, post_prune.py |
| A3 | 配置与评估管线 | 1.5 天 | `已完成` | A2 | prune_and_eval.py, prune_configs/*.json |
| DCN验证 | DCN→Conv 零重训实验 | 0.5 天 | `已完成（失败）` | A3 | DCN验证结果：AMOTA -68.5% |
| A4 | Phase 0 实验 | 2 天 | `已替代` | A3 | 被 ad-hoc 真实实验 + Phase B 微调实验替代 |
| **B.0** | **微调 pipeline 实现 + 冒烟** | **0.5 天** | **`✅ 已完成`** | A3 | `tools/finetune_pruned.py`, commit 0e31a51 |
| **B.1** | **FFN 剪枝 × queue × FT 3×3 ablation + 70/80% 扩展** | **1 天** | **`✅ 已完成`** | B.0 | AMOTA Pareto 前沿, commit f04d61e/66c38ad |
| **B.3** | **P9 decoder 层数扫描 + q=2 FT** | **0.5 天** | **`✅ 已完成`** | B.0 | 5/4/3 层数断崖分析, commit 66c38ad |
| **B.2** | **P4 准则对比（Taylor/FPGM/L1）** | **~1 天** | **`🟡 降级`** | grad_collector fix | 被微调抹平差距，Phase C 可选 |
| A5 | seg_head + FPN 扩展 | ~2 天 | `部分完成` | A3 | seg_head MLP 已验证 AMOTA 零损失 |
| A6 | 与李星峰 backbone 工作整合 | ~1 天 | `待开始` | 外部沟通 | 复用 non-DCN checkpoint + BackboneWrapper |
| **C.1** | **剪枝 × INT8 联合管线** | **~2 天** | **`🔜 Phase C 首选`** | B.1 + INT8 pipeline | joint_compress_eval.py |
| **C.2** | **蒸馏恢复（可选）** | ~2 天 | `🟡 条件触发` | C.1 | BEV 特征蒸馏 |
| C.3 | TRT engine 构建 + latency 实测 | 1 天 | `待开始` | C.1 | int8 engine + benchmark |

```
依赖关系（修正后）:
  A1 ──→ A2 ──→ A3 ──┬─→ A4 (Phase 0-B 敏感度分析)
                      ├─→ A5 (seg_head + FPN 扩展)
                      ├─→ A6 (与李星峰 backbone 整合)
                      └─→ B1 (TRT 部署) ──→ C1 (联合管线)
```

⚠️ **2026-04-14 关键调整**：
- DCN 零重训替换实验证实失败（AMOTA 0.330 → 0.104）
- 但李星峰周报证明 **DCN + 微调** 路径可行（精度恢复 + e2e 加速 21.6%）
- 搜索空间从 9 维收缩到 6 维，新增 P10 (backbone) 和 P12 (seg_mask)
- 可触及参数从 12M 扩大到 65M（5.4 倍）
- 详见 [实验反思与调整_2026-04-14.md](实验反思与调整_2026-04-14.md)

---

## 详细阶段状态

### Phase A1: DepGraph 基础设施
- **文档**: [CLAUDE_phase_A1.md](CLAUDE_phase_A1.md)
- **状态**: `已完成` (2026-04-13)
- **核心任务**:
  - [x] A.2.1 自定义剪枝器 (4 个 pruner 全部实现并验证)
  - [x] A.2.4 梯度收集器 (grad_collector.py)
  - [x] 单元测试: import + 实例化 + 剪枝操作验证
- **阻塞项**: 无
- **环境**: conda env UniV2X_2.0 (Python 3.9, PyTorch 2.0.1+cu118, torch-pruning 1.6.1 dev install)
- **注意事项**:
  - Python 3.9 不支持 `type | None` 语法，已改为不带类型注解
  - mmdet3d_plugin 的 import 必须先 `import mmdet3d`，避免 registry 重复注册错误
  - TSA concat 维度映射已验证：剪 8 通道 → sampling_offsets 输入从 512→496 (正确)
- **反思**: CUDA 自定义算子未阻断自定义剪枝器的工作——因为我们绕过了 DepGraph 的自动追踪，直接在 pruner 中手动描述依赖关系。这比尝试让 DepGraph 追踪 CUDA 算子更可靠。

### Phase A2: 剪枝执行引擎
- **文档**: [CLAUDE_phase_A2.md](CLAUDE_phase_A2.md)
- **状态**: `已完成` (2026-04-13)
- **核心任务**:
  - [x] A.2.2 剪枝主入口 (prune_model, build_pruner, _build_ratio_dict 等)
  - [x] A.2.3 解码器层数剪枝 (prune_decoder_layers) - mock 测试 6→4 通过
  - [x] A.2.5 剪枝后状态更新 (update_model_after_pruning, verify_model_consistency)
  - [x] 单元测试: 模块名分类器 + 解码器层剪枝 + 状态更新 + 一致性验证
- **阻塞项**: 无
- **注意事项**:
  - MetaPruner (非 BasePruner) 是 tp 1.6 中正确的 pruner 类
  - importance 类名: GroupTaylorImportance, MagnitudeImportance (不是 TaylorImportance)
  - head_dim 计算: 248/8=31 可以通过，但在实际部署时需要 round_to=8 保证对齐
- **反思**: 解码器层剪枝逻辑比预期简单——只需保留最后 N 层，同步 3 个 branches。post_prune 的关键是从 sampling_offsets 输出维度反推 num_heads。

### Phase A3: 配置与评估管线
- **文档**: [CLAUDE_phase_A3.md](CLAUDE_phase_A3.md)
- **状态**: `已完成` (2026-04-13)
- **核心任务**:
  - [x] A.2.6 统一配置入口 (tools/prune_and_eval.py + apply_prune_config)
  - [x] 预设配置文件 (conservative / moderate / aggressive)
  - [x] 配置加载与校验 (load_prune_config)
  - [x] 约束验证 (verify_constraints)
  - [x] 模型统计报告 (report_model_stats, count_parameters)
- **阻塞项**: 无
- **产出文件**:
  - tools/prune_and_eval.py (主入口脚本)
  - prune_configs/conservative.json (FFN 80%, ~10% 缩减)
  - prune_configs/moderate.json (FFN 60%, attn 10%, head 70%, ~25% 缩减)
  - prune_configs/aggressive.json (FFN 40%, attn 20%, head 50%, 删 1 层, ~45% 缩减)
- **注意事项**: 微调配置需在实际 checkpoint 可用后创建（依赖基线训练配置路径）
- **反思**: apply_prune_config 成功将 locked/search 维度分离融入代码。_build_ratio_dict 正确识别 FFN 1 层、attn_proj 4 层，sampling_offsets 全部进入 ignored_layers。

### Phase A4: Phase 0 实验
- **文档**: [CLAUDE_phase_A4.md](CLAUDE_phase_A4.md)
- **状态**: `代码完成 + 部分被替代` (2026-04-15)
- **核心任务**:
  - [x] tools/pruning_sensitivity_analysis.py 脚本实现
  - [x] Phase 0-A 逻辑（已被解耦分析直接替代，无需跑实验）
  - [x] Phase 0-B 逻辑（部分被 ad-hoc 实验替代：见上方"已完成的真实实验"#3-#7）
  - [x] 断点续跑支持 (checkpoint JSON)
  - [x] 可视化: 准则对比柱状图 + FFN 敏感度热力图
  - [x] per_layer_override 支持 (prune_univ2x.py 增强)
  - [x] 逻辑验证: FFN 分类 (safe/moderate/sensitive) + 交互效应计算
  - [ ] **未做**: 系统性 50+ 个逐层敏感度（每个 FFN 单独剪枝），优先级降低
- **替代说明**: 我们用更直接的"全局比例扫描"（FFN 20/30/40/50/60%）代替了"逐层单独剪枝"，因为
  - 前者更快（5 个实验 vs 50+ 个）
  - 前者直接给出可用的部署配置
  - 逐层敏感度可以从全局结果间接推断
- **执行命令**:
  ```bash
  # Phase 0-A (~2.5h)
  python tools/pruning_sensitivity_analysis.py --mode lock-dims \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth --output-dir work_dirs/phase0

  # Phase 0-B (~6h)
  python tools/pruning_sensitivity_analysis.py --mode sensitivity \
      --config projects/configs_e2e_univ2x/univ2x_coop_e2e_track.py \
      --checkpoint work_dirs/latest.pth --output-dir work_dirs/phase0
  ```
- **注意事项**:
  - B1×B2 交互的 "both" 实验需要 Phase B1 TRT 管线完成后补充
  - quant_only 使用 1.1 实验已知值 (INT8 TRT AMOTA=0.364)
  - hessian 可能 OOM, 脚本自动降级
- **反思**: (实验执行后填写)

### Phase B1: TRT 部署端
- **文档**: [CLAUDE_phase_B1.md](CLAUDE_phase_B1.md)
- **状态**: `待开始`
- **核心任务**:
  - [ ] B.2 ONNX 导出适配 (动态维度)
  - [ ] B.3 校准数据重建
  - [ ] B.4 TRT 构建适配
  - [ ] B.5 硬件约束验证工具
  - [ ] B.6 端到端验证矩阵
- **阻塞项**: 等待 A3 完成
- **反思**: (阶段完成后填写)

### Phase C1: 联合压缩管线
- **文档**: [CLAUDE_phase_C1.md](CLAUDE_phase_C1.md)
- **状态**: `待开始`
- **核心任务**:
  - [ ] C.1 联合管线串接 (joint_compress_eval.py)
  - [ ] C.2 联合配置格式 (joint_config.json)
  - [ ] C.3 联合验证 + Pareto 分析
- **阻塞项**: 等待 B1 完成
- **反思**: (阶段完成后填写)

---

## 关键指标追踪

### 主结果汇总（2026-04-14 ~ 2026-04-16, V2X-Seq-SPD val 168 samples）

> 完整数据见 [实验结果汇总_2026-04-16.md](实验结果汇总_2026-04-16.md)。此处只列 Pareto 关键点。

**🏆 Pareto 前沿（按 AMOTA 降序）**：

| 排名 | 配置 | AMOTA | Δ vs baseline | 参数缩减 | mAP |
|:---:|---|---:|---:|---:|---:|
| 1 | **FFN 30% + q=1 微调** | **0.3356** | +0.006 ⬆ | -3.55% | 0.0707 |
| 2 | **FFN 60% + q=2 微调 ⭐** | **0.3354** | **+0.006 ⬆** | **-6.96%** | 0.0689 |
| 3 | FFN 30% + q=2 微调 | 0.3306 | +0.001 | -3.55% | 0.0731 |
| **—** | **baseline (DCN)** | **0.3298** | — | 0% | 0.0724 |
| 4 | FFN 50% + q=2 微调 | 0.3129 | -0.017 | -5.74% | 0.0698 |
| 5 | FFN 70% + q=2 微调 | 0.3102 | -0.020 | -8.08% | 0.0686 |
| 6 | FFN 80% + q=2 微调 | 0.3102 | -0.020 | -9.29% | 0.0659 |
| 7 | P9 dec=5 + q=2 微调 | 0.3095 | -0.020 | -1.08% | 0.0606 |
| 8 | FFN 30% zero-ft | 0.3189 | -0.011 | -1.96% | 0.0648 |
| ... | （其他零微调和 q=1 数据略）| ... | ... | ... | ... |
| - | P9 dec=4 + q=2 微调 | 0.2151 | -0.115 ⚠ | -2.16% | 0.0467 |
| - | P9 dec=3 + q=2 微调 | 💀 0.0694 | -0.260 | -3.25% | 0.0293 |

**修正后的关键结论**（取代旧的零微调边界）：

- **✅ FFN 真实上限至少 70%** （配合 q=2 微调仍保持 AMOTA 0.3102）
- **✅ 最佳 Pareto 点：FFN 60% + q=2 微调**（参数省 6.96%，AMOTA 超过 baseline 0.006）
- **✅ queue_length 最优值随压缩率变化**：30% 用 q=1，≥50% 用 q=2
- **✅ Decoder 层数可降至 5 层**（-1.08% 参数，-0.02 AMOTA），低于 5 层断崖
- **✅ seg_head MLP 剪枝 tracking AMOTA 零影响**（但分割 IoU 有损）
- **🔜 量化维度未联合**：Phase C 核心目标

### Latency Benchmark（2026-04-15, 20 iter × CUDA Event）

| Module | Baseline (ms) | Pruned (ms) | Δ% |
|---|---:|---:|:---:|
| backbone | 34.50 | 33.39 | -3.2% |
| **bev_encoder** | **64.67** | **55.00** | **-14.9%** |
| track_head_decoder | 9.30 | 8.91 | -4.1% |
| **seg_things_mask_head** | **38.66** | **31.06** | **-19.7%** |
| **seg_stuff_mask_head** | **7.84** | **6.53** | **-16.7%** |
| seg_transformer | 32.67 | 28.83 | -11.7% |
| **e2e** | 1528.66 | 1452.71 | **-5.0%** |

模块级加速显著，e2e 受 data loading/tracking update 开销稀释。
完整报告见 [1.2_可配置剪枝_完成报告.md](1.2_可配置剪枝_完成报告.md)。

### 修订后的目标（2026-04-16 再次更新）

| 指标 | 基线 | 原目标 | 修订目标 | **Phase B 实测** |
|------|:----:|:----:|:----:|:----:|
| AMOTA (剪枝+微调后 PyTorch) | 0.330 | > 0.32 | > 0.32 | **✅ 0.3354**（超越 baseline） |
| AMOTA (剪枝+INT8 TRT) | — | > 0.31 | > 0.31 | **🔜 Phase C** |
| 参数量减少 | 0% | > 25% | > 30% | **🟡 7%**（仅 pts_bbox_head 范围）|
| e2e latency 加速 (vs FP32) | 1.0x | > 3.0x | > 3.5x | **🟡 -5% (PyTorch, 未量化)** |
| 搜索空间大小 | 155,520 | 216 | 72 | **✅ ~40**（联合可行） |
| 联合剪枝+量化实验 | 0 | 5+ | 6-8 | **🔜 Phase C** |

---

## 锁定维度记录

| 维度 | 锁定值 | 确定时间 | 确定方式 |
|------|:------:|:-------:|---------|
| P4 importance_criterion | taylor (默认) | 2026-04-14 | 解耦分析（与搜索维度无交互） |
| P5 pruning_granularity | local (默认) | 2026-04-14 | 解耦分析（不影响硬件部署） |
| P6 iterative_steps | 5 | 设计阶段 | 经验值，算法超参数 |
| P7 round_to | 8 | 设计阶段 | INT8 硬件约束 |

---

## 搜索空间分类（2026-04-16 最新）

> 完整联合优化视角的定义见 [搜索空间_联合优化_2026-04-16.md](搜索空间_联合优化_2026-04-16.md)。

| 类别 | 维度 | 当前状态 | 可取值（已验证）| 备注 |
|:---:|---|:---:|---|---|
| **剪枝 - 局部** | P1 ffn_mid_ratio | ✅ **全测完** | {1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2} | 最优点 0.4 (60% 剪) |
| **剪枝 - 局部** | seg_MLP_ratio | ✅ 零微调验证 | {1.0, 0.7} | AMOTA 零影响 |
| **剪枝 - 拓扑** | P9 decoder_num_layers | ✅ **全测完** | {6, 5} 可用 / {4, 3} 断崖 | 5 层是上限 |
| **剪枝 - 级联** | P2 / P8 / P14 | ❌ **退出搜索空间** | — | 级联成本高，ROI 低 |
| **剪枝 - 受限** | backbone | ❌ 依赖外部 | {1.0, 0.7} (李星峰工作) | 需 non-DCN checkpoint |
| **微调 - 超参** | queue_length | ✅ **全测完** | {1, 2} | 与压缩率交互 |
| **微调 - 超参** | epochs | 🟡 只测 3 | {3} 默认 | 6 epoch 待 Phase C |
| **微调 - 超参** | train_modules | ✅ | {pts_bbox_head, seg_head, both, all} | 论文级贡献 |
| **微调 - 超参** | skip_aux_heads | ✅ | {true, false} | true = 省显存 + 聚焦 track |
| **量化 - 范围** | INT8 scope | 🟡 bev_encoder 独立测 | {none, bev, heads, full} | ⚠ **未与剪枝联合** |
| **量化 - 校准** | AdaRound | ✅ 独立测 | {adaround, percentile, mse} | commit 06e4ce9 |

**Phase B 结论**：剪枝维度已基本打透，接下来**核心空白是剪枝 × 量化的联合**。

**Phase C 实验矩阵**（~6-8 个实验）：
1. FFN 60% + decoder 5 + q=2 FT（推荐组合配置）
2. FFN 60% + q=2 FT + INT8 bev_encoder
3. FFN 60% + decoder 5 + q=2 FT + INT8 full ⭐ 联合上限
4. FFN 70% + decoder 5 + q=2 FT + 蒸馏（激进）

> 注：Phase 0-A 原本用于实验决定 P4-P5-P6 值，但经过解耦分析（详见 `实施计划_1.2_可配置剪枝.md` 〇.2），
> 这些维度与搜索维度独立，不需要系统性实验。默认值足够好。
> Phase 0-B（搜索维度敏感度）仍有价值，但优先级低于 A5（seg_head 扩展）和 A6（整合李星峰工作）。

---

## 变更日志

| 日期 | 变更内容 |
|------|---------|
| 2026-04-13 | 初始版本：6 阶段划分，PROGRESS.md 创建 |
| 2026-04-13 | 搜索空间缩减：P4-P7 锁定，从 155,520 缩减到 216 个配置 |
| 2026-04-13 | 全部 6 阶段文档完成：CLAUDE_phase_A1/A2/A3/A4/B1/C1.md |
| 2026-04-13 | Phase A1 实现完成：custom_pruners.py + grad_collector.py + 全部测试通过 |
| 2026-04-13 | Phase A2 实现完成：prune_univ2x.py + post_prune.py + 全部测试通过 |
| 2026-04-13 | Phase A3 实现完成：prune_and_eval.py + 3 预设配置 + apply_prune_config + 全部测试通过 |
| 2026-04-14 | Phase A4 代码完成：pruning_sensitivity_analysis.py + per_layer_override + 可视化 + 逻辑验证通过 |
| 2026-04-14 | **DCN 零重训替换实验失败**：AMOTA 0.330 → 0.104 (-68.5%)。错因：权重语义不等价，需要微调 |
| 2026-04-14 | **读到李星峰 2026-04-11 周报**：他已完成 BackboneWrapper + non-DCN 微调，e2e 加速 21.6% |
| 2026-04-14 | **搜索空间第三次修正**：移除 P3/P8，新增 P10 (backbone) + P12 (seg_mask)，可触及参数 12M → 65M |
| 2026-04-14 | **方向调整**：新增 A5 (seg_head 扩展) 和 A6 (整合李星峰工作) 两个阶段 |
| 2026-04-14 | **重要反思文档创建**：[实验反思与调整_2026-04-14.md](实验反思与调整_2026-04-14.md) |
| 2026-04-14 | **P1 FFN 30% 真实实验成功**：AMOTA 0.330→0.319（-0.011, -3.3%），零微调 |
| 2026-04-14 | **完成报告创建**：[1.2_可配置剪枝_完成报告.md](1.2_可配置剪枝_完成报告.md) |
| 2026-04-14 | 新增工具：`test_with_pruning.py`, `eval_pkl_amota.py` |
| 2026-04-14 | 修复 4 个评估坑：test.py 非分布式禁用 / pickle dict_keys / tmp_dir GC 竞态 / pkl 格式差异 |
| 2026-04-14 | **FFN 剪枝比例扫描完成**：5 个数据点 (0/20/30/40/50/60%)，确定安全线 40%，崩溃线 60% |
| 2026-04-14 | 反思文档新增第 9 节（工程问题整理 + 时间预估修正） |
| 2026-04-15 | **seg_head mask_head MLP 剪枝扩展**：AMOTA 零损失，额外 1.59% 参数节省 |
| 2026-04-15 | **Latency benchmark 完成**：bev_encoder -14.9%, seg_things_mask_head -19.7% |
| 2026-04-15 | 新增工具：benchmark_latency.py（模块级 CUDA Event 计时）|
| 2026-04-15 | **P2 可行性分析 → 发现"局部 vs 级联"是搜索空间的根本盲区** |
| 2026-04-15 | 反思文档新增第 10/11 节：搜索空间按"实现机制"重新分类 |
| 2026-04-15 | 完成报告新增 § 3.8：P2 不能用 prune_direct，需要级联实现 |
| 2026-04-15 | **v2.1 计划诞生**：微调前置到 Phase B，`实施计划_1.2_可配置剪枝_2.md` |
| 2026-04-15 | **B.0 finetune pipeline 实现**：`tools/finetune_pruned.py`, commit `0e31a51` |
| 2026-04-15 | 修复 seg_assigner CPU/GPU 索引 bug；引入 `--train-modules / --skip-aux-heads / --queue-length` 内存优化组合 |
| 2026-04-15 | **B.0.1 突破**：P1 FFN 30% + q=1 微调 → AMOTA 0.3356 **超过 baseline 0.3298** |
| 2026-04-15 | **3×3 ablation 完成**：FFN {30,50,60}% × queue {1,2}，commit `f04d61e` |
| 2026-04-15 | **B.1 核心里程碑**：FFN 60% + q=2 = 0.3354 **再次超过 baseline**（最佳联合点）|
| 2026-04-16 | **推送 GitHub 新分支**：`pruning-finetune-b1-ablation` @ myfork, commit `58461d1` 打包基础设施 |
| 2026-04-16 | **B.1.5/B.1.6 完成**：FFN 70% + q=2 = 0.3102, FFN 80% + q=2 = 0.3102（饱和）|
| 2026-04-16 | **B.3 完成**：P9 decoder 5 可用(0.3095), 4 断崖(0.2151), 3 崩溃(0.0694)，commit `66c38ad` |
| 2026-04-16 | **文档重整**：新建 `实验结果汇总_2026-04-16.md`（主表）+ `搜索空间_联合优化_2026-04-16.md`（联合视角重定义）|
| 2026-04-16 | 反思文档新增 § 13：Phase B 完成后 4 个认知升级（零微调是伪上限 / queue×ratio 交互 / 层 vs 通道剪枝 / 消费级 GPU 部署方法论）|
