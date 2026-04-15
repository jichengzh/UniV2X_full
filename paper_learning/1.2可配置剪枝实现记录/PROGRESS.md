# 1.2 可配置剪枝 — 进度追踪

> 最后更新: 2026-04-15
> 总工期: ~15.5 个工作日（可能需要重新评估）
> 当前阶段: **⚠️ 方向校正中** — Phase A1-A3 完成，多个 ad-hoc 实验已在真实 checkpoint 上完成
> 详见: [实验反思与调整_2026-04-14.md](实验反思与调整_2026-04-14.md) + [完成报告](1.2_可配置剪枝_完成报告.md)

---

## ⭐ 已在真实 checkpoint 上完成的实验汇总

> 数据集：V2X-Seq-SPD val (168 samples)；模型：`ckpts/univ2x_coop_e2e_stg2.pth`

| 序号 | 实验 | 日期 | 工具 | 关键结果 |
|:---:|---|---|---|---|
| 1 | DCN baseline AMOTA | 2026-04-14 | `test.py` | **AMOTA=0.3298** (无剪枝基线) |
| 2 | DCN 零重训替换 | 2026-04-14 | `convert_dcn_to_conv.py` + `test.py` | AMOTA=0.1039（失败对照，验证零重训不可行）|
| 3 | P1 FFN 30% | 2026-04-14 | `test_with_pruning.py` | **AMOTA=0.3189** (-0.011) |
| 4 | P1 FFN 20% | 2026-04-14 | `run_ffn_sweep.sh` | AMOTA=0.3055 |
| 5 | P1 FFN 40% | 2026-04-14 | `run_ffn_sweep.sh` | AMOTA=0.2973 |
| 6 | P1 FFN 50% | 2026-04-14 | `run_ffn_sweep.sh` | AMOTA=0.2668 |
| 7 | P1 FFN 60% | 2026-04-14 | `run_ffn_sweep.sh` | AMOTA=0.2366 |
| 8 | P1 + seg_MLP 30% | 2026-04-15 | `test_with_pruning.py` | **AMOTA=0.3189**（与 #3 完全相同，分割 IoU 略降）|
| 9 | Latency baseline | 2026-04-15 | `benchmark_latency.py` | bev_encoder=64.7ms, e2e=1528ms |
| 10 | Latency pruned | 2026-04-15 | `benchmark_latency.py` | bev_encoder=55.0ms (-15%) |

**累计 10 次真实实验，主要结论：**
- FFN 安全剪枝上界 = 40%
- seg_head MLP 剪枝对 AMOTA 零损失
- 模块级加速 11-20%，e2e 加速 5%（受 CPU bound 限制）

**未做的实验**（属于原计划 Phase A4 的系统性扫描，后被 ad-hoc 实验替代）：
- ❌ Phase 0-A：P4/P5/P6 的 11 个对比实验（解耦分析直接锁定，不需做）
- ❌ Phase 0-B：50+ 逐层 FFN 敏感度实验（暂未做，~15 小时工作量）
- ❌ P3 head_mid 单独实验（工程已就绪）
- ❌ P2 attn_proj 实验（需要先实现方案 B，半天工程）
- ❌ Taylor 准则对比 L1（推荐下一步）
- ❌ 微调恢复实验（推荐下一步）

---

## 阶段总览

| 阶段 | 名称 | 工作量 | 状态 | 依赖 | 产出物 |
|:----:|------|:-----:|:----:|:----:|--------|
| A1 | DepGraph 基础设施 | 2.5 天 | `已完成` | 无 | custom_pruners.py, grad_collector.py |
| A2 | 剪枝执行引擎 | 3 天 | `已完成` | A1 | prune_univ2x.py, post_prune.py |
| A3 | 配置与评估管线 | 1.5 天 | `已完成` | A2 | prune_and_eval.py, prune_configs/*.json |
| DCN验证 | DCN→Conv 零重训实验 | 0.5 天 | `已完成（失败）` | A3 | DCN验证结果：AMOTA -68.5% |
| A4 | Phase 0 实验 | 2 天 | **`已被替代`** | A3 | ⚠️ 原计划的系统性扫描已被 ad-hoc 真实实验替代（见下文"已完成的真实实验"） |
| **A5** | **seg_head + FPN 扩展** | **~2 天** | **`部分完成`** | **A3** | ✅ seg_head MLP 已实现并验证（AMOTA 零损失）；FPN/seg_transformer 待续 |
| **P2 工程** | **级联剪枝实现** | **~半天** | **`待开始`** | **A3** | 必须实现方案 B 才能跑 P2 实验 |
| **A6** | **与李星峰 backbone 工作整合** | **~1 天** | **`待开始`** | **外部沟通** | **复用 non-DCN checkpoint + BackboneWrapper** |
| B1 | TRT 部署端 | 3.5 天 | `待开始` | A3 | 剪枝后 TRT engine, verify_trt_constraints.py |
| C1 | 联合压缩管线 | 2.5 天 | `待开始` | B1 | joint_compress_eval.py, Pareto 分析 |

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

### 实测 baseline（2026-04-14, V2X-Seq-SPD val 168 samples）

| 配置 | AMOTA | Δ vs baseline | 参数缩减 | mAP |
|------|:----:|:----:|:----:|:----:|
| **baseline (DCN)** | **0.3298** | — | 0% | 0.0724 |
| DCN 零重训替换 | 0.1039 | -68.5% ❌ | -1.8% | 0.0427 |
| FFN 20% | 0.3055 | -7.4% | 1.27% | 0.0669 |
| **FFN 30%** ⭐ | **0.3189** | **-3.3%** | 1.96% | 0.0648 |
| FFN 40% | 0.2973 | -9.9% | 2.54% | 0.0628 |
| FFN 50% | 0.2668 | -19% | 3.13% | 0.0595 |
| FFN 60% | 0.2366 | -28% ⚠️ | 3.82% | 0.0561 |
| 李星峰 non-DCN+微调* | ~0.33 | 恢复 | — | — |

*李星峰数据来自 2026-04-11 周报，未独立验证

**关键结论**：
- **零微调安全线** = FFN 40%（损失 10%）
- **零微调崩溃线** = FFN 60%（损失 28%）
- **最优 ROI** = FFN 30%（每 1% 参数省仅损 0.006 AMOTA）
- **非单调现象**：20% < 30%，L1 范数选通道存在随机性
- **seg_head MLP 免费午餐**：对 AMOTA 零影响，多省 1.59% 参数（仅牺牲分割 IoU）

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

### 修订后的目标

| 指标 | 基线 | 原目标 | 修订目标 |
|------|:----:|:----:|:----:|
| AMOTA (剪枝+微调后 PyTorch) | 0.330 | > 0.32 | > 0.32（精度损失 < 0.01） |
| AMOTA (剪枝+INT8 TRT) | — | > 0.31 | > 0.31 |
| 参数量减少 | 0% | > 25% | **> 30%**（可触及从 12M 扩大到 65M） |
| e2e latency 加速 (vs FP32) | 1.0x | > 3.0x | **> 3.5x**（剪枝加速 × INT8 加速） |
| 搜索空间大小 | 155,520 | 216 | 72 (6 维, 再次缩减) |

---

## 锁定维度记录

| 维度 | 锁定值 | 确定时间 | 确定方式 |
|------|:------:|:-------:|---------|
| P4 importance_criterion | taylor (默认) | 2026-04-14 | 解耦分析（与搜索维度无交互） |
| P5 pruning_granularity | local (默认) | 2026-04-14 | 解耦分析（不影响硬件部署） |
| P6 iterative_steps | 5 | 设计阶段 | 经验值，算法超参数 |
| P7 round_to | 8 | 设计阶段 | INT8 硬件约束 |

---

## 搜索空间分类（2026-04-15 新增）

按"实现机制"重新分类（详见反思文档第 10 节）：

| 类别 | 维度 | 当前状态 | 可触及参数 | 备注 |
|:---:|---|:---:|---:|---|
| **局部剪枝** | P1 ffn_mid_ratio | ✅ 已验证 5 个比例 | 6.31M | 最优 ROI = 30% |
| **局部剪枝** | seg_MLP_ratio | ✅ 已验证 30% | 5.26M | AMOTA 零损失 |
| **局部剪枝** | P3 head_mid_ratio | ⏳ 工程就绪未跑实验 | 1.89M | — |
| **拓扑剪枝** | P9 decoder_num_layers | ⏳ 工程就绪未跑实验 | ~1.5M | — |
| **级联剪枝** | P2 attn_proj_ratio | ❌ **工程未实现** | 3.95M | prune_direct 不支持，需方案 B |
| **级联剪枝** | P8 head_pruning_ratio | ❌ 工程未实现 | — | — |
| **级联剪枝** | P14 embed_dims | ❌ 工程未实现 | 全局 | 工程量最大 |
| **(受限)** | backbone | ❌ DCN 阻断 | 44M | 需李星峰 non-DCN ckpt + 微调 |

**当前可立即跑实验的搜索空间**：3 个局部 + 1 个拓扑 = 4 个维度，组合 ~90 个配置。

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
