# Plan 5 — 三新维度严格验证 (g8 plane sweep + 2:4 sparsity + TRT real INT8 AP) (/goal 长周期版)

> **版本**: v5 (2026-05-29, "破 plan v4 双轴 plateau" 直接推进版)
> **执行模式**: `/goal` 长周期, 中间不需用户介入 (除 Phase A → B 决策点 + Phase D 图 review)
> **预算**: ~52h wall (~6 day on RTX 4090 独占; 最坏 7.5 day)
> **完成判据**: 在 **g8 architecture** 上得到 **真实 lat × 真实 INT8 AP** 双轴 Pareto, R² ≥ 0.75; 或证明 g8 也 plateau (这样 plan v4 "intrinsic plateau" finding 升级为 architecture-agnostic, 更强 paper 主张)

---

## §0 必读前置 — 分层 (避免 context 爆炸)

**重要**: 6 day plan 远超单会话 context 寿命. 必须按 phase 分级 load, 不允许一次读全前置.

### §0.1 Tier 0 — 永久 in-context (整 plan 全程必须保留, 约 3.5k tokens)

**启动 + 每次 resume 都必读, 不允许 evict**:

1. **本档 plan5.md §0-§3 + 当前 phase 章节** (约 2.5-3k tokens)
2. **`plan5_state.json`** (~0.5k tokens, 当前进度 + 阈值 + 路径决策)
3. **`问题.md §7.13`** (~1k tokens, FT-固定纪律, 继承自 plan v4)

### §0.2 Tier 1 — Phase-specific load (进入 phase 时读, phase 完成后可 evict)

| Phase | 此 phase 必读 | tokens 估 |
|---|---|---|
| Phase 0 | `plan4_final_report.md` (双轴 plateau finding + 3 blindspot 总结) | 2k |
| Phase A | `问题.md §7.13` + HEAL ResNeXt width formula 推导 (`paper_learning/.../00_故事评估与实验路线_v1.md §相关`) | 2.5k |
| Phase B | TRT 2:4 sparsity 文档 (`Context7: nvidia/tensorrt §sparsity`) + `Ampere sparse tensor core` 简介 | 2k |
| Phase C | HEAL `opencood/tools/inference.py` 主流程 + voxelize-backbone-head 拆分 schema | 2.5k |
| Phase D | `e2e_bench_v1_schema.md §一-§九` (Pareto 集成列定义) | 2k |

### §0.3 Tier 2 — Lazy 引用 (出问题时才读, 平时不 load)

| 文件 | 何时 load |
|---|---|
| `AP不敏感解决方案_plan{1,2,3,4}.md` | 用户问 "之前为什么不行" 时, 否则只看本档 §2 历史反思 |
| `e2e_bench_v1.csv` (4584 行 ~600k token) | **永远不 load 进 context**. 用 pandas via Bash 读 |
| `stats_v3_plan4*/*.png` (AP+LAT 图) | 仅 Phase D 集成时取关键 2 张做 before/after 对比 |
| `CLAUDE.md` | session 启动时 system 自动注入, 不主动 read |

### §0.4 启动 checklist (agent self-verify)

启动 / resume 时 agent 自检:
- [ ] Tier 0 全部 in-context (`plan5_state.json` + 本档 §0-§3 + §7.13)
- [ ] 当前 phase 的 Tier 1 文件已 load
- [ ] **未** 把 e2e_bench_v1.csv / plan v1-4 全文 load (除非 Tier 2 触发)
- [ ] 知道下一步 = `plan5_state.json` 指向的 phase
- [ ] 不重复读 Tier 0 已经在 context 的章节

---

## §1 设计纪律 + 约束 (硬约束, 不可协商)

| # | 纪律 | 来源 | 违反后果 |
|---|---|---|---|
| C1 | **FT=8 锁定**, 全 plan 所有 anchor 同 FT epoch (g8 epoch 19→27, g32 epoch 23→31) | 问题 §7.13 (继承自 v4) | 又一次 plateau 误判 |
| C2 | LGB 特征**不含** `ft` 字段 | 问题 §7.10/7.13 | trivial R² 虚高 |
| C3 | LGB 特征**不含** `d` 字段 (除 lat predictor 阶段允许) | 问题 §7.12 | fold-leakage 假象 |
| C4 | 5-fold 必须 **GroupKFold by (architecture, plane)** | 新增 — 防 g8 同 plane 跨 Q 泄露 | 假 R² |
| C5 | **plane sweep 必须配 finetune** (不允许 mask-only zero-shot 测 AP) | 问题 §7 + 用户 2026-05-25 要求 | AP 跌报告无效 |
| C6 | TRT INT8 AP **必须从 engine forward** (不允许 fake quant proxy 替代) | 用户 2026-05-29 要求 | reporting failure (跟 M4.6 fp16-proxy-as-int8 同病) |
| C7 | 2:4 sparsity **必须 TRT --sparsity=enable build + 实测 wall-clock** | 用户 2026-05-29 要求 | sparsity gain 是 simulation 不是 deployment |
| C8 | 阈值预注册, 跑完按数字判, **不允许 post-hoc rationalize** | 继承 v4 C7 | 跟 plan v1-4 同病 |
| C9 | csv 写 `architecture` ∈ {g32, g8} + `sparsity` ∈ {dense, n2_m4} + `ap_source` ∈ {fake_quant_proxy, trt_engine_real} + `source=plan5_phaseX_pathY` | schema 扩展 | traceability 缺失 |
| **C10** | **latency bench 必须独占 GPU**: 启动前 hard guard 验证目标 GPU `util < 5%` ∧ `mem_used < 1 GB`. 不通过则 `RuntimeError` 拒绝, 不允许 `--no-isolation-check` bypass (除非用户明确授权). bench 期间禁止任何其他 CUDA 进程 (包括其他 anchor 的 finetune / inference / TRT build) | 用户 2026-05-29 要求 | 跟 finetune 共享 GPU → lat 数字虚高 50%+, paper §C 数据失效 |

---

## §2 历史经验与反思 (plan v4 双轴 plateau finding 的 3 blindspot)

### 2.1 Plan v4 主发现

- **AP 轴 plateau**: 215 anchor factorial 后 R²=0.388 (GroupKFold by triplet); 单 Q non-entropy 内 std≤0.025; entropy calibrator 触发的 10/21 triplet 随机崩溃跟 plane 无单调关系
- **LAT 轴 plateau**: bench v1 4584 anchor 用 kernel-aware proxy 拟合 R²=0.52 (<0.75 gate); cell-内 D std median 1.97 ms 但 B 维度跨 21 triplet std 仅 1.20 ms
- **结论 (v4 final)**: 双轴 plateau 是 paper §C 主要 scientific finding

### 2.2 Plan v4 后审查 — 3 个 blindspot (本 plan v5 设计动机)

| # | Blindspot | 证据 | Plan v5 对应 phase |
|---|---|---|---|
| B1 | **bench v1 全是 g32 architecture, plane 跨度被 width 公式锁死** (HEAL ResNeXt: width = int(plane × wpg / 64) × groups, wpg=4 时 plane<16 崩为 0). 21 triplet 实际 plane ∈ [16, 64], 跨度仅 4×, 不是真实剪枝跨度 (~20×) | plan v4 §A.3 + plan v3 collapse 实验 g8 extreme p=3 → lat 5ms vs p=64 → 11ms | **Phase A** g8 plane sweep |
| B2 | **没探索 GPU 真实可加速的 prune 模式 (2:4 sparsity)**, 只测了 channel-level structural pruning | TRT 8.x+ 支持 Ampere 2:4, 可叠加 INT8 再 1.7× | **Phase B** 2:4 sparsity |
| B3 | **INT8 AP 来自 PyTorch fake quant proxy, 不是 TRT engine 实跑**. fake quant 用 observer scale, TRT 用 calibrator scale + graph fusion, 数值可能差 0.5-2% | plan v4 phase A AP eval pipeline + 用户 2026-05-29 challenge | **Phase C** TRT engine real INT8 AP 闭环 |

### 2.3 共性诊断 — 5 次 plan 共同的循环 (plan v5 必须打破)

每次 plan 都基于一个"瓶颈在 X"的猜测直接跳实验. v1=FT, v2=B 极端, v3=数据加工, v4 草稿=Q. v4 实施版打破了"瓶颈直觉"循环, 用 factorial 找出 plateau. v5 必须打破"在已知 search space 内继续 finetune"的循环 — **直接扩 search space 的 3 个 architectural axis**.

---

## §3 整体路线图 (Phase 状态机)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Phase 0: Pre-flight (0.5 day)                                        │
│   • g8 baseline ckpt 健康检查 + calibration data ready 检查           │
│   • plan5_state.json 初始化                                          │
│   • H1/H2/H3 三假设 + 五阈值 pre-register                            │
│   完成信号: phase0_preflight.json 存在且 all_checks=PASS             │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase A: g8 plane sweep — 验证 H1 (lat smooth not step) (3 day)     │
│   • A.1: structural channel prune g8 baseline → plane=8/16/32/48      │
│   • A.2: finetune 1 epoch each (FT=8 → epoch 19→27)                  │
│   • A.3: bench TRT FP16/INT8 (3 Q variants × 3 D collapsed = 9 cells)│
│   • A.4: 4 plane × 9 cells = 36 new anchor; 加 baseline+extreme = 38  │
│   • A.5: fit smooth curve (3rd-order poly) lat vs plane, 计算 R²     │
│   Gate G_A: R²_smooth ≥ 0.85 → H1 confirmed (lat smooth)             │
│             R²_smooth < 0.85 → step jump 真存在, 触发 §6 path 2      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase B: 2:4 sparsity — 验证 H2 (sparsity 独立可叠加加速) (1.5 day)  │
│   • B.1: torch ASP (Automatic SParsity) 在 g8 baseline+4 prune 上    │
│         apply 2:4 mask + 1 epoch finetune                            │
│   • B.2: ONNX export + TRT --sparsity=enable build                   │
│   • B.3: bench 5 sparsified vs 5 dense, 同 (Q,D)                    │
│   • B.4: ANOVA SS(B_plane), SS(sparsity), SS(B:sparsity)             │
│   Gate G_B: sparsity-induced lat reduction ≥ 30% on at least 3/5    │
│             plane levels → H2 confirmed                              │
│             否则 → 2:4 deployment gain 不足, 触发 §6 path 3 (drop B) │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase C: TRT engine real INT8 AP — 验证 H3 (proxy gap) (2 day)      │
│   • C.1: hybrid pipeline: PyTorch voxelize → TRT INT8 backbone       │
│         → PyTorch head (含 NMS)                                      │
│   • C.2: 30 representative anchor (g8 5 plane × 3 Q × 2 D)           │
│         在 DAIR-V2X val n=1789 上跑 TRT engine forward, 算 AP        │
│   • C.3: 对比同 anchor 的 fake quant AP vs TRT real AP, 算 gap stats │
│   Gate G_C: median |AP_trt - AP_fake| ≤ 0.02 → H3 confirmed         │
│             gap > 0.02 → AP reporting 需切到 TRT real, 触发 §6 path 4│
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase D: Pareto integration + paper figure (0.5 day)                 │
│   • D.1: 整合所有 anchor (plan v4 g32 215 + plan v5 g8 38 + sparsity│
│         10 + real-AP 30 corrections) = ~290 anchor                   │
│   • D.2: 画 g8 vs g32 双 architecture Pareto frontier                │
│   • D.3: 算 dominate ratio (g8 anchors 在 (lat,AP) plane 上是否       │
│         dominate g32)                                                │
│   Gate G_D: g8 frontier 上至少 1 个 anchor 同时满足                  │
│             (lat < 0.7× g32 best) ∧ (AP ≥ 0.92 × g32 best AP)      │
│             → plan v5 主张 "g8 + (2:4 + INT8) 提供新 frontier"      │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Final report → plan5_final_report.md + paper §C 图候选 update         │
└──────────────────────────────────────────────────────────────────────┘
```

### §3.1 H1/H2/H3 三假设 + 五阈值 (Phase 0 必 register, 不允许 phase 跑完才改)

| Hypothesis | 阈值 | gate 名 | 失败 → path |
|---|---|---|---|
| H1: g8 上 lat vs plane 是 smooth (3rd-order poly fit) | R²_smooth ≥ 0.85 | G_A | path 2 (step cause 调研) |
| H2: 2:4 sparsity 提供独立可叠加 lat 加速 | reduction ≥ 30% on ≥3/5 plane | G_B | path 3 (drop sparsity dim) |
| H3: TRT real INT8 AP vs fake quant proxy gap 可忽略 | median \|gap\| ≤ 0.02 | G_C | path 4 (AP reporting 切换) |
| 衍生 D1: g8 Pareto dominate g32 | ∃ anchor: lat<0.7×g32_best ∧ AP≥0.92×g32_best | G_D | path 5 (双 architecture 共存论述) |
| 衍生 D2: cross-axis interaction (B × sparsity) negligible | SS(B:sparsity)/SS(total) < 5% | (诊断, 非 gate) | — |

---

## §3.5 上下文管理协议 (long-running /goal 关键)

继承 plan v4 §3.5, 核心 4 条:

### §3.5.1 每个 sub-task 用 background Bash + 文件持久化

不允许 agent 自己 `python script.py` 阻塞主 context. 必须:
```bash
mkdir -p /tmp/plan5_phaseA && \
nohup python scripts/phase2/plan5_phaseA_dispatcher.py \
  > /tmp/plan5_phaseA/dispatch.log 2>&1 &
```
然后用 `Bash run_in_background=true` + `Monitor` tool poll 完成信号 (file 存在 / json field set).

### §3.5.2 每个 phase 完成必写 summary file

不允许靠 agent context 记数据. 每 phase 结束写:
- `plan5_phase{0,A,B,C,D}_summary.md` (~1k tokens, 关键数字 + gate 判定 + path 选择)
- `plan5_state.json` update (phase status, last_completed_phase, current_phase, pareto_anchors_total)
- raw data → parquet/csv (NOT loaded into context)

### §3.5.3 resume 时只 reload Tier 0 + 当前 phase Tier 1

resume agent 自检:
```
1. read plan5_state.json → identify current_phase
2. read 本档 plan5.md §0-§3 + 当前 phase §X 章节
3. read 该 phase Tier 1 文件 (见 §0.2 表)
4. 不读已完成 phase 的 summary (除非需要 cross-phase reference)
5. 检测 "我已经读过 X" 不重复 read
```

### §3.5.4 错误 → 不重试 → 标 anchor failed → 继续

不允许卡在单个 anchor. anchor 失败:
- 写 `phaseX_anchors.csv` 该行 `status=failed, error=<msg>`
- 继续下一个 anchor
- phase 结束统计失败率, 失败率 > 20% → 整 phase 标 BLOCKED, 触发 §13 风险回退

---

## §4 Phase 0 — Pre-flight + 假设/阈值预注册 (0.5 day)

### §4.1 目的

- 验证 g8 baseline ckpt 可用, calibration data 可用, hybrid pipeline skeleton 可以 build
- 把 H1/H2/H3 三假设 + 五阈值写到 `plan5_state.json` (pre-register, 防 phase 跑完才改阈值)

### §4.2 实施 script

`scripts/phase2/plan5_phase0_preflight.py`:

```python
# 伪代码
def main():
    checks = {}

    # 1. g8 baseline ckpt 健康
    g8_ckpt = '/home/jichengzhi/heal_research/checkpoints/.../g8_baseline_epoch19.pth'
    checks['g8_ckpt_exists'] = Path(g8_ckpt).exists()
    checks['g8_ckpt_loadable'] = try_load_ckpt(g8_ckpt)

    # 2. g32 baseline ckpt 健康 (Pareto 集成用)
    g32_ckpt = '/home/jichengzhi/heal_research/checkpoints/.../Pyramid_m1_base_2023_08_14_04_28_12/'
    checks['g32_ckpt_exists'] = Path(g32_ckpt).exists()

    # 3. calibration data
    calib_bin = 'calibration/pyramid_calib.bin'
    checks['calib_data_ready'] = Path(calib_bin).exists() and Path(calib_bin).stat().st_size > 1e6

    # 4. TRT 版本 (>=8.5 supports --sparsity)
    trt_ver = subprocess.check_output(['trtexec', '--help']).decode()
    checks['trt_supports_sparsity'] = '--sparsity' in trt_ver

    # 5. torch ASP (NVIDIA Automatic SParsity) 可用
    try:
        from apex.contrib.sparsity import ASP  # NVIDIA APEX ASP
        checks['torch_asp_available'] = True
    except ImportError:
        checks['torch_asp_available'] = False  # 可能要 install apex

    # 6. DAIR-V2X val n=1789 ready
    val_root = '/home/jichengzhi/heal_research/dataset/DAIR-V2X-C/val'
    checks['val_n1789'] = count_samples(val_root) == 1789

    # 7. pre-register thresholds → plan5_state.json
    state = {
        'plan_version': 'v5',
        'created': '2026-05-29',
        'current_phase': 'phase_A',
        'last_completed_phase': 'phase_0',
        'pre_registered_thresholds': {
            'G_A_smooth_r2_min': 0.85,
            'G_B_sparsity_reduction_min': 0.30,
            'G_B_sparsity_pass_planes_min': 3,
            'G_C_ap_gap_max': 0.02,
            'G_D_dominate_lat_factor': 0.70,
            'G_D_dominate_ap_factor': 0.92,
        },
        'hypotheses': {
            'H1': 'g8 lat vs plane is smooth (poly3 R² >= 0.85)',
            'H2': '2:4 sparsity gives independent >=30% extra lat reduction on >=3/5 plane',
            'H3': 'TRT real INT8 AP vs fake quant proxy median gap <= 0.02',
        },
        'preflight_checks': checks,
        'all_checks_pass': all(checks.values()),
    }
    write_json('paper_learning/2. AAAI最终故事/data/plan5_state.json', state)
    write_json('.../plan5_phase0_preflight.json', checks)

if __name__ == '__main__': main()
```

### §4.3 Gate

- `all_checks_pass=True` → trigger Phase A
- 任一 fail → 写 `plan5_phase0_blocker.md`, plan halt, 等用户

---

## §5 Phase A — g8 plane sweep (3 day)

### §5.1 目的

验证 H1: g8 architecture 上 lat vs plane 是 smooth curve, 不是 step jump.

### §5.2 数据设计

| anchor | architecture | plane | finetune | Q variants | D variants | total cells |
|---|---|---|---|---|---|---|
| g8_p64 (baseline, 已有 epoch 19) | g8, wpg=16, groups=8 | 64 | 已 finetuned to baseline | 3 (fp16, int8_mm, int8_pc_wo) | 3 (D1_default, D24_BL0_default, D29_BL0_enableall) | 9 |
| g8_p48 (新) | 同上 | 48 | epoch 19→27 (8 epoch FT) | 3 | 3 | 9 |
| g8_p32 (新) | 同上 | 32 | epoch 19→27 | 3 | 3 | 9 |
| g8_p16 (新) | 同上 | 16 | epoch 19→27 | 3 | 3 | 9 |
| g8_p8 (新) | 同上 | 8 | epoch 19→27 | 3 | 3 | 9 |
| g8_p3 (extreme, 已有) | 同上 | 3 | 已 finetuned (plan v3) | 3 | 3 | 9 |

**总**: 6 plane × 3 Q × 3 D = **54 anchor** (4 plane 是新 finetune, 2 已有)

**注**: Q 收缩到 3 (排除 int8_ent 因 plan v4 发现随机崩, 排除 mix_s0/s2 因实验冗余). D 收缩到 3 (plan v4 §LAT 发现 32 D 是 step function, 取 3 cluster 代表).

### §5.3 实施 step

#### A.1 — structural channel prune (sequential, ~1h)

`scripts/phase2/plan5_phaseA_prune_g8.py`:
```python
# 对每个 plane in [48, 32, 16, 8]:
#   1. load g8 baseline ckpt
#   2. 按 L1 norm 选 top-N output channel
#   3. truncate weights to fit smaller plane
#   4. save as g8_p{plane}_pruned_unfinetuned.pth
```

#### A.2 — finetune 4 ckpt (parallel 1 GPU each, 4 GPU * 8 epoch * 2.5h/epoch = ~20h)

`scripts/phase2/plan5_phaseA_finetune.py`:
```python
# 4 ckpt finetune 同时, 用 nohup + GPU 分配:
# for plane in [48, 32, 16, 8]:
#     CUDA_VISIBLE_DEVICES=<gpu_idx> python opencood/tools/train.py \
#       --hypes_yaml lidar_pyramid_g8_p<plane>.yaml \
#       --resume <pruned_unfinetuned.pth> --epochs 8 ...
```

#### A.3 — bench 4 finetuned + 2 已有 (sequential, ~6h)

`scripts/phase2/plan5_phaseA_bench_dispatcher.py`:
```python
# for anchor in 6 plane × 3 Q × 3 D = 54:
#   1. ONNX export pyramid_backbone (1, 64, 256, 256) input
#   2. trtexec --onnx=... --saveEngine=... --<Q> --<D> ...
#   3. avgRuns=200 warmUp=200, 记 mean/p50/p99 lat
#   4. 写 anchor row 到 plan5_phaseA_anchors.csv
```

#### A.4 — smooth curve fit + gate 判

`scripts/phase2/plan5_phaseA_attribution.py`:
```python
# 1. load plan5_phaseA_anchors.csv (54 rows)
# 2. for each (Q, D) cell (9 cells):
#      fit 3rd-order poly: lat = a + b*plane + c*plane² + d*plane³
#      compute R²
# 3. weighted-average R² across 9 cells (weight = 1 since equal)
# 4. R²_smooth = mean R²
# 5. if R²_smooth >= 0.85: H1 confirmed → path 1
#    elif R²_smooth < 0.50: clear step → path 2a (zoom in, plane=4/6/10/12/14)
#    else: ambiguous → path 2b (再 finetune 4 中间点)
# 6. write plan5_phaseA_summary.md
```

### §5.4 Gate G_A

- `R²_smooth >= 0.85` → H1 confirmed, trigger Phase B (path 1)
- `0.50 <= R² < 0.85` → ambiguous, path 2b (~2 day 加测)
- `R² < 0.50` → step jump 真存在, path 2a (zoom in 找跳变点, ~2 day)

---

## §6 Phase A 出口决策树 (5 路径, 预注册)

### Path 1 — H1 confirmed (R²_smooth ≥ 0.85)

→ lat 是 smooth, "g8 上 prune 加速 linear" 假设成立
→ 进入 Phase B (2:4 sparsity)

### Path 2a — step jump 真存在 (R²_smooth < 0.50)

→ 加 4 中间 plane (4, 6, 10, 12) 重新 finetune + bench, 找物理跳变点
→ 跳变点确认后 (可能是 cuDNN kernel tactic 切换点 / Tensor Core alignment 切换点) → 进入 Phase B
→ 预算: +2 day, total Phase A 5 day

### Path 2b — ambiguous (0.50 ≤ R² < 0.85)

→ 加 2 plane (24, 40), 同 finetune, 看是否 fit smooth
→ 不强迫 fit, 接受 quasi-smooth (R²≈0.7) → 进入 Phase B
→ 预算: +1 day, total Phase A 4 day

### Path 3 — Phase B G_B FAIL (sparsity reduction <30% on most plane)

→ TRT 2:4 sparsity 在 PyramidFusion 上没 deployment gain (可能 conv 不够大触发 sparse tensor core)
→ drop 2:4 维度, 直接到 Phase C
→ 预算: -1 day

### Path 4 — Phase C G_C FAIL (real AP vs proxy gap > 0.02)

→ proxy AP 不准, 所有历史 anchor 的 INT8 AP 需要重测
→ 升级到全 anchor real AP eval (~290 anchor × 1789 sample = 重)
→ 预算: +2 day, total Phase C 4 day

### Path 5 — Phase D G_D FAIL (g8 不 dominate g32)

→ 接受 "g8 是 alternative architecture, 不 dominate", 论文论述改为 "两个 architecture 互补 Pareto"
→ 不增预算, 只改 narrative

---

## §7 Phase B — 2:4 sparsity 验证 (1.5 day)

### §7.1 目的

验证 H2: Ampere 2:4 structured sparsity 在 g8 上提供独立可叠加 lat 加速 (不只是 simulation, 是 TRT engine 真实测).

### §7.2 数据设计

| ckpt | sparsity | finetune | Q × D | new anchor |
|---|---|---|---|---|
| g8_p64 (baseline) | 2:4 mask | epoch 19→27 with ASP | 3 × 1 (D1 default) = 3 | 3 |
| g8_p48 | 2:4 mask | epoch 19→27 with ASP | 3 × 1 = 3 | 3 |
| g8_p32 | 2:4 mask | epoch 19→27 with ASP | 3 × 1 = 3 | 3 |
| g8_p16 | 2:4 mask | epoch 19→27 with ASP | 3 × 1 = 3 | 3 |
| g8_p8 | 2:4 mask | epoch 19→27 with ASP | 3 × 1 = 3 | 3 |

**总**: 5 plane × 3 Q × 1 D = **15 anchor sparsity**.
**对照**: 同 plane/Q/D 的 dense anchor (Phase A 里已有)
**比较**: lat_dense vs lat_sparse, per (plane, Q)

### §7.3 实施 step

#### B.1 — apply 2:4 mask + finetune (parallel 4 GPU, ~8h)

`scripts/phase2/plan5_phaseB_sparsify_finetune.py`:
```python
from apex.contrib.sparsity import ASP
# for each plane in [64, 48, 32, 16, 8]:
#     1. load Phase A finetuned ckpt
#     2. ASP.init_model_for_pruning(model, mask_calculator="m4n2_1d")
#     3. ASP.compute_sparse_masks()
#     4. finetune 1 epoch (sparsity-aware)
#     5. save as g8_p{plane}_sparse24_ft.pth
```

#### B.2 — ONNX export + TRT --sparsity build (sequential, ~2h)

```bash
# for each sparse ckpt:
#   1. export ONNX (mask 保留)
#   2. trtexec --onnx=... --saveEngine=... \
#               --sparsity=enable --int8 --calib=...
#   3. 记 engine size + build time
```

#### B.3 — bench 15 sparse cells + reuse 15 dense cells (~3h)

```python
# for each (plane, Q) cell, both sparse and dense:
#   trtexec --loadEngine=... --avgRuns=200 --warmUp=200
#   记 lat_p50
```

#### B.4 — ANOVA decomposition

`scripts/phase2/plan5_phaseB_attribution.py`:
```python
# 30 anchor (15 sparse + 15 dense) × 3 Q × 5 plane:
# ANOVA model: lat ~ plane + sparsity + plane:sparsity
# 计算:
#   SS(plane), SS(sparsity), SS(plane:sparsity), SS(error)
# 计算:
#   sparsity_reduction[plane, Q] = (lat_dense - lat_sparse) / lat_dense
# Gate G_B:
#   pass_planes = sum(1 for p in 5 plane if mean_Q reduction[p] >= 0.30)
#   if pass_planes >= 3: H2 confirmed → trigger Phase C
#   else: path 3 (drop sparsity, skip B 输出, 进 Phase C)
```

### §7.4 Gate G_B

- `pass_planes >= 3` (3/5 plane 的 sparsity reduction ≥30%) → H2 confirmed → Phase C
- `pass_planes < 3` → 2:4 deployment gain 不足, path 3 (drop B 输出, Phase D 不用 sparsity 维度)

---

## §8 Phase C — TRT engine real INT8 AP 闭环 (2 day)

### §8.1 目的

验证 H3: TRT INT8 engine forward 算出的 AP vs 我们一直用的 PyTorch fake quant proxy AP 的 gap. gap 小 → fake quant 是 valid proxy; gap 大 → 历史 plan v4 所有 INT8 AP 数字需重看.

### §8.2 hybrid pipeline 设计

PyramidFusion 完整 forward 拆 3 段:
```
input (raw point cloud)
  → [PyTorch] voxelize (sparse op, TRT unsupported)
  → voxel features (1, 64, 256, 256)
  → [TRT INT8 engine] pyramid_backbone forward
  → multi-scale features
  → [PyTorch] head + NMS (output bbox + score)
  → AP eval
```

`scripts/phase2/plan5_phaseC_hybrid_inference.py`:
```python
class HybridInfer:
    def __init__(self, trt_engine_path, head_ckpt_path):
        self.voxelize = HEAL_voxelize_op  # PyTorch sparse
        self.trt_engine = load_trt_engine(trt_engine_path)
        self.head = load_head(head_ckpt_path)  # PyTorch

    def __call__(self, raw_pc):
        voxel = self.voxelize(raw_pc)
        # PyTorch tensor → numpy → TRT input buffer
        # TRT execute_v2
        # TRT output buffer → PyTorch tensor
        feat = self.trt_forward(voxel)
        out = self.head(feat)
        return self.nms(out)

def eval_ap(engine_path, n=1789):
    infer = HybridInfer(engine_path, head_ckpt)
    preds, gts = [], []
    for sample in load_dair_val(n=1789):
        pred = infer(sample.pc)
        preds.append(pred); gts.append(sample.gt)
    return compute_ap_30_50_70(preds, gts)
```

### §8.3 测的 30 anchor (representative)

| plane | Q | D | sparsity | 数 |
|---|---|---|---|---|
| {64, 48, 32, 16, 8} (5) | {fp16, int8_mm, int8_pc_wo} (3) | D1_default (1) | dense (1) | 5×3×1×1 = 15 |
| {64, 32, 8} (3) | {int8_mm, int8_pc_wo} (2) | D1_default (1) | n2_m4 (1) | 3×2×1×1 = 6 |
| (extreme g8_p3) | {fp16, int8_mm, int8_pc_wo} (3) | D1_default (1) | dense (1) | 3 |
| (g32 重测 6 anchor 做 cross-architecture sanity) | {int8_mm, int8_pc_wo} (2) | D1_default (1) | dense (1) | 6 |

**总**: 30 anchor real AP eval

每个 anchor 跑 1789 sample, 每 sample ~50ms (含 voxelize) → ~90s/anchor → 30 anchor × 90s = ~45min wall

### §8.4 Gate G_C

`scripts/phase2/plan5_phaseC_ap_gap_analysis.py`:
```python
# 30 anchor:
#   ap_trt_real = eval_ap_trt(engine_path)
#   ap_fake_quant = lookup_phase_a_csv(plane, Q, D)  # plan v4/v5 phase A 存的
#   gap = ap_trt_real - ap_fake_quant
# stats:
#   median_gap, mean_gap, max_gap, |gap| > 0.02 count
# Gate G_C:
#   if median(|gap|) <= 0.02: H3 confirmed → Phase D 可以信任所有 fake quant AP
#   else: gap 大, path 4 (升级全 anchor real AP eval, 预算 +2 day)
```

---

## §9 Phase D — Pareto integration + paper figure (0.5 day)

### §9.1 整合所有 anchor

| 数据源 | 数 | architecture |
|---|---|---|
| plan v4 bench v1 (g32 主体) | 4584 | g32 |
| plan v4 phase A 215 (g32 finetuned) | 215 | g32 |
| **plan v5 Phase A** g8 dense | 54 | g8 |
| **plan v5 Phase B** g8 sparse (if G_B PASS) | 15 | g8 |
| **plan v5 Phase C** TRT real AP corrections | 30 | both |
| plan v3 g8 extreme | 2 | g8 |

**总**: ~290 unique anchor (Pareto 用 g8 + g32 各 ~30-50 个 representative)

### §9.2 Pareto frontier

`scripts/phase2/plan5_phaseD_pareto_integration.py`:
```python
# 1. load all anchor (lat_p50, ap_50, architecture, plane, Q, D, sparsity, ap_source)
# 2. Pareto: dominate(a, b) if a.lat < b.lat ∧ a.ap >= b.ap
# 3. frontier_g32 = pareto_g32_only()
# 4. frontier_g8 = pareto_g8_only()
# 5. frontier_combined = pareto_all()
# 6. compute g8 dominate ratio:
#    for anchor in frontier_combined:
#       if anchor.arch == 'g8': g8_count += 1
#    g8_ratio = g8_count / len(frontier_combined)
# 7. Gate G_D:
#    find best g32 anchor: g32_best = argmin(lat) s.t. ap >= 0.95 * max_ap
#    check if exists g8 anchor: lat < 0.7 * g32_best.lat AND ap >= 0.92 * g32_best.ap
#    if yes: G_D PASS, 'g8 + (sparsity + INT8) provides new frontier'
#    if no: G_D FAIL → path 5 (双 architecture 共存论述)
```

### §9.3 paper figure (D.3)

`scripts/phase2/plan5_phaseD_figures.py`:
```python
# Fig 1: lat vs ap scatter, color=architecture, marker=Q, size=plane
#        Pareto frontier 用粗线 over plot
# Fig 2: g8 plane sweep lat curve (Phase A 验证 H1) per (Q, D)
# Fig 3: 2:4 sparsity reduction histogram (Phase B 验证 H2)
# Fig 4: fake quant vs TRT real AP scatter (Phase C 验证 H3)
# Fig 5: Pareto frontier comparison g32-only vs g8-only vs combined
```

---

## §10 总时间预算

| Phase | wall | GPU 需 |
|---|---|---|
| 0 Pre-flight | 0.5 day (4h dev + checks) | 1 GPU |
| A g8 plane sweep | 3 day (1h prune + 20h finetune + 6h bench + 1h analysis) | 4 GPU 并发 finetune |
| B 2:4 sparsity | 1.5 day (8h ASP+FT + 2h TRT build + 3h bench + 1h analysis) | 4 GPU 并发 |
| C TRT real INT8 AP | 2 day (1 day hybrid pipeline + 45min eval + 1 day analysis) | 1 GPU + dev |
| D Pareto integration | 0.5 day (data + figures + report) | 0 GPU |

**总 wall**: ~7.5 day (含 dev overhead). 最快 6 day. 路径 2a 触发 → 9 day.

---

## §11 完成判据 (整 plan v5 finished)

任一以下成立:

| # | 判据 | 论文用途 |
|---|---|---|
| A | G_A PASS ∧ G_B PASS ∧ G_C PASS ∧ G_D PASS | paper §C 主图: "g8 + 2:4 + real INT8 提供新 Pareto frontier dominate g32" |
| B | G_A PASS ∧ G_C PASS ∧ (G_B FAIL ∨ G_D FAIL) | paper §C 副图: "g8 + INT8 是 alternative architecture, 提供互补 Pareto" |
| C | G_A FAIL (step jump confirmed) | paper §C 转 finding: "lat 在 plane 切换点有 architectural step, 找到具体 step" |
| D | G_A PASS ∧ G_C FAIL (大 gap) | paper §C 修正: "fake quant 不可靠, real INT8 AP 是必须报告 protocol" |
| E | 全部 FAIL | plan v5 升级 plan v4 finding: "intrinsic plateau 是 architecture-agnostic, 强主张" |

**任一情况都是 paper-publishable result**, 不是失败.

---

## §12 启动 / 暂停 / 恢复 操作手册 (/goal 用户接口)

### §12.1 启动

```
/goal paper_learning/2. AAAI最终故事/data/AP不敏感解决方案_plan5.md
```

agent 接受 /goal 后:
1. Read 本档 §0-§3
2. Read `plan5_state.json` (不存在则触发 Phase 0)
3. Read 当前 phase 章节
4. 启动当前 phase task (bg)
5. 用 Monitor poll completion

### §12.2 暂停

用户 `Ctrl+C` 或 close session:
- bg task 继续跑 (nohup)
- state 持久化在 `plan5_state.json`
- 下次 resume 不重跑

### §12.3 恢复

```
/goal paper_learning/2. AAAI最终故事/data/AP不敏感解决方案_plan5.md
```

agent resume:
1. Read state → identify current phase
2. Check bg task 是否还在跑 (`ps aux | grep plan5_phase`)
3. 跑完 → 进入下一 phase
4. 跑中 → Monitor poll
5. 失败 (status=failed) → 看 §13 风险回退

### §12.4 用户介入点

| 时机 | 用户需做 |
|---|---|
| Phase A 完成 G_A 判 | 0 介入 (path 1/2a/2b 自动选) |
| Phase B 完成 G_B 判 | 0 介入 (path 3 自动) |
| Phase C 完成 G_C 判 | 0 介入 (path 4 自动) |
| Phase D 图生成 | **1 介入**: 用户 review 5 张 figure, confirm acceptable → write paper §C |
| 任意 phase 失败率 > 20% | **1 介入**: 看 blocker.md, 决策是否 abort 或修 script |

---

## §13 风险 + 回退

| 风险 | 检测 | 回退 |
|---|---|---|
| g8 baseline ckpt 损坏 | Phase 0 ckpt_loadable=False | abort, 等用户重 train g8 baseline (~3 day, plan halt) |
| Apex ASP install 失败 | Phase 0 torch_asp_available=False | 用 `torch.nn.utils.prune` + manual 2:4 mask (gain 同等) |
| Finetune 不收敛 (loss NaN) | Phase A bg task log 含 "NaN" | 该 plane 标 failed, 跳过, 继续其他 plane. 失败 >2/4 → abort Phase A |
| hybrid pipeline 输出跟 PyTorch 不一致 | Phase C numerical check fail | debug 1 day max, 修不好 fallback to fake quant proxy (path 4 反之) |
| TRT --sparsity 不 build (engine size 不变) | Phase B build log 检测 sparsity flag accepted but engine size 同 dense | path 3 (drop B) |
| 全 phase 跑完 G_D FAIL | Phase D Pareto 检测 g32 dominate | path 5 (双 architecture 共存论述, plan 仍 success) |
| context 爆炸 (agent 多次重复读) | agent self-check §0.4 反复 fail | restart agent, 强制 evict 历史 phase summary |
| **bench pipe truncation 导致 partial engines** | bench bg task 用 `\| head -N` 截 stdout, 导致 python 进程 SIGPIPE 死, 但已写部分 engine 到 /tmp 污染数据 | bench 必须直接 `> file.log 2>&1` 不带 pipe; partial engines 清空重跑. **C10 隔离 guard 不够 — 还要 C10.b: bench bg 命令禁止 head/tail/grep 等 stdout-truncating filter** |
| **多用户系统其他用户占 GPU 误判 contention** | nvidia-smi 显示 8 GPU 都有 1-3GB residual, 但 plan5 没启动任何进程 | guard 用 nvidia-smi --query-compute-apps 看 pid, 排除 plan5 之外的 pid 占用; 只用 plan5 vacant 的 GPU. 若全 busy → abort, 等其他用户释放 |

---

## §14 对应实施脚本清单

| script | phase | 输入 | 输出 |
|---|---|---|---|
| `plan5_phase0_preflight.py` | 0 | — | `plan5_state.json`, `plan5_phase0_preflight.json` |
| `plan5_phaseA_prune_g8.py` | A.1 | g8 baseline ckpt | 4 pruned ckpt (g8_p48/32/16/8) |
| `plan5_phaseA_finetune.py` | A.2 | 4 pruned ckpt | 4 finetuned ckpt (epoch 19→27) |
| `plan5_phaseA_bench_dispatcher.py` | A.3 | 6 ckpt (4 new + 2 已有) | 54 anchor row → `plan5_phaseA_anchors.csv` |
| `plan5_phaseA_attribution.py` | A.4 | 54 anchor csv | `plan5_phaseA_summary.md` + gate 判 |
| `plan5_phaseB_sparsify_finetune.py` | B.1 | 5 Phase A ckpt | 5 sparse ckpt (g8_p64/48/32/16/8 + 2:4) |
| `plan5_phaseB_bench.py` | B.2-3 | 5 sparse ckpt | 15 anchor row → `plan5_phaseB_anchors.csv` |
| `plan5_phaseB_attribution.py` | B.4 | sparse vs dense csv | `plan5_phaseB_summary.md` + gate 判 |
| `plan5_phaseC_hybrid_inference.py` | C.1 | TRT engine + head ckpt | hybrid forward callable |
| `plan5_phaseC_ap_eval.py` | C.2 | 30 engine + DAIR val | `plan5_phaseC_real_ap.csv` |
| `plan5_phaseC_ap_gap_analysis.py` | C.3 | real AP + fake quant AP | `plan5_phaseC_summary.md` + gate 判 |
| `plan5_phaseD_pareto_integration.py` | D.1-2 | 全 anchor csv | `plan5_phaseD_pareto.parquet` |
| `plan5_phaseD_figures.py` | D.3 | pareto parquet | 5 PNG → `stats_v3_plan5/` |
| `plan5_phaseD_final_report.py` | D | 所有 phase summary | `plan5_final_report.md` |

---

## §15 修订历史

| 版本 | 日期 | 修订内容 |
|---|---|---|
| v5.0 | 2026-05-29 | 起草, 基于 plan v4 final report 3 blindspot |

---

## §16 跟 plan v4 的关系

- plan v4 → "在 g32 search space 内 factorial 找信号 → 双轴 plateau"
- plan v5 → "扩 search space 3 个 architectural axis (g8 / 2:4 sparsity / real INT8 AP) → 是否突破 plateau"
- 任一结果都是 paper §C 补强:
  - 突破 → "新 Pareto frontier"
  - 不突破 → "plateau is architecture-agnostic, 强主张"

plan v5 **不否定** plan v4, 是 plan v4 finding 的 architectural-extension 验证.
