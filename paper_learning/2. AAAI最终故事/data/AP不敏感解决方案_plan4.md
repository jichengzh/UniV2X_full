# Plan 4 — AP 信号瓶颈系统性归因 + 双轴 (AP + LAT) 验收 (/goal 长周期版)

> **版本**: v4 (2026-05-27 下午, FT-固定 修订版)
> **执行模式**: `/goal` 长周期, 中间不需用户介入 (除 Phase A → B 决策点 + Phase D 图 review)
> **预算**: ~22h wall (~2.5 day on 6 GPU 独占; 最坏 3.5 day)
> **完成判据**: AP 5-fold GroupKFold R² ≥ 0.65 **或** 证明 plateau intrinsic 走 reframe path (失败也算成功 — 是 paper §C 主要发现)

---

## §0 必读前置 — 分层 (避免 context 爆炸)

**重要**: 22-36h plan 远超单会话 context 寿命 (~200k tokens). agent 不可一次读全 6 份前置文档 → 必定中途 context 满 → 状态丢失. 按 phase 分级 load.

### §0.1 Tier 0 — 永久 in-context (整 plan 全程必须保留, 约 3k tokens)

**启动 + 每次 resume 都必读, 不允许 evict**:

1. **本档 plan4.md §0-§3 + 当前 phase 章节** (约 2-3k tokens)
2. **`plan4_state.json`** (~0.5k tokens, 当前进度 + sigma_FT8 + 路径决策)
3. **`问题.md §7.13`** (~1k tokens, FT-固定纪律, 防违规)

### §0.2 Tier 1 — Phase-specific load (进入 phase 时读, phase 完成后可 evict)

| Phase | 此 phase 必读 | tokens 估 |
|---|---|---|
| Pre-Phase 0 | `问题.md §7.11` (FT=4/6 noise 历史) | 1.5k |
| Phase A | `问题.md §7.10 + §7.12` (Tier B 分离 + D fold-leakage) | 3k |
| Phase B 决策 | 本档 §6 决策树 | 1k |
| Phase C | 本档 §7 + 路径 selected 子节 | 1.5k |
| Phase D | `e2e_bench_v1_schema.md §一-§九` (列定义 + entropy 崩塌警告) | 2k |
| Phase E | `数据集制作_plan §五` (Stage 2e D 网格 rationale) | 1.5k |

### §0.3 Tier 2 — Lazy 引用 (出问题时才读, 平时不 load)

| 文件 | 何时 load |
|---|---|
| `AP不敏感解决方案_plan.md` / `_plan2.md` / `_plan3.md` | 用户问 "之前为什么不行" 时, 否则只看本档 §2 历史反思 |
| `e2e_bench_v1.csv` (4584 行 ~600k token) | **永远不 load 进 context**. 用 `scripts/phase2/dataset_stats_report_bench.py` 或 pandas via Bash 读 |
| `stats_v3_bench/*.png` (14 张图) | Phase D user gate 才读 (3 张关键图) |
| `CLAUDE.md` | session 启动时 system 自动注入, 不主动 read |

### §0.4 启动 checklist (agent self-verify)

启动 / resume 时 agent 自检:
- [ ] Tier 0 全部 in-context (`plan4_state.json` + 本档 §0-§3 + §7.13)
- [ ] 当前 phase 的 Tier 1 文件已 load
- [ ] **未** 把 e2e_bench_v1.csv / 14 PNG / 历史 plan 全文 load (除非 Tier 2 触发)
- [ ] 知道下一步 = `plan4_state.json` 指向的 phase
- [ ] 不重复读 Tier 0 已经在 context 的章节 (检测 "我已经读过 X")

---

## §1 设计纪律 + 约束 (硬约束, 不可协商)

| # | 纪律 | 来源 | 违反后果 |
|---|---|---|---|
| C1 | **FT=8 锁定**, 全 plan 所有 anchor 用同一 FT epoch | 问题 §7.13 | 又一次 plateau 误判, plan 失败 |
| C2 | LGB 特征**不含** `ft` 字段 | 问题 §7.10/7.13 | trivial R² 虚高 |
| C3 | LGB 特征**不含** `d` 字段 (D 对 AP 零信号) | 问题 §7.12 | fold-leakage 假象 |
| C4 | 5-fold 必须 **GroupKFold by triplet** | 问题 §7.12 | D 复制 / triplet 复制泄露 |
| C5 | AP 主表 anchor **D = D1_default_4gb 单档** | 问题 §7.12 | D-rich 是 lat 用, AP 用 dedup |
| C6 | 所有 AP eval 用 **n=1789 DAIR val 全量** | schema §〇 | 跟 e2e_bench_v1 兼容 |
| C7 | 阈值预注册, 跑完按数字判, **不允许 post-hoc rationalize** | 用户 2026-05-27 要求 | 跟 plan v1/v2/v3 同病 |
| C8 | csv 写 `finetune_epochs=8` 列 + `source=plan4_phaseX_pathY` | schema §3.4 | traceability 缺失 |

---

## §2 历史经验与反思 (4 次尝试的 plateau)

### 2.1 Plan v1 (2026-05-21) — 失败模式 "FT sweep 假设"

- **假设**: "FT 太多抹平 AP, 找 FT* sweet spot 就行"
- **做法**: pilot 12 anchor, 3 triplet × 5 FT (4/6/8/10/15)
- **结果**: FT∈[4,15] × prune∈[14%,89%] g32 全部 plateau, std < 0.025 → 早停
- **教训**:
  - FT sweep 是 dataset 设计错误 (FT 不是 search axis, 是 deployment config)
  - 5 个 FT 档跑出 plateau 不证明"无信号", 只证明"FT 选 sweet spot 这条路不通"

### 2.2 Plan v2 (2026-05-22) — 失败模式 "B 极端必崩"

- **假设**: "g32 + 89% prune 不够极端, 改 g8 + (4,4,4) 必崩"
- **做法**: 架构改造 groups=32→8, 推 prune 到 97% (planes=4,4,4)
- **结果**: ✅ 找到 B×Q **协同** collapse (p97 + int8_pc_wo + FT=8 = 0.455, vs FP32 0.506 = -5.1pp), 但**只在窄窗口**
- **教训**:
  - 极端 B 单独不崩 (FP32 在 p97 = 0.506, plateau 内)
  - 真信号在 **B × Q 交互**, 不是 B main effect
  - 窄窗口 ≠ R² 高 (plan v3 实测 R²=0.56)

### 2.3 Plan v3 (2026-05-25) — 失败模式 "数据加工就够"

- **假设**: "FT 0/2 underfit anchor 扩 collapse 信号 + D 维度补 36 anchor → predictor R² 上去"
- **做法**: 84 anchor (含 FT=0/2 collapse + D-tactic + ft_sweep)
- **结果**: ✅ predictor R²=0.94 — **后审查发现是 FT 当特征导致的 trivial 信号**, 真实 deployment R²=0.56
- **教训**:
  - 把 FT=0/2 / D 全堆进 dataset 给 LGB → 假信号
  - **deployment-realistic 分离 (Tier B FT∈{8,15})** 才是真信号视图
  - R²=0.56 是 N=15 的 ceiling, 数据量小不是主因

### 2.4 Plan v4 草稿 (2026-05-27 上午) — 失败模式 "Q 是瓶颈"

- **假设**: "Q 7-level 太粗, 扩 16 variants 就有信号"
- **做法**: 16 Q × 21T = 336 anchor 设计
- **结果**: 用户驳回, **没证据 Q 是瓶颈, 又一次主观判断**
- **教训**:
  - 跳到 "Q 是瓶颈" 跟 plan v1 跳到 "FT 是瓶颈" 同病
  - 必须**先 factorial 归因, 后挑解决方案**

### 2.5 共性诊断 — 4 次失败的根因

**每一次 plan 都基于一个未验证假设直接跳到解决方案**. 4 次假设都不同 (FT/B/Tier 分离/Q), 但 process 一样:
1. 看到 plateau
2. 凭直觉挑一个因子说 "它是主导"
3. 设计 fix-this-factor 实验
4. 跑完发现 plateau 还在
5. 换下一个因子, GOTO 1

**plan v4 必须打破这个循环**: 先做严格 factorial 归因, **数字告诉你哪个因子主导**, 再 fix.

---

## §3 整体路线图 (Phase 状态机)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Pre-Phase 0: σ_FT8 实测 (5 seed × T_g8_p97 × FT=8)                   │
│   完成信号: phase0_noise.json 含 sigma_noise + r2_ceiling           │
│   trigger Phase A: σ_FT8 < 0.04 (否则 abort, plan 失败)              │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase A: B × Q 因素归因 (128 anchor at FT=8 锁定)                    │
│   A.1 B main (24 anchor) → A.2 Q main (15) → A.3 B×Q grid (99)      │
│   完成信号: phaseA_attribution.md 含 SS(B)%, SS(Q)%, SS(B:Q)%       │
│   trigger Phase B: 决策树 (§6) 自动选路径                            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
                  ┌───────────┼───────────┬──────────┐
                  ↓           ↓           ↓          ↓
              路径 1       路径 2a     路径 2b     路径 3       路径 4
            (B 主导)     (Q 信号)    (Q 扩 16)   (B×Q 网格)  (reframe)
                  ↓           ↓           ↓          ↓          ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase C: 按选定路径采集数据                                            │
│   完成信号: plan4_phaseC_anchors.csv 含 ≥95% 路径目标 anchor         │
│   sanity: 抽 5 anchor 跟 bench v1 对照 |Δ AP50| ≤ 0.005             │
│   trigger Phase D: csv 写好 + sanity 通过                            │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ Phase D: AP 图 + LGB v9_ap 训练 + 5 gate 验收                        │
│   完成信号: 14 PNG + lgb_v9_ap.txt + plan4_phaseD_report.md         │
│   trigger Phase E: 5 gate 全部通过 (或路径 4 已 reframe)             │
│   USER GATE: 用户 review 3 张关键图 (B1/D1/D3) 通过                   │
└──────────────────────────────────────────────────────────────────────┘
                              ↓ (可与 Phase D 并行)
┌──────────────────────────────────────────────────────────────────────┐
│ Phase E: LAT predictor 补强 (跟 AP 独立, 不依赖 FT)                   │
│   E.1 多 build sanity → E.2 D 网格补 → E.3 kernel-aware feature      │
│   完成信号: lgb_v9_lat.txt + lat 3 gate 通过                         │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ FINAL: 整体出口判定 (§11)                                              │
│   成功 path 1: AP gate 通过 + LAT gate 通过                          │
│   成功 path 2: AP reframe (path 4) + LAT gate 通过                   │
│   失败: 两轴都不达 + 不接受 reframe                                    │
└──────────────────────────────────────────────────────────────────────┘
```

**状态文件**: `paper_learning/2. AAAI最终故事/data/plan4_state.json`. agent 每个 phase 完成后更新, 启动时读它决定从哪继续.

```json
{
  "phase_0_done": false,
  "phase_0_progress": null,    // {"seeds_done": 0..5}
  "sigma_ft8": null,
  "threshold_signal": null,    // 0.06 if σ<0.02, 0.10 if 0.02-0.04

  "phase_a_done": false,
  "phase_a_progress": null,    // {"anchors_done": 0..128, "last_anchor_idx": int}
  "phase_a_attribution": null, // {"ss_b": ..., "ss_q": ..., "ss_bq": ...}
  "phase_b_path": null,        // 1 / 2a / 2b / 3 / 4
  "phase_b_path_rationale": "",

  "phase_c_done": false,
  "phase_c_progress": null,    // {"anchors_done": int, "target": int, "last_checkpoint_ts": iso}
  "phase_c_anchors": null,

  "phase_d_done": false,
  "phase_d_ap_r2": null,
  "phase_d_5gate": {},
  "phase_d_user_review": null, // null / "approved" / "redo"

  "phase_e_done": false,
  "phase_e_progress": null,    // {"sub": "E.1/E.2/E.3/E.4", "anchors_done": int}
  "phase_e_lat_r2": null,
  "phase_e_3gate": {},

  "context_compaction_count": 0,  // 触发 compaction 次数
  "last_session_id": "",            // 会话 ID, resume 时校验
  "user_abort": false,

  "final_verdict": null        // "success_path_1" / "success_path_2" / "fail"
}
```

---

## §3.5 上下文管理协议 (long-running /goal 关键)

> **核心矛盾**: plan v4 wall 22-36h, 单会话 context 寿命 ~200k tokens (10-30 hops 后开始压缩). **不主动管理 context = plan 必失败**.

### 3.5.1 三道防线

**防线 1 — 分层文件 load (§0)**: 任何时刻 in-context 文件 ≤ Tier 0 + 当前 phase Tier 1 = 总量 ≤ 8k tokens. 永不 load Tier 2 (除非显式触发).

**防线 2 — 增量 checkpoint (每 N anchor)**: phase 内部不一次性跑完再写状态, 而是每 N=10 anchor 写一次 `plan4_state.json.phase_x_progress`. Resume 时 agent 从 progress 续, 不重跑.

**防线 3 — Phase summary 替代原始数据**: 每完成一 phase, agent 写 `plan4_phase{N}_summary.md` (50-100 行, 含数字结论 + 关键决策). 跨 phase 引用从 summary 读, 不读原始 csv/JSON.

### 3.5.2 子任务委托 (delegate to bash background)

**主 agent 不直接跑长任务**, 通过 Bash run_in_background + Monitor:

| 长任务 | 委托方式 | 主 agent 看什么 |
|---|---|---|
| Phase A 128 anchor 并行 dispatcher | `bash run_in_background` (predicted 1.5h) | tail progress log (每 10 anchor 1 行) |
| HEAL finetune (Pre-Phase 0 / 路径 1) | `bash run_in_background` 单进程跑 | Monitor `loss 收敛` + `epoch 数` |
| TRT engine build + AP eval pool | dispatcher 内部 multiprocessing.Pool | 只看 dispatcher stdout, 不看 per-anchor log |
| Phase E.2 1764 anchor lat | `bash run_in_background` (predicted 7h) | 每 100 anchor checkpoint, 主 agent 醒 1 次 check |

主 agent 在子任务跑期间**不保持 100% busy**, 用 `Bash run_in_background` 后立即写 state.json `last_dispatched_at`, 然后**主动 sleep** (ScheduleWakeup 或退出循环), 等子任务通知再 wake.

### 3.5.3 Phase 完成时的 self-compaction

每个 phase 结束 (无论成功/失败), agent **强制执行**:

1. 写 `plan4_phase{N}_summary.md` — 含:
   - 一句话结论
   - 关键数字 (3-5 个)
   - 触发的下一 phase 路径决策
   - 失败 anchor 列表 (如有)
2. 更新 `plan4_state.json`
3. **主动 evict** 已完成 phase 的 Tier 1 文件 (从 context 中删除引用)
4. session_id 记录到 `last_session_id`, 用于跨会话校验

### 3.5.4 跨会话 resume 协议 (严格 3 步)

agent 收到 `/goal plan4.md` 时:

```
Step 1: 只 read plan4_state.json + plan4.md §0-§3 + §7.13 (Tier 0)
Step 2: 根据 state.json 找当前 phase, 找 phase_progress
Step 3: read 该 phase 章节 (Tier 1) + phase_{N-1}_summary.md (跨 phase 引用)
        然后 resume 工作
```

**严禁** Step 0 "我先把 6 份前置全读一遍" → 这是 plan v4 草稿的设计错误.

### 3.5.5 Context 接近上限的检测 + 主动 compaction

主 agent 每完成一个**子任务 dispatch** (不是每个 anchor) 后 self-check:
- 如果当前 conversation hop 数 > 30, **主动**写 mini-summary (10-20 行) + clear in-context 临时数据
- 触发条件: 完成 sub-phase OR > 10k tokens 临时输出 OR 收到 hook 警告

self-check 写到 state.json `context_compaction_count++`.

### 3.5.6 跨 phase 引用规则

如果 phase X 需要引用 phase Y 的数据 (Y < X):

| 引用类型 | 怎么做 |
|---|---|
| **数字结论** (R², σ, MAE) | 从 `plan4_state.json` 读 |
| **路径决策依据** | 从 `phase_{Y}_summary.md` 读 |
| **某 anchor 的详细 AP** | 从 `phase_{Y}_anchors.csv` 用 pandas (Bash) 读, 只 print 需要的 row |
| **图片** | 通过 Read tool 加载具体 PNG, 用完不 keep in context |

**绝不允许**: 把整个 phase_{Y}_anchors.csv (~500k tokens) load 进 context.

### 3.5.7 失败 + degrade 处理

如果某 phase 失败 (build fail / sanity fail / gate fail):
- agent 不立即 retry, 先写 `phase_{N}_failure_report.md` (50 行)
- 更新 state.json 标 fail + reason
- 决定: 是回退 (Phase B 路径 4 reframe) 还是 abort
- **不允许** 在主 context 里堆 100+ 行错误 stacktrace; stacktrace 写到 logs 目录

---

## §4 Pre-Phase 0 — σ_FT8 实测 + 阈值锁定

### 4.1 启动信号

- agent 启动: `plan4_state.json` 不存在或 `phase_0_done == false`
- 6 GPU 可用 (`nvidia-smi` 确认 GPU 2-7 中至少 5 个空闲, util < 20%)

### 4.2 目标

实测 FT=8 在 T_g8_p97 上的 5-seed AP 方差, 锁 Phase A "有信号"判据阈值.

### 4.3 实验设计

- **5 seed**: PYTHONHASHSEED ∈ {0, 1, 2, 3, 4} (复用 plan v3 §1.3 协议)
- **起点 ckpt**: `models/dataset_a_cache_g8/noise_ft6_p97_seed{N}/net_epoch25.pth` (FT=6 已训, 续训 2 epoch → FT=8)
- 每 seed:
  - 续训 2 epoch (epoch 25 → 27) on 1 GPU, ~10 min
  - Export ONNX
  - Build TRT FP32 engine
  - AP eval n=1789
- 输出: 5 个 AP50 + mean + std + r2_ceiling = 1 - σ²/std² (with N=15 plan v3 reference data)

### 4.4 实现

新建 `scripts/phase2/plan4_phase0_ft8_noise.py`:

```python
# 伪代码
seeds = [0, 1, 2, 3, 4]
for seed in seeds:
    src_ckpt = f"models/dataset_a_cache_g8/noise_ft6_p97_seed{seed}/net_epoch25.pth"
    out_dir = f"models/dataset_a_cache_g8/noise_ft8_p97_seed{seed}"
    # continue training: epoch 25 → 27 (2 epoch)
    run_heal_finetune(src_ckpt, out_dir, n_epoch=2, seed=seed)
    # eval
    run_e2e_eval_ap(out_dir + "/net_epoch27.pth", out_dir + "/ap.json")

aps = load_5_aps()
sigma = np.std(aps)
mean = np.mean(aps)
save_json("plan4_state.json", {
    "phase_0_done": True,
    "sigma_ft8": sigma,
    "threshold_signal": 0.06 if sigma < 0.02 else (0.10 if sigma < 0.04 else None)
})
```

### 4.5 完成信号

- `plan4_state.json` 写入 (phase_0_done=true, sigma_ft8, threshold_signal)
- `models/dataset_a_cache_g8/noise_ft8_p97_seed{0..4}/ap.json` 5 个文件存在
- `phase0_noise.md` 报告写好

### 4.6 trigger 下一 Phase (Phase A)

- **σ_FT8 < 0.02**: threshold_signal = 0.06, 启动 Phase A
- **0.02 ≤ σ_FT8 < 0.04**: threshold_signal = 0.10 (放宽), 启动 Phase A 但 Phase B 路径 4 (reframe) 概率高
- **σ_FT8 ≥ 0.04**: **abort plan v4**, 写 `phase0_abort_report.md`, 通知用户 — FT=8 不够稳, 项目 finetune-equilibrium 假设需重审

### 4.7 预算

| step | wall | GPU |
|---|---|---|
| 5 seed × 2 epoch finetune | 10 min | 5 GPU |
| 5 seed × ONNX export + TRT build | 5 min | 5 GPU |
| 5 seed × AP eval n=1789 | 5 min | 5 GPU |
| 统计 + 状态文件 + 报告 | 5 min | 0 |
| **Total** | **~30 min** | |

---

## §5 Phase A — B × Q 因素归因 (128 anchor @ FT=8 锁定)

### 5.1 启动信号

- `plan4_state.json.phase_0_done == true`
- `phase_0_noise.json.threshold_signal` ∈ {0.06, 0.10} (非 abort)
- 6 GPU 可用

### 5.2 目标

回答 3 个二元问题 (跟 σ_FT8 阈值联动):
- Q1: B (24 triplet @ FT=8, Q=fp32) 是否给 std(AP) > threshold_signal?
- Q2: Q (5 variants @ 3 triplet × FT=8) 是否给 std(AP, 排除 int8_ent) > threshold_signal?
- Q3: B × Q grid (24T × 5Q @ FT=8) LGB 拟合的 SS attribution 是否任一 > 30%?

### 5.3 实验矩阵

| sub-exp | N anchor | 因子 | 控制变量 |
|---|---|---|---|
| A.1 B main | 24 | 24 triplet 跨 prune 0%-97% | Q=fp32, FT=8, D=D1 |
| A.2 Q main | 15 | 5 Q variants on 3 anchor | T1_base / T22_p89 / T_g8_p97, FT=8, D=D1 |
| A.3 B × Q grid | 120 (dedup 99 new) | 24T × 5Q full grid | FT=8, D=D1 |

### 5.4 Anchor 实现细节

**ckpt 路径** (FT=8 锁定):

| triplet | ckpt 路径 | 备注 |
|---|---|---|
| T1_base | `/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29/net_epoch_bestval_at23.pth` | 无 prune 无 FT, 用 bestval |
| T3/T5/T7/T8/T10-T22 (17 Track A v2 g32) | `models/dataset_a_cache/ft_*/net_epoch31.pth` | **epoch 31** = baseline 23 + FT 8 |
| T2_p25/T4_p50/T6_p75 (3 HEAL 官方 g32) | 官方只到 epoch 25 (=FT 2), **补训** 25→31 至 `net_epoch31.pth` | epoch 31 = baseline 23 + FT 8 |
| T_g8_p87/p93/p97 (3 g8) | `models/dataset_a_cache_g8/ft_T_g8_p*/net_epoch27.pth` | **epoch 27** = g8 baseline 19 + FT 8 |

> ⚠️ **架构 baseline 不同**: g32 = epoch 23 (HEAL stage1), g8 = epoch 19 (plan v2). FT=8 → g32 epoch 31, g8 epoch 27. 不要混用.

**Q 5 variants**: fp32, fp16, int8_pt_wa_mm, int8_pc_wo, int8_pt_wa_ent (跟 bench v1 一致)

**D 单档**: D1_default_4gb (TRT default tactic + 4GB workspace + BL=3)

### 5.5 Engine cache 复用策略

- bench v1 的 engine 用的是 bestval ckpt, 跟 epoch 27 不同 → **必须重新 export ONNX + build engine** for T2-T22
- T1_base 已是 bestval = epoch 23, engine 可直接复用 (但需确认 cache hit)
- T_g8_p* engine 已有 (plan v2 时跑过)
- 估: 21 个新 ONNX export + 21 × 5 Q = 105 个新 engine build

### 5.6 实现

新建 `scripts/phase2/plan4_phaseA_dispatcher.py`:

```python
# 伪代码 — 复用 a11_eval_ft_sweep.py 的 6 GPU pool 结构
triplets = [
    ("T1_base", "<bestval_ckpt>"),
    *[(f"T{i}_*", f"<dataset_a_cache>/ft_<sig>/net_epoch27.pth") for i in 2..22],
    ("T_g8_p87", "<g8>/ft_T_g8_p87/net_epoch27.pth"),
    ("T_g8_p93", "<g8>/ft_T_g8_p93/net_epoch27.pth"),
    ("T_g8_p97", "<g8>/ft_T_g8_p97/net_epoch27.pth"),
]
q_variants = ["fp32", "fp16", "int8_pt_wa_mm", "int8_pc_wo", "int8_pt_wa_ent"]

# Step 1: pre-export 24 ONNX (sequential)
for trip, ckpt in triplets:
    if not exists(f"models/plan4_cache/{trip}.onnx"):
        export_onnx_pyramid_e2e(ckpt, ...)

# Step 2: 6 GPU parallel — 24 × 5 = 120 anchor
specs = [(trip, q) for trip in triplets for q in q_variants]
results = parallel_pool_run(specs, n_gpu=6, fn=run_one_anchor)

# Step 3: 写 phase_a_anchors.csv (24*5=120 rows)
write_csv("phase_a_anchors.csv", results)

# Step 4: attribution analysis
sub_a1 = filter(q="fp32")  # 24 rows
sub_a2 = filter(triplet in ["T1_base", "T22_p89", "T_g8_p97"])  # 15 rows
sub_a3 = all  # 120 rows

# A.1 B main effect
std_b = sub_a1["ap50"].std()

# A.2 Q main effect (排除 int8_ent)
sub_a2_clean = sub_a2[sub_a2["q"] != "int8_pt_wa_ent"]
std_q_per_triplet = sub_a2_clean.groupby("triplet")["ap50"].std()

# A.3 ANOVA on full grid
# fit LGB on planes_s1/s2/s3 + Q one-hot, GroupKFold by triplet
ss_b, ss_q, ss_bq, ss_err = anova_decompose(sub_a3)

# 选 Phase B 路径
path = decide_phase_b_path(std_b, std_q_per_triplet, ss_b, ss_q, ss_bq, threshold)
save_state({"phase_a_done": True, "phase_a_attribution": {...}, "phase_b_path": path})
```

### 5.7 完成信号

- `plan4_phaseA_anchors.csv` (120 行, 含 build_success + ap30/50/70)
- `phaseA_attribution.md` (写 SS%, std, ANOVA F-test, 选定路径)
- `plan4_state.json` 更新 (phase_a_done=true, phase_a_attribution, phase_b_path)
- 至少 95% 个 anchor build_success=true (允许 5-6 个失败)

### 5.8 trigger 下一 Phase (Phase B 决策)

详 §6 决策树.

### 5.9 预算

| step | wall | GPU |
|---|---|---|
| Step 1: 21 个新 ONNX export | 25 min | 1 GPU |
| Step 2: 120 anchor 真测 (engine build + AP eval) | 1h | 6 GPU 并行 |
| Step 3: 写 csv | 5 min | 0 |
| Step 4: ANOVA + 路径决策 | 15 min | 0 |
| **Total** | **~1.5h** | |

---

## §6 Phase A 出口决策树 (4 + 1 路径, 预注册)

跑完 Phase A 后, agent **不允许 hand-pick**, 必须按下表自动选路径:

| 条件 (按优先级 from top) | Path | 描述 |
|---|---|---|
| `std(A.1 B fp32) > 2 × threshold` AND `SS(B) / SS(total) > 0.30` | **1** | B 主导单因素 |
| `std(A.2 Q, 排除 ent) > threshold` in ≥ 2 triplet AND `SS(Q) > 0.30` | **2a** | Q 现 7-level 已有信号, 不扩 Q |
| `SS(B:Q) / SS(total) > 0.30` AND `SS(B) < 0.30` AND `SS(Q) < 0.30` | **3** | B × Q 协同主导 |
| 仅 int8_ent 在 A.2 出现崩塌 + `SS(Q) > 0.30 但主要来自 ent crash dummy` | **2b** | 扩 Q 到 16 variants (跨 granularity / mix / head) |
| Phase A LGB GroupKFold R² < 0.40 AND 所有 SS < 0.30 | **4** | reframe paper: plateau intrinsic |

**优先级**: 如果同时满足多个, **取首先匹配** (路径 1 优先于 2a, 等). 路径 4 是默认回退.

### 6.1 路径 1 — B 主导 (扩 B 到 28+ triplet)

**目标**: 加 5 个 g8 中间档 (p85/p90/p95/p98/p99), 把 B 跨度更均匀, 训 v9_ap.

**Anchor**: 5 新 g8 triplet × 7 Q × FT=8 × D=D1 = **35 新 anchor** (加现有 120 = **155 total**)

**新工作**:
- 5 个新 g8 structural prune (~10 min each)
- 5 个新 g8 finetune to FT=8 from g8 baseline raw (~30 min each, parallel 5 GPU)
- 5 × 7 = 35 engine build + AP eval

**预算**: 30 min prune + 30 min finetune + 1.5h eval = **2.5h**

### 6.2 路径 2a — Q 现 7-level 有信号 (重出 dataset)

**目标**: 用 24 triplet × 7 Q × FT=8 × D=D1 = **168 anchor** 替换 bench v1 (bestval) AP 列.

**新工作**: 24 × (7-5) = 48 anchor 补 (mix_s0/mix_s2 不在 Phase A 5 Q 里)

**预算**: 48 × 5 min / 6 GPU = **45 min**

### 6.3 路径 2b — Q 扩 16 variants

**目标**: 24 triplet × 16 Q × FT=8 × D=D1 = **384 anchor**

**16 Q spec** (详细见 plan v4 前一版本 § 一.1 表):
- 基础 7 (现有)
- 新 9: int8_pt_wo_mm, int8_pc_wa_mm, int8_pt_wa_kld, int8_pt_wa_pct, mix_s1, mix_s01, mix_s02, mix_s12, int8_head_fp16

**新工程** (~12h):
- ONNX Q/DQ 选 stage injection (~4h)
- TRT C++ API set_input_range (~6h)
- Custom IInt8 KLD + percentile calibrator (~2h)

**预算**: 12h 工程 + 384 × 5 min / 6 GPU = **17h** ≈ 2 day

### 6.4 路径 3 — B × Q 协同 (加密 grid)

**目标**: 28 B (24 + 4 中间) × 10 Q (7 + 3 hybrid) × FT=8 × D=D1 = **280 anchor**

**新工作**:
- 4 新 g8 triplet (路径 1 子集): ~1h
- 3 新 Q (mix_s01 / mix_s02 / int8_head_fp16): ~4h 工程
- 280 anchor eval

**预算**: ~6h

### 6.5 路径 4 — reframe paper

**目标**: 不补 AP 数据, 全力补 lat (Phase E), reframe paper §C narrative.

**新工作**: 0 新 AP. Phase C 跳过, 直接 Phase D 用 Phase A 120 anchor + (路径选定就停留) 出 14 图 + 写 reframe report.

**预算**: 0 (Phase C 跳过)

### 6.6 路径选定后状态文件

agent 写:
```json
"phase_b_path": "1" | "2a" | "2b" | "3" | "4",
"phase_b_path_rationale": "<one-line 含 SS% / std 数字>",
"phase_c_anchor_target": <int>  // 例 35 / 48 / 384 / 35 / 0
```

---

## §7 Phase C — 数据采集 (按 Phase B 路径)

### 7.1 启动信号

- `plan4_state.json.phase_a_done == true`
- `plan4_state.json.phase_b_path` ∈ {1, 2a, 2b, 3} (路径 4 直接跳 Phase D)
- 6 GPU 可用

### 7.2 实施脚本

每路径独立 dispatcher:

```
路径 1: scripts/phase2/plan4_phaseC_path1_b_extend.py
路径 2a: scripts/phase2/plan4_phaseC_path2a_q_refill.py
路径 2b: scripts/phase2/a14_q_resolution_dispatcher.py (路径 2b 用 16 Q)
路径 3: scripts/phase2/plan4_phaseC_path3_bq_dense.py
```

### 7.3 中间 sanity 协议 (强制)

- 每完成 50 anchor, 自动跑 5 anchor 跟 bench v1 比对 (同 triplet/Q, FT=8 ckpt)
- 要求 `|Δ AP50| ≤ 0.005` (跟 §0 sanity 0.0004 量级一致放宽)
- 不通过 → 停采集, 报告 `phaseC_sanity_fail.md`

### 7.4 完成信号

- `plan4_phaseC_anchors.csv` 行数 ≥ 0.95 × phase_c_anchor_target
- sanity 通过 (每 50 anchor 一次)
- agent 更新 `plan4_state.json.phase_c_done = true`

### 7.5 trigger Phase D

- 自动 trigger

### 7.6 预算 (按路径)

| 路径 | wall |
|---|---|
| 1 (B 扩) | 2.5h |
| 2a (Q refill) | 45 min |
| 2b (Q 扩 16) | 17h (含工具) |
| 3 (B×Q 加密) | 6h |
| 4 (reframe) | 0 |

---

## §8 Phase D — AP 图 + LGB v9_ap + 验收

### 8.1 启动信号

- `plan4_state.json.phase_c_done == true` OR `phase_b_path == 4`
- Phase A + C 累计 anchor ≥ 120 (基础) + 路径增量

### 8.2 14 图生成

- 脚本: `scripts/phase2/plan4_phaseD_dataset_stats.py` (复用 `dataset_stats_report_bench.py` 框架)
- 输入: 合并 Phase A + Phase C 全部 anchor (csv)
- 输出: `paper_learning/2. AAAI最终故事/data/stats_v3_plan4/` (14 PNG + 中文说明 + dataset_v4_stats.md)
- **强制 GroupKFold by triplet** (D1.6 5-fold + D2 learning curve + D3 OOD 全部)
- **强制 LGB 特征不含 ft, 不含 d** (C2/C3 约束)

### 8.3 LGB v9_ap 训练

- 脚本: `scripts/phase2/plan4_train_lgb_v9_ap.py`
- 特征:
  ```
  planes_s1, planes_s2, planes_s3, total_prune_pct,
  q_<7 or 16 one-hot>,
  q_granularity (pt/pc), q_object (wo/wa), q_calibrator (mm/ent/kld/pct),
  q_mix_s0, q_mix_s1, q_mix_s2 (binary), q_head_fp16 (binary)
  ```
- 训练: 5-fold GroupKFold by triplet
- 输出: `models/lgb_v9_ap.txt` + `lgb_v9_ap_feature_importance.csv`

### 8.4 验收 5 gate (预注册)

| # | gate | 阈值 |
|---|---|---|
| G1 | 5-fold GroupKFold R² | ≥ 0.65 |
| G2 | OOD MAE (hold 1 triplet, repeat 5 random hold) | ≤ 1.5 × in-sample MAE |
| G3 | Plateau-only R² (AP ≥ 0.50) | ≥ 0.55 |
| G4 | 强相关特征数 (|Spearman ρ| > 0.30) | ≥ 3 |
| G5 | AP std (整 dataset) | ≥ 0.10 |

**全 5 通过** → Phase D 完成
**任一不通过** → 回 Phase A 重归因 (路径 4 reframe 是合法回退)

### 8.5 user gate

- agent 完成 14 图 + LGB + 5 gate 自检后, **暂停**, 等用户 review:
  - B1 hist (新 std vs bench v1 0.116)
  - D1 5-fold R² per fold
  - D3 OOD predict-vs-real scatter
- user "ok" → trigger Phase E
- user "redo" → 回 Phase A (并附用户反馈写到状态文件)

### 8.6 完成信号

- `stats_v3_plan4/` 14 PNG + 中文说明 + dataset_v4_stats.md 存在
- `models/lgb_v9_ap.txt` 存在
- `plan4_phaseD_report.md` 含 5 gate pass/fail + 用户 review 状态
- `plan4_state.json` 更新 (phase_d_done, phase_d_ap_r2, phase_d_user_review)

### 8.7 预算

| step | wall |
|---|---|
| 14 图生成 | 30 min |
| LGB v9_ap 训练 | 5 min |
| 5 gate 自检 + 报告 | 15 min |
| 等用户 review (暂停) | 0 (异步) |
| **Total** | **~1h** + user review wait |

---

## §9 Phase E — LAT predictor 补强 (并行可启)

### 9.1 启动信号

- **可独立于 Phase A-D 启动** (LAT 跟 AP 解耦, 跟 FT 无关 — 不违反 §7.13)
- 选择启动时机:
  - **Mode A**: 跟 Phase A 并行 (节省 wall time)
  - **Mode B**: 等 Phase D 完成后顺序启 (简化调度)
- agent 默认 Mode B (简单), Mode A 留作用户手动 trigger

### 9.2 sub-tasks

#### E.1 多 build 平均 (kernel 选择噪声测量)

- 选 100 anchor (random sample from bench v1) × 3 build × bench
- 计算 per-anchor build-internal lat std
- **gate**: > 95% anchor 的 3-build std/mean ≤ 5%
- 不通过 → 看 anchor list, PIN tactic for >10% std ones
- 预算: 100 × 3 × 90s / 6 GPU = **~2h**

#### E.2 D 网格补 12 cell

- 选 12 个 D 配置 (空缺位补), × 21 triplet × 7 Q = 1764 lat anchor (无 AP)
- 复用 bench v1 ONNX + 新 D 跑 trtexec build + bench
- 预算: 1764 × 90s / 6 GPU = **~7h**

#### E.3 kernel-aware feature

- 脚本 `parse_trt_build_log.py`: 解析 build.json 提取
  - `winning_tactic` (default / cublas_lt / etc)
  - `n_layers_int8`, `n_layers_fp16`, `n_layers_fp32`
  - `n_myelin_nodes`
- 加入 LGB lat predictor 特征向量
- 预算: 3h 工程

#### E.4 LGB v9_lat 训练

- 数据: bench v1 4584 lat anchor + E.2 新 1764 = 6348 lat anchor
- 特征: planes + Q one-hot + D one-hot + kernel-aware (新)
- 分割: **5-fold GroupKFold by (triplet, q)** (问题 §7.12 教训)
- 输出: `models/lgb_v9_lat.txt`
- 预算: 10 min

### 9.3 验收 3 gate (预注册)

| # | gate | 阈值 |
|---|---|---|
| L1 | 5-fold GroupKFold R² | ≥ 0.75 |
| L2 | 3-build std/mean | ≤ 5% (≥ 95% anchor) |
| L3 | OOD MAE (hold 1 D) | ≤ 2 ms |

### 9.4 完成信号

- `models/lgb_v9_lat.txt` + `plan4_phaseE_report.md`
- L1+L2+L3 全过 (允许 L3 单独 fail, 标 reservation)
- `plan4_state.json.phase_e_done = true`

### 9.5 预算

| step | wall |
|---|---|
| E.1 多 build sanity | 2h |
| E.2 D 网格补 | 7h |
| E.3 kernel-aware 工程 | 3h |
| E.4 LGB v9_lat 训练 | 10 min |
| 报告 | 30 min |
| **Total** | **~13h** |

---

## §10 总时间预算

| Phase | wall | GPU 占用 |
|---|---|---|
| Pre-Phase 0 (σ_FT8) | 30 min | 5 GPU 10 min |
| Phase A (B × Q 归因 128 anchor) | 1.5h | 6 GPU 1h |
| Phase A 报告 + 路径决策 | 15 min | 0 |
| **Phase B 决策** | 0 (自动) | 0 |
| Phase C (按路径) | 2-17h | 6 GPU |
| Phase D (14 图 + LGB + 5 gate) | 1h | 1 GPU |
| **User gate (Phase D review)** | 0 (异步 wait) | 0 |
| Phase E (LAT 补强) | 13h | 6 GPU 9h |
| Phase E 报告 | 30 min | 0 |
| **Total (路径 1 / 2a / 3 = 典型)** | **~22h** | **2.5 day** |
| **Total (路径 2b = 最坏)** | **~36h** | **4 day** |
| **Total (路径 4 reframe)** | **~16h** | **2 day** |

---

## §11 整体最终目标 (完成判据)

### 11.1 ✅ 成功 Path 1 — 完整双轴信号

```
Phase A:  σ_FT8 < 0.04 + Phase A LGB R² ≥ 0.40
Phase D:  AP 5-fold GroupKFold R² ≥ 0.65 + OOD MAE ≤ 1.5× in-sample
          + plateau R² ≥ 0.55 + 强特征 ≥ 3 + std ≥ 0.10
Phase E:  LAT R² ≥ 0.75 + 3-build std ≤ 5%
```

**paper §C narrative**:
> "在 deployment-realistic 约束 (FT=8, D 部署 D1) 下, B × Q (or B / Q) 跨度给 AP std={MEASURED}, LGB predictor 达 R²={MEASURED}, lat predictor R²={MEASURED}. sensitivity-stratified DoE + active learning 用 K=64 anchor 即可达 R²={MEASURED}."

### 11.2 ✅ 成功 Path 2 — Reframe (路径 4 触发 + LAT 通过)

```
Phase A:  σ_FT8 < 0.04 + Phase A LGB R² < 0.40 + 所有 SS < 0.30
                                                  → 路径 4
Phase D:  跳过 (或写 reframe report)
Phase E:  LAT R² ≥ 0.75 + 3-build std ≤ 5%
```

**paper §C narrative (reframe)**:
> "Pyramid Fusion + DAIR-V2X 在 deployment-realistic 部署约束 (FT=8 + 任意 B/Q/D 组合) 下, AP intrinsic plateau (std ≈ 0.04, range ≈ 0.10 排除 entropy collapse). framework 主体 predictor 是 lat (R² ≥ 0.75 via D-rich + multi-build sanity + kernel-aware feature). AP 仅作 safety filter (rule: 避开 int8_ent + extreme g8 prune)."

这本身是 paper 主要发现, **不是失败**.

### 11.3 ❌ 失败

| 失败模式 | 触发条件 |
|---|---|
| F1 | Pre-Phase 0 σ_FT8 ≥ 0.04 (项目 finetune-equilibrium 假设崩) |
| F2 | Phase A 128 anchor build_success rate < 90% (工程故障) |
| F3 | Phase D 5 gate 全 fail AND 路径 4 也不接受 |
| F4 | Phase E lat R² < 0.65 (lat 也无法学) |
| F5 | 任一 Phase wall > 2 × 预算 |

失败时: agent 写 `plan4_failure_report.md`, 状态文件 `final_verdict = "fail"`, 暂停 + 通知用户.

---

## §12 启动 / 暂停 / 恢复 操作手册 (/goal 用户接口)

### 12.1 首次启动

```bash
# 1. 验证前置 (用户做):
ls paper_learning/2.\ AAAI最终故事/data/{问题.md,数据集制作_plan.md,e2e_bench_v1_sanity_report.md}

# 2. 启动 /goal
/goal paper_learning/2.\ AAAI最终故事/data/AP不敏感解决方案_plan4.md
```

**首次启动时 agent 行为预期** (按 §3.5.4 协议):
1. read `plan4.md` §0-§3 + §7.13 (Tier 0, ~3k tokens)
2. check `plan4_state.json` — 不存在 → 创建空 template
3. 进入 Pre-Phase 0 (因为 phase_0_done=false), read §0.2 中 Pre-Phase 0 的 Tier 1 文件 (问题.md §7.11)
4. 启动 Pre-Phase 0 dispatcher
5. **不读** Tier 2 (历史 plan / csv / png)

### 12.2 中途暂停 + 跨会话 resume

**暂停**:
- agent 完成当前 anchor / sub-step 后, **写 state.json + phase_X_progress** (不止 phase 完成才写)
- 用户随时可 Ctrl+C 或 session 结束

**Resume** (新会话 / 同会话):
```bash
/goal paper_learning/2.\ AAAI最终故事/data/AP不敏感解决方案_plan4.md
```
agent **严格按 §3.5.4 三步**:
1. read state.json + plan4.md §0-§3 + §7.13 (Tier 0 only)
2. 根据 `phase_X_progress` 找断点
3. read 当前 phase 章节 (Tier 1) + 前一 phase summary

**不允许**: 重新读全部前置 6 份, 重新跑已完成 anchor.

### 12.3 user gate (Phase D review 暂停)

- agent 完成 Phase D 14 图 + 5 gate 检查后, 暂停, 写 `plan4_state.json.phase_d_user_review = "pending"`
- 用户 review (人工看 3 张 PNG: 05/11/13_*.png) 后, **手动**编辑 state.json:
  - `phase_d_user_review = "approved"` → agent 进 Phase E
  - `phase_d_user_review = "redo"` → 写 `phase_d_user_feedback` 字段, agent 按反馈调整

### 12.4 abort (硬停)

- 用户写 `plan4_state.json.user_abort = true` → agent 完成当前 anchor 后停, 不进入下一 step
- agent 写 final state 摘要到 `plan4_aborted_summary.md`

### 12.5 上下文管理触发的特殊行为

agent 在以下情况**主动触发 self-compaction** (§3.5.5):

1. 完成一 phase: 强制写 `phase_{N}_summary.md` + evict Tier 1 文件
2. conversation hop > 30: 主动写 mini-summary 然后退出当前 loop, 让 user 重启
3. 收到 hook 警告 "context near limit": 立即写 state.json + summary, exit
4. 子任务跑期间: 主 agent **不轮询**, 用 Bash run_in_background + sleep/ScheduleWakeup 等通知

如果用户感觉 agent "丢上下文" (问相同问题 / 重做相同事):
- 检查 `plan4_state.json.context_compaction_count` 是否 > 5 (太频繁说明设计问题)
- 检查 `phase_{N}_summary.md` 是否被正确生成 + 是否被新 session read

---

## §13 风险 + 回退 (最终版)

| # | 风险 | 概率 | 影响 | 回退 |
|---|---|---|---|---|
| R1 | σ_FT8 ≥ 0.04 | 低 | F1 失败 | abort, 用户决定是否换 FT* |
| R2 | Phase A 全无信号 (R² < 0.40) | 中 | 走路径 4 (reframe) | 加 Phase E lat 即可, paper 仍有 contribution |
| R3 | Phase A SS 跨多因素都中等 | 中 | 路径选择模糊 | 选路径 3 (B × Q 加密 grid, 覆盖最广) |
| R4 | 路径 2b TRT 工程超时 | 中 | Phase C 延期 | 砍 mix variants, 接受 11 Q |
| R5 | Phase D 5 gate 全失败 | 中 | reframe 必走 | 路径 4 |
| R6 | Phase E lat R² < 0.75 | 中 | 双轴一轴不达 | paper 接受 lat R²=0.65 作 "TRT kernel noise cap" 发现 |
| R7 | GPU 不可用 | 高 (生产环境) | 暂停 | resume 等 GPU 空闲 |

---

## §14 对应实施脚本清单

```
scripts/phase2/
├── plan4_phase0_ft8_noise.py         🔲 (Pre-Phase 0)
├── plan4_phaseA_dispatcher.py        🔲 (Phase A 120 anchor)
├── plan4_phaseA_attribution.py       🔲 (ANOVA / SHAP)
├── plan4_phaseC_path1_b_extend.py    🔲 (路径 1)
├── plan4_phaseC_path2a_q_refill.py   🔲 (路径 2a)
├── a14_q_resolution_dispatcher.py    🔲 (路径 2b / 3 用)
├── plan4_phaseC_path3_bq_dense.py    🔲 (路径 3)
├── plan4_phaseD_dataset_stats.py     🔲 (wrap dataset_stats_report_bench.py)
├── plan4_train_lgb_v9_ap.py          🔲 (LGB v9 AP)
├── plan4_phaseE_multi_build.py       🔲 (E.1)
├── plan4_phaseE_d_grid_dispatcher.py 🔲 (E.2)
├── parse_trt_build_log.py            🔲 (E.3 kernel-aware)
└── plan4_train_lgb_v9_lat.py         🔲 (LGB v9 LAT)

scripts/phase1/
└── m4_8_trt_build_bench.py           ✅ (已有, --multi-build flag 待加)
```

---

## §15 修订历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1 | 2026-05-21 | plan v1 — FT sweet spot 假设, 早停 |
| v2 | 2026-05-22 | plan v2 — g8 + (4,4,4), B×Q collapse 找到但窄 |
| v3 | 2026-05-25 | plan v3 — Tier 分离 + FT=8 deployment, R²=0.56 plateau |
| v4-draft-1 | 2026-05-27 上午 | plan v4 草稿 — 含 FT sweep, **被用户驳回** |
| v4-draft-2 | 2026-05-27 下午 | plan v4 修订 — FT=8 锁定, Pre-Phase 0 + 2 因素 Phase A |
| v4-draft-3 | 2026-05-27 下午晚 | plan v4 /goal 长周期版 — 加 §0 前置, §2 历史反思, §3 状态机, §12 操作手册, 阈值统一 |
| **v4** | **2026-05-27 晚** | **plan v4 上下文管理修订 — §0 改分层 (Tier 0/1/2), 新增 §3.5 上下文管理协议 (3 道防线 + 子任务委托 + self-compaction + 跨会话 resume 协议), §12 增 context-aware 行为. 解决 22-36h 长周期 vs 单会话 context 寿命矛盾.** |
