# 数据集制作进度 (PROGRESS.md)

> 详细计划见 `数据集制作_plan.md`. 本文件按**最小可执行模块**拆分, 供 AI 逐模块产出.
> **数据口径**: `e2e_bench_v1.csv` schema (B×Q×D×HW×Model 全坐标 + lat/AP/throughput/resource 四类指标).
> **最后更新**: 2026-05-15

---

## 总览

**224 / 720 行 (31.1%)**

| Class | 已 | 目标 | 缺口 | 模块数 (待办) |
|-------|----|------|------|---------------|
| A — Pyramid 4090 + Orin | 224 | 570 | 346 | 3 |
| B — 4090 D-space | 0 | 120 | 120 | 4 |
| C — Orin D-space | 0 | 170 | 170 | 5 |
| D — 跨模型 | 0 | 100 | 100 | 2 |
| **合计** | **224** | **720** | **496** | **14** |

状态图例: ✅ 完成 / 🟡 进行中 / 🔲 未启动 / ⛔ 阻塞

---

## Class A — Pyramid 4090 + Orin (32/570)

### A.0 baseline 笛卡尔积 — ✅ 完成 (32 行)
- **范围**: 8 T × 4 Q (fp32, fp16, int8_mm, int8_ent) × 1 D (GPU/default/4GB) × rtx4090
- **产出**: `e2e_bench_v1.csv` (2026-05-14)
- **脚本**: `scripts/phase2/e2e_bench_v1_orchestrator.py`

### A.1 4090 Q 维补 — 🟢 phase1+phase2(部分) ✅ done (24/48)
- **范围**: 8 T × NEW Q × 1 D (GPU/default/4GB)
- **路径 B 完成情况**:
  - **phase1 mix ✅ done (2026-05-15, 6-GPU parallel)**: Q_mix_s0, Q_mix_s2 = 16 anchor. MinMax + interpretation A. AP50 范围 [0.529, 0.568]
  - **phase2 granularity ✅ partial (2026-05-15)**: 8 个 Q_int8_pc_wo anchor (W-only, per-channel weight, --w-only flag 实现). pt 变体 (Q_int8_pt_wa/pt_wo) **不可信测**, 根因见 `问题.md` 问题 4
- **脚本**: `a1_run_one_anchor.py` (phase1), `a1_run_one_anchor_phase2.py` + `m4_8_trt_build_bench.py --w-only` (phase2)
- **AP**: 1789 sample DAIR full eval (与 P0.5 e2e_ap_minmax_v1 同口径)
- **关键发现** (2026-05-15 最终修正后):
  - **16 mix anchor AP50 一致 mediocre** [0.529, 0.568], std 0.012 → mix precision 边界 Q/DQ **系统性损失 ~20pp** vs Q_int8_mm
  - **8 Q_int8_pc_wo anchor AP50** [0.528, 0.556] — W-only INT8 同样 ~24pp 损失. Pareto 端: T6_p75/Q_mix_s0 AP50=0.543, 155 fps, 1.75× speedup
  - **撤回伪发现 1** (问题 3): T1_base/Q_mix_s2 = 0.153 是孤儿子进程 artifact, 真值 0.549, 无 B × Q 强耦合
  - **撤回伪发现 2** (问题 2): TRT IMMA/HMMA 32-对齐假设错, 真实根因是 ResNeXt grouped conv `in_per_group` 非 pow-2 → cuDNN fallback
  - **撤回伪发现 3** (问题 4): pt_wa AP=0 不是 per-tensor 真崩, 是 ONNX Q/DQ 路径破坏 TRT 残差块融合 (Conv+Add+ReLU 融合 -25%) → 65 Conv 独立跑 + FP32 中间累积噪声 → -23pp

### A.2 4090 D 维补 3 个 — ✅ 完成 (168/240, 2026-05-16)
- **范围**: 8 T × 7 现有 Q × 3 NEW D = 168 anchor (剩 72 缺口对应 phase2 pt 变体, 走 问题 4 方案 2 后再补)
- **NEW D**: D2(with_cudnn, 8GB), D3(cublas_lt, 16GB), D4(all_enabled, 1GB)
- **产出**: `e2e_bench_v1.csv` 168 行追加 (224 行合计), `results/a2_d_expand/*.row.json`
- **脚本**: `scripts/phase1/m4_8_trt_build_bench.py --tactic` (新 flag), `scripts/phase2/a2_run_one_anchor.py` (orchestrator), `scripts/phase2/a2_dispatch_parallel.py` (6 GPU 并行)
- **执行**: 6 GPU 并行 68 min, 168/168 build_success, 0 fail
- **关键发现**:
  - D-dim 真信号: fps_delta_pct 跨 (Q, D) 在 [-23%, +14%] (mean across 8 T 在 [-23%, +2%])
  - Q_int8_pc_wo 对 tactic 最敏感 (-15% 到 -23%, W-only INT8 layer fusion 受限)
  - workspace 1-16GB 对 build_secs 影响 < 5%, 对 throughput 影响 -7% 到 +14% — 是 runtime knob 非 build knob
  - AP D-invariance 验证 ✓ (force-ap T1_base/Q_int8_mm/D3 = 0.5456, csv=0.5458, 差 0.0002)
- **副作用**: A.2 完成时 force-ap 校验暴露 问题 5 (csv A.0 AP 来自 subnet 管线 buggy), 已修复 (32 anchor 重跑 e2e_eval_ap n=1789, 128 行回填)

### A.3 Orin 移植 — ⛔ 阻塞 (0/250)
- **范围**: 8 T × 6 Q × 12 D (Orin AGX), 预期 build_success ≈ 40-50%
- **预测**: ~250 行真测 + ~250 行 negative
- **脚本**: `scripts/phase2/orin_class_a_int8_sweep.py`
- **阻塞**: TRT cache 跨版本 (4090 build = 10.13 vs Orin run = 8.5) → 需 patch magic TRT-101300 → TRT-8502 重打

---

## Class B — 4090 D-space (0/120)

### B.1 workspace 扩档 {1GB, 16GB} — 🔲 (0/24)
- **范围**: 3 T (T1/T4/T6) × 2 prec (fp16, int8_mm) × 4 tactic × 2 ws
- **脚本**: `scripts/phase2/p_4090_dspace_bench.py` (改 schema 适配 e2e_bench_v1)

### B.2 2:4 sparse INT8 路径 — 🔲 (0/24)
- **范围**: 3 T × 4 tactic × 2 ws (8GB, 16GB)
- **前置**: ASP sparse weight reload + INT8 联合校准
- **脚本**: 待新建 `scripts/phase2/dataset_b_sparse_int8.py`

### B.3 edge_only / cublas_lt only tactic — 🔲 (0/24)
- **范围**: 3 T × 2 prec × 2 tactic × 2 ws

### B.4 random_search 补 unseen triplet — 🔲 (0/48)
- **范围**: 12 random triplet × 4 D-cfg (反思 #28: 避免 hand-pick selection bias)
- **脚本**: 待新建 `scripts/phase2/dataset_b_random_search.py`

---

## Class C — Orin D-space (0/170)

### C.1 workspace 扩档 {4GB, 8GB} — 🔲 (0/36)
- **范围**: 3 T × 6 Q × 2 ws (其余 D 固定)
- **脚本**: `scripts/phase2/orin_class_a_bench_trtexec.py --ws-grid extended`

### C.2 双 IP 流水线 {A, B} — 🔲 (0/16)
- **范围**: 4 T × 2 prec × 2 scheme
- **dual_A**: backbone DLA0 + collab GPU; **dual_B**: backbone DLA1 + collab GPU

### C.3 三 IP 流水线 C — 🔲 (0/16)
- **范围**: 4 T × 2 prec × 1 scheme × 2 ws
- **triple_C**: stage01 DLA0 + stage2 DLA1 + collab GPU

### C.4 fallback policy {strict, permissive} — 🔲 (0/10)
- **范围**: 5 anchor × 2 policy
- **Orin 独有维度**, paper §B 防御实证

### C.5 单 IP DLA0/1 全图测 — 🔲 (0/20)
- **范围**: 5 unseen T × 2 prec × 2 DLA target
- **预期 build_success ≈ 50%** (DLA banks limit)

---

## Class D — 跨模型 (0/100)

### D.1 UniAD-tiny on 4090 — 🔲 (0/60)
- **范围**: 12 prune ratio × 5 D-cfg = 60
- **数据集**: OPV2V-coop val 4K samples (不是 DAIR-V2X)
- **脚本**: 待新建 `scripts/phase2/dataset_d_uniad_tiny.py`
- **前置**: M5.x 已产出的 5 个 `univ2x_tiny_*` ONNX 作为 baseline

### D.2 UniV2X full (DCN 反例) — 🔲 (0/40)
- **范围**: backbone={none, 2:4} × Q × D = 40
- **限制**: DCN 不可剪 (反思 #6 已论证)
- **角色**: framework 硬约束实证
- **脚本**: 待新建 `scripts/phase2/dataset_d_univ2x_full.py`

---

## 工程链路状态 (Schema / Pipeline 共享依赖)

| 依赖 | 状态 | 备注 |
|------|------|------|
| e2e_bench_v1 schema (41 列) | ✅ | `e2e_bench_v1_schema.md` v1.2 |
| AP eval 双 engine (subnet + collab) | ✅ | 0% PyTorch fallback |
| MinMax 主路径 calibration | ✅ | 2026-05-15 决议 (entropy 禁用) |
| TRT cross-version cache patch (4090 → Orin) | 🔲 | 阻塞 A.3 |
| DDP 4 卡训练协议 | ✅ | P0.1 已用 |
| Iterative Magnitude Pruning | 🔲 | Class D random triplet 待启用 |

---

## 修订历史

| 日期 | 变更 |
|------|------|
| 2026-05-15 | 初版 — 32/720, 15 模块拆分 |
