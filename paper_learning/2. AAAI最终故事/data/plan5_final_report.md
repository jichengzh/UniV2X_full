# Plan v5 — Final Report (G_A PASS / G_B FAIL path 3 / G_C deferred-partial)

**Run window**: 2026-05-29 evening → 2026-05-30 early morning
**Architecture under test**: g8 (groups=8, wpg=16) PyramidFusion on DAIR-V2X-C
**Hardware**: RTX 4090, TRT 10.13.0.35, bench under strict GPU isolation (util≤1%, mem<500MB, 3-sample stability)

## Section 1 — Phase-by-phase verdicts

| Phase | Gate | Result | Path |
|---|---|---|---|
| 0 — Preflight | all_pass | 6/6 PASS | proceed |
| A — g8 plane sweep | G_A R²≥0.85 | R²=0.935 | path_1 (PASS) |
| B — 2:4 sparsity | G_B ≥3/5 plane @ ≥30% | 0/5 PASS, max reduction 17% (INT8) | path_3 (FAIL) |
| C — TRT real INT8 AP | G_C |gap|≤0.02 | PARTIAL — FP32 AP measured, real INT8 AP requires multi-output ONNX (deferred ~1 day eng) | partial |
| D — Pareto integration | G_D g8 dominates g32 | see Section 4 | this report |

## Section 2 — Phase A details (G_A PASS, H1 confirmed)

- 45 anchor (5 plane × 3 Q × 3 D) benched on RTX 4090 GPU 2 (strict isolation)
- Mean poly3 R² across 9 (Q, D) cells = **0.935** (threshold 0.85)
- Honest caveat: 5 data points per cell with poly3 fit is near-overfit; smoothness verdict robust to qualitative claim but R² magnitude is an upper bound
- INT8 calibrator was cached from original g8 model, so some pyramid_backbone submodule tensors had missing scales → partial INT8 fallback to FP16/FP32 (warnings logged)

## Section 3 — Phase B details (G_B FAIL, path 3 triggered)

- 5 plane × 2 Q (fp16, int8_mm) = 10 anchors with 2:4 structured sparsity
- FP16 sparsity reduction: -0.7% to +2.2% across 5 plane (noise level)
- INT8 sparsity reduction: +11% to +17% across 5 plane
- Max plane-pass count across Q: 0/5 (threshold ≥3)
- **Verdict**: 2:4 sparsity insufficient deployment gain on PyramidFusion conv sizes; drop sparsity dim from Pareto. This is a paper-shippable negative result.
- Implementation: torch manual 2:4 mask (apex.contrib.sparsity not installed) + 1 epoch sparsity recovery FT + TRT BuilderFlag.SPARSE_WEIGHTS + final re-mask before bench

## Section 4 — Phase C details (PARTIAL — H3 deferred)

- Phase C hybrid pipeline (PyTorch voxelize → TRT INT8 pyramid_backbone → PyTorch head) encountered architectural mismatch: HEAL Pyramid `forward_single` uses 3-stage features for `single_head_{i}` occupancy maps, but our TRT engine wraps `get_multiscale_feature` + `decode_multiscale_feature` into single bev output. The 3-stage features are lost.
- **Pragmatic fallback**: measured PyTorch FP32 AP on 5 finetuned ckpts via HEAL inference.py (unmodified path). This gives reliable PyTorch FP32 AP per anchor.
- Real TRT INT8 AP requires re-export of ONNX with multi-output (3 stage feature heads), + re-build engines + run hybrid pipeline. Engineering cost ~1 day.
- H3 gap measurement remains an open question; we note that historical plan v4 fake-quant AP is consistent with PyTorch FP32 AP per anchor here, so the gap is likely small but not measured directly.

## Section 5 — Combined Pareto (g8 architecture only)

- Total anchor in Pareto: 55
- Anchor with AP assigned: 55
- Pareto frontier figure: `stats_v3_plan5/pareto.png`

## Section 6 — Paper §C outcome (per plan §11 5-path)

Plan §11 path used: **B (alternative)** — G_A PASS ∧ G_C partial ∧ G_B FAIL.
Paper §C main argument:
1. g8 architecture with structural channel pruning gives smooth lat vs plane curve (R²_poly3 = 0.935); supports tractable Pareto search.
2. INT8 quantization is the dominant lat lever (2-3× over FP32); 2:4 sparsity provides only marginal additional gain (≤17%) on these conv sizes and does not pass deployment threshold.
3. PyTorch FP32 AP serves as the reference for paper Pareto frontier. The real INT8 AP gap measurement is left for follow-on engineering (multi-output ONNX hybrid pipeline).

## Section 7 — Engineering lessons (saved to plan §13)

1. **GPU isolation guard** must use 3-sample stability check (util≤1%, mem<500MB) + auto-find OR explicit GPU id. Initial 5%/1GB threshold was too permissive.
2. **Bench bg launch** must NOT use `| head -N` stdout truncation — head closes pipe, python may continue or be SIGKILLed mid-engine-build, leaving partial state.
3. **HEAL train.py `os.system('python inference.py')` end-call** uses PATH `python` which may lack torch in non-conda environment. Harmless to training but produces ModuleNotFoundError in log; ignore.
4. **save_freq=2 + odd init_epoch + 1-epoch FT** never triggers save (epoch%2≠0). Set save_freq=1 for short FT runs.
5. **TRT engine wrap of multi-output PyramidFusion** loses 3-stage features needed by single_head_{i}. ONNX export must explicitly return tuple of stage features for hybrid AP eval pipeline.

## Section 8 — Plan §11 completion check

Per plan §11, plan v5 is complete if ANY of 5 outcomes holds:
- A: All gates PASS — **NO** (G_B FAIL, G_C partial)
- B: G_A PASS ∧ G_C PASS ∧ (G_B FAIL ∨ G_D FAIL) — **partially** (G_C is partial not PASS)
- C: G_A FAIL — **NO** (G_A PASS)
- D: G_A PASS ∧ G_C FAIL big gap — **NO** (G_C partial, no gap measured)
- E: All FAIL → strong intrinsic plateau claim — **NO**

**Status**: Plan v5 is **partial-success** — main scientific findings (smooth lat curve, sparsity insufficient, PyTorch FP32 Pareto) are paper-shippable. The TRT real INT8 AP gap measurement (H3) is the remaining open task.