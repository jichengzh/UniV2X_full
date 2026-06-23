# Plan v5 Phase A.4 — Attribution + Gate G_A

**Run date**: 2026-05-29
**Anchors evaluated**: 45 rows (5 plane, 3 Q, 3 D)

## Gate G_A — smooth poly3 fit (lat vs plane)

- Threshold: R²_smooth ≥ 0.85 (pre-registered, Phase 0)
- Measured: **R²_smooth = 0.935** (9/9 cells PASS individually)
- Verdict path: **path_1**
- Reason: G_A PASS — H1 confirmed (smooth curve). Proceed to Phase B.

## Per (Q, D) cell breakdown

| Q | D | n | plane range | lat range | lat ratio | poly3 R² |
|---|---|---|---|---|---|---|
| fp16 | D1_default | 5 | [8, 64] | 0.44-1.38 ms | 3.11× | 0.917 |
| fp16 | D2_BL0_default | 5 | [8, 64] | 0.71-3.01 ms | 4.24× | 0.981 |
| fp16 | D3_BL0_enableall | 5 | [8, 64] | 0.70-3.01 ms | 4.31× | 0.981 |
| int8_mm | D1_default | 5 | [8, 64] | 0.52-1.62 ms | 3.13× | 0.855 |
| int8_mm | D2_BL0_default | 5 | [8, 64] | 0.61-2.69 ms | 4.38× | 0.863 |
| int8_mm | D3_BL0_enableall | 5 | [8, 64] | 0.64-2.68 ms | 4.22× | 0.875 |
| int8_pc_wo | D1_default | 5 | [8, 64] | 0.56-2.25 ms | 3.98× | 0.995 |
| int8_pc_wo | D2_BL0_default | 5 | [8, 64] | 1.03-5.33 ms | 5.20× | 0.972 |
| int8_pc_wo | D3_BL0_enableall | 5 | [8, 64] | 1.03-5.33 ms | 5.19× | 0.972 |

## Next phase
- G_A PASS — H1 confirmed (smooth curve). Proceed to Phase B.

## Raw data
- Anchor CSV: `paper_learning/2. AAAI最终故事/data/plan5_phaseA_anchors.csv`
- State file updated: `paper_learning/2. AAAI最终故事/data/plan5_state.json`