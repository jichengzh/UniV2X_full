# B4 integration checklist (main control working note, 2026-06-20)

When **B1 grid** (PID 2076337 on H800, ETA ~20:00 CST, `lut_results_grid.csv`) AND
**B2 calibrated AP** (`results/ap70_model_pyramid.json`, status→CALIBRATED) both land:

## Steps
1. **Pull B1 grid** → `results/lut_results_grid.csv`:
   `sshpass -p '12345678' scp -P 30001 jichengzhi@222.95.84.215:/exdata/jichengzhi/s2_tvm/lut_results_grid.csv results/`
2. **Verify every row** `e2e_ratio > 1.05` (else load spiked → that width's tuned build was killed; re-run it). Spot-check iso_s0 reproduces tuned≈21752µs (±5%) — the harness-trust gate.
3. **Build LUT** (direct-grid format, NOT additive): `python scripts/phase2/build_latency_lut.py --gap1_json results/gap1_grid_corrected.json --lut_csv results/lut_results_grid.csv --json_out results/latency_lut_pyramid.json`. Confirm it emits `{"widths":[...]}` (direct) so `LatencyLUT._load_b1` mode=`b1_direct`.
4. **Verify B2 AP**: read `ap70_model_pyramid.json` — status CALIBRATED, b0/b1/b2 separated, held-out mixed residual reported. Expect b0≈0 (stage0 AP-insensitive). Re-eval-spot-check from Agent-APfit report.
5. **★FIX `framework/search_three_arm.py::candidate_widths` (line ~584)**: currently grid = widths with a *real* AP70 (`apm.exact`) AND priceable latency → would EXCLUDE the 12 new latency-only widths. Under the model-guided strategy the grid must = **all priceable widths** (the ~20 B1-measured widths), AP70 from the **model** (exact where known, predicted else). Change loop to iterate the LUT's priceable widths (direct grid keys ∪ seed) and call `apm.ap70(w)` for AP (which already does exact-or-model). Keep real anchors exact.
6. **Search-space-size / budget sanity**: ~20 widths × {default,tuned} = ~40 configs. Set B4 budget << 40 so convergence curves are meaningful (not instant-enumerate). If grid too small for a credible "real search not enumeration" (doc4 §0), consider: (a) framing = structural exclusion holds on any grid containing a W_g/P_g pair (the searcher autonomously locks W_g, not us hand-picking); (b) note discrete-grid limitation honestly. Decide framing when real grid size is known.
7. **Run B4**: `python -m framework.run_b4_ablation --seeds 12` → expect `data_status: real`. Verify 3 PNGs regenerate, Wilcoxon, dual iso-AP70 ratios become headline magnitudes. Check point-cloud still shows A-serial missing (pad64,tuned).
8. **Verify** (don't trust self-report): re-run driver, read JSON, confirm structural PASS + magnitudes.
9. **B5**: real-test each arm's convergence solution (real MetaSchedule tune + real finetune AP70) → confirm A-serial→W_g/dominated, A-joint→P_g. Headline = iso-AP70 latency ratio.
10. **Write back**: `4_design_ablation_proof_v1.md` §results + update `HANDOFF_three_arm_ablation_exec_v1.md`. Then Q axis (§5).

## ★★PROTOCOL-DISCREPANCY FINDING (Agent-APfit, decisive for B4 grid)
B2 calibration used a DIFFERENT finetune protocol than the stage_a anchors → the two AP sets are on DIFFERENT surfaces:
- stage_a anchors (base/p25/p50/p75): **DepGraph** proportional prune → finetune from pruned ckpt.
- B2 iso/mixed: **L1 channel-select from BASE** → finetune from L1-selected ckpt (better init).
Evidence: mixed [32,96,192] (L1-transfer) AP70=0.6288 vs pruned25 [48,96,192] (DepGraph) AP70=0.5905 — mixed has MORE
s0 reduction yet +0.034 HIGHER AP. ⇒ finetune PROTOCOL drives AP, not width alone. NNLS mixing both = invalid surface.

**DECISION — B4 rigorous grid must use ONE protocol (stage_a DepGraph).** The self-consistent grid =
**{base, p50, p75, trap25(=p25), pad64}** = 5 widths (the SMOKE_WIDTHS), AP from stage_a, latency from gap1. Contains
the W_g/P_g pair (trap25/pad64, both AP70=0.5905 by weight-identity, exact). The B2 iso/mixed points (L1-transfer) are a
SEPARATE finding (AP over-parameterization), NOT grid members. The 12 B1 latency widths have NO DepGraph AP → not grid
members yet either. So both expansions yielded LIMITS (latency non-additive; AP protocol-dependent), not a bigger grid.

**STRENGTHEN PLAN (high-ROI, do AFTER B1 latency lands ~20:00):** DepGraph-finetune the misaligned-s0 widths whose
zero-padded versions are already in the B1 latency grid → each gives a NEW W_g/P_g pair at a different AP level:
- mix_b [48,64,256] (s0=48 misaligned, W_g) ↔ pad → [64,64,256]=s1_64 (P_g). One DepGraph finetune of mix_b gives both
  (s1_64 AP = mix_b AP by weight-identity padding).
- mix_d [48,128,128] (W_g) ↔ pad → [64,128,128]=s2_128 (P_g). One finetune.
⇒ ~2 DepGraph finetunes turn the 1-pair grid into a 3-pair grid (trap25/pad64 + mix_b/s1_64 + mix_d/s2_128) across the
AP range — much stronger "structural, not cherry-picked." GATE on B1 latency first: only finetune a W_g candidate if its
misaligned version shows LOW tuning ratio (~2×) AND its padded version shows HIGH ratio (~7×+), i.e. a confirmed rank-flip.

## Grid (20 pts)
8 gap1: base/p50/p75/trap25/pad64/iso_s0/iso_s1/iso_s2.
12 new (B1, running): s0_16[16,128,256], s0_32[32,128,256], s1_32[64,32,256], s1_64[64,64,256],
s2_64[64,128,64], s2_128[64,128,128], mix_a[32,96,192]=B2 mixed, mix_b[48,64,256], mix_c[16,128,128],
mix_d[48,128,128], mix_e[64,96,128], mix_f[32,64,64].

## AP coverage (real AP70) — B2 LANDED (CALIBRATED)
Real finetuned AP70 (TRT FP16, DAIR 1789): base .6309, p50 .5641, p75 .530, trap25/pad64 .5905 (anchors);
iso_s0 .6299, iso_s1 .6336, iso_s2 .6339, mixed[32,96,192] .6288 (B2 finetune).

## ★B2 KEY FINDING — AP axis is weak (over-parameterization); AP model is MARGINAL
- Single-stage AND mixed reductions barely move AP70 (all ~0.63). iso_s0 [48,128,256]=0.6299 ≈ base 0.6309 (Δ−0.001).
  AP only drops under AGGRESSIVE UNIFORM pruning (p50 .564, p75 .530). Corroborates `project-dair-ap-axis-collapse`.
- Additive-log model MISSPECIFIED: fit MAE 0.009 but **held-out MAE 0.035** (mixed pred .594 vs real .629, systematic
  under-prediction of high-AP widths). b0=.035>b1=.022>b2=.021 (the fit smears the uniform-pruning drop across stages,
  contradicting the iso evidence that each stage alone barely matters → the truth is non-additive/saturating).
- **Structural W_g/P_g claim UNAFFECTED**: it's a latency result at fixed AP niche (trap25 vs pad64 both .5905 exact by
  weight-identity); A-serial locks trap25 via default-LATENCY ranking (42355<47407), not AP. Both arms share the AP model
  → joint-vs-serial comparison stays valid even with a biased AP axis.

## ★candidate_widths DECISION (revise step 5 above)
Given the AP model is marginal, the HEADLINE B4 search grid = the **9 real-AP70 widths** (base/p50/p75/trap25/pad64/
iso_s0/iso_s1/iso_s2/mixed), all of which also have real latency (mixed=mix_a in B1 grid). The 12 new B1 latency-only
widths have NO real AP → keep them OUT of the headline grid (model AP untrustworthy), OR include as a clearly-labeled
model-AP SENSITIVITY pass only. Do NOT let model-AP widths drive the headline. So candidate_widths' current restriction
to apm.exact is arguably CORRECT — but confirm mixed/iso are in apm.exact after B2 (they are real). Net grid ≈ 9 widths.
- Caveat to note in writeup: 9-width grid is small; defend via "structural exclusion holds on any grid w/ a W_g/P_g pair,
  searcher autonomously locks W_g (not hand-picked)". If reviewers need a bigger search, the only honest expansion is
  more real finetunes (expensive; the model-guided strategy was meant to avoid it) — flag as future work.
