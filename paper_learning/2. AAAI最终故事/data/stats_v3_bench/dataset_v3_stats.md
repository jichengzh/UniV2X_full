# Dataset v3 Stats Report (auto-generated 2026-05-26 19:47)

## Summary
- N anchors: **4704** (4584 with AP)
- Pass/fail per metric: **9 ✅ / 14**
- Verdict: **❌ Phase 4 needed (补点)**

## Coverage (A)
- A1 marginal: ✅ entropies={'planes_s3': 0.8535829957953921, 'q': 0.9999999999964027, 'd': 0.9999999999907668}
- A2 B×Q grid: ✅ filled 147/147 = 100%
- A3 B×FT grid: ✅ filled 672/672 = 100%
- A4 PCA hull: ✅ covering radius=4.865824346229248e-15

## Distribution (B)
- B1 AP histogram: ❌ std=0.11627580267910334 range=0.779501951002882
- B2 class balance: ❌ counts={'collapse (<0.3)': 341, 'degrade (0.3-0.5)': 245, 'healthy (>=0.5)': 3998}
- B3 axis box: ✅ spans={'triplet': 0.04840177048295691, 'q': 0.2550619281896459, 'd': 0.0033130845937604514}

## Information (C)
- C1 MI/Spearman: ❌ n_strong=1
- C2 corr matrix: ✅ max off-diag |r|=0.9384031756455699
- C3 interaction H: ✅ max H=0.10314144742624926

## Learnability (D)
- D1 5-fold CV: ✅ R²=0.9094456223521282 MAE=0.015333977563397402
- D2 learning curve: ✅
- D3 OOD: ❌ scenarios={'hold p97': None, 'hold int8_pc_wo': 0.010439454376092653, 'hold int8_ent': 0.24761584291187305}

## Noise (E)
- E1 noise floor: ❌ σ_noise=0.09643540519891415 R²_ceiling=0.07002126241212425

## Figure index (14 PNG in stats_v3/)
- A组 (覆盖度): 01_coverage_marginal.png, 02_coverage_grid_BxQ.png, 03_coverage_grid_BxFT.png, 04_coverage_hull_pca2d.png
- B组 (分布): 05_target_histogram.png, 06_target_class_balance.png, 07_target_by_axis_box.png
- C组 (信息量): 08_mi_spearman_per_feature.png, 09_corr_matrix.png, 10_interaction_hstat.png
- D组 (可学习性): 11_cv_r2_per_fold.png, 12_learning_curve.png, 13_ood_predict_vs_real.png
- E组 (噪声): 14_noise_floor.png

## Anchor source breakdown
- e2e_bench_v1: 4704