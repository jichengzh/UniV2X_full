# Dataset v3 Stats Report (auto-generated 2026-05-25 01:12)

## Summary
- N anchors: **83** (83 with AP)
- Pass/fail per metric: **9 ✅ / 14**
- Verdict: **❌ Phase 4 needed (补点)**

## Coverage (A)
- A1 marginal: ❌ entropies={'planes_s3': 0.8676293117096252, 'q': 0.9978518947461698, 'ft': 0.7240078598932634, 'd': 0.7298506669324165}
- A2 B×Q grid: ✅ filled 15/15 = 100%
- A3 B×FT grid: ✅ filled 15/15 = 100%
- A4 PCA hull: ✅ covering radius=0.23390936334437945

## Distribution (B)
- B1 AP histogram: ✅ std=0.20124649584337798 range=0.5718
- B2 class balance: ✅ counts={'collapse (<0.3)': 18, 'degrade (0.3-0.5)': 19, 'healthy (>=0.5)': 46}
- B3 axis box: ✅ spans={'triplet': 0.11449999999999999, 'q': 0.054800000000000015, 'ft': 0.5597, 'd': 0.05949999999999994}

## Information (C)
- C1 MI/Spearman: ✅ n_strong=6
- C2 corr matrix: ❌ max off-diag |r|=1.0
- C3 interaction H: ❌ max H=0.026585313726333947

## Learnability (D)
- D1 5-fold CV: ✅ R²=0.937167652521708 MAE=0.022851184947804623
- D2 learning curve: ✅
- D3 OOD: ❌ scenarios={'hold p97': 0.08370422143954734, 'hold int8_pc_wo': 0.023810620594241338, 'hold FT=2': 0.2656263520878475}

## Noise (E)
- E1 noise floor: ❌ σ_noise=0.09643540519891415 R²_ceiling=0.07002126241212425

## Figure index (14 PNG in stats_v3/)
- A组 (覆盖度): 01_coverage_marginal.png, 02_coverage_grid_BxQ.png, 03_coverage_grid_BxFT.png, 04_coverage_hull_pca2d.png
- B组 (分布): 05_target_histogram.png, 06_target_class_balance.png, 07_target_by_axis_box.png
- C组 (信息量): 08_mi_spearman_per_feature.png, 09_corr_matrix.png, 10_interaction_hstat.png
- D组 (可学习性): 11_cv_r2_per_fold.png, 12_learning_curve.png, 13_ood_predict_vs_real.png
- E组 (噪声): 14_noise_floor.png

## Anchor source breakdown
- v3_1.2: 36
- v3_1.1: 21
- v2_3b: 12
- v2_3a: 9
- v3_1.3: 5