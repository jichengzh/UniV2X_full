# Dataset v3 Stats Report (auto-generated 2026-05-25 15:05)

## Summary
- N anchors: **54** (54 with AP)
- Pass/fail per metric: **9 ✅ / 14**
- Verdict: **❌ Phase 4 needed (补点)**

## Coverage (A)
- A1 marginal: ✅ entropies={'planes_s3': 0.9182958340516041, 'q': 0.9823887817925145, 'd': 0.9057125980111065}
- A2 B×Q grid: ✅ filled 15/15 = 100%
- A3 B×FT grid: ✅ filled 9/9 = 100%
- A4 PCA hull: ✅ covering radius=1.3534080177941974

## Distribution (B)
- B1 AP histogram: ❌ std=0.03707423132389783 range=0.11679999999999996
- B2 class balance: ❌ counts={'collapse (<0.3)': 0, 'degrade (0.3-0.5)': 8, 'healthy (>=0.5)': 46}
- B3 axis box: ✅ spans={'triplet': 0.06330000000000002, 'q': 0.0595, 'd': 0.010500000000000065}

## Information (C)
- C1 MI/Spearman: ✅ n_strong=5
- C2 corr matrix: ❌ max off-diag |r|=1.0
- C3 interaction H: ✅ max H=0.11649356295419688

## Learnability (D)
- D1 5-fold CV: ✅ R²=0.8821458196935537 MAE=0.007018345695299959
- D2 learning curve: ✅
- D3 OOD: ❌ scenarios={'hold p97': 0.05529582635820834, 'hold int8_pc_wo': 0.02188525048741428, 'hold int8_ent': 0.044769832519632645}

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
- v2_3b: 12
- v2_3a: 6