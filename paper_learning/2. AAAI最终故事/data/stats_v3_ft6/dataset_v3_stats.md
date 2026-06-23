# Dataset v3 Stats Report (auto-generated 2026-05-26 14:32)

## Summary
- N anchors: **20** (20 with AP)
- Pass/fail per metric: **4 ✅ / 14**
- Verdict: **❌ Phase 4 needed (补点)**

## Coverage (A)
- A1 marginal: ❌ entropies={'planes_s3': 0.8112781244562475, 'q': 0.9349775297640167, 'd': 0.0}
- A2 B×Q grid: ✅ filled 15/15 = 100%
- A3 B×FT grid: ✅ filled 3/3 = 100%
- A4 PCA hull: ❌ covering radius=1.6839541342061985

## Distribution (B)
- B1 AP histogram: ❌ std=0.039039497331812834 range=0.12993382154596472
- B2 class balance: ❌ counts={'collapse (<0.3)': 0, 'degrade (0.3-0.5)': 5, 'healthy (>=0.5)': 15}
- B3 axis box: ❌ spans={'triplet': 0.05942460881212591, 'q': 0.049699999999999966, 'd': 0.0}

## Information (C)
- C1 MI/Spearman: ✅ n_strong=6
- C2 corr matrix: ❌ max off-diag |r|=1.0
- C3 interaction H: ✅ max H=0.20463743642070326

## Learnability (D)
- D1 5-fold CV: ❌ R²=0.4187477027255949 MAE=0.01587108231970153
- D2 learning curve: ❌
- D3 OOD: ❌ scenarios=NA

## Noise (E)
- E1 noise floor: ❌ σ_noise=0.023389296665267834 R²_ceiling=0.945294080150409

## Figure index (14 PNG in stats_v3/)
- A组 (覆盖度): 01_coverage_marginal.png, 02_coverage_grid_BxQ.png, 03_coverage_grid_BxFT.png, 04_coverage_hull_pca2d.png
- B组 (分布): 05_target_histogram.png, 06_target_class_balance.png, 07_target_by_axis_box.png
- C组 (信息量): 08_mi_spearman_per_feature.png, 09_corr_matrix.png, 10_interaction_hstat.png
- D组 (可学习性): 11_cv_r2_per_fold.png, 12_learning_curve.png, 13_ood_predict_vs_real.png
- E组 (噪声): 14_noise_floor.png

## Anchor source breakdown
- v3_7.11_ft_sweep: 15
- v3_7.11_noise_ft6: 5