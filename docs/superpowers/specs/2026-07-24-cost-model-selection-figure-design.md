# Cost-model selection figure design

## Objective

Show why the final latency, energy, and AP70 cost-model heads were selected
without placing the full 24-row candidate matrix in the main paper.

## Core conclusion

Under the same Gold176 grouped nested-cross-validation protocol, the retained
heads minimize the mean inner-validation selection score and are selected in
the largest number of outer folds for their respective targets.

## Main-paper summary table

Keep only three rows:

1. latency: ExtraTrees with `log1p` target encoding;
2. energy: ExtraTrees with `log1p` target encoding;
3. AP70: LightGBM with Huber loss and model-anchor residual encoding.

Columns: target, retained model, target encoding, MAE, Spearman correlation,
mean inner score, and selected outer folds.

## Figure contract

- Archetype: quantitative grid.
- Backend: Python with Matplotlib only.
- Final size: double-column, approximately 183 mm wide and 65--75 mm high.
- Layout: three horizontally aligned panels for latency, energy, and AP70.
- X-axis: mean inner-validation selection score; lower is better.
- Y-axis: the eight candidate model/encoding combinations, sorted by score.
- Point color: ExtraTrees versus LightGBM.
- Point size: number of outer folds in which the candidate was selected.
- Final retained head: star marker with a dark outline.
- Direct labels: selected-fold count next to candidates selected at least once.
- Palette: restrained, colorblind-safe blue for ExtraTrees, orange for
  LightGBM, and dark red only for the retained-head outline/accent.
- No error bars: the source report contains fold-specific selection scores but
  no repeated-seed uncertainty estimate. The plotted point is the five-fold
  mean; the figure legend must state this boundary.

## Data and selection rule

- Dataset: Gold176, 176 rows and 44 complete `(model, width)` groups.
- Evaluation: five outer folds and three inner folds, grouped by
  `(model, width)`.
- Selection score:
  `MAE / (Q0.9 - Q0.1) + 0.25 * (1 - Spearman)`.
- The full 24-row candidate matrix remains available as supplementary source
  data but is not repeated in the main paper.

## Exports

Generate:

- editable SVG;
- editable-text PDF;
- 600-dpi TIFF;
- 300-dpi PNG preview;
- source-data CSV;
- standalone Python plotting script.

## Review risks and controls

1. AP70 Huber residual does not have the lowest fixed-candidate OOF MAE.
   Therefore, the figure must state that selection used the pre-defined joint
   inner-validation score rather than an ex-post single metric.
2. Point size alone is insufficient for exact fold counts, so selected
   candidates receive explicit `n/5` labels.
3. Scores are target-specific and should not be compared numerically across
   panels.
4. Bold/star emphasis denotes the frozen selection, not universal dominance
   on every standalone metric.

## Acceptance criteria

- All 24 candidates appear exactly once.
- The retained heads are latency ExtraTrees-log, energy ExtraTrees-log, and
  AP70 LightGBM-Huber-residual.
- Their mean inner scores and selection counts are respectively
  `0.1024, 5/5`, `0.1412, 4/5`, and `0.1836, 3/5`.
- Labels remain readable at the final double-column size.
- SVG/PDF text remains editable and the PNG/TIFF previews contain no clipping
  or overlaps.
