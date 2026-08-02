# Cost-model Selection Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a publication-ready three-panel cost-model selection figure, a three-row summary table, source data, and reproducible Python exports.

**Architecture:** Store the audited 24-candidate metrics in a source CSV. A focused Matplotlib script validates the candidate matrix, derives the three retained-head summary rows, renders one score-ranked panel per target, and exports the complete figure bundle. Unit tests cover data completeness, retained-head identity, and generated artifact existence.

**Tech Stack:** Python 3, standard library `csv/pathlib/unittest`, Matplotlib.

## Global Constraints

- Use Python/Matplotlib exclusively for drawing, previewing, exporting, and QA.
- Render a double-column figure approximately 183 mm wide and 65--75 mm high.
- Plot mean inner-validation score on the x-axis; lower is better.
- Use color for model family, marker size for selected folds, and a star marker for the retained head.
- Include all 24 candidates exactly once.
- Export SVG, PDF, 600-dpi TIFF, and 300-dpi PNG plus source and summary CSV files.
- Do not imply that AP70 Huber residual has the lowest standalone OOF MAE.

---

### Task 1: Source-data contract and tests

**Files:**
- Create: `multi_agent/figure/cost_model_selection/cost_model_selection_source.csv`
- Create: `framework/tests/test_cost_model_selection_figure.py`

**Interfaces:**
- Consumes: audited fixed-candidate metrics and inner-fold selection evidence.
- Produces: a 24-row CSV with columns `target`, `candidate`, `family`, `objective`, `encoding`, `oof_mae`, `oof_spearman`, `inner_score`, `selected_folds`, and `retained`.

- [x] **Step 1: Add the complete 24-row source CSV**

Populate eight candidates for each of `latency`, `energy`, and `ap70`; mark exactly one retained candidate per target.

- [x] **Step 2: Write failing contract tests**

Test that the plotting module exposes `load_rows`, `validate_rows`, and `retained_rows`, that each target has eight candidates, and that the retained heads are ExtraTrees-log, ExtraTrees-log, and LightGBM-Huber-residual.

- [x] **Step 3: Run the tests and confirm import failure**

Run:

```bash
python -m unittest framework.tests.test_cost_model_selection_figure -v
```

Expected: failure because `make_cost_model_selection_figure.py` does not yet exist.

### Task 2: Plotting implementation

**Files:**
- Create: `multi_agent/figure/cost_model_selection/make_cost_model_selection_figure.py`

**Interfaces:**
- Consumes: `cost_model_selection_source.csv`.
- Produces:
  - `cost_model_selection_summary.csv`
  - `cost_model_selection.png`
  - `cost_model_selection.svg`
  - `cost_model_selection.pdf`
  - `cost_model_selection.tiff`

- [x] **Step 1: Implement data loading and validation**

Implement:

```python
def load_rows(path: Path) -> list[dict[str, object]]: ...
def validate_rows(rows: list[dict[str, object]]) -> None: ...
def retained_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]: ...
```

Validation must require 24 unique `(target, candidate)` records, eight candidates per target, one retained head per target, and selected-fold counts in `[0, 5]`.

- [x] **Step 2: Implement the three-panel figure**

Implement:

```python
def build_figure(rows: list[dict[str, object]]):
    ...
```

Each panel sorts candidates from low to high inner score, uses family-consistent colors, scales point sizes by selected-fold count, marks the retained head with a star, and annotates candidates selected at least once.

- [x] **Step 3: Implement publication exports**

Implement:

```python
def export_figure(fig, output_stem: Path) -> None:
    ...
```

Set editable SVG/PDF fonts and save PNG at 300 dpi and TIFF at 600 dpi.

- [x] **Step 4: Run contract tests**

Run:

```bash
python -m unittest framework.tests.test_cost_model_selection_figure -v
```

Expected: all tests pass.

### Task 3: Render and visual QA

**Files:**
- Generate: `multi_agent/figure/cost_model_selection/cost_model_selection_summary.csv`
- Generate: `multi_agent/figure/cost_model_selection/cost_model_selection.png`
- Generate: `multi_agent/figure/cost_model_selection/cost_model_selection.svg`
- Generate: `multi_agent/figure/cost_model_selection/cost_model_selection.pdf`
- Generate: `multi_agent/figure/cost_model_selection/cost_model_selection.tiff`

**Interfaces:**
- Consumes: the plotting script and validated source CSV.
- Produces: publication and preview artifacts for author review.

- [x] **Step 1: Render the full bundle**

Run:

```bash
python multi_agent/figure/cost_model_selection/make_cost_model_selection_figure.py
```

Expected: five generated artifacts and a terminal summary identifying the three retained heads.

- [x] **Step 2: Verify file signatures and dimensions**

Check that PNG/TIFF are valid raster images, SVG contains editable `<text>` elements, PDF has one page, and all files are non-empty.

- [x] **Step 3: Inspect the PNG preview**

Open the PNG and verify:

- all three panels are present;
- all candidate labels are readable;
- no text is clipped or overlapping;
- retained heads are visually dominant but not misleading;
- selected-fold annotations match `5/5`, `4/5`, and `3/5`.

- [x] **Step 4: Re-run tests after any visual adjustment**

Run:

```bash
python -m unittest framework.tests.test_cost_model_selection_figure -v
```

Expected: all tests pass after final styling.
