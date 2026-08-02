# Three Comparison Tables on One Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the TensorRT/H800, TensorRT/Orin, and TVM/H800 comparisons on one
physical page as independently numbered Tables 1, 2, and 3.

**Architecture:** Replace the two independent double-column floats with one
page-level `table*` float that contains three tabular blocks and three captions.
Populate the third block from the locked TVM/H800 `DeltaAP_max=0.10` rows.

**Tech Stack:** LaTeX, AAAI two-column template, `booktabs`, pdfTeX.

## Global Constraints

- Modify only `multi_agent/paper/Latex/AnonymousSubmission2027.tex`.
- Preserve the current Table 1 and Table 2 numeric values.
- Keep three independent table captions, labels, and reference numbers.
- Keep every caption below its table under the AAAI template.
- Keep every caption title to one rendered line.
- Round every displayed numeric cell to two decimal places.
- Leave TVM Original/default empty; do not insert PyTorch/cuDNN values.
- Represent unavailable or failed measurements as `--`, never invented values.
- Do not commit changes in the shared dirty worktree.

---

### Task 1: Build the unified three-table float

**Files:**
- Modify: `multi_agent/paper/Latex/AnonymousSubmission2027.tex:420-540`

**Interfaces:**
- Consumes:
  `multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/31*_7_14_交接文档_框架待完善项与阶段6六臂对照实施计划_v1.md`
- Produces: labels `tab:trt-main-comparison`,
  `tab:orin-stage6-comparison`, and `tab:tvm-main-comparison`

- [ ] **Step 1: Consolidate the floats**

Move the existing Table 1 and Table 2 tabular blocks into one `table*` float
with `[p]` placement. Give each tabular block its own caption and label.

- [ ] **Step 2: Add Table 3**

Add the five-route TVM/H800 table using sections 11.5 and 14.5. Keep
Original/default empty, keep Pyramid Schedule only empty, and reserve all
F-Cooper cells.

- [ ] **Step 3: Validate source content**

Run a script that checks unique labels, five rows per table, placeholders, and
the Table 3 numeric sequence against the source document.

- [ ] **Step 4: Compile and inspect**

Run pdfTeX twice, confirm the auxiliary file numbers the labels 1, 2, and 3,
render the relevant PDF page to PNG, and inspect for clipping or overlap.

- [ ] **Step 5: Review**

Review the scoped diff and obtain an independent read-only review of values,
caption disclosures, numbering, and one-page layout risk.

- [ ] **Step 6: Apply the AAAI caption and precision revision**

Keep all three captions below their respective tabular blocks, shorten each
caption to a one-line title, move measurement notes into the preceding prose,
and format every numeric cell with exactly two digits after the decimal point.
