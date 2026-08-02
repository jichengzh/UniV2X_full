# Orin Stage6 Table 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-ready Orin Stage6 comparison table containing the latest
Pyramid and CoDriving five-configuration AP70, latency, and energy results,
while reserving the same five F-Cooper rows.

**Architecture:** Insert one new `table*` immediately after the existing H800
comparison table and its interpretation. Reuse the existing
`tabular*`/`booktabs` style, but explicitly bind the new values to Orin
multiscale-backbone scope, batch 2, and the `VIN_SYS_5V0` energy rail.

**Tech Stack:** LaTeX, `booktabs`, AAAI two-column paper template.

## Global Constraints

- Modify only `multi_agent/paper/Latex/AnonymousSubmission2027.tex`.
- Preserve the existing H800 table and all H800 values.
- Use AP70, median CUDA-event latency in milliseconds, and energy in joules
  per batch-2 backbone invocation.
- Identify the joint row as an Orin FP16 control corresponding to the H800
  INT8-selected structure.
- Reserve F-Cooper cells with `--`; do not invent measurements.
- Use the unique label `tab:orin-stage6-comparison`.
- Do not commit changes in the shared dirty worktree.

---

### Task 1: Add and verify Orin Stage6 Table 2

**Files:**
- Modify:
  `multi_agent/paper/Latex/AnonymousSubmission2027.tex`

**Interfaces:**
- Consumes:
  `results/lane_c_orin_stage6_five_config_20260724/summary_fp16_joint_sudo_energy/final_summary.json`
- Produces: LaTeX table label `tab:orin-stage6-comparison`

- [ ] **Step 1: Add a grouped three-model `table*`**

  Insert five method rows with Pyramid and CoDriving AP70/latency/energy.
  Populate the corresponding F-Cooper cells with `--`.

- [ ] **Step 2: Add the measurement contract**

  State in the caption or adjacent prose that latency covers batch-2
  multiscale-backbone CUDA-event compute without transfer, and energy is
  `VIN_SYS_5V0` mean power multiplied by median latency.

- [ ] **Step 3: Validate content**

  Run:

  ```bash
  rg -n 'tab:orin-stage6-comparison|VIN_SYS_5V0|Joint FP16' \
    multi_agent/paper/Latex/AnonymousSubmission2027.tex
  ```

  Expected: one unique table label and explicit scope/precision wording.

- [ ] **Step 4: Compile the paper**

  Run the repository-available LaTeX build command from
  `multi_agent/paper/Latex`. Expected: successful PDF generation with no
  undefined reference for `tab:orin-stage6-comparison`.

- [ ] **Step 5: Review the diff**

  Confirm the H800 table is unchanged, F-Cooper contains only `--`, and the
  inserted values match the locked JSON summary.
