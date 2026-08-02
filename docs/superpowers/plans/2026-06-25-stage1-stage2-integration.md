# Stage1/Stage2 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Stage1/Stage2 integration package so Stage2 consumes only Stage1 manifest + classification, applies classifier gate, emits scoped optimization output and evidence delta, and has a fresh-clone smoke path.

**Architecture:** Add a focused `framework/stage2/` package for contracts and gate logic, leaving `framework/stage1_bridge.py` as the manifest-to-search-space adapter. The CLI `scripts/stage2_optimize_model.py` composes contracts, classifier gate, `load_stage2_search_space()`, and a deterministic smoke search summary without exposing `evidence_registry_path`, `hardware_target`, or `search_policy` as user inputs.

**Tech Stack:** Python dataclasses, JSON/YAML, pytest, existing `framework.stage1` classifier and `framework.stage1_bridge` APIs.

---

### Task 1: Stage2 Contract And Gate Tests

**Files:**
- Create: `framework/tests/test_stage2_integration_contract.py`
- Create: `framework/stage2/contracts.py`
- Create: `framework/stage2/__init__.py`

- [ ] **Step 1: Write failing tests**

Create tests that import `Stage2Input`, `Stage2Output`, `Stage2EvidenceDelta`, and `apply_stage1_gate`. Assert that `Stage2Input` only has `manifest_path` and `model_classification_path`, rejects unknown public fields, and derives `hardware_context` from `manifest.hw_capability`.

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage2_integration_contract.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'framework.stage2'`.

- [ ] **Step 3: Implement minimal contracts and gate**

Add dataclasses for `Stage2Input`, `Stage2Output`, `Stage2EvidenceRecord`, and `Stage2EvidenceDelta`. Add `apply_stage1_gate(manifest_path, classification_path)` that returns fail-closed for `SCAN_FAILED`, `serial` for `SEPARABLE_ACCELERATION`, and `joint` for `CO_ACCELERATION_REQUIRED`.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage2_integration_contract.py`.

### Task 2: Stage2 CLI

**Files:**
- Create: `scripts/stage2_optimize_model.py`
- Modify: `framework/tests/test_stage2_integration_contract.py`

- [ ] **Step 1: Add failing CLI tests**

Add subprocess tests for Pyramid, CoDriving, and a `SCAN_FAILED` model. Assert the CLI accepts only `--manifest`, `--classification`, `--out-json`, optional `--out-md`, optional `--evidence-delta-out`, and optional explicit `--demo`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run: `PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage2_integration_contract.py`.
Expected: FAIL because `scripts/stage2_optimize_model.py` does not exist.

- [ ] **Step 3: Implement CLI**

Load Stage1 manifest, load classification, apply gate, build `load_stage2_search_space()`, emit `Stage2Output`, and optionally emit `Stage2EvidenceDelta`. Never expose `evidence_registry_path`, `hardware_target`, or `search_policy`.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=/home/jichengzhi/V2X pytest -q framework/tests/test_stage2_integration_contract.py`.

### Task 3: Evidence Delta Update Contract

**Files:**
- Modify: `framework/stage2/contracts.py`
- Create: `scripts/stage2_update_evidence.py`
- Modify: `framework/tests/test_stage2_integration_contract.py`

- [ ] **Step 1: Add failing evidence delta tests**

Assert that evidence delta records require `backend`, `hardware`, `scope`, `evidence_kind`, `provenance`, `candidate_config`, and `metric`, reject demo/proxy as measured, and preserve historical TRT as historical only.

- [ ] **Step 2: Implement validation**

Add evidence delta validation and a small update script that validates and copies/merges delta JSON into a Stage1 evidence update directory without changing classifier semantics.

- [ ] **Step 3: Verify**

Run the focused pytest and py_compile commands.

### Task 4: Fresh Clone Smoke Docs

**Files:**
- Modify: `github/stage1-model-scanner-aaai/README.md`
- Modify: `github/stage1-model-scanner-aaai/README.zh-CN.md`
- Create: `github/stage1-model-scanner-aaai/docs/stage2-evidence-delta.zh-CN.md`
- Create: `github/stage1-model-scanner-aaai/docs/stage2-new-hardware.zh-CN.md`
- Copy/adapt: `framework/stage2/`, `scripts/stage2_optimize_model.py`, `scripts/stage2_update_evidence.py`

- [ ] **Step 1: Add docs**

Document that Stage2 public input is only manifest + classification, `hw_capability` is manifest-derived read-only context, demo/proxy is not paper evidence, and dense-core results cannot be promoted to full-model claims.

- [ ] **Step 2: Verify smoke commands**

Run py_compile and one CLI demo smoke in both V2X root and export repo.

### Task 5: Completion Audit

**Files:**
- Current worktree only.

- [ ] **Step 1: Verify all five gaps**

Run focused pytest, py_compile, CLI smoke for Pyramid/CoDriving/SCAN_FAILED, evidence delta validation, and export repo smoke.

- [ ] **Step 2: Decide goal status**

Only call `update_goal(status="complete")` if current evidence proves all five gaps. Leave the goal active if any proof is missing.
