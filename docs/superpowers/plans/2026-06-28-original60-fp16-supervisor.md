# Original60 FP16 Supervisor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an automated supervisor for original60 FP16 AP checkpoint generation/eval, while fixing lane failure handling and per-label port allocation so four-GPU backfill can continue without manual巡检.

**Architecture:** Keep the existing train launcher and eval watcher scripts, but add a supervisor that reads the launch queues, audits GPU3/4/5/6 availability, launches per-GPU lane runners, tracks state, and refreshes progress artifacts. Move master-port selection from fixed per-GPU values to unique per-label values, and make lane runners treat per-label failures as recorded blockers instead of lane-fatal exceptions.

**Tech Stack:** Python 3, existing V2X scripts, unittest/pytest-compatible test suite.

---

### Task 1: Lock Down Port Allocation and Lane Failure Semantics

**Files:**
- Modify: `scripts/stage2_generate_original60_fp16_ap_launch_plan.py`
- Modify: `scripts/stage2_original60_fp16_lane_runner.py`
- Test: `framework/tests/test_stage2_original60_fp16_ap_launch_plan.py`
- Create: `framework/tests/test_stage2_original60_fp16_lane_runner.py`

- [ ] Add failing tests for unique per-label master port generation and lane continuing after a blocked label.
- [ ] Run targeted tests and confirm they fail for the expected reasons.
- [ ] Implement minimal changes in launch-plan and lane-runner to satisfy the tests.
- [ ] Re-run targeted tests until green.

### Task 2: Add Supervisor Orchestration

**Files:**
- Create: `scripts/stage2_original60_fp16_supervisor.py`
- Create: `framework/tests/test_stage2_original60_fp16_supervisor.py`
- Possibly modify: `scripts/stage2_original60_fp16_ap_batch_launcher.py` if helper reuse is cleaner

- [ ] Add failing tests for GPU audit, per-GPU label selection, lane process launch metadata, and persisted supervisor state.
- [ ] Run targeted supervisor tests and confirm they fail before implementation.
- [ ] Implement the supervisor with dry-run and executable modes, low-frequency polling, and persisted state/log outputs.
- [ ] Re-run the targeted tests until green.

### Task 3: Verify End-to-End Script Surface

**Files:**
- Modify: `scripts/stage2_original60_fp16_supervisor.py`
- Possibly modify: `scripts/stage2_watch_original60_fp16_progress.py` only if state refresh integration is needed
- Test: targeted unittest commands

- [ ] Run focused tests covering launch-plan, lane-runner, and supervisor together.
- [ ] Do a syntax-level dry run of the supervisor CLI against current artifacts.
- [ ] Summarize the new command surface and expected state files for batch execution on H800.
