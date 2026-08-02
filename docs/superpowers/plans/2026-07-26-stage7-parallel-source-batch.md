# Stage7 Parallel Source Batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every frozen Stage7 source-materialization group in one four-candidate batch concurrently on its separately leased H800 GPU without changing measurement, AP, cache, feedback, or selected-event budget semantics.

**Architecture:** Keep the lease controller and one indivisible source batch as the ownership boundary. Replace the resolver's serial child loop with one bounded concurrent launch whose width equals the frozen group-job count (maximum four), wait for all launched children, then write process evidence in frozen group order and publish either one formal source result or one zero-budget retry. A failed child never cancels another already launched child; successful source artifacts remain available for the next immutable retry attempt.

**Tech Stack:** Python 3, `concurrent.futures.ThreadPoolExecutor`, pytest, existing Stage7 immutable JSON contracts, H800 dynamic UUID leases.

## Global Constraints

- Only the Stage7 source-resolution execution layer may change; Stage5/Stage3 materialization, measurement, AP, cache, and actual-feedback semantics remain frozen.
- GPU7 remains excluded and existing non-Stage7 jobs must not be stopped, modified, or preempted.
- Source-only pre-gate execution performs no latency, energy, AP, cache reveal, feedback, or T=16 budget consumption.
- A batch validates its signed lease and inventory immediately before launch and starts no child if that admission fails.
- All launched children are awaited; process receipts are written in frozen `group_jobs` order regardless of completion order.
- Any failed child prevents formal source-result publication and produces a zero-budget retry for the failed group keys; successful artifacts may be reused by the subsequent immutable attempt.
- Raw child stdout/stderr remain absent in `pre_gate_source_only`; only hashes and byte counts are persisted.
- Existing result roots and deployed attempts are never overwritten; a changed deployment SHA requires a newly isolated attempt/root.

---

### Task 1: Lock the concurrent batch contract with tests

**Files:**
- Modify: `framework/tests/test_stage7_resolve_round_sources_v2.py`

**Interfaces:**
- Consumes: `run_source_resolution(..., run_command=Callable, pre_gate_source_only=bool)`
- Produces: regression coverage for concurrent launch, ordered evidence, aggregate retry, and fail-before-launch runtime admission

- [ ] **Step 1: Add a failing overlap test**

Use a `threading.Barrier` inside the injected `run_command` to prove all unique group commands overlap. Assert the maximum active count equals the frozen group count, every command has a distinct signed `--gpu`, and process receipts remain ordered by `execution_plan["group_jobs"]`.

- [ ] **Step 2: Run the overlap test and verify RED**

Run:

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage7_resolve_round_sources_v2.py -k parallel -vv
```

Expected: the serial implementation fails because the barrier cannot be reached by all group jobs.

- [ ] **Step 3: Add aggregate-failure and runtime-admission tests**

Assert that all group commands launch before result publication, multiple failed group keys are preserved in frozen order, `selected_event_budget_delta == 0`, and a lease/inventory drift detected by the second preflight probe launches zero children.

- [ ] **Step 4: Run the focused tests and keep the expected failures**

Run:

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage7_resolve_round_sources_v2.py -k "parallel or aggregate or rechecks_inventory" -vv
```

Expected: failures identify the missing bounded concurrent batch behavior.

### Task 2: Implement bounded concurrent source launch

**Files:**
- Modify: `scripts/stage7_resolve_round_sources_v2.py`

**Interfaces:**
- Consumes: frozen ordered `group_jobs`, argv-safe commands, signed lease, injected `run_command`
- Produces: ordered `(group, completed_or_error)` outcomes after all child invocations finish

- [ ] **Step 1: Add the smallest concurrent helper**

Use `ThreadPoolExecutor(max_workers=min(4, len(commands)))`, submit one argv-safe `run_command` per frozen group, retain futures in frozen input order, and resolve every future before publishing evidence or retry state.

- [ ] **Step 2: Perform one fail-closed batch preflight**

Call `_verify_runtime` immediately before creating the executor. On drift, return `infrastructure_unavailable` for every frozen group with `materializer_invocation_count == 0`.

- [ ] **Step 3: Preserve ordered process evidence**

After all futures settle, iterate outcomes in frozen group order. Persist the existing process receipt schema and existing pre-gate raw-log suppression contract.

- [ ] **Step 4: Aggregate failures without partial formal publication**

Collect every `OSError` group as infrastructure failure and every non-zero return-code group as evidence failure. If either collection is non-empty, emit one zero-budget retry, preferring `infrastructure_unavailable` when any invocation could not start; do not call `_formal_result`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage7_resolve_round_sources_v2.py -vv
```

Expected: all resolver tests pass.

### Task 3: Reseal dependency and deployment identities

**Files:**
- Modify: `scripts/stage7_source_scheduler_v2.py`
- Verify: `framework/stage7/deployment_bundle_v2.py`
- Test: `framework/tests/test_stage7_source_scheduler_v2.py`
- Test: `framework/tests/test_stage7_deployment_bundle_v2.py`

**Interfaces:**
- Consumes: SHA256 of the verified resolver implementation
- Produces: exact scheduler pin and deployment manifest containing the same resolver bytes

- [ ] **Step 1: Compute the resolver SHA**

Run:

```bash
sha256sum scripts/stage7_resolve_round_sources_v2.py
```

- [ ] **Step 2: Update only the exact resolver pin**

Replace `PHASE3A_REPLACEMENT_PINS["scripts/stage7_resolve_round_sources_v2.py"]` with the computed digest.

- [ ] **Step 3: Verify Stage7 tests and coverage**

Run:

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage7_resolve_round_sources_v2.py framework/tests/test_stage7_source_scheduler_v2.py framework/tests/test_stage7_deployment_bundle_v2.py --cov=scripts.stage7_resolve_round_sources_v2 --cov-report=term-missing -vv
```

Expected: zero failures and at least 80% coverage for the modified resolver.

- [ ] **Step 4: Run the complete Stage7 regression**

Run:

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest framework/tests/test_stage7_*.py -q
```

Expected: zero failures.

### Task 4: Review, deploy, and observe four-GPU overlap

**Files:**
- Create only under a new isolated Stage7 v2 attempt/root on H800
- Preserve all prior local and remote result roots read-only

**Interfaces:**
- Consumes: verified local deployment bundle, exact manifest/release SHA, four free non-GPU7 H800 UUIDs
- Produces: one running Stage7 source-only batch with four concurrent materializer process trees and auditable lease/evidence state

- [ ] **Step 1: Obtain an independent code review**

Review concurrency, failure aggregation, immutable evidence ordering, raw-log suppression, GPU7 exclusion, and deployment pin propagation. Resolve all Critical and Important findings.

- [ ] **Step 2: Re-audit the H800 before switching**

Read-only inspect GPU UUIDs, active processes, F-Cooper reservation, current Stage7 PID tree, locks, leases, and result-root ownership. Stop only the current Stage7 serial attempt after the new bundle is ready and four eligible GPUs remain available.

- [ ] **Step 3: Isolate the stopped attempt and deploy verified bytes**

Move the stopped Stage7 attempt to a timestamped blocked archive, create a fresh v2 root, transfer the exact bundle, rebind owner/mode, and validate local/remote manifest SHA equality.

- [ ] **Step 4: Launch one persistent orchestrator**

Resume the same frozen selected IDs and source contracts. Do not launch a duplicate orchestrator and do not import invalid-affinity, stale-deployment, or old trajectory state.

- [ ] **Step 5: Prove real four-GPU concurrency**

Capture one timestamped audit showing four distinct non-GPU7 UUID leases, four simultaneous materializer process trees, one job per GPU, unchanged formal budget/cache/feedback counters, and no performance/AP processes.

- [ ] **Step 6: Continue the existing gate sequence**

After source materialization closes, rerun the no-GPU gate. Only after it passes may the four-arm pilot begin under the original Stage7 contract.
