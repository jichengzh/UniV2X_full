# Stage7 actual-feedback v3 core-ablation adapter implementation plan

> Supersedes `2026-07-25-stage7-scanner-deferred-core-ablation-v2.md` wherever
> that plan routes v2 through the scanner-gated Stage7 v1 controller.  Existing
> actual-feedback v3 core files and formal Stage5/Stage6 evidence remain
> read-only.

**Goal:** Close the scanner-deferred four-variant Stage7 selection/cache/
orchestration layer over the existing Stage5/Stage3 actual-feedback v3
execution primitives, prove the no-GPU integration on the formal H800 host,
then execute the frozen pilot and remaining seeds only when four eligible H800
UUID leases are available.

**Architecture:** Stage7 owns the canonical trajectory, variant selector,
selected-only exact cache, logical-to-physical request binding, scheduling and
paper aggregation.  Hardware truth remains owned by the frozen Stage5/Stage3
v3 materialization, performance, numerical/AP and promotion primitives.  A
logical round always has four selected events; the physical plan contains only
cache misses and may contain zero to four rows.  Promotion and feedback release
remain a four-row atomic barrier after exact hits and physical terminals are
merged in logical order.

**Frozen scope:** `S7-PYR-TVM`, H800, `tvm_auto`; variants `full`,
`without_surrogate`, `without_measured_feedback`, `backend_blind`; seeds
`20260718..20260720`; four rounds of four candidates.  The shared ordered
pre-scan pool contains 686 candidates and has content SHA
`3294a3e66671af2778bffd4d719ab9f45963d948cabdfb21315012274daf96f4`.

**Workspace decision:** Do not create a git worktree.  The authoritative
Stage7 implementation is currently untracked in the shared repository, so a
new worktree would silently omit the exact files being repaired.  Every task
must preserve unrelated edits and may touch only its declared files.

## Task 1: Repair the formal v2 root and canonical trajectory contract

**Owns:**

- Modify `scripts/stage7_prepare_core_ablation_v2.py`
- Modify `framework/stage7/core_ablation_v2.py` only if stricter immutable
  validation is required
- Modify `framework/tests/test_stage7_prepare_core_ablation_v2.py`
- Modify `framework/tests/test_stage7_core_ablation_v2.py`

**RED tests:**

- Preparation writes exactly one initial cache with
  `schema_version=stage7_core_cache_v2`, `entries={}`, `lineage=[]`.
- Initialization writes exactly twelve seed-root contracts under
  `variants/<variant>/seed_<seed>/trajectory_contract.json`; round artifacts
  live only under `round_NN`.
- The contract binds formal source/pre-scan/blocker/admission hashes, immutable
  input path+SHA records, scope/input/batch, measurement/AP, HV reference and
  the five frozen actual-v3 executor hashes.
- Any existing v2 root is accepted only after owner/PID/contract/cache-lineage
  recovery validation; drift fails without overwrite.

**GREEN implementation:** Add the minimum immutable contract/provenance fields
and canonical path creation.  Do not read or import v1 trajectory, scanner
state, cache, terminal status or online labels.

**Verify:**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_core_ablation_v2.py \
  framework/tests/test_stage7_prepare_core_ablation_v2.py
```

## Task 2: Generalize the four variant selectors to all rounds

**Owns:**

- Modify `framework/stage7/search_policy_v1.py`
- Add `framework/tests/test_stage7_actual_v3_selector_adapter_v2.py`

**RED tests:**

- All rounds select four unique ordered IDs from the same 686-point pre-scan
  pool and freeze a logical request SHA before cache access.
- `full` performs predicted-frontier-diversity and online refit using promoted
  actual graph feedback.
- `without_surrogate` uses seeded uniform sampling without replacement and
  makes zero surrogate/uncertainty/frontier calls.
- `without_measured_feedback` reuses the round-zero Gold176 bundle and ignores
  online metrics, failures and actual graph features except identity exclusion.
- `backend_blind` removes all capability/backend/profile-derived model inputs
  while retaining the fixed `tvm_auto` route and emits schema/leakage/delta/
  overlap audit data.

**GREEN implementation:** Extend the existing Stage5-backed selector adapter
with `round_index`, selected IDs, promoted feedback and the frozen A2 bundle.
Reuse `production_search_v1.py` and `single_target_search_v2.py`; do not
implement a second surrogate or acquisition algorithm.

**Verify:**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_search_policy_v1.py \
  framework/tests/test_stage7_actual_v3_selector_adapter_v2.py
```

## Task 3: Bind selection, exact cache and miss-only physical requests

**Owns:**

- Modify `framework/stage7/core_cache_v2.py`
- Add `framework/stage7/actual_v3_adapter_v2.py`
- Add `framework/tests/test_stage7_actual_v3_adapter_v2.py`
- Modify `framework/tests/test_stage7_core_cache_v2.py`

**RED tests:**

- Cache is not accepted by any selector API and cannot be opened until the
  ordered selected-ID/request binding is self-authenticated.
- Exact-key dimensions bind candidate/model/profile/hardware/scope/input/batch/
  genome/q_mode/source checkpoint/ONNX/build/tuning/measurement/AP/runtime
  contract identities.
- Empty cache reveals four misses.  Partial hits create a zero-to-four-row
  physical request while preserving the four-row logical request and one
  binding record per logical row.
- Terminal wrappers bind logical request, physical request, selected row,
  Stage5/Stage3 terminal artifacts, actual graph feature SHA and immutable
  cache lineage.
- Candidate failures consume one selected event with null objectives; infra
  and evidence failures preserve the same logical request SHA and consume zero.

**GREEN implementation:** Add pure, immutable adapters and validators only.
Do not execute hardware and do not duplicate Stage3 measurement/AP semantics.

**Verify:**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_core_cache_v2.py \
  framework/tests/test_stage7_actual_v3_adapter_v2.py
```

## Task 4: Wire the canonical Stage7 CLI and atomic feedback barrier

**Owns:**

- Add `scripts/stage7_core_online_ablation_v2.py`
- Add `framework/tests/test_stage7_core_online_ablation_v2.py`
- Modify `scripts/stage7_core_round_controller_v2.sh`
- Modify `framework/tests/test_stage7_core_round_controller_v2.py`

**RED tests:**

- One CLI initializes all twelve canonical trajectories, freezes one real
  four-row request for each variant, reveals cache only afterward, emits a
  row-bound miss-only v3 plan and validates executor admission.
- The controller never calls scanner-gated Stage7 v1 initialization.
- Physical terminal rows and exact hits merge in logical order, invoke
  `stage5_promote_actual_feedback_v3.promote_feedback_batch`, and pass the
  existing `single_target_search_v2.finalize_atomic_batch` four-row barrier.
- Only after the barrier succeeds may the next variant-specific request be
  emitted.  Retries cannot change the logical request SHA.
- Every command validates the complete v2 contract and all frozen executor
  hashes before writing an immutable receipt.

**GREEN implementation:** Reuse existing Python functions from Stage5/Stage7.
The shell controller may orchestrate commands but must not embed build,
measurement, numerical or AP behavior.

**Verify:**

```bash
bash -n scripts/stage7_core_round_controller_v2.sh
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_core_online_ablation_v2.py \
  framework/tests/test_stage7_core_round_controller_v2.py
```

## Task 5: Add miss-only v3 execution admission and safe H800 scheduling

**Owns:**

- Add `scripts/stage7_execute_actual_v3_misses_v2.sh`
- Add `framework/tests/test_stage7_execute_actual_v3_misses_v2.py`
- Modify `scripts/stage7_core_ablation_scheduler_v2.py`
- Modify `framework/tests/test_stage7_core_ablation_scheduler_v2.py`

**RED tests:**

- Dry-run admission proves that the miss plan can feed the existing Stage5
  source materializer, quant contract, performance-plan builder, Stage3 v3
  performance/AP executors, Stage5 finalizer and v3 promoter without launching
  CUDA.
- Runtime admission requires host `zs-nj-tap-gpu18`, model `NVIDIA H800`,
  immutable UUID leases, one job per UUID and an indivisible four-candidate
  logical round.
- GPU7 is excluded whenever the F-Cooper validation/waiter rule is active.
  Any occupied/waiting/reserved/locked UUID is excluded even at 0% utilization.
- No cross-round prefetch, at most two four-UUID batches, same-width/source
  locks, recovery by request SHA and complete occupancy/process/lease audit.

**GREEN implementation:** The execution shell expands only orchestration around
the frozen Stage5/Stage3 commands.  Do not edit the five frozen actual-v3 core
files.

**Verify:**

```bash
bash -n scripts/stage7_execute_actual_v3_misses_v2.sh
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_execute_actual_v3_misses_v2.py \
  framework/tests/test_stage7_core_ablation_scheduler_v2.py
```

## Task 6: Close finalization, coverage, review and formal H800 no-GPU dry-run

**Owns:**

- Modify `scripts/stage7_finalize_core_ablation_v2.py`
- Modify `framework/tests/test_stage7_finalize_core_ablation_v2.py`
- Add `scripts/stage7_core_no_gpu_dry_run_v2.py`
- Add `framework/tests/test_stage7_core_no_gpu_dry_run_v2.py`

**RED tests:**

- Finalizer rejects fewer/more than 12 trajectories or 192 terminal selected
  events and rejects any missing actual-v3 evidence or silent fallback.
- Non-final/dry-run output always records `paper_ready=false`,
  `core_ablation_ready=false`, `formal_v2_gpu_jobs_launched=0` and the required
  integration blocking reason.
- Final output emits raw JSON/CSV, audit bundle, descriptive paired statistics,
  paper Markdown/CSV and root-cause summary; only the fully closed contract can
  set `core_ablation_ready=true`, while `full_gear_s7_ready=false` and
  `scanner_component_status=deferred_important_fix` remain fixed.

**Verification gates:**

1. Run complete Stage7 regression.
2. Measure every changed/new Stage7 Python production file separately and
   combined with `coverage`; each result must be at least 80%.
3. Obtain an independent code review and address every Critical/Important
   finding.
4. Deploy exact reviewed files and a SHA manifest to H800.
5. With `CUDA_VISIBLE_DEVICES=''`, run the formal dry-run using the real frozen
   H800 inputs.  It must create exactly twelve trajectories, four real round-0
   requests, four misses per request, admitted miss-only plans, synthetic
   protocol-valid promotion/barrier fixtures, next-round requests and finalizer
   schema evidence without any GPU process.

## Task 6A: Relocate and materialize real sources before the no-GPU gate

H800 preflight on 2026-07-26 proved that all sixteen frozen seed-20260718
round-zero selected rows have `checkpoint_sha256=null` and
`onnx_sha256=null`; three `without_surrogate` structures have no existing ONNX
artifact.  The formal exact key cannot be constructed at selection time, and a
plan/source SHA must not be substituted for an actual checkpoint/ONNX SHA.

The operator explicitly approved source-only H800 materialization before the
complete no-GPU gate, with every newly generated artifact confined to the v2
root.  This is not a candidate-pool, seed, budget or measurement-semantic
change:

```text
prepare/initialize with CUDA hidden
  -> raw selected IDs frozen
  -> deterministic source-contract relocation into v2_root/sources
  -> relocated logical request/source-resolution plan frozen
  -> source-only resolution
  -> four-row source-ready barrier
  -> complete no-GPU exact binding/cache/admission gate rerun
  -> miss-only build/measurement/AP execution
  -> actual-v3 feedback barrier
```

**Required invariants:**

- The original ordered selected IDs remain byte-identical.  The pre-relocation
  request SHA, relocation manifest SHA, relocated request SHA and source-plan
  SHA are all persisted and cross-bound before any GPU launch or cache access.
  After relocation, the formal logical request and source plan remain
  byte-identical across source resolution and retry.
- Only source output paths are relocated.  Base checkpoint identity, width,
  training epochs, source primitive, input shape and calibration contract stay
  frozen.
- All new checkpoint/ONNX/calibration/evidence/log artifacts live under
  `v2_root/sources/pyramid/<width>/`; no Stage5/Stage6 formal result is
  modified.
- The resolver accepts no cache, latency, energy, AP, failure labels or
  terminal history.
- It reuses the existing Stage5 source materializer and only emits authenticated
  checkpoint/ONNX/calibration evidence.  A later miss executor may verify that
  evidence idempotently but may not change source identity.
- Source interruption/evidence failure consumes zero selected-event budget and
  retries under the same logical request/source-plan SHA.  No candidate may be
  replaced.
- Exact cache reveal is forbidden until all four source rows are ready and
  cross-bound to the frozen logical request.  Null or placeholder source SHA is
  always rejected.
- The source-only phase runs before the complete no-GPU gate and may use H800
  UUIDs except GPU7.  It consumes zero selected-event budget and performs no
  TVM build/tuning, latency, energy, numerical, AP or feedback work.
- After real source resolution, the complete no-GPU gate is rerun from its
  first validation step.  It must observe the real source SHA identities,
  reveal four empty-cache misses per variant only after selection, admit four
  miss-only plans, and launch zero GPU jobs during the gate.
- Synthetic promotion/barrier fixtures remain quarantined and can never append
  cache truth or satisfy formal finalization.
- Production and finalizer artifacts must bind the source-resolution plan,
  result, per-row resolved-source SHA and materializer source SHA.
- Audit counters distinguish `pre_gate_source_gpu_jobs_launched`,
  `no_gpu_gate_gpu_jobs_launched` and
  `formal_v2_measurement_gpu_jobs_launched`; zero gate jobs must not hide
  already completed source-only GPU jobs.

**Implementation surface:**

- Add a pure v2-root source-contract relocator and its RED/GREEN tests.
- Split orchestrator admission into selection-only preparation, pre-gate
  source-only resolution, and complete no-GPU gate rerun.
- Extend the source lease controller with an explicitly authenticated
  `pre_gate_source_only` mode that never binds/reveals cache and never invokes
  measurement execution.
- Reuse `stage5_materialize_round_sources_v1.sh` against the relocated request;
  keep measurement execution behind both the source-ready and no-GPU gates.
- Extend Task6 dry-run/finalizer audits and rerun every coverage/review gate.
- Record the amendment and its evidence in Doc37 before any source or
  measurement GPU job is launched.

**RED tests:**

- Relocation changes only approved output fields, preserves ordered selected
  IDs, recomputes row/request/source-plan SHAs and rejects any path outside the
  v2 root.
- Two variants selecting the same width derive the same source key and output
  tree; different widths cannot share evidence.
- Pre-gate source mode rejects cache/label inputs, GPU7, non-H800 UUIDs,
  occupied/reserved/locked UUIDs, unpinned materializer SHA and a source output
  outside `v2_root/sources`.
- A source retry preserves relocated request/source-plan SHA and consumes zero
  selected events.
- The orchestrator cannot enter the complete no-GPU gate until all four
  round-zero source-ready barriers close, and cannot enter pilot until the
  rerun gate reports 16 exact misses with zero gate GPU jobs.
- Existing Stage5/Stage6 result trees remain byte-identical in an isolation
  test.

## Task 6B: Close the persistent orchestrator and real round-controller boundary

The first H800 deployment admission exposed two integration assumptions that
the local mocked dry-run did not exercise:

1. `prepare_state.json` binds a live process identity, but every existing
   Stage7 CLI is short lived.
2. The v2 scheduler consumes static `CACHE_REVEALED` batches and currently
   launches an argv vector that does not name a valid `core_online` subcommand;
   a zero controller return code is also accepted before authenticating the
   atomic feedback barrier.

No formal prepare or GPU job may start until this task is complete.

**Owns:**

- Add `scripts/stage7_core_ablation_orchestrator_v2.py` and focused tests.
- Add a thin round worker/wrapper that derives actual-v3 executor arguments
  only from authenticated canonical round artifacts and the scheduler-owned
  four-UUID environment.
- Modify the v2 scheduler-request builder/controller boundary and v2 scheduler
  completion admission; do not modify Stage5/Stage3 execution semantics.
- Extend the deployment bundle with the exact recursive/local entrypoint
  closure needed by the orchestrator, scheduler, resolver, round worker and
  finalizer.

**Required state machine:**

```text
BOOTSTRAP -> NO_GPU_GATE -> PILOT_20260718 -> PILOT_AUDIT
          -> SEEDS_20260719_20260720 -> FINALIZE -> COMPLETE

SOURCE_PLAN_FROZEN -> SOURCE_READY -> CACHE_REVEALED
                   -> PHYSICAL_TERMINAL_READY -> FEEDBACK_COMPLETE
```

The orchestrator remains the same live PID from prepare through finalization,
revalidates its PID/start-time/cmdline plus deployment/executor pins on every
tick, treats canonical round artifacts as truth, and writes only derived
status/events after prepare.  Resource shortage waits; source and
infrastructure failures retry the same immutable request SHA without consuming
budget.  A pre-existing terminal with no barrier must never relaunch GPU work.

The round worker may call only the existing source resolver,
`stage7_execute_actual_v3_misses_v2.sh`, `core_online.finalize_round()` and
barrier validators.  It returns zero only after authenticating
`atomic_feedback_barrier.json`.  The v2 scheduler independently revalidates
that barrier before writing `feedback_complete` or adding four selected
events.

**RED tests:**

- A transient prepare/no-GPU PID is rejected; one persistent orchestrator PID
  remains valid across prepare, source, measurement and recovery ticks.
- The production scheduler request launches the real round worker with an
  argv-compatible interface, not the selection CLI.
- Source-ready is required before exact cache reveal and measurement lease.
- Controller return zero without a valid atomic barrier does not consume
  budget or become `feedback_complete`.
- Existing terminal plus missing barrier finalizes without launching a second
  GPU attempt.
- Barrier completion generates exactly one next round and the pilot gate
  blocks later seeds until all four pilot trajectories reach 16 events.
- Fake end-to-end closes four variants/four rounds with immutable request,
  source, cache, terminal and barrier lineage; all hardware calls are injected
  fakes.

**Verification:**

- Every changed/new Stage7 production Python file and their combined coverage
  are at least 80%, every production file is at most 800 lines.
- Full Stage7 regression, shell syntax, `py_compile`, isolated deployment
  import/CLI test and independent reviewer all pass.
- H800 remains at `formal_v2_gpu_jobs_launched=0` throughout Task 6B.

## Task 7: Execute the frozen GPU pilot, remaining seeds and final audit

**Owns:** Only the v2 result root and Doc37 status append.  No production-code
changes are allowed after pilot selection begins.

1. Recheck processes, locks, leases, UUIDs, GPU models, F-Cooper reservation
   state and deployed SHAs.
2. Wait until at least four continuously idle, unreserved H800 UUIDs are
   available; never stop or preempt another task.
3. Resolve and audit each frozen four-row source batch first.  Only a
   source-ready batch may perform exact cache reveal and launch build/
   measurement/AP work.  Then close all four seed-20260718 trajectories
   (64 selected events) and audit isolation/cache/request/source/terminal/
   promotion/GPU evidence.
4. Only after pilot passes, close seeds 20260719 and 20260720 without changing
   budget, pool, features, scanner state or acquisition.
5. Finalize the paper tables and append Doc37 with authoritative status and
   artifact paths.

**Completion:** Exactly 12 trajectories and 192 selected events have credible
terminal evidence; every miss has actual-v3 hardware evidence; all audits and
paper outputs close; `core_ablation_ready=true`.  Until then the goal remains
active and interim output cannot be described as paper-ready.
