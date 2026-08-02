# Stage7 Scanner-Deferred Core Ablation v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the four-row Pyramid–TVM/H800 core online-search ablation with capability scanning explicitly deferred, producing 12 trustworthy trajectories, 192 selected events, audit-complete paper tables, and no claim that the full scanner-enabled GEAR-S7 experiment is complete.

**Architecture:** Preserve the tested Stage7 v1 selection, exact-cache, feedback, and GPU lease primitives, but add a versioned v2 contract layer that binds all four variants to the same frozen 686-candidate pre-scan pool and to the immutable v1 blocker evidence. Prepare, trajectory, scheduler, controller, and finalizer paths consume that one v2 contract. The v1 blocked result root remains read-only; the v2 root starts with an empty cache and only appends exact terminal evidence after independent selection.

**Tech Stack:** Python 3.9, pytest, stdlib JSON/CSV/SHA256, existing Stage4/Stage5 search APIs, shell controller, H800 CUDA/TVM runtime.

## Global Constraints

- Formal execution host is the configured H800 worker (`$H800_HOST:$H800_PORT`); never launch Stage7 GPU work locally, on 4090, or on another host.
- The only task is `S7-PYR-TVM=Pyramid+H800 GPU+tvm_auto`.
- Active variants are exactly `full`, `without_surrogate`, `without_measured_feedback`, and `backend_blind`.
- Never execute `without_capability_scan`, INT8-only, direct/residual AP, value/rank, scalarization, NSGA-II, mixed_policy, automatic-region, CPU Lane B, ORT CPU EP, TRT, or CoDriving.
- V1 root `$V2X_RESULTS_DIR/stage7_pyramid_tvm_online_ablation_quick_v1_20260725` is immutable and read-only.
- V2 root is `$V2X_RESULTS_DIR/stage7_pyramid_tvm_online_core_ablation_quick_v2_20260725`.
- Source registry file SHA256 is `033cb9e38af39009f9919233f03c5ff77894651f6192d292dee501d974493b51`.
- Pre-scan registry artifact SHA256 is `3e584a4d06afe3ca0ba6cedce9e0a99a20aafae8b1ce7ab62b4fd00da07ac4b8`.
- Ordered 686-candidate content SHA256 is `3294a3e66671af2778bffd4d719ab9f45963d948cabdfb21315012274daf96f4`.
- V1 blocker file SHA256 is `20cadfee33e2cbbffd3e2c58ececd89282c3e3733ad5924dadc885b0f9efc5da`.
- V1 expanded-admission file SHA256 is `06761a3fdba4a945b9323941924d332e05b9f7a20321009a193696cce81ad170`.
- Every variant has `candidate_pool=pre_scan`, `scanner=deferred`, and `scanner_claim_allowed=false`.
- Seeds are exactly `{20260718, 20260719, 20260720}` with `B=4`, four rounds, and `T=16`.
- Initial cache is exactly `{"schema_version":"stage7_measurement_cache_v1","entries":{},"historical_exact_hit_count":0}`.
- Cache membership and labels remain hidden until the ordered selected IDs for that event are frozen.
- A selected exact-key cache miss requires real materialization, build, latency, energy, and full AP measurement.
- Candidate capability failure consumes one selected-event slot and has null objectives; infrastructure/evidence failure consumes no slot and retries the unchanged request SHA.
- One round is an indivisible four-candidate GPU batch. Launch only after four UUIDs are continuously idle; at eight idle UUIDs at most two independent batches may run.
- No cross-round prefetch and no crossing a measured-feedback barrier.
- Existing user changes are authoritative; do not revert or overwrite unrelated work.
- New or changed Stage7 production behavior must follow RED → GREEN → REFACTOR and new-code line coverage must be at least 80%.

---

### Task 1: Freeze the v2 Four-Variant Contract and Blocker Provenance

**Files:**
- Create: `framework/stage7/core_ablation_v2.py`
- Create: `framework/tests/test_stage7_core_ablation_v2.py`
- Modify: `framework/stage7/online_component_ablation_v1.py`
- Modify: `framework/stage7/search_policy_v1.py`

**Interfaces:**
- Produces: `CORE_VARIANTS`, `V2_ROOT`, `V1_ROOT`, `V2Contract`, `load_and_validate_v1_blocker(v1_root)`, `build_v2_contract(pre_scan_registry, v1_root)`, and `validate_v2_contract(payload)`.
- `build_v2_contract` returns an immutable mapping containing the exact four variants, formal SHAs, `scanner_deferred=True`, `scanner_claim_allowed=False`, and one shared ordered pre-scan SHA.
- Existing v1 scanner functions remain available for the read-only v1 audit and are not treated as v2 selectors.

- [ ] **Step 1: Write failing contract tests**

```python
def test_v2_contract_has_exactly_four_pre_scan_variants(v1_root, pre_scan_rows):
    contract = build_v2_contract(pre_scan_rows, v1_root)
    assert tuple(contract["variants"]) == CORE_VARIANTS
    assert "without_capability_scan" not in contract["variants"]
    assert {row["candidate_pool"] for row in contract["variant_contracts"]} == {"pre_scan"}
    assert {row["scanner"] for row in contract["variant_contracts"]} == {"deferred"}
    assert {row["scanner_claim_allowed"] for row in contract["variant_contracts"]} == {False}
    assert contract["ordered_pre_scan_sha256"] == EXPECTED_ORDERED_PRE_SCAN_SHA256


def test_v2_contract_binds_exact_v1_failure_evidence(v1_root, pre_scan_rows):
    contract = build_v2_contract(pre_scan_rows, v1_root)
    assert contract["v1_blocker"]["sha256"] == EXPECTED_BLOCKER_SHA256
    assert contract["v1_expanded_admission"]["sha256"] == EXPECTED_ADMISSION_SHA256
    assert contract["v1_blocker"]["status"] == "blocked_missing_candidate_level_scanner"
    assert contract["v1_expanded_admission"]["admission_passed"] is False
    assert contract["v1_expanded_admission"]["false_positive_unique_count"] == 3


def test_v2_contract_fails_closed_on_any_sha_or_status_drift(v1_root, pre_scan_rows):
    mutate_blocker_without_updating_expected_sha(v1_root)
    with pytest.raises(ValueError, match="v1 blocker SHA drift"):
        build_v2_contract(pre_scan_rows, v1_root)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
"$V2X_PYTHON" -m pytest -q \
  framework/tests/test_stage7_core_ablation_v2.py
```

Expected: collection/import failure because `framework.stage7.core_ablation_v2` does not exist.

- [ ] **Step 3: Implement the minimal contract module**

```python
CORE_VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
)

def load_and_validate_v1_blocker(v1_root: Path) -> dict[str, object]:
    blocker_path = v1_root / "status/without_capability_scan_blocked.json"
    admission_path = v1_root / "audits/scanner_admission_expanded_v1.json"
    # Rehash both files, require the frozen digests and required failure fields,
    # and return new copied records without mutating either source payload.

def build_v2_contract(
    pre_scan_registry: Sequence[Mapping[str, object]], v1_root: Path
) -> dict[str, object]:
    # Require 686 unique ordered IDs and the frozen candidate-content SHA.
    # Emit four copied variant records with the shared pre-scan SHA and deferred scanner fields.
```

- [ ] **Step 4: Run focused and adjacent tests**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_core_ablation_v2.py \
  framework/tests/test_stage7_online_component_ablation_v1.py \
  framework/tests/test_stage7_search_policy_v1.py
```

Expected: PASS.

- [ ] **Step 5: Record task evidence**

Write the RED command/output, GREEN command/output, changed files, and immutable SHA assertions to the SDD task report. Do not commit because Stage7 files are currently authoritative untracked shared-workspace files.

---

### Task 2: Prepare v2 Root, Frozen Empty Cache, and Four Trajectories per Seed

**Files:**
- Create: `scripts/stage7_prepare_core_ablation_v2.py`
- Create: `framework/tests/test_stage7_prepare_core_ablation_v2.py`
- Modify: `scripts/stage7_online_ablation_v1.py`
- Modify: `scripts/stage7_prepare_online_ablation_v1.py`
- Modify: `framework/stage7/search_policy_v1.py`

**Interfaces:**
- Consumes: Task 1 `build_v2_contract` and `validate_v2_contract`.
- Produces: `prepare_v2_root(v1_root, v2_root, inputs)`, `initialize_v2_trajectories(v2_root)`, `contracts/core_ablation_v2.json`, `contracts/measurement_cache_initial.json`, and 12 round-0 trajectory contracts.
- V2 preparation copies only immutable inputs/provenance into the v2 root; it never imports v1 trajectory state.

- [ ] **Step 1: Write failing prepare/isolation/cache tests**

```python
def test_prepare_v2_writes_empty_cache_and_never_imports_v1_trajectory(tmp_path, v1_root):
    v2 = tmp_path / "v2"
    result = prepare_v2_root(v1_root=v1_root, v2_root=v2, inputs=frozen_inputs())
    assert result["v2_root"] == str(v2.resolve())
    assert read_json(v2 / "contracts/measurement_cache_initial.json") == {
        "schema_version": "stage7_measurement_cache_v1",
        "entries": {},
        "historical_exact_hit_count": 0,
    }
    assert not list(v2.glob("trajectories/**/terminal_event.json"))
    assert hash_tree(v1_root) == original_v1_hash_tree


def test_initialize_v2_creates_12_round0_contracts_without_scanner_ready(v2_root):
    result = initialize_v2_trajectories(v2_root)
    assert result["trajectory_count"] == 12
    assert {(row["variant"], row["seed"]) for row in result["trajectories"]} == {
        (variant, seed) for variant in CORE_VARIANTS for seed in SEEDS
    }
    assert all(row["candidate_pool"] == "pre_scan" for row in result["trajectories"])
    assert all(row["scanner_deferred"] is True for row in result["trajectories"])
```

- [ ] **Step 2: Verify RED**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_prepare_core_ablation_v2.py
```

Expected: import failure for the new v2 prepare module.

- [ ] **Step 3: Implement minimal prepare and initialization paths**

```python
def initial_measurement_cache() -> dict[str, object]:
    return {
        "schema_version": "stage7_measurement_cache_v1",
        "entries": {},
        "historical_exact_hit_count": 0,
    }

def initialize_v2_trajectories(output_root: Path) -> dict[str, object]:
    contract = validate_v2_contract(read_json(output_root / "contracts/core_ablation_v2.json"))
    # Build only CORE_VARIANTS × SEEDS and bind every trajectory to the same pre-scan SHA.
```

Add explicit CLI subcommands `prepare-core-v2` and `init-core-v2`; do not change the behavior of archived v1 subcommands.

- [ ] **Step 4: Prove selector single-variable isolation**

Add tests that run the same ordered pre-scan pool through all four variants and assert:

```python
assert full.audit["surrogate_calls"] > 0
assert without_surrogate.audit["surrogate_calls"] == 0
assert without_surrogate.audit["uncertainty_calls"] == 0
assert without_surrogate.audit["predicted_frontier_calls"] == 0
assert without_feedback.audit["bundle_refit_calls_after_initial"] == 0
assert without_feedback.audit["actual_graph_feedback_rows"] == 0
assert backend_blind.audit["fixed_dispatch_key"] == "tvm_auto"
assert backend_blind.audit["removed_feature_names"]
assert not set(backend_blind.audit["model_feature_names"]) & set(
    backend_blind.audit["forbidden_backend_feature_names"]
)
```

- [ ] **Step 5: Run focused and full Stage7 regression**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_prepare_core_ablation_v2.py \
  framework/tests/test_stage7_search_policy_v1.py \
  framework/tests/test_stage7_cache_feedback_v1.py
```

Expected: PASS.

---

### Task 3: Selected-Only Append-Only Exact Cache and Failure Budget

**Files:**
- Create: `framework/stage7/core_cache_v2.py`
- Create: `framework/tests/test_stage7_core_cache_v2.py`
- Modify: `framework/stage7/cache_feedback_v1.py`
- Modify: `scripts/stage7_online_ablation_v1.py`

**Interfaces:**
- Consumes: frozen initial cache and existing exact cache-key builder.
- Produces: `reveal_v2_cache_after_selection(binding, cache)`, `append_v2_terminal_evidence(cache, evidence)`, and immutable cache lineage records.
- Append is allowed only for a complete, exact-contract, terminal evidence wrapper created by this v2 run.

- [ ] **Step 1: Write failing cache-lineage and budget tests**

```python
def test_cache_membership_cannot_be_revealed_before_ordered_selection_is_frozen():
    with pytest.raises(ValueError, match="selection binding must be frozen"):
        reveal_v2_cache_after_selection(unfrozen_binding(), populated_cache())


def test_selected_exact_hit_reuses_truth_without_changing_event_budget():
    selected = frozen_binding(["candidate-a"])
    reveal = reveal_v2_cache_after_selection(selected, cache_with_exact("candidate-a"))
    assert reveal["entries"][0]["disposition"] == "hit"
    assert reveal["entries"][0]["selected_event_budget_delta"] == 1
    assert reveal["entries"][0]["hardware_measurement_required"] is False


def test_candidate_failure_consumes_slot_with_null_objectives():
    terminal = finalize_candidate_failure(request(), reason="backend_capability_failure")
    assert terminal["consumes_selected_event_budget"] is True
    assert terminal["latency_ms"] is None
    assert terminal["energy_j"] is None
    assert terminal["ap70"] is None


def test_infrastructure_failure_retries_same_request_without_consuming_slot():
    retry = finalize_infrastructure_failure(request(), reason="gpu_occupancy_drift")
    assert retry["consumes_selected_event_budget"] is False
    assert retry["retry_request_sha256"] == retry["request_sha256"]
```

- [ ] **Step 2: Verify RED**

Run the new focused test and confirm the import/API failure.

- [ ] **Step 3: Implement immutable cache append and selected-only reveal**

```python
def append_v2_terminal_evidence(cache, evidence):
    validated = validate_terminal_evidence(evidence)
    key = validated["exact_cache_key_sha256"]
    if key in cache["entries"] and cache["entries"][key] != validated:
        raise ValueError("conflicting exact-cache evidence")
    return {**cache, "entries": {**cache["entries"], key: validated}}
```

Do not mutate a cache object in place and do not accept any historical wrapper lacking every protocol SHA.

- [ ] **Step 4: Run cache and controller regressions**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_core_cache_v2.py \
  framework/tests/test_stage7_cache_feedback_v1.py \
  framework/tests/test_stage7_task_round_controller_v1.py
```

Expected: PASS.

---

### Task 4: Build the 12-Trajectory Scheduler and Recoverable Round Controller

**Files:**
- Create: `scripts/stage7_core_ablation_scheduler_v2.py`
- Create: `scripts/stage7_core_round_controller_v2.sh`
- Create: `framework/tests/test_stage7_core_ablation_scheduler_v2.py`
- Create: `framework/tests/test_stage7_core_round_controller_v2.py`
- Reuse without broad mutation: `scripts/stage7_ablation_scheduler_v1.py`
- Reuse without broad mutation: `scripts/stage7_task_round_controller_v1.sh`

**Interfaces:**
- Consumes: v2 trajectory contracts and miss-only plans.
- Produces: exactly 12 trajectory queue entries, four-candidate indivisible batch requests, UUID lease audit, occupancy snapshots, lock audit, and resumable stage markers.
- The v2 wrapper may delegate to tested v1 primitives but must reject v1 roots and the archived fifth variant.

- [ ] **Step 1: Write failing queue/root/GPU barrier tests**

```python
def test_v2_queue_has_12_trajectories_and_no_a4():
    queue = build_v2_trajectory_queue(v2_root())
    assert len(queue) == 12
    assert {item.variant for item in queue} == set(CORE_VARIANTS)
    assert all(item.batch_size == 4 for item in queue)


def test_v2_scheduler_refuses_v1_root_and_less_than_four_idle_uuids():
    with pytest.raises(ValueError, match="v2 result root"):
        build_v2_trajectory_queue(v1_root())
    result = scheduler_with_snapshots(three_idle_snapshots()).schedule(batch_request())
    assert result["status"] == "waiting_for_four_idle_gpus"
    assert result["launched"] == []


def test_v2_scheduler_never_prefetches_across_feedback_barrier():
    scheduler = scheduler_with_snapshots(eight_idle_snapshots())
    result = scheduler.schedule(two_rounds_same_trajectory())
    assert len(result["launched"]) == 1
    assert result["blocked"][0]["reason"] == "feedback_barrier"
```

- [ ] **Step 2: Verify RED**

Run both new scheduler/controller test files; expected failure is missing v2 modules/scripts.

- [ ] **Step 3: Implement minimal v2 wrappers**

The Python wrapper defines only `CORE_VARIANTS`, validates `V2_ROOT`, constructs 12 queue items, and delegates UUID lease behavior to `Stage7Scheduler`. The shell wrapper validates the v2 contract SHA before every stage and uses existing atomic stage recovery.

- [ ] **Step 4: Verify shell safety and full scheduler behavior**

```bash
bash -n scripts/stage7_core_round_controller_v2.sh
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_core_ablation_scheduler_v2.py \
  framework/tests/test_stage7_core_round_controller_v2.py \
  framework/tests/test_stage7_ablation_scheduler_v1.py \
  framework/tests/test_stage7_task_round_controller_v1.py
```

Expected: PASS.

---

### Task 5: Finalize v2 Metrics, Identical-Trajectory Audit, and Paper Artifacts

**Files:**
- Create: `scripts/stage7_finalize_core_ablation_v2.py`
- Create: `framework/tests/test_stage7_finalize_core_ablation_v2.py`
- Reuse metric helpers from: `scripts/stage7_finalize_ablation_v1.py`

**Interfaces:**
- Consumes: 12 completed trajectory trees, shared exact cache, scheduler audits, and v2 frozen contracts.
- Produces: raw CSV/JSON, descriptive paired statistics, paper Markdown/CSV table, execution-cost table, completeness audit, isolation audit, and root-cause summary.
- Completion fields are `core_ablation_ready`, `full_gear_s7_ready`, and `scanner_component_status`.

- [ ] **Step 1: Write failing completion and identical-trajectory tests**

```python
def test_v2_finalizer_requires_12_trajectories_and_192_events(complete_v2_tree):
    result = finalize_v2(complete_v2_tree, output_dir())
    assert result["core_ablation_ready"] is True
    assert result["complete_trajectory_count"] == 12
    assert result["selected_event_count"] == 192
    assert result["full_gear_s7_ready"] is False
    assert result["scanner_component_status"] == "deferred_important_fix"


def test_identical_marker_requires_independent_ordered_ids_and_protocol_sha(tree):
    make_full_and_blind_ordered_ids_equal(tree)
    result = finalize_v2(tree, output_dir())
    assert result["trajectory_pairs"][0]["completed_trajectory_identical"] is True
    drift_one_selection_binding_sha(tree)
    with pytest.raises(FinalizationError, match="identical trajectory provenance"):
        finalize_v2(tree, output_dir())


def test_finalizer_rejects_surrogate_filled_failure_metrics(tree):
    inject_candidate_failure_with_latency(tree)
    with pytest.raises(FinalizationError, match="failure objectives must be null"):
        finalize_v2(tree, output_dir())
```

- [ ] **Step 2: Verify RED**

Run the new finalizer test and confirm missing v2 finalizer failure.

- [ ] **Step 3: Implement v2 finalizer with immutable outputs**

Reuse mathematical helpers for `DeltaHV-AUC`, `DeltaHV@16`, frontier recall, invalid/valid yield, iso-AP ratios, and descriptive pairing. Replace the 15/240 v1 gate with 12/192 and never emit `paper_ready=true` or `full_gear_s7_ready=true`.

- [ ] **Step 4: Verify all output schemas**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_finalize_core_ablation_v2.py \
  framework/tests/test_stage7_finalize_ablation_v1.py
```

Expected: PASS and v1 finalizer behavior remains unchanged.

---

### Task 6: Coverage, H800 Deployment, Dry-Run, Pilot, and Formal Completion

**Files:**
- Create remotely under v2 root: `deployment/stage7_v2_deploy_manifest.json`
- Create remotely under v2 root: `status/`, `contracts/`, `trajectories/`, `cache/`, `audits/`, `raw/`, `paper/`, and `summary/`
- Append final evidence to: `multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/7_14/37_7_14_交接文档_阶段7四条在线搜索组件消融实验计划_v1.md`

**Interfaces:**
- Consumes: Tasks 1–5 verified local implementation.
- Produces: deployed SHA manifest, no-GPU dry-run evidence, seed `20260718` pilot gate, remaining two seeds, 192-event closure, and the final paper artifacts.

- [ ] **Step 1: Run local full Stage7 tests and coverage**

```bash
/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m pytest -q \
  framework/tests/test_stage7_*.py

/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python -m trace \
  --count --missing --coverdir /tmp/stage7-v2-trace \
  --module pytest \
  -q \
  framework/tests/test_stage7_core_ablation_v2.py \
  framework/tests/test_stage7_prepare_core_ablation_v2.py \
  framework/tests/test_stage7_core_cache_v2.py \
  framework/tests/test_stage7_core_ablation_scheduler_v2.py \
  framework/tests/test_stage7_core_round_controller_v2.py \
  framework/tests/test_stage7_finalize_core_ablation_v2.py
```

Compute executable-line coverage separately for every new v2 Python production file from the generated `.cover` files, excluding blank, comment-only, type-only, and test lines. Include any existing v1 production file changed by Tasks 1–5 in the same calculation. Every changed production module and the combined changed-code total must be at least 80%; the shell controller is verified by its pytest behavior suite plus `bash -n`.

- [ ] **Step 2: Deploy exact tested files to H800**

Before writing the v2 root, verify again that no Stage7 process owns it and that the path is absent or contains a contract-identical resumable v2 state. Upload only the tested Stage7 files, write a manifest containing local and remote SHA256 for every file, and rehash remotely.

- [ ] **Step 3: Run no-GPU dry-run**

Set `CUDA_VISIBLE_DEVICES=""` for contract preparation, trajectory initialization, cache audit, selector audit, and scheduler simulation. Require 12 trajectories, identical pre-scan SHA, zero imported v1 trajectory, empty initial cache, and zero GPU launch.

- [ ] **Step 4: Wait for and lease four continuously idle H800 UUIDs**

Never stop or modify F-Cooper or another user’s process. The scheduler remains in a waiting state until four UUIDs pass all idle samples and locks. Persist every sample.

- [ ] **Step 5: Execute seed 20260718 four-trajectory pilot**

Run four rounds for each active variant. Every round freezes four ordered IDs before cache reveal; cache misses use real hardware and full AP. Audit the 64 selected events, single-variable isolation, feedback barriers, cache lineage, failure budget, leases, locks, and measurement windows.

- [ ] **Step 6: Execute seeds 20260719 and 20260720**

Only after the pilot gate passes, run the remaining eight trajectories under the unchanged contract. Do not change seeds, pool, budget, or scanner status.

- [ ] **Step 7: Finalize and perform requirement-by-requirement completion audit**

Require all 12 trajectories and all 192 events, then emit tables and summaries. Verify:

```text
core_ablation_ready=true
full_gear_s7_ready=false
scanner_component_status=deferred_important_fix
```

Append final status, metrics, ETA history, artifact paths, SHA manifest, and scoped scientific conclusion to Section 15. Never state that capability scanning or full GEAR-S7 has been validated.
