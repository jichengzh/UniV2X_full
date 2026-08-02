from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from framework.tests.test_stage7_source_execution_v2 import _inputs
from framework.stage7 import source_execution_v2 as source_execution
from scripts import stage7_source_lease_controller_v2 as controller


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _write_terminal_retry(
    output: Path, reason: str = "evidence_unavailable"
) -> dict[str, Any]:
    directory = output.parents[1]
    execution = json.loads(
        (directory / "source_execution_plan.json").read_text(encoding="utf-8")
    )
    plan = json.loads(
        (directory / "source_resolution_plan.json").read_text(encoding="utf-8")
    )
    lease = json.loads((output / "source_gpu_lease.json").read_text(encoding="utf-8"))
    retry = source_execution.build_zero_budget_source_retry(
        plan,
        execution,
        failed_group_keys=[
            group["source_group_execution_key"] for group in execution["group_jobs"]
        ],
        reason_code=reason,
    )
    _write_json(output / "source_resolution_retry.json", retry)
    unsigned = {
        "schema_version": "stage7_source_execution_receipt_v2",
        "status": "retry_required",
        "logical_request_sha256": execution["logical_request_sha256"],
        "source_resolution_plan_sha256": execution["source_resolution_plan_sha256"],
        "source_execution_plan_sha256": execution["source_execution_plan_sha256"],
        "source_gpu_lease_sha256": lease["source_gpu_lease_sha256"],
        "source_attempt_sha256": lease["source_attempt_sha256"],
        "materializer_sha256": "m" * 64,
        "materializer_invocation_count": 0,
        "formal_result_written": False,
        "source_result_path": None,
        "retry_path": str(output / "source_resolution_retry.json"),
        "retry_reason_code": reason,
        "source_resolution_result_sha256": None,
        "source_resolution_retry_sha256": retry["source_resolution_retry_sha256"],
        "selected_event_budget_delta": 0,
        "partial_reveal_allowed": False,
        "eligible_for_exact_cache_reveal": False,
        "eligible_for_cache_append": False,
        "eligible_for_finalization": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha(unsigned)}
    _write_json(output / "source_execution_receipt.json", receipt)
    return {
        **receipt,
        "source_execution_receipt_path": str(output / "source_execution_receipt.json"),
    }


class Lock:
    def __init__(self, label: str, events: list[str]) -> None:
        self.label = label
        self.events = events
        self.released = False
        events.append(f"lock:{label}")

    def release(self) -> None:
        assert self.released is False
        self.released = True
        self.events.append(f"release:{self.label}")


class FakeScheduler:
    def __init__(
        self,
        events: list[str],
        *,
        ready: tuple[str, ...] = ("GPU-a", "GPU-b"),
        post_lock_admitted: bool = True,
        owner_pid: int = 800,
        unavailable_lock: str | None = None,
    ) -> None:
        self.events = events
        self.ready = ready
        self.post_lock_admitted = post_lock_admitted
        self.current_owner = "runner"
        self.hostname_probe = lambda: "zs-nj-tap-gpu18"
        self.process_inspector = lambda pid: SimpleNamespace(
            pid=pid,
            owner="runner",
            start_time="123",
            alive=True,
            command=("python3", "orchestrator"),
        )
        self._last_snapshot = {
            uuid: SimpleNamespace(
                uuid=uuid,
                index=index,
                memory_used_mib=0,
                utilization_percent=0,
                compute_pids=(),
            )
            for index, uuid in enumerate(("GPU-a", "GPU-b", "GPU-c"))
        }
        self._gpu_models = {uuid: "NVIDIA H800" for uuid in self._last_snapshot}
        self.owner_pid = owner_pid
        self.unavailable_lock = unavailable_lock

    def observe_gpus(self) -> dict[str, Any]:
        self.events.append("observe")
        return copy.deepcopy(self._last_snapshot)

    def _ready_uuids(self) -> list[str]:
        self.events.append("ready")
        return list(self.ready)

    def resource_lock_factory(self, key: str) -> Lock | None:
        if self.unavailable_lock == "source":
            self.events.append("source-lock-unavailable")
            return None
        return Lock(key, self.events)

    def lock_factory(self, uuid: str) -> Lock | None:
        if self.unavailable_lock == "uuid":
            self.events.append("uuid-lock-unavailable")
            return None
        return Lock(f"uuid:{uuid}", self.events)

    def _post_lock_runtime_admission(self, uuids: tuple[str, ...]) -> bool:
        self.events.append("post-lock:" + ",".join(uuids))
        return self.post_lock_admitted


def _case(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "v2"
    round_dir = root / "variants/full/seed_20260718/round_00"
    request_bytes, binding, source_plan = _inputs(root / "sources")
    round_dir.mkdir(parents=True)
    (round_dir / "logical_request.json").write_bytes(request_bytes)
    _write_json(round_dir / "selection_binding.json", binding)
    _write_json(round_dir / "source_resolution_plan.json", source_plan)
    receipt = {
        "formal_v2_gpu_jobs_launched": 0,
        "canonical_round0_requests_written": 4,
    }
    _write_json(root / "audits/no_gpu_dry_run/dry_run_receipt.json", receipt)
    authorization = {
        "schema_version": "stage7_pre_gate_source_authorization_v2",
        "authorized_action": "source_only_materialization_before_no_gpu_gate",
        "v2_root": str(root.resolve()),
        "gpu7_excluded": True,
        "performance_measurement_allowed": False,
        "ap_allowed": False,
        "cache_reveal_allowed": False,
        "feedback_allowed": False,
        "selected_event_budget_delta": 0,
    }
    _write_json(
        root / "contracts/pre_gate_source_authorization.json",
        {**authorization, "authorization_sha256": _sha(authorization)},
    )
    return {
        "v2_root": root,
        "repo_root": tmp_path / "repo",
        "round_dir": round_dir,
        "request": json.loads(request_bytes),
        "binding": binding,
        "source_plan": source_plan,
    }


def _dependencies(
    scheduler: FakeScheduler,
    events: list[str],
    *,
    resolver_result: dict[str, Any] | None = None,
    prelease_result: dict[str, Any] | None = None,
) -> controller.SourceLeaseDependencies:
    def resolve(**kwargs: Any) -> dict[str, Any]:
        events.append("resolve:" + kwargs["environment"]["CUDA_VISIBLE_DEVICES"])
        last_lock = max(
            index for index, item in enumerate(events) if item.startswith("lock:")
        )
        prior_releases = [
            index for index, item in enumerate(events) if item.startswith("release:")
        ]
        assert not prior_releases or last_lock > max(prior_releases)
        output = Path(kwargs["output_dir"])
        if resolver_result and resolver_result.get("status") == "retry_required":
            terminal = _write_terminal_retry(
                output,
                str(resolver_result.get("retry_reason_code") or "evidence_unavailable"),
            )
            return {**terminal, **copy.deepcopy(resolver_result)}
        staged = output / "source_resolution_result.staged.json"
        _write_json(
            staged, {"formal": True, "source_resolution_result_sha256": "s" * 64}
        )
        return copy.deepcopy(
            resolver_result
            or {
                "status": "source_ready",
                "source_result_path": str(staged),
                "source_resolution_result_sha256": "s" * 64,
                "selected_event_budget_delta": 0,
            }
        )

    def bind(v2_root: Path, **kwargs: Any) -> dict[str, Any]:
        events.append("bind")
        last_lock = max(
            (index for index, item in enumerate(events) if item.startswith("lock:")),
            default=len(events),
        )
        prior_releases = [
            index for index, item in enumerate(events) if item.startswith("release:")
        ]
        assert not prior_releases or last_lock > max(prior_releases)
        source = Path(kwargs["source_result_path"])
        destination = (
            Path(v2_root)
            / "variants/full/seed_20260718/round_00/source_resolution_result.json"
        )
        destination.write_bytes(source.read_bytes())
        for filename in (
            "exact_selection_binding.json",
            "cache_snapshot_before_reveal.json",
            "cache_reveal.json",
            "miss_only_physical_request.json",
            "executor_admission.json",
        ):
            _write_json(destination.parent / filename, {})
        return {
            "controller_state": "CACHE_REVEALED",
            "source_resolution_result_sha256": "s" * 64,
        }

    return controller.SourceLeaseDependencies(
        scheduler_factory=lambda _root: scheduler,
        resolver=resolve,
        binder=bind,
        formal_result_probe=lambda *_args, **_kwargs: copy.deepcopy(prelease_result),
        formal_result_validator=lambda payload, _plan: copy.deepcopy(dict(payload)),
    )


def _run(
    case: dict[str, Any],
    dependencies: controller.SourceLeaseDependencies,
    *,
    pid: int = 800,
    pre_gate_source_only: bool = False,
) -> dict[str, Any]:
    return controller.run_source_lease_controller(
        v2_root=case["v2_root"],
        repo_root=case["repo_root"],
        variant="full",
        seed=20260718,
        round_index=0,
        orchestrator_pid=pid,
        dependencies=dependencies,
        pre_gate_source_only=pre_gate_source_only,
    )


def test_no_source_or_gpu_work_before_no_gpu_gate(tmp_path: Path) -> None:
    case = _case(tmp_path)
    (case["v2_root"] / "audits/no_gpu_dry_run/dry_run_receipt.json").unlink()
    events: list[str] = []
    scheduler = FakeScheduler(events)

    with pytest.raises(controller.SourceLeaseContractError, match="no-GPU gate"):
        _run(case, _dependencies(scheduler, events))

    assert events == []


def test_authorized_pre_gate_source_only_commits_source_without_cache_reveal(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    (case["v2_root"] / "audits/no_gpu_dry_run/dry_run_receipt.json").unlink()
    events: list[str] = []

    result = _run(
        case,
        _dependencies(FakeScheduler(events), events),
        pre_gate_source_only=True,
    )

    assert result["status"] == "source_ready"
    assert result["selected_event_budget_delta"] == 0
    assert result["pre_gate_source_gpu_jobs_launched"] == 2
    assert result["performance_measurement_jobs_launched"] == 0
    assert result["cache_membership_observed"] is False
    assert result["feedback_released"] is False
    assert "bind" not in events
    assert (case["round_dir"] / "source_resolution_result.json").is_file()
    assert (case["round_dir"] / "pre_gate_source_only_receipt.json").is_file()
    assert not any(
        (case["round_dir"] / name).exists() for name in controller.EXACT_ARTIFACTS
    )


def test_pre_gate_source_only_requires_frozen_signed_authorization(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    (case["v2_root"] / "audits/no_gpu_dry_run/dry_run_receipt.json").unlink()
    (
        case["v2_root"] / "contracts/pre_gate_source_authorization.json"
    ).unlink()
    events: list[str] = []

    with pytest.raises(
        controller.SourceLeaseContractError,
        match="pre-gate source authorization",
    ):
        _run(
            case,
            _dependencies(FakeScheduler(events), events),
            pre_gate_source_only=True,
        )

    assert events == []


@pytest.mark.parametrize("broken", (False, True))
def test_source_attempt_parent_symlink_is_rejected_before_any_gpu_or_write(
    broken: bool, tmp_path: Path
) -> None:
    case = _case(tmp_path)
    outside = tmp_path / "outside"
    if not broken:
        outside.mkdir()
    attempts = case["round_dir"] / "source_attempts"
    attempts.symlink_to(outside, target_is_directory=True)
    events: list[str] = []

    with pytest.raises(
        controller.SourceLeaseContractError,
        match="source attempt directory",
    ):
        _run(case, _dependencies(FakeScheduler(events), events))

    assert events == []
    if outside.exists():
        assert list(outside.iterdir()) == []


def test_formal_source_resolution_rejects_prior_synthetic_round_artifacts(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _write_json(
        case["round_dir"] / "synthetic_protocol_receipt.json",
        {"synthetic_nonfinal": True},
    )
    events: list[str] = []

    with pytest.raises(
        controller.SourceLeaseContractError,
        match="synthetic",
    ):
        _run(
            case,
            _dependencies(FakeScheduler(events), events),
            pre_gate_source_only=True,
        )

    assert events == []


def test_formal_existing_source_fast_path_binds_without_gpu_or_locks(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events)
    formal = {
        "schema_version": "stage7_source_resolution_result_v2",
        "row_count": 4,
        "eligible_for_exact_cache_reveal": True,
        "source_resolution_result_sha256": "s" * 64,
    }

    result = _run(
        case,
        _dependencies(
            scheduler,
            events,
            prelease_result=formal,
        ),
    )

    assert result["status"] == "cache_revealed"
    assert result["fast_path"] is True
    assert events == ["bind"]
    assert (case["round_dir"] / "source_resolution_result.json").is_file()


def test_resource_shortage_is_zero_budget_same_request_retry(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events, ready=("GPU-a",))

    result = _run(case, _dependencies(scheduler, events))

    assert result["status"] == "waiting_for_source_gpu_capacity"
    assert result["selected_event_budget_delta"] == 0
    assert (
        result["logical_request_sha256"]
        == case["request"]["measurement_request_sha256"]
    )
    assert (
        result["source_resolution_plan_sha256"]
        == case["source_plan"]["source_resolution_plan_sha256"]
    )
    assert events == ["observe", "ready"]


def test_source_locks_precede_uuid_locks_and_post_lock_drift_releases_all(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events, post_lock_admitted=False)

    result = _run(case, _dependencies(scheduler, events))

    assert result["status"] == "waiting_for_source_gpu_capacity"
    source_lock_positions = [
        index for index, event in enumerate(events) if event.startswith("lock:source:")
    ]
    uuid_lock_positions = [
        index for index, event in enumerate(events) if event.startswith("lock:uuid:")
    ]
    assert source_lock_positions and uuid_lock_positions
    assert max(source_lock_positions) < min(uuid_lock_positions)
    assert "resolve:GPU-a,GPU-b" not in events
    assert sum(event.startswith("release:") for event in events) == 4


def test_signed_lease_resolver_and_staged_bind_run_under_exact_uuid_locks(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events)

    result = _run(case, _dependencies(scheduler, events))

    assert result["status"] == "cache_revealed"
    assert result["fast_path"] is False
    assert events.index("resolve:GPU-a,GPU-b") < events.index("bind")
    assert events.index("bind") < min(
        index for index, event in enumerate(events) if event.startswith("release:")
    )
    attempt = case["round_dir"] / "source_attempts/attempt_000"
    lease = json.loads((attempt / "source_gpu_lease.json").read_text())
    assert lease["lock_owner"] == {
        "pid": 800,
        "owner": "runner",
        "start_time": "123",
    }
    assert lease["gpu_uuids"] == ["GPU-a", "GPU-b"]
    assert (case["round_dir"] / "source_resolution_result.json").is_file()


def test_resolver_retry_is_zero_budget_and_next_attempt_keeps_identity(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events)
    retry = {
        "status": "retry_required",
        "retry_reason_code": "evidence_unavailable",
        "selected_event_budget_delta": 0,
        "logical_request_sha256": case["request"]["measurement_request_sha256"],
        "source_resolution_plan_sha256": case["source_plan"][
            "source_resolution_plan_sha256"
        ],
    }
    dependencies = _dependencies(
        scheduler,
        events,
        resolver_result=retry,
    )

    first = _run(case, dependencies)
    assert first["status"] == "retry_required"
    assert first["selected_event_budget_delta"] == 0

    # A fully authenticated terminal retry admits a new immutable attempt.
    second = _run(case, dependencies)
    assert second["status"] == "retry_required"
    assert (
        case["round_dir"] / "source_attempts/attempt_001/source_gpu_lease.json"
    ).is_file()


def test_incomplete_existing_attempt_cannot_be_taken_over_even_by_same_pid(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    attempt = case["round_dir"] / "source_attempts/attempt_000"
    attempt.mkdir(parents=True)
    _write_json(attempt / "source_gpu_lease.json", {"incomplete": True})
    events: list[str] = []
    scheduler = FakeScheduler(events)

    with pytest.raises(controller.SourceLeaseContractError, match="takeover"):
        _run(case, _dependencies(scheduler, events))

    assert events == []


def test_any_earlier_incomplete_attempt_blocks_later_terminal_retry(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(
        FakeScheduler(events),
        events,
        resolver_result={
            "status": "retry_required",
            "retry_reason_code": "evidence_unavailable",
        },
    )
    assert _run(case, dependencies)["status"] == "retry_required"
    attempt0 = case["round_dir"] / "source_attempts/attempt_000"
    attempt1 = case["round_dir"] / "source_attempts/attempt_001"
    attempt1.mkdir(parents=True)
    (attempt1 / "source_gpu_lease.json").write_bytes(
        (attempt0 / "source_gpu_lease.json").read_bytes()
    )
    _write_terminal_retry(attempt1)
    (attempt0 / "source_resolution_retry.json").unlink()

    with pytest.raises(controller.SourceLeaseContractError, match="takeover"):
        _run(case, dependencies)


@pytest.mark.parametrize(
    "mutation",
    ("schema", "signature", "retry_candidates", "eligibility", "receipt_signature"),
)
def test_recovery_rejects_forged_or_incomplete_terminal_retry(
    mutation: str, tmp_path: Path
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(
        FakeScheduler(events),
        events,
        resolver_result={
            "status": "retry_required",
            "retry_reason_code": "evidence_unavailable",
        },
    )
    assert _run(case, dependencies)["status"] == "retry_required"
    attempt = case["round_dir"] / "source_attempts/attempt_000"
    target = (
        attempt / "source_execution_receipt.json"
        if mutation == "receipt_signature"
        else attempt / "source_resolution_retry.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    if mutation == "schema":
        payload["schema_version"] = "forged"
    elif mutation == "signature":
        payload["source_resolution_retry_sha256"] = "f" * 64
    elif mutation == "retry_candidates":
        payload["retry_candidate_ids"] = list(reversed(payload["retry_candidate_ids"]))
    elif mutation == "eligibility":
        payload["eligible_for_cache_append"] = True
    else:
        payload["receipt_sha256"] = "f" * 64
    _write_json(target, payload)

    with pytest.raises(controller.SourceLeaseContractError, match="retry"):
        _run(case, dependencies)


def test_existing_canonical_result_is_idempotent_fast_path(tmp_path: Path) -> None:
    case = _case(tmp_path)
    canonical = case["round_dir"] / "source_resolution_result.json"
    _write_json(
        canonical,
        {"formal": True, "source_resolution_result_sha256": "s" * 64},
    )
    events: list[str] = []
    scheduler = FakeScheduler(events)

    result = _run(case, _dependencies(scheduler, events))

    assert result["status"] == "cache_revealed"
    assert result["fast_path"] is True
    assert events == ["bind"]


def test_existing_canonical_and_exact_artifacts_need_no_rebind(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    _write_json(
        case["round_dir"] / "source_resolution_result.json",
        {"formal": True, "source_resolution_result_sha256": "s" * 64},
    )
    for filename in controller.EXACT_ARTIFACTS:
        _write_json(case["round_dir"] / filename, {})
    events: list[str] = []

    result = _run(case, _dependencies(FakeScheduler(events), events))

    assert result["status"] == "cache_revealed"
    assert result["fast_path"] is True
    assert events == []


@pytest.mark.parametrize("kind", ("source", "uuid"))
def test_foreign_source_or_uuid_lock_waits_without_resolver(
    kind: str, tmp_path: Path
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events, unavailable_lock=kind)

    result = _run(case, _dependencies(scheduler, events))

    assert result["status"] == "waiting_for_source_gpu_capacity"
    assert result["selected_event_budget_delta"] == 0
    assert not any(event.startswith("resolve:") for event in events)


@pytest.mark.parametrize(
    "resolver_result,error",
    (
        ({"status": "mystery"}, "unknown status"),
        (
            {
                "status": "retry_required",
                "logical_request_sha256": "x" * 64,
                "source_resolution_plan_sha256": "y" * 64,
                "selected_event_budget_delta": 1,
            },
            "identity or budget",
        ),
    ),
)
def test_resolver_status_or_retry_identity_drift_is_fail_closed(
    resolver_result: dict[str, Any], error: str, tmp_path: Path
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events)

    with pytest.raises(controller.SourceLeaseContractError, match=error):
        _run(
            case,
            _dependencies(
                scheduler,
                events,
                resolver_result=resolver_result,
            ),
        )

    assert sum(event.startswith("release:") for event in events) == 4


def test_orchestrator_process_identity_drift_releases_all_locks(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path)
    events: list[str] = []
    scheduler = FakeScheduler(events)
    scheduler.process_inspector = lambda _pid: None

    with pytest.raises(controller.SourceLeaseContractError, match="owner drift"):
        _run(case, _dependencies(scheduler, events))

    assert sum(event.startswith("release:") for event in events) == 4


@pytest.mark.parametrize(
    ("variant", "seed", "round_index"),
    (
        ("..", 20260718, 0),
        ("unknown", 20260718, 0),
        ("full", -1, 0),
        ("full", 20260721, 0),
        ("full", 20260718, 4),
    ),
)
def test_public_round_identity_is_limited_to_frozen_experiment(
    variant: str, seed: int, round_index: int, tmp_path: Path
) -> None:
    case = _case(tmp_path)

    with pytest.raises(controller.SourceLeaseContractError, match="round identity"):
        controller.run_source_lease_controller(
            v2_root=case["v2_root"],
            repo_root=case["repo_root"],
            variant=variant,
            seed=seed,
            round_index=round_index,
            orchestrator_pid=800,
            dependencies=_dependencies(FakeScheduler([]), []),
        )
