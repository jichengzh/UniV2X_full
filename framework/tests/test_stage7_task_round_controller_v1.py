from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = REPO_ROOT / "scripts/stage7_task_round_controller_v1.sh"
VARIANT = "full"
SEED = 20260718
REQUEST_SHA = ""
GPU_UUIDS = "GPU-aaaa,GPU-bbbb,GPU-cccc,GPU-dddd"
STAGES = (
    "validate_controller",
    "select_request",
    "validate_request",
    "reveal_cache",
    "materialize_sources",
    "build_miss_plan",
    "execute_performance",
    "execute_ap",
    "finalize_round",
    "audit_round",
)


def _canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_payload(round_index: int) -> dict[str, Any]:
    rows = [
        {
            "row_id": f"row-{round_index}-{index}",
            "task_id": "S7-PYR-TVM",
            "round_index": round_index,
        }
        for index in range(4)
    ]
    payload = {
        "schema_version": "stage7_measurement_request_v1",
        "task_id": "S7-PYR-TVM",
        "variant": VARIANT,
        "seed": SEED,
        "round_index": round_index,
        "batch_size": 4,
        "rows": rows,
    }
    return {**payload, "measurement_request_sha256": _canonical_sha(payload)}


REQUEST_SHA = _request_payload(0)["measurement_request_sha256"]


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _fake_program(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            r"""
            #!/usr/bin/env python3
            from __future__ import annotations
            import hashlib
            import json
            import os
            import sys
            from pathlib import Path

            def sha(payload):
                return hashlib.sha256(
                    json.dumps(
                        payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()

            def option(name, default=None):
                if name not in sys.argv:
                    return default
                return sys.argv[sys.argv.index(name) + 1]

            def write(path, payload):
                target = Path(path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
                    encoding="utf-8",
                )

            def read(path):
                return json.loads(Path(path).read_text(encoding="utf-8"))

            log = Path(os.environ["STAGE7_FAKE_LOG"])
            action = sys.argv[1]
            with log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {"action": action, "argv": sys.argv[2:]},
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )

            if os.environ.get("STAGE7_FAKE_INFRA_STAGE") == action:
                raise SystemExit(75)

            if action in {"init-round", "advance-round"}:
                output_root = Path(option("--output-root"))
                variant = option("--variant")
                seed = int(option("--seed"))
                round_index = (
                    int(option("--round-index"))
                    if action == "init-round"
                    else int(option("--completed-round-index")) + 1
                )
                trajectory_dir = output_root / "variants" / variant / f"seed_{seed}"
                round_dir = trajectory_dir / f"round_{round_index:02d}"
                trajectory_payload = {
                    "schema_version": "stage7_trajectory_contract_v1",
                    "task_id": "S7-PYR-TVM",
                    "variant": variant,
                    "seed": seed,
                    "trajectory_dir": str(trajectory_dir),
                }
                trajectory = {
                    **trajectory_payload,
                    "trajectory_contract_sha256": sha(trajectory_payload),
                }
                rows = [
                    {
                        "row_id": f"row-{round_index}-{index}",
                        "task_id": "S7-PYR-TVM",
                        "round_index": round_index,
                    }
                    for index in range(4)
                ]
                request_payload = {
                    "schema_version": "stage7_measurement_request_v1",
                    "task_id": "S7-PYR-TVM",
                    "variant": variant,
                    "seed": seed,
                    "round_index": round_index,
                    "batch_size": 4,
                    "rows": rows,
                }
                request = {
                    **request_payload,
                    "measurement_request_sha256": sha(request_payload),
                }
                selected = [row["row_id"] for row in rows]
                binding_payload = {
                    "schema_version": "stage7_request_binding_v1",
                    "measurement_request_sha256": request[
                        "measurement_request_sha256"
                    ],
                    "trajectory_contract_sha256": trajectory[
                        "trajectory_contract_sha256"
                    ],
                    "round_index": round_index,
                    "selected_row_ids": selected,
                }
                binding = {
                    **binding_payload,
                    "request_binding_sha256": sha(binding_payload),
                }
                write(trajectory_dir / "trajectory_contract.json", trajectory)
                write(round_dir / "measurement_request.json", request)
                write(round_dir / "stage7_request_binding.json", binding)
                tamper_contract = os.environ.get("STAGE7_FAKE_TAMPER_CONTRACT")
                if tamper_contract:
                    Path(tamper_contract).write_text("{}", encoding="utf-8")
                raise SystemExit(0)

            if action == "audit-trajectory":
                request = read(option("--request-json"))
                binding = read(option("--request-binding-json"))
                expected = option("--expected-request-sha256")
                assert request["measurement_request_sha256"] == expected
                assert binding["measurement_request_sha256"] == expected
                assert len(binding["selected_row_ids"]) == 4
                assert len(set(binding["selected_row_ids"])) == 4
                write(
                    option("--output-json"),
                    {
                        "schema_version": "stage7_trajectory_audit_v1",
                        "verdict": "pass",
                        "measurement_request_sha256": expected,
                        "selected_row_ids": binding["selected_row_ids"],
                    },
                )
                raise SystemExit(0)

            if action == "reveal-cache":
                request = read(option("--request-json"))
                hits = int(os.environ["STAGE7_FAKE_HIT_COUNT"])
                rows = []
                for index, row in enumerate(request["rows"]):
                    rows.append(
                        {
                            "row_id": row["row_id"],
                            "disposition": (
                                "exact_hit" if index < hits else "miss"
                            ),
                            **(
                                {
                                    "bound_result": {
                                        "row_id": row["row_id"],
                                        "terminal_status": "measured_success_gold",
                                    }
                                }
                                if index < hits
                                else {}
                            ),
                        }
                    )
                write(
                    option("--output-json"),
                    {
                        "schema_version": "stage7_cache_reveal_v1",
                        "measurement_request_sha256": request[
                            "measurement_request_sha256"
                        ],
                        "exact_hit_count": hits,
                        "miss_count": 4 - hits,
                        "rows": rows,
                    },
                )
                raise SystemExit(0)

            if action == "build-miss-plan":
                reveal = read(option("--cache-reveal-json"))
                misses = [
                    row for row in reveal["rows"] if row["disposition"] == "miss"
                ]
                jobs = [
                    {
                        "row_id": row["row_id"],
                        "group_id": f"group-{index}",
                    }
                    for index, row in enumerate(misses)
                ]
                write(
                    option("--output-json"),
                    {
                        "schema_version": "stage7_miss_only_performance_plan_v1",
                        "original_measurement_request_sha256": reveal[
                            "measurement_request_sha256"
                        ],
                        "miss_row_ids": [row["row_id"] for row in misses],
                        "manifest": {
                            "row_count": len(misses),
                            "genome_count": len(misses),
                            "jobs": jobs,
                        },
                        "performance_jobs": jobs,
                    },
                )
                raise SystemExit(0)

            if action == "full-plan":
                request = read(option("--request-json"))
                request_sha = (
                    "0" * 64
                    if os.environ.get("STAGE7_FAKE_PLAN_DRIFT") == "1"
                    else request["measurement_request_sha256"]
                )
                jobs = [
                    {"row_id": row["row_id"], "group_id": f"group-{index}"}
                    for index, row in enumerate(request["rows"])
                ]
                write(
                    option("--output-json"),
                    {
                        "schema_version": "fake_full_performance_plan_v1",
                        "manifest": {
                            "source_request_sha256": request_sha,
                            "row_count": 4,
                            "genome_count": 4,
                            "jobs": jobs,
                        },
                        "performance_jobs": jobs,
                    },
                )
                raise SystemExit(0)

            if action == "finalize-round":
                reveal = read(option("--cache-reveal-json"))
                misses_payload = read(option("--miss-results-json"))
                rows = [
                    row["bound_result"]
                    for row in reveal["rows"]
                    if row["disposition"] == "exact_hit"
                ] + misses_payload["rows"]
                result = {
                    "schema_version": "stage7_atomic_mixed_feedback_v1",
                    "feedback_released": True,
                    "budget_consumed": 4,
                    "measurement_request_sha256": reveal[
                        "measurement_request_sha256"
                    ],
                    "released_feedback_rows": rows,
                }
                write(option("--output-json"), result)
                write(option("--released-feedback-json"), {"rows": rows})
                raise SystemExit(0)

            output = option("--output-json")
            if action == "materialize":
                write(
                    output,
                    {
                        "schema_version": "fake_materialization_v1",
                        "status": "verified",
                    },
                )
            elif action == "performance":
                plan = read(option("--miss-plan-json"))
                write(
                    output,
                    {
                        "schema_version": "fake_performance_v1",
                        "rows": plan["performance_jobs"],
                    },
                )
            elif action == "ap":
                performance = read(option("--performance-results-json"))
                candidate_failure = (
                    os.environ.get("STAGE7_FAKE_CANDIDATE_FAILURE") == "1"
                )
                write(
                    output,
                    {
                        "schema_version": "fake_miss_results_v1",
                        "rows": [
                            {
                                "row_id": row["row_id"],
                                "terminal_status": (
                                    "runtime_failure"
                                    if candidate_failure and index == 0
                                    else "measured_success_gold"
                                ),
                            }
                            for index, row in enumerate(performance["rows"])
                        ],
                    },
                )
            elif action == "release":
                pass
            else:
                raise AssertionError(f"unknown fake action: {action}")
            """
        ).lstrip(),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _case(tmp_path: Path, *, hit_count: int, round_index: int = 0) -> dict[str, Any]:
    output_root = (tmp_path / "formal").resolve()
    trajectory_dir = output_root / "variants" / VARIANT / f"seed_{SEED}"
    round_dir = trajectory_dir / f"round_{round_index:02d}"
    round_dir.mkdir(parents=True)
    fake = _fake_program(tmp_path / "fake_stage7.py")
    log = tmp_path / "calls.jsonl"
    key_dimensions = _write_json(tmp_path / "key_dimensions.json", {"rows": {}})
    cache = _write_json(tmp_path / "cache.json", {"entries": {}})
    full_plan = round_dir / "full_plan.json"
    config = _write_json(
        tmp_path / "controller.json",
        {
            "schema_version": "stage7_task_round_controller_config_v1",
            "python_executable": sys.executable,
            "unified_cli_argv": [str(fake)],
            "trajectory_contract_json": str(
                trajectory_dir / "trajectory_contract.json"
            ),
            "key_dimensions_json": str(key_dimensions),
            "cache_json": str(cache),
            "full_plan_json": str(full_plan),
            "build_full_plan_command": [
                sys.executable,
                str(fake),
                "full-plan",
            ],
            "materialize_sources_argv": [sys.executable, str(fake), "materialize"],
            "execute_performance_argv": [
                sys.executable,
                str(fake),
                "performance",
            ],
            "execute_ap_argv": [sys.executable, str(fake), "ap"],
            "release_lease_argv": [sys.executable, str(fake), "release"],
        },
    )
    request_sha = _request_payload(round_index)["measurement_request_sha256"]
    env = {
        **os.environ,
        "STAGE7_CONTROLLER_CONFIG_JSON": str(config),
        "STAGE7_FAKE_LOG": str(log),
        "STAGE7_FAKE_HIT_COUNT": str(hit_count),
        "CUDA_VISIBLE_DEVICES": GPU_UUIDS,
    }
    command = [
        "bash",
        str(CONTROLLER),
        "--output-root",
        str(output_root),
        "--variant",
        VARIANT,
        "--seed",
        str(SEED),
        "--round-index",
        str(round_index),
        "--request-sha256",
        request_sha,
    ]
    return {
        "output_root": output_root,
        "round_dir": round_dir,
        "log": log,
        "config": config,
        "env": env,
        "command": command,
        "request_sha": request_sha,
    }


def _run(
    case: dict[str, Any],
    *,
    env_update: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        case["command"],
        cwd=cwd or case["round_dir"],
        env={**case["env"], **(env_update or {})},
        capture_output=True,
        text=True,
        timeout=30,
    )


def _actions(case: dict[str, Any]) -> list[str]:
    if not case["log"].is_file():
        return []
    return [
        json.loads(line)["action"]
        for line in case["log"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize(
    ("hit_count", "expected_runner_actions"),
    [
        (4, []),
        (2, ["materialize", "performance", "ap"]),
        (0, ["materialize", "performance", "ap"]),
    ],
)
def test_all_hit_mixed_and_all_miss_follow_only_the_required_path(
    tmp_path: Path, hit_count: int, expected_runner_actions: list[str]
) -> None:
    case = _case(tmp_path, hit_count=hit_count)

    completed = _run(case)

    assert completed.returncode == 0, completed.stderr
    actions = _actions(case)
    assert actions.count("init-round") == 1
    assert actions.count("full-plan") == 1
    assert [action for action in actions if action in {"materialize", "performance", "ap"}] == (
        expected_runner_actions
    )
    assert actions[-3:] == ["finalize-round", "audit-trajectory", "release"]
    state = json.loads(
        (
            case["output_root"] / "status/round_00_stage.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["stage"] == "audit_round"
    assert state["request_sha256"] == case["request_sha"]
    assert [entry["stage"] for entry in state["completed_stages"]] == list(STAGES)
    if hit_count == 4:
        assert state["gpu_runner_calls"] == 0


@pytest.mark.parametrize("interrupt_after", STAGES)
def test_interruption_after_every_stage_resumes_without_reselection_or_sha_change(
    tmp_path: Path, interrupt_after: str
) -> None:
    case = _case(tmp_path, hit_count=2)

    interrupted = _run(
        case,
        env_update={"STAGE7_INTERRUPT_AFTER_STAGE": interrupt_after},
    )
    assert interrupted.returncode == 99
    resumed = _run(case)

    assert resumed.returncode == 0, resumed.stderr
    request = json.loads(
        (case["round_dir"] / "measurement_request.json").read_text(encoding="utf-8")
    )
    assert request["measurement_request_sha256"] == case["request_sha"]
    assert _actions(case).count("init-round") == 1
    state = json.loads(
        (
            case["output_root"] / "status/round_00_stage.json"
        ).read_text(encoding="utf-8")
    )
    assert state["attempt"] == 2
    assert state["status"] == "complete"


def test_cwd_request_flag_binding_and_four_selected_ids_fail_before_cache(
    tmp_path: Path,
) -> None:
    wrong_cwd_case = _case(tmp_path / "cwd", hit_count=2)
    wrong_cwd = _run(wrong_cwd_case, cwd=wrong_cwd_case["output_root"])
    assert wrong_cwd.returncode != 0
    assert _actions(wrong_cwd_case) == ["release"]

    flag_case = _case(tmp_path / "flag", hit_count=2)
    flag_case["command"][-1] = "f" * 64
    wrong_flag = _run(flag_case)
    assert wrong_flag.returncode != 0
    assert "reveal-cache" not in _actions(flag_case)

    binding_case = _case(tmp_path / "binding", hit_count=2)
    stopped = _run(
        binding_case,
        env_update={"STAGE7_INTERRUPT_AFTER_STAGE": "select_request"},
    )
    assert stopped.returncode == 99
    binding_path = binding_case["round_dir"] / "stage7_request_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    _write_json(binding_path, {**binding, "request_binding_sha256": "0" * 64})
    drift = _run(binding_case)
    assert drift.returncode != 0
    assert "reveal-cache" not in _actions(binding_case)

    ids_case = _case(tmp_path / "ids", hit_count=2)
    stopped = _run(
        ids_case,
        env_update={"STAGE7_INTERRUPT_AFTER_STAGE": "select_request"},
    )
    assert stopped.returncode == 99
    binding_path = ids_case["round_dir"] / "stage7_request_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    selected = binding["selected_row_ids"][:3]
    payload = {
        key: value
        for key, value in binding.items()
        if key != "request_binding_sha256"
    }
    payload["selected_row_ids"] = selected
    _write_json(
        binding_path,
        {**payload, "request_binding_sha256": _canonical_sha(payload)},
    )
    bad_ids = _run(ids_case)
    assert bad_ids.returncode != 0
    assert "reveal-cache" not in _actions(ids_case)


def test_infrastructure_retry_preserves_request_sha_and_budget_then_resumes(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, hit_count=2)

    retry = _run(
        case,
        env_update={"STAGE7_FAKE_INFRA_STAGE": "performance"},
    )

    assert retry.returncode == 75
    state_path = case["output_root"] / "status/round_00_stage.json"
    retry_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert retry_state["status"] == "infrastructure_retry_required"
    assert retry_state["request_sha256"] == case["request_sha"]
    assert retry_state["budget_consumed"] == 0
    resumed = _run(case)
    assert resumed.returncode == 0, resumed.stderr
    assert _actions(case).count("init-round") == 1
    complete_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert complete_state["request_sha256"] == case["request_sha"]
    assert complete_state["budget_consumed"] == 4


@pytest.mark.parametrize(
    ("env_update", "expected_returncode", "expected_status"),
    [
        (
            {"STAGE7_FAKE_INFRA_STAGE": "full-plan"},
            75,
            "infrastructure_retry_required",
        ),
        ({"STAGE7_FAKE_PLAN_DRIFT": "1"}, 2, "terminal_invalid"),
    ],
)
def test_full_plan_builder_is_request_bound_and_retry_classified(
    tmp_path: Path,
    env_update: dict[str, str],
    expected_returncode: int,
    expected_status: str,
) -> None:
    case = _case(tmp_path, hit_count=2)
    assert not (case["round_dir"] / "full_plan.json").exists()

    failed = _run(case, env_update=env_update)

    assert failed.returncode == expected_returncode
    assert "build-miss-plan" not in _actions(case)
    assert not any(
        action in {"performance", "ap"} for action in _actions(case)
    )
    state = json.loads(
        (
            case["output_root"] / "status/round_00_stage.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == expected_status
    assert state["request_sha256"] == case["request_sha"]
    assert state["budget_consumed"] == 0
    if expected_status == "infrastructure_retry_required":
        resumed = _run(case)
        assert resumed.returncode == 0, resumed.stderr
        assert _actions(case).count("init-round") == 1
        assert _actions(case).count("full-plan") == 2
        resumed_state = json.loads(
            (
                case["output_root"] / "status/round_00_stage.json"
            ).read_text(encoding="utf-8")
        )
        assert resumed_state["request_sha256"] == case["request_sha"]
        assert resumed_state["status"] == "complete"


def test_true_candidate_failure_is_finalized_as_a_consumed_terminal_event(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, hit_count=0)

    completed = _run(
        case,
        env_update={"STAGE7_FAKE_CANDIDATE_FAILURE": "1"},
    )

    assert completed.returncode == 0, completed.stderr
    feedback = json.loads(
        (case["round_dir"] / "final_feedback.json").read_text(encoding="utf-8")
    )
    assert feedback["rows"][0]["terminal_status"] == "runtime_failure"
    state = json.loads(
        (
            case["output_root"] / "status/round_00_stage.json"
        ).read_text(encoding="utf-8")
    )
    assert state["status"] == "complete"
    assert state["budget_consumed"] == 4


def test_trap_calls_only_configured_owned_release_and_keeps_evidence(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, hit_count=2)

    interrupted = _run(
        case,
        env_update={"STAGE7_INTERRUPT_AFTER_STAGE": "reveal_cache"},
    )

    assert interrupted.returncode == 99
    assert _actions(case).count("release") == 1
    assert (case["round_dir"] / "cache_reveal.json").is_file()

    without_release = _case(tmp_path / "none", hit_count=2)
    config = json.loads(without_release["config"].read_text(encoding="utf-8"))
    config.pop("release_lease_argv")
    _write_json(without_release["config"], config)
    interrupted = _run(
        without_release,
        env_update={"STAGE7_INTERRUPT_AFTER_STAGE": "reveal_cache"},
    )
    assert interrupted.returncode == 99
    assert "release" not in _actions(without_release)


def test_round_barrier_blocks_advance_and_metacharacters_stay_literal(
    tmp_path: Path,
) -> None:
    barrier = _case(tmp_path / "barrier", hit_count=2, round_index=1)
    blocked = _run(barrier)
    assert blocked.returncode != 0
    assert _actions(barrier) == ["release"]

    literal = _case(tmp_path / "literal", hit_count=2)
    marker = tmp_path / "must-not-exist"
    config = json.loads(literal["config"].read_text(encoding="utf-8"))
    config["materialize_sources_argv"].append(f"$(touch {marker})")
    _write_json(literal["config"], config)
    completed = _run(literal)
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    materialize = next(
        json.loads(line)
        for line in literal["log"].read_text(encoding="utf-8").splitlines()
        if json.loads(line)["action"] == "materialize"
    )
    assert f"$(touch {marker})" in materialize["argv"]


def test_optional_pre_stage_validator_rechecks_before_next_stage_and_on_recovery(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, hit_count=2)
    contract = tmp_path / "contract.json"
    contract.write_text("trusted\n", encoding="utf-8")
    validator = tmp_path / "validator.py"
    validator.write_text(
        textwrap.dedent(
            """
            from pathlib import Path
            import os
            import sys

            stage = sys.argv[1]
            contract = Path(os.environ["STAGE7_TEST_CONTRACT"])
            if stage == "validate_controller":
                contract.write_text("tampered\\n", encoding="utf-8")
            elif contract.read_text(encoding="utf-8") != "trusted\\n":
                raise SystemExit(23)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    case["env"] = {
        **case["env"],
        "STAGE7_TEST_CONTRACT": str(contract),
        "STAGE7_PRE_STAGE_VALIDATOR_ARGV_JSON": json.dumps(
            [sys.executable, str(validator)]
        ),
    }

    interrupted = _run(
        case,
        env_update={"STAGE7_INTERRUPT_AFTER_STAGE": "validate_controller"},
    )
    assert interrupted.returncode == 99
    resumed = _run(case)

    assert resumed.returncode != 0
    assert _actions(case) == ["release", "release"]
    state = json.loads(
        (case["output_root"] / "status/round_00_stage.json").read_text(
            encoding="utf-8"
        )
    )
    assert [entry["stage"] for entry in state["completed_stages"]] == [
        "validate_controller"
    ]
    assert state["failure"]["stage"] == "select_request"


def test_shell_syntax_and_shellcheck_when_available() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(CONTROLLER)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert syntax.returncode == 0, syntax.stderr
    shellcheck = shutil.which("shellcheck")
    if shellcheck is not None:
        checked = subprocess.run(
            [shellcheck, str(CONTROLLER)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
