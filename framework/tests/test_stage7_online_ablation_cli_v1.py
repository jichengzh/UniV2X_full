from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path
from typing import Any

import pytest

from scripts import stage7_online_ablation_v1 as cli


def _write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _request_chain(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    trajectory_payload = {
        "schema_version": "stage7_search_policy_v1_trajectory_contract",
        "task_id": "S7-PYR-TVM",
        "variant": "full",
        "seed": 20260718,
        "trajectory_dir": str(root / "variants/full/seed_20260718"),
    }
    trajectory = {
        **trajectory_payload,
        "trajectory_contract_sha256": cli.policy_api.canonical_sha256(
            trajectory_payload
        ),
    }
    rows = [
        {"row_id": f"row-{index}", "task_id": "S7-PYR-TVM", "task_sha256": "f" * 64}
        for index in range(4)
    ]
    request_payload = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S7-PYR-TVM",
        "task_sha256": "f" * 64,
        "round_index": 0,
        "batch_size": 4,
        "rows": rows,
        "row_sha256": {
            row["row_id"]: cli.policy_api.canonical_sha256(row) for row in rows
        },
    }
    request = {
        **request_payload,
        "measurement_request_sha256": cli.policy_api.canonical_sha256(
            request_payload
        ),
    }
    binding = cli.policy_api.build_request_binding(
        trajectory,
        request,
        round_index=0,
        trajectory_dir=trajectory["trajectory_dir"],
    )
    return trajectory, request, binding


def test_parser_exposes_only_the_seven_stage7_commands() -> None:
    parser = cli.build_parser()
    subcommands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ).choices
    assert set(subcommands) == {
        "prepare",
        "init-round",
        "advance-round",
        "reveal-cache",
        "build-miss-plan",
        "finalize-round",
        "audit-trajectory",
    }
    assert "S5-" not in Path(cli.__file__).read_text(encoding="utf-8")


def test_atomic_json_is_idempotent_and_refuses_sha_drift(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    payload = {"schema_version": "synthetic_v1", "value": 1}

    first = cli.atomic_write_json(destination, payload)
    second = cli.atomic_write_json(destination, payload)

    assert first == second == hashlib.sha256(destination.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="drift"):
        cli.atomic_write_json(destination, {**payload, "value": 2})
    assert json.loads(destination.read_text()) == payload


def test_two_round_selection_delegates_to_existing_policy_scripts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    feedback = _write_json(root / "round0-feedback.json", {"rows": []})
    calls: list[tuple[str, dict[str, Any]]] = []

    def initialize(**kwargs: Any) -> dict[str, Any]:
        calls.append(("init", kwargs))
        return {"schema_version": "stage7_initialize_result_v1", "round": 0}

    def advance(**kwargs: Any) -> dict[str, Any]:
        calls.append(("advance", kwargs))
        return {"schema_version": "stage7_advance_result_v1", "next_round_index": 1}

    monkeypatch.setattr(cli.prepare_api, "initialize_trajectories", initialize)
    monkeypatch.setattr(cli.advance_api, "advance_trajectory", advance)
    monkeypatch.setattr(cli, "_initialize_managed_outputs", lambda *args: ())
    monkeypatch.setattr(cli, "_advance_managed_outputs", lambda *args, **kwargs: ())

    assert cli.main(
        [
            "init-round",
            "--output-root",
            str(root),
            "--variant",
            "full",
            "--seed",
            "20260718",
            "--round-index",
            "0",
        ]
    ) == 0
    assert cli.main(
        [
            "advance-round",
            "--output-root",
            str(root),
            "--variant",
            "full",
            "--seed",
            "20260718",
            "--completed-round-index",
            "0",
            "--feedback-json",
            str(feedback),
        ]
    ) == 0
    assert [name for name, _ in calls] == ["init", "advance"]
    assert calls[0][1]["output_root"] == root
    assert calls[1][1]["feedback_json"] == feedback


def test_two_round_synthetic_trajectory_uses_real_policy_advance(
    tmp_path: Path,
) -> None:
    from framework.tests.test_stage7_advance_online_ablation_v1 import (
        _feedback_for_request,
        _setup_trajectory,
    )

    output_root = _setup_trajectory(tmp_path.resolve())
    trajectory_dir = (
        output_root / "variants/without_surrogate/seed_20260718"
    )
    assert cli.main(
        [
            "init-round",
            "--output-root",
            str(output_root),
            "--variant",
            "without_surrogate",
            "--seed",
            "20260718",
            "--round-index",
            "0",
        ]
    ) == 0
    init_receipt = json.loads(
        (
            output_root
            / "status/cli_receipts/"
            "init-round_without_surrogate_20260718_00.json"
        ).read_text()
    )
    assert str(trajectory_dir / "round_00/measurement_request.json") in {
        record["absolute_path"] for record in init_receipt["outputs"]
    }
    request_shas: list[str] = []
    for round_index in (0, 1):
        request = json.loads(
            (
                trajectory_dir
                / f"round_{round_index:02d}/measurement_request.json"
            ).read_text()
        )
        request_shas.append(request["measurement_request_sha256"])
        feedback = _write_json(
            tmp_path / f"feedback-{round_index}.json",
            {"rows": _feedback_for_request(request)},
        )
        assert cli.main(
            [
                "advance-round",
                "--output-root",
                str(output_root),
                "--variant",
                "without_surrogate",
                "--seed",
                "20260718",
                "--completed-round-index",
                str(round_index),
                "--feedback-json",
                str(feedback),
            ]
        ) == 0
        assert (
            json.loads(
                (
                    trajectory_dir
                    / f"round_{round_index:02d}/measurement_request.json"
                ).read_text()
            )["measurement_request_sha256"]
            == request_shas[-1]
        )
        advance_receipt = json.loads(
            (
                output_root
                / "status/cli_receipts/"
                f"advance-round_without_surrogate_20260718_{round_index:02d}.json"
            ).read_text()
        )
        advance_outputs = {
            record["absolute_path"] for record in advance_receipt["outputs"]
        }
        assert str(
            trajectory_dir / f"round_{round_index:02d}/released_feedback.json"
        ) in advance_outputs
        assert str(
            trajectory_dir
            / f"round_{round_index + 1:02d}/measurement_request.json"
        ) in advance_outputs
    assert len(set(request_shas)) == 2
    assert (trajectory_dir / "round_02/measurement_request.json").is_file()


def test_reveal_happens_only_after_persisted_request_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    trajectory_payload, request_payload, binding_payload = _request_chain(root)
    request = _write_json(root / "request.json", request_payload)
    binding = _write_json(root / "binding.json", binding_payload)
    trajectory = _write_json(root / "trajectory.json", trajectory_payload)
    dimensions = _write_json(root / "dimensions.json", {"rows": {}})
    cache = _write_json(root / "cache.json", {"entries": {}})
    observed: list[tuple[dict[str, Any], dict[str, Any]]] = []

    def reveal(request_payload: dict[str, Any], binding_payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        observed.append((request_payload, binding_payload))
        return {
            "schema_version": "stage7_cache_reveal_v1",
            "measurement_request_sha256": request_payload["measurement_request_sha256"],
            "cache_reveal_sha256": "d" * 64,
            "rows": [],
        }

    monkeypatch.setattr(cli.cache_api, "reveal_selected_batch_cache", reveal)
    output = root / "cache-reveal.json"
    assert not output.exists()

    argv = [
        "reveal-cache",
        "--output-root",
        str(root),
        "--request-json",
        str(request),
        "--request-binding-json",
        str(binding),
        "--trajectory-contract-json",
        str(trajectory),
        "--key-dimensions-json",
        str(dimensions),
        "--cache-json",
        str(cache),
        "--output-json",
        str(output),
    ]
    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    assert observed and output.is_file()
    assert json.loads(output.read_text())["schema_version"] == "stage7_cache_reveal_v1"
    _write_json(cache, {"entries": {"new-exact-key": {}}})
    with pytest.raises(ValueError, match="drift"):
        cli.main(argv)


def test_prepare_freezes_before_scanner_audit_and_records_all_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    inputs = {
        name: _write_json(root / f"{name}.json", {"name": name})
        for name in (
            "gold",
            "graphs",
            "profiles",
            "registry",
            "closure",
            "terminal",
        )
    }
    calls: list[str] = []

    def freeze_contracts(**kwargs: Any) -> dict[str, Any]:
        calls.append("freeze")
        for path in cli.prepare_managed_outputs(
            root, include_scanner_audit=False
        ):
            cli.atomic_write_json(
                path, {"schema_version": f"synthetic_{path.name}"}
            )
        return {"schema_version": "stage7_freeze_result_v1"}

    def audit_scanner(**kwargs: Any) -> dict[str, Any]:
        calls.append("audit")
        cli.atomic_write_json(
            root / "audits/scanner_admission.json",
            {"schema_version": "stage7_scanner_admission_v1"},
        )
        return {
            "schema_version": "stage7_scanner_admission_v1",
            "status": "scanner_ready_all_pass",
        }

    monkeypatch.setattr(cli.prepare_api, "freeze_contracts", freeze_contracts)
    monkeypatch.setattr(cli.prepare_api, "audit_scanner", audit_scanner)

    assert cli.main(
        [
            "prepare",
            "--output-root",
            str(root),
            "--gold176",
            str(inputs["gold"]),
            "--graph-features",
            str(inputs["graphs"]),
            "--capability-profiles",
            str(inputs["profiles"]),
            "--source-registry",
            str(inputs["registry"]),
            "--stage4-closure",
            str(inputs["closure"]),
            "--terminal-evidence-json",
            str(inputs["terminal"]),
            "--synthetic",
        ]
    ) == 0
    assert calls == ["freeze", "audit"]
    receipt = json.loads(
        (root / "status/cli_receipts/prepare_global.json").read_text()
    )
    assert len(receipt["inputs"]) == 6
    assert receipt["input_set_sha256"] == cli.canonical_sha256(receipt["inputs"])
    output_paths = {record["absolute_path"] for record in receipt["outputs"]}
    assert str(root / "contracts/frozen_inputs.json") in output_paths
    assert str(root / "audits/scanner_admission.json") in output_paths


def test_finalize_uses_path_promotion_adapter_then_existing_atomic_finalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    paths = {
        name: _write_json(root / f"{name}.json", payload)
        for name, payload in {
            "request": {"schema_version": "request"},
            "binding": {"schema_version": "binding"},
            "reveal": {"schema_version": "reveal"},
            "misses": {"rows": [{"row_id": "miss"}]},
        }.items()
    }
    callback_order: list[str] = []

    def path_promote(request_path: Path, feedback_path: Path) -> dict[str, Any]:
        callback_order.append("promote")
        assert request_path.is_file() and feedback_path.is_file()
        return {"rows": [{"row_id": "promoted"}], "audit": {"verdict": "pass"}}

    def orchestrate(
        request: dict[str, Any],
        binding: dict[str, Any],
        reveal: dict[str, Any],
        misses: list[dict[str, Any]],
        *,
        promote_feedback_batch: Any,
        finalize_atomic_batch: Any,
    ) -> dict[str, Any]:
        promoted = promote_feedback_batch(request, misses)
        assert promoted["audit"]["verdict"] == "pass"
        callback_order.append("finalize")
        assert callable(finalize_atomic_batch)
        return {
            "schema_version": "stage7_atomic_mixed_feedback_v1",
            "feedback_released": True,
            "measurement_request_sha256": "a" * 64,
            "released_feedback_rows": promoted["rows"],
        }

    monkeypatch.setattr(cli.promotion_api, "promote_feedback_batch", path_promote)
    monkeypatch.setattr(cli.cache_api, "finalize_stage7_atomic_batch", orchestrate)
    monkeypatch.setattr(
        cli, "_validated_cache_binding", lambda request, binding: binding
    )
    output = root / "finalization.json"
    released = root / "released.json"
    work = root / "promotion"
    assert cli.main(
        [
            "finalize-round",
            "--output-root",
            str(root),
            "--request-json",
            str(paths["request"]),
            "--request-binding-json",
            str(paths["binding"]),
            "--cache-reveal-json",
            str(paths["reveal"]),
            "--miss-results-json",
            str(paths["misses"]),
            "--promotion-work-dir",
            str(work),
            "--released-feedback-json",
            str(released),
            "--output-json",
            str(output),
        ]
    ) == 0
    assert callback_order == ["promote", "finalize"]
    assert json.loads(released.read_text())["rows"] == [{"row_id": "promoted"}]
    assert (work / "promotion_audit.json").is_file()
    receipt = json.loads(
        next((root / "status/cli_receipts").glob("finalize-round_*.json")).read_text()
    )
    receipt_outputs = {record["absolute_path"] for record in receipt["outputs"]}
    assert receipt_outputs == {
        str(output),
        str(released),
        str(work / "promotion_request.json"),
        str(work / "promotion_input_feedback.json"),
        str(work / "promoted_feedback.json"),
        str(work / "promotion_audit.json"),
    }
    (work / "promotion_audit.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drift"):
        cli.main(
            [
                "finalize-round",
                "--output-root",
                str(root),
                "--request-json",
                str(paths["request"]),
                "--request-binding-json",
                str(paths["binding"]),
                "--cache-reveal-json",
                str(paths["reveal"]),
                "--miss-results-json",
                str(paths["misses"]),
                "--promotion-work-dir",
                str(work),
                "--released-feedback-json",
                str(released),
                "--output-json",
                str(output),
            ]
        )


def test_audit_trajectory_checks_request_binding_and_expected_sha(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    trajectory, request, binding = _request_chain(root)
    paths = {
        "trajectory": _write_json(root / "trajectory.json", trajectory),
        "request": _write_json(root / "request.json", request),
        "binding": _write_json(root / "binding.json", binding),
    }
    output = root / "audit.json"
    argv = [
        "audit-trajectory",
        "--output-root",
        str(root),
        "--trajectory-contract-json",
        str(paths["trajectory"]),
        "--request-json",
        str(paths["request"]),
        "--request-binding-json",
        str(paths["binding"]),
        "--expected-request-sha256",
        request["measurement_request_sha256"],
        "--output-json",
        str(output),
    ]
    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    audit = json.loads(output.read_text())
    assert audit["verdict"] == "pass"
    assert audit["selected_row_ids"] == [f"row-{index}" for index in range(4)]
    drifted_argv = list(argv)
    drifted_argv[drifted_argv.index("--expected-request-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match="expected request"):
        cli.main(drifted_argv)


@pytest.mark.parametrize("drift", ["schema", "binding_schema", "row_sha"])
def test_audit_rejects_self_consistent_malformed_measurement_request(
    tmp_path: Path, drift: str
) -> None:
    root = tmp_path.resolve()
    trajectory, request, binding = _request_chain(root)
    request_payload = dict(request)
    request_payload.pop("measurement_request_sha256")
    if drift == "schema":
        request_payload["schema_version"] = "malformed_request_v1"
    elif drift == "row_sha":
        request_payload["row_sha256"] = {
            **request_payload["row_sha256"],
            "row-0": "0" * 64,
        }
    request_payload["measurement_request_sha256"] = (
        cli.policy_api.canonical_sha256(request_payload)
    )
    binding = {
        **binding,
        "measurement_request_sha256": request_payload[
            "measurement_request_sha256"
        ],
    }
    if drift == "binding_schema":
        binding["schema_version"] = "malformed_binding_v1"
    binding_without_sha = dict(binding)
    binding_without_sha.pop("request_binding_sha256")
    binding["request_binding_sha256"] = cli.policy_api.canonical_sha256(
        binding_without_sha
    )
    paths = {
        "trajectory": _write_json(root / "trajectory.json", trajectory),
        "request": _write_json(root / "request.json", request_payload),
        "binding": _write_json(root / "binding.json", binding),
    }
    with pytest.raises(ValueError, match="schema|row SHA"):
        cli.main(
            [
                "audit-trajectory",
                "--output-root",
                str(root),
                "--trajectory-contract-json",
                str(paths["trajectory"]),
                "--request-json",
                str(paths["request"]),
                "--request-binding-json",
                str(paths["binding"]),
                "--output-json",
                str(root / "audit.json"),
            ]
        )


def test_build_miss_plan_consumes_cache_api_without_reimplementing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    paths = {
        name: _write_json(root / f"{name}.json", {"schema_version": name})
        for name in ("full-plan", "request", "binding", "reveal")
    }
    sentinel = {
        "schema_version": "stage7_miss_only_plan_v1",
        "original_measurement_request_sha256": "a" * 64,
        "miss_plan_sha256": "e" * 64,
    }
    monkeypatch.setattr(
        cli.cache_api, "derive_miss_only_plan", lambda *args: sentinel
    )
    monkeypatch.setattr(
        cli, "_validated_cache_binding", lambda request, binding: binding
    )
    output = root / "miss-plan.json"

    assert cli.main(
        [
            "build-miss-plan",
            "--output-root",
            str(root),
            "--full-plan-json",
            str(paths["full-plan"]),
            "--request-json",
            str(paths["request"]),
            "--request-binding-json",
            str(paths["binding"]),
            "--cache-reveal-json",
            str(paths["reveal"]),
            "--output-json",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text()) == sentinel


@pytest.mark.parametrize("relative", ["request.json", "../request.json"])
def test_every_path_argument_must_be_absolute(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(ValueError, match="absolute"):
        cli.main(
            [
                "audit-trajectory",
                "--output-root",
                str(tmp_path.resolve()),
                "--trajectory-contract-json",
                relative,
                "--request-json",
                str((tmp_path / "request.json").resolve()),
                "--request-binding-json",
                str((tmp_path / "binding.json").resolve()),
                "--output-json",
                str((tmp_path / "audit.json").resolve()),
            ]
        )
