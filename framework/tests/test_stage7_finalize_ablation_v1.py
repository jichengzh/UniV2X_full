from __future__ import annotations

import hashlib
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import stage7_finalize_ablation_v1 as finalizer


VARIANTS = (
    "full",
    "without_surrogate",
    "without_measured_feedback",
    "backend_blind",
    "without_capability_scan",
)
SEEDS = (20260718, 20260719, 20260720)
SHA = {
    "pre_scan": "1" * 64,
    "scan_pass": "2" * 64,
    "scanner_rule": "3" * 64,
    "scanner_decision": "4" * 64,
    "bundle": "5" * 64,
    "prediction": "6" * 64,
    "training": "7" * 64,
    "graph": "8" * 64,
}


def test_direct_cli_bootstraps_repository_imports_from_arbitrary_cwd(
    tmp_path: Path,
) -> None:
    script = Path(finalizer.__file__).resolve()

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--input-root" in completed.stdout
    assert "--output-root" in completed.stdout


def _payload_sha(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_embedded_json(path: Path, payload: dict, sha_field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != sha_field}
    embedded_sha = _payload_sha(unsigned)
    _write_json(path, {**unsigned, sha_field: embedded_sha})
    return embedded_sha


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _rewrite_round_feedback(
    trajectory_dir: Path,
    round_index: int,
    rows: list[dict],
) -> str:
    round_dir = trajectory_dir / f"round_{round_index:02d}"
    finalization_path = round_dir / "stage7_finalization.json"
    finalization = json.loads(finalization_path.read_text(encoding="utf-8"))
    finalization["released_feedback_rows"] = rows
    _rewrite_embedded_json(
        finalization_path,
        finalization,
        "stage7_finalization_sha256",
    )
    released_path = round_dir / "released_feedback.json"
    released = json.loads(released_path.read_text(encoding="utf-8"))
    released["rows"] = rows
    released["feedback_input_sha256"] = hashlib.sha256(
        finalization_path.read_bytes()
    ).hexdigest()
    released_sha = _rewrite_embedded_json(
        released_path,
        released,
        "released_feedback_sha256",
    )
    terminal_path = trajectory_dir / "trajectory_terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    all_rows = [
        row
        for current_round in range(4)
        for row in json.loads(
            (
                trajectory_dir
                / f"round_{current_round:02d}"
                / "released_feedback.json"
            ).read_text(encoding="utf-8")
        )["rows"]
    ]
    terminal["released_feedback_sha256"] = _payload_sha(all_rows)
    _rewrite_embedded_json(
        terminal_path,
        terminal,
        "trajectory_terminal_sha256",
    )
    return released_sha


def _trajectory_dir(root: Path, variant: str, seed: int) -> Path:
    return root / "variants" / variant / f"seed_{seed}"


def _build_trajectory(
    root: Path,
    variant: str,
    seed: int,
    contract_shas: dict[str, str],
) -> None:
    trajectory_dir = _trajectory_dir(root, variant, seed)
    contract_payload = {
        "schema_version": "stage7_search_policy_v1_trajectory_contract",
        "task_id": "S7-PYR-TVM",
        "variant": variant,
        "seed": seed,
        "policy_name": "predicted_frontier_diversity",
        "result_root": str(root.resolve()),
        "trajectory_dir": str(trajectory_dir.resolve()),
        "frozen_input_sha256": contract_shas["frozen_input_sha256"],
        "pre_scan_sha256": contract_shas["pre_scan_sha256"],
        "scan_pass_sha256": contract_shas["scan_pass_sha256"],
        "scanner_rule_sha256": contract_shas["scanner_rule_sha256"],
        "scanner_decision_sha256": contract_shas["scanner_decision_sha256"],
        "candidate_pool_sha256": (
            contract_shas["pre_scan_sha256"]
            if variant == "without_capability_scan"
            else contract_shas["scan_pass_sha256"]
        ),
    }
    contract_sha = _payload_sha(contract_payload)
    contract = {**contract_payload, "trajectory_contract_sha256": contract_sha}
    _write_json(trajectory_dir / "trajectory_contract.json", contract)
    previous_feedback_sha = None
    feedback_shas: list[str] = []
    all_feedback_rows: list[dict] = []
    all_selected_ids: list[str] = []
    variant_index = VARIANTS.index(variant)
    for round_index in range(4):
        round_dir = trajectory_dir / f"round_{round_index:02d}"
        selected_rows = [
            {
                "row_id": (
                    f"full-r{round_index}-e{event_index}"
                    if variant == "full"
                    else f"{variant}-{seed}-r{round_index}-e{event_index}"
                ),
                "q_mode": "fp16" if event_index % 2 == 0 else "int8",
            }
            for event_index in range(4)
        ]
        request_payload = {
            "schema_version": "stage5_measurement_request_v2",
            "task_id": "S7-PYR-TVM",
            "round_index": round_index,
            "trajectory_contract_sha256": contract_sha,
            "rows": selected_rows,
        }
        request_sha = _payload_sha(request_payload)
        request = {**request_payload, "measurement_request_sha256": request_sha}
        _write_json(round_dir / "measurement_request.json", request)
        selected_ids = [row["row_id"] for row in selected_rows]
        all_selected_ids.extend(selected_ids)
        binding_payload = {
            "schema_version": "stage7_search_policy_v1_request_binding",
            "task_id": "S7-PYR-TVM",
            "variant": variant,
            "seed": seed,
            "round_index": round_index,
            "trajectory_dir": str(trajectory_dir.resolve()),
            "measurement_request_sha256": request_sha,
            "trajectory_contract_sha256": contract_sha,
            "selected_row_ids": selected_ids,
            "selected_row_ids_sha256": _payload_sha(selected_ids),
        }
        binding_sha = _payload_sha(binding_payload)
        binding = {**binding_payload, "request_binding_sha256": binding_sha}
        _write_json(round_dir / "stage7_request_binding.json", binding)
        _write_json(
            round_dir / "cache_selection_invariance_audit.json",
            {
                "verdict": "pass",
                "selection_input_sha256": "9" * 64,
                "empty_cache_selected_ids": selected_ids,
                "populated_cache_selected_ids": selected_ids,
                "cache_labels_released_to_selector": False,
                "audit_sha256": "a" * 64,
            },
        )
        if variant == "without_measured_feedback":
            a2_payload = {
                "schema_version": "stage7_search_policy_v1_a2_frozen",
                "bundle_manifest": {"bundle_config_sha256": SHA["bundle"]},
                "bundle_sha256": SHA["bundle"],
                "candidate_predictions": [],
                "prediction_view_sha256": _payload_sha([]),
                "training_view_sha256": SHA["training"],
                "graph_feature_view_sha256": SHA["graph"],
                "anchor": {},
                "anchor_sha256": _payload_sha({}),
            }
            _write_json(
                round_dir / "a2_frozen_contract.json",
                {
                    **a2_payload,
                    "frozen_payload_sha256": _payload_sha(a2_payload),
                },
            )
        if variant == "backend_blind":
            _write_json(
                round_dir / "backend_blind_audit.json",
                {
                    "full_feature_names": ["width", "capability_profile_id"],
                    "blind_feature_names": ["width"],
                    "removed_names": ["capability_profile_id"],
                    "full_training_matrix_sha256": "1" * 64,
                    "blind_training_matrix_sha256": "2" * 64,
                    "full_candidate_matrix_sha256": "3" * 64,
                    "blind_candidate_matrix_sha256": "4" * 64,
                    "leakage_verdict": "pass",
                },
            )
        cache_entries = [
            {
                "row_id": row_id,
                "disposition": "exact_hit" if event_index < 2 else "miss",
                "revealed_after_request_sha256": request_sha,
                "trajectory_contract_sha256": contract_sha,
                "source_cache_entry_sha256": "d" * 64 if event_index < 2 else None,
                "source_evidence_sha256": "e" * 64 if event_index < 2 else None,
            }
            for event_index, row_id in enumerate(selected_ids)
        ]
        cache_payload = {
            "schema_version": "stage7_selected_batch_cache_reveal_v1",
            "measurement_request_sha256": request_sha,
            "request_binding_sha256": binding_sha,
            "trajectory_contract_sha256": contract_sha,
            "row_count": 4,
            "exact_hit_count": 2,
            "miss_count": 2,
            "rows": cache_entries,
        }
        cache_sha = _payload_sha(cache_payload)
        cache_reveal = {**cache_payload, "cache_reveal_sha256": cache_sha}
        _write_json(round_dir / "cache_reveal.json", cache_reveal)
        controller_id = f"{variant}:seed_{seed}:round_{round_index}"
        start = float(
            VARIANTS.index(variant) * 10000
            + SEEDS.index(seed) * 1000
            + round_index * 100
            + 1
        )
        gpu_uuids = [f"GPU-{index}" for index in range(4)]
        _append_jsonl(
            root / "status" / "gpu_lease_audit.jsonl",
            {
                "schema_version": "stage7_gpu_lease_audit_v1",
                "event": "batch_leased",
                "wall_time": start,
                "controller_id": controller_id,
                "request_sha256": request_sha,
                "gpu_uuids": gpu_uuids,
            },
        )
        _append_jsonl(
            root / "status" / "gpu_occupancy_snapshots.jsonl",
            {
                "schema_version": "stage7_gpu_occupancy_snapshot_v1",
                "event": "sample",
                "wall_time": start + 45.0,
                "gpus": [
                    {
                        "uuid": uuid,
                        "index": index,
                        "memory_used_mib": 256,
                        "utilization_percent": 50,
                        "compute_pids": [1000 + index],
                    }
                    for index, uuid in enumerate(gpu_uuids)
                ],
            },
        )
        _append_jsonl(
            root / "status" / "gpu_lease_audit.jsonl",
            {
                "schema_version": "stage7_gpu_lease_audit_v1",
                "event": "controller_finished",
                "wall_time": start + 90.0,
                "controller_id": controller_id,
                "returncode": 0,
            },
        )
        feedback_rows = [
            {
                **row,
                "terminal_status": "measured_success",
                "metric_source": "measured_artifacts",
                "latency_ms": 10.0 + variant_index,
                "energy_j": 10.0 + variant_index,
                "ap70": 0.5,
                "latency_artifact_sha256": "a" * 64,
                "energy_artifact_sha256": "b" * 64,
                "ap_artifact_sha256": "c" * 64,
                "cache_disposition": cache_entries[event_index]["disposition"],
            }
            for event_index, row in enumerate(selected_rows)
        ]
        finalization_payload = {
            "schema_version": "stage7_atomic_batch_feedback_v1",
            "feedback_released": True,
            "batch_quarantined": False,
            "same_request_retry": False,
            "budget_consumed": 4,
            "measurement_request_sha256": request_sha,
            "request_binding_sha256": binding_sha,
            "cache_reveal_sha256": cache_sha,
            "exact_hit_count": 2,
            "miss_count": 2,
            "released_feedback_rows": feedback_rows,
        }
        finalization_sha = _payload_sha(finalization_payload)
        finalization = {
            **finalization_payload,
            "stage7_finalization_sha256": finalization_sha,
        }
        _write_json(round_dir / "stage7_finalization.json", finalization)
        released_payload = {
            "schema_version": "stage7_released_feedback_v1",
            "variant": variant,
            "seed": seed,
            "round_index": round_index,
            "measurement_request_sha256": request_sha,
            "request_binding_sha256": binding_sha,
            "feedback_input_path": str(
                (round_dir / "stage7_finalization.json").resolve()
            ),
            "feedback_input_sha256": hashlib.sha256(
                (json.dumps(finalization, indent=2, sort_keys=True) + "\n").encode()
            ).hexdigest(),
            "rows": feedback_rows,
        }
        previous_feedback_sha = _payload_sha(released_payload)
        released_feedback = {
            **released_payload,
            "released_feedback_sha256": previous_feedback_sha,
        }
        _write_json(round_dir / "released_feedback.json", released_feedback)
        all_feedback_rows.extend(feedback_rows)
        feedback_shas.append(previous_feedback_sha)

    terminal_payload = {
        "schema_version": "stage7_trajectory_terminal_v1",
        "status": "completed_at_T16",
        "variant": variant,
        "seed": seed,
        "task_id": "S7-PYR-TVM",
        "completed_atomic_rounds": 4,
        "selected_event_count": 16,
        "selected_row_ids": all_selected_ids,
        "released_feedback_sha256": _payload_sha(all_feedback_rows),
        "trajectory_contract_sha256": contract_sha,
    }
    _write_json(
        trajectory_dir / "trajectory_terminal.json",
        {
            **terminal_payload,
            "trajectory_terminal_sha256": _payload_sha(terminal_payload),
        },
    )
def _build_tree(root: Path, *, all_pass: bool = False) -> None:
    gold_path = root / "contracts" / "synthetic_gold176.json"
    gold_rows = [
        {
            "row_id": "gold-baseline",
            "task_id": "S7-PYR-TVM",
            "model": "pyramid",
            "hardware_id": "h800",
            "dispatch_key": "tvm_auto",
            "terminal_status": "measured_success_gold",
            "metric_source": "measured_artifacts",
            "latency_ms": 20.0,
            "energy_j": 20.0,
            "ap70": 0.5,
            "latency_artifact_sha256": "a" * 64,
            "energy_artifact_sha256": "b" * 64,
            "ap_artifact_sha256": "c" * 64,
        }
    ]
    gold_sha = _write_json(gold_path, {"rows": gold_rows})
    frozen_inputs_payload = {
        "schema_version": "stage7_frozen_inputs_v1",
        "inputs": {
            "gold176": {
                "path": str(gold_path.resolve()),
                "sha256": gold_sha,
            }
        },
    }
    frozen_input_sha = _payload_sha(frozen_inputs_payload)
    _write_json(
        root / "contracts" / "frozen_inputs.json",
        {
            **frozen_inputs_payload,
            "frozen_inputs_sha256": frozen_input_sha,
        },
    )
    _write_json(
        root / "contracts" / "raw_objective_reference.json",
        {
            "schema_version": "stage7_raw_objective_reference_v1",
            "values": {
                "latency_ms": 30.0,
                "energy_j": 30.0,
                "negative_ap70": 0.0,
            },
        },
    )
    pre_scan_sha = _write_json(
        root / "contracts" / "pre_scan_candidate_registry.json",
        {"schema_version": "stage7_pre_scan_candidate_registry_v1", "rows": []},
    )
    rule_sha = _write_json(
        root / "contracts" / "scanner_rule_manifest.json",
        {"schema_version": "stage7_scanner_rule_manifest_v1", "rules": []},
    )
    decision_path = root / "contracts" / "scanner_decision_by_candidate.csv"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text("row_id,decision\n", encoding="utf-8")
    decision_sha = hashlib.sha256(decision_path.read_bytes()).hexdigest()
    contract_shas = {
        "frozen_input_sha256": frozen_input_sha,
        "pre_scan_sha256": pre_scan_sha,
        "scan_pass_sha256": SHA["scan_pass"],
        "scanner_rule_sha256": rule_sha,
        "scanner_decision_sha256": decision_sha,
    }
    _write_json(
        root / "contracts" / "frozen_candidate_pools.json",
        {
            "schema_version": "stage7_frozen_candidate_pools_v1",
            **{key: value for key, value in contract_shas.items() if key != "frozen_input_sha256"},
            "variant_pool_sha256": {
                variant: (
                    contract_shas["pre_scan_sha256"]
                    if variant == "without_capability_scan"
                    else contract_shas["scan_pass_sha256"]
                )
                for variant in VARIANTS
            },
        },
    )
    _write_json(
        root / "audits" / "scanner_admission.json",
        {
            "schema_version": "stage7_scanner_admission_v1",
            "admission_passed": True,
            "status": (
                "scanner_ready_all_pass" if all_pass else "scanner_ready_discriminative"
            ),
            "scanner_rule_file_sha256": rule_sha,
            "scanner_decision_file_sha256": decision_sha,
        },
    )
    _write_json(
        root / "audits" / "single_variable_isolation.json",
        {
            "schema_version": "stage7_single_variable_isolation_v1",
            "audits": [
                {"variant": variant, "verdict": "pass"}
                for variant in VARIANTS
                if variant != "full"
            ],
        },
    )
    for variant in VARIANTS:
        for seed in SEEDS:
            _build_trajectory(root, variant, seed, contract_shas)


def test_fourteen_of_fifteen_trajectories_fail_closed_without_partial_means(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    missing = _trajectory_dir(input_root, VARIANTS[-1], SEEDS[-1])
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing.rmdir()

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is False
    assert result["missing_matrix"] == [
        {"variant": "without_capability_scan", "seed": 20260720, "reason": "missing_trajectory"}
    ]
    assert result["aggregate"]["variants"] == {}
    assert all(row["Status"] == "incomplete" for row in result["paper_rows"])
    assert (output_root / "aggregate" / "stage7_root_cause_summary.md").is_file()


def test_fifteen_event_trajectory_rejects_partial_mean(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    feedback_path = (
        _trajectory_dir(input_root, "full", SEEDS[0])
        / "round_03"
        / "released_feedback.json"
    )
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    feedback["rows"] = feedback["rows"][:-1]
    _rewrite_embedded_json(feedback_path, feedback, "released_feedback_sha256")

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is False
    assert any(
        item["variant"] == "full"
        and item["seed"] == SEEDS[0]
        and "selected_event_count" in item["reason"]
        for item in result["missing_matrix"]
    )
    assert result["aggregate"]["variants"] == {}


@pytest.mark.parametrize(
    ("relative_path", "field", "value", "reason_fragment"),
    [
        (
            "round_00/stage7_request_binding.json",
            "selected_row_ids_sha256",
            "f" * 64,
            "binding_selected_ids_sha",
        ),
        (
            "round_00/cache_reveal.json",
            "measurement_request_sha256",
            "f" * 64,
            "cache_request_sha",
        ),
        (
            "trajectory_contract.json",
            "scanner_decision_sha256",
            "f" * 64,
            "trajectory_scanner_decision_sha256",
        ),
        (
            "trajectory_contract.json",
            "scanner_rule_sha256",
            "f" * 64,
            "trajectory_scanner_rule_sha256",
        ),
    ],
)
def test_binding_cache_scanner_and_lease_sha_drift_fail_closed(
    tmp_path: Path,
    relative_path: str,
    field: str,
    value: str,
    reason_fragment: str,
) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    target = _trajectory_dir(input_root, "full", SEEDS[0]) / relative_path
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload[field] = value
    sha_fields = {
        "trajectory_contract.json": "trajectory_contract_sha256",
        "stage7_request_binding.json": "request_binding_sha256",
        "cache_reveal.json": "cache_reveal_sha256",
    }
    sha_field = sha_fields.get(target.name)
    if sha_field is None:
        _write_json(target, payload)
    else:
        _rewrite_embedded_json(target, payload, sha_field)

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert result["aggregate"]["variants"] == {}
    assert any(reason_fragment in item["reason"] for item in result["missing_matrix"])


def test_gpu_lease_request_sha_and_occupancy_cover_measurement_window(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    lease_path = input_root / "status" / "gpu_lease_audit.jsonl"
    events = [
        json.loads(line)
        for line in lease_path.read_text(encoding="utf-8").splitlines()
    ]
    first_lease = next(index for index, event in enumerate(events) if event["event"] == "batch_leased")
    events[first_lease]["request_sha256"] = "f" * 64
    lease_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert any("gpu_lease_request_sha" in item["reason"] for item in result["missing_matrix"])


def test_same_request_infrastructure_retry_adds_cost_without_consuming_event(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    lease_path = input_root / "status" / "gpu_lease_audit.jsonl"
    lease_events = [
        json.loads(line)
        for line in lease_path.read_text(encoding="utf-8").splitlines()
    ]
    first_lease_index = next(
        index
        for index, event in enumerate(lease_events)
        if event["event"] == "batch_leased" and event["wall_time"] > 30.0
    )
    successful_lease = lease_events[first_lease_index]
    retry_lease = {
        **successful_lease,
        "wall_time": successful_lease["wall_time"] - 30.0,
    }
    retry_event = {
        "schema_version": "stage7_gpu_lease_audit_v1",
        "event": "occupancy_drift_same_request_retry",
        "wall_time": successful_lease["wall_time"] - 10.0,
        "controller_id": successful_lease["controller_id"],
        "request_sha256": successful_lease["request_sha256"],
        "selected_event_budget_consumed": 0,
    }
    lease_events[first_lease_index:first_lease_index] = [retry_lease, retry_event]
    lease_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in lease_events),
        encoding="utf-8",
    )

    occupancy_path = input_root / "status" / "gpu_occupancy_snapshots.jsonl"
    occupancy_events = [
        json.loads(line)
        for line in occupancy_path.read_text(encoding="utf-8").splitlines()
    ]
    occupancy_events.append(
        {
            **occupancy_events[0],
            "wall_time": successful_lease["wall_time"] - 20.0,
        }
    )
    occupancy_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in occupancy_events),
        encoding="utf-8",
    )

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is True, result["missing_matrix"]
    trajectories = _read_csv(
        output_root / "aggregate" / "stage7_trajectories_raw.csv"
    )
    full_seed = next(
        row
        for row in trajectories
        if row["variant"] == "full" and row["seed"] == str(SEEDS[0])
    )
    assert float(full_seed["gpu_hours"]) == pytest.approx(
        0.4 + (4 * 20.0 / 3600.0)
    )
    events = _read_csv(output_root / "aggregate" / "stage7_events_raw.csv")
    assert sum(
        row["variant"] == "full" and row["seed"] == str(SEEDS[0])
        for row in events
    ) == 16


def test_unmatched_scheduler_window_fails_closed(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    request_sha = "f" * 64
    controller_id = "unexpected-controller"
    gpu_uuids = [f"GPU-{index}" for index in range(4)]
    _append_jsonl(
        input_root / "status" / "gpu_lease_audit.jsonl",
        {
            "event": "batch_leased",
            "wall_time": 999_900.0,
            "controller_id": controller_id,
            "request_sha256": request_sha,
            "gpu_uuids": gpu_uuids,
        },
    )
    _append_jsonl(
        input_root / "status" / "gpu_occupancy_snapshots.jsonl",
        {
            "event": "sample",
            "wall_time": 999_950.0,
            "gpus": [{"uuid": uuid} for uuid in gpu_uuids],
        },
    )
    _append_jsonl(
        input_root / "status" / "gpu_lease_audit.jsonl",
        {
            "event": "controller_finished",
            "wall_time": 999_990.0,
            "controller_id": controller_id,
            "returncode": 0,
        },
    )

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert any(
        item["reason"] == "unmatched_gpu_lease_request_sha"
        for item in result["missing_matrix"]
    )


def test_output_root_with_unmanaged_file_is_rejected_without_deletion(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    unmanaged = output_root / "stale.txt"
    unmanaged.parent.mkdir(parents=True)
    unmanaged.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(ValueError, match="output_root_contains_unmanaged_files"):
        finalizer.finalize(input_root, output_root)

    assert unmanaged.read_text(encoding="utf-8") == "keep me\n"


def test_exact_cache_hit_requires_source_entry_and_evidence_sha(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    target = (
        _trajectory_dir(input_root, "full", SEEDS[0])
        / "round_00"
        / "cache_reveal.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["rows"][0]["source_evidence_sha256"] = "not-a-sha"
    _rewrite_embedded_json(target, payload, "cache_reveal_sha256")

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert any(
        "cache_hit_source_evidence_sha" in item["reason"]
        for item in result["missing_matrix"]
    )


def test_a2_bundle_prediction_training_and_graph_sha_remain_frozen(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    target = (
        _trajectory_dir(input_root, "without_measured_feedback", SEEDS[0])
        / "round_03"
        / "a2_frozen_contract.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["bundle_sha256"] = "f" * 64
    _rewrite_embedded_json(target, payload, "frozen_payload_sha256")

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert any("a2_bundle_sha256_drift" in item["reason"] for item in result["missing_matrix"])


def test_a3_leakage_verdict_must_pass_for_every_round(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    target = (
        _trajectory_dir(input_root, "backend_blind", SEEDS[0])
        / "round_02"
        / "backend_blind_audit.json"
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["leakage_verdict"] = "fail"
    _write_json(target, payload)

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert any("a3_leakage_drift" in item["reason"] for item in result["missing_matrix"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _read_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return [
        dict(zip(headers, [cell.strip() for cell in line.strip("|").split("|")]))
        for line in lines[2:]
    ]


def test_all_pass_scanner_retains_non_discriminative_note(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root, all_pass=True)

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is True
    aggregate = json.loads(
        (output_root / "aggregate" / "stage7_aggregate.json").read_text(encoding="utf-8")
    )
    note = "capability_scan_non_discriminative_on_frozen_pool"
    assert aggregate["variants"]["without_capability_scan"]["terminal_note"] == note
    execution_rows = _read_csv(output_root / "paper" / "table_stage7_execution_cost.csv")
    assert execution_rows[-1]["Terminal note"] == note
    assert note in (
        output_root / "paper" / "table_stage7_execution_cost.md"
    ).read_text(encoding="utf-8")


def test_three_seed_statistics_are_descriptive_and_paired_only(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is True
    aggregate = json.loads(
        (output_root / "aggregate" / "stage7_aggregate.json").read_text(encoding="utf-8")
    )
    summary = aggregate["variants"]["without_surrogate"]["metrics"]["delta_hv_auc"]
    assert summary["seeds"] == {
        "20260718": pytest.approx(1827.0),
        "20260719": pytest.approx(1827.0),
        "20260720": pytest.approx(1827.0),
    }
    assert summary["sample_std"] == pytest.approx(0.0)
    assert summary["median"] == pytest.approx(1827.0)
    assert summary["min"] == pytest.approx(1827.0)
    assert summary["max"] == pytest.approx(1827.0)
    paired = summary["paired_delta_vs_full"]
    assert paired["seed_deltas"] == {
        str(seed): pytest.approx(-273.0) for seed in SEEDS
    }
    assert paired["direction_improved_count"] == 0
    assert paired["direction_total"] == 3
    assert set(aggregate["variants"]["full"]["precision_distribution_by_seed"]) == {
        str(seed) for seed in SEEDS
    }


def test_outputs_never_contain_inferential_statistics(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    finalizer.finalize(input_root, output_root)

    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    )

    for forbidden in (
        "p-value",
        "wilcoxon",
        "holm",
        "significance",
        "confidence interval",
        "stability claim",
    ):
        assert forbidden not in combined


def test_success_surrogate_metric_is_rejected(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    _build_tree(input_root)
    feedback_path = (
        _trajectory_dir(input_root, "full", SEEDS[0])
        / "round_00"
        / "released_feedback.json"
    )
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    feedback["rows"][0]["metric_source"] = "surrogate_prediction"
    _rewrite_round_feedback(
        _trajectory_dir(input_root, "full", SEEDS[0]),
        0,
        feedback["rows"],
    )

    result = finalizer.finalize(input_root, tmp_path / "output")

    assert result["paper_ready"] is False
    assert any("surrogate_metric_rejected" in item["reason"] for item in result["missing_matrix"])
    assert result["aggregate"]["variants"] == {}


def test_raw_audit_json_csv_and_markdown_values_are_consistent(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is True
    expected_paths = {
        "aggregate/stage7_events_raw.csv",
        "aggregate/stage7_trajectories_raw.csv",
        "aggregate/stage7_aggregate.json",
        "aggregate/stage7_completeness_audit.json",
        "aggregate/stage7_isolation_audit.json",
        "aggregate/stage7_root_cause_summary.md",
        "paper/table_stage7_component_ablation.csv",
        "paper/table_stage7_component_ablation.md",
        "paper/table_stage7_execution_cost.csv",
        "paper/table_stage7_execution_cost.md",
    }
    actual_paths = {
        str(path.relative_to(output_root))
        for path in output_root.rglob("*")
        if path.is_file()
    }
    assert actual_paths == expected_paths
    assert len(_read_csv(output_root / "aggregate" / "stage7_events_raw.csv")) == 240
    assert len(_read_csv(output_root / "aggregate" / "stage7_trajectories_raw.csv")) == 15
    completeness = json.loads(
        (output_root / "aggregate" / "stage7_completeness_audit.json").read_text(
            encoding="utf-8"
        )
    )
    isolation = json.loads(
        (output_root / "aggregate" / "stage7_isolation_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert completeness["paper_ready"] is True
    assert completeness["missing_matrix"] == []
    assert isolation["verdict"] == "pass"

    aggregate = json.loads(
        (output_root / "aggregate" / "stage7_aggregate.json").read_text(encoding="utf-8")
    )
    paper_csv = _read_csv(output_root / "paper" / "table_stage7_component_ablation.csv")
    paper_md = _read_markdown_table(
        output_root / "paper" / "table_stage7_component_ablation.md"
    )
    cost_csv = _read_csv(output_root / "paper" / "table_stage7_execution_cost.csv")
    cost_md = _read_markdown_table(output_root / "paper" / "table_stage7_execution_cost.md")
    assert paper_csv == paper_md == aggregate["paper_rows"]
    assert cost_csv == cost_md == aggregate["execution_cost_rows"]


def test_identical_deterministic_full_ids_report_limited_effective_independence(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)

    finalizer.finalize(input_root, output_root)

    aggregate = json.loads(
        (output_root / "aggregate" / "stage7_aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate["limited_effective_independence"] is True
    assert (
        aggregate["limited_effective_independence_note"]
        == "deterministic Full trajectories are identical; effective independence is limited"
    )
    assert "effective independence is limited" in (
        output_root / "aggregate" / "stage7_root_cause_summary.md"
    ).read_text(encoding="utf-8")


def test_true_candidate_failure_consumes_one_event_without_surrogate_metrics(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    trajectory_dir = _trajectory_dir(input_root, "full", SEEDS[0])
    feedback_path = trajectory_dir / "round_03" / "released_feedback.json"
    feedback = json.loads(feedback_path.read_text(encoding="utf-8"))
    row_id = feedback["rows"][2]["row_id"]
    feedback["rows"][2] = {
        "row_id": row_id,
        "q_mode": "fp16",
        "terminal_status": "build_failure",
        "failure_kind": "build_failure",
        "consumes_selected_event_budget": True,
        "cache_disposition": "miss",
    }
    feedback_sha = _rewrite_round_feedback(
        trajectory_dir,
        3,
        feedback["rows"],
    )
    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is True
    trajectories = _read_csv(output_root / "aggregate" / "stage7_trajectories_raw.csv")
    full_seed = next(
        row
        for row in trajectories
        if row["variant"] == "full" and row["seed"] == str(SEEDS[0])
    )
    assert float(full_seed["invalid_rate"]) == pytest.approx(1 / 16)
    assert float(full_seed["valid_yield"]) == pytest.approx(15 / 16)
    raw_events = _read_csv(output_root / "aggregate" / "stage7_events_raw.csv")
    failure = next(row for row in raw_events if row["row_id"] == row_id)
    assert failure["terminal_status"] == "build_failure"
    assert failure["latency_ms"] == ""
    assert failure["energy_j"] == ""
    assert failure["ap70"] == ""


def test_iso_ap_unmatched_arm_emits_null_and_reason_without_imputation(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    _build_tree(input_root)
    for seed in SEEDS:
        trajectory_dir = _trajectory_dir(input_root, "without_surrogate", seed)
        for round_index in range(4):
            released = json.loads(
                (
                    trajectory_dir
                    / f"round_{round_index:02d}"
                    / "released_feedback.json"
                ).read_text(encoding="utf-8")
            )
            rows = [{**row, "ap70": 0.2} for row in released["rows"]]
            _rewrite_round_feedback(trajectory_dir, round_index, rows)

    result = finalizer.finalize(input_root, output_root)

    assert result["paper_ready"] is True
    aggregate = json.loads(
        (output_root / "aggregate" / "stage7_aggregate.json").read_text(encoding="utf-8")
    )
    latency = aggregate["variants"]["without_surrogate"]["metrics"][
        "iso_ap_latency_ratio"
    ]
    energy = aggregate["variants"]["without_surrogate"]["metrics"][
        "iso_ap_energy_ratio"
    ]
    assert latency["status"] == energy["status"] == "not_comparable"
    assert latency["seeds"] == {str(seed): None for seed in SEEDS}
    assert set(latency["reason_by_seed"]) == {str(seed) for seed in SEEDS}
    paper_rows = _read_csv(output_root / "paper" / "table_stage7_component_ablation.csv")
    row = next(item for item in paper_rows if item["Variant"] == "without_surrogate")
    assert row["Iso-AP latency ratio ↓"] == "not comparable"
    assert row["Iso-AP energy ratio ↓"] == "not comparable"
