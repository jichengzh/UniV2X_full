#!/usr/bin/env python3
"""Release one atomic Stage7 feedback batch and generate the next round."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.single_target_search_v2 import finalize_atomic_batch
from framework.stage7 import search_policy_v1 as policy
from framework.stage7.online_component_ablation_v1 import (
    SEEDS,
    validate_result_row,
)
from scripts import stage7_prepare_online_ablation_v1 as prepare


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [copy.deepcopy(dict(row)) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return [copy.deepcopy(dict(row)) for row in payload["rows"]]
    raise ValueError("feedback JSON must be a list or object containing rows")


def _validate_embedded_sha(payload: Mapping[str, Any], field: str, label: str) -> str:
    copied = copy.deepcopy(dict(payload))
    recorded = copied.pop(field, None)
    if recorded != policy.canonical_sha256(copied):
        raise ValueError(f"{label} SHA drift")
    return str(recorded)


def _validate_trajectory_contract(
    output_root: Path,
    trajectory_dir: Path,
    variant: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trajectory = _load_json(trajectory_dir / "trajectory_contract.json")
    _validate_embedded_sha(
        trajectory,
        "trajectory_contract_sha256",
        "trajectory contract",
    )
    if (
        trajectory.get("variant") != variant
        or trajectory.get("seed") != seed
        or trajectory.get("task_id") != "S7-PYR-TVM"
        or trajectory.get("trajectory_dir") != str(trajectory_dir)
    ):
        raise ValueError("trajectory identity drift")
    pools = _load_json(output_root / "contracts/frozen_candidate_pools.json")
    frozen_inputs = _load_json(output_root / "contracts/frozen_inputs.json")
    expected = {
        "frozen_input_sha256": frozen_inputs["frozen_inputs_sha256"],
        "pre_scan_sha256": pools["pre_scan_sha256"],
        "scan_pass_sha256": pools["scan_pass_sha256"],
        "scanner_rule_sha256": pools["scanner_rule_sha256"],
        "scanner_decision_sha256": pools["scanner_decision_sha256"],
        "candidate_pool_sha256": pools["variant_pool_sha256"][variant],
    }
    for field, value in expected.items():
        if trajectory.get(field) != value:
            if field == "frozen_input_sha256":
                raise ValueError("trajectory frozen input SHA drift")
            raise ValueError("trajectory frozen scanner/pool SHA drift")
    return trajectory, pools


def _validate_round_request(
    round_dir: Path,
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = _load_json(round_dir / "measurement_request.json")
    binding = _load_json(round_dir / "stage7_request_binding.json")
    _validate_embedded_sha(
        request,
        "measurement_request_sha256",
        "measurement request",
    )
    policy.validate_request_binding(binding, request, trajectory)
    return request, binding


def _previous_feedback(
    trajectory: Mapping[str, Any],
    trajectory_dir: Path,
    completed_round_index: int,
    current_request: Mapping[str, Any],
    current_binding: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    release_shas: list[str] = []
    chain_heads: list[str] = []
    expected_previous_feedback_sha: str | None = None
    expected_previous_binding_sha: str | None = None
    expected_prior_chain_sha = trajectory["trajectory_contract_sha256"]
    for round_index in range(completed_round_index):
        round_dir = trajectory_dir / f"round_{round_index:02d}"
        path = round_dir / "released_feedback.json"
        if not path.is_file():
            raise ValueError("previous atomic feedback round is missing")
        request, binding = _validate_round_request(round_dir, trajectory)
        _validate_round_chain_link(
            round_dir=round_dir,
            trajectory=trajectory,
            request=request,
            binding=binding,
            expected_previous_feedback_sha=expected_previous_feedback_sha,
            expected_previous_binding_sha=expected_previous_binding_sha,
            expected_prior_chain_sha=expected_prior_chain_sha,
            completed_feedback_rows=4 * round_index,
        )
        payload = _load_json(path)
        released_sha = _validate_embedded_sha(
            payload,
            "released_feedback_sha256",
            "released feedback",
        )
        expected_identity = {
            "variant": trajectory["variant"],
            "seed": trajectory["seed"],
            "round_index": round_index,
            "measurement_request_sha256": request[
                "measurement_request_sha256"
            ],
            "request_binding_sha256": binding["request_binding_sha256"],
        }
        if any(payload.get(field) != value for field, value in expected_identity.items()):
            raise ValueError("released feedback request/binding identity drift")
        released_rows = _rows(payload)
        expected_ids = [
            str(row.get("row_id") or row.get("manifest_job_id"))
            for row in request["rows"]
        ]
        actual_ids = [
            str(row.get("row_id") or row.get("manifest_job_id"))
            for row in released_rows
        ]
        if actual_ids != expected_ids:
            raise ValueError("released feedback row order drift")
        atomic_audit = finalize_atomic_batch(request, released_rows)
        if atomic_audit.get("feedback_released") is not True:
            raise ValueError("released feedback is not a valid atomic batch")
        if any(row.get("training_source") != "online_feedback" for row in released_rows):
            raise ValueError("released feedback training-source drift")
        rows.extend(released_rows)
        state = _load_json(round_dir / "round_state.json")
        chain_head = state["current_chain_head_sha256"]
        release_shas.append(released_sha)
        chain_heads.append(chain_head)
        expected_previous_feedback_sha = released_sha
        expected_previous_binding_sha = binding["request_binding_sha256"]
        expected_prior_chain_sha = chain_head
    if len(rows) != 4 * completed_round_index:
        raise ValueError("previous feedback history is not four rows per round")
    current_round_dir = (
        trajectory_dir / f"round_{completed_round_index:02d}"
    )
    _validate_round_chain_link(
        round_dir=current_round_dir,
        trajectory=trajectory,
        request=current_request,
        binding=current_binding,
        expected_previous_feedback_sha=expected_previous_feedback_sha,
        expected_previous_binding_sha=expected_previous_binding_sha,
        expected_prior_chain_sha=expected_prior_chain_sha,
        completed_feedback_rows=len(rows),
    )
    current_state = _load_json(current_round_dir / "round_state.json")
    chain_heads.append(current_state["current_chain_head_sha256"])
    return rows, {
        "previous_released_feedback_sha256": expected_previous_feedback_sha,
        "previous_request_binding_sha256": expected_previous_binding_sha,
        "prior_chain_head_sha256": expected_prior_chain_sha,
        "released_feedback_sha256s": release_shas,
        "round_chain_heads": chain_heads,
    }


def _validate_round_chain_link(
    *,
    round_dir: Path,
    trajectory: Mapping[str, Any],
    request: Mapping[str, Any],
    binding: Mapping[str, Any],
    expected_previous_feedback_sha: str | None,
    expected_previous_binding_sha: str | None,
    expected_prior_chain_sha: str,
    completed_feedback_rows: int,
) -> None:
    expected_links = {
        "previous_released_feedback_sha256": expected_previous_feedback_sha,
        "previous_request_binding_sha256": expected_previous_binding_sha,
        "prior_chain_head_sha256": expected_prior_chain_sha,
    }
    if any(binding.get(field) != value for field, value in expected_links.items()):
        raise ValueError("request binding prior-chain drift")
    if trajectory.get("variant") == "without_measured_feedback":
        _validate_a2_chain_binding(trajectory, binding)
    state = _load_json(round_dir / "round_state.json")
    _validate_embedded_sha(state, "round_state_sha256", "round state")
    expected_state = policy.build_round_state(
        trajectory,
        binding,
        status="awaiting_terminal_feedback",
        completed_feedback_rows=completed_feedback_rows,
    )
    if state != expected_state:
        raise ValueError("round state chain drift")
    if state["measurement_request_sha256"] != request["measurement_request_sha256"]:
        raise ValueError("round state measurement request drift")


def _validate_a2_chain_binding(
    trajectory: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    trajectory_dir = Path(str(trajectory["trajectory_dir"]))
    round_zero = trajectory_dir / "round_00"
    frozen = _load_json(round_zero / "a2_frozen_contract.json")
    prediction_artifact = _load_json(
        round_zero / "a2_frozen_candidate_predictions.json"
    )
    frozen_inputs = _load_json(
        trajectory_dir.parents[2] / "contracts/frozen_inputs.json"
    )
    records = frozen_inputs["inputs"]
    initial_rows = prepare._rows(
        _load_json(Path(records["gold176"]["path"])),
        "rows",
    )
    initial_graphs = prepare._rows(
        _load_json(Path(records["graph_features"]["path"])),
        "graph_features",
    )
    artifact_rows = _rows(prediction_artifact)
    policy.validate_a2_frozen_contract(
        frozen,
        initial_rows=initial_rows,
        initial_graph_features=initial_graphs,
        expected_anchor=policy.derive_a2_anchor(initial_rows),
        artifact_predictions=artifact_rows,
    )
    expected = policy.a2_frozen_binding(frozen)
    if any(binding.get(field) != value for field, value in expected.items()):
        raise ValueError("A2 frozen contract chain drift")


def _selected_ids_through(
    trajectory_dir: Path,
    completed_round_index: int,
) -> list[str]:
    selected = []
    for round_index in range(completed_round_index + 1):
        request = _load_json(
            trajectory_dir
            / f"round_{round_index:02d}/measurement_request.json"
        )
        selected.extend(
            str(row.get("row_id") or row.get("manifest_job_id"))
            for row in request.get("rows") or []
        )
    if (
        len(selected) != 4 * (completed_round_index + 1)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError("trajectory selected identities are not unique four-row rounds")
    return selected


def _released_rows(
    feedback_rows: list[dict[str, Any]],
    atomic_audit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if atomic_audit.get("feedback_released") is not True:
        return []
    released = []
    for row in feedback_rows:
        validate_result_row(row)
        released.append({**copy.deepcopy(row), "training_source": "online_feedback"})
    return released


def _write_next_round(
    *,
    output_root: Path,
    trajectory_dir: Path,
    trajectory: Mapping[str, Any],
    pools: Mapping[str, Any],
    variant: str,
    seed: int,
    next_round_index: int,
    selected_ids: set[str],
    feedback_rows: list[dict[str, Any]],
    previous_released_feedback_sha256: str,
    previous_request_binding_sha256: str,
    prior_chain_head_sha256: str,
) -> dict[str, Any]:
    frozen_inputs = _load_json(output_root / "contracts/frozen_inputs.json")
    input_paths = frozen_inputs["inputs"]
    initial_rows = prepare._rows(
        _load_json(Path(input_paths["gold176"]["path"])),
        "rows",
    )
    initial_graphs = prepare._rows(
        _load_json(Path(input_paths["graph_features"]["path"])),
        "graph_features",
    )
    profiles_payload = _load_json(Path(input_paths["capability_profiles"]["path"]))
    raw_profile, profiles = prepare._tvm_profile(profiles_payload)
    closure = _load_json(Path(input_paths["stage4_closure"]["path"]))
    a2_frozen = None
    if variant == "without_measured_feedback":
        a2_frozen = _load_json(
            trajectory_dir / "round_00/a2_frozen_contract.json"
        )
    def select_next_round(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        return policy.select_stage7_round(
            variant=variant,
            seed=seed,
            round_index=next_round_index,
            task=policy.build_stage7_task(raw_profile),
            candidate_pool=candidates,
            initial_rows=initial_rows,
            initial_graph_features=initial_graphs,
            capability_profiles=profiles,
            closure=closure,
            selected_ids=selected_ids,
            feedback_rows=feedback_rows,
            a2_frozen=a2_frozen,
        )

    selection, cache_audit = prepare._select_with_cache_invariance(
        pools["variant_pools"][variant],
        select_next_round,
    )
    request = selection["measurement_request"]
    binding = policy.build_request_binding(
        trajectory,
        request,
        round_index=next_round_index,
        trajectory_dir=trajectory_dir,
        previous_released_feedback_sha256=(
            previous_released_feedback_sha256
        ),
        previous_request_binding_sha256=previous_request_binding_sha256,
        prior_chain_head_sha256=prior_chain_head_sha256,
        a2_frozen_contract=a2_frozen,
    )
    round_dir = trajectory_dir / f"round_{next_round_index:02d}"
    payloads = {
        "candidate_manifest.json": selection["candidate_manifest"],
        "acquisition.json": selection["acquisition"],
        "measurement_request.json": request,
        "stage7_request_binding.json": binding,
        "cache_selection_invariance_audit.json": cache_audit,
    }
    if "model_bundle_manifest" in selection:
        payloads["model_bundle_manifest.json"] = selection[
            "model_bundle_manifest"
        ]
    if "predicted_candidates" in selection:
        payloads["predicted_candidates.json"] = selection[
            "predicted_candidates"
        ]
    if "a2_frozen" in selection:
        payloads["a2_frozen_contract.json"] = selection["a2_frozen"]
    if "backend_blind_audit" in selection:
        payloads["backend_blind_audit.json"] = selection[
            "backend_blind_audit"
        ]
    state = policy.build_round_state(
        trajectory,
        binding,
        status="awaiting_terminal_feedback",
        completed_feedback_rows=len(feedback_rows),
    )
    payloads["round_state.json"] = state
    for filename, payload in payloads.items():
        prepare.write_frozen_json(round_dir / filename, payload)
    return state


def advance_trajectory(
    *,
    output_root: Path,
    variant: str,
    seed: int,
    completed_round_index: int,
    feedback_json: Path,
) -> dict[str, Any]:
    if not output_root.is_absolute() or not feedback_json.is_absolute():
        raise ValueError("output root and feedback JSON paths must be absolute")
    if variant not in prepare.VARIANTS or seed not in SEEDS:
        raise ValueError("unknown Stage7 variant or seed")
    if completed_round_index not in range(4):
        raise ValueError("completed round index must be 0, 1, 2, or 3")
    policy.validate_frozen_contracts(output_root)
    trajectory_dir = output_root / "variants" / variant / f"seed_{seed}"
    trajectory, pools = _validate_trajectory_contract(
        output_root, trajectory_dir, variant, seed
    )
    round_dir = trajectory_dir / f"round_{completed_round_index:02d}"
    request, binding = _validate_round_request(round_dir, trajectory)
    previous_feedback, chain = _previous_feedback(
        trajectory,
        trajectory_dir,
        completed_round_index,
        request,
        binding,
    )
    feedback_rows = _rows(_load_json(feedback_json))
    atomic_audit = finalize_atomic_batch(request, feedback_rows)
    if atomic_audit.get("feedback_released") is not True:
        retry = {
            "schema_version": "stage7_infrastructure_retry_v1",
            "status": "infrastructure_retry_required",
            "variant": variant,
            "seed": seed,
            "round_index": completed_round_index,
            "batch_budget_consumed": 0,
            "cumulative_budget_consumed": len(previous_feedback),
            "same_request_retry": True,
            "measurement_request_sha256": request[
                "measurement_request_sha256"
            ],
            "request_binding_sha256": binding["request_binding_sha256"],
            "feedback_input_sha256": _file_sha(feedback_json),
            "resume_row_ids": atomic_audit["resume_row_ids"],
        }
        prepare.write_frozen_json(
            round_dir / "infrastructure_retry.json", retry
        )
        return retry
    released = _released_rows(feedback_rows, atomic_audit)
    feedback_payload = {
        "schema_version": "stage7_released_feedback_v1",
        "variant": variant,
        "seed": seed,
        "round_index": completed_round_index,
        "measurement_request_sha256": request[
            "measurement_request_sha256"
        ],
        "request_binding_sha256": binding["request_binding_sha256"],
        "feedback_input_path": str(feedback_json),
        "feedback_input_sha256": _file_sha(feedback_json),
        "rows": released,
    }
    feedback_payload["released_feedback_sha256"] = policy.canonical_sha256(
        feedback_payload
    )
    audit_payload = {
        **copy.deepcopy(dict(atomic_audit)),
        "measurement_request_sha256": request[
            "measurement_request_sha256"
        ],
        "request_binding_sha256": binding["request_binding_sha256"],
    }
    prepare.write_frozen_json(
        round_dir / "released_feedback.json", feedback_payload
    )
    prepare.write_frozen_json(
        round_dir / "atomic_feedback_audit.json", audit_payload
    )
    all_feedback = [*previous_feedback, *released]
    selected_ids = _selected_ids_through(
        trajectory_dir, completed_round_index
    )
    if completed_round_index == 3:
        if len(all_feedback) != 16 or len(selected_ids) != 16:
            raise ValueError("trajectory terminal requires exactly sixteen feedback rows")
        release_shas = [
            *chain["released_feedback_sha256s"],
            feedback_payload["released_feedback_sha256"],
        ]
        chain_payload = {
            "trajectory_contract_sha256": trajectory[
                "trajectory_contract_sha256"
            ],
            "final_round_chain_head_sha256": chain[
                "round_chain_heads"
            ][-1],
            "final_released_feedback_sha256": feedback_payload[
                "released_feedback_sha256"
            ],
        }
        terminal = {
            "schema_version": "stage7_trajectory_terminal_v1",
            "status": "completed_at_T16",
            "variant": variant,
            "seed": seed,
            "task_id": "S7-PYR-TVM",
            "completed_atomic_rounds": 4,
            "selected_event_count": 16,
            "selected_row_ids": selected_ids,
            "released_feedback_sha256": policy.canonical_sha256(all_feedback),
            "round_chain_heads": chain["round_chain_heads"],
            "round_released_feedback_sha256s": release_shas,
            **chain_payload,
            "final_chain_head_sha256": policy.canonical_sha256(chain_payload),
            "trajectory_contract_sha256": trajectory[
                "trajectory_contract_sha256"
            ],
            "batch_budget_consumed": 4,
            "cumulative_budget_consumed": 16,
        }
        terminal["trajectory_terminal_sha256"] = policy.canonical_sha256(
            terminal
        )
        prepare.write_frozen_json(
            trajectory_dir / "trajectory_terminal.json", terminal
        )
        return terminal
    next_state = _write_next_round(
        output_root=output_root,
        trajectory_dir=trajectory_dir,
        trajectory=trajectory,
        pools=pools,
        variant=variant,
        seed=seed,
        next_round_index=completed_round_index + 1,
        selected_ids=set(selected_ids),
        feedback_rows=all_feedback,
        previous_released_feedback_sha256=feedback_payload[
            "released_feedback_sha256"
        ],
        previous_request_binding_sha256=binding["request_binding_sha256"],
        prior_chain_head_sha256=chain["round_chain_heads"][-1],
    )
    return {
        "schema_version": "stage7_advance_result_v1",
        "status": "next_round_generated",
        "variant": variant,
        "seed": seed,
        "completed_round_index": completed_round_index,
        "batch_budget_consumed": 4,
        "cumulative_budget_consumed": len(all_feedback),
        "next_round_index": completed_round_index + 1,
        "next_measurement_request_sha256": next_state[
            "measurement_request_sha256"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=prepare.VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument(
        "--completed-round-index",
        type=int,
        choices=range(4),
        required=True,
    )
    parser.add_argument("--feedback-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = advance_trajectory(
        output_root=args.output_root.resolve(),
        variant=args.variant,
        seed=args.seed,
        completed_round_index=args.completed_round_index,
        feedback_json=args.feedback_json.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
