from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import stage7_advance_online_ablation_v1 as advance
from scripts import stage7_prepare_online_ablation_v1 as prepare
from framework.tests.test_stage7_prepare_online_ablation_v1 import (
    _inputs,
    _sha,
    _write_json,
)
from framework.tests.test_stage7_search_policy_v1 import _training_context


def _setup_trajectory(root: Path) -> Path:
    paths, expected = _inputs(root)
    registry = json.loads(paths.source_registry.read_text())
    for index, width in enumerate(
        (
            [48, 96, 192],
            [56, 112, 224],
            [64, 128, 256],
            [16, 48, 96],
            [24, 64, 128],
            [32, 80, 160],
        ),
        start=5,
    ):
        group_id = "pyramid|" + "x".join(map(str, width))
        registry["groups"].append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "source_status": "ready",
                "source_evidence_sha256": f"{index:064x}",
                "source_contract": {
                    "checkpoint_path": f"/frozen/{group_id}.pth",
                    "onnx_path": f"/frozen/{group_id}.onnx",
                },
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": width,
                    "conv_count": 20 + index,
                },
            }
        )
    _write_json(paths.source_registry, registry)
    expected["source_registry"] = hashlib.sha256(
        paths.source_registry.read_bytes()
    ).hexdigest()
    output = root / "stage7"
    prepare.freeze_contracts(
        output_root=output,
        inputs=paths,
        expected_sha256=expected,
        enforce_formal_counts=False,
    )
    terminal = root / "scanner-terminal.json"
    _write_json(terminal, {"rows": []})
    prepare.audit_scanner(
        output_root=output,
        terminal_evidence_json=terminal,
    )
    prepare.initialize_trajectories(
        output_root=output,
        variant="without_surrogate",
        seed=20260718,
    )
    return output


def _setup_a2_trajectory(root: Path) -> Path:
    paths, expected = _inputs(root)
    rows, graphs, profiles, closure = _training_context()
    _write_json(paths.gold176, {"rows": rows})
    _write_json(paths.graph_features, {"graph_features": graphs})
    _write_json(paths.capability_profiles, {"capability_profiles": profiles})
    _write_json(paths.stage4_closure, closure)
    registry = json.loads(paths.source_registry.read_text())
    for index, width in enumerate(
        ([48, 96, 192], [56, 112, 224], [64, 128, 256], [32, 80, 160]),
        start=5,
    ):
        group_id = "pyramid|" + "x".join(map(str, width))
        registry["groups"].append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                "source_status": "ready",
                "source_evidence_sha256": f"{index:064x}",
                "source_contract": {
                    "checkpoint_path": f"/frozen/{group_id}.pth",
                    "onnx_path": f"/frozen/{group_id}.onnx",
                },
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": list(width),
                    "conv_count": 20 + index,
                },
            }
        )
    _write_json(paths.source_registry, registry)
    for name, path in (
        ("gold176", paths.gold176),
        ("graph_features", paths.graph_features),
        ("capability_profiles", paths.capability_profiles),
        ("source_registry", paths.source_registry),
    ):
        expected[name] = _sha(path)
    output = root / "stage7"
    prepare.freeze_contracts(
        output_root=output,
        inputs=paths,
        expected_sha256=expected,
        enforce_formal_counts=False,
    )
    terminal = root / "scanner-terminal.json"
    _write_json(terminal, {"rows": []})
    prepare.audit_scanner(
        output_root=output,
        terminal_evidence_json=terminal,
    )
    prepare.initialize_trajectories(
        output_root=output,
        variant="without_measured_feedback",
        seed=20260718,
    )
    return output


def _feedback_for_request(request: dict, *, status: str = "measured_success_gold") -> list[dict]:
    rows = []
    for index, request_row in enumerate(request["rows"]):
        row = {
            **request_row,
            "measurement_request_row_sha256": request["row_sha256"][
                request_row["row_id"]
            ],
            "terminal_status": status,
        }
        if status == "measured_success_gold":
            row.update(
                {
                    "latency_ms": 1.0 + index,
                    "energy_j": 0.1 + 0.01 * index,
                    "ap30": 0.9,
                    "ap50": 0.8,
                    "ap70": 0.7,
                    "latency_artifact_sha256": "a" * 64,
                    "energy_artifact_sha256": "b" * 64,
                    "ap_artifact_sha256": "c" * 64,
                }
            )
        else:
            row["failure_reason"] = "synthetic infrastructure interruption"
        rows.append(row)
    return rows


class Stage7AdvanceOnlineAblationV1Tests(unittest.TestCase):
    def test_a2_round_zero_binds_all_frozen_artifact_shas_into_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_a2_trajectory(root)
            round_dir = (
                output
                / "variants/without_measured_feedback/seed_20260718/round_00"
            )
            frozen = json.loads(
                (round_dir / "a2_frozen_contract.json").read_text()
            )
            binding = json.loads(
                (round_dir / "stage7_request_binding.json").read_text()
            )
            state = json.loads((round_dir / "round_state.json").read_text())
            expected = {
                "a2_frozen_contract_sha256": frozen["frozen_payload_sha256"],
                "a2_bundle_sha256": frozen["bundle_sha256"],
                "a2_prediction_view_sha256": frozen["prediction_view_sha256"],
                "a2_training_view_sha256": frozen["training_view_sha256"],
                "a2_graph_feature_view_sha256": frozen[
                    "graph_feature_view_sha256"
                ],
                "a2_anchor_sha256": frozen["anchor_sha256"],
            }
            for field, value in expected.items():
                self.assertEqual(binding[field], value)
                self.assertEqual(state[field], value)

    def test_a2_self_consistent_payload_rewrite_cannot_escape_round_zero_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_a2_trajectory(root)
            round_dir = (
                output
                / "variants/without_measured_feedback/seed_20260718/round_00"
            )
            frozen_path = round_dir / "a2_frozen_contract.json"
            frozen = json.loads(frozen_path.read_text())
            frozen["candidate_predictions"][0]["predictions"][
                "latency_ms"
            ] += 999.0
            frozen["prediction_view_sha256"] = (
                advance.policy.canonical_sha256(
                    frozen["candidate_predictions"]
                )
            )
            frozen.pop("frozen_payload_sha256")
            frozen["frozen_payload_sha256"] = (
                advance.policy.canonical_sha256(frozen)
            )
            _write_json(frozen_path, frozen)
            _write_json(
                round_dir / "a2_frozen_candidate_predictions.json",
                {"rows": frozen["candidate_predictions"]},
            )
            request = json.loads(
                (round_dir / "measurement_request.json").read_text()
            )
            feedback = root / "feedback.json"
            _write_json(feedback, {"rows": _feedback_for_request(request)})

            with self.assertRaisesRegex(
                ValueError,
                "A2 frozen contract chain drift",
            ):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_measured_feedback",
                    seed=20260718,
                    completed_round_index=0,
                    feedback_json=feedback,
                )

    def test_a2_advance_carries_round_zero_frozen_shas_into_next_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_a2_trajectory(root)
            trajectory_dir = (
                output
                / "variants/without_measured_feedback/seed_20260718"
            )
            round_zero = trajectory_dir / "round_00"
            request = json.loads(
                (round_zero / "measurement_request.json").read_text()
            )
            feedback = root / "feedback.json"
            _write_json(feedback, {"rows": _feedback_for_request(request)})

            result = advance.advance_trajectory(
                output_root=output,
                variant="without_measured_feedback",
                seed=20260718,
                completed_round_index=0,
                feedback_json=feedback,
            )

            self.assertEqual(result["status"], "next_round_generated")
            binding_zero = json.loads(
                (round_zero / "stage7_request_binding.json").read_text()
            )
            binding_one = json.loads(
                (
                    trajectory_dir
                    / "round_01/stage7_request_binding.json"
                ).read_text()
            )
            for field in (
                "a2_frozen_contract_sha256",
                "a2_bundle_sha256",
                "a2_prediction_view_sha256",
                "a2_training_view_sha256",
                "a2_graph_feature_view_sha256",
                "a2_anchor_sha256",
            ):
                self.assertEqual(binding_one[field], binding_zero[field])

    def test_trajectory_pins_frozen_input_manifest_including_source_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            frozen_path = output / "contracts/frozen_inputs.json"
            frozen = json.loads(frozen_path.read_text())
            original_gold = Path(frozen["inputs"]["gold176"]["path"])
            relocated_gold = root / "relocated-gold.json"
            relocated_gold.write_bytes(original_gold.read_bytes())
            frozen["inputs"]["gold176"]["path"] = str(relocated_gold)
            frozen.pop("frozen_inputs_sha256")
            frozen["frozen_inputs_sha256"] = advance.policy.canonical_sha256(
                frozen
            )
            _write_json(frozen_path, frozen)
            trajectory_dir = (
                output / "variants/without_surrogate/seed_20260718"
            )
            request = json.loads(
                (trajectory_dir / "round_00/measurement_request.json").read_text()
            )
            feedback = root / "feedback.json"
            _write_json(feedback, {"rows": _feedback_for_request(request)})

            with self.assertRaisesRegex(
                ValueError,
                "freeze root input manifest drift",
            ):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=0,
                    feedback_json=feedback,
                )

    def test_prior_released_feedback_tamper_blocks_training_and_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            trajectory_dir = (
                output / "variants/without_surrogate/seed_20260718"
            )
            request_0 = json.loads(
                (trajectory_dir / "round_00/measurement_request.json").read_text()
            )
            feedback_0 = root / "feedback-0.json"
            _write_json(feedback_0, {"rows": _feedback_for_request(request_0)})
            advance.advance_trajectory(
                output_root=output,
                variant="without_surrogate",
                seed=20260718,
                completed_round_index=0,
                feedback_json=feedback_0,
            )

            released_path = trajectory_dir / "round_00/released_feedback.json"
            released = json.loads(released_path.read_text())
            released["rows"][0]["latency_ms"] = 999.0
            _write_json(released_path, released)
            request_1 = json.loads(
                (trajectory_dir / "round_01/measurement_request.json").read_text()
            )
            feedback_1 = root / "feedback-1.json"
            _write_json(feedback_1, {"rows": _feedback_for_request(request_1)})

            with self.assertRaisesRegex(ValueError, "released feedback SHA drift"):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=1,
                    feedback_json=feedback_1,
                )
            self.assertFalse((trajectory_dir / "round_02").exists())

    def test_prior_feedback_recomputed_sha_still_requires_request_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            trajectory_dir = (
                output / "variants/without_surrogate/seed_20260718"
            )
            request_0 = json.loads(
                (trajectory_dir / "round_00/measurement_request.json").read_text()
            )
            feedback_0 = root / "feedback-0.json"
            _write_json(feedback_0, {"rows": _feedback_for_request(request_0)})
            advance.advance_trajectory(
                output_root=output,
                variant="without_surrogate",
                seed=20260718,
                completed_round_index=0,
                feedback_json=feedback_0,
            )

            released_path = trajectory_dir / "round_00/released_feedback.json"
            released = json.loads(released_path.read_text())
            released["rows"].reverse()
            released.pop("released_feedback_sha256")
            released["released_feedback_sha256"] = (
                advance.policy.canonical_sha256(released)
            )
            _write_json(released_path, released)
            request_1 = json.loads(
                (trajectory_dir / "round_01/measurement_request.json").read_text()
            )
            feedback_1 = root / "feedback-1.json"
            _write_json(feedback_1, {"rows": _feedback_for_request(request_1)})

            with self.assertRaisesRegex(ValueError, "released feedback row order"):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=1,
                    feedback_json=feedback_1,
                )
            self.assertFalse((trajectory_dir / "round_02").exists())

    def test_prior_round_state_chain_tamper_blocks_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            trajectory_dir = (
                output / "variants/without_surrogate/seed_20260718"
            )
            request_0 = json.loads(
                (trajectory_dir / "round_00/measurement_request.json").read_text()
            )
            feedback_0 = root / "feedback-0.json"
            _write_json(feedback_0, {"rows": _feedback_for_request(request_0)})
            advance.advance_trajectory(
                output_root=output,
                variant="without_surrogate",
                seed=20260718,
                completed_round_index=0,
                feedback_json=feedback_0,
            )

            state_path = trajectory_dir / "round_00/round_state.json"
            state = json.loads(state_path.read_text())
            state["current_chain_head_sha256"] = "0" * 64
            _write_json(state_path, state)
            request_1 = json.loads(
                (trajectory_dir / "round_01/measurement_request.json").read_text()
            )
            feedback_1 = root / "feedback-1.json"
            _write_json(feedback_1, {"rows": _feedback_for_request(request_1)})

            with self.assertRaisesRegex(ValueError, "round state SHA drift"):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=1,
                    feedback_json=feedback_1,
                )
            self.assertFalse((trajectory_dir / "round_02").exists())

    def test_partial_or_request_sha_drift_feedback_cannot_advance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            request = json.loads(
                (
                    output
                    / "variants/without_surrogate/seed_20260718/round_00/"
                    "measurement_request.json"
                ).read_text()
            )
            feedback = root / "feedback.json"
            _write_json(feedback, {"rows": _feedback_for_request(request)[:3]})

            with self.assertRaisesRegex(ValueError, "atomic batch"):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=0,
                    feedback_json=feedback,
                )

            drifted = _feedback_for_request(request)
            drifted[-1]["measurement_request_row_sha256"] = "0" * 64
            _write_json(root / "drifted.json", {"rows": drifted})
            with self.assertRaisesRegex(ValueError, "feedback identity SHA drift"):
                advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=0,
                    feedback_json=root / "drifted.json",
                )
            self.assertFalse(
                (
                    output
                    / "variants/without_surrogate/seed_20260718/round_01"
                ).exists()
            )

    def test_infrastructure_batch_preserves_same_request_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            round_dir = (
                output / "variants/without_surrogate/seed_20260718/round_00"
            )
            request = json.loads((round_dir / "measurement_request.json").read_text())
            feedback = root / "feedback.json"
            _write_json(
                feedback,
                {"rows": _feedback_for_request(request, status="public_runner_failure")},
            )

            result = advance.advance_trajectory(
                output_root=output,
                variant="without_surrogate",
                seed=20260718,
                completed_round_index=0,
                feedback_json=feedback,
            )

            self.assertEqual(result["status"], "infrastructure_retry_required")
            self.assertEqual(result["batch_budget_consumed"], 0)
            self.assertEqual(result["cumulative_budget_consumed"], 0)
            self.assertEqual(
                result["measurement_request_sha256"],
                request["measurement_request_sha256"],
            )
            self.assertFalse((round_dir.parent / "round_01").exists())
            retry = json.loads((round_dir / "infrastructure_retry.json").read_text())
            self.assertEqual(
                retry["measurement_request_sha256"],
                request["measurement_request_sha256"],
            )

    def test_four_atomic_rounds_end_with_sixteen_unique_selected_events(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = _setup_trajectory(root)
            trajectory_dir = (
                output / "variants/without_surrogate/seed_20260718"
            )
            for round_index in range(4):
                request = json.loads(
                    (
                        trajectory_dir
                        / f"round_{round_index:02d}/measurement_request.json"
                    ).read_text()
                )
                feedback = root / f"feedback-{round_index}.json"
                _write_json(
                    feedback,
                    {"rows": _feedback_for_request(request)},
                )
                result = advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=round_index,
                    feedback_json=feedback,
                )
                repeated = advance.advance_trajectory(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                    completed_round_index=round_index,
                    feedback_json=feedback,
                )
                self.assertEqual(repeated, result)

            terminal = json.loads(
                (trajectory_dir / "trajectory_terminal.json").read_text()
            )
            self.assertEqual(terminal["status"], "completed_at_T16")
            self.assertEqual(terminal["selected_event_count"], 16)
            self.assertEqual(terminal["completed_atomic_rounds"], 4)
            self.assertEqual(len(set(terminal["selected_row_ids"])), 16)
            self.assertEqual(len(terminal["round_chain_heads"]), 4)
            self.assertEqual(
                len(terminal["round_released_feedback_sha256s"]),
                4,
            )
            self.assertEqual(
                terminal["final_round_chain_head_sha256"],
                terminal["round_chain_heads"][-1],
            )
            self.assertRegex(terminal["final_chain_head_sha256"], r"^[0-9a-f]{64}$")
            self.assertFalse((trajectory_dir / "round_04").exists())


if __name__ == "__main__":
    unittest.main()
