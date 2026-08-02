from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import stage5_backfill_actual_graph_features_v1 as backfill  # noqa: E402


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _sha_file(path)


def _formal_tree(root: Path) -> Path:
    formal = root / "formal"
    for task_index, task_id in enumerate(backfill.TASKS):
        model = "pyramid" if "PYR" in task_id else "codriving"
        model_index = 0 if model == "pyramid" else 1
        dispatch = "tvm_auto" if "TVM" in task_id else "trt_engine"
        for round_index in range(4):
            request_rows = []
            feedback_rows = []
            for offset in range(4):
                index = round_index * 4 + offset
                width = [16 + index, 32 + index, 64 + index]
                group_id = f"{model}|{'x'.join(map(str, width))}"
                q_mode = "fp16" if index % 2 == 0 else "int8"
                profile = f"h800-{dispatch}-v3"
                row_id = f"{group_id}|q={q_mode}|profile={profile}"
                onnx_path = root / "onnx" / f"{model_index}_{index}.onnx"
                onnx_path.parent.mkdir(parents=True, exist_ok=True)
                if not onnx_path.exists():
                    onnx_path.write_bytes(f"onnx-{model_index}-{index}".encode())
                source_evidence = {
                    "schema_version": "stage5_source_materialization_evidence_v1",
                    "status": "ready",
                    "group_id": group_id,
                    "model": model,
                    "width": "x".join(map(str, width)),
                    "source_plan_sha256": "",
                    "onnx_path": str(onnx_path),
                    "onnx_sha256": _sha_file(onnx_path),
                }
                source_path = root / "source" / f"{model_index}_{index}.json"
                surrogate = {
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "conv_count": 20.5,
                    "conv_flops": 1000.0 + index,
                    "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
                }
                request_row = {
                    "task_id": task_id,
                    "task_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                    "row_id": row_id,
                    "manifest_job_id": row_id,
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "q_mode": q_mode,
                    "genome": [*width, q_mode],
                    "hardware_id": "h800",
                    "dispatch_key": dispatch,
                    "capability_profile_id": profile,
                    "capability_digest": hashlib.sha256(profile.encode()).hexdigest(),
                    "materialization_kind": (
                        "pyramid_checkpoint_export"
                        if model == "pyramid"
                        else "codriving_prepare_train_export"
                    ),
                    "source_contract": {"onnx_path": str(onnx_path)},
                    "graph_features": surrogate,
                }
                request_row["source_evidence_sha256"] = backfill.source_plan_sha(
                    request_row
                )
                source_evidence["source_plan_sha256"] = request_row[
                    "source_evidence_sha256"
                ]
                source_sha = _write(source_path, source_evidence)
                request_rows.append(request_row)
                feedback_rows.append(
                    {
                        **request_row,
                        "terminal_status": "measured_success_gold",
                        "latency_ms": 1.0,
                        "energy_j": 0.5,
                        "ap30": 0.7,
                        "ap50": 0.6,
                        "ap70": 0.4,
                        "measurement_request_row_sha256": _sha_payload(request_row),
                        "materialized_source_evidence_path": str(source_path),
                        "materialized_source_evidence_sha256": source_sha,
                    }
                )
            request = {
                "schema_version": "stage5_measurement_request_v2",
                "task_id": task_id,
                "task_sha256": request_rows[0]["task_sha256"],
                "round_index": round_index,
                "batch_size": 4,
                "sample_budget": 16,
                "required_metrics": [
                    "latency_ms",
                    "energy_j",
                    "ap30",
                    "ap50",
                    "ap70",
                ],
                "atomic_feedback": True,
                "real_h800_measurement_required": True,
                "row_sha256": {
                    row["manifest_job_id"]: _sha_payload(row) for row in request_rows
                },
                "rows": request_rows,
            }
            request["measurement_request_sha256"] = _sha_payload(request)
            round_root = formal / task_id / f"round_{round_index:02d}"
            _write(round_root / "measurement_request.json", request)
            _write(
                round_root / "final/stage5_feedback_v2_final.json", feedback_rows
            )
    return formal


def _actual(path: Path) -> dict[str, object]:
    index = int(path.stem.split("_")[-1])
    return {
        "onnx_path": str(path),
        "onnx_sha256": _sha_file(path),
        "node_count": 60,
        "conv_count": 21,
        "conv_flops": 1100.0 + index,
    }


class Stage5BackfillActualGraphFeaturesV1Tests(unittest.TestCase):
    def test_backfills_64_rows_without_mutating_surrogate_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)

            with patch.object(backfill, "extract_graph_features", side_effect=_actual):
                result = backfill.build_backfill(formal)

            self.assertEqual(result["actual"]["group_count"], 32)
            self.assertEqual(result["feedback_view"]["row_count"], 64)
            self.assertEqual(result["drift"]["row_count"], 64)
            row = result["feedback_view"]["rows"][0]
            self.assertEqual(
                row["candidate_graph_features"]["graph_feature_provenance"],
                "coldstart_width_conditioned_surrogate_v1",
            )
            self.assertEqual(
                row["graph_features"]["graph_feature_provenance"],
                "materialized_onnx_extracted_v1",
            )
            self.assertEqual(row["graph_features"]["conv_count"], 21)
            self.assertEqual(len(row["candidate_graph_features_sha256"]), 64)
            self.assertEqual(len(row["materialized_graph_features_sha256"]), 64)
            self.assertTrue(row["derived_view_only"])
            self.assertEqual(row["graph_feature_backfill_round_index"], 0)
            self.assertEqual(len(row["derived_view_row_sha256"]), 64)
            self.assertGreater(result["drift"]["summary"]["max_relative_error"], 0.0)
            self.assertEqual(len(result["actual"]["group_audit"]), 32)
            self.assertTrue(any(item["row_count"] > 1 for item in result["actual"]["group_audit"]))

    def test_rejects_feedback_graph_snapshot_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            feedback_path = (
                formal
                / backfill.TASKS[0]
                / "round_00/final/stage5_feedback_v2_final.json"
            )
            feedback = json.loads(feedback_path.read_text())
            feedback[0]["graph_features"]["conv_count"] = 999
            _write(feedback_path, feedback)

            with self.assertRaisesRegex(ValueError, "feedback graph snapshot drift"):
                backfill.build_backfill(formal)

    def test_rejects_feedback_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            path = formal / backfill.TASKS[0] / "round_00/final/stage5_feedback_v2_final.json"
            feedback = json.loads(path.read_text())
            feedback[0]["model"] = "codriving"
            _write(path, feedback)

            with self.assertRaisesRegex(ValueError, "feedback request identity drift"):
                backfill.build_backfill(formal)

    def test_rejects_materialized_onnx_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            feedback_path = (
                formal
                / backfill.TASKS[0]
                / "round_00/final/stage5_feedback_v2_final.json"
            )
            feedback = json.loads(feedback_path.read_text())
            source = Path(feedback[0]["materialized_source_evidence_path"])
            payload = json.loads(source.read_text())
            Path(payload["onnx_path"]).write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "ONNX SHA mismatch"):
                backfill.build_backfill(formal)

    def test_rejects_non_terminal_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            path = formal / backfill.TASKS[0] / "round_00/final/stage5_feedback_v2_final.json"
            feedback = json.loads(path.read_text())
            feedback[0]["terminal_status"] = "pending"
            _write(path, feedback)

            with self.assertRaisesRegex(ValueError, "non-terminal status"):
                backfill.build_backfill(formal)

    def test_accepts_true_numerical_feasibility_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            path = formal / backfill.TASKS[0] / "round_00/final/stage5_feedback_v2_final.json"
            feedback = json.loads(path.read_text())
            feedback[0]["terminal_status"] = "numerical_feasibility_failure"
            feedback[0]["failure_reason"] = "real numerical gate failure"
            for field in ("latency_ms", "energy_j", "ap30", "ap50", "ap70"):
                feedback[0].pop(field)
            _write(path, feedback)

            with patch.object(backfill, "extract_graph_features", side_effect=_actual):
                result = backfill.build_backfill(formal)
            self.assertEqual(result["feedback_view"]["row_count"], 64)

    def test_rejects_public_runner_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            path = formal / backfill.TASKS[0] / "round_00/final/stage5_feedback_v2_final.json"
            feedback = json.loads(path.read_text())
            feedback[0]["terminal_status"] = "public_runner_failure"
            _write(path, feedback)

            with self.assertRaisesRegex(ValueError, "feedback batch is not terminal"):
                backfill.build_backfill(formal)

    def test_rejects_measurement_request_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            path = formal / backfill.TASKS[0] / "round_00/measurement_request.json"
            request = json.loads(path.read_text())
            request["measurement_request_sha256"] = "0" * 64
            _write(path, request)

            with self.assertRaisesRegex(ValueError, "measurement request SHA mismatch"):
                backfill.build_backfill(formal)

    def test_rejects_directory_task_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            path = formal / backfill.TASKS[0] / "round_00/measurement_request.json"
            request = json.loads(path.read_text())
            request["task_id"] = backfill.TASKS[1]
            request["measurement_request_sha256"] = _sha_payload(
                {key: value for key, value in request.items() if key != "measurement_request_sha256"}
            )
            _write(path, request)

            with self.assertRaisesRegex(ValueError, "directory task/round identity drift"):
                backfill.build_backfill(formal)

    def test_rejects_source_model_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            feedback_path = (
                formal
                / backfill.TASKS[0]
                / "round_00/final/stage5_feedback_v2_final.json"
            )
            feedback = json.loads(feedback_path.read_text())
            source_path = Path(feedback[0]["materialized_source_evidence_path"])
            source = json.loads(source_path.read_text())
            source["model"] = "codriving"
            feedback[0]["materialized_source_evidence_sha256"] = _write(
                source_path, source
            )
            _write(feedback_path, feedback)

            with self.assertRaisesRegex(ValueError, "materialized source contract mismatch"):
                backfill.build_backfill(formal)

    def test_write_outputs_preserves_historical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = _formal_tree(root)
            historical = formal / backfill.TASKS[0] / "round_00/measurement_request.json"
            before = historical.read_bytes()
            with patch.object(backfill, "extract_graph_features", side_effect=_actual):
                result = backfill.build_backfill(formal)
            backfill.write_outputs(result, root / "derived")
            self.assertEqual(historical.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
