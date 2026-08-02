from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import stage5_promote_actual_feedback_v3 as promote  # noqa: E402


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_payload(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return _sha_file(path)


def _batch(root: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    request_rows = []
    feedback_rows = []
    for index in range(4):
        width = [16 + index * 8, 32, 64]
        group_id = f"pyramid|{'x'.join(map(str, width))}"
        row_id = f"{group_id}|q=fp16|profile=h800-tvm"
        onnx_path = root / "onnx" / f"{index}.onnx"
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        onnx_path.write_bytes(f"onnx-{index}".encode())
        candidate_graph = {
            "group_id": group_id,
            "model": "pyramid",
            "width": width,
            "conv_count": 20.5,
            "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
        }
        request_row = {
            "schema_version": "stage5_candidate_row_v2",
            "task_id": "S5-PYR-TVM",
            "task_sha256": "t" * 64,
            "manifest_job_id": row_id,
            "row_id": row_id,
            "group_id": group_id,
            "model": "pyramid",
            "width": width,
            "q_mode": "fp16",
            "hardware_id": "h800",
            "dispatch_key": "tvm_auto",
            "capability_profile_id": "h800-tvm",
            "source_evidence_sha256": f"{index:064x}",
            "graph_features": candidate_graph,
        }
        source = {
            "schema_version": "stage5_source_materialization_evidence_v1",
            "status": "ready",
            "group_id": group_id,
            "model": "pyramid",
            "width": "x".join(map(str, width)),
            "source_plan_sha256": request_row["source_evidence_sha256"],
            "onnx_path": str(onnx_path),
            "onnx_sha256": _sha_file(onnx_path),
        }
        source_path = root / "source" / f"{index}.json"
        source_sha = _write(source_path, source)
        request_rows.append(request_row)
        feedback_rows.append(
            {
                **request_row,
                "training_source": "online_feedback",
                "terminal_status": "measured_success_gold",
                "latency_ms": 1.0,
                "energy_j": 0.5,
                "ap70": 0.4,
                "materialized_source_evidence_path": str(source_path),
                "materialized_source_evidence_sha256": source_sha,
            }
        )
    row_sha = {
        row["manifest_job_id"]: _sha_payload(row) for row in request_rows
    }
    for row in feedback_rows:
        row["measurement_request_row_sha256"] = row_sha[row["manifest_job_id"]]
    request = {
        "schema_version": "stage5_measurement_request_v2",
        "task_id": "S5-PYR-TVM",
        "round_index": 0,
        "batch_size": 4,
        "row_sha256": row_sha,
        "rows": request_rows,
    }
    request_path = root / "measurement_request.json"
    feedback_path = root / "feedback.json"
    _write(request_path, request)
    _write(feedback_path, feedback_rows)
    return request_path, feedback_path, request_rows


def _extract(path: Path) -> dict[str, object]:
    index = int(path.stem)
    return {
        "onnx_path": str(path),
        "onnx_sha256": _sha_file(path),
        "node_count": 60 + index,
        "conv_count": 21,
    }


class Stage5PromoteActualFeedbackV3Tests(unittest.TestCase):
    def test_promotes_materialized_features_without_mutating_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, feedback_path, request_rows = _batch(root)
            request_before = request_path.read_bytes()

            result = promote.promote_feedback_batch(
                request_path, feedback_path, extractor=_extract
            )

            self.assertEqual(request_path.read_bytes(), request_before)
            self.assertEqual(result["audit"]["promoted_row_count"], 4)
            self.assertEqual(result["audit"]["silent_surrogate_fallback_count"], 0)
            row = result["rows"][0]
            self.assertEqual(
                row["candidate_graph_features"], request_rows[0]["graph_features"]
            )
            self.assertEqual(
                row["graph_features"]["graph_feature_provenance"],
                "materialized_onnx_extracted_v1",
            )
            self.assertEqual(row["graph_features"]["conv_count"], 21)
            self.assertEqual(row["feedback_feature_contract"], "actual_feedback_v3")
            self.assertEqual(len(row["candidate_graph_features_sha256"]), 64)
            self.assertEqual(len(row["materialized_graph_features_sha256"]), 64)

    def test_rejects_feedback_candidate_snapshot_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, feedback_path, _ = _batch(root)
            feedback = json.loads(feedback_path.read_text())
            feedback[0]["graph_features"]["conv_count"] = 999
            _write(feedback_path, feedback)

            with self.assertRaisesRegex(ValueError, "feedback graph snapshot drift"):
                promote.promote_feedback_batch(
                    request_path, feedback_path, extractor=_extract
                )

    def test_rejects_extracted_onnx_sha_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path, feedback_path, _ = _batch(root)

            def wrong_extract(path: Path) -> dict[str, object]:
                return {**_extract(path), "onnx_sha256": "0" * 64}

            with self.assertRaisesRegex(ValueError, "extracted ONNX SHA mismatch"):
                promote.promote_feedback_batch(
                    request_path, feedback_path, extractor=wrong_extract
                )


if __name__ == "__main__":
    unittest.main()
