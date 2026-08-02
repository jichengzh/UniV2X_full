from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage5.independent_validation_v1 import build_validation_request


def _row(index: int) -> dict:
    width = [16 + index, 32, 64]
    row_id = f"pyramid|{'x'.join(map(str, width))}|q=fp16|profile=tvm"
    return {
        "row_id": row_id,
        "manifest_job_id": row_id,
        "task_id": "S5-PYR-TVM",
        "task_sha256": "a" * 64,
        "group_id": f"pyramid|{'x'.join(map(str, width))}",
        "model": "pyramid",
        "width": width,
        "q_mode": "fp16",
        "genome": [*width, "fp16"],
        "hardware_id": "h800",
        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
        "capability_digest": "b" * 64,
        "dispatch_key": "tvm_auto",
        "source_evidence_sha256": "c" * 64,
        "source_contract": {"source_done_marker": f"/tmp/source-{index}.done"},
    }


class Stage5IndependentValidationV1Tests(unittest.TestCase):
    def test_builds_frozen_request_from_exact_closure_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "S5-PYR-TVM"
            rows = [_row(index) for index in range(6)]
            for round_index in range(2):
                path = task_root / f"round_{round_index:02d}/measurement_request.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"rows": rows[round_index * 3 : round_index * 3 + 3]}))
            selected = [rows[4]["row_id"], rows[1]["row_id"]]

            request = build_validation_request(
                task_root,
                {
                    "task_id": "S5-PYR-TVM",
                    "task_sha256": "a" * 64,
                    "independent_validation_ids": selected,
                },
            )

        self.assertEqual(
            request["schema_version"], "stage5_independent_validation_request_v1"
        )
        self.assertEqual(request["batch_size"], 2)
        self.assertEqual([row["row_id"] for row in request["rows"]], selected)
        self.assertEqual(
            request["measurement_request_sha256"],
            hashlib.sha256(
                json.dumps(
                    {
                        key: value
                        for key, value in request.items()
                        if key != "measurement_request_sha256"
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )

    def test_rejects_selection_missing_from_formal_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "S5-PYR-TVM"
            path = task_root / "round_00/measurement_request.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"rows": [_row(0)]}))

            with self.assertRaisesRegex(ValueError, "not found in formal or cold-start requests"):
                build_validation_request(
                    task_root,
                    {
                        "task_id": "S5-PYR-TVM",
                        "task_sha256": "a" * 64,
                        "independent_validation_ids": ["missing"],
                    },
                )

    def test_accepts_frozen_coldstart_row_missing_from_online_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "S5-PYR-TVM"
            online = _row(0)
            coldstart = _row(1)
            path = task_root / "round_00/measurement_request.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"rows": [online]}))

            request = build_validation_request(
                task_root,
                {
                    "task_id": "S5-PYR-TVM",
                    "task_sha256": "a" * 64,
                    "independent_validation_ids": [coldstart["row_id"], online["row_id"]],
                },
                fallback_rows={coldstart["row_id"]: coldstart},
            )

        self.assertEqual(
            [row["row_id"] for row in request["rows"]],
            [coldstart["row_id"], online["row_id"]],
        )

    def test_rejects_fallback_mapping_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "fallback row identity mismatch"):
                build_validation_request(
                    Path(temporary),
                    {
                        "task_id": "S5-PYR-TVM",
                        "task_sha256": "a" * 64,
                        "independent_validation_ids": [_row(1)["row_id"]],
                    },
                    fallback_rows={"wrong": _row(1)},
                )

    def test_uses_manifest_identity_when_row_id_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "S5-PYR-TVM"
            row = _row(0)
            row.pop("row_id")
            path = task_root / "round_00/measurement_request.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"rows": [row]}))

            request = build_validation_request(
                task_root,
                {
                    "task_id": "S5-PYR-TVM",
                    "task_sha256": "a" * 64,
                    "independent_validation_ids": [row["manifest_job_id"]],
                },
            )

        self.assertEqual(
            set(request["row_sha256"]), {row["manifest_job_id"]}
        )


if __name__ == "__main__":
    unittest.main()
