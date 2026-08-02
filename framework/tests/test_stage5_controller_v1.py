from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = REPO_ROOT / "scripts/stage5_round0_h800_controller_v1.sh"


def _sha(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Stage5ControllerV1Tests(unittest.TestCase):
    def test_resume_requires_both_previous_and_next_request_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "round_00").mkdir()
            (root / "round_01").mkdir()
            previous_request = {"schema_version": "request", "rows": [{"id": 0}]}
            next_request = {"schema_version": "request", "rows": [{"id": 1}]}
            feedback_audit = {"schema_version": "feedback_audit", "rows": 8}
            candidate_manifest = {"schema_version": "candidates", "rows": []}
            acquisition = {"schema_version": "acquisition", "selected": []}
            model_bundle = {"bundle_config_sha256": "b" * 64}
            state = {
                "previous_measurement_request_sha256": _sha(previous_request),
                "measurement_request_sha256": _sha(next_request),
                "feedback_audit_sha256": _sha(feedback_audit),
                "candidate_manifest_sha256": _sha(candidate_manifest),
                "selection_sha256": _sha(acquisition),
                "model_bundle_config_sha256": "b" * 64,
                "model_bundle_manifest_sha256": _sha(model_bundle),
                "previous_round_feedback_verified": True,
            }
            (root / "round_00/measurement_request.json").write_text(
                json.dumps(previous_request), encoding="utf-8"
            )
            (root / "round_01/measurement_request.json").write_text(
                json.dumps(next_request), encoding="utf-8"
            )
            (root / "round_01/round_state.json").write_text(
                json.dumps(state), encoding="utf-8"
            )
            for name, payload in (
                ("feedback_audit.json", feedback_audit),
                ("candidate_manifest.json", candidate_manifest),
                ("acquisition.json", acquisition),
                ("model_bundle_manifest.json", model_bundle),
                ("predicted_candidates.json", {"rows": []}),
                ("feedback_rows.json", []),
            ):
                (root / f"round_01/{name}").write_text(json.dumps(payload), encoding="utf-8")
            env = {
                **os.environ,
                "REPO": str(REPO_ROOT),
                "ROOT": str(root),
                "PY": sys.executable,
            }
            valid = subprocess.run(
                ["bash", str(CONTROLLER)], env=env, capture_output=True, text=True
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)

            (root / "round_01/measurement_request.json").write_text(
                json.dumps({**next_request, "rows": [{"id": 2}]}), encoding="utf-8"
            )
            drifted = subprocess.run(
                ["bash", str(CONTROLLER)], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(drifted.returncode, 0)

            (root / "round_01/measurement_request.json").write_text(
                json.dumps(next_request), encoding="utf-8"
            )
            (root / "round_01/model_bundle_manifest.json").write_text(
                json.dumps({**model_bundle, "heads": {"latency": "drifted"}}),
                encoding="utf-8",
            )
            bundle_drift = subprocess.run(
                ["bash", str(CONTROLLER)], env=env, capture_output=True, text=True
            )
            self.assertNotEqual(bundle_drift.returncode, 0)


if __name__ == "__main__":
    unittest.main()
