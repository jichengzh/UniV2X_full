from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from scripts.stage6_formal_runner_v1 import build_runner_state


class Stage6FormalRunnerV1Tests(unittest.TestCase):
    def test_initial_state_covers_all_six_arms_and_two_backends(self) -> None:
        plan = {
            "schema_version": "stage6_formal_execution_plan_v1",
            "passed": True,
            "plan_sha256": "a" * 64,
            "arms": {
                "original_default": {},
                "compression_only": {},
                "schedule_only": {},
                "compress_then_tune": {},
                "tune_then_compress": {},
                "joint_shcosearch": {},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            state = build_runner_state(plan, Path(tmp))

        self.assertEqual(state["formal_stage6_started"], True)
        self.assertEqual(len(state["slots"]), 11)
        self.assertEqual(state["slots"]["original_default|pytorch_eager"], "running")
        self.assertEqual(state["slots"]["joint_shcosearch|tvm"], "evidence_bound_pending_validation")
        self.assertFalse(state["paper_table_ready"])

    def test_original_slot_requires_sha_bound_baseline(self) -> None:
        plan = {
            "schema_version": "stage6_formal_execution_plan_v1",
            "passed": True,
            "plan_sha256": "a" * 64,
            "arms": {arm: {} for arm in (
                "original_default", "compression_only", "schedule_only",
                "compress_then_tune", "tune_then_compress", "joint_shcosearch",
            )},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "native_fp32_ap").mkdir()
            (root / "native_fp32_ap/full_ap_eval_report.json").write_text("{}")
            self.assertEqual(
                build_runner_state(plan, root)["slots"]["original_default|pytorch_eager"],
                "running",
            )
            (root / "stage6_native_fp32_baseline_with_ap_v2.json").write_text(
                json.dumps({"ap_evidence_status": "full_h800_backend_execution_bound"})
            )
            self.assertEqual(
                build_runner_state(plan, root)["slots"]["original_default|pytorch_eager"],
                "complete",
            )
