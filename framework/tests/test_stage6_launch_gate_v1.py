from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.stage6_prepare_launch_gate_v1 import (
    ARM_IDS,
    audit_formal_runner_plan,
    audit_joint_closure,
    build_gate,
    classify_baseline_evidence,
)


class Stage6LaunchGateV1Tests(unittest.TestCase):
    @staticmethod
    def _baseline_payload(**overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "stage6_native_fp32_baseline_v1",
            "hardware": "NVIDIA H800",
            "input_shape": [2, 64, 128, 256],
            "scope": "pyramid_multiscale_backbone",
            "precision": "fp32",
            "backend": "pytorch_eager",
            "width": [64, 128, 256],
            "backend_tuning": False,
            "full_network_claim": False,
            "latency_p50_ms": 3.2,
            "energy_j": 1.1,
            "checkpoint_sha256": "a" * 64,
            "output_shapes": [[2, 64, 128, 256]],
            "independent_repeat_count": 3,
        }
        return {**payload, **overrides}

    def test_joint_closure_requires_closed_actual_feedback(self) -> None:
        payload = {
            "task_id": "S5-PYR-TVM",
            "closure": True,
            "sample_budget": 16,
            "batch_size": 4,
            "round_count": 4,
            "online_count": 16,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "closure.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = audit_joint_closure(path)

        self.assertTrue(audit["passed"])

    def test_4090_native_baseline_is_invalid_scope_for_h800(self) -> None:
        payload = {
            "schema_version": "stage6_native_fp32_baseline_v1",
            "gpu": "NVIDIA GeForce RTX 4090",
            "input_shape": [2, 64, 128, 256],
            "scope": "pyramid_multiscale_backbone",
            "precision": "fp32",
            "backend": "pytorch_eager",
            "width": [64, 128, 256],
            "backend_tuning": False,
            "full_network_claim": False,
            "latency_p50_ms": 1.0,
            "energy_j": 1.0,
            "checkpoint_sha256": "a" * 64,
            "output_shapes": [[2, 64, 128, 256]],
            "independent_repeat_count": 3,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = classify_baseline_evidence(path)

        self.assertEqual(audit["classification"], "invalid_scope")
        self.assertIn("hardware", audit["mismatches"])

    def test_native_baseline_rejects_wrong_width_or_tuned_evidence(self) -> None:
        payload = {
            "schema_version": "stage6_native_fp32_baseline_v1",
            "hardware": "NVIDIA H800",
            "input_shape": [2, 64, 128, 256],
            "scope": "pyramid_multiscale_backbone",
            "precision": "fp32",
            "backend": "pytorch_eager",
            "width": [48, 96, 192],
            "backend_tuning": True,
            "full_network_claim": False,
            "latency_p50_ms": 1.0,
            "energy_j": 1.0,
            "checkpoint_sha256": "a" * 64,
            "output_shapes": [[2, 48, 128, 256]],
            "independent_repeat_count": 3,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            audit = classify_baseline_evidence(path)

        self.assertEqual(audit["classification"], "invalid_scope")
        self.assertIn("width", audit["mismatches"])
        self.assertIn("backend_tuning", audit["mismatches"])

    def test_latency_energy_only_baseline_is_not_paper_table_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "baseline.json"
            path.write_text(json.dumps(self._baseline_payload()), encoding="utf-8")
            audit = classify_baseline_evidence(path)

        self.assertEqual(audit["classification"], "reusable")
        self.assertFalse(audit["paper_table_eligible"])
        self.assertEqual(
            audit["paper_table_mismatches"],
            ["ap70", "ap_report_path", "ap_report_sha256"],
        )

    def test_full_ap_requires_existing_report_with_matching_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ap_report = root / "ap.json"
            ap_report.write_text('{"ap70": 0.42}', encoding="utf-8")
            payload = self._baseline_payload(
                ap70=0.42,
                ap_report_path=str(ap_report),
                ap_report_sha256="0" * 64,
            )
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            bad_audit = classify_baseline_evidence(baseline)

            import hashlib

            payload["ap_report_sha256"] = hashlib.sha256(ap_report.read_bytes()).hexdigest()
            baseline.write_text(json.dumps(payload), encoding="utf-8")
            good_audit = classify_baseline_evidence(baseline)

        self.assertFalse(bad_audit["paper_table_eligible"])
        self.assertIn("ap_report_sha256", bad_audit["paper_table_mismatches"])
        self.assertTrue(good_audit["paper_table_eligible"])
        self.assertEqual(good_audit["paper_table_mismatches"], [])

    def test_formal_runner_plan_binds_an_existing_runner_by_sha(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "formal_runner.py"
            runner.write_text("print('stage6')\n", encoding="utf-8")

            import hashlib

            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "stage6_formal_runner_plan_v1",
                        "experiment_id": "stage6-pyramid-h800-six-arm-v1",
                        "target_model": "pyramid",
                        "hardware_id": "h800",
                        "backends": ["tvm", "trt"],
                        "arm_ids": list(ARM_IDS),
                        "formal_runner_path": str(runner),
                        "formal_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
                        "output_root": str(root / "formal"),
                        "budget_contract_bound": True,
                    }
                ),
                encoding="utf-8",
            )

            audit = audit_formal_runner_plan(plan)

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["failures"], [])

    def test_protocol_ready_does_not_require_formal_execution_but_launch_does(self) -> None:
        manifest = {
            "schema_version": "stage6_pyramid_arm_manifest_v2",
            "experiment_id": "stage6-pyramid-h800-six-arm-v1",
            "arms": [{"arm_id": arm_id} for arm_id in ARM_IDS[:-1]]
            + [
                {
                    "arm_id": ARM_IDS[-1],
                    "joint_evidence": {"tvm": "unused", "trt": "unused"},
                    "joint_summary_evidence": "unused",
                    "joint_online_rows_evidence": "unused",
                }
            ],
        }
        passed = {"passed": True, "failures": []}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps(self._baseline_payload()), encoding="utf-8")
            (root / "stage6_runner_smoke_v1.json").write_text(
                json.dumps(passed), encoding="utf-8"
            )
            runner = root / "formal_runner.py"
            runner.write_text("print('stage6')\n", encoding="utf-8")
            import hashlib

            plan = root / "formal_runner_plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "schema_version": "stage6_formal_runner_plan_v1",
                        "experiment_id": "stage6-pyramid-h800-six-arm-v1",
                        "target_model": "pyramid",
                        "hardware_id": "h800",
                        "backends": ["tvm", "trt"],
                        "arm_ids": list(ARM_IDS),
                        "formal_runner_path": str(runner),
                        "formal_runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
                        "output_root": str(root / "formal"),
                        "budget_contract_bound": True,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch(
                    "scripts.stage6_prepare_launch_gate_v1.build_stage6_manifest",
                    return_value=manifest,
                ),
                patch(
                    "scripts.stage6_prepare_launch_gate_v1.validate_stage6_manifest",
                    return_value=passed,
                ),
                patch(
                    "scripts.stage6_prepare_launch_gate_v1.audit_joint_closure",
                    return_value=passed,
                ),
                patch(
                    "scripts.stage6_prepare_launch_gate_v1.audit_joint_actual_feedback",
                    return_value=passed,
                ),
            ):
                report = build_gate(root, [baseline], formal_runner_plan=None)
                launchable_report = build_gate(root, [baseline], formal_runner_plan=plan)

        self.assertTrue(report["protocol_ready"])
        self.assertFalse(report["formal_execution_ready"])
        self.assertFalse(report["paper_table_ready"])
        self.assertFalse(report["launch_allowed"])
        self.assertIn("formal_runner_plan", report["formal_execution_gates"])
        self.assertTrue(launchable_report["protocol_ready"])
        self.assertTrue(launchable_report["formal_execution_ready"])
        self.assertTrue(launchable_report["launch_allowed"])
        self.assertFalse(launchable_report["paper_table_ready"])


if __name__ == "__main__":
    unittest.main()
