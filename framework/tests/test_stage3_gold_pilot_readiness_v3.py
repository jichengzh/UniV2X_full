from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.canonical_search_v3 import build_capability_profile


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "framework" / "stage3" / "gold_pilot_readiness_v3.py"
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage3_gold_pilot_readiness_v3.py"
PLANNER_PATH = REPO_ROOT / "scripts" / "stage2_gold_coldstart96_plan_v3.py"

SPEC = importlib.util.spec_from_file_location("stage3_gold_pilot_readiness_v3", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
PLANNER_SPEC = importlib.util.spec_from_file_location("stage2_gold_coldstart96_plan_v3", PLANNER_PATH)
assert PLANNER_SPEC and PLANNER_SPEC.loader
PLANNER = importlib.util.module_from_spec(PLANNER_SPEC)


def _load_module() -> None:
    SPEC.loader.exec_module(MODULE)


def _load_planner() -> None:
    PLANNER_SPEC.loader.exec_module(PLANNER)


def _profiles() -> list[dict]:
    return [
        build_capability_profile(
            capability_profile_id="h800-tvm-auto-v3",
            hardware_target="h800",
            compiler_fingerprint="a" * 64,
            dispatch_key="tvm_auto",
            features={"supports_int8_tensorcore": 1.0},
        ),
        build_capability_profile(
            capability_profile_id="h800-trt-v3",
            hardware_target="h800",
            compiler_fingerprint="c" * 64,
            dispatch_key="trt_engine",
            features={"supports_int8_tensorcore": 1.0},
        ),
    ]


def _plan_manifest() -> dict:
    return copy.deepcopy(PLANNER.build_gold_plan(_profiles())["manifest"])


def _full_ready_manifest() -> dict:
    manifest = _plan_manifest()
    compiler_by_profile = {
        profile["capability_profile_id"]: profile["compiler_fingerprint"]
        for profile in manifest["capability_profiles"]
    }
    for row in manifest["jobs"]:
        if row["group_id"] not in set(manifest["pilot_group_ids"]):
            continue
        row["staging_contract"] = {
            "source_checkpoint_sha256": "1" * 64,
            "source_onnx_sha256": "2" * 64,
            "profile_compiler_fingerprint": compiler_by_profile[row["capability_profile_id"]],
            "runner_kind": row["dispatch_key"],
            "ap_eval_protocol_sha256": "3" * 64,
            "latency_energy_protocol_sha256": "4" * 64,
        }
        if row["q_mode"] == "int8":
            row["staging_contract"]["calibration_sha256"] = "5" * 64
    return manifest


class Stage3GoldPilotReadinessV3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_module()
        _load_planner()

    def test_plan_manifest_without_staging_contract_emits_blocked_pilot_readiness(self) -> None:
        manifest = _plan_manifest()

        readiness = MODULE.build_gold_pilot_readiness(manifest)

        self.assertEqual(readiness["schema_version"], "stage3_gold_pilot_readiness_v3")
        self.assertEqual(readiness["status"], "blocked")
        self.assertEqual(readiness["pilot_row_count"], 8)
        self.assertEqual(len(readiness["rows"]), 8)
        self.assertTrue(readiness["blocked_gap_counts"])
        self.assertEqual(readiness["pilot_group_ids"], ["pyramid|32x32x128", "codriving|32x64x128"])
        gap_codes = {gap["code"] for row in readiness["rows"] for gap in row["gaps"]}
        self.assertIn("missing_source_checkpoint_sha256", gap_codes)
        self.assertIn("missing_source_onnx_sha256", gap_codes)
        self.assertIn("missing_ap_eval_protocol_sha256", gap_codes)
        self.assertIn("missing_latency_energy_protocol_sha256", gap_codes)
        int8_rows = [row for row in readiness["rows"] if row["q_mode"] == "int8"]
        self.assertEqual(len(int8_rows), 4)
        self.assertTrue(all(any(gap["code"] == "missing_calibration_sha256" for gap in row["gaps"]) for row in int8_rows))

    def test_complete_staging_contract_is_ready(self) -> None:
        readiness = MODULE.build_gold_pilot_readiness(_full_ready_manifest())

        self.assertEqual(readiness["status"], "ready")
        self.assertEqual(readiness["blocked_gap_counts"], {})
        self.assertTrue(all(not row["gaps"] for row in readiness["rows"]))
        self.assertEqual(
            {row["group_id"] for row in readiness["rows"]},
            {"pyramid|32x32x128", "codriving|32x64x128"},
        )

    def test_forbids_hand_rewrite_mixed_and_byoc_trt_claiming_tvm_auto(self) -> None:
        for source_kind in ("hand-rewrite", "mixed", "BYOC-TRT"):
            with self.subTest(source_kind=source_kind):
                manifest = _full_ready_manifest()
                row = next(
                    item
                    for item in manifest["jobs"]
                    if item["group_id"] == "pyramid|32x32x128"
                    and item["q_mode"] == "fp16"
                    and item["capability_profile_id"] == "h800-tvm-auto-v3"
                )
                job_id = row["job_id"]
                row["staging_contract"] = copy.deepcopy(row["staging_contract"])
                row["staging_contract"]["runner_kind"] = "tvm_auto"
                row["staging_contract"]["source_realization"] = source_kind

                readiness = MODULE.build_gold_pilot_readiness(manifest)
                readiness_row = next(item for item in readiness["rows"] if item["job_id"] == job_id)

                self.assertEqual(readiness["status"], "blocked")
                self.assertTrue(
                    any(
                        gap["code"] == "forbidden_tvm_auto_source_realization"
                        for gap in readiness_row["gaps"]
                    )
                )

    def test_cli_writes_content_addressed_artifact_for_temp_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            output_dir = Path(tmp) / "out"
            manifest_path.write_text(json.dumps(_full_ready_manifest()), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--manifest-json",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            readiness_path = Path(payload["readiness_json"])
            self.assertTrue(readiness_path.exists())
            self.assertRegex(readiness_path.name, r"gold_pilot_readiness_v3-[0-9a-f]{64}\.json")
            readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
            self.assertEqual(readiness["status"], "ready")
            self.assertEqual(readiness["pilot_group_ids"], ["pyramid|32x32x128", "codriving|32x64x128"])


if __name__ == "__main__":
    unittest.main()
