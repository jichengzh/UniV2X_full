from __future__ import annotations

import importlib.util
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from framework.stage2.canonical_search_v3 import build_capability_profile


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stage2_gold_coldstart96_plan_v3.py"
SPEC = importlib.util.spec_from_file_location("stage2_gold_coldstart96_plan_v3", SCRIPT)
assert SPEC and SPEC.loader
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


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
            compiler_fingerprint="b" * 64,
            dispatch_key="trt_engine",
            features={"supports_int8_tensorcore": 1.0},
        ),
    ]


class Stage2GoldColdstart96PlanV3Test(unittest.TestCase):
    def test_builds_24_complete_groups_and_96_rows(self) -> None:
        plan = PLANNER.build_gold_plan(_profiles())

        self.assertEqual(len(plan["manifest"]["jobs"]), 96)
        self.assertEqual(len(plan["groups"]), 24)
        self.assertEqual({group["row_count"] for group in plan["groups"]}, {4})
        self.assertEqual(len(plan["split"]["train_group_ids"]), 20)
        self.assertEqual(len(plan["split"]["holdout_group_ids"]), 4)
        self.assertEqual(len(plan["split"]["train_job_ids"]), 80)
        self.assertEqual(len(plan["split"]["holdout_job_ids"]), 16)
        expected_combinations = {
            (q_mode, profile)
            for q_mode in ("fp16", "int8")
            for profile in ("h800-tvm-auto-v3", "h800-trt-v3")
        }
        for rows in plan["groups_by_id"].values():
            self.assertEqual(
                {(row["q_mode"], row["capability_profile_id"]) for row in rows},
                expected_combinations,
            )

    def test_assigns_model_specific_width_catalogs_strata_and_pilot_groups(self) -> None:
        plan = PLANNER.build_gold_plan(_profiles())
        groups = {group["group_id"]: group for group in plan["groups"]}

        expected_pyramid = {
            "pyramid|16x16x16": "diagonal_base",
            "pyramid|16x16x64": "diagonal_base",
            "pyramid|32x32x64": "diagonal_base",
            "pyramid|32x32x128": "diagonal_base",
            "pyramid|16x64x128": "alignment_group",
            "pyramid|16x128x256": "alignment_group",
            "pyramid|32x64x256": "alignment_group",
            "pyramid|48x64x128": "alignment_group",
            "pyramid|24x48x192": "off_diagonal",
            "pyramid|24x56x128": "off_diagonal",
            "pyramid|32x32x256": "off_diagonal",
            "pyramid|40x80x160": "off_diagonal",
        }
        expected_codriving = {
            "codriving|32x32x128": "diagonal_base",
            "codriving|48x96x192": "diagonal_base",
            "codriving|64x96x192": "diagonal_base",
            "codriving|64x128x256": "diagonal_base",
            "codriving|16x32x64": "alignment_group",
            "codriving|32x64x128": "alignment_group",
            "codriving|40x64x128": "alignment_group",
            "codriving|48x64x128": "alignment_group",
            "codriving|24x32x96": "off_diagonal",
            "codriving|24x64x128": "off_diagonal",
            "codriving|56x112x224": "off_diagonal",
            "codriving|64x64x128": "off_diagonal",
        }

        self.assertEqual(
            {
                group_id: groups[group_id]["width_stratum"]
                for group_id in sorted(expected_pyramid)
            },
            expected_pyramid,
        )
        self.assertEqual(
            {
                group_id: groups[group_id]["width_stratum"]
                for group_id in sorted(expected_codriving)
            },
            expected_codriving,
        )
        self.assertEqual(
            plan["manifest"]["pilot_group_ids"],
            ["pyramid|32x32x128", "codriving|32x64x128"],
        )
        self.assertEqual(plan["contract"]["pilot_group_ids"], plan["manifest"]["pilot_group_ids"])

    def test_manifest_has_no_mixed_or_backend_token_in_strategy(self) -> None:
        plan = PLANNER.build_gold_plan(_profiles())

        for job in plan["manifest"]["jobs"]:
            self.assertEqual(job["mixed_policy_id"], "none")
            self.assertIn(job["q_mode"], {"fp16", "int8"})
            self.assertEqual(job["strategy_id"], f"q={job['q_mode']}")
            self.assertEqual(job["genome"], [*job["width"], job["q_mode"]])
            self.assertNotRegex("|".join(map(str, job["genome"])), r"tvm|trt|backend|compiler")
            self.assertNotRegex(job["strategy_id"], r"tvm|trt|backend|compiler")

    def test_ap_sharing_contract_binds_equivalence_evidence(self) -> None:
        contract = PLANNER.build_gold_plan(_profiles())["contract"]

        self.assertEqual(
            set(contract["ap_sharing_required_equivalence_fields"]),
            {
                "checkpoint_sha256",
                "source_onnx_sha256",
                "compiled_artifact_sha256_by_profile",
                "calibration_sha256",
                "dataset_manifest_sha256",
                "evaluation_protocol_sha256",
                "input_manifest_sha256",
                "tolerance_spec_sha256",
                "output_comparison_sha256",
            },
        )

    def test_locked_holdout_is_structural_and_group_isolated(self) -> None:
        plan = PLANNER.build_gold_plan(_profiles())
        holdout = set(plan["split"]["holdout_group_ids"])
        expected = {
            "pyramid|16x16x64",
            "pyramid|40x80x160",
            "codriving|24x32x96",
            "codriving|64x128x256",
        }

        self.assertEqual(holdout, expected)
        train = set(plan["split"]["train_group_ids"])
        self.assertFalse(train & holdout)
        self.assertEqual(train | holdout, set(plan["groups_by_id"]))
        self.assertEqual(len([group_id for group_id in holdout if group_id.startswith("pyramid|")]), 2)
        self.assertEqual(len([group_id for group_id in holdout if group_id.startswith("codriving|")]), 2)

    def test_writes_content_addressed_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = PLANNER.write_gold_plan(PLANNER.build_gold_plan(_profiles()), tmp)

            self.assertEqual(set(paths), {"manifest", "split", "contract"})
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertRegex(path.name, r"-[0-9a-f]{64}\.json$")
            self.assertTrue(PLANNER.audit_written_plan(paths))

    def test_audit_rejects_split_job_ids_not_matching_manifest(self) -> None:
        plan = copy.deepcopy(PLANNER.build_gold_plan(_profiles()))
        plan["split"]["train_job_ids"][0] = plan["split"]["holdout_job_ids"][0]
        with tempfile.TemporaryDirectory() as tmp:
            paths = PLANNER.write_gold_plan(plan, tmp)

            with self.assertRaisesRegex(ValueError, "split job ids do not match manifest"):
                PLANNER.audit_written_plan(paths)

    def test_cli_runs_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles = Path(tmp) / "profiles.json"
            output = Path(tmp) / "output"
            profiles.write_text(json.dumps(_profiles()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--capability-profiles",
                    str(profiles),
                    "--output-dir",
                    str(output),
                ],
                cwd=SCRIPT.parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(len(list(output.glob("*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
