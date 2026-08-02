from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "stage35_gold32_supplement_plan_v1.py"
GOLD96_MANIFEST = REPO_ROOT / "results" / "gold_coldstart96_v3_final_20260711" / (
    "gold_coldstart96_manifest_v3-d1b0495125d60962c135c3c682ccfeb106fe20babfdc408dc0215ad7518d1305.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("stage35_gold32_supplement_plan_v1", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage35Gold32SupplementPlanV1Tests(unittest.TestCase):
    def test_builds_eight_cross_model_groups_and_complete_four_arm_product(self) -> None:
        module = _load_module()
        source = json.loads(GOLD96_MANIFEST.read_text(encoding="utf-8"))
        plan = module.build_supplement_plan(source["capability_profiles"])
        manifest = plan["manifest"]

        self.assertEqual(manifest["schema_version"], "stage35_gold32_supplement_manifest_v1")
        self.assertEqual(len(manifest["jobs"]), 32)
        grouped: dict[str, list[dict]] = {}
        for row in manifest["jobs"]:
            grouped.setdefault(row["group_id"], []).append(row)
        self.assertEqual(len(grouped), 8)

        expected = {
            ("fp16", "tvm_auto"),
            ("int8", "tvm_auto"),
            ("fp16", "trt_engine"),
            ("int8", "trt_engine"),
        }
        for rows in grouped.values():
            self.assertEqual({(row["q_mode"], row["dispatch_key"]) for row in rows}, expected)
            self.assertTrue(all(row["mixed_policy_id"] == "none" for row in rows))
            self.assertTrue(all("hand" not in row["strategy_id"].lower() for row in rows))

    def test_split_is_frozen_from_existing_counterpart_widths(self) -> None:
        module = _load_module()
        source = json.loads(GOLD96_MANIFEST.read_text(encoding="utf-8"))
        plan = module.build_supplement_plan(source["capability_profiles"])
        split = plan["split"]

        self.assertEqual(len(split["train_group_ids"]), 6)
        self.assertEqual(len(split["holdout_group_ids"]), 2)
        self.assertEqual(
            set(split["holdout_group_ids"]),
            {"pyramid|64x128x256", "codriving|40x80x160"},
        )
        self.assertEqual(split["selection_basis"], "inherit_gold96_counterpart_split_before_measurement")

    def test_rows_bind_counterpart_and_checkpoint_consistent_source_contract(self) -> None:
        module = _load_module()
        source = json.loads(GOLD96_MANIFEST.read_text(encoding="utf-8"))
        rows = module.build_supplement_plan(source["capability_profiles"])["manifest"]["jobs"]

        pyramid = next(row for row in rows if row["group_id"] == "pyramid|24x64x128")
        self.assertEqual(pyramid["counterpart_group_id"], "codriving|24x64x128")
        self.assertIn("frontier_01", pyramid["source_contract"]["checkpoint_glob"])
        self.assertTrue(pyramid["source_contract"]["checkpoint_path"].endswith("/net_epoch31.pth"))
        self.assertTrue(pyramid["source_contract"]["onnx_path"].endswith("pyramid_024x064x128_multiscale.onnx"))

        base = next(row for row in rows if row["group_id"] == "pyramid|64x128x256")
        self.assertIn("Pyramid_DAIR_m1_base_2023_08_14_11_42_29", base["source_contract"]["checkpoint_path"])
        self.assertTrue(base["source_contract"]["checkpoint_path"].endswith("net_epoch_bestval_at23.pth"))

        codriving = next(row for row in rows if row["group_id"] == "codriving|24x56x128")
        self.assertEqual(codriving["counterpart_group_id"], "pyramid|24x56x128")
        self.assertTrue(codriving["source_contract"]["model_dir"].endswith("/24x56x128"))
        self.assertTrue(codriving["source_contract"]["onnx_path"].endswith("resnet_multiscale_24x56x128_final_fp32.onnx"))
        self.assertTrue(codriving["source_contract"]["calibration_npz"].endswith("stage3_calib_train_n16_float32.npz"))
        self.assertTrue(codriving["source_contract"]["calibration_summary"].endswith("stage3_calib_train_n16_float32_summary.json"))
        self.assertEqual(codriving["source_status"], "pending_checkpoint_consistent_source")

    def test_cli_writes_content_addressed_manifest_split_and_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--capability-manifest-json",
                    str(GOLD96_MANIFEST),
                    "--output-dir",
                    tmp,
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            for key in ("manifest_json", "split_json", "contract_json", "index_json"):
                self.assertTrue(Path(output[key]).is_file())
            manifest = json.loads(Path(output["manifest_json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["jobs"]), 32)


if __name__ == "__main__":
    unittest.main()
