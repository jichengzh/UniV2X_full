from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_gold32_pyramid_repair_manifest_v1.py"
PLAN_INDEX = REPO_ROOT / "results/stage35_gold32_supplement_v1_20260713/plan/gold32_supplement_plan_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("stage35_gold32_pyramid_repair_manifest_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Stage35Gold32PyramidRepairManifestV1Tests(unittest.TestCase):
    def test_overlay_replaces_only_pyramid_onnx_paths(self) -> None:
        module = _module()
        index = json.loads(PLAN_INDEX.read_text(encoding="utf-8"))
        source = json.loads(Path(index["manifest_json"]).read_text(encoding="utf-8"))
        repaired = module.apply_pyramid_shape_repair(source, "/remote/repaired")

        self.assertEqual(repaired["schema_version"], source["schema_version"])
        self.assertEqual(len(repaired["jobs"]), 32)
        self.assertEqual(repaired["repair_overlay"]["repaired_row_count"], 16)
        self.assertEqual(repaired["repair_overlay"]["input_shape"], [2, 64, 128, 256])

        for before, after in zip(source["jobs"], repaired["jobs"]):
            self.assertEqual(before["job_id"], after["job_id"])
            if before["model"] == "pyramid":
                self.assertTrue(after["source_contract"]["onnx_path"].startswith("/remote/repaired/"))
                self.assertEqual(
                    after["source_contract"]["supersedes_onnx_path"],
                    before["source_contract"]["onnx_path"],
                )
                self.assertEqual(
                    after["source_contract"]["calibration_root"],
                    before["source_contract"]["calibration_root"],
                )
            else:
                self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
