from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_canonical_architecture_v3 as planner  # noqa: E402


class Stage2CanonicalArchitectureV3Tests(unittest.TestCase):
    def test_build_outputs_emits_profiles_96_jobs_80_16_split_and_prior_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "label": "legacy",
                                "precision": "int8",
                                "width": [24, 32, 96],
                                "can_use_as_cold_start_prior": True,
                                "trusted_for_final_frontier": False,
                                "training_source_class": "historical_hand_rewrite_diagnostic",
                                "routeb_int8_latency_ms_if_measured": 3.6,
                                "phase1_old_int8tc_latency_ms_if_available": 1.7,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            outputs = planner.build_outputs(original180_audit_path=audit, seed=42)

        self.assertEqual(len(outputs["capability_profiles"]), 2)
        self.assertEqual(len(outputs["active_manifest"]["jobs"]), 96)
        self.assertEqual(len(outputs["split"]["train"]), 80)
        self.assertEqual(len(outputs["split"]["holdout"]), 16)
        self.assertEqual(len(outputs["historical180_routes"]["width_ap_prior"]), 1)
        features = [profile["features"] for profile in outputs["capability_profiles"]]
        self.assertTrue(all("int8_over_fp16_latency_ratio" not in item for item in features))
        self.assertTrue(all("int8_over_fp16_energy_ratio" not in item for item in features))
        self.assertTrue(all("backend" not in item for item in features))

    def test_write_outputs_round_trips_machine_readable_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audit = root / "audit.json"
            audit.write_text(json.dumps({"rows": []}), encoding="utf-8")
            outputs = planner.build_outputs(original180_audit_path=audit, seed=42)

            paths = planner.write_outputs(outputs, root / "out")

            self.assertEqual(set(paths), {"capability_profiles", "active_manifest", "split", "historical180_routes"})
            for path in paths.values():
                self.assertTrue(path.is_file())
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
