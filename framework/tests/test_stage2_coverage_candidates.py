from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/stage2_generate_coverage_candidates.py"
CREATED_AT = "2026-06-26T00:00:00Z"


def _load_generator_module():
    spec = importlib.util.spec_from_file_location(
        "stage2_generate_coverage_candidates",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Stage2CoverageCandidateGeneratorTest(unittest.TestCase):
    def test_default_generator_emits_60_unique_fp16_backbone_candidates(self):
        generator = _load_generator_module()
        existing_rows = [
            {
                "label": "s0_048",
                "config_id": "coverage_h800_tvm_pyramid_w48x128x256_fp16_metaschedule_tuned",
                "width": [48, 128, 256],
            },
            {
                "label": "already_frontier",
                "config_id": "coverage_h800_tvm_pyramid_w16x32x64_fp16_default",
                "width": [16, 32, 64],
            },
        ]

        candidates = generator.generate_coverage_candidates(
            limit=60,
            existing_rows=existing_rows,
            created_at=CREATED_AT,
        )

        required = {
            "candidate_id",
            "model",
            "label",
            "width",
            "arm",
            "quant_policy",
            "schedule_policy",
            "optimized_scope",
            "priority",
            "axes_required",
            "repeat_policy",
            "max_latency_repeats",
            "max_energy_repeats",
            "ap_policy",
            "created_at",
        }
        self.assertEqual(len(candidates), 60)
        self.assertTrue(all(required <= set(row) for row in candidates))
        self.assertEqual({row["model"] for row in candidates}, {"Pyramid-LiDAR"})
        self.assertEqual({row["quant_policy"] for row in candidates}, {"fp16"})
        self.assertEqual({row["optimized_scope"] for row in candidates}, {"backbone_only"})
        self.assertEqual({row["repeat_policy"] for row in candidates}, {"coverage"})
        self.assertEqual({row["max_latency_repeats"] for row in candidates}, {1})
        self.assertEqual({row["max_energy_repeats"] for row in candidates}, {1})
        self.assertEqual(
            {tuple(row["axes_required"]) for row in candidates},
            {("latency", "energy", "ap")},
        )
        self.assertEqual(
            {row["ap_policy"] for row in candidates},
            {"true_eval_or_true_import_only"},
        )

        labels = [row["label"] for row in candidates]
        candidate_ids = [row["candidate_id"] for row in candidates]
        widths = [tuple(row["width"]) for row in candidates]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))
        self.assertEqual(len(widths), len(set(widths)))
        self.assertNotIn("s0_048", labels)
        self.assertNotIn((48, 128, 256), set(widths))
        self.assertNotIn((16, 32, 64), set(widths))

        sources = {row["sample_source"] for row in candidates}
        self.assertTrue(
            {
                "single_axis_s0",
                "single_axis_s1",
                "single_axis_s2",
                "coupled_latin_hypercube",
                "frontier_neighborhood",
            }
            <= sources
        )
        self.assertIn((16, 128, 256), set(widths))
        self.assertIn((64, 32, 256), set(widths))
        self.assertIn((64, 128, 64), set(widths))

    def test_cli_writes_candidate_queue_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            existing_jsonl = tmp_path / "existing.jsonl"
            existing_jsonl.write_text(
                json.dumps(
                    {
                        "label": "s1_096",
                        "config_id": "coverage_h800_tvm_pyramid_w64x96x256_fp16_default",
                        "width": [64, 96, 256],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            out_jsonl = tmp_path / "candidates/candidate_queue.jsonl"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--existing-jsonl",
                    str(existing_jsonl),
                    "--out-jsonl",
                    str(out_jsonl),
                    "--created-at",
                    CREATED_AT,
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            rows = [
                json.loads(line)
                for line in out_jsonl.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 60)
            self.assertNotIn("s1_096", {row["label"] for row in rows})
            self.assertEqual(rows[0]["schema"], "stage2_candidate_row_v1")
            self.assertEqual(rows[0]["created_at"], CREATED_AT)


if __name__ == "__main__":
    unittest.main()
