from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_pyramid_shape_repair_v1.sh"


class Stage35PyramidShapeRepairV1Tests(unittest.TestCase):
    def test_dry_run_describes_four_h800_gpu6_repairs(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("required_host=zs-nj-tap-gpu18", completed.stdout)
        self.assertIn("gpu=6", completed.stdout)
        self.assertEqual(completed.stdout.count("--input-shape 2,64,128,256"), 4)
        self.assertEqual(completed.stdout.count("stage35_prepare_pyramid_trt_calibration_v1.py"), 4)
        for padded in ("016x032x064", "024x064x128", "032x064x128", "064x128x256"):
            self.assertIn(f"pyramid_sources_shape_repaired_v1/{padded}", completed.stdout)
            self.assertIn(f"pyramid_calibration/{padded}/spatial_features_train16.npz", completed.stdout)


if __name__ == "__main__":
    unittest.main()
