from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

from scripts import stage2_original60_fp16_train_launcher as launcher


class Stage2Original60Fp16TrainLauncherTest(unittest.TestCase):
    def test_build_env_prepends_env_python_dir_to_path(self) -> None:
        fake_env_python = "/opt/custom/bin/python"
        fake_heal_root = Path("/tmp/heal")
        base_env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": "seed"}

        with mock.patch.dict(os.environ, base_env, clear=True):
            env = launcher.build_env(
                gpu_id=5,
                heal_root=fake_heal_root,
                env_python=fake_env_python,
            )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "5")
        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(Path(fake_env_python).parent))
        self.assertTrue(env["PYTHONPATH"].startswith(str(fake_heal_root)))


if __name__ == "__main__":
    unittest.main()
