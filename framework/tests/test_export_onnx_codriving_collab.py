from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools import export_onnx_codriving_collab as exporter  # noqa: E402


class ExportOnnxCoDrivingCollabTests(unittest.TestCase):
    def test_load_hypes_prefers_opencood_yaml_parser(self) -> None:
        yaml_utils = types.ModuleType("opencood.hypes_yaml.yaml_utils")
        yaml_utils.load_yaml = lambda path: {"parsed_by": "opencood", "path": path}
        hypes_yaml = types.ModuleType("opencood.hypes_yaml")
        hypes_yaml.yaml_utils = yaml_utils
        opencood = types.ModuleType("opencood")
        opencood.hypes_yaml = hypes_yaml

        with unittest.mock.patch.dict(
            sys.modules,
            {
                "opencood": opencood,
                "opencood.hypes_yaml": hypes_yaml,
                "opencood.hypes_yaml.yaml_utils": yaml_utils,
            },
        ):
            payload = exporter.load_hypes_for_create_model("/tmp/config.yaml")

        self.assertEqual(payload, {"parsed_by": "opencood", "path": "/tmp/config.yaml"})


if __name__ == "__main__":
    unittest.main()
