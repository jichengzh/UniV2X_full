from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from unittest import mock
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96_export_codriving_onnx as export_onnx  # noqa: E402


def load_collab_export_module():
    path = REPO_ROOT / "tools" / "export_onnx_codriving_collab.py"
    spec = importlib.util.spec_from_file_location("export_onnx_codriving_collab_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V2GoldColdstart96ExportCoDrivingOnnxTests(unittest.TestCase):
    def test_default_export_kwargs_are_static_for_trt_profile_compatibility(self) -> None:
        kwargs = export_onnx.build_export_kwargs(dynamic_batch=False)

        self.assertNotIn("dynamic_axes", kwargs)
        self.assertEqual(kwargs["input_names"], ["spatial_features"])
        self.assertEqual(kwargs["output_names"], ["backbone_output"])

    def test_dynamic_batch_is_opt_in_only(self) -> None:
        kwargs = export_onnx.build_export_kwargs(dynamic_batch=True)

        self.assertEqual(kwargs["dynamic_axes"], {"spatial_features": {0: "batch"}})

    def test_collab_export_can_disable_legacy_t2lib_path_injection(self) -> None:
        module = load_collab_export_module()

        with mock.patch.dict(os.environ, {"CODRIVING_EXPORT_DISABLE_LEGACY_T2LIB": "1"}):
            self.assertEqual(module.external_python_paths(), [str(module.REPO_ROOT)])

        with mock.patch.dict(os.environ, {}, clear=True):
            paths = module.external_python_paths()

        self.assertIn(str(module.T2LIB), paths)
        self.assertIn(str(module.TVM_SP), paths)


if __name__ == "__main__":
    unittest.main()
