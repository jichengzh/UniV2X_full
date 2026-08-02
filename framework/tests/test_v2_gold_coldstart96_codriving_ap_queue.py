import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from scripts.stage2_v2_gold_coldstart96_codriving_ap_queue import (
    _prepare_import_paths,
    patch_codriving_config_text,
    parse_width,
    slice_tensor_to_shape,
    stage_train_dir,
)


class TestCoDrivingApQueue(unittest.TestCase):
    def test_patch_codriving_config_sets_backbone_filters_and_deblocks(self):
        src = """model:
  args:
    pillar_vfe:
      num_filters: [64]
    base_bev_backbone:
      layer_nums: &layer_nums [3, 4, 5]
      num_filters: &num_filters [64, 128, 256]
      num_upsample_filter: [128, 128, 128]
    resnet:
      layer_nums: *layer_nums
      num_filters: *num_filters
"""
        out = patch_codriving_config_text(src, (24, 32, 96), ckpt_path="/tmp/warmstart.pth")

        self.assertIn("num_filters: &num_filters [24, 32, 96]", out)
        self.assertIn("num_upsample_filter: [32, 32, 32]", out)
        self.assertIn("num_filters: [64]", out)
        self.assertIn("# v2_gold_coldstart_96 target width: 24x32x96", out)
        self.assertIn("# warmstart_ckpt: /tmp/warmstart.pth", out)

    def test_slice_tensor_to_shape_uses_prefix_slices(self):
        source = torch.arange(4 * 5 * 2, dtype=torch.float32).reshape(4, 5, 2)
        result = slice_tensor_to_shape(source, (2, 3, 2))

        self.assertTrue(torch.equal(result, source[:2, :3, :2]))
        self.assertEqual(tuple(result.shape), (2, 3, 2))

    def test_parse_width_accepts_x_and_comma_forms(self):
        self.assertEqual(parse_width("24x32x96"), (24, 32, 96))
        self.assertEqual(parse_width("24,32,96"), (24, 32, 96))

    def test_prepare_import_paths_keeps_site_packages_ahead_of_compat_paths(self):
        repo = Path("/tmp/v2xverse")
        with mock.patch("sys.path", ["site-packages"]):
            _prepare_import_paths(repo)

            self.assertEqual(__import__("sys").path[0], str(repo))
            self.assertLess(
                __import__("sys").path.index("site-packages"),
                __import__("sys").path.index("/data/jichengzhi_v2x/t2lib"),
            )

    def test_stage_train_dir_copies_warmstart_as_bestval_epoch_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.yaml"
            warmstart = root / "warmstart.pth"
            train_dir = root / "train"
            config.write_text("name: test\n", encoding="utf-8")
            torch.save({"weight": torch.ones(1)}, warmstart)

            report = stage_train_dir(config, warmstart, train_dir)

            self.assertEqual(report["model_dir"], str(train_dir))
            self.assertTrue((train_dir / "config.yaml").is_file())
            self.assertTrue((train_dir / "net_epoch_bestval_at0.pth").is_file())


if __name__ == "__main__":
    unittest.main()
