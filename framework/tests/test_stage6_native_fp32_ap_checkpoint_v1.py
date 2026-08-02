from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.stage2_h800_true_fp32_ap_eval import selected_checkpoint


class Stage6NativeFp32ApCheckpointV1Tests(unittest.TestCase):
    def test_explicit_checkpoint_overrides_legacy_epoch_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "net_epoch_bestval_at23.pth"
            explicit.write_bytes(b"baseline")
            (root / "net_epoch1.pth").write_bytes(b"epoch1")
            args = argparse.Namespace(ckpt_dir=str(root), checkpoint_path=str(explicit))

            self.assertEqual(selected_checkpoint(args), explicit)

    def test_explicit_checkpoint_must_belong_to_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / "outside.pth"
            outside.write_bytes(b"outside")
            args = argparse.Namespace(ckpt_dir=str(root), checkpoint_path=str(outside))
            try:
                with self.assertRaisesRegex(ValueError, "checkpoint directory"):
                    selected_checkpoint(args)
            finally:
                outside.unlink(missing_ok=True)
