from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import stage2_h800_true_fp16_ap_eval as ap_eval
from scripts import stage2_original60_fp16_eval_when_ready as watcher


class Stage2Original60Fp16CheckpointSelectionTest(unittest.TestCase):
    def test_checkpoint_epoch_parses_bestval_and_plain_epoch_names(self) -> None:
        self.assertEqual(ap_eval.checkpoint_epoch(Path("/tmp/net_epoch_bestval_at29.pth")), 29)
        self.assertEqual(ap_eval.checkpoint_epoch(Path("/tmp/net_epoch31.pth")), 31)

    def test_eval_best_checkpoint_uses_highest_epoch_net_when_only_baseline_bestval_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            (ckpt_dir / "net_epoch_bestval_at23.pth").write_text("baseline", encoding="utf-8")
            (ckpt_dir / "net_epoch31.pth").write_text("epoch31", encoding="utf-8")

            best = ap_eval.best_checkpoint(ckpt_dir)

            self.assertEqual(best.name, "net_epoch31.pth")

    def test_eval_best_checkpoint_prefers_post_baseline_bestval_over_later_plain_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            (ckpt_dir / "net_epoch_bestval_at29.pth").write_text("best29", encoding="utf-8")
            (ckpt_dir / "net_epoch31.pth").write_text("epoch31", encoding="utf-8")

            best = ap_eval.best_checkpoint(ckpt_dir)

            self.assertEqual(best.name, "net_epoch_bestval_at29.pth")

    def test_watcher_best_checkpoint_epoch_ignores_baseline_bestval_when_later_epoch_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            (ckpt_dir / "net_epoch_bestval_at23.pth").write_text("baseline", encoding="utf-8")
            (ckpt_dir / "net_epoch31.pth").write_text("epoch31", encoding="utf-8")

            best_path, best_epoch = watcher.best_checkpoint_epoch(ckpt_dir)

            self.assertIsNotNone(best_path)
            self.assertEqual(best_path.name, "net_epoch31.pth")
            self.assertEqual(best_epoch, 31)

    def test_watcher_best_checkpoint_epoch_prefers_ready_post_baseline_bestval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            (ckpt_dir / "net_epoch_bestval_at23.pth").write_text("baseline", encoding="utf-8")
            (ckpt_dir / "net_epoch_bestval_at25.pth").write_text("best25", encoding="utf-8")
            (ckpt_dir / "net_epoch27.pth").write_text("epoch27", encoding="utf-8")

            best_path, best_epoch = watcher.best_checkpoint_epoch(ckpt_dir)

            self.assertIsNotNone(best_path)
            self.assertEqual(best_path.name, "net_epoch_bestval_at25.pth")
            self.assertEqual(best_epoch, 25)


if __name__ == "__main__":
    unittest.main()
