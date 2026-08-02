from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from scripts import stage2_h800_export_checkpoint_multiscale_onnx as exporter


class Stage2H800ExportCheckpointMultiscaleOnnxTest(unittest.TestCase):
    def test_load_best_checkpoint_state_selects_latest_post_baseline_bestval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            model = nn.Linear(1, 1)
            state_25 = {"weight": torch.tensor([[25.0]]), "bias": torch.tensor([25.0])}
            state_27 = {"weight": torch.tensor([[27.0]]), "bias": torch.tensor([27.0])}
            torch.save(state_25, ckpt_dir / "net_epoch_bestval_at25.pth")
            torch.save(state_27, ckpt_dir / "net_epoch_bestval_at27.pth")

            loaded, ckpt_path, epoch = exporter._load_best_checkpoint_state(model, ckpt_dir)

            self.assertEqual(epoch, 27)
            self.assertEqual(ckpt_path.name, "net_epoch_bestval_at27.pth")
            self.assertEqual(float(loaded.weight.item()), 27.0)

    def test_explicit_checkpoint_path_preserves_valid_epoch23_base_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            model = nn.Linear(1, 1)
            explicit = ckpt_dir / "net_epoch_bestval_at23.pth"
            torch.save(
                {"weight": torch.tensor([[23.0]]), "bias": torch.tensor([23.0])},
                explicit,
            )
            torch.save(
                {"weight": torch.tensor([[1.0]]), "bias": torch.tensor([1.0])},
                ckpt_dir / "net_epoch1.pth",
            )

            loaded, ckpt_path, epoch = exporter._load_best_checkpoint_state(
                model,
                ckpt_dir,
                checkpoint_path=explicit,
            )

            self.assertEqual(epoch, 23)
            self.assertEqual(ckpt_path, explicit)
            self.assertEqual(float(loaded.weight.item()), 23.0)


if __name__ == "__main__":
    unittest.main()
