from __future__ import annotations

import torch
import torch.nn as nn

from framework.stage1.trace_plan import BoundaryValidator, TraceBoundaryDetector, WrapperSynthesizer


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, batch):
        x = batch["spatial_features"] if isinstance(batch, dict) else batch
        return {"spatial_features_2d": self.block(x)}


class _HeterLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_m1 = _Backbone()
        self.shrinker_m1 = nn.Conv2d(64, 64, 1)
        self.fusion_net = nn.Identity()
        self.cls_head = nn.Conv2d(64, 2, 1)
        self.reg_head = nn.Conv2d(64, 14, 1)
        self.dir_head = nn.Conv2d(64, 4, 1)


def test_boundary_validator_executes_forward_depgraph_and_prune_dryrun():
    full = _HeterLike()
    plan = TraceBoundaryDetector().detect(
        full,
        model_name="fcooper",
        config_path="/tmp/fcooper/config.yaml",
        ckpt_path="/tmp/fcooper/net.pth",
        ckpt_status="ok",
        input_shape=[1, 64, 16, 16],
    )
    wrapper = WrapperSynthesizer().synthesize(full, plan["selected_candidate"])
    ignored = [wrapper.heads["cls_head"], wrapper.heads["reg_head"], wrapper.heads["dir_head"]]

    validation = BoundaryValidator().validate(
        wrapper,
        torch.randn(1, 64, 16, 16),
        ignored_layers=ignored,
        run_prune=True,
    )

    assert validation["wrapper_forward_dryrun"] == "ok"
    assert validation["output_shape_sanity"] == "ok"
    assert validation["depgraph_build"] == "ok"
    assert validation["n_prunable_groups"] > 0
    assert validation["prune_dryrun"] == "ok"
    assert validation["interface_invariant_check"] == "ok"
