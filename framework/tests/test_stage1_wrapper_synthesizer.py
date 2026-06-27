from __future__ import annotations

import torch
import torch.nn as nn

from framework.stage1.trace_plan import TraceBoundaryDetector, WrapperSynthesizer


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

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


def test_wrapper_synthesizer_builds_executable_dense_path_wrapper():
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
    outputs = wrapper(torch.randn(1, 64, 16, 16))

    assert plan["selected_candidate"]["wrapper_kind"] == "heter_baseline_dense_path"
    assert isinstance(outputs, tuple)
    assert [list(item.shape) for item in outputs] == [
        [1, 2, 16, 16],
        [1, 14, 16, 16],
        [1, 4, 16, 16],
    ]
    module_names = {name for name, _module in wrapper.named_modules()}
    assert "body.backbone_m1" in module_names
    assert "body.shrinker_m1" in module_names
    assert "heads.cls_head" in module_names
