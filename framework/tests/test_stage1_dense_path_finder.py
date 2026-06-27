from __future__ import annotations

import torch.nn as nn

from framework.stage1.trace_plan import DensePathFinder, HeuristicTagger, ModuleTreeScanner


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pillar_vfe = nn.Linear(4, 4)
        self.scatter = nn.Identity()


class _MaxFusion(nn.Module):
    def forward(self, x):
        return x


class _HeterLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_m1 = _Encoder()
        self.backbone_m1 = _Backbone()
        self.shrinker_m1 = nn.Conv2d(64, 64, 1)
        self.fusion_net = _MaxFusion()
        self.cls_head = nn.Conv2d(64, 2, 1)
        self.reg_head = nn.Conv2d(64, 14, 1)
        self.dir_head = nn.Conv2d(64, 4, 1)


def test_dense_path_finder_generates_heter_candidate_and_skips_frontend_fusion():
    model = _HeterLike()
    tagger = HeuristicTagger()
    records = tagger.tag_many(ModuleTreeScanner().scan(model))
    by_path = {record.path: record for record in records}

    candidates, skipped, rejected = DensePathFinder().find(
        records=records,
        records_by_path=by_path,
        tagger=tagger,
        model_name="fcooper",
        input_shape=[1, 64, 16, 16],
    )

    assert rejected == []
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.candidate_id == "heter_baseline_dense_m1"
    assert candidate.entry == "post_scatter_bev"
    assert candidate.wrapper_kind == "heter_baseline_dense_path"
    assert candidate.included_modules == [
        "backbone_m1",
        "shrinker_m1",
        "cls_head",
        "reg_head",
        "dir_head",
    ]
    assert candidate.ignored_layers == ["cls_head", "reg_head", "dir_head"]
    by_skip = {item["name"]: item for item in skipped}
    assert by_skip["encoder_m1.pillar_vfe"]["type"] == "sparse_or_geometry_preprocess"
    assert by_skip["encoder_m1.scatter"]["type"] == "sparse_or_geometry_preprocess"
    assert by_skip["fusion_net"]["blocker_gate"] == "maxfusion_coverage_anchor"
