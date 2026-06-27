from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml

from framework.stage1.model_classifier import build_classification_report
from framework.stage1.trace_plan import (
    TRACE_PLAN_SCHEMA,
    HeuristicTagger,
    ModuleTreeScanner,
    TraceBoundaryDetector,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "results/stage1_model_predict"


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


class _Shrinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(64, 64, 1), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class MaxFusion(nn.Module):
    def forward(self, x):
        return x


class AttFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        return self.attention(x)


class Where2commRoutingFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_passing = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        return self.message_passing(x)


class V2VNetMessagePassingFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.message_passing = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        return self.message_passing(x)


class DiscoNetFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.disco_message_passing = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        return self.disco_message_passing(x)


class _MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pillar_vfe = nn.Linear(4, 4)
        self.scatter = nn.Identity()


class HeterModelBaseline(nn.Module):
    def __init__(self, fusion: nn.Module):
        super().__init__()
        self.encoder_m1 = _MockEncoder()
        self.backbone_m1 = _Backbone()
        self.shrinker_m1 = _Shrinker()
        self.fusion_net = fusion
        self.cls_head = nn.Conv2d(64, 2, 1)
        self.reg_head = nn.Conv2d(64, 14, 1)
        self.dir_head = nn.Conv2d(64, 4, 1)


def _by_name(items: list[dict]) -> dict[str, dict]:
    return {item["name"]: item for item in items}


def _detect(model_name: str, fusion: nn.Module, ckpt_status: str = "ok") -> dict:
    return TraceBoundaryDetector().detect(
        HeterModelBaseline(fusion),
        model_name=model_name,
        config_path=f"/tmp/{model_name}/config.yaml",
        ckpt_path=f"/tmp/{model_name}/net.pth",
        ckpt_status=ckpt_status,
        input_shape=[1, 64, 512, 512],
    )


def test_module_tree_scanner_and_tagger_mark_dense_sparse_and_fusion_boundaries():
    records = ModuleTreeScanner().scan(HeterModelBaseline(MaxFusion()))
    tagged = HeuristicTagger().tag_many(records)
    by_path = {record.path: record for record in tagged}

    assert "dense_path_candidate" in by_path["backbone_m1"].tags
    assert "dense_path_candidate" in by_path["shrinker_m1"].tags
    assert "ignored_candidate" in by_path["cls_head"].tags
    assert "sparse_or_geometry_preprocess" in by_path["encoder_m1.pillar_vfe"].tags
    assert "sparse_or_geometry_preprocess" in by_path["encoder_m1.scatter"].tags
    assert "fusion_or_alignment" in by_path["fusion_net"].tags


def test_fcooper_trace_plan_identifies_maxfusion_skipped_and_ckpt_ok():
    plan = _detect("fcooper", MaxFusion(), ckpt_status="ok")
    included = _by_name(plan["included_modules"])
    ignored = _by_name(plan["ignored_layers"])
    skipped = _by_name(plan["skipped_subgraphs"])

    assert plan["schema"] == TRACE_PLAN_SCHEMA
    assert plan["detector"] == "TraceBoundaryDetector.heter_baseline_v1"
    assert plan["ckpt_status"] == "ok"
    assert plan["manual_override_used"] is False
    assert plan["coverage_scope"] == "dense_core_only"
    assert plan["trace_confidence"] == "medium"
    assert plan["rejected_candidates"] == []
    assert {"backbone_m1", "shrinker_m1", "cls_head", "reg_head", "dir_head"} <= set(included)
    assert {"cls_head", "reg_head", "dir_head"} <= set(ignored)
    assert {"encoder_m1.pillar_vfe", "encoder_m1.scatter", "fusion_net"} <= set(skipped)
    assert skipped["fusion_net"]["type"] == "fusion_or_alignment"
    assert skipped["fusion_net"]["blocker_gate"] == "maxfusion_coverage_anchor"
    assert skipped["fusion_net"]["full_model_verdict_blocker"] is False


def test_attfuse_trace_plan_identifies_attention_fusion_as_full_model_blocker():
    plan = _detect("attfuse", AttFusion(), ckpt_status="ok")
    skipped = _by_name(plan["skipped_subgraphs"])

    assert "fusion_net" in skipped
    assert skipped["fusion_net"]["type"] == "attention_or_routing_fusion"
    assert skipped["fusion_net"]["blocker_gate"] == "attention_fusion_coverage_anchor"
    assert skipped["fusion_net"]["full_model_verdict_blocker"] is True
    assert skipped["encoder_m1.pillar_vfe"]["type"] == "sparse_or_geometry_preprocess"
    assert skipped["encoder_m1.scatter"]["type"] == "sparse_or_geometry_preprocess"
    assert "fusion_attention_or_routing_boundary_not_closed" in plan["review_reasons"]


def test_where2comm_missing_ckpt_stays_architecture_only_and_scan_failed(tmp_path: Path):
    plan = _detect("where2comm", Where2commRoutingFusion(), ckpt_status="missing_architecture_scan_only")
    skipped = _by_name(plan["skipped_subgraphs"])

    assert plan["ckpt_status"] == "missing_architecture_scan_only"
    assert "missing_checkpoint_architecture_only" in plan["review_reasons"]
    assert skipped["fusion_net"]["blocker_gate"] == "routing_fusion_coverage_anchor"
    assert skipped["encoder_m1.pillar_vfe"]["type"] == "sparse_or_geometry_preprocess"
    assert skipped["encoder_m1.scatter"]["type"] == "sparse_or_geometry_preprocess"

    manifest = {
        "stage": "stage1_partition",
        "model": "where2comm",
        "model_class": "HeterModelBaseline",
        "config": "/tmp/where2comm/config.yaml",
        "ckpt": "",
        "ckpt_status": "missing_architecture_scan_only",
        "scan_status": "ok",
        "trace": {
            "entry_shape": [1, 64, 512, 512],
            "skipped_modules": [],
            "skipped_subgraphs": plan["skipped_subgraphs"],
            "note": "unit-test autonomous trace plan",
        },
        "trace_plan": plan,
        "search_space_summary": {
            "n_b1_search_knobs": 1,
            "latency_status": "skipped",
        },
        "view_b1_prune_groups": [],
        "view_b1_search_groups": [],
        "view_latency": {"status": "skipped"},
        "checks": {"dryrun_prune05": {"status": "ok"}},
    }
    manifest_path = tmp_path / "where2comm_partition.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    report = build_classification_report(manifests=[manifest_path], evidence_dir=EVIDENCE_DIR)
    item = report["models"][0]
    assert item["model"] == "where2comm"
    assert item["acceleration_class"] == "SCAN_FAILED"
    assert item["ckpt_status"] == "missing_architecture_scan_only"
    assert item["manual_override_used"] is False
    assert item["coverage_scope"] == "dense_core_only"
    assert item["trace_confidence"] == "medium"
    assert item["classification"] == "ARCHITECTURE_ONLY_FUSION_UNCOVERED_UNKNOWN"
    assert "missing_checkpoint_architecture_only" in item["blockers"]
    assert any("trained checkpoint scan" in x for x in item["unsupported_conclusions"])


def test_v2vnet_and_disconet_trace_plans_mark_routing_fusion_skipped():
    for model_name, fusion in (
        ("v2vnet", V2VNetMessagePassingFusion()),
        ("disconet", DiscoNetFusion()),
    ):
        plan = _detect(model_name, fusion, ckpt_status="missing_architecture_scan_only")
        skipped = _by_name(plan["skipped_subgraphs"])

        assert plan["ckpt_status"] == "missing_architecture_scan_only"
        assert {"backbone_m1", "shrinker_m1", "cls_head", "reg_head", "dir_head"} <= {
            item["name"] for item in plan["included_modules"]
        }
        assert {"cls_head", "reg_head", "dir_head"} <= {
            item["name"] for item in plan["ignored_layers"]
        }
        assert skipped["fusion_net"]["type"] == "attention_or_routing_fusion"
        assert skipped["fusion_net"]["blocker_gate"] == "routing_fusion_coverage_anchor"
        assert skipped["fusion_net"]["full_model_verdict_blocker"] is True
        assert skipped["encoder_m1.pillar_vfe"]["type"] == "sparse_or_geometry_preprocess"
        assert skipped["encoder_m1.scatter"]["type"] == "sparse_or_geometry_preprocess"
