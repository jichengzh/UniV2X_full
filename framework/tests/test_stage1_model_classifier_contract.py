from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from framework.stage1.model_classifier import (
    BACKEND_POLICY,
    DEFAULT_MANIFESTS,
    build_classification_report,
)


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = ROOT / "results/stage1_model_predict"


def _by_model(report: dict) -> dict[str, dict]:
    return {item["model"]: item for item in report["models"]}


def test_backend_policy_allows_only_h800_tvm_for_new_measurements():
    report = build_classification_report(evidence_dir=EVIDENCE_DIR)

    assert BACKEND_POLICY["default_backend"] == "h800_tvm"
    assert report["backend_policy"]["default_backend"] == "h800_tvm"
    assert report["backend_policy"]["default_new_measurement_backend"] == "h800_tvm"
    assert report["backend_policy"]["allowed_new_measurement_backends"] == ["h800_tvm"]
    assert "trt" not in [
        backend.lower()
        for backend in report["backend_policy"]["allowed_new_measurement_backends"]
    ]


def test_model_classifier_output_contract_and_no_overpromotion():
    report = build_classification_report(evidence_dir=EVIDENCE_DIR)

    assert report["schema"] == "stage1_model_classification_v1"
    assert len(report["models"]) == 9
    assert len(DEFAULT_MANIFESTS) == 9
    assert report["no_overpromotion"] is True
    assert report["overpromotion"] == []

    required = {
        "model",
        "ckpt_status",
        "acceleration_class",
        "acceleration_class_label",
        "acceleration_class_reason",
        "classification",
        "scope",
        "backend_policy",
        "evidence_level",
        "trace_confidence",
        "coverage_scope",
        "manual_override_used",
        "review_required",
        "included_modules",
        "ignored_layers",
        "skipped_subgraphs",
        "rejected_candidates",
        "trace_plan",
        "evidence_sources",
        "historical_evidence_sources",
        "blockers",
        "required_next_probe_or_gate",
        "unsupported_conclusions",
        "no_overpromotion",
    }
    for item in report["models"]:
        assert required <= set(item)
        assert item["backend_policy"] == report["backend_policy"]
        assert item["no_overpromotion"] is True
        assert isinstance(item["blockers"], list)
        assert isinstance(item["historical_evidence_sources"], list)
        assert item["acceleration_class"] in {
            "CO_ACCELERATION_REQUIRED",
            "SEPARABLE_ACCELERATION",
            "SCAN_FAILED",
        }


def test_trt_is_historical_only_and_h800_tvm_latency_is_measured_evidence():
    report = build_classification_report(evidence_dir=EVIDENCE_DIR)
    by_model = _by_model(report)

    for item in report["models"]:
        measured_blob = repr(item.get("measured_h800_tvm", "")).lower()
        assert "trt" not in measured_blob

    pyramid = by_model["pyramid_lidar"]
    assert "historical_trt_evidence" in pyramid["evidence_level"]
    assert pyramid["historical_evidence_sources"]
    assert any("trt" in src.lower() for src in pyramid["historical_evidence_sources"])
    assert "historical_trt_evidence" in pyramid["historical_labels"]
    assert "measured_h800_tvm_latency" not in pyramid["historical_labels"]

    fcooper = by_model["fcooper"]
    assert "measured_h800_tvm_latency" in fcooper["evidence_level"]
    assert fcooper["measured_h800_tvm"]["backend"] == "h800_tvm"

    v2xvit = by_model["v2xvit"]
    assert any("true TRT INT8 AP" in item for item in v2xvit["unsupported_conclusions"])
    assert any("not done" in item.lower() or "blocked" in item.lower() for item in v2xvit["blockers"])


def test_architecture_only_models_and_codriving_stay_scoped():
    report = build_classification_report(evidence_dir=EVIDENCE_DIR)
    by_model = _by_model(report)

    codriving = by_model["codriving"]
    assert codriving["acceleration_class"] == "SEPARABLE_ACCELERATION"
    assert codriving["acceleration_class_label"] == "可分离加速"
    assert codriving["scope"] == "codriving_measured_resnet_backbone_envelope"
    assert any("cross-model" in item for item in codriving["unsupported_conclusions"])
    assert codriving["classification"] != "FULL_MODEL_SEPARABLE"

    for model in ("where2comm", "v2vnet", "disconet"):
        item = by_model[model]
        assert item["ckpt_status"] == "missing_architecture_scan_only"
        assert item["acceleration_class"] == "SCAN_FAILED"
        assert item["acceleration_class_label"] == "扫描失败"
        assert "architecture_only" in item["classification"].lower()
        assert item["measured_h800_tvm"] == {}
        assert item["evidence_level"] == "architecture_only_missing_ckpt_plus_random_init_fusion_sidecar"
        assert any("trained checkpoint scan" in x for x in item["unsupported_conclusions"])
        assert item["classification"] != "FULL_MODEL_SEPARABLE"


def test_three_class_acceleration_mapping_is_integrated_in_report():
    report = build_classification_report(evidence_dir=EVIDENCE_DIR)
    by_model = _by_model(report)

    assert report["acceleration_class_labels"] == {
        "CO_ACCELERATION_REQUIRED": "需要协同加速",
        "SEPARABLE_ACCELERATION": "可分离加速",
        "SCAN_FAILED": "扫描失败",
    }
    assert by_model["codriving"]["acceleration_class"] == "SEPARABLE_ACCELERATION"
    for model in (
        "fcooper",
        "attfuse",
        "v2xvit",
        "pyramid_lidar",
        "pyramid_camera",
    ):
        assert by_model[model]["acceleration_class"] == "CO_ACCELERATION_REQUIRED"
    for model in ("where2comm", "v2vnet", "disconet"):
        assert by_model[model]["acceleration_class"] == "SCAN_FAILED"


def test_stage1_classify_models_cli_writes_json_and_markdown(tmp_path: Path):
    out_json = tmp_path / "stage1_model_classification_v1.json"
    out_md = tmp_path / "stage1_model_classification_v1.md"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/stage1_classify_models.py"),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_json.exists()
    assert out_md.exists()


def test_source_wording_does_not_promote_no_cliff_to_model_separable():
    checked = [
        ROOT / "framework/stage1_bridge.py",
        ROOT / "framework/search_three_arm.py",
        ROOT / "framework/run_pqs_ablation.py",
    ]
    blob = "\n".join(path.read_text(encoding="utf-8") for path in checked)

    assert "SEPARABLE (全旋钮可串行)" not in blob
    assert "No cliff (separable" not in blob
    assert "no cliff (separable" not in blob
    assert "H800_TVM×4090_Q_ratio" not in blob
    assert "TRT_INT8_MinMaxCalib" not in blob
