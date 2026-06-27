from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "attention_e2e_pq_validator.py"
    spec = importlib.util.spec_from_file_location("attention_e2e_pq_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _valid_report():
    return {
        "schema_version": "attention_e2e_pq_v1",
        "dataset": {
            "name": "DAIR-V2X",
            "split": "val",
            "samples": 1789,
            "eval_script": "scripts/phase2/attention_eval.py",
        },
        "rows": [
            {
                "config": "baseline",
                "attention_prune_pct": 0,
                "quant": "fp32_pytorch",
                "quant_backend": "none",
                "finetune": "official_ckpt",
                "checkpoint_path": "/ckpt/base.pth",
                "manifest_path": "",
                "latency_scope": "e2e",
                "e2e_latency_ms": 220.0,
                "speedup": 1.0,
                "ap50": 0.7103,
                "ap70": 0.5212,
                "delta_ap50": 0.0,
                "delta_ap70": 0.0,
                "latency_command": "CUDA_VISIBLE_DEVICES=0 python bench.py",
                "latency_log": "logs/base_latency.log",
                "ap_command": "CUDA_VISIBLE_DEVICES=0 python eval.py",
                "ap_log": "logs/base_ap.log",
                "dataset_split": "DAIR val",
                "n_samples": 1789,
            },
            {
                "config": "attention-p50-fp16",
                "attention_prune_pct": 50,
                "quant": "fp16",
                "quant_backend": "pytorch",
                "finetune": "short_ft_epoch3_lr1e-4_seed42",
                "checkpoint_path": "/ckpt/attention_p50_fp16.pth",
                "manifest_path": "models/v2xvit_attention_t1/attention_p50_manifest.json",
                "prune_manifest": {
                    "hmsa_keep_heads": 4,
                    "mswin_keep_heads": {"ws4": 8, "ws8": 4, "ws16": 2},
                    "dim_256_preserved": True,
                },
                "latency_scope": "e2e",
                "e2e_latency_ms": 205.0,
                "speedup": 1.0732,
                "ap50": 0.704,
                "ap70": 0.515,
                "delta_ap50": -0.0063,
                "delta_ap70": -0.0062,
                "latency_command": "CUDA_VISIBLE_DEVICES=0 python bench.py --attention-p50",
                "latency_log": "logs/p50_fp16_latency.log",
                "ap_command": "CUDA_VISIBLE_DEVICES=0 python eval.py --attention-p50",
                "ap_log": "logs/p50_fp16_ap.log",
                "dataset_split": "DAIR val",
                "n_samples": 1789,
            },
            {
                "config": "attention-p50-int8/mixed",
                "attention_prune_pct": 50,
                "quant": "int8/mixed",
                "quant_backend": "TVM Relax int8/mixed",
                "finetune": "same_checkpoint_as_attention-p50-fp16",
                "checkpoint_path": "/ckpt/attention_p50_fp16.pth",
                "manifest_path": "models/v2xvit_attention_t1/attention_p50_manifest.json",
                "prune_manifest": {
                    "hmsa_keep_heads": 4,
                    "mswin_keep_heads": {"ws4": 8, "ws8": 4, "ws16": 2},
                    "dim_256_preserved": True,
                },
                "latency_scope": "e2e",
                "e2e_latency_ms": 190.0,
                "speedup": 1.1579,
                "ap50": 0.699,
                "ap70": 0.509,
                "delta_ap50": -0.0113,
                "delta_ap70": -0.0122,
                "latency_command": "CUDA_VISIBLE_DEVICES=0 python bench.py --tvm-int8-mixed",
                "latency_log": "logs/p50_int8_latency.log",
                "ap_command": "CUDA_VISIBLE_DEVICES=0 python eval.py --attention-p50",
                "ap_log": "logs/p50_int8_ap.log",
                "dataset_split": "DAIR val",
                "n_samples": 1789,
            },
        ],
    }


def test_validator_accepts_complete_e2e_attention_pq_report():
    mod = _load_module()

    validation = mod.validate_stop_a_report(_valid_report())

    assert validation["verdict"] == "ACCEPTABLE_E2E_SCHEMA"
    assert validation["errors"] == []


def test_validator_rejects_direct_matmul_fake_ap_and_non_tvm_quant():
    mod = _load_module()
    report = _valid_report()
    bad = report["rows"][2]
    bad["latency_scope"] = "direct_matmul"
    bad["accuracy_status"] = "SIMULATED_PRIOR_NOT_TRUE_TVM"
    bad["quant_backend"] = "TensorRT INT8"

    validation = mod.validate_stop_a_report(report)
    messages = "\n".join(issue["message"] for issue in validation["errors"])

    assert validation["verdict"] == "REJECT"
    assert "latency_scope must be e2e" in messages
    assert "fake/prior AP" in messages
    assert "INT8/mixed rows must use TVM" in messages


def test_validator_requires_real_pruned_checkpoint_manifest_and_dim_preservation():
    mod = _load_module()
    report = _valid_report()
    bad = report["rows"][1]
    bad["checkpoint_path"] = ""
    bad["manifest_path"] = ""
    bad["prune_manifest"] = {
        "hmsa_keep_heads": 4,
        "mswin_keep_heads": {"ws4": 8, "ws8": 4, "ws16": 2},
        "dim_256_preserved": False,
    }

    validation = mod.validate_stop_a_report(report)
    messages = "\n".join(issue["message"] for issue in validation["errors"])

    assert validation["verdict"] == "REJECT"
    assert "pruned rows require checkpoint_path" in messages
    assert "pruned rows require manifest_path" in messages
    assert "dim=256 must be preserved" in messages


def test_validator_flags_ap_gain_without_protocol_audit():
    mod = _load_module()
    report = _valid_report()
    row = report["rows"][1]
    row["ap50"] = 0.72
    row["ap70"] = 0.53
    row["delta_ap50"] = 0.0097
    row["delta_ap70"] = 0.0088

    validation = mod.validate_stop_a_report(report)
    warnings = "\n".join(issue["message"] for issue in validation["warnings"])

    assert validation["verdict"] == "REVISE"
    assert "AP improves over baseline" in warnings
    assert "ap_gain_audit" in warnings


def test_blocker_validator_requires_specific_blocker_and_partial_evidence():
    mod = _load_module()
    blocker = """
    # blocker
    full attention TVM INT8/mixed blocked at Relax lowering for
    mswin_bwa_full_attention: softmax boundary after int32 QK requires dequant.
    baseline e2e latency/AP evidence: results/base.json
    attention-p50-fp16 e2e latency/AP evidence: results/p50.json
    """

    validation = mod.validate_blocker_markdown(blocker)

    assert validation["verdict"] == "ACTIONABLE_BLOCKER"
    assert validation["errors"] == []


def test_blocker_validator_accepts_explicit_not_implemented_hmsa_minimal_target():
    mod = _load_module()
    blocker = """
    # Stop-B
    target: hmsa_full_attention
    backend: TVM
    stage: NOT_IMPLEMENTED
    op: HGTCavAttention relation einsum with per-type q/k/v/a ModuleList dispatch
    minimal HMSA Relax/TIR target: B=1,L=2,H=64,W=128,C=256,heads=8,dim_head=32
    baseline e2e latency/AP evidence: logs/base.json
    attention-p50-fp16 e2e latency/AP evidence: logs/p50.json
    """

    validation = mod.validate_blocker_markdown(blocker)

    assert validation["verdict"] == "ACTIONABLE_BLOCKER"
    assert validation["errors"] == []
