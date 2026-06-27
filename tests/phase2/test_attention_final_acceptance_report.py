from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(name: str):
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_e2e_report(tmp_path: Path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        """
        {
          "attention_prune_manifest": {
            "method": "manual_structured_attention_head_pruning",
            "prune_rate_pct": 50,
            "preserve_residual_dim": 256,
            "counts": {"hmsa_head": 3, "mswin_head": 9},
            "entries": [
              {"family": "hmsa_head", "keep_heads": [0, 1, 2, 3], "preserve_output_dim": 256},
              {"family": "mswin_head", "window_size": 4, "keep_heads": [0, 1, 2, 3, 4, 5, 6, 7], "preserve_output_dim": 256},
              {"family": "mswin_head", "window_size": 8, "keep_heads": [0, 1, 2, 3], "preserve_output_dim": 256},
              {"family": "mswin_head", "window_size": 16, "keep_heads": [0, 1], "preserve_output_dim": 256}
            ]
          },
          "short_finetune_manifest": {
            "steps": 100,
            "lr": 0.0001,
            "seed": 20260623,
            "train_scope": "all"
          }
        }
        """,
        encoding="utf-8",
    )
    return {
        "schema_version": "attention_e2e_checkpoint_eval_v1",
        "precision": "fp16",
        "eval_samples": 1789,
        "rows": [
            {
                "config": "baseline",
                "checkpoint_path": "/ckpt/base.pth",
                "manifest_path": None,
                "attention_prune_pct": 0,
                "quant": "fp16",
                "finetune": "official_ckpt",
                "latency_scope": "pytorch_model_forward_e2e",
                "latency_p50_ms": 38.674,
                "ap50": 0.7101,
                "ap70": 0.5216,
                "dataset_split": "DAIR val",
                "eval_samples": 1789,
                "latency_log": "/logs/base_latency.json",
                "ap_log": "/logs/base_ap.json",
                "speedup_vs_baseline": 1.0,
                "delta_ap50": 0.0,
                "delta_ap70": 0.0,
            },
            {
                "config": "attention-p50-shortft-fp16",
                "checkpoint_path": "/ckpt/p50_shortft.pth",
                "manifest_path": str(manifest),
                "attention_prune_pct": 50,
                "quant": "fp16",
                "finetune": "100_steps_lr1e-4_seed20260623_all_params",
                "latency_scope": "pytorch_model_forward_e2e",
                "latency_p50_ms": 35.8636,
                "ap50": 0.6973,
                "ap70": 0.5132,
                "dataset_split": "DAIR val",
                "eval_samples": 1789,
                "latency_log": "/logs/p50_latency.json",
                "ap_log": "/logs/p50_ap.json",
                "speedup_vs_baseline": 1.0784,
                "delta_ap50": -0.0128,
                "delta_ap70": -0.0084,
            },
        ],
    }


def _sample_subnet_report():
    return {
        "schema_version": "attention_subnet_accel_v1",
        "rows": [
            {
                "config": "attention-subnet-p50-fp16",
                "subnet_p50_ms": 0.443765,
                "speedup_vs_base_fp16": 1.7783,
                "speedup_vs_same_prune_fp16": 1.0,
            },
            {
                "config": "attention-subnet-p50-mixed-int8",
                "subnet_p50_ms": 0.462336,
                "speedup_vs_base_fp16": 1.7069,
                "speedup_vs_same_prune_fp16": 0.9598,
            },
        ],
    }


def test_acceptance_report_blocks_until_tvm_mixed_int8_e2e_row_exists(tmp_path):
    mod = _load_module("attention_final_acceptance_report")

    report = mod.build_acceptance_report(_sample_e2e_report(tmp_path), _sample_subnet_report())
    rows = {row["config"]: row for row in report["rows"]}

    assert report["gate_status"] == "BLOCKED_TVM_MIXED_INT8_E2E_MISSING"
    assert rows["attention-p50-fp16"]["source_config"] == "attention-p50-shortft-fp16"
    assert rows["attention-p50-fp16"]["speedup"] == 1.0784
    assert rows["attention-p50-fp16"]["prune_manifest"]["dim_256_preserved"] is True
    assert rows["attention-p50-fp16"]["prune_manifest"]["mswin_keep_heads"]["ws16"] == 2
    assert rows["attention-p50-int8/mixed"]["status"] == "MISSING_TVM_E2E"
    assert rows["attention-p50-int8/mixed"]["blocked_by"] == "full_model_tvm_mixed_int8_runner"
    assert report["subnet_summary"]["p50_mixed_int8_speedup_vs_same_prune_fp16"] == 0.9598
    assert report["stop_a_ready"] is False


def test_acceptance_report_adds_ap_gain_audit_when_shortft_beats_baseline(tmp_path):
    mod = _load_module("attention_final_acceptance_report")
    e2e_report = _sample_e2e_report(tmp_path)
    shortft = next(row for row in e2e_report["rows"] if row["config"] == "attention-p50-shortft-fp16")
    shortft["ap50"] = 0.7301
    shortft["ap70"] = 0.5416
    shortft["delta_ap50"] = 0.02
    shortft["delta_ap70"] = 0.02

    report = mod.build_acceptance_report(e2e_report, _sample_subnet_report())
    row = next(item for item in report["rows"] if item["config"] == "attention-p50-fp16")

    assert row["ap_gain_audit"]["same_eval_protocol"] is True
    assert row["ap_gain_audit"]["same_dataset_split"] is True
    assert row["ap_gain_audit"]["same_checkpoint_family"] is True
    assert row["ap_gain_audit"]["same_thresholds"] is True
    assert row["ap_gain_audit"]["finetune_epochs"] == "steps=100"
    assert row["ap_gain_audit"]["learning_rate"] == 0.0001
    assert row["ap_gain_audit"]["seed"] == 20260623


def test_blocked_acceptance_report_is_rejected_by_stop_a_validator(tmp_path):
    mod = _load_module("attention_final_acceptance_report")
    validator = _load_module("attention_e2e_pq_validator")

    report = mod.build_acceptance_report(_sample_e2e_report(tmp_path), _sample_subnet_report())
    validation = validator.validate_stop_a_report(report)
    messages = "\n".join(issue["message"] for issue in validation["errors"])

    assert validation["verdict"] == "REJECT"
    assert "e2e_latency_ms is required" in messages


def test_acceptance_report_can_accept_future_complete_tvm_row(tmp_path):
    mod = _load_module("attention_final_acceptance_report")
    validator = _load_module("attention_e2e_pq_validator")
    base = mod.build_acceptance_report(_sample_e2e_report(tmp_path), _sample_subnet_report())
    fp16 = next(row for row in base["rows"] if row["config"] == "attention-p50-fp16")
    tvm_row = {
        **fp16,
        "config": "attention-p50-int8/mixed",
        "quant": "int8/mixed",
        "quant_backend": "TVM Relax int8/mixed",
        "e2e_latency_ms": 34.8,
        "speedup": 1.1113,
        "ap50": 0.695,
        "ap70": 0.511,
        "delta_ap50": -0.0151,
        "delta_ap70": -0.0106,
        "latency_log": "/logs/tvm_latency.json",
        "ap_log": "/logs/tvm_ap.json",
    }

    report = mod.build_acceptance_report(
        _sample_e2e_report(tmp_path),
        _sample_subnet_report(),
        tvm_e2e_row=tvm_row,
    )
    validation = validator.validate_stop_a_report(report)

    assert report["gate_status"] == "READY_FOR_STOP_A_VALIDATION"
    assert report["stop_a_ready"] is True
    assert "Stop-A row is ready" in report["next_stop_target"]
    assert validation["verdict"] == "ACCEPTABLE_E2E_SCHEMA"


def test_markdown_keeps_missing_tvm_row_visible(tmp_path):
    mod = _load_module("attention_final_acceptance_report")

    report = mod.build_acceptance_report(_sample_e2e_report(tmp_path), _sample_subnet_report())
    md = mod.render_markdown(report)

    assert "attention-p50-int8/mixed" in md
    assert "MISSING_TVM_E2E" in md
    assert "Stop-A ready: `False`" in md


def test_markdown_includes_completed_tvm_scope_and_runtime_evidence(tmp_path):
    mod = _load_module("attention_final_acceptance_report")
    base = mod.build_acceptance_report(_sample_e2e_report(tmp_path), _sample_subnet_report())
    fp16 = next(row for row in base["rows"] if row["config"] == "attention-p50-fp16")
    tvm_row = {
        **fp16,
        "config": "attention-p50-int8/mixed",
        "quant": "int8/mixed",
        "quant_backend": "TVM Relax int8/mixed",
        "e2e_latency_ms": 20.0,
        "speedup": 1.2,
        "ap50": 0.7,
        "ap70": 0.52,
        "delta_ap50": -0.01,
        "delta_ap70": -0.001,
        "latency_log": "/logs/tvm_latency.json",
        "ap_log": "/logs/tvm_ap.json",
        "tvm_scope": "mswin",
        "quant_policy": "w8a16",
        "tvm_evidence": {
            "runtime_stats": {
                "total_tvm_call_count": 16416,
                "mswin_call_count": 16416,
                "hmsa_call_count": 0,
                "fallback_call_count": 0,
            },
            "covered_modules": {"mswin": ["a", "b"], "hmsa": []},
        },
    }

    report = mod.build_acceptance_report(
        _sample_e2e_report(tmp_path),
        _sample_subnet_report(),
        tvm_e2e_row=tvm_row,
    )
    md = mod.render_markdown(report)

    assert "TVM E2E Evidence" in md
    assert "`mswin`" in md
    assert "`w8a16`" in md
    assert "`16416`" in md
    assert "fallback" in md.lower()
