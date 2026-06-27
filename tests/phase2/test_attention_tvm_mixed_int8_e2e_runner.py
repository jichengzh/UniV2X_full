from __future__ import annotations

import importlib.util
import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "attention_tvm_mixed_int8_e2e_runner.py"
    spec = importlib.util.spec_from_file_location("attention_tvm_mixed_int8_e2e_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _measurement_payload(tmp_path: Path) -> dict:
    latency_log = tmp_path / "latency.json"
    ap_log = tmp_path / "ap.json"
    tvm_log = tmp_path / "tvm_runtime.json"
    coverage_log = tmp_path / "type_dispatch_coverage.json"
    latency_log.write_text('{"scope": "e2e"}\n', encoding="utf-8")
    ap_log.write_text('{"scope": "DAIR val"}\n', encoding="utf-8")
    tvm_log.write_text('{"runtime": "TVM Relax VM", "total_tvm_call_count": 24}\n', encoding="utf-8")
    coverage_log.write_text('{"sample_count": 1789, "observed_type_orders": [[0, 1]]}\n', encoding="utf-8")
    return {
        "schema_version": "attention_tvm_mixed_int8_e2e_measurement_v1",
        "status": "OK",
        "config": "attention-p50-int8/mixed",
        "tvm_scope": "all",
        "quant_policy": "w8a8",
        "attention_prune_pct": 50,
        "quant": "int8/mixed",
        "quant_backend": "TVM Relax int8/mixed",
        "checkpoint_path": "models/v2xvit_attention_t1/attention_p50_shortft_steps100_lr0.0001_seed20260623.pth",
        "manifest_path": "models/v2xvit_attention_t1/attention_p50_shortft_steps100_manifest_v1.json",
        "finetune": "100_steps_lr1e-4_seed20260623_all_params",
        "dataset_split": "DAIR val",
        "n_samples": 1789,
        "latency": {
            "status": "OK",
            "latency_scope": "e2e",
            "e2e_latency_ms": 34.8,
            "speedup": 1.1113,
            "latency_command": "CUDA_VISIBLE_DEVICES=6 python scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py --run-latency",
            "latency_log": str(latency_log),
        },
        "ap": {
            "status": "OK",
            "ap50": 0.695,
            "ap70": 0.511,
            "delta_ap50": -0.0151,
            "delta_ap70": -0.0106,
            "ap_command": "CUDA_VISIBLE_DEVICES=6 python scripts/phase2/attention_tvm_mixed_int8_e2e_runner.py --run-ap",
            "ap_log": str(ap_log),
            "ap_source": "real_dair_val_eval",
        },
        "prune_manifest": {
            "hmsa_keep_heads": 4,
            "mswin_keep_heads": {"ws4": 8, "ws8": 4, "ws16": 2},
            "dim_256_preserved": True,
        },
        "tvm": {
            "artifact_path": str(tvm_log),
            "runtime": "TVM Relax VM",
            "mixed_precision_policy": "int8 qkv/out projections with fp16 softmax/layernorm",
            "runtime_stats": {
                "total_tvm_call_count": 24,
                "mswin_call_count": 18,
                "hmsa_call_count": 6,
                "fallback_call_count": 0,
            },
        },
        "dynamic_type_dispatch_coverage": {
            "status": "static_type_order_observed",
            "covers_dynamic_type_dispatch": False,
            "observed_type_orders": [[0, 1]],
            "sample_count": 1789,
            "hmsa_call_count": 5367,
            "log_path": str(coverage_log),
        },
    }


def test_build_row_from_measurements_preserves_stop_a_contract(tmp_path):
    mod = _load_module()
    measurements = _measurement_payload(tmp_path)

    row = mod.build_row_from_measurements(measurements)

    assert row["config"] == "attention-p50-int8/mixed"
    assert row["attention_prune_pct"] == 50
    assert row["quant"] == "int8/mixed"
    assert row["quant_backend"] == "TVM Relax int8/mixed"
    assert row["checkpoint_path"].endswith("attention_p50_shortft_steps100_lr0.0001_seed20260623.pth")
    assert row["manifest_path"].endswith("attention_p50_shortft_steps100_manifest_v1.json")
    assert row["latency_scope"] == "e2e"
    assert row["dataset_split"] == "DAIR val"
    assert row["n_samples"] == 1789
    assert row["tvm_scope"] == "all"
    assert row["quant_policy"] == "w8a8"
    assert row["e2e_latency_ms"] == 34.8
    assert row["speedup"] == 1.1113
    assert row["ap50"] == 0.695
    assert row["ap70"] == 0.511
    assert row["delta_ap50"] == -0.0151
    assert row["delta_ap70"] == -0.0106


def test_build_row_records_commands_logs_tvm_backend_and_dynamic_hmsa_metadata(tmp_path):
    mod = _load_module()
    measurements = _measurement_payload(tmp_path)

    row = mod.build_row_from_measurements(measurements)

    assert "attention_tvm_mixed_int8_e2e_runner.py" in row["latency_command"]
    assert Path(row["latency_log"]).exists()
    assert "attention_tvm_mixed_int8_e2e_runner.py" in row["ap_command"]
    assert Path(row["ap_log"]).exists()
    assert row["tvm_evidence"]["runtime"] == "TVM Relax VM"
    assert Path(row["tvm_evidence"]["artifact_path"]).exists()
    assert row["dynamic_type_dispatch_coverage"]["covers_dynamic_type_dispatch"] is False
    assert row["dynamic_type_dispatch_coverage"]["observed_type_orders"] == [[0, 1]]


def test_mixed_precision_policy_names_disabled_hmsa_for_mswin_scope():
    mod = _load_module()

    policy = mod.mixed_precision_policy_description("mswin", "w8a16")

    assert "scope=mswin" in policy
    assert "MSwin q/k/v W8A16" in policy
    assert "HMSA TVM disabled" in policy


def test_build_row_requires_and_preserves_ap_gain_audit_for_positive_delta(tmp_path):
    mod = _load_module()
    measurements = _measurement_payload(tmp_path)
    measurements["ap"]["delta_ap50"] = 0.12
    measurements["ap"]["delta_ap70"] = 0.08

    with pytest.raises(ValueError, match="ap_gain_audit is required"):
        mod.build_row_from_measurements(measurements)

    measurements["ap_gain_audit"] = {
        "same_eval_protocol": True,
        "same_dataset_split": True,
        "same_checkpoint_family": True,
        "same_thresholds": True,
        "finetune_epochs": "steps=500",
        "learning_rate": 0.0001,
        "seed": 20260624,
    }
    row = mod.build_row_from_measurements(measurements)

    assert row["ap_gain_audit"]["finetune_epochs"] == "steps=500"
    assert row["ap_gain_audit"]["seed"] == 20260624


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["latency"].update({"latency_scope": "attention_subnet"}), "latency_scope must be e2e"),
        (lambda payload: payload["latency"].update({"latency_scope": "direct_matmul"}), "latency_scope must be e2e"),
        (lambda payload: payload["ap"].update({"ap_source": "fake_prior"}), "fake/prior AP"),
        (lambda payload: payload.update({"quant_backend": "TensorRT INT8"}), "TVM Relax int8/mixed"),
        (lambda payload: payload.update({"n_samples": 16}), "n_samples must be 1789"),
        (lambda payload: payload["tvm"].update({"runtime_stats": {"total_tvm_call_count": 0}}), "TVM runtime call count"),
        (lambda payload: payload["dynamic_type_dispatch_coverage"].update({"sample_count": 16}), "coverage sample_count must be 1789"),
        (lambda payload: payload.update({"status": "FAILED"}), "measurement status must be OK"),
        (lambda payload: payload["latency"].update({"status": "FAILED"}), "latency.status must be OK"),
        (lambda payload: payload["ap"].update({"status": "FAILED"}), "ap.status must be OK"),
        (lambda payload: payload["tvm"]["runtime_stats"].update({"fallback_call_count": 1}), "TVM fallback_call_count must be 0"),
        (lambda payload: payload.update({"tvm_scope": "all"}) or payload["tvm"]["runtime_stats"].update({"hmsa_call_count": 0}), "all scope requires both MSwin and HMSA TVM calls"),
        (lambda payload: payload.update({"tvm_scope": "hmsa"}) or payload["tvm"]["runtime_stats"].update({"mswin_call_count": 24, "hmsa_call_count": 0}), "hmsa scope requires HMSA TVM calls"),
        (lambda payload: payload.update({"tvm_scope": "mswin"}) or payload["tvm"]["runtime_stats"].update({"mswin_call_count": 0, "hmsa_call_count": 24}), "mswin scope requires MSwin TVM calls"),
    ],
)
def test_build_row_rejects_subnet_fake_direct_matmul_and_non_tvm_measurements(tmp_path, mutate, message):
    mod = _load_module()
    measurements = _measurement_payload(tmp_path)
    mutate(measurements)

    with pytest.raises(ValueError, match=message):
        mod.build_row_from_measurements(measurements)


def test_from_measurements_cli_writes_row_without_inventing_metrics(tmp_path):
    mod = _load_module()
    measurements_path = tmp_path / "measurements.json"
    out_path = tmp_path / "row.json"
    measurements_path.write_text(json.dumps(_measurement_payload(tmp_path)), encoding="utf-8")

    assert mod.main(["--from-measurements", str(measurements_path), "--out-row", str(out_path)]) == 0

    row = json.loads(out_path.read_text(encoding="utf-8"))
    assert row["e2e_latency_ms"] == 34.8
    assert row["ap50"] == 0.695
    assert row["measurement_source"] == str(measurements_path)


def test_record_type_coverage_cli_dispatches_without_row_generation(monkeypatch, tmp_path):
    mod = _load_module()
    out_log = tmp_path / "coverage.json"
    called = {}

    def fake_record(**kwargs):
        called.update(kwargs)
        return {
            "status": "OK",
            "observed_type_orders": [[0, 0]],
            "covers_dynamic_type_dispatch": False,
            "log_path": str(out_log),
        }

    monkeypatch.setattr(mod, "record_hmsa_type_dispatch_coverage", fake_record)

    rc = mod.main([
        "--record-type-coverage",
        "--device",
        "cuda:6",
        "--eval-samples",
        "8",
        "--num-workers",
        "0",
        "--out-log",
        str(out_log),
    ])

    assert rc == 0
    assert called["device"] == "cuda:6"
    assert called["eval_samples"] == 8
    assert called["num_workers"] == 0
    assert called["out_log"] == str(out_log)


def test_mswin_tvm_pilot_cli_dispatches_without_final_row(monkeypatch, tmp_path):
    mod = _load_module()
    out_log = tmp_path / "mswin_pilot.json"
    called = {}

    def fake_pilot(**kwargs):
        called.update(kwargs)
        return {
            "status": "OK",
            "latency_scope": "e2e_forward_model_with_mswin_tvm_subgraph",
            "sample_count": 2,
            "log_path": str(out_log),
        }

    monkeypatch.setattr(mod, "run_mswin_tvm_qkv_int8_forward_pilot", fake_pilot)

    rc = mod.main([
        "--run-mswin-tvm-pilot",
        "--device",
        "cuda:0",
        "--eval-samples",
        "2",
        "--num-workers",
        "0",
        "--out-log",
        str(out_log),
    ])

    assert rc == 0
    assert called["device"] == "cuda:0"
    assert called["eval_samples"] == 2
    assert called["num_workers"] == 0
    assert called["out_log"] == str(out_log)


def test_tvm_e2e_measurement_cli_dispatches_without_row_generation(monkeypatch, tmp_path):
    mod = _load_module()
    out_measurement = tmp_path / "measurement.json"
    called = {}

    def fake_measurement(**kwargs):
        called.update(kwargs)
        return {
            "status": "OK",
            "config": "attention-p50-int8/mixed",
            "n_samples": 1789,
            "measurement_path": str(out_measurement),
        }

    monkeypatch.setattr(mod, "run_tvm_mixed_int8_e2e_measurement", fake_measurement)

    rc = mod.main([
        "--run-tvm-e2e-measurement",
        "--device",
        "cuda:0",
        "--eval-samples",
        "1789",
        "--latency-samples",
        "30",
        "--latency-warmup",
        "5",
        "--tvm-scope",
        "hmsa",
        "--quant-policy",
        "w8a16",
        "--num-workers",
        "0",
        "--out-measurement",
        str(out_measurement),
    ])

    assert rc == 0
    assert called["device"] == "cuda:0"
    assert called["eval_samples"] == 1789
    assert called["latency_samples"] == 30
    assert called["latency_warmup"] == 5
    assert called["tvm_scope"] == "hmsa"
    assert called["quant_policy"] == "w8a16"
    assert called["num_workers"] == 0
    assert called["out_measurement"] == str(out_measurement)


def test_h800_bootstrap_imports_tvm_before_torch(monkeypatch):
    mod = _load_module()
    imported: list[str] = []

    fake_tvm = types.SimpleNamespace(__version__="0.20.test")
    fake_numpy = types.SimpleNamespace(__version__="1.26.test")
    fake_torch = types.SimpleNamespace(__version__="2.1.test")

    def fake_import_module(name: str):
        imported.append(name)
        if name == "tvm":
            assert "/tmp/tvm_site" in sys.path
            assert "/tmp/torch_site" not in sys.path
            return fake_tvm
        if name == "numpy":
            assert imported == ["tvm", "numpy"]
            assert "/tmp/torch_site" in sys.path
            return fake_numpy
        if name == "torch":
            assert imported == ["tvm", "numpy", "torch"]
            assert "/tmp/torch_site" in sys.path
            assert "/tmp/heal" in sys.path
            return fake_torch
        raise AssertionError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    info = mod.bootstrap_h800_tvm_then_torch(
        tvm_site="/tmp/tvm_site",
        torch_site="/tmp/torch_site",
        heal_root="/tmp/heal",
    )

    assert imported == ["tvm", "numpy", "torch"]
    assert info["tvm_version"] == "0.20.test"
    assert info["numpy_version"] == "1.26.test"
    assert info["torch_version"] == "2.1.test"
