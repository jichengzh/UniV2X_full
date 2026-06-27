from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "attention_subnet_accel_report.py"
    spec = importlib.util.spec_from_file_location("attention_subnet_accel_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sample_tvm_report():
    return {
        "backend_results": {
            "mswin_bwa_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.456245},
                "mixed_int8": {"status": "OK", "p50_ms": 0.465418},
            },
            "mswin_bwa_p50_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.244672},
                "mixed_int8": {"status": "OK", "p50_ms": 0.257077},
            },
            "hmsa_full_attention": {
                "scope": "full_attention_hmsa_relation_core_mixed_int8",
                "fp16": {"status": "OK", "p50_ms": 0.332906},
                "mixed_int8": {"status": "OK", "p50_ms": 0.348373},
            },
            "hmsa_p50_full_attention": {
                "scope": "full_attention_hmsa_relation_core_mixed_int8",
                "fp16": {"status": "OK", "p50_ms": 0.199093},
                "mixed_int8": {"status": "OK", "p50_ms": 0.205259},
            },
        }
    }


def _sample_static_hmsa_tvm_report():
    report = _sample_tvm_report()
    report["source_path"] = "results/attention_full_tvm_bench_v3_static.json"
    static_plan = {
        "scope": "hmsa_static_2agent_qkv_relation_out",
        "covers_qkv_projection": True,
        "covers_output_projection": True,
        "covers_dynamic_type_dispatch": False,
    }
    report["backend_results"]["hmsa_full_attention"].update(
        {
            "scope": "full_attention_hmsa_static_2agent_qkv_relation_core_mixed_int8",
            "caveat": "static caveat",
            "static_param_plan": static_plan,
        }
    )
    report["backend_results"]["hmsa_p50_full_attention"].update(
        {
            "scope": "full_attention_hmsa_static_2agent_qkv_relation_core_mixed_int8",
            "caveat": "static caveat",
            "static_param_plan": static_plan,
        }
    )
    return report


def test_build_subnet_rows_separates_pruning_and_mixed_int8_effects():
    mod = _load_module()

    report = mod.build_attention_subnet_report(_sample_tvm_report())

    rows = {row["config"]: row for row in report["rows"]}
    assert rows["attention-subnet-base-fp16"]["subnet_p50_ms"] == 0.789151
    assert rows["attention-subnet-p50-fp16"]["subnet_p50_ms"] == 0.443765
    assert rows["attention-subnet-p50-fp16"]["speedup_vs_base_fp16"] == 1.7783
    assert rows["attention-subnet-p50-mixed-int8"]["subnet_p50_ms"] == 0.462336
    assert rows["attention-subnet-p50-mixed-int8"]["speedup_vs_base_fp16"] == 1.7069
    assert rows["attention-subnet-p50-mixed-int8"]["speedup_vs_same_prune_fp16"] == 0.9598
    assert report["conclusion"]["pruning_effect"] == "POSITIVE_SUBNET_SPEEDUP"
    assert report["conclusion"]["mixed_int8_effect"] == "NEGATIVE_VS_SAME_PRUNE_FP16"


def test_static_hmsa_scope_is_preserved_in_report_metadata_and_rows():
    mod = _load_module()

    report = mod.build_attention_subnet_report(_sample_static_hmsa_tvm_report())
    md = mod.render_markdown(report)

    assert report["component_scope"]["hmsa_scope"] == "full_attention_hmsa_static_2agent_qkv_relation_core_mixed_int8"
    assert report["component_scope"]["hmsa_static_param_plan"]["covers_qkv_projection"] is True
    assert report["component_scope"]["hmsa_static_param_plan"]["covers_output_projection"] is True
    assert report["component_scope"]["hmsa_static_param_plan"]["covers_dynamic_type_dispatch"] is False
    assert all("hmsa_static_2agent" in row["hmsa_scope"] for row in report["rows"])
    assert "HMSA static 2-agent" in md
    assert "Dynamic HMSA type dispatch is not covered" in md


def test_markdown_includes_subnet_scope_and_caveat():
    mod = _load_module()

    report = mod.build_attention_subnet_report(_sample_tvm_report())
    md = mod.render_markdown(report)

    assert "attention subnet" in md
    assert "not full-model e2e" in md
    assert "HMSA minimal relation core" in md
    assert "1.7069" in md
