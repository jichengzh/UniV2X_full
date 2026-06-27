from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "t1_attention_pq_feasibility.py"
    spec = importlib.util.spec_from_file_location("t1_attention_pq_feasibility", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _encoder_cfg():
    return {
        "depth": 3,
        "num_blocks": 1,
        "cav_att_config": {
            "dim": 256,
            "heads": 8,
            "dim_head": 32,
            "use_hetero": True,
        },
        "pwindow_att_config": {
            "dim": 256,
            "window_size": [4, 8, 16],
            "heads": [16, 8, 4],
            "dim_head": [16, 32, 64],
            "relative_pos_embedding": True,
            "fusion_method": "naive",
        },
        "feed_forward": {"mlp_dim": 256},
    }


def test_manual_attention_scanner_enumerates_hmsa_and_mswin_structured_groups():
    mod = _load_module()

    scanner = mod.scan_manual_attention_pruning_groups(_encoder_cfg())

    hmsa = [g for g in scanner["groups"] if g["family"] == "hmsa_head"]
    mswin = [g for g in scanner["groups"] if g["family"] == "mswin_head"]
    depth = [g for g in scanner["groups"] if g["family"] == "encoder_depth"]

    assert scanner["status"] == "PASS"
    assert len(hmsa) == 3
    assert len(mswin) == 9
    assert len(depth) == 3

    first_hmsa = hmsa[0]
    assert first_hmsa["structured_unit"] == "head"
    assert first_hmsa["heads"] == 8
    assert first_hmsa["dim_head"] == 32
    assert "relation_att[:, keep_heads, :, :]" in first_hmsa["slice_rules"]
    assert "relation_msg[:, keep_heads, :, :]" in first_hmsa["slice_rules"]
    assert any("q_linears" in member for member in first_hmsa["members"])
    assert any("a_linears" in member for member in first_hmsa["members"])

    ws16 = [g for g in mswin if g["layer"] == 2 and g["window_size"] == 16][0]
    assert ws16["heads"] == 4
    assert ws16["dim_head"] == 64
    assert ws16["inner_dim"] == 256
    assert "to_qkv.out_features -> 3 * keep_heads * dim_head" in ws16["slice_rules"]
    assert "to_out[0].in_features -> keep_heads * dim_head" in ws16["slice_rules"]


def test_p_axis_gate_requires_manual_attention_scanner_not_only_candidates():
    mod = _load_module()
    groups = {
        "embed_dim": {"status": "candidate", "count": 1},
        "heads": {"status": "candidate", "count": 4},
        "depth": {"status": "candidate", "count": 3},
    }
    checks = {"feedforward_linear_depgraph": {"status": "PASS"}}

    gate = mod.gate_p_axis(groups, checks, manual_scanner=None)

    assert gate["verdict"] == "P_axis_MANUAL_SCANNER_REQUIRED"
    assert "manual_hmsa_mswin_scanner" in gate["missing"]


def test_tvm_q_gate_rejects_tensor_rt_and_accepts_direct_int8_subops():
    mod = _load_module()
    backend = {
        "linear_qkv": {
            "backend": "TensorRT",
            "fp16": {"status": "OK", "p50_ms": 0.12},
            "int8": {"status": "OK", "p50_ms": 0.07},
        },
        "linear_qkv_direct": {
            "backend": "TVM",
            "scope": "direct_matmul",
            "fp16": {"status": "OK", "p50_ms": 0.048},
            "int8": {"status": "OK", "p50_ms": 0.035},
        },
        "mswin_bwa_onnx": {
            "backend": "TVM",
            "scope": "full_attention_onnx_fp16",
            "fp16": {"status": "OK", "p50_ms": 0.19},
            "int8": {"status": "BLOCKED", "reason": "no direct full-attention int8 lowering"},
        },
    }

    gate = mod.gate_q_axis(backend)

    assert gate["verdict"] == "Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED"
    assert gate["non_tvm_ignored"] == ["linear_qkv"]
    assert gate["direct_int8_targets_ok"] == ["linear_qkv_direct"]
    assert gate["full_attention_blocked"] == ["mswin_bwa_onnx"]


def test_overall_gate_keeps_t2_blocked_until_tvm_q_and_manual_scanner_are_ready():
    mod = _load_module()
    p_gate = {"verdict": "P_axis_MANUAL_SCANNER_REQUIRED"}
    q_gate = {"verdict": "Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED"}

    overall = mod.gate_overall(p_gate, q_gate)

    assert overall["verdict"] == "T1_PQ_INCOMPLETE_DO_NOT_START_T2"
    assert overall["next"] == "Complete T1 manual scanner and TVM attention quantization evidence"


def test_coupling_table_records_prune_quant_speed_accuracy_and_sources():
    mod = _load_module()
    backend = {
        "linear_qkv_direct": {
            "backend": "TVM",
            "scope": "direct_matmul",
            "fp16": {"status": "OK", "p50_ms": 0.048},
            "int8": {"status": "OK", "p50_ms": 0.035},
        }
    }
    accuracy = {
        "q_only_fake_quant": {
            "prune_rate_pct": 0,
            "quant": "int8",
            "ap50": 0.6988,
            "ap70": 0.5101,
            "source": "results/coupling_map/C4_QgranxP_v2xvit.json",
            "status": "SIMULATED_PRIOR_NOT_TRUE_TVM",
        }
    }

    table = mod.build_attention_coupling_table(backend, accuracy)

    assert table[0]["prune_rate_pct"] == 0
    assert table[0]["quant"] == "int8"
    assert table[0]["latency_backend"] == "TVM"
    assert table[0]["speedup"] == 1.3714
    assert table[0]["accuracy_ap50"] == 0.6988
    assert table[0]["accuracy_status"] == "SIMULATED_PRIOR_NOT_TRUE_TVM"


def test_markdown_report_names_tvm_and_warns_when_accuracy_is_only_a_prior():
    mod = _load_module()
    report = {
        "gate": {
            "p_axis": {"verdict": "P_axis_READY_FOR_T2_SCANNER_ONLY"},
            "q_axis": {
                "verdict": "Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED",
                "full_attention_blocked": ["mswin_bwa_onnx"],
                "speedups": {"linear_qkv_direct": 1.3714},
            },
            "overall": {
                "verdict": "T1_PQ_INCOMPLETE_DO_NOT_START_T2",
                "next": "Complete T1 manual scanner and TVM attention quantization evidence",
            },
        },
        "backend_results": {
            "linear_qkv_direct": {
                "backend": "TVM",
                "scope": "direct_matmul",
                "fp16": {"status": "OK", "p50_ms": 0.048},
                "int8": {"status": "OK", "p50_ms": 0.035},
            },
            "mswin_bwa_onnx": {
                "backend": "TVM",
                "scope": "full_attention_onnx_fp16",
                "fp16": {"status": "OK", "p50_ms": 0.19},
                "int8": {"status": "BLOCKED", "reason": "no direct full-attention int8 lowering"},
            },
        },
        "coupling_table": [
            {
                "config": "attention_q_only_prior",
                "prune_rate_pct": 0,
                "quant": "int8",
                "speedup": 1.3714,
                "accuracy_ap50": 0.6988,
                "accuracy_ap70": 0.5101,
                "accuracy_status": "SIMULATED_PRIOR_NOT_TRUE_TVM",
            }
        ],
    }

    md = mod.build_markdown_report(report)

    assert "TensorRT" not in md
    assert "TVM" in md
    assert "SIMULATED_PRIOR_NOT_TRUE_TVM" in md
    assert "DO_NOT_START_T2" in md
