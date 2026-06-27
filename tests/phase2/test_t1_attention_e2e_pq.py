from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "t1_attention_e2e_pq.py"
    spec = importlib.util.spec_from_file_location("t1_attention_e2e_pq", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _with_heal_imports():
    heal_root = "/home/jichengzhi/heal_research/HEAL"
    if heal_root not in sys.path:
        sys.path.insert(0, heal_root)


def test_mswin_p50_surgery_preserves_output_shape_and_records_manifest():
    _with_heal_imports()
    mod = _load_module()
    from opencood.models.sub_modules.mswin import BaseWindowAttention

    attn = BaseWindowAttention(
        dim=256,
        heads=8,
        dim_head=32,
        drop_out=0.0,
        window_size=4,
        relative_pos_embedding=True,
    ).eval()
    x = torch.randn(1, 2, 8, 8, 256)
    before = attn(x)

    entry = mod.prune_mswin_bwa_heads(attn, keep_heads=[0, 1, 2, 3])
    after = attn(x)

    assert before.shape == after.shape == (1, 2, 8, 8, 256)
    assert attn.heads == 4
    assert attn.to_qkv.out_features == 3 * 4 * 32
    assert attn.to_out[0].in_features == 4 * 32
    assert entry["family"] == "mswin_head"
    assert entry["prune_rate_pct"] == 50
    assert entry["preserve_output_dim"] == 256
    assert entry["keep_heads"] == [0, 1, 2, 3]


def test_hmsa_p50_surgery_preserves_output_shape_and_records_manifest():
    _with_heal_imports()
    mod = _load_module()
    from opencood.models.sub_modules.hmsa import HGTCavAttention

    attn = HGTCavAttention(
        dim=256,
        heads=8,
        num_types=2,
        num_relations=4,
        dim_head=32,
        dropout=0.0,
    ).eval()
    x = torch.randn(1, 2, 4, 4, 256)
    mask = torch.ones(1, 1, 1, 1, 2)
    prior = torch.zeros(1, 2, 4, 4, 3)
    before = attn(x, mask, prior)

    entry = mod.prune_hmsa_heads(attn, keep_heads=[0, 1, 2, 3])
    after = attn(x, mask, prior)

    assert before.shape == after.shape == (1, 2, 4, 4, 256)
    assert attn.heads == 4
    assert attn.q_linears[0].out_features == 4 * 32
    assert attn.k_linears[1].out_features == 4 * 32
    assert attn.v_linears[0].out_features == 4 * 32
    assert attn.a_linears[0].in_features == 4 * 32
    assert tuple(attn.relation_att.shape) == (4, 4, 32, 32)
    assert tuple(attn.relation_msg.shape) == (4, 4, 32, 32)
    assert entry["family"] == "hmsa_head"
    assert entry["prune_rate_pct"] == 50
    assert entry["preserve_output_dim"] == 256
    assert entry["keep_heads"] == [0, 1, 2, 3]


def test_tvm_blocker_extraction_keeps_precise_stage_and_op():
    mod = _load_module()
    tvm_report = {
        "backend_results": {
            "mswin_bwa_full_attention": {
                "backend": "TVM",
                "scope": "full_attention_mixed_int8",
                "int8": {
                    "status": "BLOCKED",
                    "stage": "Relax LegalizeOps",
                    "op": "relax.reshape qkv -> window heads",
                    "error": "TVMError: Cannot prove symbolic reshape equality",
                },
            }
        }
    }

    blocker = mod.find_full_attention_tvm_blocker(tvm_report)

    assert blocker["target"] == "mswin_bwa_full_attention"
    assert blocker["backend"] == "TVM"
    assert blocker["stage"] == "Relax LegalizeOps"
    assert blocker["op"] == "relax.reshape qkv -> window heads"
    assert "symbolic reshape" in blocker["detail"]


def test_hmsa_tvm_blocker_fallback_names_relation_einsum_lowering():
    mod = _load_module()
    tvm_report = {
        "backend_results": {
            "hmsa_full_attention": {
                "backend": "TVM",
                "scope": "full_attention_int8_blocker",
                "description": "HGTCavAttention full hetero relation attention",
                "int8": {
                    "status": "BLOCKED",
                    "reason": "No T1 implementation yet for full TVM INT8 HMSA relation einsum and per-type projection lowering.",
                },
            }
        }
    }

    blocker = mod.find_full_attention_tvm_blocker(tvm_report)

    assert blocker["target"] == "hmsa_full_attention"
    assert blocker["stage"] == "NOT_IMPLEMENTED"
    assert "relation einsum" in blocker["op"]
    assert "per-type q/k/v/a" in blocker["op"]
    assert "B=1,L=2,H=64,W=128,C=256" in blocker["minimal_target"]


def test_hmsa_minimal_core_ok_still_blocks_dynamic_e2e_quant_row():
    mod = _load_module()
    tvm_report = {
        "backend_results": {
            "mswin_bwa_full_attention": {
                "backend": "TVM",
                "scope": "full_attention_mixed_int8",
                "fp16": {"status": "OK", "p50_ms": 0.455125},
                "mixed_int8": {"status": "OK", "p50_ms": 0.463499},
            },
            "mswin_bwa_p50_full_attention": {
                "backend": "TVM",
                "scope": "full_attention_mixed_int8",
                "fp16": {"status": "OK", "p50_ms": 0.242336},
                "mixed_int8": {"status": "OK", "p50_ms": 0.254517},
            },
            "hmsa_full_attention": {
                "backend": "TVM",
                "scope": "full_attention_hmsa_relation_core_mixed_int8",
                "fp16": {"status": "OK", "p50_ms": 0.331914},
                "mixed_int8": {"status": "OK", "p50_ms": 0.34529},
                "caveat": "dynamic type dispatch and q/k/v input projection ModuleList lowering remain separate work.",
            },
            "hmsa_p50_full_attention": {
                "backend": "TVM",
                "scope": "full_attention_hmsa_relation_core_mixed_int8",
                "fp16": {"status": "OK", "p50_ms": 0.19441},
                "mixed_int8": {"status": "OK", "p50_ms": 0.199147},
            },
        }
    }

    blocker = mod.find_full_attention_tvm_blocker(tvm_report)

    assert blocker["target"] == "attention-p50-int8/mixed"
    assert blocker["stage"] == "DYNAMIC_DISPATCH_AND_E2E_QUANT_NOT_INTEGRATED"
    assert "q/k/v/a ModuleList" in blocker["op"]
    assert "AP eval" in blocker["op"]
    assert "minimal relation core" in blocker["detail"]


def test_checkpoint_load_allows_missing_mswin_relative_indices_buffer():
    mod = _load_module()

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2)
            self.register_buffer("relative_indices", torch.ones(2, 2))

    model = Tiny()
    state = {
        "linear.weight": model.linear.weight.detach().clone(),
        "linear.bias": model.linear.bias.detach().clone(),
    }

    info = mod.load_state_dict_compatible(model, state)

    assert info["status"] == "OK"
    assert info["missing_keys"] == ["relative_indices"]
    assert info["unexpected_keys"] == []


def test_cast_output_dict_to_float32_returns_new_nested_tensors():
    mod = _load_module()
    half = torch.ones(1, 2, dtype=torch.float16)
    integer = torch.ones(1, dtype=torch.int64)
    output = {"ego": {"cls_preds": half, "count": integer, "meta": "kept"}}

    converted = mod.cast_output_dict_to_float32(output)

    assert converted is not output
    assert converted["ego"] is not output["ego"]
    assert converted["ego"]["cls_preds"].dtype == torch.float32
    assert converted["ego"]["count"].dtype == torch.int64
    assert output["ego"]["cls_preds"].dtype == torch.float16
    assert converted["ego"]["meta"] == "kept"


def test_partial_summary_includes_hmsa_minimal_core_negative_evidence():
    mod = _load_module()
    rows = [
        {"config": "baseline", "latency_p50_ms": 34.2072, "ap50": 0.5804480657, "ap70": 0.4473062150},
        {"config": "attention-p50-fp16", "latency_p50_ms": 32.8882, "ap50": 0.5421632042, "ap70": 0.3761207584},
    ]
    tvm_report = {
        "backend_results": {
            "mswin_bwa_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.455125},
                "mixed_int8": {"status": "OK", "p50_ms": 0.463499},
            },
            "mswin_bwa_p50_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.242336},
                "mixed_int8": {"status": "OK", "p50_ms": 0.254517},
            },
            "hmsa_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.331914},
                "mixed_int8": {"status": "OK", "p50_ms": 0.34529},
            },
            "hmsa_p50_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.19441},
                "mixed_int8": {"status": "OK", "p50_ms": 0.199147},
            },
        }
    }

    summary = mod.build_partial_evidence_summary(rows, tvm_report)

    assert summary["hmsa_tvm"]["base_mixed_int8_vs_fp16_speedup"] == 0.9613
    assert summary["hmsa_tvm"]["p50_mixed_int8_vs_fp16_speedup"] == 0.9762
    assert summary["hmsa_tvm"]["evidence_status"] == "MINIMAL_CORE_NEGATIVE_OR_NEUTRAL_INT8_EVIDENCE"


def test_partial_summary_records_shortft_recovery_and_latency_risk():
    mod = _load_module()
    rows = [
        {"config": "baseline", "latency_p50_ms": 34.2072, "ap50": 0.5804480657, "ap70": 0.4473062150},
        {"config": "attention-p50-fp16", "latency_p50_ms": 32.8882, "ap50": 0.5421632042, "ap70": 0.3761207584},
        {"config": "attention-p50-shortft-fp16", "latency_p50_ms": 35.751, "ap50": 0.6276087331, "ap70": 0.4774571697},
    ]

    summary = mod.build_partial_evidence_summary(rows, tvm_report=None)

    assert summary["p50_shortft"]["speedup_vs_baseline"] == 0.9568
    assert summary["p50_shortft"]["delta_ap50"] == 0.0472
    assert summary["p50_shortft"]["delta_ap70"] == 0.0302
    assert summary["p50_shortft"]["accuracy_recovered_in_pilot"] is True
    assert summary["p50_shortft"]["speed_stop_c_risk"] is True
    assert summary["p50_shortft"]["ap_gain_requires_protocol_audit"] is True


def test_stop_b_markdown_requires_real_eval_fields_and_rejects_fake_prior():
    mod = _load_module()
    rows = [
        {
            "config": "baseline",
            "checkpoint_path": "/ckpt/base.pth",
            "manifest_path": None,
            "prune_rate_pct": 0,
            "quant_backend": "none",
            "latency_scope": "pytorch_model_forward_e2e",
            "latency_p50_ms": 34.2072,
            "latency_log": "logs/base_latency.json",
            "latency_command": "python bench.py --baseline",
            "ap50": 0.5804480657,
            "ap70": 0.4473062150,
            "ap_log": "logs/base_ap.json",
            "ap_command": "python eval.py --baseline",
            "dataset_split": "DAIR val",
            "sample_count": 64,
            "accuracy_status": "REAL_EVAL_LOG_64_SAMPLE_PILOT",
        },
        {
            "config": "attention-p50-fp16",
            "checkpoint_path": "/ckpt/attention_p50.pth",
            "manifest_path": "/ckpt/attention_p50_manifest.json",
            "prune_rate_pct": 50,
            "quant_backend": "none",
            "latency_scope": "pytorch_model_forward_e2e",
            "latency_p50_ms": 32.8882,
            "latency_log": "logs/p50_latency.json",
            "latency_command": "python bench.py --p50",
            "ap50": 0.5421632042,
            "ap70": 0.3761207584,
            "ap_log": "logs/p50_ap.json",
            "ap_command": "python eval.py --p50",
            "dataset_split": "DAIR val",
            "sample_count": 64,
            "accuracy_status": "REAL_EVAL_LOG_64_SAMPLE_PILOT",
        }
    ]
    blocker = {
        "target": "hmsa_full_attention",
        "backend": "TVM",
        "stage": "NOT_IMPLEMENTED",
        "op": "HGTCavAttention relation einsum",
        "detail": "No minimal Relax/TIR attempt has been implemented yet.",
        "minimal_target": "B=1,L=2,H=64,W=128,C=256,heads=8,dim_head=32",
    }
    tvm_report = {
        "backend_results": {
            "mswin_bwa_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.456085},
                "mixed_int8": {"status": "OK", "p50_ms": 0.463605},
            },
            "mswin_bwa_p50_full_attention": {
                "fp16": {"status": "OK", "p50_ms": 0.242069},
                "mixed_int8": {"status": "OK", "p50_ms": 0.255083},
            },
        }
    }

    summary = mod.build_partial_evidence_summary(rows, tvm_report)
    md = mod.build_stop_b_markdown(rows, blocker, git_status_short="?? only", tvm_report=tvm_report)

    assert "Stop-B" in md
    assert "hmsa_full_attention" in md
    assert "relation einsum" in md
    assert "64-sample pilot" in md
    assert "Stop-C risk" in md
    assert "1.0401" in md
    assert "-0.0712" in md
    assert "MSwin mixed INT8 is slower than FP16" in md
    assert "NOT_IMPLEMENTED" in md
    assert "minimal HMSA Relax/TIR target" in md
    assert summary["p50_noft"]["speedup_vs_baseline"] == 1.0401
    assert summary["p50_noft"]["delta_ap70"] == -0.0712
    assert summary["mswin_tvm"]["base_mixed_int8_vs_fp16_speedup"] == 0.9838
    assert "C4_QgranxP" not in md
    assert "SIMULATED_PRIOR" not in md
