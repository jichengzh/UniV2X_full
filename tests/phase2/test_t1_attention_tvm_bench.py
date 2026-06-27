from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "t1_attention_tvm_bench.py"
    spec = importlib.util.spec_from_file_location("t1_attention_tvm_bench", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mswin_bwa_shape_plan_tracks_window_tokens_and_inner_dim():
    mod = _load_module()

    plan = mod.make_mswin_bwa_shape_plan(
        batch=1,
        agents=2,
        h=64,
        w=128,
        channels=256,
        heads=4,
        dim_head=64,
        window_size=16,
        prune_rate_pct=0,
    )

    assert plan["input_shape"] == [1, 2, 64, 128, 256]
    assert plan["new_h"] == 4
    assert plan["new_w"] == 8
    assert plan["window_tokens"] == 256
    assert plan["window_batches"] == 1 * 2 * 4 * 4 * 8
    assert plan["inner_dim"] == 256
    assert plan["qkv_shape_mkn"] == [1 * 2 * 64 * 128, 256, 256]
    assert plan["out_shape_mkn"] == [1 * 2 * 64 * 128, 256, 256]


def test_mswin_bwa_shape_plan_represents_p50_head_pruning():
    mod = _load_module()

    plan = mod.make_mswin_bwa_shape_plan(
        batch=1,
        agents=2,
        h=64,
        w=128,
        channels=256,
        heads=2,
        dim_head=64,
        window_size=16,
        prune_rate_pct=50,
    )

    assert plan["heads"] == 2
    assert plan["inner_dim"] == 128
    assert plan["qkv_shape_mkn"] == [1 * 2 * 64 * 128, 256, 128]
    assert plan["out_shape_mkn"] == [1 * 2 * 64 * 128, 128, 256]


def test_hmsa_shape_plan_tracks_relation_core_and_p50_target():
    mod = _load_module()

    plan = mod.make_hmsa_shape_plan(
        batch=1,
        agents=2,
        h=64,
        w=128,
        channels=256,
        heads=8,
        dim_head=32,
        num_types=2,
        num_relations=4,
        prune_rate_pct=0,
    )

    assert plan["input_shape"] == [1, 2, 64, 128, 256]
    assert plan["spatial_tokens"] == 1 * 64 * 128
    assert plan["flat_agent_tokens"] == 1 * 2 * 64 * 128
    assert plan["inner_dim"] == 256
    assert plan["relation_att_shape"] == [4, 8, 32, 32]
    assert plan["relation_pair_order"] == ["0->0", "0->1", "1->0", "1->1"]
    assert plan["qkv_shape_mkn"] == [1 * 2 * 64 * 128, 256, 256]
    assert plan["out_shape_mkn"] == [1 * 2 * 64 * 128, 256, 256]
    assert plan["core_tensor_shape"] == [8, 1 * 64 * 128, 32]

    p50 = mod.make_hmsa_shape_plan(
        batch=1,
        agents=2,
        h=64,
        w=128,
        channels=256,
        heads=4,
        dim_head=32,
        num_types=2,
        num_relations=4,
        prune_rate_pct=50,
    )

    assert p50["heads"] == 4
    assert p50["inner_dim"] == 128
    assert p50["relation_att_shape"] == [4, 4, 32, 32]
    assert p50["qkv_shape_mkn"] == [1 * 2 * 64 * 128, 256, 128]
    assert p50["out_shape_mkn"] == [1 * 2 * 64 * 128, 128, 256]


def test_hmsa_static_2agent_param_plan_includes_projection_and_relation_dispatch():
    mod = _load_module()
    plan = mod.make_hmsa_shape_plan(
        batch=1,
        agents=2,
        h=64,
        w=128,
        channels=256,
        heads=4,
        dim_head=32,
        num_types=2,
        num_relations=4,
        prune_rate_pct=50,
    )

    static_plan = mod.make_hmsa_static_2agent_param_plan(plan)

    assert static_plan["scope"] == "hmsa_static_2agent_qkv_relation_out"
    assert static_plan["static_type_order"] == [0, 1]
    assert static_plan["x_agent_shape"] == [64 * 128, 256]
    assert static_plan["qkv_weight_shapes"]["q_type0"] == [256, 128]
    assert static_plan["qkv_weight_shapes"]["v_type1"] == [256, 128]
    assert static_plan["relation_att_pair_order"] == ["0->0", "0->1", "1->0", "1->1"]
    assert static_plan["relation_att_shape"] == [4, 4, 32, 32]
    assert static_plan["out_weight_shapes"]["type0"] == [128, 256]
    assert static_plan["output_shape"] == [1, 2, 64, 128, 256]
    assert static_plan["covers_qkv_projection"] is True
    assert static_plan["covers_dynamic_type_dispatch"] is False


def test_hmsa_not_implemented_blocker_names_minimal_target():
    mod = _load_module()
    plan = {
        "input_shape": [1, 2, 64, 128, 256],
        "heads": 8,
        "dim_head": 32,
        "inner_dim": 256,
        "relation_att_shape": [4, 8, 32, 32],
        "prune_rate_pct": 0,
    }

    entry = mod.make_hmsa_not_implemented_entry(plan, precision="mixed_int8")

    assert entry["status"] == "BLOCKED"
    assert entry["stage"] == "NOT_IMPLEMENTED"
    assert entry["op"] == "HGTCavAttention relation einsum and per-type q/k/v/a dispatch"
    assert "B=1,L=2,H=64,W=128,C=256" in entry["minimal_target"]
    assert "relation_att/msg=(4,8,32,32)" in entry["minimal_target"]


def test_blocked_entry_records_stage_op_and_traceback_tail():
    mod = _load_module()

    entry = mod.make_tvm_blocked_entry(
        precision="mixed_int8",
        stage="Relax LegalizeOps",
        op="softmax_after_int8_qk",
        exc=RuntimeError("lowering failed"),
        traceback_text="line1\nline2\nline3",
    )

    assert entry["status"] == "BLOCKED"
    assert entry["stage"] == "Relax LegalizeOps"
    assert entry["op"] == "softmax_after_int8_qk"
    assert entry["error"] == "RuntimeError: lowering failed"
    assert entry["traceback"] == "line1\nline2\nline3"
