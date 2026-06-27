from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts" / "phase2" / "attention_e2e_checkpoint_eval.py"
    spec = importlib.util.spec_from_file_location("attention_e2e_checkpoint_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_eval_configs_include_baseline_noft_and_shortft():
    mod = _load_module()

    configs = mod.default_eval_configs()

    assert [cfg["config"] for cfg in configs] == [
        "baseline",
        "attention-p50-fp16",
        "attention-p50-shortft-fp16",
    ]
    assert configs[0]["pruned"] is False
    assert configs[1]["pruned"] is True
    assert configs[2]["finetune"] == "100_steps_lr1e-4_seed20260623_all_params"


def test_shortft_checkpoint_and_manifest_can_be_overridden_by_environment(monkeypatch):
    monkeypatch.setenv("V2XVIT_ATTENTION_SHORTFT_CKPT", "/tmp/h800_shortft.pth")
    monkeypatch.setenv("V2XVIT_ATTENTION_SHORTFT_MANIFEST", "/tmp/h800_shortft_manifest.json")
    monkeypatch.setenv("V2XVIT_ATTENTION_SHORTFT_FINETUNE", "500_steps_lr1e-4_seed20260624_all_params")
    mod = _load_module()

    configs = mod.default_eval_configs()

    shortft = next(cfg for cfg in configs if cfg["config"] == "attention-p50-shortft-fp16")
    assert shortft["checkpoint_path"] == "/tmp/h800_shortft.pth"
    assert shortft["manifest_path"] == "/tmp/h800_shortft_manifest.json"
    assert shortft["finetune"] == "500_steps_lr1e-4_seed20260624_all_params"


def test_build_comparison_rows_adds_speedup_and_ap_deltas():
    mod = _load_module()
    rows = [
        {"config": "baseline", "latency_p50_ms": 34.0, "ap50": 0.58, "ap70": 0.44},
        {"config": "attention-p50-fp16", "latency_p50_ms": 32.0, "ap50": 0.54, "ap70": 0.37},
        {"config": "attention-p50-shortft-fp16", "latency_p50_ms": 36.0, "ap50": 0.63, "ap70": 0.48},
    ]

    out = mod.build_comparison_rows(rows)

    by_config = {row["config"]: row for row in out}
    assert by_config["baseline"]["speedup_vs_baseline"] == 1.0
    assert by_config["attention-p50-fp16"]["speedup_vs_baseline"] == 1.0625
    assert by_config["attention-p50-fp16"]["delta_ap70"] == -0.07
    assert by_config["attention-p50-shortft-fp16"]["speedup_vs_baseline"] == 0.9444
    assert by_config["attention-p50-shortft-fp16"]["delta_ap50"] == 0.05


def test_render_markdown_marks_larger_split_not_stop_a():
    mod = _load_module()
    rows = mod.build_comparison_rows([
        {"config": "baseline", "latency_p50_ms": 34.0, "ap50": 0.58, "ap70": 0.44},
        {"config": "attention-p50-fp16", "latency_p50_ms": 32.0, "ap50": 0.54, "ap70": 0.37},
    ])
    report = {"schema_version": "attention_e2e_checkpoint_eval_v1", "eval_samples": 256, "rows": rows}

    md = mod.render_markdown(report)

    assert "not Stop-A" in md
    assert "attention-p50-fp16" in md
    assert "256" in md


def test_eval_log_tag_distinguishes_pilot_and_full_val():
    mod = _load_module()

    assert mod.eval_log_tag(256) == "eval256"
    assert mod.eval_log_tag(1789) == "full"


def test_repo_relative_output_paths_stay_under_v2x_root():
    mod = _load_module()

    path = mod.resolve_repo_path("results/example.json")

    assert str(path).endswith("/V2X/results/example.json")
