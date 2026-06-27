"""Phase T1-P/T1-Q for V2X-ViT transformer integration.

This script complements the existing T1-S results in
``results/attention_axis_feasibility_v1.json``:

* T1-P: enumerate transformer pruning families and materialize the manual
  structured scanner needed for HMSA/MSwin head pruning.
* T1-Q: record TVM-only attention quantization evidence. QDQ-ONNX and
  non-TVM engine artifacts are not accepted as this gate.

Run:
  CUDA_VISIBLE_DEVICES=6 /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python \\
      scripts/phase2/t1_attention_pq_feasibility.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
HEAL_ROOT = Path("/home/jichengzhi/heal_research/HEAL")
PYTHON = "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python"
CKPT_DIR = Path(
    "/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
    "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26"
)
CONFIG_YAML = CKPT_DIR / "config.yaml"
CKPT_FILE = CKPT_DIR / "net_epoch_bestval_at17.pth"
TVM_BENCH_SCRIPT = REPO_ROOT / "scripts/phase2/t1_attention_tvm_bench.py"
OUT_JSON = REPO_ROOT / "results/attention_axis_feasibility_v1.json"
OUT_MD = REPO_ROOT / "results/attention_axis_feasibility_v1.md"
MODEL_DIR = REPO_ROOT / "models/v2xvit_attention_t1"
CALIB_DIR = REPO_ROOT / "calibration/v2xvit_attention_t1"
COUPLING_PRIOR = REPO_ROOT / "results/coupling_map/C4_QgranxP_v2xvit.json"


class TimeoutExpired(RuntimeError):
    pass


def _round_float(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def compute_speedup(fp16_ms: Any, int8_ms: Any) -> float | None:
    fp16 = _round_float(fp16_ms, 8)
    int8 = _round_float(int8_ms, 8)
    if fp16 is None or int8 is None or int8 <= 0:
        return None
    return round(fp16 / int8, 4)


def _is_ok(entry: dict[str, Any] | None) -> bool:
    return bool(entry and entry.get("status") == "OK")


def _p50(entry: dict[str, Any] | None) -> float | None:
    if not entry:
        return None
    for key in ("p50_ms", "lat_p50_ms"):
        if key in entry:
            return _round_float(entry[key])
    return None


def _legal_keep_heads(heads: int) -> list[int]:
    return [h for h in range(heads, 0, -1) if h == heads or h % 2 == 0]


def scan_manual_attention_pruning_groups(enc_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build T1-P manual structured groups for V2X-ViT HMSA/MSwin.

    The scanner is intentionally config-driven. T2 can bind these symbolic
    members to live modules, but T1 must already prove that each structured
    unit is enumerable and has explicit slice rules.
    """
    cav_cfg = enc_cfg["cav_att_config"]
    pwin_cfg = enc_cfg["pwindow_att_config"]
    depth = int(enc_cfg["depth"])
    num_blocks = int(enc_cfg.get("num_blocks", 1))
    dim = int(cav_cfg["dim"])
    hmsa_heads = int(cav_cfg["heads"])
    hmsa_dim_head = int(cav_cfg["dim_head"])
    groups: list[dict[str, Any]] = []

    for layer in range(depth):
        for block in range(num_blocks):
            hmsa_base = f"encoder.layers.{layer}.0.layers.{block}.0.fn"
            groups.append({
                "id": f"hmsa.L{layer}.B{block}.heads",
                "family": "hmsa_head",
                "layer": layer,
                "block": block,
                "structured_unit": "head",
                "heads": hmsa_heads,
                "dim_head": hmsa_dim_head,
                "inner_dim": hmsa_heads * hmsa_dim_head,
                "legal_keep_heads": _legal_keep_heads(hmsa_heads),
                "members": [
                    f"{hmsa_base}.q_linears[*]",
                    f"{hmsa_base}.k_linears[*]",
                    f"{hmsa_base}.v_linears[*]",
                    f"{hmsa_base}.a_linears[*]",
                    f"{hmsa_base}.relation_att",
                    f"{hmsa_base}.relation_msg",
                ],
                "slice_rules": [
                    "q/k/v Linear out_features -> keep_heads * dim_head",
                    "a Linear in_features -> keep_heads * dim_head",
                    "relation_att[:, keep_heads, :, :]",
                    "relation_msg[:, keep_heads, :, :]",
                    "preserve final embed dim for residual add",
                ],
                "notes": "HGTCavAttention head pruning must slice projection rows/cols and relation tensors together.",
            })

            for idx, (ws, heads, dim_head) in enumerate(zip(
                pwin_cfg["window_size"],
                pwin_cfg["heads"],
                pwin_cfg["dim_head"],
            )):
                heads_i = int(heads)
                dim_head_i = int(dim_head)
                base = f"encoder.layers.{layer}.0.layers.{block}.1.fn.pwmsa.{idx}"
                groups.append({
                    "id": f"mswin.L{layer}.B{block}.ws{int(ws)}.heads",
                    "family": "mswin_head",
                    "layer": layer,
                    "block": block,
                    "window_index": idx,
                    "window_size": int(ws),
                    "structured_unit": "head",
                    "heads": heads_i,
                    "dim_head": dim_head_i,
                    "inner_dim": heads_i * dim_head_i,
                    "legal_keep_heads": _legal_keep_heads(heads_i),
                    "members": [
                        f"{base}.to_qkv",
                        f"{base}.to_out.0",
                        f"{base}.pos_embedding",
                    ],
                    "slice_rules": [
                        "to_qkv.out_features -> 3 * keep_heads * dim_head",
                        "to_out[0].in_features -> keep_heads * dim_head",
                        "pos_embedding unchanged",
                        "relative_indices buffer unchanged",
                        "preserve final embed dim for residual add",
                    ],
                    "notes": "BaseWindowAttention head pruning changes QKV/output channel grouping but not spatial windows.",
                })

        groups.append({
            "id": f"encoder.L{layer}.depth",
            "family": "encoder_depth",
            "layer": layer,
            "structured_unit": "encoder_layer",
            "members": [f"encoder.layers.{layer}"],
            "slice_rules": [
                "drop whole V2XFusionBlock + FeedForward layer",
                "preserve embed dim and residual shape across remaining layers",
            ],
            "notes": "Depth pruning is a whole-layer decision and requires finetune/AP validation.",
        })

    groups.append({
        "id": "transformer.embed_dim",
        "family": "embed_dim",
        "structured_unit": "channel",
        "width": dim,
        "round_to": 32,
        "legal_widths": [w for w in range(dim, 0, -32) if w >= 128],
        "members": [
            "encoder.prior_feed",
            "all PreNorm.norm",
            "all FeedForward Linear",
            "all HMSA q/k/v/a Linear",
            "all MSwin to_qkv/to_out Linear",
        ],
        "slice_rules": [
            "slice all residual-path Linear in/out features consistently",
            "slice LayerNorm normalized_shape",
            "slice HMSA relation tensors dim_head only if dim_head changes",
        ],
        "notes": "Embed pruning is wider than attention-head pruning and remains a T2 surgery item.",
    })

    counts = dict(Counter(group["family"] for group in groups))
    return {
        "status": "PASS",
        "depth": depth,
        "num_blocks": num_blocks,
        "dim": dim,
        "counts": counts,
        "groups": groups,
    }


def gate_p_axis(
    pruning_groups: dict[str, dict[str, Any]],
    depgraph_checks: dict[str, dict[str, Any]],
    manual_scanner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = ("embed_dim", "heads", "depth")
    missing = [
        name for name in required
        if pruning_groups.get(name, {}).get("count", 0) <= 0
    ]
    failed_checks = [
        name for name, item in depgraph_checks.items()
        if item.get("status") != "PASS"
    ]

    scanner_ok = bool(manual_scanner and manual_scanner.get("status") == "PASS")
    if not scanner_ok:
        missing.append("manual_hmsa_mswin_scanner")

    if not missing and not failed_checks:
        verdict = "P_axis_READY_FOR_T2_SCANNER_ONLY"
    elif not scanner_ok:
        verdict = "P_axis_MANUAL_SCANNER_REQUIRED"
    elif not missing and depgraph_checks.get("feedforward_linear_depgraph", {}).get("status") == "PASS":
        verdict = "P_axis_READY_WITH_MANUAL_SCANNER_DEPGRAPH_PARTIAL"
    else:
        verdict = "P_axis_BLOCKED"

    return {
        "verdict": verdict,
        "missing": missing,
        "failed_checks": failed_checks,
        "scanner_counts": (manual_scanner or {}).get("counts", {}),
        "next": (
            "Implement HMSA/MSwin manual structured scanner"
            if verdict == "P_axis_MANUAL_SCANNER_REQUIRED"
            else "Keep scanner as T2 input; do not start T2 until Q-axis TVM evidence is complete"
        ),
    }


def gate_q_axis(backend_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    non_tvm_ignored: list[str] = []
    direct_int8_targets_ok: list[str] = []
    full_attention_ok: list[str] = []
    full_attention_blocked: list[str] = []
    tvm_fp16_only: list[str] = []
    speedups: dict[str, float | None] = {}

    for name, item in backend_results.items():
        if item.get("backend") != "TVM":
            non_tvm_ignored.append(name)
            continue
        fp16 = item.get("fp16")
        int8 = item.get("int8")
        sp = compute_speedup(_p50(fp16), _p50(int8))
        speedups[name] = sp

        scope = str(item.get("scope", "")).lower()
        is_full_attention = "full_attention" in scope or "full_attention" in name.lower()
        is_direct_int8 = "direct" in scope or "matmul" in scope or "linear" in scope

        if is_full_attention:
            if _is_ok(fp16) and _is_ok(int8) and sp and sp > 1.05:
                full_attention_ok.append(name)
            else:
                full_attention_blocked.append(name)
            continue

        if is_direct_int8 and _is_ok(fp16) and _is_ok(int8) and sp and sp > 1.05:
            direct_int8_targets_ok.append(name)
        elif _is_ok(fp16):
            tvm_fp16_only.append(name)

    if full_attention_ok:
        verdict = "Q_axis_TVM_INT8_FULL_ATTN_READY"
    elif direct_int8_targets_ok and full_attention_blocked:
        verdict = "Q_axis_TVM_DIRECT_INT8_PARTIAL_FULL_ATTN_BLOCKED"
    elif direct_int8_targets_ok:
        verdict = "Q_axis_TVM_DIRECT_INT8_SUBOPS_ONLY"
    elif tvm_fp16_only:
        verdict = "Q_axis_TVM_FP16_ONLY_INT8_BLOCKED"
    elif backend_results:
        verdict = "Q_axis_TVM_BLOCKED_OR_NON_TVM_ONLY"
    else:
        verdict = "Q_axis_INCOMPLETE"

    return {
        "verdict": verdict,
        "non_tvm_ignored": non_tvm_ignored,
        "direct_int8_targets_ok": direct_int8_targets_ok,
        "full_attention_ok": full_attention_ok,
        "full_attention_blocked": full_attention_blocked,
        "tvm_fp16_only": tvm_fp16_only,
        "speedups": speedups,
        "next": (
            "Implement/benchmark TVM full-attention INT8 lowering; do not use non-TVM engines as substitute"
            if direct_int8_targets_ok and full_attention_blocked
            else "Run TVM attention backend benchmarks"
        ),
    }


def gate_overall(p_gate: dict[str, Any], q_gate: dict[str, Any]) -> dict[str, Any]:
    p = p_gate["verdict"]
    q = q_gate["verdict"]
    if p == "P_axis_READY_FOR_T2_SCANNER_ONLY" and q == "Q_axis_TVM_INT8_FULL_ATTN_READY":
        verdict = "T1_PQ_PASS_PROCEED_T2"
    else:
        verdict = "T1_PQ_INCOMPLETE_DO_NOT_START_T2"
    return {
        "verdict": verdict,
        "next": (
            "T2 adapter + graph_scan extension"
            if verdict == "T1_PQ_PASS_PROCEED_T2"
            else "Complete T1 manual scanner and TVM attention quantization evidence"
        ),
    }


def load_attention_accuracy_priors(path: Path = COUPLING_PRIOR) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}

    priors: dict[str, dict[str, Any]] = {}
    for anchor, item in data.get("results", {}).items():
        int8 = item.get("int8_sim", {})
        fp32 = item.get("fp32", {})
        prune_ratio = item.get("prune_ratio")
        priors[f"{anchor}_sim_int8"] = {
            "prune_rate_pct": int(round(float(prune_ratio or 0.0) * 100)),
            "quant": "int8",
            "ap50": _round_float(int8.get("ap50"), 4),
            "ap70": _round_float(int8.get("ap70"), 4),
            "fp32_ap50": _round_float(fp32.get("ap50"), 4),
            "fp32_ap70": _round_float(fp32.get("ap70"), 4),
            "source": str(path),
            "status": "SIMULATED_PRIOR_NOT_TRUE_TVM",
            "notes": item.get("caveat", ""),
        }
    return priors


def build_attention_coupling_table(
    backend_results: dict[str, dict[str, Any]],
    accuracy_priors: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    def _best_accuracy_prior(prune_rate_pct: int | None) -> dict[str, Any] | None:
        if prune_rate_pct is None:
            return None
        for item in accuracy_priors.values():
            if item.get("prune_rate_pct") == prune_rate_pct:
                return item
        return None

    direct_items = [
        (name, item)
        for name, item in backend_results.items()
        if item.get("backend") == "TVM"
        and _is_ok(item.get("fp16"))
        and _is_ok(item.get("int8"))
    ]
    direct_items.sort(key=lambda pair: (0 if "qkv" in pair[0] else 1, pair[0]))
    latency_name, latency_item = direct_items[0] if direct_items else (None, None)
    speedup = compute_speedup(
        _p50(latency_item.get("fp16")) if latency_item else None,
        _p50(latency_item.get("int8")) if latency_item else None,
    )

    rows: list[dict[str, Any]] = []
    pruned_direct_items = [
        (name, item)
        for name, item in backend_results.items()
        if item.get("backend") == "TVM"
        and item.get("prune_rate_pct") not in (None, 0)
        and _is_ok(item.get("int8"))
        and item.get("baseline_target") in backend_results
    ]
    if pruned_direct_items:
        base_prior = _best_accuracy_prior(0)
        base = backend_results.get("linear_qkv_direct")
        rows.append({
            "config": "attention_head_p0_qkv_direct",
            "prune_rate_pct": 0,
            "quant": "int8",
            "latency_backend": "TVM",
            "latency_scope": base.get("scope") if base else "direct_matmul",
            "latency_target": "linear_qkv_direct" if base else None,
            "fp16_latency_ms": _p50(base.get("fp16")) if base else None,
            "int8_latency_ms": _p50(base.get("int8")) if base else None,
            "speedup": compute_speedup(_p50(base.get("fp16")) if base else None, _p50(base.get("int8")) if base else None),
            "accuracy_ap50": base_prior.get("ap50") if base_prior else None,
            "accuracy_ap70": base_prior.get("ap70") if base_prior else None,
            "accuracy_status": "SIMULATED_PRIOR_NOT_TRUE_TVM" if base_prior else "NOT_MEASURED",
            "accuracy_source": base_prior.get("source") if base_prior else None,
            "caveat": "Base latency is real TVM direct QKV; AP is full-model fake-quant prior, not true TVM.",
        })
        for name, item in sorted(pruned_direct_items):
            baseline = backend_results.get(item.get("baseline_target"), {})
            prior = _best_accuracy_prior(int(item.get("prune_rate_pct")))
            rows.append({
                "config": name,
                "prune_rate_pct": item.get("prune_rate_pct"),
                "quant": "int8",
                "latency_backend": "TVM",
                "latency_scope": item.get("scope"),
                "latency_target": name,
                "fp16_latency_ms": _p50(baseline.get("fp16")),
                "int8_latency_ms": _p50(item.get("int8")),
                "speedup": compute_speedup(_p50(baseline.get("fp16")), _p50(item.get("int8"))),
                "accuracy_ap50": prior.get("ap50") if prior else None,
                "accuracy_ap70": prior.get("ap70") if prior else None,
                "accuracy_status": (
                    "SIMULATED_GLOBAL_PRIOR_NOT_ATTENTION_SPECIFIC_NOT_TRUE_TVM"
                    if prior else "ATTENTION_HEAD_PRUNING_AP_NOT_MEASURED"
                ),
                "accuracy_source": prior.get("source") if prior else None,
                "caveat": (
                    "Latency is real TVM direct pruned sub-op. Accuracy is not attention-head pruning AP; "
                    "global/full-model prior is shown only as a placeholder when available."
                ),
            })
        return rows

    if accuracy_priors:
        for name, acc in sorted(accuracy_priors.items()):
            rows.append({
                "config": name,
                "prune_rate_pct": acc.get("prune_rate_pct"),
                "quant": acc.get("quant", "int8"),
                "latency_backend": latency_item.get("backend") if latency_item else "TVM",
                "latency_scope": latency_item.get("scope") if latency_item else "NOT_MEASURED",
                "latency_target": latency_name,
                "fp16_latency_ms": _p50(latency_item.get("fp16")) if latency_item else None,
                "int8_latency_ms": _p50(latency_item.get("int8")) if latency_item else None,
                "speedup": speedup,
                "accuracy_ap50": acc.get("ap50"),
                "accuracy_ap70": acc.get("ap70"),
                "accuracy_status": acc.get("status", "UNKNOWN"),
                "accuracy_source": acc.get("source"),
                "caveat": (
                    "Latency is TVM direct sub-op evidence; AP prior is simulated fake-quant unless status says otherwise."
                ),
            })
    else:
        rows.append({
            "config": "attention_tvm_int8_direct_subops",
            "prune_rate_pct": 0,
            "quant": "int8",
            "latency_backend": latency_item.get("backend") if latency_item else "TVM",
            "latency_scope": latency_item.get("scope") if latency_item else "NOT_MEASURED",
            "latency_target": latency_name,
            "fp16_latency_ms": _p50(latency_item.get("fp16")) if latency_item else None,
            "int8_latency_ms": _p50(latency_item.get("int8")) if latency_item else None,
            "speedup": speedup,
            "accuracy_ap50": None,
            "accuracy_ap70": None,
            "accuracy_status": "NOT_MEASURED",
            "accuracy_source": None,
            "caveat": "No real AP result exists for TVM INT8 attention-fusion pruning yet.",
        })
    return rows


def build_markdown_report(report: dict[str, Any]) -> str:
    def _fmt_ms(entry: dict[str, Any] | None) -> str:
        value = _p50(entry)
        return "n/a" if value is None else f"{value}ms"

    gate = report.get("gate", {})
    p_gate = gate.get("p_axis", {})
    q_gate = gate.get("q_axis", {})
    overall = gate.get("overall", {})
    lines = [
        "# V2X-ViT Attention Axis Feasibility v1",
        "",
        f"- Overall: `{overall.get('verdict', 'UNKNOWN')}`",
        f"- P axis: `{p_gate.get('verdict', 'UNKNOWN')}`",
        f"- Q axis: `{q_gate.get('verdict', 'UNKNOWN')}`",
        f"- Next: {overall.get('next', 'n/a')}",
        "",
        "## T1-P Summary",
    ]

    p_axis = report.get("p_axis", {})
    for name, item in p_axis.get("pruning_groups", {}).items():
        lines.append(
            f"- `{name}`: status={item.get('status')} count={item.get('count')} "
            f"notes={item.get('notes', '')}"
        )
    scanner = p_axis.get("manual_scanner", {})
    if scanner:
        lines += ["", "## Manual HMSA/MSwin Scanner"]
        lines.append(f"- status={scanner.get('status')} counts={scanner.get('counts')}")
        for group in scanner.get("groups", [])[:8]:
            detail = (
                f"heads={group.get('heads')} dim_head={group.get('dim_head')}"
                if group.get("family") in {"hmsa_head", "mswin_head"}
                else f"unit={group.get('structured_unit')}"
            )
            lines.append(f"- `{group.get('id')}` family={group.get('family')} {detail}")

    lines += ["", "## T1-Q TVM Backend Results"]
    for name, item in report.get("backend_results", {}).items():
        if item.get("backend") != "TVM":
            continue
        fp16 = item.get("fp16", {})
        int8 = item.get("int8", {})
        sp = q_gate.get("speedups", {}).get(name)
        lines.append(
            f"- `{name}` ({item.get('scope')}): FP16={fp16.get('status')} p50={_fmt_ms(fp16)}; "
            f"INT8={int8.get('status')} p50={_fmt_ms(int8)}; speedup={sp}"
        )
        for precision, entry in (("fp16", fp16), ("int8", int8)):
            if entry.get("status") not in (None, "OK", "SKIPPED"):
                detail = entry.get("error") or entry.get("reason") or "no detail"
                lines.append(f"  - {precision}: {detail}")
            if entry.get("status") == "SKIPPED":
                lines.append(f"  - {precision} skipped: {entry.get('reason', '')}")

    ignored = q_gate.get("non_tvm_ignored", [])
    if ignored:
        lines += ["", "## Non-TVM Artifacts Ignored"]
        for name in ignored:
            lines.append(f"- `{name}`: ignored by T1-Q gate")

    blocked = q_gate.get("full_attention_blocked", [])
    if blocked:
        lines += ["", "## Full-Attention TVM INT8 Blockers"]
        for name in blocked:
            item = report.get("backend_results", {}).get(name, {})
            err = (
                item.get("fp16", {}).get("error")
                or item.get("int8", {}).get("error")
                or item.get("int8", {}).get("reason")
                or "unknown"
            )
            lines.append(f"- `{name}`: {err}")

    if report.get("coupling_table"):
        lines += [
            "",
            "## Attention Fusion Prune + INT8 Coupling Table",
            "| config | prune_rate | quant | latency_scope | speedup | AP50 | AP70 | accuracy_status |",
            "|---|---:|---|---|---:|---:|---:|---|",
        ]
        for row in report["coupling_table"]:
            lines.append(
                f"| {row.get('config')} | {row.get('prune_rate_pct')}% | {row.get('quant')} | "
                f"{row.get('latency_scope', 'n/a')} | {row.get('speedup')} | "
                f"{row.get('accuracy_ap50')} | {row.get('accuracy_ap70')} | {row.get('accuracy_status')} |"
            )

    lines += ["", "## Caveats"]
    lines.append("- Full stage1 manifest still skips V2XTransformer until Phase T2.")
    lines.append("- Q-axis gate is TVM-only; QDQ-ONNX fake quant and non-TVM engines are not accepted.")
    lines.append("- Current AP values in the coupling table are priors if marked SIMULATED_PRIOR_NOT_TRUE_TVM.")
    return "\n".join(lines) + "\n"


def _with_timeout(seconds: int, fn, *args, **kwargs):
    def _handler(_signum, _frame):
        raise TimeoutExpired(f"timeout after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _prepare_heal_imports():
    sys.path.insert(0, str(HEAL_ROOT))
    os.chdir(HEAL_ROOT)


def load_model(device: str):
    _prepare_heal_imports()
    import torch
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    model = train_utils.create_model(hypes)
    state = torch.load(CKPT_FILE, map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, hypes


def _shape_counter(linear_modules: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(f"{m['in_features']}x{m['out_features']}" for m in linear_modules)
    return dict(sorted(counts.items()))


def analyze_p_axis(model, hypes: dict[str, Any], depgraph_timeout_s: int) -> dict[str, Any]:
    import torch

    transformer = model.fusion_net.fusion_net
    encoder = transformer.encoder
    enc_cfg = hypes["model"]["args"]["v2xvit"]["transformer"]["encoder"]
    cav_cfg = enc_cfg["cav_att_config"]
    pwin_cfg = enc_cfg["pwindow_att_config"]
    dim = int(cav_cfg["dim"])
    depth = int(enc_cfg["depth"])

    linear_modules = []
    for name, module in transformer.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_modules.append({
                "name": name,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "bias": module.bias is not None,
            })

    linked_embed = [
        m for m in linear_modules
        if dim in (m["in_features"], m["out_features"])
        or m["out_features"] in (dim * 3, dim + 3)
    ]

    pruning_groups = {
        "embed_dim": {
            "status": "candidate",
            "count": 1,
            "width": dim,
            "round_to": 32,
            "int8_buildable_align": dim // 32,
            "linked_linear_count": len(linked_embed),
            "notes": "Jointly affects Linear in/out channels, LayerNorm, residual width, prior_feed, HMSA relation tensors, and MSwin to_qkv/to_out.",
        },
        "heads": {
            "status": "candidate",
            "count": 1 + len(pwin_cfg["heads"]),
            "hmsa": {
                "heads": int(cav_cfg["heads"]),
                "dim_head": int(cav_cfg["dim_head"]),
                "inner_dim": int(cav_cfg["heads"]) * int(cav_cfg["dim_head"]),
                "num_types": 2 if cav_cfg.get("use_hetero") else 1,
            },
            "mswin": [
                {
                    "window_size": int(ws),
                    "heads": int(head),
                    "dim_head": int(dim_head),
                    "inner_dim": int(head) * int(dim_head),
                }
                for ws, head, dim_head in zip(
                    pwin_cfg["window_size"], pwin_cfg["heads"], pwin_cfg["dim_head"]
                )
            ],
            "notes": "Structured head pruning is manual: slice q/k/v/a projections and relation tensors per head.",
        },
        "depth": {
            "status": "candidate",
            "count": depth,
            "layers": list(range(depth)),
            "notes": "Drop whole encoder layers; dimensions remain stable, AP requires finetune.",
        },
    }

    depgraph_checks = {
        "feedforward_linear_depgraph": _depgraph_feedforward_check(depgraph_timeout_s),
        "mswin_bwa_depgraph": _depgraph_bwa_check(depgraph_timeout_s),
    }
    manual_scanner = scan_manual_attention_pruning_groups(enc_cfg)

    return {
        "config": {
            "dim": dim,
            "depth": depth,
            "feedforward_mlp_dim": int(enc_cfg["feed_forward"]["mlp_dim"]),
            "num_blocks": int(enc_cfg["num_blocks"]),
            "use_hetero": bool(cav_cfg.get("use_hetero")),
        },
        "linear_inventory": {
            "count": len(linear_modules),
            "shape_counts": _shape_counter(linear_modules),
            "modules": linear_modules,
        },
        "pruning_groups": pruning_groups,
        "manual_scanner": manual_scanner,
        "depgraph_checks": depgraph_checks,
    }


def _depgraph_feedforward_check(timeout_s: int) -> dict[str, Any]:
    def _run():
        import torch
        import torch_pruning as tp
        from opencood.models.sub_modules.base_transformer import FeedForward

        module = FeedForward(256, 256, 0.0).eval()
        x = torch.randn(1, 2, 4, 4, 256)
        dg = tp.DependencyGraph().build_dependency(module, example_inputs=x, verbose=False)
        groups = list(dg.get_all_groups(root_module_types=[torch.nn.Linear], ignored_layers=[]))
        return {
            "status": "PASS",
            "groups": len(groups),
            "note": "torch-pruning can build isolated FFN Linear dependency graph.",
        }

    try:
        return _with_timeout(timeout_s, _run)
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _depgraph_bwa_check(timeout_s: int) -> dict[str, Any]:
    def _run():
        import torch
        import torch_pruning as tp
        from opencood.models.sub_modules.mswin import BaseWindowAttention

        module = BaseWindowAttention(256, 16, 16, 0.0, 4, True).eval()
        x = torch.randn(1, 1, 4, 4, 256)
        dg = tp.DependencyGraph().build_dependency(
            module,
            example_inputs=x,
            ignored_params=[module.pos_embedding],
            verbose=False,
        )
        groups = list(dg.get_all_groups(root_module_types=[torch.nn.Linear], ignored_layers=[]))
        return {
            "status": "PASS",
            "groups": len(groups),
            "note": "torch-pruning can trace isolated BaseWindowAttention.",
        }

    try:
        return _with_timeout(timeout_s, _run)
    except Exception as exc:
        return {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "note": "Use manual MSwin head/embed groups in T2 graph_scan.",
        }


def _get_module(root, dotted: str):
    cur = root
    for part in dotted.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _target_specs(transformer) -> dict[str, dict[str, Any]]:
    return {
        "linear_256x256": {
            "module_path": "encoder.layers.0.0.layers.0.0.fn.k_linears.0",
            "description": "HMSA k/q/v/a projection shape 256->256",
        },
        "linear_qkv": {
            "module_path": "encoder.layers.0.0.layers.0.1.fn.pwmsa.0.to_qkv",
            "description": "MSwin QKV projection shape 256->768",
        },
        "ffn": {
            "module_path": "encoder.layers.0.1.fn",
            "description": "FeedForward Linear-GELU-Linear, 256->256->256",
        },
        "mswin_bwa": {
            "module_path": "encoder.layers.0.0.layers.0.1.fn.pwmsa.2",
            "description": "BaseWindowAttention ws=16 with qkv/einsum/softmax/relative position",
        },
    }


def _ensure_calib(shape: tuple[int, ...], n_samples: int) -> Path:
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    shape_tag = "x".join(str(v) for v in shape)
    out = CALIB_DIR / f"input_{shape_tag}_n{n_samples}.npy"
    if out.exists():
        return out
    rng = np.random.default_rng(20260623)
    arr = rng.standard_normal((n_samples, *shape), dtype=np.float32)
    np.save(out, arr)
    return out


def _export_onnx(module, name: str, input_shape: tuple[int, ...], device: str) -> dict[str, Any]:
    import torch

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = MODEL_DIR / f"{name}.onnx"
    module = module.to(device).eval()
    dummy = torch.randn(*input_shape, device=device, dtype=torch.float32)

    def _do_export(export_device: str, previous_error: str | None = None) -> dict[str, Any]:
        nonlocal module
        module = module.to(export_device).eval()
        local_dummy = dummy.to(export_device)
        with torch.no_grad():
            out = module(local_dummy)
        t0 = time.time()
        torch.onnx.export(
            module,
            local_dummy,
            str(onnx_path),
            opset_version=16,
            input_names=["x"],
            output_names=["y"],
            do_constant_folding=True,
            dynamic_axes=None,
            verbose=False,
        )
        elapsed = time.time() - t0
        result = {
            "status": "OK",
            "onnx": str(onnx_path),
            "size_mb": round(onnx_path.stat().st_size / 1e6, 3),
            "export_secs": round(elapsed, 3),
            "export_device": export_device,
            "input_shape": list(input_shape),
            "output_shape": list(out.shape) if hasattr(out, "shape") else None,
        }
        if previous_error:
            result["cuda_export_error"] = previous_error
        return result

    try:
        return _do_export(device)
    except Exception as exc:
        first_error = f"{type(exc).__name__}: {exc}"
        first_tb = traceback.format_exc()[-2000:]
        if str(device).startswith("cuda"):
            try:
                result = _do_export("cpu", previous_error=first_error)
                module.to(device)
                return result
            except Exception as cpu_exc:
                return {
                    "status": "FAILED",
                    "error": f"{type(cpu_exc).__name__}: {cpu_exc}",
                    "cuda_error": first_error,
                    "traceback": traceback.format_exc()[-2000:],
                    "cuda_traceback": first_tb,
                    "input_shape": list(input_shape),
                }
        return {
            "status": "FAILED",
            "error": first_error,
            "traceback": first_tb,
            "input_shape": list(input_shape),
        }


def export_attention_onnx_targets(
    model,
    input_shape: tuple[int, ...],
    device: str,
    targets_filter: set[str] | None,
) -> dict[str, dict[str, Any]]:
    transformer = model.fusion_net.fusion_net
    specs = _target_specs(transformer)
    results: dict[str, dict[str, Any]] = {}

    for name, spec in specs.items():
        if targets_filter and name not in targets_filter:
            continue
        module = _get_module(transformer, spec["module_path"])
        item = {
            "description": spec["description"],
            "module_path": spec["module_path"],
            "export": _export_onnx(module, name, input_shape, device),
        }
        results[name] = item

    return results


def read_tvm_backend_results(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {
            "tvm_backend_missing": {
                "backend": "TVM",
                "scope": "environment",
                "description": "No TVM bench JSON was provided.",
                "fp16": {
                    "status": "SKIPPED",
                    "reason": f"Run {TVM_BENCH_SCRIPT} on H800 TVM and pass --tvm-results.",
                },
                "int8": {
                    "status": "SKIPPED",
                    "reason": f"Run {TVM_BENCH_SCRIPT} on H800 TVM and pass --tvm-results.",
                },
            }
        }
    if not path.exists():
        return {
            "tvm_backend_missing": {
                "backend": "TVM",
                "scope": "environment",
                "description": "TVM bench JSON path does not exist.",
                "fp16": {"status": "FAILED", "error": f"missing file: {path}"},
                "int8": {"status": "FAILED", "error": f"missing file: {path}"},
            }
        }
    data = json.loads(path.read_text())
    if data.get("backend_results"):
        return data["backend_results"]
    return data


def read_previous_attention_result() -> dict[str, Any] | None:
    if not OUT_JSON.exists():
        return None
    try:
        data = json.loads(OUT_JSON.read_text())
    except Exception:
        return None
    if data.get("phase") == "T1-PQ":
        return data.get("previous_s_axis")
    return data


def read_existing_t1pq_report() -> dict[str, Any] | None:
    if not OUT_JSON.exists():
        return None
    try:
        data = json.loads(OUT_JSON.read_text())
    except Exception:
        return None
    return data if data.get("phase") == "T1-PQ" else None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--w", type=int, default=128)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--calib-samples", type=int, default=4)
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--depgraph-timeout-s", type=int, default=20)
    parser.add_argument("--skip-backend", action="store_true")
    parser.add_argument(
        "--tvm-results",
        default="",
        help="Path to JSON generated by scripts/phase2/t1_attention_tvm_bench.py on H800 TVM.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export representative attention ONNX files for separate TVM experiments.",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Comma-separated subset: linear_256x256,linear_qkv,ffn,mswin_bwa",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.h % 16 != 0 or args.w % 16 != 0:
        raise ValueError("--h and --w must be divisible by 16 for mswin_bwa ws=16")

    model, hypes = load_model(args.device)
    p_axis = analyze_p_axis(model, hypes, args.depgraph_timeout_s)
    p_gate = gate_p_axis(
        p_axis["pruning_groups"],
        p_axis["depgraph_checks"],
        manual_scanner=p_axis.get("manual_scanner"),
    )

    targets_filter = {x.strip() for x in args.targets.split(",") if x.strip()} or None
    input_shape = (args.batch, args.agents, args.h, args.w, args.channels)
    backend_results = {}
    export_results = {}
    if args.export_onnx:
        export_results = export_attention_onnx_targets(
            model=model,
            input_shape=input_shape,
            device=args.device,
            targets_filter=targets_filter,
        )
    if not args.skip_backend:
        tvm_results_path = None
        if args.tvm_results:
            tvm_results_path = Path(args.tvm_results)
            if not tvm_results_path.is_absolute():
                tvm_results_path = REPO_ROOT / tvm_results_path
        backend_results.update(read_tvm_backend_results(tvm_results_path))
    q_gate = gate_q_axis(backend_results)
    overall = gate_overall(p_gate, q_gate)
    coupling_table = build_attention_coupling_table(
        backend_results,
        load_attention_accuracy_priors(),
    )

    report = {
        "phase": "T1-PQ",
        "description": "V2X-ViT transformer P/Q axis feasibility; TVM-only Q gate",
        "date": "2026-06-23",
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ckpt": str(CKPT_FILE),
        "config": str(CONFIG_YAML),
        "input_shape": list(input_shape),
        "q_backend_policy": "TVM_ONLY_NO_TENSORRT_GATE",
        "tvm_results": args.tvm_results or None,
        "previous_s_axis": read_previous_attention_result(),
        "p_axis": p_axis,
        "onnx_exports": export_results,
        "backend_results": backend_results,
        "coupling_table": coupling_table,
        "gate": {
            "p_axis": p_gate,
            "q_axis": q_gate,
            "overall": overall,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    OUT_MD.write_text(build_markdown_report(report))
    print(json.dumps(report["gate"], indent=2, ensure_ascii=False))
    print(f"[written] {OUT_JSON}")
    print(f"[written] {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
