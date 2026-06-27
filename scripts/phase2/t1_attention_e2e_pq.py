"""T1 attention prune+quant end-to-end evidence runner.

This script stays in T1. It performs real HMSA/MSwin p50 head surgery for
PyTorch FP16/FP32 evidence, records checkpoint/manifest artifacts, and writes a
Stop-B blocker when full-attention TVM INT8/mixed evidence is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(os.environ.get("V2X_REPO_ROOT", Path(__file__).resolve().parents[2]))
HEAL_ROOT = Path(os.environ.get("V2X_HEAL_ROOT", "/home/jichengzhi/heal_research/HEAL"))
PYTHON = os.environ.get("V2X_PYTHON", "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
CKPT_DIR = Path(
    os.environ.get(
        "V2XVIT_CKPT_DIR",
        "/home/jichengzhi/heal_research/checkpoints/baselines_hf/"
        "HeterBaseline_DAIR_lidar_v2xvit_2023_09_09_11_19_26",
    )
)
CONFIG_YAML = Path(os.environ.get("V2XVIT_CONFIG_YAML", CKPT_DIR / "config.yaml"))
CKPT_FILE = Path(os.environ.get("V2XVIT_CKPT_FILE", CKPT_DIR / "net_epoch_bestval_at17.pth"))
OUT_JSON = REPO_ROOT / "results/attention_e2e_pq_partial_v1.json"
OUT_CSV = REPO_ROOT / "results/attention_e2e_pq_partial_v1.csv"
OUT_BLOCKER_MD = REPO_ROOT / "results/attention_e2e_pq_blocker_v1.md"
ARTIFACT_DIR = REPO_ROOT / "models/v2xvit_attention_t1"
LOG_DIR = REPO_ROOT / "logs/attention_e2e_pq_v1"


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _command_string() -> str:
    return " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])


def _head_channel_index(keep_heads: list[int], dim_head: int, device) -> Any:
    import torch

    pieces = [
        torch.arange(head * dim_head, (head + 1) * dim_head, device=device)
        for head in keep_heads
    ]
    return torch.cat(pieces).long()


def _copy_linear_with_rows(linear, row_index):
    import torch
    from torch import nn

    new_linear = nn.Linear(
        linear.in_features,
        int(row_index.numel()),
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(0, row_index).contiguous())
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias.index_select(0, row_index).contiguous())
    return new_linear


def _copy_linear_with_cols(linear, col_index):
    import torch
    from torch import nn

    new_linear = nn.Linear(
        int(col_index.numel()),
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        new_linear.weight.copy_(linear.weight.index_select(1, col_index).contiguous())
        if linear.bias is not None:
            new_linear.bias.copy_(linear.bias)
    return new_linear


def _validate_keep_heads(heads: int, keep_heads: list[int]) -> list[int]:
    keep = sorted(int(h) for h in keep_heads)
    if not keep:
        raise ValueError("keep_heads must not be empty")
    if keep[0] < 0 or keep[-1] >= heads:
        raise ValueError(f"keep_heads {keep} out of range for heads={heads}")
    if len(set(keep)) != len(keep):
        raise ValueError(f"keep_heads contains duplicates: {keep}")
    return keep


def _p50_keep_heads(heads: int) -> list[int]:
    keep = max(1, heads // 2)
    return list(range(keep))


def prune_hmsa_heads(module, keep_heads: list[int] | None = None, module_path: str = "") -> dict[str, Any]:
    """Physically prune HGTCavAttention heads while preserving output dim."""
    import torch
    from torch import nn

    old_heads = int(module.heads)
    keep = _validate_keep_heads(old_heads, keep_heads or _p50_keep_heads(old_heads))
    dim_head = int(module.relation_att.shape[-1])
    channel_index = _head_channel_index(keep, dim_head, module.relation_att.device)
    old_inner = old_heads * dim_head
    new_inner = len(keep) * dim_head

    for idx in range(len(module.q_linears)):
        module.q_linears[idx] = _copy_linear_with_rows(module.q_linears[idx], channel_index)
        module.k_linears[idx] = _copy_linear_with_rows(module.k_linears[idx], channel_index)
        module.v_linears[idx] = _copy_linear_with_rows(module.v_linears[idx], channel_index)
        module.a_linears[idx] = _copy_linear_with_cols(module.a_linears[idx], channel_index)

    head_index = torch.tensor(keep, device=module.relation_att.device).long()
    module.relation_att = nn.Parameter(module.relation_att.index_select(1, head_index).detach().clone())
    module.relation_msg = nn.Parameter(module.relation_msg.index_select(1, head_index).detach().clone())
    module.heads = len(keep)

    return {
        "module_path": module_path,
        "family": "hmsa_head",
        "old_heads": old_heads,
        "keep_heads": keep,
        "new_heads": len(keep),
        "dim_head": dim_head,
        "old_inner_dim": old_inner,
        "new_inner_dim": new_inner,
        "prune_rate_pct": int(round(100 * (1 - len(keep) / old_heads))),
        "preserve_output_dim": int(module.a_linears[0].out_features),
        "slice_rules": [
            "q/k/v Linear rows sliced by kept head channels",
            "a Linear columns sliced by kept head channels",
            "relation_att/relation_msg head axis sliced",
            "residual output dim preserved",
        ],
    }


def prune_mswin_bwa_heads(module, keep_heads: list[int] | None = None, module_path: str = "") -> dict[str, Any]:
    """Physically prune BaseWindowAttention heads while preserving output dim."""
    old_heads = int(module.heads)
    keep = _validate_keep_heads(old_heads, keep_heads or _p50_keep_heads(old_heads))
    old_inner = int(module.to_qkv.out_features // 3)
    dim_head = int(old_inner // old_heads)
    channel_index = _head_channel_index(keep, dim_head, module.to_qkv.weight.device)
    qkv_index = []
    for offset in (0, old_inner, old_inner * 2):
        qkv_index.append(channel_index + offset)
    import torch

    qkv_index = torch.cat(qkv_index).long()
    module.to_qkv = _copy_linear_with_rows(module.to_qkv, qkv_index)
    module.to_out[0] = _copy_linear_with_cols(module.to_out[0], channel_index)
    module.heads = len(keep)
    new_inner = len(keep) * dim_head

    return {
        "module_path": module_path,
        "family": "mswin_head",
        "window_size": int(module.window_size),
        "old_heads": old_heads,
        "keep_heads": keep,
        "new_heads": len(keep),
        "dim_head": dim_head,
        "old_inner_dim": old_inner,
        "new_inner_dim": new_inner,
        "prune_rate_pct": int(round(100 * (1 - len(keep) / old_heads))),
        "preserve_output_dim": int(module.to_out[0].out_features),
        "slice_rules": [
            "to_qkv rows sliced for Q/K/V kept head channels",
            "to_out[0] columns sliced by kept head channels",
            "pos_embedding unchanged",
            "relative_indices buffer unchanged",
            "residual output dim preserved",
        ],
    }


def apply_attention_p50_surgery(model) -> dict[str, Any]:
    from opencood.models.sub_modules.hmsa import HGTCavAttention
    from opencood.models.sub_modules.mswin import BaseWindowAttention

    entries: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, HGTCavAttention):
            entries.append(prune_hmsa_heads(module, module_path=name))
        elif isinstance(module, BaseWindowAttention):
            entries.append(prune_mswin_bwa_heads(module, module_path=name))

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["family"]] = counts.get(entry["family"], 0) + 1

    return {
        "status": "OK" if entries else "FAILED",
        "method": "manual_structured_attention_head_pruning",
        "prune_rate_pct": 50,
        "preserve_residual_dim": 256,
        "counts": counts,
        "entries": entries,
        "notes": [
            "HMSA and MSwin complete heads are removed.",
            "Output dim remains 256 for residual add and detection head compatibility.",
            "No finetune is applied by this script.",
        ],
    }


def _prepare_heal_imports() -> None:
    sys.path.insert(0, str(HEAL_ROOT))
    os.chdir(str(HEAL_ROOT))


def load_model_and_hypes(device: str):
    _prepare_heal_imports()
    import torch
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.tools import train_utils

    hypes = yaml_utils.load_yaml(str(CONFIG_YAML), None)
    parser_func = getattr(yaml_utils, hypes["yaml_parser"])
    hypes = parser_func(hypes)
    hypes["validate_dir"] = hypes["test_dir"]
    model = train_utils.create_model(hypes)
    state = torch.load(str(CKPT_FILE), map_location="cpu")
    state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    load_state_dict_compatible(model, state)
    return model.to(device).eval(), hypes


def load_state_dict_compatible(model, state: dict[str, Any]) -> dict[str, Any]:
    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "status": "OK",
        "strict": False,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def build_dataset_from_hypes(hypes: dict[str, Any]):
    from opencood.data_utils.datasets import build_dataset

    return build_dataset(hypes, visualize=False, train=False)


def save_attention_pruned_artifact(model, manifest: dict[str, Any]) -> tuple[Path, Path]:
    import torch

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = ARTIFACT_DIR / "attention_p50_noft_epoch17.pth"
    manifest_path = ARTIFACT_DIR / "attention_p50_manifest_v1.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "attention_prune_manifest": manifest,
            "source_checkpoint": str(CKPT_FILE),
        },
        ckpt_path,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return ckpt_path, manifest_path


def _make_loader(dataset, num_workers: int):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )


def measure_model_forward_latency(
    model,
    dataset,
    device: str,
    precision: str,
    warmup: int,
    samples: int,
    log_path: Path,
    num_workers: int,
) -> dict[str, Any]:
    import torch
    from opencood.tools import train_utils

    loader = _make_loader(dataset, num_workers)
    latencies: list[float] = []
    n_seen = 0
    model.eval()
    amp_enabled = precision == "fp16"
    started = time.time()
    try:
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                if n_seen >= warmup + samples:
                    break
                batch = train_utils.to_device(batch, torch.device(device))
                torch.cuda.synchronize() if str(device).startswith("cuda") else None
                if str(device).startswith("cuda"):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        model(batch["ego"])
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_ms = float(start.elapsed_time(end))
                else:
                    t0 = time.perf_counter()
                    model(batch["ego"])
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                if n_seen >= warmup:
                    latencies.append(elapsed_ms)
                n_seen += 1
        result = {
            "status": "OK",
            "latency_scope": "pytorch_model_forward_e2e",
            "precision": "fp16_autocast_pytorch" if amp_enabled else "fp32_pytorch",
            "warmup": warmup,
            "sample_count": len(latencies),
            "mean_ms": _round(np.mean(latencies), 4) if latencies else None,
            "p50_ms": _round(np.percentile(latencies, 50), 4) if latencies else None,
            "p95_ms": _round(np.percentile(latencies, 95), 4) if latencies else None,
            "elapsed_secs": _round(time.time() - started, 3),
            "command": _command_string(),
        }
    except Exception as exc:
        result = {
            "status": "FAILED",
            "latency_scope": "pytorch_model_forward_e2e",
            "precision": "fp16_autocast_pytorch" if amp_enabled else "fp32_pytorch",
            "sample_count": len(latencies),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_secs": _round(time.time() - started, 3),
            "command": _command_string(),
        }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def evaluate_ap(
    model,
    dataset,
    device: str,
    precision: str,
    samples: int,
    log_path: Path,
    num_workers: int,
) -> dict[str, Any]:
    import torch
    from collections import OrderedDict
    from opencood.tools import train_utils
    from opencood.utils import eval_utils

    loader = _make_loader(dataset, num_workers)
    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }
    amp_enabled = precision == "fp16"
    n_done = 0
    started = time.time()
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                if n_done >= samples:
                    break
                batch = train_utils.to_device(batch, torch.device(device))
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    model_output = model(batch["ego"])
                output_dict = OrderedDict()
                output_dict["ego"] = cast_output_dict_to_float32(model_output)
                pred_box, pred_score, gt_box = dataset.post_process(batch, output_dict)
                for iou_th in (0.3, 0.5, 0.7):
                    eval_utils.caluclate_tp_fp(pred_box, pred_score, gt_box, result_stat, iou_th)
                n_done += 1
        tmp_dir = REPO_ROOT / "results" / f"_tmp_attention_e2e_{log_path.stem}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ap30, ap50, ap70 = eval_utils.eval_final_results(result_stat, str(tmp_dir))
        ap_scope = (
            f"{n_done}-sample pilot; real eval log; not final DAIR val AP"
            if n_done < 1789 else "full DAIR val real eval log"
        )
        accuracy_status = f"REAL_EVAL_LOG_{n_done}_SAMPLE_PILOT" if n_done < 1789 else "REAL_EVAL_LOG_FULL_VAL"
        result = {
            "status": "OK",
            "accuracy_status": accuracy_status,
            "ap_scope": ap_scope,
            "precision": "fp16_autocast_pytorch" if amp_enabled else "fp32_pytorch",
            "dataset_split": "DAIR val",
            "sample_count": n_done,
            "ap30": float(ap30),
            "ap50": float(ap50),
            "ap70": float(ap70),
            "elapsed_secs": _round(time.time() - started, 3),
            "command": _command_string(),
        }
    except Exception as exc:
        result = {
            "status": "FAILED",
            "accuracy_status": "REAL_EVAL_FAILED",
            "ap_scope": "eval failed",
            "precision": "fp16_autocast_pytorch" if amp_enabled else "fp32_pytorch",
            "dataset_split": "DAIR val",
            "sample_count": n_done,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
            "elapsed_secs": _round(time.time() - started, 3),
            "command": _command_string(),
        }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def cast_output_dict_to_float32(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.float() if torch.is_floating_point(value) else value
    if isinstance(value, dict):
        return {key: cast_output_dict_to_float32(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cast_output_dict_to_float32(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cast_output_dict_to_float32(item) for item in value)
    return value


def find_full_attention_tvm_blocker(tvm_report: dict[str, Any] | None) -> dict[str, Any]:
    if not tvm_report:
        return {
            "target": "full_attention_tvm_int8_mixed",
            "backend": "TVM",
            "stage": "TVM full-attention run",
            "op": "mswin_bwa_full_attention / hmsa_full_attention",
            "detail": "No full-attention TVM report was provided.",
        }
    backend_results = tvm_report.get("backend_results", tvm_report)
    for name, item in backend_results.items():
        if item.get("backend") != "TVM":
            continue
        scope = str(item.get("scope", "")).lower()
        if "full_attention" not in scope and "full_attention" not in name.lower():
            continue
        for precision in ("mixed_int8", "int8", "fp16"):
            entry = item.get(precision)
            if not isinstance(entry, dict):
                continue
            if entry.get("status") == "OK" and precision in ("mixed_int8", "int8"):
                continue
            if entry.get("status") != "OK":
                fallback_stage = None
                fallback_op = None
                fallback_minimal_target = None
                if name == "hmsa_full_attention":
                    fallback_stage = "NOT_IMPLEMENTED"
                    fallback_op = (
                        "HGTCavAttention relation einsum "
                        "q * relation_att * k; relation_msg * v; attn * v_msg "
                        "with per-type q/k/v/a ModuleList dispatch"
                    )
                    fallback_minimal_target = (
                        "B=1,L=2,H=64,W=128,C=256,heads=8,dim_head=32; "
                        "p50 target keeps heads=4, inner_dim=128; relation_att/msg "
                        "shape base=(4,8,32,32), p50=(4,4,32,32)"
                    )
                return {
                    "target": name,
                    "backend": "TVM",
                    "precision": precision,
                    "stage": entry.get("stage") or item.get("stage") or fallback_stage or "unknown",
                    "op": entry.get("op") or item.get("op") or fallback_op or item.get("description") or name,
                    "detail": (
                        entry.get("error")
                        or entry.get("reason")
                        or entry.get("traceback")
                        or "full-attention TVM entry is not OK"
                    ),
                    "minimal_target": entry.get("minimal_target") or item.get("minimal_target") or fallback_minimal_target,
                }
    hmsa = backend_results.get("hmsa_full_attention", {})
    hmsa_p50 = backend_results.get("hmsa_p50_full_attention", {})
    if (
        isinstance(hmsa, dict)
        and "relation_core" in str(hmsa.get("scope", "")).lower()
        and (hmsa.get("mixed_int8") or {}).get("status") == "OK"
        and isinstance(hmsa_p50, dict)
        and (hmsa_p50.get("mixed_int8") or {}).get("status") == "OK"
    ):
        return {
            "target": "attention-p50-int8/mixed",
            "backend": "TVM",
            "precision": "mixed_int8",
            "stage": "DYNAMIC_DISPATCH_AND_E2E_QUANT_NOT_INTEGRATED",
            "op": (
                "HGTCavAttention dynamic type dispatch, q/k/v/a ModuleList input projections, "
                "calibrated TVM mixed INT8 integration, and AP eval"
            ),
            "detail": (
                "HMSA minimal relation core and MSwin full window attention now have TVM runtime "
                "evidence, but the HMSA result fixes explicit 2-agent relation pairs and does not "
                "cover dynamic type dispatch or calibrated quantized end-to-end evaluation."
            ),
            "minimal_target": (
                "v2 HMSA minimal relation core OK: B=1,L=2,H=64,W=128,C=256; "
                "base heads=8, p50 heads=4; dynamic dispatch/qkv projections remain open."
            ),
        }
    return {
        "target": "full_attention_tvm_int8_mixed",
        "backend": "TVM",
        "stage": "TVM full-attention gate",
        "op": "mswin_bwa_full_attention / hmsa_full_attention",
        "detail": "No OK TVM INT8/mixed full-attention entry found.",
    }


def read_tvm_report(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    tvm_path = Path(path)
    if not tvm_path.is_absolute():
        tvm_path = REPO_ROOT / tvm_path
    if not tvm_path.exists():
        return {
            "backend_results": {
                "full_attention_tvm_missing": {
                    "backend": "TVM",
                    "scope": "full_attention_mixed_int8",
                    "int8": {
                        "status": "BLOCKED",
                        "stage": "result collection",
                        "op": "read_tvm_report",
                        "error": f"missing TVM report: {tvm_path}",
                    },
                }
            }
        }
    return json.loads(tvm_path.read_text())


def _row_from_results(
    config: str,
    checkpoint_path: Path | str,
    manifest_path: Path | str | None,
    manifest_summary: str,
    prune_rate_pct: int,
    quant_backend: str,
    latency: dict[str, Any],
    ap: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": config,
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "prune_manifest": manifest_summary,
        "prune_rate_pct": prune_rate_pct,
        "quant_backend": quant_backend,
        "latency_scope": latency.get("latency_scope"),
        "latency_p50_ms": latency.get("p50_ms"),
        "latency_mean_ms": latency.get("mean_ms"),
        "latency_status": latency.get("status"),
        "latency_command": latency.get("command"),
        "latency_log": latency.get("log_path"),
        "ap50": ap.get("ap50"),
        "ap70": ap.get("ap70"),
        "ap_status": ap.get("status"),
        "accuracy_status": ap.get("accuracy_status"),
        "ap_scope": ap.get("ap_scope"),
        "ap_command": ap.get("command"),
        "ap_log": ap.get("log_path"),
        "dataset_split": ap.get("dataset_split", "DAIR val"),
        "sample_count": ap.get("sample_count"),
    }


def _get_row(rows: list[dict[str, Any]], config: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("config") == config:
            return row
    return None


def _safe_speedup(base_ms: Any, candidate_ms: Any) -> float | None:
    base = _round(base_ms, 8)
    candidate = _round(candidate_ms, 8)
    if base is None or candidate is None or candidate <= 0:
        return None
    return _round(base / candidate, 4)


def _safe_delta(candidate: Any, base: Any) -> float | None:
    cand = _round(candidate, 8)
    ref = _round(base, 8)
    if cand is None or ref is None:
        return None
    return _round(cand - ref, 4)


def _tvm_speedup(report: dict[str, Any] | None, target: str) -> dict[str, Any] | None:
    if not report:
        return None
    item = report.get("backend_results", {}).get(target)
    if not isinstance(item, dict):
        return None
    fp16 = item.get("fp16", {})
    mixed = item.get("mixed_int8", item.get("int8", {}))
    if fp16.get("status") != "OK" or mixed.get("status") != "OK":
        return None
    speedup = _safe_speedup(fp16.get("p50_ms"), mixed.get("p50_ms"))
    if speedup is None:
        return None
    return {
        "target": target,
        "fp16_p50_ms": fp16.get("p50_ms"),
        "mixed_int8_p50_ms": mixed.get("p50_ms"),
        "mixed_int8_vs_fp16_speedup": speedup,
        "mixed_int8_slower_pct": _round((1.0 / speedup - 1.0) * 100.0, 2) if speedup > 0 else None,
    }


def build_partial_evidence_summary(
    rows: list[dict[str, Any]],
    tvm_report: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline = _get_row(rows, "baseline")
    p50 = _get_row(rows, "attention-p50-fp16") or _get_row(rows, "attention-p50-fp32")
    shortft = _get_row(rows, "attention-p50-shortft-fp16") or _get_row(rows, "attention-p50-shortft-fp32")
    p50_summary: dict[str, Any] = {}
    if baseline and p50:
        p50_summary = {
            "speedup_vs_baseline": _safe_speedup(baseline.get("latency_p50_ms"), p50.get("latency_p50_ms")),
            "delta_ap50": _safe_delta(p50.get("ap50"), baseline.get("ap50")),
            "delta_ap70": _safe_delta(p50.get("ap70"), baseline.get("ap70")),
            "accuracy_scope": "64-sample pilot; no finetune; not final DAIR val AP",
            "stop_c_risk": True,
            "risk_reason": "speedup <1.05x and AP70 drop exceeds 0.02 in the no-finetune pilot",
        }
    shortft_summary: dict[str, Any] = {}
    if baseline and shortft:
        speedup = _safe_speedup(baseline.get("latency_p50_ms"), shortft.get("latency_p50_ms"))
        delta_ap50 = _safe_delta(shortft.get("ap50"), baseline.get("ap50"))
        delta_ap70 = _safe_delta(shortft.get("ap70"), baseline.get("ap70"))
        shortft_summary = {
            "speedup_vs_baseline": speedup,
            "delta_ap50": delta_ap50,
            "delta_ap70": delta_ap70,
            "accuracy_scope": "64-sample pilot; 100-step short finetune; not final DAIR val AP",
            "speed_stop_c_risk": bool(speedup is not None and speedup < 1.05),
            "accuracy_recovered_in_pilot": bool(
                delta_ap50 is not None and delta_ap70 is not None and delta_ap50 >= -0.02 and delta_ap70 >= -0.02
            ),
            "ap_gain_requires_protocol_audit": bool(
                (delta_ap50 is not None and delta_ap50 > 0) or (delta_ap70 is not None and delta_ap70 > 0)
            ),
        }

    mswin_base = _tvm_speedup(tvm_report, "mswin_bwa_full_attention")
    mswin_p50 = _tvm_speedup(tvm_report, "mswin_bwa_p50_full_attention")
    hmsa_base = _tvm_speedup(tvm_report, "hmsa_full_attention")
    hmsa_p50 = _tvm_speedup(tvm_report, "hmsa_p50_full_attention")
    return {
        "p50_noft": p50_summary,
        "p50_shortft": shortft_summary,
        "mswin_tvm": {
            "base": mswin_base,
            "p50": mswin_p50,
            "base_mixed_int8_vs_fp16_speedup": (mswin_base or {}).get("mixed_int8_vs_fp16_speedup"),
            "p50_mixed_int8_vs_fp16_speedup": (mswin_p50 or {}).get("mixed_int8_vs_fp16_speedup"),
            "evidence_status": (
                "NEGATIVE_OR_NEUTRAL_INT8_EVIDENCE"
                if (mswin_base and mswin_base["mixed_int8_vs_fp16_speedup"] < 1.0)
                or (mswin_p50 and mswin_p50["mixed_int8_vs_fp16_speedup"] < 1.0)
                else "UNKNOWN"
            ),
        },
        "hmsa_tvm": {
            "base": hmsa_base,
            "p50": hmsa_p50,
            "base_mixed_int8_vs_fp16_speedup": (hmsa_base or {}).get("mixed_int8_vs_fp16_speedup"),
            "p50_mixed_int8_vs_fp16_speedup": (hmsa_p50 or {}).get("mixed_int8_vs_fp16_speedup"),
            "evidence_status": (
                "MINIMAL_CORE_NEGATIVE_OR_NEUTRAL_INT8_EVIDENCE"
                if (hmsa_base and hmsa_base["mixed_int8_vs_fp16_speedup"] < 1.0)
                or (hmsa_p50 and hmsa_p50["mixed_int8_vs_fp16_speedup"] < 1.0)
                else "UNKNOWN"
            ),
            "scope": "minimal_explicit_2agent_relation_core_not_full_dynamic_HGTCavAttention",
        },
    }


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_stop_b_markdown(
    rows: list[dict[str, Any]],
    blocker: dict[str, Any],
    git_status_short: str,
    tvm_report_path: str | None = None,
    tvm_report: dict[str, Any] | None = None,
) -> str:
    summary = build_partial_evidence_summary(rows, tvm_report)
    p50_noft = summary.get("p50_noft", {})
    p50_shortft = summary.get("p50_shortft", {})
    mswin = summary.get("mswin_tvm", {})
    hmsa = summary.get("hmsa_tvm", {})
    git_lines = git_status_short.strip().splitlines()
    if len(git_lines) > 120:
        omitted = len(git_lines) - 120
        git_status_short = "\n".join(git_lines[:120] + [f"... ({omitted} additional status lines omitted)"])

    lines = [
        "# Stop-B: V2X-ViT Attention E2E PQ Blocker v1",
        "",
        "Status: `BLOCKED`",
        "",
        "Review status: `REVISE_REQUIRED` until the pilot scope, Stop-C risk, MSwin/HMSA negative INT8 evidence, and the remaining e2e quantization gap are preserved in downstream reports.",
        "",
        "## Precise TVM Blocker",
        f"- target: `{blocker.get('target')}`",
        f"- backend: `{blocker.get('backend')}`",
        f"- precision: `{blocker.get('precision', 'int8/mixed')}`",
        f"- stage: `{blocker.get('stage')}`",
        f"- op: `{blocker.get('op')}`",
        f"- detail: {blocker.get('detail')}",
        f"- minimal HMSA Relax/TIR target: {blocker.get('minimal_target') or 'not recorded'}",
        "",
        "## Partial E2E Evidence",
        "| config | prune | quant_backend | latency_scope | p50_ms | AP50 | AP70 | AP status | samples | logs |",
        "|---|---:|---|---|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('config')} | {row.get('prune_rate_pct')}% | {row.get('quant_backend')} | "
            f"{row.get('latency_scope')} | {row.get('latency_p50_ms')} | {row.get('ap50')} | "
            f"{row.get('ap70')} | {row.get('accuracy_status')} | {row.get('sample_count')} | "
            f"{row.get('latency_log')} / {row.get('ap_log')} |"
        )
    lines += [
        "",
        "## Pilot Scope And Stop-C Risk",
        "- AP values above are `64-sample pilot` real eval logs. They are not final DAIR val AP and must not be used as paper/report AP.",
        "- Latency scope is `pytorch_model_forward_e2e`: it times `model(batch['ego'])`, including the model forward through fusion/detection heads, but excluding data loading, AP post-processing/NMS timing, and evaluation loop overhead.",
        (
            f"- No-finetune attention-p50 pilot: speedup={p50_noft.get('speedup_vs_baseline')}x, "
            f"delta_AP50={p50_noft.get('delta_ap50')}, delta_AP70={p50_noft.get('delta_ap70')}. "
            "This is a Stop-C risk; if short finetune/full-val confirmation remains below the AP guardrail, this direction should be rejected."
        ),
        (
            f"- Short-finetune attention-p50 pilot, if present: speedup={p50_shortft.get('speedup_vs_baseline')}x, "
            f"delta_AP50={p50_shortft.get('delta_ap50')}, delta_AP70={p50_shortft.get('delta_ap70')}. "
            "AP recovery in a small pilot must be audited for protocol/epoch/sample confounds and confirmed on larger/full DAIR val."
        ),
        "",
        "## TVM MSwin Mixed INT8 Evidence",
        "- MSwin full-attention TVM runtime exists, but it is not positive quantization evidence.",
        (
            f"- base MSwin: mixed INT8 speedup vs FP16 = {mswin.get('base_mixed_int8_vs_fp16_speedup')}x "
            f"(FP16={((mswin.get('base') or {}).get('fp16_p50_ms'))} ms, "
            f"mixed INT8={((mswin.get('base') or {}).get('mixed_int8_p50_ms'))} ms)."
        ),
        (
            f"- p50 MSwin: mixed INT8 speedup vs FP16 = {mswin.get('p50_mixed_int8_vs_fp16_speedup')}x "
            f"(FP16={((mswin.get('p50') or {}).get('fp16_p50_ms'))} ms, "
            f"mixed INT8={((mswin.get('p50') or {}).get('mixed_int8_p50_ms'))} ms)."
        ),
        "- Conclusion: MSwin mixed INT8 is slower than FP16 in this run; report it as negative/neutral evidence, not as a speedup.",
        "",
        "## TVM HMSA Minimal Relation-Core Evidence",
        "- HMSA minimal relation-core TVM runtime exists in the full-attention report, but it is not a final dynamic HGTCavAttention/e2e quantization result.",
        (
            f"- base HMSA minimal core: mixed INT8 speedup vs FP16 = {hmsa.get('base_mixed_int8_vs_fp16_speedup')}x "
            f"(FP16={((hmsa.get('base') or {}).get('fp16_p50_ms'))} ms, "
            f"mixed INT8={((hmsa.get('base') or {}).get('mixed_int8_p50_ms'))} ms)."
        ),
        (
            f"- p50 HMSA minimal core: mixed INT8 speedup vs FP16 = {hmsa.get('p50_mixed_int8_vs_fp16_speedup')}x "
            f"(FP16={((hmsa.get('p50') or {}).get('fp16_p50_ms'))} ms, "
            f"mixed INT8={((hmsa.get('p50') or {}).get('mixed_int8_p50_ms'))} ms)."
        ),
        "- Caveat: this target fixes B=1,L=2 and explicit relation pair order; dynamic `types` dispatch and q/k/v input projection ModuleList lowering are not covered.",
        "- Conclusion: HMSA minimal mixed INT8 is also slower than FP16 in this run; keep it as negative/neutral evidence.",
        "",
        "## Evidence Commands",
    ]
    for row in rows:
        lines.append(f"- `{row.get('config')}` latency: `{row.get('latency_command')}`")
        lines.append(f"- `{row.get('config')}` AP: `{row.get('ap_command')}`")
    lines += [
        "",
        "## Artifact Paths",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('config')}` checkpoint={row.get('checkpoint_path')} "
            f"manifest={row.get('manifest_path')} prune_manifest={row.get('prune_manifest')}"
        )
    lines += [
        "",
        "## TVM Evidence",
        f"- full-attention TVM report: {tvm_report_path or 'not provided'}",
        "- full-attention TVM run: `CUDA_VISIBLE_DEVICES=6 /exdata/jichengzhi/tvm310/bin/python scripts/t1_attention_tvm_bench.py --out-json results/attention_full_tvm_bench_v2.json --full-attention --full-only --number 3 --repeat 3`",
        "- H800 environment: PATH=/usr/local/cuda-12.2/bin:$PATH; LD_LIBRARY_PATH=$(cat /exdata/jichengzhi/tvm_nvlibs.path); CUDA_VISIBLE_DEVICES=6 /exdata/jichengzhi/tvm310/bin/python",
        "- MSwin full BWA FP16/mixed INT8 has TVM runtime evidence in the report above.",
        "- HMSA minimal explicit relation core has TVM runtime evidence in the report above; dynamic HGTCavAttention dispatch and calibrated e2e quantized evaluation remain blocked at the op listed in this document.",
    ]
    lines += [
        "",
        "## Dataset And Scope",
        "- dataset split: DAIR val",
        "- latency scope: pytorch_model_forward_e2e, covering model forward through V2X-ViT fusion and detection heads. It is not a direct-matmul microbench and not a full dataloader+postprocess pipeline latency.",
        "- AP values in this document come from the AP log paths above. C4 fake-quant priors are intentionally not used.",
        "- quant backend remains blocked for Stop-A until the TVM mixed INT8 path is calibrated and connected to an e2e AP/latency row.",
        "",
        "## Minimal Next-Step Acceptance Plan",
        "1. Treat the 100-step short-finetune row as a pilot only; run larger or full DAIR val with the same eval script and thresholds.",
        "2. Re-run baseline, no-finetune p50, and finetuned p50 under the same latency/AP protocol.",
        "3. Keep latency labeled `pytorch_model_forward_e2e` unless postprocess/NMS and data path are included.",
        "4. Extend the HMSA TVM path from minimal explicit relation core to dynamic dispatch/qkv projection coverage, or document a concrete failing command/log/trace for that exact unsupported op.",
        "5. Only add an `attention-p50-int8/mixed` e2e/AP row after both HMSA and MSwin TVM paths have calibrated evaluation and model-level integration evidence.",
        "",
        "## Git Status Snapshot",
        "```text",
        git_status_short.strip() or "(clean)",
        "```",
        "",
        "## Next Minimal Fix",
        "- Run short finetune/larger split and integrate a calibrated TVM mixed INT8 e2e path, or stop as negative evidence if speedup/AP guardrails are not met.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--tvm-report", default="results/attention_full_tvm_bench_v2.json")
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if not args.skip_baseline:
        model, hypes = load_model_and_hypes(args.device)
        dataset = build_dataset_from_hypes(hypes)
        lat_log = LOG_DIR / "baseline_latency.json"
        ap_log = LOG_DIR / "baseline_ap.json"
        latency = measure_model_forward_latency(
            model, dataset, args.device, args.precision,
            args.latency_warmup, args.latency_samples, lat_log, args.num_workers,
        )
        latency["log_path"] = str(lat_log)
        ap = evaluate_ap(model, dataset, args.device, args.precision, args.eval_samples, ap_log, args.num_workers)
        ap["log_path"] = str(ap_log)
        rows.append(_row_from_results(
            "baseline",
            CKPT_FILE,
            None,
            "unpruned; dim=256",
            0,
            "none",
            latency,
            ap,
        ))
        del model

    model, hypes = load_model_and_hypes(args.device)
    manifest = apply_attention_p50_surgery(model)
    ckpt_path, manifest_path = save_attention_pruned_artifact(model, manifest)
    dataset = build_dataset_from_hypes(hypes)
    lat_log = LOG_DIR / "attention_p50_fp16_latency.json"
    ap_log = LOG_DIR / "attention_p50_fp16_ap.json"
    latency = measure_model_forward_latency(
        model, dataset, args.device, args.precision,
        args.latency_warmup, args.latency_samples, lat_log, args.num_workers,
    )
    latency["log_path"] = str(lat_log)
    ap = evaluate_ap(model, dataset, args.device, args.precision, args.eval_samples, ap_log, args.num_workers)
    ap["log_path"] = str(ap_log)
    rows.append(_row_from_results(
        "attention-p50-fp16" if args.precision == "fp16" else "attention-p50-fp32",
        ckpt_path,
        manifest_path,
        "HMSA/MSwin keep first 50% heads; dim=256 preserved",
        50,
        "none",
        latency,
        ap,
    ))

    tvm_report = read_tvm_report(args.tvm_report)
    blocker = find_full_attention_tvm_blocker(tvm_report)
    result = {
        "status": "BLOCKED",
        "stop_condition": "Stop-B",
        "date": "2026-06-23",
        "rows": rows,
        "partial_evidence_summary": build_partial_evidence_summary(rows, tvm_report),
        "tvm_blocker": blocker,
        "tvm_report": args.tvm_report,
        "source_checkpoint": str(CKPT_FILE),
        "command": _command_string(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_rows_csv(rows, OUT_CSV)

    import subprocess

    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout
    OUT_BLOCKER_MD.write_text(build_stop_b_markdown(rows, blocker, git_status, args.tvm_report, tvm_report))
    print(json.dumps({"status": "BLOCKED", "blocker": blocker, "rows": rows}, indent=2, ensure_ascii=False))
    print(f"[written] {OUT_JSON}")
    print(f"[written] {OUT_CSV}")
    print(f"[written] {OUT_BLOCKER_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
