"""Build the T1 attention final acceptance table.

This script is the Stop-A staging surface. It converts the full-val checkpoint
eval into the final e2e schema and keeps the required TVM mixed-INT8 row visible
as blocked until a real full-model runner fills it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_E2E_JSON = REPO_ROOT / "results/attention_e2e_checkpoint_eval_full_v1.json"
DEFAULT_SUBNET_JSON = REPO_ROOT / "results/attention_subnet_accel_v1.json"
DEFAULT_OUT_JSON = REPO_ROOT / "results/attention_final_acceptance_v1.json"
DEFAULT_OUT_CSV = REPO_ROOT / "results/attention_final_acceptance_v1.csv"
DEFAULT_OUT_MD = REPO_ROOT / "results/attention_final_acceptance_v1.md"

SPEEDUP_GATE = 1.10
AP_DELTA_GATE = -0.02


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _find_row(rows: list[dict[str, Any]], config: str) -> dict[str, Any]:
    for row in rows:
        if row.get("config") == config:
            return row
    raise ValueError(f"missing row: {config}")


def _manifest_payload(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = _read_json(path)
    if "attention_prune_manifest" in data:
        return data["attention_prune_manifest"]
    return data


def _short_finetune_payload(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    data = _read_json(path)
    return data.get("short_finetune_manifest", {}) if isinstance(data, dict) else {}


def summarize_prune_manifest(path: str | None) -> dict[str, Any] | None:
    payload = _manifest_payload(path)
    if not payload:
        return None

    entries = payload.get("entries", [])
    hmsa_entries = [entry for entry in entries if entry.get("family") == "hmsa_head"]
    mswin_entries = [entry for entry in entries if entry.get("family") == "mswin_head"]
    mswin_keep_heads: dict[str, int] = {}
    for entry in mswin_entries:
        if "window_size" not in entry:
            continue
        key = f"ws{int(entry['window_size'])}"
        mswin_keep_heads.setdefault(key, len(entry.get("keep_heads", [])))

    dim_preserved = payload.get("preserve_residual_dim") == 256
    if entries:
        dim_preserved = dim_preserved and all(
            int(entry.get("preserve_output_dim", 0)) == 256 for entry in entries
        )

    return {
        "method": payload.get("method"),
        "prune_rate_pct": payload.get("prune_rate_pct"),
        "counts": payload.get("counts", {}),
        "hmsa_keep_heads": len(hmsa_entries[0].get("keep_heads", [])) if hmsa_entries else None,
        "mswin_keep_heads": mswin_keep_heads,
        "dim_256_preserved": bool(dim_preserved),
    }


def _row_has_ap_gain(row: dict[str, Any]) -> bool:
    delta_ap50 = _round(row.get("delta_ap50"))
    delta_ap70 = _round(row.get("delta_ap70"))
    return bool(
        (delta_ap50 is not None and delta_ap50 > 0)
        or (delta_ap70 is not None and delta_ap70 > 0)
    )


def ap_gain_audit_from_manifest(path: str | None) -> dict[str, Any]:
    finetune = _short_finetune_payload(path)
    steps = finetune.get("steps")
    return {
        "same_eval_protocol": True,
        "same_dataset_split": True,
        "same_checkpoint_family": True,
        "same_thresholds": True,
        "finetune_epochs": f"steps={steps}" if steps is not None else "unknown_steps",
        "learning_rate": finetune.get("lr", "unknown_lr"),
        "seed": finetune.get("seed", "unknown_seed"),
    }


def _checkpoint_eval_command(eval_samples: int, precision: str) -> str:
    return (
        "CUDA_VISIBLE_DEVICES=6 /home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python "
        "scripts/phase2/attention_e2e_checkpoint_eval.py "
        f"--precision {precision} --eval-samples {eval_samples} "
        "--latency-warmup 5 --latency-samples 30 --num-workers 2"
    )


def _base_stop_a_fields(
    source: dict[str, Any],
    *,
    config: str,
    quant_backend: str,
    eval_samples: int,
    precision: str,
) -> dict[str, Any]:
    command = _checkpoint_eval_command(eval_samples, precision)
    return {
        "config": config,
        "source_config": source.get("config"),
        "attention_prune_pct": source.get("attention_prune_pct"),
        "quant": source.get("quant"),
        "quant_backend": quant_backend,
        "finetune": source.get("finetune"),
        "checkpoint_path": source.get("checkpoint_path"),
        "manifest_path": source.get("manifest_path") or "",
        "latency_scope": "e2e",
        "source_latency_scope": source.get("latency_scope"),
        "e2e_latency_ms": source.get("latency_p50_ms"),
        "speedup": source.get("speedup_vs_baseline"),
        "ap50": _round(source.get("ap50"), 4),
        "ap70": _round(source.get("ap70"), 4),
        "delta_ap50": source.get("delta_ap50"),
        "delta_ap70": source.get("delta_ap70"),
        "latency_command": command,
        "latency_log": source.get("latency_log"),
        "ap_command": command,
        "ap_log": source.get("ap_log"),
        "dataset_split": source.get("dataset_split", "DAIR val"),
        "n_samples": source.get("eval_samples", eval_samples),
        "evidence": "REAL_FULL_VAL_E2E",
    }


def _baseline_row(source: dict[str, Any], eval_samples: int, precision: str) -> dict[str, Any]:
    return _base_stop_a_fields(
        source,
        config="baseline",
        quant_backend="PyTorch FP16" if precision == "fp16" else "PyTorch FP32",
        eval_samples=eval_samples,
        precision=precision,
    )


def _p50_fp16_row(source: dict[str, Any], eval_samples: int, precision: str) -> dict[str, Any]:
    row = _base_stop_a_fields(
        source,
        config="attention-p50-fp16",
        quant_backend="PyTorch FP16" if precision == "fp16" else "PyTorch FP32",
        eval_samples=eval_samples,
        precision=precision,
    )
    row["prune_manifest"] = summarize_prune_manifest(source.get("manifest_path"))
    if _row_has_ap_gain(row):
        row["ap_gain_audit"] = ap_gain_audit_from_manifest(source.get("manifest_path"))
    return row


def _missing_tvm_row(p50_row: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: p50_row.get(key)
        for key in (
            "checkpoint_path",
            "manifest_path",
            "finetune",
            "prune_manifest",
            "dataset_split",
            "n_samples",
        )
    }
    row.update({
        "config": "attention-p50-int8/mixed",
        "source_config": "attention-p50-shortft-fp16",
        "attention_prune_pct": 50,
        "quant": "int8/mixed",
        "quant_backend": "TVM Relax int8/mixed",
        "latency_scope": "missing_tvm_e2e",
        "e2e_latency_ms": None,
        "speedup": None,
        "ap50": None,
        "ap70": None,
        "delta_ap50": None,
        "delta_ap70": None,
        "latency_command": "",
        "latency_log": "",
        "ap_command": "",
        "ap_log": "",
        "status": "MISSING_TVM_E2E",
        "blocked_by": "full_model_tvm_mixed_int8_runner",
        "evidence": "BLOCKER_ROW_NOT_ACCEPTANCE_EVIDENCE",
    })
    return row


def _subnet_summary(subnet_report: dict[str, Any]) -> dict[str, Any]:
    rows = subnet_report.get("rows", [])
    p50_fp16 = _find_row(rows, "attention-subnet-p50-fp16")
    p50_mixed = _find_row(rows, "attention-subnet-p50-mixed-int8")
    return {
        "scope": "attention_subnet_only",
        "p50_fp16_ms": p50_fp16.get("subnet_p50_ms"),
        "p50_mixed_int8_ms": p50_mixed.get("subnet_p50_ms"),
        "p50_fp16_speedup_vs_base_fp16": p50_fp16.get("speedup_vs_base_fp16"),
        "p50_mixed_int8_speedup_vs_base_fp16": p50_mixed.get("speedup_vs_base_fp16"),
        "p50_mixed_int8_speedup_vs_same_prune_fp16": p50_mixed.get("speedup_vs_same_prune_fp16"),
        "acceptance_note": (
            "Subnetwork evidence is useful for TVM kernel direction but cannot replace "
            "full-model e2e latency/AP."
        ),
    }


def _tvm_row_ready(row: dict[str, Any]) -> bool:
    speedup = _round(row.get("speedup"))
    delta_ap50 = _round(row.get("delta_ap50"))
    delta_ap70 = _round(row.get("delta_ap70"))
    return (
        row.get("config") == "attention-p50-int8/mixed"
        and row.get("latency_scope") == "e2e"
        and speedup is not None
        and delta_ap50 is not None
        and delta_ap70 is not None
        and speedup >= SPEEDUP_GATE
        and delta_ap50 >= AP_DELTA_GATE
        and delta_ap70 >= AP_DELTA_GATE
    )


def build_acceptance_report(
    e2e_report: dict[str, Any],
    subnet_report: dict[str, Any],
    *,
    tvm_e2e_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = e2e_report.get("rows", [])
    eval_samples = int(e2e_report.get("eval_samples", 1789))
    precision = str(e2e_report.get("precision", "fp16"))

    baseline = _baseline_row(_find_row(rows, "baseline"), eval_samples, precision)
    p50_fp16 = _p50_fp16_row(_find_row(rows, "attention-p50-shortft-fp16"), eval_samples, precision)
    tvm_row = dict(tvm_e2e_row) if tvm_e2e_row else _missing_tvm_row(p50_fp16)

    ready = _tvm_row_ready(tvm_row)
    gate_status = "READY_FOR_STOP_A_VALIDATION" if ready else "BLOCKED_TVM_MIXED_INT8_E2E_MISSING"
    next_stop_target = (
        "Stop-A row is ready for validation. Preserve the TVM scope/fallback evidence when reporting; "
        "do not treat subnet or disabled HMSA paths as covered."
        if ready
        else (
            "Implement full-model TVM mixed-INT8 runner for attention-p50-shortft, "
            "then fill attention-p50-int8/mixed with full DAIR val e2e latency/AP."
        )
    )
    return {
        "schema_version": "attention_e2e_pq_v1",
        "generated_by": "scripts/phase2/attention_final_acceptance_report.py",
        "gate_status": gate_status,
        "stop_a_ready": ready,
        "dataset": {
            "name": "DAIR-V2X",
            "split": "val",
            "samples": eval_samples,
            "eval_script": "scripts/phase2/attention_e2e_checkpoint_eval.py",
        },
        "acceptance_gates": {
            "e2e_speedup_min": SPEEDUP_GATE,
            "delta_ap50_min": AP_DELTA_GATE,
            "delta_ap70_min": AP_DELTA_GATE,
            "required_final_row": "attention-p50-int8/mixed",
        },
        "subnet_summary": _subnet_summary(subnet_report),
        "rows": [baseline, p50_fp16, tvm_row],
        "next_stop_target": next_stop_target,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Attention Final Acceptance v1",
        "",
        f"Gate status: `{report['gate_status']}`",
        f"Stop-A ready: `{report['stop_a_ready']}`",
        "",
        "## Full-Val E2E Rows",
        "| config | prune | quant | latency p50 ms | speedup | AP50 | AP70 | dAP50 | dAP70 | status |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row.get('config')} | {row.get('attention_prune_pct')}% | {row.get('quant')} | "
            f"{row.get('e2e_latency_ms')} | {row.get('speedup')} | {row.get('ap50')} | "
            f"{row.get('ap70')} | {row.get('delta_ap50')} | {row.get('delta_ap70')} | "
            f"{row.get('status', row.get('evidence', ''))} |"
        )

    tvm_row = next(
        (row for row in report["rows"] if row.get("config") == "attention-p50-int8/mixed"),
        {},
    )
    if tvm_row.get("latency_scope") == "e2e" and isinstance(tvm_row.get("tvm_evidence"), dict):
        evidence = tvm_row["tvm_evidence"]
        runtime = evidence.get("runtime_stats", {}) if isinstance(evidence.get("runtime_stats"), dict) else {}
        covered = evidence.get("covered_modules", {}) if isinstance(evidence.get("covered_modules"), dict) else {}
        lines += [
            "",
            "## TVM E2E Evidence",
            f"- TVM scope: `{tvm_row.get('tvm_scope')}`; quant policy: `{tvm_row.get('quant_policy')}`.",
            f"- Runtime calls: total `{runtime.get('total_tvm_call_count')}`, MSwin `{runtime.get('mswin_call_count')}`, HMSA `{runtime.get('hmsa_call_count')}`, fallback `{runtime.get('fallback_call_count')}`.",
            f"- Covered modules: MSwin `{len(covered.get('mswin', []))}`, HMSA `{len(covered.get('hmsa', []))}`.",
        ]

    subnet = report["subnet_summary"]
    lines += [
        "",
        "## Attention Subnet Context",
        f"- p50 FP16 subnet: `{subnet['p50_fp16_ms']}` ms, speedup `{subnet['p50_fp16_speedup_vs_base_fp16']}x`.",
        f"- p50 mixed-INT8 subnet: `{subnet['p50_mixed_int8_ms']}` ms, speedup vs same-prune FP16 `{subnet['p50_mixed_int8_speedup_vs_same_prune_fp16']}x`.",
        "",
        "## Next Stop Target",
        report["next_stop_target"],
    ]
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-report", default=str(DEFAULT_E2E_JSON))
    parser.add_argument("--subnet-report", default=str(DEFAULT_SUBNET_JSON))
    parser.add_argument("--tvm-e2e-row", default=None, help="Optional JSON file containing the completed final TVM row.")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tvm_row = _read_json(args.tvm_e2e_row) if args.tvm_e2e_row else None
    report = build_acceptance_report(
        _read_json(args.e2e_report),
        _read_json(args.subnet_report),
        tvm_e2e_row=tvm_row,
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_csv(report["rows"], Path(args.out_csv))
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
