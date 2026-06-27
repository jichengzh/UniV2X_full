"""Build attention-subnet TVM acceleration tables.

This report is intentionally scoped to the attention subnet only. It combines
MSwin full-window attention and the current HMSA TVM target so pruning and
mixed-INT8 effects can be separated before claiming any full-model end-to-end
speedup.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN_JSON = REPO_ROOT / "results/attention_full_tvm_bench_v2.json"
DEFAULT_OUT_JSON = REPO_ROOT / "results/attention_subnet_accel_v1.json"
DEFAULT_OUT_CSV = REPO_ROOT / "results/attention_subnet_accel_v1.csv"
DEFAULT_OUT_MD = REPO_ROOT / "results/attention_subnet_accel_v1.md"


def _round(value: Any, digits: int = 4) -> float | None:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _latency(report: dict[str, Any], target: str, precision: str) -> float:
    entry = report.get("backend_results", {}).get(target, {}).get(precision, {})
    if entry.get("status") != "OK":
        raise ValueError(f"{target}.{precision} is not OK: {entry}")
    value = entry.get("p50_ms")
    if value is None:
        raise ValueError(f"{target}.{precision}.p50_ms missing")
    return float(value)


def _target_entry(report: dict[str, Any], target: str) -> dict[str, Any]:
    entry = report.get("backend_results", {}).get(target, {})
    if not entry:
        raise ValueError(f"{target} missing from backend_results")
    return entry


def _speedup(reference_ms: float, candidate_ms: float) -> float:
    if candidate_ms <= 0:
        raise ValueError("candidate_ms must be positive")
    return round(reference_ms / candidate_ms, 4)


def _component_scope(report: dict[str, Any], hmsa_target: str) -> dict[str, Any]:
    mswin_entry = _target_entry(report, "mswin_bwa_full_attention")
    hmsa_entry = _target_entry(report, hmsa_target)
    mswin_scope = mswin_entry.get("scope", "full_attention_mixed_int8")
    hmsa_scope = hmsa_entry.get("scope", "full_attention_hmsa_relation_core_mixed_int8")
    static_param_plan = hmsa_entry.get("static_param_plan") or {}
    is_static_hmsa = (
        "hmsa_static_2agent" in hmsa_scope
        or static_param_plan.get("scope") == "hmsa_static_2agent_qkv_relation_out"
    )
    if is_static_hmsa:
        hmsa_label = "HMSA static 2-agent qkv/relation/out"
        subnet_scope = "attention_subnet_mswin_full_window_plus_hmsa_static_2agent_qkv_relation_out"
        caveat = (
            "attention subnet only; not full-model e2e. HMSA static 2-agent target includes "
            "q/k/v and output projections for fixed type order [0,1]. Dynamic HMSA type "
            "dispatch is not covered."
        )
    else:
        hmsa_label = "HMSA minimal relation core"
        subnet_scope = "attention_subnet_mswin_full_window_plus_hmsa_minimal_relation_core"
        caveat = (
            "attention subnet only; not full-model e2e. HMSA target is minimal explicit "
            "2-agent relation core, not full dynamic HGTCavAttention dispatch."
        )
    return {
        "mswin_scope": mswin_scope,
        "hmsa_scope": hmsa_scope,
        "hmsa_label": hmsa_label,
        "subnet_scope": subnet_scope,
        "hmsa_static_param_plan": static_param_plan,
        "hmsa_covers_qkv_projection": bool(static_param_plan.get("covers_qkv_projection", False)),
        "hmsa_covers_output_projection": bool(static_param_plan.get("covers_output_projection", False)),
        "hmsa_covers_dynamic_type_dispatch": bool(
            static_param_plan.get("covers_dynamic_type_dispatch", False)
        ),
        "caveat": caveat,
    }


def _component_row(
    report: dict[str, Any],
    *,
    component_scope: dict[str, Any],
    config: str,
    prune_pct: int,
    quant: str,
    mswin_target: str,
    hmsa_target: str,
    precision: str,
    base_fp16_ms: float,
    same_prune_fp16_ms: float | None,
) -> dict[str, Any]:
    mswin_ms = _latency(report, mswin_target, precision)
    hmsa_ms = _latency(report, hmsa_target, precision)
    total = round(mswin_ms + hmsa_ms, 6)
    return {
        "config": config,
        "attention_prune_pct": prune_pct,
        "quant": quant,
        "backend": "TVM",
        "scope": component_scope["subnet_scope"],
        "mswin_scope": component_scope["mswin_scope"],
        "hmsa_scope": component_scope["hmsa_scope"],
        "hmsa_covers_qkv_projection": component_scope["hmsa_covers_qkv_projection"],
        "hmsa_covers_output_projection": component_scope["hmsa_covers_output_projection"],
        "hmsa_covers_dynamic_type_dispatch": component_scope["hmsa_covers_dynamic_type_dispatch"],
        "mswin_p50_ms": round(mswin_ms, 6),
        "hmsa_p50_ms": round(hmsa_ms, 6),
        "subnet_p50_ms": total,
        "speedup_vs_base_fp16": _speedup(base_fp16_ms, total),
        "speedup_vs_same_prune_fp16": _speedup(same_prune_fp16_ms, total) if same_prune_fp16_ms else 1.0,
        "evidence": "TVM_RUNTIME",
        "caveat": component_scope["caveat"],
    }


def build_attention_subnet_report(tvm_report: dict[str, Any]) -> dict[str, Any]:
    component_scope = _component_scope(tvm_report, "hmsa_full_attention")
    base_fp16_ms = round(
        _latency(tvm_report, "mswin_bwa_full_attention", "fp16")
        + _latency(tvm_report, "hmsa_full_attention", "fp16"),
        6,
    )
    p50_fp16_ms = round(
        _latency(tvm_report, "mswin_bwa_p50_full_attention", "fp16")
        + _latency(tvm_report, "hmsa_p50_full_attention", "fp16"),
        6,
    )
    rows = [
        _component_row(
            tvm_report,
            component_scope=component_scope,
            config="attention-subnet-base-fp16",
            prune_pct=0,
            quant="fp16",
            mswin_target="mswin_bwa_full_attention",
            hmsa_target="hmsa_full_attention",
            precision="fp16",
            base_fp16_ms=base_fp16_ms,
            same_prune_fp16_ms=None,
        ),
        _component_row(
            tvm_report,
            component_scope=component_scope,
            config="attention-subnet-base-mixed-int8",
            prune_pct=0,
            quant="mixed_int8",
            mswin_target="mswin_bwa_full_attention",
            hmsa_target="hmsa_full_attention",
            precision="mixed_int8",
            base_fp16_ms=base_fp16_ms,
            same_prune_fp16_ms=base_fp16_ms,
        ),
        _component_row(
            tvm_report,
            component_scope=component_scope,
            config="attention-subnet-p50-fp16",
            prune_pct=50,
            quant="fp16",
            mswin_target="mswin_bwa_p50_full_attention",
            hmsa_target="hmsa_p50_full_attention",
            precision="fp16",
            base_fp16_ms=base_fp16_ms,
            same_prune_fp16_ms=None,
        ),
        _component_row(
            tvm_report,
            component_scope=component_scope,
            config="attention-subnet-p50-mixed-int8",
            prune_pct=50,
            quant="mixed_int8",
            mswin_target="mswin_bwa_p50_full_attention",
            hmsa_target="hmsa_p50_full_attention",
            precision="mixed_int8",
            base_fp16_ms=base_fp16_ms,
            same_prune_fp16_ms=p50_fp16_ms,
        ),
    ]
    rows_by_config = {row["config"]: row for row in rows}
    p50_fp16_speedup = rows_by_config["attention-subnet-p50-fp16"]["speedup_vs_base_fp16"]
    p50_mixed_vs_same = rows_by_config["attention-subnet-p50-mixed-int8"]["speedup_vs_same_prune_fp16"]
    return {
        "schema_version": "attention_subnet_accel_v1",
        "source": tvm_report.get("source_path", "results/attention_full_tvm_bench_v2.json"),
        "backend": "TVM",
        "component_scope": component_scope,
        "rows": rows,
        "conclusion": {
            "pruning_effect": "POSITIVE_SUBNET_SPEEDUP" if p50_fp16_speedup > 1.0 else "NO_SUBNET_SPEEDUP",
            "mixed_int8_effect": (
                "POSITIVE_VS_SAME_PRUNE_FP16" if p50_mixed_vs_same > 1.0 else "NEGATIVE_VS_SAME_PRUNE_FP16"
            ),
            "subnet_best_config": max(rows, key=lambda row: row["speedup_vs_base_fp16"])["config"],
            "stop_a_ready": False,
            "reason": (
                "This is subnet TVM runtime evidence only. Full-model e2e latency/AP and dynamic "
                "HMSA TVM dispatch remain required for final acceptance."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    component_scope = report.get("component_scope", {})
    lines = [
        "# Attention Subnet TVM Acceleration v1",
        "",
        "Scope: attention subnet only, not full-model e2e.",
        f"Components: MSwin full-window attention + {component_scope.get('hmsa_label', 'HMSA target')}.",
        "",
        "| config | prune | quant | subnet p50 ms | speedup vs base fp16 | speedup vs same-prune fp16 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['config']} | {row['attention_prune_pct']}% | {row['quant']} | "
            f"{row['subnet_p50_ms']:.6f} | {row['speedup_vs_base_fp16']:.4f}x | "
            f"{row['speedup_vs_same_prune_fp16']:.4f}x |"
        )
    conclusion = report["conclusion"]
    lines += [
        "",
        "## Conclusion",
        f"- pruning_effect: `{conclusion['pruning_effect']}`",
        f"- mixed_int8_effect: `{conclusion['mixed_int8_effect']}`",
        f"- best_subnet_config: `{conclusion['subnet_best_config']}`",
        f"- stop_a_ready: `{conclusion['stop_a_ready']}`",
        "",
        f"Caveat: {component_scope.get('caveat', 'This report must not be used as final e2e evidence.')}",
    ]
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvm-report", default=str(DEFAULT_IN_JSON))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-csv", default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tvm_report_path = Path(args.tvm_report)
    tvm_report = json.loads(tvm_report_path.read_text())
    tvm_report.setdefault("source_path", str(tvm_report_path))
    report = build_attention_subnet_report(tvm_report)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    write_csv(report["rows"], Path(args.out_csv))
    Path(args.out_md).write_text(render_markdown(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
