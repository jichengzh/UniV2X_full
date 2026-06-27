"""Validate T1 attention prune+quant end-to-end evidence.

This checker encodes the handoff contract for
``results/attention_e2e_pq_v1.{csv,json,md}`` and
``results/attention_e2e_pq_blocker_v1.md``. It does not generate metrics; it
only prevents unsupported claims from entering the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_CONFIGS = {
    "baseline",
    "attention-p50-fp16",
    "attention-p50-int8/mixed",
}

REQUIRED_ROW_FIELDS = (
    "config",
    "attention_prune_pct",
    "quant",
    "finetune",
    "checkpoint_path",
    "latency_scope",
    "e2e_latency_ms",
    "speedup",
    "ap50",
    "ap70",
    "delta_ap50",
    "delta_ap70",
    "latency_command",
    "latency_log",
    "ap_command",
    "ap_log",
    "dataset_split",
    "n_samples",
)

AP_GAIN_AUDIT_FIELDS = (
    "same_eval_protocol",
    "same_dataset_split",
    "same_checkpoint_family",
    "same_thresholds",
    "finetune_epochs",
    "learning_rate",
    "seed",
)

FAKE_AP_TOKENS = ("simulated", "fake", "prior", "not_true")
NON_TVM_INT8_TOKENS = ("tensorrt", "trt", "qdq", "fake")
BLOCKER_TOKENS = (
    "onnx import",
    "relax lowering",
    "tir compile",
    "runtime correctness",
    "ap eval",
    "model surgery shape mismatch",
    "softmax",
    "relation einsum",
    "not_implemented",
)


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def _issue(row: str, message: str) -> dict[str, str]:
    return {"row": row, "message": message}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_fake_ap_status(row: dict[str, Any]) -> bool:
    status_text = " ".join(
        str(row.get(key, ""))
        for key in (
            "accuracy_status",
            "ap_status",
            "ap_source",
            "ap_source_status",
            "evidence",
        )
    ).lower()
    return any(token in status_text for token in FAKE_AP_TOKENS)


def _is_int8_row(row: dict[str, Any]) -> bool:
    return "int8" in str(row.get("quant", "")).lower()


def _validate_pruned_row(row: dict[str, Any], errors: list[dict[str, str]]) -> None:
    name = str(row.get("config", "<unknown>"))
    prune_pct = _float_or_none(row.get("attention_prune_pct"))
    if prune_pct is None or prune_pct <= 0:
        return

    if _is_missing(row.get("checkpoint_path")):
        errors.append(_issue(name, "pruned rows require checkpoint_path"))
    if _is_missing(row.get("manifest_path")):
        errors.append(_issue(name, "pruned rows require manifest_path"))

    manifest = row.get("prune_manifest")
    if not isinstance(manifest, dict):
        errors.append(_issue(name, "pruned rows require inline prune_manifest summary"))
        return
    if _is_missing(manifest.get("hmsa_keep_heads")):
        errors.append(_issue(name, "prune_manifest requires hmsa_keep_heads"))
    if _is_missing(manifest.get("mswin_keep_heads")):
        errors.append(_issue(name, "prune_manifest requires mswin_keep_heads"))
    dim_preserved = bool(
        manifest.get("dim_256_preserved")
        or manifest.get("dim_preserved")
        or manifest.get("preserve_dim_256")
    )
    if not dim_preserved:
        errors.append(_issue(name, "dim=256 must be preserved for residual/detector compatibility"))


def _validate_ap_gain_audit(row: dict[str, Any], warnings: list[dict[str, str]]) -> None:
    name = str(row.get("config", "<unknown>"))
    delta_ap50 = _float_or_none(row.get("delta_ap50"))
    delta_ap70 = _float_or_none(row.get("delta_ap70"))
    if (delta_ap50 is None or delta_ap50 <= 0) and (delta_ap70 is None or delta_ap70 <= 0):
        return

    audit = row.get("ap_gain_audit")
    if not isinstance(audit, dict):
        warnings.append(_issue(
            name,
            "AP improves over baseline; provide ap_gain_audit for epoch/lr/seed/eval protocol consistency",
        ))
        return

    missing = [key for key in AP_GAIN_AUDIT_FIELDS if _is_missing(audit.get(key))]
    false_flags = [
        key for key in (
            "same_eval_protocol",
            "same_dataset_split",
            "same_checkpoint_family",
            "same_thresholds",
        )
        if audit.get(key) is not True
    ]
    if missing or false_flags:
        warnings.append(_issue(
            name,
            "AP improves over baseline; ap_gain_audit is incomplete or inconsistent",
        ))


def _validate_metric_thresholds(row: dict[str, Any], warnings: list[dict[str, str]]) -> None:
    name = str(row.get("config", "<unknown>"))
    if name == "attention-p50-int8/mixed":
        speedup = _float_or_none(row.get("speedup"))
        if speedup is not None and speedup < 1.10:
            warnings.append(_issue(name, "pruned+int8 e2e speedup is below 1.10x Stop-C threshold"))

    for metric in ("delta_ap50", "delta_ap70"):
        delta = _float_or_none(row.get(metric))
        if delta is not None and delta < -0.02:
            warnings.append(_issue(name, f"{metric} is below -0.02 accuracy guardrail"))


def validate_stop_a_report(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the Stop-A result JSON schema and experiment口径."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        return {
            "verdict": "REJECT",
            "errors": [_issue("report", "report requires non-empty rows list")],
            "warnings": warnings,
        }

    dataset = report.get("dataset", {})
    if not isinstance(dataset, dict):
        errors.append(_issue("report", "dataset must be a dict with split/samples/eval_script"))
    else:
        for field in ("split", "samples", "eval_script"):
            if _is_missing(dataset.get(field)):
                errors.append(_issue("report", f"dataset.{field} is required"))

    configs = {str(row.get("config", "")) for row in rows}
    missing_configs = sorted(REQUIRED_CONFIGS - configs)
    if missing_configs:
        errors.append(_issue("report", f"missing required rows: {missing_configs}"))

    for row in rows:
        if not isinstance(row, dict):
            errors.append(_issue("report", "each row must be a dict"))
            continue
        name = str(row.get("config", "<unknown>"))
        for field in REQUIRED_ROW_FIELDS:
            if _is_missing(row.get(field)):
                errors.append(_issue(name, f"{field} is required"))

        scope = str(row.get("latency_scope", "")).lower()
        if scope != "e2e":
            errors.append(_issue(name, "latency_scope must be e2e for Stop-A; direct/fusion-only results are not final e2e"))

        if _has_fake_ap_status(row):
            errors.append(_issue(name, "fake/prior AP cannot be used as real attention pruning AP"))

        if _is_int8_row(row):
            backend = str(row.get("quant_backend", "")).lower()
            if "tvm" not in backend or any(token in backend for token in NON_TVM_INT8_TOKENS):
                errors.append(_issue(name, "INT8/mixed rows must use TVM runtime/build evidence"))

        _validate_pruned_row(row, errors)
        _validate_ap_gain_audit(row, warnings)
        _validate_metric_thresholds(row, warnings)

    verdict = "REJECT" if errors else ("REVISE" if warnings else "ACCEPTABLE_E2E_SCHEMA")
    return {
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
    }


def validate_blocker_markdown(text: str) -> dict[str, Any]:
    """Validate Stop-B blocker text is specific and still reports partial evidence."""
    errors: list[dict[str, str]] = []
    lowered = text.lower()

    has_concrete_blocker = any(token in lowered for token in BLOCKER_TOKENS)
    if "not_implemented" in lowered and "minimal" in lowered and "hmsa" in lowered:
        has_concrete_blocker = True
    if not has_concrete_blocker:
        errors.append(_issue("blocker", "blocker must name a concrete failing step/op"))
    if "baseline" not in lowered or ("latency" not in lowered and "ap" not in lowered):
        errors.append(_issue("blocker", "blocker must include baseline e2e latency/AP evidence"))
    if "attention-p50-fp16" not in lowered:
        errors.append(_issue("blocker", "blocker must include attention-p50-fp16 partial evidence"))
    if "tvm" not in lowered:
        errors.append(_issue("blocker", "blocker must state the TVM path being blocked"))

    return {
        "verdict": "REJECT" if errors else "ACTIONABLE_BLOCKER",
        "errors": errors,
        "warnings": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", default="results/attention_e2e_pq_v1.json")
    parser.add_argument("--blocker-md", default="results/attention_e2e_pq_blocker_v1.md")
    parser.add_argument(
        "--mode",
        choices=("auto", "stop-a", "stop-b"),
        default="auto",
        help="auto validates Stop-A if JSON exists, otherwise Stop-B if blocker exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = Path(args.report_json)
    blocker_path = Path(args.blocker_md)

    if args.mode in ("auto", "stop-a") and report_path.exists():
        validation = validate_stop_a_report(json.loads(report_path.read_text()))
    elif args.mode in ("auto", "stop-b") and blocker_path.exists():
        validation = validate_blocker_markdown(blocker_path.read_text())
    else:
        validation = {
            "verdict": "MISSING_RESULT",
            "errors": [_issue("report", "no Stop-A report JSON or Stop-B blocker markdown exists")],
            "warnings": [],
        }

    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["verdict"] in {"ACCEPTABLE_E2E_SCHEMA", "ACTIONABLE_BLOCKER"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
