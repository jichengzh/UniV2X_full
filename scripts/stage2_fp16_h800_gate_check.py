#!/usr/bin/env python3
"""Check whether Stage2 FP16 lhc_07 rewritten full-engine gates are satisfied.

This checker does not import TVM. It reads JSON artifacts produced by
stage2_fp16_tensorcore_convblock_and_engine_probe.py and reports whether the
result can be treated as an H800/sm90 full-engine TensorCore measurement.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/home/jichengzhi/V2X")
DEFAULT_EXPORT_DIR = (
    REPO_ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/"
    / "original60_quant_20260627/exports"
)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing:{path}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:  # pragma: no cover - defensive parsing path
        return None, f"parse_failed:{path}:{exc!r}"


def _target_arch(target: Any) -> str | None:
    if target is None:
        return None
    text = target if isinstance(target, str) else json.dumps(target, sort_keys=True)
    match = re.search(r'"arch"\s*:\s*"(sm_[0-9]+)"', text)
    if match:
        return match.group(1)
    match = re.search(r"sm_[0-9]+", text)
    return match.group(0) if match else None


def _counts_has_tensorcore(counts: dict[str, Any] | None) -> bool:
    counts = counts or {}
    return int(counts.get("wmma") or 0) > 0 and int(counts.get("tvm_mma_sync") or 0) > 0


def _output_compare_summary(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    items = items or []
    if not items:
        return {"present": False}
    return {
        "present": True,
        "max_abs_err_max": max(float(item.get("max_abs_err") or 0.0) for item in items),
        "mean_abs_err_max": max(float(item.get("mean_abs_err") or 0.0) for item in items),
        "mean_rel_max": max(
            float(item.get("mean_abs_err_over_original_abs_mean") or 0.0)
            for item in items
            if item.get("mean_abs_err_over_original_abs_mean") is not None
        )
        if any(item.get("mean_abs_err_over_original_abs_mean") is not None for item in items)
        else None,
        "output_count": len(items),
    }


def _check_group_conv_rewrite(export_dir: Path, require_sm90: bool) -> dict[str, Any]:
    path = export_dir / "fp16_lhc07_full_engine_group_conv_rewrite_latest.json"
    data, error = _load_json(path)
    check: dict[str, Any] = {
        "name": "group_conv_rewritten_full_engine",
        "path": str(path),
        "required_for_goal": True,
    }
    if error:
        check.update({"status": "missing", "error": error})
        return check
    assert data is not None
    default = data.get("default_full_engine") or {}
    rewritten = data.get("rewritten_full_engine") or {}
    comparison = data.get("comparison") or {}
    arch = _target_arch(data.get("target"))
    default_ms = default.get("latency_mean_ms")
    rewritten_ms = rewritten.get("latency_mean_ms")
    output_summary = _output_compare_summary(rewritten.get("output_compare"))
    target_ok = (arch == "sm_90") if require_sm90 else bool(arch)
    tensorcore_ok = bool(rewritten.get("tensorcore_gate")) or _counts_has_tensorcore(
        rewritten.get("scheduled_counts")
    )
    speed_ok = bool(comparison.get("is_faster_than_default")) or (
        isinstance(default_ms, (int, float))
        and isinstance(rewritten_ms, (int, float))
        and rewritten_ms < default_ms
    )
    check.update(
        {
            "status": "pass" if target_ok and tensorcore_ok and speed_ok else "fail",
            "artifact_status": data.get("status"),
            "target": data.get("target"),
            "target_arch": arch,
            "target_gate": target_ok,
            "default_latency_ms": default_ms,
            "rewritten_latency_ms": rewritten_ms,
            "speedup_ratio": comparison.get("speedup_ratio"),
            "speed_gate": speed_ok,
            "tensorcore_gate": tensorcore_ok,
            "rewritten_counts": rewritten.get("scheduled_counts"),
            "output_compare": output_summary,
            "notes": [
                "Requires H800/sm90 target for final table/claim." if require_sm90 else "Target arch was recorded.",
                "Output compare is required for correctness triage, but AP-safe is checked separately.",
            ],
        }
    )
    return check


def _check_one_x_one(export_dir: Path, require_sm90: bool) -> dict[str, Any]:
    path = export_dir / "fp16_lhc07_rewritten_1x1_full_engine_latest.json"
    data, error = _load_json(path)
    check: dict[str, Any] = {"name": "rewritten_1x1_full_engine", "path": str(path)}
    if error:
        check.update({"status": "missing", "error": error})
        return check
    assert data is not None
    default = data.get("original_default_engine") or {}
    rewritten = data.get("rewritten_engine") or {}
    comparison = data.get("comparison") or {}
    arch = _target_arch(data.get("target"))
    default_ms = default.get("latency_mean_ms")
    rewritten_ms = rewritten.get("latency_mean_ms")
    target_ok = (arch == "sm_90") if require_sm90 else bool(arch)
    tensorcore_ok = bool(rewritten.get("tensorcore_gate")) or _counts_has_tensorcore(
        rewritten.get("scheduled_counts")
    )
    speed_ok = (
        isinstance(default_ms, (int, float))
        and isinstance(rewritten_ms, (int, float))
        and rewritten_ms < default_ms
    )
    check.update(
        {
            "status": "pass" if target_ok and tensorcore_ok else "fail",
            "artifact_status": data.get("status"),
            "target_arch": arch,
            "target_gate": target_ok,
            "default_latency_ms": default_ms,
            "rewritten_latency_ms": rewritten_ms,
            "speedup_ratio": comparison.get("speedup_ratio"),
            "speed_gate": speed_ok,
            "tensorcore_gate": tensorcore_ok,
            "rewritten_counts": rewritten.get("scheduled_counts"),
            "interpretation": (
                "1x1 rewrite is a TensorCore route proof, not the final speed solution, "
                "unless speed_gate is true."
            ),
        }
    )
    return check


def _check_self_test(export_dir: Path) -> dict[str, Any]:
    path = export_dir / "fp16_lhc07_probe_self_test_no_tvm_latest.json"
    data, error = _load_json(path)
    check: dict[str, Any] = {"name": "self_test_no_tvm_mapping", "path": str(path)}
    if error:
        check.update({"status": "missing", "error": error})
        return check
    assert data is not None
    candidates = data.get("candidate_sample_checks") or []
    mismatch_total = sum(
        int(item.get("x_mismatch_count") or 0)
        + int(item.get("w_mismatch_count") or 0)
        + int(item.get("b_mismatch_count") or 0)
        for item in candidates
    )
    direct = data.get("direct_vs_im2col_restore") or {}
    check.update(
        {
            "status": "pass" if data.get("status") == "success" and mismatch_total == 0 else "fail",
            "artifact_status": data.get("status"),
            "candidate_count": len(candidates),
            "candidate_mismatch_total": mismatch_total,
            "direct_max_abs_err": direct.get("max_abs_err"),
            "direct_mean_abs_err": direct.get("mean_abs_err"),
            "interpretation": "No-TVM mapping self-test; not a latency/TensorCore/AP gate.",
        }
    )
    return check


def _check_accum_compare(export_dir: Path) -> dict[str, Any]:
    path = export_dir / "fp16_lhc07_group_conv_accum_compare_latest.json"
    data, error = _load_json(path)
    check: dict[str, Any] = {
        "name": "group_conv_accum_compare",
        "path": str(path),
        "required_for_goal": False,
    }
    if error:
        check.update({"status": "missing", "error": error})
        return check
    assert data is not None
    rows = data.get("rows") or []
    scale1 = next((row for row in rows if abs(float(row.get("scale") or 0.0) - 1.0) < 1e-12), None)
    fp16_rel = None
    fp32_rel = None
    fp32_improves = None
    if scale1:
        fp16_rel = (scale1.get("fp16_accum") or {}).get("mean_abs_err_over_original_abs_mean")
        fp32_rel = (scale1.get("fp32_accum") or {}).get("mean_abs_err_over_original_abs_mean")
        if fp16_rel is not None and fp32_rel is not None:
            fp32_improves = float(fp32_rel) < float(fp16_rel)
    check.update(
        {
            "status": "pass" if data.get("status") == "success" and fp32_improves else "fail",
            "artifact_status": data.get("status"),
            "target_arch": _target_arch(data.get("target")),
            "latency_scale1_ms": data.get("latency_scale1_ms"),
            "scale1_fp16_mean_rel": fp16_rel,
            "scale1_fp32_mean_rel": fp32_rel,
            "fp32_improves_error": fp32_improves,
            "interpretation": (
                "Diagnostic only: FP32 accumulation should reduce but may not eliminate drift. "
                "This is not an H800 final latency/AP gate."
            ),
        }
    )
    return check


def _check_h800_preflight(export_dir: Path) -> dict[str, Any]:
    path = export_dir / "fp16_lhc07_h800_suite_preflight_latest.json"
    data, error = _load_json(path)
    check: dict[str, Any] = {
        "name": "h800_suite_preflight",
        "path": str(path),
        "required_for_goal": True,
    }
    if error:
        check.update(
            {
                "status": "missing",
                "error": error,
                "interpretation": (
                    "Run stage2_fp16_h800_rewrite_suite_runner.py first so the final gate records "
                    "whether the active environment is H800/sm90."
                ),
            }
        )
        return check
    assert data is not None
    checks = data.get("checks") or {}
    status = "pass" if data.get("status") == "pass" else "fail"
    check.update(
        {
            "status": status,
            "artifact_status": data.get("status"),
            "failure_reasons": data.get("failure_reasons") or [],
            "python_bin_exists": checks.get("python_bin_exists"),
            "python_bin_executable": checks.get("python_bin_executable"),
            "selected_gpu_name": checks.get("selected_gpu_name"),
            "selected_gpu_compute_cap": checks.get("selected_gpu_compute_cap"),
            "target_arch": "sm_90" if checks.get("selected_gpu_compute_cap") == "9.0" else None,
            "target_gate": bool(checks.get("sm90_gate")),
            "interpretation": (
                "H800/sm90 environment preflight. This must pass before any latency row can be "
                "treated as final H800 evidence."
            ),
        }
    )
    return check


def _check_ap(export_dir: Path) -> dict[str, Any]:
    all_candidates = sorted(export_dir.glob("*fp16*ap*.json")) + sorted(export_dir.glob("*FP16*AP*.json"))
    candidates = [
        path
        for path in all_candidates
        if "lhc" in path.name.lower()
        and ("rewrite" in path.name.lower() or "rewritten" in path.name.lower())
        and ("ap" in path.name.lower())
    ]
    missing_reason = None if candidates else "missing_tvm_rewritten_backbone_to_head_postprocess_bridge"
    return {
        "name": "fp16_rewritten_ap_safe",
        "status": "missing" if not candidates else "manual_review",
        "missing_reason": missing_reason,
        "candidate_paths": [str(path) for path in candidates],
        "ignored_generic_ap_paths": [str(path) for path in all_candidates if path not in candidates],
        "required_for_goal": True,
        "interpretation": (
            "A TVM rewritten full-engine backbone/subnet to head/postprocess AP smoke/full-val artifact "
            "is required before declaring AP-safe. Generic PyTorch true-FP16 AP artifacts do not satisfy "
            "this gate."
        ),
    }


def _overall_status(checks: list[dict[str, Any]]) -> str:
    required = [item for item in checks if item.get("required_for_goal")]
    if any(item.get("status") in {"missing", "fail"} for item in required):
        return "incomplete"
    if any(item.get("status") == "manual_review" for item in required):
        return "manual_review"
    return "pass"


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Stage2 FP16 H800 rewritten full-engine gate check",
        "",
        f"- status: `{report.get('status')}`",
        f"- export_dir: `{report.get('export_dir')}`",
        f"- require_sm90: `{report.get('require_sm90')}`",
        f"- generated_at: `{report.get('generated_at')}`",
        "",
        "| check | status | target_arch | speed_gate | tensorcore_gate | notes |",
        "|---|---|---|---|---|---|",
    ]
    for item in report.get("checks", []):
        notes = item.get("interpretation") or "; ".join(item.get("notes") or [])
        lines.append(
            "| `{name}` | `{status}` | `{arch}` | `{speed}` | `{tc}` | {notes} |".format(
                name=item.get("name"),
                status=item.get("status"),
                arch=item.get("target_arch"),
                speed=item.get("speed_gate"),
                tc=item.get("tensorcore_gate"),
                notes=str(notes).replace("|", "/"),
            )
        )
    lines += [
        "",
        "## Required Remaining Gates",
        "",
        "- H800/sm90 target for rewritten full-engine latency.",
        "- rewritten full-engine latency_ms < default latency_ms on the same H800 run.",
        "- rewritten full-engine TensorCore evidence: `wmma/tvm_mma_sync > 0`.",
        "- rewritten AP smoke/full-val artifact proving AP-safe or clearly marking unacceptable AP.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    parser.add_argument("--no-require-sm90", action="store_true")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    export_dir = Path(args.export_dir)
    require_sm90 = not bool(args.no_require_sm90)
    checks = [
        _check_h800_preflight(export_dir),
        _check_self_test(export_dir),
        _check_accum_compare(export_dir),
        _check_one_x_one(export_dir, require_sm90),
        _check_group_conv_rewrite(export_dir, require_sm90),
        _check_ap(export_dir),
    ]
    report = {
        "schema": "stage2_fp16_h800_gate_check_v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export_dir": str(export_dir),
        "require_sm90": require_sm90,
        "status": _overall_status(checks),
        "checks": checks,
    }
    json_path = Path(args.output_json) if args.output_json else export_dir / "fp16_lhc07_h800_gate_check_latest.json"
    md_path = Path(args.output_md) if args.output_md else export_dir / "fp16_lhc07_h800_gate_check_latest.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(f"[done] json={json_path}")
    print(f"[done] md={md_path}")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
