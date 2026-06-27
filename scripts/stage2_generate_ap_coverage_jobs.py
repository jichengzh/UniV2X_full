#!/usr/bin/env python3
"""Generate Stage2 AP coverage jobs from artifact-ready candidates.

This script is intentionally offline: it only plans true AP eval/import work and
writes gap reports. It does not run eval commands or contact H800 hosts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import job_plan_row, read_jsonl, utc_timestamp  # noqa: E402


GAP_REPORT_SCHEMA = "stage2_axis_gap_report_v1"
AP_POLICY = "true_eval_or_true_import_only"
RETEST_TAGS = {"paper_retest", "outlier_retest", "cross_gpu_drift_check"}
PREDICTED_TOKENS = (
    "predicted",
    "prediction",
    "proxy",
    "estimated",
    "synthetic",
    "simulated",
    "fake",
    "prior",
    "not_true",
)
TRUE_SOURCE_KINDS = {
    "true_eval",
    "true_import",
    "true_anchor",
    "measured_import",
    "existing_true_eval_anchor",
}
GAP_FIELDS = [
    "candidate_id",
    "config_id",
    "label",
    "latency_status",
    "energy_status",
    "ap_status",
    "claim_status",
    "next_action",
    "reason",
    "run_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-queue", required=True)
    parser.add_argument("--artifact-registry", required=True)
    parser.add_argument("--ap-rows", action="append", default=[])
    parser.add_argument("--out-dir")
    parser.add_argument("--out-jsonl")
    parser.add_argument("--gap-report-csv")
    parser.add_argument("--gap-report-json")
    parser.add_argument("--run-id")
    parser.add_argument("--created-at")
    parser.add_argument("--allow-repeat", action="store_true")
    parser.add_argument("--rows-out-jsonl", default="rows/ap_anchor_rows_v1.jsonl")
    parser.add_argument("--manifest-path", default="candidates/candidate_queue.jsonl")
    parser.add_argument("--registry-path", default="artifacts/artifact_registry_v1.jsonl")
    parser.add_argument("--manifest-digest", default="coverage_pipeline_v1")
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _field(*rows: dict[str, Any] | None, names: str, default: Any = None) -> Any:
    for row in rows:
        if not row:
            continue
        for name in names.split("|"):
            value = row.get(name)
            if value not in (None, ""):
                return value
    return default


def _safe(value: Any) -> str:
    text = _text(value).strip() or "unknown"
    return (
        text.replace(":", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("\\", "_")
    )


def _width_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("|", ",").split(",") if part.strip()]
    else:
        parts = list(value) if isinstance(value, (list, tuple)) else []
    try:
        return tuple(int(part) for part in parts)
    except (TypeError, ValueError):
        return None


def _width_list(*rows: dict[str, Any] | None) -> list[int]:
    value = _field(*rows, names="width", default=[])
    width = _width_tuple(value)
    return list(width or [])


def _has_predicted_token(*values: Any) -> bool:
    text = " ".join(_text(value).lower() for value in values)
    return any(token in text for token in PREDICTED_TOKENS)


def _source_kind(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    return _text(
        _field(
            candidate,
            artifact,
            names="ap_source_kind|source_kind|ap_source_type|source_type",
            default="",
        )
    ).lower()


def _policy(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    return _text(_field(candidate, artifact, names="ap_policy", default=AP_POLICY))


def _source_path(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    return _text(
        _field(
            artifact,
            candidate,
            names="ap_source_path|source_path|ap_eval_source_path|ap_import_source_path",
            default="",
        )
    )


def _path_exists(path_text: str, *, base_dir: Path) -> bool:
    if not path_text or path_text == "unknown":
        return False
    path = Path(path_text).expanduser()
    candidates = [path] if path.is_absolute() else [ROOT / path, base_dir / path, Path.cwd() / path]
    return any(path.exists() for path in candidates)


def _cell_keys(row: dict[str, Any]) -> set[tuple[Any, ...]]:
    keys: set[tuple[Any, ...]] = set()
    candidate_id = row.get("candidate_id")
    config_id = row.get("config_id")
    if candidate_id:
        keys.add(("candidate_id", str(candidate_id)))
    if config_id:
        keys.add(("config_id", str(config_id)))

    label = row.get("label")
    width = _width_tuple(row.get("width"))
    dataset = row.get("dataset")
    split = row.get("split") or row.get("eval_split")
    ckpt = row.get("ckpt") or row.get("ckpt_path")
    protocol = row.get("protocol") or row.get("finetune_protocol") or row.get("eval_protocol")
    semantic_parts = [label, width, dataset, split, ckpt, protocol]
    if label and width:
        keys.add(("label_width", str(label), width))
    if all(part not in (None, "") for part in semantic_parts):
        keys.add(("semantic", *semantic_parts))
    return keys


def _true_measured_ap_row(row: dict[str, Any]) -> bool:
    if row.get("schema") != "ap_anchor_row_v1":
        return False
    if row.get("measurement_status") != "measured":
        return False
    if row.get("metric_value") is None:
        return False
    return not _has_predicted_token(
        row.get("ap_source_kind"),
        row.get("source_kind"),
        row.get("provenance"),
        row.get("notes"),
        row.get("eval_command"),
        row.get("finetune_protocol"),
    )


def _existing_ap_cells(ap_rows: list[dict[str, Any]]) -> set[tuple[Any, ...]]:
    cells: set[tuple[Any, ...]] = set()
    for row in ap_rows:
        if _true_measured_ap_row(row):
            cells.update(_cell_keys(row))
    return cells


def _artifact_score(candidate: dict[str, Any], artifact: dict[str, Any]) -> int:
    score = 0
    if candidate.get("candidate_id") and candidate.get("candidate_id") == artifact.get("candidate_id"):
        score += 100
    if candidate.get("config_id") and candidate.get("config_id") == artifact.get("config_id"):
        score += 80
    if candidate.get("label") and candidate.get("label") == artifact.get("label"):
        score += 20
    if _width_tuple(candidate.get("width")) and _width_tuple(candidate.get("width")) == _width_tuple(artifact.get("width")):
        score += 20
    if _source_path(candidate, artifact):
        score += 5
    return score


def _select_artifact(
    candidate: dict[str, Any],
    artifacts: list[dict[str, Any]],
) -> dict[str, Any] | None:
    scored = [
        (_artifact_score(candidate, artifact), artifact)
        for artifact in artifacts
        if _artifact_score(candidate, artifact) > 0
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def _artifact_ready(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> tuple[bool, str]:
    if artifact is None:
        return False, "artifact_missing"
    status = _text(_field(artifact, candidate, names="artifact_status", default="")).lower()
    if status != "ready":
        return False, f"artifact_status={status or 'unknown'}"
    quality = _text(_field(artifact, candidate, names="quality_status", default="ready")).lower()
    if quality not in {"", "ready", "ok", "pass", "valid", "claimable"}:
        return False, f"quality_status={quality}"
    return True, ""


def _ap_required(candidate: dict[str, Any]) -> bool:
    axes = candidate.get("axes_required")
    if axes is None:
        return True
    return "ap" in {str(axis).lower() for axis in _as_list(axes)}


def _repeat_allowed(candidate: dict[str, Any], artifact: dict[str, Any] | None, allow_repeat: bool) -> bool:
    if allow_repeat:
        return True
    tags: list[str] = []
    for row in (candidate, artifact or {}):
        for key in ("tag", "tags", "repeat_tag", "repeat_policy"):
            tags.extend(str(item) for item in _as_list(row.get(key)) if item)
    return bool({tag.lower() for tag in tags} & RETEST_TAGS)


def _source_rejection_reason(
    candidate: dict[str, Any],
    artifact: dict[str, Any] | None,
    *,
    base_dir: Path,
) -> str | None:
    policy = _policy(candidate, artifact)
    if policy != AP_POLICY:
        return f"ap_policy={policy} is not {AP_POLICY}"

    kind = _source_kind(candidate, artifact)
    if kind and kind not in TRUE_SOURCE_KINDS:
        if _has_predicted_token(kind):
            return f"predicted AP source is blocked: {kind}"
        return f"ap_source_kind={kind} is not an allowed true AP source kind"

    if _has_predicted_token(
        kind,
        _field(candidate, artifact, names="protocol|finetune_protocol|eval_protocol", default=""),
        _field(candidate, artifact, names="provenance|notes|eval_command", default=""),
    ):
        return "predicted AP source is blocked"

    source = _source_path(candidate, artifact)
    if not source:
        return "missing AP source_path"
    if not _path_exists(source, base_dir=base_dir):
        return f"missing AP source_path: {source}"
    return None


def _axis_status(candidate: dict[str, Any], artifact: dict[str, Any] | None, axis: str) -> str:
    row_ids = _as_list((artifact or {}).get(f"{axis}_row_ids"))
    if row_ids:
        return "measured"
    axes = {str(item).lower() for item in _as_list(candidate.get("axes_required"))}
    if not axes:
        return "unknown"
    return "required" if axis in axes else "not_required"


def _config_id(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    value = _field(candidate, artifact, names="config_id", default="")
    if value:
        return str(value)
    width = _width_list(candidate, artifact)
    label = _text(_field(candidate, artifact, names="label", default="unknown"))
    return f"coverage_ap_pyramid_{label}_w{'x'.join(str(item) for item in width)}_fp16"


def _software_point_id(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    value = _field(candidate, artifact, names="software_point_id", default="")
    if value:
        return str(value)
    width = _width_list(candidate, artifact)
    label = _text(_field(candidate, artifact, names="label", default="unknown"))
    return f"pyramid_lidar:backbone:w{'x'.join(str(item) for item in width)}:fp16:{label}"


def _width_csv(candidate: dict[str, Any], artifact: dict[str, Any] | None) -> str:
    return ",".join(str(item) for item in _width_list(candidate, artifact))


def _eval_payload_command(
    *,
    source_path: str,
    dataset: str,
    eval_split: str,
    ckpt_path: str,
    protocol: str,
) -> list[str]:
    code = (
        "import json,sys;"
        "p=sys.argv[1];"
        "txt=open(p,encoding='utf-8').read().strip();"
        "obj=json.loads(txt) if txt.startswith('{') else json.loads(txt.splitlines()[0]);"
        "val=obj.get('metric_value', obj.get('ap70', obj.get('AP70')));"
        "out={"
        "'metric': obj.get('metric','AP70'),"
        "'metric_value': val,"
        "'dataset': obj.get('dataset',sys.argv[2]),"
        "'eval_split': obj.get('eval_split', obj.get('split', sys.argv[3])),"
        "'ckpt_path': obj.get('ckpt_path',sys.argv[4]),"
        "'finetune_protocol': obj.get('finetune_protocol', obj.get('protocol', sys.argv[5])),"
        "'raw_artifact': p,"
        "'source_files': [p],"
        "'provenance': obj.get('provenance','true AP source import')"
        "};"
        "print(json.dumps(out))"
    )
    return ["python3", "-c", code, source_path, dataset, eval_split, ckpt_path, protocol]


def _job_row(
    candidate: dict[str, Any],
    artifact: dict[str, Any] | None,
    *,
    run_id: str,
    created_at: str,
    rows_out_jsonl: str,
    manifest_path: str,
    registry_path: str,
    manifest_digest: str,
) -> dict[str, Any]:
    label = _text(_field(candidate, artifact, names="label", default="unknown"))
    job_run_id = f"{run_id}_ap_{_safe(label)}"
    split = _text(_field(artifact, candidate, names="split|eval_split", default="unknown"))
    ckpt = _text(_field(artifact, candidate, names="ckpt|ckpt_path", default="unknown"))
    protocol = _text(
        _field(
            artifact,
            candidate,
            names="protocol|finetune_protocol|eval_protocol",
            default="unknown",
        )
    )
    candidate_id = _text(_field(candidate, artifact, names="candidate_id", default=""))
    model = _text(_field(candidate, artifact, names="model|model_name", default="unknown"))
    source_path = _source_path(candidate, artifact)
    eval_command = _eval_payload_command(
        source_path=source_path,
        dataset=_text(_field(artifact, candidate, names="dataset", default="unknown")),
        eval_split=split,
        ckpt_path=ckpt,
        protocol=protocol,
    )
    job_id = f"ap:{job_run_id}:{_safe(_field(candidate, artifact, names='candidate_id|config_id', default=label))}"
    command = [
        "python3",
        "scripts/stage2_generate_ap_lut.py",
        "--job-id",
        job_id,
        "--model",
        model,
        "--config-id",
        _config_id(candidate, artifact),
        "--candidate-id",
        candidate_id,
        "--software-point-id",
        _software_point_id(candidate, artifact),
        "--dense-stage",
        _text(_field(candidate, artifact, names="dense_stage", default="backbone")),
        "--optimized-scope",
        _text(_field(candidate, artifact, names="optimized_scope", default="backbone_only")),
        "--width",
        _width_csv(candidate, artifact),
        "--quant-policy",
        _text(_field(candidate, artifact, names="quant_policy", default="fp16")),
        "--schedule-policy",
        "not_applicable",
        "--backend",
        "model_eval",
        "--manifest-digest",
        manifest_digest,
        "--run-id",
        job_run_id,
        "--created-at",
        created_at,
        "--eval-command-json",
        json.dumps(eval_command),
        "--out-jsonl",
        rows_out_jsonl,
    ]
    row = job_plan_row(
        job_id=job_id,
        model=model,
        lut_kind="ap",
        job_type="generate_ap_lut",
        priority=int(_field(candidate, artifact, names="priority", default=50)),
        config_id=_config_id(candidate, artifact),
        manifest_path=manifest_path,
        registry_path=registry_path,
        candidate_id=candidate_id,
        software_point_id=_software_point_id(candidate, artifact),
        expected_output=rows_out_jsonl,
        command=command,
        max_attempts=1,
        timeout_s=21600,
        resource={"gpu": "any_h800_or_cpu_eval", "exclusive": False},
        created_at=created_at,
    )
    row.update(
        {
        "label": label,
        "width": _width_list(candidate, artifact),
        "dataset": _text(_field(artifact, candidate, names="dataset", default="unknown")),
        "split": split,
        "eval_split": split,
        "ckpt": ckpt,
        "ckpt_path": ckpt,
        "protocol": protocol,
        "source_path": source_path,
        "ap_source_kind": _source_kind(candidate, artifact) or "true_import",
        "ap_policy": AP_POLICY,
        "run_id": job_run_id,
        }
    )
    return row


def _gap_row(
    candidate: dict[str, Any],
    artifact: dict[str, Any] | None,
    *,
    ap_status: str,
    claim_status: str,
    next_action: str,
    reason: str,
    run_id: str,
) -> dict[str, str]:
    return {
        "candidate_id": _text(_field(candidate, artifact, names="candidate_id", default="")),
        "config_id": _text(_field(candidate, artifact, names="config_id", default="")),
        "label": _text(_field(candidate, artifact, names="label", default="unknown")),
        "latency_status": _axis_status(candidate, artifact, "latency"),
        "energy_status": _axis_status(candidate, artifact, "energy"),
        "ap_status": ap_status,
        "claim_status": claim_status,
        "next_action": next_action,
        "reason": reason,
        "run_id": run_id,
    }


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    out_dir = Path(args.out_dir) if args.out_dir else None
    job_queue = Path(args.out_jsonl) if args.out_jsonl else None
    gap_csv = Path(args.gap_report_csv) if args.gap_report_csv else None
    gap_json = Path(args.gap_report_json) if args.gap_report_json else None
    if out_dir is not None:
        job_queue = job_queue or out_dir / "ap_job_queue.jsonl"
        gap_csv = gap_csv or out_dir / "axis_gap_report.csv"
        gap_json = gap_json or out_dir / "axis_gap_report.json"
    if job_queue is None or gap_csv is None or gap_json is None:
        raise SystemExit(
            "--out-dir or all of --out-jsonl/--gap-report-csv/--gap-report-json is required"
        )
    return job_queue, gap_csv, gap_json


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_gap_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAP_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in GAP_FIELDS} for row in rows)


def _write_gap_json(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    run_id: str,
    created_at: str,
    queued_jobs: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": GAP_REPORT_SCHEMA,
        "run_id": run_id,
        "created_at": created_at,
        "queued_jobs": queued_jobs,
        "blocked_no_claim": sum(
            1 for row in rows if row.get("ap_status") == "blocked" and row.get("claim_status") == "no_claim"
        ),
        "rows": rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate(
    *,
    candidates: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    ap_rows: list[dict[str, Any]],
    run_id: str,
    created_at: str,
    allow_repeat: bool,
    base_dir: Path,
    rows_out_jsonl: str = "rows/ap_anchor_rows_v1.jsonl",
    manifest_path: str = "candidates/candidate_queue.jsonl",
    registry_path: str = "artifacts/artifact_registry_v1.jsonl",
    manifest_digest: str = "coverage_pipeline_v1",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    existing_cells = _existing_ap_cells(ap_rows)
    jobs: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        artifact = _select_artifact(candidate, artifacts)
        candidate_keys = _cell_keys({**(artifact or {}), **candidate})
        measured_exists = bool(candidate_keys & existing_cells) or bool(
            _as_list((artifact or {}).get("ap_row_ids"))
        )
        allowed_repeat = _repeat_allowed(candidate, artifact, allow_repeat)

        if not _ap_required(candidate):
            gap_rows.append(
                _gap_row(
                    candidate,
                    artifact,
                    ap_status="not_required",
                    claim_status="not_applicable",
                    next_action="none",
                    reason="ap axis not required",
                    run_id=run_id,
                )
            )
            continue

        ready, ready_reason = _artifact_ready(candidate, artifact)
        if not ready:
            gap_rows.append(
                _gap_row(
                    candidate,
                    artifact,
                    ap_status="blocked",
                    claim_status="no_claim",
                    next_action="wait_for_artifact_ready",
                    reason=ready_reason,
                    run_id=run_id,
                )
            )
            continue

        if measured_exists and not allowed_repeat:
            gap_rows.append(
                _gap_row(
                    candidate,
                    artifact,
                    ap_status="measured_exists",
                    claim_status="claimable",
                    next_action="skip_repeat",
                    reason="AP measured cell already exists",
                    run_id=run_id,
                )
            )
            continue

        source_reason = _source_rejection_reason(candidate, artifact, base_dir=base_dir)
        if source_reason:
            gap_rows.append(
                _gap_row(
                    candidate,
                    artifact,
                    ap_status="blocked",
                    claim_status="no_claim",
                    next_action="block_ap_no_claim_missing_true_source",
                    reason=source_reason,
                    run_id=run_id,
                )
            )
            continue

        job = _job_row(
            candidate,
            artifact,
            run_id=run_id,
            created_at=created_at,
            rows_out_jsonl=rows_out_jsonl,
            manifest_path=manifest_path,
            registry_path=registry_path,
            manifest_digest=manifest_digest,
        )
        jobs.append(job)
        gap_rows.append(
            _gap_row(
                candidate,
                artifact,
                ap_status="queued_repeat" if measured_exists else "queued",
                claim_status="pending_retest" if measured_exists else "pending",
                next_action="run_true_ap_import_or_eval",
                reason=(
                    "repeat allowed by tag or --allow-repeat"
                    if measured_exists
                    else "true AP source ready"
                ),
                run_id=str(job["run_id"]),
            )
        )

    return jobs, gap_rows


def main() -> int:
    args = parse_args()
    run_id = args.run_id or f"ap_coverage_{utc_timestamp()}"
    created_at = args.created_at or utc_timestamp()
    job_queue, gap_csv, gap_json = _output_paths(args)

    candidates = read_jsonl(args.candidate_queue)
    artifacts = read_jsonl(args.artifact_registry)
    ap_rows: list[dict[str, Any]] = []
    for path in args.ap_rows:
        ap_rows.extend(read_jsonl(path))

    jobs, gap_rows = generate(
        candidates=candidates,
        artifacts=artifacts,
        ap_rows=ap_rows,
        run_id=run_id,
        created_at=created_at,
        allow_repeat=bool(args.allow_repeat),
        base_dir=Path(args.candidate_queue).resolve().parent,
        rows_out_jsonl=args.rows_out_jsonl,
        manifest_path=args.manifest_path,
        registry_path=args.registry_path,
        manifest_digest=args.manifest_digest,
    )

    _write_jsonl(job_queue, jobs)
    _write_gap_csv(gap_csv, gap_rows)
    _write_gap_json(
        gap_json,
        rows=gap_rows,
        run_id=run_id,
        created_at=created_at,
        queued_jobs=len(jobs),
    )
    print(
        json.dumps(
            {
                "schema": "stage2_ap_coverage_job_generation_summary_v1",
                "run_id": run_id,
                "candidates": len(candidates),
                "jobs": len(jobs),
                "blocked_no_claim": sum(
                    1
                    for row in gap_rows
                    if row.get("ap_status") == "blocked"
                    and row.get("claim_status") == "no_claim"
                ),
                "out_jsonl": str(job_queue),
                "gap_report_csv": str(gap_csv),
                "gap_report_json": str(gap_json),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
