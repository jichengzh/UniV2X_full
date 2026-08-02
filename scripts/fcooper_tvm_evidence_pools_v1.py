#!/usr/bin/env python3
"""Normalize formal F-Cooper TVM feedback into byte-bound Stage6 pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SUCCESS = "measured_success_gold"
FAILURES = {"feasibility_failure", "numerical_feasibility_failure"}
POOL_TRIALS = {
    "compression_only": 0,
    "schedule_only": 64,
    "compress_then_tune_screen": 0,
    "compress_then_tune_tuned": 64,
    "gear": 64,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    rows = [
        {
            "path": str(item.relative_to(path)),
            "sha256": file_sha256(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    if not rows:
        raise ValueError(f"evidence directory is empty: {path}")
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def binding(path: Path, expected_sha: str | None = None) -> dict[str, str]:
    resolved = Path(path).resolve()
    actual = path_sha256(resolved)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"SHA mismatch: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path = Path(path)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite drifted evidence: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _legacy_trial_budget(
    row: Mapping[str, Any],
    *,
    feedback_path: Path,
) -> tuple[int, dict[str, str] | None]:
    explicit = row.get("tvm_trials")
    if explicit is not None:
        return int(explicit), None
    legacy = row.get("tvm_max_trials")
    if legacy is None:
        return -1, None
    raw = json.loads(Path(feedback_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("legacy feedback row must be a JSON object")
    recorded = str(raw.get("actual_feedback_row_sha256") or "")
    unsigned = {
        key: value
        for key, value in raw.items()
        if key != "actual_feedback_row_sha256"
    }
    if (
        raw.get("row_id") != row.get("row_id")
        or int(raw.get("tvm_max_trials", -1)) != int(legacy)
        or recorded != _canonical_sha(unsigned)
    ):
        raise ValueError("legacy tvm_max_trials lacks a valid signed feedback row")
    return int(legacy), binding(feedback_path)


def _database_path(row: Mapping[str, Any], module_path: Path) -> Path:
    label_dir = module_path.parent
    candidates = (
        label_dir / "tuning_database",
        label_dir / "ms_work_dir",
    )
    for candidate in candidates:
        if (
            (candidate / "database_workload.json").is_file()
            and (candidate / "database_tuning_record.json").is_file()
        ):
            return candidate
    if int(row.get("tvm_trials", -1)) == 0:
        marker = label_dir / "zero_trial_database_contract.json"
        write_immutable(
            marker,
            {
                "schema_version": "fcooper_tvm_zero_trial_database_contract_v1",
                "row_id": row["row_id"],
                "tvm_trials": 0,
                "reason": "automatic default schedule without MetaSchedule trials",
            },
        )
        return marker
    raise FileNotFoundError(f"TVM database is missing beside {module_path}")


def _normalized_reports(
    row: Mapping[str, Any], execution_dir: Path
) -> tuple[Path, Path]:
    performance_source = binding(
        Path(str(row["performance_result_json"])),
        str(row["performance_result_sha256"]),
    )
    ap_source = binding(
        Path(str(row["ap_report_path"])),
        str(row["ap_report_sha256"]),
    )
    performance = execution_dir / "normalized_performance_report.json"
    ap = execution_dir / "normalized_ap_report.json"
    write_immutable(
        performance,
        {
            "schema_version": "fcooper_tvm_normalized_performance_report_v1",
            "row_id": row["row_id"],
            "latency_ms": row["latency_ms"],
            "energy_j": row["energy_j"],
            "source_report": performance_source,
        },
    )
    write_immutable(
        ap,
        {
            "schema_version": "fcooper_tvm_normalized_ap_report_v1",
            "row_id": row["row_id"],
            "ap30": row["ap30"],
            "ap50": row["ap50"],
            "ap70": row["ap70"],
            "source_report": ap_source,
        },
    )
    return performance, ap


def _resolved_source_artifacts(
    row: Mapping[str, Any], provenance_path: Path
) -> tuple[dict[str, str], dict[str, str]]:
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("passed") is not True
        or provenance.get("backend_neutral_only") is not True
    ):
        raise ValueError("source reuse audit is not a passed backend-neutral contract")
    resolved = provenance.get("resolved_source_contract")
    reused = provenance.get("reused_artifacts")
    if not isinstance(resolved, Mapping) or not isinstance(reused, Mapping):
        raise ValueError("source reuse audit lacks resolved source artifacts")

    verified: dict[str, dict[str, str]] = {}
    for name, path_field in (("checkpoint", "checkpoint_path"), ("onnx", "onnx_path")):
        record = reused.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"source reuse audit lacks {name} binding")
        resolved_path = Path(str(resolved.get(path_field) or ""))
        if str(record.get("path") or "") != str(resolved_path):
            raise ValueError(f"source reuse audit {name} path drift")
        verified[name] = binding(resolved_path, str(record.get("sha256") or ""))

    if verified["checkpoint"]["sha256"] != str(row.get("checkpoint_sha256") or ""):
        raise ValueError("feedback checkpoint SHA differs from resolved source")
    return verified["checkpoint"], verified["onnx"]


def normalize_success_row(
    row: Mapping[str, Any], *, execution_dir: Path
) -> dict[str, Any]:
    if row.get("terminal_status") != SUCCESS:
        raise ValueError("success normalizer received a non-success row")
    trials, legacy_feedback = _legacy_trial_budget(
        row,
        feedback_path=execution_dir / "feedback_row.json",
    )
    if trials < 0:
        raise ValueError("feedback lacks a TVM trial budget")
    row = {
        **dict(row),
        "tvm_trials": trials,
        **(
            {"tvm_trials_source_field": "tvm_max_trials"}
            if legacy_feedback is not None
            else {}
        ),
        **(
            {
                "backend": "tvm_auto",
                "backend_source_field": "dispatch_key",
            }
            if row.get("backend") is None and row.get("dispatch_key") == "tvm_auto"
            else {}
        ),
    }
    provenance = execution_dir / "source_reuse_audit.json"
    checkpoint, _source_onnx = _resolved_source_artifacts(row, provenance)
    graph = row.get("graph_features")
    if not isinstance(graph, Mapping):
        raise ValueError("feedback lacks actual graph features")
    onnx = binding(
        Path(str(graph.get("onnx_path") or "")),
        str(graph.get("onnx_sha256") or ""),
    )
    module = Path(str(row["tvm_artifact_path"]))
    database = _database_path(row, module)
    performance, ap = _normalized_reports(row, execution_dir)
    artifacts = {
        "checkpoint": checkpoint,
        "onnx": onnx,
        "tvm_module": binding(module, str(row["tvm_artifact_sha256"])),
        "tvm_database": binding(database),
        "performance_report": binding(performance),
        "ap_report": binding(ap),
        "source_provenance": binding(provenance),
    }
    quant_path = row.get("quant_contract_path")
    if str(row.get("q_mode")) == "int8":
        if not quant_path:
            raise ValueError("INT8 feedback has no quant contract")
        artifacts["quant_contract"] = binding(
            Path(str(quant_path)),
            str(row["quant_contract_sha256"]),
        )
    if legacy_feedback is not None:
        artifacts["feedback_row"] = legacy_feedback
    return {**dict(row), "artifacts": artifacts}


def normalize_terminal_row(
    row: Mapping[str, Any], *, execution_dir: Path
) -> dict[str, Any]:
    if row.get("terminal_status") == SUCCESS:
        return normalize_success_row(row, execution_dir=execution_dir)
    if row.get("terminal_status") not in FAILURES:
        raise ValueError("feedback row is not a credible terminal observation")
    if any(
        row.get(metric) is not None
        for metric in ("ap30", "ap50", "ap70", "latency_ms", "energy_j")
    ):
        raise ValueError("terminal failure contains fabricated numerical metrics")
    if not str(row.get("failure_reason") or "").strip():
        raise ValueError("terminal failure has no reason")
    feedback_path = execution_dir / "feedback_row.json"
    trials, legacy_feedback = _legacy_trial_budget(row, feedback_path=feedback_path)
    if trials < 0:
        raise ValueError("failure feedback lacks a TVM trial budget")
    backend_derived = row.get("backend") is None and row.get("dispatch_key") == "tvm_auto"
    return {
        **dict(row),
        "tvm_trials": trials,
        **(
            {"tvm_trials_source_field": "tvm_max_trials"}
            if legacy_feedback is not None
            else {}
        ),
        **(
            {
                "backend": "tvm_auto",
                "backend_source_field": "dispatch_key",
            }
            if backend_derived
            else {}
        ),
        "artifacts": {
            "failure_contract": binding(feedback_path),
        },
    }


def _request_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(row) for row in payload], {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"unsupported request payload: {path}")
    if isinstance(payload.get("rows"), list):
        metadata = {key: value for key, value in payload.items() if key != "rows"}
        return [dict(row) for row in payload["rows"]], metadata
    requests = payload.get("requests")
    if isinstance(requests, list):
        rows: list[dict[str, Any]] = []
        for request in requests:
            nested, _ = _request_rows(Path(str(request["path"])))
            rows.extend(nested)
        return rows, {}
    raise ValueError(f"request payload has no rows: {path}")


def feedback_index(
    artifact_root: Path,
) -> dict[tuple[str, int], tuple[dict[str, Any], Path]]:
    indexed: dict[tuple[str, int], tuple[dict[str, Any], Path]] = {}
    for path in sorted(Path(artifact_root).glob("execution/*/feedback_row.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row_id = str(row.get("row_id") or "")
        if not row_id:
            continue
        trials, _legacy_feedback = _legacy_trial_budget(row, feedback_path=path)
        key = (row_id, trials)
        if key in indexed:
            raise ValueError(
                f"duplicate feedback row and trial budget: {row_id}, {trials}"
            )
        indexed[key] = (dict(row), path.parent)
    return indexed


def normalize_automatic_selection(
    *,
    selection_path: Path,
    normalized_screen_path: Path,
    tuned_row_ids: Sequence[str],
) -> dict[str, Any]:
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    normalized_screen = json.loads(
        Path(normalized_screen_path).read_text(encoding="utf-8")
    )
    if (
        not isinstance(selection, Mapping)
        or selection.get("automatic") is not True
        or list(selection.get("selected_row_ids") or []) != list(tuned_row_ids)
    ):
        raise ValueError("tuned selection is not the frozen automatic selection")
    source_pool_path = Path(str(selection.get("source_pool_path") or ""))
    source_pool = binding(
        source_pool_path, str(selection.get("source_pool_sha256") or "")
    )
    if (
        not isinstance(normalized_screen, Mapping)
        or normalized_screen.get("pool_name") != "compress_then_tune_screen"
    ):
        raise ValueError("normalized screen pool contract is invalid")
    screen_ids = {
        str(row.get("row_id") or "")
        for row in normalized_screen.get("rows", [])
        if isinstance(row, Mapping)
    }
    if not set(tuned_row_ids).issubset(screen_ids):
        raise ValueError("automatic tuned rows are not members of screen12")
    normalized_binding = binding(normalized_screen_path)
    return {
        "automatic": True,
        "selected_row_ids": list(tuned_row_ids),
        "source_pool_sha256": normalized_binding["sha256"],
        "source_normalized_screen_pool": normalized_binding,
        "source_automatic_selection": binding(selection_path),
        "source_pre_normalization_screen_pool": source_pool,
    }


def build_pool(
    *,
    name: str,
    expected_path: Path,
    artifact_root: Path,
    automatic_selection_path: Path | None = None,
    normalized_screen_path: Path | None = None,
) -> dict[str, Any]:
    expected, metadata = _request_rows(expected_path)
    indexed = feedback_index(artifact_root)
    if name not in POOL_TRIALS:
        raise ValueError(f"unsupported evidence pool: {name}")
    expected_trials = POOL_TRIALS[name]
    rows = []
    for requested in expected:
        row_id = str(requested.get("row_id") or requested.get("manifest_job_id") or "")
        key = (row_id, expected_trials)
        if key not in indexed:
            raise ValueError(
                f"{name} is missing {expected_trials}-trial feedback for {row_id}"
            )
        row, execution_dir = indexed[key]
        rows.append(normalize_terminal_row(row, execution_dir=execution_dir))
    pool = {
        "schema_version": "fcooper_tvm_stage6_evidence_pool_v1",
        "pool_name": name,
        "row_count": len(rows),
        "rows": rows,
        **{
            key: value
            for key, value in metadata.items()
            if key in {"automatic_selection", "outer_budget", "batch_size", "rounds", "atomic_feedback"}
        },
    }
    if name == "gear":
        pool.update(
            {
                "outer_budget": 16,
                "batch_size": 4,
                "rounds": 4,
                "atomic_feedback": True,
            }
        )
    if name == "compress_then_tune_tuned":
        if automatic_selection_path is None or normalized_screen_path is None:
            raise ValueError(
                "tuned pool requires the automatic selection and normalized screen"
            )
        pool["automatic_selection"] = normalize_automatic_selection(
            selection_path=automatic_selection_path,
            normalized_screen_path=normalized_screen_path,
            tuned_row_ids=[str(row["row_id"]) for row in rows],
        )
    return pool


def parse_pool(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("pool must use NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pool", type=parse_pool, action="append", required=True)
    parser.add_argument("--automatic-selection", type=Path)
    parser.add_argument("--normalized-screen-pool", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = {}
    for name, expected_path in args.pool:
        pool = build_pool(
            name=name,
            expected_path=expected_path,
            artifact_root=args.artifact_root,
            automatic_selection_path=args.automatic_selection,
            normalized_screen_path=args.normalized_screen_pool,
        )
        output = args.output_dir / f"{name}.json"
        write_immutable(output, pool)
        outputs[name] = binding(output)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
