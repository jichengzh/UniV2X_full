#!/usr/bin/env python3
"""Create an immutable repaired schedule-only feedback package.

The repair is limited to replacing a stale AP binding with an independently
measured H800 full-AP report for the exact same checkpoint and TVM module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from scripts.stage6_finalize_fcooper_tvm_v1 import (
    AP_REPEAT_ABS_TOL,
    validate_ap_repeat,
    validate_full_ap_report_identity,
)


BASE_WIDTH = [64, 128, 256, 128, 256]
SCHEDULE_TASK_ID = "S6-FCO-TVM-SCHEDULE-ONLY-MEASURE-V1"


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return dict(payload)


def _verify_binding(binding: Mapping[str, Any], label: str) -> dict[str, str]:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding is missing")
    path = Path(str(binding.get("path") or "")).resolve()
    expected = str(binding.get("sha256") or "")
    actual = _file_sha(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch")
    return {"path": str(path), "sha256": actual}


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
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


def _copy_immutable(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        if _file_sha(destination) != _file_sha(source):
            raise FileExistsError(
                f"refusing to overwrite drifted evidence: {destination}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_contract(
    source: Mapping[str, Any],
    source_path: Path,
) -> tuple[dict[str, str], dict[str, str], Path]:
    if (
        source.get("task_id") != SCHEDULE_TASK_ID
        or source.get("terminal_status") != "measured_success_gold"
        or source.get("q_mode") != "fp32"
        or source.get("width") != BASE_WIDTH
        or int(source.get("tvm_trials", -1)) != 64
    ):
        raise ValueError("source feedback is not the fixed schedule-only row")
    recorded_self_sha = str(source.get("actual_feedback_row_sha256") or "")
    without_self = {
        key: value
        for key, value in source.items()
        if key != "actual_feedback_row_sha256"
    }
    if recorded_self_sha != _canonical_sha(without_self):
        raise ValueError("source feedback self SHA mismatch")

    module_path = Path(str(source.get("tvm_artifact_path") or "")).resolve()
    module = {
        "path": str(module_path),
        "sha256": _file_sha(module_path),
    }
    if module["sha256"] != source.get("tvm_artifact_sha256"):
        raise ValueError("source feedback TVM module SHA mismatch")

    provenance_path = source_path.parent / "source_reuse_audit.json"
    provenance = _read_object(provenance_path, "source reuse audit")
    provenance_sha = _file_sha(provenance_path)
    if provenance_sha != source.get("resolved_source_evidence_sha256"):
        raise ValueError("source provenance SHA mismatch")
    resolved = provenance.get("resolved_source_contract")
    if not isinstance(resolved, Mapping):
        raise ValueError("source reuse audit lacks resolved source contract")
    checkpoint_path = Path(str(resolved.get("checkpoint_path") or "")).resolve()
    checkpoint = {
        "path": str(checkpoint_path),
        "sha256": _file_sha(checkpoint_path),
    }
    if checkpoint["sha256"] != source.get("checkpoint_sha256"):
        raise ValueError("source feedback checkpoint SHA mismatch")
    return module, checkpoint, provenance_path


def repair_schedule_feedback(
    *,
    source_feedback: Path,
    winner_validation: Path,
    output_root: Path,
) -> dict[str, Any]:
    source_feedback = Path(source_feedback).resolve()
    source = _read_object(source_feedback, "source feedback")
    module, checkpoint, provenance_path = _source_contract(source, source_feedback)

    validation_path = Path(winner_validation).resolve()
    validation = _read_object(validation_path, "winner validation")
    validation_gpu = int(validation.get("gpu_index", -1))
    if (
        validation.get("schema_version")
        != "fcooper_tvm_gpu7_winner_validation_v1"
        or validation.get("row_id") != source.get("row_id")
        or not 0 <= validation_gpu <= 7
    ):
        raise ValueError("winner validation identity mismatch")
    full_ap = validation.get("full_ap")
    if (
        not isinstance(full_ap, Mapping)
        or int(full_ap.get("gpu_index", -1)) != validation_gpu
    ):
        raise ValueError("winner validation lacks same-GPU H800 full AP")
    report_binding = _verify_binding(full_ap.get("report"), "GPU7 full AP report")
    prediction_binding = _verify_binding(
        full_ap.get("prediction"), "GPU7 prediction"
    )
    if (
        full_ap.get("prediction_sha256") != prediction_binding["sha256"]
        or full_ap.get("checkpoint_sha256") != checkpoint["sha256"]
        or full_ap.get("tvm_module_sha256") != module["sha256"]
    ):
        raise ValueError("winner validation artifact SHA mismatch")
    report = _read_object(Path(report_binding["path"]), "GPU7 full AP report")
    selected = {
        **source,
        "verified_artifacts": {
            "checkpoint": checkpoint,
            "tvm_module": module,
        },
    }
    validate_full_ap_report_identity(
        report,
        selected=selected,
        prediction_sha256=prediction_binding["sha256"],
        physical_gpu_id=validation_gpu,
    )
    for metric in ("ap30", "ap50", "ap70"):
        if float(full_ap.get(metric)) != float(report.get(metric)):
            raise ValueError(f"winner validation {metric} differs from full AP report")
    try:
        drift = validate_ap_repeat(
            source,
            report,
            absolute_tolerance=AP_REPEAT_ABS_TOL,
        )
    except ValueError as exc:
        raise ValueError(f"GPU7 AP drift invalidates repair: {exc}") from exc

    execution_dir = Path(output_root).resolve() / "execution" / "schedule_only_repaired"
    copied_provenance = execution_dir / "source_reuse_audit.json"
    _copy_immutable(provenance_path, copied_provenance)
    repair_contract_path = execution_dir / "ap_binding_repair_contract.json"
    repair_contract = {
        "schema_version": "fcooper_tvm_schedule_ap_binding_repair_v1",
        "passed": True,
        "reason": (
            "original full-AP report predates the final formal TVM module; "
            "rebind to independently measured GPU7 full AP for the exact module "
            "and checkpoint"
        ),
        "row_id": source["row_id"],
        "source_feedback": {
            "path": str(source_feedback),
            "sha256": _file_sha(source_feedback),
        },
        "winner_validation": {
            "path": str(validation_path),
            "sha256": _file_sha(validation_path),
        },
        "full_ap_report": report_binding,
        "physical_gpu_id": validation_gpu,
        "prediction": prediction_binding,
        "tvm_module": module,
        "checkpoint": checkpoint,
        "ap_drift": drift,
        "absolute_tolerance": AP_REPEAT_ABS_TOL,
    }
    _write_immutable(repair_contract_path, repair_contract)

    repaired = {
        **source,
        **{metric: float(report[metric]) for metric in ("ap30", "ap50", "ap70")},
        "ap_report_path": report_binding["path"],
        "ap_report_sha256": report_binding["sha256"],
        "ap_prediction_path": prediction_binding["path"],
        "ap_prediction_sha256": prediction_binding["sha256"],
        "source_actual_feedback_row_sha256": source[
            "actual_feedback_row_sha256"
        ],
        "evidence_repair_contract_path": str(repair_contract_path),
        "evidence_repair_contract_sha256": _file_sha(repair_contract_path),
    }
    repaired_without_self = {
        key: value
        for key, value in repaired.items()
        if key != "actual_feedback_row_sha256"
    }
    repaired["actual_feedback_row_sha256"] = _canonical_sha(repaired_without_self)
    feedback_path = execution_dir / "feedback_row.json"
    _write_immutable(feedback_path, repaired)
    return {
        "schema_version": "fcooper_tvm_schedule_ap_binding_repair_result_v1",
        "passed": True,
        "feedback_row": {
            "path": str(feedback_path),
            "sha256": _file_sha(feedback_path),
        },
        "source_reuse_audit": {
            "path": str(copied_provenance),
            "sha256": _file_sha(copied_provenance),
        },
        "repair_contract": {
            "path": str(repair_contract_path),
            "sha256": _file_sha(repair_contract_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-feedback", type=Path, required=True)
    parser.add_argument("--winner-validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = repair_schedule_feedback(
        source_feedback=args.source_feedback,
        winner_validation=args.winner_validation,
        output_root=args.output_root,
    )
    if args.result_json is not None:
        _write_immutable(args.result_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
