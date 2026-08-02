#!/usr/bin/env python3
"""Validate and aggregate the Lane C Stage6 Orin five-arm measurements."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SAMPLE_COUNT = 1500
EXPECTED_AP_SAMPLE_COUNT = 1789
EXPECTED_IDS = (
    "pyramid:original_default",
    "pyramid:compression_only",
    "pyramid:schedule_only",
    "pyramid:compress_then_tune",
    "pyramid:joint_shcosearch",
    "codriving:original_default",
    "codriving:compression_only",
    "codriving:schedule_only",
    "codriving:compress_then_tune",
    "codriving:joint_shcosearch",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_path_bound_sha256(path: Path) -> str:
    """Reproduce the pre-fix CoDriving filename-NUL-bytes digest."""
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_manifest_rows(manifest: dict[str, Any]) -> None:
    identifiers = tuple(str(row.get("id")) for row in manifest.get("rows", []))
    if identifiers != EXPECTED_IDS:
        raise ValueError(
            "manifest must contain the exact ordered 2x5 executable IDs"
        )
    for row in manifest["rows"]:
        model, arm = str(row["id"]).split(":", 1)
        if row.get("model") != model or row.get("arm") != arm:
            raise ValueError(f"manifest row identity mismatch: {row.get('id')}")


def apply_manifest_overrides(
    manifest: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    rows_by_id = {str(row["id"]): row for row in result.get("rows", [])}
    seen: set[str] = set()
    for override in overrides.get("rows", []):
        identifier = str(override.get("id"))
        if identifier in seen:
            raise ValueError(f"duplicate manifest override: {identifier}")
        seen.add(identifier)
        if identifier not in rows_by_id:
            raise ValueError(f"unknown manifest override: {identifier}")
        target = rows_by_id[identifier]
        for key in ("id", "model", "arm"):
            if key in override and override[key] != target.get(key):
                raise ValueError(
                    f"manifest override cannot change row identity: "
                    f"{identifier}"
                )
        target.update(copy.deepcopy(override))
    return result


def artifact_dir_for_spec(
    spec: dict[str, Any],
    result_root: Path,
) -> Path:
    artifact_arm = str(spec.get("artifact_arm", spec["arm"]))
    return result_root / str(spec["model"]) / artifact_arm


def latency_path_for_arm(arm_dir: Path) -> Path:
    privileged = arm_dir / "primary_latency_power_sudo.json"
    if privileged.is_file():
        return privileged
    return arm_dir / "primary_latency_power.json"


def _expected_latency_scope(
    model: str,
    widths: list[int],
    runtime: str,
) -> str:
    if runtime == "native_pytorch":
        if model in {"pyramid", "codriving"}:
            return (
                f"{model}_multiscale_backbone_compute_no_data_transfer"
            )
        raise ValueError(f"unsupported model: {model}")
    if runtime != "tensorrt":
        raise ValueError(f"unsupported runtime: {runtime}")
    signature = "x".join(str(item) for item in widths)
    if model == "pyramid":
        return (
            f"pyramid_get_multiscale_feature_{signature}"
            "_engine_compute_no_data_transfer"
        )
    if model == "codriving":
        return (
            f"codriving_backbone_resnet_{signature}"
            "_engine_compute_no_data_transfer"
        )
    raise ValueError(f"unsupported model: {model}")


def _validate_latency(
    report: dict[str, Any],
    identifier: str,
    expected_scope: str,
) -> None:
    if int(report.get("sample_count", 0)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"{identifier} latency is not 5x300 samples")
    protocol = report.get("protocol", {})
    actual = {
        "warmup": protocol.get("warmup"),
        "iters": protocol.get("iters"),
        "repeat": protocol.get("repeat"),
        "timing": protocol.get("timing"),
        "data_transfer_inside_timed_region": protocol.get(
            "data_transfer_inside_timed_region"
        ),
    }
    expected = {
        "warmup": 20,
        "iters": 300,
        "repeat": 5,
        "timing": "CUDA_event",
        "data_transfer_inside_timed_region": False,
    }
    if actual != expected:
        raise ValueError(f"{identifier} latency protocol mismatch: {actual}")
    if report.get("scope") != expected_scope:
        raise ValueError(
            f"{identifier} latency scope mismatch: {report.get('scope')}"
        )


def _validate_ap(
    report: dict[str, Any],
    *,
    identifier: str,
    runtime: str,
) -> None:
    processed = report.get("processed_samples", report.get("num_samples", 0))
    if (
        report.get("status") != "success"
        or int(processed) != EXPECTED_AP_SAMPLE_COUNT
        or int(report.get("failed_samples", 0)) != 0
        or int(report.get("fallback_samples", 0)) != 0
    ):
        raise ValueError(f"{identifier} AP is not a successful full_1789 run")
    has_trt_claim = (
        report.get("engine_ap_claim") is True
        or report.get("gates", {}).get("full_1789") is True
    )
    if runtime == "tensorrt" and not has_trt_claim:
        raise ValueError(f"{identifier} AP lacks a valid TensorRT engine claim")
    if identifier.startswith("codriving:") and runtime == "tensorrt":
        summary = report.get("output_vs_reference_error", {})
        expected_records = int(processed) * 6
        if (
            summary.get("all_finite") is not True
            or int(summary.get("shape_mismatch_count", -1)) != 0
            or int(summary.get("nonfinite_count", -1)) != 0
            or int(summary.get("num_records", -1)) != expected_records
            or int(summary.get("num_compared", -1)) != expected_records
        ):
            raise ValueError(
                f"{identifier} AP output comparison integrity failed"
            )


def _load_ap_terminal(
    *,
    arm_dir: Path,
    identifier: str,
    runtime: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    full_path = arm_dir / "ap_full" / "report.json"
    if full_path.is_file():
        report = load_json(full_path)
        _validate_ap(report, identifier=identifier, runtime=runtime)
        return (
            report,
            full_path,
            {
                "ap50": float(report["ap50"]),
                "ap70": float(report["ap70"]),
                "ap_status": "full_1789_success",
                "sanity_ap50": None,
                "sanity_ap70": None,
            },
        )

    sanity_path = arm_dir / "ap_sanity" / "report.json"
    report = load_json(sanity_path)
    output_summary = report.get("output_error_summary", {})
    is_zero_prediction_failure = (
        report.get("status") == "success"
        and int(report.get("processed_samples", 0)) == 16
        and int(report.get("failed_samples", 0)) == 0
        and int(report.get("fallback_samples", 0)) == 0
        and int(report.get("pred_nonempty_count", -1)) == 0
        and float(report.get("ap50", 1.0)) == 0.0
        and float(report.get("ap70", 1.0)) == 0.0
        and output_summary.get("all_finite") is True
    )
    if identifier != "pyramid:joint_shcosearch":
        raise ValueError(
            "sanity-zero AP terminal is only permitted for "
            "pyramid:joint_shcosearch"
        )
    if not is_zero_prediction_failure:
        raise ValueError(f"{identifier} AP is not a successful full_1789 run")
    return (
        report,
        sanity_path,
        {
            "ap50": None,
            "ap70": None,
            "ap_status": "sanity_failed_zero_predictions",
            "sanity_ap50": 0.0,
            "sanity_ap70": 0.0,
        },
    )


def _energy_fields(report: dict[str, Any]) -> tuple[float | None, str]:
    power = report.get("power_measurement", {})
    status = str(power.get("measurement_status", "unavailable"))
    if status == "available":
        value = power.get("energy_j", report.get("energy_j"))
        if value is None:
            raise ValueError("available power measurement omitted energy_j")
        return float(value), "available"
    blocker = power.get("blocker")
    if not blocker:
        raise ValueError("unavailable power measurement omitted blocker")
    return None, str(blocker)


def collect_result_row(
    spec: dict[str, Any],
    result_root: Path,
) -> dict[str, Any]:
    identifier = str(spec["id"])
    model = str(spec["model"])
    arm = str(spec["arm"])
    runtime = str(spec["runtime"])
    arm_dir = artifact_dir_for_spec(spec, result_root)
    latency_path = latency_path_for_arm(arm_dir)
    latency = load_json(latency_path)
    _validate_latency(
        latency,
        identifier,
        _expected_latency_scope(
            model,
            list(spec["widths"]),
            runtime,
        ),
    )
    ap_report, ap_path, ap_fields = _load_ap_terminal(
        arm_dir=arm_dir,
        identifier=identifier,
        runtime=runtime,
    )
    energy_j, energy_status = _energy_fields(latency)

    build_path = arm_dir / "build_report.json"
    build = load_json(build_path) if runtime == "tensorrt" else None
    engine_sha = None if build is None else str(build["engine_sha256"])
    ap_report_engine_sha = (
        ap_report.get("engine_sha256")
        or ap_report.get("sha256", {}).get("engine")
    )
    ap_engine_hash_contract = None
    if engine_sha is not None:
        engine_path = arm_dir / "backbone.engine"
        actual_engine_sha = sha256_file(engine_path)
        if engine_sha != actual_engine_sha:
            raise ValueError(f"{identifier} build engine SHA mismatch")
        if latency.get("engine_sha256") != actual_engine_sha:
            raise ValueError(f"{identifier} latency engine SHA mismatch")
        if ap_report_engine_sha is not None:
            if str(ap_report_engine_sha) == actual_engine_sha:
                ap_engine_hash_contract = "raw_file_sha256"
            elif str(ap_report_engine_sha) == legacy_path_bound_sha256(
                engine_path
            ):
                ap_engine_hash_contract = (
                    "legacy_relative_filename_nul_plus_bytes"
                )
            else:
                raise ValueError(
                    f"{identifier} AP report engine hash is not auditable"
                )
    output_error_summary = (
        ap_report.get("output_vs_reference_error")
        or ap_report.get("output_error_summary")
    )
    return {
        "id": identifier,
        "model": model,
        "arm": arm,
        "artifact_arm": str(spec.get("artifact_arm", arm)),
        "widths": list(spec["widths"]),
        "precision": str(spec["precision"]),
        "h800_precision": str(
            spec.get("h800_source_precision", spec["precision"])
        ),
        "comparison_class": str(
            spec.get("comparison_class", "same_declared_precision")
        ),
        "runtime": runtime,
        "h800": dict(spec["h800_source"]),
        "orin": {
            **ap_fields,
            "latency_median_ms": float(latency["median_ms"]),
            "latency_p90_ms": float(latency["p90_ms"]),
            "latency_p99_ms": float(latency["p99_ms"]),
            "latency_mean_ms": float(latency["mean_ms"]),
            "energy_j": energy_j,
            "energy_status": energy_status,
            "output_error_summary": output_error_summary,
        },
        "evidence_sha256": {
            "checkpoint": str(spec["source_sha256"]["checkpoint"]),
            "engine": engine_sha,
            "latency_report": sha256_file(latency_path),
            "ap_report": sha256_file(ap_path),
            "build_report": (
                sha256_file(build_path) if build is not None else None
            ),
            "ap_report_engine": ap_report_engine_sha,
        },
        "evidence_contract": {
            "ap_engine_hash": ap_engine_hash_contract,
        },
    }


def _validate_source_files(
    manifest: dict[str, Any],
    manifest_path: Path | None = None,
) -> None:
    source_root = Path(str(manifest["source_root"]))
    if manifest_path is not None and not source_root.is_absolute():
        source_root = manifest_path.parent / source_root
    for row in manifest["rows"]:
        for key, relative in row["source_files"].items():
            actual = sha256_file(source_root / relative)
            if actual != row["source_sha256"][key]:
                raise ValueError(f"{row['id']} source {key} SHA mismatch")


def collect_rows(
    manifest: dict[str, Any],
    result_root: Path,
    *,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    validate_manifest_rows(manifest)
    _validate_source_files(manifest, manifest_path)
    return [
        collect_result_row(spec, result_root)
        for spec in manifest["rows"]
    ]


def _csv_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "arm": row["arm"],
            "widths": "x".join(str(item) for item in row["widths"]),
            "precision": row["precision"],
            "h800_precision": row.get(
                "h800_precision", row["precision"]
            ),
            "comparison_class": row.get(
                "comparison_class", "same_declared_precision"
            ),
            "runtime": row["runtime"],
            "h800_ap70": row["h800"]["ap70"],
            "h800_latency_ms": row["h800"]["latency_ms"],
            "h800_energy_j": row["h800"]["energy_j"],
            "orin_ap50": row["orin"]["ap50"],
            "orin_ap70": row["orin"]["ap70"],
            "orin_ap_status": row["orin"].get(
                "ap_status", "full_1789_success"
            ),
            "orin_sanity_ap50": row["orin"].get("sanity_ap50"),
            "orin_sanity_ap70": row["orin"].get("sanity_ap70"),
            "orin_latency_median_ms": row["orin"]["latency_median_ms"],
            "orin_latency_p90_ms": row["orin"]["latency_p90_ms"],
            "orin_latency_p99_ms": row["orin"]["latency_p99_ms"],
            "orin_latency_mean_ms": row["orin"]["latency_mean_ms"],
            "orin_energy_j": row["orin"]["energy_j"],
            "orin_energy_status": row["orin"]["energy_status"],
        }
        for row in rows
    ]


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    manifest_path: Path,
    override_manifest_path: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "final_summary.json"
    csv_path = output_dir / "orin_stage6_five_config.csv"
    markdown_path = output_dir / "summary.md"
    sha_manifest_path = output_dir / "artifact_sha256.json"

    summary = {
        "schema_version": "lane_c_stage6_orin_five_config_summary_v1",
        "status": "complete" if len(rows) == 10 else "partial",
        "row_count": len(rows),
        "contracts": {
            "configuration_source": "31 document sections 11.3 and 14.3",
            "latency": (
                "batch2; warmup20; iters300; repeat5; CUDA event; "
                "multiscale-backbone compute/no data transfer"
            ),
            "ap": "DAIR-V2X full_1789; only multiscale backbone/resnet replaced",
            "energy": (
                "VIN_SYS_5V0 mean active-window power multiplied by "
                "median CUDA-event latency; joules per batch2 invocation"
            ),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "rows": rows,
    }
    if override_manifest_path is not None:
        summary["manifest_override"] = {
            "path": str(override_manifest_path),
            "sha256": sha256_file(override_manifest_path),
        }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    records = _csv_records(rows)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Lane C Orin Stage6 五配置结果",
        "",
        "| 模型 | 配置臂 | 宽度/精度 | Orin AP50 | Orin AP70 | "
        "Orin latency median / p90 / p99 / mean (ms) | Orin energy |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        orin = row["orin"]
        ap_status = orin.get("ap_status", "full_1789_success")
        if orin["ap50"] is None or orin["ap70"] is None:
            ap50_text = "NA"
            ap70_text = "NA"
        else:
            ap50_text = f"{orin['ap50']:.6f}"
            ap70_text = f"{orin['ap70']:.6f}"
        energy = (
            f"{orin['energy_j']:.6f} J"
            if orin["energy_j"] is not None
            else orin["energy_status"]
        )
        lines.append(
            "| {model} | {arm} | {widths}, {precision} | {ap50} | "
            "{ap70} | {median:.6f} / {p90:.6f} / {p99:.6f} / "
            "{mean:.6f} | {energy}; AP={ap_status} |".format(
                model=row["model"],
                arm=row["arm"],
                widths="×".join(str(item) for item in row["widths"]),
                precision=row["precision"],
                ap50=ap50_text,
                ap70=ap70_text,
                median=orin["latency_median_ms"],
                p90=orin["latency_p90_ms"],
                p99=orin["latency_p99_ms"],
                mean=orin["latency_mean_ms"],
                energy=energy,
                ap_status=ap_status,
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_files = (summary_path, csv_path, markdown_path)
    sha_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "lane_c_stage6_summary_sha256_v1",
                "files": {
                    path.name: {
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in output_files
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "sha_manifest": sha_manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--override-manifest", type=Path)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    if args.override_manifest is not None:
        override_manifest = load_json(args.override_manifest)
        expected_base_sha = override_manifest.get("base_manifest_sha256")
        actual_base_sha = sha256_file(args.manifest)
        if expected_base_sha != actual_base_sha:
            raise ValueError("override base manifest SHA mismatch")
        manifest = apply_manifest_overrides(manifest, override_manifest)
    rows = collect_rows(
        manifest,
        args.result_root,
        manifest_path=args.manifest,
    )
    outputs = write_outputs(
        rows=rows,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        override_manifest_path=args.override_manifest,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
