#!/usr/bin/env python3
"""Build the paired, content-addressed Gold Cold-start 96 v3 plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage2.canonical_search_v3 import build_active_manifest


SCHEMA_VERSION = "stage3_gold_coldstart96_plan_v3"
MODELS = ("pyramid", "codriving")
MODEL_WIDTH_STRATA = {
    "pyramid": {
        "diagonal_base": (
            "16x16x16",
            "16x16x64",
            "32x32x64",
            "32x32x128",
        ),
        "alignment_group": (
            "16x64x128",
            "16x128x256",
            "32x64x256",
            "48x64x128",
        ),
        "off_diagonal": (
            "24x48x192",
            "24x56x128",
            "32x32x256",
            "40x80x160",
        ),
    },
    "codriving": {
        "diagonal_base": (
            "32x32x128",
            "48x96x192",
            "64x96x192",
            "64x128x256",
        ),
        "alignment_group": (
            "16x32x64",
            "32x64x128",
            "40x64x128",
            "48x64x128",
        ),
        "off_diagonal": (
            "24x32x96",
            "24x64x128",
            "56x112x224",
            "64x64x128",
        ),
    },
}
MODEL_HOLDOUT_WIDTHS = {
    "pyramid": ("16x16x64", "40x80x160"),
    "codriving": ("24x32x96", "64x128x256"),
}
MODEL_PILOT_WIDTHS = {
    "pyramid": "32x32x128",
    "codriving": "32x64x128",
}
FORBIDDEN_STRATEGY_TOKENS = ("tvm", "trt", "backend", "compiler")


def _sha256_payload(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _model_widths() -> dict[str, list[str]]:
    widths_by_model: dict[str, list[str]] = {}
    for model in MODELS:
        model_strata = MODEL_WIDTH_STRATA[model]
        widths = [width for strata_widths in model_strata.values() for width in strata_widths]
        if len(widths) != 12 or len(set(widths)) != 12:
            raise ValueError(f"{model} width strata must contain exactly 12 unique widths")
        widths_by_model[model] = widths
    return widths_by_model


def _group_to_stratum() -> dict[str, str]:
    mapping = {
        f"{model}|{width}": stratum
        for model, strata in MODEL_WIDTH_STRATA.items()
        for stratum, widths in strata.items()
        for width in widths
    }
    if len(mapping) != 24:
        raise ValueError("model width strata must contain exactly 24 unique group ids")
    return mapping


def _holdout_group_ids() -> list[str]:
    group_ids = [
        f"{model}|{width}"
        for model in MODELS
        for width in MODEL_HOLDOUT_WIDTHS[model]
    ]
    if len(group_ids) != 4 or len(set(group_ids)) != 4:
        raise ValueError("holdout configuration must define four unique group ids")
    return sorted(group_ids)


def _pilot_group_ids() -> list[str]:
    group_ids = [f"{model}|{MODEL_PILOT_WIDTHS[model]}" for model in MODELS]
    if len(group_ids) != 2 or len(set(group_ids)) != 2:
        raise ValueError("pilot configuration must define two unique group ids")
    return group_ids


def _group_jobs(jobs: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in jobs:
        row = dict(source)
        grouped.setdefault(str(row["group_id"]), []).append(row)
    return grouped


def _validate_group_combinations(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]], profile_ids: set[str]
) -> None:
    expected = {(q_mode, profile_id) for q_mode in ("fp16", "int8") for profile_id in profile_ids}
    for group_id, rows in grouped.items():
        actual = {(str(row["q_mode"]), str(row["capability_profile_id"])) for row in rows}
        if len(rows) != 4 or actual != expected:
            raise ValueError(f"group is not the unique 2x2 precision/profile product: {group_id}")


def build_gold_plan(capability_profiles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create 24 complete groups with a structurally frozen 80/16 split."""

    if len(capability_profiles) != 2:
        raise ValueError("Gold Cold-start 96 requires exactly two capability profiles")
    group_to_stratum = _group_to_stratum()
    widths_by_model = _model_widths()
    manifest = build_active_manifest(
        widths_by_model=widths_by_model,
        capability_profiles=capability_profiles,
    )
    grouped = _group_jobs(manifest["jobs"])
    if len(grouped) != 24:
        raise ValueError("manifest must contain 24 complete four-row groups")
    profile_ids = {str(profile["capability_profile_id"]) for profile in capability_profiles}
    _validate_group_combinations(grouped, profile_ids)

    holdout_group_ids = sorted(group_id for group_id in grouped if group_id in set(_holdout_group_ids()))
    train_group_ids = sorted(set(grouped) - set(holdout_group_ids))
    if len(train_group_ids) != 20 or len(holdout_group_ids) != 4:
        raise ValueError("split must contain 20 train groups and 4 holdout groups")
    split_by_group = {
        **{group_id: "train" for group_id in train_group_ids},
        **{group_id: "locked_holdout" for group_id in holdout_group_ids},
    }

    jobs: list[dict[str, Any]] = []
    for source in manifest["jobs"]:
        row = dict(source)
        width_key = str(row["width_key"])
        strategy = str(row["strategy_id"]).lower()
        if any(token in strategy for token in FORBIDDEN_STRATEGY_TOKENS):
            raise ValueError("backend/compiler token leaked into strategy_id")
        jobs.append(
            {
                **row,
                "width_stratum": group_to_stratum[str(row["group_id"])],
                "split": split_by_group[str(row["group_id"])],
                "required_metrics": ["latency", "energy", "ap"],
                "genome": [*row["width"], row["q_mode"]],
                "terminal_status": "pending",
            }
        )
    manifest = {
        **manifest,
        "schema_version": "stage3_gold_coldstart96_manifest_v3",
        "stage3_plan_schema_version": SCHEMA_VERSION,
        "pilot_group_ids": _pilot_group_ids(),
        "jobs": jobs,
    }
    if not set(manifest["pilot_group_ids"]) <= set(grouped):
        raise ValueError("pilot groups must exist in grouped manifest")
    groups_by_id = _group_jobs(jobs)
    groups = [
        {
            "group_id": group_id,
            "model": rows[0]["model"],
            "width_key": rows[0]["width_key"],
            "width_stratum": rows[0]["width_stratum"],
            "split": rows[0]["split"],
            "row_count": len(rows),
            "job_ids": sorted(str(row["job_id"]) for row in rows),
        }
        for group_id, rows in sorted(groups_by_id.items())
    ]
    split = {
        "schema_version": "stage3_gold_coldstart96_split_v3",
        "selection_basis": "premeasurement_structural_strata_only",
        "train_group_ids": train_group_ids,
        "holdout_group_ids": holdout_group_ids,
        "train_job_ids": sorted(
            str(row["job_id"]) for row in jobs if row["split"] == "train"
        ),
        "holdout_job_ids": sorted(
            str(row["job_id"]) for row in jobs if row["split"] == "locked_holdout"
        ),
    }
    contract = {
        "schema_version": "stage3_gold_coldstart96_contract_v3",
        "row_count": 96,
        "group_count": 24,
        "group_size": 4,
        "train_rows": 80,
        "locked_holdout_rows": 16,
        "pilot_group_ids": list(manifest["pilot_group_ids"]),
        "pilot_row_count": 8,
        "historical_180_backend_gold_allowed": False,
        "mixed_policy_activation": "disabled",
        "ap_sharing_rule": (
            "share only within identical model,width,q_mode after TVM/TRT numerical equivalence passes"
        ),
        "ap_sharing_required_equivalence_fields": [
            "checkpoint_sha256",
            "source_onnx_sha256",
            "compiled_artifact_sha256_by_profile",
            "calibration_sha256",
            "dataset_manifest_sha256",
            "evaluation_protocol_sha256",
            "input_manifest_sha256",
            "tolerance_spec_sha256",
            "output_comparison_sha256",
        ],
        "failure_rule": "build or numerical failures terminate as feasibility evidence without AP regression",
        "completion_rule": "96/96 rows have measured-success or audited-failure terminal status",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest,
        "groups": groups,
        "groups_by_id": groups_by_id,
        "split": split,
        "contract": contract,
    }


def _write_addressed(payload: Mapping[str, Any], output_dir: Path, stem: str) -> Path:
    detached = dict(payload)
    digest = _sha256_payload(detached)
    path = output_dir / f"{stem}-{digest}.json"
    path.write_text(json.dumps(detached, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_gold_plan(plan: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": _write_addressed(plan["manifest"], destination, "gold_coldstart96_manifest_v3"),
        "split": _write_addressed(plan["split"], destination, "gold_coldstart96_split_v3"),
        "contract": _write_addressed(plan["contract"], destination, "gold_coldstart96_contract_v3"),
    }


def audit_written_plan(paths: Mapping[str, Path]) -> bool:
    payloads: dict[str, Any] = {}
    for name in ("manifest", "split", "contract"):
        path = Path(paths[name])
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not path.stem.endswith(_sha256_payload(payload)):
            raise ValueError(f"content digest mismatch for {name}")
        payloads[name] = payload
    jobs = payloads["manifest"]["jobs"]
    pilot_group_ids = payloads["manifest"].get("pilot_group_ids")
    grouped = _group_jobs(jobs)
    split = payloads["split"]
    if len(jobs) != 96 or len(grouped) != 24:
        raise ValueError("written manifest is not 24 complete groups / 96 rows")
    if pilot_group_ids != _pilot_group_ids():
        raise ValueError("written manifest pilot_group_ids mismatch")
    profile_ids = {
        str(profile["capability_profile_id"])
        for profile in payloads["manifest"]["capability_profiles"]
    }
    _validate_group_combinations(grouped, profile_ids)
    if set(split["train_group_ids"]) & set(split["holdout_group_ids"]):
        raise ValueError("train/holdout group leakage")
    if len(split["train_job_ids"]) != 80 or len(split["holdout_job_ids"]) != 16:
        raise ValueError("written split is not 80/16")
    expected_train = {str(row["job_id"]) for row in jobs if row["split"] == "train"}
    expected_holdout = {
        str(row["job_id"]) for row in jobs if row["split"] == "locked_holdout"
    }
    if set(split["train_job_ids"]) != expected_train or set(split["holdout_job_ids"]) != expected_holdout:
        raise ValueError("split job ids do not match manifest")
    expected_train_groups = {str(row["group_id"]) for row in jobs if row["split"] == "train"}
    expected_holdout_groups = {
        str(row["group_id"]) for row in jobs if row["split"] == "locked_holdout"
    }
    if set(split["train_group_ids"]) != expected_train_groups or set(split["holdout_group_ids"]) != expected_holdout_groups:
        raise ValueError("split group ids do not match manifest")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capability-profiles", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = json.loads(Path(args.capability_profiles).read_text(encoding="utf-8"))
    if not isinstance(profiles, list):
        raise ValueError("capability profile file must contain a JSON list")
    paths = write_gold_plan(build_gold_plan(profiles), args.output_dir)
    audit_written_plan(paths)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
