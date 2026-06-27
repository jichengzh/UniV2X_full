#!/usr/bin/env python3
"""Build Stage2 AP stability registry and gate artifacts.

This script is deliberately conservative:
- true_eval / true_import sources may become canonical AP rows;
- weight_identity_transfer sources are registry/source-map evidence only;
- model-fit/interpolated sources are blocked and never emitted as measured AP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    ap_anchor_row,
    stable_config_id,
    utc_timestamp,
    validate_lut_row,
    write_jsonl as write_lut_jsonl,
)

OUT_ROOT_DEFAULT = (
    "multi_agent/data/stage2_lut_generation_v1/generated/ap_stability_20260626"
)

STAGE_A_CSV = Path("multi_agent/data/sources/stage_a_ap_real.csv")
DEPGRAPH_JSON = Path("results/ap70_depgraph_expansion.json")
AP_MODEL_JSON = Path("results/ap70_model_pyramid.json")
SMOKE_AP_ROWS = Path(
    "multi_agent/data/stage2_lut_generation_v1/generated/smoke/"
    "h800_tvm_v1/ap/ap_anchor_rows_v1.jsonl"
)
ORIGINAL60_TASKS = Path(
    "multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1/"
    "artifacts/artifact_tasks_original60_v1.jsonl"
)

B2_EVAL_SOURCES = {
    "iso_s0": {
        "path": Path("results/b2_eval/b2_iso_s0_fp16.json"),
        "width": [48, 128, 256],
    },
    "iso_s1": {
        "path": Path("results/b2_eval/b2_iso_s1_fp16.json"),
        "width": [64, 96, 256],
    },
    "iso_s2": {
        "path": Path("results/b2_eval/b2_iso_s2_fp16.json"),
        "width": [64, 128, 192],
    },
}

P50B2_SOURCE = {
    "label": "p50b2_136",
    "path": Path("results/p0_2_136/p50b2_136_fp16.json"),
    "width": [32, 64, 136],
    "ckpt_path": (
        "/home/jichengzhi/heal_research/checkpoints/stage1/"
        "Pyramid_DAIR_m1_prune50b2_032_064_136_2026_06_03/"
        "net_epoch_bestval_at35.pth"
    ),
    "config_path": (
        "/home/jichengzhi/heal_research/checkpoints/stage1/"
        "Pyramid_DAIR_m1_prune50b2_032_064_136_2026_06_03/config.yaml"
    ),
}

STAGE_A_LABELS = {
    "base": "base",
    "pruned25": "trap25",
    "pruned50": "p50",
    "pruned75": "p75",
}

CANONICAL_LABELS = [
    "base",
    "p50",
    "p75",
    "trap25",
    "mix_b",
    "mix_d",
    "iso_s0",
    "iso_s1",
    "iso_s2",
    "p50b2_136",
]

REPLAY_EXPECTED_AP70 = {
    "base": 0.6308636231895992,
    "p50": 0.5641315043713827,
    "trap25": 0.5904723815291678,
    "mix_d": 0.6369383152509643,
}

FINETUNE_SMOKE = [
    {
        "job_id": "ap_ft_01",
        "label": "s0_024",
        "width": [24, 128, 256],
        "gpu_id": 0,
        "reason": "stage0 low endpoint",
    },
    {
        "job_id": "ap_ft_02",
        "label": "s0_040",
        "width": [40, 128, 256],
        "gpu_id": 1,
        "reason": "stage0 middle point",
    },
    {
        "job_id": "ap_ft_03",
        "label": "s0_056",
        "width": [56, 128, 256],
        "gpu_id": 2,
        "reason": "stage0 near-base point",
    },
    {
        "job_id": "ap_ft_04",
        "label": "s1_048",
        "width": [64, 48, 256],
        "gpu_id": 3,
        "reason": "stage1 low endpoint",
    },
    {
        "job_id": "ap_ft_05",
        "label": "s2_160",
        "width": [64, 128, 160],
        "gpu_id": 4,
        "reason": "stage2 middle point",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    parser.add_argument("--created-at", default=None)
    parser.add_argument(
        "--finetune-preflight-status",
        choices=("queued", "preflight_blocked"),
        default="queued",
    )
    parser.add_argument("--finetune-preflight-reason", default="")
    parser.add_argument("--finetune-preflight-evidence-json", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    full = ROOT / path
    if not full.exists():
        return []
    return [
        json.loads(line)
        for line in full.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_raw_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def path_digest(path: Path) -> str:
    full = ROOT / path
    if not full.exists() or not full.is_file():
        return "missing"
    h = hashlib.sha256()
    with full.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest(paths: list[str]) -> str:
    h = hashlib.sha256()
    for value in sorted(paths):
        path = Path(value)
        rel_path = path if not path.is_absolute() else path
        digest = (
            hashlib.sha256(rel_path.read_bytes()).hexdigest()
            if rel_path.is_absolute() and rel_path.exists() and rel_path.is_file()
            else path_digest(path)
        )
        h.update(value.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def registry_row(
    *,
    label: str,
    width: list[int],
    source_kind: str,
    source_path: str,
    metric_value: float | None,
    secondary_metrics: dict[str, Any] | None = None,
    quant_policy: str = "fp16",
    source_label: str | None = None,
    dataset: str = "DAIR-V2X",
    eval_split: str = "val_1789",
    num_samples: int | None = 1789,
    ckpt_path: str = "unknown",
    ckpt_digest: str = "unknown",
    config_path: str = "unknown",
    finetune_protocol: str = "unknown",
    training_budget: str = "unknown",
    claim_status: str | None = None,
    source_files: list[str] | None = None,
    raw_artifact: str | None = None,
    notes: str = "",
    created_at: str,
) -> dict[str, Any]:
    status = claim_status
    if status is None:
        if source_kind in {"true_eval", "true_import"} and ckpt_digest == "unknown":
            status = "provisional_missing_digest"
        elif source_kind == "weight_identity_transfer":
            status = "claimable_weight_identity_transfer"
        elif source_kind == "model_fit_only":
            status = "blocked_model_fit_only"
        else:
            status = "no_claim_missing_source"
    files = source_files or [source_path]
    return {
        "schema": "stage2_ap_source_registry_row_v1",
        "label": label,
        "source_label": source_label or label,
        "candidate_id": f"ap_source:pyramid_lidar:{label}:{source_kind}:{quant_policy}",
        "width": [int(v) for v in width],
        "model": "pyramid_lidar",
        "source_kind": source_kind,
        "source_path": source_path,
        "source_digest": source_digest(files),
        "dataset": dataset,
        "eval_split": eval_split,
        "num_samples": num_samples,
        "config_path": config_path,
        "ckpt_path": ckpt_path,
        "ckpt_digest": ckpt_digest,
        "quant_policy": quant_policy,
        "metric": "AP70",
        "metric_value": metric_value,
        "secondary_metrics": secondary_metrics or {},
        "finetune_protocol": finetune_protocol,
        "training_budget": training_budget,
        "claim_status": status,
        "source_files": files,
        "raw_artifact": raw_artifact or source_path,
        "created_at": created_at,
        "notes": notes,
    }


def rows_from_stage_a(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / STAGE_A_CSV).open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            source_anchor = str(item["anchor"])
            precision = str(item["precision"])
            label = STAGE_A_LABELS.get(source_anchor, source_anchor)
            if precision != "fp16":
                label = f"{label}_{precision}"
            width = [
                int(item["stage0_planes"]),
                int(item["stage1_planes"]),
                int(item["stage2_planes"]),
            ]
            protocol = (
                "pretrained"
                if source_anchor == "base"
                else "pruned_finetuned_23_epochs"
            )
            rows.append(
                registry_row(
                    label=label,
                    source_label=source_anchor,
                    width=width,
                    source_kind="true_import",
                    source_path=str(STAGE_A_CSV),
                    source_files=[str(STAGE_A_CSV)],
                    metric_value=float(item["ap70"]),
                    secondary_metrics={
                        "AP30": float(item["ap30"]),
                        "AP50": float(item["ap50"]),
                        "n_trt_path": int(item["n_trt_path"]),
                    },
                    quant_policy=precision,
                    num_samples=int(item["n_samples"]),
                    ckpt_path=str(item["ckpt"]),
                    finetune_protocol=protocol,
                    training_budget="historical_stage_a",
                    raw_artifact=str(STAGE_A_CSV),
                    created_at=created_at,
                    notes="stage_a real AP anchor imported from CSV",
                )
            )
    return rows


def rows_from_depgraph(created_at: str) -> list[dict[str, Any]]:
    data = load_json(DEPGRAPH_JSON)
    rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for item in data.get("results", []):
        label = str(item["tag"])
        by_label[label] = item
        rows.append(
            registry_row(
                label=label,
                source_label=label,
                width=[int(v) for v in item["num_filters"]],
                source_kind="true_eval",
                source_path=str(DEPGRAPH_JSON),
                source_files=[str(DEPGRAPH_JSON)],
                metric_value=float(item["ap70"]),
                secondary_metrics={
                    "AP30": float(item["ap30"]),
                    "AP50": float(item["ap50"]),
                    "n_trt_path": int(item.get("n_trt_path", 0)),
                },
                num_samples=int(item.get("n_samples", 1789)),
                ckpt_path=str(item.get("ckpt", item.get("ckpt_dir", "unknown"))),
                config_path=str(item.get("ckpt_dir", "unknown")) + "/config.yaml",
                finetune_protocol=str(item.get("protocol", data.get("protocol", "unknown"))),
                training_budget="historical_b4expand_epoch31",
                raw_artifact=str(DEPGRAPH_JSON),
                created_at=created_at,
                notes="B4 expansion true AP eval; protocol must be preserved",
            )
        )
    for pair in data.get("wg_pg_pairs_enabled", []):
        wg_tag = str(pair["W_g_tag"])
        source = by_label.get(wg_tag, {})
        rows.append(
            registry_row(
                label=str(pair["P_g_tag"]),
                source_label=wg_tag,
                width=[int(v) for v in pair["P_g_widths"]],
                source_kind="weight_identity_transfer",
                source_path=str(DEPGRAPH_JSON),
                source_files=[str(DEPGRAPH_JSON)],
                metric_value=(
                    float(source["ap70"]) if source.get("ap70") is not None else None
                ),
                secondary_metrics={
                    "AP30": source.get("ap30"),
                    "AP50": source.get("ap50"),
                },
                ckpt_path=str(source.get("ckpt", source.get("ckpt_dir", "unknown"))),
                config_path=str(source.get("ckpt_dir", "unknown")) + "/config.yaml",
                finetune_protocol="weight_identity_zero_pad_transfer",
                training_budget="inherits_W_g_AP_by_weight_identity",
                raw_artifact=str(DEPGRAPH_JSON),
                created_at=created_at,
                notes=str(pair.get("weight_identity", "")),
            )
        )
    return rows


def model_fit_related_rows(created_at: str) -> list[dict[str, Any]]:
    data = load_json(AP_MODEL_JSON)
    rows: list[dict[str, Any]] = [
        registry_row(
            label="ap70_loglinear_model",
            width=[int(v) for v in data.get("base_widths", [64, 128, 256])],
            source_kind="model_fit_only",
            source_path=str(AP_MODEL_JSON),
            source_files=[str(AP_MODEL_JSON)],
            metric_value=None,
            secondary_metrics={
                "model_form": data.get("model_form"),
                "fit_mae": data.get("fit_mae"),
                "fit_max_abs_residual": data.get("fit_max_abs_residual"),
                "n_fit_points": data.get("n_fit_points"),
            },
            dataset="not_applicable",
            eval_split="not_applicable",
            num_samples=None,
            ckpt_path="not_applicable",
            ckpt_digest="not_applicable",
            config_path="not_applicable",
            finetune_protocol="not_applicable",
            training_budget="model_fit_only_no_measured_claim",
            claim_status="blocked_model_fit_only",
            raw_artifact=str(AP_MODEL_JSON),
            created_at=created_at,
            notes="Fitted AP model; never emit as measured AP",
        )
    ]
    for item in data.get("validation_points", []):
        rows.append(
            registry_row(
                label=str(item["tag"]),
                width=[int(item["s0"]), int(item["s1"]), int(item["s2"])],
                source_kind="true_eval",
                source_path=str(AP_MODEL_JSON),
                source_files=[str(AP_MODEL_JSON)],
                metric_value=float(item["ap70"]),
                secondary_metrics={
                    "AP30": item.get("ap30"),
                    "AP50": item.get("ap50"),
                },
                num_samples=int(item.get("n_samples", 1789)),
                ckpt_path=str(item.get("ckpt", "unknown")),
                config_path=str(Path(str(item.get("ckpt", "unknown"))).parent / "config.yaml"),
                finetune_protocol=str(item.get("source", "b2_true_eval")),
                training_budget="historical_b2_epoch31",
                raw_artifact=str(AP_MODEL_JSON),
                created_at=created_at,
                notes="AP model validation point is true eval evidence, not model-fit prediction",
            )
        )
    for item in data.get("table", []):
        src = str(item.get("src", ""))
        width = [int(v) for v in item["num_filters"]]
        if "weight-identity" not in src:
            continue
        label = (
            "pad64"
            if width == [64, 96, 192]
            else "s1_64"
            if width == [64, 64, 256]
            else "s2_128"
            if width == [64, 128, 128]
            else "weight_identity_" + "_".join(str(v) for v in width)
        )
        rows.append(
            registry_row(
                label=label,
                source_label=src,
                width=width,
                source_kind="weight_identity_transfer",
                source_path=str(AP_MODEL_JSON),
                source_files=[str(AP_MODEL_JSON)],
                metric_value=float(item["ap70"]),
                finetune_protocol="weight_identity_transfer_from_model_table",
                training_budget="inherits_AP_by_weight_identity",
                raw_artifact=str(AP_MODEL_JSON),
                created_at=created_at,
                notes=src,
            )
        )
    return rows


def rows_from_b2_and_p50b2(created_at: str) -> list[dict[str, Any]]:
    model_data = load_json(AP_MODEL_JSON)
    ckpt_by_label: dict[str, str] = {}
    for item in model_data.get("fit_points", []):
        if item.get("ckpt"):
            ckpt_by_label[str(item["tag"])] = str(item["ckpt"])
    rows: list[dict[str, Any]] = []
    for label, spec in B2_EVAL_SOURCES.items():
        data = load_json(spec["path"])
        ckpt = ckpt_by_label.get(label, "unknown")
        rows.append(
            registry_row(
                label=label,
                width=list(spec["width"]),
                source_kind="true_eval",
                source_path=str(spec["path"]),
                source_files=[str(spec["path"]), str(AP_MODEL_JSON)],
                metric_value=float(data["ap70"]),
                secondary_metrics={
                    "AP30": float(data["ap30"]),
                    "AP50": float(data["ap50"]),
                    "n_trt_path": int(data.get("n_trt_path", 0)),
                },
                num_samples=int(data.get("n_samples", 1789)),
                ckpt_path=ckpt,
                config_path=str(Path(ckpt).parent / "config.yaml") if ckpt != "unknown" else "unknown",
                finetune_protocol="b2_finetune_TRT_FP16_DAIR_1789",
                training_budget="historical_b2_epoch31",
                raw_artifact=str(spec["path"]),
                created_at=created_at,
                notes="B2 true AP eval source",
            )
        )
    data = load_json(P50B2_SOURCE["path"])
    rows.append(
        registry_row(
            label=P50B2_SOURCE["label"],
            width=list(P50B2_SOURCE["width"]),
            source_kind="true_eval",
            source_path=str(P50B2_SOURCE["path"]),
            source_files=[str(P50B2_SOURCE["path"]), "scripts/phase2/p0_2_136_ap.py"],
            metric_value=float(data["ap70"]),
            secondary_metrics={
                "AP30": float(data["ap30"]),
                "AP50": float(data["ap50"]),
                "n_trt_path": int(data.get("n_trt_path", 0)),
            },
            num_samples=int(data.get("n_samples", 1789)),
            ckpt_path=P50B2_SOURCE["ckpt_path"],
            config_path=P50B2_SOURCE["config_path"],
            finetune_protocol="p50b2_136_true_eval_bestval_at35_TRT_FP16_DAIR_1789",
            training_budget="historical_p50b2_epoch48_best35",
            raw_artifact=str(P50B2_SOURCE["path"]),
            created_at=created_at,
            notes="P0-2 p50b2_136 true AP eval source",
        )
    )
    return rows


def rows_from_existing_ap_rows(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in load_jsonl(SMOKE_AP_ROWS):
        label = str(item.get("candidate_id", "")).split(":")[-1]
        rows.append(
            registry_row(
                label=label,
                width=[int(v) for v in item["width"]],
                source_kind="true_import",
                source_path=str(SMOKE_AP_ROWS),
                source_files=[str(SMOKE_AP_ROWS), *[str(v) for v in item.get("source_files", [])]],
                metric_value=float(item["metric_value"]),
                secondary_metrics=dict(item.get("secondary_metrics", {})),
                quant_policy=str(item.get("quant_policy", "fp16")),
                num_samples=item.get("num_samples"),
                ckpt_path=str(item.get("ckpt_path", "unknown")),
                ckpt_digest=str(item.get("ckpt_digest", "unknown")),
                config_path=str(item.get("config_path", "unknown")),
                finetune_protocol=str(item.get("finetune_protocol", "unknown")),
                training_budget=str(item.get("training_budget", "unknown")),
                raw_artifact=str(item.get("raw_artifact", SMOKE_AP_ROWS)),
                created_at=created_at,
                notes="existing ap_anchor_row classified as true_import source",
            )
        )
    return rows


def build_registry(created_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(rows_from_stage_a(created_at))
    rows.extend(rows_from_depgraph(created_at))
    rows.extend(rows_from_b2_and_p50b2(created_at))
    rows.extend(model_fit_related_rows(created_at))
    rows.extend(rows_from_existing_ap_rows(created_at))
    return rows


def prefer_registry_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    matches = [row for row in rows if row["label"] == label and row["quant_policy"] == "fp16"]
    priority = {"true_eval": 0, "true_import": 1}
    matches = [row for row in matches if row["source_kind"] in priority]
    if not matches:
        raise KeyError(label)
    return sorted(matches, key=lambda row: priority[row["source_kind"]])[0]


def canonical_ap_row(source: dict[str, Any], created_at: str) -> dict[str, Any]:
    label = str(source["label"])
    width = [int(v) for v in source["width"]]
    software_point_id = f"pyramid_lidar:full_model_ap:w{width[0]}x{width[1]}x{width[2]}:fp16"
    config_id = stable_config_id(
        model="pyramid_lidar",
        candidate_id=f"ap_stability:{label}",
        software_point_id=software_point_id,
        quant_policy="fp16",
        schedule_policy="not_applicable",
    )
    row = ap_anchor_row(
        config_id=config_id,
        model="pyramid_lidar",
        manifest_digest=str(source["source_digest"]),
        candidate_id=f"ap_stability:pyramid_lidar:{label}",
        software_point_id=software_point_id,
        dense_stage="backbone",
        optimized_scope="full_model_ap_eval",
        width=width,
        quant_policy="fp16",
        schedule_policy="not_applicable",
        hardware_target="not_hardware_specific",
        backend="model_eval",
        measurement_status="measured",
        metric="AP70",
        metric_value=float(source["metric_value"]),
        secondary_metrics=dict(source.get("secondary_metrics", {})),
        dataset=str(source["dataset"]),
        eval_split=str(source["eval_split"]),
        num_samples=source.get("num_samples"),
        ckpt_path=str(source["ckpt_path"]),
        ckpt_digest=str(source["ckpt_digest"]),
        config_path=str(source["config_path"]),
        finetune_protocol=str(source["finetune_protocol"]),
        training_budget=str(source["training_budget"]),
        eval_command=f"canonical_import_from_ap_source_registry label={label}",
        run_id=f"ap_stability_20260626_import_{label}",
        created_at=created_at,
        source_files=list(source.get("source_files", [])),
        raw_artifact=str(source["raw_artifact"]),
        provenance="stage2_ap_stability_true_source_registry",
        notes=(
            f"source_kind={source['source_kind']}; "
            f"claim_status={source['claim_status']}; "
            "AP value comes from true source registry, not prediction/interpolation"
        ),
        claim_status=str(source["claim_status"]),
        ap_source_kind=str(source["source_kind"]),
        source_registry_candidate_id=str(source["candidate_id"]),
    )
    validate_lut_row(row)
    return row


def build_canonical_rows(registry: list[dict[str, Any]], created_at: str) -> list[dict[str, Any]]:
    return [canonical_ap_row(prefer_registry_row(registry, label), created_at) for label in CANONICAL_LABELS]


def build_replay_gate(canonical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = {
        str(row["candidate_id"]).split(":")[-1]: row
        for row in canonical_rows
    }
    checks = []
    for label, expected in REPLAY_EXPECTED_AP70.items():
        row = by_label.get(label)
        if row is None:
            checks.append(
                {
                    "label": label,
                    "status": "missing",
                    "expected_ap70": expected,
                    "observed_ap70": None,
                    "abs_diff": None,
                    "gate": "fail",
                }
            )
            continue
        observed = float(row["metric_value"])
        diff = abs(observed - expected)
        checks.append(
            {
                "label": label,
                "width": row["width"],
                "expected_ap70": expected,
                "observed_ap70": observed,
                "abs_diff": diff,
                "threshold": 0.003,
                "gate": "pass" if diff <= 0.003 else "fail",
                "mode": "true_source_import_replay_gate_no_fresh_eval",
                "row_id": row["row_id"],
            }
        )
    passed = sum(1 for item in checks if item["gate"] == "pass")
    return {
        "schema": "stage2_ap_replay_gate_report_v1",
        "gate_status": "pass" if passed >= 3 and all(item["gate"] == "pass" for item in checks) else "fail",
        "passed_count": passed,
        "required_passed_count": 3,
        "threshold_abs_ap70": 0.003,
        "mode": "true_source_import_replay_gate_no_fresh_eval",
        "checks": checks,
    }


def build_original60_map(
    registry: list[dict[str, Any]],
    created_at: str,
    *,
    finetune_preflight_status: str,
    finetune_preflight_reason: str,
) -> list[dict[str, Any]]:
    tasks = load_jsonl(ORIGINAL60_TASKS)
    exact_sources: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    transfer_sources: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in registry:
        width = tuple(int(v) for v in row["width"])
        if row["quant_policy"] != "fp16":
            continue
        if row["source_kind"] in {"true_eval", "true_import"}:
            exact_sources.setdefault(width, []).append(row)
        elif row["source_kind"] == "weight_identity_transfer":
            transfer_sources.setdefault(width, []).append(row)
    smoke_labels = {item["label"] for item in FINETUNE_SMOKE}
    output = []
    for task in tasks:
        label = str(task["label"])
        width = tuple(int(v) for v in task["width"])
        source = None
        status = "no_claim_missing_source"
        claim_status = "no_claim_missing_source"
        next_action = "needs_finetune_or_true_eval"
        source_kind = "no_claim"
        if width in exact_sources:
            source = prefer_best_source(exact_sources[width])
            status = "exact_true_source_available"
            claim_status = str(source["claim_status"])
            next_action = "can_generate_true_import_or_true_eval_ap_row"
            source_kind = str(source["source_kind"])
        elif width in transfer_sources:
            source = prefer_best_source(transfer_sources[width])
            status = "weight_identity_transfer_available"
            claim_status = "claimable_weight_identity_transfer"
            next_action = "can_claim_transfer_after_identity_gate"
            source_kind = "weight_identity_transfer"
        elif label in smoke_labels:
            status = (
                "finetune_smoke_blocked_environment"
                if finetune_preflight_status == "preflight_blocked"
                else "finetune_smoke_queued"
            )
            claim_status = "no_claim_missing_source"
            next_action = (
                "fix_h800_ap_training_environment_then_resume"
                if finetune_preflight_status == "preflight_blocked"
                else "run_phase_c_5gpu_finetune_smoke"
            )
            source_kind = "no_claim"
        output.append(
            {
                "schema": "stage2_original60_ap_source_map_row_v1",
                "label": label,
                "candidate_id": task.get("candidate_id"),
                "config_id": task.get("config_id"),
                "width": list(width),
                "model": task.get("model", "Pyramid-LiDAR"),
                "quant_policy": task.get("quant_policy", "fp16"),
                "optimized_scope": task.get("optimized_scope", "backbone_only"),
                "artifact_status": task.get("artifact_status"),
                "ap_source_status": status,
                "claim_status": claim_status,
                "source_kind": source_kind,
                "source_label": source.get("label") if source else None,
                "source_path": source.get("source_path") if source else None,
                "metric": "AP70",
                "metric_value": source.get("metric_value") if source else None,
                "next_action": next_action,
                "created_at": created_at,
                "notes": (
                    finetune_preflight_reason
                    if label in smoke_labels and finetune_preflight_status == "preflight_blocked"
                    else
                    "AP needs full-model ckpt/config; backbone ONNX/TVM artifact alone is insufficient"
                    if source is None
                    else str(source.get("notes", ""))
                ),
            }
        )
    return output


def prefer_best_source(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {
        "true_eval": 0,
        "true_import": 1,
        "weight_identity_transfer": 2,
        "model_fit_only": 9,
    }
    return sorted(rows, key=lambda row: priority.get(str(row["source_kind"]), 99))[0]


def build_gap_report(source_map: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["ap_source_status"] for row in source_map)
    claim_counts = Counter(row["claim_status"] for row in source_map)
    return {
        "schema": "stage2_original60_ap_axis_gap_report_v2",
        "total_original60": len(source_map),
        "ap_source_status_counts": dict(sorted(status_counts.items())),
        "claim_status_counts": dict(sorted(claim_counts.items())),
        "exact_true_source_count": status_counts.get("exact_true_source_available", 0),
        "weight_identity_transfer_count": status_counts.get(
            "weight_identity_transfer_available", 0
        ),
        "finetune_smoke_blocked_environment_count": status_counts.get(
            "finetune_smoke_blocked_environment", 0
        ),
        "finetune_smoke_queued_count": status_counts.get("finetune_smoke_queued", 0),
        "no_claim_missing_source_count": status_counts.get("no_claim_missing_source", 0),
        "policy": (
            "No predicted/model-fit/interpolated AP is written as measured. "
            "Rows without full-model AP source remain no-claim until finetune/eval."
        ),
    }


def build_finetune_jobs(created_at: str) -> list[dict[str, Any]]:
    rows = []
    for item in FINETUNE_SMOKE:
        label = item["label"]
        s0, s1, s2 = item["width"]
        gpu = int(item["gpu_id"])
        ckpt_dir = (
            "/home/jichengzhi/heal_research/checkpoints/stage1/"
            f"Pyramid_DAIR_m1_stage2_ap_{label}_2026_06_26"
        )
        raw_dir = (
            "multi_agent/data/stage2_lut_generation_v1/generated/"
            f"ap_stability_20260626/raw/{label}"
        )
        rows.append(
            {
                "schema": "stage2_ap_finetune_smoke_job_v1",
                "job_id": item["job_id"],
                "label": label,
                "width": [s0, s1, s2],
                "gpu_id": gpu,
                "master_port": 29700 + gpu,
                "finetune_runs": 1,
                "epoches": 31,
                "init_epoch": 23,
                "finetune_epochs": 8,
                "dataset_train": "HEAL/DAIR-V2X-C train split train=4811",
                "dataset_eval": "DAIR-V2X-C val_1789",
                "ckpt_dir": ckpt_dir,
                "raw_dir": raw_dir,
                "status": "queued",
                "created_at": created_at,
                "commands": {
                    "preflight_record": [
                        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate --format=csv",
                        "nvidia-smi pmon -c 1",
                    ],
                    "structural_prune": [
                        f"CUDA_VISIBLE_DEVICES={gpu}",
                        "PYTHONPATH=/home/jichengzhi/heal_research/HEAL",
                        "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python",
                        "tools/structural_prune_pyramid.py",
                        "--orig-dir",
                        "/home/jichengzhi/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29",
                        "--out-dir",
                        ckpt_dir,
                        "--num-filters-new",
                        f"{s0},{s1},{s2}",
                        "--width-per-group",
                        "4",
                        "--groups",
                        "32",
                    ],
                    "train_ddp": [
                        f"CUDA_VISIBLE_DEVICES={gpu}",
                        "PYTHONPATH=/home/jichengzhi/heal_research/HEAL",
                        "/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python",
                        "-m",
                        "torch.distributed.launch",
                        "--nproc_per_node=1",
                        "--use_env",
                        f"--master_port={29700 + gpu}",
                        "/home/jichengzhi/heal_research/HEAL/opencood/tools/train_ddp.py",
                        "--hypes_yaml",
                        f"{ckpt_dir}/config.yaml",
                        "--model_dir",
                        ckpt_dir,
                        "--half",
                    ],
                },
                "notes": (
                    "GPU idle is not required for AP finetune/eval; only avoid OOM. "
                    f"Selection reason: {item['reason']}"
                ),
            }
        )
    return rows


def build_finetune_state_rows(
    finetune_jobs: list[dict[str, Any]],
    *,
    status: str,
    reason: str,
    evidence: dict[str, Any],
    created_at: str,
) -> list[dict[str, Any]]:
    if status == "queued":
        return []
    return [
        {
            "schema": "stage2_ap_finetune_smoke_job_state_v1",
            "job_id": job["job_id"],
            "label": job["label"],
            "width": job["width"],
            "gpu_id": job["gpu_id"],
            "status": status,
            "attempt": 0,
            "failure_reason": reason,
            "evidence": evidence,
            "created_at": created_at,
            "notes": "No AP value was generated; no measured AP row was written.",
        }
        for job in finetune_jobs
    ]


def build_finetune_quarantine_rows(
    finetune_jobs: list[dict[str, Any]],
    *,
    status: str,
    reason: str,
    evidence: dict[str, Any],
    created_at: str,
) -> list[dict[str, Any]]:
    if status == "queued":
        return []
    return [
        {
            "schema": "stage2_ap_quarantine_row_v1",
            "job_id": job["job_id"],
            "label": job["label"],
            "width": job["width"],
            "quarantine_status": "active",
            "claim_status": "no_claim_missing_source",
            "failure_reason": reason,
            "evidence": evidence,
            "created_at": created_at,
            "next_action": "fix H800 AP training environment, then rerun Phase C smoke",
        }
        for job in finetune_jobs
    ]


def write_summary_files(
    *,
    out_root: Path,
    registry: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    source_map: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    inv_lines = [
        "# AP Source Inventory Summary",
        "",
        "| source_kind | count |",
        "| --- | ---: |",
    ]
    for kind, count in sorted(Counter(row["source_kind"] for row in registry).items()):
        inv_lines.append(f"| {kind} | {count} |")
    inv_lines.extend(
        [
            "",
            "| label | width | source_kind | AP70 | claim_status | source_path |",
            "| --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for row in registry:
        value = row["metric_value"]
        value_text = "" if value is None else f"{float(value):.6f}"
        inv_lines.append(
            "| {label} | {width} | {kind} | {value} | {claim} | {source} |".format(
                label=row["label"],
                width=row["width"],
                kind=row["source_kind"],
                value=value_text,
                claim=row["claim_status"],
                source=row["source_path"],
            )
        )
    write_markdown(out_root / "exports/ap_source_inventory_summary.md", "\n".join(inv_lines) + "\n")

    stability_lines = [
        "# AP Stability Summary",
        "",
        f"Replay gate: `{gate['gate_status']}` ({gate['passed_count']}/{len(gate['checks'])} pass)",
        "",
        "| label | width | AP30 | AP50 | AP70 | source_kind | claim_status | ckpt_digest |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in canonical_rows:
        label = str(row["candidate_id"]).split(":")[-1]
        secondary = dict(row.get("secondary_metrics", {}))
        stability_lines.append(
            "| {label} | {width} | {ap30:.6f} | {ap50:.6f} | {ap70:.6f} | {kind} | {claim} | {digest} |".format(
                label=label,
                width=row["width"],
                ap30=float(secondary.get("AP30", 0.0) or 0.0),
                ap50=float(secondary.get("AP50", 0.0) or 0.0),
                ap70=float(row["metric_value"]),
                kind=row.get("ap_source_kind"),
                claim=row.get("claim_status"),
                digest=row.get("ckpt_digest"),
            )
        )
    write_markdown(out_root / "exports/ap_stability_summary.md", "\n".join(stability_lines) + "\n")

    csv_path = out_root / "exports/ap_stability_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "label",
                "width",
                "ap30",
                "ap50",
                "ap70",
                "source_kind",
                "claim_status",
                "ckpt_path",
                "ckpt_digest",
            ],
        )
        writer.writeheader()
        for row in canonical_rows:
            secondary = dict(row.get("secondary_metrics", {}))
            writer.writerow(
                {
                    "label": str(row["candidate_id"]).split(":")[-1],
                    "width": "/".join(str(v) for v in row["width"]),
                    "ap30": secondary.get("AP30"),
                    "ap50": secondary.get("AP50"),
                    "ap70": row["metric_value"],
                    "source_kind": row.get("ap_source_kind"),
                    "claim_status": row.get("claim_status"),
                    "ckpt_path": row.get("ckpt_path"),
                    "ckpt_digest": row.get("ckpt_digest"),
                }
            )

    review_lines = [
        "# Original60 AP Source Map Quick Review",
        "",
        "| status | count |",
        "| --- | ---: |",
    ]
    for status, count in sorted(Counter(row["ap_source_status"] for row in source_map).items()):
        review_lines.append(f"| {status} | {count} |")
    review_lines.extend(["", "| label | width | status | AP70 | next_action |", "| --- | --- | --- | ---: | --- |"])
    for row in source_map:
        value = row["metric_value"]
        value_text = "" if value is None else f"{float(value):.6f}"
        review_lines.append(
            f"| {row['label']} | {row['width']} | {row['ap_source_status']} | {value_text} | {row['next_action']} |"
        )
    write_markdown(out_root / "exports/original60_ap_source_map_quick_review.md", "\n".join(review_lines) + "\n")


def main() -> int:
    args = parse_args()
    created_at = args.created_at or utc_timestamp()
    out_root = ROOT / args.out_root
    evidence = (
        json.loads(args.finetune_preflight_evidence_json)
        if args.finetune_preflight_evidence_json
        else {}
    )

    registry = build_registry(created_at)
    canonical_rows = build_canonical_rows(registry, created_at)
    replay_gate = build_replay_gate(canonical_rows)
    finetune_jobs = build_finetune_jobs(created_at)
    source_map = build_original60_map(
        registry,
        created_at,
        finetune_preflight_status=args.finetune_preflight_status,
        finetune_preflight_reason=args.finetune_preflight_reason,
    )
    gap_report = build_gap_report(source_map)
    finetune_state_rows = build_finetune_state_rows(
        finetune_jobs,
        status=args.finetune_preflight_status,
        reason=args.finetune_preflight_reason,
        evidence=evidence,
        created_at=created_at,
    )
    finetune_quarantine_rows = build_finetune_quarantine_rows(
        finetune_jobs,
        status=args.finetune_preflight_status,
        reason=args.finetune_preflight_reason,
        evidence=evidence,
        created_at=created_at,
    )

    write_raw_jsonl(out_root / "registry/ap_source_registry_v1.jsonl", registry)
    write_lut_jsonl(out_root / "rows/ap_anchor_rows_v1.jsonl", canonical_rows)
    write_raw_json(out_root / "exports/ap_replay_gate_report.json", replay_gate)
    write_raw_jsonl(out_root / "exports/original60_ap_source_map_v1.jsonl", source_map)
    write_raw_json(out_root / "exports/original60_ap_axis_gap_report_v2.json", gap_report)
    write_raw_jsonl(out_root / "jobs/ap_finetune_smoke_queue_v1.jsonl", finetune_jobs)
    write_raw_jsonl(out_root / "jobs/ap_eval_job_queue_v1.jsonl", [])
    write_raw_jsonl(out_root / "jobs/ap_eval_job_state_v1.jsonl", [])
    write_raw_jsonl(out_root / "jobs/ap_finetune_smoke_job_state_v1.jsonl", finetune_state_rows)
    write_raw_jsonl(out_root / "quarantine/ap_unstable_or_unclaimable_v1.jsonl", finetune_quarantine_rows)
    write_summary_files(
        out_root=out_root,
        registry=registry,
        canonical_rows=canonical_rows,
        source_map=source_map,
        gate=replay_gate,
    )

    print(
        "stage2_build_ap_stability_artifacts_ok "
        f"registry_rows={len(registry)} canonical_rows={len(canonical_rows)} "
        f"gate={replay_gate['gate_status']} original60={len(source_map)} "
        f"finetune_jobs={len(finetune_jobs)} out={out_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
