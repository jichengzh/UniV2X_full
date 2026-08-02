"""Build source-aware Stage5 candidates without target-label leakage."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


W0 = (16, 24, 32, 40, 48, 56, 64)
W1 = (32, 48, 64, 80, 96, 112, 128)
W2 = (64, 96, 128, 160, 192, 224, 256)
METADATA_FIELDS = {
    "group_id",
    "model",
    "width",
    "input_dims",
    "onnx_path",
    "onnx_sha256",
    "extractor_script_sha256",
    "source_gold_sha256",
    "schema",
}
LABEL_TOKENS = ("latency", "energy", "ap30", "ap50", "ap70", "target_", "measured_")


@dataclass
class GraphFeatureSurrogate:
    feature_names: tuple[str, ...]
    model: ExtraTreesRegressor
    training_sha256: str


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_widths() -> list[list[int]]:
    return [list(width) for width in product(W0, W1, W2)]


def build_pyramid_checkpoint_inventory(
    candidates: Sequence[Mapping[str, Any]], *, checkpoint_root: str | Path
) -> dict[str, Any]:
    root = Path(checkpoint_root)
    unique: dict[str, list[int]] = {}
    for candidate in candidates:
        label = str(candidate.get("label") or "")
        width = [int(value) for value in candidate.get("width") or []]
        if not label or len(width) != 3:
            raise ValueError("Pyramid checkpoint candidates require label and three widths")
        if label in unique and unique[label] != width:
            raise ValueError(f"Pyramid label maps to multiple widths: {label}")
        unique[label] = width
    sources = []
    missing = []
    for label, width in sorted(unique.items()):
        directory = root / f"Pyramid_DAIR_m1_o60_{label}_fresh_v1"
        checkpoint = directory / "net_epoch_bestval_at1.pth"
        config = directory / "config.yaml"
        if not checkpoint.is_file() or not config.is_file():
            missing.append(label)
            continue
        sources.append(
            {
                "label": label,
                "width": width,
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _file_sha(checkpoint),
                "config_path": str(config),
                "config_sha256": _file_sha(config),
            }
        )
    return {
        "schema_version": "stage5_pyramid_checkpoint_inventory_v1",
        "candidate_count": len(unique),
        "source_count": len(sources),
        "missing_labels": missing,
        "sources": sources,
    }


def _base_features(model: str, width: Sequence[int]) -> list[float]:
    values = [float(value) for value in width]
    return [*values, float(np.log1p(np.prod(values))), float(model == "codriving")]


def fit_graph_feature_surrogate(
    graph_features: Sequence[Mapping[str, Any]], *, seed: int = 20260717
) -> GraphFeatureSurrogate:
    records = [dict(item) for item in graph_features]
    if not records:
        raise ValueError("graph feature training rows must not be empty")
    field_names = {str(name).lower() for record in records for name in record}
    leaked = sorted(
        name for name in field_names if any(token in name for token in LABEL_TOKENS)
    )
    if leaked:
        raise ValueError(f"label-like graph features are forbidden: {leaked}")
    feature_names = tuple(
        sorted(
            {
                str(name)
                for record in records
                for name, value in record.items()
                if name not in METADATA_FIELDS and _finite(value)
            }
        )
    )
    if not feature_names:
        raise ValueError("no numeric graph features available")
    matrix = np.asarray(
        [_base_features(str(row["model"]), row["width"]) for row in records], dtype=float
    )
    targets = np.asarray(
        [
            [float(row.get(name, 0.0)) if _finite(row.get(name)) else 0.0 for name in feature_names]
            for row in records
        ],
        dtype=float,
    )
    model = ExtraTreesRegressor(
        n_estimators=192,
        max_depth=10,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(matrix, targets)
    return GraphFeatureSurrogate(
        feature_names=feature_names,
        model=model,
        training_sha256=_sha(records),
    )


def predict_graph_features(
    surrogate: GraphFeatureSurrogate, *, model: str, width: Sequence[int]
) -> dict[str, Any]:
    prediction = np.asarray(
        surrogate.model.predict(np.asarray([_base_features(model, width)], dtype=float))[0],
        dtype=float,
    )
    group_id = f"{model}|{'x'.join(map(str, width))}"
    return {
        "group_id": group_id,
        "model": model,
        "width": [int(value) for value in width],
        **{
            name: float(max(0.0, value))
            for name, value in zip(surrogate.feature_names, prediction)
        },
        "graph_feature_provenance": "coldstart_width_conditioned_surrogate_v1",
        "graph_feature_surrogate_training_sha256": surrogate.training_sha256,
    }


def _padded(width: Sequence[int]) -> str:
    return "x".join(f"{int(value):03d}" for value in width)


def _materialization_evidence(
    *, group_id: str, contract: Mapping[str, Any], plan_sha: str
) -> dict[str, str] | None:
    marker = Path(str(contract.get("source_done_marker") or ""))
    evidence_path = Path(str(marker)[:-5] + "_evidence.json") if str(marker).endswith(".done") else Path()
    if not marker.is_file() or not evidence_path.is_file():
        return None
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        evidence.get("schema_version") != "stage5_source_materialization_evidence_v1"
        or evidence.get("group_id") != group_id
        or evidence.get("source_plan_sha256") != plan_sha
        or evidence.get("status") != "ready"
    ):
        return None
    bindings = (
        ("onnx_path", "onnx_path", "onnx_sha256"),
        ("calibration_npz", "calibration_path", "calibration_sha256"),
        ("calibration_summary", "calibration_summary_path", "calibration_summary_sha256"),
    )
    for contract_key, evidence_path_key, evidence_sha_key in bindings:
        artifact = Path(str(evidence.get(evidence_path_key) or ""))
        if (
            str(artifact) != str(contract.get(contract_key) or "")
            or not artifact.is_file()
            or _file_sha(artifact) != str(evidence.get(evidence_sha_key) or "")
        ):
            return None
    checkpoint = Path(str(evidence.get("checkpoint_path") or ""))
    if not checkpoint.is_file() or _file_sha(checkpoint) != str(evidence.get("checkpoint_sha256") or ""):
        return None
    trt_dir = Path(str(contract.get("trt_calibration_dir") or ""))
    if not trt_dir.is_dir() or not any(trt_dir.glob("*.npy")):
        return None
    return {"path": str(evidence_path), "sha256": _file_sha(evidence_path)}


def _pyramid_group(
    width: Sequence[int],
    *,
    source: Mapping[str, Any] | None,
    surrogate: GraphFeatureSurrogate,
    result_root: Path,
    model_root: Path | None,
    base_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values = [int(value) for value in width]
    width_key = "x".join(map(str, values))
    padded = _padded(values)
    source_dir = result_root / "pyramid_sources" / padded
    calibration_dir = result_root / "pyramid_calibration" / padded
    if source is not None:
        checkpoint = Path(str(source["checkpoint_path"]))
        expected_sha = str(source.get("checkpoint_sha256") or "")
        if len(expected_sha) != 64:
            raise ValueError(f"Pyramid checkpoint SHA missing: {width_key}")
        checkpoint_contract = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_dir": str(checkpoint.parent),
            "checkpoint_sha256": expected_sha,
            "training_required": False,
        }
        materialization_kind = "pyramid_checkpoint_export"
    else:
        if model_root is None or base_source is None:
            raise ValueError(f"Pyramid width lacks checkpoint and training contract: {width_key}")
        base_checkpoint = str(base_source.get("checkpoint_path") or "")
        base_sha = str(base_source.get("checkpoint_sha256") or "")
        base_dir = str(base_source.get("checkpoint_dir") or Path(base_checkpoint).parent)
        if not base_checkpoint or len(base_sha) != 64:
            raise ValueError("Pyramid base checkpoint contract requires path and SHA256")
        checkpoint_dir = model_root / width_key
        checkpoint_contract = {
            "checkpoint_path": str(checkpoint_dir / "stage5_best.pth"),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_sha256": None,
            "config_path": str(checkpoint_dir / "config.yaml"),
            "training_done_marker": str(checkpoint_dir / "stage5_training_complete.json"),
            "base_checkpoint_path": base_checkpoint,
            "base_checkpoint_dir": base_dir,
            "base_checkpoint_sha256": base_sha,
            "training_required": True,
            "training_epoches": 31,
            "width_per_group": 4,
            "groups": 32,
        }
        materialization_kind = "pyramid_prepare_train_export"
    contract = {
        **checkpoint_contract,
        "onnx_path": str(source_dir / f"pyramid_{padded}_multiscale.onnx"),
        "onnx_report_path": str(source_dir / "onnx_export_report.json"),
        "calibration_root": str(calibration_dir),
        "calibration_npz": str(calibration_dir / "spatial_features_train16.npz"),
        "calibration_summary": str(calibration_dir / "summary.json"),
        "trt_calibration_dir": str(calibration_dir / "trt_npy"),
        "source_done_marker": str(result_root / "source_prep" / f"pyramid_{width_key}.done"),
    }
    plan_sha = _sha({"kind": materialization_kind, "width": values, "contract": contract})
    evidence = _materialization_evidence(
        group_id=f"pyramid|{width_key}", contract=contract, plan_sha=plan_sha
    )
    ready = evidence is not None
    return {
        "group_id": f"pyramid|{width_key}",
        "model": "pyramid",
        "width": values,
        "source_status": "ready" if ready else "materializable",
        "materialization_kind": materialization_kind,
        "source_evidence_kind": "materialization_evidence" if ready else "materialization_plan",
        "source_evidence_sha256": plan_sha,
        "materialization_evidence_path": evidence["path"] if evidence else None,
        "materialization_evidence_sha256": evidence["sha256"] if evidence else None,
        "source_contract": contract,
        "graph_features": predict_graph_features(surrogate, model="pyramid", width=values),
        "original60_label": source.get("label") if source is not None else None,
    }


def _codriving_group(
    width: Sequence[int],
    *,
    surrogate: GraphFeatureSurrogate,
    result_root: Path,
    model_root: Path,
) -> dict[str, Any]:
    values = [int(value) for value in width]
    width_key = "x".join(map(str, values))
    model_dir = model_root / width_key
    contract = {
        "model_dir": str(model_dir),
        "onnx_path": str(model_dir / f"resnet_multiscale_{width_key}_final_fp32.onnx"),
        "calibration_root": str(model_dir / "calibration_source"),
        "calibration_npz": str(model_dir / "stage3_calib_train_n16_float32.npz"),
        "calibration_summary": str(model_dir / "stage3_calib_train_n16_float32_summary.json"),
        "trt_calibration_dir": str(model_dir / "trt_calibration_npy"),
        "training_done_marker": str(model_dir / "stage5_training_complete.json"),
        "source_done_marker": str(result_root / "source_prep" / f"codriving_{width_key}.done"),
    }
    plan_sha = _sha({"kind": "codriving_prepare_train_export", "width": values, "contract": contract})
    evidence = _materialization_evidence(
        group_id=f"codriving|{width_key}", contract=contract, plan_sha=plan_sha
    )
    ready = evidence is not None
    return {
        "group_id": f"codriving|{width_key}",
        "model": "codriving",
        "width": values,
        "source_status": "ready" if ready else "materializable",
        "materialization_kind": "codriving_prepare_train_export",
        "source_evidence_kind": "materialization_evidence" if ready else "materialization_plan",
        "source_evidence_sha256": plan_sha,
        "materialization_evidence_path": evidence["path"] if evidence else None,
        "materialization_evidence_sha256": evidence["sha256"] if evidence else None,
        "source_contract": contract,
        "graph_features": predict_graph_features(surrogate, model="codriving", width=values),
    }


def build_source_registry(
    *,
    graph_features: Sequence[Mapping[str, Any]],
    pyramid_inventory: Sequence[Mapping[str, Any]],
    pyramid_widths: Sequence[Sequence[int]] | None = None,
    pyramid_model_root: str | Path | None = None,
    pyramid_base_source: Mapping[str, Any] | None = None,
    codriving_widths: Sequence[Sequence[int]],
    remote_result_root: str | Path,
    codriving_model_root: str | Path,
    seed: int = 20260717,
) -> dict[str, Any]:
    surrogate = fit_graph_feature_surrogate(graph_features, seed=seed)
    result_root = Path(remote_result_root)
    model_root = Path(codriving_model_root)
    pyramid_root = Path(pyramid_model_root) if pyramid_model_root is not None else None
    inventory_by_width: dict[tuple[int, ...], dict[str, Any]] = {}
    for item in pyramid_inventory:
        key = tuple(int(value) for value in item.get("width") or [])
        if len(key) != 3 or key in inventory_by_width:
            raise ValueError("Pyramid inventory widths must be unique triplets")
        inventory_by_width[key] = dict(item)
    selected_pyramid_widths = (
        [list(width) for width in pyramid_widths]
        if pyramid_widths is not None
        else [list(width) for width in inventory_by_width]
    )
    groups = [
        _pyramid_group(
            width,
            source=inventory_by_width.get(tuple(int(value) for value in width)),
            surrogate=surrogate,
            result_root=result_root,
            model_root=pyramid_root,
            base_source=pyramid_base_source,
        )
        for width in selected_pyramid_widths
    ]
    groups.extend(
        _codriving_group(
            width,
            surrogate=surrogate,
            result_root=result_root,
            model_root=model_root,
        )
        for width in codriving_widths
    )
    group_ids = [str(group["group_id"]) for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("source registry contains duplicate model-width groups")
    payload = {
        "schema_version": "stage5_candidate_source_registry_v1",
        "group_count": len(groups),
        "graph_feature_policy": "coldstart_width_conditioned_surrogate_v1",
        "graph_feature_surrogate_training_sha256": surrogate.training_sha256,
        "source_ready_group_count": sum(group["source_status"] == "ready" for group in groups),
        "source_materializable_group_count": sum(
            group["source_status"] == "materializable" for group in groups
        ),
        "groups": sorted(groups, key=lambda group: str(group["group_id"])),
    }
    payload["registry_sha256"] = _sha(payload)
    return payload


def validate_full_source_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless both target models expose the canonical 343 widths."""
    if registry.get("schema_version") != "stage5_candidate_source_registry_v1":
        raise ValueError("unexpected source registry schema")
    groups = [dict(group) for group in registry.get("groups") or []]
    expected = {tuple(width) for width in canonical_widths()}
    counts: dict[str, int] = {}
    for model in ("pyramid", "codriving"):
        model_groups = [group for group in groups if str(group.get("model")) == model]
        widths = {tuple(int(value) for value in group.get("width") or []) for group in model_groups}
        if len(model_groups) != 343 or widths != expected:
            raise ValueError(f"{model} source registry must cover exactly 343 canonical widths")
        counts[model] = len(model_groups)
    if len(groups) != 686:
        raise ValueError("full Stage5 source registry must contain exactly 686 model-width groups")
    return {
        "schema_version": "stage5_full_source_registry_audit_v1",
        "registry_sha256": registry.get("registry_sha256"),
        "group_count": len(groups),
        "group_count_by_model": counts,
        "genome_count_by_model": {model: count * 2 for model, count in counts.items()},
    }


__all__ = [
    "GraphFeatureSurrogate",
    "build_source_registry",
    "build_pyramid_checkpoint_inventory",
    "canonical_widths",
    "fit_graph_feature_surrogate",
    "predict_graph_features",
    "validate_full_source_registry",
]
