"""Canonical Stage2 search contracts for capability-conditioned FP16/INT8 search."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "stage2_canonical_search_v3"
CAPABILITY_PROFILE_SCHEMA = "stage2_capability_profile_v3"
ACTIVE_MANIFEST_SCHEMA = "stage2_active_manifest_v3"
ACTIVE_Q_MODES = ("fp16", "int8")
FORBIDDEN_TARGET_DERIVED_CAPABILITY_FEATURES = {
    "int8_over_fp16_latency_ratio",
    "int8_over_fp16_energy_ratio",
}
FORBIDDEN_TARGET_METRIC_TOKENS = ("latency", "energy", "ap70", "map", "accuracy")


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_numeric_features(features: Mapping[str, Any]) -> None:
    if not features:
        raise ValueError("capability features must not be empty")
    for name, value in features.items():
        if not str(name):
            raise ValueError("capability feature names must be non-empty")
        if any(token in str(name).lower() for token in FORBIDDEN_TARGET_METRIC_TOKENS):
            raise ValueError(f"target-derived performance feature is forbidden: {name}")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"capability feature {name} must be numeric or null")
        if not math.isfinite(float(value)):
            raise ValueError(f"capability feature {name} must be finite")
    leaked = sorted(FORBIDDEN_TARGET_DERIVED_CAPABILITY_FEATURES & set(features))
    if leaked:
        raise ValueError(
            "target-derived performance ratios are forbidden in capability features: "
            + ",".join(leaked)
        )


def _capability_identity(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": profile["schema_version"],
        "hardware_target": profile["hardware_target"],
        "compiler_fingerprint": profile["compiler_fingerprint"],
        "features": profile["features"],
    }


def compute_capability_digest(profile: Mapping[str, Any]) -> str:
    """Hash measured capability identity while excluding dispatch-only metadata."""

    return _sha256_payload(_capability_identity(profile))


def validate_capability_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "capability_profile_id",
        "hardware_target",
        "compiler_fingerprint",
        "dispatch_key",
        "features",
        "capability_digest",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"capability profile missing fields: {missing}")
    if profile["schema_version"] != CAPABILITY_PROFILE_SCHEMA:
        raise ValueError("unexpected capability profile schema")
    for field in ("capability_profile_id", "hardware_target", "dispatch_key"):
        if not str(profile[field] or ""):
            raise ValueError(f"{field} must be non-empty")
    if not _is_sha256(profile["compiler_fingerprint"]):
        raise ValueError("compiler_fingerprint must be a SHA256 digest")
    features = profile["features"]
    if not isinstance(features, Mapping):
        raise ValueError("capability features must be a mapping")
    _validate_numeric_features(features)
    if profile["capability_digest"] != compute_capability_digest(profile):
        raise ValueError("capability_digest mismatch")
    return copy.deepcopy(dict(profile))


def build_capability_profile(
    *,
    capability_profile_id: str,
    hardware_target: str,
    compiler_fingerprint: str,
    dispatch_key: str,
    features: Mapping[str, float | int | None],
) -> dict[str, Any]:
    profile = {
        "schema_version": CAPABILITY_PROFILE_SCHEMA,
        "capability_profile_id": capability_profile_id,
        "hardware_target": hardware_target,
        "compiler_fingerprint": compiler_fingerprint,
        "dispatch_key": dispatch_key,
        "features": dict(features),
    }
    profile["capability_digest"] = compute_capability_digest(profile)
    return validate_capability_profile(profile)


def parse_width(width: str | Sequence[int]) -> tuple[int, int, int]:
    parts = width.replace("x", ",").split(",") if isinstance(width, str) else list(width)
    values = tuple(int(part) for part in parts)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"width must contain three positive integers: {width!r}")
    return values


def width_key(width: str | Sequence[int]) -> str:
    return "x".join(str(value) for value in parse_width(width))


def build_active_manifest(
    *,
    widths_by_model: Mapping[str, Sequence[str | Sequence[int]]],
    capability_profiles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    profiles = [validate_capability_profile(profile) for profile in capability_profiles]
    if len({profile["capability_profile_id"] for profile in profiles}) != len(profiles):
        raise ValueError("capability_profile_id values must be unique")
    jobs: list[dict[str, Any]] = []
    for model, widths in widths_by_model.items():
        if not str(model):
            raise ValueError("model names must be non-empty")
        normalized_widths = [width_key(width) for width in widths]
        if len(set(normalized_widths)) != len(normalized_widths):
            raise ValueError(f"duplicate widths for model {model}")
        for width in normalized_widths:
            values = list(parse_width(width))
            group_id = f"{model}|{width}"
            for q_mode in ACTIVE_Q_MODES:
                for profile in profiles:
                    profile_id = profile["capability_profile_id"]
                    jobs.append(
                        {
                            "job_id": f"{group_id}|q={q_mode}|profile={profile_id}",
                            "group_id": group_id,
                            "model": model,
                            "width": values,
                            "width_key": width,
                            "q_mode": q_mode,
                            "mixed_policy_id": "none",
                            "strategy_id": f"q={q_mode}",
                            "capability_profile_id": profile_id,
                            "capability_digest": profile["capability_digest"],
                            "dispatch_key": profile["dispatch_key"],
                        }
                    )
    return {
        "schema_version": ACTIVE_MANIFEST_SCHEMA,
        "canonical_schema_version": SCHEMA_VERSION,
        "genome_schema": ["p1", "p2", "p3", "q_mode"],
        "q_modes": list(ACTIVE_Q_MODES),
        "mixed_policy_activation": "inactive_pending_joint_gold_gate",
        "capability_profiles": profiles,
        "jobs": jobs,
    }


def assign_grouped_split(
    jobs: Sequence[Mapping[str, Any]],
    *,
    holdout_group_count: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    groups = sorted({str(job["group_id"]) for job in jobs})
    if holdout_group_count <= 0 or holdout_group_count >= len(groups):
        raise ValueError("holdout_group_count must leave non-empty train and holdout groups")
    ranked = sorted(groups, key=lambda group: _sha256_payload({"seed": seed, "group_id": group}))
    holdout_groups = set(ranked[:holdout_group_count])
    train = [copy.deepcopy(dict(job)) for job in jobs if str(job["group_id"]) not in holdout_groups]
    holdout = [copy.deepcopy(dict(job)) for job in jobs if str(job["group_id"]) in holdout_groups]
    return {"train": train, "holdout": holdout}


def route_historical_180(
    rows: Iterable[Mapping[str, Any]],
    *,
    disagreement_ratio: float,
) -> dict[str, list[dict[str, Any]]]:
    if disagreement_ratio <= 1.0:
        raise ValueError("disagreement_ratio must be greater than 1")
    routed = {
        "width_ap_prior": [],
        "disagreement_probe": [],
        "historical_ablation": [],
        "backend_gold": [],
        "final_frontier": [],
    }
    for source in rows:
        row = copy.deepcopy(dict(source))
        if bool(row.get("can_use_as_cold_start_prior")):
            routed["width_ap_prior"] = [*routed["width_ap_prior"], row]
            routed["historical_ablation"] = [*routed["historical_ablation"], row]
        automatic = row.get("routeb_int8_latency_ms_if_measured")
        historical = row.get("phase1_old_int8tc_latency_ms_if_available")
        if automatic is not None and historical is not None:
            low, high = sorted((float(automatic), float(historical)))
            if low > 0.0 and high / low >= disagreement_ratio:
                routed["disagreement_probe"] = [*routed["disagreement_probe"], row]
    return routed


__all__ = [
    "ACTIVE_MANIFEST_SCHEMA",
    "ACTIVE_Q_MODES",
    "CAPABILITY_PROFILE_SCHEMA",
    "assign_grouped_split",
    "build_active_manifest",
    "build_capability_profile",
    "compute_capability_digest",
    "parse_width",
    "route_historical_180",
    "validate_capability_profile",
    "width_key",
]
