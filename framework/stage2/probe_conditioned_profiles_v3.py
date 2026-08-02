"""Build probe-conditioned capability profiles from S1-Q and S1-P structural runs."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage2.s1_neutral_probes_v3 import PROBE_SPECS as NEUTRAL_PROBE_SPECS
from framework.stage2.s1_pruning_shape_probes_v3 import PROBE_SPECS as PRUNING_PROBE_SPECS


RUN_SCHEMA = "stage2_s1_structural_probe_run_v3"
RECORD_SCHEMA = "stage2_s1_structural_probe_record_v3"


def load_probe_run(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("probe run payload must be a JSON object")
    if str(payload.get("schema_version") or "") != RUN_SCHEMA:
        raise ValueError(f"unexpected probe run schema: {payload.get('schema_version')!r}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("probe run payload must contain non-empty records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("probe run records must be JSON objects")
        if str(record.get("schema_version") or "") != RECORD_SCHEMA:
            raise ValueError(f"unexpected probe record schema: {record.get('schema_version')!r}")
    return payload


def extract_compiler_fingerprint(run: Mapping[str, Any]) -> str:
    candidates: set[str] = set()
    top_provenance = run.get("provenance")
    if isinstance(top_provenance, Mapping):
        value = str(top_provenance.get("compiler_fingerprint") or "")
        if value:
            candidates.add(value)
        pipeline = top_provenance.get("pipeline")
        if isinstance(pipeline, Mapping):
            nested = str(pipeline.get("compiler_fingerprint") or "")
            if nested:
                candidates.add(nested)
    top_level = str(run.get("compiler_fingerprint") or "")
    if top_level:
        candidates.add(top_level)
    for record in run.get("records", []):
        provenance = record.get("provenance")
        if isinstance(provenance, Mapping):
            value = str(provenance.get("compiler_fingerprint") or "")
            if value:
                candidates.add(value)
    if not candidates:
        raise ValueError(
            "compiler fingerprint missing in probe run provenance; refuse to generate profile. "
            "re-collect the records with a real compiler fingerprint recorded at "
            "provenance.compiler_fingerprint (preferred) or per-record provenance.compiler_fingerprint "
            "from the actual compiler/build pipeline, then rerun."
        )
    if len(candidates) != 1:
        raise ValueError(f"compiler fingerprint mismatch within run provenance: {sorted(candidates)}")
    return next(iter(candidates))


def _successful(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [record for record in records if bool(record.get("build_success"))]


def _records_for_mode(records: Iterable[Mapping[str, Any]], q_mode: str) -> list[Mapping[str, Any]]:
    return [record for record in records if str(record.get("q_mode")) == q_mode]


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value <= 0.0:
        return None
    return float(numerator) / denominator_value


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _pstdev(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _rate_by_probe(
    records: Sequence[Mapping[str, Any]],
    *,
    probe_ids: Sequence[str],
    value_fn,
) -> list[float]:
    rates: list[float] = []
    by_probe = {probe_id: [] for probe_id in probe_ids}
    for record in records:
        probe_id = str(record.get("probe_id") or "")
        if probe_id in by_probe and bool(record.get("build_success")):
            value = value_fn(record)
            if value is not None:
                by_probe[probe_id].append(float(value))
    for probe_id in probe_ids:
        values = by_probe[probe_id]
        if values:
            rates.append(_mean(values))
    return rates


def aggregate_neutral_features(run: Mapping[str, Any]) -> dict[str, float | int]:
    records = [dict(record) for record in run["records"]]
    successful = _successful(records)
    fp16 = _records_for_mode(records, "fp16")
    int8 = _records_for_mode(records, "int8")
    successful_int8 = _successful(int8)

    propagation = [
        ratio
        for ratio in (
            _safe_ratio(record.get("int8_propagated_ops"), record.get("precision_eligible_ops"))
            for record in successful_int8
        )
        if ratio is not None
    ]
    qdq_fold = [
        ratio
        for ratio in (
            _safe_ratio(record.get("qdq_folded_pairs"), record.get("qdq_pairs"))
            for record in successful_int8
        )
        if ratio is not None
    ]
    reformat = [
        ratio
        for ratio in (_safe_ratio(record.get("reformat_ops"), record.get("total_ops")) for record in successful)
        if ratio is not None
    ]
    fusion = [
        ratio
        for ratio in (_safe_ratio(record.get("fused_ops"), record.get("fusible_ops")) for record in successful)
        if ratio is not None
    ]
    expected_pairs = len(NEUTRAL_PROBE_SPECS) * 2
    observed_pairs = {
        (str(record.get("probe_id") or ""), str(record.get("q_mode") or ""))
        for record in records
    }
    return {
        "s1q_probe_pair_coverage": float(len(observed_pairs) / expected_pairs),
        "s1q_build_success_coverage": float(len(successful) / len(records)),
        "s1q_fp16_build_success_coverage": float(len(_successful(fp16)) / len(fp16)) if fp16 else 0.0,
        "s1q_int8_build_success_coverage": float(len(successful_int8) / len(int8)) if int8 else 0.0,
        "s1q_int8_propagation_ratio_mean": _mean(propagation),
        "s1q_int8_propagation_observed_coverage": float(len(propagation) / len(successful_int8))
        if successful_int8
        else 0.0,
        "s1q_qdq_fold_ratio_mean": _mean(qdq_fold),
        "s1q_qdq_fold_observed_coverage": float(len(qdq_fold) / len(successful_int8))
        if successful_int8
        else 0.0,
        "s1q_reformat_rate_mean": _mean(reformat),
        "s1q_reformat_observed_coverage": float(len(reformat) / len(successful)) if successful else 0.0,
        "s1q_fusion_coverage_mean": _mean(fusion),
        "s1q_fusion_observed_coverage": float(len(fusion) / len(successful)) if successful else 0.0,
    }


def aggregate_pruning_features(run: Mapping[str, Any]) -> dict[str, float | int]:
    records = [dict(record) for record in run["records"]]
    successful = _successful(records)
    fp16 = _records_for_mode(records, "fp16")
    int8 = _records_for_mode(records, "int8")
    expected_pairs = len(PRUNING_PROBE_SPECS) * 2
    observed_pairs = {
        (str(record.get("probe_id") or ""), str(record.get("q_mode") or ""))
        for record in records
    }
    observed_probe_ids = {
        probe_id for probe_id, _ in observed_pairs if probe_id in PRUNING_PROBE_SPECS
    }
    probe_ids = list(PRUNING_PROBE_SPECS.keys())
    fp16_reformat_rates = _rate_by_probe(
        fp16,
        probe_ids=probe_ids,
        value_fn=lambda record: _safe_ratio(record.get("reformat_ops"), record.get("total_ops")),
    )
    int8_reformat_rates = _rate_by_probe(
        int8,
        probe_ids=probe_ids,
        value_fn=lambda record: _safe_ratio(record.get("reformat_ops"), record.get("total_ops")),
    )
    int8_propagation_rates = _rate_by_probe(
        int8,
        probe_ids=probe_ids,
        value_fn=lambda record: _safe_ratio(
            record.get("int8_propagated_ops"), record.get("precision_eligible_ops")
        ),
    )
    int8_qdq_fold_rates = _rate_by_probe(
        int8,
        probe_ids=probe_ids,
        value_fn=lambda record: _safe_ratio(record.get("qdq_folded_pairs"), record.get("qdq_pairs")),
    )
    fp16_build_rates = _rate_by_probe(
        fp16,
        probe_ids=probe_ids,
        value_fn=lambda record: 1.0 if bool(record.get("build_success")) else 0.0,
    )
    int8_build_rates = _rate_by_probe(
        int8,
        probe_ids=probe_ids,
        value_fn=lambda record: 1.0 if bool(record.get("build_success")) else 0.0,
    )
    return {
        "s1p_probe_pair_coverage": float(len(observed_pairs) / expected_pairs),
        "s1p_probe_group_count_observed": len(observed_probe_ids),
        "s1p_build_success_coverage": float(len(successful) / len(records)),
        "s1p_fp16_build_success_coverage": float(len(_successful(fp16)) / len(fp16)) if fp16 else 0.0,
        "s1p_int8_build_success_coverage": float(len(_successful(int8)) / len(int8)) if int8 else 0.0,
        "s1p_fp16_group_build_rate_stddev": _pstdev(fp16_build_rates),
        "s1p_int8_group_build_rate_stddev": _pstdev(int8_build_rates),
        "s1p_fp16_group_reformat_rate_stddev": _pstdev(fp16_reformat_rates),
        "s1p_int8_group_reformat_rate_stddev": _pstdev(int8_reformat_rates),
        "s1p_int8_group_propagation_rate_stddev": _pstdev(int8_propagation_rates),
        "s1p_int8_group_qdq_fold_rate_stddev": _pstdev(int8_qdq_fold_rates),
    }


def build_probe_conditioned_profile(
    *,
    capability_profile_id: str,
    hardware_target: str,
    dispatch_key: str,
    neutral_run: Mapping[str, Any],
    pruning_run: Mapping[str, Any],
) -> dict[str, Any]:
    neutral_backend = str(neutral_run.get("backend_runner") or "")
    pruning_backend = str(pruning_run.get("backend_runner") or "")
    if neutral_backend != pruning_backend:
        raise ValueError(f"backend_runner mismatch between runs: {neutral_backend!r} vs {pruning_backend!r}")
    neutral_fingerprint = extract_compiler_fingerprint(neutral_run)
    pruning_fingerprint = extract_compiler_fingerprint(pruning_run)
    if neutral_fingerprint != pruning_fingerprint:
        raise ValueError(
            f"compiler fingerprint mismatch between S1-Q and S1-P runs: "
            f"{neutral_fingerprint!r} vs {pruning_fingerprint!r}"
        )
    features = {
        **aggregate_neutral_features(neutral_run),
        **aggregate_pruning_features(pruning_run),
    }
    return build_capability_profile(
        capability_profile_id=capability_profile_id,
        hardware_target=hardware_target,
        compiler_fingerprint=neutral_fingerprint,
        dispatch_key=dispatch_key,
        features=features,
    )


def build_profile_from_run_paths(
    *,
    capability_profile_id: str,
    hardware_target: str,
    dispatch_key: str,
    neutral_run_path: str | Path,
    pruning_run_path: str | Path,
) -> dict[str, Any]:
    return build_probe_conditioned_profile(
        capability_profile_id=capability_profile_id,
        hardware_target=hardware_target,
        dispatch_key=dispatch_key,
        neutral_run=load_probe_run(neutral_run_path),
        pruning_run=load_probe_run(pruning_run_path),
    )


__all__ = [
    "aggregate_neutral_features",
    "aggregate_pruning_features",
    "build_probe_conditioned_profile",
    "build_profile_from_run_paths",
    "extract_compiler_fingerprint",
    "load_probe_run",
]
