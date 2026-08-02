"""Build the F-Cooper Stage5 source registry from the Stage1 scanner manifest."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from framework.stage1_bridge import SpaceSpec
from framework.stage5.genome_contract_v1 import canonical_group_id


WIDTH_SCHEMA = (
    "backbone.s0",
    "backbone.s1",
    "backbone.s2",
    "neck.deblock",
    "neck.output",
)
MATERIALIZATION_KIND = "fcooper_scanner_materialize_export"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _scanner_feature_prior(width: tuple[int, ...], base: tuple[int, ...]) -> dict[str, Any]:
    ratios = [value / original for value, original in zip(width, base)]
    return {
        "graph_feature_provenance": "stage1_scanner_structure_prior_v1",
        "conv_count": 31.0,
        "normalized_channel_sum": float(sum(ratios)),
        "normalized_channel_product": float(
            ratios[0] * ratios[1] * ratios[2] * ratios[3] * ratios[4]
        ),
        "backbone_width_sum": float(sum(width[:3])),
        "neck_deblock_width": float(width[3]),
        "neck_output_width": float(width[4]),
    }


def _axis_widths(spec: SpaceSpec) -> tuple[dict[str, list[int]], dict[str, int]]:
    knobs = {knob.search_group_id: knob for knob in spec.knobs}
    expected = {"backbone.s0", "backbone.s1", "backbone.s2", "neck"}
    if set(knobs) != expected:
        raise ValueError(f"F-Cooper scanner groups drift: {sorted(knobs)}")
    axes = {
        name: knobs[name].legal_widths()
        for name in ("backbone.s0", "backbone.s1", "backbone.s2")
    }
    bases = {
        name: knobs[name].base_width
        for name in ("backbone.s0", "backbone.s1", "backbone.s2")
    }
    neck_widths = sorted(set(knobs["neck"].cur_widths))
    if len(neck_widths) != 2:
        raise ValueError(
            "F-Cooper neck dependency scan must expose two interface widths"
        )
    round_to = max(1, knobs["neck"].round_to)
    max_rate = knobs["neck"].max_rate
    for name, base in (
        ("neck.deblock", neck_widths[0]),
        ("neck.output", neck_widths[1]),
    ):
        floor = max(
            round_to,
            int(math.ceil(base * (1.0 - max_rate) / round_to)) * round_to,
        )
        axes[name] = list(range(floor, base + 1, round_to))
        bases[name] = base
    return axes, bases


def build_fcooper_source_registry(
    partition_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict[str, Any]:
    partition = Path(partition_path).resolve()
    spec = SpaceSpec.from_manifest(partition)
    if spec.model != "fcooper":
        raise ValueError(f"expected F-Cooper partition, got {spec.model}")
    axis_widths, axis_bases = _axis_widths(spec)
    axes = [axis_widths[name] for name in WIDTH_SCHEMA]
    if any(not values for values in axes):
        raise ValueError("F-Cooper scanner emitted an empty structure axis")
    base = tuple(axis_bases[name] for name in WIDTH_SCHEMA)
    root = Path(artifact_root)
    groups = []
    for values in itertools.product(*axes):
        width = tuple(int(value) for value in values)
        group_id = canonical_group_id("fcooper", width, WIDTH_SCHEMA)
        tag = "x".join(map(str, width))
        materialized = root / "sources" / tag
        contract = {
            "partition_manifest": str(partition),
            "partition_manifest_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
            "checkpoint_path": str(materialized / "net_epoch_bestval_at23.pth"),
            "config_path": str(materialized / "config.yaml"),
            "onnx_path": str(materialized / f"fcooper_dense_{tag}.onnx"),
            "calibration_root": str(materialized / "calibration"),
            "calibration_summary": str(materialized / "calibration_summary.json"),
            "trt_calibration_dir": str(materialized / "calibration"),
            "source_done_marker": str(materialized / "source.ready"),
            "input_shape": [5, 64, 512, 512],
            "engine_agent_batch": 5,
            "optimized_scope": "post_scatter_backbone_shrinker",
        }
        plan = {"kind": MATERIALIZATION_KIND, "width": list(width), "contract": contract}
        groups.append(
            {
                "group_id": group_id,
                "model": "fcooper",
                "width": list(width),
                "width_schema": list(WIDTH_SCHEMA),
                "structure_widths": dict(zip(WIDTH_SCHEMA, width)),
                "source_status": "materializable",
                "materialization_kind": MATERIALIZATION_KIND,
                "source_evidence_kind": "materialization_plan",
                "source_contract": contract,
                "source_evidence_sha256": _sha(plan),
                "graph_features": {
                    "group_id": group_id,
                    "model": "fcooper",
                    "width": list(width),
                    **_scanner_feature_prior(width, base),
                },
            }
        )
    groups.sort(key=lambda item: item["group_id"])
    return {
        "schema_version": "stage5_candidate_source_registry_v1",
        "model": "fcooper",
        "partition_manifest": str(partition),
        "partition_manifest_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
        "width_schema": list(WIDTH_SCHEMA),
        "structure_group_count": len(WIDTH_SCHEMA),
        "axis_legal_widths": axis_widths,
        "structure_candidate_count": len(groups),
        "groups": groups,
    }
