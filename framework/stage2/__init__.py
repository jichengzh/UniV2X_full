"""Stage2 integration helpers with lightweight LUT imports."""

from __future__ import annotations

CONTRACT_EXPORTS: list[str] = []

try:
    from .contracts import (
        GateDecision,
        Stage2EvidenceDelta,
        Stage2EvidenceRecord,
        Stage2Input,
        Stage2Output,
        apply_stage1_gate,
        build_stage2_output,
    )

    CONTRACT_EXPORTS = [
        "GateDecision",
        "Stage2EvidenceDelta",
        "Stage2EvidenceRecord",
        "Stage2Input",
        "Stage2Output",
        "apply_stage1_gate",
        "build_stage2_output",
    ]
except ModuleNotFoundError as exc:
    missing_name = exc.name or ""
    if missing_name != "yaml" and "yaml" not in str(exc):
        raise

from .evidence_registry import (
    DSQueryResult,
    EvidenceRegistryError,
    EvidenceSource,
    Stage2CostInputs,
    Stage2EvidenceRegistry,
    build_default_registry,
    build_default_registry_dict,
    write_default_registry,
)
from .lut_productization import (
    LutProductizationError,
    ap_anchor_row,
    append_jsonl,
    coverage_from_rows,
    energy_claim_allowed_from_rows,
    energy_lut_row,
    job_plan_row,
    latency_lut_row,
    latest_job_status,
    next_queued_jobs,
    parse_width_csv,
    read_jsonl,
    run_json_command,
    stable_config_id,
    validate_lut_row,
    write_jsonl,
)

__all__ = [
    *CONTRACT_EXPORTS,
    "DSQueryResult",
    "EvidenceRegistryError",
    "EvidenceSource",
    "Stage2CostInputs",
    "Stage2EvidenceRegistry",
    "LutProductizationError",
    "ap_anchor_row",
    "append_jsonl",
    "coverage_from_rows",
    "energy_claim_allowed_from_rows",
    "energy_lut_row",
    "job_plan_row",
    "latency_lut_row",
    "latest_job_status",
    "next_queued_jobs",
    "parse_width_csv",
    "read_jsonl",
    "run_json_command",
    "stable_config_id",
    "validate_lut_row",
    "write_jsonl",
    "build_default_registry",
    "build_default_registry_dict",
    "write_default_registry",
]
