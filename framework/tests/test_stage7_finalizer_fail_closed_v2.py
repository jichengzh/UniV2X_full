from __future__ import annotations

import json
from pathlib import Path

import pytest

from framework.stage7 import ablation_statistics_v2 as statistics
from framework.stage7 import core_ablation_v2 as core
from framework.stage7 import core_cache_v2
from framework.stage7 import formal_evidence_v2 as evidence
from framework.stage7 import paper_outputs_v2 as outputs
from scripts import stage7_finalize_core_ablation_v2 as finalizer


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_modular_finalizer_facade_preserves_rendering_and_public_error() -> None:
    rows = [{"variant": "full", "values": [1, 2], "ready": True}]
    assert finalizer.FinalizationError is evidence.FinalizationError
    assert finalizer._json_text(rows) == outputs.json_text(rows)
    assert finalizer._json_text(rows) == (
        "[\n"
        "  {\n"
        '    "ready": true,\n'
        '    "values": [\n'
        "      1,\n"
        "      2\n"
        "    ],\n"
        '    "variant": "full"\n'
        "  }\n"
        "]\n"
    )
    assert finalizer._csv_text(rows) == outputs.csv_text(rows)
    assert finalizer._csv_text(rows) == (
        'variant,values,ready\r\nfull,"[1, 2]",True\r\n'
    )
    assert finalizer._describe([1.0, 2.0, 3.0]) == statistics.describe([1.0, 2.0, 3.0])


def test_non_final_dry_run_status_can_never_claim_readiness(tmp_path: Path) -> None:
    output = tmp_path / "dry-run-finalizer"
    result = finalizer.emit_non_final_dry_run(
        output,
        {
            "trajectory_count": 12,
            "round0_request_count": 4,
            "synthetic_fixture_count": 4,
        },
    )

    assert result == {
        "paper_ready": False,
        "core_ablation_ready": False,
        "full_gear_s7_ready": False,
        "scanner_component_status": "deferred_important_fix",
        "formal_v2_gpu_jobs_launched": 0,
        "blocking_reason": finalizer.INTEGRATION_BLOCKING_REASON,
    }
    written = json.loads((output / "status/finalization_status.json").read_text())
    assert written["dry_run_synthetic_non_measurement"] is True
    assert written["eligible_for_formal_finalization"] is False


def test_formal_finalizer_source_never_calls_legacy_private_validator() -> None:
    source = Path(finalizer.__file__).read_text(encoding="utf-8")
    assert "core_cache_v2._validate_exact_dimensions" not in source
    assert "validate_formal_exact_dimensions" in source
    assert "validate_committed_barrier" in source


def test_audit_source_jsonl_rejects_malformed_or_empty_content(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"schema_version":"ok"}\nnot-json\n')
    with pytest.raises(finalizer.FinalizationError, match="unavailable or invalid"):
        finalizer._read_source_rows(malformed, "audit source")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(finalizer.FinalizationError, match="one or more object rows"):
        finalizer._read_source_rows(empty, "audit source")


def test_schema_only_audit_sources_cannot_self_sign_paper_ready(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache_after_round.json"
    cache = {
        "schema_version": core_cache_v2.CACHE_SCHEMA,
        "entries": {},
        "lineage": [],
    }
    _write_json(cache_path, cache)
    with pytest.raises(finalizer.FinalizationError, match="authenticated entries"):
        finalizer._validate_semantic_audit_source(
            tmp_path,
            kind="cache",
            path=cache_path,
            schema=core_cache_v2.CACHE_SCHEMA,
            rows=[cache],
        )

    deploy_path = tmp_path / "deploy.json"
    deploy = {"schema_version": "stage7_deploy_manifest_v1"}
    _write_json(deploy_path, deploy)
    with pytest.raises(finalizer.FinalizationError, match="incomplete"):
        finalizer._validate_semantic_audit_source(
            tmp_path,
            kind="deployment",
            path=deploy_path,
            schema="stage7_deploy_manifest_v1",
            rows=[deploy],
        )


def test_missing_formal_root_raises_public_finalization_error(tmp_path: Path) -> None:
    with pytest.raises(finalizer.FinalizationError, match="formal input root"):
        finalizer.finalize_v2(tmp_path / "missing", tmp_path / "output")


def test_malformed_event_identity_raises_public_finalization_error() -> None:
    events = []
    for variant in core.CORE_VARIANTS:
        for seed in (20260718, 20260719, 20260720):
            for round_index in range(4):
                for row_index in range(4):
                    events.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "round_index": round_index,
                            "event_index": round_index * 4 + row_index,
                            "candidate_id": (
                                f"{variant}-{seed}-{round_index}-{row_index}"
                            ),
                            "terminal_evidence_kind": "actual_v3_success",
                            "cache_disposition": "miss",
                            "latency_ms": 1.0,
                            "energy_j": 1.0,
                            "ap30": 0.8,
                            "ap50": 0.7,
                            "ap70": 0.6,
                        }
                    )
    events[0] = {**events[0], "seed": "not-an-integer"}
    payload = {
        "schema_version": finalizer.FORMAL_EVIDENCE_SCHEMA,
        "events": events,
        "trajectory_count": 12,
        "selected_event_count": 192,
        "miss_count": 192,
        "actual_v3_miss_evidence_count": 192,
        "formal_v2_gpu_jobs_launched": 192,
        "silent_surrogate_fallback_count": 0,
        "ordered_pre_scan_sha256": core.EXPECTED_ORDERED_PRE_SCAN_SHA256,
        "contract_sha256": "a" * 64,
        "audits": {name: True for name in finalizer.REQUIRED_AUDIT_FLAGS},
        "initial_gold176_rows": [{"row_id": "gold"}],
        "hv_reference": [10.0, 8.0, 0.0],
    }

    with pytest.raises(finalizer.FinalizationError, match="event identity"):
        finalizer._validate_ready_evidence(payload)
