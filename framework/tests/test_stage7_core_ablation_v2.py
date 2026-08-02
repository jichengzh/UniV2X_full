from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from framework.stage7 import core_ablation_v2 as core
from framework.stage7.core_ablation_v2 import (
    CORE_VARIANTS,
    EXPECTED_ADMISSION_SHA256,
    EXPECTED_BLOCKER_SHA256,
    EXPECTED_ORDERED_PRE_SCAN_SHA256,
    EXPECTED_PRE_SCAN_REGISTRY_SHA256,
    EXPECTED_SOURCE_REGISTRY_SHA256,
    build_v2_contract,
    validate_v2_contract,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def v1_root(tmp_path: Path) -> Path:
    root = tmp_path / "v1"
    _write_json(
        root / "status" / "without_capability_scan_blocked.json",
        {
            "schema_version": "stage7_variant_blocker_v1",
            "variant": "without_capability_scan",
            "status": "blocked_missing_candidate_level_scanner",
            "selected_events_consumed": 0,
            "gpu_jobs_launched": 0,
        },
    )
    _write_json(
        root / "audits" / "scanner_admission_expanded_v1.json",
        {
            "schema_version": "stage7_scanner_admission_expanded_v1",
            "status": "blocked_scanner_admission_failed",
            "admission_passed": False,
            "false_positive_unique_count": 3,
        },
    )
    return root


@pytest.fixture
def pre_scan_rows() -> list[dict[str, object]]:
    return [
        {"row_id": f"pyramid|candidate-{index:03d}", "width": [16, 32, 64]}
        for index in range(686)
    ]


@pytest.fixture
def local_formal_fixture(monkeypatch: pytest.MonkeyPatch, v1_root: Path, pre_scan_rows: list[dict[str, object]]) -> None:
    """Make isolated fixture bytes act as the formal immutable source for unit tests."""
    monkeypatch.setattr(
        core,
        "EXPECTED_BLOCKER_SHA256",
        hashlib.sha256((v1_root / "status" / "without_capability_scan_blocked.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        core,
        "EXPECTED_ADMISSION_SHA256",
        hashlib.sha256((v1_root / "audits" / "scanner_admission_expanded_v1.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(core, "EXPECTED_ORDERED_PRE_SCAN_SHA256", core.canonical_sha256(pre_scan_rows))


def test_frozen_formal_sha_constants_are_exact() -> None:
    assert EXPECTED_SOURCE_REGISTRY_SHA256 == "033cb9e38af39009f9919233f03c5ff77894651f6192d292dee501d974493b51"
    assert EXPECTED_PRE_SCAN_REGISTRY_SHA256 == "3e584a4d06afe3ca0ba6cedce9e0a99a20aafae8b1ce7ab62b4fd00da07ac4b8"
    assert EXPECTED_ORDERED_PRE_SCAN_SHA256 == "3294a3e66671af2778bffd4d719ab9f45963d948cabdfb21315012274daf96f4"
    assert EXPECTED_BLOCKER_SHA256 == "20cadfee33e2cbbffd3e2c58ececd89282c3e3733ad5924dadc885b0f9efc5da"
    assert EXPECTED_ADMISSION_SHA256 == "06761a3fdba4a945b9323941924d332e05b9f7a20321009a193696cce81ad170"
    assert [record["sha256"] for record in core.FROZEN_ACTUAL_V3_EXECUTORS] == [
        "132321077fc1a308dc43fe3e5af3f6d74725088fb526e05e57bd31530ed40170",
        "309a7f3889be01655702d0c97217c32fbbdc4ac7b3cb873937cb89f0dd059eb0",
        "fbbd295cc881ebac6db50ee3764eb667bc921168a1981f96ec7c9d3fcb440187",
        "7d3694a82cee4e9f9265471fd32371d1012279a71a6c150bf8443240b870df35",
        "6d373c312e236222d7630d91deb8cb122d5c928b464e5d1b60f7070ea33db2e2",
    ]


def test_validate_formal_inputs_requires_exact_path_sha_records() -> None:
    valid = {
        key: {
            "path": f"/formal/{key}.json",
            "sha256": hashlib.sha256(key.encode()).hexdigest(),
        }
        for key in core.REQUIRED_IMMUTABLE_INPUT_KEYS
    }
    valid["source_registry"]["sha256"] = core.EXPECTED_SOURCE_REGISTRY_SHA256

    assert core.validate_immutable_inputs(valid) == valid

    missing = copy.deepcopy(valid)
    missing.pop("gold176")
    with pytest.raises(ValueError, match="immutable input schema"):
        core.validate_immutable_inputs(missing)
    relative = copy.deepcopy(valid)
    relative["gold176"]["path"] = "relative.json"
    with pytest.raises(ValueError, match="absolute"):
        core.validate_immutable_inputs(relative)
    bad_sha = copy.deepcopy(valid)
    bad_sha["graph_features"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA"):
        core.validate_immutable_inputs(bad_sha)


def test_v2_contract_has_exactly_four_pre_scan_variants(
    v1_root: Path, pre_scan_rows: list[dict[str, object]], local_formal_fixture: None,
) -> None:
    contract = build_v2_contract(pre_scan_rows, v1_root)

    assert tuple(contract["variants"]) == CORE_VARIANTS
    assert "without_capability_scan" not in contract["variants"]
    assert {row["candidate_pool"] for row in contract["variant_contracts"]} == {"pre_scan"}
    assert {row["scanner"] for row in contract["variant_contracts"]} == {"deferred"}
    assert {row["scanner_claim_allowed"] for row in contract["variant_contracts"]} == {False}
    assert contract["ordered_pre_scan_sha256"] == core.EXPECTED_ORDERED_PRE_SCAN_SHA256
    assert all(row["ordered_pre_scan_sha256"] == contract["ordered_pre_scan_sha256"] for row in contract["variant_contracts"])


def test_v2_contract_binds_exact_v1_failure_evidence(
    v1_root: Path, pre_scan_rows: list[dict[str, object]], local_formal_fixture: None,
) -> None:
    contract = build_v2_contract(pre_scan_rows, v1_root)

    assert contract["v1_blocker"]["sha256"] == core.EXPECTED_BLOCKER_SHA256
    assert contract["v1_expanded_admission"]["sha256"] == core.EXPECTED_ADMISSION_SHA256
    assert contract["v1_blocker"]["status"] == "blocked_missing_candidate_level_scanner"
    assert contract["v1_expanded_admission"]["admission_passed"] is False
    assert contract["v1_expanded_admission"]["false_positive_unique_count"] == 3


def test_v2_contract_fails_closed_on_any_sha_or_status_drift(
    v1_root: Path, pre_scan_rows: list[dict[str, object]], local_formal_fixture: None,
) -> None:
    blocker_path = v1_root / "status" / "without_capability_scan_blocked.json"
    blocker_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="v1 blocker SHA drift"):
        build_v2_contract(pre_scan_rows, v1_root)


def test_built_v2_contract_is_recursively_immutable(
    v1_root: Path, pre_scan_rows: list[dict[str, object]], local_formal_fixture: None,
) -> None:
    contract = build_v2_contract(pre_scan_rows, v1_root)

    with pytest.raises(TypeError):
        contract["scanner_deferred"] = False
    with pytest.raises(TypeError):
        contract["variants"][0] = "without_capability_scan"
    with pytest.raises(TypeError):
        contract["variant_contracts"][0]["scanner"] = "enabled"
    with pytest.raises(TypeError):
        contract["v1_blocker"]["status"] = "not_blocked"
    assert validate_v2_contract(contract)["contract_sha256"] == contract["contract_sha256"]


def test_validate_v2_contract_fails_closed_on_variant_or_contract_sha_drift(
    v1_root: Path, pre_scan_rows: list[dict[str, object]], local_formal_fixture: None,
) -> None:
    contract = build_v2_contract(pre_scan_rows, v1_root)
    tampered_variant = json.loads(json.dumps(contract))
    tampered_variant["variant_contracts"][0]["scanner_claim_allowed"] = True
    with pytest.raises(ValueError, match="scanner claim"):
        validate_v2_contract(tampered_variant)

    tampered_sha = json.loads(json.dumps(contract))
    tampered_sha["contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="contract SHA drift"):
        validate_v2_contract(tampered_sha)


@pytest.mark.parametrize("unexpected_field", ("scanner_active", "unknown_field"))
def test_validate_v2_contract_rejects_unknown_top_level_fields_after_rehash(
    unexpected_field: str,
    v1_root: Path,
    pre_scan_rows: list[dict[str, object]],
    local_formal_fixture: None,
) -> None:
    contract = json.loads(json.dumps(build_v2_contract(pre_scan_rows, v1_root)))
    contract[unexpected_field] = True
    unsigned = copy.deepcopy(contract)
    unsigned.pop("contract_sha256")
    contract["contract_sha256"] = core.canonical_sha256(unsigned)

    with pytest.raises(ValueError, match="exact schema"):
        validate_v2_contract(contract)
