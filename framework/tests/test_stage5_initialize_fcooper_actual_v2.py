from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stage5_initialize_fcooper_actual_v2 import (
    relabel_dispatch_profile,
    resolve_fcooper_task,
    validate_formal_registry_provenance,
)
from framework.stage2.canonical_search_v3 import build_capability_profile

ROOT = Path(__file__).resolve().parents[2]


def test_formal_initializer_hard_gates_pilot_probe_and_recovery_evidence() -> None:
    source = (ROOT / "scripts/stage5_initialize_fcooper_actual_v2.py").read_text()

    assert "pilot_online_labels_loaded" in source
    assert "probe_metrics_loaded_as_labels" in source
    assert "numeric-gate-summary-json" in source
    assert "probe-isolation-audit-json" in source
    assert "scanner-execution-json" in source
    assert 'training_view_policy="initial_coldstart_only"' in source
    assert "S5-FCO-TRT-V2" in source


def test_formal_initializer_rejects_registry_content_from_pilot() -> None:
    registry = {
        "groups": [
            {
                "source_contract": {
                    "checkpoint": (
                        "/tmp/fcooper_workpackage_a_20260723/source/checkpoint.pth"
                    )
                }
            }
        ]
    }

    with pytest.raises(ValueError, match="pilot"):
        validate_formal_registry_provenance(registry)


def test_formal_initializer_resolves_one_fixed_tvm_profile_without_genome_backend() -> None:
    profile = build_capability_profile(
        capability_profile_id="h800-tvm-fcooper-v1",
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key="tvm_auto",
        features={"supports_fp16_tensorcore": 1.0},
    )

    task = resolve_fcooper_task(
        [profile],
        task_id="S5-FCO-TVM-V1",
        dispatch_key="tvm_auto",
    )

    assert task.task_id == "S5-FCO-TVM-V1"
    assert task.capability_profile["dispatch_key"] == "tvm_auto"


def test_formal_initializer_resolves_explicit_profile_among_same_dispatch() -> None:
    profiles = [
        build_capability_profile(
            capability_profile_id=profile_id,
            hardware_target="h800",
            compiler_fingerprint=character * 64,
            dispatch_key="tvm_auto",
            features={"supports_fp16_tensorcore": 1.0},
        )
        for profile_id, character in (
            ("h800-tvm-generic", "a"),
            ("h800-tvm-fcooper", "b"),
        )
    ]

    task = resolve_fcooper_task(
        profiles,
        task_id="S5-FCO-TVM-V1",
        dispatch_key="tvm_auto",
        capability_profile_id="h800-tvm-fcooper",
    )

    assert task.capability_profile["capability_profile_id"] == "h800-tvm-fcooper"


def test_relabels_coldstart_dispatch_to_current_target_profile() -> None:
    rows = [
        {"dispatch_key": "tvm_auto", "capability_profile_id": "old", "capability_digest": "x"},
        {"dispatch_key": "trt_engine", "capability_profile_id": "trt", "capability_digest": "y"},
    ]
    profile = {
        "dispatch_key": "tvm_auto",
        "capability_profile_id": "current",
        "capability_digest": "z",
    }

    relabeled = relabel_dispatch_profile(rows, profile)

    assert relabeled[0]["capability_profile_id"] == "current"
    assert relabeled[0]["capability_digest"] == "z"
    assert relabeled[1] == rows[1]
