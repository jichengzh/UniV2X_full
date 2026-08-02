from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage7 import core_ablation_v2 as core
from framework.stage7 import deployment_bundle_v2 as deployment
from framework.stage7 import search_policy_v1 as policy
from framework.stage7.core_ablation_v2 import CORE_VARIANTS
from framework.stage7.online_component_ablation_v1 import SEEDS
from scripts import stage7_prepare_core_ablation_v2 as prepare


EXTERNAL_RELEASE_PIN = "a" * 64
EXTERNAL_MANIFEST_FILE_PIN = "b" * 64


def _prepare(**kwargs: object) -> dict[str, object]:
    return prepare.prepare_v2_root(
        expected_release_sha256=EXTERNAL_RELEASE_PIN,
        expected_manifest_file_sha256=EXTERNAL_MANIFEST_FILE_PIN,
        **kwargs,
    )


def _initialize(output_root: Path, **kwargs: object) -> dict[str, object]:
    return prepare.initialize_v2_trajectories(
        output_root,
        expected_release_sha256=EXTERNAL_RELEASE_PIN,
        expected_manifest_file_sha256=EXTERNAL_MANIFEST_FILE_PIN,
        **kwargs,
    )


def test_prepare_writer_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = importlib.import_module("framework.stage7.prepare_state_v2")
    calls: list[int] = []
    monkeypatch.setattr(state.os, "fsync", lambda descriptor: calls.append(descriptor))
    state._write_frozen_json(tmp_path / "contracts" / "artifact.json", {"ok": True})
    assert len(calls) == 2


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _process_identity(
    *,
    pid: int = 4242,
    username: str = "formal-owner",
    uid: int = 1001,
    start_time_ticks: int = 987654,
    executable: str = "/usr/bin/python3",
    cmdline: tuple[str, ...] = ("python3", "stage7-orchestrator.py"),
) -> dict[str, object]:
    return {
        "pid": pid,
        "uid": uid,
        "username": username,
        "start_time_ticks": start_time_ticks,
        "executable": executable,
        "cmdline": list(cmdline),
    }


def test_prepare_facade_modules_and_public_compatibility_surface_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    formal = importlib.import_module("framework.stage7.formal_inputs_v2")
    admission = importlib.import_module("framework.stage7.executor_admission_v2")
    state = importlib.import_module("framework.stage7.prepare_state_v2")

    assert prepare.FORMAL_INPUT_IDENTITIES == formal.FORMAL_INPUT_IDENTITIES
    assert (
        prepare.FORMAL_EMBEDDED_SOURCE_IDENTITIES
        == formal.FORMAL_EMBEDDED_SOURCE_IDENTITIES
    )
    assert prepare.PRIMITIVE_PATHS == formal.PRIMITIVE_PATHS
    assert prepare.PROTOCOL_PRIMITIVES == formal.PROTOCOL_PRIMITIVES
    assert prepare.prepare_v2_root is not state.prepare_v2_root
    assert prepare.initialize_v2_trajectories is not state.initialize_v2_trajectories
    assert (
        prepare.verify_frozen_actual_v3_executors
        is not admission.verify_frozen_actual_v3_executors
    )
    monkeypatch.setattr(sys, "argv", ["-"])
    assert hashlib.sha256(
        prepare.build_parser().format_help().encode()
    ).hexdigest() == (
        "fe90a7e4c9b9c45f7f1779949f41424f13a9d197dd73a2498ad7d47e010d30b9"
    )


@pytest.mark.parametrize(
    ("kwargs", "label"),
    (
        (
            {"expected_manifest_file_sha256": EXTERNAL_MANIFEST_FILE_PIN},
            "release",
        ),
        ({"expected_release_sha256": EXTERNAL_RELEASE_PIN}, "manifest"),
        (
            {
                "expected_release_sha256": "not-a-sha",
                "expected_manifest_file_sha256": EXTERNAL_MANIFEST_FILE_PIN,
            },
            "release",
        ),
    ),
)
def test_prepare_requires_external_deployment_pins_before_any_write(
    tmp_path: Path, kwargs: dict[str, str], label: str
) -> None:
    v2_root = tmp_path / "v2"
    with pytest.raises(ValueError, match=rf"expected {label}.*SHA256"):
        prepare.prepare_v2_root(
            v1_root=tmp_path / "v1",
            v2_root=v2_root,
            inputs={},
            **kwargs,
        )
    assert not v2_root.exists()


@pytest.mark.parametrize(
    ("kwargs", "label"),
    (
        (
            {"expected_manifest_file_sha256": EXTERNAL_MANIFEST_FILE_PIN},
            "release",
        ),
        ({"expected_release_sha256": EXTERNAL_RELEASE_PIN}, "manifest"),
    ),
)
def test_initialize_requires_external_deployment_pins_before_root_read(
    tmp_path: Path, kwargs: dict[str, str], label: str
) -> None:
    v2_root = tmp_path / "missing-v2"
    with pytest.raises(ValueError, match=rf"expected {label}.*SHA256"):
        prepare.initialize_v2_trajectories(v2_root, **kwargs)
    assert not v2_root.exists()


@pytest.mark.parametrize(
    "argv",
    (
        (
            "prepare-core-v2",
            "--v1-root",
            "/tmp/v1",
            "--v2-root",
            "/tmp/v2",
            "--inputs-json",
            "/tmp/inputs.json",
            "--orchestrator-pid",
            "42",
        ),
        ("init-core-v2", "--v2-root", "/tmp/v2"),
    ),
)
def test_prepare_cli_requires_both_external_deployment_pins(
    argv: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        prepare.build_parser().parse_args(argv)


@pytest.fixture
def executor_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "executor-repo"
    for record in core.FROZEN_ACTUAL_V3_EXECUTORS:
        source = prepare.REPO_ROOT / str(record["path"])
        target = root / str(record["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return root


@pytest.fixture
def pre_scan_rows() -> list[dict[str, object]]:
    return [
        {"row_id": f"pyramid|candidate-{index:03d}", "width": [16, 32, 64]}
        for index in range(686)
    ]


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
    _write_json(
        root / "variants" / "full" / "seed_20260718" / "trajectory_terminal.json",
        {"v1_only": True},
    )
    return root


@pytest.fixture
def frozen_inputs(
    pre_scan_rows: list[dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, object]:
    def record(name: str) -> dict[str, object]:
        path = tmp_path / "formal-inputs" / f"{name}.json"
        _write_json(path, {"fixture": name})
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    source_registry = record("source-registry")
    inputs = {
        "pre_scan_registry": pre_scan_rows,
        "gold176": record("gold176"),
        "graph_features": record("graph-features"),
        "capability_profiles": record("capability-profiles-v3"),
        "source_registry": source_registry,
        "scope_input_batch": {
            **record("scope-input-batch"),
            "task_id": "S7-PYR-TVM",
            "model": "Pyramid",
            "hardware": "H800",
            "backend_profile": "tvm_auto",
            "inference_batch": 1,
        },
        "measurement_ap": record("measurement-ap"),
        "hv_reference": record("hv-reference"),
    }
    monkeypatch.setattr(
        prepare,
        "_PINNED_FORMAL_INPUT_IDENTITIES",
        {
            key: {
                "path": str(Path(str(inputs[key]["path"])).resolve(strict=False)),
                "sha256": inputs[key]["sha256"],
            }
            for key in prepare.FORMAL_INPUT_IDENTITIES
        },
    )
    monkeypatch.setattr(
        core,
        "EXPECTED_SOURCE_REGISTRY_SHA256",
        source_registry["sha256"],
    )
    return inputs


PRIMITIVE_PATHS = {
    "materializer": "scripts/stage5_materialize_round_sources_v1.sh",
    "quant_contract": "scripts/stage3_tvm_int8_quant_contract_v3.py",
    "performance_plan_builder": "scripts/stage5_build_performance_plan_v2.py",
    "performance_executor": "scripts/stage3_execute_performance_plan_v3.py",
    "ap_plan_builder": "scripts/stage5_ap_plan_v2.py",
    "ap_executor": "scripts/stage3_execute_ap_plan_v3.py",
    "finalizer": "scripts/stage5_finalize_feedback_v2.py",
    "promoter": "scripts/stage5_promote_actual_feedback_v3.py",
    "stage5_controller": "scripts/stage5_task_round_controller_v3.sh",
    "stage5_runtime": "scripts/stage5_advance_task_round_v2.py",
}


@pytest.fixture
def embedded_source_manifest(
    tmp_path: Path,
    executor_repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    task_contract = json.loads(
        (
            prepare.REPO_ROOT / "results/stage5_single_target_search_v2_20260718/"
            "S5-PYR-TVM/task_contract.json"
        ).read_text()
    )
    stage4_closure = json.loads(
        (
            prepare.REPO_ROOT / "results/stage4_p1_p3_closure_v1_20260716/"
            "stage4_p1_p3_closure_audit.json"
        ).read_text()
    )
    objective_unsigned = {
        "schema_version": "stage7_raw_objective_reference_v1",
        "formula": {
            "objectives": [
                "minimize(latency_ms)",
                "minimize(energy_j)",
                "minimize(-ap70)",
            ],
            "reference": (
                "max_successful_gold_objective + " "max(abs(max)*0.05, 1e-9)"
            ),
            "strictly_worse_than_every_successful_gold_point": True,
        },
        # Gold176 contains 174 rows with complete objective measurements.
        "successful_gold_rows": 174,
        "values": {
            "latency_ms": 100.0,
            "energy_j": 10.0,
            "negative_ap70": 0.0,
        },
    }
    objective_reference = {
        **objective_unsigned,
        "reference_sha256": core.canonical_sha256(objective_unsigned),
    }
    sources = {
        "task_contract": (tmp_path / "sources/task_contract.json", task_contract),
        "stage4_closure": (tmp_path / "sources/stage4_closure.json", stage4_closure),
        "raw_objective_reference": (
            tmp_path / "sources/raw_objective_reference.json",
            objective_reference,
        ),
    }
    manifest: dict[str, object] = {
        "schema_version": "stage7_embedded_immutable_source_manifest_v1",
    }
    for name, (path, payload) in sources.items():
        _write_json(path, payload)
        manifest[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    monkeypatch.setattr(
        prepare,
        "_PINNED_EMBEDDED_SOURCE_IDENTITIES",
        {
            key: copy.deepcopy(manifest[key])
            for key in prepare.FORMAL_EMBEDDED_SOURCE_IDENTITIES
        },
    )
    primitives = {}
    for name, relative in PRIMITIVE_PATHS.items():
        source = prepare.REPO_ROOT / relative
        target = executor_repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        primitives[name] = {
            "path": str(target.resolve()),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    manifest["primitives"] = primitives
    return manifest


def _without_external_embedded_records(
    frozen_inputs: dict[str, object],
) -> dict[str, object]:
    return {
        key: value
        for key, value in frozen_inputs.items()
        if key not in {"scope_input_batch", "measurement_ap"}
    }


@pytest.fixture
def local_formal_fixture(
    monkeypatch: pytest.MonkeyPatch,
    v1_root: Path,
    pre_scan_rows: list[dict[str, object]],
) -> None:
    monkeypatch.setattr(
        core,
        "EXPECTED_BLOCKER_SHA256",
        hashlib.sha256(
            (v1_root / "status" / "without_capability_scan_blocked.json").read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        core,
        "EXPECTED_ADMISSION_SHA256",
        hashlib.sha256(
            (v1_root / "audits" / "scanner_admission_expanded_v1.json").read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        core,
        "EXPECTED_ORDERED_PRE_SCAN_SHA256",
        core.canonical_sha256(pre_scan_rows),
    )


def test_prepare_builds_two_embedded_contracts_from_reviewed_frozen_sources(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    embedded_source_manifest: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
) -> None:
    v2_root = tmp_path / "v2"

    result = _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=_without_external_embedded_records(frozen_inputs),
        embedded_source_manifest=embedded_source_manifest,
        repo_root=executor_repo_root,
    )

    assert result["embedded_immutable_contracts_written"] == [
        "measurement_ap",
        "scope_input_batch",
    ]
    contract = json.loads((v2_root / "contracts/core_ablation_v2.json").read_text())
    for key in ("scope_input_batch", "measurement_ap"):
        path = v2_root / f"contracts/{key}.json"
        assert contract["immutable_inputs"][key] == {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    scope = json.loads((v2_root / "contracts/scope_input_batch.json").read_text())
    measurement = json.loads((v2_root / "contracts/measurement_ap.json").read_text())
    assert scope["task_id"] == "S7-PYR-TVM"
    assert scope["source_task_id"] == "S5-PYR-TVM"
    assert scope["model"] == "pyramid"
    assert scope["hardware_id"] == "h800"
    assert scope["dispatch_key"] == "tvm_auto"
    assert scope["inference_batch"] == 1
    assert measurement["closure"]["stage4_closed"] is True
    assert measurement["raw_objective_reference"]["successful_gold_rows"] == 174
    assert measurement["raw_objective_reference"]["values"]["latency_ms"] == 100.0
    assert set(measurement["exact_key_protocol_sha256"]) == {
        "build_protocol_sha256",
        "tuning_protocol_sha256",
        "measurement_protocol_sha256",
        "ap_protocol_sha256",
        "runtime_contract_sha256",
    }
    assert all(
        len(value) == 64 for value in measurement["exact_key_protocol_sha256"].values()
    )


def test_prepare_and_initialize_authenticate_external_deployment_pins(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
) -> None:
    for relative in (
        *deployment.DEFAULT_BUNDLE_FILES,
        *deployment.DEFAULT_PRIMITIVE_FILES,
    ):
        source = prepare.REPO_ROOT / relative
        target = executor_repo_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    v2_root = tmp_path / "v2"
    bundle = prepare.build_deployment_bundle(
        v2_root=v2_root,
        frozen_repo_root=executor_repo_root,
    )
    pins = {
        "expected_release_sha256": bundle["deployment_release_sha256"],
        "expected_manifest_file_sha256": bundle[
            "deployment_manifest_file_sha256"
        ],
    }

    prepared = prepare.prepare_v2_root(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        repo_root=executor_repo_root,
        **pins,
    )
    initialized = prepare.initialize_v2_trajectories(
        v2_root,
        repo_root=executor_repo_root,
        **pins,
    )

    assert (
        prepared["deployment_release_sha256"]
        == bundle["deployment_release_sha256"]
    )
    assert (
        prepared["deployment_manifest_file_sha256"]
        == bundle["deployment_manifest_file_sha256"]
    )
    assert initialized["trajectory_count"] == 12


@pytest.mark.parametrize(
    "tamper",
    ("task_content", "primitive_content", "primitive_path", "manifest_sha"),
)
def test_embedded_contract_source_tampering_is_rejected_before_any_v2_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    embedded_source_manifest: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
    tamper: str,
) -> None:
    manifest = copy.deepcopy(embedded_source_manifest)
    if tamper == "task_content":
        Path(str(manifest["task_contract"]["path"])).write_text("{}\n")
    elif tamper == "primitive_content":
        Path(str(manifest["primitives"]["finalizer"]["path"])).write_text("# drift\n")
    elif tamper == "primitive_path":
        manifest["primitives"]["finalizer"]["path"] = str(
            (tmp_path / "outside.py").resolve()
        )
        Path(str(manifest["primitives"]["finalizer"]["path"])).write_text("outside")
        manifest["primitives"]["finalizer"]["sha256"] = hashlib.sha256(
            b"outside"
        ).hexdigest()
    else:
        manifest["stage4_closure"]["sha256"] = "0" * 64
    v2_root = tmp_path / "v2"

    with pytest.raises(ValueError, match="source|primitive|SHA|path"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=_without_external_embedded_records(frozen_inputs),
            embedded_source_manifest=manifest,
            repo_root=executor_repo_root,
        )

    assert not v2_root.exists()


def test_formal_embedded_builder_rejects_self_signed_nonproduction_sources(
    embedded_source_manifest: dict[str, object],
    executor_repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prepare,
        "_PINNED_EMBEDDED_SOURCE_IDENTITIES",
        prepare.FORMAL_EMBEDDED_SOURCE_IDENTITIES,
    )
    with pytest.raises(ValueError, match="pinned.*source|source.*identity"):
        prepare.build_embedded_immutable_contracts(
            embedded_source_manifest,
            repo_root=executor_repo_root,
        )


def test_embedded_contracts_participate_in_exact_recovery_and_init_allowlists(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    embedded_source_manifest: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
) -> None:
    v2_root = tmp_path / "v2"
    inputs = _without_external_embedded_records(frozen_inputs)
    first = _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=inputs,
        embedded_source_manifest=embedded_source_manifest,
        repo_root=executor_repo_root,
    )
    before = _hash_tree(v2_root)

    recovered = _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=inputs,
        embedded_source_manifest=embedded_source_manifest,
        repo_root=executor_repo_root,
    )
    initialized = _initialize(
        v2_root,
        repo_root=executor_repo_root,
    )

    assert first["recovery_validated"] is False
    assert recovered["recovery_validated"] is True
    assert before == {
        key: value
        for key, value in _hash_tree(v2_root).items()
        if not key.startswith("variants/") and key != "initialize_state.json"
    }
    assert initialized["trajectory_count"] == 12


def test_prepare_cli_forwards_reviewed_embedded_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs_path = tmp_path / "inputs.json"
    manifest_path = tmp_path / "embedded-source-manifest.json"
    _write_json(inputs_path, {"pre_scan_registry": []})
    _write_json(
        manifest_path,
        {"schema_version": prepare.EMBEDDED_SOURCE_SCHEMA},
    )
    captured: dict[str, object] = {}

    def prepare_root(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"schema_version": "prepared"}

    monkeypatch.setattr(prepare, "prepare_v2_root", prepare_root)
    result = prepare.main(
        [
            "prepare-core-v2",
            "--v1-root",
            str(tmp_path / "v1"),
            "--v2-root",
            str(tmp_path / "v2"),
            "--inputs-json",
            str(inputs_path),
            "--orchestrator-pid",
            "42",
            "--repo-root",
            str(tmp_path),
                "--embedded-source-manifest-json",
                str(manifest_path),
                "--expected-release-sha256",
                EXTERNAL_RELEASE_PIN,
                "--expected-manifest-file-sha256",
                EXTERNAL_MANIFEST_FILE_PIN,
            ]
        )

    assert result == 0
    assert captured["embedded_source_manifest"] == {
        "schema_version": prepare.EMBEDDED_SOURCE_SCHEMA
    }
    assert captured["expected_release_sha256"] == EXTERNAL_RELEASE_PIN
    assert (
        captured["expected_manifest_file_sha256"] == EXTERNAL_MANIFEST_FILE_PIN
    )
    assert captured["inputs"] == {"pre_scan_registry": []}
    assert json.loads(capsys.readouterr().out)["schema_version"] == "prepared"


def test_prepare_v2_writes_empty_cache_and_never_imports_v1_trajectory(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    original_v1_hash_tree = _hash_tree(v1_root)
    v2 = tmp_path / "v2"

    result = _prepare(
        v1_root=v1_root,
        v2_root=v2,
        inputs=frozen_inputs,
    )

    assert result["v2_root"] == str(v2.resolve())
    assert json.loads(
        (v2 / "contracts" / "measurement_cache_initial.json").read_text()
    ) == {
        "schema_version": "stage7_core_cache_v2",
        "entries": {},
        "lineage": [],
    }
    cache_artifacts = [
        path
        for path in v2.rglob("*.json")
        if isinstance(json.loads(path.read_text()), dict)
        and json.loads(path.read_text()).get("schema_version") == "stage7_core_cache_v2"
    ]
    assert cache_artifacts == [v2 / "contracts" / "measurement_cache_initial.json"]
    assert not list(v2.glob("trajectories/**/terminal_event.json"))
    assert not list(v2.glob("**/trajectory_terminal.json"))
    assert _hash_tree(v1_root) == original_v1_hash_tree


def test_prepare_rejects_dangling_pinned_input_before_any_v2_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    Path(str(frozen_inputs["gold176"]["path"])).unlink()
    v2_root = tmp_path / "v2"

    with pytest.raises(ValueError, match="unavailable|SHA"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
        )

    assert not v2_root.exists()


def test_prepare_rejects_pinned_input_content_drift_before_any_v2_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    Path(str(frozen_inputs["graph_features"]["path"])).write_text('{"drifted":true}\n')
    v2_root = tmp_path / "v2"

    with pytest.raises(ValueError, match="SHA"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
        )

    assert not v2_root.exists()


@pytest.mark.parametrize(
    "v1_path_kind,v2_path_kind",
    (("root", "root"), ("root", "child"), ("child", "root"), ("root", "symlink_child")),
)
def test_prepare_v2_rejects_resolved_root_overlap_before_any_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    v1_path_kind: str,
    v2_path_kind: str,
) -> None:
    child = v1_root / "v2-child"
    alias = tmp_path / "v1-alias"
    os.symlink(v1_root, alias)
    paths = {
        "root": v1_root,
        "child": child,
        "symlink_child": alias / "v2-child",
    }
    before = _hash_tree(v1_root)

    with pytest.raises(ValueError, match="overlap"):
        _prepare(
            v1_root=paths[v1_path_kind],
            v2_root=paths[v2_path_kind],
            inputs=frozen_inputs,
        )

    assert _hash_tree(v1_root) == before
    assert not child.exists()


@pytest.mark.parametrize(
    "injected",
    (
        {"trajectory_snapshot": {"rows": []}},
        {"source_registry": {"path": "/immutable/trajectory_terminal.json"}},
        {"source_registry": {"measurement_cache": {"entries": {}}}},
        {"source_registry": {"latest_feedback_event": {"rows": []}}},
    ),
)
def test_prepare_v2_rejects_nonimmutable_or_state_shaped_provenance(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    injected: dict[str, object],
) -> None:
    inputs = {**frozen_inputs, **injected}

    with pytest.raises(ValueError, match="provenance|state"):
        _prepare(
            v1_root=v1_root,
            v2_root=tmp_path / "v2",
            inputs=inputs,
        )

    assert not (tmp_path / "v2").exists()


def test_initialize_v2_creates_12_seed_root_contracts_without_scanner_ready(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    v2_root = tmp_path / "v2"
    _prepare(v1_root=v1_root, v2_root=v2_root, inputs=frozen_inputs)

    result = _initialize(v2_root)

    assert result["trajectory_count"] == 12
    assert {(row["variant"], row["seed"]) for row in result["trajectories"]} == {
        (variant, seed) for variant in CORE_VARIANTS for seed in SEEDS
    }
    assert all(row["candidate_pool"] == "pre_scan" for row in result["trajectories"])
    assert all(row["scanner_deferred"] is True for row in result["trajectories"])
    trajectory_contracts = [
        json.loads(Path(str(row["trajectory_contract_path"])).read_text())
        for row in result["trajectories"]
    ]
    assert len(trajectory_contracts) == 12
    expected_paths = {
        v2_root / "variants" / variant / f"seed_{seed}" / "trajectory_contract.json"
        for variant in CORE_VARIANTS
        for seed in SEEDS
    }
    assert {
        Path(str(row["trajectory_contract_path"])) for row in result["trajectories"]
    } == expected_paths
    assert not (v2_root / "trajectories").exists()
    assert not list(v2_root.glob("variants/*/seed_*/round_*/trajectory_contract.json"))
    assert {row["ordered_pre_scan_sha256"] for row in trajectory_contracts} == {
        core.EXPECTED_ORDERED_PRE_SCAN_SHA256
    }
    assert all(row["scanner_claim_allowed"] is False for row in trajectory_contracts)
    assert all(
        row["round_artifact_pattern"].endswith("/round_NN")
        for row in trajectory_contracts
    )


def test_formal_contract_binds_all_inputs_and_five_actual_v3_executors(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    v2_root = tmp_path / "v2"
    _prepare(v1_root=v1_root, v2_root=v2_root, inputs=frozen_inputs)
    contract = json.loads((v2_root / "contracts" / "core_ablation_v2.json").read_text())

    assert set(contract["immutable_inputs"]) == {
        "gold176",
        "graph_features",
        "capability_profiles",
        "source_registry",
        "scope_input_batch",
        "measurement_ap",
        "hv_reference",
    }
    assert all(
        Path(record["path"]).is_absolute() and len(record["sha256"]) == 64
        for record in contract["immutable_inputs"].values()
    )
    assert contract["immutable_inputs_sha256"] == core.canonical_sha256(
        contract["immutable_inputs"]
    )
    assert contract["actual_feedback_v3_executors"] == [
        dict(record) for record in core.FROZEN_ACTUAL_V3_EXECUTORS
    ]
    assert len(contract["actual_feedback_v3_executors"]) == 5


def test_prepare_hashes_executor_files_and_rejects_fixture_drift_before_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
) -> None:
    drifted_path = executor_repo_root / str(core.FROZEN_ACTUAL_V3_EXECUTORS[2]["path"])
    drifted_path.write_bytes(drifted_path.read_bytes() + b"\nreview-drift\n")
    v2_root = tmp_path / "v2"

    with pytest.raises(ValueError, match="executor.*SHA drift"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
            repo_root=executor_repo_root,
        )

    assert not v2_root.exists()


def test_prepare_existing_root_requires_exact_owner_pid_contract_and_cache_lineage(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    v2_root = tmp_path / "v2"
    identities = {4242: _process_identity()}

    def probe(pid: int) -> dict[str, object]:
        try:
            return identities[pid]
        except KeyError as error:
            raise ValueError("process is not alive") from error

    _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        owner="formal-owner",
        orchestrator_pid=4242,
        process_identity_probe=probe,
    )
    before = _hash_tree(v2_root)

    recovered = _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        owner="formal-owner",
        orchestrator_pid=4242,
        process_identity_probe=probe,
    )
    assert recovered["recovery_validated"] is True
    assert _hash_tree(v2_root) == before

    for owner, pid in (("other-owner", 4242), ("formal-owner", 4343)):
        with pytest.raises(ValueError, match="recovery|owner|PID"):
            _prepare(
                v1_root=v1_root,
                v2_root=v2_root,
                inputs=frozen_inputs,
                owner=owner,
                orchestrator_pid=pid,
                process_identity_probe=probe,
            )
        assert _hash_tree(v2_root) == before

    cache_path = v2_root / "contracts" / "measurement_cache_initial.json"
    cache = json.loads(cache_path.read_text())
    cache["lineage"].append({"forbidden": True})
    _write_json(cache_path, cache)
    drifted = _hash_tree(v2_root)
    with pytest.raises(ValueError, match="cache|lineage|recovery"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
            owner="formal-owner",
            orchestrator_pid=4242,
            process_identity_probe=probe,
        )
    assert _hash_tree(v2_root) == drifted


def test_prepare_facade_and_internal_state_emit_byte_identical_artifacts(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
) -> None:
    state = importlib.import_module("framework.stage7.prepare_state_v2")
    v2_root = tmp_path / "v2"
    identity = _process_identity()
    probe = lambda _pid: identity

    facade_prepare = _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        owner="formal-owner",
        orchestrator_pid=4242,
        repo_root=executor_repo_root,
        process_identity_probe=probe,
    )
    facade_initialize = _initialize(
        v2_root,
        repo_root=executor_repo_root,
        process_identity_probe=probe,
    )
    facade_bytes = {
        str(path.relative_to(v2_root)): path.read_bytes()
        for path in sorted(v2_root.rglob("*"))
        if path.is_file()
    }
    shutil.rmtree(v2_root)

    internal_prepare = state.prepare_v2_root(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        owner="formal-owner",
        orchestrator_pid=4242,
        repo_root=executor_repo_root,
        process_identity_probe=probe,
        formal_input_identities=prepare._PINNED_FORMAL_INPUT_IDENTITIES,
        embedded_source_identities=prepare._PINNED_EMBEDDED_SOURCE_IDENTITIES,
        executor_records=prepare.core_executor_records(),
        expected_release_sha256=EXTERNAL_RELEASE_PIN,
        expected_manifest_file_sha256=EXTERNAL_MANIFEST_FILE_PIN,
    )
    internal_initialize = state.initialize_v2_trajectories(
        v2_root,
        repo_root=executor_repo_root,
        process_identity_probe=probe,
        executor_records=prepare.core_executor_records(),
        expected_release_sha256=EXTERNAL_RELEASE_PIN,
        expected_manifest_file_sha256=EXTERNAL_MANIFEST_FILE_PIN,
    )
    internal_bytes = {
        str(path.relative_to(v2_root)): path.read_bytes()
        for path in sorted(v2_root.rglob("*"))
        if path.is_file()
    }

    assert internal_prepare == facade_prepare
    assert internal_initialize == facade_initialize
    assert internal_bytes == facade_bytes


def test_no_gpu_dryrun_default_prepare_callable_observes_old_facade_monkeypatches(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    embedded_source_manifest: dict[str, object],
    local_formal_fixture: None,
    executor_repo_root: Path,
) -> None:
    dryrun = importlib.import_module("scripts.stage7_core_no_gpu_dry_run_v2")
    v2_root = tmp_path / "v2"

    result = dryrun.prepare_and_initialize_no_gpu_root(
        v1_root=v1_root,
        v2_root=v2_root,
        prepare_inputs=_without_external_embedded_records(frozen_inputs),
        embedded_source_manifest=embedded_source_manifest,
        repo_root=executor_repo_root,
        expected_release_sha256=EXTERNAL_RELEASE_PIN,
        expected_manifest_file_sha256=EXTERNAL_MANIFEST_FILE_PIN,
        initialize=lambda *_args, **_kwargs: {
            "trajectory_count": 12,
            "trajectories": [],
        },
    )

    assert result["candidate_pool_count"] == 686
    assert result["trajectory_count"] == 12
    assert (v2_root / "contracts" / "scope_input_batch.json").is_file()


def test_prepare_recovery_rejects_dead_or_reused_pid_before_any_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    v2_root = tmp_path / "v2"
    identities = {4242: _process_identity()}

    def probe(pid: int) -> dict[str, object]:
        if pid not in identities:
            raise ValueError("process is not alive")
        return identities[pid]

    _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        owner="formal-owner",
        orchestrator_pid=4242,
        process_identity_probe=probe,
    )
    before = _hash_tree(v2_root)

    identities.clear()
    with pytest.raises(ValueError, match="alive|process identity"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
            owner="formal-owner",
            orchestrator_pid=4242,
            process_identity_probe=probe,
        )
    assert _hash_tree(v2_root) == before

    identities[4242] = _process_identity(start_time_ticks=987655)
    with pytest.raises(ValueError, match="recovery|process identity"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
            owner="formal-owner",
            orchestrator_pid=4242,
            process_identity_probe=probe,
        )
    assert _hash_tree(v2_root) == before


@pytest.mark.parametrize(
    "polluting_relative_path",
    (
        "scanner_state.json",
        "cache/legacy_cache.json",
        "variants/full/seed_20260718/trajectory_terminal.json",
        "variants/full/seed_20260718/round_00/actual_feedback.json",
        "trajectories/full/seed_20260718/v1_state.json",
    ),
)
def test_prepare_recovery_rejects_noncanonical_or_mutable_state_files(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    polluting_relative_path: str,
) -> None:
    v2_root = tmp_path / "v2"
    identity = _process_identity()
    probe = lambda _pid: identity
    _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
        owner="formal-owner",
        orchestrator_pid=4242,
        process_identity_probe=probe,
    )
    pollution = v2_root / polluting_relative_path
    _write_json(pollution, {"forbidden": True})
    before = _hash_tree(v2_root)

    with pytest.raises(ValueError, match="noncanonical|recovery"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
            owner="formal-owner",
            orchestrator_pid=4242,
            process_identity_probe=probe,
        )

    assert _hash_tree(v2_root) == before


def test_prepare_rejects_preexisting_unowned_empty_root(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
) -> None:
    v2_root = tmp_path / "v2"
    v2_root.mkdir()

    with pytest.raises(ValueError, match="pre-existing empty"):
        _prepare(
            v1_root=v1_root,
            v2_root=v2_root,
            inputs=frozen_inputs,
        )

    assert not any(v2_root.iterdir())


@pytest.mark.parametrize(
    "polluting_relative_path",
    (
        "arbitrary.json",
        "scanner/state.json",
        "cache/second_cache.json",
        "variants/full/seed_20260718/trajectory_terminal.json",
        "trajectories/v1_import.json",
    ),
)
def test_initialize_rejects_contaminated_prepared_root_without_any_write(
    tmp_path: Path,
    v1_root: Path,
    frozen_inputs: dict[str, object],
    local_formal_fixture: None,
    polluting_relative_path: str,
) -> None:
    v2_root = tmp_path / "v2"
    _prepare(
        v1_root=v1_root,
        v2_root=v2_root,
        inputs=frozen_inputs,
    )
    pollution = v2_root / polluting_relative_path
    _write_json(pollution, {"forbidden": True})
    before = _hash_tree(v2_root)

    with pytest.raises(ValueError, match="base root|noncanonical|contaminated"):
        _initialize(v2_root)

    assert _hash_tree(v2_root) == before
    assert not (v2_root / "initialize_state.json").exists()


def _profile() -> dict[str, object]:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto-formal-v3",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"compiler").hexdigest(),
        dispatch_key="tvm_auto",
        features={
            "s1p_fp16_build_success_coverage": 1.0,
            "s1q_fp16_build_success_coverage": 1.0,
            "s1p_int8_build_success_coverage": 1.0,
            "s1q_int8_build_success_coverage": 1.0,
            "operator_coverage": 0.75,
        },
    )


def _selector_context() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    profiles = [_profile()]
    profiles.append(
        build_capability_profile(
            capability_profile_id="h800-trt-formal-v3",
            hardware_target="h800",
            compiler_fingerprint=hashlib.sha256(b"trt").hexdigest(),
            dispatch_key="trt_engine",
            features={
                "s1p_fp16_build_success_coverage": 1.0,
                "s1q_fp16_build_success_coverage": 1.0,
                "s1p_int8_build_success_coverage": 1.0,
                "s1q_int8_build_success_coverage": 1.0,
                "operator_coverage": 0.5,
            },
        )
    )
    rows: list[dict[str, object]] = []
    graphs: list[dict[str, object]] = []
    groups = ([16, 32, 64], [24, 48, 96])
    for group_index, width in enumerate(groups):
        group_id = "pyramid|" + "x".join(map(str, width))
        graphs.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": width,
                "conv_count": 20 + group_index,
            }
        )
        for profile in profiles:
            for q_index, q_mode in enumerate(("fp16", "int8")):
                row_id = (
                    f"{group_id}|q={q_mode}|profile={profile['capability_profile_id']}"
                )
                rows.append(
                    {
                        "manifest_job_id": row_id,
                        "row_id": row_id,
                        "group_id": group_id,
                        "model": "pyramid",
                        "width": width,
                        "q_mode": q_mode,
                        "capability_profile_id": profile["capability_profile_id"],
                        "dispatch_key": profile["dispatch_key"],
                        "terminal_status": "measured_success_gold",
                        "training_source": "initial_coldstart",
                        "latency_ms": 2.0 + group_index + q_index,
                        "energy_j": 0.2 + 0.1 * group_index + 0.05 * q_index,
                        "ap70": 0.7 + 0.01 * group_index - 0.02 * q_index,
                    }
                )
    closure = {
        "schema_version": "stage4_p1_p3_closure_audit_v1",
        "stage4_closed": True,
        "stage5_search_ready": True,
        "canonical_value_heads": {
            "latency_ms": "extra_trees_log",
            "energy_j": "extra_trees_log",
            "ap70": "lgbm_huber_residual",
        },
        "uncertainty_policy": "lgbm_quantile_plus_group_conformal",
        "selected_acquisition_policy": "predicted_frontier_diversity",
        "training_source_rows": {"initial_coldstart": len(rows)},
    }
    registry = {
        "schema_version": "stage5_candidate_source_registry_v1",
        "groups": [
            {
                "group_id": "pyramid|" + "x".join(map(str, width)),
                "model": "pyramid",
                "width": width,
                "source_status": "ready",
                "source_evidence_sha256": f"{index + 1:064x}",
                "source_contract": {
                    "checkpoint_path": f"/frozen/{index}.pth",
                    "onnx_path": f"/frozen/{index}.onnx",
                },
                "graph_features": {
                    "group_id": "pyramid|" + "x".join(map(str, width)),
                    "model": "pyramid",
                    "width": width,
                    "conv_count": 20 + index,
                },
            }
            for index, width in enumerate(
                (
                    [16, 32, 64],
                    [24, 48, 96],
                    [32, 64, 128],
                    [40, 80, 160],
                    [48, 96, 192],
                )
            )
        ],
    }
    pool = policy.prepare_frozen_candidate_pools(
        registry, raw_profile=_profile(), measured_row_ids=set()
    )["pre_scan_candidates"]
    return rows, graphs, profiles, closure, pool


def test_v2_selector_audits_single_variable_isolation_on_one_ordered_pre_scan_pool() -> (
    None
):
    rows, graphs, profiles, closure, pool = _selector_context()
    task = policy.build_stage7_task(_profile())
    selections = {
        variant: policy.select_v2_pre_scan_round(
            variant=variant,
            seed=SEEDS[0],
            task=task,
            pre_scan_pool=pool,
            initial_rows=rows,
            initial_graph_features=graphs,
            capability_profiles=profiles,
            closure=closure,
        )
        for variant in CORE_VARIANTS
    }

    full = selections["full"]
    without_surrogate = selections["without_surrogate"]
    without_feedback = selections["without_measured_feedback"]
    backend_blind = selections["backend_blind"]
    assert {
        selection.audit["ordered_pre_scan_sha256"] for selection in selections.values()
    } == {policy.canonical_sha256(pool)}
    assert full.audit["surrogate_calls"] > 0
    assert without_surrogate.audit["surrogate_calls"] == 0
    assert without_surrogate.audit["uncertainty_calls"] == 0
    assert without_surrogate.audit["predicted_frontier_calls"] == 0
    assert without_feedback.audit["bundle_refit_calls_after_initial"] == 0
    assert without_feedback.audit["actual_graph_feedback_rows"] == 0
    assert backend_blind.audit["fixed_dispatch_key"] == "tvm_auto"
    assert backend_blind.audit["removed_feature_names"]
    assert not set(backend_blind.audit["model_feature_names"]) & set(
        backend_blind.audit["forbidden_backend_feature_names"]
    )


def test_v2_selector_fails_closed_when_full_variant_skips_observable_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, graphs, profiles, closure, pool = _selector_context()
    monkeypatch.setattr(policy, "select_stage7_round", lambda **_kwargs: {})

    with pytest.raises(ValueError, match="selector trace drift"):
        policy.select_v2_pre_scan_round(
            variant="full",
            seed=SEEDS[0],
            task=policy.build_stage7_task(_profile()),
            pre_scan_pool=pool,
            initial_rows=rows,
            initial_graph_features=graphs,
            capability_profiles=profiles,
            closure=closure,
        )
