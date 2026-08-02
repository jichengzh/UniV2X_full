"""Deployment-only bootstrap and immutable allowlist tests."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from framework.stage7 import deployment_bundle_v2 as bundle
from framework.stage7 import prepare_state_v2 as prepare_state
from scripts import stage7_core_no_gpu_dry_run_v2 as dry_run
from scripts import stage7_prepare_core_ablation_v2 as prepare_cli


def _current_external_pins(root: Path) -> dict[str, str]:
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    manifest = bundle._read_mapping(manifest_path, label="test manifest")
    return {
        "expected_release_sha256": manifest["deployment_release_sha256"],
        "expected_manifest_file_sha256": bundle._file_sha256(manifest_path),
    }


def _resign_bundle_as_transferred_from_uid(root: Path, source_uid: int) -> dict[str, str]:
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    state_path = root / "deployment/deployment_state_v2.json"
    manifest = bundle._read_mapping(manifest_path, label="test manifest")
    manifest["owner_uid"] = source_uid
    unsigned = bundle._manifest_unsigned(manifest)
    manifest["manifest_unsigned_payload_sha256"] = bundle.canonical_sha256(unsigned)
    manifest["deployment_bundle_sha256"] = bundle.canonical_sha256(unsigned)
    manifest_path.write_bytes(bundle._json_bytes(manifest))
    state = bundle._read_mapping(state_path, label="test state")
    state["owner_uid"] = source_uid
    state["deployment_manifest_sha256"] = manifest[
        "manifest_unsigned_payload_sha256"
    ]
    state["deployment_manifest_file_sha256"] = bundle._file_sha256(manifest_path)
    state["deployment_bundle_sha256"] = manifest["deployment_bundle_sha256"]
    state_unsigned = dict(state)
    state_unsigned.pop("state_sha256", None)
    state["state_sha256"] = bundle.canonical_sha256(state_unsigned)
    state_path.write_bytes(bundle._json_bytes(state))
    return _current_external_pins(root)


def test_deployment_bundle_module_exposes_canonical_builder() -> None:
    assert callable(bundle.build_deployment_bundle)


def test_transferred_bundle_owner_rebind_requires_old_pins_and_then_validates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(v2_root=root, frozen_repo_root=source)
    source_uid = os.getuid() + 10000
    old_pins = _resign_bundle_as_transferred_from_uid(root, source_uid)

    rebound = bundle.rebind_transferred_deployment_owner(
        root,
        frozen_repo_root=source,
        expected_source_release_sha256=old_pins["expected_release_sha256"],
        expected_source_manifest_file_sha256=old_pins[
            "expected_manifest_file_sha256"
        ],
        expected_source_owner_uid=source_uid,
    )

    assert rebound["source_owner_uid"] == source_uid
    assert rebound["target_owner_uid"] == os.getuid()
    assert rebound["deployment_release_sha256"] == old_pins[
        "expected_release_sha256"
    ]
    verified = bundle.validate_deployment_bundle(
        root,
        frozen_repo_root=source,
        expected_release_sha256=rebound["deployment_release_sha256"],
        expected_manifest_file_sha256=rebound[
            "deployment_manifest_file_sha256"
        ],
        expected_owner_uid=os.getuid(),
    )
    assert verified["deployment_bundle_sha256"] == rebound[
        "deployment_bundle_sha256"
    ]


def test_transferred_bundle_owner_rebind_rejects_wrong_old_manifest_pin_before_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(v2_root=root, frozen_repo_root=source)
    source_uid = os.getuid() + 10000
    old_pins = _resign_bundle_as_transferred_from_uid(root, source_uid)
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    before = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="manifest-file pin"):
        bundle.rebind_transferred_deployment_owner(
            root,
            frozen_repo_root=source,
            expected_source_release_sha256=old_pins[
                "expected_release_sha256"
            ],
            expected_source_manifest_file_sha256="0" * 64,
            expected_source_owner_uid=source_uid,
        )

    assert manifest_path.read_bytes() == before


def test_transferred_bundle_owner_rebind_preserves_all_integrity_checks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(v2_root=root, frozen_repo_root=source)
    source_uid = os.getuid() + 10000
    old_pins = _resign_bundle_as_transferred_from_uid(root, source_uid)
    target = root / "deployment/code/framework/stage7/core_ablation_v2.py"
    target.write_bytes(target.read_bytes() + b"\n# transfer-tamper\n")

    with pytest.raises(ValueError, match="byte or mode drift"):
        bundle.rebind_transferred_deployment_owner(
            root,
            frozen_repo_root=source,
            expected_source_release_sha256=old_pins[
                "expected_release_sha256"
            ],
            expected_source_manifest_file_sha256=old_pins[
                "expected_manifest_file_sha256"
            ],
            expected_source_owner_uid=source_uid,
        )


def test_builder_rejects_any_self_selected_bundle_allowlist(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    with pytest.raises(ValueError, match="exact reviewed deployment allowlist"):
        bundle.build_deployment_bundle(
            v2_root=tmp_path / "formal-v2",
            frozen_repo_root=repository,
            bundle_files=(
                "framework/__init__.py",
                "framework/stage7/__init__.py",
                "framework/stage7/deployment_bundle_v2.py",
            ),
        )


def test_default_bundle_contains_task_b_physical_projection_runtime(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"

    bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=repository,
    )

    bundled = root / "deployment/code/framework/stage7/physical_execution_v2.py"
    assert (
        bundled.read_bytes()
        == (repository / "framework/stage7/physical_execution_v2.py").read_bytes()
    )


def test_default_bundle_contains_reviewed_actual_mode_integration_bytes(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"
    verification = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=repository,
    )
    expected = {
        "framework/stage7/actual_mode_v2.py",
        "framework/stage7/actual_pipeline_support_v2.py",
        "framework/stage7/actual_pipeline_v2.py",
        "framework/stage7/no_gpu_support_v2.py",
        "framework/stage7/physical_execution_v2.py",
        "framework/stage7/physical_runtime_v2.py",
        "framework/stage7/physical_feedback_v2.py",
        "framework/stage7/source_execution_v2.py",
        "framework/stage7/source_relocation_v2.py",
        "framework/stage7/source_round_orchestration_v2.py",
        "scripts/stage7_actual_feedback_barrier_v2.py",
        "scripts/stage7_ablation_scheduler_v1.py",
        "scripts/stage7_core_ablation_orchestrator_v2.py",
        "scripts/stage7_core_ablation_scheduler_v2.py",
        "scripts/stage7_core_round_controller_v2.sh",
        "scripts/stage7_core_round_worker_v2.py",
        "scripts/stage7_core_online_ablation_v2.py",
        "scripts/stage7_core_no_gpu_dry_run_v2.py",
        "scripts/stage7_prepare_core_ablation_v2.py",
        "scripts/stage7_resolve_round_sources_v2.py",
        "scripts/stage7_h800_runtime_v2.py",
        "scripts/stage7_orchestrator_scheduler_adapter_v2.py",
        "scripts/stage7_execute_actual_v3_misses_v2.sh",
    }
    code_root = root / "deployment/code"
    listed = {
        Path(path).relative_to(code_root).as_posix()
        for path in verification["files"]
        if Path(path).is_relative_to(code_root)
    }

    assert expected <= listed


def test_default_bundle_allowlist_is_exact_task6b_runtime_closure() -> None:
    assert set(bundle.DEFAULT_BUNDLE_FILES) == {
        "framework/__init__.py",
        "framework/stage5/__init__.py",
        "framework/stage5/genome_contract_v1.py",
        "framework/stage5/measurement_plan_v1.py",
        "framework/stage5/measurement_plan_v2.py",
        "framework/stage7/__init__.py",
        "framework/stage7/ablation_statistics_v2.py",
        "framework/stage7/actual_mode_v2.py",
        "framework/stage7/actual_pipeline_support_v2.py",
        "framework/stage7/actual_pipeline_v2.py",
        "framework/stage7/actual_v3_adapter_v2.py",
        "framework/stage7/actual_v3_selector_v2.py",
        "framework/stage7/core_ablation_v2.py",
        "framework/stage7/core_cache_v2.py",
        "framework/stage7/deployment_bundle_v2.py",
        "framework/stage7/executor_admission_v2.py",
        "framework/stage7/formal_evidence_v2.py",
        "framework/stage7/formal_inputs_v2.py",
        "framework/stage7/no_gpu_support_v2.py",
        "framework/stage7/online_component_ablation_v1.py",
        "framework/stage7/paper_outputs_v2.py",
        "framework/stage7/physical_execution_v2.py",
        "framework/stage7/physical_feedback_v2.py",
        "framework/stage7/physical_runtime_v2.py",
        "framework/stage7/prepare_state_v2.py",
        "framework/stage7/search_policy_v1.py",
        "framework/stage7/source_execution_v2.py",
        "framework/stage7/source_relocation_v2.py",
        "framework/stage7/source_resolution_v2.py",
        "framework/stage7/source_round_orchestration_v2.py",
        "scripts/stage7_ablation_scheduler_v1.py",
        "scripts/stage7_actual_feedback_barrier_v2.py",
        "scripts/stage7_core_ablation_orchestrator_v2.py",
        "scripts/stage7_core_ablation_scheduler_v2.py",
        "scripts/stage7_core_no_gpu_dry_run_v2.py",
        "scripts/stage7_core_online_ablation_v2.py",
        "scripts/stage7_core_round_controller_v2.sh",
        "scripts/stage7_core_round_worker_v2.py",
        "scripts/stage7_execute_actual_v3_misses_v2.sh",
        "scripts/stage7_finalize_ablation_v1.py",
        "scripts/stage7_finalize_core_ablation_v2.py",
        "scripts/stage7_h800_runtime_v2.py",
        "scripts/stage7_orchestrator_scheduler_adapter_v2.py",
        "scripts/stage7_prepare_core_ablation_v2.py",
        "scripts/stage7_resolve_round_sources_v2.py",
        "scripts/stage7_scheduler_requests_v2.py",
        "scripts/stage7_source_lease_controller_v2.py",
        "scripts/stage7_source_scheduler_v2.py",
        "scripts/stage5_materialize_round_sources_v1.sh",
    }


def test_deployed_source_resolver_carries_the_exact_frozen_materializer(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"
    verification = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=repository,
    )
    code_root = Path(verification["bundle_code_root"])
    materializer = code_root / "scripts/stage5_materialize_round_sources_v1.sh"

    assert bundle._file_sha256(materializer) == (
        "d978e6287afc239c63471cadf729b37b4617bf46d1b8b8ea13cb4e2433bf4abe"
    )
    assert os.access(materializer, os.X_OK)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path;"
                "from scripts import stage7_resolve_round_sources_v2 as resolver;"
                f"root=Path({str(code_root)!r}).resolve();"
                "assert resolver.MATERIALIZER_PATH.resolve().is_relative_to(root);"
                "resolver._verify_frozen_dependencies()"
            ),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(code_root), str(repository))),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def _non_stage7_fallback(repository: Path, destination: Path) -> Path:
    """Expose frozen non-Stage7 dependencies without hiding bundle omissions."""
    framework_fallback = destination / "framework"
    scripts_fallback = destination / "scripts"
    framework_fallback.mkdir(parents=True)
    scripts_fallback.mkdir()
    for source in (repository / "framework").iterdir():
        if source.name in {"stage7", "__init__.py", "__pycache__"}:
            continue
        (framework_fallback / source.name).symlink_to(
            source, target_is_directory=source.is_dir()
        )
    for source in (repository / "scripts").iterdir():
        if source.name.startswith("stage7_") or source.name == "__pycache__":
            continue
        (scripts_fallback / source.name).symlink_to(
            source, target_is_directory=source.is_dir()
        )
    return destination


@pytest.mark.parametrize(
    "relative,help_text",
    (
        (
            "scripts/stage7_core_no_gpu_dry_run_v2.py",
            "Formal Stage7-to-actual-v3 integration dry-run",
        ),
        (
            "scripts/stage7_core_ablation_orchestrator_v2.py",
            "Artifact-derived persistent Stage7 core-ablation orchestrator",
        ),
        (
            "scripts/stage7_core_round_worker_v2.py",
            "usage: stage7_core_round_worker_v2.py",
        ),
    ),
)
def test_deployed_task6b_clis_import_without_any_stage7_repo_fallback(
    tmp_path: Path, relative: str, help_text: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"
    verification = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=repository,
    )
    code_root = Path(verification["bundle_code_root"])
    frozen_non_stage7 = _non_stage7_fallback(
        repository, tmp_path / "frozen-non-stage7"
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            (str(code_root), str(frozen_non_stage7))
        ),
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(code_root / relative),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert help_text in completed.stdout


def test_deployed_framework_stage7_cannot_fall_back_to_full_frozen_repo(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    sentinel = source / "framework/stage7/unbundled_sentinel.py"
    sentinel.write_text("SENTINEL = 'must-not-import' \n", encoding="utf-8")
    root = tmp_path / "formal-v2"
    verification = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            (verification["bundle_code_root"], str(source))
        ),
    }

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib;"
                "from pathlib import Path;"
                "from scripts import stage7_core_no_gpu_dry_run_v2 as dry;"
                f"dry._insert_runtime_repo_root(Path({str(source)!r}));"
                "importlib.import_module('framework.stage7.unbundled_sentinel')"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ModuleNotFoundError" in completed.stderr


def test_deployed_orchestrator_resolves_only_bundled_scheduler_adapter(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"
    verification = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=repository,
    )
    code_root = Path(verification["bundle_code_root"])
    adapter = code_root / "scripts/stage7_orchestrator_scheduler_adapter_v2.py"
    assert bundle._file_sha256(adapter) == (
        "3effb007a9c49d73b79a2933dc1cd0189e202d8a917ecfe9af678fedcff52b44"
    )
    command = (
        "from pathlib import Path;"
        "from scripts import stage7_core_ablation_orchestrator_v2 as orchestrator;"
        "from scripts import stage7_orchestrator_scheduler_adapter_v2 as adapter;"
        f"root=Path({str(code_root)!r}).resolve();"
        "assert Path(orchestrator.__file__).resolve().is_relative_to(root);"
        "assert Path(adapter.__file__).resolve().is_relative_to(root)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join((str(code_root), str(repository))),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_bundle_pins_the_job_builder_implementation() -> None:
    relative = "scripts/stage35_gold32_performance_plan_v1.py"

    assert relative in bundle.DEFAULT_PRIMITIVE_FILES
    assert bundle.EXPECTED_PRIMITIVE_SHA256[relative] == (
        "f42b2c7df7d205ed07f67872207d9dddd1336a8e59dfa64b4bf6db2fbf1412a0"
    )


def test_reviewed_stage5_measurement_modules_are_bundle_owned() -> None:
    assert {
        "framework/stage5/__init__.py",
        "framework/stage5/genome_contract_v1.py",
        "framework/stage5/measurement_plan_v1.py",
        "framework/stage5/measurement_plan_v2.py",
    } <= set(bundle.DEFAULT_BUNDLE_FILES)
    assert {
        "framework/stage5/measurement_plan_v1.py",
        "framework/stage5/measurement_plan_v2.py",
    }.isdisjoint(bundle.DEFAULT_PRIMITIVE_FILES)
    assert len(bundle.DEFAULT_PRIMITIVE_FILES) == 11


@pytest.mark.parametrize(
    "omitted",
    (
        "scripts/stage35_gold32_performance_plan_v1.py",
        "scripts/stage5_build_performance_plan_v2.py",
    ),
)
def test_bundle_rejects_omitting_any_required_execution_primitive(
    omitted: str, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"

    with pytest.raises(ValueError, match="primitive"):
        bundle.build_deployment_bundle(
            v2_root=root,
            frozen_repo_root=repository,
            primitive_files=tuple(
                relative
                for relative in bundle.DEFAULT_PRIMITIVE_FILES
                if relative != omitted
            ),
        )


@pytest.mark.parametrize(
    "relative",
    (
        "scripts/stage35_gold32_performance_plan_v1.py",
        "scripts/stage5_build_performance_plan_v2.py",
    ),
)
def test_validator_rejects_job_builder_import_chain_byte_drift(
    relative: str, tmp_path: Path
) -> None:
    source = tmp_path / "frozen-repo"
    files = _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    target = source / relative
    target.write_bytes(target.read_bytes() + b"\n# import-chain-drift\n")

    with pytest.raises(ValueError, match="primitive"):
        bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=source,
            **_current_external_pins(root),
        )


def _source_tree(root: Path) -> tuple[str, ...]:
    files = bundle.DEFAULT_BUNDLE_FILES
    repository = Path(__file__).resolve().parents[2]
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository / relative).read_bytes())
        path.chmod(0o640)
    for relative in bundle.DEFAULT_PRIMITIVE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((repository / relative).read_bytes())
        path.chmod(0o640)
    return files


def test_canonical_deployment_only_root_has_an_exact_immutable_tree(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    files = _source_tree(source)
    root = tmp_path / "formal-v2"

    verified = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )

    assert verified["bundle_code_root"] == str(root / "deployment/code")
    assert verified["frozen_repo_root"] == str(source.resolve())
    assert {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    } == {
        "deployment/deployment_manifest_v2.json",
        "deployment/deployment_state_v2.json",
        *(f"deployment/code/{relative}" for relative in files),
    }


def test_validator_requires_external_release_and_manifest_pins(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(v2_root=root, frozen_repo_root=source)

    with pytest.raises(ValueError, match="external deployment pins"):
        bundle.validate_deployment_bundle(root, frozen_repo_root=source)


def test_external_release_pin_covers_new_stage7_source_missing_on_remote(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    verified = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    (source / "framework/stage7/no_gpu_support_v2.py").unlink()

    recovered = bundle.validate_deployment_bundle(
        root,
        frozen_repo_root=source,
        expected_release_sha256=verified["deployment_release_sha256"],
        expected_manifest_file_sha256=verified[
            "deployment_manifest_file_sha256"
        ],
    )

    assert recovered["deployment_release_sha256"] == verified[
        "deployment_release_sha256"
    ]


def test_validator_rejects_fully_resigned_bundle_against_external_release_pin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    verified = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    state_path = root / "deployment/deployment_state_v2.json"
    manifest = bundle._read_mapping(manifest_path, label="manifest")
    target = root / "deployment/code/framework/stage7/core_ablation_v2.py"
    target.write_text("FORGED = True\n", encoding="utf-8")
    record = next(
        row
        for row in manifest["files"]
        if row["repo_relative_path"]
        == "framework/stage7/core_ablation_v2.py"
    )
    record["byte_count"] = target.stat().st_size
    record["sha256"] = bundle._file_sha256(target)
    manifest["ordered_file_list_sha256"] = bundle.canonical_sha256(
        manifest["files"]
    )
    forged_release = [
        {
            "repo_relative_path": row["repo_relative_path"],
            "sha256": row["sha256"],
        }
        for row in manifest["files"]
    ]
    manifest["deployment_release_sha256"] = bundle.canonical_sha256(
        forged_release
    )
    unsigned = bundle._manifest_unsigned(manifest)
    manifest["manifest_unsigned_payload_sha256"] = bundle.canonical_sha256(
        unsigned
    )
    manifest["deployment_bundle_sha256"] = bundle.canonical_sha256(unsigned)
    manifest_path.write_bytes(bundle._json_bytes(manifest))
    state = bundle._read_mapping(state_path, label="state")
    state["deployment_manifest_sha256"] = manifest[
        "manifest_unsigned_payload_sha256"
    ]
    state["deployment_manifest_file_sha256"] = bundle._file_sha256(
        manifest_path
    )
    state["deployment_bundle_sha256"] = manifest["deployment_bundle_sha256"]
    state["deployment_release_sha256"] = manifest[
        "deployment_release_sha256"
    ]
    state_unsigned = dict(state)
    state_unsigned.pop("state_sha256", None)
    state["state_sha256"] = bundle.canonical_sha256(state_unsigned)
    state_path.write_bytes(bundle._json_bytes(state))

    with pytest.raises(ValueError, match="release pin"):
        bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=source,
            expected_release_sha256=verified["deployment_release_sha256"],
            expected_manifest_file_sha256=bundle._file_sha256(manifest_path),
        )
    assert (root / "deployment/code/framework/__init__.py").read_text(
        encoding="utf-8"
    ) == '"""Stage7 deployment overlay package."""\nfrom pkgutil import extend_path\n\n__path__ = extend_path(__path__, __name__)\n'


@pytest.mark.parametrize(
    "attack", ["extra", "directory", "symlink", "hardlink", "sha", "mode", "secret"]
)
def test_bundle_validator_rejects_any_allowlist_or_integrity_drift(
    tmp_path: Path, attack: str
) -> None:
    source = tmp_path / "frozen-repo"
    files = _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(v2_root=root, frozen_repo_root=source)
    code = root / "deployment/code/framework/stage7/deployment_bundle_v2.py"
    if attack == "extra":
        (root / "deployment/code/unlisted.py").write_text("x\n", encoding="utf-8")
    elif attack == "directory":
        (root / "deployment/code/unlisted").mkdir()
    elif attack == "symlink":
        (root / "deployment/code/link.py").symlink_to(code)
    elif attack == "hardlink":
        os.link(code, root / "deployment/code/hardlinked.py")
    elif attack == "sha":
        code.write_text("drift\n", encoding="utf-8")
    elif attack == "mode":
        code.chmod(0o666)
    else:
        key_name = "API" + "_KEY"
        code.write_text(f"{key_name}='fixture-value'\n", encoding="utf-8")

    with pytest.raises(ValueError):
        bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=source,
            **_current_external_pins(root),
        )


def test_bundle_builder_recovers_only_byte_identical_stale_temp(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    temporary = target.with_suffix(".json.tmp")
    expected = b'{"safe":true}\n'
    temporary.write_bytes(expected)

    bundle._atomic_write(target, expected)

    assert target.read_bytes() == expected
    assert not temporary.exists()


def test_bundle_builder_rejects_drifted_stale_temp(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.with_suffix(".json.tmp").write_bytes(b"drift")

    with pytest.raises(ValueError, match="temporary artifact drift"):
        bundle._atomic_write(target, b"expected")


def test_bundle_builder_rejects_existing_symlink_even_when_bytes_match(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact.json"
    backing = tmp_path / "backing.json"
    backing.write_bytes(b"expected")
    target.symlink_to(backing)

    with pytest.raises(ValueError, match="regular non-symlink"):
        bundle._atomic_write(target, b"expected")


def test_prepare_accepts_only_a_validated_deployment_tree_without_mutating_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "frozen-repo"
    files = _source_tree(source)
    root = tmp_path / "formal-v2"
    before = bundle.build_deployment_bundle(
        v2_root=root, frozen_repo_root=source
    )
    deployment_bytes = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    contract = {
        "contract_sha256": "a" * 64,
        "immutable_inputs_sha256": "b" * 64,
        "actual_feedback_v3_executors": [],
        "candidate_pool_count": 0,
        "ordered_pre_scan_sha256": "c" * 64,
        "immutable_inputs": {},
    }
    monkeypatch.setattr(
        prepare_state, "verify_pinned_formal_input_identities", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        prepare_state, "verify_frozen_actual_v3_executors", lambda *_a, **_k: []
    )
    monkeypatch.setattr(
        prepare_state,
        "resolve_embedded_inputs",
        lambda inputs, **_k: (copy.deepcopy(dict(inputs)), {}),
    )
    monkeypatch.setattr(prepare_state, "_pre_scan_rows", lambda _inputs: [])
    monkeypatch.setattr(
        prepare_state, "_immutable_provenance", lambda _inputs: {"inputs": {}}
    )
    monkeypatch.setattr(
        prepare_state, "build_v2_contract", lambda *_a, **_k: copy.deepcopy(contract)
    )
    monkeypatch.setattr(
        prepare_state,
        "validate_v2_contract",
        lambda payload: copy.deepcopy(dict(payload)),
    )
    monkeypatch.setattr(
        prepare_state,
        "capture_process_identity",
        lambda *_a, **_k: {
            "pid": 7001,
            "uid": os.getuid(),
            "username": bundle.pwd.getpwuid(os.getuid()).pw_name,
            "process_identity_sha256": "d" * 64,
        },
    )

    result = prepare_state.prepare_v2_root(
        v1_root=(tmp_path / "v1").resolve(),
        v2_root=root.resolve(),
        inputs={},
        repo_root=source,
        expected_release_sha256=before["deployment_release_sha256"],
        expected_manifest_file_sha256=before[
            "deployment_manifest_file_sha256"
        ],
    )

    assert result["deployment_bundle_sha256"] == before["deployment_bundle_sha256"]
    assert deployment_bytes == {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).parts[0] == "deployment"
    }


def test_no_gpu_binding_rejects_a_bundle_missing_from_prepare_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    files = _source_tree(source)
    root = tmp_path / "formal-v2"
    verified = bundle.build_deployment_bundle(
        v2_root=root, frozen_repo_root=source
    )
    (root / "prepare_state.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="deployment"):
        dry_run.deployment_binding(
            root,
            frozen_repo_root=source,
            expected_release_sha256=verified["deployment_release_sha256"],
            expected_manifest_file_sha256=verified[
                "deployment_manifest_file_sha256"
            ],
        )


@pytest.mark.parametrize("metadata_label", ["manifest", "state"])
def test_bundle_validator_rejects_metadata_files_owned_by_a_different_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata_label: str
) -> None:
    source = tmp_path / "frozen-repo"
    files = _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(v2_root=root, frozen_repo_root=source)
    original = bundle._reject_link_or_unsafe_mode

    def owner_drift(path: Path, *, label: str) -> os.stat_result:
        metadata = original(path, label=label)
        if label != metadata_label:
            return metadata
        return SimpleNamespace(
            st_mode=metadata.st_mode,
            st_nlink=metadata.st_nlink,
            st_uid=metadata.st_uid + 1,
            st_size=metadata.st_size,
        )

    monkeypatch.setattr(bundle, "_reject_link_or_unsafe_mode", owner_drift)
    with pytest.raises(ValueError, match="owner drift"):
        bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=source,
            **_current_external_pins(root),
        )


def test_prepare_cli_exposes_explicit_bundle_and_frozen_repo_roots() -> None:
    parsed = prepare_cli.build_parser().parse_args(
        [
            "bundle-core-v2",
            "--v2-root",
            "/formal/v2",
            "--frozen-repo-root",
            "/frozen/repo",
        ]
    )

    assert parsed.command == "bundle-core-v2"
    assert parsed.v2_root == Path("/formal/v2")
    assert parsed.frozen_repo_root == Path("/frozen/repo")
