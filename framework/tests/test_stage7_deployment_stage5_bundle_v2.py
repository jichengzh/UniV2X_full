from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from framework.stage7 import deployment_bundle_v2 as bundle


REVIEWED_STAGE5_SHA256 = {
    "framework/stage5/__init__.py":
        "74dd138a48a6866c28a01a91744c1a8a2921957ea9c6c7b485590d0409a14759",
    "framework/stage5/genome_contract_v1.py":
        "1aa183c576510cf4612d3a96028a56e37b"
        "8fd4281058867b7d3b4a08660fbe7b",
    "framework/stage5/measurement_plan_v1.py":
        "7da96d705831ec834cee6d9e519753ba0c7c587b78e13a81ae6761f8dc67b15c",
    "framework/stage5/measurement_plan_v2.py":
        "4395418d9a0aedf4668bf045e1e23883779971ad823d75ef7a7ec92cc41196b8",
}


def _source_tree(root: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    for relative in (
        *bundle.DEFAULT_BUNDLE_FILES,
        *bundle.DEFAULT_PRIMITIVE_FILES,
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository / relative).read_bytes())
        destination.chmod(0o640)


def _external_pins(verified: dict[str, object]) -> dict[str, str]:
    return {
        "expected_release_sha256": str(
            verified["deployment_release_sha256"]
        ),
        "expected_manifest_file_sha256": str(
            verified["deployment_manifest_file_sha256"]
        ),
    }


def _resign_manifest_and_state(root: Path) -> dict[str, str]:
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    state_path = root / "deployment/deployment_state_v2.json"
    manifest = bundle._read_mapping(manifest_path, label="manifest")
    manifest["ordered_file_list_sha256"] = bundle.canonical_sha256(
        manifest["files"]
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
    state_unsigned = dict(state)
    state_unsigned.pop("state_sha256", None)
    state["state_sha256"] = bundle.canonical_sha256(state_unsigned)
    state_path.write_bytes(bundle._json_bytes(state))
    return {
        "expected_release_sha256": str(
            manifest["deployment_release_sha256"]
        ),
        "expected_manifest_file_sha256": bundle._file_sha256(manifest_path),
    }


def test_isolated_stage5_reviewed_modules_resolve_from_bundle_code(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path / "formal-v2"
    verified = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=repository,
    )
    code_root = Path(verified["bundle_code_root"])
    for relative, expected_sha in REVIEWED_STAGE5_SHA256.items():
        destination = code_root / relative
        assert destination.is_file()
        if not relative.endswith("/__init__.py"):
            assert bundle._file_sha256(destination) == expected_sha
    command = (
        "from pathlib import Path;"
        "import framework.stage5 as package;"
        "from framework.stage5 import genome_contract_v1 as genome;"
        "from framework.stage5 import measurement_plan_v1 as plan1;"
        "from framework.stage5 import measurement_plan_v2 as plan2;"
        f"root=Path({str(code_root)!r}).resolve();"
        "assert Path(package.__file__).resolve().is_relative_to(root);"
        "assert Path(genome.__file__).resolve().is_relative_to(root);"
        "assert Path(plan1.__file__).resolve().is_relative_to(root);"
        "assert Path(plan2.__file__).resolve().is_relative_to(root)"
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


def test_remaining_frozen_primitive_fallback_is_exactly_pinned() -> None:
    assert len(bundle.DEFAULT_PRIMITIVE_FILES) == 11
    assert set(bundle.DEFAULT_PRIMITIVE_FILES) == set(
        bundle.EXPECTED_PRIMITIVE_SHA256
    )
    assert all(
        relative not in REVIEWED_STAGE5_SHA256
        for relative in bundle.DEFAULT_PRIMITIVE_FILES
    )


def test_reviewed_cross_host_override_contract_is_exactly_four_stage5_paths(
) -> None:
    assert bundle.EXPECTED_CROSS_HOST_SOURCE_SHA256 == REVIEWED_STAGE5_SHA256
    assert set(bundle.EXPECTED_CROSS_HOST_BUNDLE_SHA256) == set(
        REVIEWED_STAGE5_SHA256
    )


def test_validator_accepts_old_remote_bytes_only_for_reviewed_stage5_overrides(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    verified = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    for relative in (
        "framework/stage5/measurement_plan_v1.py",
        "framework/stage5/measurement_plan_v2.py",
    ):
        (source / relative).write_text(
            f"# old remote bytes for {relative}\n",
            encoding="utf-8",
        )

    recovered = bundle.validate_deployment_bundle(
        root,
        frozen_repo_root=source,
        **_external_pins(verified),
    )

    assert recovered["deployment_release_sha256"] == verified[
        "deployment_release_sha256"
    ]


def test_validator_accepts_remote_stage7_and_bootstrap_source_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    verified = bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    (source / "framework/__init__.py").write_bytes(b"")
    (source / "framework/stage7/search_policy_v1.py").write_text(
        "# old remote Stage7 policy\n",
        encoding="utf-8",
    )

    recovered = bundle.validate_deployment_bundle(
        root,
        frozen_repo_root=source,
        **_external_pins(verified),
    )

    assert recovered["deployment_release_sha256"] == verified[
        "deployment_release_sha256"
    ]


def test_cross_host_source_policy_is_allowlist_bounded() -> None:
    assert bundle._cross_host_source_mismatch_allowed(
        "framework/__init__.py"
    )
    assert bundle._cross_host_source_mismatch_allowed(
        "framework/stage7/search_policy_v1.py"
    )
    assert bundle._cross_host_source_mismatch_allowed(
        "scripts/stage7_core_no_gpu_dry_run_v2.py"
    )
    assert not bundle._cross_host_source_mismatch_allowed(
        "framework/stage7/unlisted.py"
    )
    assert not bundle._cross_host_source_mismatch_allowed(
        "scripts/stage7_unlisted.py"
    )
    assert not bundle._cross_host_source_mismatch_allowed(
        "framework/stage5/arbitrary.py"
    )
    assert not bundle._cross_host_source_mismatch_allowed(
        "scripts/stage5_materialize_round_sources_v1.sh"
    )


def test_builder_rejects_unlisted_stage7_cross_host_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    unlisted = source / "framework/stage7/unlisted.py"
    unlisted.parent.mkdir(parents=True, exist_ok=True)
    unlisted.write_text("# not reviewed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exact reviewed deployment allowlist"):
        bundle.build_deployment_bundle(
            v2_root=tmp_path / "formal-v2",
            frozen_repo_root=source,
            bundle_files=(
                *bundle.DEFAULT_BUNDLE_FILES,
                "framework/stage7/unlisted.py",
            ),
        )


def test_builder_rejects_wrong_reviewed_local_override_pin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    (source / "framework/stage5/measurement_plan_v1.py").write_text(
        "# wrong reviewed local source\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cross-host override source SHA"):
        bundle.build_deployment_bundle(
            v2_root=tmp_path / "formal-v2",
            frozen_repo_root=source,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_validator_rejects_missing_or_extra_cross_host_override(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    manifest = bundle._read_mapping(manifest_path, label="manifest")
    overrides = list(manifest["cross_host_source_overrides"])
    if mutation == "missing":
        overrides.pop()
    else:
        overrides.append(
            {
                "repo_relative_path":
                    "framework/stage7/core_ablation_v2.py",
                "reviewed_source_sha256": "a" * 64,
                "bundle_sha256": "b" * 64,
            }
        )
    manifest["cross_host_source_overrides"] = overrides
    manifest_path.write_bytes(bundle._json_bytes(manifest))
    pins = _resign_manifest_and_state(root)

    with pytest.raises(ValueError, match="cross-host override contract"):
        bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=source,
            **pins,
        )


def test_validator_rejects_cross_host_override_remote_path_escape(
    tmp_path: Path,
) -> None:
    source = tmp_path / "frozen-repo"
    _source_tree(source)
    root = tmp_path / "formal-v2"
    bundle.build_deployment_bundle(
        v2_root=root,
        frozen_repo_root=source,
    )
    manifest_path = root / "deployment/deployment_manifest_v2.json"
    manifest = bundle._read_mapping(manifest_path, label="manifest")
    record = next(
        row
        for row in manifest["files"]
        if row["repo_relative_path"]
        == "framework/stage5/measurement_plan_v1.py"
    )
    record["source_path"] = str(tmp_path / "outside.py")
    manifest_path.write_bytes(bundle._json_bytes(manifest))
    pins = _resign_manifest_and_state(root)

    with pytest.raises(ValueError, match="source/destination drift"):
        bundle.validate_deployment_bundle(
            root,
            frozen_repo_root=source,
            **pins,
        )
