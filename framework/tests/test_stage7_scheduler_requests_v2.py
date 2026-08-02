from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from framework.tests import test_stage7_core_ablation_scheduler_v2 as fixture
from framework.tests import test_stage7_execute_actual_v3_misses_v2 as miss_fixture


def _artifacts(tmp_path: Path):
    module = fixture.load_module()
    root = fixture.v2_root(module, tmp_path)
    batch = fixture.request(module, root, "full:seed_20260718")
    plan = json.loads(batch.physical_plan_path.read_text())
    formal = json.loads(
        (batch.trajectory_path / "source_resolution_result.json").read_text()
    )
    return module._requests, batch, plan, formal


def test_resolved_source_helpers_validate_and_copy_only_physical_misses(
    tmp_path: Path,
) -> None:
    module, batch, plan, formal = _artifacts(tmp_path)

    assert (
        module.validate_resolved_source_lock(batch.source_lock_key, plan, formal)
        == batch.source_lock_key
    )
    bindings = module.resolved_source_bindings(plan, formal)
    assert [row["candidate_id"] for row in bindings] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
    ]
    bindings[0]["candidate_id"] = "mutated-copy"
    assert formal["rows"][0]["candidate_id"] == "row-0"


def test_resolved_source_helpers_fail_closed_on_lock_or_candidate_drift(
    tmp_path: Path,
) -> None:
    module, batch, plan, formal = _artifacts(tmp_path)

    with pytest.raises(ValueError, match="lock SHA drift"):
        module.validate_resolved_source_lock("f" * 64, plan, formal)
    incomplete = {**formal, "rows": formal["rows"][1:]}
    with pytest.raises(ValueError, match="atomic round"):
        module.resolved_source_lock_sha256(plan, incomplete)
    with pytest.raises(ValueError, match="no formal resolved source"):
        module.resolved_source_bindings(plan, incomplete)


def test_v2_measurement_request_runs_real_source_and_task4_validators(
    tmp_path: Path,
) -> None:
    from scripts import stage7_scheduler_requests_v2

    module = importlib.reload(stage7_scheduler_requests_v2)
    artifacts = miss_fixture._real_task4_artifacts(tmp_path / "formal-v2")
    root = Path(artifacts["root"]).resolve()
    round_dir = Path(artifacts["round_dir"]).resolve()
    plan = json.loads(Path(artifacts["plan"]).read_text())
    module.V2_ROOT = root
    request_sha = str(plan["logical_request_sha256"])

    batch = module.V2BatchRequest(
        trajectory_id="full:seed_20260718",
        trajectory_path=round_dir,
        round_index=0,
        request_sha256=request_sha,
        selected_row_ids=tuple(
            binding["candidate_id"] for binding in plan["logical_row_bindings"]
        ),
        command=(
            str(module.CORE_CONTROLLER),
            "--v2-root",
            str(root),
            "--variant",
            "full",
            "--seed",
            "20260718",
            "--round-index",
            "0",
            "--request-sha256",
            request_sha,
            "--expected-release-sha256",
            fixture.EXPECTED_RELEASE_SHA256,
            "--expected-manifest-file-sha256",
            fixture.EXPECTED_MANIFEST_FILE_SHA256,
        ),
        width=(16, 32, 64),
        expected_release_sha256=fixture.EXPECTED_RELEASE_SHA256,
        expected_manifest_file_sha256=fixture.EXPECTED_MANIFEST_FILE_SHA256,
        source_lock_key=str(artifacts["resolved_source_lock_sha"]),
        physical_plan_path=Path(artifacts["plan"]),
        physical_request_sha256=str(plan["physical_request_sha256"]),
    )

    assert batch.source_resolution_result_sha256 == artifacts["source_result_sha"]


def test_v2_request_rejects_obsolete_output_root_controller_contract(
    tmp_path: Path,
) -> None:
    module, batch, _plan, _formal = _artifacts(tmp_path)

    obsolete = (
        batch.command[0],
        "--output-root",
        *batch.command[2:],
    )
    with pytest.raises(ValueError, match="exact v2 controller argv"):
        fixture.replace(batch, command=obsolete)


def test_core_controller_path_resolves_relative_to_deployment_overlay(
    tmp_path: Path,
) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "stage7_scheduler_requests_v2.py"
    )
    overlay_scripts = tmp_path / "deployment" / "code" / "scripts"
    overlay_scripts.mkdir(parents=True)
    overlay_module = overlay_scripts / source.name
    overlay_module.write_bytes(source.read_bytes())
    controller = overlay_scripts / "stage7_core_round_controller_v2.sh"
    controller.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    module_name = "stage7_scheduler_requests_v2_overlay_test"
    spec = importlib.util.spec_from_file_location(module_name, overlay_module)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = loaded
    try:
        spec.loader.exec_module(loaded)
    finally:
        sys.modules.pop(module_name, None)

    assert loaded.CORE_CONTROLLER == controller.resolve()
