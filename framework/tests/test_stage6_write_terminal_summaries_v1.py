import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts/stage6_write_terminal_summaries_v1.py"
SPEC = importlib.util.spec_from_file_location("stage6_write_terminal_summaries_v1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _bundle() -> dict:
    arms = {
        arm: {
            "status": "complete",
            "independent_validation_complete": True,
            "points": [{}],
            "failure_count": 0,
        }
        for arm in MODULE.ARMS
    }
    arms["tune_then_compress"] = {
        "status": "complete_failure",
        "failure_evidence_sha_verified": True,
        "points": [],
        "failure_count": 16,
    }
    return {
        "schema_version": "stage6_paper_evidence_bundle_v1",
        "baseline": {"independent_repeat_count": 3},
        "backends": {"tvm": arms, "trt": arms},
    }


def test_build_terminals_requires_all_eleven_slots_to_be_auditable(tmp_path) -> None:
    path = tmp_path / "bundle.json"
    payload = _bundle()
    path.write_text(json.dumps(payload), encoding="utf-8")

    terminals = MODULE.build_terminals(payload, path)

    assert len(terminals) == 11
    assert terminals[("tvm", "tune_then_compress")]["status"] == "complete_failure"
    assert terminals[("trt", "joint_shcosearch")]["status"] == "complete"


def test_build_terminals_rejects_unvalidated_success(tmp_path) -> None:
    path = tmp_path / "bundle.json"
    payload = _bundle()
    payload["backends"]["tvm"]["compression_only"] = {
        "status": "complete",
        "independent_validation_complete": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="independent validation incomplete"):
        MODULE.build_terminals(payload, path)
