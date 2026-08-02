import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "scripts"
    / "stage6_export_tvm_fp32_schedule_artifact_v1.py"
)
SPEC = importlib.util.spec_from_file_location(
    "stage6_export_tvm_fp32_schedule_artifact_v1",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_skip_database_does_not_require_tuning_files(tmp_path: Path) -> None:
    database = MODULE.resolve_schedule_database(tmp_path, skip_database=True)

    assert database is None


def test_tuned_export_requires_both_database_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MODULE.resolve_schedule_database(tmp_path, skip_database=False)

    workload = tmp_path / "database_workload.json"
    records = tmp_path / "database_tuning_record.json"
    workload.write_text("{}\n", encoding="utf-8")
    records.write_text("{}\n", encoding="utf-8")

    database = MODULE.resolve_schedule_database(tmp_path, skip_database=False)

    assert database == (workload, records)
