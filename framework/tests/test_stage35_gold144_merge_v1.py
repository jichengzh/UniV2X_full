from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_gold144_merge_v1.py"
ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)
LOCKED_FIELDS = (
    "manifest_job_id",
    "latency_ms",
    "energy_j",
    "ap30",
    "ap50",
    "ap70",
    "performance_result_sha256",
    "ap_report_sha256",
)
GOLD128_SHA256 = "c" * 64


def _load_module():
    spec = importlib.util.spec_from_file_location("stage35_gold144_merge_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool(*, groups: int, locked_groups: int, targeted: bool) -> tuple[list[dict], dict]:
    rows = []
    jobs = []
    prefix = "targeted" if targeted else "gold128"
    for group_index in range(groups):
        model = "pyramid" if targeted or group_index % 2 == 0 else "codriving"
        group_id = f"{model}|{prefix}-{group_index}"
        split = "locked_holdout" if group_index >= groups - locked_groups else "train"
        for dispatch_key, q_mode in ARMS:
            backend = "tvm" if dispatch_key == "tvm_auto" else "trt"
            profile = f"h800-{backend}-probe-conditioned-v3"
            job_id = f"{group_id}|q={q_mode}|profile={profile}"
            row = {
                "schema_version": (
                    "stage35_targeted16_final_v1" if targeted else "stage35_gold128_final_v1"
                ),
                "manifest_job_id": job_id,
                "group_id": group_id,
                "model": model,
                "width": [16 + group_index, 32, 96],
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": dispatch_key,
                "terminal_status": "measured_success_gold",
                "latency_ms": 4.0 + group_index / 10,
                "energy_j": 0.5,
                "ap30": 0.8,
                "ap50": 0.7,
                "ap70": 0.6,
                "performance_result_sha256": "a" * 64,
                "ap_report_sha256": "b" * 64,
                "split": split,
                "source_pool": prefix,
                "width_stratum": "targeted_failure_boundary" if targeted else "alignment_group",
            }
            rows.append(row)
            jobs.append({
                "schema_version": (
                    "stage35_gold128_targeted_supplement_manifest_v1"
                    if targeted
                    else "stage35_gold128_manifest_v1"
                ),
                "job_id": job_id,
                "group_id": group_id,
                "model": model,
                "width": row["width"],
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": dispatch_key,
                "split": split,
                "width_stratum": row["width_stratum"],
            })
    manifest = {
        "schema_version": (
            "stage35_gold128_targeted_supplement_manifest_v1"
            if targeted
            else "stage35_gold128_manifest_v1"
        ),
        "jobs": jobs,
    }
    if not targeted:
        manifest["pilot_group_ids"] = ["pyramid|gold128-0", "codriving|gold128-1"]
    return rows, manifest


def _locked_snapshot(rows: list[dict]) -> dict[str, tuple]:
    return {
        row["manifest_job_id"]: tuple(row[field] for field in LOCKED_FIELDS)
        for row in rows
        if row["split"] == "locked_holdout"
    }


def _merge(module, gold128: list[dict], manifest128: dict, targeted16: list[dict], manifest16: dict):
    return module.merge_gold144(
        gold128,
        manifest128,
        targeted16,
        manifest16,
        gold128_sha256=GOLD128_SHA256,
    )


class Stage35Gold144MergeV1Tests(unittest.TestCase):
    def test_cli_help_runs_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_merges_144_rows_and_preserves_all_locked_values(self) -> None:
        module = _load_module()
        gold128, manifest128 = _pool(groups=32, locked_groups=6, targeted=False)
        targeted16, manifest16 = _pool(groups=4, locked_groups=0, targeted=True)
        before = _locked_snapshot(gold128)

        result = _merge(module, gold128, manifest128, targeted16, manifest16)

        self.assertEqual(result["audit"]["rows"], 144)
        self.assertEqual(result["audit"]["groups"], 36)
        self.assertEqual(result["audit"]["split_groups"], {"train": 30, "locked_holdout": 6})
        self.assertEqual(result["audit"]["source_rows"], {"gold128": 128, "targeted16": 16})
        self.assertEqual(len(result["manifest"]["jobs"]), 144)
        self.assertEqual(result["manifest"]["pilot_group_ids"], manifest128["pilot_group_ids"])
        self.assertEqual(
            result["manifest"]["source_manifests"]["gold128_sha256"], GOLD128_SHA256
        )
        self.assertEqual(_locked_snapshot(result["rows"]), before)
        self.assertEqual({row["schema_version"] for row in result["rows"]}, {"stage35_gold144_final_v1"})

    def test_rejects_group_overlap_and_duplicate_manifest_id(self) -> None:
        module = _load_module()
        for failure in ("overlap", "duplicate"):
            gold128, manifest128 = _pool(groups=32, locked_groups=6, targeted=False)
            targeted16, manifest16 = _pool(groups=4, locked_groups=0, targeted=True)
            if failure == "overlap":
                old_group = targeted16[0]["group_id"]
                new_group = gold128[0]["group_id"]
                for row in targeted16:
                    if row["group_id"] == old_group:
                        row["group_id"] = new_group
                for job in manifest16["jobs"]:
                    if job["group_id"] == old_group:
                        job["group_id"] = new_group
                message = "overlap"
            else:
                duplicate = gold128[0]["manifest_job_id"]
                targeted16[0]["manifest_job_id"] = duplicate
                manifest16["jobs"][0]["job_id"] = duplicate
                message = "duplicate"
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, message):
                _merge(module, gold128, manifest128, targeted16, manifest16)

    def test_rejects_incomplete_group_and_non_final_targeted_row(self) -> None:
        module = _load_module()
        for failure in ("incomplete", "non-final"):
            gold128, manifest128 = _pool(groups=32, locked_groups=6, targeted=False)
            targeted16, manifest16 = _pool(groups=4, locked_groups=0, targeted=True)
            if failure == "incomplete":
                targeted16.pop()
                manifest16["jobs"].pop()
                message = "exactly 16|four-arm"
            else:
                targeted16[0]["terminal_status"] = "pending_ap"
                message = "non-terminal|non-final|measured_success_gold"
            with self.subTest(failure=failure), self.assertRaisesRegex(ValueError, message):
                _merge(module, gold128, manifest128, targeted16, manifest16)

    def test_rejects_any_holdout_split_change(self) -> None:
        module = _load_module()
        gold128, manifest128 = _pool(groups=32, locked_groups=6, targeted=False)
        targeted16, manifest16 = _pool(groups=4, locked_groups=0, targeted=True)
        locked_group = next(job["group_id"] for job in manifest128["jobs"] if job["split"] == "locked_holdout")
        for job in manifest128["jobs"]:
            if job["group_id"] == locked_group:
                job["split"] = "train"

        with self.assertRaisesRegex(ValueError, "holdout|split|binding"):
            _merge(module, gold128, manifest128, targeted16, manifest16)

    def test_rejects_duplicate_id_inside_gold128(self) -> None:
        module = _load_module()
        gold128, manifest128 = _pool(groups=32, locked_groups=6, targeted=False)
        targeted16, manifest16 = _pool(groups=4, locked_groups=0, targeted=True)
        gold128[1]["manifest_job_id"] = gold128[0]["manifest_job_id"]
        manifest128["jobs"][1]["job_id"] = manifest128["jobs"][0]["job_id"]

        with self.assertRaisesRegex(ValueError, "duplicate"):
            _merge(module, gold128, manifest128, targeted16, manifest16)

    def test_writes_all_outputs_and_csv_uses_row_key_union(self) -> None:
        module = _load_module()
        gold128, manifest128 = _pool(groups=32, locked_groups=6, targeted=False)
        targeted16, manifest16 = _pool(groups=4, locked_groups=0, targeted=True)
        targeted16[-1]["later_only"] = "kept"
        result = _merge(module, gold128, manifest128, targeted16, manifest16)

        with tempfile.TemporaryDirectory() as directory:
            module.write_outputs(result, directory)
            output = Path(directory)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {
                    "gold144_final.json",
                    "gold144_final.jsonl",
                    "gold144_final.csv",
                    "gold144_manifest.json",
                    "gold144_audit.json",
                },
            )
            with (output / "gold144_final.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            audit = json.loads((output / "gold144_audit.json").read_text(encoding="utf-8"))

        self.assertEqual(csv_rows[-1]["later_only"], "kept")
        self.assertEqual(audit["rows"], 144)


if __name__ == "__main__":
    unittest.main()
