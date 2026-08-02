from __future__ import annotations

import importlib.util
import csv
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_gold128_merge_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage35_gold128_merge_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset(prefix: str, groups: int, locked_groups: int) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    jobs: list[dict] = []
    for group_index in range(groups):
        model = "pyramid" if group_index % 2 == 0 else "codriving"
        width = [16 + group_index, 32, 64]
        group_id = f"{model}|{prefix}{group_index}"
        split = "locked_holdout" if group_index >= groups - locked_groups else "train"
        for backend, q_mode in (("tvm_auto", "fp16"), ("tvm_auto", "int8"), ("trt_engine", "fp16"), ("trt_engine", "int8")):
            profile = "h800-tvm-probe-conditioned-v3" if backend == "tvm_auto" else "h800-trt-probe-conditioned-v3"
            job_id = f"{group_id}|q={q_mode}|profile={profile}"
            rows.append({
                "schema_version": f"{prefix}_final",
                "manifest_job_id": job_id,
                "group_id": group_id,
                "model": model,
                "width": width,
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": backend,
                "terminal_status": "measured_success_gold",
                "latency_ms": 1.0,
                "energy_j": 0.2,
                "ap30": 0.8,
                "ap50": 0.7,
                "ap70": 0.6,
                "performance_result_sha256": "a" * 64,
                "ap_report_sha256": "b" * 64,
            })
            jobs.append({
                "job_id": job_id,
                "group_id": group_id,
                "model": model,
                "width": width,
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": backend,
                "split": split,
                "width_stratum": "cross_model_common_anchor" if prefix == "g32" else "alignment_group",
            })
    manifest = {
        "schema_version": f"{prefix}_manifest",
        "jobs": jobs,
        "pilot_group_ids": [f"pyramid|{prefix}0", f"codriving|{prefix}1"],
    }
    return rows, manifest


class Stage35Gold128MergeV1Tests(unittest.TestCase):
    def test_merge_preserves_grouped_split_and_source_pool(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)

        result = module.merge_gold128(gold96, manifest96, gold32, manifest32)

        self.assertEqual(result["audit"]["rows"], 128)
        self.assertEqual(result["audit"]["groups"], 32)
        self.assertEqual(result["audit"]["split_groups"], {"train": 26, "locked_holdout": 6})
        self.assertEqual(result["audit"]["source_rows"], {"gold96": 96, "gold32": 32})
        self.assertEqual({row["schema_version"] for row in result["rows"]}, {"stage35_gold128_final_v1"})
        self.assertEqual({row["source_pool"] for row in result["rows"]}, {"gold96", "gold32"})
        self.assertTrue(all(row["split"] in {"train", "locked_holdout"} for row in result["rows"]))
        self.assertEqual(result["manifest"]["pilot_group_ids"], manifest96["pilot_group_ids"])

    def test_merge_rejects_overlapping_groups(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)
        overlap = manifest96["jobs"][0]["group_id"]
        for row in gold32[:4]:
            row["group_id"] = overlap
        for row in manifest32["jobs"][:4]:
            row["group_id"] = overlap

        with self.assertRaisesRegex(ValueError, "overlap"):
            module.merge_gold128(gold96, manifest96, gold32, manifest32)

    def test_merge_rejects_row_manifest_binding_mismatch(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)
        gold96[0] = {**gold96[0], "group_id": manifest96["jobs"][-1]["group_id"]}

        with self.assertRaisesRegex(ValueError, "row/manifest binding mismatch"):
            module.merge_gold128(gold96, manifest96, gold32, manifest32)

    def test_merge_rejects_locked_holdout_feasibility_failure(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)
        locked_id = next(
            job["job_id"] for job in manifest32["jobs"] if job["split"] == "locked_holdout"
        )
        row = next(item for item in gold32 if item["manifest_job_id"] == locked_id)
        row["terminal_status"] = "feasibility_failure"

        with self.assertRaisesRegex(ValueError, "locked holdout"):
            module.merge_gold128(gold96, manifest96, gold32, manifest32)

    def test_merge_rejects_per_model_split_imbalance(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)
        locked_pyramid = next(
            job["group_id"] for job in manifest96["jobs"]
            if job["split"] == "locked_holdout" and job["model"] == "pyramid"
        )
        train_codriving = next(
            job["group_id"] for job in manifest96["jobs"]
            if job["split"] == "train" and job["model"] == "codriving"
        )
        for job in manifest96["jobs"]:
            if job["group_id"] == locked_pyramid:
                job["split"] = "train"
            elif job["group_id"] == train_codriving:
                job["split"] = "locked_holdout"

        with self.assertRaisesRegex(ValueError, "per-model split"):
            module.merge_gold128(gold96, manifest96, gold32, manifest32)

    def test_write_outputs_emits_json_jsonl_csv_manifest_and_audit(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)
        result = module.merge_gold128(gold96, manifest96, gold32, manifest32)
        with tempfile.TemporaryDirectory() as directory:
            module.write_outputs(result, directory)
            names = {path.name for path in Path(directory).iterdir()}
            self.assertEqual(names, {
                "gold128_final.json", "gold128_final.jsonl", "gold128_final.csv",
                "gold128_manifest.json", "gold128_audit.json",
            })
            audit = json.loads((Path(directory) / "gold128_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["rows"], 128)

    def test_write_outputs_preserves_fields_present_only_in_later_rows(self) -> None:
        module = _load_module()
        gold96, manifest96 = _dataset("g96", 24, 4)
        gold32, manifest32 = _dataset("g32", 8, 2)
        gold32[0] = {
            **gold32[0],
            "ap_checkpoint_epoch": 31,
            "ap_label_repair_schema": "stage35_gold128_ap_label_repair_v2",
        }
        result = module.merge_gold128(gold96, manifest96, gold32, manifest32)

        with tempfile.TemporaryDirectory() as directory:
            module.write_outputs(result, directory)
            with (Path(directory) / "gold128_final.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[96]["ap_checkpoint_epoch"], "31")
        self.assertEqual(
            rows[96]["ap_label_repair_schema"],
            "stage35_gold128_ap_label_repair_v2",
        )


if __name__ == "__main__":
    unittest.main()
