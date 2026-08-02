from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/stage35_gold128_targeted_ap_plan_v1.py"
ROOT = REPO_ROOT / "results/stage35_gold128_targeted_supplement_v1_20260714"
MANIFEST = ROOT / "plan/targeted_supplement_manifest.json"
PERFORMANCE_JOBS = ROOT / "plan/targeted_performance_jobs.jsonl"


def _module():
    spec = importlib.util.spec_from_file_location("stage35_gold128_targeted_ap_plan_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _successful_performance_states(jobs: list[dict], root: Path) -> list[dict]:
    states: list[dict] = []
    for index, job in enumerate(jobs):
        if job["runner_key"].startswith("trt_"):
            artifact = root / f"artifact-{index}.engine"
        elif job["runner_key"] == "tvm_fp16":
            artifact = root / f"artifact-{index}.so"
        else:
            artifact = root / f"route-{index}/route_b_int8_auto_decomp.vmexec"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"artifact")
        result = root / f"result-{index}.json"
        result.write_text(json.dumps({"artifact_path": str(artifact)}), encoding="utf-8")
        states.append({"job_id": job["job_id"], "status": "success", "result_json": str(result)})
    return states


class Stage35Gold128TargetedApPlanV1Tests(unittest.TestCase):
    def test_builds_exactly_four_complete_ready_groups(self) -> None:
        module = _module()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        jobs = _read_jsonl(PERFORMANCE_JOBS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = module.build_targeted_ap_plan(
                manifest,
                performance_jobs=jobs,
                performance_state_rows=_successful_performance_states(jobs, root),
                output_root=root / "ap",
            )

        self.assertEqual(len(rows), 16)
        self.assertEqual({row["ap_terminal"] for row in rows}, {"ready"})
        self.assertEqual(
            [row["manifest_job_id"] for row in rows],
            [row["job_id"] for row in manifest["jobs"]],
        )
        grouped: dict[str, set[tuple[str, str]]] = {}
        source_by_id = {row["job_id"]: row for row in manifest["jobs"]}
        for row in rows:
            source = source_by_id[row["manifest_job_id"]]
            grouped.setdefault(source["group_id"], set()).add(
                (source["dispatch_key"], source["q_mode"])
            )
            self.assertEqual(source["split"], "train")
            self.assertEqual(row["source_contract"], source["source_contract"])
        expected_arms = {
            ("tvm_auto", "fp16"),
            ("tvm_auto", "int8"),
            ("trt_engine", "fp16"),
            ("trt_engine", "int8"),
        }
        self.assertEqual(len(grouped), 4)
        self.assertTrue(all(arms == expected_arms for arms in grouped.values()))

    def test_fails_fast_when_any_performance_row_is_not_successful(self) -> None:
        module = _module()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        jobs = _read_jsonl(PERFORMANCE_JOBS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = _successful_performance_states(jobs, root)
            states[-1] = {"job_id": jobs[-1]["job_id"], "status": "confirmed_failure"}
            with self.assertRaisesRegex(ValueError, "all 16 targeted AP rows must be ready"):
                module.build_targeted_ap_plan(
                    manifest,
                    performance_jobs=jobs,
                    performance_state_rows=states,
                    output_root=root / "ap",
                )

    def test_rejects_non_train_or_incomplete_four_arm_group(self) -> None:
        module = _module()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        bad_manifest = {**manifest, "jobs": [dict(row) for row in manifest["jobs"]]}
        bad_manifest["jobs"][0]["split"] = "locked_holdout"
        with self.assertRaisesRegex(ValueError, "targeted supplement rows must all be train"):
            module.validate_targeted_manifest(bad_manifest)

        incomplete = {**manifest, "jobs": [dict(row) for row in manifest["jobs"][:-1]]}
        with self.assertRaisesRegex(ValueError, "complete four-arm groups"):
            module.validate_targeted_manifest(incomplete)


if __name__ == "__main__":
    unittest.main()
