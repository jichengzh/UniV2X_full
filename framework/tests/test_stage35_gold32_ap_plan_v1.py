from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/stage35_gold32_ap_plan_v1.py"
ROOT = REPO_ROOT / "results/stage35_gold32_supplement_v1_20260713"
MANIFEST = ROOT / "repair_v1/gold32_supplement_manifest_pyramid_shape_repaired_v1.json"
JOB_FILES = (
    ROOT / "performance_plans/gold32_performance_batch_01_jobs.jsonl",
    ROOT / "repair_v1/performance_plans/gold32_performance_batch_02_jobs.jsonl",
)


def _module():
    spec = importlib.util.spec_from_file_location("stage35_gold32_ap_plan_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Stage35Gold32ApPlanV1Tests(unittest.TestCase):
    def test_builds_32_ready_rows_using_manifest_source_contracts(self) -> None:
        module = _module()
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        jobs = [row for path in JOB_FILES for row in _read_jsonl(path)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            states = []
            for index, job in enumerate(jobs):
                artifact = root / (
                    f"artifact-{index}.engine" if job["runner_key"].startswith("trt_")
                    else f"artifact-{index}.so" if job["runner_key"] == "tvm_fp16"
                    else f"route-{index}/route_b_int8_auto_decomp.vmexec"
                )
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"artifact")
                result = root / f"result-{index}.json"
                result.write_text(json.dumps({"artifact_path": str(artifact)}), encoding="utf-8")
                states.append({"job_id": job["job_id"], "status": "success", "result_json": str(result)})

            rows = module.build_gold32_ap_plan(
                manifest,
                performance_jobs=jobs,
                performance_state_rows=states,
                output_root=root / "ap",
            )

        self.assertEqual(len(rows), 32)
        self.assertEqual({row["ap_terminal"] for row in rows}, {"ready"})
        by_id = {row["manifest_job_id"]: row for row in rows}
        source_by_id = {row["job_id"]: row for row in manifest["jobs"]}
        self.assertTrue(all(
            row["source_contract"] == source_by_id[row["manifest_job_id"]]["source_contract"]
            for row in rows
        ))

        pyramid_id = "pyramid|24x64x128|q=fp16|profile=h800-trt-probe-conditioned-v3"
        pyramid_source = next(row for row in manifest["jobs"] if row["job_id"] == pyramid_id)
        pyramid_command = by_id[pyramid_id]["sanity_command"]
        self.assertEqual(
            pyramid_command[pyramid_command.index("--ckpt-dir") + 1],
            pyramid_source["source_contract"]["checkpoint_dir"],
        )

        codriving_id = "codriving|16x64x128|q=fp16|profile=h800-tvm-probe-conditioned-v3"
        codriving_source = next(row for row in manifest["jobs"] if row["job_id"] == codriving_id)
        codriving_command = by_id[codriving_id]["sanity_command"]
        self.assertEqual(
            codriving_command[codriving_command.index("--model-dir") + 1],
            codriving_source["source_contract"]["model_dir"],
        )


if __name__ == "__main__":
    unittest.main()
