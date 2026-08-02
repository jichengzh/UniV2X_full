from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "stage35_finalize_gold32_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage35_finalize_gold32_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    jobs = []
    for group_index in range(8):
        model = "pyramid" if group_index < 4 else "codriving"
        width = [16 + group_index, 32, 64]
        group_id = f"{model}|{'x'.join(map(str, width))}"
        for backend, q_mode in (("tvm", "fp16"), ("tvm", "int8"), ("trt", "fp16"), ("trt", "int8")):
            profile = f"h800-{backend}-probe-conditioned-v3"
            jobs.append({
                "job_id": f"{group_id}|q={q_mode}|profile={profile}",
                "group_id": group_id,
                "model": model,
                "width": width,
                "q_mode": q_mode,
                "capability_profile_id": profile,
                "dispatch_key": "tvm_auto" if backend == "tvm" else "trt_engine",
            })
    return {"schema_version": "stage35_gold32_supplement_manifest_v1", "jobs": jobs}


class Stage35FinalizeGold32V1Tests(unittest.TestCase):
    def test_cli_help_runs_from_repo_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validates_32_rows_and_writes_gold32_named_outputs(self) -> None:
        module = _load_module()
        manifest = _manifest()
        plan = [
            {"manifest_job_id": row["job_id"], "performance_job_id": f"perf-{index}"}
            for index, row in enumerate(manifest["jobs"])
        ]

        result = module.finalize_gold32(
            manifest,
            ap_plan_rows=plan,
            performance_state_rows=[],
            ap_state_rows=[],
        )

        self.assertEqual(result["summary"], {"measured": 0, "failure": 0, "pending": 32, "total": 32})
        self.assertEqual(len(result["group_audit"]), 8)
        self.assertTrue(all(row["schema_version"] == "stage35_gold32_final_v1" for row in result["rows"]))
        with tempfile.TemporaryDirectory() as tmp:
            module.write_gold32_outputs(result, tmp)
            output = Path(tmp)
            self.assertTrue((output / "gold32_final.json").is_file())
            self.assertTrue((output / "gold32_final.jsonl").is_file())
            self.assertTrue((output / "gold32_final.csv").is_file())
            audit = json.loads((output / "gold32_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["schema_version"], "stage35_gold32_final_v1")

    def test_rejects_incomplete_manifest(self) -> None:
        module = _load_module()
        manifest = _manifest()
        manifest["jobs"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 32 jobs"):
            module.finalize_gold32(
                manifest,
                ap_plan_rows=[],
                performance_state_rows=[],
                ap_state_rows=[],
            )


if __name__ == "__main__":
    unittest.main()
