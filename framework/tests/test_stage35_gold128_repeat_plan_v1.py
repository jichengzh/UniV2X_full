from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold128_repeat_plan_v1 as repeat_plan  # noqa: E402


def _source_jobs() -> list[dict]:
    rows = []
    for category, groups in repeat_plan.ANCHOR_GROUPS.items():
        for group_id in groups:
            model, width = group_id.split("|")
            for runner, backend, q_mode in (
                ("tvm_fp16", "tvm_auto", "fp16"),
                ("tvm_int8", "tvm_auto", "int8"),
                ("trt_fp16", "trt_engine", "fp16"),
                ("trt_int8", "trt_engine", "int8"),
            ):
                old_root = f"/old/{group_id}"
                result = f"{old_root}/{runner}/result.json"
                rows.append({
                    "job_id": f"{group_id}|{runner}",
                    "manifest_job_id": f"{group_id}|q={q_mode}|profile={backend}",
                    "group_id": group_id,
                    "model": model,
                    "width_key": width,
                    "q_mode": q_mode,
                    "dispatch_key": backend,
                    "runner_key": runner,
                    "remote_artifact_root": old_root,
                    "expected_result_json": result,
                    "command": [
                        "python3", "runner.py", "--label", f"{model}_{width}",
                        "--out-dir", f"{old_root}/{runner}", "--calib", f"/calib/{group_id}",
                    ],
                })
    return rows


def _gold_rows(jobs: list[dict]) -> list[dict]:
    return [{
        "manifest_job_id": row["manifest_job_id"],
        "performance_result_json": (
            f"/gold/repair/{row['model']}/{row['width_key']}/build/"
            f"{row['model']}_{row['width_key']}_scaleaware/route_b_int8_auto_decomp_result.json"
            if row["runner_key"] == "tvm_int8"
            else f"/baseline/{row['job_id']}.json"
        ),
    } for row in jobs]


class Stage35Gold128RepeatPlanV1Tests(unittest.TestCase):
    def test_builds_32_new_jobs_without_reusing_measurement_outputs(self) -> None:
        gold = _gold_rows(_source_jobs())
        output_root = "/new/repeat"
        rows = repeat_plan.build_repeat_plan(_source_jobs(), gold, output_root=output_root, gpu=6)

        self.assertEqual(len(rows), 32)
        self.assertEqual(len({row["job_id"] for row in rows}), 32)
        self.assertEqual({row["repeat_category"] for row in rows}, set(repeat_plan.ANCHOR_GROUPS))
        self.assertTrue(all(row["expected_result_json"].startswith(output_root) for row in rows))
        self.assertTrue(all("/old/" not in " ".join(row["command"]) for row in rows))
        self.assertTrue(all("/calib/" in " ".join(row["command"]) for row in rows))
        self.assertTrue(all(row["assigned_gpu"] == 6 for row in rows))
        self.assertTrue(all(row["baseline_result_json"].startswith(("/baseline/", "/gold/repair/")) for row in rows))

    def test_tvm_int8_repeat_preserves_repaired_scaleaware_contract(self) -> None:
        jobs = _source_jobs()
        gold = _gold_rows(jobs)

        rows = repeat_plan.build_repeat_plan(jobs, gold, output_root="/new/repeat", gpu=6)
        tvm_int8 = [row for row in rows if row["runner_key"] == "tvm_int8"]

        self.assertEqual(len(tvm_int8), 8)
        for row in tvm_int8:
            command = row["command"]
            self.assertTrue(command[command.index("--label") + 1].endswith("_scaleaware"))
            quant_path = command[command.index("--tensor-quant-params-json") + 1]
            self.assertEqual(
                quant_path,
                str(Path(row["baseline_result_json"]).parents[2] / "tensor_quant_params.json"),
            )
            self.assertEqual(command[command.index("--warmup") + 1], "20")
            self.assertEqual(command[command.index("--number") + 1], "20")
            self.assertEqual(command[command.index("--repeat") + 1], "5")

    def test_rejects_missing_anchor_arm(self) -> None:
        jobs = _source_jobs()[:-1]
        gold = _gold_rows(jobs)
        with self.assertRaisesRegex(ValueError, "four-arm"):
            repeat_plan.build_repeat_plan(jobs, gold, output_root="/new", gpu=6)


if __name__ == "__main__":
    unittest.main()
