import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "fcooper_tvm_gear_round_watcher_v1.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("fcooper_tvm_gear_round_watcher_v1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FCooperTvmGearRoundWatcherTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def template_manifest(self):
        return {
            "schema_version": "fcooper_tvm_gpu_job_manifest_v1",
            "barrier_order": ["round_02"],
            "gpu_pool": list(range(8)),
            "jobs": [
                {
                    "job_id": f"gear-r02-{index}",
                    "request_json": "/old/round_02/measurement_request.json",
                    "row_index": index,
                    "request_kind": "t16",
                    "max_trials": 64,
                    "priority": 100,
                    "barrier_id": "round_02",
                    "runner_python": "/python",
                    "runner_script": "/runner.py",
                    "runner_common_args": {"artifact-root": "/artifacts"},
                }
                for index in range(4)
            ],
        }

    def test_build_next_manifest_preserves_frozen_runner_contract(self):
        template = self.template_manifest()
        template["manifest_sha256"] = self.module.canonical_sha256(template)
        before = copy.deepcopy(template)
        request = self.root / "round_03" / "measurement_request.json"

        result = self.module.build_next_manifest(
            template,
            round_index=3,
            request_path=request,
        )

        self.assertEqual(template, before)
        self.assertEqual(result["barrier_order"], ["round_03"])
        self.assertEqual(result["gpu_pool"], list(range(8)))
        self.assertEqual(
            [job["job_id"] for job in result["jobs"]],
            [f"gear-r03-{index}" for index in range(4)],
        )
        self.assertTrue(all(job["barrier_id"] == "round_03" for job in result["jobs"]))
        self.assertTrue(
            all(job["request_json"] == str(request.resolve()) for job in result["jobs"])
        )
        self.assertTrue(all(job["max_trials"] == 64 for job in result["jobs"]))
        recorded = result.pop("manifest_sha256")
        self.assertEqual(recorded, self.module.canonical_sha256(result))

    def test_scheduler_audit_requires_exactly_four_successes(self):
        audit = {
            "schema_version": "fcooper_tvm_gpu_scheduler_audit_v1",
            "status_counts": {
                "succeeded": 4,
                "pending": 0,
                "running": 0,
                "failed_infrastructure": 0,
                "failed_terminal": 0,
                "blocked_dependency": 0,
            },
        }

        self.module.validate_scheduler_audit(audit)

        failed = copy.deepcopy(audit)
        failed["status_counts"]["succeeded"] = 3
        failed["status_counts"]["failed_infrastructure"] = 1
        with self.assertRaisesRegex(ValueError, "four successful"):
            self.module.validate_scheduler_audit(failed)

    def test_scheduler_audit_distinguishes_running_from_failed_terminal(self):
        running = {
            "schema_version": "fcooper_tvm_gpu_scheduler_audit_v1",
            "status_counts": {
                "succeeded": 2,
                "pending": 0,
                "running": 2,
                "failed_infrastructure": 0,
                "failed_terminal": 0,
                "blocked_dependency": 0,
            },
        }
        failed = copy.deepcopy(running)
        failed["status_counts"].update(
            {"succeeded": 3, "running": 0, "failed_infrastructure": 1}
        )

        self.assertEqual(self.module.scheduler_audit_state(running), "running")
        self.assertEqual(self.module.scheduler_audit_state(failed), "failed")

    def test_write_manifest_is_idempotent_but_rejects_drift(self):
        path = self.root / "manifest.json"
        manifest = self.template_manifest()
        manifest["manifest_sha256"] = self.module.canonical_sha256(manifest)

        self.module.write_manifest_idempotently(path, manifest)
        first = path.read_bytes()
        self.module.write_manifest_idempotently(path, manifest)
        self.assertEqual(path.read_bytes(), first)

        drifted = copy.deepcopy(manifest)
        drifted["gpu_pool"] = [0]
        payload = {key: value for key, value in drifted.items() if key != "manifest_sha256"}
        drifted["manifest_sha256"] = self.module.canonical_sha256(payload)
        with self.assertRaisesRegex(ValueError, "drift"):
            self.module.write_manifest_idempotently(path, drifted)

    def test_partial_transition_outputs_are_rejected(self):
        paths = [self.root / name for name in ("feedback.json", "audit.json", "history.json")]
        paths[0].write_text(json.dumps({"rows": []}), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "partial"):
            self.module.transition_outputs_complete(paths)

        for path in paths[1:]:
            path.write_text("{}", encoding="utf-8")
        self.assertTrue(self.module.transition_outputs_complete(paths))

    def test_validate_advanced_round_binds_all_five_outputs(self):
        round_dir = self.root / "round_03"
        round_dir.mkdir()
        rows = [{"row_id": f"row-{index}"} for index in range(4)]
        request = {
            "schema_version": "stage5_measurement_request_v2",
            "task_id": "S5-FCO-TVM-V1",
            "round_index": 3,
            "batch_size": 4,
            "rows": rows,
        }
        request["measurement_request_sha256"] = self.module.canonical_sha256(request)
        state = {
            "schema_version": "stage5_fcooper_formal_round_state_v2",
            "task_id": "S5-FCO-TVM-V1",
            "round_index": 3,
            "status": "awaiting_recovered_source_measurement",
            "selected_row_ids": [row["row_id"] for row in rows],
            "measurement_request_sha256": request["measurement_request_sha256"],
        }
        for name, payload in {
            "measurement_request.json": request,
            "round_state.json": state,
            "acquisition.json": {},
            "candidate_manifest.json": {},
            "predicted_candidates.json": {},
        }.items():
            (round_dir / name).write_text(json.dumps(payload), encoding="utf-8")

        self.module.validate_advanced_round(
            round_dir,
            round_index=3,
            task_id="S5-FCO-TVM-V1",
        )

        (round_dir / "candidate_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "partial"):
            self.module.validate_advanced_round(
                round_dir,
                round_index=3,
                task_id="S5-FCO-TVM-V1",
            )

    def test_build_next_manifest_rejects_signed_non_t16_template(self):
        template = self.template_manifest()
        template["jobs"][0]["request_kind"] = "stage6-control"
        template["manifest_sha256"] = self.module.canonical_sha256(template)

        with self.assertRaisesRegex(ValueError, "t16"):
            self.module.build_next_manifest(
                template,
                round_index=3,
                request_path=self.root / "request.json",
            )

    def test_finds_orphan_scheduler_by_manifest_command_line(self):
        proc = self.root / "proc"
        matching = proc / "123"
        unrelated = proc / "456"
        matching.mkdir(parents=True)
        unrelated.mkdir(parents=True)
        manifest = self.root / "manifest.json"
        matching.joinpath("cmdline").write_bytes(
            b"python\0/repo/fcooper_tvm_gpu_scheduler_v1.py\0"
            + str(manifest.resolve()).encode()
            + b"\0"
        )
        unrelated.joinpath("cmdline").write_bytes(b"python\0other.py\0")

        self.assertEqual(
            self.module.find_running_scheduler(
                manifest,
                proc_root=proc,
            ),
            123,
        )


if __name__ == "__main__":
    unittest.main()
