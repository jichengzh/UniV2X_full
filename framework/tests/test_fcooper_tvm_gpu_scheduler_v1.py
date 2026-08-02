import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/fcooper_tvm_gpu_scheduler_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fcooper_tvm_gpu_scheduler", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeProcess:
    next_pid = 1000

    def __init__(self, command, **_kwargs):
        self.command = list(command)
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None

    def poll(self):
        return self.returncode


class FakePopen:
    def __init__(self):
        self.processes = []

    def __call__(self, command, **kwargs):
        process = FakeProcess(command, **kwargs)
        self.processes.append(process)
        return process


class MutableGpuProbe:
    def __init__(self, module, free):
        self.module = module
        self.free = set(free)

    def __call__(self):
        return {
            gpu: self.module.GpuStatus(
                index=gpu,
                uuid=f"GPU-{gpu}",
                memory_used_mib=0 if gpu in self.free else 10000,
                memory_total_mib=80000,
                utilization_percent=0 if gpu in self.free else 80,
                compute_pids=() if gpu in self.free else (9000 + gpu,),
            )
            for gpu in range(8)
        }


class FCooperTvmGpuSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.schedulers = []
        self.request = self.root / "request.json"
        self._write_request()

    def tearDown(self):
        for scheduler in self.schedulers:
            for handle in scheduler.log_handles.values():
                handle.close()
        self.temporary.cleanup()

    def _write_request(self):
        rows = [
            {"row_id": f"row-{index}", "manifest_job_id": f"row-{index}"}
            for index in range(4)
        ]
        self.request.write_text(json.dumps({"rows": rows, "batch_size": 4}))

    def job(
        self,
        job_id,
        row_index,
        *,
        kind="stage6-control",
        priority=10,
        barrier_id=None,
        release_marker=None,
    ):
        job = {
            "job_id": job_id,
            "request_json": str(self.request),
            "row_index": row_index,
            "request_kind": kind,
            "max_trials": 64,
            "priority": priority,
            "barrier_id": barrier_id,
            "runner_python": "/usr/bin/python3",
            "runner_script": "/repo/scripts/fcooper_execute_tvm_measurement_row_v1.py",
            "runner_common_args": {
                "artifact-root": str(self.root / "artifacts"),
                "code-root": "/repo",
                "heal-root": "/heal",
            },
        }
        if release_marker is not None:
            job["gpu_release_marker"] = str(release_marker)
        return job

    def manifest(self, jobs, barrier_order=None):
        payload = {
            "schema_version": "fcooper_tvm_gpu_job_manifest_v1",
            "gpu_pool": list(range(8)),
            "barrier_order": barrier_order or [],
            "jobs": jobs,
        }
        payload["manifest_sha256"] = self.module.canonical_sha256(payload)
        path = self.root / "jobs.json"
        path.write_text(json.dumps(payload))
        return path

    def scheduler(self, manifest, probe, popen, **kwargs):
        scheduler = self.module.Scheduler(
            manifest_path=manifest,
            state_path=self.root / "state.json",
            events_path=self.root / "events.jsonl",
            audit_path=self.root / "audit.json",
            gpu_probe=probe,
            popen_factory=popen,
            monotonic=lambda: 100.0,
            wall_time=lambda: 1_700_000_000.0,
            **kwargs,
        )
        self.schedulers.append(scheduler)
        return scheduler

    def test_leases_only_free_gpus_and_prioritizes_gear(self):
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0, 2})
        manifest = self.manifest(
            [
                self.job("control", 0, priority=10),
                self.job("gear", 1, kind="t16", priority=100, barrier_id="round-0"),
            ],
            barrier_order=["round-0"],
        )

        scheduler = self.scheduler(manifest, probe, popen)
        scheduler.tick()

        self.assertEqual(len(popen.processes), 2)
        self.assertEqual(scheduler.state["jobs"]["gear"]["gpu"], 0)
        self.assertEqual(scheduler.state["jobs"]["control"]["gpu"], 2)
        self.assertNotIn(1, scheduler.leased_gpus())
        gear_command = popen.processes[0].command
        self.assertEqual(gear_command[gear_command.index("--gpu") + 1], "0")

    def test_fake_nvidia_smi_reports_compute_occupancy(self):
        class Result:
            def __init__(self, stdout):
                self.stdout = stdout

        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command)
            if "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu" in command:
                return Result("0, GPU-a, 0, 80000, 0\n1, GPU-b, 200, 80000, 0\n")
            return Result("GPU-b, 321\n")

        statuses = self.module.query_nvidia_smi(run_command=fake_run)

        self.assertEqual(len(calls), 2)
        self.assertTrue(
            self.module.gpu_is_free(
                statuses[0],
                memory_threshold_mib=1024,
                utilization_threshold_percent=5,
            )
        )
        self.assertFalse(
            self.module.gpu_is_free(
                statuses[1],
                memory_threshold_mib=1024,
                utilization_threshold_percent=5,
            )
        )

    def test_completion_refills_gpu_without_waiting_for_poll_interval(self):
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0})
        manifest = self.manifest(
            [self.job("first", 0, priority=20), self.job("second", 1, priority=10)]
        )
        scheduler = self.scheduler(manifest, probe, popen)
        scheduler.tick()
        self.assertEqual(len(popen.processes), 1)

        popen.processes[0].returncode = 0
        scheduler.tick()

        self.assertEqual(scheduler.state["jobs"]["first"]["status"], "succeeded")
        self.assertEqual(len(popen.processes), 2)
        self.assertEqual(scheduler.state["jobs"]["second"]["gpu"], 0)

    def test_gear_barrier_blocks_later_round_but_not_controls(self):
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0, 2, 3})
        manifest = self.manifest(
            [
                self.job("r0", 0, kind="t16", priority=100, barrier_id="round-0"),
                self.job("r1", 1, kind="t16", priority=100, barrier_id="round-1"),
                self.job("control", 2, priority=10),
            ],
            barrier_order=["round-0", "round-1"],
        )
        scheduler = self.scheduler(manifest, probe, popen)
        scheduler.tick()

        self.assertTrue(
            all(
                item.command[item.command.index("--request-json") + 1]
                == str(self.request)
                for item in popen.processes
            )
        )
        self.assertEqual(scheduler.state["jobs"]["r0"]["status"], "running")
        self.assertEqual(scheduler.state["jobs"]["control"]["status"], "running")
        self.assertEqual(scheduler.state["jobs"]["r1"]["status"], "pending")

        for process in popen.processes:
            process.returncode = 0
        scheduler.tick()
        self.assertEqual(scheduler.state["jobs"]["r1"]["status"], "running")

    def test_resume_is_idempotent_and_rejects_request_drift(self):
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0})
        manifest = self.manifest([self.job("one", 0)])
        scheduler = self.scheduler(manifest, probe, popen)
        scheduler.tick()
        popen.processes[0].returncode = 0
        scheduler.tick()

        resumed_popen = FakePopen()
        resumed = self.scheduler(manifest, probe, resumed_popen)
        resumed.tick()
        self.assertEqual(resumed.state["jobs"]["one"]["status"], "succeeded")
        self.assertEqual(resumed_popen.processes, [])

        self._write_request()
        payload = json.loads(self.request.read_text())
        payload["rows"][0]["row_id"] = "mutated"
        self.request.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "request drift"):
            self.scheduler(manifest, probe, FakePopen())

    def test_infrastructure_failure_retries_same_job_but_scientific_failure_does_not(self):
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0})
        manifest = self.manifest([self.job("retry", 0)])
        scheduler = self.scheduler(
            manifest, probe, popen, max_infrastructure_attempts=2
        )
        scheduler.tick()
        popen.processes[0].returncode = 75
        scheduler.tick()
        self.assertEqual(scheduler.state["jobs"]["retry"]["status"], "running")
        self.assertEqual(len(popen.processes), 2)
        self.assertEqual(scheduler.state["jobs"]["retry"]["attempts"], 2)

        popen.processes[1].returncode = 2
        scheduler.tick()
        self.assertEqual(scheduler.state["jobs"]["retry"]["status"], "failed_terminal")
        self.assertEqual(len(popen.processes), 2)

    def test_failed_gear_barrier_blocks_later_round(self):
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0})
        manifest = self.manifest(
            [
                self.job("r0", 0, kind="t16", priority=100, barrier_id="round-0"),
                self.job("r1", 1, kind="t16", priority=100, barrier_id="round-1"),
            ],
            barrier_order=["round-0", "round-1"],
        )
        scheduler = self.scheduler(manifest, probe, popen)
        scheduler.tick()
        popen.processes[0].returncode = 2
        done = scheduler.tick()

        self.assertTrue(done)
        self.assertEqual(scheduler.state["jobs"]["r0"]["status"], "failed_terminal")
        self.assertEqual(
            scheduler.state["jobs"]["r1"]["status"], "blocked_dependency"
        )
        self.assertEqual(len(popen.processes), 1)

    def test_gpu7_drain_and_explicit_exclusive_reservation(self):
        manifest = self.manifest(
            [self.job("ordinary", 0, priority=20), self.job("repeat", 1, priority=10)]
        )
        probe = MutableGpuProbe(self.module, free={7})
        popen = FakePopen()
        drained = self.scheduler(manifest, probe, popen, drain_gpu7=True)
        drained.tick()
        self.assertEqual(popen.processes, [])

        exclusive_popen = FakePopen()
        exclusive = self.module.Scheduler(
            manifest_path=manifest,
            state_path=self.root / "exclusive-state.json",
            events_path=self.root / "exclusive-events.jsonl",
            audit_path=self.root / "exclusive-audit.json",
            gpu_probe=probe,
            popen_factory=exclusive_popen,
            monotonic=lambda: 100.0,
            wall_time=lambda: 1_700_000_000.0,
            gpu7_exclusive_job_id="repeat",
        )
        self.schedulers.append(exclusive)
        exclusive.tick()
        self.assertEqual(len(exclusive_popen.processes), 1)
        command = exclusive_popen.processes[0].command
        self.assertEqual(command[command.index("--row-index") + 1], "1")

    def test_explicit_cpu_phase_marker_releases_gpu_and_audit_is_emitted(self):
        marker = self.root / "gpu.done"
        popen = FakePopen()
        probe = MutableGpuProbe(self.module, free={0})
        manifest = self.manifest(
            [
                self.job("gpu-then-cpu", 0, priority=20, release_marker=marker),
                self.job("next", 1, priority=10),
            ]
        )
        scheduler = self.scheduler(manifest, probe, popen)
        scheduler.tick()
        marker.write_text("gpu phase complete\n")
        scheduler.tick()

        self.assertEqual(len(popen.processes), 2)
        self.assertTrue(scheduler.state["jobs"]["gpu-then-cpu"]["gpu_released"])
        audit = json.loads((self.root / "audit.json").read_text())
        self.assertGreaterEqual(audit["effective_parallelism"]["peak_running_jobs"], 2)
        self.assertIn("gpu_hours", audit)


if __name__ == "__main__":
    unittest.main()
