from __future__ import annotations

import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_route_b_fp16_auto_runner as runner  # noqa: E402


class Stage2RouteBFp16AutoRunnerTests(unittest.TestCase):
    def test_parse_width_accepts_fcooper_five_axis_genome(self) -> None:
        self.assertEqual(
            runner.parse_width("64,128,256,128,256"),
            (64, 128, 256, 128, 256),
        )

    def test_parse_width_rejects_empty_or_non_positive_axes(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            runner.parse_width("")
        with self.assertRaisesRegex(ValueError, "positive"):
            runner.parse_width("64,0,256")

    def test_detect_group_conv_split_size_from_primfunc_text_without_base_width_constants(self) -> None:
        grouped = """
        T.Buffer((T.int64(96), T.int64(3), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(96), T.int64(64), T.int64(128)), "float16")
        """
        regular = """
        T.Buffer((T.int64(64), T.int64(3), T.int64(3), T.int64(3)), "float16")
        T.Buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(128)), "float16")
        """

        self.assertEqual(runner.detect_group_conv_split_size(grouped), 3)
        self.assertIsNone(runner.detect_group_conv_split_size(regular))

    def test_phase_commands_keep_width_onnx_and_label_parameterized(self) -> None:
        args = Namespace(
            label="smbo_16x32x64",
            width="16,32,64",
            onnx=Path("/tmp/models/smbo_16x32x64_backbone.onnx"),
            out_dir=Path("/tmp/route_b_fp16"),
            gpu="6",
            tvm_site=Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"),
            max_trials=7,
            seed=11,
            fix="split",
            warmup=3,
            iters=5,
            repeat=2,
            measure_energy=False,
            energy_iters=300,
            ref_so=None,
            wait_idle=False,
            idle_timeout_s=900,
        )

        paths = runner.plan_paths(args)
        build_cmd = runner.self_phase_command(Path("/usr/bin/python3"), args, "build")
        measure_cmd = runner.self_phase_command(Path("/usr/bin/python3"), args, "measure")

        self.assertEqual(paths["label_dir"], Path("/tmp/route_b_fp16/smbo_16x32x64"))
        self.assertEqual(paths["result_json"].name, "route_b_fp16_auto_result.json")
        self.assertIn("--width", build_cmd)
        self.assertIn("16,32,64", build_cmd)
        self.assertIn("--onnx", build_cmd)
        self.assertIn("/tmp/models/smbo_16x32x64_backbone.onnx", build_cmd)
        self.assertNotIn("smbo_64x128x256_backbone.onnx", " ".join(build_cmd))
        self.assertIn("--phase", measure_cmd)
        self.assertIn("measure", measure_cmd)

    def test_phase_command_forwards_energy_flags_when_requested(self) -> None:
        args = Namespace(
            label="smbo_16x32x64",
            width="16,32,64",
            onnx=Path("/tmp/models/smbo_16x32x64_backbone.onnx"),
            out_dir=Path("/tmp/route_b_fp16"),
            gpu="6",
            tvm_site=Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"),
            max_trials=7,
            seed=11,
            fix="split",
            warmup=3,
            iters=5,
            repeat=2,
            measure_energy=True,
            energy_iters=123,
            ref_so=None,
            wait_idle=False,
            idle_timeout_s=900,
        )

        measure_cmd = runner.self_phase_command(Path("/usr/bin/python3"), args, "measure")

        self.assertIn("--measure-energy", measure_cmd)
        self.assertIn("--energy-iters", measure_cmd)
        self.assertIn("123", measure_cmd)

    def test_fp32_schedule_path_is_explicit_and_not_named_fp16(self) -> None:
        args = Namespace(
            label="fcooper_schedule_only",
            width="64,128,256,128,256",
            onnx=Path("/tmp/fcooper.onnx"),
            out_dir=Path("/tmp/route_b"),
            gpu="3",
            tvm_site=Path("/opt/tvm/site-packages"),
            max_trials=64,
            seed=7,
            fix="none",
            precision="fp32",
            warmup=3,
            iters=5,
            repeat=2,
            measure_energy=False,
            energy_iters=300,
            ref_so=None,
            wait_idle=False,
            idle_timeout_s=900,
        )

        paths = runner.plan_paths(args)
        command = runner.self_phase_command(Path("/usr/bin/python3"), args, "build")

        self.assertEqual(paths["artifact_so"].name, "route_b_fp32_auto.so")
        self.assertIn("--precision", command)
        self.assertIn("fp32", command)

    def test_base_env_uses_selected_gpu_and_preserves_existing_ld_library_path(self) -> None:
        env = runner.build_h800_env(
            gpu="4",
            tvm_site=Path("/opt/tvm/site-packages"),
            existing_env={"LD_LIBRARY_PATH": "/already/there", "PATH": "/bin"},
            nvlibs_text="/cuda/lib:/driver/lib",
        )

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "4")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertTrue(env["PATH"].startswith("/usr/local/cuda-12.2/bin:"))
        self.assertIn("/opt/tvm/site-packages/tvm/lib", env["LD_LIBRARY_PATH"])
        self.assertIn("/cuda/lib:/driver/lib", env["LD_LIBRARY_PATH"])
        self.assertTrue(env["LD_LIBRARY_PATH"].endswith("/already/there"))

    def test_plan_phase_result_is_success_status_for_cli_smoke(self) -> None:
        args = Namespace(
            label="smbo_16x32x64",
            width="16,32,64",
            onnx=Path("/tmp/models/smbo_16x32x64_backbone.onnx"),
            out_dir=Path("/tmp/route_b_fp16"),
            gpu="6",
            tvm_site=Path("/exdata/jichengzhi/tvm310/lib/python3.10/site-packages"),
            max_trials=7,
            seed=11,
            fix="split",
            warmup=3,
            iters=5,
            repeat=2,
            measure_energy=False,
            energy_iters=300,
            ref_so=None,
            wait_idle=False,
            idle_timeout_s=900,
        )

        result = runner.run_plan(args)

        self.assertEqual(result["status"], "planned")

    def test_parse_args_defaults_keep_energy_opt_in(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["stage2_route_b_fp16_auto_runner.py"],
        ):
            args = runner.parse_args()

        self.assertFalse(args.measure_energy)
        self.assertEqual(args.energy_iters, 300)

    def test_measure_compiled_vm_energy_reports_machine_readable_success_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp)

            def fake_measure_energy(
                measure_dir: Path,
                module: object,
                dev: object,
                runtime_args: list[object],
                *,
                gpu: str,
                measure_iters: int,
            ) -> dict[str, object]:
                self.assertEqual(measure_dir, label_dir)
                self.assertEqual(gpu, "6")
                self.assertEqual(measure_iters, 123)
                (label_dir / "idle_power_samples.csv").write_text("ts,power_w\n0.0,100.0\n", encoding="utf-8")
                (label_dir / "active_power_samples.csv").write_text(
                    "ts,power_w\n0.1,120.0\n0.2,121.0\n",
                    encoding="utf-8",
                )
                return {
                    "idle_watt_avg": 100.0,
                    "watt_avg": 120.5,
                    "watt_delta_avg": 20.5,
                    "watt_p50": 120.0,
                    "watt_p90": 121.0,
                    "sample_window_ms": 5100,
                    "energy_J": 0.9,
                    "joule_per_inference": 0.9,
                    "measure_iters": 125,
                    "requested_measure_iters": 123,
                    "min_active_s": 5.0,
                    "warmup_iters": 20,
                }

            energy = runner.measure_compiled_vm_energy(
                label_dir=label_dir,
                vm=object(),
                dev=object(),
                runtime_args=[object()],
                gpu="6",
                measure_iters=123,
                energy_impl=fake_measure_energy,
            )

        self.assertEqual(energy["status"], "success")
        self.assertEqual(energy["joules_per_inference"], 0.9)
        self.assertEqual(energy["joule_per_inference"], 0.9)
        self.assertEqual(energy["requested_measure_iters"], 123)
        self.assertEqual(energy["completed_measure_iters"], 125)
        self.assertEqual(energy["active_sample_count"], 2)
        self.assertEqual(energy["idle_sample_count"], 1)
        self.assertEqual(energy["telemetry_source"], "nvidia-smi power.draw polling 50ms")
        self.assertEqual(energy["power_samples_csv"]["active"], str(label_dir / "active_power_samples.csv"))

    def test_measure_compiled_vm_energy_failure_is_explicit_and_not_estimated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label_dir = Path(tmp)

            def boom(*args: object, **kwargs: object) -> dict[str, object]:
                raise RuntimeError("nvml unavailable")

            energy = runner.measure_compiled_vm_energy(
                label_dir=label_dir,
                vm=object(),
                dev=object(),
                runtime_args=[object()],
                gpu="6",
                measure_iters=123,
                energy_impl=boom,
            )

        self.assertEqual(energy["status"], "failed")
        self.assertEqual(energy["requested_measure_iters"], 123)
        self.assertEqual(energy["completed_measure_iters"], 0)
        self.assertIsNone(energy["joules_per_inference"])
        self.assertIn("nvml unavailable", energy["error"])

    def test_run_measure_marks_gold_incomplete_when_energy_measurement_fails(self) -> None:
        class FakeTensor:
            def __init__(self, value: float) -> None:
                self._value = value

            def numpy(self) -> object:
                import numpy as np

                return np.array([self._value], dtype="float32")

        class FakeDev:
            def sync(self) -> None:
                return None

        class FakeEvalResult:
            results = [0.001, 0.002, 0.003]

        class FakeVM:
            def __getitem__(self, key: str):
                def run(*args: object) -> list[FakeTensor]:
                    return [FakeTensor(1.0)]

                return run

            def time_evaluator(self, name: str, dev: object, number: int, repeat: int):
                def evaluate(*args: object) -> FakeEvalResult:
                    return FakeEvalResult()

                return evaluate

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            label_dir = tmp_path / "smbo_16x32x64"
            label_dir.mkdir(parents=True, exist_ok=True)
            artifact_so = label_dir / "route_b_fp16_auto.so"
            ref_so = label_dir / "route_b_fp16_default_ref.so"
            artifact_so.write_text("artifact", encoding="utf-8")
            ref_so.write_text("ref", encoding="utf-8")
            (label_dir / "input_specs.json").write_text(
                '[{"name":"input0","shape":[1,1,1,1],"dtype":"float32"}]',
                encoding="utf-8",
            )
            args = Namespace(
                phase="measure",
                label="smbo_16x32x64",
                width="16,32,64",
                onnx=Path("/tmp/models/smbo_16x32x64_backbone.onnx"),
                out_dir=tmp_path,
                gpu="6",
                tvm_site=Path("/tmp/tvm-site"),
                max_trials=7,
                seed=11,
                fix="split",
                warmup=1,
                iters=3,
                repeat=1,
                measure_energy=True,
                energy_iters=123,
                ref_so=None,
                wait_idle=False,
                idle_timeout_s=900,
            )
            fake_tvm = types.SimpleNamespace(cuda=lambda index: FakeDev())
            fake_relax = object()
            with (
                patch.object(runner, "build_h800_env", return_value={}),
                patch.object(runner, "_configure_python_imports"),
                patch.object(runner, "_load_vm", side_effect=[FakeVM(), FakeVM()]),
                patch.object(runner, "_make_runtime_args", return_value=[object()]),
                patch.object(
                    runner,
                    "measure_compiled_vm_energy",
                    return_value={
                        "status": "failed",
                        "error": "RuntimeError('nvml unavailable')",
                        "requested_measure_iters": 123,
                        "completed_measure_iters": 0,
                        "joules_per_inference": None,
                    },
                ),
                patch.dict(
                    sys.modules,
                    {
                        "tvm": fake_tvm,
                        "tvm.tirx": types.SimpleNamespace(),
                    },
                ),
                patch.object(fake_tvm, "relax", fake_relax, create=True),
            ):
                result = runner.run_measure(args)

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["gold_measurement_complete"])
        self.assertEqual(result["gold_measurement_gaps"], ["energy_failed"])
        self.assertEqual(result["energy"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
