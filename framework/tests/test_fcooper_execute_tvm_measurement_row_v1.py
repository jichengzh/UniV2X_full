import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/fcooper_execute_tvm_measurement_row_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fcooper_tvm_row", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FCooperExecuteTvmMeasurementRowTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def row(self, q_mode="fp16"):
        row = {
            "row_id": "row-1",
            "manifest_job_id": "row-1",
            "task_id": "S5-FCO-TVM-V1",
            "task_sha256": "t" * 64,
            "model": "fcooper",
            "hardware_id": "h800",
            "dispatch_key": "tvm_auto",
            "capability_profile_id": "h800-tvm-fcooper-v1",
            "group_id": "fcooper|64x128x256x128x256",
            "width": [64, 128, 256, 128, 256],
            "width_schema": [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
            ],
            "q_mode": q_mode,
            "source_contract": {},
            "source_evidence_sha256": "s" * 64,
        }
        return row

    def request(self, row=None):
        row = row or self.row()
        request = {
            "schema_version": "stage5_measurement_request_v2",
            "task_id": "S5-FCO-TVM-V1",
            "task_sha256": row["task_sha256"],
            "atomic_feedback": True,
            "real_h800_measurement_required": True,
            "batch_size": 4,
            "rows": [],
        }
        request["rows"] = [
            {**row, "row_id": f"row-{index}", "manifest_job_id": f"row-{index}"}
            for index in range(4)
        ]
        request["row_sha256"] = {
            item["row_id"]: self.module.sha256_payload(item)
            for item in request["rows"]
        }
        request["measurement_request_sha256"] = self.module.sha256_payload(request)
        return request

    def test_validates_tvm_formal_request(self):
        request = self.request()
        row = self.module.validate_tvm_request(request, row_index=2, request_kind="t16")
        self.assertEqual(row["row_id"], "row-2")

    def test_rejects_wrong_dispatch_or_trt_performance_fields(self):
        request = self.request()
        request["rows"][0]["dispatch_key"] = "trt_engine"
        request["row_sha256"]["row-0"] = self.module.sha256_payload(request["rows"][0])
        request["measurement_request_sha256"] = self.module.sha256_payload(
            {key: value for key, value in request.items() if key != "measurement_request_sha256"}
        )
        with self.assertRaisesRegex(ValueError, "TVM profile"):
            self.module.validate_tvm_request(request, row_index=0, request_kind="t16")

        row = self.row()
        row["engine_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "TRT performance"):
            self.module.reject_cross_backend_performance(row)

    def test_builds_precision_specific_commands(self):
        common = {
            "python": Path("/venv/python"),
            "code_root": Path("/repo"),
            "onnx": Path("/tmp/prepared.onnx"),
            "out_dir": Path("/tmp/out"),
            "label": "row",
            "width": (64, 128, 256, 128, 256),
            "gpu": 3,
            "max_trials": 64,
        }
        fp16 = self.module.build_route_command(q_mode="fp16", **common)
        self.assertIn("stage2_route_b_fp16_auto_runner.py", " ".join(map(str, fp16)))
        self.assertEqual(fp16[fp16.index("--max-trials") + 1], "64")
        self.assertIn("--measure-energy", fp16)

        fp32 = self.module.build_route_command(q_mode="fp32", **common)
        self.assertEqual(fp32[fp32.index("--precision") + 1], "fp32")

        int8 = self.module.build_route_command(
            q_mode="int8",
            quant_contract=Path("/tmp/quant.json"),
            **common,
        )
        self.assertIn("stage2_route_b_int8_auto_decomp.py", " ".join(map(str, int8)))
        self.assertEqual(int8[int8.index("--max-trials") + 1], "64")
        self.assertEqual(
            int8[int8.index("--tensor-quant-params-json") + 1],
            "/tmp/quant.json",
        )

    def test_quant_contract_command_batches_four_observed_outputs(self):
        command = self.module.build_quant_contract_command(
            python=Path("/venv/python"),
            code_root=Path("/repo"),
            onnx=Path("/tmp/prepared.onnx"),
            calibration_summary=Path("/tmp/calibration.json"),
            calibration_dir=Path("/tmp/calibration"),
            output_json=Path("/tmp/quant.json"),
        )
        self.assertEqual(
            command[command.index("--max-outputs-per-run") + 1],
            "4",
        )

    def test_extracts_latency_energy_and_requires_success(self):
        route = {
            "status": "success",
            "build_success": True,
            "latency": {"latency_ms_p50": 2.5},
            "energy": {"status": "success", "joules_per_inference": 0.4},
        }
        self.assertEqual(self.module.route_metrics(route), (2.5, 0.4))
        route["energy"]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "energy"):
            self.module.route_metrics(route)

    def test_first_route_shape_accepts_mapping_or_sequence(self):
        self.assertEqual(
            self.module.first_route_shape({"spatial_features": [5, 64, 512, 512]}, [1]),
            [5, 64, 512, 512],
        )
        self.assertEqual(self.module.first_route_shape([[1, 2, 3]], [9]), [1, 2, 3])

    def test_ap_bridge_command_uses_separate_tvm_worker_environment(self):
        command = self.module.build_fp16_ap_command(
            python=Path("/py39/python"),
            tvm_python=Path("/py310/python"),
            code_root=Path("/repo"),
            config=Path("/tmp/config.yaml"),
            checkpoint=Path("/tmp/model.pth"),
            artifact=Path("/tmp/model.so"),
            input_shape="5,64,128,128",
            output_shape="5,128,64,64",
            precision="fp16",
            artifact_output_dtype="float32",
            output_json=Path("/tmp/ap.json"),
            gpu_id=0,
            tvm_site=Path("/tvm/site"),
            tvm_lib_dirs=(Path("/cuda/lib"), Path("/tvm/lib")),
        )

        self.assertEqual(command[0], "/py39/python")
        self.assertEqual(command[command.index("--tvm-worker-python") + 1], "/py310/python")
        self.assertEqual(command[command.index("--tvm-site") + 1], "/tvm/site")
        self.assertEqual(
            command[command.index("--artifact-output-dtype") + 1],
            "float32",
        )
        self.assertEqual(command.count("--tvm-lib-dir"), 2)

    def test_stale_route_attempt_is_archived_before_retry(self):
        route_root = self.root / "route"
        label_dir = route_root / "row_int8"
        label_dir.mkdir(parents=True)
        (label_dir / "route_b_int8_auto_decomp_result.json").write_text(
            json.dumps({"status": "failed"})
        )
        (label_dir / "tuning_database").mkdir()

        archived = self.module.archive_stale_route_attempt(
            route_root=route_root,
            label="row_int8",
        )

        self.assertFalse(label_dir.exists())
        self.assertIsNotNone(archived)
        self.assertTrue((archived / "tuning_database").is_dir())
        self.assertTrue((archived / "route_b_int8_auto_decomp_result.json").is_file())

    def test_source_reuse_audit_binds_all_backend_neutral_files(self):
        source = self.root / "source"
        source.mkdir()
        paths = {}
        for name in ("model.onnx", "model.pth", "config.yaml", "recovery.json"):
            path = source / name
            path.write_text(name)
            paths[name] = path
        paths["recovery.json"].write_text(json.dumps({"elapsed_seconds": 12.5}))
        recovered = source / "recovered.pth"
        recovered.write_text("recovered")
        report = {
            "schema_version": "fcooper_source_export_v2",
            "status": "success",
            "formal_measurement_eligible": True,
            "onnx_path": str(paths["model.onnx"]),
            "onnx_sha256": self.module.sha256_file(paths["model.onnx"]),
            "checkpoint_path": str(recovered),
            "checkpoint_sha256": self.module.sha256_file(recovered),
        }
        report_path = source / "source_export_report.json"
        report_path.write_text(json.dumps(report))
        row = self.row()
        row["source_contract"] = {
            "onnx_path": str(paths["model.onnx"]),
            "checkpoint_path": str(paths["model.pth"]),
            "config_path": str(paths["config.yaml"]),
            "recovery_training_report_path": str(paths["recovery.json"]),
            "source_export_report": str(report_path),
        }
        audit = self.module.audit_reused_source(row)
        self.assertTrue(audit["passed"])
        self.assertTrue(audit["backend_neutral_only"])
        self.assertGreater(audit["recovery_training_seconds_saved"], 0)
        self.assertEqual(
            audit["resolved_source_contract"]["checkpoint_path"],
            str(recovered.resolve()),
        )

    def test_missing_source_runs_frozen_recovery_contract_in_new_root(self):
        row = self.row()
        row["source_contract"] = {
            "checkpoint_path": "/missing/checkpoint.pth",
            "config_path": "/missing/config.yaml",
            "onnx_path": "/missing/model.onnx",
        }
        artifact_root = self.root / "formal"
        execution = artifact_root / "execution" / "row"
        execution.mkdir(parents=True)
        source_config = self.root / "base.yaml"
        source_checkpoint = self.root / "base.pth"
        recovery_contract = self.root / "recovery.json"
        for path in (source_config, source_checkpoint, recovery_contract):
            path.write_text(path.name)
        args = SimpleNamespace(
            artifact_root=artifact_root,
            source_config=source_config,
            source_checkpoint=source_checkpoint,
            recovery_contract=recovery_contract,
        )
        calls = []

        def prepare_source(args, *, width, source_dir, execution, base_env, timings):
            calls.append((width, source_dir))
            source_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = source_dir / "recovered_checkpoint.pth"
            config = source_dir / "config.yaml"
            onnx = source_dir / "fcooper_dense_64x128x256x128x256.onnx"
            report = source_dir / "source_export_report.json"
            for path in (checkpoint, config, onnx):
                path.write_text(path.name)
            report.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "formal_measurement_eligible": True,
                        "checkpoint_sha256": self.module.sha256_file(checkpoint),
                        "onnx_sha256": self.module.sha256_file(onnx),
                    }
                )
            )
            timings["recovery_training_seconds"] = 9.0
            return checkpoint, onnx, None

        prepared_row, audit = self.module.ensure_backend_neutral_source(
            args,
            row=row,
            width=(64, 128, 256, 128, 256),
            execution=execution,
            base_env={},
            timings={},
            prepare_source=prepare_source,
        )

        self.assertEqual(len(calls), 1)
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["source_reused_from_trt_v2"])
        self.assertEqual(
            prepared_row["source_contract"]["onnx_path"],
            str(artifact_root / "sources/64x128x256x128x256/fcooper_dense_64x128x256x128x256.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
