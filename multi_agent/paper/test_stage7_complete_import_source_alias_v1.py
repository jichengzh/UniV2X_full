import hashlib
import json
import os
import fcntl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stage7_complete_import_source_alias_v1 import (
    EVIDENCE_FIELDS,
    acquire_uuid_lock,
    atomic_json,
    build_evidence,
    checkpoint_alias_name,
    copy_authenticated,
    create_epoch_alias,
    gpu_is_idle,
    load_json,
    main,
    require_file_sha,
    run_logged,
    select_import_row,
    verify_validation_code_root,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ImportSourceAliasTests(unittest.TestCase):
    def test_select_import_row_requires_one_exact_import_only_group(self) -> None:
        request = {
            "rows": [
                {
                    "group_id": "pyramid|40x96x64",
                    "model": "pyramid",
                    "source_contract": {"training_required": False},
                },
                {
                    "group_id": "pyramid|16x32x64",
                    "model": "pyramid",
                    "source_contract": {"training_required": True},
                },
            ]
        }
        row = select_import_row(request, "pyramid|40x96x64")
        self.assertEqual(row["group_id"], "pyramid|40x96x64")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            select_import_row(request, "pyramid|missing")
        with self.assertRaisesRegex(ValueError, "training_required=false"):
            select_import_row(request, "pyramid|16x32x64")

    def test_checkpoint_alias_name_preserves_authenticated_epoch_basename(self) -> None:
        self.assertEqual(
            checkpoint_alias_name("/trusted/net_epoch_bestval_at1.pth"),
            "net_epoch_bestval_at1.pth",
        )
        with self.assertRaisesRegex(ValueError, "epoch-named"):
            checkpoint_alias_name("/trusted/stage5_best.pth")

    def test_build_evidence_has_exact_schema_and_primary_checkpoint_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {
                name: root / name
                for name in ("stage5_best.pth", "model.onnx", "calibration.npz", "summary.json")
            }
            for name, path in artifacts.items():
                path.write_bytes(name.encode("utf-8"))
            evidence = build_evidence(
                group_id="pyramid|40x96x64",
                model="pyramid",
                width=[40, 96, 64],
                source_plan_sha256="a" * 64,
                checkpoint_path=artifacts["stage5_best.pth"],
                onnx_path=artifacts["model.onnx"],
                calibration_path=artifacts["calibration.npz"],
                calibration_summary_path=artifacts["summary.json"],
            )
            self.assertEqual(set(evidence), EVIDENCE_FIELDS)
            self.assertEqual(
                evidence["checkpoint_path"],
                str(artifacts["stage5_best.pth"].resolve()),
            )
            self.assertEqual(
                evidence["checkpoint_sha256"], _sha(artifacts["stage5_best.pth"])
            )
            self.assertNotIn("provenance", evidence)

    def test_json_and_authenticated_copy_helpers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pth"
            destination = root / "destination.pth"
            alias = root / "net_epoch_bestval_at1.pth"
            source.write_bytes(b"authenticated")
            digest = _sha(source)
            copy_authenticated(source, destination, digest)
            copy_authenticated(source, destination, digest)
            create_epoch_alias(destination, alias, digest)
            create_epoch_alias(destination, alias, digest)
            self.assertEqual(os.stat(destination).st_ino, os.stat(alias).st_ino)
            require_file_sha(destination, digest, "destination")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                require_file_sha(destination, "0" * 64, "destination")
            payload_path = root / "payload.json"
            atomic_json(payload_path, {"ready": True})
            self.assertEqual(load_json(payload_path), {"ready": True})
            payload_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                load_json(payload_path)

    @mock.patch("stage7_complete_import_source_alias_v1.subprocess.run")
    def test_gpu_idle_and_logged_command(self, run: mock.Mock) -> None:
        run.return_value = SimpleNamespace(stdout="", returncode=0)
        self.assertTrue(gpu_is_idle(5))
        run.return_value = SimpleNamespace(stdout="123\n", returncode=0)
        self.assertFalse(gpu_is_idle(5))
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "command.log"
            run_logged(["true"], env={"PATH": os.environ["PATH"]}, log=log)
            self.assertIn('COMMAND ["true"]', log.read_text(encoding="utf-8"))

    def test_uuid_lock_matches_scheduler_nonblocking_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_root = Path(temporary)
            first = acquire_uuid_lock("GPU-test_1", lock_root=lock_root)
            self.assertIsNotNone(first)
            self.assertIsNone(acquire_uuid_lock("GPU-test_1", lock_root=lock_root))
            assert first is not None
            fcntl.flock(first.fileno(), fcntl.LOCK_UN)
            first.close()
            third = acquire_uuid_lock("GPU-test_1", lock_root=lock_root)
            self.assertIsNotNone(third)
            assert third is not None
            fcntl.flock(third.fileno(), fcntl.LOCK_UN)
            third.close()
            with self.assertRaisesRegex(ValueError, "unsafe GPU UUID"):
                acquire_uuid_lock("GPU/escape", lock_root=lock_root)

    def test_validation_code_root_must_match_deployment_pins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            validation = root / "validation"
            deployment = root / "deployment"
            relative_files = (
                "framework/stage5/measurement_plan_v2.py",
                "framework/stage7/source_resolution_v2.py",
                "scripts/stage5_materialize_round_sources_v1.sh",
            )
            for relative in relative_files:
                for code_root in (validation, deployment):
                    path = code_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(relative, encoding="utf-8")
            verify_validation_code_root(validation, deployment)
            (validation / relative_files[0]).write_text("drift", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity drift"):
                verify_validation_code_root(validation, deployment)

    def test_main_completes_exact_import_contract_with_primary_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v2_root = root / "formal"
            source_root = v2_root / "sources/pyramid/040x096x064"
            checkpoint_dir = source_root / "checkpoint"
            imported_root = root / "import"
            imported_checkpoint = imported_root / "net_epoch_bestval_at1.pth"
            imported_config = imported_root / "config.yaml"
            imported_root.mkdir(parents=True)
            imported_checkpoint.write_bytes(b"checkpoint")
            imported_config.write_bytes(b"config")
            checkpoint = checkpoint_dir / "stage5_best.pth"
            config = checkpoint_dir / "config.yaml"
            onnx = source_root / "onnx/model.onnx"
            report = source_root / "onnx/report.json"
            calibration = source_root / "calibration/features.npz"
            summary = source_root / "calibration/summary.json"
            trt_dir = source_root / "calibration/trt_npy"
            marker = source_root / "source_ready.done"
            evidence = source_root / "source_ready_evidence.json"
            request_path = root / "logical_request.json"
            plan_path = root / "source_resolution_plan.json"
            audit_path = root / "audit.json"
            log_path = root / "repair.log"
            source_contract = {
                "training_required": False,
                "checkpoint_path": str(checkpoint),
                "checkpoint_dir": str(checkpoint_dir),
                "config_path": str(config),
                "import_checkpoint_path": str(imported_checkpoint),
                "import_checkpoint_sha256": _sha(imported_checkpoint),
                "checkpoint_sha256": _sha(imported_checkpoint),
                "import_config_path": str(imported_config),
                "import_config_sha256": _sha(imported_config),
                "source_done_marker": str(marker),
                "onnx_path": str(onnx),
                "onnx_report_path": str(report),
                "calibration_npz": str(calibration),
                "calibration_summary": str(summary),
                "calibration_root": str(calibration.parent),
                "trt_calibration_dir": str(trt_dir),
            }
            row = {
                "row_id": "pyramid|40x96x64|q=fp16|profile=test",
                "group_id": "pyramid|40x96x64",
                "model": "pyramid",
                "width": [40, 96, 64],
                "source_evidence_sha256": "a" * 64,
                "source_contract": source_contract,
            }
            plan_row = {
                "candidate_id": row["row_id"],
                "group_id": row["group_id"],
                "source_evidence_path": str(evidence),
            }
            atomic_json(request_path, {"rows": [row]})
            atomic_json(
                plan_path,
                {"source_resolution_plan_sha256": "b" * 64, "rows": [plan_row]},
            )

            def fake_run_logged(command, *, env, log):
                del env, log
                if "stage2_h800_export_checkpoint_multiscale_onnx.py" in command[1]:
                    onnx.parent.mkdir(parents=True, exist_ok=True)
                    onnx.write_bytes(b"onnx")
                    atomic_json(report, {"status": "success"})
                elif "stage3_pyramid_calibration_export_v3.py" in command[1]:
                    calibration.parent.mkdir(parents=True, exist_ok=True)
                    calibration.write_bytes(b"calibration")
                    atomic_json(summary, {"schema": "stage3_pyramid_calibration_export_v3"})
                else:
                    trt_dir.mkdir(parents=True, exist_ok=True)
                    for index in range(15):
                        (trt_dir / f"batch2_{index:03d}.npy").write_bytes(b"npy")

            arguments = [
                "stage7_complete_import_source_alias_v1.py",
                "--v2-root",
                str(v2_root),
                "--validation-code-root",
                str(root / "reviewed_repo"),
                "--request",
                str(request_path),
                "--source-plan",
                str(plan_path),
                "--group-id",
                row["group_id"],
                "--gpu",
                "5",
                "--expected-gpu-uuid",
                "GPU-test",
                "--audit-json",
                str(audit_path),
                "--log",
                str(log_path),
            ]
            uuid_result = SimpleNamespace(stdout="GPU-test\n", returncode=0)
            uuid_lock = tempfile.TemporaryFile(mode="w+")
            with (
                mock.patch.object(sys, "argv", arguments),
                mock.patch(
                    "stage7_complete_import_source_alias_v1.verify_validation_code_root"
                ),
                mock.patch(
                    "stage7_complete_import_source_alias_v1.validate_plan_binding",
                    return_value=(row, plan_row),
                ),
                mock.patch(
                    "stage7_complete_import_source_alias_v1.validate_written_evidence"
                ) as validate_evidence,
                mock.patch(
                    "stage7_complete_import_source_alias_v1.gpu_is_idle",
                    return_value=True,
                ),
                mock.patch(
                    "stage7_complete_import_source_alias_v1.subprocess.run",
                    return_value=uuid_result,
                ),
                mock.patch(
                    "stage7_complete_import_source_alias_v1.acquire_uuid_lock",
                    return_value=uuid_lock,
                ),
                mock.patch(
                    "stage7_complete_import_source_alias_v1.run_logged",
                    side_effect=fake_run_logged,
                ),
            ):
                self.assertEqual(main(), 0)
            written = load_json(evidence)
            self.assertEqual(set(written), EVIDENCE_FIELDS)
            self.assertEqual(written["checkpoint_path"], str(checkpoint.resolve()))
            self.assertEqual(load_json(audit_path)["selected_event_budget_delta"], 0)
            validate_evidence.assert_called_once()
            self.assertEqual(
                validate_evidence.call_args.args[0], root / "reviewed_repo"
            )


if __name__ == "__main__":
    unittest.main()
