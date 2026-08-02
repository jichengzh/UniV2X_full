from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_materialize_round_sources_v1.sh"


def _request(root: Path) -> dict:
    rows = []
    specs = {
        "pyramid": {
            "width": [32, 64, 96],
            "source_contract": {
                "checkpoint_path": "/ckpt/net_epoch_bestval_at1.pth",
                "checkpoint_dir": "/ckpt",
                "checkpoint_sha256": "a" * 64,
                "onnx_path": f"{root}/pyramid/model.onnx",
                "onnx_report_path": f"{root}/pyramid/export.json",
                "calibration_root": f"{root}/pyramid/calibration",
                "calibration_npz": f"{root}/pyramid/calibration/features.npz",
                "calibration_summary": f"{root}/pyramid/calibration/summary.json",
                "trt_calibration_dir": f"{root}/pyramid/calibration/trt_npy",
                "source_done_marker": f"{root}/pyramid.done",
            },
        },
        "codriving": {
            "width": [16, 48, 64],
            "source_contract": {
                "model_dir": f"{root}/codriving/16x48x64",
                "onnx_path": f"{root}/codriving/16x48x64/model.onnx",
                "calibration_root": f"{root}/codriving/16x48x64/calibration_source",
                "calibration_npz": f"{root}/codriving/16x48x64/features.npz",
                "calibration_summary": f"{root}/codriving/16x48x64/summary.json",
                "trt_calibration_dir": f"{root}/codriving/16x48x64/trt_calibration_npy",
                "training_done_marker": f"{root}/codriving/16x48x64/stage5_training_complete.json",
                "source_done_marker": f"{root}/codriving.done",
            },
        },
    }
    profiles = (("tvm_auto", "tvm"), ("trt_engine", "trt"))
    for model, spec in specs.items():
        width_key = "x".join(map(str, spec["width"]))
        kind = (
            "pyramid_checkpoint_export"
            if model == "pyramid"
            else "codriving_prepare_train_export"
        )
        source_plan_sha = hashlib.sha256(
            json.dumps(
                {"kind": kind, "width": spec["width"], "contract": spec["source_contract"]},
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for dispatch, profile in profiles:
            for q_mode in ("fp16", "int8"):
                rows.append(
                    {
                        "manifest_job_id": f"{model}|{width_key}|{dispatch}|{q_mode}",
                        "group_id": f"{model}|{width_key}",
                        "model": model,
                        "width": spec["width"],
                        "dispatch_key": dispatch,
                        "capability_profile_id": profile,
                        "q_mode": q_mode,
                        "source_status": "materializable",
                        "source_evidence_sha256": source_plan_sha,
                        "source_contract": spec["source_contract"],
                    }
                )
    return {
        "schema_version": "stage5_measurement_request_v1",
        "group_count": 2,
        "row_count": 8,
        "rows": rows,
    }


class Stage5MaterializeRoundSourcesV1Tests(unittest.TestCase):
    def test_source_lock_is_keyed_by_group_not_gpu(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(".stage5_source_group.lock", source)
        self.assertNotIn(".stage5_source_gpu${GPU}.lock", source)

    def test_pyramid_training_runs_from_heal_root_for_relative_dataset_paths(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('(cd "$HEAL_ROOT" &&', source)

    def test_pyramid_epoch_patch_matches_indented_yaml_and_reuses_cutoff_checkpoint(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn(r'r"(^\s*epoches:\s*)\d+"', source)
        self.assertIn('target_checkpoint="$checkpoint_dir/net_epoch${epoches}.pth"', source)
        self.assertIn('if [[ -s "$target_checkpoint" && -s "$config" ]]; then', source)

    def test_pyramid_export_uses_epoch_named_checkpoint_for_trained_alias(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('export_checkpoint="$checkpoint"', source)
        self.assertIn(
            'export_checkpoint=$(latest_pyramid_checkpoint_at_or_before "$checkpoint_dir" "$epoches")',
            source,
        )
        self.assertIn('--checkpoint-path "$export_checkpoint"', source)

    def _dry_run(self, model: str) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            request = Path(temporary) / "request.json"
            request.write_text(json.dumps(_request(Path(temporary))), encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--request",
                    str(request),
                    "--model",
                    model,
                    "--gpu",
                    "6",
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_pyramid_dry_run_uses_checkpoint_export_and_calibration(self) -> None:
        output = self._dry_run("pyramid")

        self.assertIn("stage2_h800_export_checkpoint_multiscale_onnx.py", output)
        self.assertIn("stage3_pyramid_calibration_export_v3.py", output)
        self.assertIn("stage35_prepare_pyramid_trt_calibration_v1.py", output)
        self.assertNotIn("im2col", output.lower())

    def test_codriving_dry_run_uses_prepare_train_export_pipeline(self) -> None:
        output = self._dry_run("codriving")

        self.assertIn("prepare-one", output)
        self.assertIn("opencood/tools/train.py", output)
        self.assertIn("stage2_v2_gold_coldstart96_codriving_calib_export.py", output)
        self.assertIn("stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py", output)
        self.assertNotIn("mixed", output.lower())
        self.assertNotIn("int8_tc", output.lower())

    def test_v2_single_task_request_materializes_one_explicit_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_payload = _request(root)
            pyramid_rows = [
                row for row in request_payload["rows"] if row["model"] == "pyramid"
            ]
            request_payload = {
                "schema_version": "stage5_measurement_request_v2",
                "task_id": "S5-PYR-TVM",
                "batch_size": 4,
                "sample_budget": 16,
                "atomic_feedback": True,
                "real_h800_measurement_required": True,
                "required_metrics": [
                    "latency_ms",
                    "energy_j",
                    "ap30",
                    "ap50",
                    "ap70",
                ],
                "rows": [
                    {
                        **row,
                        "task_id": "S5-PYR-TVM",
                        "task_sha256": "b" * 64,
                        "row_id": row["manifest_job_id"],
                        "hardware_id": "h800",
                        "capability_digest": "c" * 64,
                        "dispatch_key": "tvm_auto",
                        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
                        "genome": [*row["width"], row["q_mode"]],
                    }
                    for row in pyramid_rows
                ],
            }
            request_payload["task_sha256"] = "b" * 64
            request_payload["row_sha256"] = {
                row["row_id"]: hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for row in request_payload["rows"]
            }
            request_payload["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            request = root / "request_v2.json"
            request.write_text(json.dumps(request_payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--request",
                    str(request),
                    "--model",
                    "pyramid",
                    "--group-id",
                    "pyramid|32x64x96",
                    "--gpu",
                    "6",
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("stage2_h800_export_checkpoint_multiscale_onnx.py", completed.stdout)

    def test_v2_unseen_pyramid_width_trains_before_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_payload = _request(root)
            pyramid_rows = [
                row for row in request_payload["rows"] if row["model"] == "pyramid"
            ]
            checkpoint_dir = root / "pyramid_models/32x64x96"
            contract = {
                **pyramid_rows[0]["source_contract"],
                "checkpoint_path": str(checkpoint_dir / "stage5_best.pth"),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_sha256": None,
                "config_path": str(checkpoint_dir / "config.yaml"),
                "training_done_marker": str(checkpoint_dir / "stage5_training_complete.json"),
                "base_checkpoint_path": "/base/net_epoch_bestval_at23.pth",
                "base_checkpoint_dir": "/base",
                "base_checkpoint_sha256": "d" * 64,
                "training_required": True,
                "training_epoches": 31,
                "width_per_group": 4,
                "groups": 32,
            }
            source_sha = hashlib.sha256(
                json.dumps(
                    {
                        "kind": "pyramid_prepare_train_export",
                        "width": [32, 64, 96],
                        "contract": contract,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            rows = []
            for row in pyramid_rows:
                rows.append(
                    {
                        **row,
                        "task_id": "S5-PYR-TVM",
                        "task_sha256": "b" * 64,
                        "row_id": row["manifest_job_id"],
                        "hardware_id": "h800",
                        "capability_digest": "c" * 64,
                        "dispatch_key": "tvm_auto",
                        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
                        "genome": [*row["width"], row["q_mode"]],
                        "materialization_kind": "pyramid_prepare_train_export",
                        "source_contract": contract,
                        "source_evidence_sha256": source_sha,
                    }
                )
            request_payload = {
                "schema_version": "stage5_measurement_request_v2",
                "task_id": "S5-PYR-TVM",
                "task_sha256": "b" * 64,
                "batch_size": 4,
                "sample_budget": 16,
                "atomic_feedback": True,
                "real_h800_measurement_required": True,
                "required_metrics": [
                    "latency_ms",
                    "energy_j",
                    "ap30",
                    "ap50",
                    "ap70",
                ],
                "rows": rows,
            }
            request_payload["row_sha256"] = {
                row["row_id"]: hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for row in rows
            }
            request_payload["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            request = root / "request_v2_train.json"
            request.write_text(json.dumps(request_payload), encoding="utf-8")
            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--request",
                    str(request),
                    "--model",
                    "pyramid",
                    "--group-id",
                    "pyramid|32x64x96",
                    "--gpu",
                    "6",
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("structural_prune_pyramid.py", completed.stdout)
        self.assertIn("opencood/tools/train_ddp.py", completed.stdout)
        self.assertIn("stage2_h800_export_checkpoint_multiscale_onnx.py", completed.stdout)

    def test_valid_source_evidence_returns_before_waiting_for_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for name in ("checkpoint", "model", "calibration", "summary"):
                path = root / f"{name}.bin"
                path.write_bytes(name.encode())
                artifacts[name] = path
            marker = root / "pyramid.done"
            evidence = root / "pyramid_evidence.json"
            contract = {
                "checkpoint_path": str(artifacts["checkpoint"]),
                "checkpoint_dir": str(root),
                "checkpoint_sha256": hashlib.sha256(
                    artifacts["checkpoint"].read_bytes()
                ).hexdigest(),
                "onnx_path": str(artifacts["model"]),
                "onnx_report_path": str(root / "export.json"),
                "calibration_root": str(root),
                "calibration_npz": str(artifacts["calibration"]),
                "calibration_summary": str(artifacts["summary"]),
                "trt_calibration_dir": str(root / "trt"),
                "source_done_marker": str(marker),
                "training_required": False,
            }
            source_plan_sha = hashlib.sha256(
                json.dumps(
                    {
                        "kind": "pyramid_checkpoint_export",
                        "width": [32, 64, 96],
                        "contract": contract,
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            evidence.write_text(
                json.dumps(
                    {
                        "schema_version": "stage5_source_materialization_evidence_v1",
                        "group_id": "pyramid|32x64x96",
                        "model": "pyramid",
                        "width": "32x64x96",
                        "source_plan_sha256": source_plan_sha,
                        "checkpoint_path": str(artifacts["checkpoint"]),
                        "checkpoint_sha256": contract["checkpoint_sha256"],
                        "onnx_path": str(artifacts["model"]),
                        "onnx_sha256": hashlib.sha256(
                            artifacts["model"].read_bytes()
                        ).hexdigest(),
                        "calibration_path": str(artifacts["calibration"]),
                        "calibration_sha256": hashlib.sha256(
                            artifacts["calibration"].read_bytes()
                        ).hexdigest(),
                        "calibration_summary_path": str(artifacts["summary"]),
                        "calibration_summary_sha256": hashlib.sha256(
                            artifacts["summary"].read_bytes()
                        ).hexdigest(),
                        "status": "ready",
                    }
                ),
                encoding="utf-8",
            )
            marker.touch()
            rows = []
            for index, q_mode in enumerate(("fp16", "int8", "fp16", "int8")):
                row_id = f"pyramid|32x64x96|q={q_mode}|copy={index}"
                rows.append(
                    {
                        "manifest_job_id": row_id,
                        "row_id": row_id,
                        "group_id": "pyramid|32x64x96",
                        "model": "pyramid",
                        "width": [32, 64, 96],
                        "q_mode": q_mode,
                        "task_id": "S5-PYR-TVM",
                        "task_sha256": "b" * 64,
                        "hardware_id": "h800",
                        "capability_digest": "c" * 64,
                        "dispatch_key": "tvm_auto",
                        "capability_profile_id": "h800-tvm-probe-conditioned-v3",
                        "genome": [32, 64, 96, q_mode],
                        "source_status": "materializable",
                        "source_evidence_sha256": source_plan_sha,
                        "source_contract": contract,
                    }
                )
            payload = {
                "schema_version": "stage5_measurement_request_v2",
                "task_id": "S5-PYR-TVM",
                "task_sha256": "b" * 64,
                "batch_size": 4,
                "sample_budget": 16,
                "atomic_feedback": True,
                "real_h800_measurement_required": True,
                "required_metrics": [
                    "latency_ms",
                    "energy_j",
                    "ap30",
                    "ap50",
                    "ap70",
                ],
                "rows": rows,
            }
            payload["row_sha256"] = {
                row["row_id"]: hashlib.sha256(
                    json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                for row in rows
            }
            payload["measurement_request_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            request = root / "request.json"
            request.write_text(json.dumps(payload), encoding="utf-8")
            fake_bin = root / "bin"
            fake_bin.mkdir()
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text("#!/usr/bin/env bash\nexit 42\n", encoding="utf-8")
            nvidia_smi.chmod(0o755)

            completed = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--request",
                    str(request),
                    "--model",
                    "pyramid",
                    "--group-id",
                    "pyramid|32x64x96",
                    "--gpu",
                    "0",
                ],
                cwd=REPO,
                env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
