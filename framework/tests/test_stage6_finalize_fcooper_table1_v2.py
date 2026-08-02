import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import shutil

from scripts.stage6_finalize_fcooper_table1_v2 import (
    TASK_ID,
    _median_performance,
    _ensure_final_outputs_absent,
    _prepare_final_output_target,
    _validate_schedule_identity,
    _validate_gpu_guard_audits,
    _validate_independent_ap,
    _validate_legacy_timing_remeasurement,
    _validate_search_replay_timing,
    _validate_terminal_pool,
    _summarize_phase_timings,
    _normalize_reused_phase_timings,
    normalize_terminal_row,
    repeat_paths_for_row,
    select_gear_candidate,
    verify_success_evidence,
)


def success(row_id: str, *, task_id: str = TASK_ID) -> dict:
    return {
        "row_id": row_id,
        "task_id": task_id,
        "training_source": "online_feedback",
        "terminal_status": "measured_success_gold",
        "width": [32, 64, 128, 64, 128],
        "q_mode": "int8",
        "ap30": 0.7,
        "ap50": 0.65,
        "ap70": 0.60,
        "latency_ms": 1.0,
        "energy_j": 0.2,
    }


class FCooperTableFinalizerV2Test(unittest.TestCase):
    def test_phase_timing_summary_requires_monotonic_fields(self) -> None:
        rows = [
            {
                "terminal_status": "measured_success_gold",
                "phase_timings_seconds": {
                    "recovery_initialization_seconds": 1.0,
                    "recovery_training_seconds": 2.0,
                    "onnx_export_seconds": 3.0,
                    "trt_build_performance_energy_seconds": 4.0,
                    "full_ap_seconds": 5.0,
                },
            }
        ]
        summary = _summarize_phase_timings("formal_t16", rows, expected_count=1)
        self.assertEqual(summary["total_monotonic_seconds"], 15.0)
        self.assertTrue(summary["passed"])

    def test_compression_timing_recovery_requires_bound_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "training.json"
            training.write_text(json.dumps({"elapsed_seconds": 12.5}))
            initialization = root / "initialization.json"
            initialization_report = {
                "status": "ready_for_recovery_training",
                "width": [32, 64, 64, 32, 64],
                "source_config_sha256": "a" * 64,
                "source_checkpoint_sha256": "b" * 64,
                "recovery_contract_sha256": "c" * 64,
            }
            initialization.write_text(json.dumps(initialization_report))
            log = root / "initialization.log"
            log.write_text("ok\n")
            command = ["python", "recovery.py"]
            row = {
                "terminal_status": "measured_success_gold",
                "width": [32, 64, 64, 32, 64],
                "recovery_training_report_path": str(training),
                "recovery_training_report_sha256": hashlib.sha256(
                    training.read_bytes()
                ).hexdigest(),
                "phase_timings_seconds": {
                    "onnx_export_seconds": 3.0,
                    "trt_build_performance_energy_seconds": 4.0,
                    "full_ap_seconds": 5.0,
                },
            }
            audit = {
                "schema_version": "fcooper_recovery_initialization_timing_v2",
                "passed": True,
                "entry_count": 1,
                "source_config_sha256": "a" * 64,
                "source_checkpoint_sha256": "b" * 64,
                "recovery_contract_sha256": "c" * 64,
                "entries": [
                    {
                        "width": row["width"],
                        "status": "success",
                        "elapsed_seconds": 1.5,
                        "timing_kind": "independent_monotonic_remeasurement",
                        "command": command,
                        "command_sha256": hashlib.sha256(
                            json.dumps(command, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "report_path": str(initialization),
                        "report_sha256": hashlib.sha256(
                            initialization.read_bytes()
                        ).hexdigest(),
                        "log_path": str(log),
                        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    }
                ]
            }

            normalized, provenance = _normalize_reused_phase_timings(
                "compression_only",
                [row],
                compression_initialization_audit=audit,
            )

            self.assertEqual(
                normalized[0]["phase_timings_seconds"][
                    "recovery_initialization_seconds"
                ],
                1.5,
            )
            self.assertEqual(
                normalized[0]["phase_timings_seconds"][
                    "recovery_training_seconds"
                ],
                12.5,
            )
            self.assertEqual(provenance["recovered_training_rows"], 1)

    def test_reused_screen_records_zero_skipped_source_work(self) -> None:
        row = {
            "terminal_status": "measured_success_gold",
            "task_id": "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2",
            "phase": "screen",
            "training_source": "online_feedback",
            "control_source_task_id": "S5-FCO-TRT-V2",
            "width": [32, 64, 64, 32, 64],
            "checkpoint_sha256": "a" * 64,
            "graph_features": {"onnx_sha256": "b" * 64},
            "recovery_training_report_sha256": "c" * 64,
            "phase_timings_seconds": {
                "trt_build_performance_energy_seconds": 4.0,
                "full_ap_seconds": 5.0,
            },
        }
        source_row = {
            "terminal_status": "measured_success_gold",
            "width": row["width"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "graph_features": {"onnx_sha256": "b" * 64},
            "recovery_training_report_sha256": "c" * 64,
        }

        normalized, provenance = _normalize_reused_phase_timings(
            "compress_then_tune_screen",
            [row],
            reuse_source_rows=[source_row],
        )

        timings = normalized[0]["phase_timings_seconds"]
        self.assertEqual(timings["recovery_initialization_seconds"], 0.0)
        self.assertEqual(timings["recovery_training_seconds"], 0.0)
        self.assertEqual(timings["onnx_export_seconds"], 0.0)
        self.assertEqual(provenance["zero_work_reuse_rows"], 1)
        self.assertEqual(provenance["sha_bound_reuse_rows"], 1)

    def test_reused_screen_rejects_missing_sha_bound_source_provenance(
        self,
    ) -> None:
        row = {
            "terminal_status": "measured_success_gold",
            "task_id": "S6-FCO-TRT-COMPRESS-THEN-TUNE-SCREEN-V2",
            "phase": "screen",
            "training_source": "online_feedback",
            "control_source_task_id": "S5-FCO-TRT-V2",
            "width": [32, 64, 64, 32, 64],
            "checkpoint_sha256": "a" * 64,
            "graph_features": {"onnx_sha256": "b" * 64},
            "recovery_training_report_sha256": "c" * 64,
            "phase_timings_seconds": {
                "trt_build_performance_energy_seconds": 4.0,
                "full_ap_seconds": 5.0,
            },
        }

        with self.assertRaisesRegex(ValueError, "SHA-bound source provenance"):
            _normalize_reused_phase_timings(
                "compress_then_tune_screen",
                [row],
                reuse_source_rows=[],
            )

    def test_cost_model_replay_timing_requires_four_matching_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "controls/resource_audit/replay.json"
            path.parent.mkdir(parents=True)
            rounds = []
            for index in range(4):
                expected = (
                    root
                    / "search"
                    / TASK_ID
                    / f"round_{index:02d}/measurement_request.json"
                )
                replayed = (
                    root
                    / "replay"
                    / "search"
                    / TASK_ID
                    / f"round_{index:02d}/measurement_request.json"
                )
                log = root / "replay" / f"round_{index:02d}.log"
                expected.parent.mkdir(parents=True, exist_ok=True)
                replayed.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "selected_row_ids": ["a", "b", "c", "d"],
                    "measurement_request_sha256": f"{index:x}" * 64,
                }
                expected.write_text(json.dumps(payload))
                replayed.write_text(json.dumps(payload))
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text("ok\n")
                rounds.append(
                    {
                        "round_index": index,
                        "elapsed_seconds": 2.5,
                        "request_semantics_match": True,
                        "selected_row_ids": ["a", "b", "c", "d"],
                        "measurement_request_sha256": f"{index:x}" * 64,
                        "expected_request_path": str(expected),
                        "expected_file_sha256": hashlib.sha256(
                            expected.read_bytes()
                        ).hexdigest(),
                        "replayed_request_path": str(replayed),
                        "replayed_file_sha256": hashlib.sha256(
                            replayed.read_bytes()
                        ).hexdigest(),
                        "log_path": str(log),
                        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    }
                )
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "fcooper_cost_model_replay_timing_v2",
                        "status": "passed",
                        "timing_kind": (
                            "deterministic_same_input_replay_monotonic"
                        ),
                        "original_online_refit_timing_available": False,
                        "replay_matches_original_requests": True,
                        "total_elapsed_seconds": 10.0,
                        "rounds": rounds,
                    }
                )
            )
            audit = _validate_search_replay_timing(path)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["round_count"], 4)

    def test_legacy_timing_remeasurement_requires_seven_monotonic_runs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            labels = [
                *(f"original_default_repeat_{index}" for index in range(3)),
                *(f"schedule_only_repeat_{index}" for index in range(3)),
                "schedule_only_full_ap",
            ]
            for label in labels:
                old = root / f"{label}.old.json"
                new = root / f"{label}.new.json"
                log = root / f"{label}.log"
                log.write_text("log")
                output_root = root / label
                output_root.mkdir()
                if label.startswith("original_default"):
                    output = output_root / "repeat.json"
                    command = ["python", "runner.py", "--output-json", str(output)]
                elif label == "schedule_only_full_ap":
                    output = output_root / "ap.json"
                    engine = (
                        root
                        / "schedule_only_repeat_0/engine/compiled.engine"
                    )
                    engine.parent.mkdir(parents=True, exist_ok=True)
                    engine.write_text("engine")
                    command = [
                        "python",
                        "runner.py",
                        "--engine",
                        str(engine),
                        "--output-json",
                        str(output),
                    ]
                else:
                    output = output_root / "performance.json"
                    engine_dir = output_root / "engine"
                    engine_dir.mkdir()
                    (engine_dir / "compiled.engine").write_text("engine")
                    command = [
                        "python",
                        "runner.py",
                        "--artifact-dir",
                        str(engine_dir),
                        "--out",
                        str(output),
                    ]
                output.write_text("{}")
                old.write_text(
                    json.dumps(
                        {
                            "schema_version": "fcooper_gpu_exclusivity_gate_v1",
                            "status": "completed_exclusive",
                            "evidence_scope": "sampled_process_exclusivity",
                            "gpu_index": 7,
                            "return_code": 0,
                            "runtime_sample_count": 1,
                            "runtime_observations": [],
                            "residual_processes": [],
                            "command": command,
                        }
                    )
                )
                new.write_text(
                    json.dumps(
                        {
                            "schema_version": "fcooper_gpu_exclusivity_gate_v1",
                            "status": "completed_exclusive",
                            "evidence_scope": "sampled_process_exclusivity",
                            "gpu_index": 7,
                            "return_code": 0,
                            "runtime_sample_count": 1,
                            "runtime_seconds": 1.0,
                            "runtime_observations": [],
                            "residual_processes": [],
                            "command": command,
                            "log_path": str(log),
                        }
                    )
                )
                records.append(
                    {
                        "label": label,
                        "timing_kind": (
                            "independent_gpu7_monotonic_remeasurement"
                        ),
                        "runtime_seconds": 1.0,
                        "legacy_audit_path": str(old),
                        "legacy_audit_sha256": hashlib.sha256(
                            old.read_bytes()
                        ).hexdigest(),
                        "timing_audit_path": str(new),
                        "timing_audit_sha256": hashlib.sha256(
                            new.read_bytes()
                        ).hexdigest(),
                        "log_path": str(log),
                        "log_sha256": hashlib.sha256(
                            log.read_bytes()
                        ).hexdigest(),
                    }
                )
            summary = root / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "fcooper_legacy_gpu7_timing_remeasurement_v2"
                        ),
                        "status": "passed",
                        "does_not_replace_selected_metrics": True,
                        "records": records,
                        "total_runtime_seconds": 7.0,
                    }
                )
            )
            audit = _validate_legacy_timing_remeasurement(summary)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["record_count"], 7)

            records[-1] = {**records[0]}
            summary.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "fcooper_legacy_gpu7_timing_remeasurement_v2"
                        ),
                        "status": "passed",
                        "does_not_replace_selected_metrics": True,
                        "records": records,
                        "total_runtime_seconds": 7.0,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _validate_legacy_timing_remeasurement(summary)

    def test_terminal_pool_rejects_duplicate_row_identity(self) -> None:
        duplicate = {
            "row_id": "same",
            "terminal_status": "feasibility_failure",
            "failure_reason": "credible",
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_terminal_pool(
                [duplicate, duplicate],
                evidence_root=Path("/unused"),
                expected_count=2,
            )

    def test_gpu_guard_audits_are_required_and_sha_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            label = "gear"
            expected_artifacts = []
            for suffix in ("repeat_0", "repeat_1", "repeat_2", "full_ap"):
                log = root / f"{label}_{suffix}.log"
                log.write_text(f"{suffix}\n")
                artifact = root / f"{label}_{suffix}.json"
                artifact.write_text("{}\n")
                expected_artifacts.append(artifact)
                audit = root / f"{label}_{suffix}_guard.json"
                audit.write_text(
                    json.dumps(
                        {
                            "schema_version": "fcooper_gpu_exclusivity_gate_v1",
                            "status": "completed_exclusive",
                            "evidence_scope": "sampled_process_exclusivity",
                            "gpu_index": 7,
                            "return_code": 0,
                            "command_pid": 123,
                            "command": [
                                "python",
                                "runner.py",
                                "--output-json",
                                str(artifact),
                            ],
                            "log_path": str(log),
                            "runtime_sample_count": 1,
                            "runtime_observations": [],
                            "residual_processes": [],
                        }
                    )
                )

            evidence = _validate_gpu_guard_audits(
                root,
                label=label,
                repeat_paths=expected_artifacts[:3],
                full_ap_path=expected_artifacts[3],
            )

            self.assertEqual(len(evidence), 4)
            self.assertTrue(all(len(row["audit_sha256"]) == 64 for row in evidence))
            (root / "gear_repeat_1_guard.json").write_text(
                json.dumps(
                    {
                        "schema_version": "fcooper_gpu_exclusivity_gate_v1",
                        "status": "runtime_interference",
                        "gpu_index": 7,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "exclusive"):
                _validate_gpu_guard_audits(
                    root,
                    label=label,
                    repeat_paths=expected_artifacts[:3],
                    full_ap_path=expected_artifacts[3],
                )

    def test_gpu_guard_rejects_stale_artifact_binding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = []
            for index in range(3):
                artifact = root / f"repeat-{index}.json"
                artifact.write_text("{}")
                expected.append(artifact)
                log = root / f"log-{index}.txt"
                log.write_text("ok")
                (root / f"gear_repeat_{index}_guard.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "fcooper_gpu_exclusivity_gate_v1",
                            "status": "completed_exclusive",
                            "evidence_scope": "sampled_process_exclusivity",
                            "gpu_index": 7,
                            "return_code": 0,
                            "command_pid": 123,
                            "command": ["python", "--out", "/stale/performance.json"],
                            "log_path": str(log),
                            "runtime_sample_count": 1,
                            "runtime_observations": [],
                            "residual_processes": [],
                        }
                    )
                )

            with self.assertRaisesRegex(ValueError, "artifact"):
                _validate_gpu_guard_audits(
                    root,
                    label="gear",
                    repeat_paths=expected,
                    full_ap_path=None,
                )

    def test_final_outputs_refuse_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            paths = [
                output / "results.csv",
                output / "audit.json",
                output / "bundle.json",
            ]
            _ensure_final_outputs_absent(paths)
            paths[1].write_text("existing")
            with self.assertRaises(FileExistsError):
                _ensure_final_outputs_absent(paths)

    def test_partial_final_output_is_quarantined_before_atomic_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "final"
            output.mkdir()
            partial = output / "fcooper_stage6_trt_delta_ap_0.10_v2.csv"
            partial.write_text("method\n")

            quarantine = _prepare_final_output_target(output)

            self.assertFalse(output.exists())
            self.assertIsNotNone(quarantine)
            assert quarantine is not None
            self.assertEqual((quarantine / partial.name).read_text(), "method\n")
            shutil.rmtree(quarantine)

    def test_schedule_identity_is_bound_again_at_finalization(self) -> None:
        row = {
            "row_id": "schedule-row",
            "manifest_job_id": "schedule-row",
            "schedule_baseline_derivation": (
                "scanner_unique_original_structure_to_fixed_fp32"
            ),
            "schedule_baseline_derivation_sha256": "d" * 64,
        }
        plan = {
            "fixed_row_id": "schedule-row",
            "fixed_manifest_job_id": "schedule-row",
            "schedule_baseline_derivation": row[
                "schedule_baseline_derivation"
            ],
            "schedule_baseline_derivation_sha256": "d" * 64,
        }

        _validate_schedule_identity(row, plan)
        with self.assertRaisesRegex(ValueError, "schedule-only identity"):
            _validate_schedule_identity(
                {**row, "schedule_baseline_derivation_sha256": "e" * 64},
                plan,
            )

    def test_gear_requires_exactly_16_formal_online_rows(self) -> None:
        rows = [success(f"online-{index:02d}") for index in range(16)]
        rows.append(success("probe", task_id="S5-FCO-PROBE"))

        with self.assertRaisesRegex(ValueError, "exactly 16"):
            select_gear_candidate(rows, ap70_ref=0.63, max_ap_drop=0.10)

        selected = select_gear_candidate(
            rows[:16], ap70_ref=0.63, max_ap_drop=0.10
        )
        self.assertIn(selected["row_id"], {row["row_id"] for row in rows[:16]})

    def test_repeat_paths_are_derived_from_selected_row_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = "formal-row-7"
            for index in range(3):
                path = (
                    root
                    / selected
                    / f"same_gpu7_repeat_{index}"
                    / "performance.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            unrelated = (
                root / "old-probe" / "same_gpu7_repeat_0" / "performance.json"
            )
            unrelated.parent.mkdir(parents=True)
            unrelated.write_text("{}")

            paths = repeat_paths_for_row(root, selected)

            self.assertEqual(len(paths), 3)
            self.assertTrue(all(selected in str(path) for path in paths))

    def test_three_repeats_must_all_be_bound_to_gpu7(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(3):
                path = Path(tmp) / f"repeat-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "gpu_abs": 6 if index == 2 else 7,
                            "precision": "int8",
                            "builder_optimization_level": 5,
                            "lat_p50_ms": 1.0,
                            "energy_j": 0.2,
                            "artifact_sha256": {
                                "source_onnx": "a" * 64,
                                "compiled_engine": "b" * 64,
                            },
                        }
                    )
                )
                paths.append(path)

            with self.assertRaisesRegex(ValueError, "GPU7"):
                _median_performance(
                    paths,
                    expected_onnx_sha256="a" * 64,
                    expected_precision="int8",
                    expected_builder_level=5,
                )

    def test_success_evidence_verifies_bytes_and_pruned_recovery_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {}
            for field in ("performance", "ap", "source", "engine", "checkpoint"):
                path = root / field
                path.write_bytes(field.encode())
                files[field] = path
            files["contract"] = root / "contract.json"
            files["config"] = root / "config.yaml"
            files["initial"] = root / "initial.pth"
            files["config"].write_text("model: fcooper\n")
            files["initial"].write_bytes(b"initial")
            files["contract"].write_text(
                json.dumps(
                    {
                        "seed": 20260723,
                        "start_epoch": 23,
                        "minimum_epochs": 4,
                        "recovery_epochs": 8,
                        "amp_fp16": True,
                    }
                )
            )
            epoch_records = []
            for epoch in range(24, 28):
                epoch_checkpoint = root / f"net_epoch{epoch}.pth"
                epoch_checkpoint.write_bytes(f"epoch-{epoch}".encode())
                epoch_records.append(
                    {
                        "epoch": epoch,
                        "train_loss": 1.0,
                        "validation_loss": 0.9,
                        "elapsed_seconds": 1.0,
                        "checkpoint_path": str(epoch_checkpoint),
                        "checkpoint_sha256": hashlib.sha256(
                            epoch_checkpoint.read_bytes()
                        ).hexdigest(),
                    }
                )
            files["recovery"] = root / "recovery"
            files["recovery"].write_text(
                json.dumps(
                    {
                        "schema_version": "fcooper_recovery_training_report_v2",
                        "status": "success",
                        "initialization_policy": "scanner_dependency_l1_v2",
                        "recovery_contract_path": str(files["contract"]),
                        "recovery_contract_sha256": hashlib.sha256(
                            files["contract"].read_bytes()
                        ).hexdigest(),
                        "config_path": str(files["config"]),
                        "config_sha256": hashlib.sha256(
                            files["config"].read_bytes()
                        ).hexdigest(),
                        "initial_checkpoint_path": str(files["initial"]),
                        "initial_checkpoint_sha256": hashlib.sha256(
                            files["initial"].read_bytes()
                        ).hexdigest(),
                        "recovered_checkpoint_path": str(files["checkpoint"]),
                        "epochs_completed": 4,
                        "recovered_checkpoint_sha256": hashlib.sha256(
                            files["checkpoint"].read_bytes()
                        ).hexdigest(),
                        "seed": 20260723,
                        "amp_fp16": True,
                        "dataset": {
                            "train_root": "/data/train",
                            "validation_root": "/data/val",
                            "train_samples": 10,
                            "validation_samples": 5,
                            "full_train_split": True,
                            "full_validation_split": True,
                        },
                        "epoch_records": epoch_records,
                        "elapsed_seconds": 4.0,
                    }
                )
            )
            source_payload = {
                "checkpoint_path": str(files["checkpoint"]),
                "checkpoint_sha256": hashlib.sha256(
                    files["checkpoint"].read_bytes()
                ).hexdigest(),
                "recovery_training_report_path": str(files["recovery"]),
                "recovery_training_report_sha256": hashlib.sha256(
                    files["recovery"].read_bytes()
                ).hexdigest(),
            }
            files["source"].write_text(json.dumps(source_payload))
            row = {
                **success("pruned"),
                "performance_result_json": str(files["performance"]),
                "performance_result_sha256": hashlib.sha256(
                    files["performance"].read_bytes()
                ).hexdigest(),
                "ap_report_path": str(files["ap"]),
                "ap_report_sha256": hashlib.sha256(files["ap"].read_bytes()).hexdigest(),
                "materialized_source_evidence_path": str(files["source"]),
                "materialized_source_evidence_sha256": hashlib.sha256(
                    files["source"].read_bytes()
                ).hexdigest(),
                "engine_path": str(files["engine"]),
                "engine_sha256": hashlib.sha256(
                    files["engine"].read_bytes()
                ).hexdigest(),
                "checkpoint_sha256": source_payload["checkpoint_sha256"],
                "recovery_training_report_sha256": source_payload[
                    "recovery_training_report_sha256"
                ],
                "graph_features": {"group_id": "g"},
            }
            row["materialized_graph_features_sha256"] = hashlib.sha256(
                json.dumps(
                    row["graph_features"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()

            audit = verify_success_evidence(row, root=root)

            self.assertTrue(audit["verified"])
            files["recovery"].write_text("drift")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                verify_success_evidence(row, root=root)

    def test_credible_failure_has_no_invented_metrics(self) -> None:
        failure = {
            "row_id": "failed",
            "terminal_status": "feasibility_failure",
            "failure_reason": "TensorRT rejected the materialized graph",
        }

        normalized = normalize_terminal_row(failure)

        self.assertEqual(normalized["terminal_status"], "feasibility_failure")
        for metric in ("ap30", "ap50", "ap70", "latency_ms", "energy_j"):
            self.assertIsNone(normalized[metric])

    def test_independent_ap_requires_full_2170_and_matching_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "ap.json"
            engine = root / "compiled.engine"
            config = root / "config.yaml"
            checkpoint = root / "checkpoint.pth"
            engine.write_bytes(b"engine")
            config.write_bytes(b"config")
            checkpoint.write_bytes(b"checkpoint")
            hashes = {
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
            expected = {
                "ap30": 0.7,
                "ap50": 0.65,
                "ap70": 0.6,
                "checkpoint_sha256": hashes["checkpoint_sha256"],
            }
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "fcooper_trt_ap_report_v1",
                        "status": "success_full",
                        "dataset_samples": 2170,
                        "processed_samples": 2170,
                        "fallback_samples": 0,
                        "failed_samples": 0,
                        "engine_samples": 2170,
                        "engine_calls": 2170,
                        "numerical_contract": {
                            "requested_samples": 2170,
                            "fallback_samples": 0,
                            "full_dataset_engine_execution": True,
                            "silent_fallback_forbidden": True,
                        },
                        **hashes,
                        "ap30": 0.7,
                        "ap50": 0.65,
                        "ap70": 0.6001,
                    }
                )
            )
            audit = _validate_independent_ap(
                path,
                expected=expected,
                engine_path=engine,
                config_path=config,
                checkpoint_path=checkpoint,
            )
            self.assertEqual(audit["processed_samples"], 2170)
            payload = json.loads(path.read_text())
            payload["ap70"] = 0.5
            path.write_text(
                json.dumps(payload)
            )
            with self.assertRaisesRegex(ValueError, "metric drift"):
                _validate_independent_ap(
                    path,
                    expected=expected,
                    engine_path=engine,
                    config_path=config,
                    checkpoint_path=checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
