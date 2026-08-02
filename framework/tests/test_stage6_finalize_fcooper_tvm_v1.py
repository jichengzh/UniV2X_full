import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage6_finalize_fcooper_tvm_v1 import (
    EvidenceError,
    _validate_original,
    _validate_gear,
    _validate_pool,
    _validate_tuned_selection,
    _validate_winner_package,
    apply_validated_winner,
    validate_full_ap_report_identity,
    validate_legacy_trial_budget,
    validate_resource_audit,
    validate_search_initialization,
    validate_precondition_audits,
    validate_ap_reference,
    validate_paper_arm_constraints,
    validate_control_provenance,
    validate_ap_repeat,
    build_three_model_rows,
    extract_reference_rows,
    select_winner,
    validate_success_evidence,
    validate_terminal_row,
)


SUCCESS = "measured_success_gold"


class LegacyTrialBudgetTests(unittest.TestCase):
    def test_accepts_self_signed_tvm_max_trials(self):
        raw = {
            "row_id": "gear-row",
            "tvm_max_trials": 64,
            "dispatch_key": "tvm_auto",
            "terminal_status": SUCCESS,
        }
        raw["actual_feedback_row_sha256"] = hashlib.sha256(
            json.dumps(
                raw,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

        validate_legacy_trial_budget(
            {
                "row_id": "gear-row",
                "tvm_trials": 64,
                "tvm_max_trials": 64,
                "tvm_trials_source_field": "tvm_max_trials",
                "backend": "tvm_auto",
                "backend_source_field": "dispatch_key",
            },
            raw,
        )

    def test_rejects_unsigned_trial_budget_drift(self):
        raw = {
            "row_id": "gear-row",
            "tvm_max_trials": 32,
            "terminal_status": SUCCESS,
            "actual_feedback_row_sha256": "0" * 64,
        }
        with self.assertRaisesRegex(EvidenceError, "legacy TVM trial budget"):
            validate_legacy_trial_budget(
                {
                    "row_id": "gear-row",
                    "tvm_trials": 64,
                    "tvm_max_trials": 64,
                    "tvm_trials_source_field": "tvm_max_trials",
                    "backend": "tvm_auto",
                    "backend_source_field": "dispatch_key",
                },
                raw,
            )


def _write(path: Path, payload) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(root: Path, name: str, payload=b"evidence") -> dict[str, str]:
    path, sha = _write(root / name, payload)
    return {"path": path, "sha256": sha}


def _directory_artifact(root: Path, name: str) -> dict[str, str]:
    path = root / name
    _write(path / "database.json", {"records": 1})
    manifest = [
        {
            "path": str(item.relative_to(path)),
            "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    digest = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {"path": str(path), "sha256": digest}


def _success_row(root: Path, row_id: str, latency: float, energy: float) -> dict:
    artifacts = {
        name: _artifact(root, f"{row_id}/{name}.bin")
        for name in (
            "checkpoint",
            "onnx",
            "tvm_module",
        )
    }
    artifacts["tvm_database"] = _directory_artifact(
        root, f"{row_id}/ms_work_dir"
    )
    performance_source = _artifact(
        root,
        f"{row_id}/performance_source.json",
        {
            "schema": "route_b_fp16_auto_result_v1",
            "precision": "fp16",
            "status": "success",
            "build_success": True,
            "gold_measurement_complete": True,
            "artifact_path": artifacts["tvm_module"]["path"],
            "artifact_digest": artifacts["tvm_module"]["sha256"],
            "latency": {"latency_ms_p50": latency},
            "energy": {"joule_per_inference": energy},
        },
    )
    artifacts["performance_report"] = _artifact(
        root,
        f"{row_id}/performance.json",
        {
            "schema_version": "fcooper_tvm_normalized_performance_report_v1",
            "row_id": row_id,
            "latency_ms": latency,
            "energy_j": energy,
            "source_report": performance_source,
        },
    )
    source_ap_report = _artifact(
        root,
        f"{row_id}/full_ap.json",
        {
            "schema_version": "fcooper_tvm_fp16_ap_report_v1",
            "status": "success_full",
            "dataset": "OPV2V",
            "split": "test",
            "dataset_samples": 2170,
            "requested_samples": 2170,
            "processed_samples": 2170,
            "backend_calls": 2170,
            "failed_samples": 0,
            "fallback_samples": 0,
            "gates": {"full": True, "status": "success_full"},
            "artifact_path": artifacts["tvm_module"]["path"],
            "artifact_sha256": artifacts["tvm_module"]["sha256"],
            "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
            "sha256": {
                "artifact": artifacts["tvm_module"]["sha256"],
                "checkpoint": artifacts["checkpoint"]["sha256"],
            },
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.60,
        },
    )
    artifacts["ap_report"] = _artifact(
        root,
        f"{row_id}/ap.json",
        {
            "schema_version": "fcooper_tvm_normalized_ap_report_v1",
            "row_id": row_id,
            "ap30": 0.9,
            "ap50": 0.8,
            "ap70": 0.60,
            "source_report": source_ap_report,
        },
    )
    artifacts["source_provenance"] = _artifact(
        root,
        f"{row_id}/source.json",
        {"backend_neutral_source": True, "row_id": row_id},
    )
    return {
        "row_id": row_id,
        "task_id": "S6-FCO-TVM-COMPRESSION-ONLY-MEASURE-V1",
        "model": "fcooper",
        "hardware_id": "h800",
        "dispatch_key": "tvm_auto",
        "capability_profile_id": "h800-tvm-fcooper-probe-conditioned-v1",
        "training_source": "online_feedback",
        "terminal_status": SUCCESS,
        "backend": "tvm_auto",
        "width": [32, 64, 128, 64, 128],
        "q_mode": "fp16",
        "ap30": 0.9,
        "ap50": 0.8,
        "ap70": 0.60,
        "latency_ms": latency,
        "energy_j": energy,
        "tvm_trials": 64,
        "artifacts": artifacts,
    }


class WinnerSelectionTests(unittest.TestCase):
    def test_ap_reference_must_match_measured_original(self):
        validate_ap_reference(0.633, {"ap70": 0.633})
        with self.assertRaisesRegex(EvidenceError, "AP reference"):
            validate_ap_reference(0.50, {"ap70": 0.633})

    def test_successful_ap_violation_cannot_be_paper_ready(self):
        validate_paper_arm_constraints(
            {
                "gear": {
                    "terminal_status": "feasibility_failure",
                    "ap_constraint_satisfied": False,
                }
            }
        )
        with self.assertRaisesRegex(EvidenceError, "AP constraint"):
            validate_paper_arm_constraints(
                {
                    "gear": {
                        "terminal_status": SUCCESS,
                        "ap_constraint_satisfied": False,
                    }
                }
            )

    def test_energy_breaks_one_percent_latency_tie(self):
        rows = [
            {
                "row_id": "fast",
                "terminal_status": SUCCESS,
                "ap70": 0.60,
                "latency_ms": 10.0,
                "energy_j": 8.0,
            },
            {
                "row_id": "efficient",
                "terminal_status": SUCCESS,
                "ap70": 0.61,
                "latency_ms": 10.09,
                "energy_j": 3.0,
            },
            {
                "row_id": "outside",
                "terminal_status": SUCCESS,
                "ap70": 0.62,
                "latency_ms": 10.11,
                "energy_j": 1.0,
            },
        ]
        selected = select_winner(rows, ap70_ref=0.65, max_ap_drop=0.10)
        self.assertEqual(selected["row_id"], "efficient")
        self.assertEqual(selected["selection_status"], "selected_feasible")

    def test_ap_repeat_accepts_sub_milli_drift_and_rejects_larger_drift(self):
        drift = validate_ap_repeat(
            {"ap30": 0.9, "ap50": 0.8, "ap70": 0.6},
            {"ap30": 0.8998, "ap50": 0.7995, "ap70": 0.5991},
        )
        self.assertAlmostEqual(drift["ap70"], -0.0009)
        with self.assertRaisesRegex(EvidenceError, "AP repeat drift"):
            validate_ap_repeat(
                {"ap30": 0.9, "ap50": 0.8, "ap70": 0.6},
                {"ap30": 0.9, "ap50": 0.8, "ap70": 0.5989},
            )

    def test_independent_repeat_must_preserve_winner_identity(self):
        rows = [
            {
                "row_id": "initial",
                "terminal_status": SUCCESS,
                "ap70": 0.60,
                "latency_ms": 1.0,
                "energy_j": 1.0,
            },
            {
                "row_id": "alternative",
                "terminal_status": SUCCESS,
                "ap70": 0.60,
                "latency_ms": 1.05,
                "energy_j": 0.5,
            },
        ]
        selected = select_winner(rows, ap70_ref=0.65, max_ap_drop=0.10)
        with self.assertRaisesRegex(EvidenceError, "winner identity"):
            apply_validated_winner(
                rows,
                selected,
                {
                    "latency_ms": 1.10,
                    "energy_j": 1.0,
                    "ap30": 0.9,
                    "ap50": 0.8,
                    "ap70": 0.60,
                },
                ap70_ref=0.65,
                max_ap_drop=0.10,
            )


class EvidenceValidationTests(unittest.TestCase):
    def test_pool_performance_source_must_bind_measured_tvm_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            performance_path = Path(row["artifacts"]["performance_report"]["path"])
            performance = json.loads(performance_path.read_text())
            source_path = Path(performance["source_report"]["path"])
            source = json.loads(source_path.read_text())
            source["artifact_digest"] = "0" * 64
            source_path.write_text(json.dumps(source, sort_keys=True) + "\n")
            performance["source_report"]["sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            performance_path.write_text(
                json.dumps(performance, sort_keys=True) + "\n"
            )
            row["artifacts"]["performance_report"]["sha256"] = hashlib.sha256(
                performance_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(EvidenceError, "performance source identity"):
                validate_success_evidence(row, root=root)

    def test_search_initialization_binds_gold176_and_single_tvm_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_id = "h800-tvm-fcooper-probe-conditioned-v1"
            capability_digest = "a" * 64
            payloads = {
                "task_contract": {
                    "schema_version": "stage5_search_task_contract_v2",
                    "task_id": "S5-FCO-TVM-V1",
                    "target_model": "fcooper",
                    "hardware_id": "h800",
                    "dispatch_key": "tvm_auto",
                    "capability_profile_id": profile_id,
                    "capability_digest": capability_digest,
                    "genome_schema": [
                        "backbone.s0",
                        "backbone.s1",
                        "backbone.s2",
                        "neck.deblock",
                        "neck.output",
                        "q_mode",
                    ],
                    "batch_size": 4,
                    "round_count": 4,
                    "sample_budget": 16,
                    "main_search_early_stopping": False,
                },
                "initialization_summary": {
                    "schema_version": "stage5_fcooper_formal_v2_initialization_summary",
                    "task_id": "S5-FCO-TVM-V1",
                    "coldstart_rows": 176,
                    "coldstart_audit": {
                        "rows_sha256": "b" * 64,
                        "graph_features_sha256": "c" * 64,
                    },
                    "training_view_policy": "initial_coldstart_only",
                    "pilot_online_labels_loaded": False,
                    "cross_model_online_labels_loaded": False,
                    "probe_metrics_loaded_as_labels": False,
                    "probe_rows_excluded_from_online_budget": 4,
                    "eligible_genomes": 3580,
                    "selected_row_ids": [
                        f"candidate-{index}|profile={profile_id}"
                        for index in range(4)
                    ],
                    "source_registry_provenance_audit": {
                        "pilot_references": 0,
                    },
                },
                "capability_profile": {
                    "schema_version": "fcooper_tvm_capability_profiles_v1",
                    "capability_profiles": [
                        {
                            "schema_version": "stage2_capability_profile_v3",
                            "capability_profile_id": profile_id,
                            "capability_digest": capability_digest,
                            "dispatch_key": "tvm_auto",
                            "hardware_target": "h800",
                            "features": {
                                "structure_axis_count": 5,
                                "probe_count": 5,
                                "int8_automatic_route_coverage": 1.0,
                                "int8_fallback_conv_count": 0,
                            },
                        }
                    ],
                },
            }
            bindings = {}
            for name, payload in payloads.items():
                path, sha = _write(root / f"{name}.json", payload)
                bindings[name] = {"path": path, "sha256": sha}

            audit = validate_search_initialization(bindings, root=root)
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["coldstart_rows"], 176)
            self.assertEqual(audit["genome_schema"][-1], "q_mode")

            summary_path = Path(bindings["initialization_summary"]["path"])
            summary = json.loads(summary_path.read_text())
            summary["pilot_online_labels_loaded"] = True
            summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n")
            bindings["initialization_summary"]["sha256"] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(EvidenceError, "search initialization"):
                validate_search_initialization(bindings, root=root)

    def test_precondition_audits_close_capability_and_label_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = {
                "capability_probe": {
                    "schema_version": "fcooper_tvm_capability_probe_audit_v1",
                    "passed": True,
                    "all_probes_terminal": True,
                    "label_free": True,
                    "dispatch_key": "tvm_auto",
                    "profile_id": "h800-tvm-fcooper-probe-conditioned-v1",
                    "probe_count": 5,
                    "probes": [
                        {
                            "precision": precision,
                            "role": role,
                            "build_run_passed": True,
                            "correctness_passed": True,
                            **(
                                {
                                    "automatic_route_passed": True,
                                    "fallback_conv_count": 0,
                                }
                                if precision == "int8"
                                else {}
                            ),
                        }
                        for precision, role in (
                            ("fp16", "base"),
                            ("fp16", "boundary"),
                            ("fp32", "original"),
                            ("int8", "base"),
                            ("int8", "boundary"),
                        )
                    ],
                },
                "probe_isolation": {
                    "schema_version": "fcooper_tvm_probe_isolation_audit_v1",
                    "passed": True,
                    "status": "passed",
                    "profile_id": "h800-tvm-fcooper-probe-conditioned-v1",
                    "probe_metrics_allowed_as_cost_model_labels": False,
                    "probe_rows_allowed_as_winner": False,
                    "probe_rows_allowed_in_t16_budget": False,
                    "policy": {
                        "probe_labels_allowed_in_training": False,
                        "probe_performance_fields_exported_to_profile": False,
                        "probe_rows_allowed_as_winner": False,
                        "probe_rows_allowed_in_t16": False,
                    },
                },
                "recovery_numeric_gate": {
                    "schema_version": "fcooper_backend_neutral_recovery_numeric_gate_v1",
                    "passed": True,
                    "status": "passed",
                    "t16_search_allowed": True,
                    "excludes": [
                        "backend_performance",
                        "backend_accuracy",
                        "TRT_artifacts",
                        "probe_labels",
                    ],
                },
                "control_provenance": {
                    "schema_version": "stage6_fcooper_tvm_request_rebind_audit_v1",
                    "backend": "tvm",
                    "dispatch_key": "tvm_auto",
                    "capability_profile_id": "h800-tvm-fcooper-probe-conditioned-v1",
                    "candidate_counts": {
                        "compression_only": 16,
                        "compress_then_tune": 12,
                        "schedule_only": 1,
                    },
                    "genome_sequence_equal": True,
                    "input_manifests_modified": False,
                    "old_measurement_results_read": False,
                    "source_contract_sequence_equal": True,
                    "source_evidence_sequence_equal": True,
                    "trt_performance_artifact_fields": [],
                    "row_identity_mappings": {},
                },
            }
            bindings = {}
            for name, payload in payloads.items():
                path, sha = _write(root / f"{name}.json", payload)
                bindings[name] = {"path": path, "sha256": sha}

            audits = validate_precondition_audits(bindings, root=root)
            self.assertEqual(set(audits["artifacts"]), set(payloads))

            isolation_path = Path(bindings["probe_isolation"]["path"])
            isolation = json.loads(isolation_path.read_text())
            isolation["probe_rows_allowed_as_winner"] = True
            isolation_path.write_text(json.dumps(isolation, sort_keys=True) + "\n")
            bindings["probe_isolation"]["sha256"] = hashlib.sha256(
                isolation_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(EvidenceError, "probe isolation"):
                validate_precondition_audits(bindings, root=root)

    def test_control_provenance_must_cover_exact_control_pool_rows(self):
        pools = {
            "compression_only": [{"row_id": f"compression-{index}"} for index in range(16)],
            "compress_then_tune_screen": [
                {"row_id": f"screen-{index}"} for index in range(12)
            ],
            "schedule_only": [{"row_id": "schedule-0"}],
        }
        provenance = {
            "row_identity_mappings": {
                "compression_only": [
                    {"rebound_row_id": row["row_id"]}
                    for row in pools["compression_only"]
                ],
                "compress_then_tune": [
                    {"rebound_row_id": row["row_id"]}
                    for row in pools["compress_then_tune_screen"]
                ],
                "schedule_only": [
                    {"rebound_row_id": pools["schedule_only"][0]["row_id"]}
                ],
            }
        }
        validate_control_provenance(provenance, pools)
        provenance["row_identity_mappings"]["compression_only"][0][
            "rebound_row_id"
        ] = "not-formal"
        with self.assertRaisesRegex(EvidenceError, "control provenance"):
            validate_control_provenance(provenance, pools)

    def test_gear_rounds_are_reconstructed_from_sha_bound_atomic_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            round_bindings = []
            round_states = []
            released_by_round = []
            for round_index in range(4):
                released = []
                for row_index in range(4):
                    row_id = f"round-{round_index}-row-{row_index}"
                    feedback_sha = hashlib.sha256(row_id.encode()).hexdigest()
                    rows.append(
                        {
                            "row_id": row_id,
                            "task_id": "S5-FCO-TVM-V1",
                            "training_source": "online_feedback",
                            "actual_feedback_row_sha256": feedback_sha,
                        }
                    )
                    released.append(
                        {
                            "row_id": row_id,
                            "actual_feedback_row_sha256": feedback_sha,
                        }
                    )
                path, sha = _write(
                    root / f"round_{round_index:02d}/atomic_batch_audit.json",
                    {
                        "schema_version": "stage5_atomic_batch_audit_v2",
                        "budget_consumed": 4,
                        "feedback_released": True,
                        "batch_quarantined": False,
                        "released_feedback_rows": released,
                    },
                )
                round_bindings.append({"path": path, "sha256": sha})
                released_by_round.append([row["row_id"] for row in released])
                if round_index > 0:
                    state_path, state_sha = _write(
                        root / f"round_{round_index:02d}/round_state.json",
                        {
                            "schema_version": "stage5_fcooper_formal_round_state_v2",
                            "task_id": "S5-FCO-TVM-V1",
                            "round_index": round_index,
                            "completed_feedback_rows": 4 * round_index,
                            "actual_graph_feedback_rows": 4 * round_index,
                            "budget_consumed": 4 * round_index,
                            "cross_model_online_labels_loaded": False,
                            "pilot_online_labels_loaded": False,
                            "probe_metrics_loaded_as_labels": False,
                            "atomic_release_audit": {
                                "schema_version": "stage5_fcooper_atomic_release_validation_v2",
                                "round_index": round_index - 1,
                                "released_row_ids": released_by_round[round_index - 1],
                            },
                            "selected_row_ids": released_by_round[round_index],
                            "feedback_evidence_audit": {
                                "schema_version": "stage5_fcooper_formal_feedback_evidence_audit_v2",
                                "feedback_rows": 4 * round_index,
                                "verified_actual_feedback_rows": 4 * round_index,
                                "recovered_pruned_rows": 4 * round_index,
                                "prefix_only_measurement_rows": 0,
                            },
                        },
                    )
                    round_states.append(
                        {"path": state_path, "sha256": state_sha}
                    )

            audit = _validate_gear(
                rows,
                {
                    "outer_budget": 16,
                    "batch_size": 4,
                    "rounds": 4,
                    "atomic_feedback": True,
                },
                round_bindings=round_bindings,
                round_state_bindings=round_states,
                root=root,
            )

            self.assertEqual(audit["round_row_counts"], [4, 4, 4, 4])
            self.assertEqual(len(audit["verified_round_audits"]), 4)
            self.assertEqual(len(audit["verified_round_states"]), 3)

    def test_pool_measurement_does_not_require_winner_only_repeat_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            validated = validate_success_evidence(row, root=root)
            self.assertTrue(validated["credible_terminal_status"])

    def test_fp32_row_rejects_fp16_performance_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            row["q_mode"] = "fp32"
            with self.assertRaisesRegex(
                EvidenceError, "FP performance source identity"
            ):
                validate_success_evidence(row, root=root)

    def test_fp32_row_rejects_fp16_full_ap_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            row["q_mode"] = "fp32"

            normalized_path = Path(row["artifacts"]["performance_report"]["path"])
            normalized = json.loads(normalized_path.read_text())
            source_path = Path(normalized["source_report"]["path"])
            source = json.loads(source_path.read_text())
            source["precision"] = "fp32"
            source_path.write_text(json.dumps(source, sort_keys=True) + "\n")
            normalized["source_report"]["sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            normalized_path.write_text(json.dumps(normalized, sort_keys=True) + "\n")
            row["artifacts"]["performance_report"]["sha256"] = hashlib.sha256(
                normalized_path.read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(
                EvidenceError, "2170-sample full AP source"
            ):
                validate_success_evidence(row, root=root)

    def test_formal_identity_mismatch_blocks_success_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            row["hardware_id"] = "rtx4090"
            with self.assertRaisesRegex(EvidenceError, "formal row identity"):
                validate_success_evidence(row, root=root)

    def test_partial_ap_report_blocks_success_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            normalized_path = Path(row["artifacts"]["ap_report"]["path"])
            normalized = json.loads(normalized_path.read_text())
            source_path = Path(normalized["source_report"]["path"])
            source = json.loads(source_path.read_text())
            source["processed_samples"] = 16
            source_path.write_text(json.dumps(source, sort_keys=True) + "\n")
            normalized["source_report"]["sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            normalized_path.write_text(json.dumps(normalized, sort_keys=True) + "\n")
            row["artifacts"]["ap_report"]["sha256"] = hashlib.sha256(
                normalized_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(EvidenceError, "2170-sample full AP"):
                validate_success_evidence(row, root=root)

    def test_pool_full_ap_must_bind_same_module_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            normalized_path = Path(row["artifacts"]["ap_report"]["path"])
            normalized = json.loads(normalized_path.read_text())
            source_path = Path(normalized["source_report"]["path"])
            source = json.loads(source_path.read_text())
            source["artifact_sha256"] = "f" * 64
            source["sha256"]["artifact"] = "f" * 64
            source_path.write_text(json.dumps(source, sort_keys=True) + "\n")
            normalized["source_report"]["sha256"] = hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            normalized_path.write_text(json.dumps(normalized, sort_keys=True) + "\n")
            row["artifacts"]["ap_report"]["sha256"] = hashlib.sha256(
                normalized_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(EvidenceError, "full AP source identity"):
                validate_success_evidence(row, root=root)

    def test_failure_row_requires_formal_fcooper_h800_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = {
                "row_id": "failed",
                "task_id": "S5-FCO-TVM-V1",
                "model": "other",
                "hardware_id": "h800",
                "dispatch_key": "tvm_auto",
                "capability_profile_id": "h800-tvm-fcooper-probe-conditioned-v1",
                "training_source": "online_feedback",
                "terminal_status": "feasibility_failure",
                "failure_reason": "formal build failure",
                "ap70": None,
                "latency_ms": None,
                "energy_j": None,
                "artifacts": {
                    "failure_contract": _artifact(
                        root,
                        "failed/failure.json",
                        {"failure_reason": "formal build failure"},
                    )
                },
            }
            with self.assertRaisesRegex(EvidenceError, "formal row identity"):
                validate_terminal_row(row, root=root)

    def test_missing_tvm_database_blocks_success_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            del row["artifacts"]["tvm_database"]
            with self.assertRaisesRegex(EvidenceError, "tvm_database"):
                validate_success_evidence(row, root=root)

    def test_trt_performance_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            row["engine_sha256"] = "a" * 64
            with self.assertRaisesRegex(EvidenceError, "TRT"):
                validate_success_evidence(row, root=root)

    def test_metric_drift_from_bound_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            row["latency_ms"] = 1.0
            with self.assertRaisesRegex(EvidenceError, "performance report"):
                validate_success_evidence(row, root=root)

    def test_pool_schema_and_identity_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _success_row(root, "candidate", 2.0, 0.5)
            row["tvm_trials"] = 0
            path, sha = _write(
                root / "pool.json",
                {
                    "schema_version": "fcooper_tvm_stage6_evidence_pool_v1",
                    "pool_name": "wrong_pool",
                    "rows": [row] * 16,
                },
            )
            with self.assertRaisesRegex(EvidenceError, "pool identity"):
                _validate_pool(
                    "compression_only",
                    {"path": path, "sha256": sha},
                    root=root,
                )

    def test_tuned_selection_requires_untampered_selector_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            screen_path, screen_sha = _write(
                root / "screen.json",
                {
                    "schema_version": "fcooper_tvm_stage6_evidence_pool_v1",
                    "pool_name": "compress_then_tune_screen",
                    "rows": [],
                },
            )
            source_path, source_sha = _write(root / "source_screen.json", {"rows": []})
            selector_path, selector_sha = _write(
                root / "selector.json",
                {
                    "automatic": True,
                    "selected_row_ids": ["a"],
                    "source_pool_path": source_path,
                    "source_pool_sha256": source_sha,
                },
            )
            selection = {
                "automatic": True,
                "selected_row_ids": ["a"],
                "source_pool_sha256": screen_sha,
                "source_normalized_screen_pool": {
                    "path": screen_path,
                    "sha256": screen_sha,
                },
                "source_automatic_selection": {
                    "path": selector_path,
                    "sha256": selector_sha,
                },
                "source_pre_normalization_screen_pool": {
                    "path": source_path,
                    "sha256": source_sha,
                },
            }
            Path(selector_path).write_text(json.dumps({"automatic": False}))

            with self.assertRaisesRegex(EvidenceError, "automatic selection"):
                _validate_tuned_selection(
                    [{"row_id": "a"}],
                    [{"row_id": "a"}],
                    screen_binding={"path": screen_path, "sha256": screen_sha},
                    tuned_payload={"automatic_selection": selection},
                    root=root,
                )

    def test_independent_validation_accepts_exclusive_gpu0(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = validate_success_evidence(
                _success_row(root, "candidate", 2.0, 0.5),
                root=root,
            )
            artifacts = selected["verified_artifacts"]
            manifest_sha_fields = {
                f"{name}_sha256": artifacts[name]["sha256"]
                for name in ("checkpoint", "onnx", "tvm_module", "tvm_database")
            }
            report_sha_fields = {
                "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                "onnx_sha256": artifacts["onnx"]["sha256"],
                "module_sha256": artifacts["tvm_module"]["sha256"],
                "database_sha256": artifacts["tvm_database"]["sha256"],
            }
            repeats = []
            for index in range(3):
                report = _artifact(
                    root,
                    f"validation/repeat_{index}.json",
                    {
                        "schema_version": "fcooper_tvm_independent_performance_v1",
                        "row_id": selected["row_id"],
                        "gpu_index": 0,
                        "latency_ms": 2.0,
                        "energy_j": 0.5,
                        **report_sha_fields,
                    },
                )
                repeats.append(
                    {
                        "gpu_index": 0,
                        "latency_ms": 2.0,
                        "energy_j": 0.5,
                        "report": report,
                        **manifest_sha_fields,
                    }
                )
            prediction = _artifact(root, "validation/predictions.jsonl")
            full_ap_report = _artifact(
                root,
                "validation/full_ap.json",
                {
                    "schema_version": "fcooper_tvm_fp16_ap_report_v1",
                    "status": "success_full",
                    "dataset": "OPV2V",
                    "split": "test",
                    "processed_samples": 2170,
                    "requested_samples": 2170,
                    "dataset_samples": 2170,
                    "backend_calls": 2170,
                    "failed_samples": 0,
                    "fallback_samples": 0,
                    "execution_device": {
                        "physical_gpu_id": 0,
                        "cuda_visible_devices": "0",
                    },
                    "numerical_contract": {
                        "silent_fallback_forbidden": True,
                        "artifact_compute_dtype": "float16",
                    },
                    "artifact_path": artifacts["tvm_module"]["path"],
                    "artifact_sha256": artifacts["tvm_module"]["sha256"],
                    "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                    "prediction_sha256": prediction["sha256"],
                    "sha256": {
                        "artifact": artifacts["tvm_module"]["sha256"],
                        "checkpoint": artifacts["checkpoint"]["sha256"],
                        "prediction": prediction["sha256"],
                    },
                    "ap30": selected["ap30"],
                    "ap50": selected["ap50"],
                    "ap70": selected["ap70"],
                },
            )
            validation = {
                "schema_version": "fcooper_tvm_gpu7_winner_validation_v1",
                "row_id": selected["row_id"],
                "gpu_index": 0,
                "exclusive_wait": {
                    "physical_gpu_id": 0,
                    "exclusive": True,
                    "checks": 1,
                    "waited_seconds": 0.1,
                },
                "performance_repeats": repeats,
                "full_ap": {
                    "gpu_index": 0,
                    "ap30": selected["ap30"],
                    "ap50": selected["ap50"],
                    "ap70": selected["ap70"],
                    "report": full_ap_report,
                    "prediction": prediction,
                    "prediction_sha256": prediction["sha256"],
                    **manifest_sha_fields,
                },
            }
            validation_path, validation_sha = _write(
                root / "validation/manifest.json",
                validation,
            )

            audit = _validate_winner_package(
                selected,
                {"path": validation_path, "sha256": validation_sha},
                root=root,
            )

            self.assertTrue(audit["passed"])
            self.assertEqual(audit["physical_gpu_id"], 0)

    def test_validation_wrapper_cannot_hide_report_from_another_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = validate_success_evidence(
                _success_row(root, "candidate", 2.0, 0.5),
                root=root,
            )
            artifacts = selected["verified_artifacts"]
            sha_fields = {
                f"{name}_sha256": artifacts[name]["sha256"]
                for name in ("checkpoint", "onnx", "tvm_module", "tvm_database")
            }
            repeats = []
            for index in range(3):
                report = _artifact(
                    root,
                    f"validation/repeat_{index}.json",
                    {
                        "schema_version": "fcooper_tvm_independent_performance_v1",
                        "row_id": selected["row_id"],
                        "gpu_index": 6,
                        "latency_ms": 2.0,
                        "energy_j": 0.5,
                        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                        "onnx_sha256": artifacts["onnx"]["sha256"],
                        "module_sha256": artifacts["tvm_module"]["sha256"],
                        "database_sha256": artifacts["tvm_database"]["sha256"],
                    },
                )
                repeats.append(
                    {
                        "gpu_index": 7,
                        "latency_ms": 2.0,
                        "energy_j": 0.5,
                        "report": report,
                        **sha_fields,
                    }
                )
            prediction = _artifact(root, "validation/predictions.jsonl")
            validation = {
                "schema_version": "fcooper_tvm_gpu7_winner_validation_v1",
                "row_id": selected["row_id"],
                "gpu_index": 7,
                "performance_repeats": repeats,
                "full_ap": {
                    "gpu_index": 7,
                    "ap30": selected["ap30"],
                    "ap50": selected["ap50"],
                    "ap70": selected["ap70"],
                    "report": artifacts["ap_report"],
                    "prediction": prediction,
                    "prediction_sha256": prediction["sha256"],
                    **sha_fields,
                },
            }
            validation_path, validation_sha = _write(
                root / "validation/manifest.json",
                validation,
            )

            with self.assertRaisesRegex(EvidenceError, "report identity"):
                _validate_winner_package(
                    selected,
                    {"path": validation_path, "sha256": validation_sha},
                    root=root,
                )

    def test_resource_audit_requires_full_formal_budget_and_all_gpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler_audit = _artifact(root, "resource/scheduler_audit.json")
            scheduler_state = _artifact(root, "resource/scheduler_state.json")
            scheduler_manifest = _artifact(root, "resource/scheduler_manifest.json")
            payload = {
                "schema_version": "fcooper_tvm_stage6_resource_audit_v1",
                "passed": True,
                "outer_budget": 16,
                "formal_measurement": {
                    "row_observations": 49,
                    "unique_row_trial_pairs": 49,
                    "tvm_trials": 1344,
                    "row_trial_pairs": [
                        {"row_id": f"row-{index}", "tvm_trials": 64}
                        for index in range(49)
                    ],
                    "pool_bindings": {
                        name: {}
                        for name in (
                            "compression_only",
                            "schedule_only",
                            "compress_then_tune_screen",
                            "compress_then_tune_tuned",
                            "gear",
                        )
                    },
                },
                "source_reuse": {
                    "audited_success_rows": 49,
                    "reused_from_trt_v2_rows": 30,
                    "generated_in_tvm_v1_rows": 19,
                },
                "scheduler": {
                    "component_count": 1,
                    "components": [
                        {
                            "audit_path": scheduler_audit["path"],
                            "audit_sha256": scheduler_audit["sha256"],
                            "state_path": scheduler_state["path"],
                            "state_sha256": scheduler_state["sha256"],
                            "manifest_path": scheduler_manifest["path"],
                            "manifest_sha256": scheduler_manifest["sha256"],
                        }
                    ],
                    "gpu_hours": 1.0,
                    "wall_clock_seconds": 10.0,
                    "effective_parallelism": {
                        "average_gpu_jobs": 1.0,
                        "peak_running_jobs": 1,
                    },
                    "queue": {
                        "component_count": 1,
                        "queued_jobs": 49,
                        "maximum_start_delay_seconds": 0.0,
                    },
                    "infrastructure_retry_count": 0,
                    "nonterminal_or_abandoned_components": [],
                    "formal_pair_coverage": {
                        "expected": 49,
                        "successful": 49,
                        "missing": [],
                    },
                },
                "per_gpu": {
                    str(index): {
                        "busy_seconds": 10.0 if index == 0 else 0.0,
                        "jobs_started": 49 if index == 0 else 0,
                        "job_attempts": (
                            [
                                {"row_id": f"row-{row}", "tvm_trials": 64}
                                for row in range(49)
                            ]
                            if index == 0
                            else []
                        ),
                    }
                    for index in range(8)
                },
            }
            path, sha = _write(root / "resource.json", payload)

            audit, _ = validate_resource_audit(
                {"path": path, "sha256": sha},
                root=root,
            )
            self.assertEqual(audit["formal_measurement"]["tvm_trials"], 1344)

            payload["per_gpu"]["0"]["job_attempts"].pop()
            payload["per_gpu"]["0"]["jobs_started"] -= 1
            path, sha = _write(root / "resource_missing_attempt.json", payload)
            with self.assertRaisesRegex(EvidenceError, "GPU attempt coverage"):
                validate_resource_audit(
                    {"path": path, "sha256": sha},
                    root=root,
                )
            payload["per_gpu"]["0"]["job_attempts"].append(
                {"row_id": "row-48", "tvm_trials": 64}
            )
            payload["per_gpu"]["0"]["jobs_started"] += 1

            payload["scheduler"]["nonterminal_or_abandoned_components"] = ["round_02"]
            path, sha = _write(root / "resource_nonterminal.json", payload)
            with self.assertRaisesRegex(EvidenceError, "resource audit"):
                validate_resource_audit(
                    {"path": path, "sha256": sha},
                    root=root,
                )
            payload["scheduler"]["nonterminal_or_abandoned_components"] = []

            payload["per_gpu"].pop("7")
            path, sha = _write(root / "resource_bad.json", payload)
            with self.assertRaisesRegex(EvidenceError, "resource audit"):
                validate_resource_audit(
                    {"path": path, "sha256": sha},
                    root=root,
                )

    def test_full_ap_report_is_bound_to_selected_artifact_and_prediction(self):
        selected = {
            "row_id": "row",
            "q_mode": "fp16",
            "verified_artifacts": {
                "checkpoint": {"sha256": "a" * 64},
                "tvm_module": {
                    "path": "/formal/model.so",
                    "sha256": "b" * 64,
                },
            },
        }
        report = {
            "schema_version": "fcooper_tvm_fp16_ap_report_v1",
            "status": "success_full",
            "processed_samples": 2170,
            "requested_samples": 2170,
            "dataset_samples": 2170,
            "backend_calls": 2170,
            "failed_samples": 0,
            "fallback_samples": 0,
            "execution_device": {
                "physical_gpu_id": 7,
                "cuda_visible_devices": "7",
            },
            "artifact_path": "/formal/model.so",
            "artifact_sha256": "c" * 64,
            "checkpoint_sha256": "a" * 64,
            "prediction_sha256": "d" * 64,
            "sha256": {
                "artifact": "c" * 64,
                "checkpoint": "a" * 64,
                "prediction": "d" * 64,
            },
        }
        with self.assertRaisesRegex(EvidenceError, "full AP report identity"):
            validate_full_ap_report_identity(
                report,
                selected=selected,
                prediction_sha256="d" * 64,
            )

    def test_full_ap_report_must_self_attest_physical_gpu7(self):
        selected = {
            "row_id": "row",
            "q_mode": "fp16",
            "verified_artifacts": {
                "checkpoint": {"sha256": "a" * 64},
                "tvm_module": {
                    "path": "/formal/model.so",
                    "sha256": "b" * 64,
                },
            },
        }
        report = {
            "schema_version": "fcooper_tvm_fp16_ap_report_v1",
            "status": "success_full",
            "dataset": "OPV2V",
            "split": "test",
            "processed_samples": 2170,
            "requested_samples": 2170,
            "dataset_samples": 2170,
            "backend_calls": 2170,
            "failed_samples": 0,
            "fallback_samples": 0,
            "execution_device": {
                "physical_gpu_id": 6,
                "cuda_visible_devices": "6",
            },
            "numerical_contract": {
                "silent_fallback_forbidden": True,
                "artifact_compute_dtype": "float16",
            },
            "artifact_path": "/formal/model.so",
            "artifact_sha256": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "prediction_sha256": "d" * 64,
            "sha256": {
                "artifact": "b" * 64,
                "checkpoint": "a" * 64,
                "prediction": "d" * 64,
            },
        }
        with self.assertRaisesRegex(EvidenceError, "full AP report identity"):
            validate_full_ap_report_identity(
                report,
                selected=selected,
                prediction_sha256="d" * 64,
            )

    def test_int8_full_ap_accepts_real_bridge_schema(self):
        selected = {
            "row_id": "row",
            "q_mode": "int8",
            "verified_artifacts": {
                "checkpoint": {"sha256": "a" * 64},
                "tvm_module": {
                    "path": "/formal/model.vmexec",
                    "sha256": "b" * 64,
                },
            },
        }
        report = {
            "schema_version": "fcooper_tvm_int8_ap_report_v1",
            "status": "success",
            "dataset": "OPV2V",
            "split": "test",
            "processed_samples": 2170,
            "requested_samples": 2170,
            "vm_calls": 2170,
            "failed_samples": 0,
            "fallback_samples": 0,
            "ap_measured": True,
            "gates": {"full_2170": True, "passed": True},
            "execution_device": {
                "physical_gpu_id": 7,
                "cuda_visible_devices": "7",
            },
            "numerical_contract": {
                "fallback_forbidden": True,
                "graph_input_dtype": "uint8",
                "graph_output_dtype": "uint8",
            },
            "artifact_path": "/formal/model.vmexec",
            "artifact_sha256": "b" * 64,
            "checkpoint_sha256": "a" * 64,
            "prediction_sha256": "d" * 64,
        }
        validate_full_ap_report_identity(
            report,
            selected=selected,
            prediction_sha256="d" * 64,
        )


class NativeOriginalAdmissionTests(unittest.TestCase):
    def _native_contract(self, root: Path) -> tuple[dict, dict]:
        row_id = "fcooper-original-default"
        checkpoint = _artifact(root, "native/checkpoint.pth")
        config = _artifact(root, "native/config.yaml")
        ap_report = _artifact(
            root,
            "native/ap_report.json",
            {
                "ap30": 0.91,
                "ap50": 0.82,
                "ap70": 0.63,
                "checkpoint_sha256": checkpoint["sha256"],
                "config_sha256": config["sha256"],
                "dataset_samples": 2170,
            },
        )
        evaluation_report = _artifact(root, "native/evaluation.yaml")
        row = {
            "row_id": row_id,
            "terminal_status": SUCCESS,
            "backend": "pytorch_cuda_cudnn",
            "optimized_scope": "post_scatter_backbone_shrinker",
            "width": [64, 128, 256, 128, 256],
            "q_mode": "fp32",
            "ap30": 0.91,
            "ap50": 0.82,
            "ap70": 0.63,
            "latency_ms": 11.8,
            "energy_j": 6.7,
            "artifacts": {
                "checkpoint": checkpoint,
                "config": config,
                "ap_report": ap_report,
                "evaluation_report": evaluation_report,
            },
        }
        contract_path, contract_sha = _write(
            root / "native/original_contract.json",
            {
                "schema_version": "fcooper_tvm_native_original_admission_v1",
                "same_scope_sha_admission": True,
                "row": row,
            },
        )
        return row, {"path": contract_path, "sha256": contract_sha}

    def _native_validation(self, root: Path, row: dict) -> dict:
        artifacts = row["artifacts"]
        repeats = []
        for index, (latency, energy) in enumerate(
            ((11.7, 6.8), (11.8, 6.7), (11.9, 6.6))
        ):
            report = _artifact(
                root,
                f"native/repeat_{index}.json",
                {
                    "gpu_abs": 7,
                    "latency_ms": latency,
                    "energy_j": energy,
                    "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                    "config_sha256": artifacts["config"]["sha256"],
                },
            )
            repeats.append(
                {
                    "gpu_index": 7,
                    "latency_ms": latency,
                    "energy_j": energy,
                    "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                    "config_sha256": artifacts["config"]["sha256"],
                    "report": report,
                }
            )
        return {
            "schema_version": "fcooper_native_gpu7_validation_admission_v1",
            "row_id": row["row_id"],
            "gpu_index": 7,
            "performance_repeats": repeats,
            "full_ap": {
                "gpu_index": 7,
                "ap30": row["ap30"],
                "ap50": row["ap50"],
                "ap70": row["ap70"],
                "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
                "config_sha256": artifacts["config"]["sha256"],
                "report": artifacts["ap_report"],
                "evaluation_report": artifacts["evaluation_report"],
            },
        }

    def test_native_original_uses_separate_gpu7_admission_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, contract_binding = self._native_contract(root)
            original, _ = _validate_original(contract_binding, root=root)
            validation = self._native_validation(root, source)
            path, sha = _write(root / "native/validation.json", validation)

            audit = _validate_winner_package(
                original, {"path": path, "sha256": sha}, root=root
            )

            self.assertTrue(audit["passed"])
            self.assertEqual(audit["evidence_kind"], "reused_native_fp32_gpu7")
            self.assertEqual(audit["repeat_count"], 3)
            self.assertEqual(audit["latency_ms"], 11.8)
            self.assertFalse(audit["prediction_required"])

    def test_native_original_rejects_tvm_backend_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, contract_binding = self._native_contract(root)
            contract = json.loads(Path(contract_binding["path"]).read_text())
            contract["row"]["backend"] = "tvm_auto"
            Path(contract_binding["path"]).write_text(
                json.dumps(contract, sort_keys=True) + "\n"
            )
            contract_binding["sha256"] = hashlib.sha256(
                Path(contract_binding["path"]).read_bytes()
            ).hexdigest()

            with self.assertRaisesRegex(EvidenceError, "native FP32"):
                _validate_original(contract_binding, root=root)


class ThreeModelTableTests(unittest.TestCase):
    @staticmethod
    def _reference(model: str) -> dict:
        methods = [
            "original_default",
            "compression_only",
            "schedule_only",
            "compress_then_tune",
            "tune_then_compress",
            "joint SHCoSearch",
        ]
        return {
            "paper_ready": True,
            "backends": {
                "tvm": {
                    "paper_ready": True,
                    "tables": {
                        "delta_0.10": [
                            {
                                "method": method,
                                "AP70": 0.6,
                                "latency_ms": 2.0,
                                "energy_j": 0.4,
                                "outcome": "selected",
                                "model": model,
                            }
                            for method in methods
                        ]
                    },
                }
            },
        }

    def test_extracts_five_rows_and_renames_joint_to_gear(self):
        rows = extract_reference_rows(self._reference("pyramid"), model="pyramid")
        self.assertEqual(len(rows), 5)
        self.assertNotIn("Tune -> Compress", {row["method"] for row in rows})
        self.assertIn("GEAR", {row["method"] for row in rows})

    def test_reference_rows_reject_explicit_cross_model_identity(self):
        payload = self._reference("codriving")
        with self.assertRaisesRegex(EvidenceError, "model identity"):
            extract_reference_rows(payload, model="pyramid")

    def test_combined_table_has_three_models_and_fifteen_rows(self):
        pyramid = extract_reference_rows(
            self._reference("pyramid"), model="pyramid"
        )
        codriving = extract_reference_rows(
            self._reference("codriving"), model="codriving"
        )
        fcooper = [
            {
                "model": "fcooper",
                "method": method,
                "terminal_status": SUCCESS,
                "AP70": 0.6,
                "latency_ms": 2.0,
                "energy_j": 0.4,
            }
            for method in (
                "Original/default",
                "Compression only",
                "Schedule only",
                "Compress -> Tune",
                "GEAR",
            )
        ]
        combined = build_three_model_rows(pyramid, codriving, fcooper)
        self.assertEqual(len(combined), 15)
        self.assertEqual(
            {row["model"] for row in combined},
            {"pyramid", "codriving", "fcooper"},
        )
        for model in ("pyramid", "codriving", "fcooper"):
            self.assertEqual(
                len([row for row in combined if row["model"] == model]), 5
            )


if __name__ == "__main__":
    unittest.main()
