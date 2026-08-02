from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import stage35_gold144_sufficiency_v1 as stage35  # noqa: E402


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)


def _gold144_fixture() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []
    pilot_group_ids: list[str] = []
    model_groups = {
        "codriving": ([*range(13)], [13, 14, 15]),
        "pyramid": ([*range(17)], [17, 18, 19]),
    }
    for model, (train_indices, locked_indices) in model_groups.items():
        pilot_group_ids.append(f"{model}|g0")
        for group_index in [*train_indices, *locked_indices]:
            group_id = f"{model}|g{group_index}"
            split = "locked_holdout" if group_index in locked_indices else "train"
            source_pool = (
                "targeted16"
                if model == "pyramid" and group_index in {13, 14, 15, 16}
                else "gold128"
            )
            for arm_index, (backend, q_mode) in enumerate(ARMS):
                job_id = f"{group_id}|{backend}|{q_mode}"
                common = {
                    "group_id": group_id,
                    "model": model,
                    "width": [16 + group_index, 32, 64],
                    "q_mode": q_mode,
                    "dispatch_key": backend,
                    "capability_profile_id": f"h800-{backend}",
                    "split": split,
                    "source_pool": source_pool,
                    "width_stratum": "test",
                }
                jobs.append({
                    "job_id": job_id,
                    **common,
                })
                rows.append({
                    "schema_version": "stage35_gold144_final_v1",
                    "manifest_job_id": job_id,
                    "terminal_status": "measured_success_gold",
                    "latency_ms": 1.0 + group_index / 100.0 + arm_index / 1000.0,
                    "energy_j": 0.2 + group_index / 1000.0 + arm_index / 10000.0,
                    "ap30": 0.8,
                    "ap50": 0.7,
                    "ap70": 0.5,
                    "performance_result_sha256": "a" * 64,
                    "ap_report_sha256": "b" * 64,
                    **common,
                })
    return rows, {
        "schema_version": "stage35_gold144_manifest_v1",
        "jobs": jobs,
        "pilot_group_ids": pilot_group_ids,
        "source_manifests": {"gold128_sha256": "c" * 64},
    }


def _source_gold128(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in rows if row["source_pool"] == "gold128"]


def _repeat_audit(
    source_rows: list[dict[str, object]], directory: Path, source_sha256: str
) -> dict[str, object]:
    performance_artifact = directory / "repeat-performance.json"
    performance_artifact.write_text("{}\n", encoding="utf-8")
    performance_sha256 = hashlib.sha256(performance_artifact.read_bytes()).hexdigest()
    ap_artifact = directory / "repeat-ap.json"
    ap_artifact.write_text("{}\n", encoding="utf-8")
    ap_sha256 = hashlib.sha256(ap_artifact.read_bytes()).hexdigest()
    categories = ("base", "small_channel", "alignment_trap", "large_model")
    performance_rows = []
    for index, baseline in enumerate(source_rows[:32]):
        performance_rows.append({
            "manifest_job_id": baseline["manifest_job_id"],
            "repeat_category": categories[index % len(categories)],
            "baseline_latency_ms": baseline["latency_ms"],
            "baseline_energy_j": baseline["energy_j"],
            "repeat_result_json": str(performance_artifact),
            "repeat_result_sha256": performance_sha256,
        })
    ap_rows = []
    for index, baseline in enumerate(source_rows[:4]):
        ap_rows.append({
            "source_manifest_job_id": baseline["manifest_job_id"],
            "repeat_category": categories[index],
            "baseline_ap30": baseline["ap30"],
            "baseline_ap50": baseline["ap50"],
            "baseline_ap70": baseline["ap70"],
            "baseline_ap_report_sha256": baseline["ap_report_sha256"],
            "repeat_report_path": str(ap_artifact),
            "repeat_report_sha256": ap_sha256,
        })
    return {
        "schema_version": "stage35_gold128_repeat_audit_v1",
        "source_gold_sha256": source_sha256,
        "qualified": True,
        "performance_repeat_qualified": True,
        "terminal_rows": 32,
        "performance_checks": {"stable": True},
        "rows": performance_rows,
        "ap_repeat_audit": {
            "schema_version": "stage35_gold128_ap_repeat_audit_v1",
            "qualified": True,
            "terminal_rows": 4,
            "checks": {"stable": True},
            "rows": ap_rows,
        },
    }


def _passing_evidence() -> dict[str, object]:
    return {
        "learning": {
            target: {
                "mae_improvement_26_to_30": 0.04,
                "spearman_30": 0.92,
                "topk_recall_30": 0.8,
            }
            for target in ("ap70", "latency_ms", "energy_j")
        },
        "calibrated_beats_global_only_count": 24,
        "calibrated_beats_local_only_count": 16,
        "transfer_task_count": 24,
        "pareto_recall_p10": 0.85,
        "hv_regret_p90": 0.08,
        "coverage_90_min": 0.82,
        "fully_covered_group_rate_90_min": 0.82,
    }


class Stage35Gold144SufficiencyV1Tests(unittest.TestCase):
    def test_manifest_sha256_must_match_frozen_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"schema_version":"test"}\n')
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            self.assertEqual(stage35.validate_manifest_sha256(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "manifest SHA256"):
                stage35.validate_manifest_sha256(path, "0" * 64)

    def test_gold144_contract_accepts_36_complete_groups_and_17_13_train_balance(self) -> None:
        rows, manifest = _gold144_fixture()

        stage35.validate_dataset_contract(rows, manifest)
        context = stage35.manifest_context(manifest)
        stage35.validate_context(context)
        selected, locked = stage35.manifest_group_split(
            context["train_pools"],
            locked_groups=context["locked_groups"],
            reference_groups=context["reference_groups"],
            train_count=30,
            seed=1,
        )

        self.assertEqual(stage35.LEARNING_SIZES, (6, 12, 18, 24, 26, 30))
        self.assertEqual(stage35.K_GROUPS, (2, 4, 6, 8))
        self.assertEqual(len(selected), 30)
        self.assertEqual(len(locked), 6)
        self.assertEqual(
            {model: len(groups) for model, groups in context["train_pools"].items()},
            {"codriving": 13, "pyramid": 17},
        )

    def test_gold144_contract_rejects_incomplete_four_arm_group(self) -> None:
        rows, manifest = _gold144_fixture()
        rows.pop()
        manifest["jobs"].pop()

        with self.assertRaisesRegex(ValueError, "144|four-arm"):
            stage35.validate_dataset_contract(rows, manifest)

    def test_gold144_contract_rejects_wrong_manifest_schema(self) -> None:
        rows, manifest = _gold144_fixture()
        manifest["schema_version"] = "stage35_gold128_manifest_v1"

        with self.assertRaisesRegex(ValueError, "stage35_gold144_manifest_v1"):
            stage35.validate_dataset_contract(rows, manifest)

    def test_gold144_contract_rejects_wrong_row_schema(self) -> None:
        rows, manifest = _gold144_fixture()
        rows[0]["schema_version"] = "stage35_gold128_final_v1"

        with self.assertRaisesRegex(ValueError, "stage35_gold144_final_v1"):
            stage35.validate_dataset_contract(rows, manifest)

    def test_lock_decision_uses_26_to_30_plateau_and_30_group_ranking(self) -> None:
        evidence = _passing_evidence()
        accepted = stage35.lock_decision(evidence, repeat_audit_qualified=True)
        self.assertEqual(accepted["decision"], "lock_gold144_for_stage4")

        evidence["learning"]["latency_ms"]["mae_improvement_26_to_30"] = 0.08
        rejected = stage35.lock_decision(evidence, repeat_audit_qualified=True)
        self.assertEqual(rejected["decision"], "targeted_supplement_required")
        self.assertFalse(rejected["checks"]["learning_plateau"])

        evidence = _passing_evidence()
        evidence["learning"]["ap70"]["spearman_30"] = 0.89
        rejected = stage35.lock_decision(evidence, repeat_audit_qualified=True)
        self.assertFalse(rejected["checks"]["ranking"])

    def test_original_gold128_repeat_audit_binding_is_accepted(self) -> None:
        rows, manifest = _gold144_fixture()
        source_rows = _source_gold128(rows)
        with tempfile.TemporaryDirectory() as directory:
            audit = _repeat_audit(source_rows, Path(directory), "c" * 64)

            self.assertTrue(stage35.validate_repeat_audit(
                audit,
                rows,
                gold144_manifest=manifest,
                source_gold_rows=source_rows,
                source_gold_sha256="c" * 64,
            ))

    def test_identical_feasibility_failure_without_metrics_is_frozen_evidence_match(self) -> None:
        failure = {
            "split": "train",
            "terminal_status": "feasibility_failure",
            "latency_ms": None,
            "energy_j": None,
            "ap30": None,
            "ap50": None,
            "ap70": None,
            "performance_result_sha256": None,
            "ap_report_sha256": None,
        }

        self.assertTrue(stage35._evidence_matches(failure, dict(failure)))

    def test_frozen_evidence_rejects_width_or_backend_binding_drift(self) -> None:
        source = {
            "group_id": "pyramid|16x64x128",
            "model": "pyramid",
            "width": [16, 64, 128],
            "q_mode": "fp16",
            "dispatch_key": "tvm_auto",
            "capability_profile_id": "tvm-profile",
            "split": "train",
            "terminal_status": "feasibility_failure",
            "failure_reason": "shape lowering failed",
            "latency_ms": None,
            "energy_j": None,
            "ap30": None,
            "ap50": None,
            "ap70": None,
            "performance_result_sha256": None,
            "ap_report_sha256": None,
        }
        changed_width = {**source, "width": [24, 64, 128]}
        changed_backend = {**source, "dispatch_key": "trt_engine"}

        self.assertFalse(stage35._evidence_matches(source, changed_width))
        self.assertFalse(stage35._evidence_matches(source, changed_backend))

    def test_dataset_contract_rejects_width_stratum_manifest_drift(self) -> None:
        rows, manifest = _gold144_fixture()
        manifest["jobs"][0]["width_stratum"] = "changed-after-freeze"

        with self.assertRaisesRegex(ValueError, "width_stratum"):
            stage35.validate_dataset_contract(rows, manifest)

    def test_any_original_locked_holdout_change_rejects_repeat_audit(self) -> None:
        rows, manifest = _gold144_fixture()
        source_rows = _source_gold128(rows)
        changed = [dict(row) for row in rows]
        locked = next(row for row in changed if row["split"] == "locked_holdout")
        locked["performance_result_sha256"] = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            audit = _repeat_audit(source_rows, Path(directory), "c" * 64)

            with self.assertRaisesRegex(ValueError, "locked holdout.*changed"):
                stage35.validate_repeat_audit(
                    audit,
                    changed,
                    gold144_manifest=manifest,
                    source_gold_rows=source_rows,
                    source_gold_sha256="c" * 64,
                )

    def test_any_non_repeat_gold128_frozen_evidence_change_is_rejected(self) -> None:
        rows, manifest = _gold144_fixture()
        source_rows = _source_gold128(rows)
        changed = [dict(row) for row in rows]
        changed_by_id = {row["manifest_job_id"]: row for row in changed}
        untouched_by_repeat = source_rows[-1]
        changed_by_id[untouched_by_repeat["manifest_job_id"]]["ap50"] = 0.699
        with tempfile.TemporaryDirectory() as directory:
            audit = _repeat_audit(source_rows, Path(directory), "c" * 64)

            with self.assertRaisesRegex(ValueError, "Gold128.*changed"):
                stage35.validate_repeat_audit(
                    audit,
                    changed,
                    gold144_manifest=manifest,
                    source_gold_rows=source_rows,
                    source_gold_sha256="c" * 64,
                )

    def test_gold128_source_pool_ids_must_match_all_source_rows(self) -> None:
        rows, manifest = _gold144_fixture()
        source_rows = _source_gold128(rows)
        changed = [dict(row) for row in rows]
        changed[0]["source_pool"] = "targeted16"
        with tempfile.TemporaryDirectory() as directory:
            audit = _repeat_audit(source_rows, Path(directory), "c" * 64)

            with self.assertRaisesRegex(ValueError, "source_pool=gold128.*IDs"):
                stage35.validate_repeat_audit(
                    audit,
                    changed,
                    gold144_manifest=manifest,
                    source_gold_rows=source_rows,
                    source_gold_sha256="c" * 64,
                )

    def test_gold144_manifest_must_bind_repeat_source_gold128_sha256(self) -> None:
        rows, manifest = _gold144_fixture()
        source_rows = _source_gold128(rows)
        manifest["source_manifests"]["gold128_sha256"] = "d" * 64
        with tempfile.TemporaryDirectory() as directory:
            audit = _repeat_audit(source_rows, Path(directory), "c" * 64)

            with self.assertRaisesRegex(ValueError, "provenance.*Gold128 SHA256"):
                stage35.validate_repeat_audit(
                    audit,
                    rows,
                    gold144_manifest=manifest,
                    source_gold_rows=source_rows,
                    source_gold_sha256="c" * 64,
                )

    def test_report_uses_gold144_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = stage35.write_report(
                Path(directory), {"decision": "targeted_supplement_required"}
            )

            self.assertEqual(path.name, "gold144_sufficiency_report.json")
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["decision"],
                "targeted_supplement_required",
            )


if __name__ == "__main__":
    unittest.main()
