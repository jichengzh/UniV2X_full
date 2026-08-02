from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/stage35_gold176_sufficiency_v1.py"
if MODULE_PATH.is_file():
    spec = importlib.util.spec_from_file_location("stage35_gold176_sufficiency_v1", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    stage35 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage35)
else:
    stage35 = None


ARMS = (
    ("tvm_auto", "fp16"),
    ("tvm_auto", "int8"),
    ("trt_engine", "fp16"),
    ("trt_engine", "int8"),
)


def _gold176_fixture() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    jobs: list[dict[str, object]] = []
    pilot_group_ids: list[str] = []
    model_groups = {
        "codriving": (range(17), range(17, 20), 13),
        "pyramid": (range(21), range(21, 24), 17),
    }
    for model, (train_indices, locked_indices, base_train_count) in model_groups.items():
        pilot_group_ids.append(f"{model}|g0")
        for group_index in [*train_indices, *locked_indices]:
            group_id = f"{model}|g{group_index}"
            split = "locked_holdout" if group_index in locked_indices else "train"
            source_pool = (
                "gold144"
                if split == "locked_holdout" or group_index < base_train_count
                else "targeted32"
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
                jobs.append({"job_id": job_id, **common})
                rows.append({
                    "schema_version": "stage35_gold176_final_v1",
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
        "schema_version": "stage35_gold176_manifest_v1",
        "jobs": jobs,
        "pilot_group_ids": pilot_group_ids,
        "source_manifests": {"gold144_sha256": "c" * 64},
    }


def _source_gold144(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        if row["source_pool"] != "gold144":
            continue
        source = dict(row)
        source["schema_version"] = "stage35_gold144_final_v1"
        source["source_pool"] = "gold128"
        output.append(source)
    return output


def _passing_evidence() -> dict[str, object]:
    return {
        "learning": {
            target: {
                "mae_improvement_32_to_34": 0.03,
                "mae_improvement_34_to_38": 0.04,
                "spearman_38": 0.92,
                "topk_recall_38": 0.8,
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


@unittest.skipIf(stage35 is None, "Gold176 sufficiency module not implemented")
class Stage35Gold176SufficiencyBehaviorTests(unittest.TestCase):
    def test_contract_accepts_44_complete_groups_and_21_17_train_balance(self) -> None:
        rows, manifest = _gold176_fixture()

        stage35.validate_dataset_contract(rows, manifest)
        context = stage35.manifest_context(manifest)
        stage35.validate_context(context)
        selected, locked = stage35.manifest_group_split(
            context["train_pools"],
            locked_groups=context["locked_groups"],
            reference_groups=context["reference_groups"],
            train_count=38,
            seed=1,
        )

        self.assertEqual(stage35.LEARNING_SIZES, (6, 12, 18, 24, 28, 30, 32, 34, 38))
        self.assertEqual(stage35.K_GROUPS, (2, 4, 6, 8))
        self.assertEqual(len(selected), 38)
        self.assertEqual(len(locked), 6)
        self.assertEqual(
            {model: len(groups) for model, groups in context["train_pools"].items()},
            {"codriving": 17, "pyramid": 21},
        )

    def test_contract_rejects_incomplete_four_arm_group(self) -> None:
        rows, manifest = _gold176_fixture()
        rows.pop()
        manifest["jobs"].pop()

        with self.assertRaisesRegex(ValueError, "176|four-arm"):
            stage35.validate_dataset_contract(rows, manifest)

    def test_gold144_source_rows_are_frozen_by_manifest_digest(self) -> None:
        rows, manifest = _gold176_fixture()
        source = _source_gold144(rows)

        self.assertTrue(stage35.validate_gold144_source(
            rows,
            manifest,
            source_gold144_rows=source,
            source_gold144_sha256="c" * 64,
        ))

        changed = [dict(row) for row in rows]
        base_row = next(row for row in changed if row["source_pool"] == "gold144")
        base_row["latency_ms"] = float(base_row["latency_ms"]) + 0.1
        with self.assertRaisesRegex(ValueError, "Gold144.*changed"):
            stage35.validate_gold144_source(
                changed,
                manifest,
                source_gold144_rows=source,
                source_gold144_sha256="c" * 64,
            )

    def test_lock_decision_uses_balanced_and_extension_plateau_with_full_ranking(self) -> None:
        accepted = stage35.lock_decision(
            _passing_evidence(),
            repeat_audit_qualified=True,
            holdout_independent=True,
        )
        self.assertEqual(accepted["decision"], "lock_gold176_for_stage4")

        evidence = _passing_evidence()
        evidence["learning"]["ap70"]["topk_recall_38"] = 0.74
        rejected = stage35.lock_decision(
            evidence,
            repeat_audit_qualified=True,
            holdout_independent=True,
        )
        self.assertEqual(rejected["decision"], "targeted_supplement_required")
        self.assertFalse(rejected["checks"]["ranking"])

    def test_lock_requires_new_repeat_audit_and_independent_holdout(self) -> None:
        missing_repeat = stage35.lock_decision(
            _passing_evidence(),
            repeat_audit_qualified=None,
            holdout_independent=True,
        )
        self.assertEqual(missing_repeat["decision"], "repeat_audit_required")

        reused_holdout = stage35.lock_decision(
            _passing_evidence(),
            repeat_audit_qualified=True,
            holdout_independent=False,
        )
        self.assertEqual(reused_holdout["decision"], "independent_holdout_required")

    def test_candidate_selected_from_sufficiency_report_is_not_independent(self) -> None:
        candidate = {
            "selection_policy": "freeze_before_targeted_measurement",
            "source_sufficiency_report_sha256": "d" * 64,
        }

        assessment = stage35.assess_holdout_independence(candidate)

        self.assertFalse(assessment["holdout_independent"])
        self.assertEqual(
            assessment["reason"], "candidate_selection_used_prior_holdout_report"
        )

    def test_gate_comparison_reports_resolved_and_unresolved_failures(self) -> None:
        baseline = {
            "decision": "targeted_supplement_required",
            "checks": {"ranking": False, "pareto_recall": False, "learning_plateau": True},
        }
        current = {
            "decision": "targeted_supplement_required",
            "checks": {"ranking": True, "pareto_recall": False, "learning_plateau": False},
        }

        comparison = stage35.compare_gate_reports(baseline, current)

        self.assertEqual(comparison["resolved_failures"], ["ranking"])
        self.assertEqual(comparison["unresolved_failures"], ["pareto_recall"])
        self.assertEqual(comparison["new_regressions"], [])
        self.assertEqual(comparison["non_comparable_gates"], ["learning_plateau"])
        self.assertFalse(comparison["all_previous_failures_resolved"])

    def test_report_writes_gold176_and_gate_comparison_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report_path = stage35.write_report(output, {"decision": "test"})
            comparison_path = stage35.write_gate_comparison(output, {"resolved_failures": []})

            self.assertEqual(report_path.name, "gold176_sufficiency_report.json")
            self.assertEqual(comparison_path.name, "gold144_vs_gold176_gate_comparison.json")
            self.assertEqual(json.loads(report_path.read_text())["decision"], "test")


class Stage35Gold176SufficiencyModuleTests(unittest.TestCase):
    def test_module_exists(self) -> None:
        self.assertIsNotNone(stage35, "scripts/stage35_gold176_sufficiency_v1.py is missing")


if __name__ == "__main__":
    unittest.main()
