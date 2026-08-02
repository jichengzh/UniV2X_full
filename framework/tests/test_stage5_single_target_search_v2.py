from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5 import single_target_search_v2 as stage5


def _profile(dispatch_key: str = "tvm_auto") -> dict:
    return build_capability_profile(
        capability_profile_id=f"h800-{dispatch_key}-probe-conditioned-v3",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(dispatch_key.encode()).hexdigest(),
        dispatch_key=dispatch_key,
        features={"int8_propagation": 0.25, "qdq_fold": 0.5},
    )


def _registry() -> dict:
    groups = []
    for model in ("pyramid", "codriving"):
        for width in ([16, 32, 64], [24, 48, 96], [32, 64, 128]):
            group_id = f"{model}|{'x'.join(map(str, width))}"
            groups.append(
                {
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "source_status": "ready",
                    "source_evidence_sha256": "a" * 64,
                    "source_contract": {"onnx_path": f"/remote/{group_id}.onnx"},
                    "graph_features": {
                        "group_id": group_id,
                        "model": model,
                        "width": width,
                        "conv_count": 27 if model == "pyramid" else 24,
                        "conv_macs": float(width[0] * width[1] * width[2]),
                    },
                }
            )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


def _task() -> stage5.SearchTask:
    return stage5.SearchTask(
        task_id="S5-PYR-TVM",
        target_model="pyramid",
        hardware_id="h800",
        capability_profile=_profile("tvm_auto"),
        sample_budget=16,
        batch_size=4,
        round_count=4,
    )


def _fcooper_registry() -> dict:
    width_schema = [
        "backbone.s0",
        "backbone.s1",
        "backbone.s2",
        "neck.deblock",
        "neck.output",
    ]
    groups = []
    for width in (
        [64, 128, 256, 128, 256],
        [32, 32, 32, 32, 64],
        [64, 64, 128, 64, 128],
    ):
        assignments = dict(zip(width_schema, width))
        group_id = "fcooper|" + "|".join(
            f"{name}={assignments[name]}" for name in width_schema
        )
        groups.append(
            {
                "group_id": group_id,
                "model": "fcooper",
                "width": list(width),
                "width_schema": width_schema,
                "structure_widths": assignments,
                "source_status": "ready",
                "source_evidence_sha256": "b" * 64,
                "source_contract": {"onnx_path": f"/remote/{group_id}.onnx"},
                "graph_features": {
                    "group_id": group_id,
                    "model": "fcooper",
                    "width": list(width),
                    "conv_count": 31,
                    "conv_macs": float(width[0] * width[1] * width[2] + width[3]),
                },
            }
        )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


def _fcooper_task() -> stage5.SearchTask:
    return stage5.SearchTask(
        task_id="S5-FCO-TRT",
        target_model="fcooper",
        hardware_id="h800",
        capability_profile=_profile("trt_engine"),
        sample_budget=16,
        batch_size=4,
        round_count=4,
    )


class Stage5SingleTargetSearchV2Tests(unittest.TestCase):
    def test_initial_training_view_accepts_only_frozen_gold176(self) -> None:
        rows = [
            {
                "manifest_job_id": f"gold-{index}",
                "training_source": "initial_coldstart",
            }
            for index in range(176)
        ]

        frozen = stage5.freeze_initial_coldstart(rows)

        self.assertEqual(len(frozen), 176)
        self.assertEqual({row["training_source"] for row in frozen}, {"initial_coldstart"})
        self.assertNotEqual(id(frozen[0]), id(rows[0]))

    def test_initial_training_view_rejects_feedback_or_wrong_cardinality(self) -> None:
        rows = [
            {"manifest_job_id": f"gold-{index}", "training_source": "initial_coldstart"}
            for index in range(176)
        ]
        mixed = [*rows, {"manifest_job_id": "feedback-0", "training_source": "online_feedback"}]

        with self.assertRaisesRegex(ValueError, "exactly 176"):
            stage5.freeze_initial_coldstart(mixed)
        with self.assertRaisesRegex(ValueError, "initial_coldstart"):
            stage5.freeze_initial_coldstart([*rows[:-1], mixed[-1]])

    def test_task_fixes_model_hardware_profile_and_budget(self) -> None:
        task = _task()

        audit = stage5.validate_search_task(task)

        self.assertEqual(audit["target_model"], "pyramid")
        self.assertEqual(audit["dispatch_key"], "tvm_auto")
        self.assertEqual(audit["genome_schema"], ["w0", "w1", "w2", "q_mode"])
        self.assertEqual(audit["sample_budget"], 16)
        self.assertEqual(audit["batch_size"], 4)
        self.assertEqual(audit["round_count"], 4)
        self.assertNotIn("backend", audit["genome_schema"])

    def test_task_rejects_any_budget_other_than_frozen_b4_t16(self) -> None:
        with self.assertRaisesRegex(ValueError, "B=4, T=16"):
            stage5.validate_search_task(
                stage5.SearchTask(
                    "S5-PYR-TVM", "pyramid", "h800", _profile(),
                    sample_budget=20, batch_size=5, round_count=4,
                )
            )

    def test_frozen_coldstart_artifacts_require_pinned_content_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = root / "rows.json"
            graphs = root / "graphs.json"
            rows.write_text(json.dumps([{"row_id": "gold"}]), encoding="utf-8")
            graphs.write_text(json.dumps([{"group_id": "group"}]), encoding="utf-8")
            row_sha = hashlib.sha256(rows.read_bytes()).hexdigest()
            graph_sha = hashlib.sha256(graphs.read_bytes()).hexdigest()

            audit = stage5.verify_frozen_coldstart_artifacts(
                rows, graphs, expected_rows_sha256=row_sha,
                expected_graph_features_sha256=graph_sha,
            )
            self.assertEqual(audit["rows_sha256"], row_sha)
            with self.assertRaisesRegex(ValueError, "rows SHA"):
                stage5.verify_frozen_coldstart_artifacts(
                    rows, graphs, expected_rows_sha256="0" * 64,
                    expected_graph_features_sha256=graph_sha,
                )

    def test_feedback_history_rejects_profile_model_and_source_drift(self) -> None:
        task = _task()
        contract = stage5.validate_search_task(task)
        profile_id = task.capability_profile["capability_profile_id"]
        rows = [
            {
                "manifest_job_id": f"feedback-{index}",
                "task_id": task.task_id,
                "model": task.target_model,
                "hardware_id": task.hardware_id,
                "capability_profile_id": profile_id,
                "dispatch_key": "tvm_auto",
                "task_sha256": contract["task_sha256"],
                "training_source": "online_feedback",
                "terminal_status": "measured_success_gold",
            }
            for index in range(4)
        ]
        audit = stage5.validate_task_feedback_history(rows, task=task, completed_rounds=1)
        self.assertEqual(audit["feedback_rows"], 4)

        for field, value in (
            ("model", "codriving"),
            ("capability_profile_id", "wrong-profile"),
            ("training_source", "initial_coldstart"),
        ):
            drifted = [dict(row) for row in rows]
            drifted[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                stage5.validate_task_feedback_history(
                    drifted, task=task, completed_rounds=1
                )

    def test_candidate_manifest_is_single_profile_and_excludes_rows_not_width_groups(self) -> None:
        measured = {"pyramid|16x32x64|q=fp16|profile=h800-tvm_auto-probe-conditioned-v3"}

        manifest = stage5.build_task_candidate_manifest(
            _registry(), task=_task(), measured_row_ids=measured
        )

        self.assertEqual(manifest["task_id"], "S5-PYR-TVM")
        self.assertEqual(manifest["eligible_row_count"], 5)
        self.assertEqual({row["model"] for row in manifest["rows"]}, {"pyramid"})
        self.assertEqual({row["dispatch_key"] for row in manifest["rows"]}, {"tvm_auto"})
        self.assertEqual(
            {row["capability_profile_id"] for row in manifest["rows"]},
            {"h800-tvm_auto-probe-conditioned-v3"},
        )
        self.assertIn(
            "pyramid|16x32x64|q=int8|profile=h800-tvm_auto-probe-conditioned-v3",
            {row["row_id"] for row in manifest["rows"]},
        )
        self.assertTrue(all(len(row["genome"]) == 4 for row in manifest["rows"]))

    def test_candidate_manifest_does_not_remove_stage4_holdout_groups(self) -> None:
        manifest = stage5.build_task_candidate_manifest(
            _registry(),
            task=_task(),
            measured_row_ids=set(),
        )

        self.assertIn(
            "pyramid|16x32x64|q=fp16|profile=h800-tvm_auto-probe-conditioned-v3",
            {row["row_id"] for row in manifest["rows"]},
        )
        self.assertNotIn("frozen_independent_holdout", {row["reason"] for row in manifest["excluded"]})

    def test_fcooper_candidate_manifest_uses_scanner_derived_four_axis_genome(self) -> None:
        task = _fcooper_task()
        audit = stage5.validate_search_task(task)

        manifest = stage5.build_task_candidate_manifest(
            _fcooper_registry(), task=task, measured_row_ids=set()
        )

        self.assertEqual(
            audit["genome_schema"],
            [
                "backbone.s0",
                "backbone.s1",
                "backbone.s2",
                "neck.deblock",
                "neck.output",
                "q_mode",
            ],
        )
        self.assertEqual(manifest["eligible_row_count"], 6)
        self.assertTrue(all(len(row["width"]) == 5 for row in manifest["rows"]))
        self.assertTrue(all(len(row["genome"]) == 6 for row in manifest["rows"]))
        self.assertTrue(
            all(
                row["width_schema"]
                == [
                    "backbone.s0",
                    "backbone.s1",
                    "backbone.s2",
                    "neck.deblock",
                    "neck.output",
                ]
                for row in manifest["rows"]
            )
        )
        self.assertEqual(
            manifest["rows"][0]["structure_widths"],
            dict(zip(manifest["rows"][0]["width_schema"], manifest["rows"][0]["width"])),
        )

    def test_fcooper_candidate_manifest_rejects_three_width_template(self) -> None:
        registry = _fcooper_registry()
        registry["groups"][0]["width"] = [64, 128, 256]

        with self.assertRaisesRegex(ValueError, "structure identity"):
            stage5.build_task_candidate_manifest(
                registry, task=_fcooper_task(), measured_row_ids=set()
            )

    def test_acquisition_selects_four_independent_genomes_without_pair_expansion(self) -> None:
        manifest = stage5.build_task_candidate_manifest(
            _registry(), task=_task(), measured_row_ids=set()
        )
        predicted = []
        for index, row in enumerate(manifest["rows"]):
            predicted.append(
                {
                    **row,
                    "predictions": {
                        "latency_ms": 1.0 + index,
                        "energy_j": 0.1 + index,
                        "ap70": 0.8 - index * 0.01,
                    },
                    "prediction_intervals": {
                        key: {"lower": 0.0, "median": 0.5, "upper": 1.0}
                        for key in ("latency_ms", "energy_j", "ap70")
                    },
                }
            )

        selection = stage5.select_task_batch(
            predicted,
            measured_rows=[],
            measured_graph_features=[],
            task=_task(),
        )

        self.assertEqual(selection["selected_row_count"], 4)
        self.assertEqual(len({row["row_id"] for row in selection["selected_rows"]}), 4)
        self.assertTrue(all(row["model"] == "pyramid" for row in selection["selected_rows"]))
        self.assertNotIn("required_arm_product", selection)
        widths = [tuple(row["width"]) for row in selection["selected_rows"]]
        self.assertLess(len(set(widths)), 4)

    def test_atomic_batch_feedback_only_enters_training_after_all_four_terminal(self) -> None:
        manifest = stage5.build_task_candidate_manifest(
            _registry(), task=_task(), measured_row_ids=set()
        )
        rows = manifest["rows"][:4]
        request = stage5.build_measurement_request(
            task=_task(), selected_rows=rows, round_index=0
        )
        partial = [
            {
                **row,
                "measurement_request_row_sha256": request["row_sha256"][row["row_id"]],
                "terminal_status": "measured_success_gold",
                "latency_ms": 1.0,
                "energy_j": 0.2,
                "ap30": 0.9,
                "ap50": 0.8,
                "ap70": 0.7,
            }
            for row in rows[:3]
        ]

        with self.assertRaisesRegex(ValueError, "atomic batch"):
            stage5.finalize_atomic_batch(request, partial)

        terminal = [
            *partial,
            {
                **rows[3],
                "measurement_request_row_sha256": request["row_sha256"][rows[3]["row_id"]],
                "terminal_status": "feasibility_failure",
                "failure_reason": "true shape feasibility failure",
            },
        ]
        audit = stage5.finalize_atomic_batch(request, terminal)

        self.assertTrue(audit["feedback_released"])
        self.assertEqual(audit["budget_consumed"], 4)
        self.assertEqual(audit["successful_rows"], 3)
        self.assertEqual(audit["feasibility_terminal_rows"], 1)

    def test_public_runner_failure_quarantines_batch_without_consuming_budget(self) -> None:
        manifest = stage5.build_task_candidate_manifest(
            _registry(), task=_task(), measured_row_ids=set()
        )
        request = stage5.build_measurement_request(
            task=_task(), selected_rows=manifest["rows"][:4], round_index=0
        )
        feedback = [
            {
                **row,
                "measurement_request_row_sha256": request["row_sha256"][row["row_id"]],
                "terminal_status": "public_runner_failure",
                "failure_reason": "shared entrypoint regression",
            }
            for row in request["rows"]
        ]

        audit = stage5.finalize_atomic_batch(request, feedback)

        self.assertFalse(audit["feedback_released"])
        self.assertTrue(audit["batch_quarantined"])
        self.assertEqual(audit["budget_consumed"], 0)
        self.assertEqual(audit["resume_row_ids"], [row["row_id"] for row in request["rows"]])

    def test_atomic_batch_rejects_mutated_frozen_request_row(self) -> None:
        manifest = stage5.build_task_candidate_manifest(
            _registry(), task=_task(), measured_row_ids=set()
        )
        request = stage5.build_measurement_request(
            task=_task(), selected_rows=manifest["rows"][:4], round_index=0
        )
        request["rows"][0]["width"][0] += 1
        feedback = [
            {
                **row,
                "measurement_request_row_sha256": request["row_sha256"][row["row_id"]],
                "terminal_status": "measured_success_gold",
                "latency_ms": 1.0,
                "energy_j": 0.2,
                "ap30": 0.9,
                "ap50": 0.8,
                "ap70": 0.7,
            }
            for row in request["rows"]
        ]

        with self.assertRaisesRegex(ValueError, "request row SHA drift"):
            stage5.finalize_atomic_batch(request, feedback)

    def test_acquisition_rejects_nested_graph_metric_leakage(self) -> None:
        manifest = stage5.build_task_candidate_manifest(
            _registry(), task=_task(), measured_row_ids=set()
        )
        predicted = []
        for index, row in enumerate(manifest["rows"]):
            graph = {**row["graph_features"]}
            if index == 0:
                graph["latency_ms"] = 0.0
            predicted.append(
                {
                    **row,
                    "graph_features": graph,
                    "predictions": {
                        "latency_ms": 1.0 + index,
                        "energy_j": 0.1 + index,
                        "ap70": 0.8 - index * 0.01,
                    },
                    "prediction_intervals": {
                        key: {"lower": 0.0, "median": 0.5, "upper": 1.0}
                        for key in ("latency_ms", "energy_j", "ap70")
                    },
                }
            )

        with self.assertRaisesRegex(ValueError, "nested graph label"):
            stage5.select_task_batch(
                predicted,
                measured_rows=[],
                measured_graph_features=[],
                task=_task(),
            )


if __name__ == "__main__":
    unittest.main()
