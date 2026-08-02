from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_v2_gold_coldstart96 as cold96  # noqa: E402
import stage2_codriving_int8_provenance as provenance  # noqa: E402


def valid_int8_compile_summary() -> dict:
    signature = "input=2x64x256x512|weight=32x64x3x3"
    return {
        "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
        "conv_precision_plan": {"fused_conv2d": "int8"},
        "int8_signature_plan": {"fused_conv2d": signature},
        "int8_calibration": {
            "schema": provenance.INT8_CALIBRATION_SCHEMA,
            "quantization_semantics": provenance.INT8_QUANTIZATION_SEMANTICS,
            "calibration_source": "/tmp/calib.npz",
            "calibration_summary": "/tmp/calib_summary.json",
            "calibration_split": "train",
            "calibration_split_source": "/data/train.json",
            "calibration_samples": 16,
            "calibration_source_sha256": "a" * 64,
            "calibration_summary_sha256": "b" * 64,
            "spatial_features_shape": [16, 2, 64, 256, 512],
            "qmin": -127,
            "qmax": 127,
            "scales_by_signature": {
                signature: {"input_scale": 0.1, "weight_scale": 0.01},
            },
        },
    }


class V2GoldColdstart96PlanTests(unittest.TestCase):
    def test_build_plan_has_12_widths_8_strategies_and_no_backend_in_strategy_id(self) -> None:
        plan = cold96.build_plan()

        self.assertEqual(plan["schema_version"], "v2_gold_coldstart_96_launch_manifest_v1")
        self.assertEqual(len(plan["widths"]), 12)
        self.assertEqual(len(plan["strategy_combinations"]), 8)
        self.assertEqual(len(plan["jobs"]), 96)

        for job in plan["jobs"]:
            self.assertEqual(job["required_metrics"], ["latency_ms", "energy_j", "ap70"])
            self.assertTrue(job["trusted_for_final_frontier"])
            self.assertNotIn("tvm", job["strategy_id"])
            self.assertNotIn("trt", job["strategy_id"])
            self.assertNotIn("cutlass", job["strategy_id"])
            self.assertNotIn("backend", job["strategy_id"])
            self.assertEqual(job["strategy_id"], cold96.strategy_id(job["q_mode"], job["mixed_policy_id"]))

    def test_validate_training_rows_requires_96_complete_gold_rows(self) -> None:
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=cold96.DEFAULT_STRATEGY_COMBINATIONS[:2])
        complete = []
        for job in plan["jobs"]:
            complete.append(
                {
                    **{key: job[key] for key in cold96.JOB_ID_KEYS},
                    "strategy_id": job["strategy_id"],
                    "latency_ms": 1.0,
                    "energy_j": 0.1,
                    "ap70": 0.5,
                    "trusted_for_final_frontier": True,
                    "source_file": "result.json",
                    "ap_source_file": "ap.yaml",
                    "build_status": "success",
                    "run_status": "success",
                    "historical_prior_flag": False,
                }
            )

        audit = cold96.validate_training_rows(plan, complete)

        self.assertTrue(audit["complete"])
        self.assertEqual(audit["summary"]["complete_rows"], 2)
        self.assertEqual(audit["summary"]["missing_rows"], 0)
        self.assertEqual(audit["summary"]["failed_rows"], 0)

    def test_validate_training_rows_rejects_missing_metrics_prior_and_backend_strategy_id(self) -> None:
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=cold96.DEFAULT_STRATEGY_COMBINATIONS[:2])
        rows = [
            {
                **{key: plan["jobs"][0][key] for key in cold96.JOB_ID_KEYS},
                "strategy_id": "q=fp32|mixed=none",
                "latency_ms": 1.0,
                "energy_j": None,
                "ap70": 0.5,
                "trusted_for_final_frontier": True,
                "source_file": "bad.json",
                "build_status": "success",
                "run_status": "success",
                "historical_prior_flag": False,
            },
            {
                **{key: plan["jobs"][1][key] for key in cold96.JOB_ID_KEYS},
                "strategy_id": "q=int8|mixed=all_eligible_conv|backend=trt",
                "latency_ms": 1.0,
                "energy_j": 0.2,
                "ap70": 0.4,
                "trusted_for_final_frontier": True,
                "source_file": "bad2.json",
                "build_status": "success",
                "run_status": "success",
                "historical_prior_flag": True,
            },
        ]

        audit = cold96.validate_training_rows(plan, rows)

        self.assertFalse(audit["complete"])
        self.assertEqual(audit["summary"]["complete_rows"], 0)
        self.assertEqual(audit["summary"]["invalid_rows"], 2)
        reasons = {reason for issue in audit["invalid_rows"] for reason in issue["reasons"]}
        self.assertIn("missing_metric:energy_j", reasons)
        self.assertIn("backend_token_in_strategy_id", reasons)
        self.assertIn("historical_prior_in_gold", reasons)

    def test_validate_training_rows_rejects_ap70_without_ap_source_file(self) -> None:
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=cold96.DEFAULT_STRATEGY_COMBINATIONS[:1])
        job = plan["jobs"][0]
        rows = [
            {
                **{key: job[key] for key in cold96.JOB_ID_KEYS},
                "strategy_id": job["strategy_id"],
                "latency_ms": 1.0,
                "energy_j": 0.1,
                "ap70": 0.5,
                "trusted_for_final_frontier": True,
                "source_file": "latency_energy.json",
                "build_status": "success",
                "run_status": "success",
                "historical_prior_flag": False,
            }
        ]

        audit = cold96.validate_training_rows(plan, rows)

        self.assertFalse(audit["complete"])
        self.assertEqual(audit["summary"]["invalid_rows"], 1)
        reasons = {reason for issue in audit["invalid_rows"] for reason in issue["reasons"]}
        self.assertEqual(reasons, {"missing_ap_source_file"})

    def test_collect_existing_trt_row_keeps_ap70_as_required_metric(self) -> None:
        combo = next(item for item in cold96.DEFAULT_STRATEGY_COMBINATIONS if item["combo_id"] == "trt_fp16")
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=[combo])
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "codriving_trt_qxs8_20260708"
            result_dir.mkdir(parents=True)
            source = result_dir / "codriving_trt_16x32x64_fp16_20260708.json"
            source.write_text(
                json.dumps(
                    {
                        "framework": "tensorrt",
                        "precision": "fp16",
                        "build_success": True,
                        "lat_p50_ms": 0.35,
                        "energy_j": 0.10,
                    }
                ),
                encoding="utf-8",
            )

            rows = cold96.collect_existing_rows(plan, Path(tmpdir))
            audit = cold96.validate_training_rows(plan, rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["latency_ms"], 0.35)
        self.assertEqual(rows[0]["energy_j"], 0.10)
        self.assertIsNone(rows[0]["ap70"])
        self.assertFalse(audit["complete"])
        self.assertEqual(audit["summary"]["invalid_rows"], 1)
        reasons = {reason for issue in audit["invalid_rows"] for reason in issue["reasons"]}
        self.assertIn("missing_metric:ap70", reasons)

    def test_collect_existing_routeb_latency_row_does_not_invent_energy_or_ap(self) -> None:
        combo = next(item for item in cold96.DEFAULT_STRATEGY_COMBINATIONS if item["combo_id"] == "tvm_routeb_int8_all")
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=[combo])
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "codriving_routeb_qxs8_backboneonly_20260708"
            result_dir.mkdir(parents=True)
            source = result_dir / "codriving_routeb_16x32x64_backboneonly_20260708.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": "codriving_whole_engine_tc_v1",
                        "results": [
                            {"precision": "fp16", "status": "success", "latency_ms_p50": 2.5},
                            {"precision": "int8", "status": "success", "latency_ms_p50": 2.8},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = cold96.collect_existing_rows(plan, Path(tmpdir))
            audit = cold96.validate_training_rows(plan, rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["latency_ms"], 2.8)
        self.assertIsNone(rows[0]["energy_j"])
        self.assertIsNone(rows[0]["ap70"])
        self.assertFalse(audit["complete"])
        reasons = {reason for issue in audit["invalid_rows"] for reason in issue["reasons"]}
        self.assertIn("missing_metric:energy_j", reasons)
        self.assertIn("missing_metric:ap70", reasons)

    def test_collect_existing_cutlass_prefers_success_json_over_failed_sidecar(self) -> None:
        combo = next(item for item in cold96.DEFAULT_STRATEGY_COMBINATIONS if item["combo_id"] == "cutlass_fp16")
        plan = cold96.build_plan(widths=["24x64x128"], strategy_combinations=[combo])
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = Path(tmpdir) / "v2_gold_coldstart_96_20260708" / "cutlass_energy_raw" / "24x64x128"
            result_dir.mkdir(parents=True)
            (result_dir / "cutlass_fp16_energy_gpu4.failed.json").write_text(
                json.dumps({"status": "failed", "reason": "gpu_not_idle"}),
                encoding="utf-8",
            )
            (result_dir / "cutlass_fp16_energy_gpu5.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "latency_ms_p50": 1.2,
                        "energy_j": 0.3,
                    }
                ),
                encoding="utf-8",
            )

            rows = cold96.collect_existing_rows(plan, Path(tmpdir))
            audit = cold96.validate_training_rows(plan, rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_file"], str(result_dir / "cutlass_fp16_energy_gpu5.json"))
        self.assertEqual(rows[0]["latency_ms"], 1.2)
        self.assertEqual(rows[0]["energy_j"], 0.3)
        self.assertEqual(rows[0]["build_status"], "success")
        self.assertEqual(rows[0]["run_status"], "success")
        self.assertFalse(audit["complete"])
        reasons = {reason for issue in audit["invalid_rows"] for reason in issue["reasons"]}
        self.assertEqual(reasons, {"missing_metric:ap70"})

    def test_apply_codriving_fp_ap_overlays_only_baseline_and_cutlass_rows(self) -> None:
        combos = [
            item
            for item in cold96.DEFAULT_STRATEGY_COMBINATIONS
            if item["combo_id"] in {"tvm_fp32_baseline", "cutlass_fp16", "tvm_routeb_fp16", "trt_fp16"}
        ]
        plan = cold96.build_plan(widths=["24x32x96"], strategy_combinations=combos)
        rows = [cold96._base_row(job, Path("/tmp/source.json")) for job in plan["jobs"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            ap_dir = Path(tmpdir) / "v2_gold_coldstart_96_20260708" / "codriving_ap_raw" / "24x32x96"
            ap_dir.mkdir(parents=True)
            (ap_dir / "eval_intermediate_epoch5.yaml").write_text(
                "ap30: 0.61\nap_50: 0.52\nap_70: 0.37\n",
                encoding="utf-8",
            )

            cold96.apply_codriving_ap_overlays(rows, Path(tmpdir))

        baseline = next(row for row in rows if row["combo_id"] == "tvm_fp32_baseline")
        cutlass = next(row for row in rows if row["combo_id"] == "cutlass_fp16")
        routeb = next(row for row in rows if row["combo_id"] == "tvm_routeb_fp16")
        trt = next(row for row in rows if row["combo_id"] == "trt_fp16")
        self.assertEqual(baseline["ap70"], 0.37)
        self.assertEqual(cutlass["ap70"], 0.37)
        self.assertIsNone(routeb["ap70"])
        self.assertIsNone(trt["ap70"])

    def test_apply_codriving_trt_hybrid_ap_overlays_only_tensorrt_rows(self) -> None:
        combos = [
            item
            for item in cold96.DEFAULT_STRATEGY_COMBINATIONS
            if item["combo_id"] in {"trt_int8_all", "tvm_routeb_int8_all"}
        ]
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=combos)
        rows = [cold96._base_row(job, Path("/tmp/source.json")) for job in plan["jobs"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            ap_dir = Path(tmpdir) / "v2_gold_coldstart_96_20260708" / "codriving_trt_hybrid_ap_raw" / "16x32x64"
            ap_dir.mkdir(parents=True)
            (ap_dir / "trt_int8_all_final.json").write_text(
                json.dumps(
                    {
                        "schema": "v2_gold_coldstart_96_codriving_trt_hybrid_ap_eval_v1",
                        "ap70": 0.41,
                        "n_done": 1789,
                        "n_trt_path": 1789,
                        "n_fallback_path": 0,
                    }
                ),
                encoding="utf-8",
            )

            cold96.apply_codriving_ap_overlays(rows, Path(tmpdir))

        trt = next(row for row in rows if row["backend_context"] == "tensorrt")
        tvm = next(row for row in rows if row["backend_context"] == "tvm")
        self.assertEqual(trt["ap70"], 0.41)
        self.assertIsNone(tvm["ap70"])
        self.assertIn("trt_int8_all_final.json", trt["ap_source_file"])

    def test_apply_codriving_tvm_resnet_ap_overlays_rejects_legacy_dummy_scale_int8(self) -> None:
        combos = [
            item
            for item in cold96.DEFAULT_STRATEGY_COMBINATIONS
            if item["combo_id"] in {"tvm_routeb_int8_top25", "trt_int8_all"}
        ]
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=combos)
        rows = [cold96._base_row(job, Path("/tmp/source.json")) for job in plan["jobs"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            ap_dir = Path(tmpdir) / "v2_gold_coldstart_96_20260708" / "codriving_tvm_routeb_resnet_ap_raw" / "16x32x64"
            ap_dir.mkdir(parents=True)
            (ap_dir / "tvm_routeb_int8_top25_final.json").write_text(
                json.dumps(
                    {
                        "schema": "v2_gold_coldstart_96_codriving_tvm_resnet_ap_eval_v1",
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "ap70": 0.0,
                        "n_done": 1789,
                        "n_tvm_path": 1789,
                        "n_fallback_path": 0,
                    }
                ),
                encoding="utf-8",
            )

            cold96.apply_codriving_ap_overlays(rows, Path(tmpdir))

        tvm = next(row for row in rows if row["backend_context"] == "tvm")
        trt = next(row for row in rows if row["backend_context"] == "tensorrt")
        self.assertIsNone(tvm["ap70"])
        self.assertIsNone(trt["ap70"])

    def test_apply_codriving_tvm_resnet_ap_overlays_accepts_calibrated_int8(self) -> None:
        combo = next(
            item
            for item in cold96.DEFAULT_STRATEGY_COMBINATIONS
            if item["combo_id"] == "tvm_routeb_int8_top25"
        )
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=[combo])
        rows = [cold96._base_row(job, Path("/tmp/source.json")) for job in plan["jobs"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            ap_dir = Path(tmpdir) / "v2_gold_coldstart_96_20260708" / "codriving_tvm_routeb_resnet_ap_raw" / "16x32x64"
            ap_dir.mkdir(parents=True)
            (ap_dir / "tvm_routeb_int8_top25_final.json").write_text(
                json.dumps(
                    {
                        "schema": "v2_gold_coldstart_96_codriving_tvm_resnet_ap_eval_v1",
                        "tag": "tvm_routeb_int8_top25",
                        "mode": "mixed_top25_flops",
                        "precision": "mixed",
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "ap70": 0.39,
                        "n_done": 1789,
                        "n_tvm_path": 1789,
                        "n_fallback_path": 0,
                        "compile_summary": valid_int8_compile_summary(),
                    }
                ),
                encoding="utf-8",
            )

            cold96.apply_codriving_ap_overlays(rows, Path(tmpdir))

        self.assertEqual(rows[0]["ap70"], 0.39)
        self.assertEqual(rows[0]["ap_pipeline_scope"], "tvm_routeb_resnet_in_full_pytorch_eval")

    def test_apply_codriving_tvm_resnet_ap_overlays_rejects_fp16_payload_on_int8_path(self) -> None:
        combo = next(
            item
            for item in cold96.DEFAULT_STRATEGY_COMBINATIONS
            if item["combo_id"] == "tvm_routeb_int8_all"
        )
        plan = cold96.build_plan(widths=["16x32x64"], strategy_combinations=[combo])
        rows = [cold96._base_row(job, Path("/tmp/source.json")) for job in plan["jobs"]]

        with tempfile.TemporaryDirectory() as tmpdir:
            ap_dir = Path(tmpdir) / "v2_gold_coldstart_96_20260708" / "codriving_tvm_routeb_resnet_ap_raw" / "16x32x64"
            ap_dir.mkdir(parents=True)
            (ap_dir / "tvm_routeb_int8_all_final.json").write_text(
                json.dumps(
                    {
                        "schema": "v2_gold_coldstart_96_codriving_tvm_resnet_ap_eval_v1",
                        "tag": "tvm_routeb_int8_all",
                        "precision": "fp16",
                        "pipeline_scope": "tvm_routeb_resnet_in_full_pytorch_eval",
                        "ap70": 0.41,
                        "n_done": 1789,
                        "n_tvm_path": 1789,
                        "n_fallback_path": 0,
                        "compile_summary": {},
                    }
                ),
                encoding="utf-8",
            )

            cold96.apply_codriving_ap_overlays(rows, Path(tmpdir))

        self.assertIsNone(rows[0]["ap70"])


if __name__ == "__main__":
    unittest.main()
