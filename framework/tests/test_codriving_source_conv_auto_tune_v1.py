from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import stage2_codriving_source_conv_auto_tune_v1 as runner  # noqa: E402


class CoDrivingSourceConvAutoTuneV1Tests(unittest.TestCase):
    def test_real_anchor_contracts_match_top_two_codriving_convs(self) -> None:
        first, second = runner.ANCHORS

        self.assertEqual(first.input_shape, (2, 64, 256, 512))
        self.assertEqual(first.weight_shape, (32, 64, 3, 3))
        self.assertEqual(first.output_shape, (2, 32, 128, 256))
        self.assertEqual(first.strides, (2, 2))
        self.assertEqual(second.input_shape, first.output_shape)
        self.assertEqual(second.output_shape, (2, 32, 128, 256))

    def test_fp16_and_int8_region_realizations_cover_the_same_two_convs(self) -> None:
        self.assertEqual(runner.REALIZATIONS["fp16_rank2_region"], ("fp16_region", "rank2_region"))
        self.assertEqual(runner.REALIZATIONS["int8_rank2_region"], ("int8_region", "rank2_region"))
        self.assertEqual(runner.expected_tensorcore_conv_count("fp16_rank2_region"), 2)
        self.assertEqual(runner.expected_tensorcore_conv_count("int8_rank2_region"), 2)
        self.assertEqual(runner.expected_tensorcore_conv_count("fp16_rank1"), 1)

        fp16_signature = runner.region_structural_signature("fp16_rank2_region")
        int8_signature = runner.region_structural_signature("int8_rank2_region")
        self.assertEqual(fp16_signature["anchors"], int8_signature["anchors"])
        self.assertEqual(fp16_signature["region_input_shape"], int8_signature["region_input_shape"])
        self.assertEqual(fp16_signature["region_output_shape"], int8_signature["region_output_shape"])
        self.assertEqual(fp16_signature["semantic_ops"], int8_signature["semantic_ops"])
        self.assertEqual(fp16_signature["quantization_ops"], [])
        self.assertGreater(len(int8_signature["quantization_ops"]), 0)

    def test_source_guard_rejects_explicit_im2col_constructs(self) -> None:
        runner.validate_source_ir_text("R.nn.conv2d(x, weight)")

        for forbidden in (
            "x_col",
            "w_mat",
            "im2col",
            'name="matmul"',
            "@T.prim_func",
            "T.prim_func",
            "R.call_tir",
            "relax.call_tir",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaisesRegex(ValueError, "explicit TE/im2col"):
                    runner.validate_source_ir_text(f"R.nn.conv2d(x, weight) {forbidden}")

    def test_tir_product_classifier_requires_real_tensorcore_evidence(self) -> None:
        tensorcore = runner.classify_tir_products(
            "T.tvm_mma_sync(...) T.tvm_load_matrix_sync(...) wmma.matrix_a"
        )
        fallback = runner.classify_tir_products("for i in T.serial(128): accumulator += value")

        self.assertEqual(tensorcore["path_class"], "TENSORCORE_MMA")
        self.assertTrue(tensorcore["tensorcore_evidence"])
        self.assertEqual(fallback["path_class"], "SCALAR_OR_FALLBACK")
        self.assertFalse(fallback["tensorcore_evidence"])

    def test_region_gate_requires_tensorcore_evidence_for_each_conv_primfunc(self) -> None:
        scheduled = """
        @T.prim_func(private=True)
        def fused_conv2d_first(...):
            conv2d_nchw_intermediate[...] = value
        @T.prim_func(private=True)
        def fused_conv2d_second(...):
            T.tvm_mma_sync(...)
        """
        products = runner.classify_tir_products(scheduled)
        result = {
            "lowering_origin": "source_ir_automatic",
            "source_ir_sha256": "a" * 64,
            "schedule_trace_sha256": "b" * 64,
            "tuning_database_sha256": "c" * 64,
            "build_status": "success",
            "numerical_status": "pass",
            "expected_tensorcore_conv_count": 2,
            "tir_products": products,
        }

        self.assertEqual(products["conv_primfunc_count"], 2)
        self.assertEqual(products["tensorcore_conv_count"], 1)
        self.assertFalse(runner.automatic_candidate_eligible(result))

    def test_candidate_gate_requires_source_origin_trace_db_and_numerical_pass(self) -> None:
        result = {
            "lowering_origin": "source_ir_automatic",
            "source_ir_sha256": "a" * 64,
            "schedule_trace_sha256": "b" * 64,
            "tuning_database_sha256": "c" * 64,
            "build_status": "success",
            "numerical_status": "pass",
            "tir_products": {"tensorcore_evidence": True},
        }

        self.assertTrue(runner.automatic_candidate_eligible(result))
        self.assertFalse(
            runner.automatic_candidate_eligible({**result, "lowering_origin": "explicit_te_constructed"})
        )
        self.assertFalse(
            runner.automatic_candidate_eligible({**result, "tir_products": {"tensorcore_evidence": False}})
        )

    def test_runtime_support_imports_from_direct_script_execution_path(self) -> None:
        configure_tvm_env, wait_gpu_idle = runner.load_runtime_support()

        self.assertTrue(callable(configure_tvm_env))
        self.assertTrue(callable(wait_gpu_idle))

    def test_diagnostic_calibration_never_enters_search_training(self) -> None:
        result = {
            "action": "measure",
            "calibration_kind": "diagnostic",
            "lowering_origin": "source_ir_automatic",
            "source_ir_sha256": "a" * 64,
            "schedule_trace_sha256": "b" * 64,
            "tuning_database_sha256": "c" * 64,
            "build_status": "success",
            "numerical_status": "pass",
            "expected_tensorcore_conv_count": 1,
            "tir_products": {"tensorcore_evidence": True, "tensorcore_conv_count": 1},
        }

        self.assertTrue(runner.automatic_candidate_eligible(result))
        self.assertFalse(runner.search_training_eligible(result))


if __name__ == "__main__":
    unittest.main()
