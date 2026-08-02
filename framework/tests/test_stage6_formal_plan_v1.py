from __future__ import annotations

import unittest

from framework.stage6.formal_plan_v1 import build_formal_plan


def candidate(index: int, backend: str) -> dict:
    q_mode = "int8" if index % 2 else "fp16"
    width = [16 + index, 32 + index, 64 + index]
    profile = f"h800-{backend}-probe-conditioned-v3"
    return {
        "genome": [*width, q_mode],
        "width": width,
        "q_mode": q_mode,
        "group_id": f"pyramid|{'x'.join(map(str, width))}",
        "manifest_job_id": (
            f"pyramid|{'x'.join(map(str, width))}|q={q_mode}|profile={profile}"
        ),
        "capability_profile_id": profile,
        "dispatch_key": "tvm_auto" if backend == "tvm" else "trt_engine",
        "source_contract": {"onnx_path": f"/{index}.onnx"},
        "source_evidence_sha256": str(index).zfill(64),
        "graph_features": {
            "parameter_elements": 1000 + index,
            "conv_flops": 2000 + index,
        },
        "predictions": {"ap70": 0.5 + index / 1000},
    }


class Stage6FormalPlanV1Tests(unittest.TestCase):
    def test_builds_backend_symmetric_frozen_arm_plans(self) -> None:
        tvm = [candidate(index, "tvm") for index in range(20)]
        trt = [candidate(index, "trt") for index in range(20)]

        plan = build_formal_plan(tvm, trt, expected_pool_size=20)

        self.assertTrue(plan["passed"])
        self.assertEqual(plan["effective_candidate_pool_size"], 20)
        self.assertEqual(len(plan["arms"]["compression_only"]["selected_genomes"]), 16)
        self.assertEqual(len(plan["arms"]["compress_then_tune"]["screened_genomes"]), 12)
        self.assertEqual(len(plan["arms"]["compress_then_tune"]["locked_genomes"]), 4)
        self.assertEqual(len(plan["arms"]["tune_then_compress"]["attempt_genomes"]), 16)
        self.assertEqual(
            plan["backend_plans"]["tvm"]["compression_only"],
            plan["backend_plans"]["trt"]["compression_only"],
        )

    def test_rejects_backend_candidate_pool_drift(self) -> None:
        tvm = [candidate(index, "tvm") for index in range(20)]
        trt = [candidate(index, "trt") for index in range(19)]

        with self.assertRaisesRegex(ValueError, "candidate genome sets differ"):
            build_formal_plan(tvm, trt, expected_pool_size=20)

    def test_rejects_backend_dependent_ap_surrogate(self) -> None:
        tvm = [candidate(index, "tvm") for index in range(20)]
        trt = [candidate(index, "trt") for index in range(20)]
        trt[0]["predictions"]["ap70"] += 0.1

        with self.assertRaisesRegex(ValueError, "AP surrogate drift"):
            build_formal_plan(tvm, trt, expected_pool_size=20)
