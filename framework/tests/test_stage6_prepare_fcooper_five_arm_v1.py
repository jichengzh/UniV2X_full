import unittest

from scripts.stage6_prepare_fcooper_five_arm_v1 import (
    genome_key,
    rank_backend_neutral_candidates,
)


def candidate(width: list[int], q_mode: str) -> dict:
    return {
        "model": "fcooper",
        "width": width,
        "width_schema": [
            "backbone.s0",
            "backbone.s1",
            "backbone.s2",
            "neck.deblock",
            "neck.output",
        ],
        "q_mode": q_mode,
        "genome": [*width, q_mode],
        "row_id": f"{width}|{q_mode}",
        "source_evidence_sha256": "a" * 64,
    }


class FCooperFiveArmPlanTest(unittest.TestCase):
    def test_genome_key_preserves_all_five_scanner_axes(self) -> None:
        row = candidate([64, 128, 256, 128, 256], "fp16")

        self.assertEqual(
            genome_key(row),
            (64, 128, 256, 128, 256, "fp16"),
        )

    def test_backend_neutral_ranking_uses_ap_and_bit_cost_only(self) -> None:
        first = candidate([64, 128, 256, 128, 256], "fp16")
        second = candidate([32, 64, 128, 64, 128], "int8")
        rows = [first, second]
        ap = {genome_key(first): 0.63, genome_key(second): 0.62}
        graph_costs = {
            tuple(first["width"]): {
                "parameter_elements": 1000.0,
                "conv_flops": 2000.0,
            },
            tuple(second["width"]): {
                "parameter_elements": 200.0,
                "conv_flops": 400.0,
            },
        }

        ranked = rank_backend_neutral_candidates(rows, ap, graph_costs)

        self.assertEqual(ranked[0]["genome"], second["genome"])
        self.assertNotIn("latency_ms", ranked[0])
        self.assertNotIn("energy_j", ranked[0])


if __name__ == "__main__":
    unittest.main()
