import json
import tempfile
import unittest
from pathlib import Path

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage5.fcooper_space_v1 import WIDTH_SCHEMA
from scripts.stage6_prepare_fcooper_five_arm_v2 import (
    CONTROL_TASK_IDS,
    TASK_ID,
    build_five_arm_protocol,
    candidates_from_scanner_registry,
    fixed_original_candidate,
    load_frozen_gold176,
    select_tuned_remeasurements,
)


def scanner_group(index: int) -> dict:
    width = [32, 32, 32 + 32 * index, 32, 64]
    group_id = "|".join(
        ["fcooper", *[f"{axis}={value}" for axis, value in zip(WIDTH_SCHEMA, width)]]
    )
    return {
        "group_id": group_id,
        "model": "fcooper",
        "width": width,
        "width_schema": list(WIDTH_SCHEMA),
        "structure_widths": dict(zip(WIDTH_SCHEMA, width)),
        "source_status": "materializable",
        "materialization_kind": "fcooper_scanner_materialize_export",
        "source_evidence_kind": "materialization_plan",
        "source_evidence_sha256": f"{index + 1:064x}",
        "source_contract": {"scanner_group": index},
        "graph_features": {
            "group_id": group_id,
            "model": "fcooper",
            "width": width,
            "conv_count": 20 + index,
        },
    }


def successful_screen(row_id: str, rank: int) -> dict:
    return {
        "row_id": row_id,
        "task_id": CONTROL_TASK_IDS[("compress_then_tune", "screen")],
        "terminal_status": "measured_success_gold",
        "width": [32, 32, 32, 32, 64],
        "q_mode": "int8",
        "ap70": 0.60,
        "latency_ms": 1.0 + rank / 100,
        "energy_j": 0.2 + rank / 1000,
        "checkpoint_sha256": "a" * 64,
        "recovery_training_report_sha256": "b" * 64,
    }


class FCooperFiveArmPlanV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = build_capability_profile(
            capability_profile_id="h800-test",
            hardware_target="h800",
            compiler_fingerprint="a" * 64,
            dispatch_key="trt_engine",
            features={"tensor_core": 1},
        )

    def test_candidate_pool_is_expanded_only_from_scanner_registry(self) -> None:
        registry = {
            "schema_version": "stage5_candidate_source_registry_v1",
            "model": "fcooper",
            "width_schema": list(WIDTH_SCHEMA),
            "groups": [scanner_group(0), scanner_group(1)],
        }

        rows = candidates_from_scanner_registry(registry, self.profile)

        self.assertEqual(len(rows), 4)
        self.assertEqual({row["task_id"] for row in rows}, {TASK_ID})
        self.assertEqual(
            {tuple(row["width"]) for row in rows},
            {tuple(group["width"]) for group in registry["groups"]},
        )

    def test_scanner_registry_rejects_legacy_pilot_paths(self) -> None:
        group = scanner_group(0)
        group["source_contract"]["checkpoint_path"] = (
            "/tmp/fcooper_workpackage_a_20260723/checkpoint.pth"
        )
        registry = {
            "schema_version": "stage5_candidate_source_registry_v1",
            "model": "fcooper",
            "width_schema": list(WIDTH_SCHEMA),
            "groups": [group],
        }

        with self.assertRaisesRegex(ValueError, "forbidden pilot"):
            candidates_from_scanner_registry(registry, self.profile)

    def test_protocol_has_16_compression_and_unresolved_12_plus_4(self) -> None:
        ranked = [
            {
                "row_id": f"row-{index:02d}",
                "genome": [32, 32, 32 + index, 32, 64, "int8"],
            }
            for index in range(20)
        ]

        protocol = build_five_arm_protocol(ranked, candidate_pool_size=20)

        self.assertEqual(
            len(protocol["arms"]["compression_only"]["selected_row_ids"]), 16
        )
        tune = protocol["arms"]["compress_then_tune"]
        self.assertEqual(len(tune["screen_row_ids"]), 12)
        self.assertEqual(tune["tuned_row_ids"], [])
        self.assertEqual(tune["phase_status"], "awaiting_12_screen_observations")
        self.assertNotIn("locked_genomes", tune)
        self.assertEqual(
            protocol["arms"]["original_default"]["fixed_width"],
            [64, 128, 256, 128, 256],
        )
        self.assertEqual(
            protocol["arms"]["schedule_only"]["fixed_width"],
            [64, 128, 256, 128, 256],
        )

    def test_tuned_four_are_selected_after_all_12_screen_observations(self) -> None:
        screen_ids = [f"row-{index:02d}" for index in range(12)]
        feedback = [
            successful_screen(row_id, rank)
            for rank, row_id in enumerate(reversed(screen_ids))
        ]

        selected = select_tuned_remeasurements(
            feedback,
            screen_row_ids=screen_ids,
            ap70_ref=0.63,
            max_ap_drop=0.10,
        )

        self.assertEqual(
            [row["row_id"] for row in selected],
            ["row-11", "row-10", "row-09", "row-08"],
        )

    def test_tuned_four_cannot_be_selected_before_12_observations(self) -> None:
        screen_ids = [f"row-{index:02d}" for index in range(12)]

        with self.assertRaisesRegex(ValueError, "exactly 12"):
            select_tuned_remeasurements(
                [successful_screen(row_id, rank) for rank, row_id in enumerate(screen_ids[:11])],
                screen_row_ids=screen_ids,
                ap70_ref=0.63,
                max_ap_drop=0.10,
            )

    def test_tuned_four_reject_stage5_task_identity(self) -> None:
        screen_ids = [f"row-{index:02d}" for index in range(12)]
        feedback = [
            {
                **successful_screen(row_id, rank),
                "task_id": TASK_ID,
            }
            for rank, row_id in enumerate(screen_ids)
        ]

        with self.assertRaisesRegex(ValueError, "task identity drift"):
            select_tuned_remeasurements(
                feedback,
                screen_row_ids=screen_ids,
                ap70_ref=0.63,
                max_ap_drop=0.10,
            )

    def test_coldstart_loader_rejects_nonfrozen_stage5_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = root / "rows.json"
            graphs = root / "graphs.json"
            rows.write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "row_id": "stage5-online-row",
                                "terminal_status": "measured_success_gold",
                                "training_source": "online_feedback",
                            }
                        ]
                    }
                )
            )
            graphs.write_text(json.dumps({"graph_features": []}))

            with self.assertRaisesRegex(ValueError, "Gold176"):
                load_frozen_gold176(rows, graphs)

    def test_schedule_fp32_is_deterministically_derived_from_scanner_base_group(
        self,
    ) -> None:
        base = {
            "task_id": TASK_ID,
            "task_sha256": "a" * 64,
            "model": "fcooper",
            "hardware_id": "h800",
            "group_id": "fcooper|scanner-original",
            "width": [64, 128, 256, 128, 256],
            "width_schema": list(WIDTH_SCHEMA),
            "source_evidence_sha256": "b" * 64,
        }
        rows = [
            {
                **base,
                "row_id": f"base-{q_mode}",
                "manifest_job_id": f"base-{q_mode}",
                "q_mode": q_mode,
            }
            for q_mode in ("fp16", "int8")
        ]

        selected = fixed_original_candidate(rows)

        self.assertEqual(selected["q_mode"], "fp32")
        self.assertEqual(selected["width"], [64, 128, 256, 128, 256])
        self.assertEqual(
            selected["schedule_baseline_derivation"],
            "scanner_unique_original_structure_to_fixed_fp32",
        )
        self.assertEqual(selected["row_id"], selected["manifest_job_id"])
        self.assertEqual(len(selected["schedule_baseline_derivation_sha256"]), 64)
        self.assertNotIn(selected["row_id"], {"base-fp16", "base-int8"})


if __name__ == "__main__":
    unittest.main()
