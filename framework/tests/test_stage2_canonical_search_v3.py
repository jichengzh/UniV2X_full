from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from framework.stage2 import canonical_search_v3 as canonical  # noqa: E402


WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
    "24x64x128",
    "40x64x128",
    "48x64x128",
    "64x64x128",
)


def _profile(profile_id: str, dispatch_key: str, int8_ratio: float) -> dict:
    return canonical.build_capability_profile(
        capability_profile_id=profile_id,
        hardware_target="h800",
        compiler_fingerprint="a" * 64,
        dispatch_key=dispatch_key,
        features={
            "supports_fp16_tensorcore": 1.0,
            "supports_int8_tensorcore": 1.0,
            "int8_precision_propagation_ratio": 1.0,
            "int8_channel_alignment": int8_ratio,
            "build_success_rate": 1.0,
        },
    )


class Stage2CanonicalSearchV3Tests(unittest.TestCase):
    def test_capability_digest_uses_numeric_features_not_dispatch_name(self) -> None:
        tvm = _profile("h800-profile-a", "tvm_auto", 1.29)
        renamed_dispatch = {**tvm, "dispatch_key": "renamed_internal_runner"}
        changed_feature = copy.deepcopy(tvm)
        changed_feature["features"]["int8_channel_alignment"] = 0.70

        self.assertEqual(canonical.compute_capability_digest(tvm), tvm["capability_digest"])
        self.assertEqual(canonical.compute_capability_digest(renamed_dispatch), tvm["capability_digest"])
        self.assertNotEqual(canonical.compute_capability_digest(changed_feature), tvm["capability_digest"])

    def test_target_workload_performance_ratios_are_rejected_as_capability_features(self) -> None:
        with self.assertRaisesRegex(ValueError, "target-derived performance"):
            canonical.build_capability_profile(
                capability_profile_id="leaked",
                hardware_target="h800",
                compiler_fingerprint="a" * 64,
                dispatch_key="runner",
                features={"int8_over_fp16_latency_ratio": 1.29},
            )
        with self.assertRaisesRegex(ValueError, "target-derived performance"):
            canonical.build_capability_profile(
                capability_profile_id="leaked-ap",
                hardware_target="h800",
                compiler_fingerprint="a" * 64,
                dispatch_key="runner",
                features={"ap70": 0.5},
            )

    def test_active_manifest_is_exactly_two_models_twelve_widths_two_q_two_profiles(self) -> None:
        profiles = [_profile("h800-profile-a", "tvm_auto", 1.29), _profile("h800-profile-b", "trt", 0.70)]
        manifest = canonical.build_active_manifest(
            widths_by_model={"pyramid": WIDTHS, "codriving": WIDTHS},
            capability_profiles=profiles,
        )

        self.assertEqual(manifest["genome_schema"], ["p1", "p2", "p3", "q_mode"])
        self.assertEqual(manifest["q_modes"], ["fp16", "int8"])
        self.assertEqual(len(manifest["jobs"]), 96)
        self.assertEqual(len({job["group_id"] for job in manifest["jobs"]}), 24)
        for job in manifest["jobs"]:
            self.assertIn(job["strategy_id"], {"q=fp16", "q=int8"})
            self.assertEqual(job["mixed_policy_id"], "none")
            self.assertNotIn("tvm", job["strategy_id"])
            self.assertNotIn("trt", job["strategy_id"])
            self.assertNotIn("backend", job["strategy_id"])
        counts = {group: 0 for group in {job["group_id"] for job in manifest["jobs"]}}
        for job in manifest["jobs"]:
            counts[job["group_id"]] += 1
        self.assertEqual(set(counts.values()), {4})

    def test_grouped_split_is_exactly_80_train_16_holdout_without_group_leakage(self) -> None:
        profiles = [_profile("h800-profile-a", "tvm_auto", 1.29), _profile("h800-profile-b", "trt", 0.70)]
        manifest = canonical.build_active_manifest(
            widths_by_model={"pyramid": WIDTHS, "codriving": WIDTHS},
            capability_profiles=profiles,
        )

        split = canonical.assign_grouped_split(manifest["jobs"], holdout_group_count=4, seed=42)

        self.assertEqual(len(split["train"]), 80)
        self.assertEqual(len(split["holdout"]), 16)
        train_groups = {row["group_id"] for row in split["train"]}
        holdout_groups = {row["group_id"] for row in split["holdout"]}
        self.assertFalse(train_groups & holdout_groups)

    def test_historical_180_router_never_emits_backend_gold(self) -> None:
        rows = [
            {
                "label": "a",
                "precision": "int8",
                "width": [24, 32, 96],
                "can_use_as_cold_start_prior": True,
                "trusted_for_final_frontier": False,
                "training_source_class": "historical_hand_rewrite_diagnostic",
                "routeb_int8_latency_ms_if_measured": 3.6,
                "phase1_old_int8tc_latency_ms_if_available": 1.7,
            },
            {
                "label": "b",
                "precision": "fp16",
                "width": [32, 64, 128],
                "can_use_as_cold_start_prior": True,
                "trusted_for_final_frontier": False,
                "training_source_class": "legacy_or_rewritten_fp16_diagnostic",
                "routeb_int8_latency_ms_if_measured": None,
                "phase1_old_int8tc_latency_ms_if_available": None,
            },
        ]

        routed = canonical.route_historical_180(rows, disagreement_ratio=1.5)

        self.assertEqual(len(routed["width_ap_prior"]), 2)
        self.assertEqual(len(routed["historical_ablation"]), 2)
        self.assertEqual([row["label"] for row in routed["disagreement_probe"]], ["a"])
        self.assertEqual(routed["backend_gold"], [])
        self.assertEqual(routed["final_frontier"], [])


if __name__ == "__main__":
    unittest.main()
