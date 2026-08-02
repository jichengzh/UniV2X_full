from __future__ import annotations

import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from framework.stage2 import probe_conditioned_profiles_v3 as profiles  # noqa: E402
import stage2_finalize_capability_profiles_v3 as cli  # noqa: E402


def _record(
    *,
    probe_id: str,
    q_mode: str,
    build_success: bool,
    total_ops: int,
    reformat_ops: int,
    fused_ops: int,
    fusible_ops: int,
    int8_propagated_ops: int | None = None,
    precision_eligible_ops: int | None = None,
    qdq_folded_pairs: int | None = None,
    qdq_pairs: int | None = None,
) -> dict:
    return {
        "schema_version": "stage2_s1_structural_probe_record_v3",
        "probe_id": probe_id,
        "q_mode": q_mode,
        "build_success": build_success,
        "total_ops": total_ops,
        "reformat_ops": reformat_ops,
        "fused_ops": fused_ops,
        "fusible_ops": fusible_ops,
        "int8_propagated_ops": int8_propagated_ops,
        "precision_eligible_ops": precision_eligible_ops,
        "qdq_folded_pairs": qdq_folded_pairs,
        "qdq_pairs": qdq_pairs,
        "backend_runner": "fixture",
        "probe_seconds": 1.0,
        "build_seconds": 1.0,
        "onnx_sha256": "a" * 64,
        "error": None,
    }


def _run_payload(*, backend_runner: str, compiler_fingerprint: str, records: list[dict]) -> dict:
    return {
        "schema_version": "stage2_s1_structural_probe_run_v3",
        "backend_runner": backend_runner,
        "contains_latency_or_energy_measurement": False,
        "manifest_sha256": "b" * 64,
        "wall_seconds": 1.0,
        "provenance": {
            "compiler_fingerprint": compiler_fingerprint,
            "collector": "unit-test",
        },
        "records": [{**record, "backend_runner": backend_runner} for record in records],
    }


def _neutral_payload(*, backend_runner: str, compiler_fingerprint: str) -> dict:
    return _run_payload(
        backend_runner=backend_runner,
        compiler_fingerprint=compiler_fingerprint,
        records=[
            _record(
                probe_id="P1",
                q_mode="fp16",
                build_success=True,
                total_ops=4,
                reformat_ops=1,
                fused_ops=1,
                fusible_ops=2,
            ),
            _record(
                probe_id="P2",
                q_mode="fp16",
                build_success=False,
                total_ops=5,
                reformat_ops=3,
                fused_ops=0,
                fusible_ops=2,
            ),
            _record(
                probe_id="P1",
                q_mode="int8",
                build_success=True,
                total_ops=5,
                reformat_ops=1,
                fused_ops=1,
                fusible_ops=2,
                int8_propagated_ops=3,
                precision_eligible_ops=4,
                qdq_folded_pairs=2,
                qdq_pairs=4,
            ),
            _record(
                probe_id="P2",
                q_mode="int8",
                build_success=True,
                total_ops=4,
                reformat_ops=0,
                fused_ops=0,
                fusible_ops=1,
                int8_propagated_ops=2,
                precision_eligible_ops=2,
                qdq_folded_pairs=1,
                qdq_pairs=2,
            ),
        ],
    )


def _pruning_payload(*, backend_runner: str, compiler_fingerprint: str) -> dict:
    fp16_reformat = {
        "aligned_channels": 0,
        "misaligned_channels": 1,
        "small_channels": 2,
        "group_packed": 1,
        "group_unpacked": 3,
        "off_diagonal_two_stage_boundary": 2,
    }
    int8_reformat = {
        "aligned_channels": 0,
        "misaligned_channels": 1,
        "small_channels": 2,
        "group_packed": 1,
        "group_unpacked": 3,
        "off_diagonal_two_stage_boundary": 2,
    }
    int8_prop = {
        "aligned_channels": (4, 4),
        "misaligned_channels": (2, 4),
        "small_channels": (0, 4),
        "group_packed": (4, 4),
        "group_unpacked": (1, 4),
        "off_diagonal_two_stage_boundary": (3, 4),
    }
    int8_qdq = {
        "aligned_channels": (2, 2),
        "misaligned_channels": (1, 2),
        "small_channels": (0, 2),
        "group_packed": (1, 2),
        "group_unpacked": (0, 2),
        "off_diagonal_two_stage_boundary": (2, 3),
    }
    records: list[dict] = []
    for probe_id, reformat_ops in fp16_reformat.items():
        records.append(
            _record(
                probe_id=probe_id,
                q_mode="fp16",
                build_success=True,
                total_ops=4,
                reformat_ops=reformat_ops,
                fused_ops=0,
                fusible_ops=1,
            )
        )
    for probe_id, reformat_ops in int8_reformat.items():
        propagated, eligible = int8_prop[probe_id]
        folded, pairs = int8_qdq[probe_id]
        records.append(
            _record(
                probe_id=probe_id,
                q_mode="int8",
                build_success=True,
                total_ops=4,
                reformat_ops=reformat_ops,
                fused_ops=0,
                fusible_ops=1,
                int8_propagated_ops=propagated,
                precision_eligible_ops=eligible,
                qdq_folded_pairs=folded,
                qdq_pairs=pairs,
            )
        )
    return _run_payload(
        backend_runner=backend_runner,
        compiler_fingerprint=compiler_fingerprint,
        records=records,
    )


class Stage2ProbeConditionedProfilesV3Tests(unittest.TestCase):
    def test_build_probe_conditioned_profile_aggregates_structural_numeric_features_only(self) -> None:
        profile = profiles.build_probe_conditioned_profile(
            capability_profile_id="h800-probe-conditioned",
            hardware_target="h800",
            dispatch_key="runner",
            neutral_run=_neutral_payload(backend_runner="tvm", compiler_fingerprint="c" * 64),
            pruning_run=_pruning_payload(backend_runner="tvm", compiler_fingerprint="c" * 64),
        )

        self.assertEqual(profile["schema_version"], "stage2_capability_profile_v3")
        self.assertEqual(profile["compiler_fingerprint"], "c" * 64)
        self.assertEqual(profile["dispatch_key"], "runner")
        self.assertAlmostEqual(profile["features"]["s1q_build_success_coverage"], 0.75)
        self.assertAlmostEqual(profile["features"]["s1q_fp16_build_success_coverage"], 0.5)
        self.assertAlmostEqual(profile["features"]["s1q_int8_build_success_coverage"], 1.0)
        self.assertAlmostEqual(profile["features"]["s1q_int8_propagation_ratio_mean"], 0.875)
        self.assertAlmostEqual(profile["features"]["s1q_qdq_fold_ratio_mean"], 0.5)
        self.assertAlmostEqual(profile["features"]["s1q_reformat_rate_mean"], 0.15)
        self.assertAlmostEqual(profile["features"]["s1q_fusion_coverage_mean"], 1.0 / 3.0)
        self.assertAlmostEqual(profile["features"]["s1p_probe_pair_coverage"], 1.0)
        self.assertEqual(profile["features"]["s1p_probe_group_count_observed"], 6)

        fp16_rates = [0.0, 0.25, 0.5, 0.25, 0.75, 0.5]
        int8_rates = [0.0, 0.25, 0.5, 0.25, 0.75, 0.5]
        int8_prop_rates = [1.0, 0.5, 0.0, 1.0, 0.25, 0.75]
        int8_qdq_rates = [1.0, 0.5, 0.0, 0.5, 0.0, 2.0 / 3.0]
        self.assertAlmostEqual(
            profile["features"]["s1p_fp16_group_reformat_rate_stddev"],
            statistics.pstdev(fp16_rates),
        )
        self.assertAlmostEqual(
            profile["features"]["s1p_int8_group_reformat_rate_stddev"],
            statistics.pstdev(int8_rates),
        )
        self.assertAlmostEqual(
            profile["features"]["s1p_int8_group_propagation_rate_stddev"],
            statistics.pstdev(int8_prop_rates),
        )
        self.assertAlmostEqual(
            profile["features"]["s1p_int8_group_qdq_fold_rate_stddev"],
            statistics.pstdev(int8_qdq_rates),
        )

        serialized = json.dumps(profile["features"], sort_keys=True)
        self.assertNotIn("latency", serialized)
        self.assertNotIn("energy", serialized)
        self.assertNotIn("ap", serialized)
        self.assertNotIn("backend", serialized)
        self.assertNotIn("tvm", serialized)
        self.assertNotIn("trt", serialized)
        self.assertTrue(all(isinstance(value, (int, float)) for value in profile["features"].values()))

    def test_missing_compiler_fingerprint_blocks_profile_generation(self) -> None:
        blocked = _neutral_payload(backend_runner="tvm", compiler_fingerprint="d" * 64)
        blocked.pop("provenance")

        with self.assertRaisesRegex(ValueError, "compiler fingerprint missing"):
            profiles.build_probe_conditioned_profile(
                capability_profile_id="blocked",
                hardware_target="h800",
                dispatch_key="runner",
                neutral_run=blocked,
                pruning_run=_pruning_payload(backend_runner="tvm", compiler_fingerprint="d" * 64),
            )

        try:
            profiles.build_probe_conditioned_profile(
                capability_profile_id="blocked",
                hardware_target="h800",
                dispatch_key="runner",
                neutral_run=blocked,
                pruning_run=_pruning_payload(backend_runner="tvm", compiler_fingerprint="d" * 64),
            )
        except ValueError as exc:
            self.assertIn("provenance.compiler_fingerprint", str(exc))
            self.assertIn("re-collect", str(exc))

    def test_build_profiles_from_run_paths_requires_matching_backend_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            neutral = root / "neutral.json"
            pruning = root / "pruning.json"
            neutral.write_text(
                json.dumps(_neutral_payload(backend_runner="tvm", compiler_fingerprint="e" * 64)),
                encoding="utf-8",
            )
            pruning.write_text(
                json.dumps(_pruning_payload(backend_runner="trt", compiler_fingerprint="e" * 64)),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "backend_runner mismatch"):
                profiles.build_profile_from_run_paths(
                    capability_profile_id="bad",
                    hardware_target="h800",
                    dispatch_key="runner",
                    neutral_run_path=neutral,
                    pruning_run_path=pruning,
                )

    def test_cli_writes_two_profiles_and_blocks_on_real_missing_fingerprint_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            neutral_tvm = root / "neutral_tvm.json"
            pruning_tvm = root / "pruning_tvm.json"
            neutral_trt = root / "neutral_trt.json"
            pruning_trt = root / "pruning_trt.json"
            out_path = root / "capability_profiles_v3.json"
            neutral_tvm.write_text(
                json.dumps(_neutral_payload(backend_runner="tvm", compiler_fingerprint="1" * 64)),
                encoding="utf-8",
            )
            pruning_tvm.write_text(
                json.dumps(_pruning_payload(backend_runner="tvm", compiler_fingerprint="1" * 64)),
                encoding="utf-8",
            )
            neutral_trt.write_text(
                json.dumps(_neutral_payload(backend_runner="trt", compiler_fingerprint="2" * 64)),
                encoding="utf-8",
            )
            pruning_trt.write_text(
                json.dumps(_pruning_payload(backend_runner="trt", compiler_fingerprint="2" * 64)),
                encoding="utf-8",
            )

            exit_code = cli.main(
                [
                    "--hardware-target",
                    "h800",
                    "--neutral-tvm",
                    str(neutral_tvm),
                    "--pruning-tvm",
                    str(pruning_tvm),
                    "--neutral-trt",
                    str(neutral_trt),
                    "--pruning-trt",
                    str(pruning_trt),
                    "--out",
                    str(out_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 2)
            self.assertEqual(
                {item["capability_profile_id"] for item in payload},
                {"h800-tvm-probe-conditioned-v3", "h800-trt-probe-conditioned-v3"},
            )

            blocked_path = root / "blocked.json"
            blocked = _neutral_payload(backend_runner="tvm", compiler_fingerprint="3" * 64)
            blocked.pop("provenance")
            blocked_path.write_text(json.dumps(blocked), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "compiler fingerprint missing"):
                cli.build_profiles(
                    hardware_target="h800",
                    neutral_tvm_path=blocked_path,
                    pruning_tvm_path=pruning_tvm,
                    neutral_trt_path=neutral_trt,
                    pruning_trt_path=pruning_trt,
                )


if __name__ == "__main__":
    unittest.main()
