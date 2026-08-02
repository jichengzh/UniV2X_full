from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from framework.stage2.canonical_search_v3 import build_capability_profile
from framework.stage7 import search_policy_v1 as policy
from framework.tests.test_stage7_search_policy_v1 import _training_context
from scripts import stage7_prepare_online_ablation_v1 as prepare


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile() -> dict:
    return build_capability_profile(
        capability_profile_id="h800-tvm-auto-formal-v3",
        hardware_target="h800",
        compiler_fingerprint=hashlib.sha256(b"compiler").hexdigest(),
        dispatch_key="tvm_auto",
        features={
            "s1p_fp16_build_success_coverage": 1.0,
            "s1q_fp16_build_success_coverage": 1.0,
            "s1p_int8_build_success_coverage": 1.0,
            "s1q_int8_build_success_coverage": 1.0,
        },
    )


def _registry() -> dict:
    groups = []
    for index, width in enumerate(
        ([16, 32, 64], [24, 48, 96], [32, 64, 128], [40, 80, 160])
    ):
        group_id = "pyramid|" + "x".join(map(str, width))
        groups.append(
            {
                "group_id": group_id,
                "model": "pyramid",
                "width": list(width),
                "source_status": "ready",
                "source_evidence_sha256": f"{index + 1:064x}",
                "source_contract": {
                    "checkpoint_path": f"/frozen/{group_id}.pth",
                    "onnx_path": f"/frozen/{group_id}.onnx",
                },
                "graph_features": {
                    "group_id": group_id,
                    "model": "pyramid",
                    "width": list(width),
                    "conv_count": 20 + index,
                },
            }
        )
    return {"schema_version": "stage5_candidate_source_registry_v1", "groups": groups}


def _inputs(root: Path, *, malformed_gold: bool = False) -> tuple[prepare.FrozenInputPaths, dict]:
    paths = prepare.FrozenInputPaths(
        gold176=root / "gold.json",
        graph_features=root / "graphs.json",
        capability_profiles=root / "profiles.json",
        source_registry=root / "registry.json",
        stage4_closure=root / "closure.json",
    )
    if malformed_gold:
        paths.gold176.write_text("{not-json", encoding="utf-8")
    else:
        _write_json(
            paths.gold176,
            {
                "rows": [
                    {
                        "row_id": (
                            "pyramid|16x32x64|q=fp16|profile="
                            "h800-tvm-auto-formal-v3"
                        ),
                        "manifest_job_id": (
                            "pyramid|16x32x64|q=fp16|profile="
                            "h800-tvm-auto-formal-v3"
                        ),
                        "group_id": "pyramid|16x32x64",
                        "model": "pyramid",
                        "dispatch_key": "tvm_auto",
                        "terminal_status": "measured_success_gold",
                        "latency_ms": 2.0,
                        "energy_j": 0.2,
                        "ap70": 0.7,
                    }
                ]
            },
        )
    _write_json(paths.graph_features, {"graph_features": []})
    _write_json(paths.capability_profiles, {"capability_profiles": [_profile()]})
    _write_json(paths.source_registry, _registry())
    _write_json(paths.stage4_closure, {"schema_version": "synthetic_closure"})
    expected = {
        "gold176": _sha(paths.gold176),
        "graph_features": _sha(paths.graph_features),
        "capability_profiles": _sha(paths.capability_profiles),
        "source_registry": _sha(paths.source_registry),
    }
    return paths, expected


def _fake_selection(**kwargs: object) -> dict:
    task = kwargs["task"]
    candidate_pool = list(kwargs["candidate_pool"])
    selected = candidate_pool[:4]
    request = policy.build_measurement_request(
        task=task,
        selected_rows=selected,
        round_index=int(kwargs["round_index"]),
    )
    result = {
        "schema_version": "fake_stage7_selection",
        "candidate_manifest": {"rows": candidate_pool},
        "acquisition": {
            "policy": "predicted_frontier_diversity",
            "selected_row_ids": [row["row_id"] for row in selected],
            "selected_rows": selected,
        },
        "measurement_request": request,
    }
    if kwargs["variant"] == "without_measured_feedback":
        result["a2_frozen"] = policy._a2_frozen_payload(
            {"bundle_config_sha256": "a" * 64},
            candidate_pool,
            list(kwargs["initial_rows"]),
            list(kwargs["initial_graph_features"]),
            {},
        )
    return result


class Stage7PrepareOnlineAblationV1Tests(unittest.TestCase):
    def test_self_consistent_manifest_and_pool_rewrite_cannot_replace_freeze_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            relocated = root / "relocated-gold.json"
            relocated.write_bytes(paths.gold176.read_bytes())
            frozen_path = output / "contracts/frozen_inputs.json"
            frozen = json.loads(frozen_path.read_text())
            frozen["inputs"]["gold176"]["path"] = str(relocated)
            frozen.pop("frozen_inputs_sha256")
            frozen["frozen_inputs_sha256"] = policy.canonical_sha256(frozen)
            _write_json(frozen_path, frozen)

            pools_path = output / "contracts/frozen_candidate_pools.json"
            pools = json.loads(pools_path.read_text())
            pools["variant_pools"]["full"].reverse()
            pools["variant_pool_sha256"]["full"] = policy.canonical_sha256(
                pools["variant_pools"]["full"]
            )
            pools.pop("frozen_candidate_pools_sha256")
            pools["frozen_candidate_pools_sha256"] = policy.canonical_sha256(
                pools
            )
            _write_json(pools_path, pools)

            with self.assertRaisesRegex(ValueError, "freeze root input manifest drift"):
                policy.validate_frozen_contracts(output)

    def test_pool_reorder_with_all_embedded_shas_recomputed_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            pools_path = output / "contracts/frozen_candidate_pools.json"
            pools = json.loads(pools_path.read_text())
            pools["variant_pools"]["full"].reverse()
            pools["variant_pool_sha256"]["full"] = policy.canonical_sha256(
                pools["variant_pools"]["full"]
            )
            pools.pop("frozen_candidate_pools_sha256")
            pools["frozen_candidate_pools_sha256"] = policy.canonical_sha256(
                pools
            )
            _write_json(pools_path, pools)

            with self.assertRaisesRegex(
                ValueError,
                "full ordered deterministic pool drift",
            ):
                policy.validate_frozen_contracts(output)

    def test_deterministic_contract_bytes_cannot_be_self_consistently_reformatted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            pre_scan = output / "contracts/pre_scan_candidate_registry.json"
            payload = json.loads(pre_scan.read_text())
            pre_scan.write_text(
                json.dumps(payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            pre_scan.with_suffix(".sha256").write_text(
                hashlib.sha256(pre_scan.read_bytes()).hexdigest() + "\n",
                encoding="ascii",
            )

            with self.assertRaisesRegex(
                ValueError,
                "pre-scan deterministic bytes drift",
            ):
                policy.validate_frozen_contracts(output)

    def test_cache_invariance_reruns_selector_on_adversarial_exact_membership(self) -> None:
        candidates = policy.prepare_frozen_candidate_pools(
            _registry(),
            raw_profile=_profile(),
            measured_row_ids=set(),
        )["variant_pools"]["full"]
        observed: list[list[dict]] = []

        def selector(rows: list[dict]) -> dict:
            observed.append([dict(row) for row in rows])
            clean = prepare.selection_candidate_view(rows)
            selected = clean[:4]
            return {
                "acquisition": {
                    "selected_row_ids": [row["row_id"] for row in selected],
                }
            }

        selection, audit = prepare._select_with_cache_invariance(
            candidates,
            selector,
        )

        self.assertEqual(len(observed), 2)
        self.assertTrue(
            all("cache_membership" not in row for row in observed[0])
        )
        self.assertEqual(
            {row["cache_membership"] for row in observed[1]},
            {True, False},
        )
        self.assertTrue(
            all("cached_labels" in row for row in observed[1])
        )
        self.assertEqual(audit["selector_invocation_count"], 2)
        self.assertEqual(
            selection["acquisition"]["selected_row_ids"],
            audit["empty_cache_selected_ids"],
        )
        self.assertEqual(
            audit["empty_cache_selected_ids"],
            audit["populated_cache_selected_ids"],
        )

    def test_frozen_contract_validator_rehashes_manifests_rows_and_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )

            audit = policy.validate_frozen_contracts(output)

            self.assertEqual(audit["pre_scan_count"], 8)
            self.assertEqual(audit["variant_pool_counts"]["full"], 7)
            pools_path = output / "contracts/frozen_candidate_pools.json"
            pools = json.loads(pools_path.read_text())
            pools["variant_pools"]["full"][0]["width"][0] += 8
            _write_json(pools_path, pools)
            with self.assertRaisesRegex(ValueError, "frozen candidate pools SHA drift"):
                policy.validate_frozen_contracts(output)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            paths.source_registry.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_registry file SHA drift"):
                prepare.initialize_trajectories(
                    output_root=output,
                    variant="without_surrogate",
                    seed=20260718,
                )

    def test_freeze_persists_scanner_rule_and_decisions_before_loading_gold_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root, malformed_gold=True)
            output = root / "stage7"

            with self.assertRaises(json.JSONDecodeError):
                prepare.freeze_contracts(
                    output_root=output,
                    inputs=paths,
                    expected_sha256=expected,
                    enforce_formal_counts=False,
                )

            self.assertTrue((output / "contracts/scanner_rule_manifest.json").is_file())
            self.assertTrue(
                (output / "contracts/scanner_decision_by_candidate.csv").is_file()
            )
            self.assertFalse((output / "contracts/raw_objective_reference.json").exists())

    def test_freeze_uses_exact_scanner_shape_and_gold_only_changes_selectability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"

            result = prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )

            rule = json.loads(
                (output / "contracts/scanner_rule_manifest.json").read_text()
            )
            pools = json.loads(
                (output / "contracts/frozen_candidate_pools.json").read_text()
            )
            reference = json.loads(
                (output / "contracts/raw_objective_reference.json").read_text()
            )
            self.assertEqual(
                rule["static_shape_contract"],
                {
                    "width_alignment": [8, 16, 32],
                    "width_min": [16, 32, 64],
                    "width_max": [64, 128, 256],
                },
            )
            self.assertEqual(pools["pre_scan_count"], 8)
            self.assertEqual(len(pools["variant_pools"]["without_capability_scan"]), 7)
            self.assertGreater(reference["values"]["latency_ms"], 2.0)
            self.assertGreater(reference["values"]["energy_j"], 0.2)
            self.assertGreater(reference["values"]["negative_ap70"], -0.7)
            self.assertEqual(result["status"], "frozen")

            repeated = prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            self.assertEqual(repeated, result)

            paths.source_registry.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                prepare.freeze_contracts(
                    output_root=output,
                    inputs=paths,
                    expected_sha256=expected,
                    enforce_formal_counts=False,
                )

    def test_gold_exclusion_count_only_includes_ids_in_frozen_candidate_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            gold = json.loads(paths.gold176.read_text())
            gold["rows"].append(
                {
                    **gold["rows"][0],
                    "row_id": (
                        "pyramid|64x128x256|q=fp16|profile="
                        "h800-tvm-auto-formal-v3"
                    ),
                    "manifest_job_id": (
                        "pyramid|64x128x256|q=fp16|profile="
                        "h800-tvm-auto-formal-v3"
                    ),
                    "group_id": "pyramid|64x128x256",
                }
            )
            _write_json(paths.gold176, gold)
            expected["gold176"] = _sha(paths.gold176)

            result = prepare.freeze_contracts(
                output_root=root / "stage7",
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )

            self.assertEqual(result["gold_excluded_count"], 1)
            pools = json.loads(
                (root / "stage7/contracts/frozen_candidate_pools.json").read_text()
            )
            self.assertEqual(
                pools["gold_measured_ids_outside_frozen_pool"],
                [
                    "pyramid|64x128x256|q=fp16|profile="
                    "h800-tvm-auto-formal-v3"
                ],
            )

    def test_scanner_audit_verifies_frozen_decision_sha_before_terminal_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            terminal = root / "terminal.json"
            terminal.write_text("{not-json", encoding="utf-8")
            (output / "contracts/scanner_rule_manifest.sha256").write_text(
                "0" * 64 + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "scanner rule SHA drift"):
                prepare.audit_scanner(
                    output_root=output,
                    terminal_evidence_json=terminal,
                )

    def test_initialize_generates_all_fifteen_round_zero_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            terminal = root / "terminal.json"
            _write_json(terminal, {"rows": []})
            prepare.audit_scanner(
                output_root=output,
                terminal_evidence_json=terminal,
            )

            with mock.patch.object(
                prepare.policy, "select_stage7_round", side_effect=_fake_selection
            ):
                result = prepare.initialize_trajectories(output_root=output)

            self.assertEqual(result["trajectory_count"], 15)
            for variant in (
                "full",
                "without_surrogate",
                "without_measured_feedback",
                "backend_blind",
                "without_capability_scan",
            ):
                for seed in (20260718, 20260719, 20260720):
                    round_dir = (
                        output
                        / "variants"
                        / variant
                        / f"seed_{seed}"
                        / "round_00"
                    )
                    self.assertTrue((round_dir / "measurement_request.json").is_file())
                    binding = json.loads(
                        (round_dir / "stage7_request_binding.json").read_text()
                    )
                    trajectory = json.loads(
                        (round_dir.parent / "trajectory_contract.json").read_text()
                    )
                    self.assertEqual(binding["seed"], seed)
                    self.assertEqual(binding["variant"], variant)
                    for field in (
                        "pre_scan_sha256",
                        "scan_pass_sha256",
                        "scanner_rule_sha256",
                        "scanner_decision_sha256",
                        "candidate_pool_sha256",
                    ):
                        self.assertEqual(len(trajectory[field]), 64)

    def test_initialize_all_fifteen_uses_real_five_variant_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, expected = _inputs(root)
            rows, graphs, profiles, closure = _training_context()
            _write_json(paths.gold176, {"rows": rows})
            _write_json(paths.graph_features, {"graph_features": graphs})
            _write_json(
                paths.capability_profiles,
                {"capability_profiles": profiles},
            )
            _write_json(paths.stage4_closure, closure)
            for name, path in (
                ("gold176", paths.gold176),
                ("graph_features", paths.graph_features),
                ("capability_profiles", paths.capability_profiles),
            ):
                expected[name] = _sha(path)
            output = root / "stage7"
            prepare.freeze_contracts(
                output_root=output,
                inputs=paths,
                expected_sha256=expected,
                enforce_formal_counts=False,
            )
            terminal = root / "terminal.json"
            _write_json(terminal, {"rows": []})
            prepare.audit_scanner(
                output_root=output,
                terminal_evidence_json=terminal,
            )

            result = prepare.initialize_trajectories(output_root=output)

            self.assertEqual(result["trajectory_count"], 15)
            expected_policy = {
                "full": "predicted_frontier_diversity",
                "without_surrogate": "uniform_random_without_replacement",
                "without_measured_feedback": "predicted_frontier_diversity",
                "backend_blind": "predicted_frontier_diversity",
                "without_capability_scan": "predicted_frontier_diversity",
            }
            for variant, policy_name in expected_policy.items():
                for seed in (20260718, 20260719, 20260720):
                    round_dir = (
                        output / "variants" / variant / f"seed_{seed}/round_00"
                    )
                    acquisition = json.loads(
                        (round_dir / "acquisition.json").read_text()
                    )
                    cache_audit = json.loads(
                        (
                            round_dir
                            / "cache_selection_invariance_audit.json"
                        ).read_text()
                    )
                    self.assertEqual(acquisition["policy"], policy_name)
                    self.assertEqual(cache_audit["selector_invocation_count"], 2)
                    self.assertEqual(cache_audit["verdict"], "pass")
            self.assertFalse(
                (
                    output
                    / "variants/without_surrogate/seed_20260718/round_00/"
                    "model_bundle_manifest.json"
                ).exists()
            )
            self.assertTrue(
                (
                    output
                    / "variants/without_measured_feedback/seed_20260718/"
                    "round_00/a2_frozen_contract.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    output
                    / "variants/backend_blind/seed_20260718/round_00/"
                    "backend_blind_audit.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
