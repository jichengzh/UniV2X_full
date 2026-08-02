from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from framework.stage5 import source_registry_v1 as registry


def _graphs() -> list[dict]:
    rows = []
    for model in ("pyramid", "codriving"):
        for width in ([16, 32, 64], [24, 48, 96], [32, 64, 128]):
            group_id = f"{model}|{'x'.join(map(str, width))}"
            rows.append(
                {
                    "group_id": group_id,
                    "model": model,
                    "width": width,
                    "conv_count": 27 if model == "pyramid" else 24,
                    "conv_macs": float(width[0] * width[1] * width[2]),
                    "group_conv_count": 3 if model == "pyramid" else 0,
                }
            )
    return rows


class Stage5SourceRegistryV1Tests(unittest.TestCase):
    def test_full_registry_has_343_widths_per_model_independent_of_existing_checkpoints(self) -> None:
        widths = registry.canonical_widths()
        result = registry.build_source_registry(
            graph_features=_graphs(),
            pyramid_inventory=[],
            pyramid_widths=widths,
            pyramid_model_root="/remote/pyramid",
            pyramid_base_source={
                "checkpoint_path": "/remote/base/net_epoch_bestval_at23.pth",
                "checkpoint_sha256": "b" * 64,
                "checkpoint_dir": "/remote/base",
            },
            codriving_widths=widths,
            remote_result_root="/remote/stage5",
            codriving_model_root="/remote/codriving",
            seed=3,
        )

        self.assertEqual(result["group_count"], 686)
        for model in ("pyramid", "codriving"):
            model_groups = [group for group in result["groups"] if group["model"] == model]
            self.assertEqual(len(model_groups), 343)
            self.assertEqual({tuple(group["width"]) for group in model_groups}, {tuple(w) for w in widths})
        unseen = next(group for group in result["groups"] if group["group_id"] == "pyramid|16x32x64")
        self.assertEqual(unseen["materialization_kind"], "pyramid_prepare_train_export")
        self.assertEqual(unseen["source_status"], "materializable")
        self.assertEqual(unseen["source_contract"]["base_checkpoint_sha256"], "b" * 64)
        self.assertTrue(unseen["source_contract"]["checkpoint_path"].endswith("/16x32x64/stage5_best.pth"))

        audit = registry.validate_full_source_registry(result)
        self.assertEqual(audit["group_count_by_model"], {"codriving": 343, "pyramid": 343})
        self.assertEqual(audit["genome_count_by_model"], {"codriving": 686, "pyramid": 686})

    def test_pyramid_inventory_binds_each_width_to_real_checkpoint_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "Pyramid_DAIR_m1_o60_frontier_01_fresh_v1"
            directory.mkdir()
            checkpoint = directory / "net_epoch_bestval_at1.pth"
            checkpoint.write_bytes(b"real-checkpoint")
            (directory / "config.yaml").write_text("model: pyramid\n", encoding="utf-8")

            inventory = registry.build_pyramid_checkpoint_inventory(
                [{"label": "frontier_01", "width": [40, 80, 160]}],
                checkpoint_root=root,
            )

        self.assertEqual(len(inventory["sources"]), 1)
        self.assertEqual(inventory["missing_labels"], [])
        self.assertEqual(
            inventory["sources"][0]["checkpoint_sha256"],
            hashlib.sha256(b"real-checkpoint").hexdigest(),
        )

    def test_graph_surrogate_rejects_label_like_features(self) -> None:
        graphs = _graphs()
        graphs[0]["latency_ms"] = 1.0

        with self.assertRaisesRegex(ValueError, "label-like"):
            registry.fit_graph_feature_surrogate(graphs, seed=1)

    def test_registry_builds_pyramid_checkpoint_and_codriving_materialization_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "net_epoch_bestval_at1.pth"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            pyramid_inventory = [
                {
                    "label": "frontier_01",
                    "width": [40, 80, 160],
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha,
                }
            ]
            result = registry.build_source_registry(
                graph_features=_graphs(),
                pyramid_inventory=pyramid_inventory,
                codriving_widths=[[40, 80, 160], [48, 96, 192]],
                remote_result_root="/remote/stage5",
                codriving_model_root="/remote/codriving",
                seed=3,
            )

        self.assertEqual(result["schema_version"], "stage5_candidate_source_registry_v1")
        self.assertEqual(result["group_count"], 3)
        pyramid = next(group for group in result["groups"] if group["model"] == "pyramid")
        codriving = [group for group in result["groups"] if group["model"] == "codriving"]
        self.assertEqual(pyramid["source_contract"]["checkpoint_path"], str(checkpoint))
        self.assertEqual(pyramid["source_status"], "materializable")
        self.assertTrue(all(group["source_status"] == "materializable" for group in codriving))
        self.assertTrue(all(group["materialization_kind"] == "codriving_prepare_train_export" for group in codriving))
        self.assertTrue(all("latency_ms" not in group["graph_features"] for group in result["groups"]))
        self.assertTrue(all(len(group["source_evidence_sha256"]) == 64 for group in result["groups"]))

    def test_default_width_space_is_343_unique_widths(self) -> None:
        widths = registry.canonical_widths()

        self.assertEqual(len(widths), 343)
        self.assertEqual(len({tuple(width) for width in widths}), 343)
        self.assertIn([16, 32, 64], widths)
        self.assertIn([64, 128, 256], widths)

    def test_ready_status_requires_sha_bound_materialization_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pth"
            checkpoint.write_bytes(b"checkpoint")
            inventory = [{
                "label": "frontier_01",
                "width": [40, 80, 160],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }]
            first = registry.build_source_registry(
                graph_features=_graphs(),
                pyramid_inventory=inventory,
                codriving_widths=[],
                remote_result_root=root / "stage5",
                codriving_model_root=root / "codriving",
                seed=3,
            )
            group = first["groups"][0]
            contract = group["source_contract"]
            for key in ("onnx_path", "calibration_npz", "calibration_summary"):
                path = Path(contract[key])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(key.encode())
            trt_dir = Path(contract["trt_calibration_dir"])
            trt_dir.mkdir(parents=True, exist_ok=True)
            (trt_dir / "sample.npy").write_bytes(b"sample")
            marker = Path(contract["source_done_marker"])
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
            evidence_path = Path(contract["source_done_marker"].replace(".done", "_evidence.json"))
            evidence = {
                "schema_version": "stage5_source_materialization_evidence_v1",
                "group_id": group["group_id"],
                "source_plan_sha256": group["source_evidence_sha256"],
                "status": "ready",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "onnx_path": contract["onnx_path"],
                "onnx_sha256": hashlib.sha256(Path(contract["onnx_path"]).read_bytes()).hexdigest(),
                "calibration_path": contract["calibration_npz"],
                "calibration_sha256": hashlib.sha256(Path(contract["calibration_npz"]).read_bytes()).hexdigest(),
                "calibration_summary_path": contract["calibration_summary"],
                "calibration_summary_sha256": hashlib.sha256(Path(contract["calibration_summary"]).read_bytes()).hexdigest(),
            }
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            second = registry.build_source_registry(
                graph_features=_graphs(),
                pyramid_inventory=inventory,
                codriving_widths=[],
                remote_result_root=root / "stage5",
                codriving_model_root=root / "codriving",
                seed=3,
            )

        ready = second["groups"][0]
        self.assertEqual(ready["source_status"], "ready")
        self.assertEqual(ready["source_evidence_kind"], "materialization_evidence")
        self.assertEqual(len(ready["materialization_evidence_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
