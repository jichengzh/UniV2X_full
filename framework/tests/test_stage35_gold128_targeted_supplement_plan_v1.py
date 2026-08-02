from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "stage35_gold128_targeted_supplement_plan_v1.py"
CANDIDATES = ROOT / "results/stage35_gold128_targeted_supplement_v1_20260714/candidate_plan.json"
SOURCE_AUDIT = ROOT / "results/stage35_gold128_targeted_supplement_v1_20260714/source_prep/source_preparation_audit.json"
BASE_MANIFEST = ROOT / "results/stage35_gold128_v2_20260714/gold128_manifest.json"
BASE_GOLD = ROOT / "results/stage35_gold128_v2_20260714/gold128_final.json"
CAPABILITIES = ROOT / "results/gold_coldstart96_v3_final_20260711/" / (
    "gold_coldstart96_manifest_v3-d1b0495125d60962c135c3c682ccfeb106fe20babfdc408dc0215ad7518d1305.json"
)
REMOTE_RESULT_ROOT = "/home/jichengzhi/V2X/results/stage35_gold128_targeted_supplement_v1_20260714"


def _load_module():
    spec = importlib.util.spec_from_file_location("targeted_plan", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage35Gold128TargetedSupplementPlanV1Tests(unittest.TestCase):
    def test_builds_four_train_groups_with_complete_four_arm_product(self) -> None:
        module = _load_module()
        candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))["capability_profiles"]

        result = module.build_targeted_plan(
            candidates,
            capabilities,
            source_prep_audit=source_audit,
            source_prep_audit_sha256=_sha256(SOURCE_AUDIT),
            base_gold_manifest=base_manifest,
            base_gold_sha256=_sha256(BASE_GOLD),
            remote_result_root=REMOTE_RESULT_ROOT,
            gpus=[6, 7],
        )

        self.assertEqual(len(result["manifest"]["jobs"]), 16)
        self.assertEqual(len(result["performance_jobs"]), 16)
        grouped: dict[str, list[dict]] = {}
        for row in result["manifest"]["jobs"]:
            grouped.setdefault(row["group_id"], []).append(row)
        self.assertEqual(len(grouped), 4)
        expected = {
            ("tvm_auto", "fp16"), ("tvm_auto", "int8"),
            ("trt_engine", "fp16"), ("trt_engine", "int8"),
        }
        self.assertTrue(all({(r["dispatch_key"], r["q_mode"]) for r in rows} == expected for rows in grouped.values()))
        self.assertTrue(all(row["split"] == "train" for row in result["manifest"]["jobs"]))
        self.assertTrue(all(row["source_status"] == "checkpoint_consistent_source_prepared" for row in result["manifest"]["jobs"]))
        self.assertTrue(all(row["source_prep_evidence"]["status"] == "prepared" for row in result["manifest"]["jobs"]))
        self.assertEqual({row["assigned_gpu"] for row in result["performance_jobs"]}, {6, 7})
        self.assertEqual(
            result["manifest"]["source_prep_audit_sha256"], _sha256(SOURCE_AUDIT)
        )

    def test_rejects_candidate_that_overlaps_locked_holdout(self) -> None:
        module = _load_module()
        candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))["capability_profiles"]
        locked = next(row for row in base_manifest["jobs"] if row["split"] == "locked_holdout")
        candidates = {**candidates, "groups": [dict(row) for row in candidates["groups"]]}
        candidates["groups"][0].update({
            "group_id": locked["group_id"],
            "width": locked["width"],
        })
        with self.assertRaisesRegex(ValueError, "overlap locked holdout"):
            module.build_targeted_plan(
                candidates,
                capabilities,
                source_prep_audit=source_audit,
                source_prep_audit_sha256=_sha256(SOURCE_AUDIT),
                base_gold_manifest=base_manifest,
                base_gold_sha256=_sha256(BASE_GOLD),
                remote_result_root=REMOTE_RESULT_ROOT,
                gpus=[6, 7],
            )

    def test_rejects_source_audit_path_that_does_not_match_generated_contract(self) -> None:
        module = _load_module()
        candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))["capability_profiles"]
        source_audit["groups"][0]["files"]["onnx_path"]["path"] += ".stale"

        with self.assertRaisesRegex(ValueError, "path"):
            module.build_targeted_plan(
                candidates,
                capabilities,
                source_prep_audit=source_audit,
                source_prep_audit_sha256="a" * 64,
                base_gold_manifest=base_manifest,
                base_gold_sha256=_sha256(BASE_GOLD),
                remote_result_root=REMOTE_RESULT_ROOT,
                gpus=[6, 7],
            )

    def test_rejects_non_hex_source_audit_sha256(self) -> None:
        module = _load_module()
        candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))["capability_profiles"]
        source_audit["groups"][0]["files"]["onnx_path"]["sha256"] = "z" * 64

        with self.assertRaisesRegex(ValueError, "SHA256|evidence"):
            module.build_targeted_plan(
                candidates,
                capabilities,
                source_prep_audit=source_audit,
                source_prep_audit_sha256="a" * 64,
                base_gold_manifest=base_manifest,
                base_gold_sha256=_sha256(BASE_GOLD),
                remote_result_root=REMOTE_RESULT_ROOT,
                gpus=[6, 7],
            )

    def test_accepts_exact_remote_audit_paths_without_local_files(self) -> None:
        module = _load_module()
        candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))["capability_profiles"]
        remote_root = "/remote-only/stage35-targeted"
        remote_candidates = copy.deepcopy(candidates)
        remote_audit = copy.deepcopy(source_audit)
        for candidate, audit_row in zip(remote_candidates["groups"], remote_audit["groups"]):
            old_checkpoint = candidate["checkpoint_path"]
            candidate["checkpoint_dir"] = f"/remote-only/checkpoints/{candidate['group_id']}"
            candidate["checkpoint_path"] = f"{candidate['checkpoint_dir']}/best.pth"
            audit_row["files"]["checkpoint_path"]["path"] = candidate["checkpoint_path"]
            for item in audit_row["files"].values():
                if item["path"] != candidate["checkpoint_path"]:
                    item["path"] = item["path"].replace(REMOTE_RESULT_ROOT, remote_root)
            self.assertNotEqual(old_checkpoint, candidate["checkpoint_path"])

        result = module.build_targeted_plan(
            remote_candidates,
            capabilities,
            source_prep_audit=remote_audit,
            source_prep_audit_sha256="a" * 64,
            base_gold_manifest=base_manifest,
            base_gold_sha256=_sha256(BASE_GOLD),
            remote_result_root=remote_root,
            gpus=[6, 7],
        )

        self.assertEqual(len(result["manifest"]["jobs"]), 16)

    def test_rejects_candidate_bound_to_different_base_gold_file(self) -> None:
        module = _load_module()
        candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))
        source_audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf-8"))
        base_manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
        capabilities = json.loads(CAPABILITIES.read_text(encoding="utf-8"))["capability_profiles"]

        with self.assertRaisesRegex(ValueError, "Gold128.*SHA256|source_gold128_sha256"):
            module.build_targeted_plan(
                candidates,
                capabilities,
                source_prep_audit=source_audit,
                source_prep_audit_sha256=_sha256(SOURCE_AUDIT),
                base_gold_manifest=base_manifest,
                base_gold_sha256="d" * 64,
                remote_result_root=REMOTE_RESULT_ROOT,
                gpus=[6, 7],
            )


if __name__ == "__main__":
    unittest.main()
