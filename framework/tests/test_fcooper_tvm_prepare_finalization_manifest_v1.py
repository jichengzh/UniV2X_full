import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "fcooper_tvm_prepare_finalization_manifest_v1.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "fcooper_tvm_prepare_finalization_manifest_v1",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FCooperTvmPrepareFinalizationManifestTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def artifact(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name}) + "\n")
        return path

    def test_builds_all_sha_bound_finalizer_inputs(self):
        pools = {
            name: self.artifact(f"pools/{name}.json")
            for name in (
                "compression_only",
                "schedule_only",
                "compress_then_tune_screen",
                "compress_then_tune_tuned",
                "gear",
            )
        }
        validations = {
            name: self.artifact(f"validations/{name}.json")
            for name in (
                "original_default",
                "compression_only",
                "schedule_only",
                "compress_then_tune",
                "gear",
            )
        }
        pyramid = self.artifact("stage6_pyramid_formal/reference.json")
        codriving = self.artifact("stage6_codriving_formal/reference.json")
        original = self.artifact("original/contract.json")
        resource = self.artifact("audits/resource.json")
        gear_round_audits = [
            self.artifact(f"gear/round_{index:02d}/atomic_batch_audit.json")
            for index in range(4)
        ]
        gear_round_states = [
            self.artifact(f"gear/round_{index:02d}/round_state.json")
            for index in range(1, 4)
        ]
        precondition_audits = {
            name: self.artifact(f"preconditions/{name}.json")
            for name in (
                "capability_probe",
                "probe_isolation",
                "recovery_numeric_gate",
                "control_provenance",
            )
        }
        search_initialization = {
            name: self.artifact(f"search/{name}.json")
            for name in (
                "task_contract",
                "initialization_summary",
                "capability_profile",
            )
        }

        manifest = self.module.build_manifest(
            evidence_root=self.root,
            ap70_ref=0.633,
            original_contract=original,
            pools=pools,
            winner_validations=validations,
            pyramid_reference=pyramid,
            codriving_reference=codriving,
            resource_audit=resource,
            gear_round_audits=gear_round_audits,
            gear_round_states=gear_round_states,
            precondition_audits=precondition_audits,
            search_initialization=search_initialization,
        )

        self.assertEqual(
            manifest["schema_version"],
            "fcooper_tvm_stage6_finalize_manifest_v1",
        )
        self.assertEqual(set(manifest["pools"]), set(pools))
        self.assertEqual(set(manifest["winner_validations"]), set(validations))
        self.assertEqual(
            manifest["resource_audit"]["sha256"],
            hashlib.sha256(resource.read_bytes()).hexdigest(),
        )
        self.assertEqual(len(manifest["gear_round_audits"]), 4)
        self.assertEqual(
            manifest["gear_round_audits"][2]["sha256"],
            hashlib.sha256(gear_round_audits[2].read_bytes()).hexdigest(),
        )
        self.assertEqual(len(manifest["gear_round_states"]), 3)
        self.assertEqual(
            set(manifest["precondition_audits"]),
            set(precondition_audits),
        )
        self.assertEqual(
            set(manifest["search_initialization"]),
            set(search_initialization),
        )
        self.assertEqual(
            manifest["search_initialization"]["initialization_summary"]["sha256"],
            hashlib.sha256(
                search_initialization["initialization_summary"].read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            manifest["reference_artifacts"]["pyramid_tvm"]["model"],
            "pyramid",
        )
        binding = manifest["pools"]["gear"]
        self.assertEqual(
            binding["sha256"],
            hashlib.sha256(pools["gear"].read_bytes()).hexdigest(),
        )

    def test_rejects_reference_path_without_model_identity(self):
        artifact = self.artifact("references/unknown.json")
        with self.assertRaisesRegex(ValueError, "model identity"):
            self.module.reference_binding(artifact, model="pyramid")


if __name__ == "__main__":
    unittest.main()
