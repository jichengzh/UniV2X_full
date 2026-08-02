import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from stage7_repair_ap_checkpoint_alias_v1 import (
    atomic_json,
    file_sha256,
    repair_alias,
    source_paths,
)


class RepairApCheckpointAliasTests(unittest.TestCase):
    def test_source_paths_rejects_escape_and_builds_canonical_group_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint_dir, evidence = source_paths(root, "pyramid|64x128x224")
            self.assertEqual(
                checkpoint_dir,
                root / "sources/pyramid/064x128x224/checkpoint",
            )
            self.assertEqual(
                evidence,
                root / "sources/pyramid/064x128x224/source_ready_evidence.json",
            )
            for invalid in ("pyramid|../../tmp", "codriving|64x128x224", "pyramid|0x1x2"):
                with self.assertRaisesRegex(ValueError, "group_id"):
                    source_paths(root, invalid)

    def test_repair_adds_non_bestval_alias_without_changing_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint_dir, evidence_path = source_paths(
                root, "pyramid|64x128x224"
            )
            checkpoint_dir.mkdir(parents=True)
            primary = checkpoint_dir / "stage5_best.pth"
            imported_name = checkpoint_dir / "net_epoch_bestval_at1.pth"
            primary.write_bytes(b"frozen-checkpoint")
            os.link(primary, imported_name)
            digest = file_sha256(primary)
            atomic_json(
                evidence_path,
                {
                    "status": "ready",
                    "group_id": "pyramid|64x128x224",
                    "checkpoint_path": str(primary),
                    "checkpoint_sha256": digest,
                },
            )
            audit_path = root / "audits/repair.json"

            result = repair_alias(
                v2_root=root,
                group_id="pyramid|64x128x224",
                expected_checkpoint_sha256=digest,
                audit_path=audit_path,
            )
            alias = checkpoint_dir / "net_epoch1.pth"

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["selected_event_budget_delta"], 0)
            self.assertFalse(result["scientific_contract_changed"])
            self.assertEqual(file_sha256(alias), digest)
            self.assertEqual(os.stat(primary).st_ino, os.stat(alias).st_ino)
            self.assertEqual(
                hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                result["source_evidence_file_sha256"],
            )
            self.assertEqual(json.loads(audit_path.read_text()), result)
            self.assertEqual(
                repair_alias(
                    v2_root=root,
                    group_id="pyramid|64x128x224",
                    expected_checkpoint_sha256=digest,
                    audit_path=audit_path,
                )["alias_sha256"],
                digest,
            )

    def test_repair_fails_closed_on_evidence_or_epoch_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint_dir, evidence_path = source_paths(
                root, "pyramid|32x80x128"
            )
            checkpoint_dir.mkdir(parents=True)
            primary = checkpoint_dir / "stage5_best.pth"
            primary.write_bytes(b"checkpoint")
            digest = file_sha256(primary)
            atomic_json(
                evidence_path,
                {
                    "status": "ready",
                    "group_id": "pyramid|32x80x128",
                    "checkpoint_path": str(primary),
                    "checkpoint_sha256": digest,
                },
            )
            with self.assertRaisesRegex(ValueError, "one matching bestval"):
                repair_alias(
                    v2_root=root,
                    group_id="pyramid|32x80x128",
                    expected_checkpoint_sha256=digest,
                    audit_path=root / "audit.json",
                )
            for epoch in (1, 2):
                os.link(primary, checkpoint_dir / f"net_epoch_bestval_at{epoch}.pth")
            with self.assertRaisesRegex(ValueError, "one matching bestval"):
                repair_alias(
                    v2_root=root,
                    group_id="pyramid|32x80x128",
                    expected_checkpoint_sha256=digest,
                    audit_path=root / "audit.json",
                )


if __name__ == "__main__":
    unittest.main()
