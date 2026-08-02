from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.stage5_apply_single_config_repair_v1 import apply_repair


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class Stage5ApplySingleConfigRepairV1Tests(unittest.TestCase):
    def test_replaces_only_selected_configuration_and_preserves_backups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task, repair = root / "task", root / "repair"
            for repeat in range(3):
                target = task / f"repeat_{repeat}"
                source = repair / f"repeat_{repeat}"
                old = {"manifest_job_id": "target", "job_id": "job", "value": "old"}
                keep = {"manifest_job_id": "keep", "job_id": "keep-job"}
                new = {**old, "value": "new"}
                _jsonl(target / "performance_jobs.jsonl", [old, keep])
                _jsonl(target / "performance_state.jsonl", [{"job_id": "job", "status": "success"}, {"job_id": "keep-job", "status": "success"}])
                (target / "performance_manifest.json").write_text(json.dumps({"jobs": [old, keep]}))
                _jsonl(source / "performance_jobs.jsonl", [new])
                _jsonl(source / "performance_state.jsonl", [{"job_id": "job", "status": "success", "value": "new"}])
                (source / "performance_manifest.json").write_text(json.dumps({"jobs": [new]}))
            _jsonl(task / "ap/ap_state.jsonl", [{"job_id": "target", "status": "failed"}, {"job_id": "keep"}])
            _jsonl(repair / "ap/ap_state.jsonl", [{"job_id": "target", "record_type": "job_terminal", "stage": "full", "status": "success"}])

            audit = apply_repair(task, repair, "target")

            jobs = [json.loads(line) for line in (task / "repeat_0/performance_jobs.jsonl").read_text().splitlines()]
            ap = [json.loads(line) for line in (task / "ap/ap_state.jsonl").read_text().splitlines()]
            self.assertEqual(jobs[0]["value"], "new")
            self.assertEqual(jobs[1]["manifest_job_id"], "keep")
            self.assertEqual(len([row for row in ap if row.get("job_id") == "target"]), 1)
            self.assertEqual(audit["performance_repeat_count"], 3)
            self.assertTrue((task / "ap/ap_state.jsonl.pre_quant_contract_repair_v1").is_file())


if __name__ == "__main__":
    unittest.main()
