from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/stage5_full_budget_scheduler_v3.sh"


class Stage5FullBudgetSchedulerV3Tests(unittest.TestCase):
    def test_plan_after_active_first_batch_contains_exact_remaining_budget(self) -> None:
        completed = subprocess.run(
            ["bash", str(SCRIPT), "--plan-only"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        jobs = [line for line in completed.stdout.splitlines() if line.startswith("RUN ")]
        self.assertEqual(len(jobs), 15)
        self.assertEqual(jobs[0], "RUN S5-PYR-TVM round=1")
        self.assertEqual(jobs[-1], "RUN S5-COD-TRT round=3")

    def test_retries_same_request_after_controller_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary) / "repo"
            root = Path(temporary) / "formal"
            scripts = repo / "scripts"
            scripts.mkdir(parents=True)
            fake_controller = scripts / "stage5_task_round_controller_v3.sh"
            fake_controller.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} == --validate-only ]] && exit 0
mkdir -p "$ROOT/controller" "$ROOT/$TASK"
tag="${TASK}_round$(printf '%02d' "$ROUND_INDEX")"
attempt_file="$ROOT/controller/${tag}.attempts"
attempt=$(( $(cat "$attempt_file" 2>/dev/null || echo 0) + 1 ))
printf '%s\n' "$attempt" >"$attempt_file"
if [[ "$TASK" == S5-PYR-TVM && "$ROUND_INDEX" == 1 && "$attempt" == 1 ]]; then
  printf 'infrastructure failure\n' >"$ROOT/controller/${tag}.failed"
  exit 1
fi
rm -f "$ROOT/controller/${tag}.failed"
date -Is >"$ROOT/controller/${tag}.done"
if [[ "$ROUND_INDEX" == 3 ]]; then
  cat >"$ROOT/$TASK/task_budget_terminal.json" <<JSON
{"status":"budget_exhausted","budget_consumed":16,"round_count":4}
JSON
fi
""",
                encoding="utf-8",
            )
            fake_controller.chmod(0o755)
            request = root / "S5-PYR-TVM/round_01/measurement_request.json"
            request.parent.mkdir(parents=True)
            request.write_text('{"frozen":true}\n', encoding="utf-8")
            completed = subprocess.run(
                ["bash", str(SCRIPT)],
                cwd=REPO,
                env={
                    **os.environ,
                    "REPO": str(repo),
                    "ROOT": str(root),
                    "GPU_POOL": "0,1,2,3",
                    "MAX_INFRA_ATTEMPTS": "2",
                    "RETRY_DELAY_S": "0",
                },
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            attempts = root / "controller/S5-PYR-TVM_round01.attempts"
            self.assertEqual(attempts.read_text(encoding="utf-8").strip(), "2")
            terminal = root / "controller/full_budget_scheduler_terminal.json"
            self.assertTrue(terminal.is_file())
            retry_audit = root / "controller/S5-PYR-TVM_round01.retry_audit.jsonl"
            self.assertEqual(len(retry_audit.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
