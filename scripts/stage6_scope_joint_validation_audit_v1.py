#!/usr/bin/env python3
"""Create a Pyramid-only passed view of the broader Stage5 validation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage6.independent_validation_v1 import (  # noqa: E402
    scope_passed_validation_tasks,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.source_audit.read_text(encoding="utf-8"))
    scoped = scope_passed_validation_tasks(source, task_ids=args.task_id)
    payload = {
        **scoped,
        "source_audit_path": str(args.source_audit),
        "source_audit_sha256": _sha(args.source_audit),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "all_tasks_passed": True,
                "task_count": payload["task_count"],
                "configuration_count": payload["configuration_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
