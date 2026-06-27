#!/usr/bin/env python3
"""Validate and archive a Stage2 evidence delta for Stage1 refresh."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.contracts import (
    STAGE2_EVIDENCE_DELTA_SCHEMA,
    Stage2EvidenceDelta,
    Stage2EvidenceRecord,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", required=True, help="Stage2 evidence delta JSON.")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "results/stage1_model_predict/stage2_evidence_delta"),
        help="Archive directory for validated Stage2 evidence deltas.",
    )
    return parser.parse_args()


def _load_and_validate(path: Path) -> Stage2EvidenceDelta:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != STAGE2_EVIDENCE_DELTA_SCHEMA:
        raise ValueError(f"unexpected schema: {data.get('schema')}")
    records = [
        Stage2EvidenceRecord(
            backend=record["backend"],
            hardware=record["hardware"],
            scope=record["scope"],
            evidence_kind=record["evidence_kind"],
            provenance=record["provenance"],
            candidate_config=record["candidate_config"],
            metric=record["metric"],
        )
        for record in data.get("records", [])
    ]
    return Stage2EvidenceDelta(model=str(data.get("model", "unknown")), records=records)


def main() -> int:
    args = _parse_args()
    delta_path = Path(args.delta)
    delta = _load_and_validate(delta_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{delta.model}_stage2_evidence_delta_v1.json"
    shutil.copyfile(delta_path, out_path)
    print(
        "stage2_update_evidence_ok "
        f"model={delta.model} "
        f"records={len(delta.records)} "
        f"out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
