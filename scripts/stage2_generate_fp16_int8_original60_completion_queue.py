#!/usr/bin/env python3
"""Generate FP16/INT8 original60 completion queue and review artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import utc_timestamp  # noqa: E402
from framework.stage2.original60_quant_completion import (  # noqa: E402
    build_completion_jobs,
    summarize_completion_jobs,
    write_completion_outputs,
)


DEFAULT_OUTPUT_ROOT = (
    ROOT / "multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--latency-rows",
        default=None,
        help="Canonical latency JSONL. Defaults to <output-root>/rows/latency_original60_quant_rows_v1.jsonl.",
    )
    parser.add_argument(
        "--energy-rows",
        default=None,
        help="Canonical energy JSONL. Defaults to <output-root>/rows/energy_original60_quant_rows_v1.jsonl.",
    )
    parser.add_argument(
        "--ap-rows",
        default=None,
        help="Canonical AP JSONL. Defaults to <output-root>/rows/ap_original60_quant_rows_v1.jsonl.",
    )
    parser.add_argument("--created-at", default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    rows_dir = output_root / "rows"
    latency_path = Path(args.latency_rows) if args.latency_rows else rows_dir / "latency_original60_quant_rows_v1.jsonl"
    energy_path = Path(args.energy_rows) if args.energy_rows else rows_dir / "energy_original60_quant_rows_v1.jsonl"
    ap_path = Path(args.ap_rows) if args.ap_rows else rows_dir / "ap_original60_quant_rows_v1.jsonl"
    created_at = args.created_at or utc_timestamp()

    jobs = build_completion_jobs(
        latency_rows=read_jsonl(latency_path),
        energy_rows=read_jsonl(energy_path),
        ap_rows=read_jsonl(ap_path),
        created_at=created_at,
    )
    written = write_completion_outputs(
        output_root=output_root,
        jobs=jobs,
        created_at=created_at,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "summary": summarize_completion_jobs(jobs),
                "outputs": {key: str(value) for key, value in written.items()},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
