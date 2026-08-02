#!/usr/bin/env python3
"""Build the Stage3 pilot readiness staging contract from a v3 gold manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "framework" / "stage3" / "gold_pilot_readiness_v3.py"
SPEC = importlib.util.spec_from_file_location("stage3_gold_pilot_readiness_v3", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    readiness = MODULE.build_gold_pilot_readiness(manifest, manifest_path=args.manifest_json)
    readiness_path = MODULE.write_gold_pilot_readiness(readiness, args.output_dir)
    print(
        json.dumps(
            {
                "readiness_json": str(readiness_path),
                "status": readiness["status"],
                "pilot_row_count": readiness["pilot_row_count"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
