#!/usr/bin/env python3
"""Finalize probe-conditioned capability profiles from S1-Q and S1-P structural runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from framework.stage2.probe_conditioned_profiles_v3 import build_profile_from_run_paths  # noqa: E402


def build_profiles(
    *,
    hardware_target: str,
    neutral_tvm_path: str | Path,
    pruning_tvm_path: str | Path,
    neutral_trt_path: str | Path,
    pruning_trt_path: str | Path,
    tvm_dispatch_key: str = "tvm",
    trt_dispatch_key: str = "trt",
) -> list[dict]:
    return [
        build_profile_from_run_paths(
            capability_profile_id=f"{hardware_target}-tvm-probe-conditioned-v3",
            hardware_target=hardware_target,
            dispatch_key=tvm_dispatch_key,
            neutral_run_path=neutral_tvm_path,
            pruning_run_path=pruning_tvm_path,
        ),
        build_profile_from_run_paths(
            capability_profile_id=f"{hardware_target}-trt-probe-conditioned-v3",
            hardware_target=hardware_target,
            dispatch_key=trt_dispatch_key,
            neutral_run_path=neutral_trt_path,
            pruning_run_path=pruning_trt_path,
        ),
    ]


def write_profiles(profiles: list[dict], out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-target", required=True)
    parser.add_argument("--neutral-tvm", type=Path, required=True)
    parser.add_argument("--pruning-tvm", type=Path, required=True)
    parser.add_argument("--neutral-trt", type=Path, required=True)
    parser.add_argument("--pruning-trt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tvm-dispatch-key", default="tvm")
    parser.add_argument("--trt-dispatch-key", default="trt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    profiles = build_profiles(
        hardware_target=args.hardware_target,
        neutral_tvm_path=args.neutral_tvm,
        pruning_tvm_path=args.pruning_tvm,
        neutral_trt_path=args.neutral_trt,
        pruning_trt_path=args.pruning_trt,
        tvm_dispatch_key=args.tvm_dispatch_key,
        trt_dispatch_key=args.trt_dispatch_key,
    )
    out_path = write_profiles(profiles, args.out)
    print(
        json.dumps(
            {
                "profile_count": len(profiles),
                "out": str(out_path),
                "profile_ids": [profile["capability_profile_id"] for profile in profiles],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
