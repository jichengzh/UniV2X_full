#!/usr/bin/env python3
"""Run one AP evaluation command and append a Stage2 AP anchor row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    LutProductizationError,
    ap_anchor_row,
    append_jsonl,
    parse_width_csv,
    run_json_command,
    utc_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--software-point-id", required=True)
    parser.add_argument("--dense-stage", required=True)
    parser.add_argument("--width", required=True)
    parser.add_argument("--quant-policy", required=True)
    parser.add_argument("--schedule-policy", default="not_applicable")
    parser.add_argument("--optimized-scope", default="rsu_dense_core")
    parser.add_argument("--backend", default="model_eval")
    parser.add_argument("--manifest-digest", default="unknown")
    parser.add_argument("--run-id")
    parser.add_argument("--created-at")
    parser.add_argument("--eval-command-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    return parser.parse_args()


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    return None if value is None else int(value)


def main() -> int:
    args = parse_args()
    payload = run_json_command(args.eval_command_json)
    if payload.get("metric_value") is None:
        raise LutProductizationError("AP eval command must emit metric_value")

    run_id = args.run_id or str(
        payload.get("run_id") or args.job_id or f"ap_{utc_timestamp()}"
    )
    row = ap_anchor_row(
        config_id=args.config_id,
        model=args.model,
        manifest_digest=args.manifest_digest,
        candidate_id=args.candidate_id,
        software_point_id=args.software_point_id,
        dense_stage=args.dense_stage,
        optimized_scope=args.optimized_scope,
        width=parse_width_csv(args.width),
        quant_policy=args.quant_policy,
        schedule_policy=args.schedule_policy,
        backend=args.backend,
        measurement_status="measured",
        metric=payload.get("metric", "AP70"),
        metric_value=float(payload["metric_value"]),
        secondary_metrics=dict(payload.get("secondary_metrics", {})),
        dataset=payload.get("dataset", "unknown"),
        eval_split=payload.get("eval_split", "unknown"),
        num_samples=_optional_int(payload, "num_samples"),
        ckpt_path=payload.get("ckpt_path", "unknown"),
        ckpt_digest=payload.get("ckpt_digest", "unknown"),
        finetune_protocol=payload.get("finetune_protocol", "none"),
        training_budget=payload.get("training_budget", "none"),
        eval_command=payload.get("eval_command", ""),
        provenance=payload.get("provenance", "DAIR validation AP anchor"),
        run_id=run_id,
        created_at=args.created_at or str(payload.get("created_at") or utc_timestamp()),
        source_files=list(payload.get("source_files", [])),
        raw_artifact=payload.get("raw_artifact"),
        notes=payload.get("notes", "AP anchor; schedule-independent"),
    )
    append_jsonl(args.out_jsonl, row)
    print(json.dumps({"schema": "lut_generation_result_v1", "row_id": row["row_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
