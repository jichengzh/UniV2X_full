#!/usr/bin/env python3
"""Import existing AP anchors into canonical Stage2 AP rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.lut_productization import (  # noqa: E402
    ap_anchor_row,
    stable_config_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--metric", default="AP70")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-split", required=True)
    parser.add_argument("--finetune-protocol", required=True)
    return parser.parse_args()


def _items(data: dict[str, object]) -> list[dict[str, object]]:
    for key in ("table", "anchors", "fit_points"):
        value = data.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _width(item: dict[str, object]) -> list[int]:
    raw = item.get("num_filters")
    if isinstance(raw, list):
        return [int(value) for value in raw]
    return [
        int(item[key])
        for key in ("s0", "s1", "s2")
        if item.get(key) is not None
    ]


def main() -> int:
    args = parse_args()
    source = Path(args.source_json)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text(encoding="utf-8"))
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    rows = []
    for item in _items(data):
        value = item.get("ap70")
        if value is None:
            continue
        width = _width(item)
        label = str(item.get("label", item.get("tag", "_".join(str(v) for v in width))))
        software_point_id = f"{label}:w{'x'.join(str(value) for value in width)}:fp16"
        config_id = stable_config_id(
            model=args.model,
            candidate_id=label,
            software_point_id=software_point_id,
            quant_policy="fp16",
            schedule_policy="not_applicable",
        )
        rows.append(
            ap_anchor_row(
                config_id=config_id,
                model=args.model,
                manifest_digest=digest,
                candidate_id=label,
                software_point_id=software_point_id,
                dense_stage=str(item.get("dense_stage", "model")),
                width=width,
                quant_policy="fp16",
                schedule_policy="not_applicable",
                backend="model_eval",
                measurement_status="measured",
                metric=args.metric,
                metric_value=float(value),
                secondary_metrics={},
                dataset=args.dataset,
                eval_split=args.eval_split,
                num_samples=None,
                ckpt_path=str(item.get("ckpt_path", "unknown")),
                ckpt_digest=item.get("ckpt_digest"),
                finetune_protocol=args.finetune_protocol,
                training_budget=str(item.get("training_budget", "unknown")),
                eval_command="imported_existing_ap_anchor",
                run_id=f"import_ap_{source.stem}",
                created_at=created_at,
                source_files=[str(source)],
                raw_artifact=str(source),
                provenance=str(
                    item.get("source", item.get("_source", data.get("_source", "imported_ap_anchor")))
                ),
                notes="imported from existing Stage2 AP anchor/model file",
            )
        )
    write_jsonl(args.out_jsonl, rows)
    print(f"stage2_import_ap_anchors_ok rows={len(rows)} out={args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
