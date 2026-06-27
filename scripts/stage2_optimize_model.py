#!/usr/bin/env python3
"""Stage2 thin integration CLI.

Public input is limited to a Stage1 manifest and an optional Stage1
classification report. Missing classification fails closed unless an explicit
demo path is added by future code; this script does not infer evidence registry,
hardware target, or user-facing search policy from CLI flags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.stage2.contracts import Stage2Input, build_stage2_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Stage1 partition manifest path.",
    )
    parser.add_argument(
        "--classification",
        default=None,
        help="Stage1 model classification JSON path. Missing input fails closed.",
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Stage2Output JSON path.",
    )
    parser.add_argument(
        "--out-md",
        default=None,
        help="Optional Markdown summary path.",
    )
    parser.add_argument(
        "--evidence-delta-out",
        default=None,
        help="Optional Stage2 evidence delta JSON path.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Reserved explicit demo flag; output remains marked demo/proxy only.",
    )
    return parser.parse_args()


def _write_json(path: str | Path, data: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_md(data: dict) -> str:
    status = data["optimization_status"]
    claim = data["claim_boundaries"]
    lines = [
        f"# Stage2 Optimization — {data['model']}",
        "",
        f"- schema: `{data['schema']}`",
        f"- allowed: `{status['allowed']}`",
        f"- mode: `{status['mode']}`",
        f"- reason: `{status['reason']}`",
        f"- optimized_scope: `{data['optimized_scope']}`",
        f"- full_model_claim_allowed: `{claim['full_model_claim_allowed']}`",
        f"- hardware_context: `{data['hardware_context'].get('name', 'unknown')}`",
        "",
        "This output is scoped to the Stage1 manifest and classification input.",
        "Dense-core evidence is not promoted to full-model speedup.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    stage2_input = Stage2Input(
        manifest_path=Path(args.manifest),
        model_classification_path=Path(args.classification) if args.classification else None,
    )
    output = build_stage2_output(stage2_input)
    data = output.to_dict()
    if args.demo:
        data["demo_mode"] = True
        data["evidence_delta"]["notes"].append("explicit --demo was used")
    _write_json(args.out_json, data)
    if args.evidence_delta_out:
        _write_json(args.evidence_delta_out, data["evidence_delta"])
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_render_md(data), encoding="utf-8")
    print(
        "stage2_optimize_model_ok "
        f"model={data['model']} "
        f"allowed={data['optimization_status']['allowed']} "
        f"mode={data['optimization_status']['mode']} "
        f"out={Path(args.out_json)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
