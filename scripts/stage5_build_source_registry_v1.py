#!/usr/bin/env python3
"""Build the source-aware Stage5 Pyramid/CoDriving candidate registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.stage5.source_registry_v1 import (
    build_pyramid_checkpoint_inventory,
    build_source_registry,
    canonical_widths,
)


DEFAULT_GRAPH = (
    REPO_ROOT
    / "results/stage35_gold144_targeted_supplement_v2_20260714/final_gold176_v1"
    / "graph_features.json"
)
DEFAULT_CANDIDATES = (
    REPO_ROOT
    / "multi_agent/data/stage2_lut_generation_v1/generated/coverage_pipeline_v1"
    / "candidates/candidate_queue.jsonl"
)
DEFAULT_CHECKPOINT_ROOT = Path("/home/jichengzhi/heal_research/checkpoints/stage1")
DEFAULT_CODRIVING_ROOT = Path(
    "/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709"
)
DEFAULT_RESULT_ROOT = REPO_ROOT / "results/stage5_single_target_search_v2_gold176_20260718"
DEFAULT_PYRAMID_MODEL_ROOT = Path(
    "/home/jichengzhi/heal_research/checkpoints/stage5/pyramid_gold176_search_v1"
)
DEFAULT_PYRAMID_BASE_DIR = Path(
    "/home/jichengzhi/heal_research/checkpoints/stage1/"
    "Pyramid_DAIR_m1_base_2023_08_14_11_42_29"
)


def _rows(payload: Any, field: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get(field), list):
        return [dict(item) for item in payload[field]]
    raise ValueError(f"expected list or object containing {field}")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_idempotent(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to overwrite drifted registry artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-features-json", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--pyramid-candidate-jsonl", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--pyramid-checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--pyramid-model-root", type=Path, default=DEFAULT_PYRAMID_MODEL_ROOT)
    parser.add_argument("--pyramid-base-dir", type=Path, default=DEFAULT_PYRAMID_BASE_DIR)
    parser.add_argument("--codriving-model-root", type=Path, default=DEFAULT_CODRIVING_ROOT)
    parser.add_argument("--remote-result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_RESULT_ROOT / "candidate_source_registry_full.json")
    parser.add_argument("--inventory-json", type=Path, default=DEFAULT_RESULT_ROOT / "pyramid_checkpoint_inventory.json")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--allow-missing-pyramid", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_features = _rows(
        json.loads(args.graph_features_json.read_text(encoding="utf-8")),
        "graph_features",
    )
    inventory = build_pyramid_checkpoint_inventory(
        _jsonl(args.pyramid_candidate_jsonl),
        checkpoint_root=args.pyramid_checkpoint_root,
    )
    if inventory["missing_labels"] and not args.allow_missing_pyramid:
        raise ValueError(
            "Pyramid checkpoint inventory is incomplete: "
            + ",".join(inventory["missing_labels"])
        )
    base_checkpoint = args.pyramid_base_dir / "net_epoch_bestval_at23.pth"
    if not base_checkpoint.is_file():
        raise ValueError(f"Pyramid base checkpoint is missing: {base_checkpoint}")
    result = build_source_registry(
        graph_features=graph_features,
        pyramid_inventory=inventory["sources"],
        pyramid_widths=canonical_widths(),
        pyramid_model_root=args.pyramid_model_root,
        pyramid_base_source={
            "checkpoint_path": str(base_checkpoint),
            "checkpoint_dir": str(args.pyramid_base_dir),
            "checkpoint_sha256": hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
        },
        codriving_widths=canonical_widths(),
        remote_result_root=args.remote_result_root,
        codriving_model_root=args.codriving_model_root,
        seed=args.seed,
    )
    _write_idempotent(args.inventory_json, inventory)
    _write_idempotent(args.output_json, result)
    print(
        json.dumps(
            {
                "registry": str(args.output_json),
                "inventory": str(args.inventory_json),
                "group_count": result["group_count"],
                "pyramid_source_count": inventory["source_count"],
                "source_ready_group_count": result["source_ready_group_count"],
                "source_materializable_group_count": result[
                    "source_materializable_group_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
