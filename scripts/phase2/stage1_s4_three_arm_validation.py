#!/usr/bin/env python3
"""Run S4 mini three-arm validation from measured S2 anchor cells.

The S4 input is the existing H800/TVM S2 24-cell matrix. This script does not
create new latency measurements. It evaluates three search policies over the
measured candidate cells:

- local_only: serial local baseline P -> Q -> S.
- pair_search: best of P x S with local Q fixed, and Q x S with local P fixed.
- joint_search: full measured P x Q x S mini search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "results/stage1_model_predict/s2_anchor_scan/"
    "stage1_s2_anchor_scan_v1_20260623_2110.json"
)
DEFAULT_OUT_JSON = (
    ROOT
    / "results/stage1_model_predict/s4_three_arm_validation/"
    "stage1_s4_three_arm_validation_v1.json"
)
DEFAULT_OUT_MD = DEFAULT_OUT_JSON.with_suffix(".md")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _candidate_rows(cells: list[dict[str, Any]], probe_id: str, batch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell.get("probe_id") != probe_id or int(cell.get("batch", -1)) != batch:
            continue
        for arm_key, schedule_name in (
            ("default", "default"),
            ("own_tuned", "own_tuned"),
            ("schedule_swap", "fp16_tuned_swap"),
        ):
            arm = cell.get(arm_key, {}) or {}
            if not arm.get("ok"):
                continue
            value = arm.get("value", {}) or {}
            mean_us = value.get("mean_us")
            if mean_us is None or float(mean_us) <= 0:
                continue
            rows.append(
                {
                    "probe_id": probe_id,
                    "batch": batch,
                    "p_label": cell["p_label"],
                    "width": int(cell["width"]),
                    "precision": cell["precision"],
                    "schedule": schedule_name,
                    "latency_us": float(mean_us),
                    "source_arm": arm_key,
                }
            )
    return rows


def _winner(rows: list[dict[str, Any]], *, arm: str, note: str) -> dict[str, Any]:
    if not rows:
        return {"arm": arm, "status": "NO_CANDIDATES", "note": note}
    ranked = sorted(rows, key=lambda item: item["latency_us"])
    winner = dict(ranked[0])
    winner.update(
        {
            "arm": arm,
            "status": "OK",
            "rank": [
                {
                    "p_label": item["p_label"],
                    "width": item["width"],
                    "precision": item["precision"],
                    "schedule": item["schedule"],
                    "latency_us": round(item["latency_us"], 6),
                }
                for item in ranked
            ],
            "note": note,
        }
    )
    winner["latency_us"] = round(winner["latency_us"], 6)
    return winner


def _same_config(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ("p_label", "width", "precision", "schedule")
    return all(a.get(key) == b.get(key) for key in keys)


def _analyze_probe_batch(cells: list[dict[str, Any]], probe_id: str, batch: int) -> dict[str, Any]:
    rows = _candidate_rows(cells, probe_id, batch)
    p_baseline_rows = [
        row for row in rows if row["precision"] == "fp16" and row["schedule"] == "default"
    ]
    p_pick = _winner(
        p_baseline_rows,
        arm="local_p_pick",
        note="P selected under fp16/default schedule baseline.",
    )

    q_rows = [
        row
        for row in rows
        if row["p_label"] == p_pick.get("p_label") and row["schedule"] == "default"
    ]
    q_pick = _winner(
        q_rows,
        arm="local_q_pick",
        note="Q selected at local P with default schedule.",
    )

    s_rows = [
        row
        for row in rows
        if row["p_label"] == p_pick.get("p_label")
        and row["precision"] == q_pick.get("precision")
    ]
    local_only = _winner(
        s_rows,
        arm="local_only",
        note="Serial local baseline P(fp16/default) -> Q(default) -> S.",
    )

    ps_rows = [row for row in rows if row["precision"] == q_pick.get("precision")]
    ps_pair = _winner(
        ps_rows,
        arm="pair_search_ps",
        note="P x S pair search with local Q fixed.",
    )
    qs_rows = [row for row in rows if row["p_label"] == p_pick.get("p_label")]
    qs_pair = _winner(
        qs_rows,
        arm="pair_search_qs",
        note="Q x S pair search with local P fixed.",
    )
    pair_options = [item for item in (ps_pair, qs_pair) if item.get("status") == "OK"]
    pair_search = _winner(
        pair_options,
        arm="pair_search",
        note="Best of the two measured pair arms: P x S or Q x S.",
    )

    joint_search = _winner(
        rows,
        arm="joint_search",
        note="Full measured P x Q x S mini search over existing S2 cells.",
    )

    local_to_pair_change = not _same_config(local_only, pair_search)
    pair_to_joint_change = not _same_config(pair_search, joint_search)
    local_to_joint_change = not _same_config(local_only, joint_search)
    latency_gain_local_to_joint = None
    if local_only.get("status") == "OK" and joint_search.get("status") == "OK":
        latency_gain_local_to_joint = round(
            float(local_only["latency_us"]) / float(joint_search["latency_us"]),
            6,
        )

    if local_to_joint_change:
        implication = "LOCAL_ONLY_UNSAFE_PAIR_OR_JOINT_REQUIRED"
    elif pair_to_joint_change:
        implication = "PAIR_SEARCH_INSUFFICIENT_JOINT_REQUIRED"
    else:
        implication = "LOCAL_OR_PAIR_MATCHES_JOINT_FOR_THIS_MEASURED_BATCH"

    return {
        "probe_id": probe_id,
        "batch": batch,
        "candidate_count": len(rows),
        "arms": {
            "local_p_pick": p_pick,
            "local_q_pick": q_pick,
            "local_only": local_only,
            "pair_search_ps": ps_pair,
            "pair_search_qs": qs_pair,
            "pair_search": pair_search,
            "joint_search": joint_search,
        },
        "winner_changes": {
            "local_to_pair": local_to_pair_change,
            "pair_to_joint": pair_to_joint_change,
            "local_to_joint": local_to_joint_change,
        },
        "latency_gain_local_to_joint": latency_gain_local_to_joint,
        "predictor_relevant_rule_change": local_to_joint_change or pair_to_joint_change,
        "implication": implication,
    }


def _anchor_rollup(results: list[dict[str, Any]]) -> dict[str, Any]:
    winner_change_batches = [
        item["batch"] for item in results if item["winner_changes"]["local_to_joint"]
    ]
    pair_to_joint_change_batches = [
        item["batch"] for item in results if item["winner_changes"]["pair_to_joint"]
    ]
    max_gain = max(
        (
            float(item["latency_gain_local_to_joint"])
            for item in results
            if item.get("latency_gain_local_to_joint") is not None
        ),
        default=1.0,
    )
    if pair_to_joint_change_batches:
        verdict = "JOINT_SEARCH_REQUIRED_FOR_MEASURED_LATENCY_ANCHOR"
    elif winner_change_batches:
        verdict = "PAIR_SEARCH_REQUIRED_LOCAL_ONLY_UNSAFE"
    else:
        verdict = "LOCAL_MATCHES_JOINT_IN_THIS_MINI_VALIDATION"
    return {
        "verdict": verdict,
        "winner_change_batches": winner_change_batches,
        "pair_to_joint_change_batches": pair_to_joint_change_batches,
        "max_latency_gain_local_to_joint": round(max_gain, 6),
        "scope": "latency-only synthetic H800/TVM schedule anchor; not AP/HV or full-model proof",
    }


def build_report(input_json: Path) -> dict[str, Any]:
    source = _load_json(input_json)
    cells = source.get("cells", [])
    anchors = source.get("anchors", [])
    if not isinstance(cells, list) or not isinstance(anchors, list):
        raise ValueError("S2 input must contain list fields: cells and anchors")

    probe_ids = [str(item["probe_id"]) for item in anchors if isinstance(item, dict)]
    batches = sorted({int(cell["batch"]) for cell in cells if isinstance(cell, dict)})
    results: list[dict[str, Any]] = []
    for probe_id in probe_ids:
        for batch in batches:
            results.append(_analyze_probe_batch(cells, probe_id, batch))

    rollup = {
        probe_id: _anchor_rollup([item for item in results if item["probe_id"] == probe_id])
        for probe_id in probe_ids
    }

    return {
        "schema": "stage1_s4_three_arm_validation_v1",
        "generated_on": "2026-06-23",
        "input": _rel(input_json),
        "measurement_status": "REUSE_EXISTING_H800_TVM_S2_MEASURED_CELLS",
        "runner": _rel(Path(__file__)),
        "policy": {
            "no_new_s2_matrix": True,
            "latency_only": True,
            "no_ap_or_hv_claim": True,
            "scope": "two high-risk synthetic schedule anchors from S2",
        },
        "anchors": anchors,
        "arm_definitions": {
            "local_only": "P selected at fp16/default, Q selected at local P/default, S selected at local P/Q.",
            "pair_search": "Best of P x S with local Q fixed and Q x S with local P fixed.",
            "joint_search": "Best measured P x Q x S candidate.",
        },
        "results": results,
        "anchor_rollup": rollup,
        "global_conclusion": (
            "S4 latency-only validation finds local-only winner changes for measured "
            "BaseBEVBackbone and neck/deconv anchors. Pair search matches joint in "
            "this mini matrix, so S5 should require at least pair-level schedule "
            "calibration for these regimes and must not emit static low-risk."
        ),
        "unsupported_conclusions": [
            "Does not prove full-model irreducible coupling.",
            "Does not provide AP/HV joint-vs-serial evidence.",
            "Does not cover attention/fusion/custom skipped subgraphs.",
        ],
    }


def _fmt_config(item: dict[str, Any]) -> str:
    if item.get("status") != "OK":
        return str(item.get("status"))
    return (
        f"{item.get('p_label')} w{item.get('width')} / {item.get('precision')} / "
        f"{item.get('schedule')} / {item.get('latency_us')} us"
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage1 S4 Mini Three-Arm Validation v1",
        "",
        "S4 复用 S2 的 H800/TVM 实测 24-cell 矩阵；本报告不新增 latency 测量，也不扩展 S2 大矩阵。",
        "",
        f"- measurement_status: `{report['measurement_status']}`",
        f"- input: `{report['input']}`",
        f"- runner: `{report['runner']}`",
        f"- latency_only: `{report['policy']['latency_only']}`",
        f"- no_ap_or_hv_claim: `{report['policy']['no_ap_or_hv_claim']}`",
        "",
        "## Arm Definitions",
        "",
        f"- local-only: {report['arm_definitions']['local_only']}",
        f"- pair-search: {report['arm_definitions']['pair_search']}",
        f"- joint-search: {report['arm_definitions']['joint_search']}",
        "",
        "## Results",
        "",
        "| probe | batch | candidates | local-only winner | pair-search winner | joint-search winner | local->joint change | pair->joint change | gain local/joint | implication |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---|",
    ]
    for item in report["results"]:
        arms = item["arms"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item['probe_id']}`",
                    str(item["batch"]),
                    str(item["candidate_count"]),
                    _fmt_config(arms["local_only"]),
                    _fmt_config(arms["pair_search"]),
                    _fmt_config(arms["joint_search"]),
                    str(item["winner_changes"]["local_to_joint"]),
                    str(item["winner_changes"]["pair_to_joint"]),
                    str(item.get("latency_gain_local_to_joint")),
                    f"`{item['implication']}`",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Anchor Rollup", ""])
    lines.extend(
        [
            "| probe | verdict | winner change batches | pair->joint batches | max gain |",
            "|---|---|---|---|---:|",
        ]
    )
    for probe_id, item in report["anchor_rollup"].items():
        lines.append(
            f"| `{probe_id}` | `{item['verdict']}` | {item['winner_change_batches']} | "
            f"{item['pair_to_joint_change_batches']} | {item['max_latency_gain_local_to_joint']} |"
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            report["global_conclusion"],
            "",
            "Unsupported conclusions:",
        ]
    )
    lines.extend(f"- {item}" for item in report["unsupported_conclusions"])
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(Path(args.input_json))
    _write_json(Path(args.out_json), report)
    _write_text(Path(args.out_md), render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
