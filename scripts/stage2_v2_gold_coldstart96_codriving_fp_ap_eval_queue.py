#!/usr/bin/env python3
"""Launch CoDriving FP AP inference after v2 gold finetune jobs finish."""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REMOTE_REPO = Path("/exdata/jichengzhi/V2Xverse_pyramid")
REMOTE_OUT_ROOT = REMOTE_REPO / "output/codriving_v2_gold_ap_20260709"
REMOTE_RESULT_ROOT = Path("/home/jichengzhi/V2X/results/v2_gold_coldstart_96_20260708")
REMOTE_AP_RAW_ROOT = REMOTE_RESULT_ROOT / "codriving_ap_raw"
REMOTE_LOG_ROOT = REMOTE_OUT_ROOT / "logs"
PYTHON = "python3"
POLL_SECS = 300

DEFAULT_WIDTHS = (
    "16x32x64",
    "24x32x96",
    "32x32x128",
    "32x64x128",
    "48x96x192",
    "56x112x224",
    "64x96x192",
    "64x128x256",
    "24x64x128",
    "40x64x128",
    "48x64x128",
    "64x64x128",
)

TRAIN_GPU_BY_WIDTH = {
    "16x32x64": 5,
    "32x64x128": 5,
    "64x96x192": 5,
    "48x64x128": 5,
    "24x32x96": 6,
    "48x96x192": 6,
    "64x128x256": 6,
    "64x64x128": 6,
    "32x32x128": 7,
    "56x112x224": 7,
    "24x64x128": 7,
    "40x64x128": 7,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_jobs(widths: Iterable[str] = DEFAULT_WIDTHS, eval_gpu: int = 4) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for width in widths:
        train_gpu = TRAIN_GPU_BY_WIDTH[width]
        model_dir = REMOTE_OUT_ROOT / width
        jobs.append(
            {
                "width": width,
                "eval_gpu": int(eval_gpu),
                "train_gpu": train_gpu,
                "model_dir": str(model_dir),
                "train_log": str(REMOTE_LOG_ROOT / f"train_{width}_gpu{train_gpu}.log"),
                "eval_log": str(REMOTE_LOG_ROOT / f"fp_ap_eval_{width}_gpu{eval_gpu}.log"),
                "raw_ap_dir": str(REMOTE_AP_RAW_ROOT / width),
            }
        )
    return jobs


def command_for(job: dict[str, Any], poll_secs: int = POLL_SECS) -> str:
    train_log = _q(job["train_log"])
    model_dir = _q(job["model_dir"])
    eval_log = _q(job["eval_log"])
    raw_ap_dir = _q(job["raw_ap_dir"])
    width = _q(job["width"])
    eval_gpu = int(job["eval_gpu"])
    repo = _q(REMOTE_REPO)
    return "\n".join(
        [
            f"echo '[wait] {width}'",
            f"while ! grep -q 'Training Finished' {train_log}; do",
            f"  if [ -f {train_log} ] && grep -q 'Traceback\\|RuntimeError\\|ModuleNotFoundError' {train_log}; then",
            f"    echo '[warn] training log has errors before finish for {width}'",
            "  fi",
            f"  sleep {int(poll_secs)}",
            "done",
            f"mkdir -p {raw_ap_dir}",
            f"if ls {raw_ap_dir}/eval_intermediate_epoch*.yaml >/dev/null 2>&1; then",
            f"  echo '[skip] existing AP yaml for {width}'",
            "else",
            f"  echo '[eval] {width}'",
            f"  cd {repo}",
            "  export PYTHONPATH=/exdata/jichengzhi/tp_lib:/data/jichengzhi_v2x/t2lib:/exdata/jichengzhi/V2Xverse_pyramid:.",
            (
                f"  CUDA_VISIBLE_DEVICES={eval_gpu} {PYTHON} -u opencood/tools/inference.py "
                f"--model_dir {model_dir} --fusion_method intermediate > {eval_log} 2>&1"
            ),
            f"  cp -f {model_dir}/eval_intermediate_epoch*.yaml {raw_ap_dir}/",
            "fi",
        ]
    )


def render_launcher(jobs: list[dict[str, Any]], poll_secs: int = POLL_SECS) -> str:
    body = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for job in jobs:
        body.append(command_for(job, poll_secs=poll_secs))
        body.append("")
    return "\n".join(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-gpu", type=int, default=4)
    parser.add_argument("--poll-secs", type=int, default=POLL_SECS)
    parser.add_argument("--widths", nargs="*", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--out", type=Path, default=None, help="Write launcher shell script instead of stdout only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = build_jobs(args.widths, eval_gpu=args.eval_gpu)
    payload = {
        "schema": "v2_gold_coldstart_96_codriving_fp_ap_eval_queue_v1",
        "created_at_utc": utc_now(),
        "jobs": jobs,
    }
    script = render_launcher(jobs, poll_secs=args.poll_secs)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(script + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if args.out is None:
        print(script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
