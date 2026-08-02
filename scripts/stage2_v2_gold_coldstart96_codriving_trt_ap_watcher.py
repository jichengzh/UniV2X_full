#!/usr/bin/env python3
"""Run final CoDriving FP and TensorRT AP jobs as each v2 width finishes training."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable

from stage2_codriving_int8_provenance import report_has_valid_int8_calibration


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
TVM_ROUTE_B_TAG_BY_MODE = {
    "fp16": "tvm_routeb_fp16",
    "int8_all": "tvm_routeb_int8_all",
    "mixed_top25_flops": "tvm_routeb_int8_top25",
    "mixed_top50_flops": "tvm_routeb_int8_top50",
}
TVM_ROUTE_B_PRECISION_BY_MODE = {
    "fp16": "fp16",
    "int8_all": "int8",
    "mixed_top25_flops": "mixed",
    "mixed_top50_flops": "mixed",
}


def latest_bestval_ckpt(model_dir: Path) -> Path:
    candidates = list(model_dir.glob("net_epoch_bestval_at*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no net_epoch_bestval_at*.pth under {model_dir}")

    def epoch(path: Path) -> int:
        match = re.search(r"_at(\d+)\.pth$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch)


def trt_ap_raw_path(results_root: Path, width: str, mode: str) -> Path:
    tag_by_mode = {
        "fp16": "trt_fp16_final.json",
        "int8": "trt_int8_all_final.json",
    }
    if mode not in tag_by_mode:
        raise ValueError(f"unsupported TRT AP mode: {mode}")
    return results_root / "codriving_trt_hybrid_ap_raw" / width / tag_by_mode[mode]


def tvm_routeb_ap_raw_path(results_root: Path, width: str, mode: str) -> Path:
    if mode not in TVM_ROUTE_B_TAG_BY_MODE:
        raise ValueError(f"unsupported TVM RouteB AP mode: {mode}")
    filename = f"{TVM_ROUTE_B_TAG_BY_MODE[mode]}_final.json"
    return results_root / "codriving_tvm_routeb_resnet_ap_raw" / width / filename


def hybrid_report_complete(path: Path, min_samples: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    n_done = int(payload.get("n_done") or 0)
    n_trt_path = int(payload.get("n_trt_path") or 0)
    n_fallback_path = int(payload.get("n_fallback_path") or 0)
    return (
        payload.get("ap70") is not None
        and n_done >= min_samples
        and n_trt_path == n_done
        and n_fallback_path == 0
    )


def tvm_report_complete(path: Path, min_samples: int, expected_mode: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    n_done = int(payload.get("n_done") or 0)
    n_tvm_path = int(payload.get("n_tvm_path") or 0)
    n_fallback_path = int(payload.get("n_fallback_path") or 0)
    if expected_mode is not None and (
        payload.get("mode") != expected_mode
        or payload.get("tag") != TVM_ROUTE_B_TAG_BY_MODE.get(expected_mode)
        or payload.get("precision") != TVM_ROUTE_B_PRECISION_BY_MODE.get(expected_mode)
    ):
        return False
    return (
        payload.get("ap70") is not None
        and payload.get("pipeline_scope") == "tvm_routeb_resnet_in_full_pytorch_eval"
        and n_done >= min_samples
        and n_tvm_path == n_done
        and n_fallback_path == 0
        and report_has_valid_int8_calibration(payload)
    )


def training_finished(out_root: Path, width: str) -> bool:
    for log in (out_root / "logs").glob(f"train_{width}_gpu*.log"):
        try:
            if "Training Finished" in log.read_text(encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return False


def run_logged(
    cmd: list[str],
    *,
    log: Path,
    cwd: Path,
    env: dict[str, str],
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n[cmd] " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def copy_eval_yamls(model_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in model_dir.glob("eval_intermediate_epoch*.yaml"):
        shutil.copy2(path, target_dir / path.name)


def fp_ap_complete(results_root: Path, width: str) -> bool:
    return any((results_root / "codriving_ap_raw" / width).glob("eval_intermediate_epoch*.yaml"))


def base_env(gpu: str, pythonpath: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = pythonpath
    return env


def conda_opencood_env(gpu: str, local_v2x: Path, repo_root: Path, conda_python: str) -> dict[str, str]:
    env = base_env(gpu, f"{local_v2x}:{repo_root}")
    conda_bin = str(Path(conda_python).parent)
    env["PATH"] = f"{conda_bin}:{env.get('PATH', '')}"
    env["CODRIVING_EXPORT_DISABLE_LEGACY_T2LIB"] = "1"
    return env


def tvm_eval_env(gpu: str, local_v2x: Path, repo_root: Path) -> dict[str, str]:
    env = base_env(
        gpu,
        ":".join(
            [
                str(local_v2x),
                "/exdata/jichengzhi/tp_lib",
                "/data/jichengzhi_v2x/t2lib",
                "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages",
                str(repo_root),
                ".",
            ]
        ),
    )
    cuda_runtime = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
    tvm_lib = "/exdata/jichengzhi/tvm310/lib/python3.10/site-packages/tvm/lib"
    env["PATH"] = "/usr/local/cuda-12.2/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = f"{cuda_runtime}:{tvm_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["LD_PRELOAD"] = f"{cuda_runtime}/libcudart.so.12"
    return env


def run_width(args: argparse.Namespace, width: str) -> None:
    out_root = Path(args.out_root)
    model_dir = out_root / width
    results_root = Path(args.results_root)
    log_dir = out_root / "logs" / "final_ap_watcher" / width
    conda_env = conda_opencood_env(args.gpu, Path(args.local_v2x), Path(args.repo_root), args.conda_python)
    tvm_env = tvm_eval_env(args.gpu, Path(args.local_v2x), Path(args.repo_root))

    if not fp_ap_complete(results_root, width):
        run_logged(
            [
                args.conda_python,
                "-u",
                "opencood/tools/inference.py",
                "--model_dir",
                str(model_dir),
                "--fusion_method",
                "intermediate",
            ],
            log=log_dir / "fp_ap.log",
            cwd=Path(args.repo_root),
            env=conda_env,
        )
        copy_eval_yamls(model_dir, results_root / "codriving_ap_raw" / width)

    onnx = model_dir / f"collab_{width}_final_fp32.onnx"
    if not onnx.is_file():
        run_logged(
            [
                args.conda_python,
                "-u",
                str(Path(args.local_v2x) / "tools/export_onnx_codriving_collab.py"),
                "--ckpt",
                str(latest_bestval_ckpt(model_dir)),
                "--hypes",
                str(model_dir / "config.yaml"),
                "--out",
                str(onnx),
                "--device",
                "cuda:0",
            ],
            log=log_dir / "onnx_export.log",
            cwd=Path(args.repo_root),
            env=conda_env,
        )

    calib = model_dir / f"collab_calib_train_final_n{args.calib_samples}_float16.npz"
    calib_summary = model_dir / f"collab_calib_train_final_n{args.calib_samples}_float16_summary.json"
    if not calib.is_file() or not calib_summary.is_file():
        run_logged(
            [
                args.conda_python,
                "-u",
                str(Path(args.local_v2x) / "scripts/stage2_v2_gold_coldstart96_codriving_calib_export.py"),
                "--repo-root",
                str(args.repo_root),
                "--width",
                width,
                "--model-dir",
                str(model_dir),
                "--output",
                str(calib),
                "--summary",
                str(calib_summary),
                "--n-samples",
                str(args.calib_samples),
                "--num-workers",
                str(args.num_workers),
                "--progress-every",
                str(max(1, min(args.calib_samples, 4))),
                "--storage-dtype",
                "float16",
                "--split",
                "train",
            ],
            log=log_dir / "calib_export.log",
            cwd=Path(args.repo_root),
            env=conda_env,
        )

    engine_root = out_root / "trt_collab_engines_final"
    for mode in ("fp16", "int8"):
        engine = engine_root / width / f"collab_{width}_{mode}.engine"
        if not engine.is_file():
            cmd = [
                args.conda_python,
                "-u",
                str(Path(args.local_v2x) / "scripts/stage2_v2_gold_coldstart96_codriving_trt_collab_builder.py"),
                "--onnx",
                str(onnx),
                "--width",
                width,
                "--out-root",
                str(engine_root),
                "--mode",
                mode,
                "--workspace-mb",
                str(args.workspace_mb),
                "--summary",
                str(model_dir / f"trt_collab_{mode}_final_summary.json"),
            ]
            if mode == "int8":
                cmd.extend(["--calib-data", str(calib)])
            run_logged(cmd, log=log_dir / f"trt_build_{mode}.log", cwd=Path(args.repo_root), env=conda_env)

        out_json = model_dir / f"trt_hybrid_ap_final_{mode}.json"
        if not hybrid_report_complete(out_json, min_samples=args.ap_samples):
            run_logged(
                [
                    args.conda_python,
                    "-u",
                    str(Path(args.local_v2x) / "scripts/stage2_v2_gold_coldstart96_codriving_trt_hybrid_ap_eval.py"),
                    "--repo-root",
                    str(args.repo_root),
                    "--width",
                    width,
                    "--tag",
                    f"trt_{mode}_final" if mode == "fp16" else "trt_int8_all_final",
                    "--model-dir",
                    str(model_dir),
                    "--engine",
                    str(engine),
                    "--n-samples",
                    str(args.ap_samples),
                    "--num-workers",
                    str(args.num_workers),
                    "--progress-every",
                    str(args.progress_every),
                    "--eval-dir",
                    str(model_dir / f"trt_hybrid_ap_final_{mode}_eval"),
                    "--out-json",
                    str(out_json),
                ],
                log=log_dir / f"trt_hybrid_ap_{mode}.log",
                cwd=Path(args.repo_root),
                env=conda_env,
            )
        raw = trt_ap_raw_path(results_root, width, mode)
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json, raw)

    for mode in ("fp16", "int8_all", "mixed_top25_flops", "mixed_top50_flops"):
        out_json = model_dir / f"tvm_resnet_ap_final_{mode}.json"
        onnx = model_dir / f"resnet_multiscale_{width}_final_fp32.onnx"
        if not tvm_report_complete(out_json, min_samples=args.ap_samples, expected_mode=mode):
            run_logged(
                [
                    args.system_python,
                    "-u",
                    str(Path(args.local_v2x) / "scripts/stage2_v2_gold_coldstart96_codriving_tvm_resnet_ap_eval.py"),
                    "--repo-root",
                    str(args.repo_root),
                    "--width",
                    width,
                    "--mode",
                    mode,
                    "--model-dir",
                    str(model_dir),
                    "--onnx",
                    str(onnx),
                    "--calib-npz",
                    str(calib),
                    "--calib-summary",
                    str(calib_summary),
                    "--n-samples",
                    str(args.ap_samples),
                    "--num-workers",
                    str(args.num_workers),
                    "--progress-every",
                    str(args.progress_every),
                    "--tvm-gpu",
                    "0",
                    "--eval-dir",
                    str(model_dir / f"tvm_resnet_ap_final_{mode}_eval"),
                    "--out-json",
                    str(out_json),
                ],
                log=log_dir / f"tvm_resnet_ap_{mode}.log",
                cwd=Path(args.repo_root),
                env=tvm_env,
            )
        raw = tvm_routeb_ap_raw_path(results_root, width, mode)
        raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_json, raw)


def run_watcher(args: argparse.Namespace, widths: Iterable[str]) -> None:
    pending = list(widths)
    while pending:
        progressed = False
        for width in list(pending):
            if not training_finished(Path(args.out_root), width):
                continue
            print(f"[final-ap] start {width}", flush=True)
            run_width(args, width)
            pending.remove(width)
            progressed = True
            print(f"[final-ap] done {width} remaining={len(pending)}", flush=True)
        if pending and not progressed:
            print(f"[final-ap] waiting remaining={len(pending)} sleep={args.poll_secs}s", flush=True)
            time.sleep(args.poll_secs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/exdata/jichengzhi/V2Xverse_pyramid"))
    parser.add_argument("--local-v2x", type=Path, default=Path("/home/jichengzhi/V2X"))
    parser.add_argument("--out-root", type=Path, default=Path("/exdata/jichengzhi/V2Xverse_pyramid/output/codriving_v2_gold_ap_20260709"))
    parser.add_argument("--results-root", type=Path, default=Path("/home/jichengzhi/V2X/results/v2_gold_coldstart_96_20260708"))
    parser.add_argument("--conda-python", default="/home/jichengzhi/miniconda3/envs/UniV2X_2.0/bin/python")
    parser.add_argument("--system-python", default="/usr/bin/python3")
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--widths", nargs="*", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--poll-secs", type=int, default=300)
    parser.add_argument("--calib-samples", type=int, default=16)
    parser.add_argument("--ap-samples", type=int, default=1789)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workspace-mb", type=int, default=4096)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_watcher(args, args.widths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
