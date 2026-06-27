from __future__ import annotations
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/calibration/latency_h800_tvm_v1')
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
GPU = '1'
MEASURE_ITERS = 500
REPEAT = 5
WARMUP = 1
JOBS = [
    {
        'label': 's1_64',
        'onnx_path': '/exdata/jichengzhi/s2_tvm/models/s1_64_backbone.onnx',
        'work_dir': '/exdata/jichengzhi/s2_tvm/ms_bumped_s1_64',
        'width': [64, 64, 256],
    },
    {
        'label': 'mix_b',
        'onnx_path': '/exdata/jichengzhi/s2_tvm/models/mix_b_backbone.onnx',
        'work_dir': '/exdata/jichengzhi/s2_tvm/ms_bumped_mix_b',
        'width': [48, 64, 256],
    },
]

for sub in ['raw', 'logs', 'jobs', 'compare']:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)


def now_local() -> str:
    return datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f'{cmd} failed')
    return proc.stdout


def gpu_snapshot() -> dict[str, tuple[int, int]]:
    out = run_text(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', '--format=csv,noheader,nounits'])
    snap = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3:
            snap[parts[0]] = (int(float(parts[1])), int(float(parts[2])))
    return snap


def target_has_pmon(gpu: str) -> bool:
    proc = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'], capture_output=True, text=True, check=False)
    text = proc.stdout or ''
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[0] == gpu and parts[1] != '-':
            return True
    return False


def wait_gpu_idle(gpu: str, *, required_samples: int = 3, interval_s: int = 5, timeout_s: int = 900) -> list[dict[str, object]]:
    deadline = time.time() + timeout_s
    clean = 0
    history = []
    while time.time() < deadline:
        snap = gpu_snapshot()
        util, mem = snap.get(gpu, (999, 999999))
        has_pmon = target_has_pmon(gpu)
        sample = {'ts': now_local(), 'gpu': gpu, 'util': util, 'mem_mib': mem, 'has_pmon': has_pmon}
        history.append(sample)
        print(json.dumps({'event': 'idle_check', **sample}), flush=True)
        if util <= 5 and mem <= 1024 and not has_pmon:
            clean += 1
            if clean >= required_samples:
                return history
        else:
            clean = 0
        time.sleep(interval_s)
    raise RuntimeError(f'GPU{gpu} not idle before timeout; last={history[-5:]}')


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding='utf-8')


def read_inputs(onnx_model):
    init = {i.name for i in onnx_model.graph.initializer}
    return {
        i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim)
        for i in onnx_model.graph.input
        if i.name not in init
    }


def summarize_times(results_s: list[float]) -> dict[str, object]:
    vals = [float(x) * 1e6 for x in results_s]
    vals_sorted = sorted(vals)
    mid = len(vals_sorted) // 2
    if len(vals_sorted) % 2:
        p50 = vals_sorted[mid]
    else:
        p50 = (vals_sorted[mid - 1] + vals_sorted[mid]) / 2.0
    return {
        'us': round(p50, 3),
        'mean_us': round(sum(vals) / len(vals), 3),
        'min_us': round(min(vals), 3),
        'max_us': round(max(vals), 3),
        'repeats_us': [round(v, 3) for v in vals],
    }


def time_vm(vm, args, dev, number: int, repeat: int) -> dict[str, object]:
    for _ in range(WARMUP):
        vm['main'](*args)
        dev.sync()
    vf = vm.time_evaluator('main', dev, number=number, repeat=repeat)
    result = vf(*args)
    return summarize_times(list(result.results))


def payload(result: dict[str, object], raw: Path, schedule: str) -> dict[str, object]:
    prefix = 'default' if schedule == 'default' else 'tuned'
    strategy = 'relax_default' if schedule == 'default' else 'relax_metaschedule_reuse_existing_ms_db'
    build_key = 'build_default_s' if schedule == 'default' else 'build_tuned_s'
    return {
        'batch_size': 1,
        'build_status': 'success',
        'build_time_s': result.get(build_key),
        'input_shape': result.get('input_shape', {}),
        'latency_max_us': result.get(f'{prefix}_max_us'),
        'latency_mean_us': result.get(f'{prefix}_mean_us'),
        'latency_min_us': result.get(f'{prefix}_min_us'),
        'latency_p50_us': result.get(f'{prefix}_us'),
        'measure_iters': MEASURE_ITERS,
        'notes': f"latency calibration medium clean retry g1; {result['label']}; {schedule}; backbone-only; no fresh tune",
        'provenance': f'H800 TVM latency calibration medium clean retry g1; {schedule}; reuse existing artifacts',
        'raw_artifact': str(raw),
        'repeat': REPEAT,
        'run_id': result.get('run_id'),
        'source_files': [result['onnx_path'], result['work_dir'], str(raw / 'latency_result.json')],
        'tvm_strategy': strategy,
        'tvm_target': 'cuda',
        'warmup_iters': WARMUP,
    }


def measure(job: dict[str, object]) -> dict[str, object]:
    label = str(job['label'])
    run_id = f'calib_h800_tvm_pyramid_{label}_fp16_medium_clean_retry_g1_{STAMP}'
    raw = ROOT / 'raw' / run_id
    raw.mkdir(parents=True, exist_ok=True)
    state = {'schema': 'lut_job_state_row_v1', 'job_id': f'latency:calibration_h800_tvm_pyramid_{label}_fp16_medium_clean_retry_g1', 'status': 'running', 'attempt': 1, 'started_at': now_local(), 'finished_at': None, 'error': None, 'log_path': str(ROOT / 'logs' / f'{label}.medium_clean_retry_g1.log.json'), 'output_row_id': None}
    with (ROOT / 'jobs' / 'job_state_medium_clean_retry_g1.jsonl').open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(state, sort_keys=True) + '\n')
    result: dict[str, object] = {
        'schema': 'stage2_h800_calibration_latency_result_v1',
        'status': 'started',
        'model': 'pyramid_lidar',
        'label': label,
        'gpu': GPU,
        'run_id': run_id,
        'onnx_path': job['onnx_path'],
        'work_dir': job['work_dir'],
        'width': job['width'],
        'warmup_iters': WARMUP,
        'measure_iters': MEASURE_ITERS,
        'repeat': REPEAT,
        'batch_size': 1,
    }
    try:
        idle_history = wait_gpu_idle(GPU)
        result['idle_history_tail'] = idle_history[-3:]
        write_text(raw / 'nvidia_smi_preflight.csv', run_text(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits']))
        subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um'], stdout=(raw / 'nvidia_smi_pmon_preflight.txt').open('w'), stderr=subprocess.STDOUT, text=True, check=False)
        write_text(raw / 'compute_apps_preflight.csv', subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_bus_id,pid,process_name,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=False).stdout)
        result['preflight_status'] = 'clean'
        write_text(raw / 'hostname.txt', socket.gethostname() + '\n')
        write_text(raw / 'env.json', json.dumps({'CUDA_VISIBLE_DEVICES': GPU, 'python': sys.executable}, indent=2) + '\n')
        write_text(raw / 'command.json', json.dumps({'schema': 'stage2_h800_latency_medium_retry_command_v1', 'mode': 'medium_clean_retry_g1_reuse_existing_ms_db', 'gpu': GPU, 'label': label, 'run_id': run_id, 'onnx_path': job['onnx_path'], 'work_dir': job['work_dir'], 'measure_iters': MEASURE_ITERS, 'repeat': REPEAT}, indent=2) + '\n')

        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
        import numpy as np
        import onnx
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401

        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        model = onnx.load(str(job['onnx_path']))
        shapes = read_inputs(model)
        result['input_shape'] = {k: list(v) for k, v in shapes.items()}
        rng = np.random.RandomState(0)
        feeds_np = {k: rng.rand(*v).astype('float32') for k, v in shapes.items()}
        mod0 = from_onnx(model, shape_dict=shapes, keep_params_in_input=False)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in shapes]

        t0 = time.time()
        with tvm.transform.PassContext(opt_level=3):
            ex = relax.build(mod0, target='cuda')
        result['build_default_s'] = round(time.time() - t0, 3)
        vm = relax.VirtualMachine(ex, dev)
        default_stats = time_vm(vm, args, dev, MEASURE_ITERS, REPEAT)
        result.update({
            'default_us': default_stats['us'],
            'default_mean_us': default_stats['mean_us'],
            'default_min_us': default_stats['min_us'],
            'default_max_us': default_stats['max_us'],
            'default_repeats_us': default_stats['repeats_us'],
        })
        del vm, ex
        dev.sync()

        seq = tvm.transform.Sequential([
            relax.transform.LegalizeOps(),
            relax.transform.AnnotateTIROpPattern(),
            relax.transform.FuseOps(),
            relax.transform.FuseTIR(),
        ])
        t1 = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
            scheduled = relax.transform.MetaScheduleApplyDatabase(work_dir=str(job['work_dir']))(modt)
            ex2 = tvm.compile(scheduled, target=target)
        result['build_tuned_s'] = round(time.time() - t1, 3)
        vm2 = relax.VirtualMachine(ex2, dev)
        tuned_stats = time_vm(vm2, args, dev, MEASURE_ITERS, REPEAT)
        result.update({
            'tuned_us': tuned_stats['us'],
            'tuned_mean_us': tuned_stats['mean_us'],
            'tuned_min_us': tuned_stats['min_us'],
            'tuned_max_us': tuned_stats['max_us'],
            'tuned_repeats_us': tuned_stats['repeats_us'],
        })
        result['ratio'] = round(float(result['default_us']) / float(result['tuned_us']), 6)
        result['status'] = 'success'
        write_text(raw / 'latency_result.json', json.dumps(result, indent=2, sort_keys=True) + '\n')
        write_text(raw / 'measurement_payload_default.json', json.dumps(payload(result, raw, 'default'), indent=2, sort_keys=True) + '\n')
        write_text(raw / 'measurement_payload_tuned.json', json.dumps(payload(result, raw, 'metaschedule_tuned'), indent=2, sort_keys=True) + '\n')
        print(json.dumps({'event': 'job_success', 'label': label, 'default_us': result['default_us'], 'tuned_us': result['tuned_us'], 'ratio': result['ratio'], 'run_id': run_id}), flush=True)
        final_state = dict(state, status='succeeded', finished_at=now_local())
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        write_text(raw / 'latency_result.json', json.dumps(result, indent=2, sort_keys=True) + '\n')
        write_text(raw / 'measurement_payload_default.json', json.dumps(payload(result, raw, 'default'), indent=2, sort_keys=True) + '\n')
        write_text(raw / 'measurement_payload_tuned.json', json.dumps(payload(result, raw, 'metaschedule_tuned'), indent=2, sort_keys=True) + '\n')
        print(json.dumps({'event': 'job_failed', 'label': label, 'error': repr(exc), 'run_id': run_id}), flush=True)
        final_state = dict(state, status='failed', finished_at=now_local(), error=repr(exc))
    with (ROOT / 'jobs' / 'job_state_medium_clean_retry_g1.jsonl').open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(final_state, sort_keys=True) + '\n')
    write_text(ROOT / 'logs' / f'{label}.medium_clean_retry_g1.log.json', json.dumps({'schema': 'lut_job_log_v1', 'job_id': state['job_id'], 'raw': str(raw), 'result': result, 'returncode': 0 if result.get('status') == 'success' else 1}, indent=2, sort_keys=True) + '\n')
    return result


def main() -> int:
    results = []
    for job in JOBS:
        results.append(measure(job))
        time.sleep(15)
    summary = {'schema': 'calibration_latency_medium_clean_retry_g1_summary_v1', 'created_at': now_local(), 'gpu': GPU, 'jobs': len(JOBS), 'results': results}
    write_text(ROOT / 'compare' / 'latency_calibration_medium_clean_retry_g1_summary_v1.json', json.dumps(summary, indent=2, sort_keys=True) + '\n')
    return 0 if all(r.get('status') == 'success' for r in results) else 1

if __name__ == '__main__':
    raise SystemExit(main())
