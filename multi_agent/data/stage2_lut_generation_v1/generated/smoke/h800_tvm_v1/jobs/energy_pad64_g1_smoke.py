from __future__ import annotations
import csv
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/jichengzhi/V2X/multi_agent/data/stage2_lut_generation_v1/generated/smoke/h800_tvm_v1')
GPU = '1'
LABEL = 'pad64'
WIDTH = [64, 96, 192]
ONNX = '/exdata/jichengzhi/s2_tvm/models/trap25_pad64_backbone.onnx'
WORK_DIR = '/exdata/jichengzhi/s2_tvm/ms_work_2e_pad64_retest'
LATENCY_RUN_ID = 'smoke_h800_tvm_pyramid_pad64_fp16_tuned_20260625_192000'
MEASURE_ITERS = 1500
WARMUP_ITERS = 50
REPEAT = 1
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_ID = f'smoke_h800_tvm_energy_pyramid_{LABEL}_fp16_tuned_{STAMP}'
RAW = ROOT / 'raw' / RUN_ID
RAW.mkdir(parents=True, exist_ok=True)


def now_local():
    return datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S%z')


def run_text(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f'{cmd} failed')
    return proc.stdout


def gpu_snapshot():
    out = run_text(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw,pstate','--format=csv,noheader,nounits'])
    snap = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            snap[parts[0]] = {'util': int(float(parts[1])), 'mem_mib': int(float(parts[2])), 'power_w': float(parts[3]), 'pstate': parts[4]}
    return snap


def target_has_pmon(gpu):
    proc = subprocess.run(['nvidia-smi','pmon','-c','1','-s','um'], capture_output=True, text=True, check=False)
    for line in (proc.stdout or '').splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        parts = s.split()
        if len(parts) >= 2 and parts[0] == gpu and parts[1] != '-':
            return True
    return False


def wait_idle():
    clean = 0
    hist = []
    deadline = time.time() + 900
    while time.time() < deadline:
        snap = gpu_snapshot().get(GPU, {'util': 999, 'mem_mib': 999999})
        has_pmon = target_has_pmon(GPU)
        item = {'ts': now_local(), 'gpu': GPU, 'util': snap['util'], 'mem_mib': snap['mem_mib'], 'has_pmon': has_pmon}
        hist.append(item)
        print(json.dumps({'event':'idle_check', **item}), flush=True)
        if snap['util'] <= 5 and snap['mem_mib'] <= 1024 and not has_pmon:
            clean += 1
            if clean >= 3:
                return hist
        else:
            clean = 0
        time.sleep(5)
    raise RuntimeError(f'GPU{GPU} not idle before timeout; tail={hist[-5:]}')


def query_power_w():
    out = run_text(['nvidia-smi', f'--id={GPU}', '--query-gpu=power.draw', '--format=csv,noheader,nounits'])
    return float(out.strip().splitlines()[0].strip())


def sample_idle(seconds=5.0, interval=0.2):
    rows = []
    end = time.time() + seconds
    while time.time() < end:
        ts = time.time()
        try:
            rows.append((ts, query_power_w()))
        except Exception:
            pass
        time.sleep(interval)
    return rows


def write_power_csv(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['timestamp_s','power_w'])
        for ts, w in rows:
            writer.writerow([f'{ts:.6f}', f'{w:.3f}'])


def power_stats(rows):
    vals = [w for _, w in rows]
    vals_sorted = sorted(vals)
    if not vals:
        return {'avg': None, 'p50': None, 'p90': None}
    p50 = vals_sorted[len(vals_sorted)//2]
    p90 = vals_sorted[min(len(vals_sorted)-1, int(0.9*(len(vals_sorted)-1)))]
    return {'avg': sum(vals)/len(vals), 'p50': p50, 'p90': p90}


def read_inputs(m):
    init = {i.name for i in m.graph.initializer}
    return {i.name: tuple(d.dim_value for d in i.type.tensor_type.shape.dim) for i in m.graph.input if i.name not in init}


def main():
    state_path = ROOT / 'jobs' / 'job_state_energy_pad64_g1.jsonl'
    state = {'schema':'lut_job_state_row_v1','job_id':'energy:smoke_h800_tvm_pyramid_pad64_fp16_tuned_g1','status':'running','attempt':1,'started_at':now_local(),'finished_at':None,'error':None,'log_path':str(ROOT/'logs'/'energy_smoke_pad64_g1.log'),'output_row_id':None}
    with state_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(state, sort_keys=True)+'\n')
    result = {'schema':'stage2_h800_energy_smoke_result_v1','status':'started','gpu':GPU,'model':'pyramid_lidar','label':LABEL,'width':WIDTH,'onnx_path':ONNX,'work_dir':WORK_DIR,'run_id':RUN_ID,'latency_run_id':LATENCY_RUN_ID,'warmup_iters':WARMUP_ITERS,'measure_iters':MEASURE_ITERS,'repeat':REPEAT,'batch_size':1}
    try:
        idle_hist = wait_idle()
        result['idle_history_tail'] = idle_hist[-3:]
        (RAW/'hostname.txt').write_text(socket.gethostname()+'\n')
        (RAW/'nvidia_smi_preflight.csv').write_text(run_text(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,pstate','--format=csv']))
        subprocess.run(['nvidia-smi','pmon','-c','1','-s','um'], stdout=(RAW/'nvidia_smi_pmon_preflight.txt').open('w'), stderr=subprocess.STDOUT, text=True, check=False)
        (RAW/'env.json').write_text(json.dumps({'CUDA_VISIBLE_DEVICES':GPU,'LD_LIBRARY_PATH_set':bool(os.environ.get('LD_LIBRARY_PATH')),'PATH_prefix':os.environ.get('PATH','').split(':')[:3]}, indent=2)+'\n')
        (RAW/'command.json').write_text(json.dumps({'schema':'stage2_h800_energy_smoke_command_v1','mode':'nvidia_smi_power_draw_tuned_vm','gpu':GPU,'model':'pyramid_lidar','label':LABEL,'width':WIDTH,'onnx_path':ONNX,'work_dir':WORK_DIR,'run_id':RUN_ID,'latency_run_id':LATENCY_RUN_ID,'measure_iters':MEASURE_ITERS}, indent=2)+'\n')
        idle_rows = sample_idle(5.0, 0.2)
        write_power_csv(RAW/'idle_power_samples.csv', idle_rows)
        idle = power_stats(idle_rows)
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
        import numpy as np
        import onnx
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        import tvm.s_tir.tensor_intrin.cuda  # noqa: F401
        dev = tvm.cuda(0)
        target = tvm.target.Target.from_device(dev)
        m = onnx.load(ONNX)
        shapes = read_inputs(m)
        result['input_shape'] = {k:list(v) for k,v in shapes.items()}
        rng = np.random.RandomState(0)
        feeds_np = {k: rng.rand(*v).astype('float32') for k,v in shapes.items()}
        mod0 = from_onnx(m, shape_dict=shapes, keep_params_in_input=False)
        seq = tvm.transform.Sequential([relax.transform.LegalizeOps(), relax.transform.AnnotateTIROpPattern(), relax.transform.FuseOps(), relax.transform.FuseTIR()])
        t_build = time.time()
        with target, tvm.transform.PassContext(opt_level=3):
            modt = seq(mod0)
            sched = relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR)(modt)
            ex = tvm.compile(sched, target=target)
        result['build_time_s'] = round(time.time() - t_build, 3)
        vm = relax.VirtualMachine(ex, dev)
        args = [tvm.runtime.tensor(feeds_np[k], device=dev) for k in shapes]
        for _ in range(WARMUP_ITERS):
            vm['main'](*args)
        dev.sync()
        samples = []
        stop = {'value': False}
        def poller():
            while not stop['value']:
                try:
                    samples.append((time.time(), query_power_w()))
                except Exception:
                    pass
                time.sleep(0.05)
        th = threading.Thread(target=poller, daemon=True)
        th.start()
        t0 = time.time()
        vm['main'](*args)
        # Use one time_evaluator batch for a stable active window.
        vf = vm.time_evaluator('main', dev, number=MEASURE_ITERS, repeat=1)
        timing = vf(*args)
        dev.sync()
        active_duration = time.time() - t0
        stop['value'] = True
        th.join(timeout=1.0)
        write_power_csv(RAW/'active_power_samples.csv', samples)
        active = power_stats(samples)
        time_eval_s = float(timing.results[0])
        joule = max(0.0, ((active['avg'] or 0.0) - (idle['avg'] or 0.0)) * active_duration / MEASURE_ITERS)
        result.update({'status':'success','active_duration_s':round(active_duration, 6),'active_samples':len(samples),'idle_samples':len(idle_rows),'idle_watt_avg':round(idle['avg'], 4),'watt_avg':round(active['avg'], 4),'watt_p50':round(active['p50'], 4),'watt_p90':round(active['p90'], 4),'sample_window_ms':int(round(active_duration*1000)),'time_evaluator_s':round(time_eval_s, 6),'joule_per_inference':round(joule, 8)})
        telemetry = {'run_id':RUN_ID,'latency_run_id':LATENCY_RUN_ID,'latency_config_id':'smoke_h800_tvm_pyramid_lidar_pad64_fp16_metaschedule_tuned','joule_per_inference':result['joule_per_inference'],'watt_avg':result['watt_avg'],'watt_p50':result['watt_p50'],'watt_p90':result['watt_p90'],'idle_watt_avg':result['idle_watt_avg'],'idle_baseline_policy':'subtract_idle_avg_5s_pre_window','sample_window_ms':result['sample_window_ms'],'telemetry_source':'nvidia-smi power.draw polling 50ms','power_cap_watt':None,'clock_policy':'default','warmup_iters':WARMUP_ITERS,'measure_iters':MEASURE_ITERS,'repeat':REPEAT,'provenance':'H800 power telemetry smoke aligned with tuned TVM VM','source_files':[ONNX, WORK_DIR, str(RAW/'energy_result.json'), str(RAW/'idle_power_samples.csv'), str(RAW/'active_power_samples.csv')],'raw_artifact':str(RAW),'notes':'energy smoke telemetry; target GPU idle at preflight; other GPUs may be occupied so not paper-grade energy'}
        (RAW/'energy_result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        (RAW/'telemetry_payload.json').write_text(json.dumps(telemetry, indent=2, sort_keys=True)+'\n')
        print(json.dumps({'event':'energy_success','label':LABEL,'joule_per_inference':result['joule_per_inference'],'watt_avg':result['watt_avg'],'run_id':RUN_ID}), flush=True)
        final = dict(state, status='succeeded', finished_at=now_local())
    except Exception as exc:
        result['status'] = 'failed'; result['error'] = repr(exc); result['traceback'] = traceback.format_exc()
        (RAW/'energy_result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        print(json.dumps({'event':'energy_failed','error':repr(exc),'run_id':RUN_ID}), flush=True)
        final = dict(state, status='failed', finished_at=now_local(), error=repr(exc))
    with state_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(final, sort_keys=True)+'\n')
    (ROOT/'logs'/'energy_smoke_pad64_g1.log.json').write_text(json.dumps({'schema':'lut_job_log_v1','job_id':state['job_id'],'raw':str(RAW),'result':result,'returncode':0 if result.get('status') == 'success' else 1}, indent=2, sort_keys=True)+'\n')
    (ROOT/'compare'/'energy_smoke_pad64_g1_summary_v1.json').write_text(json.dumps({'schema':'energy_smoke_pad64_g1_summary_v1','created_at':now_local(),'result':result}, indent=2, sort_keys=True)+'\n')
    return 0 if result.get('status') == 'success' else 1

if __name__ == '__main__':
    raise SystemExit(main())
