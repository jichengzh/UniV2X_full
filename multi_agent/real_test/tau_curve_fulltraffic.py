#!/usr/bin/env python3
"""
tau_curve_fulltraffic.py — τ_ego 延迟→V2X驾驶分曲线 (FULL-traffic 版, 多slot并行)

依据 4090 te0-clean 拆分定论: 保留 ambient(_1) + 只跑 te0-clean 路线 + N-repeat
→ 干净单调退化曲线 (100→78), 而 _1notraffic 会抹平信号。

env 可配:
  ROUTE_FILE     路线库文件(每行一个 route id), 缺省内置 57 条
  TAUS           逗号分隔延迟档(ms), 缺省 "0,100,300,400,500"
  NREP           每格重复次数, 缺省 3
  TAG_PREFIX     结果目录前缀, 缺省 "ftclean"
  SCEN_SUFFIX    scenario_parameter 后缀, 缺省 "_1notraffic"; full-traffic 用 "_1"
  GPULIST        逗号分隔 GPU id, 缺省 "0,1,2,3,4"
  SLOTS_PER_GPU  每 GPU 并行 slot 数, 缺省 1
  PORT_BASE      CARLA 起始端口, 缺省 3000

每 slot: 1 GPU(可与他 slot 共卡), CARLA + process_b 同卡; eval(进程A) 无 GPU。
N-repeat = 每次独立 RESULT_NAME(..._n{rep}), repeat 参数恒 0。断点续跑。
"""
import os, subprocess, time, json, sys, logging, socket

VXDIR   = "/data/jichengzhi_v2x/V2Xverse"
LOGDIR  = "/data/jichengzhi_v2x/ftclean_logs"
PROGLOG = "/data/jichengzhi_v2x/ftclean_progress.log"
V2XENV  = "/data/jichengzhi_v2x/envs/v2xverse/bin"
T2LIB   = "/data/jichengzhi_v2x/t2lib"
CARLA_L = "/data/jichengzhi_v2x/h800_carla_launch.sh"
os.makedirs(LOGDIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(PROGLOG, mode='a'), logging.StreamHandler(sys.stdout)])
log = logging.info

# ── config ───────────────────────────────────────────────────────────────────
DEFAULT_ROUTES = [0,1,2,3,4,5,6,17,18,20,24,26,28,29,30,31,100,101,102,103,104,105,
    106,107,108,109,110,111,112,113,135,136,137,138,139,141,142,143,144,145,147,160,
    161,162,300,302,303,304,307,311,312,314,316,317,319,325,326]
TAG_PREFIX  = os.environ.get("TAG_PREFIX", "ftclean")
CFG_TAG     = os.environ.get("CFG_TAG", "te")   # te=tau_ego(plan延迟), tp=tau_perc(感知延迟)
SCEN_SUFFIX = os.environ.get("SCEN_SUFFIX", "_1notraffic")
NREP        = int(os.environ.get("NREP", "3"))
TAUS        = [int(x) for x in os.environ.get("TAUS", "0,100,300,400,500").split(",")]
PORT_BASE   = int(os.environ.get("PORT_BASE", "3000"))

rf = os.environ.get("ROUTE_FILE", "")
if rf and os.path.isfile(rf):
    ROUTES = [int(l.strip()) for l in open(rf) if l.strip() and not l.startswith("#")]
else:
    ROUTES = DEFAULT_ROUTES

# slots from GPULIST × SLOTS_PER_GPU
_gpus = [int(x) for x in os.environ.get("GPULIST", "0,1,2,3,4").split(",")]
_spg  = int(os.environ.get("SLOTS_PER_GPU", "1"))
SLOTS = []           # (gpu, carla_port, b_port)
_p = 0
for g in _gpus:
    for _ in range(_spg):
        SLOTS.append((g, PORT_BASE + _p*200, 5600 + _p*20)); _p += 1
NSLOT = len(SLOTS)

slot_busy = [False]*NSLOT
slot_job  = [None]*NSLOT
slot_epid = [None]*NSLOT
slot_start = [0.0]*NSLOT
RUN_TIMEOUT = int(os.environ.get("RUN_TIMEOUT", "900"))  # 单run超时(s): 防崩route卡死slot

def tag_of(tau, route, rep):
    return f"{TAG_PREFIX}_{CFG_TAG}{tau}_r{route}_n{rep}"
def result_path(tau, route, rep):
    return os.path.join(VXDIR, f"results/results_driving_{tag_of(tau,route,rep)}/"
        f"v2x_final/town05_short_collab/r{route}_repeat0/ego_vehicle_0/results.json")
def already_done(tau, route, rep):
    rp = result_path(tau, route, rep)
    if not os.path.isfile(rp): return False
    try: return bool(json.load(open(rp))['_checkpoint']['global_record'].get('status',''))
    except Exception: return False
def kill_pattern(pat): subprocess.run(['pkill','-9','-f',pat], capture_output=True)
def wait_port(port, timeout=180):
    for _ in range(timeout):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1): return True
        except OSError: time.sleep(1)
    return False
def wait_b_ready(logfile, timeout=300):
    for _ in range(timeout//2):
        if os.path.isfile(logfile):
            with open(logfile, errors='replace') as f:
                if '=== PROCESS_B_READY ===' in f.read(): return True
        time.sleep(2)
    return False

def start_slot(slot, tau, route, rep):
    gpu, cport, bport = SLOTS[slot]
    tag = tag_of(tau, route, rep)
    clog = f"{LOGDIR}/carla_s{slot}_p{cport}.log"
    blog = f"{LOGDIR}/b_s{slot}_p{bport}_{tag}.log"
    elog = f"{LOGDIR}/eval_s{slot}_{tag}.log"
    log(f"START slot={slot} tau={tau} r={route} n={rep} GPU={gpu} cport={cport} bport={bport}")
    kill_pattern(f"CarlaUE4.*world-port={cport}")
    kill_pattern(f"process_b_server.*--port {bport}")
    time.sleep(1)
    if os.path.isfile(blog): os.remove(blog)
    env_b = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
        'PYTHONPATH': f"{T2LIB}:{VXDIR}:{VXDIR}/simulation/leaderboard",
        'ROUTES': f"simulation/leaderboard/data/evaluation_routes/town05_short_r{route}.xml",
        'SAVE_PATH': f"/data/jichengzhi_v2x/bsave/b_save_s{slot}"}
    with open(blog,'w') as bf:
        bp = subprocess.Popen(['python3',
            f"{VXDIR}/simulation/leaderboard/team_code/closedloop/process_b_server.py",
            '--port', str(bport), '--gpu', str(gpu),
            '--pnp-config', f"simulation/leaderboard/team_code/agent_config/pnp_config_codriving_{CFG_TAG}{tau}_l1.yaml"],
            cwd=VXDIR, stdout=bf, stderr=bf, stdin=subprocess.DEVNULL, env=env_b, start_new_session=True)
    log(f"  B pid={bp.pid} loading...")
    if not wait_b_ready(blog):
        log(f"  ERROR B not ready bport={bport}"); return False
    with open(clog,'w') as cf:
        subprocess.Popen(['setsid','bash',CARLA_L,str(gpu),str(cport)],
            stdout=cf, stderr=cf, stdin=subprocess.DEVNULL, start_new_session=True)
    if not wait_port(cport):
        log(f"  ERROR CARLA not ready cport={cport}"); return False
    log(f"  CARLA ready cport={cport}")
    env_a = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'USE_INFER_SERVER':'1',
        'INFER_SERVER_PORT': str(bport), 'RECORD_PATH':'',
        'PATH': f"{V2XENV}:{os.environ.get('PATH','')}"}
    with open(elog,'w') as ef:
        ep = subprocess.Popen(['bash', f"{VXDIR}/scripts/eval_driving_e2e.sh",
            str(route), str(cport), tag, '0', f"codriving_{CFG_TAG}{tau}_l1", SCEN_SUFFIX],
            cwd=VXDIR, stdout=ef, stderr=ef, stdin=subprocess.DEVNULL, env=env_a, start_new_session=True)
    slot_epid[slot]=ep.pid; slot_busy[slot]=True; slot_job[slot]=(tau,route,rep)
    slot_start[slot]=time.time()
    log(f"  eval pid={ep.pid}")
    return True

def free_slot(slot):
    gpu, cport, bport = SLOTS[slot]
    kill_pattern(f"process_b_server.*--port {bport}")
    kill_pattern(f"CarlaUE4.*world-port={cport}")
    slot_busy[slot]=False; slot_job[slot]=None; slot_epid[slot]=None

def check_done(slot):
    job = slot_job[slot]
    if job is None: return False
    tau, route, rep = job
    rp = result_path(tau, route, rep)
    if os.path.isfile(rp):
        try:
            st = json.load(open(rp))['_checkpoint']['global_record'].get('status','')
            if st:
                log(f"DONE slot={slot} tau={tau} r={route} n={rep} status={st}")
                free_slot(slot); return True
        except Exception: pass
    epid = slot_epid[slot]
    if epid is not None:
        try: os.kill(epid, 0)
        except ProcessLookupError:
            log(f"WARN slot={slot} eval died (tau={tau} r={route} n={rep})")
            free_slot(slot); return True
    # 超时保护: slot busy 超过 RUN_TIMEOUT 仍无 results → 判崩溃, 强制 kill+跳过
    if time.time() - slot_start[slot] > RUN_TIMEOUT:
        log(f"TIMEOUT slot={slot} tau={tau} r={route} n={rep} (>{RUN_TIMEOUT}s) 强制跳过")
        if epid is not None:
            try: os.killpg(os.getpgid(epid), 9)
            except Exception: pass
        try:  # 写超时标记 results.json, 防重启无限重试该崩route
            rp = result_path(tau, route, rep)
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            json.dump({"_checkpoint":{"global_record":{"status":"TIMEOUT_SKIP","scores":{},"infractions":{}}}}, open(rp,"w"))
        except Exception: pass
        free_slot(slot); return True
    return False

def main():
    all_jobs = [(t,r,n) for t in TAUS for r in ROUTES for n in range(1,NREP+1)]
    done = {j for j in all_jobs if already_done(*j)}
    jobs = [j for j in all_jobs if j not in done]
    total = len(jobs)
    log("=== TAU CURVE FULL-TRAFFIC START ===")
    log(f"  ROUTES({len(ROUTES)}) TAUS={TAUS} NREP={NREP} SCEN={SCEN_SUFFIX} TAG={TAG_PREFIX}")
    log(f"  SLOTS({NSLOT})={SLOTS}")
    log(f"  total={len(all_jobs)} done={len(done)} pending={total}")
    job_idx=0; completed=0
    while completed < total:
        for slot in range(NSLOT):
            if slot_busy[slot] and check_done(slot):
                completed += 1; log(f"Progress: {completed}/{total}")
        for slot in range(NSLOT):
            if not slot_busy[slot] and job_idx < total:
                t,r,n = jobs[job_idx]; job_idx += 1
                if not start_slot(slot, t, r, n):
                    log(f"WARN start failed, requeue (tau={t} r={r} n={n})")
                    job_idx -= 1; time.sleep(20)
        time.sleep(10)
    log(f"=== COMPLETE: {completed}/{total} new + {len(done)} skipped ===")

if __name__ == '__main__':
    main()
