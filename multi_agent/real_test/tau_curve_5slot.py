#!/usr/bin/env python3
"""
tau_curve_5slot.py — τ_ego 延迟→V2X驾驶分曲线, 5-slot 并行 (GPU 0-4)

依据 HANDOFF_tau_curve_final_plan_v1.md:
  - 平台 H800, 配置 _1notraffic (删 ambient, 保留 Scenario3 脚本行人 + RSU)
  - 每 slot 1 GPU, CARLA + process_b 共卡 (同 GPU); eval(进程A) 无 GPU
  - N-repeat = 每次独立 RESULT_NAME (..._n{rep}_notraffic), repeat 参数恒 0

可配置 (env):
  ROUTE_FILE   路线库文件(每行一个 route id); 缺省用内置候选
  TAUS         逗号分隔延迟档(ms), 缺省 "0,100,200,300,400,500"
  NREP         每格重复次数, 缺省 3
  TAG_PREFIX   结果目录前缀, 缺省 "taucurve"

断点续跑: 已有 results.json + 非空 status 的 (tau,route,rep) 跳过。
"""
import os, subprocess, time, json, sys, logging, socket

VXDIR   = "/data/jichengzhi_v2x/V2Xverse"
LOGDIR  = "/data/jichengzhi_v2x/taucurve_logs"
PROGLOG = "/data/jichengzhi_v2x/taucurve_progress.log"
V2XENV  = "/data/jichengzhi_v2x/envs/v2xverse/bin"
T2LIB   = "/data/jichengzhi_v2x/t2lib"
CARLA_L = "/data/jichengzhi_v2x/h800_carla_launch.sh"

os.makedirs(LOGDIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S',
    handlers=[logging.FileHandler(PROGLOG, mode='a'), logging.StreamHandler(sys.stdout)],
)
log = logging.info

# 5 slots: GPU 0-4, CARLA + B co-located per GPU
# (gpu, carla_port, b_port)
SLOTS = [
    (0, 3000, 5600),
    (1, 3200, 5620),
    (2, 3400, 5640),
    (3, 3600, 5660),
    (4, 3800, 5680),
]
NSLOT = len(SLOTS)

# ── config ───────────────────────────────────────────────────────────────────
DEFAULT_ROUTES = [0,1,3,5,7,11,13,16,17,24,136,146,164,302,307,310,311,317,325,326]
TAG_PREFIX = os.environ.get("TAG_PREFIX", "taucurve")
NREP       = int(os.environ.get("NREP", "3"))
TAUS       = [int(x) for x in os.environ.get("TAUS", "0,100,200,300,400,500").split(",")]

rf = os.environ.get("ROUTE_FILE", "")
if rf and os.path.isfile(rf):
    ROUTES = [int(l.strip()) for l in open(rf) if l.strip() and not l.startswith("#")]
else:
    ROUTES = DEFAULT_ROUTES

slot_busy  = [False] * NSLOT
slot_job   = [None]  * NSLOT   # (tau, route, rep)
slot_epid  = [None]  * NSLOT


def tag_of(tau, route, rep):
    return f"{TAG_PREFIX}_te{tau}_r{route}_n{rep}_notraffic"

def result_path(tau, route, rep):
    t = tag_of(tau, route, rep)
    return os.path.join(VXDIR,
        f"results/results_driving_{t}/v2x_final/town05_short_collab/"
        f"r{route}_repeat0/ego_vehicle_0/results.json")

def already_done(tau, route, rep):
    rp = result_path(tau, route, rep)
    if not os.path.isfile(rp):
        return False
    try:
        d = json.load(open(rp))
        return bool(d['_checkpoint']['global_record'].get('status', ''))
    except Exception:
        return False

def kill_pattern(pat):
    subprocess.run(['pkill', '-9', '-f', pat], capture_output=True)

def wait_port(port, timeout=180):
    for _ in range(timeout):
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False

def wait_b_ready(logfile, timeout=300):
    for _ in range(timeout // 2):
        if os.path.isfile(logfile):
            with open(logfile, errors='replace') as f:
                if '=== PROCESS_B_READY ===' in f.read():
                    return True
        time.sleep(2)
    return False


def start_slot(slot, tau, route, rep):
    gpu, cport, bport = SLOTS[slot]
    tag  = tag_of(tau, route, rep)
    clog = f"{LOGDIR}/carla_s{slot}_p{cport}.log"
    blog = f"{LOGDIR}/b_s{slot}_p{bport}_{tag}.log"
    elog = f"{LOGDIR}/eval_s{slot}_{tag}.log"

    log(f"START slot={slot} tau={tau} route={route} rep={rep} GPU={gpu} cport={cport} bport={bport}")

    kill_pattern(f"CarlaUE4.*world-port={cport}")
    kill_pattern(f"process_b_server.*--port {bport}")
    time.sleep(1)

    # ① process_b FIRST (full GPU for model load before CARLA)
    if os.path.isfile(blog):
        os.remove(blog)
    env_b = {**os.environ,
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'PYTHONPATH': f"{T2LIB}:{VXDIR}:{VXDIR}/simulation/leaderboard",
        'ROUTES': f"simulation/leaderboard/data/evaluation_routes/town05_short_r{route}.xml",
        'SAVE_PATH': f"/data/jichengzhi_v2x/bsave/b_save_s{slot}"}
    with open(blog, 'w') as bf:
        bp = subprocess.Popen(
            ['python3',
             f"{VXDIR}/simulation/leaderboard/team_code/closedloop/process_b_server.py",
             '--port', str(bport), '--gpu', str(gpu),
             '--pnp-config',
             f"simulation/leaderboard/team_code/agent_config/pnp_config_codriving_te{tau}_l1.yaml"],
            cwd=VXDIR, stdout=bf, stderr=bf, stdin=subprocess.DEVNULL,
            env=env_b, start_new_session=True)
    log(f"  B server pid={bp.pid} — waiting model load...")
    if not wait_b_ready(blog):
        log(f"  ERROR B not ready port={bport} — abort slot")
        return False

    # ② CARLA (B already loaded)
    with open(clog, 'w') as cf:
        subprocess.Popen(['setsid', 'bash', CARLA_L, str(gpu), str(cport)],
            stdout=cf, stderr=cf, stdin=subprocess.DEVNULL, start_new_session=True)
    if not wait_port(cport):
        log(f"  ERROR CARLA not ready port={cport} — abort slot")
        return False
    log(f"  CARLA ready port={cport}")

    # ③ eval (Process A, no GPU, _1notraffic, default scenarios json)
    env_a = {**os.environ,
        'CUDA_VISIBLE_DEVICES': '',
        'USE_INFER_SERVER':  '1',
        'INFER_SERVER_PORT': str(bport),
        'RECORD_PATH':       '',
        'PATH': f"{V2XENV}:{os.environ.get('PATH', '')}"}
    with open(elog, 'w') as ef:
        ep = subprocess.Popen(
            ['bash', f"{VXDIR}/scripts/eval_driving_e2e.sh",
             str(route), str(cport), tag, '0',
             f"codriving_te{tau}_l1", "_1notraffic"],
            cwd=VXDIR, stdout=ef, stderr=ef, stdin=subprocess.DEVNULL,
            env=env_a, start_new_session=True)
    slot_epid[slot] = ep.pid
    slot_busy[slot] = True
    slot_job[slot]  = (tau, route, rep)
    log(f"  eval pid={ep.pid}")
    return True


def free_slot(slot):
    gpu, cport, bport = SLOTS[slot]
    kill_pattern(f"process_b_server.*--port {bport}")
    kill_pattern(f"CarlaUE4.*world-port={cport}")
    slot_busy[slot] = False
    slot_job[slot]  = None
    slot_epid[slot] = None

def check_done(slot):
    job = slot_job[slot]
    if job is None:
        return False
    tau, route, rep = job
    rp = result_path(tau, route, rep)
    if os.path.isfile(rp):
        try:
            d  = json.load(open(rp))
            st = d['_checkpoint']['global_record'].get('status', '')
            if st:
                log(f"DONE slot={slot} tau={tau} r={route} n={rep} status={st}")
                free_slot(slot)
                return True
        except Exception:
            pass
    epid = slot_epid[slot]
    if epid is not None:
        try:
            os.kill(epid, 0)
        except ProcessLookupError:
            log(f"WARN slot={slot} eval pid={epid} died (tau={tau} r={route} n={rep}) — mark done")
            free_slot(slot)
            return True
    return False


def main():
    all_jobs = [(tau, r, rep) for tau in TAUS for r in ROUTES for rep in range(1, NREP+1)]
    done_set = {j for j in all_jobs if already_done(*j)}
    jobs     = [j for j in all_jobs if j not in done_set]
    total    = len(jobs)

    log("=== TAU CURVE 5-SLOT START ===")
    log(f"  ROUTES({len(ROUTES)})={ROUTES}")
    log(f"  TAUS={TAUS}  NREP={NREP}  TAG_PREFIX={TAG_PREFIX}")
    log(f"  Total={len(all_jobs)}  done={len(done_set)}  pending={total}")

    job_idx = 0
    completed = 0
    while completed < total:
        for slot in range(NSLOT):
            if slot_busy[slot] and check_done(slot):
                completed += 1
                log(f"Progress: {completed}/{total}")
        for slot in range(NSLOT):
            if not slot_busy[slot] and job_idx < total:
                tau, route, rep = jobs[job_idx]
                job_idx += 1
                if not start_slot(slot, tau, route, rep):
                    log(f"WARN start_slot failed — requeue (tau={tau} r={route} n={rep})")
                    job_idx -= 1
                    time.sleep(20)
        time.sleep(10)
    log(f"=== TAU CURVE COMPLETE: {completed}/{total} new + {len(done_set)} skipped ===")


if __name__ == '__main__':
    main()
