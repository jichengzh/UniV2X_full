# 复现 H800 闭环仿真 — 操作文档 (v1, 2026-06-11)

> 给任何接手者(含其他 AI)。读完能在 H800(8×H800, sm90/Hopper) 上跑通 V2Xverse 闭环仿真并产出真实 driving score。
> **本文头条 = 一个常见误判的纠正**(见 §0), 不读这条会得出"H800 跑不了仿真"的错误结论。

---

## §0 ★必读 — 为什么"cu113 在 sm90 上卡死"不代表 H800 跑不了仿真

**H800 上有两个 Python 环境, 分属两个进程, 千万别搞混:**

| 环境 | torch | 用途 | 能在 sm90 GPU 跑吗 |
|------|-------|------|------------------|
| `/data/jichengzhi_v2x/envs/v2xverse` (py3.7) | **1.10.1 / cu113** | **进程 A: CARLA 客户端 + leaderboard + pnp_agent I/O** (CPU, 不碰 GPU 感知) | ❌ 不能(也不需要) |
| `/data/jichengzhi_v2x/t2lib` (py3.10, pip --target 目录) | **2.1.2 / cu121** | **进程 B: codriving 感知整脑** (GPU 推理) | ✅ **能**(arch_list 含 sm_90) |

**误判来源**: 只检查了 v2xverse(cu113) 那个环境, 发现它在 sm90 上第一个 CUDA 算子就卡死 → 结论"H800 跑不了"。**这是对的, 但无关** —— cu113 是 2021 栈, 没有 Hopper kernel, 我们从来不用它做 GPU 感知。
**真相**: 闭环用**双进程拆分**绕开了这个不兼容 —— cu113 只在进程 A 做 CARLA I/O(CPU); 真正的 GPU 感知在进程 B 用 cu121(sm90 原生)。所以"cu113 不支持 sm90"恰恰是**我们设计双进程的原因**, 不是阻塞。

**现场自证(任何人可复跑)**:
```bash
cd /data/jichengzhi_v2x/V2Xverse
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=/data/jichengzhi_v2x/t2lib python3 -c \
"import torch; print(torch.__version__, torch.cuda.get_arch_list()); \
x=torch.randn(512,512,device='cuda'); print('sm90 OK', float((x@x).sum()))"
# 期望: 2.1.2+cu121  ['sm_50',...,'sm_90']  sm90 OK <number>
```
> 2026-06-11 实测: 输出 `2.1.2+cu121 [... 'sm_90']`, GPU matmul 成功, device=NVIDIA H800。闭环当日多次跑出真实 DS(11.95/14.88)、Efficiency 53.86%、Comfort 0.034、IPC 4600+帧@~130ms。

**注**: pypi 不通 / 磁盘 100%(62G 剩) 都是真的, 但**只影响"装新环境", 不影响"跑已建好的 t2lib"**。t2lib 已建好且 sm90 可跑, 无需任何 pip install 即可运行闭环。

---

## §1 连接 + 资源纪律
- SSH: `sshpass -p '12345678' ssh -p 30001 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215`(密码每会话确认)
- **跑前必 `nvidia-smi`**: GPU 0/1 常被他人占; 用 util≈0、mem≤50MiB 的空闲卡。CARLA / 感知 / 各占一张(算力隔离)。
- **长任务必 detached**: `setsid nohup <cmd> >log 2>&1 </dev/null &` + 轮询日志。前台 ssh 长命令会被掐断。
- 磁盘 `/data` 99-100% 满, 写大文件前 `df -h /data`; 不要装新环境到 home(`/` 也满)。

---

## §2 三个进程 (闭环 = CARLA + 进程B + 进程A)

### 进程 ①: CARLA 0.9.10 headless (一张卡)
```bash
# launcher 已在: /data/jichengzhi_v2x/h800_carla_launch.sh  (args: GPU PORT)
setsid bash /data/jichengzhi_v2x/h800_carla_launch.sh 2 3000 \
  >/data/jichengzhi_v2x/carla_g2_p3000.log 2>&1 </dev/null &
# 轮询端口就绪(CARLA 启动 ~30-60s):
for i in $(seq 1 90); do nc -z -w1 127.0.0.1 3000 && { echo "CARLA ready"; break; }; sleep 1; done
```
> launcher 内容 = `CUDA_VISIBLE_DEVICES=$GPU ./CarlaUE4.sh -prefer-nvidia -opengl -RenderOffScreen -graphicsadapter=$GPU -world-port=$PORT -nosound -quality-level=Low`。H800 无 Vulkan, 走 OpenGL/EGL OK。

### 进程 ②: 感知 server (process B, 另一张卡, t2lib/cu121)
```bash
cd /data/jichengzhi_v2x/V2Xverse
CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH=/data/jichengzhi_v2x/t2lib:/data/jichengzhi_v2x/V2Xverse:/data/jichengzhi_v2x/V2Xverse/simulation/leaderboard \
ROUTES=simulation/leaderboard/data/evaluation_routes/town05_short_r146.xml \
SAVE_PATH=/tmp/b_save \
setsid nohup python3 simulation/leaderboard/team_code/closedloop/process_b_server.py \
  --port 5557 --gpu 3 \
  --pnp-config simulation/leaderboard/team_code/agent_config/pnp_config_codriving_norsu.yaml \
  >/data/jichengzhi_v2x/process_b_g3_p5557.log 2>&1 </dev/null &
# 等就绪: 轮询日志出现 === PROCESS_B_READY ===  (载模型~30-60s)
```
> process_b_server 内部: 注入 carla_shim → import 整个 `PnP_infer`(无真 carla) → 构建 codriving 感知+planner(cu121/GPU) → 包进 zmq InferServer。

### 进程 ③: leaderboard eval (process A, py3.7/v2xverse env)
```bash
cd /data/jichengzhi_v2x/V2Xverse
CUDA_VISIBLE_DEVICES=4 \
USE_INFER_SERVER=1 INFER_SERVER_PORT=5557 \
PATH=/data/jichengzhi_v2x/envs/v2xverse/bin:$PATH RECORD_PATH="" \
setsid nohup bash scripts/eval_driving_e2e.sh 146 3000 myrun 0 codriving_norsu _1 \
  >/data/jichengzhi_v2x/eval_myrun.log 2>&1 </dev/null &
# 参数: eval_driving_e2e.sh <routeId> <carla_port> <tag> <repeat> <agent_cfg> <scen_param>
# USE_INFER_SERVER=1 → pnp_agent 用 IPC proxy 转发给进程B, 不在 A 建模型(A 无 GPU 感知)
```
> 算力隔离: CARLA GPU2 / B GPU3 / A 不用 GPU(感知全在 B)。三者不同卡, 时延测量纯净。

---

## §3 加指标 (Efficiency / Comfort, 可选)
在进程 ③ 加 `METRIC_LOG=1 METRIC_LOG_PATH=<dir>`:
```bash
... USE_INFER_SERVER=1 INFER_SERVER_PORT=5557 METRIC_LOG=1 METRIC_LOG_PATH=/data/jichengzhi_v2x/metric_run1 \
   bash scripts/eval_driving_e2e.sh 146 3000 myrun 0 codriving_norsu _1 ...
# 跑完后离线算指标:
cd /data/jichengzhi_v2x/V2Xverse
PYTHONPATH=/data/jichengzhi_v2x/t2lib python3 \
  simulation/leaderboard/team_code/closedloop/bench2drive_metrics.py /data/jichengzhi_v2x/metric_run1/metric_info.json
# 输出 Efficiency(%) + Comfort(0-1)
```

---

## §4 验证 (怎么确认真跑通)
```bash
# (a) B server 活着? (任何时候)
cd /data/jichengzhi_v2x/V2Xverse
PYTHONPATH=/data/jichengzhi_v2x/t2lib:.:simulation/leaderboard:simulation/leaderboard/team_code \
python3 simulation/leaderboard/team_code/closedloop/process_b_client_test.py --port 5557
# 期望: ping response: OK / === PROCESS_B_CLIENT_OK ===

# (b) 跑完读 driving score:
python3 -c "import json; d=json.load(open('results/results_driving_myrun/v2x_final/town05_short_collab/r146_repeat0/ego_vehicle_0/results.json')); r=d['_checkpoint']['global_record']; print('DS',r['scores']['score_composed'],'RC',r['scores']['score_route'],'status',r['status'])"

# (c) 进程 A 日志应见 [InferProxy] frame=... latency_ms=... (每帧打到 B 的 RTT)
```

---

## §5 关键文件清单 (H800: /data/jichengzhi_v2x/V2Xverse/)
| 文件 | 作用 |
|------|------|
| `simulation/leaderboard/team_code/closedloop/process_b_server.py` | 进程B: 起感知 server |
| `simulation/leaderboard/team_code/closedloop/infer_server.py` | zmq ROUTER/DEALER + CUDA计时(被 server 用) |
| `simulation/leaderboard/team_code/closedloop/carla_shim/__init__.py` | 让 PnP_infer 在 py3.10 无真 carla 下 import |
| `simulation/leaderboard/team_code/closedloop/process_b_client_test.py` | ping/验证 |
| `simulation/leaderboard/team_code/closedloop/bench2drive_metrics.py` | 离线 Efficiency/Comfort |
| `simulation/leaderboard/team_code/pnp_agent_e2e.py` | 进程A agent (USE_INFER_SERVER 门控 proxy + METRIC_LOG logger) |
| `scripts/eval_driving_e2e.sh` | 进程A 启动包装 |
| `checkpoints/codriving/{perception,planner}/` | codriving 感知+planner 权重(51M) |
| `/data/jichengzhi_v2x/h800_carla_launch.sh` | CARLA 启动器 |

torch2 移植补丁(已打, 见 PROGRESS_h800_sim_smoke_v1.md): opencood spconv1→2 import(4文件) + pcdet_utils THC删/重编 sm90 + box_overlaps cython重编 + codriving_attn 删 turtle + utils/__init__ 绕 birdeye。

---

## §6 已知约束 / 排错
- **第一次在某张卡冷启 CARLA 可能 sensor queue timeout** → 先跑一条 warmup route 再正式跑。
- **看到 cu113/torch1.10 在 sm90 卡死** → 正常, 那是进程A的CPU env, 不要用它跑感知(见 §0)。
- **磁盘满** → 别装新环境; 跑现有 t2lib 不需装包。清 `results/`、旧 log 腾空间。
- **pypi 不通** → 跑闭环不需联网; t2lib 已自足。要装新包才需代理 `--proxy http://127.0.0.1:7890`。
- **连接易断** → 全部 detached + 轮询, 别前台跑长命令。
- **GPU 被占** → nvidia-smi 选空闲卡, 改 `--gpu`/`CUDA_VISIBLE_DEVICES` 与 `--graphicsadapter`/端口。
- **canonical 栈** (当前已起, 可直接复用): CARLA GPU2:3000 + B server GPU3:5557(norsu config)。

---

## §7 一句话给"判 H800 跑不了"的接手者
你检查的是 cu113 那个 CPU 客户端环境(它确实不支持 sm90, 但本就不该用它跑感知)。真正的感知跑在 **t2lib (torch2.1+cu121, sm90 原生)** 的**独立进程 B**, 经 zmq 与 CARLA 客户端解耦。按 §2 三进程起栈、§4 验证, 即可复现真实 driving score。**H800 闭环 2026-06-11 当日已多次实跑验证。**
