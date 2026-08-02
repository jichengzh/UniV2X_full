# 协同加速框架 — H800 调用与评测指南 (README)

> 目的:一站式说明**我们的软硬件协同加速搜索框架**在 H800 上怎么起、怎么测、怎么出 Pareto。
> 每次新 session 冷启动**先读本文**,不要再靠记忆猜命令。
> 纯 SSH 访问的深度排障参考 [`../multi_agent/methods/.../RUNBOOK_stage2_h800_server_access_v1_zh.md`](../multi_agent/methods/design/stage1-model-predict/auto-tuning/progress/RUNBOOK_stage2_h800_server_access_v1_zh.md)(本文格式沿用它,但**命令以本文为准**,RUNBOOK 部分内容已过时)。

---

## 0. 框架是什么(一句话)

在**单一联合软件空间 P×Q**(P=宽度 w0,w1,w2;Q=精度 {fp32,fp16,int8_tc})上做 **surrogate-assisted NSGA-II** 搜索,内层由 **TVM** 对每个 (w,q) tune 出唯一调度 s,**在 H800 空闲卡真测 latency/energy**,回流重训 cost model,迭代出 **RSU-感知 Pareto 前沿**(AP max / latency min / energy min 三极值)。

- **后端 = TVM**(dp4a/WMMA 张量化),**不是 TRT**。legacy TRT 数据只作历史参考。
- **精度 q 是搜索维**(进 NSGA-II 基因),不按精度拆多次跑。
- **验证靠 baseline 对照**(random-search hypervolume + 三臂消融),**不做 ground-truth 穷举**(穷举不 scale;见 `9_7_14` §9)。

---

## 1. H800 服务器访问

| 项 | 值 |
|----|----|
| host | `<PRIVATE_HOST>` |
| port | `30001` |
| user | `<LOCAL_USER>` |
| 认证 | **密码**(每 session 向用户口头确认;**永不明文写进命令/磁盘**) |
| python | `${V2X_HOME}/miniconda3/envs/UniV2X_2.0/bin/python` |

**连接纪律(强制)**:
- 密码用八进制注入环境变量,不出现在命令行明文:`printf -v PW '<REDACTED_LEGACY_SECRET>'`(示例编码,实际以用户给的为准)。
- 强制密码认证禁 pubkey:`-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new`。
- 加 `-o ConnectTimeout=60 -o ServerAliveInterval=30`,并给 `timeout` 包一层防挂死。

```bash
printf -v PW '<REDACTED_LEGACY_SECRET>'   # 占位,实际密码以用户为准
timeout 60 ssh -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no -o StrictHostKeyChecking=accept-new \
  -o ConnectTimeout=60 -o ServerAliveInterval=30 -p 30001 ${V2X_REMOTE_USER}@<PRIVATE_HOST> '<远端命令>'
```

**scp 同理**(注意端口是大写 `-P`):
```bash
scp -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=accept-new -P 30001 <本地文件> ${V2X_REMOTE_USER}@<PRIVATE_HOST>:${V2X_ROOT}/<路径>
```

---

## 2. ★H800 是独立 checkout(最容易踩的坑)

**H800 的 `${V2X_ROOT}` 与本地 4090 不是同一份**。经验:
- H800 **有**:`framework/measure_config.py`、`tools/structural_prune_pyramid.py`、HEAL repo、DAIR 数据、训练 ckpt。
- H800 **可能缺**:本地新建/新改的文件(如 `framework/feasibility_gate.py`、`cost_model/train/*table*.json`、你刚写的搜索脚本)。

**因此**:
1. 新写的脚本要么**自包含**(不 import 本地新模块),要么随用随 `scp` 过去。
2. 脚本依赖的输入表(training table 等)运行前**先 scp 到 H800 对应路径**(目录不存在先 `mkdir -p`)。
3. 起跑前用一次 `ls` 确认脚本 + 输入表都在 H800。

---

## 3. GPU preflight 硬规则

| 用途 | 卡要求 |
|------|--------|
| **latency / energy 实测** | **完全空闲卡独占**(`util 0%` 且 `mem ≤ 50MiB`),否则数被污染作废 |
| **AP 微调 / 训练** | 可共卡(不污染 AP);起 worker 前确认剩余显存 > 15GB |

- 可用卡:**GPU 3, 4, 5**;GPU5 常被 user01 占,跑前必查。
```bash
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i 3,4,5
```
- **实测线**只用空闲卡且串行独占;**微调线**可铺满含共卡。分线调度见 [[feedback-gpu-sharing-ap-vs-latency]]。

---

## 4. 核心调用(从底层到顶层三层)

> 所有命令都在 H800 上、`cd ${V2X_ROOT}`、用上面的 `$PY`。

### L0 — 单配置真测 latency/energy:`framework/measure_config.py`

最底层原子操作:给一个 (宽度, 精度) → export ONNX → TVM tune(复用 original60 tuned DB ~70s)→ 实测 lat(default+tuned)+ energy。

```bash
$PY framework/measure_config.py --width 16,64,256 --precision int8_tc --gpu 3
```
- `--precision {fp32,fp16,int8,int8_tc}`。
  - **`int8_tc`** = MatmulInt8Tensorization(WMMA,**公平口径**,tensorcore_gate=True)。
  - **`int8`** = native SIMT(慢 ~2×,**不公平**,勿用于精度轴对比)。
- 输出 = 末尾一段**多行 pretty JSON**:`{lat_tuned_ms, lat_default_ms, energy_j, build_success, quant_method, tensorcore_gate}`。解析要取**最后一个完整 `{...}` 块**(别只读单行)。
- 默认 input_hw=[128,256](§2 自洽口径)。

### L1 — ★主入口:联合 P×Q NSGA-II 搜索:`scripts/stage2_smbo_joint_nsga2_v1.py`

**这是出 Pareto 的正解搜索器**(v2)。单一 1029 联合空间,精度进基因,surrogate-assisted NSGA-II,内层调 L0 真测。

```bash
# reuse 模式:只用已测语料 + surrogate,瞬时出预测前沿(验证逻辑,不测新点)
$PY scripts/stage2_smbo_joint_nsga2_v1.py --measure reuse --pop 24 --budget 60 --seed 0

# real 模式:NSGA-II 提候选 → H800 空闲卡真测新宽度 → 回流重训 → 迭代(约 1–2.5h)
$PY -u scripts/stage2_smbo_joint_nsga2_v1.py --measure real --pop 24 --budget 60 \
     --k-per-gen 4 --gpu 3 --seed 0
```
- `--pop` 种群、`--budget` **新真测次数上限**、`--k-per-gen` 每代真测数、`--gpu` 空闲卡、`--seed`。
- **务必加 `-u`**(unbuffered),否则后台 stdout 缓冲、log 看着是空的(GPU 100% 才是真进度)。
- 产物在 `.../original60_quant_20260627/smbo_joint_nsga2/`:
  - `joint_nsga2_pop{P}_b{B}_{mode}_seed{S}.json` — 联合前沿 + 三极值 + baseline hypervolume 对照 + trace。
  - `newly_measured_*.json` — 本次新真测点(可回灌语料)。

> ⚠️ 旧脚本 `scripts/stage2_smbo_loop_v1.py`(v1)是**按精度拆三次跑 + 穷举**,方法已废(见 `9_7_14` §9),**别再用它当搜索主入口**,仅作历史参考。

### L2 — 前沿点补真 AP(finetune,gold 协议):prune + train_ddp + inference

前沿成员的真 AP70。优先复用 `data/stage_a_ap_real.parquet` 已有对角锚;缺的走 DAIR val n=1789 收敛 finetune。

```bash
# ① 结构剪枝(从 V2X cwd;产 config.yaml + init ckpt@23)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=${V2X_HOME}/heal_research/HEAL \
  $PY tools/structural_prune_pyramid.py \
  --orig-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29 \
  --out-dir ${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_stage2_ap_<TAG> \
  --num-filters-new 16,64,256 --width-per-group 4 --groups 32

# ② finetune(★必须从 HEAL cwd 跑,dataset 是相对路径;--half)
cd ${V2X_HOME}/heal_research/HEAL
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD $PY -m torch.distributed.launch --nproc_per_node=1 \
  --use_env --master_port=29713 opencood/tools/train_ddp.py \
  --hypes_yaml <CK>/config.yaml --model_dir <CK> --half

# ③ AP eval(PyTorch,DAIR val 1789,避 TRT 保 TVM 纪律)
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD $PY opencood/tools/inference.py \
  --model_dir <CK> --fusion_method intermediate   # 输出 AP@IOU 0.3/0.5/0.7
```
- gold 金标准 = `Pyramid_DAIR_m1_base_2023_08_14_11_42_29`(DAIR 版,勿用 OPV2V 的 `..._04_28_12`)。
- 批量 finetune 队列工具:`scripts/stage2_h800_ap_finetune_smoke_runner.py`(消费 job jsonl,`--max-workers` 并行)。

### L3 — 方法有效性验证:消融对照(非 ground-truth)

```bash
$PY framework/run_pqs_ablation.py      # 三臂 A-joint / A-serial / A-noS,Wilcoxon p 值 + HV
```
- 证明"联合搜 > 串行 > 无 S 轴";配合 L1 搜索器自带的 **random-search bootstrap hypervolume** 对照。
- **不做全枚举 ground truth**(领域惯例只在 NAS-Bench 类小 benchmark 用;真实空间比 baseline 即可)。

---

## 5. 典型端到端流程(出一条 Pareto)

```
① preflight: nvidia-smi 确认 GPU3/4 空闲;确认 H800 有脚本+训练表(缺则 scp)
② reuse 冒烟: --measure reuse 秒验搜索逻辑 + baseline HV
③ 真搜索:   nohup $PY -u ...joint_nsga2... --measure real --gpu 3 &  (~1–2.5h)
④ 监控:     GPU util(100%=在测) + 轮询输出 json 出现(stdout 有缓冲别只看 log)
⑤ 补 AP:    对前沿 off-diagonal 成员跑 L2 finetune + inference
⑥ 装配画图: 合并前沿(lat/energy 真测 + AP gold/finetune)→ 三目标非支配 → 出 csv/json/png
```

---

## 6. 常见坑(本会话真踩过)

| 坑 | 症状 | 解 |
|----|------|----|
| stdout 缓冲 | 后台 log 空但 GPU 100% | 跑 python 加 `-u` |
| H800 独立 checkout | `ModuleNotFoundError` / 找不到表 | 脚本自包含 + 先 scp 依赖 |
| train_ddp cwd 错 | `FileNotFoundError: dataset/my_dair_v2x/.../train.json` 秒崩 | train_ddp **从 HEAL cwd 跑** |
| int8 口径不公平 | int8 比 fp16 还慢 | 用 `--precision int8_tc` 非 `int8` |
| measure_config JSON | parser 拿不到 lat | 取**最后一个完整 `{...}` 块**(多行 pretty) |
| GPU 被占 | energy/lat 被污染 | 只在 util0%/mem≤50MiB 卡测;GPU5 常被 user01 占 |
| launch 脚本不查 rc | 失败却打 DONE | 每步查 `$?`,失败写 FAIL 并 exit |
| rtk hook | ssh/cat 输出被搅乱 | 关键数值用 python ast/heredoc + /usr/bin/* 读,别凭搅乱 stdout 下判断 |

---

## 7. 关键路径速查

| 项 | 路径 |
|----|------|
| gold 金标准(DAIR) | `${V2X_HOME}/heal_research/checkpoints/stage1/Pyramid_DAIR_m1_base_2023_08_14_11_42_29` |
| HEAL repo | `${V2X_HOME}/heal_research/HEAL`(train_ddp/inference 从此 cwd 跑) |
| DAIR 数据 | `HEAL/dataset/my_dair_v2x/v2x_c/cooperative-vehicle-infrastructure/{train,val}.json` |
| 训练表(surrogate 语料) | `multi_agent/data/stage2_lut_generation_v1/generated/original60_quant_20260627/cost_model/train/original60_training_table_latest.json` |
| 搜索/测量产物根 | `.../original60_quant_20260627/{smbo_joint_nsga2, smbo_loop}/` |
| gold AP 锚 | `data/stage_a_ap_real.parquet` |
| 主入口搜索器 | `scripts/stage2_smbo_joint_nsga2_v1.py` |
| 单点真测 | `framework/measure_config.py` |
| AP finetune runner | `scripts/stage2_h800_ap_finetune_smoke_runner.py` |
| 消融 | `framework/run_pqs_ablation.py` |

---

## 8. 快速冒烟(确认框架能跑)

```bash
# 本地:搜索逻辑(不碰 H800)
$PY scripts/stage2_smbo_joint_nsga2_v1.py --measure reuse --pop 24 --budget 60 --max-gen 8
# → 应打印 front_size=8 + HV(nsga2) >= HV(rand)

# H800:单点真测
$PY framework/measure_config.py --width 16,64,256 --precision int8_tc --gpu 3
# → build_success:true + lat_tuned_ms ~2.8ms
```
