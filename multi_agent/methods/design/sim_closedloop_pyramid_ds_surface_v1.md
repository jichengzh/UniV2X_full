# Pyramid DS(AP, τ) 二维曲面实验设计 v1

> 编制: 2026-06-21, sim-integrator (T4)
> 依据: `HANDOFF_codesign_nextstage_v1.md §5+§6`、勘误 `closedloop_b4_plugin_v1.md §顶部`
> 状态: 移植 P1–P5 已完成(H800 核验通过); τ_perc sweep 配置已建; 本文规划 2D 曲面实验。

---

## §1 目标与动机

当前 DS 模型 (`closedloop_b4_plugin_v1.md §2`) 存在两个根本性问题:
1. **用的是 CoDriving 的 τ→DS 曲线**: 隐含 Pyramid≡CoDriving 假设 (不成立)。
2. **β=0 (纯延迟模型)**: 同延迟不同 AP 得同 DS，即漏检多=漏检少一个待遇 (明显错误)。

**正解**: 在 CARLA 闭环里同时扫 **感知精度(AP)** × **RSU 推理延迟(τ_perc)**，
实测 Pyramid 的驾驶分响应，建 **DS(AP, τ)** 真实联合曲面。

---

## §2 实验矩阵

### 2.1 AP 维度 (Pyramid on V2Xverse, 感知精度档)

| AP 档 | 模型配置 | V2Xverse 训练 | 目标 AP50_car (val) | 状态 |
|---|---|---|---|---|
| **AP_base** | 无剪枝 (baseline) | 已训 40ep (ckpt epoch33) | ~0.91 | ✅ 可用 |
| **AP_p50** | 50% channel pruned + finetune | ★ 待训 | ~0.85–0.90 (估) | ❌ 待做 |
| **AP_p75** | 75% channel pruned + finetune | ★ 待训 | ~0.78–0.85 (估) | ❌ 待做 |
| **AP_single** | 单车(disable_rsu=true) | 不需要重训 | 低基准 | ⏳ 可用现有ckpt |

**AP_p50/p75 训练步骤** (复用 H800 t2lib 4卡训练栈):
```bash
# 1. DepGraph 剪枝 (从 base V2Xverse ckpt)
python3 tools/prune_pyramid_v2xverse.py \
  --ckpt opencood/logs/v2xverse_pyramid_multiclass_2026_06_13_01_43_57/net_epoch_bestval_at33.pth \
  --prune_ratio 0.5 --output_dir opencood/logs/v2xverse_pyramid_p50/

# 2. V2Xverse finetune (40ep, 4卡DDP)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run \
  --nproc_per_node=4 --master_port=29516 opencood/tools/train_ddp.py \
  -y opencood/hypes_yaml/v2xverse/pyramid_multiclass_config.yaml \
  --model_dir opencood/logs/v2xverse_pyramid_p50 > p50_finetune.log 2>&1
```
预算: 40ep × ~9min/ep × 4卡 ≈ 6小时/档, 两档 ~12h 总训练。

---

### 2.2 τ 维度 (RSU perception latency 注入)

| τ 档 (ms) | 物理含义 | config 文件 (已建) | 对应 Orin e2e |
|---|---|---|---|
| **0** | 理想(无延迟基线) | `pnp_config_pyramid_sweep_d0.yaml` | — |
| **108** | 最优链(TRT FP16, E7 实测) | `pnp_config_pyramid_sweep_d108.yaml` | Orin 108ms |
| **219** | FP32 baseline (E7 实测) | `pnp_config_pyramid_sweep_d219.yaml` | Orin 219ms |
| **500** | 降级场景上界 | `pnp_config_pyramid_sweep_d500.yaml` | — |

配置文件已落盘: H800 `/exdata/jichengzhi/V2Xverse_pyramid/simulation/leaderboard/team_code/agent_config/`
本机 git: `/home/jichengzhi/V2Xverse/simulation/leaderboard/team_code/agent_config/`

**注意**: 所有 sweep 配置已 **移除 `disable_rsu: true`** (RSU 启用), 保留 `latency_inject_ms: X`。
口径标注: 所有注入延迟标 `latency_source='injected_from_E7'` (非真实在线测量)。

---

### 2.3 路线集

使用 **6 条确定性路线** (已知排除 ambient 非确定性干扰的路线):

| Route | Pyramid τ=0 DS | 选择理由 |
|---|---|---|
| r0 | 100 | 简单基线, DS=100 天花板 |
| r2 | 100 | 直路 |
| r5 | 100 | 左转 |
| r10 | 100 | Pyramid 鲁棒(CoDriving=7.4, 对照价值高) |
| r18 | 25 | 低 DS 路线, 延迟效应放大 |
| r31 | 100 | 直路补充 |

**[不使用]**: r1(DS=36 但高方差), r3(DS=50), r112(DS=60)——方差大;
r146 等: Scenario13 实例化失败, 与模型无关。

---

### 2.4 完整实验矩阵

**全量** (建议最终结论用):
```
4 AP档 × 4 τ档 × 6 route × 1 repeat = 96 runs
≈ 96 × 12min/run = 19h GPU 时间
```

**最小烟雾** (验证 RSU 可用+曲面形状, 先做):
```
2 AP档(base+p50) × 3 τ档(0,219,500) × 3 route(r0,r10,r18) × 1 rep
= 18 runs ≈ 3.5h
```

**↑ 建议先跑最小烟雾确认 RSU 启用后 Pyramid 工作正常,再决定是否上全量。**

---

## §3 关键前提核验 (实验启动前必做)

### 3.1 RSU 兼容性验证 (★最重要)

当前状态: Pyramid 仅在 `disable_rsu: true` (norsu) 下测过闭环。
τ_perc sweep 需要 RSU 启用。**需验证 Pyramid 在 RSU 模式下能否正常驱动 CARLA**。

验证命令 (H800, 单路线 r0, τ=0, RSU enabled):
```bash
# 1. 起 CARLA
bash /data/jichengzhi_v2x/h800_carla_launch.sh 5 4450

# 2. 起 process B (Pyramid + RSU enabled)
cd /exdata/jichengzhi/V2Xverse_pyramid
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/data/jichengzhi_v2x/t2lib:.:simulation/leaderboard \
  setsid nohup python3 simulation/leaderboard/team_code/closedloop/process_b_server.py \
    --port 5557 --gpu 5 \
    --pnp-config simulation/leaderboard/team_code/agent_config/pnp_config_pyramid_sweep_d0.yaml \
    > /tmp/pb_pyramid_rsu_smoke.log 2>&1 </dev/null &
# 等待 PROCESS_B_READY

# 3. 起 eval
CUDA_VISIBLE_DEVICES=5 USE_INFER_SERVER=1 INFER_SERVER_PORT=5557 \
  PATH=/data/jichengzhi_v2x/envs/v2xverse/bin:$PATH \
  bash scripts/eval_driving_e2e.sh 0 4450 pyramid_rsu_smoke 0 \
  simulation/leaderboard/team_code/agent_config/pnp_config_pyramid_sweep_d0.yaml _1
```

**通过标准**: route 完成 (不秒退), results.json 有 DS/RC 值, 无 Traceback。

**失败处理**:
- 如 Pyramid 感知在 RSU 模式下崩 → 检查 `heter_pyramid_collab_multiclass.py` 的 agent_modality_list 处理
- 如需重训(带 RSU data) → 增加训练代价约 6-8h

---

### 3.2 数据前提

- `disable_rsu: true` 去掉后, V2Xverse 会在仿真中生成真实 RSU
- Pyramid 感知模型以 `max_cav: 5` 训练, 应能处理 RSU 作为额外 CAV
- 如 RSU 位置 (rsu_distance=12, rsu_height=7.5) 导致感知崩溃 → 尝试减少 RSU 数量或距离

---

## §4 执行时序 (准确)

```
[NOW 完成] 创建 pnp_config_pyramid_sweep_d{0,108,219,500}.yaml  ✅
[~1h] 核验 RSU smoke (H800, r0, τ=0)  ← 下一步
  └── 若通: 进入 §4.1
  └── 若崩: 调试 RSU 兼容 (~4-8h) → 可能重训
[~12h] 训练 Pyramid p50 + p75 (H800, 4卡)
[~20h] 全量 τ×route sweep (AP_base × 4τ × 6route)
[~20h] AP_p50/p75 τ×route sweep
[~2h] 拟合 DS(AP, τ) 曲面 + 出图
```

**总工期(乐观)**: 约 5-7 天 (含训练与 sweep 运行)
**总工期(有障碍)**: 10-14 天 (RSU 不兼容需调试+重训)

---

## §5 DS(AP, τ) 曲面拟合方法

**目标**: 从 `(AP_i, τ_j) → DS_ij` 离散测点插值出连续曲面。

**推荐方法**:
1. **双线性插值** (简单, 论文可用): `DS = a + b*AP + c*τ + d*AP*τ`
2. **RBF 插值** (平滑, 4×4=16点足够): `scipy.interpolate.RBFInterpolator`
3. **GP 回归** (带置信区间): 若 repeat≥3 则用 GPyTorch

**关键轴**:
- X 轴: τ_perc (ms), 0–500
- Y 轴: AP50_car (V2Xverse val), 0.78–0.91
- Z 轴: DS (driving score), 0–100
- 辅助: collision_rate

**替换目标**: 替换 `closedloop_b4_plugin_v1.md §2` 的线性 DS(τ) 估算模型。
真曲面建立后, B4/B5 的 DS 数字从 "model-estimated (CoDriving)" → "真实 Pyramid 闭环实测"。

---

## §6 指标入库规划

新增列 (需与 data-orchestrator 议定 schema):
```
closedloop_route_id (str)
closedloop_repeat_idx (int)
perc_latency_ms (float)  # 注入值
perc_latency_source (str)  # 'injected_from_E7'
ds_closedloop (float)
rc_closedloop (float)
collision_flag (int)  # 0/1 per route
v2xverse_rsu_enabled (bool)
model_prune_ratio (float)  # 0/0.5/0.75
pyramid_ap50_v2x_val (float)  # 感知精度实测
```

**并入规则**: 闭环指标存 `results/closedloop_pyramid_tau_sweep_v1.csv`,
由 data-orchestrator 审格式后 JOIN 到 dataset_v2 (以 model_config + τ 为 key)。
**不擅自改主表列** —— 先与 data-orchestrator 议定再动。

---

## §7 与 B4 的关联

B4 当前 DS 数字 (trap25 DS≈85.6 vs pad64 DS≈95.6) 是 **model-estimated (CoDriving τ→DS)**。
当本曲面建成后:
1. 查 Pyramid(AP=0.590, τ=535ms) → 得真实 DS_trap25
2. 查 Pyramid(AP=0.590, τ=215ms) → 得真实 DS_pad64
3. 若真实曲面仍显示 DS_pad64 ≫ DS_trap25 → B4 结论得真实支持
4. 若真实曲面区分度不同 → 修正 DS 结论

**过渡期**: 在曲面建成前, B4 DS 轴标注 "model-estimated" 不可信跨 AP 比较。
B4 主结论 (HV/Wilcoxon, 对内 DS) 不依赖 DS 曲面, 仍有效。
