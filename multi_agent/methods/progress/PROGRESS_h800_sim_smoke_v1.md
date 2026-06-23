# 本次会话进度报告 — H800 闭环仿真迁移 (裸 smoke 里程碑达成)
**时间**: 2026-06-10
**会话时长**: ~130 分钟

## 任务
读 `HANDOFF_h800_torch2_sim_migration_v1.md` → 启动 V2Xverse 闭环仿真到 H800 的迁移。
用户拍板: 4090 已不可用全迁 H800; 第一里程碑 = **裸闭环 smoke 跑通**(单 route 出 driving score, 证 CARLA(py3.7)↔感知(py3.10) IPC 架构通)。

## 已完成(全部真实测+主控独立复核)

### ① 代码/资产迁移 ✅
- V2Xverse 代码(432MB)+codriving ckpt(51MB) rsync 到 `H800:/data/jichengzhi_v2x/V2Xverse`; CARLA(16G,已在)未重传; `external_paths/carla_root`→H800 CARLA 软链已建。

### ② CARLA py3.7 客户端连通 ✅(独立复核)
- GPU2 起 headless CARLA(port3000); py3.7 client 连上(client==server ver `784d9b9f`), 同步 tick 5 帧。算力隔离(CARLA 独占 GPU2)验证。

### ③ process-B 感知 torch2 移植 ✅(独立复核 B_MODEL_OK)
- V2Xverse opencood 与 HEAL 同源 → HEAL §3 补丁 1:1 + 2 处专属(turtle 死导入; utils/__init__ 引 birdeye 绕过)。.so: iou3d/roiaware/box_overlaps 复用 HEAL, pointnet2 sm90 重编。
- `center_point_codriving` load(missing=2/unexpected=0)+planner load+forward(fused_feature[1,128,96,288])。
- ⚠️ 2 missing key `fusion_net.naive_communication.gaussian_filter` 同 ckpt 在 4090 也缺(非回归), 通信门控保真度待查。

### ④ process-B IPC 服务 ✅(独立复核 ping OK)
- carla shim(`closedloop/carla_shim/`) 让 `PnP_infer` 整脑在 py3.10 无真 carla 下 import+构建; 包进 zmq InferServer 常驻(GPU3 port5557)。ping→OK→PROCESS_B_CLIENT_OK; CUDA 计时活。

### ⑤ process-A proxy + 单 route 闭环 smoke ✅(独立读 results.json 复核)
- A 端 InferProxy(py3.7, USE_INFER_SERVER 门控替换 self.infer); 真实 sensor 经 msgpack_numpy 送 B, 收回 control 重建 carla.VehicleControl。
- **真实闭环结果(r146, norsu, 43.2s game-time)**: `score_composed=11.95, score_route=19.92, score_penalty=0.6, status=Failed`(t=13.15s 撞 vehicle.bmw.isetta→blocked)。
- **延迟实测**(InferProxy in-CARLA, measured): N=219 帧, 稳态 mean 102.4ms / p50 101.6ms / min 87.5 / max 276; 首帧 594ms(冷启)。
- 算力隔离三卡: CARLA GPU2 / B 感知 GPU3 / A leaderboard GPU4, 零 GPU 争用。

## 当前状态
- **里程碑(裸 smoke)达成**: 双进程 IPC 架构端到端跑通并出真实 DS。低分/撞车是 norsu ego-only 在难 route r146 的**预期**, 里程碑只要求"跑通出 DS"非高分。
- 所有 5 步均主控独立复核(ping/B_MODEL_OK/results.json 亲自读), 非轻信 agent 自报。

## 遗留问题 / 下一步
1. **提升 smoke 保真度**: 查 `gaussian_filter` 2 missing key 是否在推理路径(影响通信门控); 选更易 route 或开 RSU 验证非撞车正常完赛。
2. **接入 RSU 协作路**(当前 norsu): B 用带 RSU 的 config 重建, A 端 spawn_rsu 串入 payload。
3. **τ_ego/τ_RSU 时延注入移植**: 把 4090 上已建的 L1 控制器 + τ_ego + RSU 时延 sweep 移到 H800(见 `sim_ego_rsu_latency_strategy_v1.md`, `l1_architecture_problems_v1.md` —— L1 转向无视 planner→τ_ego 信号平 的架构问题仍是真正科学瓶颈, 迁移未解决它)。
4. **IPC 序列化耗时单列**: 时延实验前需把 msgpack round-trip 与 forward 计时分离(infer_server 已分离 forward)。
5. **Pyramid(项目 hero 模型)入闭环**: 当前用 codriving 感知; HEAL Pyramid 入 V2Xverse 闭环是后续(~1800 LOC, 见 framework_analysis §4)。

## 关键文件清单
| 文件路径 (H800: /data/jichengzhi_v2x/V2Xverse/) | 改动 | 状态 |
|---|---|---|
| simulation/leaderboard/team_code/closedloop/carla_shim/__init__.py | 新增 | 已验证 |
| simulation/leaderboard/team_code/closedloop/process_b_server.py | 新增 | 已验证(常驻GPU3) |
| simulation/leaderboard/team_code/closedloop/process_b_client_test.py | 新增 | 已验证 |
| team_code/closedloop/infer_server.py | 复用(已有) | 已验证 |
| simulation/leaderboard/team_code/pnp_infer_action_e2e.py | 修改(torch2安全copy/no_grad/detach/buffer_rgba; A-proxy门控) | 已验证 |
| simulation/leaderboard/team_code/pnp_agent_e2e.py | 修改(USE_INFER_SERVER 门控 self.infer→proxy) | 已验证 |
| opencood/{utils/spconv_utils,models/sub_modules/matcher_v2,matcher_v3,sparse_backbone_3d}.py | 修改(spconv2.x import) | 已验证 |
| opencood/models/fuse_modules/codriving_attn.py | 修改(删 turtle 导入) | 已验证 |
| opencood/pcdet_utils/*, utils/box_overlaps | 重编/复用 .so (py3.10/sm90) | 已验证 |
| opencood/data_utils/...(post_process/inference_utils/voxel_postprocessor/base_postprocessor) | 修改(detach/safe_copy/no_grad) | 已验证 |
| 本地 multi_agent/real_test/sim_h800_ipc_split_v1.md | 新增(架构设计+进度) | — |
| H800:/data/jichengzhi_v2x/smoke_ipc_norsu_r146_results.json | 结果产物 | 已读核 |

## 复跑命令(cheat-sheet)
```
# CARLA(GPU2): setsid bash /data/jichengzhi_v2x/h800_carla_launch.sh 2 3000 >log 2>&1 </dev/null &
# B server(GPU3): CUDA_VISIBLE_DEVICES=3 ROUTES=<xml> SAVE_PATH=<dir> python3 .../closedloop/process_b_server.py --port 5557 --gpu 3
# A smoke: USE_INFER_SERVER=1 INFER_SERVER_PORT=5557 + v2xverse py3.7 env + scripts/eval_driving_e2e.sh r146 <carla_port> <tag> 0 codriving_norsu _1
```
