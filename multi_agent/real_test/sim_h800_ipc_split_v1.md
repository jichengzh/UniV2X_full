# H800 闭环迁移 — 双进程 IPC 拆分设计 (v1, 2026-06-10)

> 背景: 4090 已不可用, 闭环仿真整体迁 H800。H800(sm90) 感知必须 torch2/py3.10, 而 CARLA 客户端只支持 py3.7 → 必须双进程 IPC 解耦。
> 本文定迁移**第一里程碑(裸闭环 smoke)**的拆分边界。母文档: `HANDOFF_h800_torch2_sim_migration_v1.md` §5; 复用 `closedloop/infer_server.py`(zmq ROUTER/DEALER + msgpack_numpy, 已建)。
> 状态: 设计已定。进度: ① 传输+CARLA客户端连通+CARLA算力隔离 ✅; ② **process-B 模型 torch2 已跑通并独立复核** ✅(`B_MODEL_OK`, GPU4: center_point_codriving load missing=2/unexpected=0, planner load, forward fused_feature[1,128,96,288]); ③ process-B IPC server(carla shim + PnP_infer 整脑) 实现中; ④ A 端 proxy + 单 route smoke 待启。

> ★ B 模型移植结论(可复现): V2Xverse opencood 的 torch2 补丁 = HEAL §3 补丁 + 2 处 V2Xverse 专属: (a) `codriving_attn.py` 删 `from turtle import update`(headless 无 tkinter); (b) `team_code/utils/__init__` 引 `carla_birdeye_view`→离线测试须绕过 __init__ 直接 importlib 加载 yaml_utils。.so: iou3d_nms/roiaware_pool3d/box_overlaps 与 HEAL 源码 md5 同→直接 copy HEAL .so; pointnet2_batch/stack 源码不同→THC 删后 sm90 重编。**2 个 missing key `fusion_net.naive_communication.gaussian_filter.{w,b}` 是同一 ckpt 在 4090 也缺(strict=False 随机初始化)→ 非迁移回归, 但闭环保真需留意通信门控行为**(待查是否在推理路径)。

---

## 一、已验证的事实 (本会话实测)
- ✅ 代码+codriving ckpt(51M) 已传 `H800:/data/jichengzhi_v2x/V2Xverse`; `external_paths/carla_root`→`/data/jichengzhi_v2x/carla` 已建。
- ✅ **CARLA py3.7 客户端在 H800 连通**: GPU2 起 CARLA headless(port3000, 渲染落 GPU2 1054MiB/100%), v2xverse env py3.7 client 连上(client_ver==server_ver 784d9b9f), 同步 tick 5 帧 OK。→ **算力隔离(CARLA 一张卡)已验证**。
- ✅ V2Xverse 的 `opencood` 与 HEAL **同源**: 同 4 个 spconv 文件 / 同 pcdet_utils(THC) / 同 box_overlaps cython → **HEAL §3 torch2 补丁 1:1 适用**, HEAL 已编译的 py3.10 .so 可作参考/直接复用。

## 二、拆分边界 (★核心决策)

闭环每 tick 的控制流 (`pnp_agent_e2e.py:run_step`):
```
A(py3.7,CARLA): tick(input_data) → ego_data[](numpy 传感器 dict + pose + measurements)
                                  → self.infer.get_action_from_list_inter(ego_data, rsu_data, step, ts) → control_all
                  非推理帧: self.infer.run_l1_step(ego_data_l1, ts) → l1_ctrl
```

**决策: 在 `self.infer`(AgentInter 整脑) 处切。整脑放 process B(torch2/GPU/py3.10), A 只留 CARLA I/O。**

依据 (实测代码):
1. `get_action_from_list_inter` 主感知路 (L724-764) = numpy 传感器 → collate → `perception_model(...)` → `post_process_multiclass` → 检测框, **carla-free**。特权 `collect_actor_data()` 已注释(L731-733) → 活体世界查询不在学习路径上。
2. 之后 检测框→BEV occupancy(大量 `.cuda()`)→`planning_model`→V2X_Controller→`carla.VehicleControl`。所有 `.cuda()` 在 B 跑(正确且快); 一帧一次往返。
3. carla 耦合仅: `carla.VehicleControl`(输出, L1013/1214)、`carla.Location/Vector3D/Rotation`(几何, L61…)、红绿灯 `self._vehicle.get_traffic_light_state()`(L1610/1746, 疑似 expert 路, 默认关) → **B 装 carla shim** 即可, 不需真 carla。

**为何不在"仅 model.forward"处切**: 感知/规划两个 forward 之间夹大量 `.cuda()` BEV/warp, 切 model 需两次往返 + 把这些 op 退到 A 的 CPU(torch1.10), 反而更乱更慢。整脑入 B 一次往返最干净。

## 三、IPC 形态 (复用 infer_server.py)
- **B = 常驻服务**: 持有 AgentInter 实例(模型 + L1 plan store + prev_control + lat_aligner 全在 B, 跨帧有状态)。zmq ROUTER。RPC 面 = `{get_action_from_list_inter, run_l1_step}`(method 名 + args)。
- **A = pnp_agent_e2e + leaderboard + CARLA**(py3.7)。把 `self.infer` 换成 IPC proxy: 转发 method+args(ego_data/rsu_data numpy 经 msgpack_numpy)→ B → 收回 control dict → A 用真 `carla.VehicleControl` 重建。
- **control 序列化**: B 返回 `{vehicle_idx: {throttle, steer, brake, ...}}` 纯 dict(B 的 VehicleControl 是 shim); A 端重建 carla.VehicleControl。
- **carla shim (B 端)**: 提供 `VehicleControl`(plain struct)、`Location/Vector3D/Rotation`(纯 python 几何)、`libcarla.TrafficLightState` 常量; `sys.modules['carla']=shim` + stub `srunner.../CarlaDataProvider`、`agents.navigation.local_planner.RoadOption` 于 import 前注入。活体世界查询若真在路径上→由 A 把所需 actor/world 状态塞进 payload(smoke 阶段默认 shim 返回空, 学习路径不依赖)。

## 四、算力隔离
CARLA 渲染 GPU_a(已验证 GPU2), perception process B GPU_b(另一张空闲卡, `CUDA_VISIBLE_DEVICES`), 互不抢 → 时延测量纯净。

## 五、实现步骤 (待 B 模型 torch2 跑通后)
1. [Task#3] B 端 codriving 感知+planner torch2 load+forward 跑通 (HEAL §3 补丁; 进行中)。
2. carla shim 模块 + import 注入; 确认 AgentInter 在 B 能 import(无真 carla)。
3. B server: 包 AgentInter 为 infer_fn, 起 InferServer(持状态)。
4. A 端: `self.infer` → IPC proxy(InferClient.submit/poll), control dict→carla.VehicleControl。
5. 单 route(town05_short r146 或更易 route) 闭环 smoke → results.json 出 DS = 里程碑验收。

## 六、诚实 caveat
- 红绿灯 `self._vehicle.get_traffic_light_state()` 是否在 MotionNet 控制路径上未最终确证(疑 expert 路)。若在→需 A 把 ego 的 traffic_light_state 塞进 payload。smoke 先 shim 返回 None 试跑, 撞到再补。
- payload 含多相机 RGB + LiDAR, 逐帧 msgpack 体积/序列化耗时未测; smoke 可接受, 时延实验前需单列 IPC 序列化耗时(infer_server 已分离 forward 计时)。
- B 整脑有跨帧状态(L1/lat_aligner) → IPC 必须保证调用顺序与单进程一致(同步 REQ/REP 即可; 异步 submit/poll 仅在引 τ_ego 流水时再用)。
