# 交接: τ_ego 延迟→V2X驾驶分曲线 — 最终实验方案 (HANDOFF_tau_curve_final_plan_v1)

> 写于 2026-06-13。承接 H800 闭环根因调查闭合(无缺陷, 详见 `multi_agent/real_test/h800_rootcause_investigation_v1.md`)。
> ★本文 = 正式跑延迟曲线的可执行方案 + 拉起的 agent + 工作流 + 出图口径。接手先读 §1 前提、§2 方案、§4 工作流。

---

## §0 一句话
在 **H800**(无缺陷, 与4090等价, 八卡空闲)上, 用 **`_1notraffic` 确定配置** + **多路线小N** 跑 τ_ego={0,100,200,300,400,500} 扫描, 出一张 **延迟→碰撞率/DS/效率** 曲线证明推理延迟伤害 V2X 驾驶。

## §1 已settled前提(不再重验, 来自根因调查)
- **H800 无框架/感知/IPC缺陷**: 感知 b-test 等价(6e-5); 同步模式下 IPC墙钟≠仿真staleness, **H800 te0 = 4090 te0 = 真零延迟基线**(两边te0=100印证)。⇒ **平台选H800纯看算力, 无正确性代价**。
- **方差源 = CARLA ambient背景traffic**(B1/B2/B3穷尽定源, 两机对称)。`_1notraffic`(vehicle_amount=0且pedestrian_amount=0)删ambient后: te0确定=100, **保留Scenario3脚本行人+RSU协同**(spawn_rsu独立, disable_rsu=False), 失败纯脚本行人碰撞。
- **延迟效应route-dependent + 单路非单调是伪象**: 失败阈值由避让余量定(r6@te219崩/r3@te500崩); 边际档高方差因避让余量~0.23m落CARLA物理非确定带。⇒ **必须多路线聚合 + 看碰撞率, 别盯单路DS**。

## §2 ★最终实验方案 (确定)
| 项 | 值 | 理由 |
|---|---|---|
| **平台** | H800 八卡(6槽并行框架现成) | 空闲+等价; 4090满 |
| **配置** | `scenario_parameter_1notraffic.yaml` | 确定baseline+保留V2X场景 |
| **路线库** | **~12条** 含Scenario3行人横穿、中等避让余量的route(冻结) | 见§3选库 |
| **延迟档** | **6档**: te0,100,200,300,400,500 (Δ_ego=0,2,4,6,8,10帧) | 看曲线形状 |
| **重复N** | **每格 N=3** | ★统计功效来自路线总体(12×3=36样本/档), 非单格N; 失败是二值, N=3够采概率 |
| **总跑量** | 12×6×3 = **216 run** (~4h, 6槽H800) | 可接受 |
| **指标** | ①碰撞率(主) ②DS mean±95%CI ③Efficiency | 见§5 |

**为什么N=3不是10**: 残余方差只在"可能撞"的边际档, 且是二值(撞/不撞)。每档碰撞率由 **12条路×3次=36样本** 估, 比单路跑10次稳一个量级。单格N只需粗采概率(0/3..3/3), 真功效在route总体。

## §3 选库标准 (executor+supervisor 跑前冻结一次)
源: `multi_agent/real_test/route_scenario_classification_v1.csv`(105路场景类型/强度)。选~12条满足:
- 含 **Scenario3 行人横穿**(延迟敏感信号最干净, 行人走KeepVelocity确定);
- **中等避让余量**: 排除 r6 类剃刀边际(~0.23m全在噪声带, 无信息)、排除 r104 类永不敏感(te500都=100);
- 跑前用 te0 `_1notraffic` ×1 快筛: **te0必须=100/0撞**(确定基线), 否则换route。
- 冻结成 `tau_curve_route_lib_v1.txt`, 之后所有档跑同一库。

## §4 工作流 + 拉起的 agent
**保留并复用现有 team `h800-rootcause` 的两个 agent + 已起的 doc-curator:**
| agent | 角色 | 本阶段任务 |
|---|---|---|
| **sim-executor** | 执行手 | 跑 216-run 矩阵(6槽并行, 每run独立重启CARLA, _1notraffic, te0-500×N3), 逐格落 results.json + 回报DS/colped |
| **rc-supervisor** | 编排+核验+聚合 | 选库冻结/盯进度/查异常/聚合每档碰撞率+DS mean±CI+Eff/防n=1误读 |
| **doc-curator** | 文档 | (本轮已在订正旧"感知退化"条; 完后并入本方法学) |
| **team-lead(主控)** | 复核+出图 | 抽查results.json/读全量数据/绘最终曲线/对用户负责 |

**步骤**: ①supervisor选库冻结(§3) → ②executor跑216矩阵(分批, 每批抄送进度) → ③主控逐批复核results.json(尤其colped归因脚本行人) → ④supervisor聚合统计 → ⑤主控/analyst出图(§5)。**纪律**: SAVE_PATH/TMPDIR在/data; bracket-trick杀进程; 1卡1组; 用过即清; 不轻信自报、复跑核验。

## §5 ★最终曲线 (出图口径)
**主图**: x=τ_ego(0→500ms), 三条y(双轴):
- **碰撞率**(撞脚本行人的run占比, 跨12路×N3): 预期随延迟**上升**(更多route越阈值) —— 这是"延迟伤安全"的主证据。
- **mean DS ± 95%CI**: 预期**下降**(注: 极延迟可能因边际概率/保守回升, 用CI诚实呈现)。
- **Efficiency**: 预期**下降**(极延迟ego卡顿, 即便没撞效率也崩 → 补足"碰撞率plateau时延迟仍伤效率"的故事)。

**附图**: 每条route的小多图(各自阈值) + 标注route-dependent。
**口径标注**(必写): 平台H800/配置_1notraffic/冻结库版本/N=3/DS来源ego_vehicle_0/results.json的score_composed/失败均colped(脚本行人)。
**脚本**: 复用 `bench2drive_metrics.py` 算Efficiency; 聚合+绘图主控用matplotlib写一次性脚本, 落 `multi_agent/figure/`。

## §6 资源/接入
- H800: `sshpass -p '12345678' ssh -p 30001 -o StrictHostKeyChecking=no jichengzhi@222.95.84.215`(密码每会话确认); repo `/data/jichengzhi_v2x/V2Xverse`。
- 双进程: A=CARLA+leaderboard(py3.7 v2xverse env, CPU); B=感知(t2lib torch2.1/cu121 sm90 GPU); zmq IPC(USE_INFER_SERVER=1)。
- 跑批框架: `/data/jichengzhi_v2x/strong_sweep_6slot.py` 改延迟档/route/config 即用; eval `scripts/eval_driving_e2e.sh <route> <port> <tag> <repeat> codriving_te{TAU}_l1 _1notraffic`。
- te配置: `pnp_config_codriving_te{0,100,200,300,400,500}_l1.yaml`(te0/108/136/219有, 缺的照模板sed生成 `tau_ego_ms`)。
- DS读: `results/.../r{N}_repeat0/ego_vehicle_0/results.json` 的 `_checkpoint.global_record.scores.score_composed`(顶层那份空)。

## §7 一句话给接手者
平台已无悬念(H800=4090, 选H800因卡空); 配置已定(_1notraffic); 真功效在**多路线小N+碰撞率**, 别再纠结单路N。冻库→跑216矩阵→复核colped→聚合→出三线图(碰撞率↑/DS↓/Eff↓)。这才是项目要的"延迟→V2X驾驶退化"干净曲线。
