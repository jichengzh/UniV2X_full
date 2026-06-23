# HANDOFF: 闭环 AP 退化旋钮 (感知精度可调) — 给 DS 接上"AP 轴" (v1, 2026-06-21)

> 自足交接。背景: 闭环 DS 此前只受 latency(τ)调控,默认了"感知精度=CoDriving 原值"。
> 本阶段新建**可调感知精度(AP)旋钮**: 像设 τ 阈值一样,主动丢检测框拉低 AP,
> 模拟剪枝/量化的精度损失 → DS 同时受 AP 与 latency 影响。**最小验证已通过**。
> 配套: [[project-tau-perc-signal]](延迟轴) / HANDOFF_latency_sim_faithfulness_v1(延迟保真度)。

---

## 0. 一句话现状
在 CoDriving 闭环里实现了 **conf 排序丢框 + 特征一致退化** 的 AP 旋钮(env/config 门控),
离线标定出 knob→AP 单调曲线(drop 0→0.9 把 veh AP50 从 0.84 拨到 0.27),
闭环 4 臂验证 **AP↗ 则 DS↘ 单调、盲驾臂崩**(因果链通)。
**全部隔离在 `/exdata/jichengzhi/V2Xverse_apknob`,仿真主框架 `/data/jichengzhi_v2x/V2Xverse` 零污染(git diff 已验空)**。

## 1. 机制 (注入点 + 算子)
**为何是感知输出端**: planner 吃的是"检测框渲染的 occupancy 栅格 + 融合 BEV 特征(`fused_feature`)"
的混合输入(end2end planner, `codriving/models/planning_end2end.py`: `x_enhanced=cat(occ, feat)`)。
AP 在 `pred_box_tensor` 上算 → 注入点 = 感知输出后、`turn_traffic_into_map` 前。

**算子** (`pnp_infer_action_e2e.py`, env `AP_DROP`/`AP_FEATMASK` 或 config `simulation.ap_drop`/`ap_featmask`):
1. 按 `pred_score` 升序丢最低 conf 的 `ap_drop` 比例框(弱目标先丢=量化真实退化模式)。
2. `ap_featmask=1`: 把被丢框的空间位置在 `fused_feature` 上一并置零。
   **★ 关键(实测厘清)**: planner 在"occ 与 feat 一致"上训练;只丢框不动特征(featmask=0)
   制造矛盾输入(occ 说没车/feat 说有车)→ 训练分布外 → **高估**危害。真剪枝模型 occ 与 feat
   同源一致退化 → **featmask=1(一致退化)才忠实**。默认 featmask=1。
3. drop≥0.999: 整 feature 置零 = 盲驾 sanity 臂。

注入默认 `ap_drop=0.0 = no-op`,不影响延迟实验。

## 2. 离线 knob→AP 标定 (veh, 200帧, CoDriving 感知)
`opencood/tools/inference_apcalib.py`(copy 自 inference_multiclass.py + env 门控同款丢框算子)。
| drop | veh AP50 | veh AP70 |
|---|---|---|
| 0.0 | 0.841 | 0.722 |
| 0.25 | 0.783 | 0.683 |
| 0.5 | 0.559 | 0.516 |
| 0.75 | 0.310 | 0.291 |
| 0.9 | 0.266 | 0.250 |
跑法: `CUDA_VISIBLE_DEVICES=N PYTHONPATH=/data/jichengzhi_v2x/t2lib:<COPY> APCALIB_TESTDIR=/exdata/jichengzhi/v2xverse_dataset/ APCALIB_NMAX=200 AP_DROP=<d> python3 opencood/tools/inference_apcalib.py --model_dir checkpoints/codriving/perception --fusion_method intermediate`

## 3. ★闭环有效性验证 (ap0-clean 交集 3 路 [3,18,104], N=2, full-traffic _1)
| 臂 | drop | featmask | ~vehAP50 | honest DS | RC | 灾难% | 均碰撞 | ΔDS |
|---|---|---|---|---|---|---|---|---|
| ap0 | 0 | — | 0.84 | 91.7 | 100 | 0 | 0.02 | — |
| ap50 | 0.5 | 1(box+feat) | 0.56 | 83.3 | 100 | 0 | 0.05 | −8.3 |
| ap55 | 0.5 | 0(box-only) | 0.56 | 67.7 | 100 | 0 | 0.14 | −24.0 |
| ap100 | 1.0 | 1(blind) | 0.27 | 37.2 | 37 | 33% | — | −54.4 |

**结论**: ① AP 调控有效—DS 随 AP 单调退化,盲驾臂 DS 崩+RC 塌+33% 灾难超时=接线正确、AP→DS 因果通。
② featmask=1 忠实(见 §1.2)。**caveat: 仅 3 路 N=2(6 reps/臂),方向/单调清楚但数值粗;
ap50 仅 −8.3(AP50 −0.28)信号偏浅 → 正式实验需扩路集+N+更细 drop 档。**

## 4. 隔离副本 — 以后所有 AP 实验都在这跑 (主框架不碰)
- **副本**: `/exdata/jichengzhi/V2Xverse_apknob`(rsync 自主框架 741M, 排除 results;含 AP 注入 agent + 4 个 ap configs + apcalib)。
- **harness**: `/exdata/jichengzhi/tau_curve_apknob.py`(VXDIR/LOGDIR/PROGLOG 已指副本)。
- **跑闭环 AP sweep**(范本, 复用 tau harness 全部断点续跑/诚实DS/超时逻辑):
  ```
  cd /exdata/jichengzhi
  CFG_TAG=ap TAUS=0,50,55,100 NREP=2 ROUTE_FILE=<routes.txt> \
    TAG_PREFIX=apval SCEN_SUFFIX=_1 GPULIST=2,3,4 SLOTS_PER_GPU=2 \
    PORT_BASE=4500 BPORT_BASE=5700 RUN_TIMEOUT=900 \
    setsid nohup python3 /exdata/jichengzhi/tau_curve_apknob.py > apval_run.log 2>&1 < /dev/null &
  ```
  config 命名 `pnp_config_codriving_ap{code}_l1.yaml`(simulation 段 `ap_drop`/`ap_featmask`);
  结果 `V2Xverse_apknob/results/results_driving_apval_ap{code}_r{R}_n{N}/...`。
- **加新 drop 档**: 仿 `/tmp/make_ap_configs.py` 在副本 agent_config 下生成新 `ap{code}` config。
- **分析**: `/tmp/ap_ds_analysis.py`(诚实DS+碰撞+接标定; 改 BASE 指副本 results)。

## 5. 下一步 (正式 AP×latency 联合曲线)
1. 扩路集(用 tp0-clean 全集 ~29 路)+ N≥3 + 更细 drop 档(0/0.1/0.2/0.3/0.5)→ 干净 AP→DS 曲线。
2. **AP×τ 二维网格**(像 ego×RSU 那张热图): 同时拨 AP 与 latency → 验证"加速换来的 AP 损失
   vs 延迟降低"的净 DS 权衡 = 软硬件协同在闭环的最终落点。
3. 用真剪枝/量化 Pyramid 的误差画像(recall-vs-距离/FP率/定位误差)校准算子 → 强忠实版。

## 6. 主框架纯净性 (已核验)
- `git status --short pnp_infer_action_e2e.py` = 空; AP configs / inference_apcalib.py 已从主框架删除。
- 主框架的 τ_perc 延迟注入(前序工作)不受影响,延迟实验照常。
