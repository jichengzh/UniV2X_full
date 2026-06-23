"""把 complete_points (latency+AP, 单值精度) 与 perstage_AP_v2 (per-stage 精度 AP)
映射进统一 schema_v2, 输出 multi_agent/data/dataset_v2.{csv,parquet}。

设计: 扁平列只放当前在测维度; 跨模型/更多模块用 config_json 结构化引用承载, 不预留空列。
每行标 latency_kind / ap_valid / is_real_measured / source, 不强行 join 不同测量。
"""
import json
import pandas as pd
from pathlib import Path

# ★ 2026-06-12: UniV2X 已退化为断裂符号链接空壳, 真实仓库=/home/jichengzhi/V2X (见 memory project-real-repo-is-v2x)。
# 所有输入(results/、data/、multi_agent/data/sources/)与输出均在 V2X 下。
ROOT = Path("/home/jichengzhi/V2X")
COLS = [
    # 标识
    "config_id", "model_class", "dataset_src", "config_label", "triplet",
    # 剪枝 B1
    "prune_rate", "prune_object", "prune_criterion",
    "stage0_planes", "stage1_planes", "stage2_planes",
    "deblocks_keep", "shrink_keep", "params_total", "finetune_epochs", "ckpt_status",
    # 量化 B2
    "stage0_prec", "stage1_prec", "stage2_prec", "q_mode",
    "q_granularity", "q_object", "calibrator",
    # 硬件 D
    "hardware", "d_scheme", "d_tactic", "d_workspace_gb",
    # 标签 latency
    "latency_kind", "lat_p50_ms", "lat_p99_ms", "lat_mean_ms",
    # 标签 throughput (★非冗余维度: kind 区分口径, batch>1/pipelined 才脱离 1/lat)
    "throughput_fps", "throughput_kind", "batch",
    # 标签 accuracy
    "ap30", "ap50", "ap70", "n_ap_samples", "ap_pipeline",
    "ap_baseline_ref", "delta_ap50_vs_baseline",
    # TP 几何误差项 (supervisor ISS-020 PASS: 剪枝轴SNR=8.5×信号成立; INT8无信号SNR=0.45×)
    # 定位: 预测器约束信号(标regime), 非成本Pareto目标轴
    # mATE: 米; mASE: 0-1; mAOE: 弧度[0,π/2]
    # mAOE 全集点估计(1789帧) + CI(N=350共同子集bootstrap 95%, 方案C)
    "mATE", "mASE", "mAOE", "mAOE_ci_lo", "mAOE_ci_hi", "mAOE_n_tp",
    # mAOE_basis: measured(本config逐engine直接评估) / exact_reuse(同planes+prec EXACT复用)
    "mAOE_basis", "metric_pipeline",
    # 标签 energy (★Pareto 第 5 轴; NVML board 实测, 仅 E4 有)
    "mean_power_w", "energy_per_frame_mj", "perf_per_watt_fps_per_w",
    # 标签 其它
    "engine_size_mb", "fp16_layer_count", "int8_layer_count", "build_secs",
    # 派生诊断: 有效吞吐 GMAC/ms + nominal GMAC (hw 实算, 非主指标) + E 难度分箱 AP70
    "eff_gmac_per_ms", "nominal_gmac", "ap70_by_range",
    # 有效性/溯源
    "is_real_measured", "ap_valid", "regime", "ap_reuse_basis", "source", "exec_path", "ts",
    "onnx_path", "engine_path", "config_json",
    # ---- 闭环驾驶指标 (Sim-D, cl_ 前缀; model_class=codriving_v2xverse, regime=sim_closedloop) ----
    # ★dataset_v2(全表)= 最全面统计源, 保留全部闭环列 (8 元数据 + 15 cl_ = 23 列)。
    # ★与感知 AP/latency/energy 不同任务口径, 严禁混入同列/同曲线/同 Pareto 图。
    # 现为空列 (CL_INGEST=False); 时延补全 + GO 门达标后填 (GO 门见 real_test/sim_test_design_v1.md §5)。
    # 精简学习视图 dataset_v2_learning 闭环只取 DS+RC(+latency_inject_ms 作τ特征)且剔 norsu 行, 见 LEARN_DROP。
    # 元数据 (8):
    "latency_inject_ms", "latency_ms_source", "isolation",
    "sim_route_id", "sim_arm", "sim_route_set", "n_repeat", "rsu_enabled",
    # 性能 (15 cl_): DS=driving_score=route_completion×infraction_penalty×100
    "cl_driving_score", "cl_route_completion", "cl_infraction_penalty",
    "cl_collision_ped", "cl_collision_veh", "cl_collision_layout",
    "cl_red_light", "cl_outside_lanes",
    # 机制健康度探针 (ZOH 时延回灌): norsu 行这些 = NaN
    "cl_zoh_age_mean", "cl_delta_frames", "cl_zoh_held_frac",
    "cl_audit_frames", "cl_duration_system_s", "cl_route_length_m", "cl_status",
    # 末尾备注
    "notes",
]

# ★ 精简学习视图 (dataset_v2_learning): 每个评价维度只留一个规范值, 供后续预测器/选择器学习。
# 全表 dataset_v2 保留全部冗余列不动; 学习视图 = 全表 - 下列 8 个冗余指标重复列。
# 保留: ap70 / lat_p50_ms(+latency_kind) / throughput_fps(+throughput_kind) /
#       energy_per_frame_mj / engine_size_mb / mATE,mASE,mAOE(约束信号,非主目标轴) / cl_* 闭环列。
LEARN_DROP = [
    # —— 开环 5 指标: 每维度只留一个规范值 ——
    "ap30", "ap50",                                  # 精度 -> 留 ap70 (信号最强, 见 schema §简化)
    "lat_p99_ms", "lat_mean_ms",                     # 延迟 -> 留 lat_p50_ms (满覆盖+抗尾部)
    "mean_power_w", "perf_per_watt_fps_per_w",       # 能耗 -> 留 energy_per_frame_mj (每帧真成本)
    "params_total",                                  # 体积 -> 留 engine_size_mb (真部署约束+反映量化)
    "delta_ap50_vs_baseline",                        # ap50 已删, 该派生 delta 失去基准, 弃 (预测器学绝对 ap70)
    # —— 闭环: 学习视图只留 DS + cl_route_completion (用户拍板) + latency_inject_ms(τ特征) ——
    # 其余闭环列 (次要违规/碰撞分解/ZOH探针/诊断/冗余元数据) 仅留全表 dataset_v2; norsu 行另在下方按行剔除。
    "latency_ms_source", "isolation", "sim_route_id", "sim_arm", "sim_route_set", "n_repeat", "rsu_enabled",
    "cl_infraction_penalty", "cl_collision_ped", "cl_collision_veh", "cl_collision_layout",
    "cl_red_light", "cl_outside_lanes",
    "cl_zoh_age_mean", "cl_delta_frames", "cl_zoh_held_frac",
    "cl_audit_frames", "cl_duration_system_s", "cl_route_length_m", "cl_status",
]

def q_mode_of(p0, p1, p2, label):
    if "automix" in str(label):
        return "auto"
    if "forced" in str(label):
        return "uniform_forced"
    return "uniform" if (p0 == p1 == p2) else "per_stage_mixed"

def cfg_json(r):
    return json.dumps({
        "prune": {"stage_planes": [r["stage0_planes"], r["stage1_planes"], r["stage2_planes"]],
                  "deblocks_keep": r["deblocks_keep"], "shrink_keep": r["shrink_keep"],
                  "object": r["prune_object"], "criterion": r["prune_criterion"]},
        "quant": {"stage_prec": [r["stage0_prec"], r["stage1_prec"], r["stage2_prec"]],
                  "granularity": r["q_granularity"], "object": r["q_object"], "calibrator": r["calibrator"]},
        "hw": {"hardware": r["hardware"], "d_scheme": r["d_scheme"]},
    }, ensure_ascii=False)

# ---------- collab2 口径统一 (任务 A) ----------
# 源 1 的 6 个 anchor 原本是单 agent body_subnet 口径 (input (1,64,128,256)),
# 与源 2 的 27 个 body_subnet_collab2 (双 agent (2,64,128,256)+t_ego+fusion)
# 不可比, 无法同放一条 Pareto。这里把这 6 个统一改用 collab2 uniform 延迟:
#   - 5 个直接复用 results/perstage_quant_latency_global_v2.csv 的
#     global_fp16/int8_automix (TRT-auto uniform, collab2);
#   - p25 (48,96,192) fp16 不在原 perstage triplet 里, 由
#     scripts/phase1/m4_9_bench_p25_collab2.py 新测 (同 stage_a_cache 引擎,
#     CUDA-Event warmup200/measure200, GPU6 idle), 写回同一 global CSV。
# AP / 结构 / engine_size 仍沿用源 1 的真测值 (单 agent body 与 collab2
# 同一 backbone, 检测精度一致); 仅 latency 列换成 collab2。
_gl_collab2 = pd.read_csv(ROOT / "results/perstage_quant_latency_global_v2.csv")
# key: (latency-triplet, fp16/int8) -> collab2 latency row
_GL_LOOKUP = {}
for _, gr in _gl_collab2.iterrows():
    prec_tag = "INT8" if "int8" in str(gr["config_label"]) else "FP16"
    _GL_LOOKUP[(gr["triplet"], prec_tag)] = gr
# anchor triplet (源 1) -> latency-triplet (collab2 global) 映射
_ANCHOR_LAT_TRIP = {
    "T1_base": "base", "T2_p25": "pruned25",
    "T4_p50": "pruned50", "T6_p75": "pruned75",
}

rows = []

# ---------- 源 1: complete_points (AP + 结构真测; latency 换 collab2) ----------
cp = pd.read_csv(ROOT / "multi_agent/data/sources/complete_points_pyramid_v1.csv")
for _, r in cp.iterrows():
    prec = str(r["q_bits"]).upper()
    pruned = float(r["prune_rate"]) > 0
    # 取 collab2 uniform latency (复用 global CSV; p25 为新测)
    lat_trip = _ANCHOR_LAT_TRIP.get(r["triplet"])
    gr = _GL_LOOKUP.get((lat_trip, prec))
    if gr is None:
        raise ValueError(
            f"no collab2 latency for anchor {r['anchor_id']} "
            f"(triplet={r['triplet']}->{lat_trip}, prec={prec})")
    is_new = "new-measure" in str(gr["source"])
    lat_note = (f"latency=body_subnet_collab2 ({'NEW real-measure' if is_new else 'reuse collab2 global'}): "
                f"{gr['config_label']}/{gr['triplet']}; "
                f"idle_gpu[{gr['gpu_idle_verified']}]; "
                f"warmup{gr['n_warmup']}/measure{gr['n_measure']}; input={gr['input_shape']}; "
                f"AP/结构沿用源1单agent真测 (同 backbone, 精度一致)")
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{int(r.stage0_planes)}-{int(r.stage1_planes)}-{int(r.stage2_planes)}_{prec}",
        "model_class": "pyramid_fusion", "dataset_src": "complete_points_v1",
        "config_label": r["anchor_id"], "triplet": r["triplet"],
        "prune_rate": r["prune_rate"], "prune_object": r["prune_object"],
        "prune_criterion": "l1_norm",
        "stage0_planes": r.stage0_planes, "stage1_planes": r.stage1_planes, "stage2_planes": r.stage2_planes,
        "deblocks_keep": 1.0, "shrink_keep": 1.0, "params_total": r["params_total_new"],
        "finetune_epochs": 23 if pruned else None,
        "ckpt_status": "pruned_finetuned" if pruned else "pretrained",
        "stage0_prec": prec, "stage1_prec": prec, "stage2_prec": prec, "q_mode": "uniform",
        "q_granularity": r["q_granularity"], "q_object": r["q_object"], "calibrator": r["calibrator"],
        "hardware": r["hardware"], "d_scheme": r["d_scheme"], "d_tactic": r["d_tactic"],
        "d_workspace_gb": r["d_workspace_gb"],
        "latency_kind": "body_subnet_collab2",
        "lat_p50_ms": gr["lat_p50_ms"], "lat_p99_ms": gr["lat_p99_ms"],
        "lat_mean_ms": gr["lat_mean_ms"], "throughput_fps": gr["throughput_fps"],
        "ap30": r["ap30"], "ap50": r["ap50"], "ap70": r["ap70"],
        "n_ap_samples": 1789, "ap_pipeline": "DAIR_val_1789_TRT",
        "ap_baseline_ref": None, "delta_ap50_vs_baseline": None,
        "engine_size_mb": gr["engine_size_mb"], "int8_layer_count": None, "fp16_layer_count": None,
        "build_secs": r["build_secs"],
        "is_real_measured": True, "ap_valid": True,
        "source": (f"{r['source']} | latency:{gr['source']}"),
        "exec_path": r["exec_path"],
        "ts": r["ts"], "onnx_path": r["onnx_path"],
        "engine_path": gr["engine_path"], "notes": lat_note,
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 2: perstage_AP_v2 (per-stage 精度, AP-only) ----------
PRUNE_RATE = {"T_baseline": 0.0, "T_prune50p": 0.5, "T_prune75": 0.75}
# AP triplet 名 -> latency triplet 名映射
LAT_TRIP = {"T_baseline": "base", "T_prune50p": "pruned50", "T_prune75": "pruned75"}

# 实测 latency (clean idle GPU, CUDA-Event, warmup200/measure200).
# ★ 口径: collab 2-agent (spatial_features (2,64,128,256)+t_ego)+fusion ->
#   latency_kind=body_subnet_collab2, 与 complete_points 的 (1,64,128,256) 单 agent
#   不同列可比; perstage 集合内部彼此可比。
# 21 个 per-stage 混精/forced 引擎来自 perstage_quant_ap_cache_v2;
# 6 个 global automix (TRT-auto uniform) 引擎交叉引用自 stage_a_cache。
_lat_ps = pd.read_csv(ROOT / "results/perstage_quant_latency_v2.csv")
_lat_gl = pd.read_csv(ROOT / "results/perstage_quant_latency_global_v2.csv")
_lat = pd.concat([_lat_ps, _lat_gl], ignore_index=True)
# key: (latency-triplet, config_label)
LAT_LOOKUP = {(r["triplet"], r["config_label"]): r for _, r in _lat.iterrows()}

ps = pd.read_csv(ROOT / "results/perstage_quant_AP_real_v2.csv")
for _, r in ps.iterrows():
    p0, p1, p2 = r["stage0_prec"], r["stage1_prec"], r["stage2_prec"]
    pr = PRUNE_RATE.get(r["triplet"], None)
    pruned = (pr or 0) > 0
    fp16c = intc = None
    s = str(r.get("fp16_int8_summary", "") or "")
    if "FP16" in s and "INT8" in s:  # 例: "93 FP16 / 61 INT8"
        try:
            fp16c = int(s.split("FP16")[0].strip())
            intc = int(s.split("/")[1].strip().split("INT8")[0].strip())
        except Exception:
            pass
    # LEFT-JOIN 实测 latency (按 latency-triplet + config_label)
    lat_trip = LAT_TRIP.get(r["triplet"])
    lr = LAT_LOOKUP.get((lat_trip, r["config_label"]))
    if lr is not None:
        lat_kind = lr["latency_kind"]
        lat_p50 = lr["lat_p50_ms"]; lat_p99 = lr["lat_p99_ms"]
        lat_mean = lr["lat_mean_ms"]; lat_fps = lr["throughput_fps"]
        eng_size = lr["engine_size_mb"]; eng_path = lr["engine_path"]
        lat_src = lr["source"]
        lat_note = (f"latency: {lat_kind}; idle_gpu[{lr['gpu_idle_verified']}]; "
                    f"warmup{lr['n_warmup']}/measure{lr['n_measure']}; "
                    f"input={lr['input_shape']}; {str(lr.get('notes','') or '')}")
    else:
        lat_kind = "NA_pending_idle_gpu"
        lat_p50 = lat_p99 = lat_mean = lat_fps = None
        eng_size = eng_path = None
        lat_src = None
        lat_note = "AP-only; latency 待 clean-GPU 实测 (无缓存引擎)"

    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{int(r.stage0_planes)}-{int(r.stage1_planes)}-{int(r.stage2_planes)}_{p0[0]}{p1[0]}{p2[0]}",
        "model_class": "pyramid_fusion", "dataset_src": "perstage_AP_v2",
        "config_label": r["config_label"], "triplet": r["triplet"],
        "prune_rate": pr, "prune_object": "channel" if pruned else "none",
        "prune_criterion": "l1_norm" if pruned else None,
        "stage0_planes": r.stage0_planes, "stage1_planes": r.stage1_planes, "stage2_planes": r.stage2_planes,
        "deblocks_keep": 1.0, "shrink_keep": 1.0, "params_total": None,
        "finetune_epochs": 23 if pruned else None,
        "ckpt_status": "pruned_finetuned" if pruned else "pretrained",
        "stage0_prec": p0, "stage1_prec": p1, "stage2_prec": p2,
        "q_mode": q_mode_of(p0, p1, p2, r["config_label"]),
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A", "calibrator": "minmax",
        "hardware": "rtx4090", "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "latency_kind": lat_kind,
        "lat_p50_ms": lat_p50, "lat_p99_ms": lat_p99, "lat_mean_ms": lat_mean,
        "throughput_fps": lat_fps,
        "ap30": r["ap30"], "ap50": r["ap50"], "ap70": r["ap70"],
        "n_ap_samples": r["n_samples"], "ap_pipeline": "DAIR_val_1789_TRT",
        "ap_baseline_ref": "forced_all_int8_same_pipeline",
        "delta_ap50_vs_baseline": r["delta_ap50_vs_forced_int8"],
        "engine_size_mb": eng_size, "fp16_layer_count": fp16c, "int8_layer_count": intc,
        "build_secs": None,
        "is_real_measured": True, "ap_valid": True,
        "source": (str(r["source"]) + (f" | {lat_src}" if lat_src else "")),
        "exec_path": "quant_config_mixed",
        "ts": None, "onnx_path": None, "engine_path": eng_path,
        "notes": lat_note,
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 5: P0-1 p25(48/96/192) INT8 耦合点 (hw collab2 lat/energy + AP) ----------
# hw Task#2 真测 results/P0_1_p25_int8_trap_4090.csv (GPU7 全空闲, body_subnet_collab2 口径) 3 配置。
# 填平 B1×B2 耦合矩阵唯一无 INT8 的 prune0.25 档:
#   - p25_int8_automix = 真前沿完整点(regime=front): AP 取 stage_a pruned25 int8 金标准
#     (同引擎 stage_a_cache/pruned25_int8.engine -> AP/lat 同 config, 铁实完整点);
#   - p25_c_all_int8_forced = 消融点(regime=ablation_guardrail_off, ISS-013 被支配点):
#     AP 取 results/p0_1_p25_forced_int8_ap.json (同 forced engine), 仅预测器训练+消融, 不进前沿;
#   - p25_fp16_rebaseline: 不新增行(源1已有 p25 FP16), 仅用其 GPU7-clean latency 覆盖源1旧值(2.946->2.905)。
_p0_trap = {r["config_label"]: r for _, r in
            pd.read_csv(ROOT / "results/P0_1_p25_int8_trap_4090.csv").iterrows()}
_sa = pd.read_csv(ROOT / "data/stage_a_ap_real.csv")
_sa_p25_i8 = _sa[(_sa.anchor == "pruned25") & (_sa.precision == "int8")].iloc[0]
_fj_p25 = json.load(open(ROOT / "results/p0_1_p25_forced_int8_ap.json"))

def _p25_int8_row(cfg_label, qmode, regime, ap, ap_src, note):
    hw = _p0_trap[cfg_label]
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_48-96-192_III_{qmode}",
        "model_class": "pyramid_fusion", "dataset_src": "P0_1_p25_trap",
        "config_label": cfg_label, "triplet": "T2_p25",
        "prune_rate": 0.25, "prune_object": "channel", "prune_criterion": "l1_norm",
        "stage0_planes": 48, "stage1_planes": 96, "stage2_planes": 192,
        "deblocks_keep": 1.0, "shrink_keep": 1.0, "params_total": None,
        "finetune_epochs": 23, "ckpt_status": "pruned_finetuned",
        "stage0_prec": "INT8", "stage1_prec": "INT8", "stage2_prec": "INT8", "q_mode": qmode,
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A", "calibrator": "minmax",
        "hardware": "rtx4090", "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "regime": regime,
        "latency_kind": "body_subnet_collab2",
        "lat_p50_ms": hw["lat_p50_ms"], "lat_p99_ms": hw["lat_p99_ms"], "lat_mean_ms": hw["lat_mean_ms"],
        "throughput_fps": hw["throughput_fps"],
        "ap30": float(ap[0]), "ap50": float(ap[1]), "ap70": float(ap[2]),
        "n_ap_samples": 1789, "ap_pipeline": "DAIR_val_1789_TRT",
        "ap_baseline_ref": None, "delta_ap50_vs_baseline": None,
        "mean_power_w": hw["mean_power_w"], "energy_per_frame_mj": hw["energy_per_frame_mj"],
        "perf_per_watt_fps_per_w": hw["perf_per_watt_fps_per_w"],
        "engine_size_mb": hw["engine_size_mb"], "build_secs": None,
        "is_real_measured": True, "ap_valid": True,
        "source": f"lat+energy:P0_1_p25_int8_trap_4090.csv(hw GPU7 collab2 NVML-idle-excl); ap:{ap_src}",
        "exec_path": "trt_engine_collab2_nvml", "ts": None, "onnx_path": None,
        "engine_path": hw["engine_path"], "notes": note,
    })
    row["config_json"] = cfg_json(row)
    return row

rows.append(_p25_int8_row(
    "p25_int8_automix", "auto", "front",
    (_sa_p25_i8.ap30, _sa_p25_i8.ap50, _sa_p25_i8.ap70),
    "stage_a_ap_real(gold; 同引擎 stage_a_cache/pruned25_int8.engine)",
    "P0-1 耦合矩阵补格: prune0.25×INT8(TRT-auto) 完整点. AP(stage_a 金标准)+lat/energy(hw collab2 GPU7) 同引擎. "
    "陷阱: INT8 加速仅 1.06x(对齐档 1.25-1.57x), 非对齐 48 通道 kernel-selection cliff(profile 实证)."))
rows.append(_p25_int8_row(
    "p25_c_all_int8_forced", "uniform_forced", "ablation_guardrail_off",
    (_fj_p25["ap30"], _fj_p25["ap50"], _fj_p25["ap70"]),
    "p0_1_p25_forced_int8_ap.json(同 forced engine, 0FP16/155INT8)",
    "P0-1 消融(ISS-013): forced-all-int8 是 Pareto 被支配点(auto-int8 双轴支配它), 仅预测器训练+护栏消融, 不进前沿. "
    "forced lat 4.339ms=0.67x(比自身 FP16 还慢)."))

# ---------- 源 6: P0-2 [32,64,136] 完整点 (非对齐 stage2=136 但无 cliff) ----------
# 1×1-conv 主导的非 32 倍数宽度: INT8 1.34x (在对齐档 1.25-1.57x 内, 无 kernel-cliff)
# -> 反证耦合陷阱 scope 限 grouped-conv(3x3 g32); 1×1 conv 非对齐不触发。
# AP 真测 results/p0_2_136_ap.json (逐 engine, 4dp); lat+energy results/P0_2_136_hw.csv (collab2 GPU1 NVML)。
_p136_ap = {r["prec"]: r for r in json.load(open(ROOT / "results/p0_2_136_ap.json"))["rows"]}
_p136_hw = {str(r["precision"]).upper(): r for _, r in
            pd.read_csv(ROOT / "results/P0_2_136_hw.csv").iterrows()}
for _prec in ("FP16", "INT8"):
    _ap = _p136_ap[_prec.lower()]
    _hw = _p136_hw[_prec]
    _lat = float(_hw["lat_p50_ms"]); _pw = float(_hw["mean_power_w"])
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_32-64-136_{_prec}",
        "model_class": "pyramid_fusion", "dataset_src": "P0_2_136",
        "config_label": f"p50b2_136_{_prec.lower()}", "triplet": "T_prune50b2_136",
        "prune_rate": 0.5, "prune_object": "channel", "prune_criterion": "l1_norm",
        "stage0_planes": 32, "stage1_planes": 64, "stage2_planes": 136,
        "deblocks_keep": 1.0, "shrink_keep": 1.0, "params_total": None,
        "finetune_epochs": 35, "ckpt_status": "pruned_finetuned",
        "stage0_prec": _prec, "stage1_prec": _prec, "stage2_prec": _prec, "q_mode": "uniform",
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A", "calibrator": "minmax",
        "hardware": "rtx4090", "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "regime": "front",
        "latency_kind": "body_subnet_collab2",
        "lat_p50_ms": _lat, "lat_p99_ms": _hw["lat_p99_ms"], "lat_mean_ms": None,
        "throughput_fps": round(1000.0 / _lat, 1),
        "ap30": _ap["ap30"], "ap50": _ap["ap50"], "ap70": _ap["ap70"],
        "n_ap_samples": 1789, "ap_pipeline": "DAIR_val_1789_TRT",
        "mean_power_w": _pw, "energy_per_frame_mj": _hw["energy_per_frame_mj"],
        "perf_per_watt_fps_per_w": round(1000.0 / _lat / _pw, 3),
        "eff_gmac_per_ms": _hw["eff_gmac_per_ms"], "nominal_gmac": _hw["nominal_gmac"],
        "engine_size_mb": _hw["engine_size_mb"], "build_secs": None,
        "is_real_measured": True, "ap_valid": True,
        "source": "ap:p0_2_136_ap.json(逐engine 4dp); lat+energy:P0_2_136_hw.csv(collab2 GPU1 NVML-idle-excl)",
        "exec_path": "trt_engine_collab2_nvml", "ts": None, "onnx_path": None,
        "engine_path": _ap.get("engine"),
        "notes": (f"P0-2 非对齐宽度完整点: stage2=136 非32倍数。INT8 加速 1.34x(对齐档 1.25-1.57x 内, "
                  "无 kernel-cliff)→ 反证耦合陷阱 scope 限 grouped-conv(3x3 g32), 1×1 conv 非对齐不触发。"),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 7: A 剪枝×forced-INT8 消融 (regime=ablation_guardrail_off, 非前沿) ----------
# 深剪 pyramid_backbone (cliff系) × forced-all-int8 (护栏 OFF)。
# AP 4dp 真测 pathA_forced_int8_ap.json; lat+energy P_ab_hw_bench.csv (collab2 GPU7)。
# 纪律: cliff系≠backbone系不拼曲线; 禁与 2dp fp16-ref 做 delta (2dp-vs-4dp 假象)。
_pab_hw = {r["config"]: r for _, r in pd.read_csv(ROOT / "results/P_ab_hw_bench.csv").iterrows()}
_pa_tiers = json.load(open(ROOT / "results/pathA_forced_int8_ap.json"))["tiers"]
_PA_HWKEY = {"cliff2_c": "cliff2_c_forced_int8", "prune90": "prune90_forced_int8",
             "prune95": "prune95_forced_int8"}
for t in _pa_tiers:
    if t.get("status") != "ok":
        continue
    pl = t["planes"]; hw = _pab_hw[_PA_HWKEY[t["tag"]]]
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{pl[0]}-{pl[1]}-{pl[2]}_III_forcedA",
        "model_class": "pyramid_fusion", "dataset_src": "pathA_forced_int8",
        "config_label": f"A_{t['tag']}_forced_int8", "triplet": f"A_{t['tag']}",
        "prune_rate": None, "prune_object": "channel_pyramid_backbone", "prune_criterion": "l1_norm",
        "stage0_planes": pl[0], "stage1_planes": pl[1], "stage2_planes": pl[2],
        "deblocks_keep": 1.0, "shrink_keep": 1.0, "params_total": t.get("pb_params_M"),
        "finetune_epochs": 40, "ckpt_status": "pruned_finetuned",
        "stage0_prec": "INT8", "stage1_prec": "INT8", "stage2_prec": "INT8", "q_mode": "uniform_forced",
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A", "calibrator": "minmax",
        "hardware": "rtx4090", "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "regime": "ablation_guardrail_off",
        "latency_kind": "body_subnet_collab2",
        "lat_p50_ms": hw["lat_p50_ms"], "lat_p99_ms": hw["lat_p99_ms"], "lat_mean_ms": hw["lat_mean_ms"],
        "throughput_fps": hw["throughput_fps"],
        "ap30": t["ap30"], "ap50": t["ap50"], "ap70": t["ap70"],
        "n_ap_samples": 1789, "ap_pipeline": "DAIR_val_1789_TRT",
        "mean_power_w": hw["mean_power_w"], "energy_per_frame_mj": hw["energy_per_frame_mj"],
        "perf_per_watt_fps_per_w": hw["perf_per_watt_fps_per_w"],
        "eff_gmac_per_ms": hw["eff_gmac_per_ms"], "nominal_gmac": hw["nominal_gmac"],
        "engine_size_mb": hw["engine_size_mb"],
        "is_real_measured": True, "ap_valid": True,
        "source": f"ap:pathA_forced_int8_ap.json({t.get('fp16_int8_summary')}); lat+energy:P_ab_hw_bench.csv(collab2 GPU7)",
        "exec_path": "trt_engine_forced_int8", "engine_path": hw["engine_path"],
        "notes": (f"A 消融: forced-all-int8(护栏OFF) on 深剪 pyramid_backbone({t['tag']}, pb={t.get('pb_params_M')}M)。"
                  "regime=ablation_guardrail_off 非前沿; cliff系≠backbone系不拼曲线; 不与 2dp fp16-ref 做 delta。"),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 8: B 护栏消融 head-INT8/rest-FP16 (regime=ablation; 不崩+不省延迟) ----------
# detection heads 强制 INT8, 其余 stage 保 FP16。AP pathB_head_int8_ablation.json; lat+energy P_ab_hw_bench.csv。
_pb_tiers = json.load(open(ROOT / "results/pathB_head_int8_ablation.json"))["tiers"]
_PB_HWKEY = {"base": "base_headINT8_restFP16", "pruned50": "p50_headINT8_restFP16"}
_PB_PR = {"base": 0.0, "pruned50": 0.5}
for t in _pb_tiers:
    if t.get("status") != "ok":
        continue
    pl = t["planes"]; hw = _pab_hw[_PB_HWKEY[t["tag"]]]; pr = _PB_PR[t["tag"]]
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{pl[0]}-{pl[1]}-{pl[2]}_headINT8_B",
        "model_class": "pyramid_fusion", "dataset_src": "pathB_head_int8",
        "config_label": f"B_{t['tag']}_headINT8_restFP16", "triplet": f"B_{t['tag']}",
        "prune_rate": pr, "prune_object": "channel" if pr > 0 else "none",
        "prune_criterion": "l1_norm" if pr > 0 else None,
        "stage0_planes": pl[0], "stage1_planes": pl[1], "stage2_planes": pl[2],
        "deblocks_keep": 1.0, "shrink_keep": 1.0,
        "finetune_epochs": 23 if pr > 0 else None,
        "ckpt_status": "pruned_finetuned" if pr > 0 else "pretrained",
        "stage0_prec": "FP16", "stage1_prec": "FP16", "stage2_prec": "FP16",
        "q_mode": "head_int8_forced",
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A(head only)", "calibrator": "minmax",
        "hardware": "rtx4090", "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "regime": "ablation_guardrail_off",
        "latency_kind": "body_subnet_collab2",
        "lat_p50_ms": hw["lat_p50_ms"], "lat_p99_ms": hw["lat_p99_ms"], "lat_mean_ms": hw["lat_mean_ms"],
        "throughput_fps": hw["throughput_fps"],
        "ap30": t["ap30"], "ap50": t["ap50"], "ap70": t["ap70"],
        "n_ap_samples": 1789, "ap_pipeline": "DAIR_val_1789_TRT",
        "ap_baseline_ref": "fp16_same_planes", "delta_ap50_vs_baseline": None,
        "mean_power_w": hw["mean_power_w"], "energy_per_frame_mj": hw["energy_per_frame_mj"],
        "perf_per_watt_fps_per_w": hw["perf_per_watt_fps_per_w"],
        "eff_gmac_per_ms": hw["eff_gmac_per_ms"], "nominal_gmac": hw["nominal_gmac"],
        "engine_size_mb": hw["engine_size_mb"],
        "is_real_measured": True, "ap_valid": True,
        "source": (f"ap:pathB_head_int8_ablation.json({t.get('head_int8_summary')}, "
                   f"Δap70_vs_fp16={t.get('delta_ap70_vs_fp16')}); lat+energy:P_ab_hw_bench.csv(collab2 GPU7)"),
        "exec_path": "trt_engine_head_int8", "engine_path": hw["engine_path"],
        "notes": (f"B 消融: head-INT8/rest-FP16(护栏OFF on detection heads, {t.get('head_int8_summary')})。"
                  f"Δap70_vs_fp16={t.get('delta_ap70_vs_fp16')}(噪声内, head-INT8 不崩) 但不省延迟。regime=ablation 非前沿。"),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 1+2 的 33 行: 标 throughput 口径 ----------
# 这 33 行全是 batch=1 单流, 没有流水/批处理 -> throughput 退化为 1000/latency。
# 标 throughput_kind=inv_latency 显式声明"此处吞吐=1/延迟"(不是真冗余, 是欠采样)。
for r in rows:
    r["batch"] = 1
    r["throughput_kind"] = "inv_latency"  # 单流 batch1: throughput == 1000/lat_p50

# ---------- 源 3: E 类 — 能耗 (E4) + 解耦吞吐 (E3 异构流水) ----------
# 按 schema 纪律"不强行 join 不同测量": E4/E3 与 collab2-body 是不同测量,
# 故作为新行追加, 各自带 latency_kind / throughput_kind / source, 而非并到 33 行上。
# - E4 (rtx4090, NVML board 功率实测): batch1 单流(throughput=inv_latency) +
#   batch2 真批处理(throughput=batched, 脱离 1/lat) -> 提供能耗第 5 轴。
# - E3 (orin_agx, 跨进程 DLA0||DLA1 流水): interframe < e2e 单帧延迟 ->
#   throughput=pipelined, 真实证明吞吐 != 1/latency。AP/energy 未测(留空)。
def _planes_from_engine(name, triplet):
    import re
    m = re.search(r"engine_(\d{3})_(\d{3})_(\d{3})", str(name))
    if m:
        return [int(m.group(1)), int(m.group(2)), int(m.group(3))]
    return {  # doe6 引擎名不含 planes, 用 triplet 映射
        "T1_base": [64, 128, 256], "T2_p25": [48, 96, 192],
        "T4_p50": [32, 64, 128], "T6_p75": [16, 32, 64],
    }[triplet]

# core AP 查表 (planes, prec3) -> (ap30, ap50, ap70)。AP 由架构+权重+val集 决定,
# 与硬件/测量路径/batch 无关, 故同配置可安全共享给 E4/E3 行补全 AP 轴。
# 仅 EXACT (planes + 三段精度) 匹配, 绝不近似借用(如 [32,64,136]≠[32,64,128] 不匹配)。
_CORE_AP = {}
for _r in rows:
    if _r.get("ap70") is not None:
        _k = ((int(_r["stage0_planes"]), int(_r["stage1_planes"]), int(_r["stage2_planes"])),
              (_r["stage0_prec"], _r["stage1_prec"], _r["stage2_prec"]))
        _CORE_AP.setdefault(_k, (_r["ap30"], _r["ap50"], _r["ap70"]))

import os as _os
_E4_ENG_DIRS = ["output/doe_dataset_v1/real_engines", "models/p0_random_cache"]
def _find_eng_mb(name):
    for d in _E4_ENG_DIRS:
        p = ROOT / d / f"{name}.engine"
        if p.exists():
            return round(p.stat().st_size / 1e6, 3), str(p)
    return None, None

_E4_PR = {"T1_base": 0.0, "T2_p25": 0.25, "T4_p50": 0.5, "T6_p75": 0.75,
          "T_baseline": 0.0, "T_prune50": 0.5, "T_prune75": 0.75}
e4 = pd.read_csv(ROOT / "results/E4_energy_4090.csv")
for _, r in e4.iterrows():
    pl = _planes_from_engine(r["engine"], r["triplet"])
    pr = _E4_PR.get(r["triplet"], 0.0)
    prec = str(r["precision"]).upper()
    bt = int(r["batch"])
    tk = "batched" if bt > 1 else "inv_latency"
    eng_mb, _eng_full = _find_eng_mb(str(r["engine"]).replace(".engine", ""))
    _ap = _CORE_AP.get((tuple(pl), (prec, prec, prec)))  # EXACT 同配置才补 AP
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{pl[0]}-{pl[1]}-{pl[2]}_{prec}_b{bt}_E4",
        "model_class": "pyramid_fusion", "dataset_src": "E4_energy_v1",
        "config_label": f"{r['triplet']}_{prec}_b{bt}", "triplet": r["triplet"],
        "prune_rate": pr, "prune_object": "channel" if pr > 0 else "none",
        "prune_criterion": "l1_norm" if pr > 0 else None,
        "stage0_planes": pl[0], "stage1_planes": pl[1], "stage2_planes": pl[2],
        "deblocks_keep": 1.0, "shrink_keep": 1.0,
        "ckpt_status": "pruned_finetuned" if pr > 0 else "pretrained",
        "stage0_prec": prec, "stage1_prec": prec, "stage2_prec": prec,
        "q_mode": "uniform", "q_granularity": "per_channel_w/per_tensor_a",
        "q_object": "W+A", "calibrator": "minmax",
        "hardware": "rtx4090", "d_scheme": "GPU", "d_tactic": "default", "d_workspace_gb": 4,
        "latency_kind": "engine_board_energy",
        "lat_p50_ms": r["latency_p50_ms"], "lat_p99_ms": None, "lat_mean_ms": None,
        "throughput_fps": r["throughput_fps"], "throughput_kind": tk, "batch": bt,
        "ap30": (_ap[0] if _ap else None), "ap50": (_ap[1] if _ap else None),
        "ap70": (_ap[2] if _ap else None), "n_ap_samples": (1789 if _ap else None),
        "ap_pipeline": ("DAIR_val_1789_TRT (config-matched from core: 同 planes+精度 ckpt; "
                        "AP 与 hw/batch/测量路径无关)" if _ap else None),
        "ap_baseline_ref": None, "delta_ap50_vs_baseline": None,
        "mean_power_w": r["mean_power_w"], "energy_per_frame_mj": r["energy_per_frame_mj"],
        "perf_per_watt_fps_per_w": r["perf_per_watt_fps_per_w"],
        "engine_size_mb": eng_mb, "is_real_measured": True, "ap_valid": bool(_ap),
        "source": f"E4_energy_4090.csv:{r['source']}", "exec_path": "trt_engine_nvml_power",
        "engine_path": r["engine"],
        "notes": (f"NVML board 功率实测; engine_size 实测自磁盘; batch={bt}"
                  + ("(真批处理, throughput 脱离 1/lat)" if bt > 1 else "(单流, throughput=1/lat)")
                  + ("; AP 同配置补自 core(架构+精度 exact 匹配)" if _ap else
                     "; AP 无法补(core 无此 exact 配置: FP32 未测 / [32,64,136]≠core[32,64,128] 异架构)")),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# E3 异构 DLA 流水: (config, planes, prune_rate, e2e单帧延迟ms, interframe ms, throughput fps) 全真测
_E3_PIPE = [
    ("016_032_064", [16, 32, 64], 0.75, 23.71, 16.05, 62.3),
    ("032_064_136", [32, 64, 136], 0.5, 50.94, 32.17, 31.1),
]
for cfgname, pl, pr, e2e_lat, interf, tput in _E3_PIPE:
    _ap = _CORE_AP.get((tuple(pl), ("FP16", "FP16", "FP16")))  # DLA FP16, EXACT 架构匹配才补
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{pl[0]}-{pl[1]}-{pl[2]}_FP16_orinDLApipe_E3",
        "model_class": "pyramid_fusion", "dataset_src": "E3_orin_dla_pipe_v1",
        "config_label": f"dla_xproc_{cfgname}", "triplet": f"dla_split_{cfgname}",
        "prune_rate": pr, "prune_object": "channel" if pr > 0 else "none",
        "prune_criterion": "l1_norm" if pr > 0 else None,
        "stage0_planes": pl[0], "stage1_planes": pl[1], "stage2_planes": pl[2],
        "deblocks_keep": 1.0, "shrink_keep": 1.0,
        "ckpt_status": "pruned_finetuned" if pr > 0 else "pretrained",
        "stage0_prec": "FP16", "stage1_prec": "FP16", "stage2_prec": "FP16",
        "q_mode": "uniform", "q_granularity": "per_tensor", "q_object": "W+A", "calibrator": "none",
        "hardware": "orin_agx", "d_scheme": "DLA0||DLA1_xproc_pipeline",
        "d_tactic": "dla_fp16", "d_workspace_gb": 2,
        "latency_kind": "dla_pipeline_e2e_single_frame",
        "lat_p50_ms": e2e_lat, "lat_p99_ms": None, "lat_mean_ms": None,
        "throughput_fps": tput, "throughput_kind": "pipelined", "batch": 1,
        "ap30": (_ap[0] if _ap else None), "ap50": (_ap[1] if _ap else None),
        "ap70": (_ap[2] if _ap else None), "n_ap_samples": (1789 if _ap else None),
        "ap_pipeline": ("DAIR_val_1789_TRT (config-matched from core: 同 planes FP16 ckpt; "
                        "AP 与 hw 无关)" if _ap else None),
        "ap_baseline_ref": None, "delta_ap50_vs_baseline": None,
        "mean_power_w": None, "energy_per_frame_mj": None, "perf_per_watt_fps_per_w": None,
        "engine_size_mb": None, "is_real_measured": True, "ap_valid": bool(_ap),
        "source": "E3_orin_dla_pipeline.csv:real_xproc_shm_handoff", "exec_path": "orin_dla_xproc_pipeline",
        "engine_path": None,
        "notes": (f"异构 DLA0||DLA1 跨进程流水; e2e单帧延迟={e2e_lat}ms 但 interframe={interf}ms "
                  f"-> throughput={tput} QPS != 1000/lat({1000/e2e_lat:.1f}). 真实证明吞吐解耦于延迟。"
                  + ("; AP 同架构补自 core" if _ap else "; AP 无 exact 匹配([32,64,136] 异架构)")
                  + " energy 未测(Orin 无能耗采集); engine_size 未填(DLA stage-split 引擎)。"),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 9: Orin AGX FP16 跨硬件完整点 + 能耗 (E6) ----------
# hardware=orin_agx; latency_kind=body_subnet_collab2_orin (与 4090 collab2 不混比)。
# 能耗 = module-total(VIN_SYS_5V0 整模块)对 4090 整板, 非 GPU-rail; nvpmodel 限 30W。
# FP16: AP 借 4090 FP16 (fp16_crossplatform_exact, P0_3 output-match near-identical 验证)。
# INT8: ap_valid=False (Orin INT8 输出 NOT match 4090, cross-TRT divergence; 透明标, 仅 lat/energy 有效)。
_e6 = pd.read_csv(ROOT / "results/E6_orin_p03_lat_energy.csv")
_om = {(r["config"], r["precision"]): r for _, r in
       pd.read_csv(ROOT / "results/P0_3_orin_output_match.csv").iterrows()}
_ORIN_PL = {"base": [64, 128, 256], "p50": [32, 64, 128], "p75": [16, 32, 64]}
_ORIN_PR = {"base": 0.0, "p50": 0.5, "p75": 0.75}
for _, r in _e6.iterrows():
    cfg = r["config"]; prec = str(r["precision"]).upper(); pl = _ORIN_PL[cfg]; pr = _ORIN_PR[cfg]
    om = _om.get((cfg, r["precision"]))
    ap_valid = bool(om["ap_valid"]) if om is not None else (prec == "FP16")
    _ap = _CORE_AP.get((tuple(pl), (prec, prec, prec))) if ap_valid else None
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_{pl[0]}-{pl[1]}-{pl[2]}_{prec}_orin_E6",
        "model_class": "pyramid_fusion", "dataset_src": "E6_orin_p03",
        "config_label": f"orin_{cfg}_{prec.lower()}", "triplet": f"orin_{cfg}",
        "prune_rate": pr, "prune_object": "channel" if pr > 0 else "none",
        "prune_criterion": "l1_norm" if pr > 0 else None,
        "stage0_planes": pl[0], "stage1_planes": pl[1], "stage2_planes": pl[2],
        "deblocks_keep": 1.0, "shrink_keep": 1.0,
        "finetune_epochs": 23 if pr > 0 else None,
        "ckpt_status": "pruned_finetuned" if pr > 0 else "pretrained",
        "stage0_prec": prec, "stage1_prec": prec, "stage2_prec": prec, "q_mode": "uniform",
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A", "calibrator": "minmax",
        "hardware": "orin_agx", "d_scheme": "GPU", "d_tactic": "orin_30w", "d_workspace_gb": 2,
        "regime": "front",
        "latency_kind": "body_subnet_collab2_orin",
        "lat_p50_ms": r["lat_gpu_compute_median_ms"], "lat_p99_ms": None, "lat_mean_ms": None,
        "throughput_fps": r["throughput_qps"], "throughput_kind": "inv_latency", "batch": 1,
        "ap30": (_ap[0] if _ap else None), "ap50": (_ap[1] if _ap else None),
        "ap70": (_ap[2] if _ap else None), "n_ap_samples": (1789 if _ap else None),
        "ap_pipeline": ("DAIR_val_1789_TRT (crossplatform: 4090 FP16 AP, P0_3 output-match verified)"
                        if _ap else None),
        "mean_power_w": r["power_module_vin_w"], "energy_per_frame_mj": r["energy_per_frame_module_mj"],
        "perf_per_watt_fps_per_w": r["perf_per_watt_module_fps_per_w"],
        "engine_size_mb": None,
        "is_real_measured": True, "ap_valid": ap_valid,
        "source": (f"lat+energy:E6_orin_p03_lat_energy.csv(orin {r['power_mode']} tegrastats-VIN_SYS module-total); "
                   + ("ap:4090_fp16_crossplatform(P0_3 output-match near-identical)" if _ap
                      else "ap:orin_int8_NOT_match_4090(ap_valid=False)")),
        "exec_path": "orin_trt_collab2_tegrastats", "engine_path": r["engine"],
        "notes": (f"Orin AGX {r['power_mode']} 跨硬件点; 能耗=module-total(VIN_SYS_5V0 整模块)对 4090 整板, 非 GPU-rail。"
                  + (" INT8 ap_valid=False: Orin INT8 输出与 4090 NOT match(cross-TRT divergence), 仅 lat/energy 有效。"
                     if prec == "INT8" else " FP16 输出与 4090 near-identical(P0_3 验证), AP 借 4090 FP16。")),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 10: #12 collab2 多流 throughput (主前沿诚实 2D 的证据) ----------
# collab2 多 context 并发: 峰值仅 1.106x(n=2, SM@1=94% 已饱和)-> throughput 塌回 ~1/lat
# -> 主前沿诚实 2D(lat,energy)。throughput_kind=batched_request(非主 Pareto, 仅 throughput 轴证据)。
# 注: P12_subnet_batch_sweep.csv = subnet_capability batch ablation (NOT collab2 部署吞吐), 仅脚注, 不入表。
_p12 = pd.read_csv(ROOT / "results/P12_collab2_request_batch.csv")
_base_ap = _CORE_AP.get(((64, 128, 256), ("FP16", "FP16", "FP16")))
for _, r in _p12.iterrows():
    n = int(r["n_concurrent_requests"])
    if n == 1:
        continue  # n=1 == 已有 base_fp16 collab2 点, 不重复
    row = {c: None for c in COLS}
    row.update({
        "config_id": f"pyr_64-128-256_FP16_collab2req{n}_P12",
        "model_class": "pyramid_fusion", "dataset_src": "P12_collab2_throughput",
        "config_label": f"collab2_req{n}", "triplet": "T1_base",
        "prune_rate": 0.0, "prune_object": "none",
        "stage0_planes": 64, "stage1_planes": 128, "stage2_planes": 256,
        "deblocks_keep": 1.0, "shrink_keep": 1.0, "ckpt_status": "pretrained",
        "stage0_prec": "FP16", "stage1_prec": "FP16", "stage2_prec": "FP16", "q_mode": "uniform",
        "q_granularity": "per_channel_w/per_tensor_a", "q_object": "W+A", "calibrator": "none",
        "hardware": "rtx4090", "d_scheme": "GPU_multi_context", "d_tactic": "default", "d_workspace_gb": 4,
        "regime": "ablation_throughput_saturation",
        "latency_kind": "body_subnet_collab2",
        "lat_p50_ms": r["per_request_lat_p50_ms"], "lat_p99_ms": None, "lat_mean_ms": None,
        "throughput_fps": r["total_throughput_req_s"], "throughput_kind": "batched_request", "batch": n,
        "ap30": (_base_ap[0] if _base_ap else None), "ap50": (_base_ap[1] if _base_ap else None),
        "ap70": (_base_ap[2] if _base_ap else None), "n_ap_samples": (1789 if _base_ap else None),
        "ap_pipeline": ("DAIR_val_1789_TRT (config-matched base FP16 core; AP 与并发无关)" if _base_ap else None),
        "is_real_measured": True, "ap_valid": bool(_base_ap),
        "ap_reuse_basis": "core_exact_reuse_4090",
        "source": f"P12_collab2_request_batch.csv:{r['source']}",
        "exec_path": "trt_multi_context_concurrent", "engine_path": r["engine"],
        "notes": (f"#12 collab2 多流: {n} 并发 request, throughput={r['total_throughput_req_s']} req/s "
                  f"(decoupling vs n=1 = {r['decoupling_vs_n1']}x; SM@1 已饱和 util={r['gpu_util_pct']}%)。"
                  "峰值仅 1.106x -> throughput 塌回 ~1/lat -> 主前沿诚实 2D(lat,energy)。"
                  "throughput_kind=batched_request 非主 Pareto。"),
    })
    row["config_json"] = cfg_json(row)
    rows.append(row)

# ---------- 源 13: V2X-ViT A-1/A-2 AP 行 (Task #4, team-lead 授权 2026-06-05) ----------
# 事实源(json): results/v2xvit_baseline_a1.json / v2xvit_a2_p50.json / v2xvit_a2_p75.json
# 口径: fp32_pytorch, DAIR val 1789 全集, 逐行独立真测; epoch 锁定各见 json (ISS-024)
# ★纪律:
#   (1) model_class=v2x_vit, 严禁与 pyramid_fusion 混口径/混表
#   (2) p75 actual_filters=[64,32,64] 非单调(L1+round_to=32 致 stage0 留满)→ 禁拼"剪枝率→AP"单调曲线
#   (3) latency_kind=NA_pending_trt_build (A-3 TRT build HOLD 在用户桌上; 仅 AP 有效)
#   (4) mATE/mASE/mAOE+CI 直接从 json 读入(逐行独立真测, DAIR_val_1789 同人口全集 CI, ISS-024规范)
#   (5) v2xvit_dair_real.json 含 PyTorch 分段计时(e2e=61.86ms / fusion_net=27.39ms)供参考但不入 lat 列;
#       A-3 TRT build 后由 hw 补真测 latency
# 注: "四行"如含 v2xvit_dair_real.json 计时行则该行独立于 AP 口径,
#     本次按 "以 3 个 AP json 为准" 入 3 行; 计时信息记在 notes

_V2XVIT_SRCS = [
    (ROOT / "results/v2xvit_baseline_a1.json", "v2xvit_base_fp32",  0.0,  None, "pretrained"),
    (ROOT / "results/v2xvit_a2_p50.json",      "v2xvit_p50_fp32",   0.5,  25,   "pruned_finetuned"),
    (ROOT / "results/v2xvit_a2_p75.json",      "v2xvit_p75_fp32",   0.75, 25,   "pruned_finetuned"),
]

def _v2xvit_cfg_json(backbone_base, actual_f, prune_ratio, epoch_used, ckpt_path):
    return json.dumps({
        "model_type": "v2x_vit",
        "backbone": {
            "base_filters": backbone_base,
            "actual_filters": actual_f,
            "prune_ratio_nominal": prune_ratio,
            "prune_criterion": "l1_norm" if prune_ratio > 0 else None,
        },
        "quant": {"precision": "fp32_pytorch", "trt_build": "pending_A3"},
        "epoch_lock": epoch_used,
        "ckpt_path": ckpt_path,
    }, ensure_ascii=False)

for _v2x_jp, _cfg_lbl, _pr, _ft_ep, _ckpt_st in _V2XVIT_SRCS:
    _jd = json.load(open(_v2x_jp))
    # actual_filters: base json 无此字段 -> 默认 [64,128,256]; 剪枝行从 json 读
    _af = list(_jd["actual_filters"]) if "actual_filters" in _jd else [64, 128, 256]
    _s0, _s1, _s2 = int(_af[0]), int(_af[1]), int(_af[2])
    _pruned = _pr > 0
    _epoch_used = str(_jd.get("epoch_used", "unknown"))
    _non_mono = _pruned and (_af[0] > _af[1] or _af[1] > _af[2])

    _row = {c: None for c in COLS}
    _row.update({
        "config_id":         f"v2xvit_{_s0}-{_s1}-{_s2}_FP32",
        "model_class":       "v2x_vit",
        "dataset_src":       "v2xvit_A1A2_ap",
        "config_label":      _cfg_lbl,
        "triplet":           f"v2xvit_{_jd['anchor']}",
        "prune_rate":        _pr,
        "prune_object":      "channel_backbone" if _pruned else "none",
        "prune_criterion":   "l1_norm" if _pruned else None,
        "stage0_planes":     _s0, "stage1_planes": _s1, "stage2_planes": _s2,
        "deblocks_keep":     None, "shrink_keep": None,
        "params_total":      None,
        "finetune_epochs":   _ft_ep,
        "ckpt_status":       _ckpt_st,
        # FP32 PyTorch: 无 TRT 量化
        "stage0_prec": "FP32", "stage1_prec": "FP32", "stage2_prec": "FP32",
        "q_mode":            "fp32_pytorch",
        "q_granularity":     None, "q_object": None, "calibrator": None,
        "hardware":          "rtx4090",
        "d_scheme":          "GPU_pytorch",
        "d_tactic":          None,
        "d_workspace_gb":    None,
        # latency: NOT_MEASURED (A-3 TRT build HOLD)
        "latency_kind":      "NA_pending_trt_build",
        "lat_p50_ms":        None, "lat_p99_ms": None, "lat_mean_ms": None,
        "throughput_fps":    None, "throughput_kind": None, "batch": 1,
        # AP from json (4dp 真测)
        "ap30":              round(float(_jd["ap30"]), 6),
        "ap50":              round(float(_jd["ap50"]), 6),
        "ap70":              round(float(_jd["ap70"]), 6),
        "n_ap_samples":      int(_jd["n_samples"]),
        "ap_pipeline":       "DAIR_val_1789_PyTorch_FP32",
        "ap_baseline_ref":   None, "delta_ap50_vs_baseline": None,
        # TP 几何误差从 json 直接读 (逐行独立真测, 同人口全集 CI, ISS-024规范)
        "mATE":              round(float(_jd["mATE"]), 6),
        "mASE":              round(float(_jd["mASE"]), 6),
        "mAOE":              round(float(_jd["mAOE"]), 6),
        "mAOE_ci_lo":        round(float(_jd["mAOE_ci_lo"]), 6),
        "mAOE_ci_hi":        round(float(_jd["mAOE_ci_hi"]), 6),
        "mAOE_n_tp":         int(_jd["n_tp"]),
        "mAOE_basis":        "measured",
        "metric_pipeline":   (
            "mAOE:DAIR_val_1789_全集_PyTorch_FP32; "
            "CI:bootstrap_95%_同人口全集(ISS-024规范); "
            f"epoch_lock={_epoch_used}(ISS-024)"
        ),
        # 能耗: 未测
        "mean_power_w":             None,
        "energy_per_frame_mj":      None,
        "perf_per_watt_fps_per_w":  None,
        "engine_size_mb":           None,
        # 有效性/溯源
        "is_real_measured":  True,
        "ap_valid":          True,
        "regime":            "ap_reference_pending_trt",
        "ap_reuse_basis":    "independent_measured",
        "source":            (
            f"ap:{_v2x_jp.name}(v2x_vit DAIR_val_1789 "
            f"PyTorch_FP32 independent_measured epoch={_epoch_used})"
        ),
        "exec_path":         "pytorch_fp32_inference",
        "ts":                None,
        "onnx_path":         None,
        "engine_path":       _jd.get("ckpt_path"),
        "config_json":       _v2xvit_cfg_json(
            [64, 128, 256], _af, _pr, _epoch_used, _jd.get("ckpt_path")
        ),
        "notes":             (
            f"V2X-ViT A-{'1' if not _pruned else '2'} "
            f"{'baseline' if not _pruned else f'pruned(nominal={_pr}; actual_filters={_af})'} "
            f"DAIR_val_1789 PyTorch_FP32. epoch_lock={_epoch_used}(ISS-024). "
            f"latency=NA_pending_trt_build(A-3 TRT build HOLD 在用户桌上). "
            + ("★p75实测actual_filters=[64,32,64]非单调(L1+round_to=32 stage0留满); "
               "禁拼单调剪枝率→AP曲线. " if _non_mono else "")
            + ("参考:v2xvit_dair_real.json有PyTorch分段计时(e2e p50=56.61ms/"
               "fusion_net p50=24.02ms/NMS p50=22.21ms); "
               "A-3 TRT build后由hw补真测lat. " if not _pruned else "")
            + "model_class=v2x_vit 不与 pyramid_fusion 混口径/混表."
        ),
    })
    rows.append(_row)
print(f"[v2xvit A1/A2] 追加 {len(_V2XVIT_SRCS)} 行 "
      f"(AP-only, latency pending A-3 TRT build; model_class=v2x_vit 独立口径)")

# ---------- 源 14: V2X-ViT base PyTorch 分段计时行 (Task #4 补录 第四行, team-lead 授权 2026-06-05) ----------
# 授权原文: "team-lead 授权 data-orchestrator 补录第四行: v2xvit_dair_real.json 的
#           PyTorch 分段计时入 dataset_v2, latency_kind=forward_hook_pytorch_fp32"
# 数值严格以 data/v2x_baseline_timing/v2xvit_dair_real.json 为准;
# AP 借用 A-1 (同 ckpt + 同 DAIR_val_1789 + FP32 → exact_reuse)
_TIMING_JP  = ROOT / "data/v2x_baseline_timing/v2xvit_dair_real.json"
_tdj        = json.load(open(_TIMING_JP))
_A1_JP_reuse = ROOT / "results/v2xvit_baseline_a1.json"
_a1jd_reuse  = json.load(open(_A1_JP_reuse))

# e2e_walltime (入库值用 p50; mean 留于 notes / breakdown)
_t_e2e_p50  = float(_tdj["e2e_walltime"]["p50_ms"])    # 56.609 ms
_t_e2e_p99  = float(_tdj["e2e_walltime"]["p99_ms"])    # 156.217 ms
_t_e2e_mean = float(_tdj["e2e_walltime"]["mean_ms"])   # 61.863 ms (audit 引用的是此均值)

# forward_stages CUDA Event p50
_t_enc  = float(_tdj["forward_stages_cuda_event"]["encoder"]["p50_ms"])       # 2.803 ms
_t_bb   = float(_tdj["forward_stages_cuda_event"]["backbone"]["p50_ms"])      # 2.914 ms
_t_shr  = float(_tdj["forward_stages_cuda_event"]["shrinker_m1"]["p50_ms"])   # 0.891 ms
_t_fus  = float(_tdj["forward_stages_cuda_event"]["fusion_net"]["p50_ms"])    # 24.024 ms
_t_nms  = float(_tdj["postproc_substages_cuda_event"]["nms"]["p50_ms"])       # 22.209 ms

_timing_row = {c: None for c in COLS}
_timing_row.update({
    "config_id":       "v2xvit_64-128-256_FP32_hook_timing",
    "model_class":     "v2x_vit",
    "dataset_src":     "v2xvit_dair_real_timing",
    "config_label":    "v2xvit_base_fp32_hook_e2e",
    "triplet":         "v2xvit_base_hook",
    "prune_rate":      0.0,
    "prune_object":    "none",
    "prune_criterion": None,
    "stage0_planes":   64, "stage1_planes": 128, "stage2_planes": 256,
    "finetune_epochs": None,
    "ckpt_status":     "pretrained",
    "stage0_prec":     "FP32", "stage1_prec": "FP32", "stage2_prec": "FP32",
    "q_mode":          "fp32_pytorch",
    "hardware":        "rtx4090",
    "d_scheme":        "GPU_pytorch",
    "latency_kind":    "forward_hook_pytorch_fp32",
    "lat_p50_ms":      round(_t_e2e_p50,  3),   # 56.609 ms (e2e_walltime p50)
    "lat_p99_ms":      round(_t_e2e_p99,  3),   # 156.217 ms
    "lat_mean_ms":     round(_t_e2e_mean, 3),   # 61.863 ms (即 audit 引均值)
    "throughput_fps":  round(1000.0 / _t_e2e_p50, 3),
    "throughput_kind": "inv_latency",
    "batch":           1,
    # AP: 借用 A-1 (同 ckpt + 同 val 集 + FP32 → deterministic exact_reuse)
    "ap30": round(float(_a1jd_reuse["ap30"]), 6),
    "ap50": round(float(_a1jd_reuse["ap50"]), 6),
    "ap70": round(float(_a1jd_reuse["ap70"]), 6),
    "n_ap_samples":    int(_a1jd_reuse["n_samples"]),
    "ap_pipeline":     "DAIR_val_1789_PyTorch_FP32",
    "mATE":            round(float(_a1jd_reuse["mATE"]), 6),
    "mASE":            round(float(_a1jd_reuse["mASE"]), 6),
    "mAOE":            round(float(_a1jd_reuse["mAOE"]), 6),
    "mAOE_ci_lo":      round(float(_a1jd_reuse["mAOE_ci_lo"]), 6),
    "mAOE_ci_hi":      round(float(_a1jd_reuse["mAOE_ci_hi"]), 6),
    "mAOE_n_tp":       int(_a1jd_reuse["n_tp"]),
    "mAOE_basis":      "measured",
    "metric_pipeline": (
        f"latency:v2xvit_dair_real.json(PyTorch_FP32_hook CUDA-event "
        f"warmup={int(_tdj['warmup'])} measure={int(_tdj['measure'])} {_tdj['device']}); "
        f"AP:v2xvit_baseline_a1.json(DAIR_val_1789 PyTorch_FP32 independent_measured "
        f"epoch_lock={_a1jd_reuse['epoch_used']}(ISS-024))"
    ),
    "is_real_measured": True,
    "ap_valid":         True,
    "regime":           "ap_reference_pending_trt",
    "ap_reuse_basis":   "core_exact_reuse_4090",
    "source": (
        f"timing:v2xvit_dair_real.json(forward_hook_pytorch_fp32 e2e_p50="
        f"{round(_t_e2e_p50,3)}ms warmup={int(_tdj['warmup'])} measure={int(_tdj['measure'])}); "
        f"ap:v2xvit_baseline_a1.json(同ckpt core_exact_reuse_4090)"
    ),
    "exec_path":   "pytorch_fp32_inference_hook",
    "engine_path": _tdj.get("ckpt"),
    "config_json": json.dumps({
        "model_type": "v2x_vit",
        "backbone": {"actual_filters": [64, 128, 256], "prune_ratio_nominal": 0.0},
        "quant": {"precision": "fp32_pytorch", "trt_build": "pending_A3"},
        "timing_breakdown_p50_ms": {
            "e2e_walltime":  round(_t_e2e_p50,  3),
            "encoder":       round(_t_enc,  3),
            "backbone":      round(_t_bb,   3),
            "shrinker_m1":   round(_t_shr,  3),
            "fusion_net":    round(_t_fus,  3),
            "nms":           round(_t_nms,  3),
        },
        "timing_mean_ms": {
            "e2e_walltime": round(_t_e2e_mean, 3),
            "fusion_net":   round(float(_tdj["forward_stages_cuda_event"]["fusion_net"]["mean_ms"]), 3),
            "nms":          round(float(_tdj["postproc_substages_cuda_event"]["nms"]["mean_ms"]), 3),
        },
        "timing_meta": {
            "warmup":  int(_tdj["warmup"]),
            "measure": int(_tdj["measure"]),
            "device":  _tdj["device"],
            "ckpt":    _tdj.get("ckpt"),
        },
    }, ensure_ascii=False),
    "notes": (
        f"V2X-ViT base PyTorch_FP32 hook 级分段计时行(latency_kind=forward_hook_pytorch_fp32). "
        f"e2e_walltime: p50={round(_t_e2e_p50,3)}ms p99={round(_t_e2e_p99,3)}ms "
        f"mean={round(_t_e2e_mean,3)}ms. "
        f"breakdown(p50): encoder={round(_t_enc,3)}ms backbone={round(_t_bb,3)}ms "
        f"shrinker_m1={round(_t_shr,3)}ms fusion_net={round(_t_fus,3)}ms "
        f"nms={round(_t_nms,3)}ms. "
        f"★口径交叉注(ISS-035 supervisor 裁决): 同 run mean 口径 = "
        f"e2e 61.86ms/fusion 27.39ms/NMS 23.75ms(审计 v1.3 用 mean); "
        f"本行入库用 json p50: e2e=56.61ms/fusion=24.02ms/NMS=22.21ms. "
        f"fusion_net p99=111ms 重尾致 mean>p50, 两组勿当矛盾测量混比(同一 run 不同统计量). "
        f"AP借用A-1(同ckpt+同val集+FP32, core_exact_reuse_4090). "
        f"latency_kind=forward_hook_pytorch_fp32 ≠ collab2 ≠ e2e_trt, 不与任何TRT行比较. "
        f"model_class=v2x_vit 不与 pyramid_fusion 混口径/混表."
    ),
})
rows.append(_timing_row)
print(f"[v2xvit timing] 追加 1 行 "
      f"(latency_kind=forward_hook_pytorch_fp32; e2e_p50={round(_t_e2e_p50,3)}ms; "
      f"AP core_exact_reuse_4090 from A-1; model_class=v2x_vit)")

df = pd.DataFrame(rows)[COLS]

# ---------- 去重: complete_points_v1 ↔ perstage_AP_v2 的 uniform 配置完全重复 ----------
# 两源在 uniform(FFF/III)配置上录了同一数据点(同 config + 同 collab2 口径 + AP/lat 数值一致)。
# 保留 complete_points_v1(有 params_total/build_secs/onnx 等更全溯源), 删 perstage 的重复份。
# 注: E4(engine_board_energy 单agent口径)与 core(collab2 双agent口径)虽同 config 但 latency_kind
# 不同(scope 不同, E4≈0.65×core)且 E4 带能耗 -> 不是重复, 不删。
_sig = df.apply(lambda r: (
    int(r.stage0_planes), int(r.stage1_planes), int(r.stage2_planes),
    r.stage0_prec, r.stage1_prec, r.stage2_prec, r.hardware, int(r.batch), r.latency_kind,
    round(r.ap70, 4) if pd.notna(r.ap70) else None,
    round(r.lat_p50_ms, 4) if pd.notna(r.lat_p50_ms) else None), axis=1)
_n_before = len(df)
df = df[~_sig.duplicated(keep="first")].reset_index(drop=True)
print(f"[dedup] 删除完全重复行 {_n_before - len(df)} 个 (complete↔perstage uniform 重叠)")

# ---------- 源 4: collab2 能耗 (E5) + P2 补 AP -> 填回已有行(去重后)----------
# E5: 28 个 collab2 引擎在双 agent 输入下的 NVML 板级能耗(frame=1 inference, 非/2)。
# 按 engine_path 精确匹配填回 collab2 行 -> 这 28 行的 energy 与其 collab2 延迟同口径自洽。
import json as _json
_e5p = ROOT / "results/E5_collab2_energy.csv"
if _e5p.exists():
    _e5 = pd.read_csv(_e5p).drop_duplicates("engine_path").set_index("engine_path")
    _n = 0
    for i in df.index:
        ep = df.at[i, "engine_path"]
        if df.at[i, "latency_kind"] == "body_subnet_collab2" and ep in _e5.index:
            er = _e5.loc[ep]
            df.at[i, "mean_power_w"] = round(float(er["mean_power_w"]), 2)
            df.at[i, "energy_per_frame_mj"] = round(float(er["energy_per_frame_mj"]), 4)
            df.at[i, "perf_per_watt_fps_per_w"] = round(float(er["perf_per_watt_fps_per_w"]), 3)
            df.at[i, "notes"] = str(df.at[i, "notes"]) + " | energy: E5 collab2 双agent NVML 实测(frame=1)"
            _n += 1
    print(f"[E5] 填回 collab2 能耗 {_n} 行")

# P2: T1_base_FP32 AP(真测)+ [32,64,136] AP(finetune 收敛后才填; 否则跳过)
_p2p = ROOT / "results/P2_ap_fill.json"
if _p2p.exists():
    _p2 = _json.load(open(_p2p))
    def _fill_ap(planes, prec_is_fp32, ap, tag):
        n = 0
        for i in df.index:
            pl = (int(df.at[i, "stage0_planes"]), int(df.at[i, "stage1_planes"]), int(df.at[i, "stage2_planes"]))
            ok = pl == planes and ((df.at[i, "stage0_prec"] == "FP32") == prec_is_fp32)
            if ok and pd.isna(df.at[i, "ap70"]):
                df.at[i, "ap30"], df.at[i, "ap50"], df.at[i, "ap70"] = ap
                df.at[i, "ap_valid"] = True
                df.at[i, "n_ap_samples"] = 1789
                df.at[i, "ap_pipeline"] = f"DAIR_val_1789 ({tag})"
                df.at[i, "notes"] = str(df.at[i, "notes"]) + f" | AP: {tag}"
                n += 1
        return n
    _f32 = _p2.get("t1_base_fp32", {})
    if _f32.get("status") == "ok":
        nn = _fill_ap((64, 128, 256), True, (_f32["ap30"], _f32["ap50"], _f32["ap70"]), "P2 FP32 真测 canonical")
        print(f"[P2] FP32 AP 填回 {nn} 行")
    _a136 = _p2.get("planes_032_064_136", {})
    if _a136.get("status") == "ok" and _a136.get("ap70") is not None:
        nn = _fill_ap((32, 64, 136), False, (_a136["ap30"], _a136["ap50"], _a136["ap70"]), "P2 [32,64,136] finetune 真测")
        print(f"[P2] [32,64,136] AP 填回 {nn} 行")
    else:
        print(f"[P2] [32,64,136] AP 状态={_a136.get('status')} -> 暂不填(finetune 收敛后重跑 builder)")

# ---------- 源 11: E 难子集 距离分箱 AP (负结果) -> 填入 ap70_by_range 诊断列 ----------
# 诚实记: spread 随距离收缩(非放大), 远距离 AP floor 现象 -> 非前沿 trade-off, 不新增行。
# 把分箱 ap70 写入对应 collab2 uniform 行的 ap70_by_range (JSON {bin: ap70})。
_peJ = ROOT / "results/pathE_distance_binned_ap.json"
if _peJ.exists():
    _pe = _json.load(open(_peJ))["configs"]
    _PE_MATCH = {  # pathE config -> (planes, prec)
        "base_fp16": ((64, 128, 256), "FP16"), "base_int8": ((64, 128, 256), "INT8"),
        "pruned50_int8": ((32, 64, 128), "INT8"), "pruned75_int8": ((16, 32, 64), "INT8"),
    }
    _BINS = ["near", "mid", "r50_60", "r60_80", "r80plus", "far50plus", "full"]
    _n_pe = 0
    for cfg, (pl, prec) in _PE_MATCH.items():
        if cfg not in _pe:
            continue
        binned = {b: round(_pe[cfg][b]["ap70"], 4) for b in _BINS if b in _pe[cfg]}
        for i in df.index:
            if (df.at[i, "latency_kind"] == "body_subnet_collab2"
                    and (int(df.at[i, "stage0_planes"]), int(df.at[i, "stage1_planes"]),
                         int(df.at[i, "stage2_planes"])) == pl
                    and df.at[i, "stage0_prec"] == prec
                    and df.at[i, "q_mode"] in ("uniform", "auto")  # p75-int8 唯一 collab2 行是 auto(global_automix); 排除 forced/mixed
                    and df.at[i, "dataset_src"] != "P12_collab2_throughput"  # 排除多流 throughput-probe 行
                    and pd.isna(df.at[i, "ap70_by_range"])):
                df.at[i, "ap70_by_range"] = _json.dumps(binned, ensure_ascii=False)
                df.at[i, "notes"] = str(df.at[i, "notes"]) + (
                    " | E 距离分箱 ap70(pathE 负结果: spread 随距离收缩非放大, 远距离 AP floor)")
                _n_pe += 1
                break
    print(f"[pathE] 填入 ap70_by_range 分箱 {_n_pe} 行 (负结果)")

# ---------- 后处理: regime 填充 + ap:TODO 修正 + p25 FP16 latency 覆盖 ----------
# (1) regime: 源5 已显式标 front/ablation_guardrail_off; 其余按 q_mode 填(selector 过滤前沿用)。
#     uniform/auto = front(可上前沿); uniform_forced = ablation_guardrail_off(护栏OFF, ISS-013 被支配);
#     per_stage_mixed = ablation_perstage_dominated(被 TRT-auto 支配, 非前沿)。
def _regime_of(qmode):
    if qmode == "uniform_forced":
        return "ablation_guardrail_off"
    if qmode == "per_stage_mixed":
        return "ablation_perstage_dominated"
    return "front"
_n_reg = 0
for i in df.index:
    if pd.isna(df.at[i, "regime"]) or df.at[i, "regime"] is None:
        df.at[i, "regime"] = _regime_of(df.at[i, "q_mode"]); _n_reg += 1
print(f"[regime] 填充 {_n_reg} 行 (源5 已显式标); 分布: {df['regime'].value_counts().to_dict()}")

# (2) ap:TODO 陈旧标注修正(只改标注不改数值): AP 实际已从 stage_a 金标准填入且数值对。
_n_todo = 0
for i in df.index:
    s = df.at[i, "source"]
    if isinstance(s, str) and "ap:TODO" in s and pd.notna(df.at[i, "ap70"]):
        df.at[i, "source"] = s.replace("ap:TODO", "ap:stage_a_ap_real(gold)"); _n_todo += 1
print(f"[ap:TODO] 修正陈旧 source 标注 {_n_todo} 行 (数值不变)")

# (3) p25 FP16 collab2 latency 覆盖为 hw GPU7-clean rebaseline (2.946->2.905, team-lead 指定; 旧值 noise 内)。
_hwf = _p0_trap["p25_fp16_rebaseline"]
_n_ov = 0
for i in df.index:
    if (df.at[i, "latency_kind"] == "body_subnet_collab2" and int(df.at[i, "stage0_planes"]) == 48
            and df.at[i, "stage0_prec"] == "FP16"):
        df.at[i, "lat_p50_ms"] = _hwf["lat_p50_ms"]; df.at[i, "lat_p99_ms"] = _hwf["lat_p99_ms"]
        df.at[i, "lat_mean_ms"] = _hwf["lat_mean_ms"]; df.at[i, "throughput_fps"] = _hwf["throughput_fps"]
        df.at[i, "notes"] = str(df.at[i, "notes"]) + " | lat 更新为 hw GPU7-clean rebaseline 2.905ms(旧 2.946 noise内, 同 collab2)"
        _n_ov += 1
print(f"[p25 FP16 lat] 覆盖为 GPU7-clean {_n_ov} 行")

# (4) ap_reuse_basis 全表回填(ISS-017 supervisor 建议): AP 溯源强度机器可读列。
#     枚举: independent_measured(本 config 逐 engine 独立真测) /
#           stage_a_gold_exact_reuse(uniform 配置 AP 取自 4090 stage_a 金标准, 同 config) /
#           core_exact_reuse_4090(E4/E3 跨 scope 借 4090 同 config core AP) /
#           fp16_crossplatform_exact(Orin FP16 借 4090 FP16, 数学一致) /
#           orin_int8_gate1_scale_contract[+spotcheck](Orin INT8 双闸门, 由 Orin 源块设) /
#           none(无 AP / ap_valid=False)。
def _ap_reuse_basis(r):
    if pd.isna(r["ap70"]):
        return "none"
    src = str(r["source"]); ds = str(r["dataset_src"]); hw = str(r["hardware"])
    if hw == "orin_agx":
        return "fp16_crossplatform_exact" if r["stage0_prec"] == "FP16" else "orin_int8_pending_gate"
    if ds in ("E4_energy_v1", "E3_orin_dla_pipe_v1"):
        return "core_exact_reuse_4090"
    if ("reference_only" in src or "prior_eval" in src or "stage_a" in src
            or ds == "complete_points_v1"):
        return "stage_a_gold_exact_reuse"
    return "independent_measured"  # perstage real_mixed/forced + p25 forced: 逐 engine 真测
_n_b = 0
for i in df.index:
    if pd.isna(df.at[i, "ap_reuse_basis"]) or df.at[i, "ap_reuse_basis"] is None:
        df.at[i, "ap_reuse_basis"] = _ap_reuse_basis(df.loc[i]); _n_b += 1
print(f"[ap_reuse_basis] 全表回填 {_n_b} 行; 分布: {df['ap_reuse_basis'].value_counts().to_dict()}")

# ---------- 源 12: mATE/mASE/mAOE (supervisor ISS-020/024/029 终核 PASS) ----------
# 定位: 预测器约束信号(标 regime), 非成本 Pareto 目标轴。
#   剪枝轴 SNR=14.2×(ISS-020终核, 全档均半宽口径; 备用13.1×avg-half/17.1×pooled-SE)。
#   pairwise CI 全不重叠, 除 p25↔p50(相邻细档不可分辨, 须诚实标)。
#   INT8: p25/p50/p75 SNR=0.58/0.05/0.82×(噪声级,CI重叠,非单调变号), 无量化轴信号;
#         base档 Δ+0.0030(2.4×半宽CI不重叠)但非单调+量级仅5%, 按ISS-020#4不构成轴信号。
# 评估与 stage_a 同路径: stage_a_cache 引擎 + 1618 TRT collab + 171 PyTorch fallback。
# 方案C(修正版 ISS-024/025): mAOE=全集1789点估计 + mAOE_ci_lo/hi=同人口全集bootstrap 95%CI。
# mAOE_basis: measured(本config直接评估,8行) / exact_reuse(同planes+prec EXACT复用,17行)。
# 填入规则: (anchor → planes) EXACT 匹配 + stage0==stage1==stage2 (uniform) + ap_valid=True
#           + q_mode 无 forced (forced 引擎量化不同, mAOE 不可借用; Orin INT8 不可借用)。
# ★ n_tp 增长归因(ISS-024修正): 权重错配修复使检测质量恢复 → TP 变多
#    (pruned 6行 +112~+851; base 2行微变 fp16=-6/int8=+21)。与幸存者偏差是两个不同概念。
# ⚠️ pruned50 INT8 幸存者偏差(ISS-024修正后仍存在): INT8 n_tp=25577 比 FP16 26154 少 577
#    → mATE/mAOE 不可与 FP16 直接比(survivor bias 与 ISS-024 epoch修正无关)。
# ⚠️ ISS-029 pruned50 AP70 口径: eval_script fp16=0.5730/int8=0.5636 vs stage_a +0.009
#    (两精度同向; 其余6行 ≤0.002); AP 列保留 stage_a 值(fp16=0.5641/int8=0.5542), 须标 caveat。
# ⚠️ pruned75 AP70 口径: 修正版 eval_script vs stage_a 差≤0.0002(旧污染偏差已消除);
#    AP 列保留 stage_a 值。
_TP_ANCHOR_PLANES = {
    "base": (64, 128, 256),
    "pruned25": (48, 96, 192),
    "pruned50": (32, 64, 128),
    "pruned75": (16, 32, 64),
}

def _maoe_basis(r_ds, r_cl, r_pl0):
    """mAOE_basis: measured(8行直接评估) vs exact_reuse(17行EXACT复用)"""
    if r_ds == "complete_points_v1":
        return "measured"   # 6 rows: base/p25/p50/p75×fp16 + base/p50×int8
    if r_ds == "P0_1_p25_trap" and "automix" in str(r_cl) and "forced" not in str(r_cl):
        return "measured"   # 1 row: p25 int8 automix
    if r_ds == "perstage_AP_v2" and "automix" in str(r_cl) and int(r_pl0) == 16:
        return "measured"   # 1 row: p75 int8 global_int8_automix (dedup后唯一perstage global)
    return "exact_reuse"    # 17 rows: E4/E6/E3/P12

_tp_errs_p = ROOT / "results/tp_errors_corrected_full.csv"  # ISS-024修正版(单文件含同人口CI)
if _tp_errs_p.exists():
    _tp_errs = {(str(r["anchor"]), str(r["precision"])): r
                for _, r in pd.read_csv(_tp_errs_p).iterrows()}
    _n_mao = 0
    for i in df.index:
        p0, p1, p2 = df.at[i, "stage0_prec"], df.at[i, "stage1_prec"], df.at[i, "stage2_prec"]
        # 只填 uniform (三段精度一致) 且 ap_valid=True
        if p0 != p1 or p1 != p2 or not df.at[i, "ap_valid"]:
            continue
        # 排除 forced 配置: forced 引擎量化不同, mAOE 不可借用
        qm = str(df.at[i, "q_mode"])
        if "forced" in qm:
            continue
        prec = str(p0).lower()  # "fp16" / "int8"
        if prec not in ("fp16", "int8"):
            continue
        pl = (int(df.at[i, "stage0_planes"]), int(df.at[i, "stage1_planes"]),
              int(df.at[i, "stage2_planes"]))
        anchor = next((k for k, v in _TP_ANCHOR_PLANES.items() if v == pl), None)
        if anchor is None:
            continue
        tp_row = _tp_errs.get((anchor, prec))
        if tp_row is None:
            continue
        survivor_bias = (anchor == "pruned50" and prec == "int8")
        p75_caveat = (anchor == "pruned75")
        p50_ap_caveat = (anchor == "pruned50")  # ISS-029: AP70 偏 stage_a gold +0.009, 两精度同向
        df.at[i, "mATE"] = round(float(tp_row["mATE"]), 6)
        df.at[i, "mASE"] = round(float(tp_row["mASE"]), 6)
        df.at[i, "mAOE"] = round(float(tp_row["mAOE"]), 6)
        df.at[i, "mAOE_n_tp"] = int(tp_row["n_tp"])
        # 方案C(修正版 ISS-024): CI 列同人口全集bootstrap 95%CI (与全集点估计同子集)
        if "mAOE_ci_lo" in tp_row.index and pd.notna(tp_row["mAOE_ci_lo"]):
            df.at[i, "mAOE_ci_lo"] = round(float(tp_row["mAOE_ci_lo"]), 6)
            df.at[i, "mAOE_ci_hi"] = round(float(tp_row["mAOE_ci_hi"]), 6)
        # mAOE_basis: 溯源强度标记 (8 measured + 17 exact_reuse = 25)
        df.at[i, "mAOE_basis"] = _maoe_basis(
            df.at[i, "dataset_src"], df.at[i, "config_label"], df.at[i, "stage0_planes"])
        df.at[i, "metric_pipeline"] = (
            "mAOE:DAIR_val_1789_全集(1618_TRT_collab+171_PyTorch_fallback); "
            "CI:同人口全集_bootstrap_B=1000_95%CI(ISS-024修正,共同GT); "
            "mAOE=min(|Δyaw|mod_π,π-|Δyaw|mod_π); scripts/phase2/eval_tp_errors_corrected_full.py"
            + ("; WARN:pruned50-INT8_survivor_bias(n_tp-577,不可比)" if survivor_bias else "")
            + ("; WARN:pruned75_AP70_eval_vs_stage_a_diff≤0.0002,AP保留gold" if p75_caveat else "")
            + ("; WARN:ISS-029_pruned50_AP70_eval偏stage_a+0.009(两精度同向),AP保留stage_a" if p50_ap_caveat else "")
        )
        df.at[i, "notes"] = str(df.at[i, "notes"]) + (
            " | mATE/mASE/mAOE+CI: TP几何误差(ISS-020/024/029终核PASS,剪枝约束信号)"
            + (" ⚠️pruned50-INT8幸存者偏差(n_tp-577)" if survivor_bias else "")
            + (" ⚠️pruned75-AP口径差(修正后≤0.0002)" if p75_caveat else "")
            + (" ⚠️ISS-029:pruned50-AP70偏gold+0.009" if p50_ap_caveat else "")
        )
        _n_mao += 1
    print(f"[mAOE] 填入 mATE/mASE/mAOE {_n_mao} 行; "
          f"mAOE 非空={df['mAOE'].notna().sum()}/{len(df)}")
    # 信号强度报告 (front 行 uniform FP16 4 anchor, 排除幸存者偏差行)
    # 用 normalized span = (max-min)/baseline 计算相对退化幅度
    _front_fp16 = df[
        (df["regime"] == "front") & (df["stage0_prec"] == "FP16") & df["mAOE"].notna() &
        (df["stage0_prec"] == df["stage1_prec"]) & (df["stage1_prec"] == df["stage2_prec"]) &
        (df["dataset_src"] == "complete_points_v1")  # 只用4 anchor基准行,避免E4/E6重复计算
    ].sort_values("prune_rate")
    if len(_front_fp16) >= 2:
        _base = _front_fp16.iloc[0]  # prune_rate=0, base
        _span_ap70_norm = (_front_fp16["ap70"].max() - _front_fp16["ap70"].min()) / _base["ap70"]
        _span_maoe_norm = (_front_fp16["mAOE"].max() - _front_fp16["mAOE"].min()) / _base["mAOE"]
        print(f"  [ISS-020 PASS] front FP16 4 anchor: "
              f"AP70 span={_span_ap70_norm:.3f}, mAOE span={_span_maoe_norm:.3f} "
              f"(剪枝约束信号, 非trade-off轴)")
else:
    print("[mAOE] 跳过: results/tp_errors_corrected_full.csv 不存在")

# ---------- 源 15: 闭环驾驶 sweep (Sim-D, cl_ 前缀) — gated ----------
# GO 门 (sim_test_design_v1.md §5): ① 路侧真接入 control ✅(§4.4) + ② ≥1 route DS 随时延可分辨 🔶待探针。
# 未达门前 pilot 行不入主表 (r0 八档 DS 全平=100, 场景太易非有效信号)。
# 时延补全 + GO 门达标后, 把 CL_INGEST 置 True 即自动接入 (列已就位)。
CL_INGEST = False
CL_SWEEP_CSV = Path("/home/jichengzhi/V2Xverse/results/closedloop_sweep_v1.csv")
if CL_INGEST and CL_SWEEP_CSV.exists():
    _cl = pd.read_csv(CL_SWEEP_CSV)
    _cl_rows = []
    for _, c in _cl.iterrows():
        if str(c["status"]) != "Completed":
            continue  # 质检门: 非 Completed 不入库
        _norsu = (str(c["arm"]) == "norsu")
        _row = {col: None for col in COLS}
        _row.update({
            "config_id": f"codriving_v2xverse_r{c['route_id']}_{c['arm']}",
            "model_class": "codriving_v2xverse",  # ★严禁与 pyramid_fusion/v2x_vit 混表
            "dataset_src": "closedloop_sweep_v1",
            "config_label": f"r{c['route_id']}_{c['arm']}",
            "latency_kind": "injected_sim_arm",   # 注入档 ≠ 实测 latency, lat_p50_ms 留空
            "regime": "sim_closedloop",            # 完全隔离感知 Pareto 候选池
            "is_real_measured": True,
            "ap_valid": False,                     # 闭环任务无 AP
            # 闭环元数据 (8)
            "latency_inject_ms": (None if _norsu else float(c["inject_ms"])),
            "latency_ms_source": "injected_from_E7",   # Orin E7 真测为注入值依据 (≠现场实测)
            "isolation": str(c.get("isolation", "single_card_shared")),
            "sim_route_id": int(c["route_id"]),
            "sim_arm": str(c["arm"]),
            "sim_route_set": "tau_ego_route_library_v1",  # ★演进自旧 6route: tau-ego 大路线库 (~105 条)
            "n_repeat": 1,
            "rsu_enabled": (not _norsu),           # ★norsu=单车 ego-only 消融基线(非部署模式); 学习视图按行剔除
            # 闭环性能 (15 cl_)
            "cl_driving_score": float(c["driving_score"]),
            "cl_route_completion": float(c["route_completion"]),
            "cl_infraction_penalty": float(c["infraction_penalty"]),
            "cl_collision_ped": float(c["collisions_pedestrian"]),
            "cl_collision_veh": float(c["collisions_vehicle"]),
            "cl_collision_layout": float(c["collisions_layout"]),
            "cl_red_light": float(c["red_light"]),
            "cl_outside_lanes": float(c["outside_route_lanes"]),
            "cl_zoh_age_mean": (None if _norsu else float(c["zoh_age_mean"])),
            "cl_delta_frames": (None if _norsu else float(c["delta_mean"])),
            "cl_zoh_held_frac": (None if _norsu else float(c["zoh_held_frac"])),
            "cl_audit_frames": int(c["audit_frames"]),
            "cl_duration_system_s": float(c["duration_system"]),
            "cl_route_length_m": float(c["route_length"]),
            "cl_status": str(c["status"]),
            "source": "cl:closedloop_sweep_v1(V2Xverse/results)",
            "notes": "Sim-D 闭环驾驶; 注入时延档; 与感知指标不同任务口径不可混比",
        })
        _cl_rows.append(_row)
    if _cl_rows:
        df = pd.concat([df, pd.DataFrame(_cl_rows, columns=COLS)], ignore_index=True)
    print(f"[closedloop] 接入 {len(_cl_rows)} 行 (CL_INGEST=True)")
else:
    print(f"[closedloop] 跳过接入 (CL_INGEST={CL_INGEST}); cl_ 列已就位, 等时延补全+GO门达标")

out_csv = ROOT / "multi_agent/data/dataset_v2.csv"
df.to_csv(out_csv, index=False)
df.to_parquet(ROOT / "multi_agent/data/dataset_v2.parquet")

# ---------- 精简学习视图: 每维度只留一个规范值 (供预测器/选择器学习) ----------
# 闭环只学 RSU-present 的 τ_ego→驾驶主曲线 → 剔除 norsu(单车 ego-only 消融基线)行。
# (norsu 完整保留在全表 dataset_v2 作 V2X 收益对照; mask 在删列前用全表 df 算, 同 index 对齐)
_norsu_mask = (df["regime"] == "sim_closedloop") & (df["rsu_enabled"] == False)
df_learn = df.drop(columns=[c for c in LEARN_DROP if c in df.columns])
df_learn = df_learn[~_norsu_mask.values].reset_index(drop=True)
learn_csv = ROOT / "multi_agent/data/dataset_v2_learning.csv"
df_learn.to_csv(learn_csv, index=False)
df_learn.to_parquet(ROOT / "multi_agent/data/dataset_v2_learning.parquet")
print(f"dataset_v2_learning: {df_learn.shape[0]} 行 x {df_learn.shape[1]} 列 "
      f"(全表 {df.shape[1]} - {len(LEARN_DROP)} 冗余指标列) -> {learn_csv}")

print(f"dataset_v2: {df.shape[0]} 行 x {df.shape[1]} 列 -> {out_csv}")
print("\n按完整度统计:")
has_lat = df["lat_p50_ms"].notna()
has_ap = df["ap50"].notna()
print(f"  完整点 (lat+AP): {(has_lat & has_ap).sum()}")
print(f"  仅 AP:           {(~has_lat & has_ap).sum()}")
print(f"  仅 latency:      {(has_lat & ~has_ap).sum()}")
print("\n按 dataset_src:")
print(df.groupby("dataset_src").size().to_string())
print("\n5 轴 Pareto 指标覆盖 (非空行数):")
for m, col in [("AP(ap70)", "ap70"), ("latency", "lat_p50_ms"),
               ("throughput", "throughput_fps"), ("energy", "energy_per_frame_mj"),
               ("model_size", "engine_size_mb")]:
    print(f"  {m:14s}: {df[col].notna().sum()}/{len(df)}")
print("\nthroughput_kind 分布:")
print(df["throughput_kind"].value_counts().to_string())
print(f"\n能耗行(energy 非空)= {df['energy_per_frame_mj'].notna().sum()}; "
      f"pipelined 吞吐行 = {(df['throughput_kind']=='pipelined').sum()}; "
      f"batched 吞吐行 = {(df['throughput_kind']=='batched').sum()}")
print(f"\nTP 误差指标覆盖 (ISS-020 PASS, 方案C):")
print(f"  mAOE 非空: {df['mAOE'].notna().sum()}/{len(df)}")
print(f"  mAOE_ci_lo/hi 非空: {df['mAOE_ci_lo'].notna().sum()}/{len(df)}")
print(f"  mAOE_basis 分布: {df['mAOE_basis'].value_counts().to_dict()}")
print(f"  forced行 mAOE非空(应为0): {df[df['q_mode'].str.contains('forced',na=False)]['mAOE'].notna().sum()}")
print(f"  Orin INT8 mAOE非空(应为0): {df[(df['hardware']=='orin_agx')&(df['stage0_prec']=='INT8')]['mAOE'].notna().sum()}")
_mao_front = df[(df['regime']=='front') & df['mAOE'].notna()]
print(f"  front 行 mAOE 覆盖: {len(_mao_front)} 行")
print(f"\n闭环 (Sim-D, cl_) 覆盖:")
_cl_mask = df['regime'] == 'sim_closedloop'
print(f"  sim_closedloop 行: {_cl_mask.sum()} (cl_driving_score 非空: {df['cl_driving_score'].notna().sum()})")
print(f"  全表 cl_ 性能列 (最全面): {sum(c.startswith('cl_') for c in COLS)} 列 + 8 元数据 = 23 闭环列; 时延补全后填")
print(f"  学习视图闭环只留: cl_driving_score + cl_route_completion (+latency_inject_ms 作τ特征), 剔 norsu 行")
print(f"\nmodel_class 分布:")
print(df['model_class'].value_counts().to_string())
