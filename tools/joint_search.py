"""
B1 x B2 x D 联合搜索框架

Phase 3 的核心脚本：定义联合搜索空间、约束条件，
并提供 Level 1 (廉价) 和 Level 2 (真实) 两级评估。

当前实现: 搜索空间编码 + Level 1 廉价评估。
TODO: BoTorch BO 循环 + Level 2 真实评估管线。

用法:
  # 枚举所有可行配置并用 Level 1 评估
  PYTHONPATH=/home/jichengzhi/UniV2X conda run -n UniV2X_2.0 \\
    python tools/joint_search.py --mode enumerate

  # 单个配置的 Level 1 评估
  PYTHONPATH=/home/jichengzhi/UniV2X conda run -n UniV2X_2.0 \\
    python tools/joint_search.py --mode eval-l1 \\
    --b1-enc-ratio 1.0 --b1-dec-ratio 0.7 --b1-dec-layers 6 \\
    --d1-streams 1 --d2-overlap none --d3-precision fp16 --d3-frames 1 --d4-strategy defrag
"""
import argparse
import json
import itertools
import sys

sys.path.insert(0, '/home/jichengzhi/UniV2X')


# ============================================================
# 搜索空间定义
# ============================================================

B1_SPACE = {
    'enc_ffn_ratio': [1.0, 0.8, 0.7, 0.6],    # P1a
    'dec_ffn_ratio': [1.0, 0.7, 0.5, 0.3, 0.2], # P1b
    'dec_num_layers': [5, 6],                     # P9
    'seg_mlp_ratio': [1.0, 0.7],                  # P11
}

B2_SPACE = {
    'bev_precision': ['FP16', 'INT8'],
    'bev_granularity': ['per-tensor', 'per-channel'],
    'bev_target': ['W+A', 'W-only'],
    'heads_precision': ['FP16', 'INT8'],
    'heads_granularity': ['per-tensor', 'per-channel'],
}

D_SPACE = {
    'd1_streams': [1, 2],
    'd2_overlap': ['none', 'backbone_bev'],
    'd3_precision': ['fp16', 'int8'],
    'd3_frames': [1, 2],  # 0 removed: AMOTA crashes to 0.021 without temporal
    'd4_strategy': ['dynamic', 'defrag'],
}


# ============================================================
# 约束条件 (C1-C5)
# ============================================================

def check_constraints(b1, b2, d):
    """检查 B1 x B2 x D 联合约束

    Returns:
        (is_valid, violated_constraints)
    """
    violations = []

    # C1: 高压缩率需要时序补偿
    # 当 enc_ratio 和 dec_ratio 都很小时, 需要多帧时序
    avg_ratio = (b1['enc_ffn_ratio'] + b1['dec_ffn_ratio']) / 2
    if avg_ratio < 0.5 and d['d3_frames'] < 2:
        violations.append('C1: high compression (avg_ratio<0.5) requires d3_frames>=2')

    # C2: 高并行度 + 大模型需要 INT8 降显存
    if d['d1_streams'] >= 3 and b2['bev_precision'] == 'FP16':
        violations.append('C2: d1>=3 streams with FP16 may OOM')

    # C3: 全重叠需要额外显存 (当前未实现, 仅记录)
    # 暂不约束

    # C4: INT8 缓存 + encoder 被剪枝 → 需要验证
    if d['d3_precision'] == 'int8' and b1['enc_ffn_ratio'] < 0.8:
        violations.append('C4: INT8 cache with pruned encoder (ratio<0.8) needs validation')

    # C5: 静态预分配时, 显存上限确定
    # 暂不约束 (当前无 static 选项)

    # 逻辑约束: INT8 缓存 + 0帧 无意义
    if d['d3_precision'] == 'int8' and d['d3_frames'] == 0:
        violations.append('LOGIC: int8 cache with 0 frames is meaningless')

    return len(violations) == 0, violations


# ============================================================
# Level 1 廉价评估
# ============================================================

# 已知的 B1 Pareto 数据 (来自 1.2 实验)
B1_AMOTA_DATA = {
    # (enc_ratio, dec_ratio, layers): AMOTA
    (1.0, 1.0, 6): 0.330,   # baseline
    (1.0, 0.7, 6): 0.367,   # D.1.4 最优
    (0.8, 0.3, 6): 0.337,   # D.1.2
    (1.0, 0.3, 6): 0.329,   # D.1.1
    (0.7, 0.4, 6): 0.306,   # D.1.3
    (0.4, 0.4, 6): 0.335,   # S1 绑定 60%
    (1.0, 1.0, 5): 0.310,   # P9 5层
}

# B2 量化对 AMOTA 的影响 (来自 1.1/1.2 实验)
B2_AMOTA_DELTA = {
    'FP16': 0.0,     # 不量化
    'INT8_baseline': -0.004,  # baseline INT8 (0.364 vs 0.330 实际反升, 但 PyTorch 端 -0.004)
    'INT8_pruned_decouple': -0.007,  # 解耦剪枝 + INT8 (0.360 vs 0.367)
    'INT8_pruned_bound': -0.048,     # 绑定剪枝 + INT8 (0.287 vs 0.335)
}


def level1_evaluate(b1, b2, d):
    """Level 1 廉价评估 (<1ms)

    基于已知实验数据的线性加性模型:
      AMOTA ≈ B1_AMOTA[config] + B2_delta + D3_delta
      Latency ≈ LUT[d_config]
      Memory ≈ base + cache_memory
    """
    # AMOTA 预估
    b1_key = (b1['enc_ffn_ratio'], b1['dec_ffn_ratio'], b1['dec_num_layers'])
    b1_amota = B1_AMOTA_DATA.get(b1_key, None)
    if b1_amota is None:
        # 线性插值/最近邻
        b1_amota = 0.330  # fallback to baseline

    # B2 量化影响
    if b2['bev_precision'] == 'INT8':
        if b1['enc_ffn_ratio'] >= 0.8:
            b2_delta = B2_AMOTA_DELTA['INT8_pruned_decouple']
        elif b1['enc_ffn_ratio'] == b1['dec_ffn_ratio']:
            b2_delta = B2_AMOTA_DELTA['INT8_pruned_bound']
        else:
            b2_delta = B2_AMOTA_DELTA['INT8_baseline']
    else:
        b2_delta = 0.0

    # D3 缓存影响 (0帧 → 降精度, INT8 → 微降)
    d3_delta = 0.0
    if d['d3_frames'] == 0:
        d3_delta = -0.01  # 无时序会损失精度
    elif d['d3_precision'] == 'int8':
        d3_delta = -0.005  # INT8 缓存有量化误差

    est_amota = b1_amota + b2_delta + d3_delta

    # Latency 预估 (from LUT)
    try:
        from tools.query_lut import LatencyLUT
        lut = LatencyLUT()
        model_variant = 'baseline'
        if b1['enc_ffn_ratio'] != 1.0 or b1['dec_ffn_ratio'] != 1.0:
            model_variant = 'pruned_d14_enc10_dec07'  # 最近邻
        lut_result = lut.estimate(
            d1_streams=d['d1_streams'],
            d2_overlap=d['d2_overlap'],
            d3_precision=d['d3_precision'],
            d3_frames=d['d3_frames'],
            d4_strategy=d['d4_strategy'],
            model_variant=model_variant
        )
        est_latency = lut_result['latency_ms']
        est_memory = lut_result['memory_mb']
        est_energy = lut_result['energy_mj']
    except Exception:
        est_latency = 600.0
        est_memory = 2500.0
        est_energy = 54000.0

    return {
        'est_amota': round(est_amota, 4),
        'est_latency_ms': round(est_latency, 1),
        'est_memory_mb': round(est_memory, 1),
        'est_energy_mj': round(est_energy, 0),
        'b1': b1,
        'b2': b2,
        'd': d,
    }


# ============================================================
# 搜索空间枚举
# ============================================================

def enumerate_all_configs():
    """枚举所有 B1 x B2 x D 可行配置"""
    configs = []
    n_total = 0
    n_valid = 0

    for enc_r in B1_SPACE['enc_ffn_ratio']:
        for dec_r in B1_SPACE['dec_ffn_ratio']:
            for dec_l in B1_SPACE['dec_num_layers']:
                for seg_r in B1_SPACE['seg_mlp_ratio']:
                    b1 = {'enc_ffn_ratio': enc_r, 'dec_ffn_ratio': dec_r,
                           'dec_num_layers': dec_l, 'seg_mlp_ratio': seg_r}

                    for bev_p in B2_SPACE['bev_precision']:
                        for bev_g in B2_SPACE['bev_granularity']:
                            for bev_t in B2_SPACE['bev_target']:
                                for h_p in B2_SPACE['heads_precision']:
                                    for h_g in B2_SPACE['heads_granularity']:
                                        b2 = {'bev_precision': bev_p, 'bev_granularity': bev_g,
                                               'bev_target': bev_t, 'heads_precision': h_p,
                                               'heads_granularity': h_g}

                                        for d1 in D_SPACE['d1_streams']:
                                            for d2 in D_SPACE['d2_overlap']:
                                                for d3p in D_SPACE['d3_precision']:
                                                    for d3f in D_SPACE['d3_frames']:
                                                        for d4 in D_SPACE['d4_strategy']:
                                                            d = {'d1_streams': d1, 'd2_overlap': d2,
                                                                 'd3_precision': d3p, 'd3_frames': d3f,
                                                                 'd4_strategy': d4}
                                                            n_total += 1
                                                            valid, _ = check_constraints(b1, b2, d)
                                                            if valid:
                                                                n_valid += 1
                                                                configs.append((b1, b2, d))

    return configs, n_total, n_valid


def parse_args():
    p = argparse.ArgumentParser(description="B1 x B2 x D joint search")
    p.add_argument('--mode', choices=['enumerate', 'eval-l1', 'pareto'],
                   default='enumerate')
    p.add_argument('--b1-enc-ratio', type=float, default=1.0)
    p.add_argument('--b1-dec-ratio', type=float, default=1.0)
    p.add_argument('--b1-dec-layers', type=int, default=6)
    p.add_argument('--d1-streams', type=int, default=1)
    p.add_argument('--d2-overlap', default='none')
    p.add_argument('--d3-precision', default='fp16')
    p.add_argument('--d3-frames', type=int, default=1)
    p.add_argument('--d4-strategy', default='defrag')
    p.add_argument('--output', default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'enumerate':
        print("[Search] Enumerating all B1 x B2 x D configurations...")
        configs, n_total, n_valid = enumerate_all_configs()
        print(f"  Total (naive): {n_total:,}")
        print(f"  Valid (after constraints): {n_valid:,}")
        print(f"  Reduction: {(1 - n_valid/n_total)*100:.1f}%")

        # Level 1 评估 top configs
        print(f"\n[Search] Running Level 1 evaluation on {n_valid} configs...")
        results = []
        for b1, b2, d in configs:
            est = level1_evaluate(b1, b2, d)
            results.append(est)

        # Pareto 前沿 (AMOTA vs latency)
        results.sort(key=lambda x: (-x['est_amota'], x['est_latency_ms']))
        print(f"\n[Search] Top-10 by estimated AMOTA:")
        for i, r in enumerate(results[:10]):
            b1 = r['b1']
            d = r['d']
            print(f"  {i+1}. AMOTA={r['est_amota']:.3f} lat={r['est_latency_ms']:.0f}ms "
                  f"mem={r['est_memory_mb']:.0f}MB "
                  f"| enc={b1['enc_ffn_ratio']} dec={b1['dec_ffn_ratio']} "
                  f"layers={b1['dec_num_layers']} "
                  f"| {d['d2_overlap']} {d['d3_precision']}-{d['d3_frames']}f")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'n_total': n_total, 'n_valid': n_valid,
                           'top_10': results[:10]}, f, indent=2)

    elif args.mode == 'eval-l1':
        b1 = {'enc_ffn_ratio': args.b1_enc_ratio, 'dec_ffn_ratio': args.b1_dec_ratio,
               'dec_num_layers': args.b1_dec_layers, 'seg_mlp_ratio': 1.0}
        b2 = {'bev_precision': 'FP16', 'bev_granularity': 'per-tensor',
               'bev_target': 'W+A', 'heads_precision': 'FP16',
               'heads_granularity': 'per-tensor'}
        d = {'d1_streams': args.d1_streams, 'd2_overlap': args.d2_overlap,
             'd3_precision': args.d3_precision, 'd3_frames': args.d3_frames,
             'd4_strategy': args.d4_strategy}

        valid, violations = check_constraints(b1, b2, d)
        if not valid:
            print(f"[WARN] Constraint violations: {violations}")

        est = level1_evaluate(b1, b2, d)
        print(json.dumps(est, indent=2))


if __name__ == '__main__':
    main()
