"""
Latency LUT 查询接口

从 calibration/latency_lut.json 查询给定配置的预估延迟/显存/能耗。
用于联合搜索中的 Level 1 廉价评估。

用法:
  from tools.query_lut import LatencyLUT
  lut = LatencyLUT('calibration/latency_lut.json')
  est = lut.estimate(d1_streams=2, d2_overlap='backbone_bev', d3_precision='int8', d3_frames=1, d4_strategy='defrag')
  print(est)  # {'latency_ms': ..., 'memory_mb': ..., 'energy_mj': ...}
"""
import json
import os
import sys

sys.path.insert(0, '/home/jichengzhi/UniV2X')


class LatencyLUT:
    """D 空间 Latency LUT 查询

    基于 calibration/latency_lut.json 中的实测数据，
    组合各维度的增量效应来预估给定配置的延迟/显存/能耗。

    预估模型 (线性叠加):
      latency = base_latency + delta_d1 + delta_d2 + delta_d3 + delta_d4
      memory  = base_memory  + delta_mem_d1 + delta_mem_d3 + delta_mem_d4
      energy  = latency * avg_power
    """

    def __init__(self, lut_path='calibration/latency_lut.json'):
        with open(lut_path) as f:
            self.data = json.load(f)

    def estimate(self, d1_streams=1, d2_overlap='none',
                 d3_precision='fp16', d3_frames=1,
                 d4_strategy='dynamic',
                 model_variant='baseline'):
        """预估给定 D 配置的延迟/显存/能耗

        Args:
            d1_streams: 1 或 2
            d2_overlap: 'none' 或 'backbone_bev'
            d3_precision: 'fp16' 或 'int8'
            d3_frames: 0, 1, 或 2
            d4_strategy: 'dynamic' 或 'defrag'
            model_variant: 'baseline' 或 'pruned_d14_enc10_dec07'

        Returns:
            dict with latency_ms, memory_mb, power_w, energy_mj
        """
        # D1: 多 stream
        d1_data = self.data.get('D1_multi_stream', {}).get(model_variant, {})
        d1_key = f'stream_{d1_streams}'
        d1_entry = d1_data.get(d1_key, d1_data.get('stream_1', {}))
        base_latency = d1_entry.get('latency_ms', 600)
        base_memory = d1_entry.get('peak_memory_mb', 2500)
        base_power = d1_entry.get('avg_power_w', 90)

        # D2: 流水线重叠 (理论值)
        d2_data = self.data.get('D2_pipeline_overlap', {}).get(model_variant, {})
        if d2_overlap == 'backbone_bev':
            d2_entry = d2_data.get('backbone_bev_overlap', {})
            theoretical_steady = d2_entry.get('theoretical_steady_state_ms', base_latency)
            # 实际延迟 = 理论稳态 + Python 开销
            # Python 开销 = actual - (backbone + non_backbone)
            actual = d2_data.get('no_overlap', {}).get('actual_latency_ms', base_latency)
            backbone = d2_entry.get('backbone_ms', 32)
            non_backbone = d2_entry.get('non_backbone_ms', 125)
            python_overhead = actual - backbone - non_backbone
            d2_latency = theoretical_steady + python_overhead
            d2_extra_mem = d2_entry.get('extra_memory_mb', 50) if 'extra_memory_mb' in d2_entry else 0
        else:
            d2_latency = base_latency
            d2_extra_mem = 0

        # D3: 时序缓存
        # 每帧 BEV 缓存: FP16=20.5MB, INT8=10.2MB
        per_frame_mb = 10.2 if d3_precision == 'int8' else 20.5
        d3_cache_mem = per_frame_mb * d3_frames

        # D4: 显存策略
        d4_data = self.data.get('D4_memory_strategy', {}).get('baseline', {})
        d4_entry = d4_data.get(d4_strategy, {})
        d4_latency_std = d4_entry.get('latency_std_ms', 30)
        d4_waste_pct = d4_entry.get('memory_waste_pct', 12)

        # 组合预估
        est_latency = d2_latency  # D2 重叠效应替代 base
        est_memory = base_memory + d2_extra_mem + d3_cache_mem
        est_power = base_power
        est_energy = est_latency * est_power  # ms * W = mJ

        return {
            'latency_ms': round(est_latency, 1),
            'latency_std_ms': round(d4_latency_std, 1),
            'memory_mb': round(est_memory, 1),
            'cache_mb': round(d3_cache_mem, 1),
            'power_w': round(est_power, 1),
            'energy_mj': round(est_energy, 0),
            'memory_waste_pct': round(d4_waste_pct, 1),
            'config': {
                'd1_streams': d1_streams,
                'd2_overlap': d2_overlap,
                'd3_precision': d3_precision,
                'd3_frames': d3_frames,
                'd4_strategy': d4_strategy,
                'model': model_variant,
            }
        }

    def estimate_all_d_configs(self, model_variant='baseline'):
        """枚举所有 D 配置的预估值"""
        results = []
        for d1 in [1, 2]:
            for d2 in ['none', 'backbone_bev']:
                for d3_prec in ['fp16', 'int8']:
                    for d3_frames in [0, 1, 2]:
                        if d3_prec == 'int8' and d3_frames == 0:
                            continue  # INT8 + 0帧无意义
                        for d4 in ['dynamic', 'defrag']:
                            est = self.estimate(
                                d1_streams=d1, d2_overlap=d2,
                                d3_precision=d3_prec, d3_frames=d3_frames,
                                d4_strategy=d4, model_variant=model_variant
                            )
                            results.append(est)
        return results


if __name__ == '__main__':
    lut = LatencyLUT()

    print("=== Single query ===")
    est = lut.estimate(d1_streams=1, d2_overlap='backbone_bev',
                       d3_precision='int8', d3_frames=1, d4_strategy='defrag')
    print(json.dumps(est, indent=2))

    print("\n=== All D configs (baseline) ===")
    all_configs = lut.estimate_all_d_configs()
    print(f"Total configs: {len(all_configs)}")

    # 按 latency 排序显示 Top-5
    sorted_configs = sorted(all_configs, key=lambda x: x['latency_ms'])
    print("\nTop-5 lowest latency:")
    for c in sorted_configs[:5]:
        cfg = c['config']
        print(f"  {cfg['d1_streams']}stream/{cfg['d2_overlap']}/{cfg['d3_precision']}-{cfg['d3_frames']}f/{cfg['d4_strategy']}"
              f" → lat={c['latency_ms']}ms, mem={c['memory_mb']}MB, energy={c['energy_mj']}mJ")

    # 按 memory 排序显示 Top-5
    sorted_mem = sorted(all_configs, key=lambda x: x['memory_mb'])
    print("\nTop-5 lowest memory:")
    for c in sorted_mem[:5]:
        cfg = c['config']
        print(f"  {cfg['d1_streams']}stream/{cfg['d2_overlap']}/{cfg['d3_precision']}-{cfg['d3_frames']}f/{cfg['d4_strategy']}"
              f" → lat={c['latency_ms']}ms, mem={c['memory_mb']}MB, energy={c['energy_mj']}mJ")
