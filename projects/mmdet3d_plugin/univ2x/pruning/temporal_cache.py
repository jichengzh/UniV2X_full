"""
D3 时序缓存管理器

控制 prev_bev 的存储精度（FP16/INT8）和帧数（0/1/2），
用于 D 空间搜索中评估不同缓存策略对精度/存储的 trade-off。

与 B2 量化的区别:
  - B2 控制模型层的计算精度（forward 中权重和激活的位宽）
  - D3 控制帧间缓存的存储精度（BEV encoder 输出后、下一帧使用前的存储方式）

使用方式:
    from projects.mmdet3d_plugin.univ2x.pruning.temporal_cache import TemporalCacheManager

    cache_mgr = TemporalCacheManager(precision='int8', max_frames=2)

    # 在 univ2x_track.py 的 forward_test 中:
    # 原: self.prev_bev = frame_res["bev_embed"]
    # 改: cache_mgr.store(frame_res["bev_embed"])
    #     self.prev_bev = cache_mgr.retrieve()
"""
import torch


class TemporalCacheManager:
    """时序 BEV 缓存管理器

    Args:
        precision: 'fp16' 或 'int8'，控制缓存存储精度
        max_frames: 最大缓存帧数（0=不缓存, 1=前一帧, 2=前两帧）
        calibrate_scale: 是否在首帧自动校准 INT8 scale（否则使用固定 scale）
        fixed_scale: 固定的量化 scale（仅在 calibrate_scale=False 时使用）
    """

    def __init__(self, precision='fp16', max_frames=1,
                 calibrate_scale=True, fixed_scale=None):
        self.precision = precision
        self.max_frames = max_frames
        self.calibrate_scale = calibrate_scale
        self.cache = []  # list of cached BEV tensors
        self.scale = fixed_scale  # INT8 量化 scale (per-tensor)
        self.n_stored = 0  # 累计存储次数
        self._memory_saved_bytes = 0  # 累计节省的显存

    def store(self, bev):
        """BEV encoder 输出后调用，将 bev 存入缓存

        Args:
            bev: BEV 特征 tensor，shape 通常为 (H*W, B, C) 或 (B, H*W, C)
                 数据类型为 FP16 或 FP32
        """
        if self.max_frames == 0:
            return

        if self.precision == 'int8':
            bev_cached = self._quantize_to_int8(bev)
        else:
            # FP16 存储 (如果输入是 FP32 则降精度)
            bev_cached = bev.half() if bev.dtype == torch.float32 else bev.clone()

        self.cache.append(bev_cached)
        self.n_stored += 1

        # 超出 max_frames 则丢弃最旧的
        while len(self.cache) > self.max_frames:
            self.cache.pop(0)

    def retrieve(self):
        """下一帧 TemporalSelfAttention 使用前调用

        Returns:
            最近一帧的 BEV 特征 (反量化为原始精度), 或 None (无缓存)
        """
        if not self.cache or self.max_frames == 0:
            return None

        # 返回最近一帧 (index -1)
        cached = self.cache[-1]

        if self.precision == 'int8':
            return self._dequantize_from_int8(cached)
        else:
            return cached

    def retrieve_all(self):
        """返回所有缓存帧 (按时间从旧到新)

        Returns:
            list of BEV tensors, 或空列表
        """
        if not self.cache or self.max_frames == 0:
            return []

        result = []
        for cached in self.cache:
            if self.precision == 'int8':
                result.append(self._dequantize_from_int8(cached))
            else:
                result.append(cached)
        return result

    def reset(self):
        """重置缓存（场景切换时调用）"""
        self.cache.clear()
        self.scale = None if self.calibrate_scale else self.scale

    def _quantize_to_int8(self, bev):
        """对称 INT8 量化 (per-tensor)"""
        with torch.no_grad():
            if self.calibrate_scale and self.scale is None:
                # 首帧校准 scale
                abs_max = bev.abs().amax()
                self.scale = abs_max / 127.0
                if self.scale == 0:
                    self.scale = torch.tensor(1.0, device=bev.device)

            # 量化
            bev_int8 = torch.clamp(
                torch.round(bev.float() / self.scale), -127, 127
            ).to(torch.int8)

            # 记录节省的显存
            orig_bytes = bev.nelement() * bev.element_size()
            new_bytes = bev_int8.nelement() * 1  # INT8 = 1 byte
            self._memory_saved_bytes += (orig_bytes - new_bytes)

            return bev_int8

    def _dequantize_from_int8(self, bev_int8):
        """INT8 反量化为 FP16"""
        return bev_int8.half() * self.scale.half()

    @property
    def memory_stats(self):
        """返回显存统计"""
        cache_bytes = 0
        for c in self.cache:
            cache_bytes += c.nelement() * (1 if c.dtype == torch.int8 else c.element_size())
        return {
            'num_cached_frames': len(self.cache),
            'max_frames': self.max_frames,
            'precision': self.precision,
            'cache_size_mb': round(cache_bytes / (1024 ** 2), 2),
            'total_memory_saved_mb': round(self._memory_saved_bytes / (1024 ** 2), 2),
            'n_stored': self.n_stored,
        }

    def __repr__(self):
        stats = self.memory_stats
        return (f"TemporalCacheManager(precision={self.precision}, "
                f"max_frames={self.max_frames}, "
                f"cached={stats['num_cached_frames']}, "
                f"cache_size={stats['cache_size_mb']}MB)")
