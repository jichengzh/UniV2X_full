"""通用 latency 测量工具 (阶段 1.3)

为阶段 1 L3 设计:在任意 nn.Module 上测 latency, 支持:
- P50/P95/P99 + 标准差
- 模块级分布(forward hook 拆解 backbone/encoder/decoder/heads)
- 10+ 次重复 + warmup 控制 + CUDA Event 同步
- 剪枝前后对比 (estimate_speedup)

设计原则:
- 不依赖具体模型(只要是 nn.Module 都能测)
- 不依赖 mmcv/mmdet 的 build_model(直接接受 model + dummy input)
- 复用 1.3 已有的 latency_lut.json 字段命名(latency_ms / peak_memory_mb / ...)

用法示例:
    from tools.measure_latency import LatencyBenchmark
    bench = LatencyBenchmark(n_warmup=10, n_runs=20, device='cuda')
    metrics = bench.run(model, dummy_input)         # 整体 latency
    module_metrics = bench.run_per_module(model, dummy_input,
                                           module_names=['backbone', 'encoder', 'decoder'])
"""

from __future__ import annotations

import contextlib
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Any, Optional

import torch
import torch.nn as nn


# ============================================================
# 数据结构
# ============================================================

@dataclass
class LatencyMetrics:
    """单次 latency 测量结果."""

    n_runs: int
    n_warmup: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    peak_memory_mb: Optional[float] = None
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"LatencyMetrics(P50={self.p50_ms:.2f}ms P95={self.p95_ms:.2f}ms "
            f"mean={self.mean_ms:.2f}±{self.std_ms:.2f}ms n={self.n_runs})"
        )


@dataclass
class ModuleLatency:
    """单个模块的 latency 切片."""

    module_name: str
    metrics: LatencyMetrics
    pct_of_total: float  # 占整体 latency 的百分比


# ============================================================
# 主类
# ============================================================

class LatencyBenchmark:
    """通用 latency 测量器."""

    def __init__(
        self,
        n_warmup: int = 10,
        n_runs: int = 20,
        device: str = "cuda",
        sync_each_run: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            n_warmup: 预热次数 (CUDA 上首次推理可能编译/选 tactic, 不计入)
            n_runs: 测量次数 (10+ 才能稳定 P95)
            device: 'cuda' 或 'cuda:0' / 'cpu'
            sync_each_run: 每次推理后是否 cuda.synchronize (确保独立测时)
            verbose: 是否打印进度
        """
        self.n_warmup = n_warmup
        self.n_runs = n_runs
        self.device = device
        self.sync_each_run = sync_each_run
        self.verbose = verbose
        self.is_cuda = "cuda" in str(device)

    # ---------------- 核心计时 ----------------

    def _time_one_run(self, model: nn.Module, inp: Any) -> float:
        """单次前向, 返回毫秒. CUDA 上用 Event, CPU 上用 perf_counter."""
        if self.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            with torch.no_grad():
                _ = model(inp) if not isinstance(inp, dict) else model(**inp)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)  # ms
        else:
            import time
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(inp) if not isinstance(inp, dict) else model(**inp)
            return (time.perf_counter() - t0) * 1000

    def _aggregate(self, samples: list[float]) -> LatencyMetrics:
        """从样本数组聚合统计指标."""
        s = sorted(samples)
        n = len(s)
        return LatencyMetrics(
            n_runs=n,
            n_warmup=self.n_warmup,
            mean_ms=float(statistics.mean(s)),
            std_ms=float(statistics.stdev(s) if n > 1 else 0.0),
            p50_ms=float(s[n // 2]),
            p90_ms=float(s[int(n * 0.90)] if n > 10 else s[-1]),
            p95_ms=float(s[int(n * 0.95)] if n > 20 else s[-1]),
            p99_ms=float(s[int(n * 0.99)] if n > 100 else s[-1]),
            min_ms=float(s[0]),
            max_ms=float(s[-1]),
            device=str(self.device),
        )

    # ---------------- 公共 API ----------------

    def run(self, model: nn.Module, dummy_input: Any) -> LatencyMetrics:
        """整体 latency 测量."""
        model = model.to(self.device).eval()
        # 输入也放到设备
        dummy_input = self._to_device(dummy_input)

        # warmup
        if self.verbose:
            print(f"[warmup] {self.n_warmup} runs ...")
        for _ in range(self.n_warmup):
            self._time_one_run(model, dummy_input)

        # 主测量 + 显存峰值
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        samples: list[float] = []
        for i in range(self.n_runs):
            t = self._time_one_run(model, dummy_input)
            samples.append(t)
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  run {i+1}/{self.n_runs}: {t:.2f}ms")

        metrics = self._aggregate(samples)
        if self.is_cuda:
            metrics.peak_memory_mb = float(
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
            )
        return metrics

    def run_per_module(
        self,
        model: nn.Module,
        dummy_input: Any,
        module_names: list[str],
    ) -> tuple[LatencyMetrics, list[ModuleLatency]]:
        """模块级 latency 分布(用 forward hook).

        Args:
            module_names: 形如 ['backbone', 'encoder', 'decoder'] 的子模块名字
                         (会用 model.get_submodule(name) 取)

        Returns:
            (whole_metrics, [ModuleLatency, ...]) 整体指标 + 每模块切片
        """
        model = model.to(self.device).eval()
        dummy_input = self._to_device(dummy_input)

        # 找到对应子模块
        target_modules: dict[str, nn.Module] = {}
        for name in module_names:
            try:
                target_modules[name] = model.get_submodule(name)
            except AttributeError:
                if self.verbose:
                    print(f"[warn] module '{name}' not found, skipped")

        # 为每个目标模块挂 hook 收集 latency
        module_samples: dict[str, list[float]] = {n: [] for n in target_modules}
        events: dict[str, tuple] = {}

        def make_hooks(name: str):
            def pre_hook(_mod, _inp):
                if self.is_cuda:
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    events[name] = (s, e)
                else:
                    import time
                    events[name] = (time.perf_counter(), None)

            def post_hook(_mod, _inp, _out):
                if self.is_cuda:
                    s, e = events[name]
                    e.record()
                    torch.cuda.synchronize()
                    module_samples[name].append(s.elapsed_time(e))
                else:
                    import time
                    t0, _ = events[name]
                    module_samples[name].append((time.perf_counter() - t0) * 1000)

            return pre_hook, post_hook

        # 注册 hook
        handles = []
        for name, mod in target_modules.items():
            pre, post = make_hooks(name)
            handles.append(mod.register_forward_pre_hook(pre))
            handles.append(mod.register_forward_hook(post))

        try:
            # 整体测量(同时也收集模块级)
            whole_metrics = self.run(model, dummy_input)
        finally:
            for h in handles:
                h.remove()

        # 聚合每模块
        module_results: list[ModuleLatency] = []
        for name in module_names:
            if name not in module_samples or not module_samples[name]:
                continue
            samples = module_samples[name][-self.n_runs:]  # 去掉 warmup
            m = self._aggregate(samples)
            pct = (m.p50_ms / whole_metrics.p50_ms * 100) if whole_metrics.p50_ms > 0 else 0
            module_results.append(ModuleLatency(name, m, pct))

        return whole_metrics, module_results

    def estimate_speedup(
        self,
        model_a: nn.Module,
        model_b: nn.Module,
        dummy_input: Any,
        label_a: str = "baseline",
        label_b: str = "pruned",
    ) -> dict:
        """对比两个模型 (剪枝前/后) 的加速比."""
        if self.verbose:
            print(f"[bench] {label_a} ...")
        ma = self.run(model_a, dummy_input)
        if self.verbose:
            print(f"[bench] {label_b} ...")
        mb = self.run(model_b, dummy_input)

        speedup_p50 = ma.p50_ms / mb.p50_ms if mb.p50_ms > 0 else float("nan")
        delta_p50 = (ma.p50_ms - mb.p50_ms) / ma.p50_ms * 100 if ma.p50_ms > 0 else 0

        return {
            label_a: ma.to_dict(),
            label_b: mb.to_dict(),
            "speedup_p50": speedup_p50,
            "delta_p50_pct": delta_p50,
        }

    # ---------------- 工具 ----------------

    def _to_device(self, inp: Any) -> Any:
        """把任意输入(tensor/list/dict)放到设备上."""
        if isinstance(inp, torch.Tensor):
            return inp.to(self.device)
        if isinstance(inp, dict):
            return {k: self._to_device(v) for k, v in inp.items()}
        if isinstance(inp, (list, tuple)):
            return type(inp)(self._to_device(x) for x in inp)
        return inp


# ============================================================
# CLI
# ============================================================

def _build_dummy_resnet50(input_size=(1, 3, 800, 450)) -> tuple[nn.Module, torch.Tensor]:
    """sanity 测试用: 标准 R50 + 800×450 输入."""
    import torchvision.models as tvm
    model = tvm.resnet50(weights=None)
    return model, torch.randn(*input_size)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="通用 latency 测量")
    p.add_argument("--model", default="resnet50_dummy", help="目前仅支持 resnet50_dummy")
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-runs", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--per-module", action="store_true", help="启用模块级分解")
    p.add_argument("--input-size", type=int, nargs=4, default=[1, 3, 800, 450])
    p.add_argument("--output", default=None, help="可选: 输出 json 文件")
    args = p.parse_args()

    if args.model == "resnet50_dummy":
        model, inp = _build_dummy_resnet50(tuple(args.input_size))
    else:
        raise NotImplementedError(f"model={args.model} 暂未实现")

    bench = LatencyBenchmark(
        n_warmup=args.n_warmup,
        n_runs=args.n_runs,
        device=args.device,
        verbose=True,
    )

    print(f"\n=== Latency benchmark: {args.model} {tuple(args.input_size)} on {args.device} ===")

    if args.per_module:
        # R50 子模块: layer1-4 + fc
        modules = ["layer1", "layer2", "layer3", "layer4", "fc"]
        whole, per = bench.run_per_module(model, inp, modules)
        print(f"\n[whole] {whole}")
        print(f"\n[per-module] (按 P50 占比降序)")
        per_sorted = sorted(per, key=lambda m: -m.pct_of_total)
        for ml in per_sorted:
            print(f"  {ml.module_name:10s}  P50={ml.metrics.p50_ms:6.2f}ms  P95={ml.metrics.p95_ms:6.2f}ms  ({ml.pct_of_total:.1f}% of whole)")
        result = {"whole": whole.to_dict(), "per_module": [{"name": m.module_name, "pct": m.pct_of_total, **m.metrics.to_dict()} for m in per]}
    else:
        whole = bench.run(model, inp)
        print(f"\n[whole] {whole}")
        result = {"whole": whole.to_dict()}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
