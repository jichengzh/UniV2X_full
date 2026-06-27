"""Stage1 子流程 B — 步骤 S5b: 逐层时延 profiling (内建工作流步骤).

动机 (用户 Q, 2026-06-15): partition manifest 原本是纯结构产物 (params/对齐/耦合),
无任何延迟信息 → 无法回答"动哪里最值(时间)"。本模块把"逐层 CUDA-Event 计时 +
按 manifest 自身单元(b2 量化单元 / b1 搜索旋钮 / bucket)聚合"做成 scan() 的内建步骤,
让每次重跑扫描自动产出时延旁注。方法学对齐 results/pyramid_m1_submodule_profile.json
(fp32 CUDA-Event chain hook timing)。

★ 诚实纪律:
  - 时延是**描述性旁注 (descriptive sidecar)**, 不参与合法性判定 (非约束)。这保持
    Stage1 "只做结构耦合, 性能反馈推迟到搜索回路"的叙事 (method §3.1)。
  - cuda → CUDA-Event 真测, status=ok; 须空闲卡 (util 0%/mem≤50MiB), profile 前快照
    GPU util/mem 写入 provenance, 占用高时 status 标 ok_but_contended 警告。
  - cpu → perf_counter 计时, status=estimated_cpu, 明确"不代表 GPU 部署延迟, 仅供
    相对结构参考"。
  - random weight / single-agent trace 核 / body-only (不含 encoder voxelize + NMS);
    ms 是相对优先级提示, 非部署 e2e。口径全部写进 provenance.note。
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from framework.stage1.graph_scan import _search_key  # 复用 b1 旋钮归并键

_RESULTS = Path("/home/jichengzhi/V2X/results")
_TOP_K = 15          # yaml 内只列前 K 个热点层, 全量明细落 sidecar json


# ---------------------------------------------------------------------------
# GPU 占用快照 (诚实: 记录测时环境)
# ---------------------------------------------------------------------------

def _gpu_snapshot(device: str) -> dict:
    if not str(device).startswith("cuda"):
        return {"device": device}
    idx = 0
    if ":" in str(device):
        try:
            idx = int(str(device).split(":")[1])
        except ValueError:
            idx = 0
    snap = {"device": device, "gpu_index": idx}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        name, util, mem = (c.strip() for c in out.split(","))
        snap.update({"gpu_name": name, "util_pct_at_start": int(util),
                     "mem_used_mib_at_start": int(mem)})
        snap["idle_ok"] = int(util) <= 5 and int(mem) <= 50
    except Exception as e:  # noqa: BLE001
        snap["snapshot_err"] = f"{type(e).__name__}: {e}"
    return snap


# ---------------------------------------------------------------------------
# 逐层计时 (forward hook + CUDA Event / perf_counter)
# ---------------------------------------------------------------------------

def _leaf_modules(net: nn.Module) -> list[tuple[str, nn.Module]]:
    out = []
    for n, m in net.named_modules():
        if n == "":
            continue
        if len(list(m.children())) == 0:   # 叶子模块 (无子模块) = 真正出 kernel 的算子
            out.append((n, m))
    return out


class _Timer:
    """逐叶子模块计时器: cuda 用 Event, cpu 用 perf_counter。"""

    def __init__(self, leaves, use_cuda: bool):
        self.leaves = leaves
        self.use_cuda = use_cuda
        self.totals: dict[str, float] = {n: 0.0 for n, _ in leaves}
        self.handles = []
        self._record = False        # warmup 阶段不累加
        self._pending = []           # cuda: 本次 forward 的 (name,start,end) 待 sync 后结算
        self._cpu_start: dict[str, float] = {}

    def _pre(self, name):
        def hook(_m, _inp):
            if not self._record:
                return
            if self.use_cuda:
                s = torch.cuda.Event(enable_timing=True)
                s.record()
                self._pending.append([name, s, None])
            else:
                self._cpu_start[name] = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(_m, _inp, _out):
            if not self._record:
                return
            if self.use_cuda:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                for rec in reversed(self._pending):
                    if rec[0] == name and rec[2] is None:
                        rec[2] = e
                        break
            else:
                t0 = self._cpu_start.get(name)
                if t0 is not None:
                    self.totals[name] += (time.perf_counter() - t0) * 1e3
        return hook

    def attach(self):
        for n, m in self.leaves:
            self.handles.append(m.register_forward_pre_hook(self._pre(n)))
            self.handles.append(m.register_forward_hook(self._post(n)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def settle_cuda_iter(self):
        """cuda: 一次 forward + sync 后, 结算 pending events 累加到 totals。"""
        if not self.use_cuda:
            return
        torch.cuda.synchronize()
        for name, s, e in self._pending:
            if e is not None:
                self.totals[name] += s.elapsed_time(e)   # ms
        self._pending = []


def _forward(net, x):
    return net(x) if not isinstance(x, (list, tuple)) else net(*x)


def _coverage(adapter, status: str) -> dict:
    skipped = adapter.typed_skipped_subgraphs() if hasattr(adapter, "typed_skipped_subgraphs") else []
    return {
        "coverage_scope": "trace_net_only",
        "trace_net_latency_pct": 100.0 if status != "skipped" else None,
        "full_model_latency_pct": None,
        "skipped_subgraphs_accounted_separately": True,
        "n_skipped_subgraphs": len(skipped),
        "note": (
            "Trace-net latency coverage only; this is not full-model coverage. "
            "Skipped sparse/fusion/attention/custom subgraphs are excluded from "
            "the hook-timed trace net and must be gated separately."
        ),
    }


def profile_latency(net: nn.Module, x, adapter, device: str,
                    warmup: int = 30, measure: int = 100) -> dict:
    """对已建好的 trace net 跑逐层计时, 返回 view_latency dict。

    复用调用方已建的 (net, x) — 即 scan() S0 那次 forward 的同一对象, 不重建模型。
    """
    use_cuda = str(device).startswith("cuda")
    snap = _gpu_snapshot(device)
    status = "ok" if use_cuda else "estimated_cpu"
    if use_cuda and not snap.get("idle_ok", True):
        status = "ok_but_contended"   # 占用高 → 数据可信度降级, 不阻断

    leaves = _leaf_modules(net)
    timer = _Timer(leaves, use_cuda)
    timer.attach()

    if use_cuda:
        e2e_s = torch.cuda.Event(enable_timing=True)
        e2e_e = torch.cuda.Event(enable_timing=True)

    e2e_total = 0.0
    try:
        with torch.no_grad():
            for _ in range(max(0, warmup)):           # warmup: 不记录
                _forward(net, x)
            if use_cuda:
                torch.cuda.synchronize()
            timer._record = True
            for _ in range(max(1, measure)):
                if use_cuda:
                    e2e_s.record()
                    _forward(net, x)
                    e2e_e.record()
                    timer.settle_cuda_iter()
                    e2e_total += e2e_s.elapsed_time(e2e_e)
                else:
                    t0 = time.perf_counter()
                    _forward(net, x)
                    e2e_total += (time.perf_counter() - t0) * 1e3
            timer._record = False
    finally:
        timer.detach()

    n = max(1, measure)
    per_layer_ms = {k: v / n for k, v in timer.totals.items()}
    e2e_ms = e2e_total / n
    attributed = sum(per_layer_ms.values())

    # 逐层 → bucket / b2 单元 / b1 旋钮 聚合
    by_bucket: dict[str, float] = {}
    by_knob: dict[str, float] = {}
    layer_rows = []
    for name, ms in per_layer_ms.items():
        mod = dict(net.named_modules()).get(name)
        op = type(mod).__name__ if mod is not None else "?"
        bucket = adapter.semantic_bucket(name)
        knob = _search_key(name, bucket)
        by_bucket[bucket] = by_bucket.get(bucket, 0.0) + ms
        by_knob[knob] = by_knob.get(knob, 0.0) + ms
        layer_rows.append({"layer": name, "op": op, "bucket": bucket,
                           "search_knob": knob, "latency_ms": round(ms, 4)})

    def _pct(v):
        return round(100.0 * v / e2e_ms, 1) if e2e_ms > 0 else None

    by_b2 = [{"unit": b, "latency_ms": round(ms, 4), "pct_of_e2e": _pct(ms)}
             for b, ms in sorted(by_bucket.items(), key=lambda kv: -kv[1])]
    by_b1 = [{"search_group_id": k, "latency_ms": round(ms, 4), "pct_of_e2e": _pct(ms)}
             for k, ms in sorted(by_knob.items(), key=lambda kv: -kv[1])]
    layer_rows.sort(key=lambda r: -r["latency_ms"])

    # 全量明细落 sidecar json (诚实: yaml 只留 top-K, 体积可控)
    sidecar = _RESULTS / f"stage1_latency_{adapter.name}.json"
    try:
        _RESULTS.mkdir(parents=True, exist_ok=True)
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump({"model": adapter.name, "status": status,
                       "provenance": snap, "warmup": warmup, "measure": measure,
                       "e2e_forward_ms": e2e_ms, "attributed_ms": attributed,
                       "per_layer": layer_rows}, f, ensure_ascii=False, indent=2)
        sidecar_path = str(sidecar)
    except Exception as e:  # noqa: BLE001
        sidecar_path = f"write_fail: {type(e).__name__}: {e}"

    return {
        "status": status,
        "provenance": {
            "method": "cuda_event_forward_hook" if use_cuda else "perf_counter_forward_hook",
            **snap,
            "dtype": "fp32",
            "warmup": warmup, "measure": measure,
            "input_shape": list(x.shape) if torch.is_tensor(x) else "multi",
            "note": ("描述性旁注, 非合法性约束 (Stage1 结构纯粹性不变); "
                     "random weight / single-agent trace 核 / body-only "
                     "(不含 encoder voxelize + NMS); ms 为相对优先级提示, 非部署 e2e。"
                     + (" ⚠️ CPU 计时不代表 GPU 部署延迟, 仅供相对结构参考。"
                        if not use_cuda else "")
                     + (" ⚠️ 测时 GPU 非空闲(util/mem 见快照), 可信度降级。"
                        if status == "ok_but_contended" else "")),
        },
        "e2e_forward_ms": round(e2e_ms, 4),
        "attributed_ms": round(attributed, 4),
        "unattributed_ms": round(max(0.0, e2e_ms - attributed), 4),
        "by_b2_quant_unit": by_b2,        # 对齐 view_b2_quant_units
        "by_b1_search_knob": by_b1,       # 对齐 view_b1_search_groups
        "top_layers": layer_rows[:_TOP_K],
        "full_detail_sidecar": sidecar_path,
        "coverage": _coverage(adapter, status),
    }


__all__ = ["profile_latency"]
