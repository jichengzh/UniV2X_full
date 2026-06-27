"""Three-arm multi-objective search kernel — prune-width (P) × schedule (S).

B3 of the three-arm ablation (see
multi_agent/methods/design/auto-tuning/4_design_ablation_proof_v1.md).

This is a REAL SEARCH (NSGA-II on a cost model), NOT an enumeration of known
points.  The three arms share ONE explorer kernel, ONE cost model, and ONE total
evaluation budget; they differ ONLY in search-space structure:

  A-joint  : search P × S jointly.
  A-noS    : search P only; S frozen to `default` (dlight) always.
  A-serial : stage 1 = search P under DEFAULT schedule and LOCK the resulting
             default-Pareto widths (single-point lock = strongest local optimum;
             Top-k lock = relaxed variant); stage 2 = only then tune S on the
             locked widths.  Widths discarded in stage 1 (e.g. pad64, dominated
             under default by trap25) are NEVER tuned -> their tuned branch is
             structurally unreachable.

DISCIPLINE (doc4 §3, non-negotiable):
  * The global / reference Pareto is HIDDEN from every arm.  Each arm sees only
    the cost model (latency LUT + AP model) and a finite eval budget.
  * A reference Pareto + a fixed HV nadir are computed OFFLINE purely to SCORE
    hypervolume; they never enter any arm's search.
  * A-serial is a STRONG baseline: each stage uses the genuine best-available
    choice (default-Pareto front of widths, not a deliberately bad pick), so its
    failure is attributable to serial *structure*, not a weak baseline.

Interfaces (forward-compatible with B1 / B2 outputs, with seed fallbacks):
  * LatencyLUT.latency(width, sched) -> microseconds.
      Reads results/latency_lut_pyramid.json (B1) if present (direct-grid or
      additive per-stage format); else falls back to the 8 seed points in
      gap1_grid_corrected.json (exact width match).
  * APModel.ap70(width) -> float.
      Reads an AP-model file (B2) if present; else falls back to the known AP70
      anchors with monotone interpolation by total backbone channels.

Pure Python, no GPU dependency, no third-party deps.  Importable; run as
`python -m framework.search_three_arm` (or directly) for the smoke test.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Optional closed-loop driving-score axis (B7 / Agent-CL). Lazy/guarded so the
# kernel still imports if the helper or its data is absent.
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from scripts.phase2.closedloop_objective_query import (
        backbone_to_e2e_latency as _cl_e2e,
        driving_score as _cl_ds,
    )
    _HAS_CL = True
except Exception:                                          # pragma: no cover
    _HAS_CL = False

SEED_GRID_JSON = RESULTS / "gap1_grid_corrected.json"
LATENCY_LUT_JSON = RESULTS / "latency_lut_pyramid.json"        # B1 (optional)

# B2 (optional). Agent-AP writes ap70_model_pyramid.json (canonical);
# accept ap_model_pyramid.json as an alias. First existing wins.
_AP_MODEL_CANDIDATES = (
    RESULTS / "ap70_model_pyramid.json",
    RESULTS / "ap_model_pyramid.json",
)
AP_MODEL_JSON = next((p for p in _AP_MODEL_CANDIDATES if p.exists()),
                     _AP_MODEL_CANDIDATES[0])

Width = tuple[int, int, int]
SCHEDULES = ("default", "tuned")


# ============================================================================
# Seed data
# ============================================================================
def load_seed_grid(path: Path = SEED_GRID_JSON, key_scale: int = 1) -> dict[Width, dict]:
    """Load the 8 real width points (W_g probe) keyed by num_filters tuple.

    key_scale>1 (bridge-native cur_width mode): re-key every width by ×key_scale
    so the whole searcher operates in the manifest's native cur_width space
    (cur_width = key_scale × num_filters, the bottleneck expansion).  Default 1 =
    legacy num_filters keys, bit-identical to the pre-bridge path.
    """
    data = json.loads(path.read_text())
    out: dict[Width, dict] = {}
    for g in data["grid"]:
        w = tuple(int(x) * key_scale for x in g["num_filters"])
        out[w] = {
            "label": g["label"],
            "default_us": float(g["default_us"]),
            "tuned_us": float(g["tuned_us"]),
            "ap70": (None if g.get("ap70") is None else float(g["ap70"])),
        }
    return out


# ============================================================================
# Latency LUT interface  (B1: results/latency_lut_pyramid.json, else seed)
# ============================================================================
class LatencyLUT:
    """latency(width, sched) -> us.

    Resolution order:
      1. B1 LUT file, if present. Two accepted formats:
         (a) direct grid: {"widths":[{"num_filters":[...],"default_us":..,"tuned_us":..}]}
         (b) additive per-stage: {"per_stage":{"0":{"16":{"default":..,"tuned":..},...},
             "1":{...},"2":{...}}}  -> Lat = sum_k interp(f_k, width_k, sched)
      2. Seed grid exact width match.
    For widths not resolvable by either, raises ValueError (smoke only queries
    seed/anchor widths, which always resolve).
    """

    def __init__(self, lut_path: Path = LATENCY_LUT_JSON,
                 seed_path: Path = SEED_GRID_JSON, key_scale: int = 1):
        self.key_scale = int(key_scale)        # >1 → bridge-native cur_width keys
        self.seed = load_seed_grid(seed_path, key_scale=self.key_scale)
        self.mode = "seed"
        self.direct: dict[Width, dict] = {}
        self.per_stage: dict[int, dict[int, dict[str, float]]] = {}
        self.additive_error: Optional[float] = None
        if lut_path.exists():
            self._load_b1(lut_path)

    def _load_b1(self, path: Path) -> None:
        data = json.loads(path.read_text())
        s = self.key_scale
        if "widths" in data or "grid" in data:
            rows = data.get("widths") or data.get("grid")
            for r in rows:
                w = tuple(int(x) * s for x in r["num_filters"])
                self.direct[w] = {"default_us": float(r["default_us"]),
                                  "tuned_us": float(r["tuned_us"])}
            self.mode = "b1_direct"
        if "per_stage" in data:
            for k, tbl in data["per_stage"].items():
                self.per_stage[int(k)] = {
                    int(wk) * s: {"default": float(v["default"]), "tuned": float(v["tuned"])}
                    for wk, v in tbl.items()
                }
            self.additive_error = data.get("additive_error")
            self.mode = "b1_additive" if not self.direct else "b1_direct+additive"

    @staticmethod
    def _interp(table: dict[int, dict[str, float]], x: int, sched: str) -> float:
        """Piecewise-linear interpolation over a per-stage f_k table."""
        if x in table:
            return table[x][sched]
        xs = sorted(table)
        if x <= xs[0]:
            return table[xs[0]][sched]
        if x >= xs[-1]:
            return table[xs[-1]][sched]
        lo = max(v for v in xs if v <= x)
        hi = min(v for v in xs if v >= x)
        if lo == hi:
            return table[lo][sched]
        t = (x - lo) / (hi - lo)
        return (1 - t) * table[lo][sched] + t * table[hi][sched]

    def latency(self, width: Width, sched: str) -> float:
        width = tuple(int(x) for x in width)
        if sched not in SCHEDULES:
            raise ValueError(f"unknown schedule {sched!r}")
        key = "default_us" if sched == "default" else "tuned_us"
        # B1 LUT takes precedence over the seed fallback when present.
        if width in self.direct:
            return self.direct[width][key]
        if self.per_stage:
            return sum(self._interp(self.per_stage[k], width[k], sched)
                       for k in range(len(width)))
        if width in self.seed:
            return self.seed[width][key]
        raise ValueError(f"latency: width {width} not resolvable (mode={self.mode})")


# ============================================================================
# AP model interface  (B2: results/ap_model_pyramid.json, else anchors)
# ============================================================================
class APModel:
    """ap70(width) -> float.

    Resolution order:
      1. B2 AP-model file if present. Accepted formats:
         (a) {"table":[{"num_filters":[...],"ap70":..}]}  -> exact + monotone interp
         (b) {"anchors":[{"num_filters":[...],"ap70":..}]} -> same
      2. Known AP70 anchors from the seed grid + monotone interpolation by total
         backbone channels (sum of num_filters).  Anchors: base 0.631 /
         trap25 0.590 / pad64 0.590 / p50 0.564 / p75 0.530.
    """

    def __init__(self, ap_path: Path = AP_MODEL_JSON,
                 seed_path: Path = SEED_GRID_JSON, key_scale: int = 1):
        self.key_scale = int(key_scale)        # >1 → bridge-native cur_width keys
        self.exact: dict[Width, float] = {}
        seed = load_seed_grid(seed_path, key_scale=self.key_scale)
        for w, info in seed.items():
            if info["ap70"] is not None:
                self.exact[w] = info["ap70"]
        self.mode = "anchors"
        if ap_path.exists():
            self._load_b2(ap_path)
        # monotone interpolation support: (proxy, ap) sorted, dups averaged
        self._build_interp()

    def _load_b2(self, path: Path) -> None:
        data = json.loads(path.read_text())
        s = self.key_scale
        rows = data.get("table") or data.get("anchors") or []
        for r in rows:
            self.exact[tuple(int(x) * s for x in r["num_filters"])] = float(r["ap70"])
        if rows:
            self.mode = "b2"

    @staticmethod
    def _proxy(w: Width) -> int:
        return sum(w)  # total backbone channels (AP is backbone-channel driven)

    def _build_interp(self) -> None:
        agg: dict[int, list[float]] = {}
        for w, ap in self.exact.items():
            agg.setdefault(self._proxy(w), []).append(ap)
        self._xs = sorted(agg)
        self._ys = [sum(agg[x]) / len(agg[x]) for x in self._xs]

    def ap70(self, width: Width) -> float:
        width = tuple(int(x) for x in width)
        if width in self.exact:
            return self.exact[width]
        x = self._proxy(width)
        xs, ys = self._xs, self._ys
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if xs[i] >= x:
                t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
                return (1 - t) * ys[i - 1] + t * ys[i]
        return ys[-1]


# ============================================================================
# Cost model  (one per arm-run; identical data/logic across arms)
# ============================================================================
@dataclass
class CostModel:
    """Shared cost model. Minimization objective vector = (latency_us, -ap70).

    Caches by (width, sched) so repeated genomes don't spend budget.  Maintains
    an eval log (every unique point visited) and a convergence log (running best
    hypervolume vs unique-eval count).  hv_ref is the fixed nadir used to score
    HV — computed OFFLINE, never derived from the searcher's own progress.
    """
    lut: LatencyLUT
    apm: APModel
    hv_ref: tuple[float, float]
    use_energy: bool = False
    use_closedloop: bool = False        # attach model-estimated driving score
    cl_beta: float = 0.0                # AP70 sensitivity exponent (0 = latency-only)
    cl_scale_tier: str = "fp32"         # H800 backbone us -> Orin e2e ms tier
    cache: dict = field(default_factory=dict)
    eval_log: list[dict] = field(default_factory=list)
    conv_log: list[dict] = field(default_factory=list)
    archive: list[tuple[float, float]] = field(default_factory=list)
    n_unique: int = 0

    def evaluate(self, width: Width, sched: str) -> tuple[tuple[float, float], dict]:
        width = tuple(int(x) for x in width)
        key = (width, sched)
        if key in self.cache:
            return self.cache[key]
        ap = self.apm.ap70(width)
        lat = self.lut.latency(width, sched)
        obj = (lat, -ap)            # minimize latency, minimize -ap70
        self.n_unique += 1
        rec = {"n": self.n_unique, "width": width, "sched": sched,
               "ap70": ap, "lat_us": lat}
        if self.use_energy:
            rec["energy_j"] = self._energy(lat, sched)
        # Closed-loop driving-score axis (plan B:附加 rec 字段, 不进 HV / 搜索).
        # model-estimated only (CoDriving τ-curve + H800→Orin scaling); see
        # multi_agent/methods/design/closedloop_b4_plugin_v1.md.
        if self.use_closedloop and _HAS_CL:
            e2e_ms = _cl_e2e(lat, self.cl_scale_tier)
            rec["e2e_orin_ms_est"] = round(e2e_ms, 1)
            rec["ds_model"] = round(_cl_ds(ap, e2e_ms, beta=self.cl_beta), 2)
        self.eval_log.append(rec)
        self.archive.append(obj)
        hv = hypervolume_2d(self.archive, self.hv_ref)
        self.conv_log.append({"n": self.n_unique, "hv": hv})
        self.cache[key] = (obj, rec)
        return self.cache[key]

    @staticmethod
    def _energy(lat_us: float, sched: str) -> float:
        # crude optional proxy: energy ~ latency * representative power
        return lat_us * 1e-6 * 250.0  # J/frame at ~250W

    def budget_used(self) -> int:
        return self.n_unique


# ============================================================================
# Pareto / hypervolume utilities (2D, minimization)
# ============================================================================
def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if a dominates b (minimization, both objectives)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


def nondominated_idx(objs: list[tuple[float, float]]) -> list[int]:
    idx = []
    for i, o in enumerate(objs):
        if not any(j != i and dominates(objs[j], o) for j in range(len(objs))):
            idx.append(i)
    return idx


def hypervolume_2d(points_min: list[tuple[float, float]],
                   ref: tuple[float, float]) -> float:
    """2D hypervolume (area dominated) for minimization, bounded by nadir `ref`."""
    front = [points_min[i] for i in nondominated_idx(points_min)]
    front = [p for p in front if p[0] < ref[0] and p[1] < ref[1]]
    if not front:
        return 0.0
    front.sort(key=lambda p: p[0], reverse=True)   # f1 descending
    hv, prev_f1 = 0.0, ref[0]
    for f1, f2 in front:
        hv += (prev_f1 - f1) * (ref[1] - f2)
        prev_f1 = f1
    return hv


def fast_nondominated_sort(objs: list[tuple[float, float]]) -> list[list[int]]:
    n = len(objs)
    S = [[] for _ in range(n)]
    ndom = [0] * n
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                ndom[p] += 1
        if ndom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(objs: list[tuple[float, float]], front: list[int]) -> dict[int, float]:
    cd = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for m in range(2):
        order = sorted(front, key=lambda i: objs[i][m])
        cd[order[0]] = cd[order[-1]] = math.inf
        lo, hi = objs[order[0]][m], objs[order[-1]][m]
        span = (hi - lo) or 1.0
        for k in range(1, len(order) - 1):
            cd[order[k]] += (objs[order[k + 1]][m] - objs[order[k - 1]][m]) / span
    return cd


# ============================================================================
# Generic NSGA-II explorer kernel (shared by all three arms)
# ============================================================================
def nsga2(cost: CostModel,
          decode: Callable[[tuple[int, ...]], tuple[Width, str]],
          gene_cards: list[int],
          pop_size: int,
          budget: int,
          rng: random.Random,
          max_gens: int = 200,
          forced_init: Optional[list[tuple[int, ...]]] = None) -> None:
    """Drive exploration on `cost` until `budget` unique evals or space exhausted.

    Results are read back from `cost.eval_log`; this function only steers the
    search (genome -> (width, sched) via `decode`).

    forced_init: genomes to seed the initial population with (multi-start: lets
    A-serial's stage-1 greedy begin from a specified width pick).
    """
    space = 1
    for c in gene_cards:
        space *= c

    def rand_genome() -> tuple[int, ...]:
        return tuple(rng.randrange(c) for c in gene_cards)

    def obj_of(g: tuple[int, ...]) -> tuple[float, float]:
        w, s = decode(g)
        return cost.evaluate(w, s)[0]

    # initial population (unique genomes), seeded with any forced-init genomes
    pop: list[tuple[int, ...]] = []
    seen: set = set()
    for g in (forced_init or []):
        g = tuple(g)
        if g not in seen and len(g) == len(gene_cards):
            seen.add(g)
            pop.append(g)
    guard = 0
    while len(pop) < min(pop_size, space) and guard < pop_size * 50:
        guard += 1
        g = rand_genome()
        if g in seen:
            continue
        seen.add(g)
        pop.append(g)
    for g in pop:
        if cost.budget_used() >= budget:
            break
        obj_of(g)

    def tournament() -> tuple[int, ...]:
        a, b = rng.choice(pop), rng.choice(pop)
        ra, rb = rank[a], rank[b]
        if ra != rb:
            return a if ra < rb else b
        return a if cd.get(a, 0) >= cd.get(b, 0) else b

    def crossover(p1, p2):
        return tuple(p1[i] if rng.random() < 0.5 else p2[i] for i in range(len(gene_cards)))

    def mutate(g):
        g = list(g)
        for i, c in enumerate(gene_cards):
            if c > 1 and rng.random() < 1.0 / len(gene_cards):
                g[i] = rng.randrange(c)
        return tuple(g)

    stall = 0
    for _ in range(max_gens):
        if cost.budget_used() >= budget or len(seen) >= space:
            break
        objs = [obj_of(g) for g in pop]              # cache hits
        fronts = fast_nondominated_sort(objs)
        rank = {}
        cd = {}
        for r, fr in enumerate(fronts):
            for i in fr:
                rank[pop[i]] = r
            for i, v in crowding_distance(objs, fr).items():
                cd[pop[i]] = v

        offspring = []
        guard = 0
        while len(offspring) < pop_size and guard < pop_size * 20:
            guard += 1
            child = mutate(crossover(tournament(), tournament()))
            offspring.append(child)

        before = cost.budget_used()
        for g in offspring:
            if cost.budget_used() >= budget:
                break
            obj_of(g)
            seen.add(g)
        new_evals = cost.budget_used() - before
        stall = stall + 1 if new_evals == 0 else 0
        if stall >= 3:
            break

        # environmental selection: keep best pop_size of pop+offspring
        union = list(dict.fromkeys(pop + offspring))
        uobjs = [obj_of(g) for g in union]
        ufronts = fast_nondominated_sort(uobjs)
        nxt: list[tuple[int, ...]] = []
        for fr in ufronts:
            if len(nxt) + len(fr) <= pop_size:
                nxt.extend(union[i] for i in fr)
            else:
                ucd = crowding_distance(uobjs, fr)
                ordered = sorted(fr, key=lambda i: ucd[i], reverse=True)
                nxt.extend(union[i] for i in ordered[:pop_size - len(nxt)])
                break
        pop = nxt if nxt else pop


# ============================================================================
# The three arms
# ============================================================================
@dataclass
class ArmResult:
    name: str
    cost: CostModel
    pareto: list[dict]                  # final deliverable Pareto (records)
    visited: list[tuple[Width, str]]    # every (W,S) evaluated
    final_hv: float
    locked_widths: Optional[list[Width]] = None


def _front_records(records: list[dict]) -> list[dict]:
    objs = [(r["lat_us"], -r["ap70"]) for r in records]
    keep = nondominated_idx(objs)
    return sorted((records[i] for i in keep), key=lambda r: r["lat_us"])


def run_joint(lut, apm, width_grid, budget, rng, hv_ref, pop_size, use_energy=False,
              use_closedloop=False, cl_beta=0.0, cl_scale_tier="fp32"):
    cost = CostModel(lut, apm, hv_ref, use_energy, use_closedloop=use_closedloop,
                     cl_beta=cl_beta, cl_scale_tier=cl_scale_tier)
    decode = lambda g: (width_grid[g[0]], SCHEDULES[g[1]])
    nsga2(cost, decode, [len(width_grid), len(SCHEDULES)], pop_size, budget, rng)
    pareto = _front_records(cost.eval_log)
    visited = [(r["width"], r["sched"]) for r in cost.eval_log]
    return ArmResult("A-joint", cost, pareto, visited,
                     hypervolume_2d(cost.archive, hv_ref))


def run_noS(lut, apm, width_grid, budget, rng, hv_ref, pop_size, use_energy=False,
            use_closedloop=False, cl_beta=0.0, cl_scale_tier="fp32"):
    cost = CostModel(lut, apm, hv_ref, use_energy, use_closedloop=use_closedloop,
                     cl_beta=cl_beta, cl_scale_tier=cl_scale_tier)
    decode = lambda g: (width_grid[g[0]], "default")
    nsga2(cost, decode, [len(width_grid)], pop_size, budget, rng)
    pareto = _front_records(cost.eval_log)
    visited = [(r["width"], r["sched"]) for r in cost.eval_log]
    return ArmResult("A-noS", cost, pareto, visited,
                     hypervolume_2d(cost.archive, hv_ref))


def run_serial(lut, apm, width_grid, budget, rng, hv_ref, pop_size,
               stage1_frac=0.5, topk=None, use_energy=False, start_width=None,
               use_closedloop=False, cl_beta=0.0, cl_scale_tier="fp32"):
    """Serial greedy: lock default-Pareto widths in stage 1, tune S in stage 2.

    topk=None  -> single-point lock (lock exactly the default-Pareto front).
    topk=k     -> relaxed lock: also keep the k best dominated widths (by
                  default latency) so the lock set is larger.
    start_width -> multi-start: force stage-1's initial population to include
                  this width (greedy-start perturbation). If the lock set is
                  invariant across start widths, the local optimum is structural,
                  not a bad start.
    The whole point: widths dominated under default (e.g. pad64) are excluded
    from the lock set and their tuned branch is NEVER evaluated.
    """
    cost = CostModel(lut, apm, hv_ref, use_energy, use_closedloop=use_closedloop,
                     cl_beta=cl_beta, cl_scale_tier=cl_scale_tier)
    b1 = max(1, int(budget * stage1_frac))

    # --- stage 1: width search under DEFAULT schedule ---
    decode1 = lambda g: (width_grid[g[0]], "default")
    forced = None
    if start_width is not None:
        sw = tuple(int(x) for x in start_width)
        if sw in width_grid:
            forced = [(width_grid.index(sw),)]
    nsga2(cost, decode1, [len(width_grid)], pop_size, b1, rng, forced_init=forced)

    default_recs = [r for r in cost.eval_log if r["sched"] == "default"]
    locked = [r["width"] for r in _front_records(default_recs)]   # default-Pareto
    if topk:
        # relaxed: append best-by-default-latency widths not already locked
        extra = sorted((r for r in default_recs if r["width"] not in locked),
                       key=lambda r: r["lat_us"])
        for r in extra[:topk]:
            if r["width"] not in locked:
                locked.append(r["width"])

    # --- stage 2: schedule search on the LOCKED widths only ---
    if locked:
        decode2 = lambda g: (locked[g[0]], SCHEDULES[g[1]])
        nsga2(cost, decode2, [len(locked), len(SCHEDULES)], pop_size, budget, rng)

    # deliverable = nondominated over evaluated points whose width is locked
    deliverable = [r for r in cost.eval_log if r["width"] in set(locked)]
    pareto = _front_records(deliverable)
    visited = [(r["width"], r["sched"]) for r in cost.eval_log]
    # HV scored over the deliverable archive (what serial can actually output)
    final_hv = hypervolume_2d([(r["lat_us"], -r["ap70"]) for r in deliverable], hv_ref)
    return ArmResult("A-serial", cost, pareto, visited, final_hv, locked_widths=locked)


# ============================================================================
# Offline scoring helpers (HIDDEN from arms — discipline)
# ============================================================================
def compute_hv_ref(width_grid, lut, apm) -> tuple[float, float]:
    """Fixed nadir for HV scoring. Uses default (worst) latency + min AP.

    Computed offline; identical across arms; never fed into any search."""
    lats = [lut.latency(w, "default") for w in width_grid]
    aps = [apm.ap70(w) for w in width_grid]
    return (max(lats) * 1.05, -(min(aps) - 0.02))


def reference_pareto(width_grid, lut, apm, hv_ref):
    """Global Pareto over the full P×S space. For REPORTING/scoring only."""
    recs = []
    for w in width_grid:
        for s in SCHEDULES:
            recs.append({"width": w, "sched": s,
                         "ap70": apm.ap70(w), "lat_us": lut.latency(w, s)})
    front = _front_records(recs)
    hv = hypervolume_2d([(r["lat_us"], -r["ap70"]) for r in recs], hv_ref)
    return front, hv


# ============================================================================
# Grid derivation (grid-agnostic: auto picks up B1/B2; else seed fallback)
# ============================================================================
def candidate_widths(lut: LatencyLUT, apm: APModel) -> list[Width]:
    """The prune-width search grid = every width with a *defined* AP70 (B2 table
    or seed anchors — NOT interpolated) whose latency is resolvable (default +
    tuned). This is what the search arms see; it auto-expands when B1/B2 land and
    falls back to the 5 AP-known seed anchors otherwise.
    """
    out: list[Width] = []
    for w in apm.exact:                      # widths with a real (non-interp) AP
        try:
            lut.latency(w, "default")
            lut.latency(w, "tuned")
        except ValueError:
            continue                          # AP known but latency unpriceable
        out.append(w)
    return sorted(out)


def detect_wg_pg_pairs(width_grid: list[Width], lut: LatencyLUT, apm: APModel,
                       ap_tol: float = 1e-4) -> list[dict]:
    """Auto-detect rank-flip W_g/P_g pairs in the priceable grid (grid-agnostic).

    A pair = two grid widths that
      * share (s1, s2) and differ only in s0,
      * have EQUAL AP70 within `ap_tol` (zero-pad weight-identity: padding the
        misaligned s0 up to an aligned channel count changes no weights), and
      * exhibit a DEFAULT->TUNED rank flip: the width that is *faster under the
        default schedule* (= W_g, what A-serial ranks/locks in stage 1) becomes
        the *slower-tuned* one, while its aligned partner (= P_g) tunes far
        better.  A-serial, ranking widths by default latency, locks W_g and
        never tunes P_g; A-joint tunes both and keeps P_g.

    Returns one dict per pair, sorted by descending iso-AP latency ratio
    (W_g_tuned / P_g_tuned) — the speedup A-joint gets over A-serial at equal AP.
    """
    g = list(width_grid)
    pairs: list[dict] = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            wa, wb = g[i], g[j]
            if wa[1:] != wb[1:] or wa[0] == wb[0]:
                continue
            if abs(apm.ap70(wa) - apm.ap70(wb)) > ap_tol:
                continue
            da, db = lut.latency(wa, "default"), lut.latency(wb, "default")
            ta, tb = lut.latency(wa, "tuned"), lut.latency(wb, "tuned")
            if (da < db) == (ta < tb):          # same order -> no rank flip
                continue
            wg, pg = (wa, wb) if da < db else (wb, wa)   # W_g = default-faster
            wg_t, pg_t = lut.latency(wg, "tuned"), lut.latency(pg, "tuned")
            pairs.append({
                "wg": wg, "pg": pg, "ap70": round(apm.ap70(wg), 4),
                "wg_default_us": lut.latency(wg, "default"),
                "pg_default_us": lut.latency(pg, "default"),
                "wg_tuned_us": wg_t, "pg_tuned_us": pg_t,
                "wg_tuned_ratio": round(lut.latency(wg, "default") / wg_t, 3),
                "pg_tuned_ratio": round(lut.latency(pg, "default") / pg_t, 3),
                "iso_ap_latency_ratio": round(wg_t / pg_t, 3),
            })
    return sorted(pairs, key=lambda p: -p["iso_ap_latency_ratio"])


# ============================================================================
# Bridge: stage1 partition manifest → configured searcher (native cur_width space)
# ============================================================================
def build_from_manifest(manifest_path,
                        seed_path: Path = SEED_GRID_JSON,
                        lut_path: Path = LATENCY_LUT_JSON,
                        ap_path: Path = AP_MODEL_JSON,
                        q_path=None) -> dict:
    """Configure (LatencyLUT, APModel, QLookup) from a stage1 manifest.

    Replaces the hand-curated num_filters parametrization with the manifest's
    native cur_width space + manifest-derived int8 buildability.  Returns a dict
    {spec, key_scale, bridge_int8_align, lut, apm, qlut, gating_knob}.  Pure
    parsing — no GPU.  Derivation (all from the manifest, zero per-model hardcode):

      * gating knob   = the cliff knob (legal ⊋ int8-buildable) with the smallest
        base width (= stage0, the searcher's s0 axis).  None → no int8
        buildability cliff in traced dense space; this is not a model-level
        separability proof.
      * key_scale     = gating.base_width // base_s0  (cur_width = scale×num_filters,
        the bottleneck expansion; base_s0 = max s0 in the seed grid).
      * bridge_int8_align = gating.int8_buildable_align (already in cur_width space).
    """
    from framework.stage1_bridge import SpaceSpec        # local import: no GPU dep

    spec = SpaceSpec.from_manifest(manifest_path)
    cliff = [k for k in spec.knobs
             if set(k.legal_widths()) - set(k.buildable_int8_widths())]
    gating = min(cliff, key=lambda k: k.base_width) if cliff else None

    base_s0 = max((w[0] for w in load_seed_grid(seed_path)), default=1)
    if gating is not None:
        if gating.base_width % base_s0:
            raise ValueError(
                f"non-integer cur_width/num_filters ratio: gating base "
                f"{gating.base_width} not divisible by seed base_s0 {base_s0}")
        key_scale = gating.base_width // base_s0
        bridge_align = gating.int8_buildable_align
    else:
        # No int8 buildability cliff in traced dense space: int8 builds for every
        # legal dense width. Gate on pack_factor alone (dp4a per-group packing,
        # groups=1) so all widths %pack==0 are buildable — matches measured
        # all-build, exactly replacing the old QLookupCoDriving
        # (can_build_int8≡True). key_scale=1.
        key_scale = 1
        bridge_align = spec.hw.int8_pack_factor

    q_kwargs = {} if q_path is None else {"q_path": q_path}
    lut = LatencyLUT(lut_path, seed_path, key_scale=key_scale)
    apm = APModel(ap_path, seed_path, key_scale=key_scale)
    qlut = QLookup(key_scale=key_scale, bridge_int8_align=bridge_align, **q_kwargs)
    # ★ stage1→stage2 打通: 逐-knob 耦合分数驱动内环分流 (高耦合旋钮联合搜/低耦合串行)。
    dispatch = spec.dispatch_plan()
    return {"spec": spec, "key_scale": key_scale, "bridge_int8_align": bridge_align,
            "lut": lut, "apm": apm, "qlut": qlut, "gating_knob": gating,
            "dispatch_plan": dispatch, "joint_knobs": [d["knob"] for d in dispatch
                                                       if d["dispatch"] == "joint"]}


# ============================================================================
# Smoke test
# ============================================================================
SMOKE_WIDTHS: list[Width] = [
    (64, 128, 256),   # base
    (32, 64, 128),    # p50
    (16, 32, 64),     # p75
    (48, 96, 192),    # trap25  (W_g)
    (64, 96, 192),    # pad64   (P_g)
]


def _label(w: Width, seed) -> str:
    return seed.get(w, {}).get("label", str(w))


def _fmt_front(front, seed):
    return ", ".join(
        f"{_label(r['width'], seed)}/{r['sched']}({r['lat_us']:.0f}us,AP70={r['ap70']:.3f})"
        for r in front)


def run_smoke(seeds=(0, 1, 2, 3, 4), budget=40, pop_size=8, verbose=True):
    seed_grid = load_seed_grid()
    lut = LatencyLUT()
    apm = APModel()
    hv_ref = compute_hv_ref(SMOKE_WIDTHS, lut, apm)
    ref_front, ref_hv = reference_pareto(SMOKE_WIDTHS, lut, apm, hv_ref)

    print("=" * 84)
    print("THREE-ARM SEARCH SMOKE  (P × S, Pyramid backbone seed space)")
    print("=" * 84)
    print(f"LUT mode={lut.mode}  AP mode={apm.mode}  "
          f"hv_ref(lat,-ap)={hv_ref[0]:.0f},{hv_ref[1]:.3f}")
    print(f"Reference (global P×S) Pareto [HIDDEN from arms, scoring only]:")
    print(f"  {_fmt_front(ref_front, seed_grid)}")
    print(f"  reference HV = {ref_hv:.4e}\n")

    summary = {a: [] for a in ("A-joint", "A-noS", "A-serial")}
    visited_union = {a: set() for a in summary}

    for sd in seeds:
        if verbose:
            print(f"--- seed {sd} ---")
        for runner, name in ((run_joint, "A-joint"), (run_noS, "A-noS"),
                             (run_serial, "A-serial")):
            rng = random.Random(sd)
            res = runner(lut, apm, SMOKE_WIDTHS, budget, rng, hv_ref, pop_size)
            summary[name].append(res)
            visited_union[name].update(res.visited)
            if verbose:
                extra = ""
                if res.locked_widths is not None:
                    extra = "  locked=" + ",".join(_label(w, seed_grid)
                                                    for w in res.locked_widths)
                print(f"  {name:9s} HV={res.final_hv:.4e}  "
                      f"evals={res.cost.budget_used():2d}{extra}")
                print(f"    Pareto: {_fmt_front(res.pareto, seed_grid)}")

    # ---- aggregate report ----
    print("\n" + "=" * 84)
    print("AGGREGATE  (mean HV over seeds, normalized to reference HV)")
    print("=" * 84)
    for name in ("A-joint", "A-serial", "A-noS"):
        hvs = [r.final_hv for r in summary[name]]
        mean = sum(hvs) / len(hvs)
        print(f"  {name:9s} mean HV={mean:.4e}  ({mean / ref_hv * 100:5.1f}% of ref)  "
              f"min={min(hvs):.3e} max={max(hvs):.3e}")

    print("\n" + "=" * 84)
    print("VISITED (W,S) POINT SETS  (union over seeds) — proves arms differ")
    print("=" * 84)
    for name in ("A-joint", "A-noS", "A-serial"):
        pts = sorted((_label(w, seed_grid), s) for (w, s) in visited_union[name])
        print(f"  {name:9s} ({len(pts):2d}): "
              + ", ".join(f"{lab}/{s}" for lab, s in pts))

    # ---- the structural claim ----
    pad64 = (64, 96, 192)
    joint_has = (pad64, "tuned") in visited_union["A-joint"]
    serial_has = (pad64, "tuned") in visited_union["A-serial"]
    serial_locks_trap = all(
        (48, 96, 192) in (r.locked_widths or []) and pad64 not in (r.locked_widths or [])
        for r in summary["A-serial"])
    print("\n" + "=" * 84)
    print("STRUCTURAL CLAIM  (P_g = pad64-tuned reachability)")
    print("=" * 84)
    print(f"  A-joint  visits (pad64, tuned) : {joint_has}   [P_g reached]")
    print(f"  A-serial visits (pad64, tuned) : {serial_has}   "
          f"[should be False — structurally excluded]")
    print(f"  A-serial locks trap25 (W_g) & drops pad64 every seed: {serial_locks_trap}")
    ok = joint_has and (not serial_has) and serial_locks_trap
    jhv = sum(r.final_hv for r in summary["A-joint"]) / len(summary["A-joint"])
    shv = sum(r.final_hv for r in summary["A-serial"]) / len(summary["A-serial"])
    print(f"  mean HV  A-joint {jhv:.4e} >= A-serial {shv:.4e} : {jhv >= shv}")
    print("\n  RESULT: " + ("PASS — A-serial structurally misses (P_g=pad64, tuned); "
                            "A-joint reaches it." if ok else "FAIL — see above."))
    print("=" * 84)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Three-arm P×S search (smoke)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--q-mode", action="store_true",
                    help="Run Q-extended P×Q×S smoke test")
    args = ap.parse_args()
    if args.q_mode:
        from framework.search_three_arm import run_smoke_pqs
        ok = run_smoke_pqs(seeds=tuple(range(args.seeds)), budget=args.budget,
                           pop_size=args.pop, verbose=not args.quiet)
    else:
        ok = run_smoke(seeds=tuple(range(args.seeds)), budget=args.budget,
                       pop_size=args.pop, verbose=not args.quiet)
    raise SystemExit(0 if ok else 1)


# ============================================================================
# Q-axis (quantization) extension  — T1 P×Q×S
# ============================================================================
# DISCIPLINE:
#   * New INT8 latency evidence must be H800 TVM only.
#   * Historical RTX 4090 TRT ratios may be kept as historical context only and
#     are disabled as an active latency backend unless explicitly opted in.
#   * INT8 AP penalty: constant -0.007 AP70 (median from e2e pipeline, 8 widths).
#     Labeled historical evidence — derived from 4090 e2e pipeline, not stage_a full eval.
#   * CROSS-HARDWARE WARNING: historical Q_ratio comes from RTX 4090 TRT; H800
#     INT8 speedup may differ. Do not use it as a new H800 backend.

QUANT_MODES = ("fp16", "int8")
Q_LUT_JSON = RESULTS / "latency_lut_pyramid_q.json"

# Median INT8 AP70 penalty from e2e pipeline measurements (8 widths, minmax calib).
# Source: results/a0_refresh_ap/ + models/e2e_cache/*.bench.json
_INT8_AP_DELTA_MEDIAN = -0.008   # historical DAIR val 1789 TRT INT8 MinMax: base=-0.0081, p50=-0.0099, p75=-0.0064, median=-0.0081
_INT8_AP_DELTA_LABEL = "historical_trt_evidence:median_DAIR_val_1789_int8_minmax_ap_delta"


class QLookup:
    """Q-ratio table: fp16_p50_ms / int8_p50_ms per width (4090 backbone subnet).

    speedup(width) -> float, default 1.0 (no speedup) for unknown widths.
    int8_lat(h800_tvm_us, width) -> float  estimated H800 INT8 latency (us).

    Cross-hardware WARNING: historical ratios from RTX 4090 TRT are not a new
    H800 backend and are disabled for active latency by default.
    """

    # Fallback table for widths not in the Q-LUT file (from session measurements)
    _BUILTIN: dict[Width, float] = {
        (64, 128, 256): 1.266 / 0.803,   # base 1.577x
        (48, 96, 192):  2.919 / 2.728,   # pruned25/trap25 1.070x (alignment trap)
        (32, 64, 128):  1.009 / 0.787,   # pruned50 1.281x
        (16, 32, 64):   0.779 / 0.639,   # pruned75 1.218x
        (32, 64, 136):  1.164 / 0.868,   # p50b2_136 1.341x
    }

    # ── Structural INT8-buildability constraint (H800 TVM NCHWc dp4a path) ──
    # Pyramid stage0 is a grouped conv; the per-group input channel count is
    # in_per_g = s0 // GROUP_DIVISOR.  The NCHWc int8 layout packs 4 int8 along
    # the channel axis (IC_BN=4 for dp4a), so int8 is only BUILDABLE when
    # in_per_g is divisible by 4.  MEASURED (results/q_int8_dp4a_pairs.csv):
    #   s0=48 -> in_per_g=3 -> NCHWc IC_BN=4 IMPOSSIBLE -> NOT_APPLICABLE (real)
    #   s0=64 -> in_per_g=4 -> builds real WMMA/dp4a int8, 1.45x over fp16 (real)
    # s0=16/32 (in_per_g 1/2) are excluded by the SAME rule (inferred, same path).
    # This is the categorical Q-axis coupling: the misaligned widths a serial
    # optimizer locks under fp16-default can NEVER reach the int8 path.
    GROUP_DIVISOR = 16
    # Real H800 stage0 int8 speedup over fp16 (measured WMMA, max_rel_err=0.0).
    MEASURED_INT8_SPEEDUP = 1.449   # results/q_int8_ms_stage0_result.json

    def __init__(self, q_path: Path = Q_LUT_JSON, key_scale: int = 1,
                 bridge_int8_align: Optional[int] = None,
                 bridge_pack_factor: int = 4):
        self.key_scale = int(key_scale)         # >1 → bridge-native cur_width keys
        # Bridge-native int8 buildability (manifest int8_buildable_align). When set,
        # can_build_int8/in_per_g operate natively on cur_width keys instead of the
        # legacy GROUP_DIVISOR rule. None → legacy in_per_g%4 path (bit-identical).
        self.bridge_int8_align = (None if bridge_int8_align is None
                                  else int(bridge_int8_align))
        self.bridge_pack_factor = int(bridge_pack_factor)
        self._ratios: dict[Width, float] = {}
        self._complete: dict[Width, bool] = {}  # True if fp16+int8 both measured
        # Direct H800 TVM int8 latencies (real, preferred over 4090 ratio estimates)
        # Keyed by (width_tuple, sched) -> lat_us
        self._h800_int8: dict[tuple, float] = {}
        # When set, INT8 latency for BUILDABLE widths = H800 fp16 / uniform_int8_speedup
        # (keeps the Pareto axis H800-pure: uses the one real H800 int8 measurement
        #  as a uniform per-width proxy, instead of cross-hardware 4090 ratios).
        self.uniform_int8_speedup: Optional[float] = None
        self.allow_historical_trt_ratio_fallback: bool = False
        # When True, evaluate() treats non-buildable INT8 as structurally unreachable.
        self.enforce_int8_buildable: bool = False
        self._mode = "builtin"
        # load _BUILTIN as baseline (re-keyed ×key_scale in bridge-native mode)
        for w, r in self._BUILTIN.items():
            self._ratios[tuple(x * self.key_scale for x in w)] = r
            self._complete[tuple(x * self.key_scale for x in w)] = True
        if q_path.exists():
            self._load(q_path)

    def _load(self, path: Path) -> None:
        data = json.loads(path.read_text())
        s = self.key_scale
        for entry in data.get("widths", []):
            w = tuple(int(x) * s for x in entry["num_filters"])
            fp16 = entry.get("fp16_p50_ms")
            int8 = entry.get("int8_p50_ms")
            if fp16 and int8 and int8 > 0:
                self._ratios[w] = fp16 / int8
                self._complete[w] = True
            elif fp16 and not int8:
                # FP16 known, INT8 pending — keep builtin if available
                self._complete[w] = False
            # Load real H800 TVM int8 measurements if present (preferred source)
            h8d = entry.get("h800_tvm_int8_default_us")
            h8t = entry.get("h800_tvm_int8_tuned_us")
            if h8d is not None and h8d > 0:
                self._h800_int8[(w, "default")] = h8d
            if h8t is not None and h8t > 0:
                self._h800_int8[(w, "tuned")] = h8t
        self._mode = "q_lut"

    def speedup(self, width: Width) -> float:
        """INT8/FP16 speedup ratio (>1 = INT8 faster). Default 1.0 if unknown."""
        w = tuple(int(x) for x in width)
        return self._ratios.get(w, 1.0)

    def has_direct_h800(self, width: Width, sched: str = "tuned") -> bool:
        """True if real H800 TVM int8 measurement is available for this width+sched."""
        return (tuple(int(x) for x in width), sched) in self._h800_int8

    def in_per_g(self, width: Width) -> int:
        """Per-group input channel count of the stage0 grouped conv.

        Bridge-native mode (bridge_int8_align set): width[0] is cur_width;
        in_per_g = cur_width × pack_factor / int8_buildable_align (= cur_width /
        groups, since int8_buildable_align = lcm(int8_align, pack×groups)).
        Legacy mode: num_filters // GROUP_DIVISOR.
        """
        if self.bridge_int8_align is not None:
            return int(width[0]) * self.bridge_pack_factor // self.bridge_int8_align
        return int(width[0]) // self.GROUP_DIVISOR

    def can_build_int8(self, width: Width) -> bool:
        """True iff the H800 TVM NCHWc int8 (dp4a) path can be BUILT for this width.

        Bridge-native mode: cur_width % int8_buildable_align == 0 (manifest-derived,
        no per-model rule). Legacy mode: in_per_g % 4 == 0 (GROUP_DIVISOR). Both
        agree on the measured grid (cur_width%128 ⟺ s0%64 ⟺ in_per_g%4), verified
        against results/q_int8_dp4a_pairs.csv.
        """
        if self.bridge_int8_align is not None:
            return int(width[0]) % self.bridge_int8_align == 0
        return self.in_per_g(width) % 4 == 0

    def int8_lat(self, h800_tvm_us: float, width: Width, sched: str = "tuned") -> float:
        """H800 INT8 latency.

        Preference order:
          1. Real H800 TVM int8 measurement (h800_tvm_int8_*_us in the Q-LUT).
          2. uniform_int8_speedup proxy (H800_FP16 / measured stage0 speedup) —
             H800-pure, used by the P×Q×S ablation driver.
          3. Historical TRT ratio fallback only when explicitly enabled; otherwise
             return neutral FP16 latency so no new backend is implied.
        """
        key = (tuple(int(x) for x in width), sched)
        if key in self._h800_int8:
            return self._h800_int8[key]  # real H800 TVM int8
        if self.uniform_int8_speedup:
            return h800_tvm_us / self.uniform_int8_speedup
        if self.allow_historical_trt_ratio_fallback:
            return h800_tvm_us / self.speedup(width)
        return h800_tvm_us

    def is_complete(self, width: Width) -> bool:
        """True if both FP16 and INT8 measured for this width."""
        return self._complete.get(tuple(int(x) for x in width), False)

    def known_widths(self) -> list[Width]:
        return list(self._ratios)


# ── Q-extended CostModel ──────────────────────────────────────────────────────

@dataclass
class CostModelPQS:
    """P×Q×S cost model: evaluate(width, sched, quant) -> (lat_us, -ap70).

    INT8 latency uses H800 TVM measurements or an explicit H800 stage0 proxy.
    Historical TRT ratio fallback is disabled by default and must be labeled
    historical if explicitly enabled.
    INT8 AP uses a historical TRT AP delta context label.
    """
    lut: LatencyLUT
    apm: APModel
    qlut: QLookup
    hv_ref: tuple[float, float]
    cache: dict = field(default_factory=dict)
    eval_log: list[dict] = field(default_factory=list)
    conv_log: list[dict] = field(default_factory=list)
    archive: list[tuple[float, float]] = field(default_factory=list)
    n_unique: int = 0

    def evaluate(self, width: Width, sched: str,
                 quant: str = "fp16") -> tuple[tuple[float, float], dict]:
        width = tuple(int(x) for x in width)
        if quant not in QUANT_MODES:
            raise ValueError(f"unknown quant {quant!r}")
        key = (width, sched, quant)
        if key in self.cache:
            return self.cache[key]

        ap_fp16 = self.apm.ap70(width)
        lat_fp16 = self.lut.latency(width, sched)

        if quant == "fp16":
            lat = lat_fp16
            ap  = ap_fp16
            lat_source = "H800_TVM_exact"
            ap_source  = "stage_a_fp16"
        elif (self.qlut.enforce_int8_buildable
              and not self.qlut.can_build_int8(width)):
            # Structurally UNREACHABLE: in_per_g % 4 != 0 -> NCHWc int8 cannot
            # be built on the H800 TVM dp4a path.  Sentinel = HV nadir latency
            # (strictly dominated, contributes 0 hypervolume) so the optimizer
            # can NEVER place this on a Pareto front.  This is the categorical
            # Q-axis coupling: misaligned widths have no int8 path at all.
            lat = self.hv_ref[0]
            ap  = ap_fp16 + _INT8_AP_DELTA_MEDIAN
            lat_source = "int8_UNBUILDABLE_structural(in_per_g%4!=0)"
            ap_source  = _INT8_AP_DELTA_LABEL
        else:  # int8, buildable
            lat = self.qlut.int8_lat(lat_fp16, width, sched=sched)
            ap  = ap_fp16 + _INT8_AP_DELTA_MEDIAN
            if self.qlut.has_direct_h800(width, sched):
                lat_source = "H800_TVM_int8_real"
            elif self.qlut.uniform_int8_speedup:
                lat_source = (f"H800_TVM_fp16/{self.qlut.uniform_int8_speedup}"
                              "(measured_stage0_speedup_proxy)")
            elif self.qlut.allow_historical_trt_ratio_fallback:
                lat_source = "historical_trt_evidence:4090_q_ratio_context"
            else:
                lat_source = "H800_TVM_int8_unmeasured_neutral_fp16_latency"
            ap_source  = _INT8_AP_DELTA_LABEL

        obj = (lat, -ap)
        self.n_unique += 1
        rec = {
            "n": self.n_unique, "width": width, "sched": sched, "quant": quant,
            "ap70": ap, "lat_us": lat,
            "lat_source": lat_source, "ap_source": ap_source,
            "int8_speedup": round(self.qlut.speedup(width), 4) if quant == "int8" else 1.0,
        }
        self.eval_log.append(rec)
        self.archive.append(obj)
        hv = hypervolume_2d(self.archive, self.hv_ref)
        self.conv_log.append({"n": self.n_unique, "hv": hv})
        self.cache[key] = (obj, rec)
        return self.cache[key]

    def budget_used(self) -> int:
        return self.n_unique


# ── Extended NSGA-II (3-gene decode) ─────────────────────────────────────────

def nsga2_pqs(cost: CostModelPQS,
              decode: Callable[[tuple[int, ...]], tuple[Width, str, str]],
              gene_cards: list[int],
              pop_size: int,
              budget: int,
              rng: random.Random,
              max_gens: int = 200,
              forced_init: Optional[list[tuple[int, ...]]] = None) -> None:
    """NSGA-II over P×Q×S (3-gene genome). Drives CostModelPQS."""
    space = 1
    for c in gene_cards:
        space *= c

    def rand_genome() -> tuple[int, ...]:
        return tuple(rng.randrange(c) for c in gene_cards)

    def obj_of(g: tuple[int, ...]) -> tuple[float, float]:
        w, s, q = decode(g)
        return cost.evaluate(w, s, q)[0]

    pop: list[tuple[int, ...]] = []
    seen: set = set()
    for g in (forced_init or []):
        g = tuple(g)
        if g not in seen and len(g) == len(gene_cards):
            seen.add(g); pop.append(g)
    guard = 0
    while len(pop) < min(pop_size, space) and guard < pop_size * 50:
        guard += 1
        g = rand_genome()
        if g not in seen:
            seen.add(g); pop.append(g)
    for g in pop:
        if cost.budget_used() >= budget:
            break
        obj_of(g)

    def tournament():
        a, b = rng.choice(pop), rng.choice(pop)
        ra, rb = rank[a], rank[b]
        if ra != rb:
            return a if ra < rb else b
        return a if cd.get(a, 0) >= cd.get(b, 0) else b

    def crossover(p1, p2):
        return tuple(p1[i] if rng.random() < 0.5 else p2[i]
                     for i in range(len(gene_cards)))

    def mutate(g):
        g = list(g)
        for i, c in enumerate(gene_cards):
            if c > 1 and rng.random() < 1.0 / len(gene_cards):
                g[i] = rng.randrange(c)
        return tuple(g)

    stall = 0
    for _ in range(max_gens):
        if cost.budget_used() >= budget or len(seen) >= space:
            break
        objs = [obj_of(g) for g in pop]
        fronts = fast_nondominated_sort(objs)
        rank = {}; cd = {}
        for r, fr in enumerate(fronts):
            for i in fr:
                rank[pop[i]] = r
            for i, v in crowding_distance(objs, fr).items():
                cd[pop[i]] = v
        offspring = []
        guard = 0
        while len(offspring) < pop_size and guard < pop_size * 20:
            guard += 1
            offspring.append(mutate(crossover(tournament(), tournament())))
        before = cost.budget_used()
        for g in offspring:
            if cost.budget_used() >= budget:
                break
            obj_of(g); seen.add(g)
        stall = stall + 1 if cost.budget_used() == before else 0
        if stall >= 3:
            break
        union = list(dict.fromkeys(pop + offspring))
        uobjs = [obj_of(g) for g in union]
        ufronts = fast_nondominated_sort(uobjs)
        nxt: list[tuple[int, ...]] = []
        for fr in ufronts:
            if len(nxt) + len(fr) <= pop_size:
                nxt.extend(union[i] for i in fr)
            else:
                ucd = crowding_distance(uobjs, fr)
                ordered = sorted(fr, key=lambda i: ucd[i], reverse=True)
                nxt.extend(union[i] for i in ordered[:pop_size - len(nxt)])
                break
        pop = nxt if nxt else pop


# ── Three P×Q×S arms ─────────────────────────────────────────────────────────

def _front_records_pqs(records: list[dict]) -> list[dict]:
    objs = [(r["lat_us"], -r["ap70"]) for r in records]
    keep = nondominated_idx(objs)
    return sorted((records[i] for i in keep), key=lambda r: r["lat_us"])


def run_joint_pqs(lut, apm, qlut, width_grid, budget, rng, hv_ref, pop_size):
    """A-joint P×Q×S: explore all three dimensions jointly."""
    cost = CostModelPQS(lut, apm, qlut, hv_ref)
    def decode(g):
        return width_grid[g[0]], SCHEDULES[g[1]], QUANT_MODES[g[2]]
    nsga2_pqs(cost, decode,
              [len(width_grid), len(SCHEDULES), len(QUANT_MODES)],
              pop_size, budget, rng)
    pareto = _front_records_pqs(cost.eval_log)
    visited = [(r["width"], r["sched"], r["quant"]) for r in cost.eval_log]
    return ArmResult("A-joint-PQS", cost, pareto, visited,
                     hypervolume_2d(cost.archive, hv_ref))


def run_noS_pqs(lut, apm, qlut, width_grid, budget, rng, hv_ref, pop_size):
    """A-noS P×Q only (S=default always): reveals Q-axis value without S tuning."""
    cost = CostModelPQS(lut, apm, qlut, hv_ref)
    def decode(g):
        return width_grid[g[0]], "default", QUANT_MODES[g[1]]
    nsga2_pqs(cost, decode,
              [len(width_grid), len(QUANT_MODES)],
              pop_size, budget, rng)
    pareto = _front_records_pqs(cost.eval_log)
    visited = [(r["width"], r["sched"], r["quant"]) for r in cost.eval_log]
    return ArmResult("A-noS-PQS", cost, pareto, visited,
                     hypervolume_2d(cost.archive, hv_ref))


def run_serial_pqs(lut, apm, qlut, width_grid, budget, rng, hv_ref, pop_size,
                   stage1_frac=0.5, topk=None, tune_quant_stage2=True):
    """A-serial P×Q×S (FAIR serial): stage1=P under FP16 default,
    stage2 = schedule × quant on the LOCKED widths.

    Crucially, Q IS explored in stage 2 (tune_quant_stage2=True) — serial is NOT
    artificially forbidden from trying INT8.  It still loses, because the widths
    it locked in stage 1 (chosen FP16-default-fast = the misaligned s0=48 greedy
    trap) are STRUCTURALLY int8-unbuildable (in_per_g=3 ÷4 fails).  The aligned
    P_g (s0=64) that owns BOTH the schedule rank-flip AND the int8 path was
    dropped in stage 1 → unreachable.  This is the 3-axis (P×Q×S) coupling:
    one alignment property gates pruning, schedule-tuning, AND quant together.

    tune_quant_stage2=False reproduces the strict (Q-locked) serial.
    """
    cost = CostModelPQS(lut, apm, qlut, hv_ref)
    b1 = max(1, int(budget * stage1_frac))

    # Stage 1: width search under DEFAULT schedule + FP16 (the cheap proxy a
    # stepwise optimizer uses to pick P before touching S or Q).
    def decode1(g):
        return width_grid[g[0]], "default", "fp16"
    nsga2_pqs(cost, decode1, [len(width_grid)], pop_size, b1, rng)

    default_recs = [r for r in cost.eval_log
                    if r["sched"] == "default" and r["quant"] == "fp16"]
    locked = [r["width"] for r in _front_records_pqs(default_recs)]
    if topk:
        extra = sorted((r for r in default_recs if r["width"] not in locked),
                       key=lambda r: r["lat_us"])
        for r in extra[:topk]:
            if r["width"] not in locked:
                locked.append(r["width"])

    # Stage 2: schedule (× quant) search on the locked widths.
    if locked:
        if tune_quant_stage2:
            def decode2(g):
                return locked[g[0]], SCHEDULES[g[1]], QUANT_MODES[g[2]]
            cards = [len(locked), len(SCHEDULES), len(QUANT_MODES)]
        else:
            def decode2(g):
                return locked[g[0]], SCHEDULES[g[1]], "fp16"
            cards = [len(locked), len(SCHEDULES)]
        nsga2_pqs(cost, decode2, cards, pop_size, budget, rng)

    deliverable = [r for r in cost.eval_log if r["width"] in set(locked)]
    pareto = _front_records_pqs(deliverable)
    visited = [(r["width"], r["sched"], r["quant"]) for r in cost.eval_log]
    final_hv = hypervolume_2d([(r["lat_us"], -r["ap70"]) for r in deliverable], hv_ref)
    arm = ArmResult("A-serial-PQS", cost, pareto, visited, final_hv,
                    locked_widths=locked)
    return arm


# ── Q-axis rank-flip detection ────────────────────────────────────────────────

def detect_q_rank_flip_pairs(width_grid: list[Width],
                              lut: LatencyLUT, apm: APModel, qlut: QLookup,
                              ap_tol: float = 1e-4,
                              sched: str = "tuned") -> list[dict]:
    """Detect Q-axis rank-flips: pairs where W_g/P_g ordering flips between FP16 and INT8.

    A Q-rank-flip pair satisfies:
      * Same (s1, s2), differ only in s0 (same as P×S pair criterion)
      * AP70 within ap_tol (weight-identity)
      * Under FP16: wa_lat < wb_lat  (wa is FP16-faster = W_g)
      * Under INT8: wa_lat_int8 > wb_lat_int8  (wb is INT8-faster = P_g)
        i.e. the INT8 speedup of P_g >> INT8 speedup of W_g

    This is independent of the P×S rank-flip (which requires schedule tuning).
    The Q-rank-flip can exist even at schedule='default'.
    """
    pairs: list[dict] = []
    g = list(width_grid)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            wa, wb = g[i], g[j]
            if wa[1:] != wb[1:] or wa[0] == wb[0]:
                continue
            if abs(apm.ap70(wa) - apm.ap70(wb)) > ap_tol:
                continue
            # FP16 latencies
            la_fp16 = lut.latency(wa, sched)
            lb_fp16 = lut.latency(wb, sched)
            # INT8 latencies — pass sched so direct H800 TVM lookup uses correct key
            la_int8 = qlut.int8_lat(lut.latency(wa, sched), wa, sched=sched)
            lb_int8 = qlut.int8_lat(lut.latency(wb, sched), wb, sched=sched)
            # FP16 rank
            wg_fp16, pg_fp16 = (wa, wb) if la_fp16 < lb_fp16 else (wb, wa)
            # INT8 rank
            faster_int8 = wa if la_int8 < lb_int8 else wb
            if faster_int8 == wg_fp16:
                continue   # same order in FP16 and INT8 → no Q rank-flip
            # Q rank-flip confirmed: W_g is FP16-faster but INT8-slower
            is_measured = (qlut.has_direct_h800(wg_fp16, sched) and
                           qlut.has_direct_h800(pg_fp16, sched))
            pairs.append({
                "wg": wg_fp16, "pg": pg_fp16,
                "ap70": round(apm.ap70(wg_fp16), 4),
                "sched": sched,
                "wg_fp16_us": lut.latency(wg_fp16, sched),
                "pg_fp16_us": lut.latency(pg_fp16, sched),
                "wg_int8_us": qlut.int8_lat(lut.latency(wg_fp16, sched), wg_fp16, sched=sched),
                "pg_int8_us": qlut.int8_lat(lut.latency(pg_fp16, sched), pg_fp16, sched=sched),
                "wg_int8_speedup": round(qlut.speedup(wg_fp16), 4),
                "pg_int8_speedup": round(qlut.speedup(pg_fp16), 4),
                "q_rank_flip_ratio": round(
                    qlut.int8_lat(lut.latency(wg_fp16, sched), wg_fp16, sched=sched) /
                    qlut.int8_lat(lut.latency(pg_fp16, sched), pg_fp16, sched=sched), 3),
                "wg_int8_complete": qlut.is_complete(wg_fp16),
                "pg_int8_complete": qlut.is_complete(pg_fp16),
                "source": (
                    "H800_TVM_INT8_measured"
                    if is_measured
                    else (
                        "historical_trt_evidence:4090_q_ratio_context"
                        if qlut.allow_historical_trt_ratio_fallback
                        else "H800_TVM_int8_unmeasured_no_new_backend"
                    )
                ),
            })
    return sorted(pairs, key=lambda p: -p["q_rank_flip_ratio"])


def compute_hv_ref_pqs(width_grid, lut, apm, qlut) -> tuple[float, float]:
    """Nadir for P×Q×S HV scoring: worst (lat, -ap) over full space."""
    lats, aps = [], []
    for w in width_grid:
        for s in SCHEDULES:
            for q in QUANT_MODES:
                lat_fp16 = lut.latency(w, s)
                lat = lat_fp16 if q == "fp16" else qlut.int8_lat(lat_fp16, w)
                ap  = apm.ap70(w) + (_INT8_AP_DELTA_MEDIAN if q == "int8" else 0.0)
                lats.append(lat); aps.append(ap)
    return (max(lats) * 1.05, -(min(aps) - 0.02))


def candidate_widths_pqs(lut: LatencyLUT, apm: APModel,
                          qlut: QLookup) -> list[Width]:
    """Widths with known AP, resolvable H800 TVM latency, AND known Q-ratio.

    Stricter than candidate_widths: drops widths with unknown INT8 speedup
    (Q-ratio defaults to 1.0, which may hide rank-flips).
    """
    out: list[Width] = []
    for w in apm.exact:
        try:
            lut.latency(w, "default"); lut.latency(w, "tuned")
        except ValueError:
            continue
        if not qlut.is_complete(w):
            continue   # INT8 speedup not yet measured → excluded for now
        out.append(w)
    return sorted(out)


# ── P×Q×S Smoke test ─────────────────────────────────────────────────────────

# Smoke widths for P×Q×S: same base set but must all have Q-ratios
SMOKE_WIDTHS_PQS: list[Width] = [
    (64, 128, 256),   # base    — Q-ratio 1.577x (complete)
    (32, 64, 128),    # p50     — Q-ratio 1.281x (complete)
    (16, 32, 64),     # p75     — Q-ratio 1.218x (complete)
    (48, 96, 192),    # trap25  — Q-ratio 1.070x (W_g pair1, s1=96 alignment trap)
    (64, 96, 192),    # pad64   — Q-ratio 1.095x (P_g pair1)
    (48, 64, 256),    # mix_b   — Q-ratio 1.156x (W_g pair2, s1=64)
    (64, 64, 256),    # s1_64   — Q-ratio 1.216x (P_g pair2)
    (48, 128, 128),   # mix_d   — Q-ratio 1.153x (W_g pair3, s2=128; Q-rank-flip pair!)
    (64, 128, 128),   # s2_128  — Q-ratio 1.393x (P_g pair3; INT8-faster at default → Q-flip 1.081x)
]


def run_smoke_pqs(seeds=(0, 1, 2, 3, 4), budget=60, pop_size=8, verbose=True):
    """P×Q×S three-arm smoke test. Tests Q-axis rank-flip detectability."""
    seed_data = load_seed_grid()
    lut = LatencyLUT()
    apm = APModel()
    qlut = QLookup()

    # Check Q-coverage
    complete = [w for w in SMOKE_WIDTHS_PQS if qlut.is_complete(w)]
    pending  = [w for w in SMOKE_WIDTHS_PQS if not qlut.is_complete(w)]
    if pending:
        print(f"[WARNING] {len(pending)} widths have pending Q measurements "
              f"(speedup=1.0 fallback): {pending}")
        print(f"  → Run hw-optimizer to measure: pad64 [64,96,192]")

    hv_ref = compute_hv_ref_pqs(SMOKE_WIDTHS_PQS, lut, apm, qlut)

    print("=" * 84)
    print("THREE-ARM SEARCH SMOKE  (P × Q × S, Q-extended Pyramid search space)")
    print("=" * 84)
    print(f"LUT mode={lut.mode}  AP mode={apm.mode}  Q_LUT Q-complete={len(complete)}/5")
    print(f"INT8 speedup per width:")
    for w in SMOKE_WIDTHS_PQS:
        sp = qlut.speedup(w)
        flag = "✓" if qlut.is_complete(w) else "⚠(estimated=1.0)"
        print(f"  {w}: {sp:.3f}x {flag}")
    print(f"hv_ref(lat,-ap)={hv_ref[0]:.0f},{hv_ref[1]:.3f}")

    # Q-axis rank-flip analysis (before search)
    # ★ Check at "default" schedule: this is where the alignment trap matters.
    # At "tuned", pad64 is already FP16-faster → no FP16-ordering to flip.
    # At "default": trap25 FP16-default is faster; if pad64 INT8-default becomes faster → Q-flip.
    q_pairs = detect_q_rank_flip_pairs(SMOKE_WIDTHS_PQS, lut, apm, qlut, sched="default")
    s_pairs = detect_wg_pg_pairs(SMOKE_WIDTHS_PQS, lut, apm)
    print(f"\nP×S rank-flip pairs (tuned schedule): {len(s_pairs)}")
    for p in s_pairs:
        print(f"  S-flip: W_g={p['wg']} vs P_g={p['pg']} "
              f"ratio={p['iso_ap_latency_ratio']}x")
    print(f"Q-axis rank-flip pairs (FP16 vs INT8): {len(q_pairs)}")
    for p in q_pairs:
        flag = "ESTIMATED" if not p["wg_int8_complete"] or not p["pg_int8_complete"] else "MEASURED"
        print(f"  Q-flip: W_g={p['wg']}(INT8={p['wg_int8_speedup']:.3f}x) "
              f"vs P_g={p['pg']}(INT8={p['pg_int8_speedup']:.3f}x) "
              f"ratio={p['q_rank_flip_ratio']}x [{flag}]")

    summary = {a: [] for a in ("A-joint-PQS", "A-noS-PQS", "A-serial-PQS")}

    for sd in seeds:
        if verbose:
            print(f"\n--- seed {sd} ---")
        rng_j = random.Random(sd)
        rng_n = random.Random(sd)
        rng_s = random.Random(sd)
        res_j = run_joint_pqs(lut, apm, qlut, SMOKE_WIDTHS_PQS, budget, rng_j, hv_ref, pop_size)
        res_n = run_noS_pqs(lut, apm, qlut, SMOKE_WIDTHS_PQS, budget, rng_n, hv_ref, pop_size)
        res_s = run_serial_pqs(lut, apm, qlut, SMOKE_WIDTHS_PQS, budget, rng_s, hv_ref, pop_size)
        for res, name in ((res_j, "A-joint-PQS"), (res_n, "A-noS-PQS"), (res_s, "A-serial-PQS")):
            summary[name].append(res)
            if verbose:
                extra = ""
                if res.locked_widths is not None:
                    extra = "  locked=" + str([w for w in res.locked_widths])
                quants = {r["quant"] for r in res.cost.eval_log}
                print(f"  {name:15s} HV={res.final_hv:.4e}  "
                      f"evals={res.cost.budget_used():2d}  "
                      f"Q={sorted(quants)}{extra}")

    print("\n" + "=" * 84)
    print("AGGREGATE  (mean HV, normalized)")
    print("=" * 84)
    ref_hv_j = sum(r.final_hv for r in summary["A-joint-PQS"]) / len(summary["A-joint-PQS"])
    for name in ("A-joint-PQS", "A-noS-PQS", "A-serial-PQS"):
        hvs = [r.final_hv for r in summary[name]]
        mean = sum(hvs) / len(hvs)
        pct  = mean / ref_hv_j * 100 if ref_hv_j > 0 else 0
        print(f"  {name:15s} mean HV={mean:.4e} ({pct:5.1f}% of A-joint)")

    # Structural claim: A-joint reaches INT8 for P_g; A-serial never does
    pad64 = (64, 96, 192)
    j_int8_pg = any(r["quant"] == "int8" and r["width"] == pad64
                    for res in summary["A-joint-PQS"] for r in res.cost.eval_log)
    s_int8_pg = any(r["quant"] == "int8" and r["width"] == pad64
                    for res in summary["A-serial-PQS"] for r in res.cost.eval_log)
    s_q_never = all(r["quant"] == "fp16"
                    for res in summary["A-serial-PQS"] for r in res.cost.eval_log)
    print("\n" + "=" * 84)
    print("Q-STRUCTURAL CLAIM  (A-serial Q-axis exclusion)")
    print("=" * 84)
    print(f"  A-joint-PQS  visits (pad64, *, int8): {j_int8_pg}   [Q-P_g reached]")
    print(f"  A-serial-PQS visits (pad64, *, int8): {s_int8_pg}   [should be False]")
    print(f"  A-serial-PQS Q is always fp16: {s_q_never}  [should be True]")
    ok = j_int8_pg and (not s_int8_pg) and s_q_never
    print(f"\n  RESULT: " + ("PASS — A-serial structurally excludes Q-axis; "
                              "A-joint-PQS explores INT8." if ok else "FAIL — see above."))
    if q_pairs:
        print(f"\n  Q-rank-flip pairs confirmed: {len(q_pairs)} "
              f"(A-serial misses INT8 on W_g's aligned partner P_g)")
    elif pending:
        print(f"\n  [NOTE] pad64 Q-ratio pending — Q-rank-flip unconfirmed until "
              f"hw-optimizer completes build")
    print("=" * 84)
    return ok


__all__ = [
    # P×S (original)
    "LatencyLUT", "APModel", "CostModel", "ArmResult",
    "run_joint", "run_noS", "run_serial",
    "nsga2", "hypervolume_2d", "compute_hv_ref", "reference_pareto",
    "candidate_widths", "detect_wg_pg_pairs", "SMOKE_WIDTHS", "SCHEDULES",
    "run_smoke",
    # P×Q×S extension (T1)
    "QUANT_MODES", "QLookup", "CostModelPQS",
    "run_joint_pqs", "run_noS_pqs", "run_serial_pqs",
    "nsga2_pqs", "detect_q_rank_flip_pairs",
    "candidate_widths_pqs", "compute_hv_ref_pqs",
    "SMOKE_WIDTHS_PQS", "Q_LUT_JSON",
    "run_smoke_pqs",
]
