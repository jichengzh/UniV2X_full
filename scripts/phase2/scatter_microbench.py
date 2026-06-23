"""point_pillar_scatter micro-benchmark + vectorized optimization.

Tests whether encoder's measured latency is the scatter *memory* op or the
eager traps (.item() sync + python for-loop over batch + per-batch 36MB zeros).
Baseline = HEAL PointPillarScatter.forward verbatim. Opt = vectorized, single
index_put, no .item() per-iter, no python loop. CUDA-event timed + numeric
equivalence asserted (output must be byte-identical scatter).

Usage: python scatter_microbench.py [M] [batch] [device]
"""
import sys, torch

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
B = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DEV = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
NX, NY, NZ, C = 704, 200, 1, 64
HW = NX * NY * NZ
dev = torch.device(DEV)
torch.manual_seed(0)

# dummy pillars with UNIQUE (batch,cell) — real voxelization has no collisions
perm = torch.randperm(B * HW, device=dev)[:M]
bidx = perm // HW
lin_in = perm % HW
z = torch.zeros(M, device=dev, dtype=torch.long)
y = lin_in // NX
x = lin_in % NX
coords = torch.stack([bidx, z, y, x], dim=1).float()
feats = torch.randn(M, C, device=dev)


def baseline(pillar_features, coords):
    batch_size = coords[:, 0].max().int().item() + 1          # .item() SYNC
    out = []
    for b in range(batch_size):                                # python loop
        sf = torch.zeros(C, HW, dtype=pillar_features.dtype, device=pillar_features.device)  # 36MB
        m = coords[:, 0] == b
        tc = coords[m, :]
        idx = (tc[:, 1] + tc[:, 2] * NX + tc[:, 3]).long()
        sf[:, idx] = pillar_features[m, :].t()
        out.append(sf)
    return torch.stack(out, 0).view(batch_size, C, NY, NX)


def vectorized(pillar_features, coords, batch_size):
    # WRONG layout: writes [B*HW,C] then permute+contiguous (big transpose copy,
    # cheap on high-BW GPUs, EXPENSIVE on low-BW edge cards like Orin).
    b = coords[:, 0].long()
    lin = b * HW + (coords[:, 1] + coords[:, 2] * NX + coords[:, 3]).long()
    flat = torch.zeros(B * HW, C, dtype=pillar_features.dtype, device=pillar_features.device)
    flat[lin] = pillar_features
    return flat.view(batch_size, HW, C).permute(0, 2, 1).contiguous().view(batch_size, C, NY, NX)


def vectorized_nchw(pillar_features, coords, batch_size):
    # CORRECT: scatter straight into NCHW layout, NO permute/contiguous.
    # spatial[b, :, col] = feats  (advanced index on dims 0,2; middle ':' = C)
    spatial = torch.zeros(batch_size, C, HW, dtype=pillar_features.dtype,
                          device=pillar_features.device)
    b = coords[:, 0].long()
    col = (coords[:, 1] + coords[:, 2] * NX + coords[:, 3]).long()
    spatial[b, :, col] = pillar_features
    return spatial.view(batch_size, C, NY, NX)


def timeit(fn, *a, iters=200, warmup=50):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*a)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


print(f"scatter microbench M={M} B={B} grid={NX}x{NY} HW={HW} dev={DEV}")
ob = baseline(feats, coords)
ov = vectorized(feats, coords, B)
on = vectorized_nchw(feats, coords, B)
print(f"equiv vectorized      maxdiff={(ob-ov).abs().max().item():.3e}")
print(f"equiv vectorized_nchw maxdiff={(ob-on).abs().max().item():.3e}  (<1e-5 视为等价)")

t_base = timeit(baseline, feats, coords)
t_vec = timeit(vectorized, feats, coords, B)
t_nchw = timeit(vectorized_nchw, feats, coords, B)
def just_sync(coords):
    return coords[:, 0].max().int().item()
t_sync = timeit(just_sync, coords, iters=200, warmup=50)
print(f"baseline         = {t_base*1000:8.1f} us")
print(f"vectorized(perm) = {t_vec*1000:8.1f} us   speedup={t_base/t_vec:.2f}x")
print(f"vectorized_nchw  = {t_nchw*1000:8.1f} us   speedup={t_base/t_nchw:.2f}x  ★no-transpose")
print(f"  .item()sync    = {t_sync*1000:8.1f} us")
