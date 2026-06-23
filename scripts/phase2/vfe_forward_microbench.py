"""PillarVFE forward micro-benchmark on Orin — isolate VFE eager cost and the
cudnn-disabled-BN trap, to fully decompose encoder=17.49ms = VFE-fwd + scatter.

Reproduces pillar_vfe.py PFNLayer forward (DAIR m1: single layer):
  Linear(10->64) on [M,32,10] -> BN1d(64) [cudnn toggled OFF in HEAL!] -> ReLU
  -> max over point dim -> [M,64]
baseline = with the `torch.backends.cudnn.enabled=False` wrap (HEAL verbatim).
opt      = cudnn left enabled (drop the trap).

Usage: python vfe_forward_microbench.py [M] [device]
"""
import sys, torch, torch.nn as nn, torch.nn.functional as F

M = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
P, DIN, DOUT = 32, 10, 64
dev = torch.device(DEV)
torch.manual_seed(0)
x = torch.randn(M, P, DIN, device=dev)
lin = nn.Linear(DIN, DOUT).to(dev).eval()
bn = nn.BatchNorm1d(DOUT).to(dev).eval()


def pfn(inputs, toggle_cudnn):
    h = lin(inputs)                                  # [M,32,64]
    if toggle_cudnn:
        torch.backends.cudnn.enabled = False         # HEAL trap
        h = bn(h.permute(0, 2, 1)).permute(0, 2, 1)
        torch.backends.cudnn.enabled = True
    else:
        h = bn(h.permute(0, 2, 1)).permute(0, 2, 1)
    h = F.relu(h)
    return torch.max(h, dim=1, keepdim=True)[0]      # [M,1,64]


def timeit(fn, *a, iters=200, warmup=50):
    with torch.no_grad():
        for _ in range(warmup):
            fn(*a)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn(*a)
        e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"VFE-forward microbench M={M} dev={DEV} (Linear({DIN}->{DOUT})+BN+ReLU+max)")
with torch.no_grad():
    a = pfn(x, True); b = pfn(x, False)
print(f"equiv (toggle vs no-toggle) maxdiff={(a-b).abs().max().item():.3e}")
t_trap = timeit(pfn, x, True)
t_opt = timeit(pfn, x, False)
print(f"baseline(cudnn-OFF trap) = {t_trap*1000:8.1f} us")
print(f"opt(cudnn enabled)       = {t_opt*1000:8.1f} us   speedup={t_trap/t_opt:.2f}x")
