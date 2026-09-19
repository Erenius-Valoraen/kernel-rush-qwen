"""Prefill GEMM sweep: Triton vs cuBLAS at M=8192 on a Modal H100."""
import os
import modal
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy"))
app = modal.App("kernel-rush-gemm", image=image)

SRC = r'''
import sys, itertools, torch, triton, triton.language as tl
import torch.nn.functional as F

@triton.jit
def mm(x_ptr, w_ptr, o_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM); num_n = N // BN
    group = GM * num_n
    gid = pid // group
    first_m = gid * GM
    gsize = tl.minimum(num_m - first_m, GM)
    pm = first_m + (pid % group) % gsize
    pn = (pid % group) // gsize
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    mmask = offs_m < M
    acc = tl.zeros([BM, BN], tl.float32)
    xr = x_ptr + offs_m[:, None].to(tl.int64) * K
    wr = w_ptr + offs_n[:, None].to(tl.int64) * K
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(xr + offs_k[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(wr + offs_k[None, :])
        acc += tl.dot(x, tl.trans(w))
    tl.store(o_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc.to(tl.bfloat16), mask=mmask[:, None])

def t(f, reps=5):
    f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps): f()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / reps

M = int(sys.argv[1])
for name, (N, K) in dict(qkv=(6144, 2560), o=(2560, 4096), gu=(19456, 2560), down=(2560, 9728)).items():
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16); w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fl = 2 * M * N * K / 1e12
    tc = t(lambda: F.linear(x, w))
    res = []
    for bm, bn, bk, gm, wp, st in itertools.product((64, 128), (64, 128), (64, 128), (4, 8), (8, 16), (3, 5)):
        try:
            g = (triton.cdiv(M, bm) * (N // bn),)
            res.append((t(lambda: mm[g](x, w, o, M, N, K, BM=bm, BN=bn, BK=bk, GM=gm, num_warps=wp, num_stages=st), 3),
                        f"bm{bm} bn{bn} bk{bk} gm{gm} w{wp} st{st}"))
        except Exception as e:
            pass
    res.sort()
    print(f"RESULT M={M} {name}: cublas {tc:.3f}ms ({fl / tc * 1e3:.0f} TFLOPS) | " +
          " ; ".join(f"{c} {v:.3f}ms ({fl / v * 1e3:.0f})" for v, c in res[:3]), flush=True)
'''

@app.function(gpu="H100", timeout=1500)
def run(m: int):
    import subprocess
    open("/tmp/g.py", "w").write(SRC)
    r = subprocess.run(["python", "/tmp/g.py", str(m)], capture_output=True, text=True)
    print(r.stdout[-3000:]); print(r.stderr[-800:])

@app.local_entrypoint()
def main(m: int = 8192):
    run.remote(m)
