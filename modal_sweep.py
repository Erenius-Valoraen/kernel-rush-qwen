"""Sweep decode-matmul kernel configs on a Modal H100 (random weights, no model load).

  modal run modal_sweep.py --ms 1,16
"""

import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy")
    .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"])
)
app = modal.App("kernel-rush-sweep", image=image)


@app.function(gpu="H100", timeout=1500)
def sweep(ms: str, names: str):
    import sys, itertools
    sys.path.insert(0, "/root/engine")
    import torch
    import torch.nn.functional as F
    import triton
    from kernels.gemv import _gemv_kernel, _gemv_swiglu_kernel
    from kernels.ops import silu_mul

    dev = "cuda"
    H, I, QKV, QD, V = 2560, 9728, 6144, 4096, 151936
    shapes = dict(qkv=(QKV, H, 36), o=(H, QD, 36), gu=(2 * I, H, 36), down=(H, I, 36), lm=(V, H, 3))
    gk, sk = _gemv_kernel.fn, _gemv_swiglu_kernel.fn

    def timeit(fn, reps=7):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1))
        return best * 1e3

    for M in (int(m) for m in ms.split(",")):
        BM = max(16, triton.next_power_of_2(M))
        for name in names.split(","):
            N, K, copies = shapes[name]
            ws = [torch.randn((N, K), device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
            x = torch.randn((M, K), device=dev, dtype=torch.bfloat16)
            gb = N * K * 2 / 1e9
            res = []
            if name == "gu":
                t = timeit(lambda: [silu_mul(F.linear(x, w)) for w in ws]) / copies
            else:
                t = timeit(lambda: [F.linear(x, w) for w in ws]) / copies
            res.append((t, "cublas"))
            n_out = N // 2 if name == "gu" else N
            splits = (1,) if name in ("gu", "lm") else (1, 2, 4, 8)
            for split, bn, bk, warps, stages in itertools.product(
                    splits, (16, 32, 64, 128), (64, 128, 256), (4, 8), (2, 4)):
                if (K // split) % bk or n_out % bn:
                    continue
                if split > 1:
                    out = torch.empty((split, M, N), device=dev, dtype=torch.float32)
                else:
                    out = torch.empty((M, n_out), device=dev, dtype=torch.bfloat16)

                def run():
                    for w in ws:
                        if name == "gu":
                            sk[(n_out // bn,)](x, w, out, M, n_out, K, BM=BM, BN=bn, BK=bk,
                                               DOT_F32=False, num_warps=warps, num_stages=stages)
                        else:
                            gk[(N // bn, split)](x, w, out, M, N, K, K // split, BM=BM, BN=bn, BK=bk,
                                                 PARTIAL=split > 1, DOT_F32=False,
                                                 num_warps=warps, num_stages=stages)
                try:
                    res.append((timeit(run, 4) / copies, f"s{split} bn{bn} bk{bk} w{warps} st{stages}"))
                except Exception as e:
                    pass
            res.sort()
            cub = [r for r in res if r[1] == "cublas"][0][0]
            print(f"M={M} {name}: cublas {cub:.1f}us ({gb / cub * 1e6 / 1e3:.2f} TB/s) | best: " +
                  " ; ".join(f"{c} {t:.1f}us ({gb / t * 1e3:.2f}TB/s)" for t, c in res[:5]), flush=True)
            del ws
            torch.cuda.empty_cache()


@app.local_entrypoint()
def main(ms: str = "1,4,16", names: str = "qkv,o,gu,down,lm"):
    calls = [sweep.spawn(m, n) for m in ms.split(",") for n in names.split(",")]
    for c in calls:
        c.get()
