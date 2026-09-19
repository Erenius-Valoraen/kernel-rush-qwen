"""Time the residual+RMSNorm kernel variants on a Modal H100."""
import os
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
app = modal.App("kernel-rush-norm", image=image)


@app.function(gpu="H100", timeout=900)
def run():
    import sys
    sys.path.insert(0, "/root/engine")
    import torch, triton
    from kernels.ops import _add_rmsnorm_kernel
    H = 2560
    def timeit(fn, n=72):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n): fn()
        best = 1e9
        for _ in range(7):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1) * 1e3 / n)
        return best
    w = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    for M in (1, 4, 16, 64):
        x = torch.randn(M, H, device="cuda", dtype=torch.bfloat16); y = torch.empty_like(x)
        d16 = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        for S in (0, 2, 4):
            d = d16 if S == 0 else torch.randn(S, M, H, device="cuda", dtype=torch.float32)
            row = []
            for blk, wp, st in ((4096, 8, 3), (4096, 4, 3), (4096, 16, 3), (4096, 2, 3), (4096, 1, 3), (4096, 8, 1), (4096, 16, 1)):
                try:
                    t = timeit(lambda: _add_rmsnorm_kernel[(M,)](x, d, w, y, H, M * H, 1e-6, HAS_DELTA=True, DSPLIT=S,
                                                                BLOCK=blk, num_warps=wp, num_stages=st))
                    row.append(f"w{wp}s{st}:{t:.1f}")
                except Exception as e:
                    row.append(f"w{wp}s{st}:ERR")
            print(f"RESULT M={M} dsplit={S}: " + "  ".join(row) + " us/launch", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
