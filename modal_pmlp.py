"""Persistent fused MLP vs separate gate/up + down GEMVs on a Modal H100."""
import os
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
app = modal.App("kernel-rush-pmlp", image=image)


@app.function(gpu="H100", timeout=900)
def run():
    import sys
    sys.path.insert(0, "/root/engine")
    import torch
    from kernels.gemv import gemv, gemv_swiglu
    from kernels.mlp import PersistentMLP, _mlp_kernel
    H, I, L = 2560, 9728, 12
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print("SMs", sms)
    def timeit(fn):
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        best = 1e9
        for _ in range(7):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1) * 1e3 / L)
        return best
    wg = [torch.randn(2 * I, H, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
    wd = [torch.randn(H, I, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(L)]
    for M in (1, 16):
        h = torch.randn(M, H, device="cuda", dtype=torch.bfloat16)
        ref = gemv(gemv_swiglu(h, wg[0]), wd[0], 4).sum(0)
        t_sep = timeit(lambda: [gemv(gemv_swiglu(h, wg[l]), wd[l], 4) for l in range(L)])
        print(f"RESULT M={M} separate: {t_sep:.1f} us/layer", flush=True)
        for G in (sms, 2 * sms, 3 * sms, 4 * sms):
            for split, bn, bn2 in ((4, 16, 16), (4, 32, 32), (2, 32, 32), (8, 32, 16)):
                try:
                    pm = PersistentMLP(M, H, I, "cuda", G, split=split, bn=bn, bn2=bn2)
                    got = pm(h, wg[0], wd[0]).sum(0)
                    torch.cuda.synchronize()
                    err = (got - ref).abs().max().item()
                    t = timeit(lambda: [pm(h, wg[l], wd[l]) for l in range(L)])
                    print(f"RESULT M={M} pmlp G={G} split={split} bn={bn}/{bn2}: {t:.1f} us/layer  err {err:.4f}", flush=True)
                except Exception as e:
                    print(f"RESULT M={M} pmlp G={G} split={split} bn={bn}/{bn2}: ERR {type(e).__name__} {str(e)[:100]}", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
