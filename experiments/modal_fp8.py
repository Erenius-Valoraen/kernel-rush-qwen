"""FP8 microbench on real weights: decode GEMV (graph-timed) and prefill _scaled_mm."""
import os
import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("safetensors==0.5.3", "numpy", "huggingface_hub")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-fp8", image=image)


@app.function(gpu="H100", timeout=1200, volumes={"/weights": vol})
def run(ms: str):
    import sys, glob, json, torch
    import torch.nn.functional as F
    sys.path.insert(0, "/root/engine")
    from safetensors import safe_open
    from kernels.gemv import gemv, gemv_swiglu
    from kernels import fp8

    snap = glob.glob("/weights/hf/**/model.safetensors.index.json", recursive=True)[0]
    root = os.path.dirname(snap)
    index = json.load(open(snap))["weight_map"]

    def get(name):
        with safe_open(os.path.join(root, index[name]), "pt", device="cuda") as f:
            return f.get_tensor(name)

    def tg(f, reps=36):
        """us per call, replayed from a CUDA graph of `reps` calls."""
        f(); torch.cuda.synchronize()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            f()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                f()
        torch.cuda.synchronize()
        best = 1e9
        for _ in range(5):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1) / reps)
        return best * 1e3

    L = "model.layers.17."
    mats = {
        "qkv": (torch.cat([get(L + f"self_attn.{n}_proj.weight") for n in "qkv"], 0).contiguous(), 2),
        "o": (get(L + "self_attn.o_proj.weight"), 4),
        "down": (get(L + "mlp.down_proj.weight"), 4),
        "gu": (torch.cat([get(L + "mlp.gate_proj.weight"), get(L + "mlp.up_proj.weight")], 0).contiguous(), 0),
    }
    for name, (w, split) in mats.items():
        N, K = w.shape
        q = fp8.quantize(w)
        deq = q[0].float() * q[1][:, None]
        rel = ((deq - w.float()).norm() / w.float().norm()).item()
        print(f"RESULT {name} [{N},{K}] weight rel err {rel:.4f}", flush=True)
        for M in (int(m) for m in ms.split(",")):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
            try:
                if split == 0:
                    fr, fq = (lambda: gemv_swiglu(x, w)), (lambda: fp8.gemv_swiglu_fp8(x, q))
                else:
                    fr, fq = (lambda: gemv(x, w, split)), (lambda: fp8.gemv_fp8(x, q, split))
                ref, got = fr().float(), fq().float()
                if split > 1:
                    ref, got = ref.sum(0), got.sum(0)
                err = ((got - ref).norm() / ref.norm()).item()
                t0, t1 = tg(fr), tg(fq)
                print(f"RESULT   M={M}: bf16 {t0:.1f}us  fp8 {t1:.1f}us  speedup {t0 / t1:.2f}x  out rel err {err:.4f}", flush=True)
            except Exception as e:
                print(f"RESULT   M={M}: FAILED " + repr(e)[-600:], flush=True)
        for M in (512, 8192):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
            try:
                ref = F.linear(x, w).float(); got = fp8.linear_fp8(x, q).float()
                err = ((got - ref).norm() / ref.norm()).item()
                t0, t1 = tg(lambda: F.linear(x, w), 8), tg(lambda: fp8.linear_fp8(x, q), 8)
                x8, xs = fp8.quant_rows(x)
                t2 = tg(lambda: torch._scaled_mm(x8, q[0].t(), scale_a=xs, scale_b=q[1].view(1, -1), out_dtype=torch.bfloat16), 8)
                print(f"RESULT   prefill M={M}: cublas {t0:.0f}us  fp8 {t1:.0f}us (mm only {t2:.0f}us)  speedup {t0 / t1:.2f}x  rel err {err:.4f}", flush=True)
            except Exception as e:
                print(f"RESULT   prefill M={M}: FAILED " + repr(e)[-600:], flush=True)


@app.local_entrypoint()
def main(ms: str = "1,4,16,64"):
    run.remote(ms)
