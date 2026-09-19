"""Lossless 12-bit GEMV microbench on real weights: escapes, exactness, speed."""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("safetensors==0.5.3", "numpy", "huggingface_hub")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-df12", image=image)


@app.function(gpu="H100", timeout=1200, volumes={"/weights": vol})
def run(ms: str):
    import sys, glob, json, torch
    sys.path.insert(0, "/root/engine")
    from safetensors import safe_open
    from kernels.gemv import gemv
    from kernels import df12

    snap = glob.glob("/weights/hf/**/model.safetensors.index.json", recursive=True)[0]
    root = os.path.dirname(snap)
    index = json.load(open(snap))["weight_map"]

    def get(name):
        with safe_open(os.path.join(root, index[name]), "pt", device="cuda") as f:
            return f.get_tensor(name)

    def t(f, reps=20):
        f(); torch.cuda.synchronize()
        best = 1e9
        for _ in range(3):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(reps):
                f()
            e1.record(); e1.synchronize()
            best = min(best, e0.elapsed_time(e1) / reps)
        return best * 1e3

    L = "model.layers.17."
    mats = {
        "qkv": (torch.cat([get(L + f"self_attn.{n}_proj.weight") for n in "qkv"], 0).contiguous(), 2),
        "o": (get(L + "self_attn.o_proj.weight"), 4),
        "gu": (torch.cat([get(L + "mlp.gate_proj.weight"), get(L + "mlp.up_proj.weight")], 0).contiguous(), 1),
        "down": (get(L + "mlp.down_proj.weight"), 4),
        "lm": (get("model.embed_tokens.weight"), 1),
    }
    for name, (w, split) in mats.items():
        N, K = w.shape
        p = df12.pack(w, max_esc=10 ** 9)
        W = p[3].shape[1]
        nesc = int((p[4] != 0).sum())
        exact = bool(torch.equal(df12.unpack(p).view(torch.int16), w.view(torch.int16)))
        nbytes = sum(a.numel() * a.element_size() for a in p)
        print(f"RESULT {name} [{N},{K}] escapes={nesc} ({nesc / w.numel():.2e}) maxrow={W} "
              f"roundtrip_exact={exact} bytes={nbytes / (w.numel() * 2):.3f}x", flush=True)
        for M in (int(m) for m in ms.split(",")):
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
            try:
                ref = gemv(x, w, split)
                got = df12.gemv_df12(x, p, split)
                if split > 1:
                    ref, got = ref.sum(0), got.sum(0)
                err = (got.float() - ref.float()).abs().max().item()
                same = bool(torch.equal(got, ref))
                t0 = t(lambda: gemv(x, w, split))
                t1 = t(lambda: df12.gemv_df12(x, p, split))
                print(f"RESULT   M={M}: bf16 {t0:.1f}us  df12 {t1:.1f}us  speedup {t0 / t1:.2f}x  "
                      f"maxerr {err:.3g} (ref max {ref.float().abs().max().item():.3g}) bit_equal={same}", flush=True)
            except Exception as e:
                print(f"RESULT   M={M}: FAILED " + repr(e)[-700:], flush=True)


@app.local_entrypoint()
def main(ms: str = "1,4,16,64"):
    run.remote(ms)
