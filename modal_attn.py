"""Decode-attention sweep at large batch on a Modal H100."""
import os
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
app = modal.App("kernel-rush-attn", image=image)


@app.function(gpu="H100", timeout=1500)
def run(shapes: str):
    import sys, itertools
    sys.path.insert(0, "/root/engine")
    import torch
    from kernels.fused_attn import FusedDecodeAttention
    nq, nkv, d, L = 32, 8, 128, 8
    for shp in shapes.split(","):
        B, ctx = (int(v) for v in shp.split("x"))
        cap = -(-(ctx + 16) // 128) * 128
        kc = torch.randn(L, B, nkv, cap, d, device="cuda", dtype=torch.bfloat16)
        vc = torch.randn_like(kc)
        qkv = torch.randn(B, (nq + 2 * nkv) * d, device="cuda", dtype=torch.bfloat16)
        qw = torch.ones(d, device="cuda", dtype=torch.bfloat16); kw = qw.clone()
        cos = torch.randn(cap + 8, d, device="cuda", dtype=torch.bfloat16); sin = cos.clone()
        pos = torch.full((B,), ctx, device="cuda", dtype=torch.int32)
        gb = B * (ctx + 1) * nkv * d * 2 * 2 / 1e9
        res = []
        for sms, (bn, w, st) in itertools.product((B * nkv // 2, B * nkv, B * nkv * 2, B * nkv * 4, B * nkv * 8),
                                                  [(32, 4, 3), (64, 4, 2), (64, 4, 3), (64, 8, 3), (128, 4, 2),
                                                   (128, 4, 3), (128, 8, 3), (256, 8, 2), (256, 8, 3)]):
            try:
                att = FusedDecodeAttention(B, 1, cap, nq, nkv, d, "cuda", max(1, sms // 2))
                att.cfg = (bn, w, st); att.tuned = True
                def f():
                    for l in range(L):
                        att(qkv, qw, kw, cos, sin, kc[l], vc[l], pos, 1e-6)
                f(); torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    f()
                best = 1e9
                for _ in range(5):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); g.replay(); e1.record(); e1.synchronize()
                    best = min(best, e0.elapsed_time(e1) / L)
                res.append((best, f"nsplit{att.nsplit} bn{bn} w{w} st{st}"))
            except Exception as e:
                res.append((1e9, f"ERR {type(e).__name__} nsplit? bn{bn} w{w} st{st}"))
        res.sort()
        cur = [r for r in res if "bn64 w4 st2" in r[1]]
        print(f"RESULT B={B} ctx={ctx} ({gb * 1e3:.0f}MB/layer): " +
              " ; ".join(f"{c} {t * 1e3:.0f}us ({gb / t * 1e3 / 1e3:.2f}TB/s)" for t, c in res[:4]) +
              " || current-ish: " + " ; ".join(f"{c} {t * 1e3:.0f}us" for t, c in cur[:5]), flush=True)


@app.local_entrypoint()
def main(shapes: str = "16x640,32x640,64x640,32x2080,8x2080,1x640"):
    calls = [run.spawn(s) for s in shapes.split(",")]
    for c in calls:
        c.get()
