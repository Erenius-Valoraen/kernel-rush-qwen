"""Try each flash-prefill config in its own subprocess (compiler crashes abort the process)."""
import os
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("numpy")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
app = modal.App("kernel-rush-fa", image=image)

CHILD = r'''
import sys, torch
sys.path.insert(0, "/root/engine")
import torch.nn.functional as F
from kernels.flash_prefill import _flash_prefill_kernel
bq, bk, w, st, G, S = (int(a) for a in sys.argv[1:7])
nq, nkv, d = 32, 8, 128
torch.manual_seed(0)
q = torch.randn(G * S, nq * d, device="cuda", dtype=torch.bfloat16)
kc = torch.randn(G, nkv, S + 64, d, device="cuda", dtype=torch.bfloat16)
vc = torch.randn(G, nkv, S + 64, d, device="cuda", dtype=torch.bfloat16)
out = torch.empty_like(q)
fn = _flash_prefill_kernel.fn
def tri():
    fn[(triton.cdiv(S, bq), nq, G)](q, kc, vc, out, S, 0, kc.stride(0), kc.stride(1), d ** -0.5,
        NQ=nq, NKV=nkv, D=d, BQ=bq, BKV=bk, DOT_F32=False, num_warps=w, num_stages=st)
    return out
import triton
def sdpa():
    qq = q.view(G, S, nq, d).transpose(1, 2)
    return F.scaled_dot_product_attention(qq, kc[:, :, :S], vc[:, :, :S], is_causal=True, enable_gqa=True)
def t(f):
    f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(5): f()
    e1.record(); e1.synchronize()
    return e0.elapsed_time(e1) / 5
ref = sdpa().transpose(1, 2).reshape(G * S, nq * d).float()
err = (tri().float() - ref).abs().max().item()
print(f"RESULT bq{bq} bk{bk} w{w} st{st} G{G} S{S}: triton {t(tri):.2f}ms sdpa {t(sdpa):.2f}ms err {err:.4f}", flush=True)
'''

@app.function(gpu="H100", timeout=1200)
def run(gs: str):
    import subprocess, itertools
    open("/tmp/child.py", "w").write(CHILD)
    for shp in gs.split(","):
        G, S = shp.split("x")
        for bq, bk, w, st in itertools.product((32, 64, 128), (64, 128, 256), (4, 8), (2, 3)):
            r = subprocess.run(["python", "/tmp/child.py", str(bq), str(bk), str(w), str(st), G, S],
                               capture_output=True, text=True, timeout=300)
            line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
            print(line[0] if line else f"CRASH bq{bq} bk{bk} w{w} st{st} G{G} S{S}: {(r.stderr or '')[-160:].strip()!r}", flush=True)

@app.local_entrypoint()
def main(gs: str = "4x2048,16x512"):
    run.remote(gs)
