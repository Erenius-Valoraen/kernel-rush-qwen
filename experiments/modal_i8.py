"""INT8-weight GEMV variants + per-tensor FP8 prefill microbench."""
import os
import modal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("safetensors==0.5.3", "numpy", "huggingface_hub")
         .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-i8", image=image)

SRC = r'''
import sys, os, glob, json, itertools, torch, triton, triton.language as tl
import torch.nn.functional as F
sys.path.insert(0, "/root/engine")
from safetensors import safe_open
from kernels.gemv import gemv

@triton.jit
def k_i8(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    w_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    x_base = x_ptr + offs_m[:, None] * K
    if MODE == 2:
        acc2 = tl.zeros([BN, BK], tl.float32)
        for kk in range(0, K, BK):
            offs_k = kk + tl.arange(0, BK)
            x = tl.load(x_ptr + offs_k).to(tl.float32)
            w = tl.load(w_base + offs_k[None, :]).to(tl.float32)
            acc2 += w * x[None, :]
        acc = tl.sum(acc2, axis=1)[None, :]
    else:
        acc = tl.zeros([BM, BN], tl.float32)
        for kk in range(0, K, BK):
            offs_k = kk + tl.arange(0, BK)
            x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
            w = tl.load(w_base + offs_k[None, :])
            if MODE == 0:
                acc += tl.dot(x, tl.trans(w.to(tl.bfloat16)))
            elif MODE == 1:
                acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
            elif MODE == 3:
                acc += tl.dot(x, tl.trans(w.to(tl.float32).to(tl.bfloat16)))
            elif MODE == 4:
                acc += tl.sum(x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2)
            elif MODE == 5:
                acc += tl.dot(x, tl.trans(w.to(tl.float16).to(tl.bfloat16)))
    acc = acc * tl.load(s_ptr + offs_n)[None, :]
    dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
    tl.store(dst, acc.to(tl.bfloat16), mask=mmask[:, None])

def tg(f, reps=36):
    f(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): f()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize()
        best = min(best, e0.elapsed_time(e1) / reps)
    return best * 1e3

snap = glob.glob("/weights/hf/**/model.safetensors.index.json", recursive=True)[0]
root = os.path.dirname(snap); index = json.load(open(snap))["weight_map"]
def get(name):
    with safe_open(os.path.join(root, index[name]), "pt", device="cuda") as f:
        return f.get_tensor(name)
L = "model.layers.17."
mats = {"gu": (torch.cat([get(L + "mlp.gate_proj.weight"), get(L + "mlp.up_proj.weight")], 0).contiguous(), 1),
        "down": (get(L + "mlp.down_proj.weight"), 4),
        "qkv": (torch.cat([get(L + f"self_attn.{n}_proj.weight") for n in "qkv"], 0).contiguous(), 2)}
for name, (w, split) in mats.items():
    N, K = w.shape
    wf = w.float(); s = wf.abs().amax(1) / 127.0
    w8 = torch.round(wf / s[:, None]).clamp_(-127, 127).to(torch.int8).contiguous()
    rel = (((w8.float() * s[:, None]) - wf).norm() / wf.norm()).item()
    print(f"RESULT {name} [{N},{K}] int8 weight rel err {rel:.4f}", flush=True)
    q8 = (wf / (wf.abs().max() / 448)).to(torch.float8_e4m3fn)
    sw = (wf.abs().max() / 448).float().reshape(())
    for M in (1, 4, 16, 32, 64, 128, 512, 8192):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
        sx = (x.abs().max().float() / 448).reshape(())
        x8 = (x.float() / sx).to(torch.float8_e4m3fn)
        reps = 36 if M <= 128 else 8
        t0 = tg(lambda: F.linear(x, w), reps)
        tt = tg(lambda: gemv(x, w, split), reps) if M <= 128 else 0
        try:
            t1 = tg(lambda: torch._scaled_mm(x8, q8.t(), scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16), reps)
        except Exception as e:
            t1 = -1; print("ERR", repr(e)[-300:])
        print(f"RESULT   M={M}: cublas {t0:.1f}us  triton-bf16 {tt:.1f}us  fp8 scaled_mm {t1:.1f}us", flush=True)
'''

@app.function(gpu="H100", timeout=1500, volumes={"/weights": vol})
def run():
    import subprocess
    open("/tmp/g.py", "w").write(SRC)
    r = subprocess.run(["python", "/tmp/g.py"], capture_output=True, text=True)
    print(r.stdout[-6000:]); print(r.stderr[-1500:])

@app.local_entrypoint()
def main():
    run.remote()
    run.remote()
