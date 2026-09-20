"""Speed ceiling for an int8 KV cache: decode attention inner loop, bf16 cache vs
int8-per-token cache (dequant in-kernel). One new query per (batch, kv-head, group),
reading a cache of length L. Mirrors the fused_attn read loop's arithmetic.
"""
import modal

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124"))
app = modal.App("kernel-rush-kvspeed", image=image)

SRC = r'''
import sys, torch, triton, triton.language as tl

NKV, GROUP, D = 8, 4, 128     # Qwen3-4B
NLAYER = 36

@triton.jit
def attn_bf16(kc, vc, q, out, L, stride_b, stride_h, scale,
              BN: tl.constexpr, G: tl.constexpr, DD: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * stride_h
    offs_d = tl.arange(0, DD)
    offs_g = tl.arange(0, 16); gm = offs_g < G
    qv = tl.load(q + pid.to(tl.int64) * G * DD + offs_g[:, None] * DD + offs_d[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
    m_i = tl.full([16], -1e30, tl.float32); l_i = tl.zeros([16], tl.float32); acc = tl.zeros([16, DD], tl.float32)
    for n0 in range(0, L, BN):
        offs_n = n0 + tl.arange(0, BN); nmask = offs_n < L
        row = base + offs_n[:, None].to(tl.int64) * DD
        k = tl.load(kc + row + offs_d[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        s = tl.dot(qv, tl.trans(k)) * scale
        s = tl.where(nmask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None]); l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vc + row + offs_d[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    tl.store(out + pid.to(tl.int64) * G * DD + offs_g[:, None] * DD + offs_d[None, :], (acc / l_i[:, None]).to(tl.bfloat16), mask=gm[:, None])

@triton.jit
def attn_int8(kc, ks, vc, vs, q, out, L, stride_b, stride_h, ss_b, ss_h, scale,
              BN: tl.constexpr, G: tl.constexpr, DD: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * stride_h; sbase = pid.to(tl.int64) * ss_h
    offs_d = tl.arange(0, DD); offs_g = tl.arange(0, 16); gm = offs_g < G
    qv = tl.load(q + pid.to(tl.int64) * G * DD + offs_g[:, None] * DD + offs_d[None, :], mask=gm[:, None], other=0.0).to(tl.float32)
    m_i = tl.full([16], -1e30, tl.float32); l_i = tl.zeros([16], tl.float32); acc = tl.zeros([16, DD], tl.float32)
    for n0 in range(0, L, BN):
        offs_n = n0 + tl.arange(0, BN); nmask = offs_n < L
        row = base + offs_n[:, None].to(tl.int64) * DD
        ksc = tl.load(ks + sbase + offs_n, mask=nmask, other=0.0)
        k = tl.load(kc + row + offs_d[None, :], mask=nmask[:, None], other=0).to(tl.float32) * ksc[:, None]
        s = tl.dot(qv, tl.trans(k)) * scale
        s = tl.where(nmask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None]); l_i = l_i * alpha + tl.sum(p, 1)
        vsc = tl.load(vs + sbase + offs_n, mask=nmask, other=0.0)
        v = tl.load(vc + row + offs_d[None, :], mask=nmask[:, None], other=0) .to(tl.float32) * vsc[:, None]
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    tl.store(out + pid.to(tl.int64) * G * DD + offs_g[:, None] * DD + offs_d[None, :], (acc / l_i[:, None]).to(tl.bfloat16), mask=gm[:, None])

@triton.jit
def attn_fp8(kc, vc, q, out, L, stride_h, scale, vscale,
             BN: tl.constexpr, G: tl.constexpr, DD: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * stride_h
    offs_d = tl.arange(0, DD); offs_g = tl.arange(0, 16); gm = offs_g < G
    qv = tl.load(q + pid.to(tl.int64) * G * DD + offs_g[:, None] * DD + offs_d[None, :], mask=gm[:, None], other=0.0)
    m_i = tl.full([16], -1e30, tl.float32); l_i = tl.zeros([16], tl.float32); acc = tl.zeros([16, DD], tl.float32)
    for n0 in range(0, L, BN):
        offs_n = n0 + tl.arange(0, BN); nmask = offs_n < L
        row = base + offs_n[:, None].to(tl.int64) * DD
        k = tl.load(kc + row + offs_d[None, :], mask=nmask[:, None], other=0.0).to(tl.bfloat16)
        s = tl.dot(qv, tl.trans(k)) * scale
        s = tl.where(nmask[None, :], s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1)); alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None]); l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vc + row + offs_d[None, :], mask=nmask[:, None], other=0.0).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    tl.store(out + pid.to(tl.int64) * G * DD + offs_g[:, None] * DD + offs_d[None, :], (acc * vscale / l_i[:, None]).to(tl.bfloat16), mask=gm[:, None])

def tg(f, reps=50):
    f(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): f()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps): f()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(5):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); e1.synchronize(); best = min(best, e0.elapsed_time(e1) / reps)
    return best * 1e3

for B, L in [(1, 512), (4, 2048), (16, 512), (16, 640), (2, 4096), (8, 1024), (32, 384)]:
    P = B * NKV                          # programs (one per batch, kv-head), all layers looped outside
    kc = torch.randn(P, L, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(P, L, D, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(P, GROUP, D, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(P, GROUP, D, device="cuda", dtype=torch.bfloat16)
    ks = (kc.abs().amax(-1) / 127).clamp_min(1e-6); ki = torch.round(kc.float() / ks[..., None]).clamp(-127, 127).to(torch.int8)
    vs = (vc.abs().amax(-1) / 127).clamp_min(1e-6); vi = torch.round(vc.float() / vs[..., None]).clamp(-127, 127).to(torch.int8)
    sc = D ** -0.5
    kmax = float(kc.abs().amax())/448; vmax = float(vc.abs().amax())/448
    kf = (kc.float()/kmax).clamp(-448,448).to(torch.float8_e4m3fn); vf = (vc.float()/vmax).clamp(-448,448).to(torch.float8_e4m3fn)
    skq = sc*kmax
    def rb(): attn_bf16[(P,)](kc, vc, q, out, L, kc.stride(0), kc.stride(0), sc, BN=64, G=GROUP, DD=D, num_warps=4, num_stages=3)
    def rf(): attn_fp8[(P,)](kf, vf, q, out, L, kf.stride(0), skq, vmax, BN=64, G=GROUP, DD=D, num_warps=4, num_stages=3)
    def ri(): attn_int8[(P,)](ki, ks, vi, vs, q, out, L, ki.stride(0), ki.stride(0), ks.stride(0), ks.stride(0), sc, BN=64, G=GROUP, DD=D, num_warps=4, num_stages=3)
    # correctness of the int8 path vs bf16 read
    rb(); ob = out.clone(); ri(); oi = out.clone(); rf(); of = out.clone()
    ei = (ob.float()-oi.float()).abs().max().item(); ef = (ob.float()-of.float()).abs().max().item()
    tb, ti, tf = tg(rb), tg(ri), tg(rf)
    print(f"RESULT B={B:2d} L={L:4d}: bf16 {tb:7.1f}us | int8 {ti:7.1f}us ({tb/ti:.2f}x err{ei:.3f}) | fp8pt {tf:7.1f}us ({tb/tf:.2f}x err{ef:.3f})", flush=True)
'''

@app.function(gpu="H100", timeout=1200)
def run():
    import subprocess
    open("/tmp/k.py", "w").write(SRC)
    r = subprocess.run(["python", "/tmp/k.py"], capture_output=True, text=True)
    print(r.stdout[-6000:]); print(r.stderr[-2000:])

@app.local_entrypoint()
def main():
    run.remote()
