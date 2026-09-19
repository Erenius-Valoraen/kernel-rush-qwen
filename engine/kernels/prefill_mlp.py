"""Prefill gate/up projection with the SwiGLU epilogue fused in.

act[M, I] = bf16(bf16(silu(bf16(x @ Wg^T))) * bf16(x @ Wu^T)), W = [Wg; Wu].
Each program owns a BM x BN tile of act and accumulates the matching gate and
up tiles side by side, so the [M, 2I] intermediate never touches HBM.
"""

import os

import torch
import triton
import triton.language as tl

if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = [triton.Config({"BM": 32, "BN": 32, "BK": 32, "GM": 4}, num_warps=4, num_stages=1)]
else:
    _CONFIGS = [
        triton.Config({"BM": 128, "BN": 64, "BK": 64, "GM": 8}, num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GM": 8}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 128, "BK": 64, "GM": 8}, num_warps=4, num_stages=4),
    ]


@triton.autotune(configs=_CONFIGS, key=["M", "I", "K"], warmup=5, rep=20)
@triton.jit
def _gu_swiglu_kernel(x_ptr, w_ptr, out_ptr, M, I, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
                      DOT_F32: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = I // BN
    group = GM * num_n
    gid = pid // group
    first_m = gid * GM
    gsize = tl.minimum(num_m - first_m, GM)
    pm = first_m + (pid % group) % gsize
    pn = (pid % group) // gsize
    offs_m = pm * BM + tl.arange(0, BM)
    offs_n = pn * BN + tl.arange(0, BN)
    mmask = offs_m < M
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    x_rows = x_ptr + offs_m[:, None].to(tl.int64) * K
    g_rows = w_ptr + offs_n[:, None].to(tl.int64) * K
    u_rows = w_ptr + (I + offs_n[:, None]).to(tl.int64) * K
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_rows + offs_k[None, :], mask=mmask[:, None], other=0.0)
        wg = tl.load(g_rows + offs_k[None, :])
        wu = tl.load(u_rows + offs_k[None, :])
        if DOT_F32:
            acc_g += tl.dot(x.to(tl.float32), tl.trans(wg.to(tl.float32)))
            acc_u += tl.dot(x.to(tl.float32), tl.trans(wu.to(tl.float32)))
        else:
            acc_g += tl.dot(x, tl.trans(wg))
            acc_u += tl.dot(x, tl.trans(wu))
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :],
             (s * u).to(tl.bfloat16), mask=mmask[:, None])


def gu_swiglu(x, w_gu):
    """x [M, K] bf16 contiguous, w_gu [2I, K] -> act [M, I] bf16."""
    M, K = x.shape
    I = w_gu.shape[0] // 2
    out = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * (I // meta["BN"]),)
    _gu_swiglu_kernel[grid](x, w_gu, out, M, I, K,
                            DOT_F32=os.environ.get("TRITON_INTERPRET") == "1")
    return out
