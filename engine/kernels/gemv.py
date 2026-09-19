"""Skinny-GEMM ("GEMV") kernels for decode: y[M, N] = x[M, K] @ W[N, K]^T, M <= 128.

Decode is weight-bandwidth bound, so each program streams a [BN, K_SPLIT]
slab of W exactly once. Wide-K / narrow-N projections are split over K to
fill the GPU; their fp32 partials are summed by the consumer kernel (RoPE or
residual+RMSNorm), which rounds to bf16 exactly where the reference's
Linear output would be rounded. The gate/up projection computes both halves
in one program and applies SwiGLU in the epilogue.
"""

import os

import torch
import triton
import triton.language as tl

_DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"

if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = [triton.Config({"BN": 32, "BK": 64}, num_warps=4, num_stages=1)]
else:
    _CONFIGS = [
        triton.Config({"BN": 32, "BK": 128}, num_warps=4, num_stages=4),
        triton.Config({"BN": 64, "BK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BN": 16, "BK": 128}, num_warps=4, num_stages=4),
        triton.Config({"BN": 32, "BK": 64}, num_warps=4, num_stages=6),
    ]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "K_SPLIT"])
@triton.jit
def _gemv_kernel(x_ptr, w_ptr, out_ptr, M, N, K, K_SPLIT,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PARTIAL: tl.constexpr, DOT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mmask = offs_m < M
    acc = tl.zeros([BM, BN], tl.float32)
    k0 = pid_k * K_SPLIT
    x_base = x_ptr + offs_m[:, None] * K
    w_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    for kk in range(0, K_SPLIT, BK):
        offs_k = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(w_base + offs_k[None, :])
        if DOT_F32:
            acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
        else:
            acc += tl.dot(x, tl.trans(w))
    if PARTIAL:
        dst = out_ptr + (pid_k * M + offs_m)[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc, mask=mmask[:, None])
    else:
        dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc.to(tl.bfloat16), mask=mmask[:, None])


@triton.autotune(configs=_CONFIGS, key=["M", "I", "K"])
@triton.jit
def _gemv_swiglu_kernel(x_ptr, w_ptr, out_ptr, M, I, K,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        DOT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mmask = offs_m < M
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    x_base = x_ptr + offs_m[:, None] * K
    g_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    u_base = w_ptr + (I + offs_n[:, None]).to(tl.int64) * K
    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)
        x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
        wg = tl.load(g_base + offs_k[None, :])
        wu = tl.load(u_base + offs_k[None, :])
        if DOT_F32:
            xf = x.to(tl.float32)
            acc_g += tl.dot(xf, tl.trans(wg.to(tl.float32)))
            acc_u += tl.dot(xf, tl.trans(wu.to(tl.float32)))
        else:
            acc_g += tl.dot(x, tl.trans(wg))
            acc_u += tl.dot(x, tl.trans(wu))
    # reference: bf16(bf16(silu(bf16(gate))) * bf16(up))
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    dst = out_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :]
    tl.store(dst, (s * u).to(tl.bfloat16), mask=mmask[:, None])


def _bm(M):
    return max(16, triton.next_power_of_2(M))


def gemv(x, w, split=1):
    """x [M, K] bf16, w [N, K] bf16. split == 1: returns bf16 [M, N].
    split > 1: returns fp32 partials [split, M, N] (sum them, then round)."""
    M, K = x.shape
    N = w.shape[0]
    assert K % split == 0
    k_split = K // split
    if split == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        out = torch.empty((split, M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (N // meta["BN"], split)
    _gemv_kernel[grid](x, w, out, M, N, K, k_split, BM=_bm(M),
                       PARTIAL=split > 1, DOT_F32=_DOT_F32)
    return out


def gemv_swiglu(x, w_gu):
    """x [M, K], w_gu [2I, K] = [gate; up] -> bf16(silu(gate) * up) [M, I]."""
    M, K = x.shape
    I = w_gu.shape[0] // 2
    out = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (I // meta["BN"],)
    _gemv_swiglu_kernel[grid](x, w_gu, out, M, I, K, BM=_bm(M), DOT_F32=_DOT_F32)
    return out
