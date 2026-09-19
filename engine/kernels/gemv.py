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

from kernels import pdl
from kernels.pdl import pdl_launch, pdl_wait, prefetch_l2

_DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"

if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = [triton.Config({"BN": 32, "BK": 64}, num_warps=4, num_stages=1)]
else:
    _CONFIGS = [
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn, bk, w, st in [
            (32, 128, 4, 4), (64, 128, 4, 3), (16, 128, 4, 4),
            (128, 64, 8, 4), (16, 256, 4, 3), (32, 256, 8, 3),
        ]
    ]


def _prune(configs, named_args, **kwargs):
    ks = named_args.get("K_SPLIT", named_args.get("K"))
    n = named_args.get("N", named_args.get("I"))
    ok = [c for c in configs if ks % c.kwargs["BK"] == 0 and n % c.kwargs["BN"] == 0]
    if not ok:
        raise ValueError(f"no config divides K_SPLIT={ks}, N={n}")
    return ok


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "K_SPLIT"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=5, rep=20)
@triton.jit
def _gemv_kernel(x_ptr, w_ptr, out_ptr, M, N, K, K_SPLIT,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PARTIAL: tl.constexpr, DOT_F32: tl.constexpr, PDL: tl.constexpr = False,
                 PREFETCH: tl.constexpr = False):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    # weights are read-only: stream this program's slab into L2 while the
    # predecessor kernel is still finishing
    prefetch_l2(w_ptr + (pid_n * BN + tl.arange(0, BN)).to(tl.int64) * K + pid_k * K_SPLIT,
                K_SPLIT * 2, PREFETCH)
    z = pdl_wait(PDL)
    pdl_launch(PDL)
    offs_m = tl.arange(0, BM) + z
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


@triton.autotune(configs=_CONFIGS, key=["M", "I", "K"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=5, rep=20)
@triton.jit
def _gemv_swiglu_kernel(x_ptr, w_ptr, out_ptr, M, I, K,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        DOT_F32: tl.constexpr, PDL: tl.constexpr = False,
                        PREFETCH: tl.constexpr = False):
    pid_n = tl.program_id(0)
    rows = (pid_n * BN + tl.arange(0, BN)).to(tl.int64)
    prefetch_l2(w_ptr + rows * K, K * 2, PREFETCH)
    prefetch_l2(w_ptr + (I + rows) * K, K * 2, PREFETCH)
    z = pdl_wait(PDL)
    pdl_launch(PDL)
    offs_m = tl.arange(0, BM) + z
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
    pdl.before_launch()
    _gemv_kernel[grid](x, w, out, M, N, K, k_split, BM=_bm(M),
                       PARTIAL=split > 1, DOT_F32=_DOT_F32, PDL=pdl.compiled(),
                       PREFETCH=pdl.prefetch_on())
    pdl.after_launch()
    return out


def gemv_swiglu(x, w_gu):
    """x [M, K], w_gu [2I, K] = [gate; up] -> bf16(silu(gate) * up) [M, I]."""
    M, K = x.shape
    I = w_gu.shape[0] // 2
    out = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (I // meta["BN"],)
    pdl.before_launch()
    _gemv_swiglu_kernel[grid](x, w_gu, out, M, I, K, BM=_bm(M), DOT_F32=_DOT_F32,
                              PDL=pdl.compiled(), PREFETCH=pdl.prefetch_on())
    pdl.after_launch()
    return out


# ---------------------------------------------------------------------------
# M == 1 variants on CUDA cores: elementwise FMA into a [BN, BK] fp32 tile,
# one reduction at the end. No shared-memory staging, wide vector loads.
# ---------------------------------------------------------------------------

if os.environ.get("TRITON_INTERPRET") == "1":
    _M1_CONFIGS = [triton.Config({"BN": 16, "BK": 64}, num_warps=4, num_stages=1)]
else:
    _M1_CONFIGS = [
        triton.Config({"BN": bn, "BK": bk}, num_warps=w, num_stages=st)
        for bn, bk, w, st in [
            (8, 512, 4, 1), (16, 256, 4, 1), (16, 512, 8, 1), (32, 256, 8, 1),
            (4, 1024, 4, 1), (8, 512, 4, 3), (16, 128, 4, 1), (8, 64, 4, 1),
        ]
    ]


@triton.autotune(configs=_M1_CONFIGS, key=["N", "K", "K_SPLIT"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=5, rep=20)
@triton.jit
def _gemv_m1_kernel(x_ptr, w_ptr, out_ptr, N, K, K_SPLIT,
                    BN: tl.constexpr, BK: tl.constexpr, PARTIAL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros([BN, BK], tl.float32)
    k0 = pid_k * K_SPLIT
    w_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    for kk in range(0, K_SPLIT, BK):
        offs_k = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        w = tl.load(w_base + offs_k[None, :]).to(tl.float32)
        acc += w * x[None, :]
    res = tl.sum(acc, axis=1)
    if PARTIAL:
        tl.store(out_ptr + pid_k * N + offs_n, res)
    else:
        tl.store(out_ptr + offs_n, res.to(tl.bfloat16))


@triton.autotune(configs=_M1_CONFIGS, key=["I", "K"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=5, rep=20)
@triton.jit
def _gemv_m1_swiglu_kernel(x_ptr, w_ptr, out_ptr, I, K,
                           BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc_g = tl.zeros([BN, BK], tl.float32)
    acc_u = tl.zeros([BN, BK], tl.float32)
    g_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    u_base = w_ptr + (I + offs_n[:, None]).to(tl.int64) * K
    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k).to(tl.float32)[None, :]
        acc_g += tl.load(g_base + offs_k[None, :]).to(tl.float32) * x
        acc_u += tl.load(u_base + offs_k[None, :]).to(tl.float32) * x
    g = tl.sum(acc_g, axis=1).to(tl.bfloat16).to(tl.float32)
    u = tl.sum(acc_u, axis=1).to(tl.bfloat16).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs_n, (s * u).to(tl.bfloat16))


def gemv_m1(x, w, split=1):
    """Same contract as gemv() for a single row."""
    M, K = x.shape
    assert M == 1 and K % split == 0
    N = w.shape[0]
    if split == 1:
        out = torch.empty((1, N), device=x.device, dtype=torch.bfloat16)
    else:
        out = torch.empty((split, 1, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (N // meta["BN"], split)
    _gemv_m1_kernel[grid](x, w, out, N, K, K // split, PARTIAL=split > 1)
    return out


def gemv_m1_swiglu(x, w_gu):
    M, K = x.shape
    assert M == 1
    I = w_gu.shape[0] // 2
    out = torch.empty((1, I), device=x.device, dtype=torch.bfloat16)
    grid = lambda meta: (I // meta["BN"],)
    _gemv_m1_swiglu_kernel[grid](x, w_gu, out, I, K)
    return out


# ---------------------------------------------------------------------------
# Persistent row-block GEMV: exactly G programs (a multiple of the SM count),
# program p owns output rows [p*N//G, (p+1)*N//G) over the full K. Every SM
# gets the same number of bytes, so no SM idles in a tail wave.
# ---------------------------------------------------------------------------


@triton.jit
def _gemv_rows_kernel(x_ptr, w_ptr, out_ptr, M, N, K, G,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      SWIGLU: tl.constexpr, M1: tl.constexpr, DOT_F32: tl.constexpr):
    pid = tl.program_id(0)
    r0 = (pid * N) // G
    r1 = ((pid + 1) * N) // G
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    for n0 in range(r0, r1, BN):
        offs_n = n0 + tl.arange(0, BN)
        nmask = offs_n < r1
        w_rows = w_ptr + offs_n[:, None].to(tl.int64) * K
        if SWIGLU:
            u_rows = w_ptr + (N + offs_n[:, None]).to(tl.int64) * K
        if M1:
            acc = tl.zeros([BN, BK], tl.float32)
            acc_u = tl.zeros([BN, BK], tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                x = tl.load(x_ptr + offs_k).to(tl.float32)[None, :]
                acc += tl.load(w_rows + offs_k[None, :], mask=nmask[:, None], other=0.0).to(tl.float32) * x
                if SWIGLU:
                    acc_u += tl.load(u_rows + offs_k[None, :], mask=nmask[:, None], other=0.0).to(tl.float32) * x
            res = tl.sum(acc, axis=1)[None, :]
            res_u = tl.sum(acc_u, axis=1)[None, :]
        else:
            res = tl.zeros([BM, BN], tl.float32)
            res_u = tl.zeros([BM, BN], tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mmask[:, None], other=0.0)
                w = tl.load(w_rows + offs_k[None, :], mask=nmask[:, None], other=0.0)
                if DOT_F32:
                    res += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
                else:
                    res += tl.dot(x, tl.trans(w))
                if SWIGLU:
                    wu = tl.load(u_rows + offs_k[None, :], mask=nmask[:, None], other=0.0)
                    if DOT_F32:
                        res_u += tl.dot(x.to(tl.float32), tl.trans(wu.to(tl.float32)))
                    else:
                        res_u += tl.dot(x, tl.trans(wu))
        omask = mmask[:, None] & nmask[None, :]
        dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        if SWIGLU:
            g = res.to(tl.bfloat16).to(tl.float32)
            u = res_u.to(tl.bfloat16).to(tl.float32)
            s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
            tl.store(dst, (s * u).to(tl.bfloat16), mask=omask)
        else:
            tl.store(dst, res.to(tl.bfloat16), mask=omask)


def gemv_rows(x, w, programs, swiglu=False, bn=16, bk=256, num_warps=4):
    """Persistent GEMV over `programs` programs; bf16 [M, N] out
    (N = I and SwiGLU applied if swiglu, with w = [gate; up])."""
    M, K = x.shape
    N = w.shape[0] // 2 if swiglu else w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    m1 = M == 1
    _gemv_rows_kernel[(programs,)](
        x, w, out, M, N, K, programs, BM=max(16, triton.next_power_of_2(M)),
        BN=bn, BK=bk if m1 else 128, SWIGLU=swiglu, M1=m1, DOT_F32=_DOT_F32,
        num_warps=num_warps, num_stages=3,
    )
    return out


# ---------------------------------------------------------------------------
# TMA variant (Hopper): weight tiles arrive through the Tensor Memory
# Accelerator via Triton 3.1's experimental descriptor loads. One 128-byte
# descriptor per weight, built on the host once and cached.
# ---------------------------------------------------------------------------

TMA_BN, TMA_BK = 64, 128
_tma_descs = {}


def _tma_desc(w):
    key = (w.data_ptr(), tuple(w.shape))
    d = _tma_descs.get(key)
    if d is None:
        import numpy as np
        buf = np.empty(128, dtype=np.int8)
        triton.runtime.driver.active.utils.fill_2d_tma_descriptor(
            w.data_ptr(), w.shape[0], w.shape[1], TMA_BN, TMA_BK, w.element_size(), buf)
        d = _tma_descs[key] = torch.tensor(buf, device=w.device)
    return d


@triton.jit
def _gemv_tma_kernel(x_ptr, w_desc, out_ptr, M, N, K, K_SPLIT,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     PARTIAL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BN
    k0 = pid_k * K_SPLIT
    offs_m = tl.arange(0, BM)
    offs_n = n0 + tl.arange(0, BN)
    mmask = offs_m < M
    acc = tl.zeros([BM, BN], tl.float32)
    for kk in range(0, K_SPLIT, BK):
        offs_k = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mmask[:, None], other=0.0)
        w = tl._experimental_descriptor_load(w_desc, [n0, k0 + kk], [BN, BK], tl.bfloat16)
        acc += tl.dot(x, tl.trans(w))
    if PARTIAL:
        dst = out_ptr + (pid_k * M + offs_m)[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc, mask=mmask[:, None])
    else:
        dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc.to(tl.bfloat16), mask=mmask[:, None])


def gemv_tma(x, w, split=1):
    """Same contract as gemv(); requires N % 64 == 0 and (K / split) % 128 == 0."""
    M, K = x.shape
    N = w.shape[0]
    if N % TMA_BN or K % split or (K // split) % TMA_BK:
        raise ValueError("shape not TMA-tileable")
    if split == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        out = torch.empty((split, M, N), device=x.device, dtype=torch.float32)
    _gemv_tma_kernel[(N // TMA_BN, split)](
        x, _tma_desc(w), out, M, N, K, K // split, BM=max(16, triton.next_power_of_2(M)),
        BN=TMA_BN, BK=TMA_BK, PARTIAL=split > 1, num_warps=4, num_stages=4)
    return out
