"""Decode GEMVs with the residual add and RMSNorm folded in, so a decoder
layer is 4 matmul launches + 1 attention launch instead of 7.

Producer (EP): y = x @ W^T for a projection whose output is added to the
residual stream (o_proj, down_proj). Split-K partials are written as before;
the LAST program to finish an N-tile (per-tile atomic counter) sums them in
fixed order, rounds to bf16 (the reference's Linear output), adds the
residual in fp32 and rounds (the reference's bf16 residual add), writes the
new residual row tile to res_out and atomically accumulates its squares
into ss_out[row]. Reordering only; every rounding point is the reference's.

Consumer (NORM): the next projection (qkv, gate/up, lm head) reads the new
residual and ss, computes rstd = rsqrt(ss/K + eps) per row and normalises
each K-tile on the fly: h = bf16(bf16(x * rstd) * w_ln) - exactly what
Qwen3RMSNorm produces - before the dot. Program (0, 0) of a consumer also
zeroes the *other* ss buffer, which the next producer accumulates into.
"""

import os

import torch
import triton
import triton.language as tl

from kernels import pdl
from kernels.gemv import _CONFIGS, _DOT_F32, _bm, _prune
from kernels.pdl import pdl_launch, pdl_wait


@triton.jit
def _fence_all(FENCE: tl.constexpr):
    if FENCE:
        dummy = tl.arange(0, 128)
        tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0, $1;", "=r,r", [dummy],
                                  dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _norm_tile(x, rstd, ln_ptr, offs_k, NORM: tl.constexpr):
    if NORM:
        w_ln = tl.load(ln_ptr + offs_k).to(tl.float32)
        return ((x.to(tl.float32) * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
                * w_ln[None, :]).to(tl.bfloat16)
    else:
        return x


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "K_SPLIT"],
                 prune_configs_by={"early_config_prune": _prune}, warmup=5, rep=20)
@triton.jit
def _gemv_fused_kernel(x_ptr, w_ptr, out_ptr, M, N, K, K_SPLIT,
                       ln_ptr, ss_in_ptr, zero_ss_ptr,
                       res_in_ptr, res_out_ptr, ss_out_ptr, cnt_ptr, eps,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                       SPLIT: tl.constexpr, NORM: tl.constexpr, EP: tl.constexpr,
                       ZERO_SS: tl.constexpr, DOT_F32: tl.constexpr, PDL: tl.constexpr,
                       FENCE: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    z = pdl_wait(PDL)
    pdl_launch(PDL)
    offs_m = tl.arange(0, BM) + z
    offs_n = pid_n * BN + tl.arange(0, BN)
    mmask = offs_m < M
    if ZERO_SS:
        if pid_n == 0 and pid_k == 0:
            tl.store(zero_ss_ptr + offs_m, tl.zeros([BM], tl.float32), mask=mmask)
    if NORM:
        ss = tl.load(ss_in_ptr + offs_m, mask=mmask, other=1.0)
        rstd = tl.math.rsqrt(ss / K + eps)
    else:
        rstd = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, BN], tl.float32)
    k0 = pid_k * K_SPLIT
    x_base = x_ptr + offs_m[:, None] * K
    w_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    for kk in range(0, K_SPLIT, BK):
        offs_k = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
        x = _norm_tile(x, rstd, ln_ptr, offs_k, NORM)
        w = tl.load(w_base + offs_k[None, :])
        if DOT_F32:
            acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
        else:
            acc += tl.dot(x, tl.trans(w))
    if EP:
        tile = offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        if SPLIT > 1:
            tl.store(out_ptr + (pid_k * M) * N + tile, acc, mask=mmask[:, None])
            _fence_all(FENCE)
            tl.debug_barrier()
            done = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel")
            tl.debug_barrier()
            if done == SPLIT - 1:
                _fence_all(FENCE)
                y = tl.zeros([BM, BN], tl.float32)
                for sp in tl.static_range(SPLIT):
                    y += tl.load(out_ptr + (sp * M) * N + tile, mask=mmask[:, None], other=0.0,
                                 cache_modifier=".cg")
                ybf = y.to(tl.bfloat16).to(tl.float32)
                r = tl.load(res_in_ptr + tile, mask=mmask[:, None], other=0.0).to(tl.float32)
                xn = (r + ybf).to(tl.bfloat16)
                tl.store(res_out_ptr + tile, xn, mask=mmask[:, None])
                xf = xn.to(tl.float32)
                tl.atomic_add(ss_out_ptr + offs_m, tl.sum(xf * xf, axis=1), mask=mmask)
                tl.atomic_xchg(cnt_ptr + pid_n, 0)
        else:
            ybf = acc.to(tl.bfloat16).to(tl.float32)
            r = tl.load(res_in_ptr + tile, mask=mmask[:, None], other=0.0).to(tl.float32)
            xn = (r + ybf).to(tl.bfloat16)
            tl.store(res_out_ptr + tile, xn, mask=mmask[:, None])
            xf = xn.to(tl.float32)
            tl.atomic_add(ss_out_ptr + offs_m, tl.sum(xf * xf, axis=1), mask=mmask)
    else:
        if SPLIT > 1:
            dst = out_ptr + (pid_k * M + offs_m)[:, None].to(tl.int64) * N + offs_n[None, :]
            tl.store(dst, acc, mask=mmask[:, None])
        else:
            dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
            tl.store(dst, acc.to(tl.bfloat16), mask=mmask[:, None])


@triton.autotune(configs=_CONFIGS, key=["M", "I", "K"],
                 prune_configs_by={"early_config_prune": _prune}, warmup=5, rep=20)
@triton.jit
def _gemv_swiglu_fused_kernel(x_ptr, w_ptr, out_ptr, M, I, K,
                              ln_ptr, ss_in_ptr, zero_ss_ptr, eps,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                              NORM: tl.constexpr, ZERO_SS: tl.constexpr,
                              DOT_F32: tl.constexpr, PDL: tl.constexpr):
    pid_n = tl.program_id(0)
    z = pdl_wait(PDL)
    pdl_launch(PDL)
    offs_m = tl.arange(0, BM) + z
    offs_n = pid_n * BN + tl.arange(0, BN)
    mmask = offs_m < M
    if ZERO_SS:
        if pid_n == 0:
            tl.store(zero_ss_ptr + offs_m, tl.zeros([BM], tl.float32), mask=mmask)
    if NORM:
        ss = tl.load(ss_in_ptr + offs_m, mask=mmask, other=1.0)
        rstd = tl.math.rsqrt(ss / K + eps)
    else:
        rstd = tl.zeros([BM], tl.float32)
    acc_g = tl.zeros([BM, BN], tl.float32)
    acc_u = tl.zeros([BM, BN], tl.float32)
    x_base = x_ptr + offs_m[:, None] * K
    g_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    u_base = w_ptr + (I + offs_n[:, None]).to(tl.int64) * K
    for kk in range(0, K, BK):
        offs_k = kk + tl.arange(0, BK)
        x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
        x = _norm_tile(x, rstd, ln_ptr, offs_k, NORM)
        wg = tl.load(g_base + offs_k[None, :])
        wu = tl.load(u_base + offs_k[None, :])
        if DOT_F32:
            xf = x.to(tl.float32)
            acc_g += tl.dot(xf, tl.trans(wg.to(tl.float32)))
            acc_u += tl.dot(xf, tl.trans(wu.to(tl.float32)))
        else:
            acc_g += tl.dot(x, tl.trans(wg))
            acc_u += tl.dot(x, tl.trans(wu))
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    dst = out_ptr + offs_m[:, None].to(tl.int64) * I + offs_n[None, :]
    tl.store(dst, (s * u).to(tl.bfloat16), mask=mmask[:, None])


class FusedLayerBuffers:
    """Residual ping-pong rows, sum-of-squares accumulators and split-K
    counters for one M."""

    def __init__(self, M, H, n_max, device):
        self.res = torch.zeros((2, M, H), device=device, dtype=torch.bfloat16)
        self.ss = torch.zeros((2, M), device=device, dtype=torch.float32)
        self.cnt = torch.zeros((n_max // 8,), device=device, dtype=torch.int32)


def gemv_fused(x, w, split, *, norm=None, zero_ss=None, ep=None, eps=1e-6):
    """x [M, K] bf16, w [N, K] bf16.
    norm=(ln_w, ss_in): normalise x rows on the fly.
    zero_ss=ss: program (0,0) zeroes it.
    ep=(res_in, res_out, ss_out, cnt): residual epilogue; returns None.
    Otherwise returns bf16 [M, N] (split 1) or fp32 partials [split, M, N]."""
    M, K = x.shape
    N = w.shape[0]
    assert K % split == 0
    dev = x.device
    dummy_f = x if norm is None else norm[0]
    if ep is not None or split > 1:
        out = torch.empty((split, M, N), device=dev, dtype=torch.float32)
    else:
        out = torch.empty((M, N), device=dev, dtype=torch.bfloat16)
    ln_ptr, ss_in = (norm if norm is not None else (dummy_f, x))
    res_in, res_out, ss_out, cnt = (ep if ep is not None else (x, x, x, x))
    grid = lambda meta: (N // meta["BN"], split)
    pdl.before_launch()
    _gemv_fused_kernel[grid](
        x, w, out, M, N, K, K // split,
        ln_ptr, ss_in, zero_ss if zero_ss is not None else x,
        res_in, res_out, ss_out, cnt, eps,
        BM=_bm(M), SPLIT=split, NORM=norm is not None, EP=ep is not None,
        ZERO_SS=zero_ss is not None, DOT_F32=_DOT_F32, PDL=pdl.compiled(), FENCE=not _DOT_F32)
    pdl.after_launch()
    return None if ep is not None else out


def gemv_swiglu_fused(x, w_gu, *, norm=None, zero_ss=None, eps=1e-6):
    M, K = x.shape
    I = w_gu.shape[0] // 2
    out = torch.empty((M, I), device=x.device, dtype=torch.bfloat16)
    ln_ptr, ss_in = (norm if norm is not None else (x, x))
    grid = lambda meta: (I // meta["BN"],)
    pdl.before_launch()
    _gemv_swiglu_fused_kernel[grid](
        x, w_gu, out, M, I, K, ln_ptr, ss_in, zero_ss if zero_ss is not None else x, eps,
        BM=_bm(M), NORM=norm is not None, ZERO_SS=zero_ss is not None,
        DOT_F32=_DOT_F32, PDL=pdl.compiled())
    pdl.after_launch()
    return out
