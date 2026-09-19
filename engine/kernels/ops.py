"""Fused Triton kernels for the Qwen3 decode/prefill path.

Every kernel reproduces the reference's BF16 rounding points exactly (see
``rmsnorm.py`` for why that matters); only reduction order differs.
"""

import os

import torch
import triton
import triton.language as tl

# Triton's CPU interpreter (used only for local testing) mis-computes bf16
# tl.dot; upcast there. On GPU the bf16 MMA path is used.
_DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"


# ---------------------------------------------------------------------------
# RMSNorm, optionally fused with the preceding residual add.
#
# Reference:  x = residual + delta            (bf16 + bf16 -> bf16)
#             y = w * bf16(x_f32 * rsqrt(mean(x_f32^2) + eps))
# ---------------------------------------------------------------------------


@triton.jit
def _add_rmsnorm_kernel(x_ptr, d_ptr, w_ptr, y_ptr, n_cols, eps,
                        HAS_DELTA: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offs = row * n_cols + cols
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    if HAS_DELTA:
        d = tl.load(d_ptr + offs, mask=mask, other=0.0)
        x = (x.to(tl.float32) + d.to(tl.float32)).to(tl.bfloat16)
        tl.store(x_ptr + offs, x, mask=mask)
    xf = x.to(tl.float32)
    var = tl.sum(xf * xf, axis=0) / n_cols
    normed = (xf * tl.math.rsqrt(var + eps)).to(tl.bfloat16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(y_ptr + offs, y, mask=mask)


def add_rmsnorm(x, delta, weight, eps):
    """If delta is given: x += delta (in place, bf16). Returns rmsnorm(x)*w."""
    M, N = x.shape
    y = torch.empty_like(x)
    _add_rmsnorm_kernel[(M,)](
        x, delta if delta is not None else x, weight, y, N, eps,
        HAS_DELTA=delta is not None, BLOCK=triton.next_power_of_2(N), num_warps=8,
    )
    return y


# ---------------------------------------------------------------------------
# Per-head Q/K RMSNorm + RoPE, K/V written straight into the cache.
#
# qkv: [M, (NQ + 2*NKV) * D] rows; row r belongs to sequence b0 + r // S at
# position pos0 + r % S (pos0 read from a device pointer so a CUDA graph can
# replay it at a new position).
# ---------------------------------------------------------------------------


@triton.jit
def _qk_norm_rope_kernel(qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, q_out_ptr,
                         kc_ptr, vc_ptr, pos_ptr, S, b0,
                         stride_cb, stride_ch, eps,
                         NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    b = b0 + row // S
    pos = tl.load(pos_ptr) + row % S
    row64 = row.to(tl.int64)
    src = qkv_ptr + row64 * ((NQ + 2 * NKV) * D) + head * D
    offs = tl.arange(0, HALF)

    if head < NQ + NKV:
        x1 = tl.load(src + offs).to(tl.float32)
        x2 = tl.load(src + HALF + offs).to(tl.float32)
        var = (tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D
        rstd = tl.math.rsqrt(var + eps)
        if head < NQ:
            w_ptr = qw_ptr
        else:
            w_ptr = kw_ptr
        w1 = tl.load(w_ptr + offs).to(tl.float32)
        w2 = tl.load(w_ptr + HALF + offs).to(tl.float32)
        n1 = ((x1 * rstd).to(tl.bfloat16).to(tl.float32) * w1).to(tl.bfloat16).to(tl.float32)
        n2 = ((x2 * rstd).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)
        c1 = tl.load(cos_ptr + pos * D + offs).to(tl.float32)
        c2 = tl.load(cos_ptr + pos * D + HALF + offs).to(tl.float32)
        s1 = tl.load(sin_ptr + pos * D + offs).to(tl.float32)
        s2 = tl.load(sin_ptr + pos * D + HALF + offs).to(tl.float32)
        # q*cos + rotate_half(q)*sin, each product rounded to bf16 first.
        a1 = (n1 * c1).to(tl.bfloat16).to(tl.float32)
        r1 = (-n2 * s1).to(tl.bfloat16).to(tl.float32)
        o1 = (a1 + r1).to(tl.bfloat16)
        a2 = (n2 * c2).to(tl.bfloat16).to(tl.float32)
        r2 = (n1 * s2).to(tl.bfloat16).to(tl.float32)
        o2 = (a2 + r2).to(tl.bfloat16)
        if head < NQ:
            dst = q_out_ptr + row64 * (NQ * D) + head * D
        else:
            dst = kc_ptr + b.to(tl.int64) * stride_cb + (head - NQ) * stride_ch + pos.to(tl.int64) * D
        tl.store(dst + offs, o1)
        tl.store(dst + HALF + offs, o2)
    else:
        v1 = tl.load(src + offs)
        v2 = tl.load(src + HALF + offs)
        dst = vc_ptr + b.to(tl.int64) * stride_cb + (head - NQ - NKV) * stride_ch + pos.to(tl.int64) * D
        tl.store(dst + offs, v1)
        tl.store(dst + HALF + offs, v2)


def qk_norm_rope_cache(qkv, q_w, k_w, cos, sin, k_cache, v_cache, pos_t, S, b0,
                       eps, nq, nkv, d):
    """k_cache/v_cache: [B, NKV, CAP, D] views for one layer."""
    M = qkv.shape[0]
    q_out = torch.empty((M, nq * d), device=qkv.device, dtype=qkv.dtype)
    _qk_norm_rope_kernel[(M, nq + 2 * nkv)](
        qkv, q_w, k_w, cos, sin, q_out, k_cache, v_cache, pos_t, S, b0,
        k_cache.stride(0), k_cache.stride(1), eps,
        NQ=nq, NKV=nkv, D=d, num_warps=1,
    )
    return q_out


# ---------------------------------------------------------------------------
# SwiGLU: gu = [gate | up] -> bf16(bf16(silu(gate)) * up)
# ---------------------------------------------------------------------------


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, I, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < I
    g = tl.load(gu_ptr + row * 2 * I + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * I + I + cols, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + row * I + cols, (s * u).to(tl.bfloat16), mask=mask)


def silu_mul(gu):
    M, two_i = gu.shape
    I = two_i // 2
    out = torch.empty((M, I), device=gu.device, dtype=gu.dtype)
    BLOCK = 1024
    _silu_mul_kernel[(M, triton.cdiv(I, BLOCK))](gu, out, I, BLOCK=BLOCK, num_warps=4)
    return out


# ---------------------------------------------------------------------------
# Split-K grouped-query decode attention over a fixed-capacity cache, reading
# the valid length (pos + 1) from device memory so CUDA graphs can replay it.
# ---------------------------------------------------------------------------


@triton.jit
def _decode_attn_kernel(q_ptr, kc_ptr, vc_ptr, o_ptr, m_ptr, l_ptr, pos_ptr,
                        stride_cb, stride_ch, scale, CHUNK,
                        NKV: tl.constexpr, GROUP: tl.constexpr, GPAD: tl.constexpr,
                        D: tl.constexpr, BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr,
                        DOT_F32: tl.constexpr):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    b = pid // NKV
    kvh = pid % NKV
    seq_len = tl.load(pos_ptr) + 1
    start = split * CHUNK
    end = tl.minimum(start + CHUNK, seq_len)

    offs_g = tl.arange(0, GPAD)
    offs_d = tl.arange(0, D)
    gmask = offs_g < GROUP
    q = tl.load(q_ptr + b * (NKV * GROUP * D) + (kvh * GROUP + offs_g)[:, None] * D + offs_d[None, :],
                mask=gmask[:, None], other=0.0)

    m_i = tl.full([GPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([GPAD], tl.float32)
    acc = tl.zeros([GPAD, D], tl.float32)
    base = b.to(tl.int64) * stride_cb + kvh * stride_ch
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        kv_offs = base + offs_n[:, None].to(tl.int64) * D + offs_d[None, :]
        k = tl.load(kc_ptr + kv_offs, mask=nmask[:, None], other=0.0)
        if DOT_F32:
            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale
        else:
            s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(nmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + kv_offs, mask=nmask[:, None], other=0.0)
        if DOT_F32:
            pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
        else:
            pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    part = (pid * NSPLIT + split) * GROUP
    tl.store(m_ptr + part + offs_g, m_i, mask=gmask)
    tl.store(l_ptr + part + offs_g, l_i, mask=gmask)
    tl.store(o_ptr + (part + offs_g)[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None])


@triton.jit
def _decode_combine_kernel(o_ptr, m_ptr, l_ptr, out_ptr,
                           NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr,
                           NSPLIT: tl.constexpr, SPAD: tl.constexpr):
    pid = tl.program_id(0)          # b * NQ + head
    b = pid // (NKV * GROUP)
    head = pid % (NKV * GROUP)
    kvh = head // GROUP
    g = head % GROUP
    offs_s = tl.arange(0, SPAD)
    smask = offs_s < NSPLIT
    idx = ((b * NKV + kvh) * NSPLIT + offs_s) * GROUP + g
    m = tl.load(m_ptr + idx, mask=smask, other=float("-inf"))
    l = tl.load(l_ptr + idx, mask=smask, other=0.0)
    m_max = tl.max(m, axis=0)
    w = tl.where(m > float("-inf"), tl.exp(m - m_max), 0.0)
    denom = tl.sum(w * l, axis=0)
    offs_d = tl.arange(0, D)
    o = tl.load(o_ptr + idx[:, None] * D + offs_d[None, :], mask=smask[:, None], other=0.0)
    out = tl.sum(o * w[:, None], axis=0) / denom
    tl.store(out_ptr + pid * D + offs_d, out.to(tl.bfloat16))


class DecodeAttention:
    """Preallocated split-K workspace for one (batch, capacity) shape."""

    BLOCK_N = 64

    def __init__(self, batch, capacity, nq, nkv, d, device, num_sms):
        self.batch, self.nq, self.nkv, self.d = batch, nq, nkv, d
        self.group = nq // nkv
        nblocks = triton.cdiv(capacity, self.BLOCK_N)
        target = max(1, triton.cdiv(2 * num_sms, batch * nkv))
        nsplit = max(1, min(nblocks, target))
        self.chunk = triton.cdiv(nblocks, nsplit) * self.BLOCK_N
        self.nsplit = triton.cdiv(capacity, self.chunk)
        parts = batch * nkv * self.nsplit * self.group
        self.o = torch.empty((parts, d), device=device, dtype=torch.float32)
        self.m = torch.empty((parts,), device=device, dtype=torch.float32)
        self.l = torch.empty((parts,), device=device, dtype=torch.float32)
        self.scale = d ** -0.5

    def __call__(self, q, k_cache, v_cache, pos_t):
        B = self.batch
        out = torch.empty((B, self.nq * self.d), device=q.device, dtype=q.dtype)
        _decode_attn_kernel[(B * self.nkv, self.nsplit)](
            q, k_cache, v_cache, self.o, self.m, self.l, pos_t,
            k_cache.stride(0), k_cache.stride(1), self.scale, self.chunk,
            NKV=self.nkv, GROUP=self.group, GPAD=16, D=self.d,
            BLOCK_N=self.BLOCK_N, NSPLIT=self.nsplit, DOT_F32=_DOT_F32, num_warps=4, num_stages=2,
        )
        _decode_combine_kernel[(B * self.nq,)](
            self.o, self.m, self.l, out,
            NKV=self.nkv, GROUP=self.group, D=self.d, NSPLIT=self.nsplit,
            SPAD=max(2, triton.next_power_of_2(self.nsplit)), num_warps=2,
        )
        return out
