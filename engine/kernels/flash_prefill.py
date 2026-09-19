"""Causal GQA flash attention for prefill, reading K/V straight from the cache.

q: [G*S, NQ*D] contiguous bf16 (rows ordered sequence-major), K/V cache
[B, NKV, CAP, D]; sequences b0..b0+G-1, positions 0..S-1. Output
[G*S, NQ*D] contiguous, laid out exactly as o_proj wants it - no repeat_kv,
no transposes, no copies. Scores and the softmax accumulate in fp32.
"""

import os

import torch
import triton
import triton.language as tl

_INTERP = os.environ.get("TRITON_INTERPRET") == "1"

if _INTERP:
    _CFGS = [triton.Config({"BQ": 16, "BKV": 16}, num_warps=4, num_stages=1)]
else:
    _CFGS = [
        triton.Config({"BQ": bq, "BKV": bk}, num_warps=w, num_stages=st)
        for bq, bk, w, st in [(64, 64, 4, 3), (128, 128, 8, 3), (64, 128, 4, 3), (128, 64, 8, 2),
                              (32, 128, 4, 3), (64, 64, 8, 2)]
    ]


@triton.autotune(configs=_CFGS, key=["S", "NQ", "NKV"], warmup=5, rep=20)
@triton.jit
def _flash_prefill_kernel(q_ptr, kc_ptr, vc_ptr, o_ptr, S, b0,
                          stride_cb, stride_ch, scale,
                          NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
                          BQ: tl.constexpr, BKV: tl.constexpr, DOT_F32: tl.constexpr):
    qb = tl.program_id(0)            # query block within the sequence
    head = tl.program_id(1)
    g = tl.program_id(2)             # sequence within this chunk
    kvh = head // (NQ // NKV)
    offs_q = qb * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)
    qmask = offs_q < S
    row = (g.to(tl.int64) * S + offs_q) * (NQ * D) + head * D
    q = tl.load(q_ptr + row[:, None] + offs_d[None, :], mask=qmask[:, None], other=0.0)
    base = (b0 + g).to(tl.int64) * stride_cb + kvh * stride_ch

    m_i = tl.full([BQ], float("-inf"), tl.float32)
    l_i = tl.zeros([BQ], tl.float32)
    acc = tl.zeros([BQ, D], tl.float32)
    hi = tl.minimum((qb + 1) * BQ, S)            # causal: keys < hi
    for k0 in range(0, hi, BKV):
        offs_k = k0 + tl.arange(0, BKV)
        kv_off = base + offs_k[:, None].to(tl.int64) * D + offs_d[None, :]
        kmask = offs_k < hi
        k = tl.load(kc_ptr + kv_off, mask=kmask[:, None], other=0.0)
        if DOT_F32:
            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale
        else:
            s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        valid = (offs_k[None, :] <= offs_q[:, None]) & kmask[None, :]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + kv_off, mask=kmask[:, None], other=0.0)
        if DOT_F32:
            pv = tl.dot(p, v.to(tl.float32))
        else:
            pv = tl.dot(p.to(tl.bfloat16), v).to(tl.float32)
        acc = acc * alpha[:, None] + pv
        m_i = m_new
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = (acc / l_safe[:, None]).to(tl.bfloat16)
    tl.store(o_ptr + row[:, None] + offs_d[None, :], out, mask=qmask[:, None])


def flash_prefill(q, k_cache, v_cache, S, b0, G, nq, nkv, d):
    """q [G*S, NQ*D] -> attention output [G*S, NQ*D]."""
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(S, meta["BQ"]), nq, G)
    _flash_prefill_kernel[grid](q, k_cache, v_cache, out, S, b0,
                                k_cache.stride(0), k_cache.stride(1), d ** -0.5,
                                NQ=nq, NKV=nkv, D=d, DOT_F32=_INTERP)
    return out
