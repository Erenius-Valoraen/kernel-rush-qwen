"""Fused decode attention: Q/K per-head RMSNorm + RoPE, K/V cache write,
split-K causal GQA attention and the cross-split combine in ONE launch.

Program (b, kvh, split) handles the GROUP query heads of KV head kvh for the
T new tokens of sequence b, over keys [split*CHUNK, (split+1)*CHUNK). A new
token's K/V is written by the split whose range holds its position, which is
the only program that reads it. The last split program to finish for
(b, kvh), detected with an atomic counter, merges all partial results.
"""

import os

import torch
import triton
import triton.language as tl

if os.environ.get("TRITON_INTERPRET") == "1":
    _CONFIGS = [triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=1)]
else:
    _CONFIGS = [
        triton.Config({"BLOCK_N": bn}, num_warps=w, num_stages=st)
        for bn, w, st in [(64, 4, 2), (64, 4, 3), (32, 4, 3), (128, 4, 2), (128, 8, 3), (64, 8, 3)]
    ]

from kernels.ops import _DOT_F32, DecodeAttention, _load_rows


@triton.jit
def _fence(FENCE: tl.constexpr):
    """GPU-scope acq_rel fence by every thread (publishing partials safely)."""
    if FENCE:
        dummy = tl.arange(0, 128)
        tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0, $1;", "=r,r", [dummy],
                                  dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _norm_rope_half(x1, x2, w_ptr, cos_ptr, sin_ptr, pos, offs_h, mask2,
                    eps, D: tl.constexpr, HALF: tl.constexpr):
    """x1, x2: [R, HALF] fp32 halves of R bf16 head vectors -> roped bf16 halves.

    Same rounding points as Qwen3RMSNorm followed by apply_rotary_pos_emb."""
    var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D
    rstd = tl.math.rsqrt(var + eps)[:, None]
    w1 = tl.load(w_ptr + offs_h).to(tl.float32)[None, :]
    w2 = tl.load(w_ptr + HALF + offs_h).to(tl.float32)[None, :]
    n1 = ((x1 * rstd).to(tl.bfloat16).to(tl.float32) * w1).to(tl.bfloat16).to(tl.float32)
    n2 = ((x2 * rstd).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)
    cp = cos_ptr + pos[:, None] * D + offs_h[None, :]
    sp = sin_ptr + pos[:, None] * D + offs_h[None, :]
    c1 = tl.load(cp, mask=mask2, other=0.0).to(tl.float32)
    c2 = tl.load(cp + HALF, mask=mask2, other=0.0).to(tl.float32)
    s1 = tl.load(sp, mask=mask2, other=0.0).to(tl.float32)
    s2 = tl.load(sp + HALF, mask=mask2, other=0.0).to(tl.float32)
    o1 = ((n1 * c1).to(tl.bfloat16).to(tl.float32)
          + (-n2 * s1).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    o2 = ((n2 * c2).to(tl.bfloat16).to(tl.float32)
          + (n1 * s2).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    return o1, o2


@triton.jit
def _fused_attn_kernel(qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr,
                       kc_ptr, vc_ptr, pos_ptr, o_ptr, m_ptr, l_ptr, cnt_ptr, out_ptr,
                       stride_cb, stride_ch, split_stride, scale, eps, CHUNK,
                       NKV: tl.constexpr, GROUP: tl.constexpr, T: tl.constexpr,
                       RPAD: tl.constexpr, TPAD: tl.constexpr, D: tl.constexpr,
                       BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr, QSPLIT: tl.constexpr,
                       DOT_F32: tl.constexpr, FENCE: tl.constexpr = False):
    pid = tl.program_id(0)
    split = tl.program_id(1)
    b = pid // NKV
    kvh = pid % NKV
    HALF: tl.constexpr = D // 2
    NQ: tl.constexpr = NKV * GROUP
    ROWS: tl.constexpr = GROUP * T
    W: tl.constexpr = (NQ + 2 * NKV) * D
    p0 = tl.load(pos_ptr + b)
    start = split * CHUNK
    end = tl.minimum(start + CHUNK, p0 + T)
    offs_h = tl.arange(0, HALF)
    offs_d = tl.arange(0, D)
    cache_base = b.to(tl.int64) * stride_cb + kvh * stride_ch

    # ---- new K/V of the tokens whose positions fall in this split
    offs_t = tl.arange(0, TPAD)
    tpos = p0 + offs_t
    tmask = (offs_t < T) & (tpos >= start) & (tpos < end)
    tmask2 = tmask[:, None] & (offs_h[None, :] < HALF)
    trow = qkv_ptr + (b * T + offs_t)[:, None].to(tl.int64) * W
    kptr = trow + (NQ + kvh) * D + offs_h[None, :]
    k1 = _load_rows(kptr, 0, tmask2, QSPLIT, split_stride).to(tl.float32)
    k2 = _load_rows(kptr + HALF, 0, tmask2, QSPLIT, split_stride).to(tl.float32)
    k1, k2 = _norm_rope_half(k1, k2, kw_ptr, cos_ptr, sin_ptr, tpos, offs_h, tmask2, eps, D, HALF)
    dst = cache_base + tpos[:, None].to(tl.int64) * D + offs_h[None, :]
    tl.store(kc_ptr + dst, k1, mask=tmask2)
    tl.store(kc_ptr + dst + HALF, k2, mask=tmask2)
    vptr = trow + (NQ + NKV + kvh) * D + offs_h[None, :]
    v1 = _load_rows(vptr, 0, tmask2, QSPLIT, split_stride)
    v2 = _load_rows(vptr + HALF, 0, tmask2, QSPLIT, split_stride)
    tl.store(vc_ptr + dst, v1, mask=tmask2)
    tl.store(vc_ptr + dst + HALF, v2, mask=tmask2)
    tl.debug_barrier()

    # ---- queries of this KV head's GROUP heads for the T tokens
    offs_r = tl.arange(0, RPAD)
    rmask = offs_r < ROWS
    j = offs_r // GROUP
    g = offs_r % GROUP
    rmask2 = rmask[:, None] & (offs_h[None, :] < HALF)
    qptr = (qkv_ptr + (b * T + j)[:, None].to(tl.int64) * W
            + (kvh * GROUP + g)[:, None] * D + offs_h[None, :])
    q1 = _load_rows(qptr, 0, rmask2, QSPLIT, split_stride).to(tl.float32)
    q2 = _load_rows(qptr + HALF, 0, rmask2, QSPLIT, split_stride).to(tl.float32)
    qa, qb = _norm_rope_half(q1, q2, qw_ptr, cos_ptr, sin_ptr, p0 + j, offs_h, rmask2, eps, D, HALF)
    limit = p0 + j

    m_i = tl.full([RPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([RPAD], tl.float32)
    acc = tl.zeros([RPAD, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        row_off = cache_base + offs_n[:, None].to(tl.int64) * D
        ka = tl.load(kc_ptr + row_off + offs_h[None, :], mask=nmask[:, None], other=0.0)
        kb = tl.load(kc_ptr + row_off + HALF + offs_h[None, :], mask=nmask[:, None], other=0.0)
        if DOT_F32:
            s = tl.dot(qa.to(tl.float32), tl.trans(ka.to(tl.float32)))
            s += tl.dot(qb.to(tl.float32), tl.trans(kb.to(tl.float32)))
        else:
            s = tl.dot(qa, tl.trans(ka))
            s += tl.dot(qb, tl.trans(kb))
        s = s * scale
        valid = nmask[None, :] & (offs_n[None, :] <= limit[:, None])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + row_off + offs_d[None, :], mask=nmask[:, None], other=0.0)
        if DOT_F32:
            pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
        else:
            pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    out_rows = (b * T + j) * NQ + kvh * GROUP + g          # output head rows
    if NSPLIT == 1:
        res = acc / l_i[:, None]
        tl.store(out_ptr + out_rows[:, None].to(tl.int64) * D + offs_d[None, :],
                 res.to(tl.bfloat16), mask=rmask[:, None])
    else:
        part = (pid * NSPLIT + split) * ROWS
        tl.store(m_ptr + part + offs_r, m_i, mask=rmask)
        tl.store(l_ptr + part + offs_r, l_i, mask=rmask)
        tl.store(o_ptr + (part + offs_r)[:, None] * D + offs_d[None, :], acc, mask=rmask[:, None])
        _fence(FENCE)
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
        tl.debug_barrier()
        if done == NSPLIT - 1:
            _fence(FENCE)
            # every other split has published its partials: merge them
            base_r = pid * NSPLIT * ROWS + offs_r
            m_max = tl.full([RPAD], float("-inf"), tl.float32)
            for sp in range(0, NSPLIT):
                ms = tl.load(m_ptr + base_r + sp * ROWS, mask=rmask, other=float("-inf"),
                             cache_modifier=".cg")
                m_max = tl.maximum(m_max, ms)
            den = tl.zeros([RPAD], tl.float32)
            tot = tl.zeros([RPAD, D], tl.float32)
            for sp in range(0, NSPLIT):
                ms = tl.load(m_ptr + base_r + sp * ROWS, mask=rmask, other=float("-inf"),
                             cache_modifier=".cg")
                ls = tl.load(l_ptr + base_r + sp * ROWS, mask=rmask, other=0.0,
                             cache_modifier=".cg")
                o_s = tl.load(o_ptr + (base_r + sp * ROWS)[:, None] * D + offs_d[None, :],
                              mask=rmask[:, None], other=0.0, cache_modifier=".cg")
                w = tl.where(ms > float("-inf"), tl.exp(ms - m_max), 0.0)
                den += w * ls
                tot += w[:, None] * o_s
            res = tot / den[:, None]
            tl.store(out_ptr + out_rows[:, None].to(tl.int64) * D + offs_d[None, :],
                     res.to(tl.bfloat16), mask=rmask[:, None])
            tl.atomic_xchg(cnt_ptr + pid, 0)


class FusedDecodeAttention(DecodeAttention):
    """Norm + RoPE + cache write + attention + combine for T tokens per sequence."""

    # (BLOCK_N, num_warps, num_stages) candidates, picked by timing once
    CONFIGS = [(64, 4, 2), (128, 8, 3), (64, 4, 3)]

    def __init__(self, batch, t, capacity, nq, nkv, d, device, num_sms):
        super().__init__(batch, t, capacity, nq, nkv, d, device, num_sms)
        self.cnt = torch.zeros((batch * nkv,), device=device, dtype=torch.int32)
        self.tpad = max(2, triton.next_power_of_2(t))
        self.cfg = self.CONFIGS[0]
        self.tuned = False

    def tune(self, qkv, q_w, k_w, cos, sin, k_cache, v_cache, pos_t, eps):
        """Pick the fastest config at the current positions (call outside graph
        capture, with positions at the longest context of interest)."""
        if self.tuned or _DOT_F32:
            return
        self.tuned = True
        best, best_t = self.cfg, None
        for cfg in self.CONFIGS:
            try:
                self.cfg = cfg
                self(qkv, q_w, k_w, cos, sin, k_cache, v_cache, pos_t, eps)
                torch.cuda.synchronize()
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e0.record()
                for _ in range(10):
                    self(qkv, q_w, k_w, cos, sin, k_cache, v_cache, pos_t, eps)
                e1.record()
                e1.synchronize()
                t = e0.elapsed_time(e1)
                if best_t is None or t < best_t:
                    best, best_t = cfg, t
            except Exception:  # pragma: no cover
                pass
        self.cfg = best

    def __call__(self, qkv, q_w, k_w, cos, sin, k_cache, v_cache, pos_t, eps):
        """qkv: bf16 [B*T, W] or fp32 split-K partials [S, B*T, W]. Returns [B*T, NQ*D]."""
        B, T = self.batch, self.t
        qsplit = qkv.shape[0] if qkv.dim() == 3 else 0
        M, W = qkv.shape[-2], qkv.shape[-1]
        out = torch.empty((B * T, self.nq * self.d), device=qkv.device, dtype=torch.bfloat16)
        _fused_attn_kernel[(B * self.nkv, self.nsplit)](
            qkv, q_w, k_w, cos, sin, k_cache, v_cache, pos_t,
            self.o, self.m, self.l, self.cnt, out,
            k_cache.stride(0), k_cache.stride(1), M * W, self.scale, eps, self.chunk,
            NKV=self.nkv, GROUP=self.group, T=T, RPAD=self.rpad, TPAD=self.tpad,
            D=self.d, BLOCK_N=self.cfg[0], NSPLIT=self.nsplit, QSPLIT=qsplit,
            DOT_F32=_DOT_F32, FENCE=not _DOT_F32, num_warps=self.cfg[1], num_stages=self.cfg[2],
        )
        return out
