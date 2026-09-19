"""Persistent decode megakernel: one launch per decode step (T = 1, M = B <= 16).

Exactly G programs (G <= #SMs, so all are co-resident and spin-waits cannot
deadlock) walk every layer's phases in order:

  P1  qkv   = rmsnorm(x + delta, ln1) @ Wqkv^T          (rows of 6144 split over G)
  P2  attn  = norm/rope/cache-write + split-K GQA + combine (work items over G)
  P3  o     = attn @ Wo^T                                 (rows of H)
  P4  act   = swiglu(rmsnorm(x + o, ln2) @ [Wg; Wu]^T)    (rows of I)
  P5  delta = act @ Wd^T                                  (rows of H)
  P6  final rmsnorm + lm head rows + argmax partials; last program reduces,
      writes the next token, advances positions and resets the counters.

Phases are ordered by monotonic global counters (one per phase), replacing
~9 kernel boundaries per layer. Every rounding point matches the reference
(and the multi-kernel path): each Linear output is rounded to bf16, residual
adds are bf16, RMSNorm casts before the weight multiply.
"""

import os

import torch
import triton
import triton.language as tl

from kernels.fused_attn import _norm_rope_half

_DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _wait(ptr, target):
    while tl.atomic_add(ptr, 0, sem="acquire") < target:
        pass


@triton.jit
def _signal(ptr):
    tl.debug_barrier()
    tl.atomic_add(ptr, 1, sem="release")


@triton.jit
def _dot(a, b, DOT_F32: tl.constexpr):
    if DOT_F32:
        return tl.dot(a.to(tl.float32), tl.trans(b.to(tl.float32)))
    else:
        return tl.dot(a, tl.trans(b))


@triton.jit
def _residual_chunk(res_ptr, emb_ptr, tok, d_ptr, offs_m, mmask, offs_k, H,
                    FIRST: tl.constexpr, HAS_D: tl.constexpr):
    """bf16 residual stream chunk [BM, BK]: FIRST -> embedding rows, else
    res (+ d), with the reference's bf16 rounding of the add."""
    m2 = mmask[:, None]
    if FIRST:
        x = tl.load(emb_ptr + tok[:, None] * H + offs_k[None, :], mask=m2, other=0.0)
    else:
        x = tl.load(res_ptr + offs_m[:, None] * H + offs_k[None, :], mask=m2, other=0.0,
                    cache_modifier=".cg")
        if HAS_D:
            d = tl.load(d_ptr + offs_m[:, None] * H + offs_k[None, :], mask=m2, other=0.0,
                        cache_modifier=".cg")
            x = (x.to(tl.float32) + d.to(tl.float32)).to(tl.bfloat16)
    return x


@triton.jit
def _normed_rows(res_ptr, emb_ptr, tok, d_ptr, ln_ptr, res_out_ptr, w_ptr, out_ptr,
                 M, H, N, r0, r1, eps, write_res,
                 FIRST: tl.constexpr, HAS_D: tl.constexpr, SWIGLU: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_F32: tl.constexpr):
    """out[:, r0:r1] = rmsnorm(residual) @ W[r0:r1]^T (SWIGLU: silu(g)*u with
    up rows at W[N + r]). Writes the residual to res_out if write_res."""
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    # row statistics (identical in every program: same data, same order)
    ss = tl.zeros([BM], tl.float32)
    for k0 in range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = _residual_chunk(res_ptr, emb_ptr, tok, d_ptr, offs_m, mmask, offs_k, H, FIRST, HAS_D)
        if write_res:
            tl.store(res_out_ptr + offs_m[:, None] * H + offs_k[None, :], x, mask=mmask[:, None])
        xf = x.to(tl.float32)
        ss += tl.sum(xf * xf, axis=1)
    rstd = tl.math.rsqrt(ss / H + eps)[:, None]
    for n0 in range(r0, r1, BN):
        offs_n = n0 + tl.arange(0, BN)
        nmask = offs_n < r1
        acc = tl.zeros([BM, BN], tl.float32)
        acc_u = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, H, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = _residual_chunk(res_ptr, emb_ptr, tok, d_ptr, offs_m, mmask, offs_k, H, FIRST, HAS_D)
            w_ln = tl.load(ln_ptr + offs_k).to(tl.float32)[None, :]
            h = ((x.to(tl.float32) * rstd).to(tl.bfloat16).to(tl.float32) * w_ln).to(tl.bfloat16)
            w = tl.load(w_ptr + offs_n[:, None].to(tl.int64) * H + offs_k[None, :],
                        mask=nmask[:, None], other=0.0)
            acc += _dot(h, w, DOT_F32)
            if SWIGLU:
                wu = tl.load(w_ptr + (N + offs_n[:, None]).to(tl.int64) * H + offs_k[None, :],
                             mask=nmask[:, None], other=0.0)
                acc_u += _dot(h, wu, DOT_F32)
        omask = mmask[:, None] & nmask[None, :]
        if SWIGLU:
            g = acc.to(tl.bfloat16).to(tl.float32)
            u = acc_u.to(tl.bfloat16).to(tl.float32)
            s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
            res = (s * u).to(tl.bfloat16)
        else:
            res = acc.to(tl.bfloat16)
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], res, mask=omask)


@triton.jit
def _plain_rows(x_ptr, w_ptr, out_ptr, M, K, N, r0, r1,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DOT_F32: tl.constexpr):
    """out[:, r0:r1] = bf16(x @ W[r0:r1]^T), x bf16 [M, K]."""
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    for n0 in range(r0, r1, BN):
        offs_n = n0 + tl.arange(0, BN)
        nmask = offs_n < r1
        acc = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mmask[:, None],
                        other=0.0, cache_modifier=".cg")
            w = tl.load(w_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :],
                        mask=nmask[:, None], other=0.0)
            acc += _dot(x, w, DOT_F32)
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.bfloat16),
                 mask=mmask[:, None] & nmask[None, :])


@triton.jit
def _attn_item(qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, kc_ptr, vc_ptr, pos_ptr,
               o_ptr, m_ptr, l_ptr, cnt_ptr, out_ptr, item,
               stride_cb, stride_ch, scale, eps, CHUNK,
               NKV: tl.constexpr, GROUP: tl.constexpr, RPAD: tl.constexpr, D: tl.constexpr,
               BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr, DOT_F32: tl.constexpr):
    """One (b, kvh, split) unit of fused decode attention for T = 1.
    Returns 1 if this call completed the (b, kvh) output, else 0."""
    pid = item // NSPLIT
    split = item % NSPLIT
    b = pid // NKV
    kvh = pid % NKV
    HALF: tl.constexpr = D // 2
    NQ: tl.constexpr = NKV * GROUP
    ROWS: tl.constexpr = GROUP
    W: tl.constexpr = (NQ + 2 * NKV) * D
    p0 = tl.load(pos_ptr + b)
    start = split * CHUNK
    end = tl.minimum(start + CHUNK, p0 + 1)
    offs_h = tl.arange(0, HALF)
    offs_d = tl.arange(0, D)
    cache_base = b.to(tl.int64) * stride_cb + kvh * stride_ch
    row = qkv_ptr + b.to(tl.int64) * W

    # new K/V (only the split holding position p0 writes and reads it)
    offs_t = tl.arange(0, 2)
    tpos = p0 + offs_t
    tmask = (offs_t < 1) & (tpos >= start) & (tpos < end)
    tmask2 = tmask[:, None] & (offs_h[None, :] < HALF)
    kptr = row + (NQ + kvh) * D + offs_h[None, :] + offs_t[:, None] * 0
    k1 = tl.load(kptr, mask=tmask2, other=0.0, cache_modifier=".cg").to(tl.float32)
    k2 = tl.load(kptr + HALF, mask=tmask2, other=0.0, cache_modifier=".cg").to(tl.float32)
    k1, k2 = _norm_rope_half(k1, k2, kw_ptr, cos_ptr, sin_ptr, tpos, offs_h, tmask2, eps, D, HALF)
    dst = cache_base + tpos[:, None].to(tl.int64) * D + offs_h[None, :]
    tl.store(kc_ptr + dst, k1, mask=tmask2)
    tl.store(kc_ptr + dst + HALF, k2, mask=tmask2)
    vptr = row + (NQ + NKV + kvh) * D + offs_h[None, :] + offs_t[:, None] * 0
    tl.store(vc_ptr + dst, tl.load(vptr, mask=tmask2, other=0.0, cache_modifier=".cg"), mask=tmask2)
    tl.store(vc_ptr + dst + HALF, tl.load(vptr + HALF, mask=tmask2, other=0.0, cache_modifier=".cg"),
             mask=tmask2)
    tl.debug_barrier()

    offs_r = tl.arange(0, RPAD)
    rmask = offs_r < ROWS
    rmask2 = rmask[:, None] & (offs_h[None, :] < HALF)
    qptr = row + (kvh * GROUP + offs_r)[:, None] * D + offs_h[None, :]
    q1 = tl.load(qptr, mask=rmask2, other=0.0, cache_modifier=".cg").to(tl.float32)
    q2 = tl.load(qptr + HALF, mask=rmask2, other=0.0, cache_modifier=".cg").to(tl.float32)
    qa, qb = _norm_rope_half(q1, q2, qw_ptr, cos_ptr, sin_ptr, p0 + offs_r * 0, offs_h, rmask2,
                             eps, D, HALF)

    m_i = tl.full([RPAD], float("-inf"), tl.float32)
    l_i = tl.zeros([RPAD], tl.float32)
    acc = tl.zeros([RPAD, D], tl.float32)
    for n0 in range(start, end, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        nmask = offs_n < end
        row_off = cache_base + offs_n[:, None].to(tl.int64) * D
        ka = tl.load(kc_ptr + row_off + offs_h[None, :], mask=nmask[:, None], other=0.0,
                     cache_modifier=".cg")
        kb = tl.load(kc_ptr + row_off + HALF + offs_h[None, :], mask=nmask[:, None], other=0.0,
                     cache_modifier=".cg")
        s = (_dot(qa, ka, DOT_F32) + _dot(qb, kb, DOT_F32)) * scale
        s = tl.where(nmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(vc_ptr + row_off + offs_d[None, :], mask=nmask[:, None], other=0.0,
                    cache_modifier=".cg")
        if DOT_F32:
            pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
        else:
            pv = tl.dot(p.to(tl.bfloat16), v)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    out_rows = b * NQ + kvh * GROUP + offs_r
    completed = 0
    if NSPLIT == 1:
        tl.store(out_ptr + out_rows[:, None] * D + offs_d[None, :],
                 (acc / l_i[:, None]).to(tl.bfloat16), mask=rmask[:, None])
        completed = 1
    else:
        part = (pid * NSPLIT + split) * ROWS
        tl.store(m_ptr + part + offs_r, m_i, mask=rmask)
        tl.store(l_ptr + part + offs_r, l_i, mask=rmask)
        tl.store(o_ptr + (part + offs_r)[:, None] * D + offs_d[None, :], acc, mask=rmask[:, None])
        tl.debug_barrier()
        done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
        tl.debug_barrier()
        if done == NSPLIT - 1:
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
                ls = tl.load(l_ptr + base_r + sp * ROWS, mask=rmask, other=0.0, cache_modifier=".cg")
                o_s = tl.load(o_ptr + (base_r + sp * ROWS)[:, None] * D + offs_d[None, :],
                              mask=rmask[:, None], other=0.0, cache_modifier=".cg")
                w = tl.where(ms > float("-inf"), tl.exp(ms - m_max), 0.0)
                den += w * ls
                tot += w[:, None] * o_s
            tl.store(out_ptr + out_rows[:, None] * D + offs_d[None, :],
                     (tot / den[:, None]).to(tl.bfloat16), mask=rmask[:, None])
            tl.atomic_xchg(cnt_ptr + pid, 0)
            completed = 1
    return completed


@triton.jit
def _mega_kernel(tok_ptr, pos_ptr, emb_ptr,
                 wqkv_ptr, wo_ptr, wgu_ptr, wd_ptr, ln1_ptr, ln2_ptr, qn_ptr, kn_ptr, fn_ptr,
                 cos_ptr, sin_ptr, kc_ptr, vc_ptr,
                 res_ptr, qkv_ptr, att_ptr, o_ptr, act_ptr, dl_ptr, hfin_ptr,
                 po_ptr, pm_ptr, pl_ptr, acnt_ptr, ctr_ptr, bval_ptr, bidx_ptr,
                 M, G, n_layers, V, eps, scale, CHUNK, stride_cl, stride_cb, stride_ch,
                 H: tl.constexpr, I: tl.constexpr, NQKV: tl.constexpr,
                 NKV: tl.constexpr, GROUP: tl.constexpr, D: tl.constexpr, NSPLIT: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, BLOCK_N: tl.constexpr,
                 DOT_F32: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    tok = tl.load(tok_ptr + offs_m, mask=mmask, other=0)
    NQ: tl.constexpr = NKV * GROUP
    c_qkv = ctr_ptr + 0
    c_att = ctr_ptr + 1
    c_o = ctr_ptr + 2
    c_gu = ctr_ptr + 3
    c_dn = ctr_ptr + 4
    c_lm = ctr_ptr + 5
    n_items = M * NKV * NSPLIT
    # residual ping-pong: res[0] after P1 (x + delta), res[1] after P4 (x + o)
    res_a = res_ptr
    res_b = res_ptr + M * H
    # per-layer pointers, advanced each iteration (64-bit pointer arithmetic)
    wq = wqkv_ptr
    wo = wo_ptr
    wgu = wgu_ptr
    wd = wd_ptr
    kc = kc_ptr
    vc = vc_ptr
    for l in range(0, n_layers):
        # ---- P1: qkv
        r0 = (pid * NQKV) // G
        r1 = ((pid + 1) * NQKV) // G
        if l == 0:
            _normed_rows(res_b, emb_ptr, tok, dl_ptr, ln1_ptr, res_a, wq, qkv_ptr,
                         M, H, NQKV, r0, r1, eps, pid == 0,
                         True, False, False, BM, BN, BK, DOT_F32)
        else:
            _wait(c_dn, l * G)
            _normed_rows(res_b, emb_ptr, tok, dl_ptr, ln1_ptr + l * H, res_a, wq, qkv_ptr,
                         M, H, NQKV, r0, r1, eps, pid == 0,
                         False, True, False, BM, BN, BK, DOT_F32)
        _signal(c_qkv)
        # ---- P2: attention
        _wait(c_qkv, (l + 1) * G)
        n_done = 0
        for item in range(pid, n_items, G):
            n_done += _attn_item(qkv_ptr, qn_ptr + l * D, kn_ptr + l * D, cos_ptr, sin_ptr,
                                 kc, vc, pos_ptr,
                                 po_ptr, pm_ptr, pl_ptr, acnt_ptr, att_ptr, item,
                                 stride_cb, stride_ch, scale, eps, CHUNK,
                                 NKV, GROUP, 16, D, BLOCK_N, NSPLIT, DOT_F32)
        tl.debug_barrier()
        if n_done > 0:
            tl.atomic_add(c_att, n_done, sem="release")
        # ---- P3: o projection
        _wait(c_att, (l + 1) * M * NKV)
        r0 = (pid * H) // G
        r1 = ((pid + 1) * H) // G
        _plain_rows(att_ptr, wo, o_ptr, M, NQ * D, H, r0, r1,
                    BM, BN, BK, DOT_F32)
        _signal(c_o)
        # ---- P4: gate/up + swiglu (residual += o)
        _wait(c_o, (l + 1) * G)
        r0 = (pid * I) // G
        r1 = ((pid + 1) * I) // G
        _normed_rows(res_a, emb_ptr, tok, o_ptr, ln2_ptr + l * H, res_b,
                     wgu, act_ptr,
                     M, H, I, r0, r1, eps, pid == 0,
                     False, True, True, BM, BN, BK, DOT_F32)
        _signal(c_gu)
        # ---- P5: down projection
        _wait(c_gu, (l + 1) * G)
        r0 = (pid * H) // G
        r1 = ((pid + 1) * H) // G
        _plain_rows(act_ptr, wd, dl_ptr, M, I, H, r0, r1,
                    BM, BN, BK, DOT_F32)
        _signal(c_dn)
        wq += NQKV * H
        wo += H * NQ * D
        wgu += 2 * I * H
        wd += H * I
        kc += stride_cl
        vc += stride_cl

    # ---- P6: final norm + lm head + argmax
    _wait(c_dn, n_layers * G)
    ss = tl.zeros([BM], tl.float32)
    for k0 in range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = _residual_chunk(res_b, emb_ptr, tok, dl_ptr, offs_m, mmask, offs_k, H, False, True)
        xf = x.to(tl.float32)
        ss += tl.sum(xf * xf, axis=1)
    rstd = tl.math.rsqrt(ss / H + eps)[:, None]
    if pid == 0:
        for k0 in range(0, H, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = _residual_chunk(res_b, emb_ptr, tok, dl_ptr, offs_m, mmask, offs_k, H, False, True)
            w_ln = tl.load(fn_ptr + offs_k).to(tl.float32)[None, :]
            h = ((x.to(tl.float32) * rstd).to(tl.bfloat16).to(tl.float32) * w_ln).to(tl.bfloat16)
            tl.store(hfin_ptr + offs_m[:, None] * H + offs_k[None, :], h, mask=mmask[:, None])
    r0 = (pid * V) // G
    r1 = ((pid + 1) * V) // G
    best_v = tl.full([BM], float("-inf"), tl.float32)
    best_i = tl.zeros([BM], tl.int32)
    for n0 in range(r0, r1, BN):
        offs_n = n0 + tl.arange(0, BN)
        nmask = offs_n < r1
        acc = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, H, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = _residual_chunk(res_b, emb_ptr, tok, dl_ptr, offs_m, mmask, offs_k, H, False, True)
            w_ln = tl.load(fn_ptr + offs_k).to(tl.float32)[None, :]
            h = ((x.to(tl.float32) * rstd).to(tl.bfloat16).to(tl.float32) * w_ln).to(tl.bfloat16)
            w = tl.load(emb_ptr + offs_n[:, None].to(tl.int64) * H + offs_k[None, :],
                        mask=nmask[:, None], other=0.0)
            acc += _dot(h, w, DOT_F32)
        logit = acc.to(tl.bfloat16).to(tl.float32)
        logit = tl.where(nmask[None, :], logit, float("-inf"))
        tmax = tl.max(logit, axis=1)
        cand = tl.where(logit == tmax[:, None], offs_n[None, :], 2147483647)
        targ = tl.min(cand, axis=1)
        better = tmax > best_v
        best_i = tl.where(better, targ, best_i)
        best_v = tl.where(better, tmax, best_v)
    tl.store(bval_ptr + pid * BM + offs_m, best_v)
    tl.store(bidx_ptr + pid * BM + offs_m, best_i)
    tl.debug_barrier()
    fin = tl.atomic_add(c_lm, 1, sem="acq_rel")
    if fin == G - 1:
        bv = tl.full([BM], float("-inf"), tl.float32)
        bi = tl.zeros([BM], tl.int32)
        for p in range(0, G):
            v = tl.load(bval_ptr + p * BM + offs_m, cache_modifier=".cg")
            i = tl.load(bidx_ptr + p * BM + offs_m, cache_modifier=".cg")
            better = v > bv
            bi = tl.where(better, i, bi)
            bv = tl.where(better, v, bv)
        tl.store(tok_ptr + offs_m, bi.to(tl.int64), mask=mmask)
        pos = tl.load(pos_ptr + offs_m, mask=mmask, other=0)
        tl.store(pos_ptr + offs_m, pos + 1, mask=mmask)
        for c in range(0, 6):
            tl.atomic_xchg(ctr_ptr + c, 0)


class MegaDecode:
    """Buffers + launcher for the persistent decode step of one (B, capacity)."""

    def __init__(self, eng, st, attn_ws, programs):
        dev = eng.device
        B = st.batch
        self.B, self.G = B, programs
        H = eng.embed.shape[1]
        I = eng.layers[0]["down"].shape[1]
        self.H, self.I = H, I
        self.nqkv = eng.layers[0]["qkv"].shape[0]
        self.res = torch.zeros((2, B, H), device=dev, dtype=torch.bfloat16)
        self.qkv = torch.zeros((B, self.nqkv), device=dev, dtype=torch.bfloat16)
        self.att = torch.zeros((B, eng.nq * eng.d), device=dev, dtype=torch.bfloat16)
        self.o = torch.zeros((B, H), device=dev, dtype=torch.bfloat16)
        self.act = torch.zeros((B, I), device=dev, dtype=torch.bfloat16)
        self.dl = torch.zeros((B, H), device=dev, dtype=torch.bfloat16)
        self.hfin = torch.zeros((B, H), device=dev, dtype=torch.bfloat16)
        self.ctr = torch.zeros((8,), device=dev, dtype=torch.int32)
        self.bval = torch.zeros((programs, 16), device=dev, dtype=torch.float32)
        self.bidx = torch.zeros((programs, 16), device=dev, dtype=torch.int32)
        self.ws = attn_ws          # split-K attention workspace (o/m/l/cnt, chunk, nsplit)

    def __call__(self, eng, st, tok, pos):
        ws = self.ws
        kc, vc = st.k_cache, st.v_cache
        _mega_kernel[(self.G,)](
            tok, pos, eng.embed,
            eng.w_qkv, eng.w_o, eng.w_gu, eng.w_down, eng.w_ln1, eng.w_ln2, eng.w_qn, eng.w_kn,
            eng.final_norm, eng.cos, eng.sin, kc, vc,
            self.res, self.qkv, self.att, self.o, self.act, self.dl, self.hfin,
            ws.o, ws.m, ws.l, ws.cnt, self.ctr, self.bval, self.bidx,
            self.B, self.G, eng.n_layers, eng.lm_head.shape[0], eng.eps, ws.scale, ws.chunk,
            kc.stride(0), kc.stride(1), kc.stride(2),
            H=self.H, I=self.I, NQKV=self.nqkv, NKV=eng.nkv, GROUP=eng.nq // eng.nkv, D=eng.d,
            NSPLIT=ws.nsplit, BM=16, BN=32, BK=128, BLOCK_N=64, DOT_F32=_DOT_F32,
            num_warps=8, num_stages=3,
        )
