"""On-device n-gram drafting and exact acceptance for speculative decoding.

Each sequence keeps its tokens (prompt + generated) in hist[b, :hlen[b]].
A step is: draft (this file) -> verify forward over T tokens -> accept (this
file), all inside one CUDA graph, so steps can be replayed back-to-back with
no host round trip.

draft:  find the latest earlier occurrence of the longest suffix (up to 4
        tokens) of the sequence; propose the T-1 tokens that followed it.
        Input row = [last token, drafts...] at positions hlen-1 ...
accept: the model's argmax at every row is exact greedy for that prefix, so
        keep argmax[0..a] where drafts[0..a-1] matched argmax[0..a-1].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _draft_kernel(hist_ptr, hlen_ptr, inp_ptr, pos_ptr, HCAP,
                  T: tl.constexpr, TPAD: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    L = tl.load(hlen_ptr + b)
    base = hist_ptr + b.to(tl.int64) * HCAP
    t0 = tl.load(base + L - 1)
    t1 = tl.load(base + L - 2, mask=L >= 2, other=-1)
    t2 = tl.load(base + L - 3, mask=L >= 3, other=-1)
    t3 = tl.load(base + L - 4, mask=L >= 4, other=-1)
    best = tl.full([BLOCK], -1, tl.int32)
    for i0 in range(0, L - 1, BLOCK):
        i = i0 + tl.arange(0, BLOCK)
        valid = i <= L - 2
        h0 = tl.load(base + i, mask=valid, other=-2)
        h1 = tl.load(base + i - 1, mask=valid & (i >= 1), other=-3)
        h2 = tl.load(base + i - 2, mask=valid & (i >= 2), other=-4)
        h3 = tl.load(base + i - 3, mask=valid & (i >= 3), other=-5)
        m1 = valid & (h0 == t0)
        m2 = m1 & (h1 == t1) & (L >= 2)
        m3 = m2 & (h2 == t2) & (L >= 3)
        m4 = m3 & (h3 == t3) & (L >= 4)
        n = m1.to(tl.int32) + m2.to(tl.int32) + m3.to(tl.int32) + m4.to(tl.int32)
        score = tl.where(n > 0, n * 16777216 + i, -1)
        best = tl.maximum(best, score)
    top = tl.max(best, axis=0)
    start = (top % 16777216) + 1
    k = tl.arange(0, TPAD)
    # continuation of the match; when it runs off the end of the history the
    # text is repeating with period (L - start): wrap around cyclically
    period = tl.maximum(L - start, 1)
    src = start + (k - 1) % period
    dmask = (k >= 1) & (k < T) & (top >= 0) & (src < L)
    d = tl.load(base + src, mask=dmask, other=0)
    d = tl.where(dmask, d, t0)
    tok = tl.where(k == 0, t0, d)
    tl.store(inp_ptr + b * T + k, tok.to(tl.int64), mask=k < T)
    tl.store(pos_ptr + b, L - 1)


@triton.jit
def _accept_kernel(hist_ptr, hlen_ptr, lim_ptr, inp_ptr, out_ptr, HCAP,
                   T: tl.constexpr, TPAD: tl.constexpr):
    b = tl.program_id(0)
    L = tl.load(hlen_ptr + b)
    lim = tl.load(lim_ptr + b)
    k = tl.arange(0, TPAD)
    o = tl.load(out_ptr + b * T + k, mask=k < T, other=-1)
    d = tl.load(inp_ptr + b * T + k + 1, mask=k < T - 1, other=-2)
    eq = (o == d) & (k < T - 1)
    a = tl.min(tl.where(eq, T - 1, k), axis=0)      # accepted drafts
    wmask = (k <= a) & (k < T) & (L + k < lim)
    tl.store(hist_ptr + b.to(tl.int64) * HCAP + L + k, o.to(tl.int32), mask=wmask)
    tl.store(hlen_ptr + b, tl.minimum(L + a + 1, lim))


def draft(hist, hlen, inp, pos, t):
    B, hcap = hist.shape
    _draft_kernel[(B,)](hist, hlen, inp, pos, hcap, T=t,
                        TPAD=max(2, triton.next_power_of_2(t)), BLOCK=1024, num_warps=4)


def accept(hist, hlen, lim, inp, out, t):
    B, hcap = hist.shape
    _accept_kernel[(B,)](hist, hlen, lim, inp, out, hcap, T=t,
                         TPAD=max(2, triton.next_power_of_2(t)), num_warps=1)
