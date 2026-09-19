"""Persistent fused MLP for decode: act = SwiGLU(h @ [gate; up]^T), then
partial down-projection, in ONE launch of exactly G programs (G <= #SMs, so
every program is resident and spin-waiting cannot deadlock).

Phase 1: gate/up tiles (BN columns of I each) are dealt round-robin, in
column order; each finished tile bumps the counter of its CHUNK of columns.
Phase 2: down-projection work items (BN2 output rows x one K-split of I) start
as soon as the act chunks they read are complete, overlapping the tail of
phase 1 instead of waiting for a kernel boundary. Output is fp32 split-K
partials [S, M, H], reduced by the next residual+RMSNorm kernel. The last
program to finish resets the counters for the next launch.
"""

import os

import torch
import triton
import triton.language as tl

_DOT_F32 = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _mlp_kernel(h_ptr, wgu_ptr, wd_ptr, act_ptr, out_ptr, cnt_ptr, done_ptr,
                M, H, I, G,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                BN2: tl.constexpr, BK2: tl.constexpr, CHUNK: tl.constexpr,
                SPLIT: tl.constexpr, DOT_F32: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    TILES_PER_CHUNK: tl.constexpr = CHUNK // BN

    # ---- phase 1: act[:, n0:n0+BN] = swiglu
    n_tiles = I // BN
    for t in range(pid, n_tiles, G):
        n0 = t * BN
        offs_n = n0 + tl.arange(0, BN)
        g_rows = wgu_ptr + offs_n[:, None].to(tl.int64) * H
        u_rows = wgu_ptr + (I + offs_n[:, None]).to(tl.int64) * H
        acc_g = tl.zeros([BM, BN], tl.float32)
        acc_u = tl.zeros([BM, BN], tl.float32)
        for k0 in range(0, H, BK):
            offs_k = k0 + tl.arange(0, BK)
            x = tl.load(h_ptr + offs_m[:, None] * H + offs_k[None, :], mask=mmask[:, None], other=0.0)
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
        tl.store(act_ptr + offs_m[:, None] * I + offs_n[None, :], (s * u).to(tl.bfloat16),
                 mask=mmask[:, None])
        tl.debug_barrier()
        tl.atomic_add(cnt_ptr + n0 // CHUNK, 1, sem="release")

    # ---- phase 2: out[s, :, r0:r0+BN2] = act[:, ks] @ Wd[r0:r0+BN2, ks]^T
    k_per_split = I // SPLIT
    n_items = (H // BN2) * SPLIT
    for item in range(pid, n_items, G):
        sp = item % SPLIT
        r0 = (item // SPLIT) * BN2
        offs_r = r0 + tl.arange(0, BN2)
        d_rows = wd_ptr + offs_r[:, None].to(tl.int64) * I
        acc = tl.zeros([BM, BN2], tl.float32)
        kbeg = sp * k_per_split
        for k0 in range(kbeg, kbeg + k_per_split, BK2):
            c = k0 // CHUNK
            while tl.atomic_add(cnt_ptr + c, 0, sem="acquire") < TILES_PER_CHUNK:
                pass
            offs_k = k0 + tl.arange(0, BK2)
            a = tl.load(act_ptr + offs_m[:, None] * I + offs_k[None, :], mask=mmask[:, None],
                        other=0.0, cache_modifier=".cg")
            w = tl.load(d_rows + offs_k[None, :])
            if DOT_F32:
                acc += tl.dot(a.to(tl.float32), tl.trans(w.to(tl.float32)))
            else:
                acc += tl.dot(a, tl.trans(w))
        tl.store(out_ptr + (sp * M + offs_m)[:, None].to(tl.int64) * H + offs_r[None, :], acc,
                 mask=mmask[:, None])

    # ---- the last program out resets the chunk counters for the next launch
    tl.debug_barrier()
    finished = tl.atomic_add(done_ptr, 1, sem="acq_rel")
    if finished == G - 1:
        n_chunks = I // CHUNK
        for c in range(0, n_chunks):
            tl.atomic_xchg(cnt_ptr + c, 0)
        tl.atomic_xchg(done_ptr, 0)


class PersistentMLP:
    """Workspace for one M. Call with (h [M, H] bf16, w_gu [2I, H], w_down [H, I])."""

    def __init__(self, M, H, I, device, programs, split=4, bn=16, bn2=16, chunk=512):
        assert I % chunk == 0 and chunk % bn == 0 and I % split == 0
        assert (I // split) % 64 == 0 and H % bn2 == 0
        self.M, self.H, self.I, self.G = M, H, I, programs
        self.split, self.bn, self.bn2, self.chunk = split, bn, bn2, chunk
        self.cnt = torch.zeros((I // chunk,), device=device, dtype=torch.int32)
        self.done = torch.zeros((1,), device=device, dtype=torch.int32)

    def __call__(self, h, w_gu, w_down):
        M, H, I = self.M, self.H, self.I
        act = torch.empty((M, I), device=h.device, dtype=torch.bfloat16)
        out = torch.empty((self.split, M, H), device=h.device, dtype=torch.float32)
        _mlp_kernel[(self.G,)](
            h, w_gu, w_down, act, out, self.cnt, self.done, M, H, I, self.G,
            BM=max(16, triton.next_power_of_2(M)), BN=self.bn, BK=128,
            BN2=self.bn2, BK2=64, CHUNK=self.chunk, SPLIT=self.split,
            DOT_F32=_DOT_F32, num_warps=4, num_stages=3,
        )
        return out
