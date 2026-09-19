"""Lossless 12-bit weights for decode GEMVs.

A bf16 value is sign(1) | exponent(8) | mantissa(7). Within one weight row the
exponents sit in a narrow band, so each weight is stored as
  * one byte   sign | mantissa            (wa [N, K]   uint8)
  * one nibble exponent - row_base        (we [N, K/2] uint8, low nibble first)
with a per-row base exponent (wb [N] uint8). Codes 0..14 are exponent offsets;
code 15 marks a weight outside the band: it contributes zero in the dense
stream and its exact bf16 value comes from a small per-row side list
(ec [N, W] int32 columns, ev [N, W] bf16 values, zero padded).

The kernel rebuilds the exact bf16 bit pattern, so the product uses the same
weights as the bf16 kernels: 25% fewer bytes streamed per decode step and no
change to the math.
"""

import torch
import triton
import triton.language as tl

from kernels.gemv import _CONFIGS, _prune, _bm, _DOT_F32

MAX_ESC = 16          # side-list width above which a matrix stays bf16


@torch.no_grad()
def pack(w, max_esc=MAX_ESC):
    """w [N, K] bf16 (K even) -> (wa, we, wb, ec, ev) or None if a row needs
    more than max_esc side entries."""
    N, K = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    exp = (bits >> 7) & 0xFF
    # best 15-wide exponent window per row: cumulative histogram over 256 bins
    hist = torch.zeros((N, 256 + 15), device=w.device, dtype=torch.int32)
    hist[:, :256].scatter_add_(1, exp.to(torch.int64), torch.ones_like(exp))
    csum = torch.cumsum(hist, 1)
    csum = torch.cat([torch.zeros((N, 1), device=w.device, dtype=csum.dtype), csum], 1)
    inside = csum[:, 15:15 + 256] - csum[:, 0:256]          # window [b, b+15)
    base = torch.argmax(inside, 1).to(torch.int32)           # [N]
    code = exp - base[:, None]
    esc = (code < 0) | (code > 14)
    n_esc = esc.sum(1)
    W = int(n_esc.max().item())
    if W > max_esc:
        return None
    W = max(2, triton.next_power_of_2(W))
    code = torch.where(esc, torch.full_like(code, 15), code)
    wa = (((bits >> 8) & 0x80) | (bits & 0x7F)).to(torch.uint8)
    we = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    ec = torch.zeros((N, W), device=w.device, dtype=torch.int32)
    ev = torch.zeros((N, W), device=w.device, dtype=torch.bfloat16)
    rows, cols = torch.nonzero(esc, as_tuple=True)
    if rows.numel():
        start = torch.cumsum(n_esc, 0) - n_esc
        slot = torch.arange(rows.numel(), device=w.device) - start[rows]
        ec[rows, slot] = cols.to(torch.int32)
        ev[rows, slot] = w[rows, cols]
    return wa.contiguous(), we.contiguous(), base.to(torch.uint8).contiguous(), ec, ev


@torch.no_grad()
def unpack(p):
    """Inverse of pack (for tests): exact bf16 [N, K]."""
    wa, we, wb, ec, ev = p
    N, K = wa.shape
    code = torch.stack([we & 15, we >> 4], -1).reshape(N, K).to(torch.int32)
    a = wa.to(torch.int32)
    bits = ((a & 0x80) << 8) | ((wb.to(torch.int32)[:, None] + code) << 7) | (a & 0x7F)
    bits = torch.where(code == 15, torch.zeros_like(bits), bits)
    w = bits.to(torch.int16).view(torch.bfloat16).clone()
    rows = torch.arange(N, device=wa.device)[:, None].expand_as(ec)
    nz = ev != 0
    w[rows[nz], ec[nz].to(torch.int64)] = ev[nz]
    return w


@triton.jit
def _decode(a, e, b, BN: tl.constexpr, BK: tl.constexpr):
    """a [BN, BK] u8, e [BN, BK/2] u8, b [BN] u8 -> bf16 [BN, BK]."""
    code = tl.reshape(tl.join(e & 15, e >> 4), [BN, BK]).to(tl.int32)
    ai = a.to(tl.int32)
    bits = ((ai & 0x80) << 8) | ((b.to(tl.int32)[:, None] + code) << 7) | (ai & 0x7F)
    bits = tl.where(code == 15, 0, bits)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "K_SPLIT", "W"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=5, rep=20)
@triton.jit
def _df12_kernel(x_ptr, wa_ptr, we_ptr, wb_ptr, ec_ptr, ev_ptr, out_ptr, M, N, K, K_SPLIT,
                 W: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 PARTIAL: tl.constexpr, DOT_F32: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    rows = offs_n.to(tl.int64)
    k0 = pid_k * K_SPLIT
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    b = tl.load(wb_ptr + offs_n)
    acc = tl.zeros([BM, BN], tl.float32)
    x_base = x_ptr + offs_m[:, None] * K
    for kk in range(0, K_SPLIT, BK):
        offs_k = k0 + kk + tl.arange(0, BK)
        offs_h = (k0 + kk) // 2 + tl.arange(0, BK // 2)
        x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
        a = tl.load(wa_ptr + rows[:, None] * K + offs_k[None, :])
        e = tl.load(we_ptr + rows[:, None] * (K // 2) + offs_h[None, :])
        w = _decode(a, e, b, BN, BK)
        if DOT_F32:
            acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
        else:
            acc += tl.dot(x, tl.trans(w))
    if pid_k == 0:
        offs_w = tl.arange(0, W)
        cols = tl.load(ec_ptr + rows[:, None] * W + offs_w[None, :])           # [BN, W]
        vals = tl.load(ev_ptr + rows[:, None] * W + offs_w[None, :]).to(tl.float32)
        xs = tl.load(x_ptr + offs_m[:, None, None] * K + cols[None, :, :],
                     mask=mmask[:, None, None], other=0.0).to(tl.float32)   # [BM, BN, W]
        acc += tl.sum(xs * vals[None, :, :], axis=2)
    if PARTIAL:
        dst = out_ptr + (pid_k * M + offs_m)[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc, mask=mmask[:, None])
    else:
        dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc.to(tl.bfloat16), mask=mmask[:, None])


def gemv_df12(x, p, split=1):
    """Same contract as kernels.gemv.gemv with packed weights p."""
    wa, we, wb, ec, ev = p
    M, K = x.shape
    N = wa.shape[0]
    assert K % split == 0
    if split == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        out = torch.empty((split, M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (N // meta["BN"], split)
    _df12_kernel[grid](x, wa, we, wb, ec, ev, out, M, N, K, K // split, W=ec.shape[1],
                       BM=_bm(M), PARTIAL=split > 1, DOT_F32=_DOT_F32)
    return out
