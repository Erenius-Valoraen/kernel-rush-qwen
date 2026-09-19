"""INT8 weight-only decode GEMVs (per-output-row scales), same tiling and
output contracts as kernels.gemv: bf16 [M, N], or fp32 split-K partials."""

import torch
import triton
import triton.language as tl

from kernels.gemv import _CONFIGS, _prune, _bm


@torch.no_grad()
def quantize_rows(w):
    """w [N, K] bf16 -> (w8 int8 [N, K], scale fp32 [N]), w ~= w8 * scale[:, None]."""
    wf = w.float()
    scale = wf.abs().amax(1).clamp_min(1e-12) / 127.0
    w8 = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return w8.contiguous(), scale.contiguous()


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "K_SPLIT"],
                 prune_configs_by={"early_config_prune": _prune},
                 warmup=5, rep=20)
@triton.jit
def _gemv_i8_kernel(x_ptr, w_ptr, s_ptr, out_ptr, M, N, K, K_SPLIT,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    PARTIAL: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    k0 = pid_k * K_SPLIT
    w_base = w_ptr + offs_n[:, None].to(tl.int64) * K
    offs_m = tl.arange(0, BM)
    mmask = offs_m < M
    acc = tl.zeros([BM, BN], tl.float32)
    x_base = x_ptr + offs_m[:, None] * K
    for kk in range(0, K_SPLIT, BK):
        offs_k = k0 + kk + tl.arange(0, BK)
        x = tl.load(x_base + offs_k[None, :], mask=mmask[:, None], other=0.0)
        w = tl.load(w_base + offs_k[None, :]).to(tl.float32).to(tl.bfloat16)
        acc += tl.dot(x, tl.trans(w))
    acc = acc * tl.load(s_ptr + offs_n)[None, :]
    if PARTIAL:
        dst = out_ptr + (pid_k * M + offs_m)[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc, mask=mmask[:, None])
    else:
        dst = out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
        tl.store(dst, acc.to(tl.bfloat16), mask=mmask[:, None])


def gemv_i8(x, q, split=1):
    """Same contract as kernels.gemv.gemv; q = quantize_rows(W)."""
    w8, s = q
    M, K = x.shape
    N = w8.shape[0]
    assert K % split == 0
    if split == 1:
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        out = torch.empty((split, M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (N // meta["BN"], split)
    _gemv_i8_kernel[grid](x, w8, s, out, M, N, K, K // split, BM=_bm(M), PARTIAL=split > 1)
    return out
