"""FP8 (e4m3) prefill GEMMs through cuBLASLt (torch._scaled_mm).

Weights get one scale per matrix. Prefill activations use one dynamic scale per call (their own range, so
nothing is ever clamped); one fused kernel scales and casts. Decode stays on the bf16 kernels.
"""

import os

import torch
import triton
import triton.language as tl

FP8_MAX = 448.0
HEADROOM = float(os.environ.get("ENGINE_FP8_HEADROOM", "4"))


@torch.no_grad()
def quantize_weight(w):
    """w [N, K] bf16 -> (w8 [N, K] float8_e4m3fn, scale fp32 scalar), w ~= w8 * scale."""
    wf = w.float()
    scale = (wf.abs().max().clamp_min(1e-12) / FP8_MAX).reshape(())
    w8 = (wf / scale).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return w8.contiguous(), scale


class ActScale:
    """Activation scale for one call: x8 = clamp(x * inv), x ~= x8 * scale.
    Computed on device from this tensor's own range (no sync, graph-safe)."""

    def __init__(self, x, headroom=1.0):
        mn, mx = torch.aminmax(x)
        amax = torch.maximum(-mn, mx).float().clamp_min(1e-6) * headroom
        self.amax = amax
        self.inv = (FP8_MAX / amax).reshape(1)             # fp32, on device
        self.scale = (amax / FP8_MAX).reshape(())


@triton.jit
def _cast_kernel(x_ptr, y_ptr, inv_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    inv = tl.load(inv_ptr)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv
    x = tl.minimum(tl.maximum(x, -448.0), 448.0)
    tl.store(y_ptr + offs, x.to(tl.float8e4nv), mask=mask)


def _cast_triton(x, a):
    y = torch.empty(x.shape, device=x.device, dtype=torch.float8_e4m3fn)
    n = x.numel()
    _cast_kernel[(triton.cdiv(n, 4096),)](x, y, a.inv, n, BLOCK=4096, num_warps=4)
    return y


def _cast_torch(x, a):
    return (x.float() * a.inv).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


_cast = None


def pick_cast(log):
    """Validate the fused cast against torch on this GPU and keep the faster one."""
    global _cast
    _cast = _cast_torch
    try:
        x = torch.randn((4096, 2560), device="cuda", dtype=torch.bfloat16) * 3
        a = ActScale(x)
        ref = _cast_torch(x, a).float()
        got = _cast_triton(x, a).float()
        bad = (got - ref).abs().max().item()
        ts = {}
        for name, fn in (("torch", _cast_torch), ("triton", _cast_triton)):
            fn(x, a)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(5):
                fn(x, a)
            e1.record()
            e1.synchronize()
            ts[name] = e0.elapsed_time(e1) / 5
        # one e4m3 step at the top of the range is 32: allow rounding-mode ties only
        if bad <= 32.0 and torch.isfinite(got).all() and ts["triton"] < ts["torch"]:
            _cast = _cast_triton
        log(f"fp8 cast: {ts} maxdiff={bad:.3g} -> {_cast.__name__}")
    except Exception as e:  # pragma: no cover
        log(f"fp8 triton cast unavailable: {e!r}")


class SharedScale:
    """Scale slots living in caller-owned device tensors (read by decode graphs)."""

    def __init__(self, inv, scale):
        self.inv, self.scale = inv, scale


def linear_fp8(x, q, a=None, amax_out=None):
    """bf16 [M, N] ~= x @ W^T with q = quantize_weight(W); a = given scale, or
    None to scale by this tensor's own range (running max kept in amax_out)."""
    w8, ws = q
    if a is None:
        a = ActScale(x)
        if amax_out is not None:
            amax_out.copy_(torch.maximum(amax_out, a.amax.reshape(1)))
    return torch._scaled_mm(_cast(x, a), w8.t(), scale_a=a.scale, scale_b=ws,
                            out_dtype=torch.bfloat16)
