"""Qwen3's RMSNorm in Triton, written to match the reference exactly.

Nothing imports this. It is here to demonstrate the two things the baseline
never shows: how a module beside ``engine.py`` is vendored and imported, and how
closely a fused kernel has to follow the reference's arithmetic to stay inside
the tie margin.

To use it, swap it in for the ``Qwen3RMSNorm`` modules on the loaded model in
``Engine.__init__``. Delete this package if you would rather start clean.
"""

import torch
import triton
import triton.language as tl

#: One row must fit in one block. Qwen3 4B norms 2560 columns (hidden) and 128
#: (per-head q/k norm), so both land well inside this.
MAX_BLOCK = 8192


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, row_stride, n_cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    offsets = row * row_stride + cols

    # The reference reduces in fp32 over the whole row. Masked lanes load as
    # zero, so they contribute nothing to the sum of squares.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / n_cols
    normed = x * tl.math.rsqrt(variance + eps)

    # Cast placement, and the whole reason this file exists. The reference ends:
    #
    #     return self.weight * hidden_states.to(input_dtype)
    #
    # so the normalised value is rounded to bfloat16 *before* the weight
    # multiply, not after. Keeping the product in fp32 and rounding once at the
    # end is the obvious version, is strictly more accurate, and is wrong: it
    # computes a different function, and on some prompt it moves a logit further
    # than the 2.0 tie margin allows. Reorder arithmetic freely; do not
    # reformulate it.
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, normed.to(y_ptr.dtype.element_ty) * weight, mask=mask)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the last dimension, matching ``Qwen3RMSNorm.forward``.

    ``x`` is any shape whose last dimension matches ``weight``; ``weight`` is
    the module's learned gain, in the same dtype as ``x``.
    """
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).contiguous()
    n_rows, n_cols = rows.shape
    block = triton.next_power_of_2(n_cols)
    if block > MAX_BLOCK:
        raise ValueError(f"a row must fit in one block; {n_cols} columns does not")
    out = torch.empty_like(rows)
    _rms_norm_kernel[(n_rows,)](
        rows,
        weight,
        out,
        rows.stride(0),
        n_cols,
        eps,
        BLOCK=block,
        num_warps=max(4, min(16, block // 256)),
    )
    return out.reshape(shape)
