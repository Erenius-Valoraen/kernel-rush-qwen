# Qwen3 4B implementation guide

This guide connects the baseline in `engine/engine.py` to the operations you
would replace in an optimized engine. The engine contract (included in the
full starter and available on the event's
Docs page) defines correctness, timing, and packaging.

Design against the pinned checkpoint and Transformers 4.51.3, not a newer
`main` branch. To inspect the exact installed implementation in the benchmark
runtime:

```python
import inspect
from transformers.models.qwen3 import modeling_qwen3

print(inspect.getsourcefile(modeling_qwen3.Qwen3ForCausalLM))
print(inspect.getsource(modeling_qwen3.Qwen3Attention.forward))
```

## Architecture

The checkpoint is a dense, decoder-only Transformer. Dimensions come from the
[pinned configuration](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/config.json);
operation order follows [Transformers 4.51.3](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen3/modeling_qwen3.py).

| Property | Value |
| --- | ---: |
| Decoder layers | 36 |
| Hidden width (`H`) | 2,560 |
| MLP width (`I`) | 9,728 |
| Vocabulary (`V`) | 151,936 |
| Query heads (`Nq`) | 32 |
| Key/value heads (`Nkv`) | 8 |
| Head width (`D`) | 128 |
| Query heads per KV head | 4 |
| Activation | SwiGLU (`silu(gate) * up`) |
| Normalization | RMSNorm, epsilon `1e-6` |
| Position encoding | RoPE, theta `5,000,000` |
| Sliding window | none |
| Linear biases | none |
| Weight dtype | BF16 |
| Input/output embedding | tied |

`H` is not `Nq * D`: the query projection expands 2,560 hidden values to
4,096 query values, and the attention output projection maps 4,096 back to
2,560.

## Execution graph

Let `B` be batch size, `T` the tokens processed by this forward, and `L` the
tokens already cached. A forward passes through:

```text
IDs [B,T] -> embedding -> 36 decoder layers -> final RMSNorm
          -> last position -> tied LM head -> logits [B,1,V] -> argmax
```

Each decoder layer has two residual branches:

```text
n = input_layernorm(x)
q, k, v = projections(n)
q, k = per_head_rmsnorm(q), per_head_rmsnorm(k)
q, k = rope(q, k, absolute_positions)
k, v = update_layer_cache(k, v)
a = x + o_proj(grouped_query_attention(q, k, v))
m = post_attention_layernorm(a)
y = a + down_proj(silu(gate_proj(m)) * up_proj(m))
```

On the first baseline iteration, `T` is the prompt length and the cache is
empty. This prefill processes every prompt token, but `logits_to_keep=1` sends
only the last position through the LM head. Every later decode iteration has
`T=1`; projections and the MLP process one token while attention reads the
complete cache.

## Tensor shapes

These are logical shapes; a transpose may produce non-contiguous strides.
Activations are BF16 unless noted. RMSNorm reductions and RoPE angle
construction use FP32 internally.

| Operation or value | Shape |
| --- | --- |
| Input IDs | `[B, T]`, int64 |
| Hidden state | `[B, T, 2560]` |
| Q before heads | `[B, T, 4096]` |
| K and V before heads | `[B, T, 1024]` each |
| Q after reshape/transpose | `[B, 32, T, 128]` |
| New K and V | `[B, 8, T, 128]` each |
| RoPE cosine and sine | `[1, T, 128]` |
| Per-layer K/V cache after update | `[B, 8, L+T, 128]` each |
| Logical attention result | `[B, 32, T, 128]` |
| Before/after attention output projection | `[B,T,4096]` / `[B,T,2560]` |
| MLP gate, up, and product | `[B, T, 9728]` each |
| MLP output | `[B, T, 2560]` |
| Logits with `logits_to_keep=1` | `[B, 1, 151936]` |
| Greedy token | `[B, 1]`, int64 |

Query head `h` uses KV head `h // 4`. Transformers 4.51.3 expands the eight
K/V heads to 32 before PyTorch SDPA; a custom grouped-query attention kernel
can apply the mapping directly.

The complete BF16 cache payload is:

```text
36 * 2 (K,V) * B * sequence_length * 8 * 128 * 2 bytes
= 147,456 * B * sequence_length bytes
```

That is 144 KiB per cached token per sequence, before allocator overhead and
attention temporaries.

## Modules and weights

After loading, define `base = self.model.model` and `layer = base.layers[i]`.
PyTorch linear weights use `[out_features, in_features]`: `linear(x) = x @ W.T`.

| Object | Weight shape |
| --- | --- |
| `base.embed_tokens` | `[151936, 2560]` |
| `layer.input_layernorm` | `[2560]` |
| `layer.self_attn.q_proj` | `[4096, 2560]` |
| `layer.self_attn.k_proj`, `layer.self_attn.v_proj` | `[1024, 2560]` each |
| `layer.self_attn.q_norm`, `layer.self_attn.k_norm` | `[128]` each |
| `layer.self_attn.o_proj` | `[2560, 4096]` |
| `layer.post_attention_layernorm` | `[2560]` |
| `layer.mlp.gate_proj`, `layer.mlp.up_proj` | `[9728, 2560]` each |
| `layer.mlp.down_proj` | `[2560, 9728]` |
| `base.norm` | `[2560]` |
| `self.model.lm_head` | `[151936, 2560]`, tied to embedding |

Use the loaded modules instead of assuming which safetensors shard holds a
name. This preserves tied weights.

## How the platform reaches your kernels

The platform calls your Python engine. Your engine imports and launches the
kernels; files in `kernels/` are not discovered or installed automatically.

```text
connected repository's engine/ contents -> extracted submission directory
  -> import engine.Engine -> Engine(model_path)
  -> generate(warmup_ids, max_new_tokens)
  -> generate(sample_ids, max_new_tokens), repeated for measured samples
       -> your Python wrappers -> your GPU kernel launches
       -> yield one list of B token IDs per output step
```

Each workload gets a fresh engine process. Its engine instance survives warmup
and that workload's samples, so weights and compiled kernels can be reused.
Reset prompt-dependent state at the start of every generation. The runner
streams your yielded lists to the timing process; it does not call individual
norm, attention, or MLP kernels. Correctness and speed are judged on complete
generation, including prefill and the time until tokens reach that process.

Only `engine.py` exporting `Engine.__init__(model_path)` and
`Engine.generate(input_ids, max_new_tokens)` is a platform interface. There is
no kernel registry, required kernel function name, manifest, or
`build_kernel` hook. You may keep Transformers, replace selected modules, or
implement the entire forward yourself. The reference model is independent;
changing your model does not change the judge.

### Source format and responsibilities

Use Python modules for host code and Triton `@triton.jit` functions in `.py`
files for custom GPU code. A typical repository layout is:

```text
engine/
  engine.py                Engine and generation loop
  kernels/
    __init__.py
    rmsnorm.py             Python wrapper + Triton device function
    attention.py
    decode.py              optional fused decode implementation
```

Inside `engine.py`, import with `from kernels.rmsnorm import rms_norm`: the
contents of `engine/` become the submission root on `sys.path`.
The installed runtime is PyTorch 2.5.1, Triton 3.1.0, Transformers 4.51.3,
and CUDA 12.4 on one H100. Imports alone do not replace a model operation;
install an adapter or call the wrapper from your own forward path.

Ship source, not precompiled `.so`, cubin, or PTX artifacts. Standalone
`.cu`/`.cuh` files are outside the archive allowlist. A requirements file
does not install dependencies, and the execution sandbox has no network.
Triton compiles device code at runtime; launch each specialization during
warmup to avoid first-use compilation in measured samples. Check APIs against
Triton 3.1.0 before using examples written for newer releases.

The engine must load weights from `model_path`, allocate activations and cache,
supply positions and masks, launch kernels in dependency order, select tokens,
and yield host integer lists. If you use extra CUDA streams, arrange their
dependencies before consuming or yielding results.

## Map model operations to kernels

A model module and a GPU kernel are different boundaries: one module may launch
many kernels, and one fused kernel may implement several operations. Let
`M = B * T` when flattening batch and token dimensions for matrix operations.
The following are possible designs, not required platform signatures.

| Target | Kernel inputs → outputs | Initial GPU work mapping |
| --- | --- | --- |
| Hidden RMSNorm | `x[M,2560], weight[2560], eps → y[M,2560]` | One program per row; reduce over hidden width. |
| Q/K head norm + RoPE | Q `[B,T,32,128]`, K `[B,T,8,128]`, gains, positions → normalized and rotated Q/K | One program per token/head; optionally write K directly to its cache slot. |
| Q/K/V projections | `x[M,2560]`, three weights → `q[M,4096], k/v[M,1024]` | Tiles over rows and output channels; optionally pack weights for one combined projection. |
| Attention | Q `[B,32,T,128]`, cached K/V `[B,8,L+T,128]`, valid length/mask → `[B,T,4096]` | Tiles over batch, query heads, and query positions; reduce across visible keys with softmax. |
| Attention output + residual | attention `[M,4096]`, weight `[2560,4096]`, residual `[M,2560] → [M,2560]` | Matrix tiles with a residual epilogue; preserve the reference rounding boundaries. |
| SwiGLU activation | gate and up `[M,9728] → product[M,9728]` | Elementwise tiles; fuse SiLU and multiplication. Projection fusion is a separate choice. |
| MLP projections | `[M,2560] → gate/up[M,9728] → [M,2560]` | Matrix tiles for gate/up and down; activation fusion must preserve BF16 casts. |
| LM head + token selection | final hidden `[B,2560]`, tied weight `[151936,2560] → IDs[B]` | Vocabulary tiles and a global argmax reduction; preserve lowest-index selection on exact ties. |

Prefill has many rows; decode has only `B` rows. Start with separate dispatch
for these two cases and measure both. A single tiling strategy need not serve
both well.

### Define your own kernel interface

For each wrapper, document tensor shape, dtype, strides, device, output
ownership, and mutation. For example, an allocation-free norm wrapper could be
`rms_norm_out(x, weight, out, eps)`: contiguous BF16 CUDA tensors, normalization
over the final dimension, caller-owned output of the same shape, no input/output
aliasing. Inside, compute the launch grid and call
`_kernel[grid](...)`. The bundled `rms_norm` instead allocates and returns an
output; adapt it if you need persistent graph buffers.

For attention, also specify cache layout, capacity versus valid length, absolute
positions, and which K/V slots are written. Either accept actual strides or
require a layout and convert explicitly. Account for conversion costs in your
timing.

When replacing a Transformers module, preserve that module's Python call
interface. In 4.51.3, a norm or MLP returns a tensor; `self_attn.forward`
returns `(attention_output, attention_weights)`, with `None` for unused
weights on the baseline SDPA path. Replacing attention also transfers
responsibility for its cache updates. In a fully custom engine, your internal
signatures are yours to choose.

## Are megakernels possible?

Yes, the submission contract permits fused blocks, a whole-layer kernel, or a
custom kernel implementing one complete decode step. The platform imposes no
kernel-count or model-module boundary. A megakernel still has to fit the pinned
source-only runtime, produce valid tokens, and pass the same timing and memory
gates. There is no platform-provided megakernel scheduler.

The difficult part is synchronization across dependent stages. A whole-model
step needs reductions, matrix products, attention, residuals, all 36 layers,
and vocabulary selection. Ordinary GPU blocks cannot assume other blocks are
resident or use a block-local barrier as a grid-wide barrier. A spin-wait on
work assigned to an unscheduled block can deadlock. See NVIDIA's
[block execution model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
and [synchronization guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html);
these describe the constraints, not an extra launch capability supplied here.

Design any persistent schedule around synchronization mechanisms actually
available in this runtime, register/shared-memory limits, and occupancy.
Fusing more work can reduce launches yet increase spills or reduce matrix
throughput. Separate launches ordered on one stream provide dependency
boundaries; CUDA graph replay can reduce their host overhead without requiring
one device kernel.

Start with a fused operation or layer, then a graph of a single-token decode
step. A full-step megakernel is a further option once profiling shows launch
or intermediate-memory costs justify it. Keep a separate prefill path if its
larger matrix operations benefit from different kernels.

A megakernel does not change the streaming interface: `generate` still yields
one host list per output step. Computing the entire continuation before the
first yield delays TTFT by the whole generation and can fail its latency gate.
A persistent kernel spanning multiple tokens would need a working incremental
device-to-host handoff; returning a final GPU token buffer alone is insufficient.
Stop all work for a sample before it finishes, and reuse no prompt content in
the next call.

## How to replace Transformers operations

### Replace one leaf module first

Keep Transformers' positions, cache, and attention dispatch while replacing a
leaf. Put this adapter above `Engine` in `engine/engine.py`:

```python
import torch
from kernels.rmsnorm import rms_norm


class FusedRMSNorm(torch.nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.variance_epsilon = reference.variance_epsilon

    def forward(self, x):
        return rms_norm(x, self.weight, self.variance_epsilon)
```

After loading the model in `Engine.__init__`, replace its norms:

```python
base = self.model.model
base.norm = FusedRMSNorm(base.norm)
for layer in base.layers:
    layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
    layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
    layer.self_attn.q_norm = FusedRMSNorm(layer.self_attn.q_norm)
    layer.self_attn.k_norm = FusedRMSNorm(layer.self_attn.k_norm)
```

The adapter reuses each learned weight and epsilon. Start with one norm to
isolate failures, then expand to the others. Benchmark the result: this is an
integration example, with no measured speedup guarantee.

### Bypass the top-level wrapper

After leaf replacements match baseline, remove output-object and generic model
dispatch while retaining the loaded layers. Put this helper above `Engine`;
it assumes the baseline's `.eval()` model and `attn_implementation="sdpa"`:

```python
from transformers import DynamicCache


@torch.inference_mode()
def qwen_forward(model, input_ids, cache, first_position):
    base = model.model
    x = base.embed_tokens(input_ids)
    length = input_ids.shape[1]
    cache_position = torch.arange(
        first_position, first_position + length, device=input_ids.device
    )
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = base.rotary_emb(x, position_ids)

    for layer in base.layers:
        x = layer(
            x,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]

    x = base.norm(x)
    return model.lm_head(x[:, -1:, :]), cache
```

Use it inside `Engine.generate`:

```python
current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
cache = DynamicCache()  # Fresh for every call, including after warmup.
position = 0
for _ in range(max_new_tokens):
    logits, cache = qwen_forward(self.model, current, cache, position)
    position += current.shape[1]
    current = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    yield current[:, 0].tolist()
```

This example supports a full, unpadded prefill into an empty cache, followed
by single-token decode. The first call consumes `S` prompt tokens and emits
the first output; the next consumes that output at absolute position `S`.
The final emitted token does not need another forward.

**Do not reuse `attention_mask=None` for chunked prefill, multi-token
verification, or a static cache.** The SDPA adapter infers causality from query
length, which is insufficient for those cases. Supply a mask that permits
key position `j` only when `j <= L + query_index` and the slot is initialized.

Python layer dispatch, dynamic-cache growth, and the per-step GPU-to-host
transfer in `.tolist()` remain. Measure whether removing the wrapper helps.

### Replace the cache and capture decode

Preallocate per-layer K and V as
`[B, 8, prompt_length + max_new_tokens, 128]`. Write new values at
`[:, :, L:L+T, :]` and expose only `[:, :, :L+T, :]` to attention. Prefill must
remain causal; decode must never read unused capacity.

Allocate shape-dependent buffers during warmup, when batch and prompt length
are known. Reset logical length and positions before every `generate` call;
warmup and measured prompts must never share cached content.

For CUDA graphs, a growing Python slice such as `:L+T` still changes the
attention shape. Use fixed-capacity tensors with a changing mask or a custom
kernel that reads a device-side length. Keep graph input/output addresses
stable, update token IDs and positions in place, and keep `.tolist()` and
`yield` outside capture.

### Replace full blocks last

Preserve these semantics:

- Q and K are RMS-normalized per 128-value head before RoPE; V is not.
- RoPE uses absolute cache position and theta `5,000,000`.
- attention scale is `1 / sqrt(128)` and query head `h` uses KV head `h // 4`.
- MLP is `down(silu(gate(x)) * up(x))`.
- both residual additions remain outside their pre-norm branches.
- projections have no biases and the embedding/LM-head weight remains tied.
- generation is argmax for exactly `max_new_tokens`; EOS is an ordinary ID.

Fusing operations may reorder arithmetic, but it must not change the formula.
The RMSNorm example shows a critical cast boundary: reduce and normalize in
FP32, cast the normalized value to the input dtype, then multiply by the
learned weight.

## Check each change

Compare against an untouched baseline under inference mode: prefill logits,
several cached decode steps, and two consecutive `generate` calls with different
prompts. Test all public shapes and inspect TTFT, TPOT, throughput, and memory.
A local comparison helps locate errors; the platform's teacher-forced replay
on your emitted prefix is the correctness judge.

The examples here are implementation scaffolds. They are not H100 performance
results or a guarantee that a custom kernel passes the numerical tolerance.
