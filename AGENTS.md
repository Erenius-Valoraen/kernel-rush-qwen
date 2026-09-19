# Working in this repository

Read the engine contract once, in full, before your first edit — the **Docs**
page at https://htn.dryft.ai/docs, or the included
`QWEN_ENGINE_CONTRACT.md`.
This file is the part you must not get wrong.

Then read `OPTIMIZATION_GUIDE.md` before replacing model operations. It maps
the pinned Transformers 4.51.3 implementation to the exact Qwen3 dimensions,
tensor shapes, execution graph, cache semantics, and module boundaries.

## Invariants

Break one of these and the run fails outright; no amount of speed compensates.

1. **`engine.py` exports `class Engine`, at the archive root.** Two methods,
   `__init__(self, model_path)` and `generate(self, input_ids, max_new_tokens)`.
   No manifest, no config, no other entry point.
2. **`generate` yields exactly `max_new_tokens` times.** One list per step, one
   token id per sequence, in the order the sequences arrived. Never stop early
   on an end-of-sequence token; it is an ordinary id here.
3. **Every token must be native Qwen's greedy choice on your own prefix**, or
   within 2.0 logits of the argmax there. The judge replays your sequence
   teacher-forced after your engine is dead. One bad position fails the
   workload.
4. **Nothing leaves the container.** No network, no downloads, no pip. Vendor
   pure-Python or Triton source in `engine/`.
5. **`engine/` holds only what the engine imports.** Agent code, notes,
   experiment logs and results belong outside it.

## The gates

A workload fails, and takes the run with it, on any of:

| Gate | Limit |
| --- | --- |
| Wrong token | any position outside the 2.0 tie margin |
| Time to first token | above 1.10× native's median |
| Time per output token | above 1.10× native's median |
| Peak GPU memory | above 90% of the device |
| Sample spread | above 25% across the five official samples |
| Load plus one warmup | above 300 seconds |
| One sample | above 300 seconds |

Two of those bite optimizations that look like wins. A change that trades
latency for throughput can pass on tokens per second and still fail the TTFT or
TPOT gate. A change whose cost varies with the prompt — a cache that sometimes
recompiles, a heuristic that sometimes takes a slow path — can pass a
single-sample public run and fail the 25% spread gate on the official five.

## What is measured

Six workloads, each a fixed batch, prompt length and output length. Only the
hidden three are scored; the public three are feedback and never rank.

| Workload | batch | prompt | output | scored |
| --- | ---: | ---: | ---: | --- |
| public-0 | 1 | 512 | 32 | no |
| public-1 | 4 | 2048 | 32 | no |
| public-2 | 16 | 512 | 128 | no |

Use public runs to measure improvements. Hidden workloads determine the official
score, so avoid assumptions about their shapes.

Each workload gets a fresh process: load, one warmup generation of the same
shape, then the samples. Loading and warmup share a 300-second budget and are
untimed. Load and relayout weights in `__init__`; allocate shape-dependent
buffers, autotune, and capture graphs during warmup, when shapes are known.
Reset cache state for each subsequent `generate` call.

## Where the wins are

In rough order of payoff against these shapes:

- **Per-step overhead.** The baseline pays full Python and Transformers
  dispatch on every decode step. CUDA graphs over the decode step, or a hand-
  rolled step that skips the `Qwen3ForCausalLM` wrapper, attack the largest
  cost at batch 1 and 4.
- **Prefill.** Outputs are 32 to 128 tokens against prompts of 512 to 2048, so
  prefill is a large share of total time. Chunking it, or overlapping it with
  the first decode steps, moves the total and the TTFT together.
- **KV cache layout.** The baseline uses whatever `past_key_values` Transformers
  hands back. A preallocated, correctly strided cache removes per-step
  allocation and concatenation.
- **Fused kernels.** RMSNorm, rotary embedding and the attention epilogue are
  small, repeated, and safe to fuse. `engine/kernels/rmsnorm.py` is a worked
  example.
- **Speculative decoding with exact verification.** Legal and passes rule 3 by
  construction, because verification reproduces the same argmax. Expensive to
  get right; leave it until the cheap wins are taken.

Do not reach for quantization, cache eviction, approximate or sparse attention,
or a smaller draft model used without verification. All of them shift logits by
whole units and fail rule 3.

## Numerics

The tie margin is 2.0 logits, and it was calibrated on native Qwen against
itself: replayed teacher-forced, native's own tokens sit up to 0.75 logits below
the replay's argmax at a handful of positions in every twelve thousand, purely
because the cached decode path and one full forward accumulate BF16 sums in
different orders. That is the size of the budget you are spending when you
reorder arithmetic.

So reorderings are affordable and reformulations are not. Changing the order of
a reduction, or accumulating in fp32 where the reference accumulated in fp32, is
within budget. Changing *where* the cast to BF16 happens is a reformulation and
can move a logit by more than the noise floor — see the comment in
`engine/kernels/rmsnorm.py`, which matches the reference's cast placement exactly and
explains why the obvious version does not.

## Workflow

1. Change `engine/engine.py`, or add a module beside it.
2. Package and submit a **public** run. It is one sample per workload, about
   two minutes, and it never touches the leaderboard.
3. Read the per-workload report: tokens per second, TTFT, TPOT, peak memory,
   and the native ratios. A public run reports the latency ratios instead of
   enforcing them, so check them yourself before an official run.
4. When public results are clean and the ratios have headroom, request an
   **official** run: five samples per workload, the hidden shapes scored, the
   spread gate live.

Use `bin/dryft` to package, submit, and wait on a run. `agent/loop.py` is
where you record what each edit did and decide the next change; the hidden
scores are the only ones that count, and they move for reasons the public
three will not always show you.

## Failure codes

`incorrect_output`, `candidate_error`, `timeout`, `latency_limit`,
`memory_limit` and `unstable_timing` are your engine's fault. `harness_error`
and infrastructure codes are the platform's — retry those rather than rewriting
anything. The log page carries a bounded tail of your engine's own stdout and
stderr, so log what you need to diagnose a failure from it alone.
