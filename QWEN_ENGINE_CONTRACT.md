# Qwen3 4B decode benchmark

The platform runs one benchmark: **Decode Qwen3 4B faster.** Participants
submit an inference engine for the pinned model, `Qwen/Qwen3-4B-Instruct-2507`
at revision `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16 weights, one H100.
The engine owns the whole generation loop. The judge owns the prompts, the
clock, the reference, and the rule that the output must be what native Qwen
would have produced under greedy decoding.

There is one benchmark and one leaderboard, and no versions of either. When
its rules change, the definition is updated in place, earlier results are
invalidated, and teams run again.

## What you submit

A tar.gz whose root holds `engine.py`, plus any Python or Triton source you
need. No weights, no credentials, no compiled binaries, no Docker images, no
research agent. The archive is limited to 2 MiB compressed and 200 files.

There is no manifest. The format is fixed and nothing in it was ever yours to
choose: one benchmark, one runtime, and one entry point. The archive root holds
`engine.py`, and `engine.py` exports a class with two methods:

```python
class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yield a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence in input_ids has the
        same length. Do not stop at end-of-sequence tokens.
        """
```

The runtime is the platform's: Python 3.11, CUDA 12.4, PyTorch 2.5.1, Triton
3.1.0, Transformers 4.51.3, safetensors 0.5.3, tokenizers 0.21.1. There is no
network, so there is nothing to install; vendor pure-Python or Triton source
in the archive. The archive root is on `sys.path`, so they are imported by
module name. `model_path` is a directory holding the checkpoint exactly as
published by Hugging Face. Your engine may load it into any layout it likes,
capture CUDA graphs, precompute anything that does not depend on prompts,
and warm up. Loading plus one warmup generation must finish within 300
seconds and the engine must stay under 90% of the GPU's memory.

The starter repository's `engine/engine.py` is native Qwen: it loads the model
with Transformers and runs a plain greedy loop with a KV cache. Submitted
unchanged, it gives you a starting throughput measurement. Improve it with KV
cache layout and paging, CUDA graphs for decode steps, fused kernels, chunked
or overlapped prefill, speculative decoding with exact verification, or
custom Triton for any block. What you may not change is the answer.

## Integrating custom kernels

The platform imports `engine.Engine`, constructs it with `model_path`, and
calls `generate` for warmup and measured samples. Your engine imports and
launches its kernels; the platform does not discover kernel files, register
operation replacements, or supply a kernel-level calling convention.

You may replace individual Transformers modules, use fused Triton kernels,
or implement an entire decode step as a megakernel. The same source-only
runtime, output stream, correctness rule, and resource gates apply.
Your engine owns tensor layouts, buffers, cache updates, synchronization,
and conversion to the token lists it yields.

See the [implementation guide](OPTIMIZATION_GUIDE.md#how-the-platform-reaches-your-kernels)
for the loading sequence, source format, operation-to-kernel mapping,
wrapper interfaces, and megakernel constraints.

## What is measured

Public workloads use these fixed batches, prompt lengths and output lengths.
Hidden workloads determine the official score; their shapes are not published:

| Workload | batch | prompt tokens | output tokens | visible |
| --- | ---: | ---: | ---: | --- |
| public-0 | 1 | 512 | 32 | yes |
| public-1 | 4 | 2048 | 32 | yes |
| public-2 | 16 | 512 | 128 | yes |

Prompts are token ids the judge derives from a fixed corpus with a fresh
random seed for every sample, drawn after your engine has loaded. Your engine
never sees text and never sees the same prompt twice.

For each workload the judge starts your engine in a fresh process, lets it
load and warm up on a prompt of the same shape, then runs the workload's
samples: five for an official run, one for a public run. The clock lives in
a separate trusted process that writes the prompt ids to your engine and
reads the token stream back. Time to first token is when the first step
arrives; time per output token is the rest of the stream divided by the
remaining steps; total time is the whole stream. Tokens per second is
`batch × output tokens / total time`. Native Qwen is measured by exactly the
same clock through exactly the same pipe, in the same container, so the
per-step overhead of the protocol is paid equally by both sides.

Per workload the report carries the median of your samples, their spread,
time to first token, time per output token, tokens per second, and peak GPU
memory. The score is the geometric mean of output tokens per second across
the three hidden workloads, with equal weight for each. Higher is faster.
Throughput includes prefill: `batch × output tokens / median generation seconds`.
Native timings set latency gates, not the score. Public workloads never rank.

## The quality rule

Your tokens must be the greedy choice native Qwen makes at every step, given
the tokens you produced so far. After your samples finish, the judge replays
your token sequence through native Qwen, teacher-forced, and checks each
output position: the token you emitted must be the argmax of the native
logits there, or within `tieMarginLogits` (currently 2.0) of the argmax. The
tie margin exists because BF16 arithmetic in a different order can flip a
genuine near-tie; it does not admit approximations. It was calibrated on
native Qwen itself: replayed teacher-forced, native's own greedy tokens sit
up to 0.75 logits below the replay's argmax at a handful of positions in
every twelve thousand, because the cached decode path and one full forward
accumulate BF16 sums in different orders. A quantized model, a pruned
cache, or an approximate attention shifts logits by whole units on some
prompt, and a single failing position fails the workload.

Because the replay follows your own prefix, a near-tie flip does not cascade:
later positions are judged on what you actually generated. A speculative
decoder with exact verification passes by construction.

## Gates and failure

A workload fails, and with it the run, on any of:

- a token that is neither the native argmax nor within the tie margin;
- time to first token or time per output token above 1.10 times native's,
  on the medians (official runs only; a public run reports the ratios);
- peak GPU memory above 90% of the device;
- more than 25% spread across the five samples;
- an engine that fails to load, exceeds its load budget, crashes, emits the
  wrong number of steps or tokens, or exceeds 300 seconds on one sample.

Failure codes distinguish your engine's faults (`incorrect_output`,
`candidate_error`, `timeout`, `latency_limit`, `memory_limit`,
`unstable_timing`) from the platform's (`harness_error`, infrastructure), and
the log page shows your engine's stdout and stderr, bounded.

## Trust boundary

Your engine runs inside the timing container, as an unprivileged user, with
no network, no secrets, and read-only access to the harness and the
checkpoint. The trusted client that feeds it prompts and reads its tokens
runs as root in the same container, so your process cannot stop, trace, or
patch it, and cannot alter the harness or the interpreter it runs on. The
native reference is a separate trusted process that is loaded and warmed
before your engine starts and is timed only after your engine has been
killed and confirmed gone, so nothing you leave running can slow it. The
replay that judges your tokens happens in that same reference process after
your engine is gone. Each workload gets a fresh engine process; each
submission gets a fresh container, destroyed with the run.

The judge does not lock clocks or normalize power. It pairs every workload's
native measurement with yours in the same container, alternating which side
goes first from one workload to the next, and reports the spread. Timing
noise is a property of the hardware; the gates and the reported spread are
how the benchmark stays honest about it.

Protocol `dryft.model.generation/3` (a child process timed by a token stream)
and input derivation `qwen-prompts/1` are bound into every report, and a run
carries a signed attestation of the container it ran in. Runs from an earlier
protocol are not comparable and are not ranked.
