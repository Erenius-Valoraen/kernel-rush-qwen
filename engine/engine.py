"""Fast Qwen3 4B engine.

Hand-rolled forward over the checkpoint's weights:
  * fused QKV and gate/up weights, cuBLAS GEMMs (same as the reference);
  * Triton kernels for residual-add+RMSNorm, Q/K norm+RoPE+cache write,
    SwiGLU and split-K causal GQA attention over the cache, all matching the
    reference's BF16 rounding points;
  * a preallocated fixed-capacity KV cache with per-sequence positions;
  * decode steps captured as CUDA graphs.

Two decode modes, chosen per workload during the (untimed) warmup call by
timing both on the warmup prompt:
  * plain: one token per step, graph advances its own positions on device and
    steps are replayed back-to-back while the host streams tokens out;
  * speculative: each sequence proposes T-1 draft tokens by n-gram lookup in
    its own prompt + output, one graph verifies all T positions, and the
    longest prefix matching the model's own argmax is accepted plus the
    model's next token. Exact greedy by construction.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from kernels.gemv import gemv, gemv_swiglu
from kernels.ops import DecodeAttention, add_rmsnorm, qk_norm_rope_cache, silu_mul

PREFILL_TOKENS = 8192      # rows per prefill chunk (whole sequences per chunk)
LOOKAHEAD = 8              # plain mode: steps enqueued ahead of the one yielded
MAX_NGRAM = 4
SPEC_MARGIN = 0.95         # speculative must beat plain by 5% on warmup
GEMV_MAX_M = 128           # Triton skinny GEMMs up to this many rows
SPLIT_QKV, SPLIT_O, SPLIT_DOWN = 2, 4, 4
CALIBRATION_BUDGET_S = 150.0


def _log(msg):
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _spec_candidates(batch):
    env = os.environ.get("ENGINE_SPEC_T")
    if env is not None:
        return [int(t) for t in env.split(",")]
    if batch <= 4:
        return [1, 4, 8]
    if batch <= 16:
        return [1, 3, 5]
    if batch <= 32:
        return [1, 3]
    return [1]


class _Drafter:
    """Most-recent-occurrence n-gram lookup over one sequence's tokens."""

    def __init__(self, tokens):
        self.toks = []
        self.tables = [None] + [{} for _ in range(MAX_NGRAM)]
        self.extend(tokens)

    def extend(self, new):
        toks, tables = self.toks, self.tables
        for t in new:
            # n-grams ending at the current last token now have a continuation
            i = len(toks) - 1
            for n in range(1, MAX_NGRAM + 1):
                if i - n + 1 < 0:
                    break
                tables[n][tuple(toks[i - n + 1:i + 1])] = i + 1
            toks.append(t)

    def draft(self, k):
        toks = self.toks
        L = len(toks)
        for n in range(min(MAX_NGRAM, L), 0, -1):
            p = self.tables[n].get(tuple(toks[L - n:]))
            if p is not None:
                out = toks[p:p + k]
                if len(out) < k:
                    out = out + [out[-1] if out else toks[-1]] * (k - len(out))
                return out
        return [toks[-1]] * k


class _Runner:
    """Graph + static buffers for T tokens per sequence per step."""

    def __init__(self, eng, st, t, use_gemv):
        dev = eng.device
        self.t = t
        self.use_gemv = use_gemv
        B = st.batch
        self.attn = DecodeAttention(B, t, st.capacity, eng.nq, eng.nkv, eng.d, dev, eng.num_sms)
        self.pos = torch.zeros((B,), device=dev, dtype=torch.int32)
        if t == 1:
            self.tok = torch.zeros((B,), device=dev, dtype=torch.int64)
        else:
            self.inbuf = torch.zeros((B, t + 1), device=dev, dtype=torch.int64)
            self.out = torch.zeros((B, t), device=dev, dtype=torch.int64)
        self.graph = None

    def step(self, eng, st):
        if self.t == 1:
            nxt = eng._forward_step(st, self.tok, self.pos, 1, self.attn, self.use_gemv)
            self.tok.copy_(nxt)
            self.pos.add_(1)
        else:
            self.pos.copy_(self.inbuf[:, self.t])
            toks = self.inbuf[:, :self.t].reshape(-1)
            nxt = eng._forward_step(st, toks, self.pos, self.t, self.attn, self.use_gemv)
            self.out.copy_(nxt.view(st.batch, self.t))

    def run(self, eng, st):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.step(eng, st)

    def capture(self, eng, st):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            # Autotuning happens here, so run at the longest context.
            for _ in range(2):
                if self.t > 1:
                    self.inbuf.zero_()
                    self.inbuf[:, self.t] = st.capacity - self.t
                else:
                    self.pos.fill_(st.capacity - 2)
                self.step(eng, st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.step(eng, st)
        torch.cuda.synchronize()
        self.graph = g


class _State:
    """KV cache and runners for one (batch, capacity) shape."""

    def __init__(self, eng, batch, capacity):
        dev = eng.device
        self.batch, self.capacity = batch, capacity
        self.k_cache = torch.zeros((eng.n_layers, batch, eng.nkv, capacity, eng.d),
                                   device=dev, dtype=torch.bfloat16)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.zero_pos = torch.zeros((batch,), device=dev, dtype=torch.int32)
        self.first = torch.zeros((batch,), device=dev, dtype=torch.int64)
        self.runners = {}
        self.pf_len = None        # prompt length the prefill graph was captured for
        self.pf_graph = None
        self.pf_ids = None
        self.pf_runs = 0
        self.mode = None          # chosen (T, use_gemv) for this shape

    def runner(self, eng, t, use_gemv):
        r = self.runners.get((t, use_gemv))
        if r is None:
            r = self.runners[(t, use_gemv)] = _Runner(eng, self, t, use_gemv)
            if eng.use_graphs:
                try:
                    r.capture(eng, self)
                except Exception as e:  # pragma: no cover
                    _log(f"graph capture failed for T={t}, running eager: {e!r}")
                    r.graph = None
        return r


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda = self.device.type == "cuda"
        self.use_graphs = self.cuda and os.environ.get("ENGINE_NO_GRAPH") != "1"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        ).eval().to(self.device)
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.nq = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.d = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.eps = cfg.rms_norm_eps
        self.num_sms = (torch.cuda.get_device_properties(self.device).multi_processor_count
                        if self.cuda else 132)

        base = model.model
        self.rotary = base.rotary_emb
        self.embed = base.embed_tokens.weight
        self.lm_head = model.lm_head.weight
        self.final_norm = base.norm.weight
        self.layers = []
        with torch.no_grad():
            for layer in base.layers:
                at, mlp = layer.self_attn, layer.mlp
                self.layers.append(dict(
                    ln1=layer.input_layernorm.weight,
                    ln2=layer.post_attention_layernorm.weight,
                    qkv=torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous(),
                    o=at.o_proj.weight,
                    qn=at.q_norm.weight,
                    kn=at.k_norm.weight,
                    gu=torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous(),
                    down=mlp.down_proj.weight,
                ))
                at.q_proj = at.k_proj = at.v_proj = None
                mlp.gate_proj = mlp.up_proj = None
        del model
        if self.cuda:
            torch.cuda.empty_cache()

        self.state = None
        self.host_buf = None
        self.rope_len = 0
        self._ensure_rope(8192)

    # ------------------------------------------------------------------ utils

    @torch.inference_mode()
    def _ensure_rope(self, n):
        if n <= self.rope_len:
            return
        n = max(n, 2 * self.rope_len)
        pos = torch.arange(n, device=self.device).unsqueeze(0)
        dummy = torch.empty((1,), device=self.device, dtype=torch.bfloat16)
        cos, sin = self.rotary(dummy, pos)
        self.cos = cos[0].contiguous()
        self.sin = sin[0].contiguous()
        self.rope_len = n
        if self.state is not None:          # graphs captured the old tables
            self.state = None

    def _get_state(self, batch, needed):
        st = self.state
        if st is not None and st.batch == batch and st.capacity >= needed:
            return st
        self.state = None
        if self.cuda:
            torch.cuda.empty_cache()
        capacity = -(-needed // 128) * 128
        self.state = _State(self, batch, capacity)
        return self.state

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    # ---------------------------------------------------------------- forward

    def _prefill(self, ids, st):
        """ids: [B, S] on device. Fills the cache, writes first tokens to st.first.

        The first run for a prompt length is eager (compiles kernels); the
        second captures a CUDA graph that later calls replay."""
        S = ids.shape[1]
        if not self.use_graphs:
            return self._prefill_eager(ids, st)
        if st.pf_len != S:
            st.pf_len, st.pf_graph, st.pf_runs = S, None, 0
            st.pf_ids = torch.empty_like(ids)
        if st.pf_graph is not None:
            st.pf_ids.copy_(ids)
            st.pf_graph.replay()
            return
        st.pf_runs += 1
        if st.pf_runs < 2:
            return self._prefill_eager(ids, st)
        try:
            st.pf_ids.copy_(ids)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._prefill_eager(st.pf_ids, st)
            torch.cuda.synchronize()
            st.pf_graph = g
            g.replay()
        except Exception as e:  # pragma: no cover
            _log(f"prefill graph capture failed: {e!r}")
            st.pf_runs = -10 ** 9
            self._prefill_eager(ids, st)

    def _prefill_eager(self, ids, st):
        B, S = ids.shape
        per = max(1, PREFILL_TOKENS // S)
        nq, nkv, d = self.nq, self.nkv, self.d
        for b0 in range(0, B, per):
            g = min(per, B - b0)
            x = F.embedding(ids[b0:b0 + g].reshape(-1), self.embed)
            delta = None
            for li, L in enumerate(self.layers):
                h = add_rmsnorm(x, delta, L["ln1"], self.eps)
                qkv = F.linear(h, L["qkv"])
                kc, vc = st.k_cache[li], st.v_cache[li]
                q = qk_norm_rope_cache(qkv, L["qn"], L["kn"], self.cos, self.sin, kc, vc,
                                       st.zero_pos, S, b0, self.eps, nq, nkv, d)
                q = q.view(g, S, nq, d).transpose(1, 2).contiguous()
                k = _repeat_kv(kc[b0:b0 + g, :, :S], nq // nkv).contiguous()
                v = _repeat_kv(vc[b0:b0 + g, :, :S], nq // nkv).contiguous()
                a = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=d ** -0.5)
                a = a.transpose(1, 2).reshape(g * S, nq * d)
                o = F.linear(a, L["o"])
                h = add_rmsnorm(x, o, L["ln2"], self.eps)
                delta = F.linear(silu_mul(F.linear(h, L["gu"])), L["down"])
            xl = x.view(g, S, -1)[:, -1].contiguous()
            dl = delta.view(g, S, -1)[:, -1].contiguous()
            h = add_rmsnorm(xl, dl, self.final_norm, self.eps)
            logits = F.linear(h, self.lm_head)
            st.first[b0:b0 + g] = torch.argmax(logits, dim=-1)

    def _forward_step(self, st, toks, pos, t, attn, use_gemv):
        """toks: [B*T] ids, sequence b's token j at position pos[b]+j. Returns argmax [B*T]."""
        if use_gemv:
            return self._forward_step_gemv(st, toks, pos, t, attn)
        nq, nkv, d = self.nq, self.nkv, self.d
        x = F.embedding(toks, self.embed)
        h = add_rmsnorm(x, None, self.layers[0]["ln1"], self.eps)
        n = len(self.layers)
        for li, L in enumerate(self.layers):
            qkv = F.linear(h, L["qkv"])
            kc, vc = st.k_cache[li], st.v_cache[li]
            q = qk_norm_rope_cache(qkv, L["qn"], L["kn"], self.cos, self.sin, kc, vc,
                                   pos, t, 0, self.eps, nq, nkv, d)
            a = attn(q, kc, vc, pos)
            o = F.linear(a, L["o"])
            h = add_rmsnorm(x, o, L["ln2"], self.eps)
            delta = F.linear(silu_mul(F.linear(h, L["gu"])), L["down"])
            nw = self.layers[li + 1]["ln1"] if li + 1 < n else self.final_norm
            h = add_rmsnorm(x, delta, nw, self.eps)
        logits = F.linear(h, self.lm_head)
        return torch.argmax(logits, dim=-1)

    def _forward_step_gemv(self, st, toks, pos, t, attn):
        nq, nkv, d = self.nq, self.nkv, self.d
        x = F.embedding(toks, self.embed)
        h = add_rmsnorm(x, None, self.layers[0]["ln1"], self.eps)
        n = len(self.layers)
        for li, L in enumerate(self.layers):
            qkv = gemv(h, L["qkv"], SPLIT_QKV)
            kc, vc = st.k_cache[li], st.v_cache[li]
            q = qk_norm_rope_cache(qkv, L["qn"], L["kn"], self.cos, self.sin, kc, vc,
                                   pos, t, 0, self.eps, nq, nkv, d)
            a = attn(q, kc, vc, pos)
            h = add_rmsnorm(x, gemv(a, L["o"], SPLIT_O), L["ln2"], self.eps)
            delta = gemv(gemv_swiglu(h, L["gu"]), L["down"], SPLIT_DOWN)
            nw = self.layers[li + 1]["ln1"] if li + 1 < n else self.final_norm
            h = add_rmsnorm(x, delta, nw, self.eps)
        logits = gemv(h, self.lm_head)
        return torch.argmax(logits, dim=-1)

    # ------------------------------------------------------------ decode loops

    def _plain(self, st, ids, S, n, use_gemv):
        r = st.runner(self, 1, use_gemv)
        self._prefill(ids, st)
        r.tok.copy_(st.first)
        r.pos.fill_(S)
        B = st.batch
        if not self.cuda:
            yield st.first.tolist()
            for _ in range(n - 1):
                r.run(self, st)
                yield r.tok.tolist()
            return
        if self.host_buf is None or self.host_buf.shape[0] < n or self.host_buf.shape[1] != B:
            self.host_buf = torch.empty((n, B), dtype=torch.int64, pin_memory=True)
        host = self.host_buf
        events = [None] * n
        stream = torch.cuda.current_stream()

        def launch(i):
            if i > 0:
                r.run(self, st)
            host[i].copy_(r.tok, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(stream)
            events[i] = ev

        launch(0)
        launched = 1
        for i in range(n):
            while launched < n and launched <= i + LOOKAHEAD:
                launch(launched)
                launched += 1
            events[i].synchronize()
            yield host[i].tolist()

    def _spec(self, st, ids, input_ids, S, n, t, use_gemv, stats=None):
        r = st.runner(self, t, use_gemv)
        B = st.batch
        k = t - 1
        self._prefill(ids, st)
        first_host = st.first.to("cpu", non_blocking=True) if self.cuda else st.first
        ev = None
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record()
        drafters = [_Drafter(p) for p in input_ids]     # overlaps the prefill
        if ev is not None:
            ev.synchronize()
        first = first_host.tolist()
        yield first
        if n == 1:
            return
        outs = [[tk] for tk in first]
        for dr, tk in zip(drafters, first):
            dr.extend([tk])
        pos = [S] * B
        emitted = 1
        steps = accepted = 0
        inbuf_host = torch.empty((B, t + 1), dtype=torch.int64, pin_memory=self.cuda)
        out_host = torch.empty((B, t), dtype=torch.int64, pin_memory=self.cuda)
        inb = inbuf_host.numpy()
        while emitted < n:
            drafts = []
            for b in range(B):
                if len(outs[b]) >= n:        # finished: harmless work at position 0
                    dr = [0] * k
                    inb[b, 0] = 0
                    inb[b, t] = 0
                else:
                    dr = drafters[b].draft(k)
                    inb[b, 0] = outs[b][-1]
                    inb[b, t] = pos[b]
                inb[b, 1:t] = dr
                drafts.append(dr)
            r.inbuf.copy_(inbuf_host, non_blocking=True)
            r.run(self, st)
            out_host.copy_(r.out, non_blocking=True)
            if self.cuda:
                e = torch.cuda.Event()
                e.record()
                e.synchronize()
            res = out_host.tolist()
            steps += 1
            for b in range(B):
                if len(outs[b]) >= n:
                    continue
                row, dr = res[b], drafts[b]
                a = 0
                while a < k and dr[a] == row[a]:
                    a += 1
                new = row[:a + 1]
                outs[b].extend(new)
                drafters[b].extend(new)
                pos[b] += a + 1
                accepted += a
            ready = min(min(len(o) for o in outs), n)
            while emitted < ready:
                yield [o[emitted] for o in outs]
                emitted += 1
        if stats is not None:
            stats["steps"] = steps
            stats["accepted"] = accepted
            stats["outs"] = outs

    # ---------------------------------------------------------------- choose

    def _calibrate(self, st, ids, input_ids, S, n):
        """Time each decode mode on the warmup prompt; keep the fastest."""
        B = st.batch
        start = time.perf_counter()
        modes = []
        for t in _spec_candidates(B):
            if t > 1 and n < 4:
                continue
            modes.append((t, False))
            if self.cuda and B * t <= GEMV_MAX_M and os.environ.get("ENGINE_NO_GEMV") != "1":
                modes.append((t, True))
        if not self.cuda and os.environ.get("ENGINE_TEST_GEMV") == "1":
            modes = [(t, True) for t, _ in modes if B * t <= GEMV_MAX_M]
        best, best_time, report = (1, False), None, []
        for t, g in modes:
            if time.perf_counter() - start > CALIBRATION_BUDGET_S:
                report.append(f"T={t} gemv={g}: skipped (budget)")
                continue
            try:
                st.runner(self, t, g)
                stats = {}
                for rep in range(2):             # first pass warms caches
                    self._sync()
                    t0 = time.perf_counter()
                    stats = {}
                    if t == 1:
                        gen = self._plain(st, ids, S, n, g)
                    else:
                        gen = self._spec(st, ids, input_ids, S, n, t, g, stats)
                    for _ in gen:
                        pass
                    self._sync()
                    dt = time.perf_counter() - t0
                eff = dt if t == 1 else dt / SPEC_MARGIN
                acc = (f" acc/step={stats['accepted'] / max(1, stats['steps']) / B:.2f}"
                       if stats else "")
                report.append(f"T={t} gemv={g}: {dt * 1e3:.1f}ms{acc}")
                if best_time is None or eff < best_time:
                    best, best_time = (t, g), eff
            except Exception as e:  # pragma: no cover
                report.append(f"T={t} gemv={g}: failed {e!r}")
        st.mode = best
        _log(f"B={B} S={S} n={n} calibration ({time.perf_counter() - start:.1f}s): "
             f"{'; '.join(report)} -> {best}")

    # --------------------------------------------------------------- generate

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        n = max_new_tokens
        if n <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        tmax = max(_spec_candidates(B))
        self._ensure_rope(-(-(S + n + tmax) // 128) * 128 + 1)
        st = self._get_state(B, S + n + tmax)
        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        if st.mode is None:
            if n > 1 and os.environ.get("ENGINE_NO_CALIBRATE") != "1":
                self._calibrate(st, ids, input_ids, S, n)
            else:
                st.mode = (int(os.environ.get("ENGINE_MODE", "1")),
                           os.environ.get("ENGINE_TEST_GEMV") == "1")

        t, g = st.mode
        if t == 1 or n == 1:
            yield from self._plain(st, ids, S, n, g)
        else:
            yield from self._spec(st, ids, input_ids, S, n, t, g)
