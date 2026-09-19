"""Fast Qwen3 4B engine.

Hand-rolled forward over the checkpoint's weights:
  * fused QKV and gate/up weights, cuBLAS GEMMs (same as the reference);
  * Triton kernels for residual-add+RMSNorm, Q/K norm+RoPE+cache write,
    SwiGLU and split-K GQA decode attention, all matching the reference's
    BF16 rounding points;
  * a preallocated fixed-capacity KV cache;
  * the whole decode step (embedding -> argmax) captured in one CUDA graph
    that advances its own position on device, replayed back-to-back while
    the host streams finished tokens out of pinned memory.
"""

import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from kernels.ops import DecodeAttention, add_rmsnorm, qk_norm_rope_cache, silu_mul

PREFILL_TOKENS = 8192      # rows per prefill chunk (whole sequences per chunk)
LOOKAHEAD = 8              # decode steps enqueued ahead of the one being yielded


def _log(msg):
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


def _repeat_kv(x, n_rep):
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class _State:
    """Buffers (and the decode graph) for one (batch, capacity) shape."""

    def __init__(self, eng, batch, capacity):
        dev = eng.device
        self.batch, self.capacity = batch, capacity
        self.k_cache = torch.zeros((eng.n_layers, batch, eng.nkv, capacity, eng.d),
                                   device=dev, dtype=torch.bfloat16)
        self.v_cache = torch.zeros_like(self.k_cache)
        self.tok = torch.zeros((batch,), device=dev, dtype=torch.int64)
        self.pos = torch.zeros((1,), device=dev, dtype=torch.int32)
        self.attn = DecodeAttention(batch, capacity, eng.nq, eng.nkv, eng.d, dev, eng.num_sms)
        self.graph = None


class Engine:
    def __init__(self, model_path: str) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_graphs = self.device.type == "cuda" and os.environ.get("ENGINE_NO_GRAPH") != "1"
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
                        if self.device.type == "cuda" else 132)

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
                # Drop the unfused copies.
                at.q_proj = at.k_proj = at.v_proj = None
                mlp.gate_proj = mlp.up_proj = None
        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        self.zero_pos = torch.zeros((1,), device=self.device, dtype=torch.int32)
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
        # Graphs captured the old table's address.
        if self.state is not None:
            self.state.graph = None

    def _get_state(self, batch, needed):
        st = self.state
        if st is not None and st.batch == batch and st.capacity >= needed:
            return st
        self.state = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        capacity = -(-needed // 128) * 128
        self.state = _State(self, batch, capacity)
        return self.state

    # ---------------------------------------------------------------- forward

    def _prefill(self, ids, st):
        """ids: [B, S] on device. Fills the cache, writes first tokens to st.tok."""
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
                                       self.zero_pos, S, b0, self.eps, nq, nkv, d)
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
            st.tok[b0:b0 + g] = torch.argmax(logits, dim=-1)
        st.pos.fill_(S)

    def _decode_step(self, st):
        """One token for every sequence at position st.pos; advances st.pos."""
        nq, nkv, d = self.nq, self.nkv, self.d
        x = F.embedding(st.tok, self.embed)
        h = add_rmsnorm(x, None, self.layers[0]["ln1"], self.eps)
        n = len(self.layers)
        for li, L in enumerate(self.layers):
            qkv = F.linear(h, L["qkv"])
            kc, vc = st.k_cache[li], st.v_cache[li]
            q = qk_norm_rope_cache(qkv, L["qn"], L["kn"], self.cos, self.sin, kc, vc,
                                   st.pos, 1, 0, self.eps, nq, nkv, d)
            a = st.attn(q, kc, vc, st.pos)
            o = F.linear(a, L["o"])
            h = add_rmsnorm(x, o, L["ln2"], self.eps)
            delta = F.linear(silu_mul(F.linear(h, L["gu"])), L["down"])
            nw = self.layers[li + 1]["ln1"] if li + 1 < n else self.final_norm
            h = add_rmsnorm(x, delta, nw, self.eps)
        logits = F.linear(h, self.lm_head)
        st.tok.copy_(torch.argmax(logits, dim=-1))
        st.pos.add_(1)

    def _capture(self, st):
        # Compile every kernel specialization outside capture (on scratch
        # state), then restore; the cache contents do not matter here.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                st.pos.fill_(0)
                self._decode_step(st)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._decode_step(st)
        torch.cuda.synchronize()
        st.graph = g

    # --------------------------------------------------------------- generate

    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        n = max_new_tokens
        if n <= 0:
            return
        B, S = len(input_ids), len(input_ids[0])
        self._ensure_rope(S + n + 1)
        st = self._get_state(B, S + n)

        if self.use_graphs and st.graph is None and n > 1:
            try:
                self._capture(st)
            except Exception as e:  # pragma: no cover - fall back to eager
                _log(f"graph capture failed, running eager: {e!r}")
                self.use_graphs = False
                st.graph = None

        cuda = self.device.type == "cuda"
        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        self._prefill(ids, st)

        if not cuda:
            yield st.tok.tolist()
            for _ in range(n - 1):
                self._decode_step(st)
                yield st.tok.tolist()
            return

        if self.host_buf is None or self.host_buf.shape[0] < n or self.host_buf.shape[1] != B:
            self.host_buf = torch.empty((n, B), dtype=torch.int64, pin_memory=True)
        host = self.host_buf
        events = [None] * n
        stream = torch.cuda.current_stream()

        def launch(i):
            if i > 0:
                if st.graph is not None:
                    st.graph.replay()
                else:
                    self._decode_step(st)
            host[i].copy_(st.tok, non_blocking=True)
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
