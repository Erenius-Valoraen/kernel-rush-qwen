"""One-off patch: adds the trained draft head to engine/engine.py."""
import ast

p = 'engine/engine.py'
s = open(p).read()


def rep(old, new, count=1):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new, count)


rep("CALIB_REPS = 4                # timed repetitions per decode mode (min taken)",
    '''CALIB_REPS = int(os.environ.get("ENGINE_CALIB_REPS", "2"))   # timed repetitions per decode mode (min taken)
# Draft head: a small MLP trained during warmup on the model's own greedy text
# proposes the token after next; the model verifies it (exact greedy output).
HEAD = os.environ.get("ENGINE_HEAD", "1") == "1"
HEAD_TRAIN_S = float(os.environ.get("ENGINE_HEAD_TRAIN_S", "15"))
HEAD_ROUNDS = int(os.environ.get("ENGINE_HEAD_ROUNDS", "5"))
HEAD_BATCH = int(os.environ.get("ENGINE_HEAD_BATCH", "256"))
HEAD_STEPS = int(os.environ.get("ENGINE_HEAD_STEPS", "96"))
HEAD_VOCAB = 32768
HEAD_MARGIN = 0.97            # head speculation must beat the plain mode by 3% on warmup''')

rep('''        logits = F.linear(h, self.lm_head)
        return torch.argmax(logits, dim=-1)

    def _attend''', '''        self._last_h = h
        logits = F.linear(h, self.lm_head)
        return torch.argmax(logits, dim=-1)

    def _attend''')
rep('''        logits = linear_fp8(h, self.lm8) if FP8_LM else gemv(h, self.lm_head, 1)''',
    '''        self._last_h = h
        logits = linear_fp8(h, self.lm8) if FP8_LM else gemv(h, self.lm_head, 1)''')
rep('''        logits = _gemm_run("lm", plan["lm"], h, self.lm_head)''',
    '''        self._last_h = h
        logits = _gemm_run("lm", plan["lm"], h, self.lm_head)''')

rep("            if self.t == 1 and MULTI > 1:",
    "            if self.t == 1 and MULTI > 1 and not getattr(self, \"no_multi\", False):")

rep('''class _State:
    """KV cache and runners for one (batch, capacity) shape."""''', '''class _TrainRunner(_Runner):
    """Plain decode that also keeps each step's final hidden state (head training data)."""

    no_multi = True

    def __init__(self, eng, st, plan):
        super().__init__(eng, st, 1, plan)
        self.hbuf = torch.zeros((st.batch, eng.embed.shape[1]), device=eng.device, dtype=torch.bfloat16)

    def step(self, eng, st):
        super().step(eng, st)
        self.hbuf.copy_(eng._last_h)


class _HeadRunner(_Runner):
    """T=2 verify steps; the draft for the next step comes from the trained head."""

    def __init__(self, eng, st, plan):
        super().__init__(eng, st, 2, plan)
        B = st.batch
        self.draft = torch.zeros((B,), device=eng.device, dtype=torch.int64)
        self.prev = torch.zeros((B,), device=eng.device, dtype=torch.int32)
        self.ar = torch.arange(B, device=eng.device)

    def step(self, eng, st):
        B = st.batch
        last = self.hist.gather(1, (self.hlen - 1).long().unsqueeze(1)).squeeze(1)
        self.inp[:, 0] = last
        self.inp[:, 1] = self.draft
        self.pos.copy_(self.hlen - 1)
        nxt = eng._forward_step(st, self.inp.view(-1), self.pos, 2, self.attn, self.use_gemv)
        self.out.copy_(nxt.view(B, 2))
        h = eng._last_h.view(B, 2, -1)
        self.prev.copy_(self.hlen)
        spec_accept(self.hist, self.hlen, self.lim, self.inp, self.out, 2)
        idx = (self.hlen - self.prev - 1).clamp_(0, 1).long()
        self.draft.copy_(eng._head_draft(h[self.ar, idx], self.out[self.ar, idx]))


class _State:
    """KV cache and runners for one (batch, capacity) shape."""''')

rep('''            r = self.runners[(t, use_gemv)] = _Runner(eng, self, t, use_gemv)''',
    '''            if use_gemv == "head":
                r = self.runners[(t, use_gemv)] = _HeadRunner(eng, self, eng.head_plan)
            elif isinstance(use_gemv, tuple):           # ("train", plan)
                r = self.runners[(t, use_gemv)] = _TrainRunner(eng, self, use_gemv[1])
            else:
                r = self.runners[(t, use_gemv)] = _Runner(eng, self, t, use_gemv)''')

rep('''        r.lim.fill_(S + n)
''', '''        r.lim.fill_(S + n)
        if isinstance(r, _HeadRunner):
            r.draft.copy_(st.first)
''')

HEAD_CODE = '''    # ------------------------------------------------------------ draft head

    def _head_draft(self, h, g):
        """h [B, H] final hidden that produced token g [B] -> proposed token after g."""
        x = torch.cat([h, F.embedding(g, self.embed) * self.head_escale], -1)
        z = h + F.linear(F.silu(F.linear(x, self.head_a)), self.head_b)
        return self.head_ids[torch.argmax(F.linear(z, self.head_lm), dim=-1)]

    def _train_head(self, input_ids, plan):
        """Generate greedy continuations of pieces of the warmup prompt with the
        engine itself, then fit the head to predict the token after next."""
        t_start = time.perf_counter()
        dev = self.device
        B0, S = len(input_ids), len(input_ids[0])
        slen = max(8, min(96, S - 1))
        Bt, n = HEAD_BATCH, HEAD_STEPS
        gen = torch.Generator().manual_seed(0)
        prompt = torch.tensor(input_ids, dtype=torch.int64)
        keep, state = self.state, None
        Hs, Ns, Ts = [], [], []
        try:
            self.state = None
            state = _State(self, Bt, -(-(slen + n + 2) // 128) * 128)
            r = state.runner(self, 1, ("train", False))   # cuBLAS path: any batch size
            hb = torch.empty((n, Bt, self.embed.shape[1]), device=dev, dtype=torch.bfloat16)
            tb = torch.empty((n + 1, Bt), device=dev, dtype=torch.int64)
            for _ in range(HEAD_ROUNDS):
                rows = torch.randint(0, B0, (Bt,), generator=gen)
                offs = torch.randint(0, S - slen + 1, (Bt,), generator=gen)
                seeds = torch.stack([prompt[a, b:b + slen] for a, b in zip(rows.tolist(), offs.tolist())]).to(dev)
                self._prefill_eager(seeds, state)
                r.tok.copy_(state.first)
                r.pos.fill_(slen)
                tb[0].copy_(state.first)
                for j in range(n):
                    r.run(self, state)
                    hb[j].copy_(r.hbuf)             # hidden that produced tb[j + 1]
                    tb[j + 1].copy_(r.tok)
                Hs.append(hb[:-1].reshape(-1, hb.shape[-1]).clone())
                Ns.append(tb[1:-1].reshape(-1).clone())
                Ts.append(tb[2:].reshape(-1).clone())
        finally:
            del state
            self.state = keep
            torch.cuda.empty_cache()
        t_gen = time.perf_counter() - t_start
        with torch.inference_mode(False), torch.enable_grad():
            H = torch.cat(Hs).clone()
            Nx = torch.cat(Ns).clone()
            Tg = torch.cat(Ts).clone()
            E = self.embed.detach()
            Hd = E.shape[1]
            escale = float(H.float().pow(2).mean().sqrt() / E[Nx[:4096]].float().pow(2).mean().sqrt())
            a = (torch.randn((Hd, 2 * Hd), device=dev) * (2 * Hd) ** -0.5).requires_grad_(True)
            b = torch.zeros((Hd, Hd), device=dev).requires_grad_(True)
            opt = torch.optim.AdamW([a, b], lr=1e-3, weight_decay=0.0)
            t0 = time.perf_counter()
            steps, N = 0, H.shape[0]
            while time.perf_counter() - t0 < HEAD_TRAIN_S:
                idx = torch.randint(0, N, (4096,), device=dev)
                h = H[idx]
                x = torch.cat([h, E[Nx[idx]] * escale], -1)
                z = h + F.linear(F.silu(F.linear(x, a.to(torch.bfloat16))), b.to(torch.bfloat16))
                loss = F.cross_entropy(F.linear(z, E).float(), Tg[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                for grp in opt.param_groups:
                    grp["lr"] = 1e-3 * max(0.05, 1 - (time.perf_counter() - t0) / HEAD_TRAIN_S)
                opt.step()
                steps += 1
            final_loss = float(loss)
            seen = torch.unique(torch.cat([Tg, prompt.reshape(-1).to(dev)]))
            mask = torch.ones((E.shape[0],), device=dev, dtype=torch.bool)
            mask[seen] = False
            fill = torch.nonzero(mask).reshape(-1)[:max(0, HEAD_VOCAB - seen.numel())]
            ids = torch.cat([seen, fill])[:HEAD_VOCAB].contiguous()
            self.head_ids = ids
            self.head_lm = E[ids].contiguous()
            self.head_a = a.detach().to(torch.bfloat16).contiguous()
            self.head_b = b.detach().to(torch.bfloat16).contiguous()
            self.head_escale = escale
        del H, Nx, Tg, Hs, Ns, Ts
        torch.cuda.empty_cache()
        self.head_plan = plan
        _log(f"draft head: {N} samples gen {t_gen:.1f}s, {steps} steps loss {final_loss:.3f}, "
             f"vocab {seen.numel()} seen, total {time.perf_counter() - t_start:.1f}s")

    def _try_head(self, st, ids, input_ids, S, n):
        """Train the head; keep head speculation if it beats the chosen mode on the warmup prompt."""
        t, g = st.mode
        plan = g if g in ("fixed", "fp8") else ("fp8" if self.fp8_dec_layers and st.batch >= 2 else "fixed")
        self._train_head(input_ids, plan)

        def timed(gen_fn):
            best = None
            for rep_i in range(3):
                self._sync()
                t0 = time.perf_counter()
                for _ in gen_fn():
                    pass
                self._sync()
                if rep_i:
                    dt = time.perf_counter() - t0
                    best = dt if best is None else min(best, dt)
            return best

        if t == 1:
            base = timed(lambda: self._plain(st, ids, S, n, g))
        else:
            base = timed(lambda: self._spec(st, ids, input_ids, S, n, t, g))
        stats = {}
        head = timed(lambda: self._spec(st, ids, input_ids, S, n, 2, "head", stats))
        acc = stats["accepted"] / max(1, stats["steps"]) / st.batch if stats else 0.0
        _log(f"head speculation: {head * 1e3:.1f}ms (acc/step {acc:.2f}) vs {base * 1e3:.1f}ms {st.mode}")
        if head < base * HEAD_MARGIN:
            st.mode = (2, "head")

    # ------------------------------------------------------------ decode loops
'''
rep('''    # ------------------------------------------------------------ decode loops
''', HEAD_CODE)

rep('''                self._calibrate(st, ids, input_ids, S, n)
''', '''                self._calibrate(st, ids, input_ids, S, n)
                if (HEAD and self.cuda and n >= 8 and S >= 16 and st.mode[1] is not False
                        and not self._late()):
                    try:
                        self._try_head(st, ids, input_ids, S, n)
                    except Exception as e:  # pragma: no cover
                        _log(f"draft head unavailable: {e!r}")
                        self.state = st
''')
ast.parse(s)
open(p, 'w').write(s)
print("patched")
