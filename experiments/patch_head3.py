import ast
p = 'engine/engine.py'
s = open(p).read()
def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)

rep('HEAD_TRAIN_S = float(os.environ.get("ENGINE_HEAD_TRAIN_S", "8"))', 'HEAD_TRAIN_S = float(os.environ.get("ENGINE_HEAD_TRAIN_S", "12"))')
rep('HEAD_ROUNDS = int(os.environ.get("ENGINE_HEAD_ROUNDS", "4"))', 'HEAD_ROUNDS = int(os.environ.get("ENGINE_HEAD_ROUNDS", "8"))\nHEAD_T = int(os.environ.get("ENGINE_HEAD_T", "3"))          # verify width: 1 real token + HEAD_T-1 chained drafts')
# runner
rep('''        super().__init__(eng, st, 2, plan)
        B = st.batch
        self.draft = torch.zeros((B,), device=eng.device, dtype=torch.int64)''', '''        super().__init__(eng, st, HEAD_T, plan)
        B = st.batch
        self.draft = torch.zeros((B, HEAD_T - 1), device=eng.device, dtype=torch.int64)''')
rep('''        self.inp[:, 1] = self.draft
        self.pos.copy_(self.hlen - 1)
        nxt = eng._forward_step(st, self.inp.view(-1), self.pos, 2, self.attn, self.use_gemv)
        self.out.copy_(nxt.view(B, 2))
        h = eng._last_h.view(B, 2, -1)
        self.prev.copy_(self.hlen)
        spec_accept(self.hist, self.hlen, self.lim, self.inp, self.out, 2)
        idx = (self.hlen - self.prev - 1).clamp_(0, 1).long()''', '''        self.inp[:, 1:] = self.draft
        self.pos.copy_(self.hlen - 1)
        nxt = eng._forward_step(st, self.inp.view(-1), self.pos, HEAD_T, self.attn, self.use_gemv)
        self.out.copy_(nxt.view(B, HEAD_T))
        h = eng._last_h.view(B, HEAD_T, -1)
        self.prev.copy_(self.hlen)
        spec_accept(self.hist, self.hlen, self.lim, self.inp, self.out, HEAD_T)
        idx = (self.hlen - self.prev - 1).clamp_(0, HEAD_T - 1).long()''')
rep('''            r.draft.copy_(st.first)
''', '''            r.draft.copy_(st.first.unsqueeze(1).expand_as(r.draft))
''')
# chained drafting
rep('''        x = torch.cat([h, F.embedding(g, self.embed) * self.head_escale], -1)
        z = h + F.linear(F.silu(F.linear(x, self.head_a)), self.head_b)
        return self.head_ids[torch.argmax(F.linear(z, self.head_lm), dim=-1)]''', '''        z, tok, outs = h, g, []
        for _ in range(HEAD_T - 1):
            x = torch.cat([z, F.embedding(tok, self.embed) * self.head_escale], -1)
            z = z + F.linear(F.silu(F.linear(x, self.head_a)), self.head_b)
            tok = self.head_ids[torch.argmax(F.linear(z, self.head_lm), dim=-1)]
            outs.append(tok)
        return torch.stack(outs, 1)''')
# data: also the token two ahead of the target
rep('''                Hs.append(hb[:-1].reshape(-1, hb.shape[-1]).clone())
                Ns.append(tb[1:-1].reshape(-1).clone())
                Ts.append(tb[2:].reshape(-1).clone())''', '''                Hs.append(hb[:-2].reshape(-1, hb.shape[-1]).clone())
                Ns.append(tb[1:-2].reshape(-1).clone())
                Ts.append(tb[2:-1].reshape(-1).clone())
                T2s.append(tb[3:].reshape(-1).clone())''')
rep('''        Hs, Ns, Ts = [], [], []''', '''        Hs, Ns, Ts, T2s = [], [], [], []''')
rep('''            Tg = torch.cat(Ts).clone()''', '''            Tg = torch.cat(Ts).clone()
            Tg2 = torch.cat(T2s).clone()''')
rep('''            seen = torch.unique(torch.cat([Tg, prompt.reshape(-1).to(dev)]))''', '''            seen = torch.unique(torch.cat([Tg, Tg2, prompt.reshape(-1).to(dev)]))''')
rep('''            Tsub = remap[Tg]''', '''            Tsub, Tsub2 = remap[Tg], remap[Tg2]''')
rep('''                z = h + F.linear(F.silu(F.linear(x, a.to(torch.bfloat16))), b.to(torch.bfloat16))
                loss = F.cross_entropy(F.linear(z, Esub).float(), Tsub[idx])''', '''                a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
                z = h + F.linear(F.silu(F.linear(x, a16)), b16)
                loss = F.cross_entropy(F.linear(z, Esub).float(), Tsub[idx])
                if HEAD_T > 2:          # second chained draft: the head runs on its own output
                    x2 = torch.cat([z, E[Tg[idx]] * escale], -1)
                    z2 = z + F.linear(F.silu(F.linear(x2, a16)), b16)
                    loss = loss + 0.5 * F.cross_entropy(F.linear(z2, Esub).float(), Tsub2[idx])''')
rep('''        del H, Nx, Tg, Hs, Ns, Ts''', '''        del H, Nx, Tg, Tg2, Hs, Ns, Ts, T2s''')
rep('''        head = timed(lambda: self._spec(st, ids, input_ids, S, n, 2, "head", stats))''', '''        head = timed(lambda: self._spec(st, ids, input_ids, S, n, HEAD_T, "head", stats))''')
rep('''            st.mode = (2, "head")''', '''            st.mode = (HEAD_T, "head")''')
rep('''        tmax = max(_spec_candidates(B))''', '''        tmax = max(max(_spec_candidates(B)), 4)''')
ast.parse(s)
open(p, 'w').write(s)
print("patched3")
