"""Adds sparse-tree speculation (ENGINE_TREE): verify the head's top-2 for the
first draft token as a query-only leaf. Layout [last, d1a, d2, d1b]."""
import ast
p = 'engine/engine.py'
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a[:70]
    s = s.replace(a, b, 1)

rep('HEAD_MARGIN = 0.90', 'TREE = os.environ.get("ENGINE_TREE", "1") == "1"    # top-2 sparse-tree first draft\nHEAD_MARGIN = 0.90')

# --- top-2 tree draft: d1a (top1), d1b (top2) after g; d2 = top1 after d1a
rep('''    def _head_draft(self, h, g, k):''', '''    def _head_tree_draft(self, h, g):
        """h [B,H] hidden that produced g [B] -> (d1a top1, d2 top1-after-d1a, d1b top2)."""
        x = torch.cat([h, F.embedding(g, self.embed) * self.head_escale], -1)
        z1 = h + F.linear(F.silu(F.linear(x, self.head_a)), self.head_b)
        l1 = F.linear(z1, self.head_lm)
        i1 = torch.argmax(l1, dim=-1)
        l1b = l1.scatter(-1, i1[:, None], float("-inf"))
        i2 = torch.argmax(l1b, dim=-1)
        d1a, d1b = self.head_ids[i1], self.head_ids[i2]
        x2 = torch.cat([z1, F.embedding(d1a, self.embed) * self.head_escale], -1)
        z2 = z1 + F.linear(F.silu(F.linear(x2, self.head_a)), self.head_b)
        d2 = self.head_ids[torch.argmax(F.linear(z2, self.head_lm), dim=-1)]
        return d1a, d2, d1b

    def _head_draft(self, h, g, k):''')

# --- tree runner
rep('''class _State:
    """KV cache and runners for one (batch, capacity) shape."""''', '''class _TreeHeadRunner(_Runner):
    """Sparse-tree speculation, width 4: [last, d1a, d2, d1b]. d1b is the head's
    2nd-best first token, verified as a query-only leaf (attends only `last`, no
    cache slot). Advances up to 3 tokens/step; the next draft seeds from the
    hidden of the last accepted token's branch."""

    REL_POS = [0, 1, 2, 1]
    REL_LIM = [-1, 0, 1, 0]
    WRITE = [1, 1, 1, 0]

    def __init__(self, eng, st, plan):
        super().__init__(eng, st, 4, plan)
        B = st.batch
        self.ar = torch.arange(B, device=eng.device)
        self.d1a = torch.zeros((B,), device=eng.device, dtype=torch.int64)
        self.d2 = torch.zeros((B,), device=eng.device, dtype=torch.int64)
        self.d1b = torch.zeros((B,), device=eng.device, dtype=torch.int64)
        if hasattr(self.attn, "set_tree"):
            self.attn.set_tree(self.REL_POS, self.REL_LIM, self.WRITE)

    def reset_draft(self, first):
        self.d1a.copy_(first); self.d2.copy_(first); self.d1b.copy_(first)

    def step(self, eng, st):
        B = st.batch
        last = self.hist.gather(1, (self.hlen - 1).long().unsqueeze(1)).squeeze(1)
        self.inp[:, 0] = last
        self.inp[:, 1] = self.d1a
        self.inp[:, 2] = self.d2
        self.inp[:, 3] = self.d1b
        self.pos.copy_(self.hlen - 1)
        nxt = eng._forward_step(st, self.inp.view(-1), self.pos, 4, self.attn, self.use_gemv)
        out = nxt.view(B, 4)
        h = eng._last_h.view(B, 4, -1)
        d1a, d2, d1b = self.inp[:, 1], self.inp[:, 2], self.inp[:, 3]
        g1 = out[:, 0]
        ma = g1 == d1a
        mb = (~ma) & (g1 == d1b)
        a2 = ma & (out[:, 1] == d2)
        nacc = 1 + ma.long() * (1 + a2.long()) + mb.long()
        tok0 = g1
        tok1 = torch.where(mb, out[:, 3], out[:, 1])
        tok2 = out[:, 2]
        base = self.hlen.long()
        cap = self.lim.long()

        def commit(off, tok, cond):
            pos = base + off
            ok = cond & (pos < cap)
            idx = torch.where(ok, pos, torch.zeros_like(pos))
            self.hist[self.ar, idx] = torch.where(ok, tok.to(torch.int32), self.hist[self.ar, idx])

        commit(0, tok0, torch.ones(B, dtype=torch.bool, device=self.ar.device))
        commit(1, tok1, nacc >= 2)
        commit(2, tok2, nacc >= 3)
        self.hlen.copy_(torch.minimum(base + nacc, cap).to(torch.int32))
        row = torch.where(a2, torch.full_like(g1, 2),
                          torch.where(ma, torch.ones_like(g1),
                                      torch.where(mb, torch.full_like(g1, 3), torch.zeros_like(g1))))
        nd1a, nd2, nd1b = eng._head_tree_draft(h[self.ar, row], out[self.ar, row])
        self.d1a.copy_(nd1a); self.d2.copy_(nd2); self.d1b.copy_(nd1b)


class _State:
    """KV cache and runners for one (batch, capacity) shape."""''')

# --- dispatch
rep('''            if use_gemv == "head":
                r = self.runners[(t, use_gemv)] = _HeadRunner(eng, self, eng.head_plan, t)''',
    '''            if use_gemv == "tree":
                r = self.runners[(t, use_gemv)] = _TreeHeadRunner(eng, self, eng.head_plan)
            elif use_gemv == "head":
                r = self.runners[(t, use_gemv)] = _HeadRunner(eng, self, eng.head_plan, t)''')

# --- _spec reset: generalize the draft seed
rep('''        if isinstance(r, _HeadRunner):
            r.draft.copy_(st.first.unsqueeze(1).expand_as(r.draft))''',
    '''        if isinstance(r, _TreeHeadRunner):
            r.reset_draft(st.first)
        elif isinstance(r, _HeadRunner):
            r.draft.copy_(st.first.unsqueeze(1).expand_as(r.draft))''')

# --- calibration: try the tree width alongside the chained head widths
rep('''        best_w, best_t, report = None, base * HEAD_MARGIN, []''',
    '''        best_w, best_t, report = None, base * HEAD_MARGIN, []
        self._tree_win = False''')
rep('''        _log(f"head speculation: {'; '.join(report)} vs {base * 1e3:.1f}ms {st.mode} -> {best_w}")
        if best_w is not None:
            st.mode = (best_w, "head")''',
    '''        if TREE:
            stats = {}
            dt = timed(lambda: self._spec(st, ids, input_ids, S, n, 4, "tree", stats))
            acc = stats["accepted"] / max(1, stats["steps"]) / st.batch if stats else 0.0
            report.append(f"tree: {dt * 1e3:.1f}ms acc/step {acc:.2f}")
            if dt < best_t:
                best_w, best_t, self._tree_win = 4, dt, True
        _log(f"head speculation: {'; '.join(report)} vs {base * 1e3:.1f}ms {st.mode} -> {best_w}")
        if best_w is not None:
            st.mode = (best_w, "tree" if self._tree_win else "head")''')

ast.parse(s)
open(p, 'w').write(s)
print("patched tree")
