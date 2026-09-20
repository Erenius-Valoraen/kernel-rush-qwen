import ast, re
p = 'engine/engine.py'
s = open(p).read()
def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)

rep('HEAD_T = int(os.environ.get("ENGINE_HEAD_T", "3"))          # verify width: 1 real token + HEAD_T-1 chained drafts',
    'HEAD_T = int(os.environ.get("ENGINE_HEAD_T", "3"))          # widest verify step tried: 1 real token + T-1 chained drafts\nHEAD_T3_MAX_B = 16          # wider verify steps cost too many rows beyond this batch')
# runner: per-instance T
rep('''    def __init__(self, eng, st, plan):
        super().__init__(eng, st, HEAD_T, plan)
        B = st.batch
        self.draft = torch.zeros((B, HEAD_T - 1), device=eng.device, dtype=torch.int64)''',
'''    def __init__(self, eng, st, plan, t):
        super().__init__(eng, st, t, plan)
        B = st.batch
        self.draft = torch.zeros((B, t - 1), device=eng.device, dtype=torch.int64)''')
a = s.index("class _HeadRunner(_Runner):"); b = s.index("class _State:")
blk = s[a:b]
blk = blk.replace("self.pos, HEAD_T, self.attn", "self.pos, T, self.attn").replace("nxt.view(B, HEAD_T)", "nxt.view(B, T)") \
         .replace("eng._last_h.view(B, HEAD_T, -1)", "eng._last_h.view(B, T, -1)").replace("self.inp, self.out, HEAD_T)", "self.inp, self.out, T)") \
         .replace(".clamp_(0, HEAD_T - 1)", ".clamp_(0, T - 1)").replace("self.out[self.ar, idx]))", "self.out[self.ar, idx], T - 1))") \
         .replace("        B = st.batch\n        last =", "        B, T = st.batch, self.t\n        last =")
assert "HEAD_T" not in blk
s = s[:a] + blk + s[b:]
rep("_HeadRunner(eng, self, eng.head_plan)", "_HeadRunner(eng, self, eng.head_plan, t)")
rep('''    def _head_draft(self, h, g):''', '''    def _head_draft(self, h, g, k):''')
rep('''        for _ in range(HEAD_T - 1):''', '''        for _ in range(k):''')
rep('''    def _train_head(self, input_ids, plan):''', '''    def _train_head(self, input_ids, plan, chain):''')
rep('''                if HEAD_T > 2:          # second chained draft''', '''                if chain:               # second chained draft''')
old = s[s.index("        self._train_head(input_ids, plan)\n"):s.index("    # ------------------------------------------------------------ decode loops")]
new = '''        widths = [w for w in (2, 3) if w <= HEAD_T and (w == 2 or st.batch <= HEAD_T3_MAX_B)]
        self._train_head(input_ids, plan, chain=max(widths) > 2)

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
        best_w, best_t, report = None, base * HEAD_MARGIN, []
        for w in widths:
            stats = {}
            dt = timed(lambda: self._spec(st, ids, input_ids, S, n, w, "head", stats))
            acc = stats["accepted"] / max(1, stats["steps"]) / st.batch if stats else 0.0
            report.append(f"T={w}: {dt * 1e3:.1f}ms acc/step {acc:.2f}")
            if dt < best_t:
                best_w, best_t = w, dt
        _log(f"head speculation: {'; '.join(report)} vs {base * 1e3:.1f}ms {st.mode} -> {best_w}")
        if best_w is not None:
            st.mode = (best_w, "head")

'''
s = s.replace(old, new, 1)
ast.parse(s)
open(p, 'w').write(s)
print("patched4", s.count("HEAD_T"))
