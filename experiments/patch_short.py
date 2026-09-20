import ast
p = 'engine/engine.py'
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a[:70]
    s = s.replace(a, b, 1)

# 1. keep the prefill's last-position hidden state: the head drafts the first verify step from it
rep('''        self.first = torch.zeros((batch,), device=dev, dtype=torch.int64)''',
    '''        self.first = torch.zeros((batch,), device=dev, dtype=torch.int64)
        self.pf_h = torch.zeros((batch, eng.embed.shape[1]), device=dev, dtype=torch.bfloat16)''')
rep('''            logits = F.linear(h, self.lm_head)
            st.first[b0:b0 + g] = torch.argmax(logits, dim=-1)''',
    '''            logits = F.linear(h, self.lm_head)
            st.pf_h[b0:b0 + g] = h
            st.first[b0:b0 + g] = torch.argmax(logits, dim=-1)''')
rep('''            r.draft.copy_(st.first.unsqueeze(1).expand_as(r.draft))''',
    '''            r.draft.copy_(self._head_draft(st.pf_h, st.first, r.t - 1))''')
# 2. queue the first verify steps behind the prefill before waiting for the first token
rep('''        ev0.synchronize()               # nothing may delay the first token''',
    '''        while launched < min(LOOKAHEAD, max(1, -(-(n - 1) // t))):
            launch(launched)
            launched += 1
        ev0.synchronize()               # queued work never delays the first token's copy''')
rep('''        emitted = 1
        while launched < min(LOOKAHEAD, max(1, (n - 1) // 2)):
            launch(launched)
            launched += 1
''', '''        emitted = 1
''')
# 3. never keep more steps in flight than the slowest sequence can still need
rep('''launched - done < max(1, (n - min_len) // 2):''', '''launched - done < max(1, -(-(n - min_len) // t)):''')
ast.parse(s)
open(p, 'w').write(s)
print("patched short")
