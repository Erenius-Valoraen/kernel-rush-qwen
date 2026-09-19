import ast
p = 'engine/engine.py'
s = open(p).read()
def rep(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)

rep('HEAD_TRAIN_S = float(os.environ.get("ENGINE_HEAD_TRAIN_S", "15"))', 'HEAD_TRAIN_S = float(os.environ.get("ENGINE_HEAD_TRAIN_S", "8"))')
rep('HEAD_ROUNDS = int(os.environ.get("ENGINE_HEAD_ROUNDS", "5"))', 'HEAD_ROUNDS = int(os.environ.get("ENGINE_HEAD_ROUNDS", "4"))')
# train against the draft vocabulary only
rep('''            a = (torch.randn((Hd, 2 * Hd), device=dev) * (2 * Hd) ** -0.5).requires_grad_(True)''',
'''            seen = torch.unique(torch.cat([Tg, prompt.reshape(-1).to(dev)]))
            mask = torch.ones((E.shape[0],), device=dev, dtype=torch.bool)
            mask[seen] = False
            fill = torch.nonzero(mask).reshape(-1)[:max(0, HEAD_VOCAB - seen.numel())]
            ids = torch.cat([seen, fill])[:HEAD_VOCAB].contiguous()
            Esub = E[ids].contiguous()
            remap = torch.zeros((E.shape[0],), device=dev, dtype=torch.int64)
            remap[ids] = torch.arange(ids.numel(), device=dev)
            Tsub = remap[Tg]
            a = (torch.randn((Hd, 2 * Hd), device=dev) * (2 * Hd) ** -0.5).requires_grad_(True)''')
rep('''                loss = F.cross_entropy(F.linear(z, E).float(), Tg[idx])''', '''                loss = F.cross_entropy(F.linear(z, Esub).float(), Tsub[idx])''')
rep('''            seen = torch.unique(torch.cat([Tg, prompt.reshape(-1).to(dev)]))
            mask = torch.ones((E.shape[0],), device=dev, dtype=torch.bool)
            mask[seen] = False
            fill = torch.nonzero(mask).reshape(-1)[:max(0, HEAD_VOCAB - seen.numel())]
            ids = torch.cat([seen, fill])[:HEAD_VOCAB].contiguous()
            self.head_ids = ids
            self.head_lm = E[ids].contiguous()''', '''            self.head_ids = ids
            self.head_lm = Esub''')
# leaner calibration when the fp8 plan exists
rep('''            if self.pdl_ok and n >= 4 and not self._late()''', '''            if self.pdl_ok and n >= 4 and not self.fp8_dec_layers and not self._late()''')
rep('''        spec_ts = [t for t in _spec_candidates(B)''', '''        if self.fp8_dec_layers and B >= 2 and len(modes) > 2:
            modes = [m for m in modes if m[1] in ("fixed", "fp8")]
        spec_ts = [t for t in _spec_candidates(B)''')
ast.parse(s)
open(p, 'w').write(s)
print("patched2")
