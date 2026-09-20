import ast
p = 'engine/engine.py'
s = open(p).read()
def rep(a, b):
    global s
    assert a in s, a[:70]
    s = s.replace(a, b, 1)

rep('HEAD_VOCAB = ', '''# Workloads of one run share the container: keep the head and part of its training set
# (warmup-prompt continuations only) so later workloads train on more varied text.
HEAD_CACHE = os.environ.get("ENGINE_HEAD_CACHE", "/tmp/kernel_rush_head")
HEAD_CACHE_KEEP = float(os.environ.get("ENGINE_HEAD_CACHE_KEEP", "0.6"))     # share of new samples kept
HEAD_CACHE_MAX = int(os.environ.get("ENGINE_HEAD_CACHE_MAX", "600000"))     # samples
HEAD_VOCAB = ''')

rep('''            Tg2 = torch.cat(T2s).clone()
            E = self.embed.detach()''', '''            Tg2 = torch.cat(T2s).clone()
            fresh = (H, Nx, Tg, Tg2)
            cached = self._head_cache_load()
            if cached is not None:
                H = torch.cat([H, cached["H"]]); Nx = torch.cat([Nx, cached["Nx"]])
                Tg = torch.cat([Tg, cached["Tg"]]); Tg2 = torch.cat([Tg2, cached["Tg2"]])
            E = self.embed.detach()''')
rep('''            escale = float(H.float().pow(2).mean().sqrt() / E[Nx[:4096]].float().pow(2).mean().sqrt())''',
    '''            if cached is not None:
                escale = cached["escale"]
            else:
                escale = float(H[:65536].float().pow(2).mean().sqrt() / E[Nx[:4096]].float().pow(2).mean().sqrt())''')
rep('''            a = (torch.randn((Hd, 2 * Hd), device=dev) * (2 * Hd) ** -0.5).requires_grad_(True)
            b = torch.zeros((Hd, Hd), device=dev).requires_grad_(True)''', '''            if cached is not None:
                a = cached["a"].float().clone().requires_grad_(True)
                b = cached["b"].float().clone().requires_grad_(True)
            else:
                a = (torch.randn((Hd, 2 * Hd), device=dev) * (2 * Hd) ** -0.5).requires_grad_(True)
                b = torch.zeros((Hd, Hd), device=dev).requires_grad_(True)''')
rep('''            self.head_escale = escale
''', '''            self.head_escale = escale
            self._head_cache_save(fresh, cached, a.detach(), b.detach(), escale)
            n_cached = 0 if cached is None else cached["H"].shape[0]
            del fresh, cached
''')
rep('''        _log(f"draft head: {N} samples gen''', '''        _log(f"draft head cache: {n_cached} earlier samples")
        _log(f"draft head: {N} samples gen''')
rep('''    def _train_head(self, input_ids, plan, chain):''', '''    def _head_cache_load(self):
        if not HEAD_CACHE:
            return None
        try:
            path = os.path.join(HEAD_CACHE, "state.pt")
            if not os.path.exists(path):
                return None
            d = torch.load(path, map_location=self.device, weights_only=True)
            if d["a"].shape != (self.embed.shape[1], 2 * self.embed.shape[1]):
                return None
            return d
        except Exception as e:  # pragma: no cover
            _log(f"head cache unreadable: {e!r}")
            return None

    def _head_cache_save(self, fresh, cached, a, b, escale):
        if not HEAD_CACHE:
            return
        try:
            os.makedirs(HEAD_CACHE, exist_ok=True)
            H, Nx, Tg, Tg2 = fresh
            k = int(H.shape[0] * HEAD_CACHE_KEEP)
            idx = torch.randperm(H.shape[0], device=H.device)[:k]
            keep = dict(H=H[idx], Nx=Nx[idx], Tg=Tg[idx], Tg2=Tg2[idx])
            if cached is not None:
                for key in keep:
                    keep[key] = torch.cat([keep[key], cached[key]])[:HEAD_CACHE_MAX]
            d = {key: v.cpu() for key, v in keep.items()}
            d.update(a=a.cpu(), b=b.cpu(), escale=float(escale))
            tmp = os.path.join(HEAD_CACHE, f"state.{os.getpid()}.tmp")
            torch.save(d, tmp)
            os.replace(tmp, os.path.join(HEAD_CACHE, "state.pt"))
        except Exception as e:  # pragma: no cover
            _log(f"head cache not saved: {e!r}")

    def _train_head(self, input_ids, plan, chain):''')
ast.parse(s)
open(p, 'w').write(s)
print("patched cache")
