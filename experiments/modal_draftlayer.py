"""Feasibility: 1-layer attention draft (init from the last decoder layer) vs the MLP head."""
import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "huggingface_hub", "numpy", "accelerate"))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-draftlayer", image=image)


@app.function(gpu="H100", timeout=3000, volumes={"/weights": vol})
def run():
    import copy, time, torch, urllib.request
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    def corpus(url):
        t = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")[20000:1_500_000]
        return tok(t, add_special_tokens=False)["input_ids"]
    wp = corpus("https://www.gutenberg.org/cache/epub/2600/pg2600.txt")
    pp = corpus("https://www.gutenberg.org/cache/epub/1342/pg1342.txt")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    for p in model.parameters():
        p.requires_grad_(False)
    E = model.model.embed_tokens.weight
    g = torch.Generator().manual_seed(0)

    @torch.inference_mode()
    def gen(seeds, n):
        ids = torch.tensor(seeds).cuda()
        L = ids.shape[1]
        seq = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0)
        hs = []
        for b in range(0, seq.shape[0], 32):
            hs.append(model.model(input_ids=seq[b:b + 32]).last_hidden_state[:, L - 1:-1])
        return torch.cat(hs), seq[:, L:]                     # hs[:, j] predicts toks[:, j]

    def dataset(corp, nseed, slen, n, base=None):
        seeds = []
        for _ in range(nseed):
            if base is None:
                s = int(torch.randint(0, len(corp) - slen - 1, (1,), generator=g)); seeds.append(corp[s:s + slen])
            else:
                row = base[int(torch.randint(0, len(base), (1,), generator=g))]
                s = int(torch.randint(0, len(row) - slen, (1,), generator=g)); seeds.append(row[s:s + slen])
        hs, ts = [], []
        for b in range(0, nseed, 256):
            h, t = gen(seeds[b:b + 256], n); hs.append(h); ts.append(t)
        return torch.cat(hs).clone(), torch.cat(ts).clone()

    starts = torch.randint(0, len(wp) - 513, (16,), generator=g).tolist()
    warm = [wp[s:s + 512] for s in starts]
    H_tr, T_tr = dataset(wp, 2048, 96, 96, base=warm)
    H_a, T_a = dataset(wp, 256, 512, 96)
    H_b, T_b = dataset(pp, 256, 512, 96)
    print(f"RESULT data {tuple(H_tr.shape)}", flush=True)
    rot = model.model.rotary_emb
    fnorm = model.model.norm

    class Draft(torch.nn.Module):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
            self.fuse = torch.nn.Linear(5120, 2560, bias=False)
            with torch.no_grad():
                self.fuse.weight.zero_()
                self.fuse.weight[:, :2560].copy_(torch.eye(2560))
            if kind == "layer":
                self.layer = copy.deepcopy(model.model.layers[35]).float()
                for p in self.layer.parameters():
                    p.requires_grad_(True)
            else:
                self.a = torch.nn.Linear(2560, 2560, bias=False)
                self.b = torch.nn.Linear(2560, 2560, bias=False)
                torch.nn.init.zeros_(self.b.weight)

        def forward(self, h, nxt):                      # h [B, L, H], nxt [B, L]
            x = self.fuse(torch.cat([h.float(), E[nxt].float() * 30.0], -1))
            if self.kind == "layer":
                B, L, _ = x.shape
                pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pe = rot(x, pos)
                    y = self.layer(x, position_ids=pos, position_embeddings=pe)[0]
                    return fnorm(y)
            return h.float() + self.b(F.silu(self.a(x)))

    def evaluate(d, Hs, Ts):
        accs = []
        with torch.inference_mode():
            for b in range(0, Hs.shape[0], 64):
                z = d(Hs[b:b + 64, :-1], Ts[b:b + 64, :-1])
                pred = F.linear(z.to(torch.bfloat16), E).argmax(-1)
                accs.append((pred == Ts[b:b + 64, 1:]).float()[:, 8:])       # skip the first positions (no context yet)
        return torch.cat(accs).mean().item()

    for kind, secs, lr in (("mlp", 20, 1e-3), ("layer", 30, 1e-4), ("layer", 30, 3e-5), ("layer", 60, 1e-4)):
        d = Draft(kind).cuda()
        opt = torch.optim.AdamW([p for p in d.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        t0 = time.perf_counter(); steps = 0
        bs = 128
        while time.perf_counter() - t0 < secs:
            idx = torch.randint(0, H_tr.shape[0], (bs,), device="cuda")
            z = d(H_tr[idx, :-1], T_tr[idx, :-1])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = F.linear(z, E)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), T_tr[idx, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True); loss.backward()
            for gp in opt.param_groups:
                gp["lr"] = lr * max(0.05, 1 - (time.perf_counter() - t0) / secs)
            opt.step(); steps += 1
        d.eval()
        print(f"RESULT {kind} {secs}s lr{lr} ({steps} steps, loss {loss.item():.2f}): train {evaluate(d, H_tr[:256], T_tr[:256]):.3f}  "
              f"same-corpus {evaluate(d, H_a, T_a):.3f}  other-book {evaluate(d, H_b, T_b):.3f}", flush=True)
        del d, opt
        torch.cuda.empty_cache()


@app.local_entrypoint()
def main():
    run.remote()
