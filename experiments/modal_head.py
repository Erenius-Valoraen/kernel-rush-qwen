"""Feasibility: train a draft head in ~60s on self-generated greedy text; measure held-out acceptance."""
import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "huggingface_hub", "numpy", "accelerate"))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-head", image=image)


@app.function(gpu="H100", timeout=2400, volumes={"/weights": vol})
def run():
    import time, torch, urllib.request
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    def corpus(url):
        t = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")[20000:1_500_000]
        return tok(t, add_special_tokens=False)["input_ids"]
    wp = corpus("https://www.gutenberg.org/cache/epub/2600/pg2600.txt")     # War and Peace
    pp = corpus("https://www.gutenberg.org/cache/epub/1342/pg1342.txt")     # Pride and Prejudice
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    E = model.model.embed_tokens.weight
    g = torch.Generator().manual_seed(0)

    @torch.inference_mode()
    def gen(seeds, n):
        """greedy continuation; returns hidden h_t (post-norm) at generated steps and tokens."""
        ids = torch.tensor(seeds).cuda()
        L = ids.shape[1]
        seq = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0)
        toks = seq[:, L:]                                  # g_1..g_n
        hs = []
        for b in range(0, seq.shape[0], 32):               # teacher-forced pass: normed last hidden
            hs.append(model.model(input_ids=seq[b:b + 32]).last_hidden_state[:, L - 1:-1])
        return torch.cat(hs), toks                          # hs[:, j] predicts toks[:, j]

    def dataset(corp, nseed, slen, n, base=None):
        if base is None:                                  # seeds anywhere in the corpus
            starts = torch.randint(0, len(corp) - slen - 1, (nseed,), generator=g).tolist()
            seeds = [corp[s:s + slen] for s in starts]
        else:                                             # seeds cut from a "warmup prompt" (16 x 512 tokens)
            seeds = []
            for _ in range(nseed):
                row = base[int(torch.randint(0, len(base), (1,), generator=g))]
                s = int(torch.randint(0, len(row) - slen, (1,), generator=g))
                seeds.append(row[s:s + slen])
        hs, toks = [], []
        for b in range(0, nseed, 256):
            h, t = gen(seeds[b:b + 256], n)
            hs.append(h); toks.append(t)
        return torch.cat(hs), torch.cat(toks)

    t0 = time.perf_counter()
    starts = torch.randint(0, len(wp) - 513, (16,), generator=g).tolist()
    warm = [wp[s:s + 512] for s in starts]
    H_tr, T_tr = dataset(wp, 1536, 96, 96, base=warm)
    t_gen = time.perf_counter() - t0
    H_a, T_a = dataset(wp, 256, 512, 96)          # same corpus, other chunks (like measured samples)
    H_b, T_b = dataset(pp, 256, 512, 96)          # different book
    print(f"RESULT data: train {tuple(H_tr.shape)} gen {t_gen:.1f}s (HF generate; the engine is ~10x faster)", flush=True)
    # hidden_states[-1] from generate: check whether it is already normed by comparing lm_head argmax
    chk = (model.lm_head(H_tr[:64].reshape(-1, H_tr.shape[-1])).argmax(-1) == T_tr[:64].reshape(-1)).float().mean().item()
    print(f"RESULT sanity: lm_head(h_t) predicts g_(t+1) with acc {chk:.3f} (expect ~1.0)", flush=True)

    def pairs(Hs, Ts, mode):
        # at step t: h_t produced token g_(t+1)=Ts[:,t+1]; target = g_(t+2)=Ts[:,t+2]
        h = Hs[:, :-1].reshape(-1, Hs.shape[-1]); nxt = Ts[:, :-1].reshape(-1); tgt = Ts[:, 1:].reshape(-1)
        return h, nxt, tgt

    class Head(torch.nn.Module):
        def __init__(self, mode, width=2560):
            super().__init__()
            self.mode = mode
            d_in = 2560 * (2 if mode == "eagle" else 1)
            self.a = torch.nn.Linear(d_in, width, bias=False)
            self.b = torch.nn.Linear(width, 2560, bias=False)
            torch.nn.init.zeros_(self.b.weight)
        def forward(self, h, nxt):
            x = torch.cat([h, E[nxt] * 30.0], -1) if self.mode == "eagle" else h
            return h + self.b(F.silu(self.a(x)))

    for mode in ("medusa", "eagle"):
        for secs in (30, 75):
            head = Head(mode).cuda().to(torch.bfloat16)
            master = [p.detach().float().clone().requires_grad_(True) for p in head.parameters()]
            opt = torch.optim.AdamW(master, lr=1e-3, weight_decay=0.0)
            h, nxt, tgt = pairs(H_tr, T_tr, mode)
            N = h.shape[0]
            t0 = time.perf_counter(); step = 0
            total = None
            while time.perf_counter() - t0 < secs:
                idx = torch.randint(0, N, (4096,), device="cuda")
                for p, m in zip(head.parameters(), master):
                    p.data.copy_(m.data)
                logits = F.linear(head(h[idx], nxt[idx]), E.detach()).float()
                loss = F.cross_entropy(logits, tgt[idx])
                grads = torch.autograd.grad(loss, list(head.parameters()))
                for m, gr in zip(master, grads):
                    m.grad = gr.float()
                frac = (time.perf_counter() - t0) / secs
                for gp in opt.param_groups:
                    gp["lr"] = 1e-3 * max(0.05, 1 - frac)
                opt.step(); step += 1
            for p, m in zip(head.parameters(), master):
                p.data.copy_(m.data)
            res = []
            with torch.inference_mode():
                for name, (Hs, Ts) in (("train", (H_tr[:256], T_tr[:256])), ("same-corpus", (H_a, T_a)), ("other-book", (H_b, T_b))):
                    hh, nn_, tt = pairs(Hs, Ts, mode)
                    acc = []
                    for b in range(0, hh.shape[0], 8192):
                        acc.append((F.linear(head(hh[b:b + 8192], nn_[b:b + 8192]), E).argmax(-1) == tt[b:b + 8192]).float())
                    res.append(f"{name} {torch.cat(acc).mean().item():.3f}")
            print(f"RESULT {mode} {secs}s ({step} steps, {N} samples, loss {loss.item():.2f}): acceptance " + "  ".join(res), flush=True)


@app.local_entrypoint()
def main():
    run.remote()
