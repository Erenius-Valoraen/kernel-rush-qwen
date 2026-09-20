"""Study: does training the draft head on continuations of several prompts (as a cross-workload
cache would provide) generalise better than one prompt's worth?"""
import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "huggingface_hub", "numpy", "accelerate"))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-head-div", image=image)


@app.function(gpu="H100", timeout=3000, volumes={"/weights": vol})
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
    wp = corpus("https://www.gutenberg.org/cache/epub/2600/pg2600.txt")
    pp = corpus("https://www.gutenberg.org/cache/epub/1342/pg1342.txt")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
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
        return torch.cat(hs), seq[:, L:]                      # hs[:, j] predicts toks[:, j]

    def from_prompt(corp, nseed, slen=96, n=96):
        """nseed continuations seeded from pieces of one 16x512 'warmup prompt'."""
        st = torch.randint(0, len(corp) - 513, (16,), generator=g).tolist()
        base = [corp[q:q + 512] for q in st]
        seeds = []
        for _ in range(nseed):
            row = base[int(torch.randint(0, 16, (1,), generator=g))]
            s = int(torch.randint(0, 512 - slen, (1,), generator=g))
            seeds.append(row[s:s + slen])
        hs, ts = [], []
        for b in range(0, nseed, 256):
            h, t = gen(seeds[b:b + 256], n)
            hs.append(h); ts.append(t)
        return torch.cat(hs).clone(), torch.cat(ts).clone()

    def anywhere(corp, nseed, slen=512, n=96):
        st = torch.randint(0, len(corp) - slen - 1, (nseed,), generator=g).tolist()
        h, t = gen([corp[q:q + slen] for q in st], n)
        return h.clone(), t.clone()

    chunks = [from_prompt(wp, 768) for _ in range(9)]          # nine "workloads", 768 seeds each
    H_a, T_a = anywhere(wp, 256)
    H_b, T_b = anywhere(pp, 256)
    print("RESULT data ready", flush=True)

    def pairs(Hs, Ts):
        return Hs[:, :-1].reshape(-1, Hs.shape[-1]), Ts[:, :-1].reshape(-1), Ts[:, 1:].reshape(-1)

    def train(Hx, Tx, secs, init=None):
        a = (torch.randn((2560, 5120), device="cuda") * 5120 ** -0.5) if init is None else init[0].clone()
        b = torch.zeros((2560, 2560), device="cuda") if init is None else init[1].clone()
        a.requires_grad_(True); b.requires_grad_(True)
        opt = torch.optim.AdamW([a, b], lr=1e-3, weight_decay=0.0)
        h, nxt, tgt = pairs(Hx, Tx)
        t0 = time.perf_counter()
        with torch.enable_grad():
            while time.perf_counter() - t0 < secs:
                idx = torch.randint(0, h.shape[0], (4096,), device="cuda")
                x = torch.cat([h[idx], E[nxt[idx]] * 30.0], -1)
                z = h[idx] + F.linear(F.silu(F.linear(x, a.to(torch.bfloat16))), b.to(torch.bfloat16))
                loss = F.cross_entropy(F.linear(z, E).float(), tgt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                for gp in opt.param_groups:
                    gp["lr"] = 1e-3 * max(0.05, 1 - (time.perf_counter() - t0) / secs)
                opt.step()
        return a.detach(), b.detach()

    @torch.inference_mode()
    def acc(ab, Hs, Ts):
        a, b = ab[0].to(torch.bfloat16), ab[1].to(torch.bfloat16)
        h, nxt, tgt = pairs(Hs, Ts)
        out = []
        for i in range(0, h.shape[0], 8192):
            x = torch.cat([h[i:i + 8192], E[nxt[i:i + 8192]] * 30.0], -1)
            z = h[i:i + 8192] + F.linear(F.silu(F.linear(x, a)), b)
            out.append((F.linear(z, E).argmax(-1) == tgt[i:i + 8192]).float())
        return torch.cat(out).mean().item()

    def report(name, ab):
        print(f"RESULT {name}: same-corpus {acc(ab, H_a, T_a):.3f}  other-book {acc(ab, H_b, T_b):.3f}", flush=True)

    report("1 prompt, 12s", train(chunks[0][0], chunks[0][1], 12))
    for k in (3, 9):
        Hx = torch.cat([c[0] for c in chunks[:k]]); Tx = torch.cat([c[1] for c in chunks[:k]])
        report(f"{k} prompts pooled, 12s", train(Hx, Tx, 12))
        report(f"{k} prompts pooled, 40s", train(Hx, Tx, 40))
    # continual: carry only the weights from workload to workload, 12 s each on the new prompt's data
    ab = None
    for k in range(9):
        ab = train(chunks[k][0], chunks[k][1], 12, init=ab)
        if k in (0, 2, 5, 8):
            report(f"weights carried over, after workload {k + 1}", ab)
    # continual with a replay buffer: a quarter of every earlier workload's data kept
    ab, keepH, keepT = None, [], []
    for k in range(9):
        Hx = torch.cat([chunks[k][0]] + keepH); Tx = torch.cat([chunks[k][1]] + keepT)
        ab = train(Hx, Tx, 12, init=ab)
        keepH.append(chunks[k][0][:192]); keepT.append(chunks[k][1][:192])
        if k in (2, 5, 8):
            report(f"weights + replay buffer, after workload {k + 1}", ab)


@app.local_entrypoint()
def main():
    run.remote()
