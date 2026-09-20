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

    @torch.inference_mode()
    def sample_text(seeds, n, temp):
        ids = torch.tensor(seeds).cuda()
        out = model.generate(ids, max_new_tokens=n, do_sample=True, temperature=temp, top_p=0.95, pad_token_id=0)
        return out[:, ids.shape[1]:].tolist()

    st = torch.randint(0, len(wp) - 513, (16,), generator=g).tolist()
    base = [wp[q:q + 512] for q in st]

    def pieces(nseed, slen=96):
        out = []
        for _ in range(nseed):
            row = base[int(torch.randint(0, 16, (1,), generator=g))]
            s0 = int(torch.randint(0, 512 - slen, (1,), generator=g))
            out.append(row[s0:s0 + slen])
        return out

    def greedy_data(seeds, n=96):
        hs, ts = [], []
        for b0 in range(0, len(seeds), 256):
            h, t = gen(seeds[b0:b0 + 256], n)
            hs.append(h); ts.append(t)
        return torch.cat(hs).clone(), torch.cat(ts).clone()

    real = greedy_data(pieces(1536))
    synth = {}
    for temp in (0.8, 1.2):
        new_seeds = []
        sp = pieces(1536)
        for b0 in range(0, 1536, 256):
            new_seeds += [x[-96:] for x in sample_text(sp[b0:b0 + 256], 160, temp)]
        synth[temp] = greedy_data(new_seeds)
    longer = greedy_data(pieces(768), n=192)
    H_a, T_a = anywhere(wp, 256)
    H_b, T_b = anywhere(pp, 256)
    print("RESULT data ready", flush=True)

    def pairs(Hs, Ts):
        return Hs[:, :-1].reshape(-1, Hs.shape[-1]), Ts[:, :-1].reshape(-1), Ts[:, 1:].reshape(-1)

    ha, na, ta = pairs(H_a, T_a)
    hb, nb, tb = pairs(H_b, T_b)

    def fwd(a, b, hh, nn_):
        x = torch.cat([hh, E[nn_] * 30.0], -1)
        return hh + F.linear(F.silu(F.linear(x, a.to(torch.bfloat16))), b.to(torch.bfloat16))

    def run_one(name, sets, secs=14):
        h = torch.cat([pairs(*d)[0] for d in sets]); nxt = torch.cat([pairs(*d)[1] for d in sets]); tgt = torch.cat([pairs(*d)[2] for d in sets])
        a = (torch.randn((2560, 5120), device="cuda") * 5120 ** -0.5).requires_grad_(True)
        b = torch.zeros((2560, 2560), device="cuda").requires_grad_(True)
        opt = torch.optim.AdamW([a, b], lr=1e-3, weight_decay=0.0)
        t0 = time.perf_counter()
        with torch.enable_grad():
            while time.perf_counter() - t0 < secs:
                idx = torch.randint(0, h.shape[0], (4096,), device="cuda")
                loss = F.cross_entropy(F.linear(fwd(a, b, h[idx], nxt[idx]), E).float(), tgt[idx])
                opt.zero_grad(set_to_none=True); loss.backward()
                for gp in opt.param_groups:
                    gp["lr"] = 1e-3 * max(0.05, 1 - (time.perf_counter() - t0) / secs)
                opt.step()
        res = []
        with torch.inference_mode():
            for nm, (hh, nn_, tt) in (("same-corpus", (ha, na, ta)), ("other-book", (hb, nb, tb))):
                acc = [(F.linear(fwd(a, b, hh[i:i + 8192], nn_[i:i + 8192]), E).argmax(-1) == tt[i:i + 8192]).float() for i in range(0, hh.shape[0], 8192)]
                res.append(f"{nm} {torch.cat(acc).mean().item():.3f}")
        print(f"RESULT {name} ({h.shape[0]} samples): " + "  ".join(res), flush=True)

    half = lambda d: (d[0][:768], d[1][:768])
    run_one("prompt pieces only", [real])
    run_one("half pieces + half synthetic seeds T=0.8", [half(real), half(synth[0.8])])
    run_one("half pieces + half synthetic seeds T=1.2", [half(real), half(synth[1.2])])
    run_one("pieces + synthetic T=0.8 (2x data)", [real, synth[0.8]])
    run_one("pieces + synthetic T=1.2 (2x data)", [real, synth[1.2]])
    run_one("pieces + both synthetic (3x data)", [real, synth[0.8], synth[1.2]], 20)
    run_one("half pieces + longer continuations (192)", [half(real), longer])


@app.local_entrypoint()
def main():
    run.remote()
