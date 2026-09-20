"""Feasibility: does an FP8 (e4m3) KV cache keep the judge's logit gap under 2.0?

Patches SDPA to round K and V through FP8 with several scale granularities, replays
greedy decode teacher-forced through the real model, and reports the worst gap
between the chosen token's logit and the max logit (the judge's actual check).
Quantizing the GQA-expanded K/V per-token over the last dim is faithful to a real
per-token-per-KV-head FP8 cache (the 4 query heads in a group share identical K/V).
"""
import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "huggingface_hub", "numpy", "accelerate"))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-kvfp8", image=image)


@app.function(gpu="H100", timeout=2400, volumes={"/weights": vol})
def run():
    import torch, urllib.request
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    text = urllib.request.urlopen("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", timeout=60).read().decode("utf-8", "ignore")[20000:2_000_000]
    corpus = tok(text, add_special_tokens=False)["input_ids"]
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    g = torch.Generator().manual_seed(0)
    FMAX = 448.0

    cfg = {"mode": "off"}
    _sdpa = F.scaled_dot_product_attention

    def q_pertoken(t):                          # scale per (…, L, :) over last dim
        s = t.abs().amax(-1, keepdim=True).float().clamp_min(1e-6) / FMAX
        return (t.float() / s).clamp_(-FMAX, FMAX).to(torch.float8_e4m3fn).float().mul_(s).to(t.dtype)

    def q_pertensor(t):
        s = t.abs().amax().float().clamp_min(1e-6) / FMAX
        return (t.float() / s).clamp_(-FMAX, FMAX).to(torch.float8_e4m3fn).float().mul_(s).to(t.dtype)

    def q_perhead(t):                           # scale per (…, H, 1, :) over L and D
        s = t.abs().amax((-2, -1), keepdim=True).float().clamp_min(1e-6) / FMAX
        return (t.float() / s).clamp_(-FMAX, FMAX).to(torch.float8_e4m3fn).float().mul_(s).to(t.dtype)

    def q_int8_pertoken(t):
        s = t.abs().amax(-1, keepdim=True).float().clamp_min(1e-6) / 127.0
        return torch.round(t.float() / s).clamp_(-127, 127).mul_(s).to(t.dtype)

    QS = {"pertoken": q_pertoken, "pertensor": q_pertensor, "perhead": q_perhead, "int8_pertoken": q_int8_pertoken}

    def patched(q, k, v, *a, **kw):
        m = cfg["mode"]
        if m != "off":
            qz = QS[m]
            k = qz(k)
            if cfg.get("v", True):
                v = qz(v)
        return _sdpa(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = patched

    def prompts(B, S):
        st = torch.randint(0, len(corpus) - S - 1, (B,), generator=g).tolist()
        return torch.tensor([corpus[s:s + S] for s in st]).cuda()

    @torch.inference_mode()
    def worst_gap(B, S, n):
        cfg["mode"] = "off"
        ids = prompts(B, S)
        seq = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=0)   # true greedy (fp16 cache)
        out = seq[:, S:]
        # teacher-force the greedy sequence under each quant mode; gap vs argmax at each output position
        res = {}
        for m in ["off"] + list(QS):
            cfg["mode"] = m
            worst, over = 0.0, 0
            for b in range(0, B, 4):
                lg = model(input_ids=seq[b:b + 4]).logits[:, S - 1:-1].float()
                ch = lg.gather(-1, out[b:b + 4].unsqueeze(-1)).squeeze(-1)
                gap = lg.max(-1).values - ch
                worst = max(worst, gap.max().item()); over += int((gap > 1.0).sum())
            res[m] = (worst, over, out.numel())
        return res

    for (B, S, n) in [(4, 512, 96), (8, 2048, 64), (16, 512, 128), (2, 4096, 48)]:
        r = worst_gap(B, S, n)
        print(f"RESULT B={B} S={S} n={n}:")
        for m, (w, o, tot) in r.items():
            print(f"    {m:16s} worst gap {w:.3f}  gaps>1.0: {o}/{tot}", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
