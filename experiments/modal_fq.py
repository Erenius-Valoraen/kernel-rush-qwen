"""Fake-quant study: which FP8 scheme for gate/up keeps the judge's gap small?"""
import os
import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", extra_index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "huggingface_hub", "numpy", "accelerate"))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-fq", image=image)


@app.function(gpu="H100", timeout=2400, volumes={"/weights": vol})
def run():
    import torch, urllib.request
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    text = urllib.request.urlopen("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", timeout=60).read().decode("utf-8", "ignore")[:2_000_000]
    corpus = tok(text, add_special_tokens=False)["input_ids"]
    g = torch.Generator().manual_seed(1)
    NSEQ, S = 96, 640
    starts = torch.randint(0, len(corpus) - S - 1, (NSEQ,), generator=g).tolist()
    ids = torch.tensor([corpus[s:s + S] for s in starts]).cuda()
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").eval().cuda()
    layers = model.model.layers
    FMAX = 448.0

    def fq(x, scale):
        return ((x.float() / scale).clamp(-FMAX, FMAX).to(torch.float8_e4m3fn).float() * scale).to(torch.bfloat16)

    cfg = {}

    def make(li, mlp):
        def fwd(x):
            c = cfg
            if not c or li not in c["layers"]:
                return mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))
            shp = x.shape
            h = x.reshape(-1, shp[-1])
            if c["act"] == "pt":
                hq = fq(h, h.abs().max().float() * c.get("head", 1.0) / FMAX)
            elif c["act"] == "tok":
                hq = fq(h, h.abs().amax(1, keepdim=True).float() / FMAX)
            else:
                hq = h
            outs = []
            k = c.get("topk", 0)
            if k:
                rows = h.abs().amax(1).topk(k).indices
                if c.get("rescale"):      # scale from the remaining rows only
                    m = torch.ones(h.shape[0], dtype=torch.bool, device=h.device); m[rows] = False
                    hq = fq(h, h[m].abs().max().float() / FMAX)
            for lin in (mlp.gate_proj, mlp.up_proj):
                w = lin.weight
                wq = w
                if c["w"] == "pt":
                    wq = fq(w, w.abs().max().float() / FMAX)
                elif c["w"] == "ch":
                    wq = fq(w, w.abs().amax(1, keepdim=True).float() / FMAX)
                o = F.linear(hq, wq)
                if k:
                    o[rows] = F.linear(h[rows], w)
                outs.append(o)
            act = F.silu(outs[0]) * outs[1]
            if c.get("down"):
                wd = mlp.down_proj.weight
                aq = fq(act, act.abs().max().float() * c.get("head", 1.0) / FMAX) if c["down"] == "pt" else fq(act, act.abs().amax(1, keepdim=True).float() / FMAX)
                wd = fq(wd, wd.abs().max().float() / FMAX)
                return F.linear(aq, wd).reshape(shp)
            return mlp.down_proj(act).reshape(shp)
        return fwd

    for li, L in enumerate(layers):
        L.mlp.forward = make(li, L.mlp)

    @torch.inference_mode()
    def logits():
        out = []
        for b in range(0, NSEQ, 4):
            out.append(model(input_ids=ids[b:b + 4]).logits[:, 511:].float().cpu())   # decode-like positions
        return torch.cat(out)

    base = logits()
    bmax = base.max(-1).values
    ALL = set(range(36))
    variants = {
        "w=pt a=pt(dyn)": dict(layers=ALL, w="pt", act="pt"),
        "w=pt a=pt top8 bf16": dict(layers=ALL, w="pt", act="pt", topk=8),
        "w=pt a=pt top32 bf16": dict(layers=ALL, w="pt", act="pt", topk=32),
        "w=pt a=pt top32 bf16 rescale": dict(layers=ALL, w="pt", act="pt", topk=32, rescale=True),
        "w=pt a=pt top128 bf16 rescale": dict(layers=ALL, w="pt", act="pt", topk=128, rescale=True),
        "w=pt a=tok": dict(layers=ALL, w="pt", act="tok"),
        "w=pt a=pt L8-27": dict(layers=set(range(8, 28)), w="pt", act="pt"),
        "w=pt a=pt L8-27 top32 rescale": dict(layers=set(range(8, 28)), w="pt", act="pt", topk=32, rescale=True),
        "only L0": dict(layers={0}, w="pt", act="pt"),
        "only L0 top32 rescale": dict(layers={0}, w="pt", act="pt", topk=32, rescale=True),
    }
    for name, c in variants.items():
        cfg.clear()
        if c:
            cfg.update(c)
        lg = logits()
        pick = lg.argmax(-1, keepdim=True)
        gap = bmax - base.gather(-1, pick).squeeze(-1)
        rms = (lg - base).pow(2).mean().sqrt().item()
        top = (lg.max(-1).values - bmax).abs().max().item()
        dt = (lg.max(-1).values - bmax).abs()
        worst = torch.topk(dt.flatten(), 3)
        where = [(int(i) // dt.shape[1], 511 + int(i) % dt.shape[1], round(float(v), 2), round(float(gap.flatten()[i]), 2),
                  repr(tok.decode([int(ids[int(i) // dt.shape[1], 511 + int(i) % dt.shape[1]])]))) for v, i in zip(worst.values, worst.indices)]
        print(f"RESULT   top dtop (seq,pos,dtop,gap,token): {where}", flush=True)
        print(f"RESULT {name:32s} worst gap {gap.max().item():.3f}  gaps>1: {int((gap > 1).sum())}  >0.5: {int((gap > 0.5).sum())} of {gap.numel()}  "
              f"rms dlogit {rms:.4f}  max |dtop| {top:.3f}", flush=True)


@app.local_entrypoint()
def main():
    run.remote()
