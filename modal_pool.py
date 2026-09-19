"""Offline estimate: n-gram draft acceptance with own history vs a shared pool of other outputs."""
import os
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3-4B-Instruct-2507"; REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1", "huggingface_hub", "numpy", "accelerate")
    .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"]))
vol = modal.Volume.from_name("qwen3-4b-weights", create_if_missing=True)
app = modal.App("kernel-rush-pool", image=image)


@app.function(gpu="H100", timeout=1200, volumes={"/weights": vol})
def run(S: int, N: int):
    import sys, urllib.request, torch
    os.environ["ENGINE_NO_CALIBRATE"] = "1"
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf", allow_patterns=["*.json", "*.safetensors", "*.txt"])
    sys.path.insert(0, "/root/engine")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    text = ""
    for url in ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
                "https://www.gutenberg.org/cache/epub/84/pg84.txt"):
        text += urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
    corpus = tok(text[:4_000_000], add_special_tokens=False)["input_ids"]
    from engine import Engine
    eng = Engine(path)
    g = torch.Generator().manual_seed(1)
    B, rounds = 16, 4
    prompts, outs = [], []
    for _ in range(rounds):
        starts = torch.randint(0, len(corpus) - S - 1, (B,), generator=g).tolist()
        ids = [corpus[s:s + S] for s in starts]
        steps = list(eng.generate(ids, N))
        o = torch.tensor(steps).T.tolist()
        prompts += ids; outs += o
    print("sample outputs:")
    for o in outs[:4]:
        print("   ", repr(tok.decode(o[:48])))

    def candidates(ctx, pool, C, D, NG=4):
        """Up to C chains (distinct first token) of depth D: latest matches of the longest suffixes."""
        cands, seen = [], set()
        for n in range(NG, 0, -1):
            key = tuple(ctx[-n:])
            j = len(ctx) - n - 1
            while j >= 0 and len(cands) < C:
                if tuple(ctx[j:j + n]) == key:
                    ch = ctx[j + n:j + n + D]
                    if ch and ch[0] not in seen:
                        seen.add(ch[0]); cands.append(ch)
                j -= 1
            if pool is not None and len(cands) < C:
                for ch in pool.get(key, ()):
                    if ch[0] not in seen and len(cands) < C:
                        seen.add(ch[0]); cands.append(list(ch[:D]))
            if len(cands) >= C:
                break
        return cands

    def simulate(prompt, out, pool, C, D):
        hist = list(prompt); i = 0; steps = 0
        while i < len(out):
            best = 0
            for ch in candidates(hist, pool, C, D):
                a = 0
                while a < len(ch) and i + a < len(out) and ch[a] == out[i + a]:
                    a += 1
                best = max(best, a)
            take = min(best + 1, len(out) - i)
            hist += out[i:i + take]; i += take; steps += 1
        return steps

    def build_pool(seqs, NG=4, K=8, W=4):
        pool = {}
        for s_ in seqs:
            for n in range(1, NG + 1):
                for j in range(len(s_) - n):
                    lst = pool.setdefault(tuple(s_[j:j + n]), [])
                    ch = tuple(s_[j + n:j + n + K])
                    if ch and len(lst) < W and all(c[0] != ch[0] for c in lst):
                        lst.append(ch)
        return pool

    import statistics
    test_p, test_o = prompts[-16:], outs[-16:]
    pool = build_pool(outs[:-16])
    for nn in (32, 128):
        print(f"--- first {nn} output tokens, S={S}; tokens/step: mean over seqs | batch-of-16 lockstep (slowest seq)")
        for C, D in ((1, 3), (1, 7), (2, 3), (3, 2), (4, 3), (7, 1), (8, 3), (16, 3)):
            for nm, pl in (("own", None), ("own+pool", pool)):
                st_ = [simulate(p, o[:nn], pl, C, D) for p, o in zip(test_p, test_o)]
                print(f"   C={C:2d} D={D} slots={C * D:2d} {nm:9s}: mean {nn / statistics.mean(st_):.3f} | lockstep {nn / max(st_):.3f}")


@app.local_entrypoint()
def main(s: int = 512, n: int = 128):
    run.remote(s, n)
