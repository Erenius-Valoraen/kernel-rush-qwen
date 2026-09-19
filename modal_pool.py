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

    def simulate(prompt, out, pool, K=4, NG=4):
        """tokens emitted per verify step with longest-suffix n-gram drafting (K drafts)."""
        hist = list(prompt); steps = 0; i = 0
        index = {}
        def add(seq, upto_from=0):
            pass
        while i < len(out):
            # draft
            ctx = hist
            draft = []
            for n in range(NG, 0, -1):
                key = tuple(ctx[-n:])
                pos = None
                # own history (latest occurrence before end)
                for j in range(len(ctx) - n - 1, -1, -1):
                    if tuple(ctx[j:j + n]) == key:
                        pos = ("own", j + n); break
                if pos is None and pool is not None:
                    p = pool.get(key)
                    if p is not None:
                        pos = ("pool", p)
                if pos is not None:
                    if pos[0] == "own":
                        draft = ctx[pos[1]:pos[1] + K]
                    else:
                        draft = list(pos[1][:K])
                    break
            a = 0
            while a < len(draft) and i + a < len(out) and draft[a] == out[i + a]:
                a += 1
            take = min(a + 1, len(out) - i)
            hist += out[i:i + take]; i += take; steps += 1
        return len(out) / steps

    def build_pool(seqs, NG=4, K=8):
        pool = {}
        for s in seqs:
            for n in range(1, NG + 1):
                for j in range(len(s) - n):
                    pool.setdefault(tuple(s[j:j + n]), tuple(s[j + n:j + n + K]))
        return pool

    import statistics
    own = [simulate(p, o, None) for p, o in zip(prompts[-16:], outs[-16:])]
    pool = build_pool(outs[:-16])
    shared = [simulate(p, o, pool) for p, o in zip(prompts[-16:], outs[-16:])]
    print(f"S={S} N={N}: tokens/step own-history mean {statistics.mean(own):.3f} min {min(own):.2f} | "
          f"+pool of {len(outs) - 16} earlier outputs mean {statistics.mean(shared):.3f} min {min(shared):.2f}")
    for n_first in (16, 32, 64):
        o2 = [simulate(p, o[:n_first], None) for p, o in zip(prompts[-16:], outs[-16:])]
        s2 = [simulate(p, o[:n_first], pool) for p, o in zip(prompts[-16:], outs[-16:])]
        print(f"   first {n_first} tokens: own {statistics.mean(o2):.3f}  +pool {statistics.mean(s2):.3f} (min {min(s2):.2f})")


@app.local_entrypoint()
def main(s: int = 512, n: int = 128):
    run.remote(s, n)
