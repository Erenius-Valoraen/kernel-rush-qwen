"""Run our engine on a Modal H100 with full logs.

  modal run modal_bench.py                       # 3 public shapes, timing + correctness
  modal run modal_bench.py --shapes 1x512x32     # pick shapes (BxSxN, comma separated)
  modal run modal_bench.py --check 0             # skip the HF reference replay
  modal run modal_bench.py --env ENGINE_DIAG=0,ENGINE_FORCE_PLAN=fusedpdl
"""

import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "triton==3.1.0", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("transformers==4.51.3", "safetensors==0.5.3", "tokenizers==0.21.1",
                 "huggingface_hub", "numpy", "accelerate")
    .add_local_dir(os.environ.get("ENGINE_DIR") or os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("qwen3-4b-weights", create_if_missing=True)
app = modal.App("kernel-rush-bench", image=image)


@app.function(gpu="H100", timeout=900, volumes={"/weights": vol})
def bench(shapes: str, check: int, env: str, samples: int):
    import sys, time, glob
    for kv in filter(None, env.split(",")):
        k, v = kv.split("=", 1)
        os.environ[k] = v
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf",
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    vol.commit()
    sys.path.insert(0, "/root/engine")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    import urllib.request
    text = ""
    for url in ("https://www.gutenberg.org/cache/epub/1342/pg1342.txt",      # Pride and Prejudice
                "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"):     # War and Peace
        try:
            text += urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "ignore")
        except Exception as e:
            print("corpus download failed", e)
    if len(text) < 100000:
        for f in sorted(glob.glob("/usr/share/common-licenses/*")):
            text += open(f, errors="ignore").read()
    text = text[:3_000_000]
    if os.environ.get("BENCH_CORPUS") == "qa":
        NL = chr(10)
        qs = ["What is the capital of France?", "Name a primary color.", "What is 2 + 2?", "Is water wet?",
              "What day comes after Monday?", "Spell the word cat.", "What is the opposite of hot?"]
        text = "".join("<|im_start|>user" + NL + q + "<|im_end|>" + NL + "<|im_start|>assistant" + NL
                       + ("Yes." if i % 2 else "It is simple.") + "<|im_end|>" + NL
                       for i, q in enumerate(qs * 3000))
    corpus = tok(text, add_special_tokens=False)["input_ids"]
    print(f"corpus tokens: {len(corpus)}  gpu: {torch.cuda.get_device_name(0)}")

    from engine import Engine
    gen = torch.Generator().manual_seed(0)

    def prompts(B, S):
        starts = torch.randint(0, len(corpus) - S - 1, (B,), generator=gen).tolist()
        return [corpus[s:s + S] for s in starts]

    results = {}
    for shp in shapes.split(","):
        B, S, N = (int(v) for v in shp.split("x"))
        t0 = time.perf_counter()
        eng = Engine(path)
        list(eng.generate(prompts(B, S), N))
        torch.cuda.synchronize()
        print(f"[{shp}] load+warmup {time.perf_counter() - t0:.1f}s  mode={eng.state.mode}")
        tot, ttft, outs, ins = [], [], [], []
        for _ in range(samples):
            ids = prompts(B, S)
            t0 = time.perf_counter()
            first = None
            steps = []
            for s in eng.generate(ids, N):
                if first is None:
                    first = time.perf_counter() - t0
                steps.append(s)
            dt = time.perf_counter() - t0
            tot.append(dt); ttft.append(first); outs.append(steps); ins.append(ids)
        tot.sort(); ttft.sort()
        med, f = tot[len(tot) // 2], ttft[len(ttft) // 2]
        print(f"[{shp}] total {med * 1e3:.1f}ms  ttft {f * 1e3:.1f}ms  tpot {(med - f) / max(1, N - 1) * 1e3:.3f}ms  "
              f"tok/s {B * N / med:.1f}  peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB")
        results[shp] = (ins, outs)
        del eng
        torch.cuda.empty_cache()

    if check:
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                   attn_implementation="sdpa").eval().cuda()
        for shp, (all_ins, all_outs) in results.items():
            worst, over1, npos = 0.0, 0, 0
            for ins, outs in zip(all_ins, all_outs):
                S = len(ins[0])
                out = torch.tensor(outs).T
                full = torch.cat([torch.tensor(ins), out], 1).cuda()
                with torch.inference_mode():
                    for b in range(0, full.shape[0], 4):
                        lg = ref(input_ids=full[b:b + 4]).logits[:, S - 1:-1].float()
                        ch = lg.gather(-1, out[b:b + 4].cuda().unsqueeze(-1)).squeeze(-1)
                        gap = lg.max(-1).values - ch
                        worst = max(worst, gap.max().item())
                        over1 += int((gap > 1.0).sum()); npos += gap.numel()
            print(f"[{shp}] correctness: worst gap = {worst:.3f} (limit 2.0), gaps>1.0: {over1} of {npos} positions")


@app.local_entrypoint()
def main(shapes: str = "1x512x32,4x2048x32,16x512x128", check: int = 1, env: str = "", samples: int = 3):
    bench.remote(shapes, check, env, samples)
