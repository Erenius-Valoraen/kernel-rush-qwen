"""One short H100 run: measure vLLM and HF Transformers greedy decode at the three
public workloads, tokens/sec = batch * output / total generation time (prefill+decode),
median of 3. Reuses the cached weights volume. Our engine's numbers come from the
official judge (same metric), and this run's HF numbers cross-check against it.
"""
import modal

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REV = "cdbee75f17c01a7cc42f958dc650907174af0554"
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("vllm", "transformers==4.51.3", "huggingface_hub"))
vol = modal.Volume.from_name("qwen3-4b-weights")
app = modal.App("kernel-rush-compare", image=image)

SHAPES = [(1, 512, 32), (4, 2048, 32), (16, 512, 128)]


@app.function(gpu="H100", timeout=900, volumes={"/weights": vol})
def run():
    import time, glob, os, torch

    snap = glob.glob("/weights/hf/**/config.json", recursive=True)
    path = os.path.dirname(snap[0]) if snap else MODEL
    print("model path:", path, flush=True)
    g = torch.Generator().manual_seed(0)
    prompts = {S: [torch.randint(0, 151000, (S,), generator=g).tolist() for _ in range(max(b for b, _, _ in SHAPES))]
               for S in {s for _, s, _ in SHAPES}}

    def tps(fn, B, N, reps=3):
        fn()  # warmup
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            fn(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return B * N / (sorted(ts)[len(ts) // 2])

    results = {}

    # ---- HuggingFace Transformers (greedy) ----
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                              attn_implementation="sdpa").eval().cuda()
    for B, S, N in SHAPES:
        ids = torch.tensor([prompts[S][i] for i in range(B)], device="cuda")
        def go():
            with torch.inference_mode():
                hf.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False,
                            use_cache=True, pad_token_id=0)
        results[("hf", B, S, N)] = tps(go, B, N)
        print(f"RESULT hf   B={B} S={S} N={N}: {results[('hf', B, S, N)]:.0f} tok/s", flush=True)
    del hf
    import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- vLLM (greedy, default settings) ----
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=4096, disable_log_stats=True)
        for B, S, N in SHAPES:
            sp = SamplingParams(temperature=0.0, max_tokens=N, min_tokens=N, ignore_eos=True)
            batch = [{"prompt_token_ids": prompts[S][i]} for i in range(B)]
            def go():
                llm.generate(batch, sp, use_tqdm=False)
            results[("vllm", B, S, N)] = tps(go, B, N)
            print(f"RESULT vllm B={B} S={S} N={N}: {results[('vllm', B, S, N)]:.0f} tok/s", flush=True)
    except Exception as e:
        print("VLLM FAILED:", repr(e)[:500], flush=True)

    print("SUMMARY", {f"{k[0]}_{k[1]}x{k[2]}x{k[3]}": round(v) for k, v in results.items()}, flush=True)


@app.local_entrypoint()
def main():
    run.remote()
