"""Profile prefill on a Modal H100: modal run modal_prof.py --b 8 --s 2048"""
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
    .add_local_dir(os.path.join(HERE, "engine"), "/root/engine", ignore=["__pycache__"])
)
vol = modal.Volume.from_name("qwen3-4b-weights", create_if_missing=True)

app = modal.App("kernel-rush-prof", image=image)


@app.function(gpu="H100", timeout=900, volumes={"/weights": vol})
def prof(b: int, s: int, env: str):
    import sys
    for kv in filter(None, env.split(",")):
        k, v = kv.split("=", 1)
        os.environ[k] = v
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REV, cache_dir="/weights/hf",
                             allow_patterns=["*.json", "*.safetensors", "*.txt"])
    sys.path.insert(0, "/root/engine")
    from engine import Engine
    eng = Engine(path)
    ids = torch.randint(0, 100000, (b, s), device="cuda")
    with torch.inference_mode():
        st = eng._get_state(b, s + 64)
        for _ in range(2):
            eng._prefill_eager(ids, st)
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(3):
            eng._prefill_eager(ids, st)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 3
        print(f"prefill {b}x{s}: {dt * 1e3:.1f} ms  ({b * s / dt / 1e3:.1f}k tok/s)")
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
            eng._prefill_eager(ids, st)
            torch.cuda.synchronize()
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=70))


@app.local_entrypoint()
def main(b: int = 8, s: int = 2048, env: str = ""):
    prof.remote(b, s, env)
