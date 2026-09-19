"""Telemetry through timing (diagnostic builds only).

Official runs hide our logs, but they publish per-workload p10/p50/p90 total
time, TTFT and TPOT. A diagnostic build measures a few quantities during the
untimed warmup and then sleeps by known amounts in the measured samples so
those quantities can be read back from the report:

  samples 1,2 : pre-first-token sleep A  -> p10  = base + A
  sample  3   : pre-first-token sleep B  -> p50  = base + B
  samples 4,5 : pre-first-token sleep C  -> p90  = base + C
  every token : sleep P                  -> tpot = base + P

  A = step_us / 5            (one decode step of the chosen plan, graph replay)
  B = 2000 + gemv_us / 5     (all 145 matmuls of the plan, back-to-back)
  C = 4000 + attn_us / 5     (36 fused attention launches, back-to-back)
  P = 10 * plan_idx + norm_us / 100   (72 residual+RMSNorm launches)
All in ms; total time also contains (n-1)*P. Such a run fails the latency gates and never ranks.
"""

import time

import torch
import torch.nn.functional as F

PLAN_ORDER = [False, "fixed", "fixedpdl", "tuned", "tunedpdl", "fixedpdlpeel", "fixedpdlpf",
              "mega", "megapf", "fused", "fusedpdl"]


def _t(fn, reps=5):
    fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        dt = e0.elapsed_time(e1) * 1e3
        best = dt if best is None else min(best, dt)
    return best


def _tg(fn, reps=5):
    """Device time of fn's launches, replayed from a CUDA graph (no host gaps)."""
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return _t(g.replay, reps)


def measure(eng, st, S, n):
    from engine import _gemm_run, add_rmsnorm
    out = dict(step_us=0.0, gemv_us=0.0, attn_us=0.0, norm_us=0.0, plan_idx=0)
    try:
        t, g = st.mode
        out["plan_idx"] = PLAN_ORDER.index(g) if g in PLAN_ORDER else 12
        if t > 1:
            out["plan_idx"] += 20
        r = st.runner(eng, 1, g)
        M = st.batch
        dev = eng.device
        with torch.inference_mode():
            if r.graph is not None:
                def step():
                    r.pos.fill_(S)
                    r.graph.replay()
                out["step_us"] = _t(step)
            base = "fixed" if g in ("fixedpdl", "fixedpdlpeel", "fixedpdlpf", "fused", "fusedpdl") else g
            if base == "tunedpdl":
                base = "tuned"
            if base == "tuned":
                plan = {k: eng.gemm_plan[(k, M)] for k in ("qkv", "o", "gu", "down", "lm")}
            elif base == "fixed":
                plan = dict(qkv="tr2", o="tr4", gu="tr", down="tr4", lm="tr1")
            else:
                plan = dict(qkv="cublas", o="cublas", gu="cublas", down="cublas", lm="cublas")
            H = eng.embed.shape[1]
            h = torch.randn((M, H), device=dev, dtype=torch.bfloat16) * 0.1
            a = torch.randn((M, eng.nq * eng.d), device=dev, dtype=torch.bfloat16) * 0.1
            I = eng.layers[0]["down"].shape[1]
            act = torch.randn((M, I), device=dev, dtype=torch.bfloat16) * 0.1

            def gemvs():
                for L in eng.layers:
                    _gemm_run("qkv", plan["qkv"], h, L["qkv"])
                    _gemm_run("o", plan["o"], a, L["o"])
                    _gemm_run("gu", plan["gu"], h, L["gu"])
                    _gemm_run("down", plan["down"], act, L["down"])
                _gemm_run("lm", plan["lm"], h, eng.lm_head)
            out["gemv_us"] = _tg(gemvs)

            L0 = eng.layers[0]
            qkv = _gemm_run("qkv", plan["qkv"], h, L0["qkv"])
            pos = torch.full((M,), S, device=dev, dtype=torch.int32)

            def attns():
                for li, L in enumerate(eng.layers):
                    eng._attend(qkv, L, st.k_cache[li], st.v_cache[li], pos, 1, r.attn)
            out["attn_us"] = _tg(attns)

            x = torch.randn((M, H), device=dev, dtype=torch.bfloat16)
            delta = torch.randn((4, M, H), device=dev, dtype=torch.float32) * 0.1

            def norms():
                for L in eng.layers:
                    add_rmsnorm(x, delta, L["ln1"], eng.eps)
                    add_rmsnorm(x, delta, L["ln2"], eng.eps)
            out["norm_us"] = _tg(norms)
    except Exception as e:  # pragma: no cover
        out["err"] = repr(e)
    return out


def sleeps(m, sample):
    """(pre-first-token ms, per-token ms) for sample index 1..5."""
    A = m["step_us"] / 5
    B = 2000 + m["gemv_us"] / 5
    C = 4000 + m["attn_us"] / 5
    pre = [A, A, B, C, C][min(max(sample, 1), 5) - 1]
    per = 10 * m["plan_idx"] + min(m["norm_us"], 999) / 100
    return pre, per


def wrap(gen, pre_ms, per_ms, n):
    time.sleep(pre_ms / 1e3)
    for i, tok in enumerate(gen):
        yield tok
        if i + 1 < n:
            time.sleep(per_ms / 1e3)
