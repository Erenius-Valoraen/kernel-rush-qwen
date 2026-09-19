"""One-off microbenchmarks of our own kernels on synthetic inputs (no prompts)."""

import time

import torch
import torch.nn.functional as F


def _t(fn, iters=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def _graph_t(fn, iters=20):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return _t(g.replay, iters)


def run(eng):
    from kernels.gemv import gemv, gemv_swiglu
    import sys
    mod = sys.modules[type(eng).__module__]
    SPLIT_QKV, SPLIT_O, SPLIT_DOWN, _State = mod.SPLIT_QKV, mod.SPLIT_O, mod.SPLIT_DOWN, mod._State
    out = []
    L = eng.layers[0]
    dev = eng.device
    shapes = [("qkv", L["qkv"], SPLIT_QKV), ("o", L["o"], SPLIT_O),
              ("gu", L["gu"], 1), ("dn", L["down"], SPLIT_DOWN), ("lm", eng.lm_head, 1)]
    for M in (1, 16, 64):
        row = [f"M{M}"]
        for name, w, sp in shapes:
            x = torch.randn(M, w.shape[1], device=dev, dtype=torch.bfloat16)
            tb = _t(lambda: F.linear(x, w))
            if name == "gu":
                tg = _t(lambda: gemv_swiglu(x, w))
            else:
                tg = _t(lambda: gemv(x, w, sp))
            gb = w.numel() * 2 / 1e3
            row.append(f"{name}:cb{tb:.0f}/tr{tg:.0f}us({gb / min(tb, tg):.0f}GB/s)")
        out.append(" ".join(row))
    # empty graph round trip (host/driver overhead under the sandbox)
    z = torch.zeros(1, device=dev)
    out.append(f"tinygraph {_graph_t(lambda: z.add_(1)):.1f}us")
    t0 = time.perf_counter()
    for _ in range(50):
        e = torch.cuda.Event()
        e.record()
        e.synchronize()
    out.append(f"evsync {(time.perf_counter() - t0) / 50 * 1e6:.1f}us")
    # full decode steps
    for B, ctx in ((1, 600), (16, 600), (4, 2100)):
        st = _State(eng, B, ctx + 128)
        for g in (False, True):
            r = st.runner(eng, 1, g)
            r.pos.fill_(ctx)
            tstep = _t(r.graph.replay, 20) if r.graph is not None else -1
            out.append(f"step B{B} ctx{ctx} gemv={int(g)}: {tstep:.0f}us")
        for T in (4,):
            r = st.runner(eng, T, True)
            r.inbuf[:, T] = ctx
            out.append(f"spec T{T} B{B}: {_t(r.graph.replay, 20):.0f}us")
        del st
        torch.cuda.empty_cache()
    # prefill
    st = _State(eng, 4, 2048 + 64)
    ids = torch.randint(0, 1000, (4, 2048), device=dev)
    out.append(f"prefill4x2048 {_t(lambda: eng._prefill_eager(ids, st), 3) / 1e3:.1f}ms")
    return " | ".join(out)
