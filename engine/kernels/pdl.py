"""Programmatic Dependent Launch (PDL) for our Triton kernels inside CUDA graphs.

Between two back-to-back kernels, a normal graph edge makes the second wait
until the first has fully drained: launch latency + the first kernel's tail
wave + the second's ramp-up all sit on the critical path (~3 us each, ~260
boundaries per decode step). With a *programmatic* edge (CUDA >= 12.3), the
dependent grid may launch as soon as every block of its predecessor has
executed `griddepcontrol.launch_dependents`; it then runs its prologue and
blocks in `griddepcontrol.wait` until the predecessor has completed and its
memory is visible. Results are identical; only the gap shrinks.

PyTorch 2.5 exposes neither, so:
  * kernels carry a PDL constexpr: `pdl_wait()` is the first thing they do
    (every later load of predecessor data is ordered after it) followed by
    `pdl_launch()`;
  * while torch captures the graph, `before_launch()` rewrites the capture
    stream's current dependency into a programmatic edge (driver API
    cuStreamGetCaptureInfo_v3 / cuStreamUpdateCaptureDependencies_v2) - only
    when that dependency is the node our previous PDL kernel produced, so
    cuBLAS/torch nodes keep ordinary edges.
Everything is off unless `enable()` succeeds (Hopper, driver >= 12.3).
"""

import ctypes
import os

import torch
import triton
import triton.language as tl

_ACTIVE = False
_lib = None
_last_node = None


class _EdgeData(ctypes.Structure):
    _fields_ = [("from_port", ctypes.c_ubyte), ("to_port", ctypes.c_ubyte),
                ("type", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte * 5)]


_CAPTURE_ACTIVE = 1
_PORT_PROGRAMMATIC = 1
_TYPE_PROGRAMMATIC = 1
_SET_DEPS = 1


def enable():
    """Try to switch PDL on; returns True if available."""
    global _ACTIVE, _lib
    if os.environ.get("ENGINE_NO_PDL") == "1" or not torch.cuda.is_available():
        return False
    try:
        if torch.cuda.get_device_capability()[0] < 9:
            return False
        lib = ctypes.CDLL("libcuda.so.1")
        ver = ctypes.c_int()
        lib.cuDriverGetVersion(ctypes.byref(ver))
        if ver.value < 12030:
            return False
        lib.cuStreamGetCaptureInfo_v3.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
            ctypes.POINTER(ctypes.POINTER(_EdgeData)), ctypes.POINTER(ctypes.c_size_t)]
        lib.cuStreamUpdateCaptureDependencies_v2.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_EdgeData),
            ctypes.c_size_t, ctypes.c_uint]
        _lib = lib
        # the PTX must compile and run here, or every kernel would break
        x = torch.zeros(64, device="cuda", dtype=torch.int32)
        _selftest_kernel[(1,)](x, PDL=True)
        torch.cuda.synchronize()
        if int(x.sum().item()) != 64:
            _lib = None
            return False
        _ACTIVE = False     # edge rewriting is switched on per capture
        return True
    except Exception:
        _ACTIVE = False
        _lib = None
        return False


def active():
    return _ACTIVE


def set_active(flag):
    """Enable/disable edge rewriting (kernels compiled with PDL stay correct
    either way: griddepcontrol.wait is a no-op without a programmatic edge)."""
    global _ACTIVE, _last_node
    _ACTIVE = bool(flag) and _lib is not None
    _last_node = None


def _deps():
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    status = ctypes.c_int()
    cid = ctypes.c_uint64()
    graph = ctypes.c_void_p()
    deps = ctypes.POINTER(ctypes.c_void_p)()
    edges = ctypes.POINTER(_EdgeData)()
    n = ctypes.c_size_t()
    r = _lib.cuStreamGetCaptureInfo_v3(stream, ctypes.byref(status), ctypes.byref(cid),
                                       ctypes.byref(graph), ctypes.byref(deps),
                                       ctypes.byref(edges), ctypes.byref(n))
    if r != 0 or status.value != _CAPTURE_ACTIVE:
        return stream, None
    return stream, [deps[i] for i in range(n.value)]


def before_launch():
    """Called right before launching a PDL-compiled kernel."""
    if not _ACTIVE or not torch.cuda.is_current_stream_capturing():
        return
    stream, deps = _deps()
    if deps is None or len(deps) != 1 or _last_node is None or deps[0] != _last_node:
        return
    node = (ctypes.c_void_p * 1)(deps[0])
    edge = (_EdgeData * 1)()
    edge[0].from_port = _PORT_PROGRAMMATIC
    edge[0].to_port = 0
    edge[0].type = _TYPE_PROGRAMMATIC
    _lib.cuStreamUpdateCaptureDependencies_v2(stream, node, edge, 1, _SET_DEPS)


def after_launch():
    """Called right after launching a PDL-compiled kernel: remember its node."""
    global _last_node
    if not _ACTIVE or not torch.cuda.is_current_stream_capturing():
        return
    _, deps = _deps()
    _last_node = deps[0] if deps and len(deps) == 1 else None


def compiled():
    """Whether kernels should be compiled with the PDL instructions."""
    return _lib is not None


@triton.jit
def pdl_wait(PDL: tl.constexpr):
    """Block until the programmatic predecessor finished (no-op otherwise).
    Returns an opaque 0 that callers add to their base offsets, so no load of
    predecessor data can be hoisted above the wait."""
    if PDL:
        z = tl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;", "=r,r", [tl.full([1], 0, tl.int32)],
            dtype=tl.int32, is_pure=False, pack=1)
        return tl.sum(z, axis=0)
    else:
        return 0


@triton.jit
def pdl_launch(PDL: tl.constexpr):
    """Allow the dependent grid to start launching."""
    if PDL:
        tl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;", "=r,r",
            [tl.full([1], 0, tl.int32)], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _selftest_kernel(x_ptr, PDL: tl.constexpr):
    z = pdl_wait(PDL)
    pdl_launch(PDL)
    offs = tl.arange(0, 64) + z
    tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1)
