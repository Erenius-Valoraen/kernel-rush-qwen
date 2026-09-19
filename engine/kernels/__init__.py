"""Modules your engine imports, vendored beside ``engine.py``.

The archive root is on ``sys.path``, so this package is importable by name —
``from kernels.rmsnorm import rms_norm`` — as long as ``kernels/`` ships inside
the archive. Nothing here is imported by the baseline engine.
"""
