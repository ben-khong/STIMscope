"""Locate the compiled extension, or carry on without it.

The repo has to be useful on a machine where nobody has run CMake — a fresh
clone, a CI box, a reviewer skimming the code. So the import is attempted and
the failure is recorded rather than raised, and every caller that needs the
native path checks ``HAVE_NATIVE`` first. ``clpipe.background`` supplies a NumPy
implementation of the identical algorithm for when it is missing.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["HAVE_NATIVE", "IMPORT_ERROR", "backend_name", "describe", "native"]

native = None
IMPORT_ERROR: Optional[str] = None

try:
    from . import _native as native  # type: ignore[no-redef]
except ImportError as exc:  # not built yet, or built for another interpreter
    IMPORT_ERROR = str(exc)

HAVE_NATIVE = native is not None


def backend_name() -> str:
    """Which implementation ``clpipe.background`` will pick: cuda, cpp, or numpy."""
    if not HAVE_NATIVE:
        return "numpy"
    return "cuda" if native.cuda_available() else "cpp"


def describe() -> str:
    """One human-readable line about the build, for the CLI and build script."""
    if not HAVE_NATIVE:
        return (
            f"native extension not built ({IMPORT_ERROR}); "
            "using the NumPy reference. Run scripts/build_native.sh to compile it."
        )

    if native.cuda_available():
        detail = "CUDA backend active"
    elif native.built_with_cuda():
        detail = "compiled with CUDA but no device visible, running on CPU"
    else:
        detail = "CPU-only build"

    if native.built_with_onnxruntime():
        detail += f", ONNX Runtime {native.onnxruntime_version()}"
    else:
        detail += ", analytic segmenter only"

    return f"native extension v{native.__version__} loaded: {detail}"
