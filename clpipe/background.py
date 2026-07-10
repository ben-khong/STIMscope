"""Background subtraction: the NumPy reference, the backend picker, and the stage.

``NumpyBackgroundSubtractor`` mirrors ``background_apply_cpu`` in C++ line for
line. It exists for two reasons: the repo stays runnable on a machine that has
never run CMake, and the native backends have something to be checked against
that is obviously correct by inspection.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .native import HAVE_NATIVE, native
from .quality import QualityReport, to_gray
from .sources import Frame

__all__ = [
    "NumpyBackgroundSubtractor",
    "make_subtractor",
    "BackgroundStage",
]


class NumpyBackgroundSubtractor:
    """Vectorised NumPy version of the running-average model.

    Note that this is *not* the honest "before" baseline for the CUDA
    comparison: NumPy dispatches to compiled, often SIMD, loops underneath, so
    it sits somewhere between the scalar C++ loop and the GPU. The benchmark
    reports all three separately for exactly that reason.
    """

    def __init__(self, rows: int, cols: int, alpha: float = 0.05, threshold: int = 25) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must lie in [0, 255]")

        self.rows = rows
        self.cols = cols
        self.alpha = np.float32(alpha)
        self.threshold = np.float32(threshold)
        self.backend = "numpy"

        self._model = np.zeros((rows, cols), dtype=np.float32)
        self._seeded = False

    @property
    def seeded(self) -> bool:
        return self._seeded

    @property
    def model(self) -> np.ndarray:
        return self._model

    def reset(self) -> None:
        self._model.fill(0.0)
        self._seeded = False

    def apply(self, frame: np.ndarray, mask: np.ndarray) -> int:
        if frame.shape != (self.rows, self.cols):
            raise ValueError("frame shape does not match the subtractor's geometry")
        if mask.shape != (self.rows, self.cols) or mask.dtype != np.uint8:
            raise ValueError("mask must be a uint8 array matching the frame geometry")

        pixels = frame.astype(np.float32)

        # First frame seeds the model: with a zeroed model every pixel would
        # read as foreground, and the loop would act on a mask of everything.
        if not self._seeded:
            self._model[...] = pixels
            mask.fill(0)
            self._seeded = True
            return 0

        delta = pixels - self._model
        foreground = np.abs(delta) > self.threshold

        np.multiply(foreground, 255, out=mask, dtype=np.uint8, casting="unsafe")
        # Model held still under foreground pixels, matching the C++ and CUDA paths.
        self._model += np.where(foreground, np.float32(0.0), self.alpha * delta)

        return int(np.count_nonzero(foreground))


def make_subtractor(
    rows: int,
    cols: int,
    alpha: float = 0.05,
    threshold: int = 25,
    prefer_cuda: bool = True,
    force_backend: Optional[str] = None,
) -> Any:
    """Pick the best available implementation.

    ``force_backend`` is for the benchmark and the parity tests: "numpy", "cpp"
    and "cuda" each pin one path so the three can be timed against each other.
    """
    if force_backend == "numpy":
        return NumpyBackgroundSubtractor(rows, cols, alpha, threshold)

    if force_backend in ("cpp", "cuda"):
        if not HAVE_NATIVE:
            raise RuntimeError(
                f"backend {force_backend!r} requested but the native extension is not built"
            )
        if force_backend == "cuda" and not native.cuda_available():
            raise RuntimeError("CUDA backend requested but no device is available")
        return native.BackgroundSubtractor(
            rows, cols, alpha, threshold, prefer_cuda=(force_backend == "cuda")
        )

    if force_backend is not None:
        raise ValueError(f"unknown backend {force_backend!r}")

    if HAVE_NATIVE:
        return native.BackgroundSubtractor(rows, cols, alpha, threshold, prefer_cuda)
    return NumpyBackgroundSubtractor(rows, cols, alpha, threshold)


class BackgroundStage:
    """Pipeline stage: gated frame in, foreground mask out.

    Sits downstream of ``QualityStage`` and upstream of segmentation. The mask
    buffer is allocated once and reused for every frame — a per-frame
    allocation inside a real-time loop is a source of jitter that shows up as
    p99 latency long before it shows up as a mean.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        threshold: int = 25,
        force_backend: Optional[str] = None,
        downstream: Optional[Callable[[Frame, np.ndarray, int], None]] = None,
    ) -> None:
        self.alpha = alpha
        self.threshold = threshold
        self.force_backend = force_backend
        self.downstream = downstream

        self.frames = 0
        self.foreground_total = 0
        self._subtractor: Any = None
        self._mask: Optional[np.ndarray] = None

    @property
    def backend(self) -> str:
        if self._subtractor is None:
            return "uninitialised"
        return getattr(self._subtractor, "backend", "unknown")

    @property
    def mask(self) -> Optional[np.ndarray]:
        """The most recent foreground mask. Reused in place, so copy it to keep it."""
        return self._mask

    @property
    def mean_foreground_fraction(self) -> float:
        if self.frames == 0 or self._mask is None:
            return 0.0
        return self.foreground_total / (self.frames * self._mask.size)

    def __call__(self, frame: Frame, report: Optional[QualityReport] = None) -> int:
        gray = self._prepare(frame.data)

        if self._subtractor is None:
            rows, cols = gray.shape
            self._subtractor = make_subtractor(
                rows, cols, self.alpha, self.threshold, force_backend=self.force_backend
            )
            self._mask = np.zeros((rows, cols), dtype=np.uint8)

        count = int(self._subtractor.apply(gray, self._mask))
        self.frames += 1
        self.foreground_total += count

        if self.downstream is not None:
            self.downstream(frame, self._mask, count)
        return count

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        """Hand the native side a 2D uint8 buffer it can borrow.

        A monochrome scientific camera already delivers exactly that, so this is
        a no-op and the frame crosses into C++ untouched. A colour camera cannot
        avoid a copy — the grayscale conversion has to write somewhere.
        """
        gray = to_gray(image)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        if not gray.flags["C_CONTIGUOUS"]:
            gray = np.ascontiguousarray(gray)
        return gray
