"""Closing the loop: segmentation to actuation, and the feedback path.

The loop is:

    camera sees activity -> segmenter says where -> DMD illuminates there
      -> the sample changes -> the camera sees the new state next frame

Which is the whole point, and the thing that distinguishes this from a
detector that fires once. Nothing here decides *what* to do with a frame in
isolation; each cycle's output changes the input to the next one.

Two pieces live here. ``NumpyController`` is a fallback implementation of one
cycle so the repo runs uncompiled. ``ClosedLoopStage`` is the pipeline stage
that sits downstream of background subtraction and drives whichever controller
is available, then feeds the projected pattern back to the sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np

from .native import HAVE_NATIVE, native
from .quality import QualityReport, to_gray
from .segmentation import NumpySegmenter
from .sources import Frame

__all__ = ["CycleResult", "NumpyController", "make_controller", "ClosedLoopStage",
           "Stimulable"]


@dataclass(frozen=True)
class CycleResult:
    active_pixels: int
    stimulated_pixels: int
    peak_confidence: int
    segment_ms: float
    actuate_ms: float

    @property
    def total_ms(self) -> float:
        return self.segment_ms + self.actuate_ms


class Stimulable(Protocol):
    """A sample that responds to projected light.

    ``SyntheticSource`` implements this. A real rig does not need to: there the
    sample is the sample, and the feedback path is physics rather than a method
    call. The protocol exists so the simulated loop has the same shape as the
    real one instead of a special case threaded through the stage.
    """

    def apply_stimulation(self, pattern: np.ndarray) -> None: ...


class NumpyController:
    """One cycle, in NumPy. Same contract as the native controller."""

    def __init__(self, rows: int, cols: int, segmenter: str = "builtin",
                 activation_threshold: int = 128) -> None:
        if segmenter not in ("builtin", "", "numpy"):
            raise RuntimeError(
                f"segmenter spec {segmenter!r} looks like a model path, but the native "
                "extension is not built. Run scripts/build_native.sh with "
                "-DCLPIPE_ONNXRUNTIME_ROOT=..., or pass 'builtin'."
            )

        self.rows = rows
        self.cols = cols
        self.activation_threshold = activation_threshold
        self.segmenter = "numpy"

        self._segmenter = NumpySegmenter(rows, cols)
        self._pattern = np.zeros((rows, cols), dtype=np.uint8)
        self._scratch = np.zeros((rows, cols), dtype=np.uint8)
        self.patterns_written = 0

    def process(self, frame: np.ndarray, motion: Optional[np.ndarray] = None,
                confidence: Optional[np.ndarray] = None) -> CycleResult:
        import time

        target = self._scratch if confidence is None else confidence

        start = time.perf_counter()
        result = self._segmenter.run(frame, target, self.activation_threshold)
        segment_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        fire = target >= self.activation_threshold
        if motion is not None:
            # The motion mask is a veto: stimulate only where the segmenter says
            # active *and* something actually changed.
            fire &= motion != 0
        np.multiply(fire, 255, out=self._pattern, dtype=np.uint8, casting="unsafe")
        self.patterns_written += 1
        actuate_ms = (time.perf_counter() - start) * 1000.0

        return CycleResult(
            active_pixels=result.active_pixels,
            stimulated_pixels=int(np.count_nonzero(fire)),
            peak_confidence=result.peak_confidence,
            segment_ms=segment_ms,
            actuate_ms=actuate_ms,
        )

    def blank(self) -> None:
        self._pattern.fill(0)
        self.patterns_written += 1

    def last_pattern(self) -> np.ndarray:
        view = self._pattern.view()
        view.flags.writeable = False
        return view


def make_controller(rows: int, cols: int, segmenter: str = "builtin",
                    activation_threshold: int = 128, force_numpy: bool = False) -> Any:
    """Native controller when the extension is built, NumPy otherwise."""
    if HAVE_NATIVE and not force_numpy:
        return native.ClosedLoopController(rows, cols, segmenter, activation_threshold)
    return NumpyController(rows, cols, segmenter, activation_threshold)


class ClosedLoopStage:
    """Pipeline stage: gated frame plus motion mask in, DMD pattern out.

    Sits downstream of ``BackgroundStage``. Its signature matches that stage's
    ``downstream`` hook, so wiring the loop together is one assignment.
    """

    def __init__(
        self,
        segmenter: str = "builtin",
        activation_threshold: int = 128,
        sample: Optional[Stimulable] = None,
        use_motion_veto: bool = True,
        force_numpy: bool = False,
        on_cycle: Optional[Callable[[Frame, CycleResult], None]] = None,
    ) -> None:
        self.segmenter_spec = segmenter
        self.activation_threshold = activation_threshold
        self.sample = sample
        self.use_motion_veto = use_motion_veto
        self.force_numpy = force_numpy
        self.on_cycle = on_cycle

        self.cycles = 0
        self.stimulated_total = 0
        self.segment_ms: list[float] = []
        self.actuate_ms: list[float] = []

        self._controller: Any = None
        self._confidence: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    @property
    def backend(self) -> str:
        if self._controller is None:
            return "uninitialised"
        return self._controller.segmenter

    @property
    def confidence(self) -> Optional[np.ndarray]:
        """Most recent per-pixel activity map. Reused in place; copy to keep it."""
        return self._confidence

    @property
    def pattern(self) -> Optional[np.ndarray]:
        if self._controller is None:
            return None
        return self._controller.last_pattern()

    def latency_summary(self) -> str:
        if not self.segment_ms:
            return "no cycles run"
        segment = sorted(self.segment_ms)
        actuate = sorted(self.actuate_ms)
        median = len(segment) // 2
        p95 = int(0.95 * (len(segment) - 1))
        return (
            f"segment p50={segment[median]:.3f}ms p95={segment[p95]:.3f}ms  "
            f"actuate p50={actuate[median]:.3f}ms p95={actuate[p95]:.3f}ms"
        )

    # ------------------------------------------------------------------ #

    def __call__(self, frame: Frame, motion: Optional[np.ndarray] = None,
                 foreground: int = 0) -> CycleResult:
        gray = self._prepare(frame.data)
        self._ensure_controller(gray.shape)

        veto = motion if (self.use_motion_veto and motion is not None) else None
        raw = self._controller.process(gray, veto, self._confidence)

        result = CycleResult(
            active_pixels=int(raw.active_pixels),
            stimulated_pixels=int(raw.stimulated_pixels),
            peak_confidence=int(raw.peak_confidence),
            segment_ms=float(raw.segment_ms),
            actuate_ms=float(raw.actuate_ms),
        )

        self.cycles += 1
        self.stimulated_total += result.stimulated_pixels
        self.segment_ms.append(result.segment_ms)
        self.actuate_ms.append(result.actuate_ms)

        # The feedback edge. On a real rig this is physics and there is nothing
        # to call; in simulation the sample has to be told what was projected at
        # it, or the loop is not closed and the whole thing is just a detector.
        if self.sample is not None:
            self.sample.apply_stimulation(self._controller.last_pattern())

        if self.on_cycle is not None:
            self.on_cycle(frame, result)
        return result

    def on_rejected_frame(self) -> None:
        """No usable image means no basis for stimulating. Mirrors go dark."""
        if self._controller is not None:
            self._controller.blank()
            if self.sample is not None:
                self.sample.apply_stimulation(self._controller.last_pattern())

    # ------------------------------------------------------------------ #

    def _ensure_controller(self, shape: tuple) -> None:
        if self._controller is not None:
            return
        rows, cols = shape
        self._controller = make_controller(
            rows, cols, self.segmenter_spec, self.activation_threshold,
            force_numpy=self.force_numpy,
        )
        self._confidence = np.zeros((rows, cols), dtype=np.uint8)

    @staticmethod
    def _prepare(image: np.ndarray) -> np.ndarray:
        gray = to_gray(image)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        if not gray.flags["C_CONTIGUOUS"]:
            gray = np.ascontiguousarray(gray)
        return gray
