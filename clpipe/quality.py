"""Diagnostic analysis: decide whether a frame is worth processing at all.

The check is deliberately cheap. It runs on every frame in the hot path, so
anything expensive here eats the budget that the inference stage needs. Mean
grayscale intensity catches the two failure modes that actually occur on an
imaging rig: the shutter did not open (near-zero) and the illumination spiked
or the gain ran away (near-saturation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .sources import Frame

__all__ = ["QualityReport", "IntensityGate", "QualityStage", "to_gray"]

# OpenCV hands back BGR, so the luminance weights are reversed relative to the
# usual RGB ordering. Used only by the NumPy fallback path.
_BGR_LUMA = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Single-channel view of ``image``, using OpenCV when it is available."""
    if image.ndim == 2:
        return image
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"unsupported frame shape {image.shape}")

    try:
        import cv2
    except ImportError:
        return image[..., :3].astype(np.float32) @ _BGR_LUMA

    code = cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(image, code)


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    mean_intensity: float
    reason: Optional[str] = None


class IntensityGate:
    """Rejects frames whose mean intensity falls outside ``[low, high]``.

    Defaults of 10 and 245 on an 8-bit sensor: below 10 the frame carries no
    signal to segment, above 245 the sensor is clipped and the segmentation
    output would be meaningless rather than merely noisy.
    """

    def __init__(self, low: float = 10.0, high: float = 245.0) -> None:
        if low >= high:
            raise ValueError("low must be strictly less than high")
        self.low = low
        self.high = high

    def __call__(self, frame: Frame) -> QualityReport:
        return self.evaluate(frame.data)

    def evaluate(self, image: np.ndarray) -> QualityReport:
        mean = float(to_gray(image).mean())
        if mean < self.low:
            return QualityReport(False, mean, "blank")
        if mean > self.high:
            return QualityReport(False, mean, "overexposed")
        return QualityReport(True, mean, None)


class QualityStage:
    """Pipeline stage: gate frames, forward the survivors, tally both.

    ``downstream`` is the composition point for the rest of the system. In this
    part of the build it is usually ``None`` or a print callback; later it
    becomes the C++ inference call.
    """

    def __init__(
        self,
        gate: Optional[IntensityGate] = None,
        downstream: Optional[Callable[[Frame, QualityReport], None]] = None,
        on_reject: Optional[Callable[[Frame, QualityReport], None]] = None,
    ) -> None:
        self.gate = gate or IntensityGate()
        self.downstream = downstream
        # Rejection is not a no-op downstream. A closed-loop system holding a
        # stale pattern on the mirrors while it cannot see the sample is worse
        # than one that goes dark, so the hook exists to let it go dark.
        self.on_reject = on_reject
        self.accepted = 0
        self.rejected = 0
        self.reasons: dict[str, int] = {}

    def __call__(self, frame: Frame) -> QualityReport:
        report = self.gate(frame)
        if not report.ok:
            self.rejected += 1
            self.reasons[report.reason or "unknown"] = (
                self.reasons.get(report.reason or "unknown", 0) + 1
            )
            if self.on_reject is not None:
                self.on_reject(frame, report)
            return report

        self.accepted += 1
        if self.downstream is not None:
            self.downstream(frame, report)
        return report
