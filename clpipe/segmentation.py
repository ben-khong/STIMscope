"""NumPy reference for the analytic segmenter.

Mirrors ``BoxContrastSegmenter`` in cpp/src/segmenter.cpp operation for
operation, including the integer divisions. That is what lets
tests/test_segmentation.py assert *exact* equality between the two rather than a
tolerance — and an exact assertion is worth engineering for, because a tolerance
quietly absorbs the class of bug where the two implementations drift apart by a
little bit on every frame.

It also keeps the repo runnable with nothing compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["SegmentationResult", "NumpySegmenter"]


@dataclass(frozen=True)
class SegmentationResult:
    active_pixels: int
    peak_confidence: int


class NumpySegmenter:
    """Difference of box means over an integral image.

    Responds to compact bright structure and ignores slow illumination
    gradients across the field, which is the useful half of what a trained
    activity segmenter does and is entirely transparent about how it got there.
    """

    def __init__(self, rows: int, cols: int, fine_radius: int = 2, coarse_radius: int = 8,
                 gain: int = 4) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0 <= fine_radius < coarse_radius:
            raise ValueError("require 0 <= fine_radius < coarse_radius")
        if gain <= 0:
            raise ValueError("gain must be positive")

        self.rows = rows
        self.cols = cols
        self.fine_radius = fine_radius
        self.coarse_radius = coarse_radius
        self.gain = gain
        self.name = "numpy"

        # Box bounds depend only on geometry, so resolve them once.
        self._fine = self._box_indices(fine_radius)
        self._coarse = self._box_indices(coarse_radius)

    def _box_indices(self, radius: int) -> Tuple[np.ndarray, ...]:
        r = np.arange(self.rows)
        c = np.arange(self.cols)
        r0 = np.maximum(0, r - radius)
        r1 = np.minimum(self.rows - 1, r + radius)
        c0 = np.maximum(0, c - radius)
        c1 = np.minimum(self.cols - 1, c + radius)
        area = ((r1 - r0 + 1)[:, None] * (c1 - c0 + 1)[None, :]).astype(np.int64)
        return r0, r1, c0, c1, area

    @staticmethod
    def _box_mean(integral: np.ndarray, box: Tuple[np.ndarray, ...]) -> np.ndarray:
        r0, r1, c0, c1, area = box
        total = (
            integral[np.ix_(r1 + 1, c1 + 1)]
            - integral[np.ix_(r0, c1 + 1)]
            - integral[np.ix_(r1 + 1, c0)]
            + integral[np.ix_(r0, c0)]
        )
        # Floor division on non-negative values matches C++ integer division,
        # which truncates toward zero.
        return total // area

    def run(self, frame: np.ndarray, confidence: np.ndarray,
            activation_threshold: int = 128) -> SegmentationResult:
        if frame.shape != (self.rows, self.cols):
            raise ValueError("frame shape does not match the segmenter's geometry")
        if confidence.shape != (self.rows, self.cols) or confidence.dtype != np.uint8:
            raise ValueError("confidence must be a uint8 array matching the frame geometry")

        integral = np.zeros((self.rows + 1, self.cols + 1), dtype=np.int64)
        integral[1:, 1:] = frame.astype(np.int64).cumsum(axis=0).cumsum(axis=1)

        fine = self._box_mean(integral, self._fine)
        coarse = self._box_mean(integral, self._coarse)

        response = np.clip((fine - coarse) * self.gain, 0, 255)
        confidence[...] = response.astype(np.uint8)

        return SegmentationResult(
            active_pixels=int(np.count_nonzero(confidence >= activation_threshold)),
            peak_confidence=int(confidence.max()),
        )
