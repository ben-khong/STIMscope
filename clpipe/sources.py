"""Frame sources: where the pipeline's ingest stage gets its data.

Two implementations ship here:

``SyntheticSource``
    A deterministic, dependency-light stand-in for a scientific camera. It
    renders drifting Gaussian blobs (a crude stand-in for tissue activity) on a
    dim background, and deliberately emits a configurable fraction of blank and
    overexposed frames so the diagnostic stage downstream has something to
    reject. This is what lets the whole repo run on a laptop with no hardware.

``CameraSource``
    A thin wrapper over ``cv2.VideoCapture`` for when real hardware is present.

Both satisfy the same ``FrameSource`` protocol, so the pipeline does not care
which one it is fed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

__all__ = ["Frame", "FrameSource", "SyntheticSource", "CameraSource"]


@dataclass(frozen=True)
class Frame:
    """One captured image plus the metadata the pipeline needs to reason about it.

    ``timestamp`` is a ``time.perf_counter()`` reading taken as close to
    acquisition as possible. Every latency number the pipeline reports is
    measured against it, so it must be stamped at capture time and never
    refreshed.
    """

    index: int
    timestamp: float
    data: np.ndarray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __repr__(self) -> str:  # keep queue/debug dumps readable
        return f"Frame(index={self.index}, shape={self.data.shape}, dtype={self.data.dtype})"


class FrameSource(ABC):
    """A pull-based stream of frames."""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """Return the next frame, or ``None`` once the stream is exhausted."""

    def close(self) -> None:
        """Release any underlying hardware handle. Safe to call twice."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame


class SyntheticSource(FrameSource):
    """Generates camera-like frames without a camera.

    Parameters
    ----------
    fps:
        Frames are paced to this rate when ``realtime`` is True, so the
        pipeline experiences realistic inter-frame gaps. Set ``realtime=False``
        in tests and benchmarks to generate as fast as possible.
    blank_rate / overexposed_rate:
        Fraction of frames that come out unusable. These exist so the
        diagnostic stage is exercised rather than merely present.
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        fps: float = 30.0,
        n_frames: Optional[int] = 300,
        n_blobs: int = 3,
        sigma_range: tuple = (4.0, 12.0),
        blank_rate: float = 0.02,
        overexposed_rate: float = 0.02,
        background: int = 40,
        noise: float = 6.0,
        seed: int = 0,
        realtime: bool = True,
        stimulus_gain: float = 0.35,
        recovery_rate: float = 0.06,
    ) -> None:
        if not 0.0 <= blank_rate + overexposed_rate <= 1.0:
            raise ValueError("blank_rate + overexposed_rate must lie in [0, 1]")

        self.width = width
        self.height = height
        self.fps = fps
        self.n_frames = n_frames
        self.blank_rate = blank_rate
        self.overexposed_rate = overexposed_rate
        self.background = background
        self.noise = noise
        self.realtime = realtime

        # Response to projected light. Illuminated regions dim, then recover
        # once the light goes away. This is a caricature of a real preparation,
        # not a model of one -- but it has the property the loop needs: the
        # sample's next state depends on what was projected at it, so feedback
        # is observable instead of asserted.
        self.stimulus_gain = stimulus_gain
        self.recovery_rate = recovery_rate

        self._rng = np.random.default_rng(seed)
        self._index = 0
        self._next_deadline: Optional[float] = None
        self._closed = False
        self._suppression = np.zeros((height, width), dtype=np.float32)

        # Precompute the coordinate grid once; re-deriving it per frame is the
        # single easiest way to make a synthetic source slower than the camera
        # it is standing in for.
        yy, xx = np.mgrid[0:height, 0:width]
        self._xx = xx.astype(np.float32)
        self._yy = yy.astype(np.float32)

        self._centers = np.column_stack(
            [
                self._rng.uniform(0.2 * width, 0.8 * width, n_blobs),
                self._rng.uniform(0.2 * height, 0.8 * height, n_blobs),
            ]
        ).astype(np.float32)
        self._velocity = self._rng.normal(0.0, 1.8, (n_blobs, 2)).astype(np.float32)
        # Compact activity rather than broad glow. The scale matters: the
        # segmenter downstream is a centre-surround operator with a 17-pixel
        # outer window, and structure much wider than that is indistinguishable
        # from illumination and is correctly ignored.
        self._sigma = self._rng.uniform(
            float(sigma_range[0]), float(sigma_range[1]), n_blobs).astype(np.float32)
        self._amplitude = self._rng.uniform(110.0, 190.0, n_blobs).astype(np.float32)

    def read(self) -> Optional[Frame]:
        if self._closed:
            return None
        if self.n_frames is not None and self._index >= self.n_frames:
            return None

        self._pace()
        image = self._render()

        roll = self._rng.random()
        if roll < self.blank_rate:
            image = self._degenerate(low=True)
        elif roll < self.blank_rate + self.overexposed_rate:
            image = self._degenerate(low=False)

        frame = Frame(index=self._index, timestamp=time.perf_counter(), data=image)
        self._index += 1
        return frame

    def close(self) -> None:
        self._closed = True

    def apply_stimulation(self, pattern: np.ndarray) -> None:
        """Record light projected at the sample. Satisfies ``Stimulable``.

        Suppression approaches 1 asymptotically under continued illumination
        rather than climbing without bound, so a region that stays lit settles
        instead of going negative and inverting the signal.
        """
        if pattern.shape != self._suppression.shape:
            raise ValueError("stimulation pattern shape does not match the frame geometry")
        lit = pattern > 0
        if not lit.any():
            return
        self._suppression[lit] += self.stimulus_gain * (1.0 - self._suppression[lit])

    @property
    def suppression(self) -> np.ndarray:
        """Per-pixel activity suppression, 0 (untouched) to 1 (fully damped)."""
        return self._suppression

    # ------------------------------------------------------------------ #

    def _pace(self) -> None:
        """Sleep so frames arrive at roughly ``self.fps``."""
        if not self.realtime or self.fps <= 0:
            return
        period = 1.0 / self.fps
        now = time.perf_counter()
        if self._next_deadline is None:
            self._next_deadline = now + period
            return
        remaining = self._next_deadline - now
        if remaining > 0:
            time.sleep(remaining)
            self._next_deadline += period
        else:
            # We fell behind. Resync rather than accumulating debt forever.
            self._next_deadline = time.perf_counter() + period

    def _render(self) -> np.ndarray:
        self._advance_blobs()

        # Recover a little each frame. Regions no longer being illuminated drift
        # back toward full activity, which is what makes the loop oscillate
        # rather than latch: stimulate, dim, drop below threshold, recover.
        if self.recovery_rate > 0:
            self._suppression *= (1.0 - self.recovery_rate)

        activity = np.zeros((self.height, self.width), dtype=np.float32)
        for (cx, cy), sigma, amp in zip(self._centers, self._sigma, self._amplitude):
            dx = self._xx - cx
            dy = self._yy - cy
            activity += amp * np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

        canvas = np.full((self.height, self.width), float(self.background), dtype=np.float32)
        canvas += activity * (1.0 - self._suppression)

        if self.noise > 0:
            canvas += self._rng.normal(0.0, self.noise, canvas.shape).astype(np.float32)

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def _advance_blobs(self) -> None:
        self._centers += self._velocity
        # Reflect off the edges so activity stays in frame.
        for axis, limit in enumerate((self.width, self.height)):
            out_low = self._centers[:, axis] < 0
            out_high = self._centers[:, axis] > limit
            self._velocity[out_low | out_high, axis] *= -1.0
            np.clip(self._centers[:, axis], 0, limit, out=self._centers[:, axis])

    def _degenerate(self, low: bool) -> np.ndarray:
        """A blank (shutter closed) or overexposed (lamp spike) frame."""
        level = 3.0 if low else 252.0
        noisy = self._rng.normal(level, 2.0, (self.height, self.width))
        return np.clip(noisy, 0, 255).astype(np.uint8)


class CameraSource(FrameSource):
    """Real acquisition via OpenCV. Imported lazily so ``cv2`` stays optional."""

    def __init__(self, device: int | str = 0, width: Optional[int] = None,
                 height: Optional[int] = None, fps: Optional[float] = None) -> None:
        import cv2  # local import: the synthetic path must not require OpenCV

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(device)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open capture device {device!r}")

        if width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, fps)
        # A shallow buffer keeps us near the live edge: on a closed-loop system
        # a stale frame is worse than a dropped one.
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._index = 0

    def read(self) -> Optional[Frame]:
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        frame = Frame(index=self._index, timestamp=time.perf_counter(), data=image)
        self._index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
