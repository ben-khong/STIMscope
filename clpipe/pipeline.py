"""The concurrent ingest/process pipeline.

One thread pulls frames off the source and pushes them onto a bounded queue;
another pops them and runs the processing chain. The point is overlap: the rig
is acquiring frame N+1 while frame N is still being analysed, so the wall-clock
cost of a cycle is ``max(acquire, process)`` instead of their sum.

The queue is bounded on purpose. An unbounded queue on a live stream does not
prevent backpressure, it just converts it into unbounded latency and memory
growth. When the consumer falls behind, the honest options are to block the
producer or to drop frames; for a closed-loop system acting on the present
state of the sample, dropping the *oldest* frame is correct. A stale frame
steers the hardware toward a state that no longer exists.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .sources import Frame, FrameSource

__all__ = ["Metrics", "Pipeline"]

_SENTINEL = object()
_MAX_LATENCY_SAMPLES = 100_000


@dataclass
class Metrics:
    """Counters for one run.

    Each counter is written by exactly one thread (``captured``/``dropped`` by
    the ingest thread, the rest by the processing thread), so no lock is needed
    beyond CPython's guarantees on attribute assignment.
    """

    captured: int = 0
    dropped: int = 0
    processed: int = 0
    wall_time: float = 0.0
    latencies: List[float] = field(default_factory=list, repr=False)

    def record_latency(self, seconds: float) -> None:
        if len(self.latencies) < _MAX_LATENCY_SAMPLES:
            self.latencies.append(seconds)

    @property
    def fps(self) -> float:
        return self.processed / self.wall_time if self.wall_time > 0 else 0.0

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.captured if self.captured else 0.0

    def latency_ms(self, percentile: float = 50.0) -> float:
        """End-to-end latency: capture timestamp to end of processing."""
        if not self.latencies:
            return 0.0
        ordered = sorted(self.latencies)
        k = min(len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))))
        return ordered[k] * 1000.0

    def summary(self) -> str:
        return (
            f"captured={self.captured} processed={self.processed} "
            f"dropped={self.dropped} ({self.drop_rate:.1%}) "
            f"throughput={self.fps:.1f} fps  "
            f"latency p50={self.latency_ms(50):.2f}ms p95={self.latency_ms(95):.2f}ms"
        )


class Pipeline:
    """Runs ``handler`` over every frame from ``source`` on a separate thread.

    Parameters
    ----------
    queue_size:
        Depth of the buffer between the two threads. Small on purpose: it is a
        shock absorber for jitter, not a backlog.
    drop_when_full:
        ``True`` discards the oldest queued frame to make room for the newest
        (real-time behaviour). ``False`` blocks the producer instead, which is
        what you want when recording to disk and no frame may be lost.
    """

    def __init__(
        self,
        source: FrameSource,
        handler: Callable[[Frame], Any],
        queue_size: int = 8,
        drop_when_full: bool = True,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be >= 1")

        self._source = source
        self._handler = handler
        self._drop_when_full = drop_when_full
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None
        self.metrics = Metrics()

    # ------------------------------------------------------------------ #

    def run(self) -> Metrics:
        """Run to completion. Blocks. Ctrl-C shuts both threads down cleanly."""
        started = time.perf_counter()

        ingest = threading.Thread(target=self._ingest_loop, name="ingest", daemon=True)
        process = threading.Thread(target=self._process_loop, name="process", daemon=True)
        ingest.start()
        process.start()

        try:
            while process.is_alive():
                process.join(timeout=0.1)
        except KeyboardInterrupt:
            self._stop.set()
            process.join(timeout=2.0)

        ingest.join(timeout=2.0)
        self.metrics.wall_time = time.perf_counter() - started
        self._source.close()

        if self._error is not None:
            raise self._error
        return self.metrics

    def stop(self) -> None:
        """Ask both threads to wind down at the next opportunity."""
        self._stop.set()

    # ------------------------------------------------------------------ #

    def _ingest_loop(self) -> None:
        try:
            for frame in self._source:
                if self._stop.is_set():
                    break
                self.metrics.captured += 1
                if not self._offer(frame):
                    break
        except BaseException as exc:  # surface it on the calling thread
            self._error = exc
            self._stop.set()
        finally:
            self._close_queue()

    def _offer(self, frame: Frame) -> bool:
        """Enqueue ``frame``, dropping the oldest if configured to. False = give up."""
        while not self._stop.is_set():
            try:
                if self._drop_when_full:
                    # Non-blocking on purpose. Waiting even briefly to enqueue a
                    # live frame is the same mistake as buffering it: by the time
                    # the slot frees up the frame is already stale.
                    self._queue.put_nowait(frame)
                else:
                    self._queue.put(frame, timeout=0.05)
                return True
            except queue.Full:
                if not self._drop_when_full:
                    continue  # block until the consumer catches up
                try:
                    self._queue.get_nowait()
                    self.metrics.dropped += 1
                except queue.Empty:
                    pass  # consumer beat us to it; retry the put
        return False

    def _close_queue(self) -> None:
        """Push the sentinel so the consumer terminates. Never blocks forever."""
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            try:
                self._queue.put(_SENTINEL, timeout=0.05)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                    self.metrics.dropped += 1
                except queue.Empty:
                    pass

    def _process_loop(self) -> None:
        while True:
            try:
                item = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue

            if item is _SENTINEL:
                return

            try:
                self._handler(item)
            except BaseException as exc:
                self._error = exc
                self._stop.set()
                return

            self.metrics.processed += 1
            self.metrics.record_latency(time.perf_counter() - item.timestamp)
