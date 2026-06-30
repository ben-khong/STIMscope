import time

import numpy as np
import pytest

from clpipe import Frame, FrameSource, Pipeline, SyntheticSource


class ListSource(FrameSource):
    """Replays a fixed number of frames, optionally paced like a real camera.

    ``interval`` matters: with no pacing the producer is infinitely fast, so in
    drop mode it will always outrun the consumer no matter how cheap the
    handler is. Real sources have a frame period.
    """

    def __init__(self, count, shape=(8, 8), interval=0.0):
        self._count = count
        self._shape = shape
        self._interval = interval
        self._i = 0
        self.closed = False

    def read(self):
        if self._i >= self._count:
            return None
        if self._interval:
            time.sleep(self._interval)
        frame = Frame(self._i, time.perf_counter(), np.full(self._shape, 128, dtype=np.uint8))
        self._i += 1
        return frame

    def close(self):
        self.closed = True


def test_every_frame_reaches_the_handler_when_consumer_keeps_up():
    seen = []
    metrics = Pipeline(ListSource(50, interval=0.002), handler=seen.append, queue_size=8).run()

    assert [f.index for f in seen] == list(range(50))
    assert metrics.captured == 50
    assert metrics.processed == 50
    assert metrics.dropped == 0


def test_source_is_closed_after_the_run():
    source = ListSource(5)
    Pipeline(source, handler=lambda f: None).run()
    assert source.closed


def test_slow_consumer_drops_frames_rather_than_growing_the_queue():
    seen = []

    def slow(frame):
        time.sleep(0.004)
        seen.append(frame.index)

    metrics = Pipeline(ListSource(200), handler=slow, queue_size=2, drop_when_full=True).run()

    assert metrics.dropped > 0
    assert metrics.processed + metrics.dropped == metrics.captured
    # Dropping the oldest means what survives is the freshest available frame,
    # so the tail of the stream must still get through.
    assert max(seen) >= 190


def test_no_drop_mode_blocks_the_producer_instead():
    seen = []

    def slow(frame):
        time.sleep(0.002)
        seen.append(frame.index)

    metrics = Pipeline(ListSource(40), handler=slow, queue_size=2, drop_when_full=False).run()

    assert metrics.dropped == 0
    assert [f for f in seen] == list(range(40))


def test_handler_exceptions_surface_on_the_calling_thread():
    def explode(frame):
        raise RuntimeError("segmentation model not loaded")

    with pytest.raises(RuntimeError, match="segmentation model not loaded"):
        Pipeline(ListSource(10), handler=explode).run()


def test_concurrency_actually_overlaps_acquisition_and_processing():
    """A 5ms source and a 5ms handler should finish nearer 5ms/frame than 10ms."""

    class SlowSource(FrameSource):
        def __init__(self, count):
            self._i = 0
            self._count = count

        def read(self):
            if self._i >= self._count:
                return None
            time.sleep(0.005)
            frame = Frame(self._i, time.perf_counter(), np.zeros((4, 4), dtype=np.uint8))
            self._i += 1
            return frame

    metrics = Pipeline(SlowSource(40), handler=lambda f: time.sleep(0.005), queue_size=4).run()

    serial_estimate = 40 * 0.010
    assert metrics.wall_time < serial_estimate * 0.85


def test_synthetic_source_emits_some_unusable_frames():
    source = SyntheticSource(width=64, height=64, n_frames=400, realtime=False,
                             blank_rate=0.05, overexposed_rate=0.05, seed=1)
    means = [float(frame.data.mean()) for frame in source]

    assert len(means) == 400
    assert any(m < 10 for m in means), "expected some blank frames"
    assert any(m > 245 for m in means), "expected some overexposed frames"
    assert any(10 <= m <= 245 for m in means), "expected mostly usable frames"


def test_synthetic_source_is_deterministic_for_a_given_seed():
    def first_frame(seed):
        return SyntheticSource(width=32, height=32, n_frames=1, realtime=False, seed=seed).read()

    assert np.array_equal(first_frame(7).data, first_frame(7).data)
    assert not np.array_equal(first_frame(7).data, first_frame(8).data)
