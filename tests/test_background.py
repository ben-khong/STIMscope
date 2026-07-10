import numpy as np
import pytest

from clpipe import BackgroundStage, Frame, NumpyBackgroundSubtractor, make_subtractor


def frame_of(value, shape=(16, 16)):
    return np.full(shape, value, dtype=np.uint8)


def test_first_frame_seeds_the_model_and_reports_no_foreground():
    sub = NumpyBackgroundSubtractor(16, 16)
    mask = np.empty((16, 16), dtype=np.uint8)

    assert sub.apply(frame_of(100), mask) == 0
    assert sub.seeded
    assert not mask.any()
    assert np.allclose(sub.model, 100.0)


def test_detects_a_changed_region():
    sub = NumpyBackgroundSubtractor(16, 16, threshold=25)
    mask = np.empty((16, 16), dtype=np.uint8)
    sub.apply(frame_of(100), mask)

    moved = frame_of(100)
    moved[4:8, 4:8] = 200
    count = sub.apply(moved, mask)

    assert count == 16
    assert mask[4:8, 4:8].all() and mask[4, 4] == 255
    assert not mask[0, 0]


def test_subthreshold_drift_is_absorbed_not_flagged():
    sub = NumpyBackgroundSubtractor(8, 8, alpha=0.5, threshold=25)
    mask = np.empty((8, 8), dtype=np.uint8)
    sub.apply(frame_of(100, (8, 8)), mask)

    assert sub.apply(frame_of(110, (8, 8)), mask) == 0
    assert sub.model[0, 0] == pytest.approx(105.0)  # 100 + 0.5 * 10


def test_model_is_held_still_under_foreground():
    """A stationary object must not dissolve into the background."""
    sub = NumpyBackgroundSubtractor(8, 8, alpha=0.5, threshold=25)
    mask = np.empty((8, 8), dtype=np.uint8)
    sub.apply(frame_of(100, (8, 8)), mask)

    for _ in range(20):
        assert sub.apply(frame_of(200, (8, 8)), mask) == 64

    assert sub.model[0, 0] == pytest.approx(100.0)


def test_reset_forgets_the_model():
    sub = NumpyBackgroundSubtractor(8, 8)
    mask = np.empty((8, 8), dtype=np.uint8)
    sub.apply(frame_of(100, (8, 8)), mask)
    sub.reset()

    assert not sub.seeded
    assert sub.apply(frame_of(250, (8, 8)), mask) == 0  # reseeds instead of flagging everything


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"alpha": 1.5}, {"threshold": -1}, {"threshold": 300}])
def test_rejects_nonsense_parameters(kwargs):
    with pytest.raises(ValueError):
        NumpyBackgroundSubtractor(8, 8, **kwargs)


def test_make_subtractor_rejects_an_unknown_backend():
    with pytest.raises(ValueError):
        make_subtractor(8, 8, force_backend="opencl")


def test_stage_initialises_lazily_and_accumulates():
    stage = BackgroundStage(force_backend="numpy", threshold=25)
    assert stage.backend == "uninitialised"

    base = frame_of(100, (32, 32))
    stage(Frame(0, 0.0, base))
    assert stage.backend == "numpy"

    moved = base.copy()
    moved[0:8, 0:8] = 220
    assert stage(Frame(1, 0.0, moved)) == 64
    assert stage.frames == 2
    assert stage.foreground_total == 64


def test_stage_forwards_the_mask_downstream():
    seen = []
    stage = BackgroundStage(
        force_backend="numpy",
        downstream=lambda frame, mask, count: seen.append((frame.index, count, mask.shape)),
    )

    base = frame_of(80, (16, 16))
    stage(Frame(0, 0.0, base))
    changed = base.copy()
    changed[:4] = 255
    stage(Frame(1, 0.0, changed))

    assert seen == [(0, 0, (16, 16)), (1, 64, (16, 16))]


def test_stage_handles_three_channel_frames():
    stage = BackgroundStage(force_backend="numpy")
    colour = np.full((16, 16, 3), 120, dtype=np.uint8)
    stage(Frame(0, 0.0, colour))
    assert stage.mask.shape == (16, 16)
