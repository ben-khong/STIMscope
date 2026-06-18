import numpy as np
import pytest

from clpipe import Frame, IntensityGate, QualityStage, to_gray


def make_frame(value, shape=(32, 32), index=0):
    return Frame(index=index, timestamp=0.0, data=np.full(shape, value, dtype=np.uint8))


def test_accepts_a_normally_exposed_frame():
    report = IntensityGate()(make_frame(128))
    assert report.ok
    assert report.reason is None
    assert report.mean_intensity == pytest.approx(128.0)


def test_rejects_blank_frame():
    report = IntensityGate()(make_frame(3))
    assert not report.ok
    assert report.reason == "blank"


def test_rejects_overexposed_frame():
    report = IntensityGate()(make_frame(250))
    assert not report.ok
    assert report.reason == "overexposed"


@pytest.mark.parametrize("value,expected", [(10, True), (9, False), (245, True), (246, False)])
def test_thresholds_are_inclusive_at_the_boundary(value, expected):
    assert IntensityGate()(make_frame(value)).ok is expected


def test_low_must_be_below_high():
    with pytest.raises(ValueError):
        IntensityGate(low=200, high=100)


def test_handles_three_channel_frames():
    image = np.full((16, 16, 3), 200, dtype=np.uint8)
    assert to_gray(image).shape == (16, 16)
    assert IntensityGate()(Frame(0, 0.0, image)).ok


def test_stage_tallies_and_forwards_only_accepted_frames():
    forwarded = []
    stage = QualityStage(downstream=lambda frame, report: forwarded.append(frame.index))

    for index, value in enumerate([128, 2, 200, 255, 90]):
        stage(make_frame(value, index=index))

    assert stage.accepted == 3
    assert stage.rejected == 2
    assert stage.reasons == {"blank": 1, "overexposed": 1}
    assert forwarded == [0, 2, 4]
