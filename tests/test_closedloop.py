import numpy as np
import pytest

from clpipe import BackgroundStage, ClosedLoopStage, Frame, QualityStage, SyntheticSource
from clpipe.closedloop import NumpyController, make_controller
from clpipe.native import HAVE_NATIVE, native


def blob_frame(size=64, level=60, blob_level=230):
    image = np.full((size, size), level, dtype=np.uint8)
    image[24:34, 24:34] = blob_level
    return np.ascontiguousarray(image)


# --------------------------------------------------------------------------- #
# One cycle
# --------------------------------------------------------------------------- #


def test_a_cycle_turns_mirrors_on_where_there_is_activity():
    controller = NumpyController(64, 64)
    result = controller.process(blob_frame())

    assert result.active_pixels > 0
    assert result.stimulated_pixels == result.active_pixels

    pattern = controller.last_pattern()
    assert pattern[28, 28] == 255
    assert pattern[2, 2] == 0


def test_the_projected_pattern_is_read_only():
    """It aliases the device buffer. Editing it from Python would be a lie."""
    controller = NumpyController(32, 32)
    controller.process(blob_frame(32))

    with pytest.raises(ValueError):
        controller.last_pattern()[0, 0] = 255


def test_the_motion_mask_vetoes_stimulation():
    controller = NumpyController(64, 64)
    frame = blob_frame()

    unvetoed = controller.process(frame, None)
    assert unvetoed.stimulated_pixels > 0

    vetoed = controller.process(frame, np.zeros((64, 64), dtype=np.uint8))
    assert vetoed.active_pixels == unvetoed.active_pixels  # the segmenter still fires
    assert vetoed.stimulated_pixels == 0                   # but nothing is projected


def test_blank_clears_every_mirror():
    controller = NumpyController(64, 64)
    controller.process(blob_frame())
    assert controller.last_pattern().any()

    controller.blank()
    assert not controller.last_pattern().any()


def test_a_model_path_without_the_extension_is_refused():
    with pytest.raises(RuntimeError, match="native extension is not built"):
        NumpyController(32, 32, segmenter="models/real.onnx")


def test_timing_is_recorded_per_stage():
    result = NumpyController(64, 64).process(blob_frame())

    assert result.segment_ms > 0
    assert result.actuate_ms > 0
    assert result.total_ms == pytest.approx(result.segment_ms + result.actuate_ms)


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


def test_stage_initialises_lazily_and_accumulates():
    stage = ClosedLoopStage(force_numpy=True)
    assert stage.backend == "uninitialised"

    stage(Frame(0, 0.0, blob_frame()))
    assert stage.backend == "numpy"
    assert stage.cycles == 1
    assert stage.stimulated_total > 0
    assert stage.confidence.shape == (64, 64)


def test_stage_handles_three_channel_frames():
    stage = ClosedLoopStage(force_numpy=True)
    stage(Frame(0, 0.0, np.full((32, 32, 3), 100, dtype=np.uint8)))
    assert stage.pattern.shape == (32, 32)


def test_a_rejected_frame_blanks_the_device():
    """No usable image means no basis for illuminating the sample."""
    stage = ClosedLoopStage(force_numpy=True)
    stage(Frame(0, 0.0, blob_frame()))
    assert stage.pattern.any()

    stage.on_rejected_frame()
    assert not stage.pattern.any()


def test_stage_composes_behind_the_gate_and_background_stage():
    """The wiring the CLI uses: gate -> background -> closed loop."""
    loop = ClosedLoopStage(force_numpy=True)
    background = BackgroundStage(force_backend="numpy", downstream=loop)
    gate = QualityStage(downstream=lambda frame, report: background(frame, report))

    source = SyntheticSource(width=64, height=64, n_frames=30, realtime=False, seed=2)
    for frame in source:
        gate(frame)

    assert gate.accepted + gate.rejected == 30
    assert background.frames == gate.accepted
    assert loop.cycles == gate.accepted


# --------------------------------------------------------------------------- #
# The loop is actually closed
# --------------------------------------------------------------------------- #


def run_loop(connect_feedback: bool, frames: int = 80, seed: int = 5):
    source = SyntheticSource(
        width=64, height=64, n_frames=frames, realtime=False, seed=seed,
        blank_rate=0.0, overexposed_rate=0.0, n_blobs=2,
    )
    stage = ClosedLoopStage(
        force_numpy=True,
        use_motion_veto=False,
        sample=source if connect_feedback else None,
    )

    stimulated = []
    for frame in source:
        stimulated.append(stage(frame).stimulated_pixels)
    return source, stimulated


def test_closing_the_feedback_path_changes_the_sample():
    """The load-bearing test for the word 'closed'.

    With the feedback edge connected, what the DMD projects reaches the sample
    and the next frame is different because of it. Without it, the identical
    pipeline over the identical frames is just a detector. Comparing the two is
    the only way to show the loop is doing something rather than merely being
    described as a loop.
    """
    closed_source, closed = run_loop(connect_feedback=True)
    _, open_loop = run_loop(connect_feedback=False)

    assert closed != open_loop, "feedback had no effect on stimulation over time"
    assert closed_source.suppression.max() > 0.2, "sample was never suppressed"

    early = float(np.mean(closed[:15]))
    late = float(np.mean(closed[-15:]))
    assert late < early, f"stimulation did not settle: {early:.1f} -> {late:.1f}"


def test_the_open_loop_sample_is_never_touched():
    open_source, _ = run_loop(connect_feedback=False)
    assert open_source.suppression.max() == 0.0


def test_suppression_saturates_rather_than_running_away():
    """Continued illumination must settle, not invert the signal."""
    source = SyntheticSource(width=32, height=32, n_frames=1, realtime=False,
                             recovery_rate=0.0)
    lit = np.full((32, 32), 255, dtype=np.uint8)

    for _ in range(200):
        source.apply_stimulation(lit)

    assert source.suppression.max() <= 1.0


# --------------------------------------------------------------------------- #
# Native and ONNX
# --------------------------------------------------------------------------- #

requires_native = pytest.mark.skipif(
    not HAVE_NATIVE, reason="native extension not built; run scripts/build_native.sh"
)


@requires_native
def test_make_controller_prefers_the_native_implementation():
    assert make_controller(32, 32).segmenter == "builtin"
    assert make_controller(32, 32, force_numpy=True).segmenter == "numpy"


@requires_native
def test_native_and_numpy_controllers_agree():
    native_controller = native.ClosedLoopController(64, 64, "builtin", 128)
    numpy_controller = NumpyController(64, 64)
    frame = blob_frame()

    a = native_controller.process(frame, None, None)
    b = numpy_controller.process(frame, None, None)

    assert a.active_pixels == b.active_pixels
    assert a.stimulated_pixels == b.stimulated_pixels
    assert np.array_equal(native_controller.last_pattern(), numpy_controller.last_pattern())


@requires_native
def test_native_pattern_view_is_read_only_and_does_not_copy():
    controller = native.ClosedLoopController(32, 32, "builtin", 128)
    controller.process(blob_frame(32), None, None)

    first = controller.last_pattern()
    assert not first.flags.writeable
    # Same buffer every call: it is a view onto the device, not a snapshot.
    assert first.ctypes.data == controller.last_pattern().ctypes.data


@requires_native
def test_native_dmd_stats_track_what_was_written():
    controller = native.ClosedLoopController(64, 64, "builtin", 128)
    result = controller.process(blob_frame(), None, None)
    stats = controller.dmd_stats

    assert stats.patterns_written == 1
    assert stats.last_mirrors_on == result.stimulated_pixels

    controller.blank()
    assert controller.dmd_stats.last_mirrors_on == 0


onnx_model = pytest.mark.skipif(
    not (HAVE_NATIVE and native.built_with_onnxruntime()),
    reason="extension built without ONNX Runtime",
)


@onnx_model
def test_onnx_segmenter_agrees_broadly_with_the_analytic_one(tmp_path):
    """Agreement in the bulk, not pixel equality.

    The stand-in model computes centre-surround contrast in float and pads with
    zeros; the analytic version uses truncating integer means and clips its
    boxes at the border. So they disagree at the edges and on pixels sitting
    right at the threshold. Asserting overlap rather than equality is the honest
    assertion -- and it is still enough to catch a model that is wired up wrong,
    which is what this test is for.
    """
    import subprocess
    import sys

    model = tmp_path / "stub.onnx"
    subprocess.run(
        [sys.executable, "scripts/make_stub_model.py", "--output", str(model)],
        check=True, capture_output=True,
    )

    frame = blob_frame()
    analytic = native.ClosedLoopController(64, 64, "builtin", 128)
    onnx = native.ClosedLoopController(64, 64, str(model), 128)
    assert onnx.segmenter.startswith("onnx:")

    analytic.process(frame, None, None)
    onnx.process(frame, None, None)

    a = analytic.last_pattern() > 0
    b = onnx.last_pattern() > 0
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)

    assert union > 0
    assert intersection / union > 0.75, f"IoU {intersection / union:.2f} is too low"
