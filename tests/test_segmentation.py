import numpy as np
import pytest

from clpipe.native import HAVE_NATIVE, native
from clpipe.segmentation import NumpySegmenter


def blob_frame(size=64, level=60, blob_level=220, box=(20, 30, 30, 40)):
    image = np.full((size, size), level, dtype=np.uint8)
    r0, r1, c0, c1 = box
    image[r0:r1, c0:c1] = blob_level
    return np.ascontiguousarray(image)


def test_flat_field_produces_no_activity():
    """Uniform illumination is not activity, however bright it is."""
    seg = NumpySegmenter(32, 32)
    confidence = np.zeros((32, 32), dtype=np.uint8)

    result = seg.run(np.full((32, 32), 200, dtype=np.uint8), confidence)

    assert result.active_pixels == 0
    assert result.peak_confidence == 0


def test_gradient_is_rejected_but_a_blob_is_not():
    """The point of centre-surround: ignore slow gradients, keep compact structure."""
    seg = NumpySegmenter(64, 64)
    confidence = np.zeros((64, 64), dtype=np.uint8)

    gradient = np.tile(np.linspace(20, 200, 64, dtype=np.uint8), (64, 1))
    gradient_result = seg.run(np.ascontiguousarray(gradient), confidence)

    blob_result = seg.run(blob_frame(), confidence)

    assert gradient_result.active_pixels == 0
    assert blob_result.active_pixels > 0


def test_confidence_peaks_inside_the_blob():
    seg = NumpySegmenter(64, 64)
    confidence = np.zeros((64, 64), dtype=np.uint8)
    seg.run(blob_frame(), confidence)

    assert confidence[24, 34] > confidence[5, 5]
    assert confidence[5, 5] == 0


def test_output_is_a_map_not_a_label():
    """A classifier would answer one number; this has to say *where*."""
    seg = NumpySegmenter(64, 64)
    confidence = np.zeros((64, 64), dtype=np.uint8)

    left = seg.run(blob_frame(box=(20, 30, 5, 15)), confidence)
    left_centroid = np.argwhere(confidence >= 128).mean(axis=0)

    right = seg.run(blob_frame(box=(20, 30, 45, 55)), confidence)
    right_centroid = np.argwhere(confidence >= 128).mean(axis=0)

    assert left.active_pixels > 0 and right.active_pixels > 0
    assert right_centroid[1] - left_centroid[1] > 30


@pytest.mark.parametrize("kwargs", [
    {"fine_radius": 8, "coarse_radius": 4},
    {"fine_radius": -1},
    {"gain": 0},
])
def test_rejects_nonsense_parameters(kwargs):
    with pytest.raises(ValueError):
        NumpySegmenter(32, 32, **kwargs)


# --------------------------------------------------------------------------- #
# Parity with the compiled implementations
# --------------------------------------------------------------------------- #

requires_native = pytest.mark.skipif(
    not HAVE_NATIVE, reason="native extension not built; run scripts/build_native.sh"
)


@requires_native
def test_builtin_matches_the_numpy_reference_exactly():
    """Exact, not approximate.

    Both sides do integer box means with the same truncating division, so there
    is no rounding to accommodate. A tolerance here would quietly absorb the
    class of bug where the two drift apart a little on every frame.
    """
    rng = np.random.default_rng(7)
    controller = native.ClosedLoopController(64, 64, "builtin", 128)
    reference = NumpySegmenter(64, 64)

    native_confidence = np.zeros((64, 64), dtype=np.uint8)
    numpy_confidence = np.zeros((64, 64), dtype=np.uint8)

    for _ in range(10):
        frame = np.ascontiguousarray(
            rng.integers(0, 256, (64, 64), dtype=np.uint8))

        native_result = controller.process(frame, None, native_confidence)
        numpy_result = reference.run(frame, numpy_confidence, 128)

        assert np.array_equal(native_confidence, numpy_confidence)
        assert native_result.active_pixels == numpy_result.active_pixels
        assert native_result.peak_confidence == numpy_result.peak_confidence


@requires_native
def test_a_model_path_without_onnxruntime_raises_rather_than_falling_back():
    if native.built_with_onnxruntime():
        pytest.skip("this build has ONNX Runtime")

    with pytest.raises(Exception, match="ONNX Runtime"):
        native.ClosedLoopController(32, 32, "some/model.onnx", 128)


@requires_native
def test_a_missing_model_file_fails_at_construction():
    if not native.built_with_onnxruntime():
        pytest.skip("build has no ONNX Runtime")

    with pytest.raises(Exception):
        native.ClosedLoopController(32, 32, "definitely/not/here.onnx", 128)
