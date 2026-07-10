"""Native-extension tests.

Every test here is skipped when the extension has not been compiled, so a fresh
clone runs green. The CUDA parity test is skipped a second time when no device
is visible, which is the normal case on a laptop.
"""

import numpy as np
import pytest

from clpipe import NumpyBackgroundSubtractor
from clpipe.native import HAVE_NATIVE, native

pytestmark = pytest.mark.skipif(
    not HAVE_NATIVE, reason="native extension not built; run scripts/build_native.sh"
)


@pytest.fixture
def stream():
    """A short sequence with a bright square that appears, moves, then leaves."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(40):
        img = (rng.normal(90, 4, (64, 64))).clip(0, 255).astype(np.uint8)
        if 5 <= i < 30:
            row = 4 + i
            img[row : row + 10, 20:30] = 220
        frames.append(np.ascontiguousarray(img))
    return frames


# --------------------------------------------------------------------------- #
# The zero-copy claim
# --------------------------------------------------------------------------- #


def test_arrays_cross_the_boundary_without_being_copied():
    array = np.zeros((32, 32), dtype=np.uint8)
    assert native.probe_pointer(array) == array.ctypes.data


def test_a_row_slice_is_also_borrowed_in_place():
    """A crop has a padded row stride but contiguous rows: still no copy."""
    parent = np.zeros((32, 32), dtype=np.uint8)
    crop = parent[8:16, :]
    assert native.probe_pointer(crop) == crop.ctypes.data


def test_wrong_dtype_is_rejected_rather_than_silently_converted():
    with pytest.raises(TypeError):
        native.probe_pointer(np.zeros((8, 8), dtype=np.float32))


def test_column_sliced_array_is_rejected():
    parent = np.zeros((16, 16), dtype=np.uint8)
    sub = native.BackgroundSubtractor(16, 8)
    mask = np.zeros((16, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="contiguous rows"):
        sub.apply(parent[:, ::2], mask)


def test_readonly_mask_is_rejected():
    sub = native.BackgroundSubtractor(8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask.flags.writeable = False
    with pytest.raises(ValueError, match="writeable"):
        sub.apply(np.zeros((8, 8), dtype=np.uint8), mask)


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def test_reports_its_backend():
    sub = native.BackgroundSubtractor(8, 8)
    assert sub.backend in ("cpu", "cuda")
    assert sub.rows == 8 and sub.cols == 8
    assert not sub.seeded


def test_shape_mismatch_is_caught():
    sub = native.BackgroundSubtractor(16, 16)
    with pytest.raises(ValueError, match="geometry"):
        sub.apply(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8))


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"threshold": 900}])
def test_constructor_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        native.BackgroundSubtractor(8, 8, **kwargs)


def test_cpp_matches_the_numpy_reference_exactly(stream):
    cpp = native.BackgroundSubtractor(64, 64, 0.05, 25, prefer_cuda=False)
    ref = NumpyBackgroundSubtractor(64, 64, 0.05, 25)

    cpp_mask = np.zeros((64, 64), dtype=np.uint8)
    ref_mask = np.zeros((64, 64), dtype=np.uint8)

    for frame in stream:
        assert cpp.apply(frame, cpp_mask) == ref.apply(frame, ref_mask)
        assert np.array_equal(cpp_mask, ref_mask)


def test_reset_reseeds_the_native_model(stream):
    sub = native.BackgroundSubtractor(64, 64, prefer_cuda=False)
    mask = np.zeros((64, 64), dtype=np.uint8)

    for frame in stream[:10]:
        sub.apply(frame, mask)
    assert sub.seeded

    sub.reset()
    assert not sub.seeded
    assert sub.apply(stream[20], mask) == 0  # reseeded, so nothing is foreground


# --------------------------------------------------------------------------- #
# CUDA
# --------------------------------------------------------------------------- #

requires_cuda = pytest.mark.skipif(
    not (HAVE_NATIVE and native.cuda_available()), reason="no CUDA device available"
)


@requires_cuda
def test_cuda_backend_is_selected_when_asked():
    assert native.BackgroundSubtractor(32, 32, prefer_cuda=True).backend == "cuda"
    assert native.BackgroundSubtractor(32, 32, prefer_cuda=False).backend == "cpu"


@requires_cuda
def test_cuda_agrees_with_the_cpu_backend(stream):
    """Near-identical, not bit-identical.

    nvcc contracts `bg + alpha * delta` into an FMA, which rounds once where the
    host does it twice. The models can therefore drift by an ULP or so, and a
    pixel sitting exactly on the threshold can land on either side. Asserting
    bit equality here would produce a test that fails on some GPUs and passes on
    others, which is worse than useless — so the tolerance is stated explicitly.
    """
    gpu = native.BackgroundSubtractor(64, 64, 0.05, 25, prefer_cuda=True)
    cpu = native.BackgroundSubtractor(64, 64, 0.05, 25, prefer_cuda=False)

    gpu_mask = np.zeros((64, 64), dtype=np.uint8)
    cpu_mask = np.zeros((64, 64), dtype=np.uint8)

    for index, frame in enumerate(stream):
        gpu_count = gpu.apply(frame, gpu_mask)
        cpu_count = cpu.apply(frame, cpu_mask)

        disagreements = int(np.count_nonzero(gpu_mask != cpu_mask))
        assert disagreements <= 2, f"frame {index}: {disagreements} pixels disagree"
        assert abs(gpu_count - cpu_count) <= 2
