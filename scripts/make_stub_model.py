#!/usr/bin/env python3
"""Emit a stand-in segmentation model so the ONNX path can actually be run.

The trained model this pipeline was built around belongs to the lab and is not
something this repo can carry. That is fine for demonstrating the deployment
path, which is the part that was mine: what matters is that a real .onnx file
goes in, a per-pixel map comes out, and the loop closes around it at frame rate.

So this writes a model with the right *shape* contract and a deliberately
transparent operator: NCHW float input in [0, 1], NCHW float output in [0, 1],
one channel each, spatial dims dynamic. The operator is a centre-surround
convolution — a tight box average minus a wide one — which is the same thing
clpipe's analytic segmenter computes. That correspondence is on purpose: it
means the ONNX and built-in paths can be checked against each other, so a
failure in test_segmentation.py points at the deployment code rather than at a
disagreement about what the answer should have been.

Swapping in the real model needs no code change, only a different path.

    python scripts/make_stub_model.py --output models/stub_segmenter.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def centre_surround_kernel(fine_radius: int, coarse_radius: int) -> np.ndarray:
    """A (1, 1, k, k) kernel whose response is mean(fine box) - mean(coarse box)."""
    size = 2 * coarse_radius + 1
    kernel = np.zeros((size, size), dtype=np.float32)

    centre = coarse_radius
    coarse_area = float(size * size)
    kernel -= 1.0 / coarse_area

    fine_size = 2 * fine_radius + 1
    fine_area = float(fine_size * fine_size)
    lo = centre - fine_radius
    hi = centre + fine_radius + 1
    kernel[lo:hi, lo:hi] += 1.0 / fine_area

    return kernel.reshape(1, 1, size, size)


def build(output: Path, fine_radius: int, coarse_radius: int, gain: float) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    kernel = centre_surround_kernel(fine_radius, coarse_radius)

    initializers = [
        numpy_helper.from_array(kernel, name="kernel"),
        numpy_helper.from_array(np.array([gain], dtype=np.float32), name="gain"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="lo"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="hi"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            inputs=["input", "kernel"],
            outputs=["contrast"],
            kernel_shape=[kernel.shape[2], kernel.shape[3]],
            # Pad so the output keeps the input's spatial dims; edge pixels see a
            # truncated window, exactly as the analytic version's clipped boxes do.
            pads=[coarse_radius] * 4,
            name="centre_surround",
        ),
        helper.make_node("Mul", inputs=["contrast", "gain"], outputs=["scaled"], name="scale"),
        helper.make_node("Clip", inputs=["scaled", "lo", "hi"], outputs=["output"], name="clip"),
    ]

    # Dynamic spatial dims: the same model file works for any frame size, so
    # changing camera binning does not mean regenerating the model.
    graph = helper.make_graph(
        nodes,
        name="stub_segmenter",
        inputs=[helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 1, "height", "width"])],
        outputs=[helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 1, "height", "width"])],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        producer_name="clpipe-make-stub-model",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8  # compatible with ONNX Runtime 1.12 and newer
    model.doc_string = (
        "Stand-in segmentation model: centre-surround contrast, clipped to [0, 1]. "
        "Not trained on anything. Replace with a real model of the same shape."
    )

    onnx.checker.check_model(model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("models/stub_segmenter.onnx"))
    parser.add_argument("--fine-radius", type=int, default=2)
    parser.add_argument("--coarse-radius", type=int, default=8)
    parser.add_argument("--gain", type=float, default=4.0)
    args = parser.parse_args()

    try:
        build(args.output, args.fine_radius, args.coarse_radius, args.gain)
    except ImportError:
        print("this script needs the 'onnx' package: pip install onnx")
        return 1

    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    print("run it with:  clpipe --closed-loop --segmenter " + str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
