#pragma once

// Segmentation: where in the frame is there activity?
//
// The distinction that matters here is segmentation versus classification. A
// classifier answers "is something happening in this frame" with one label per
// frame, which is useless downstream — the DMD needs to know *where* to aim.
// A segmenter answers per pixel, and that map is what steers the mirrors.
//
// The engine behind the map is deliberately swappable. `Segmenter` is an
// interface with two implementations: a built-in analytic operator that is
// always compiled in, and an ONNX Runtime backend that loads a trained model
// from disk when the extension was built with it. The pipeline does not know
// or care which one it got. That seam is the point: the model itself is the
// lab's, and it can be replaced without touching the deployment path around it.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "clpipe/image_view.hpp"

namespace clpipe {

using ConfidenceView = View2D<std::uint8_t>;  ///< per-pixel activity, 0-255

struct SegmentationResult {
    std::size_t active_pixels = 0;   ///< pixels at or above the activation threshold
    std::uint8_t peak_confidence = 0;
};

class Segmenter {
public:
    virtual ~Segmenter();

    /// Write a per-pixel confidence map for `frame` into `confidence`.
    virtual SegmentationResult run(ImageView frame, ConfidenceView confidence,
                                   std::uint8_t activation_threshold) = 0;

    virtual const char* name() const = 0;
};

/// True when the extension was compiled against ONNX Runtime.
bool built_with_onnxruntime();

/// Version string of the linked ONNX Runtime, or an empty string without it.
std::string onnxruntime_version();

/// Build a segmenter.
///
/// `spec` is either "builtin" or a filesystem path to a .onnx model. Passing a
/// model path to a build without ONNX Runtime throws rather than silently
/// falling back — a closed-loop rig quietly running the wrong operator is worse
/// than one that refuses to start.
std::unique_ptr<Segmenter> make_segmenter(const std::string& spec, int rows, int cols);

// --------------------------------------------------------------------------
// Built-in analytic segmenter.
//
// Difference of box means: a tight box average minus a wide one, which
// responds to compact bright structure and ignores slow illumination gradients
// across the field. Both radii are computed from one integral image, so cost is
// independent of the window size.
//
// All arithmetic is integer, including the divisions. That is not an accident:
// the NumPy reference in clpipe/segmentation.py performs the same integer
// operations in the same order, so the two agree bit for bit and the parity
// test can assert exact equality rather than a tolerance.
// --------------------------------------------------------------------------
class BoxContrastSegmenter : public Segmenter {
public:
    BoxContrastSegmenter(int rows, int cols, int fine_radius = 2, int coarse_radius = 8,
                         int gain = 4);
    ~BoxContrastSegmenter() override;

    SegmentationResult run(ImageView frame, ConfidenceView confidence,
                           std::uint8_t activation_threshold) override;

    const char* name() const override { return "builtin"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace clpipe
