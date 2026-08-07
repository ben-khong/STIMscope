#include <algorithm>
#include <stdexcept>
#include <vector>

#include "clpipe/segmenter.hpp"
#include "clpipe/segmenter_onnx.hpp"

namespace clpipe {

Segmenter::~Segmenter() = default;

bool built_with_onnxruntime() {
#ifdef CLPIPE_WITH_ONNXRUNTIME
    return true;
#else
    return false;
#endif
}

std::string onnxruntime_version() {
#ifdef CLPIPE_WITH_ONNXRUNTIME
    return onnx_runtime_version_string();
#else
    return std::string();
#endif
}

// --------------------------------------------------------------------------

struct BoxContrastSegmenter::Impl {
    int rows = 0;
    int cols = 0;
    int fine_radius = 2;
    int coarse_radius = 8;
    int gain = 4;

    // (rows + 1) x (cols + 1) so the four-corner lookup needs no bounds tests.
    std::vector<std::int64_t> integral;

    std::int64_t at(int r, int c) const {
        return integral[static_cast<std::size_t>(r) * (cols + 1) + c];
    }

    /// Integer mean over the box of `radius` centred on (r, c), clipped to the image.
    std::int64_t box_mean(int r, int c, int radius) const {
        const int r0 = std::max(0, r - radius);
        const int c0 = std::max(0, c - radius);
        const int r1 = std::min(rows - 1, r + radius);
        const int c1 = std::min(cols - 1, c + radius);

        const std::int64_t total = at(r1 + 1, c1 + 1) - at(r0, c1 + 1) - at(r1 + 1, c0) + at(r0, c0);
        const std::int64_t area = static_cast<std::int64_t>(r1 - r0 + 1) * (c1 - c0 + 1);
        return total / area;  // non-negative, so truncation is floor
    }
};

BoxContrastSegmenter::BoxContrastSegmenter(int rows, int cols, int fine_radius,
                                           int coarse_radius, int gain)
    : impl_(new Impl()) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }
    if (fine_radius < 0 || coarse_radius <= fine_radius) {
        throw std::invalid_argument("require 0 <= fine_radius < coarse_radius");
    }
    if (gain <= 0) {
        throw std::invalid_argument("gain must be positive");
    }

    impl_->rows = rows;
    impl_->cols = cols;
    impl_->fine_radius = fine_radius;
    impl_->coarse_radius = coarse_radius;
    impl_->gain = gain;
    impl_->integral.assign(static_cast<std::size_t>(rows + 1) * (cols + 1), 0);
}

BoxContrastSegmenter::~BoxContrastSegmenter() = default;

SegmentationResult BoxContrastSegmenter::run(ImageView frame, ConfidenceView confidence,
                                             std::uint8_t activation_threshold) {
    if (frame.rows != impl_->rows || frame.cols != impl_->cols) {
        throw std::invalid_argument("frame shape does not match the segmenter's geometry");
    }
    if (confidence.rows != impl_->rows || confidence.cols != impl_->cols) {
        throw std::invalid_argument("confidence shape does not match the segmenter's geometry");
    }

    const int rows = impl_->rows;
    const int cols = impl_->cols;

    // Build the integral image. Row-major, one pass, running row sum.
    for (int r = 0; r < rows; ++r) {
        const std::uint8_t* src = frame.row(r);
        std::int64_t row_total = 0;
        const std::size_t prev = static_cast<std::size_t>(r) * (cols + 1);
        const std::size_t curr = static_cast<std::size_t>(r + 1) * (cols + 1);
        impl_->integral[curr] = 0;
        for (int c = 0; c < cols; ++c) {
            row_total += static_cast<std::int64_t>(src[c]);
            impl_->integral[curr + c + 1] = impl_->integral[prev + c + 1] + row_total;
        }
    }

    SegmentationResult result;
    const std::int64_t gain = impl_->gain;

    for (int r = 0; r < rows; ++r) {
        std::uint8_t* out = confidence.row(r);
        for (int c = 0; c < cols; ++c) {
            const std::int64_t fine = impl_->box_mean(r, c, impl_->fine_radius);
            const std::int64_t coarse = impl_->box_mean(r, c, impl_->coarse_radius);

            std::int64_t response = (fine - coarse) * gain;
            if (response < 0) response = 0;
            if (response > 255) response = 255;

            const std::uint8_t value = static_cast<std::uint8_t>(response);
            out[c] = value;

            if (value >= activation_threshold) {
                ++result.active_pixels;
            }
            if (value > result.peak_confidence) {
                result.peak_confidence = value;
            }
        }
    }

    return result;
}

// --------------------------------------------------------------------------

std::unique_ptr<Segmenter> make_segmenter(const std::string& spec, int rows, int cols) {
    if (spec.empty() || spec == "builtin") {
        return std::unique_ptr<Segmenter>(new BoxContrastSegmenter(rows, cols));
    }

#ifdef CLPIPE_WITH_ONNXRUNTIME
    return make_onnx_segmenter(spec, rows, cols);
#else
    // Deliberately not a fallback. Silently substituting a different operator
    // for the model someone asked for would make the rig stimulate on the wrong
    // basis without anyone noticing.
    throw std::runtime_error(
        "segmenter spec '" + spec +
        "' looks like a model path, but this extension was built without ONNX Runtime. "
        "Rebuild with -DCLPIPE_ONNXRUNTIME_ROOT=/path/to/onnxruntime, or pass 'builtin'.");
#endif
}

}  // namespace clpipe
