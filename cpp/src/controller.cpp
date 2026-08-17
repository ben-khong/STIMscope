#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "clpipe/controller.hpp"

namespace clpipe {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

}  // namespace

struct ClosedLoopController::Impl {
    int rows = 0;
    int cols = 0;
    std::uint8_t activation_threshold = 128;

    std::unique_ptr<Segmenter> segmenter;
    std::unique_ptr<SimulatedDmd> dmd;

    // Scratch for the pattern and for the confidence map when the caller did
    // not supply one. Allocated once; see the note in controller.hpp about
    // per-frame allocation in a loop with a deadline.
    std::vector<std::uint8_t> pattern;
    std::vector<std::uint8_t> scratch_confidence;
};

ClosedLoopController::ClosedLoopController(int rows, int cols,
                                           const std::string& segmenter_spec,
                                           std::uint8_t activation_threshold)
    : impl_(new Impl()) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }

    impl_->rows = rows;
    impl_->cols = cols;
    impl_->activation_threshold = activation_threshold;
    impl_->segmenter = make_segmenter(segmenter_spec, rows, cols);
    impl_->dmd.reset(new SimulatedDmd(rows, cols));

    const std::size_t count = static_cast<std::size_t>(rows) * cols;
    impl_->pattern.assign(count, 0);
    impl_->scratch_confidence.assign(count, 0);
}

ClosedLoopController::~ClosedLoopController() = default;

CycleResult ClosedLoopController::process(ImageView frame, PatternView motion,
                                          ConfidenceView confidence) {
    if (frame.rows != impl_->rows || frame.cols != impl_->cols) {
        throw std::invalid_argument("frame shape does not match the controller's geometry");
    }

    ConfidenceView target = confidence;
    if (target.data == nullptr) {
        target = ConfidenceView(impl_->scratch_confidence.data(), impl_->rows, impl_->cols);
    } else if (target.rows != impl_->rows || target.cols != impl_->cols) {
        throw std::invalid_argument("confidence shape does not match the controller's geometry");
    }

    const bool use_motion = motion.data != nullptr;
    if (use_motion && (motion.rows != impl_->rows || motion.cols != impl_->cols)) {
        throw std::invalid_argument("motion shape does not match the controller's geometry");
    }

    CycleResult result;

    const auto segment_start = Clock::now();
    const SegmentationResult segmentation =
        impl_->segmenter->run(frame, target, impl_->activation_threshold);
    result.segment_ms = ms_since(segment_start);
    result.active_pixels = segmentation.active_pixels;
    result.peak_confidence = segmentation.peak_confidence;

    const auto actuate_start = Clock::now();

    std::size_t stimulated = 0;
    for (int r = 0; r < impl_->rows; ++r) {
        const std::uint8_t* conf = target.row(r);
        const std::uint8_t* changed = use_motion ? motion.row(r) : nullptr;
        std::uint8_t* out = impl_->pattern.data() + static_cast<std::size_t>(r) * impl_->cols;

        for (int c = 0; c < impl_->cols; ++c) {
            const bool active = conf[c] >= impl_->activation_threshold;
            const bool moved = !use_motion || changed[c] != 0;
            const bool fire = active && moved;
            out[c] = fire ? 255 : 0;
            if (fire) ++stimulated;
        }
    }

    impl_->dmd->write(PatternView(impl_->pattern.data(), impl_->rows, impl_->cols));
    result.actuate_ms = ms_since(actuate_start);
    result.stimulated_pixels = stimulated;

    return result;
}

void ClosedLoopController::blank() {
    std::fill(impl_->pattern.begin(), impl_->pattern.end(), static_cast<std::uint8_t>(0));
    impl_->dmd->blank();
}

const char* ClosedLoopController::segmenter_name() const { return impl_->segmenter->name(); }

DmdStats ClosedLoopController::dmd_stats() const { return impl_->dmd->stats(); }

const std::uint8_t* ClosedLoopController::last_pattern() const {
    return impl_->dmd->last_pattern();
}

int ClosedLoopController::rows() const { return impl_->rows; }
int ClosedLoopController::cols() const { return impl_->cols; }

std::uint8_t ClosedLoopController::activation_threshold() const {
    return impl_->activation_threshold;
}

void ClosedLoopController::set_activation_threshold(std::uint8_t value) {
    impl_->activation_threshold = value;
}

}  // namespace clpipe
