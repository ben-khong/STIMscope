#include <algorithm>
#include <stdexcept>
#include <vector>

#include "clpipe/dmd.hpp"

namespace clpipe {

DmdDevice::~DmdDevice() = default;

struct SimulatedDmd::Impl {
    int rows = 0;
    int cols = 0;
    std::vector<std::uint8_t> pattern;
    DmdStats stats;
};

SimulatedDmd::SimulatedDmd(int rows, int cols) : impl_(new Impl()) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }
    impl_->rows = rows;
    impl_->cols = cols;
    impl_->pattern.assign(static_cast<std::size_t>(rows) * cols, 0);
}

SimulatedDmd::~SimulatedDmd() = default;

void SimulatedDmd::write(PatternView pattern) {
    if (pattern.rows != impl_->rows || pattern.cols != impl_->cols) {
        throw std::invalid_argument("pattern shape does not match the device geometry");
    }

    std::uint64_t on = 0;
    for (int r = 0; r < pattern.rows; ++r) {
        const std::uint8_t* src = pattern.row(r);
        std::uint8_t* dst = impl_->pattern.data() + static_cast<std::size_t>(r) * impl_->cols;
        for (int c = 0; c < pattern.cols; ++c) {
            // Normalise to 0/255. A real device takes one bit per mirror; keeping
            // the buffer strictly binary means the simulated sample downstream
            // never has to guess what a value of 37 was supposed to mean.
            const std::uint8_t value = src[c] != 0 ? 255 : 0;
            dst[c] = value;
            if (value) ++on;
        }
    }

    impl_->stats.patterns_written += 1;
    impl_->stats.mirrors_on_total += on;
    impl_->stats.last_mirrors_on = on;
}

void SimulatedDmd::blank() {
    std::fill(impl_->pattern.begin(), impl_->pattern.end(), static_cast<std::uint8_t>(0));
    impl_->stats.patterns_written += 1;
    impl_->stats.last_mirrors_on = 0;
}

DmdStats SimulatedDmd::stats() const { return impl_->stats; }

const std::uint8_t* SimulatedDmd::last_pattern() const { return impl_->pattern.data(); }

int SimulatedDmd::rows() const { return impl_->rows; }

int SimulatedDmd::cols() const { return impl_->cols; }

}  // namespace clpipe
