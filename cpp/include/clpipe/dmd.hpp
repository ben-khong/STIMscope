#pragma once

// The actuation end of the loop: a digital micromirror device.
//
// A DMD is an array of mirrors, one per addressable point, each either steering
// light at the sample or dumping it. So a pattern is a binary image the same
// shape as the field of view, and "aim the stimulation at the segmented region"
// reduces to writing the thresholded confidence map to the device.
//
// `DmdDevice` is an interface for the same reason `Segmenter` is: the vendor SDK
// is not something this repo can carry, and the closed-loop logic should not be
// written against it anyway. `SimulatedDmd` records what would have been written
// so the whole loop can run, and be tested, with no hardware present.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "clpipe/image_view.hpp"

namespace clpipe {

using PatternView = View2D<const std::uint8_t>;  ///< 0 = mirror off, non-zero = on

struct DmdStats {
    std::uint64_t patterns_written = 0;
    std::uint64_t mirrors_on_total = 0;  ///< summed across every pattern written
    std::uint64_t last_mirrors_on = 0;
};

class DmdDevice {
public:
    virtual ~DmdDevice();

    /// Push one pattern to the device. Must not block longer than the frame period.
    virtual void write(PatternView pattern) = 0;

    /// Turn every mirror off. Called on shutdown and on loss of a valid frame —
    /// when the pipeline cannot see the sample, it must stop illuminating it.
    virtual void blank() = 0;

    virtual DmdStats stats() const = 0;
    virtual const char* name() const = 0;
};

/// In-memory stand-in. Keeps the last pattern so the simulated sample can read it.
class SimulatedDmd : public DmdDevice {
public:
    SimulatedDmd(int rows, int cols);
    ~SimulatedDmd() override;

    void write(PatternView pattern) override;
    void blank() override;
    DmdStats stats() const override;
    const char* name() const override { return "simulated"; }

    /// The most recently written pattern, as a borrowed view. Valid until the
    /// next write().
    const std::uint8_t* last_pattern() const;
    int rows() const;
    int cols() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace clpipe
