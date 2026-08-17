#pragma once

// One turn of the loop, in C++.
//
//   frame in -> segment -> threshold -> gate by the motion mask -> write the DMD
//
// Everything above the line is Python: acquisition, the diagnostic gate,
// background subtraction, orchestration. Everything from here down runs without
// touching the interpreter, and the binding releases the GIL around it, so the
// ingest thread keeps pulling frames while a cycle is in flight.
//
// The controller owns the timing instrumentation too. A closed-loop system is
// judged on whether it hits its deadline, not on its average throughput, so the
// per-stage costs are recorded separately: a loop that misses its budget needs
// to say which stage ate it.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "clpipe/dmd.hpp"
#include "clpipe/image_view.hpp"
#include "clpipe/segmenter.hpp"

namespace clpipe {

struct CycleResult {
    std::size_t active_pixels = 0;     ///< pixels the segmenter called active
    std::size_t stimulated_pixels = 0; ///< mirrors actually turned on after gating
    std::uint8_t peak_confidence = 0;
    double segment_ms = 0.0;
    double actuate_ms = 0.0;

    double total_ms() const { return segment_ms + actuate_ms; }
};

class ClosedLoopController {
public:
    /// @param segmenter_spec  "builtin" or a path to a .onnx model.
    /// @param activation_threshold  Confidence at or above which a pixel is a target.
    ClosedLoopController(int rows, int cols, const std::string& segmenter_spec = "builtin",
                         std::uint8_t activation_threshold = 128);
    ~ClosedLoopController();

    ClosedLoopController(const ClosedLoopController&) = delete;
    ClosedLoopController& operator=(const ClosedLoopController&) = delete;

    /// Run one cycle.
    ///
    /// `motion` is the background-subtraction mask from upstream and acts as a
    /// veto: a pixel is only stimulated where the segmenter says "active" *and*
    /// something actually changed there. Pass an empty view to skip the veto.
    /// `confidence`, if non-null data, receives the raw segmenter output, which
    /// is what the Python side surfaces for display and tests.
    CycleResult process(ImageView frame, PatternView motion, ConfidenceView confidence);

    /// Turn every mirror off. The pipeline calls this when a frame fails the
    /// diagnostic gate: no usable image means no basis for stimulating.
    void blank();

    const char* segmenter_name() const;
    DmdStats dmd_stats() const;
    const std::uint8_t* last_pattern() const;
    int rows() const;
    int cols() const;
    std::uint8_t activation_threshold() const;
    void set_activation_threshold(std::uint8_t value);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace clpipe
