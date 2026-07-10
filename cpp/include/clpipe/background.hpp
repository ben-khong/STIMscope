#pragma once

// Running-average background subtraction.
//
// The model is one float per pixel. For each incoming frame:
//
//   diff = |pixel - model|
//   mask = diff > threshold ? 255 : 0
//   model += alpha * (pixel - model)      // only where the pixel is background
//
// Holding the model still under foreground pixels stops a stationary object
// from dissolving into the background over a few hundred frames.
//
// Every pixel's update depends only on that pixel's own history. There is no
// neighbourhood term, no reduction, no sequential dependency between pixels.
// That independence is exactly what makes the operation a good fit for one GPU
// thread per pixel, and it is why this stage was the one worth moving.

#include <cstddef>
#include <cstdint>
#include <memory>

#include "clpipe/image_view.hpp"

namespace clpipe {

enum class Backend { Cpu, Cuda };

const char* backend_name(Backend backend);

/// True when the extension was built with CUDA *and* a usable device is present.
bool cuda_available();

/// True when the extension was compiled with the CUDA backend included at all.
bool built_with_cuda();

/// Scalar reference implementation. Exposed so the benchmark can call it directly.
std::size_t background_apply_cpu(ImageView frame, ModelView model, MaskView mask, float alpha,
                                 int threshold);

class BackgroundSubtractor {
public:
    /// @param prefer_cuda  Use the GPU backend when one is available. Set false to
    ///                     force the CPU path, which is how the benchmark measures both.
    BackgroundSubtractor(int rows, int cols, float alpha = 0.05f, int threshold = 25,
                         bool prefer_cuda = true);
    ~BackgroundSubtractor();

    BackgroundSubtractor(const BackgroundSubtractor&) = delete;
    BackgroundSubtractor& operator=(const BackgroundSubtractor&) = delete;

    /// Writes the foreground mask and returns the number of foreground pixels.
    /// The first frame seeds the model and reports zero foreground.
    std::size_t apply(ImageView frame, MaskView mask);

    /// Forget the model; the next frame seeds it again.
    void reset();

    Backend backend() const;
    bool seeded() const;
    int rows() const;
    int cols() const;
    float alpha() const;
    int threshold() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace clpipe
