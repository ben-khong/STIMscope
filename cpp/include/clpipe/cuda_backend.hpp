#pragma once

// The seam between ordinary C++ and the .cu translation unit.
//
// Everything below is declared in plain C++ types so that background.cpp can be
// compiled by the host compiler with no CUDA headers in scope. Only
// background_cuda.cu includes <cuda_runtime.h>. When the extension is built
// without CUDA, the whole block compiles away and `available()` is a constant
// false, so there is no stub object file and no dead symbol to link.

#include <cstddef>
#include <cstdint>

namespace clpipe {
namespace cuda {

#ifdef CLPIPE_WITH_CUDA

/// Opaque handle to the device-resident model, staging buffers and counter.
struct Buffers;

/// True if the runtime reports at least one usable device.
bool available();

/// Allocate device state for an n-pixel image. Throws std::runtime_error on failure.
Buffers* create(std::size_t n_pixels);
void destroy(Buffers* buffers) noexcept;

/// Copy `frame` into the model as the initial background estimate.
void seed(Buffers* buffers, const std::uint8_t* frame, std::size_t n_pixels);

/// One frame: upload, launch, download the mask. Returns the foreground count.
std::size_t apply(Buffers* buffers, const std::uint8_t* frame, std::uint8_t* mask,
                  std::size_t n_pixels, float alpha, int threshold);

#else

struct Buffers;
inline bool available() { return false; }

#endif  // CLPIPE_WITH_CUDA

}  // namespace cuda
}  // namespace clpipe
