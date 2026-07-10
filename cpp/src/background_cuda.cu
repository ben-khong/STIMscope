// Hand-written CUDA backend for background subtraction.
//
// One thread per pixel. The per-pixel update is fully independent, so there is
// no cooperation needed for the arithmetic itself — the only shared quantity is
// the foreground count, and that is reduced inside each block so the kernel
// performs one atomic per block rather than one per foreground pixel.
//
// Device state (model, staging buffers, counter) is allocated once at
// construction and reused for every frame. Allocating per frame would dominate
// the measurement and would also introduce unbounded jitter into a loop that is
// supposed to hit a deadline.

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "clpipe/cuda_backend.hpp"

namespace clpipe {
namespace cuda {
namespace {

constexpr unsigned int kBlockSize = 256;  // power of two: required by the reduction below

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + what + ": " +
                                 cudaGetErrorString(status));
    }
}

/// Foreground mask + running-average model update, one pixel per thread.
///
/// Mirrors background_apply_cpu exactly, including holding the model still
/// under foreground pixels. The parity test in tests/test_native.py compares
/// the two outputs pixel for pixel, so any divergence here is a test failure
/// rather than a silent behaviour change.
__global__ void background_update_kernel(const unsigned char* __restrict__ frame,
                                         float* __restrict__ model,
                                         unsigned char* __restrict__ mask,
                                         unsigned int* __restrict__ foreground_count,
                                         std::size_t n, float alpha, float threshold) {
    __shared__ unsigned int scratch[kBlockSize];

    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    unsigned int flag = 0u;

    // No early return: every thread in the block must reach the __syncthreads()
    // calls in the reduction below, including the tail threads past the end of
    // the image.
    if (i < n) {
        const float pixel = static_cast<float>(frame[i]);
        const float bg = model[i];
        const float delta = pixel - bg;
        const bool is_foreground = fabsf(delta) > threshold;

        mask[i] = is_foreground ? 255 : 0;
        if (!is_foreground) {
            model[i] = fmaf(alpha, delta, bg);
        }
        flag = is_foreground ? 1u : 0u;
    }

    scratch[threadIdx.x] = flag;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && scratch[0] != 0u) {
        atomicAdd(foreground_count, scratch[0]);
    }
}

__global__ void seed_kernel(const unsigned char* __restrict__ frame, float* __restrict__ model,
                            std::size_t n) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        model[i] = static_cast<float>(frame[i]);
    }
}

}  // namespace

struct Buffers {
    std::size_t n = 0;
    unsigned char* d_frame = nullptr;
    unsigned char* d_mask = nullptr;
    float* d_model = nullptr;
    unsigned int* d_count = nullptr;
    cudaStream_t stream = nullptr;
};

bool available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        // Clear the sticky error so a later legitimate call is not misdiagnosed.
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

Buffers* create(std::size_t n_pixels) {
    Buffers* buffers = new Buffers();
    buffers->n = n_pixels;

    try {
        check(cudaMalloc(&buffers->d_frame, n_pixels), "malloc frame");
        check(cudaMalloc(&buffers->d_mask, n_pixels), "malloc mask");
        check(cudaMalloc(&buffers->d_model, n_pixels * sizeof(float)), "malloc model");
        check(cudaMalloc(&buffers->d_count, sizeof(unsigned int)), "malloc counter");
        check(cudaMemset(buffers->d_model, 0, n_pixels * sizeof(float)), "memset model");
        check(cudaStreamCreate(&buffers->stream), "stream create");
    } catch (...) {
        destroy(buffers);
        throw;
    }

    return buffers;
}

void destroy(Buffers* buffers) noexcept {
    if (buffers == nullptr) {
        return;
    }
    if (buffers->stream != nullptr) cudaStreamDestroy(buffers->stream);
    if (buffers->d_count != nullptr) cudaFree(buffers->d_count);
    if (buffers->d_model != nullptr) cudaFree(buffers->d_model);
    if (buffers->d_mask != nullptr) cudaFree(buffers->d_mask);
    if (buffers->d_frame != nullptr) cudaFree(buffers->d_frame);
    delete buffers;
}

void seed(Buffers* buffers, const unsigned char* frame, std::size_t n_pixels) {
    check(cudaMemcpyAsync(buffers->d_frame, frame, n_pixels, cudaMemcpyHostToDevice,
                          buffers->stream),
          "seed upload");

    const unsigned int blocks =
        static_cast<unsigned int>((n_pixels + kBlockSize - 1) / kBlockSize);
    seed_kernel<<<blocks, kBlockSize, 0, buffers->stream>>>(buffers->d_frame, buffers->d_model,
                                                            n_pixels);
    check(cudaGetLastError(), "seed launch");
    check(cudaStreamSynchronize(buffers->stream), "seed sync");
}

std::size_t apply(Buffers* buffers, const unsigned char* frame, unsigned char* mask,
                  std::size_t n_pixels, float alpha, int threshold) {
    if (n_pixels != buffers->n) {
        throw std::invalid_argument("frame size does not match the allocated device buffers");
    }

    cudaStream_t stream = buffers->stream;

    // On a Jetson, host and device share physical DRAM, so these copies are
    // memory-to-memory rather than PCIe traffic. They can be removed entirely by
    // allocating the frame buffer with cudaHostAllocMapped and handing the
    // device the mapped pointer; that is worth doing once the camera driver can
    // be made to write into a pinned buffer directly.
    check(cudaMemcpyAsync(buffers->d_frame, frame, n_pixels, cudaMemcpyHostToDevice, stream),
          "frame upload");
    check(cudaMemsetAsync(buffers->d_count, 0, sizeof(unsigned int), stream), "counter reset");

    const unsigned int blocks =
        static_cast<unsigned int>((n_pixels + kBlockSize - 1) / kBlockSize);
    background_update_kernel<<<blocks, kBlockSize, 0, stream>>>(
        buffers->d_frame, buffers->d_model, buffers->d_mask, buffers->d_count, n_pixels, alpha,
        static_cast<float>(threshold));
    check(cudaGetLastError(), "kernel launch");

    unsigned int foreground = 0;
    check(cudaMemcpyAsync(mask, buffers->d_mask, n_pixels, cudaMemcpyDeviceToHost, stream),
          "mask download");
    check(cudaMemcpyAsync(&foreground, buffers->d_count, sizeof(unsigned int),
                          cudaMemcpyDeviceToHost, stream),
          "counter download");
    check(cudaStreamSynchronize(stream), "sync");

    return static_cast<std::size_t>(foreground);
}

}  // namespace cuda
}  // namespace clpipe
