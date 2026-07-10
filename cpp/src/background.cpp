#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "clpipe/background.hpp"
#include "clpipe/cuda_backend.hpp"

namespace clpipe {

const char* backend_name(Backend backend) {
    return backend == Backend::Cuda ? "cuda" : "cpu";
}

bool built_with_cuda() {
#ifdef CLPIPE_WITH_CUDA
    return true;
#else
    return false;
#endif
}

bool cuda_available() { return cuda::available(); }

struct BackgroundSubtractor::Impl {
    int rows = 0;
    int cols = 0;
    float alpha = 0.05f;
    int threshold = 25;
    bool seeded = false;
    Backend backend = Backend::Cpu;

    // Host-side model, used by the CPU backend. The CUDA backend keeps its own
    // copy resident on the device instead: shipping a float-per-pixel model
    // across the bus twice per frame would cost more than the arithmetic it
    // feeds, and would make the GPU measurement meaningless.
    std::vector<float> model;

#ifdef CLPIPE_WITH_CUDA
    cuda::Buffers* device = nullptr;
#endif
};

BackgroundSubtractor::BackgroundSubtractor(int rows, int cols, float alpha, int threshold,
                                           bool prefer_cuda)
    : impl_(new Impl()) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("rows and cols must be positive");
    }
    if (!(alpha > 0.0f && alpha <= 1.0f)) {
        throw std::invalid_argument("alpha must lie in (0, 1]");
    }
    if (threshold < 0 || threshold > 255) {
        throw std::invalid_argument("threshold must lie in [0, 255]");
    }

    impl_->rows = rows;
    impl_->cols = cols;
    impl_->alpha = alpha;
    impl_->threshold = threshold;

    const std::size_t n = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

#ifdef CLPIPE_WITH_CUDA
    if (prefer_cuda && cuda::available()) {
        impl_->device = cuda::create(n);
        impl_->backend = Backend::Cuda;
        return;
    }
#else
    (void)prefer_cuda;
#endif

    impl_->model.assign(n, 0.0f);
    impl_->backend = Backend::Cpu;
}

BackgroundSubtractor::~BackgroundSubtractor() {
#ifdef CLPIPE_WITH_CUDA
    if (impl_ && impl_->device != nullptr) {
        cuda::destroy(impl_->device);
        impl_->device = nullptr;
    }
#endif
}

std::size_t BackgroundSubtractor::apply(ImageView frame, MaskView mask) {
    if (frame.rows != impl_->rows || frame.cols != impl_->cols) {
        throw std::invalid_argument("frame shape does not match the subtractor's geometry");
    }
    if (mask.rows != impl_->rows || mask.cols != impl_->cols) {
        throw std::invalid_argument("mask shape does not match the subtractor's geometry");
    }

    [[maybe_unused]] const std::size_t n = frame.count();

    // First frame: the sample has no history, so the best available estimate of
    // the background is the frame itself. Seeding with zeros instead would make
    // the entire first frame read as foreground.
    if (!impl_->seeded) {
        if (impl_->backend == Backend::Cuda) {
#ifdef CLPIPE_WITH_CUDA
            if (!frame.contiguous()) {
                throw std::invalid_argument("CUDA backend requires a contiguous frame");
            }
            cuda::seed(impl_->device, frame.data, n);
#endif
        } else {
            for (int r = 0; r < frame.rows; ++r) {
                const std::uint8_t* src = frame.row(r);
                float* bg = impl_->model.data() + static_cast<std::ptrdiff_t>(r) * frame.cols;
                for (int c = 0; c < frame.cols; ++c) {
                    bg[c] = static_cast<float>(src[c]);
                }
            }
        }

        for (int r = 0; r < mask.rows; ++r) {
            std::fill_n(mask.row(r), mask.cols, static_cast<std::uint8_t>(0));
        }
        impl_->seeded = true;
        return 0;
    }

    if (impl_->backend == Backend::Cuda) {
#ifdef CLPIPE_WITH_CUDA
        if (!frame.contiguous() || !mask.contiguous()) {
            throw std::invalid_argument("CUDA backend requires contiguous frame and mask");
        }
        return cuda::apply(impl_->device, frame.data, mask.data, n, impl_->alpha,
                           impl_->threshold);
#else
        throw std::logic_error("CUDA backend selected in a build without CUDA");
#endif
    }

    ModelView model(impl_->model.data(), impl_->rows, impl_->cols);
    return background_apply_cpu(frame, model, mask, impl_->alpha, impl_->threshold);
}

void BackgroundSubtractor::reset() {
    impl_->seeded = false;
    std::fill(impl_->model.begin(), impl_->model.end(), 0.0f);
}

Backend BackgroundSubtractor::backend() const { return impl_->backend; }
bool BackgroundSubtractor::seeded() const { return impl_->seeded; }
int BackgroundSubtractor::rows() const { return impl_->rows; }
int BackgroundSubtractor::cols() const { return impl_->cols; }
float BackgroundSubtractor::alpha() const { return impl_->alpha; }
int BackgroundSubtractor::threshold() const { return impl_->threshold; }

}  // namespace clpipe
