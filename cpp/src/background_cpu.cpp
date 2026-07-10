// The scalar, sequential implementation.
//
// This is deliberately the obvious loop: one pixel at a time, in row order, on
// one core. It is the baseline the GPU path is measured against, so it should
// stay readable rather than become a hand-vectorised strawman. It is also the
// fallback that runs on any machine without a GPU, which is most laptops, so it
// has to be correct rather than merely illustrative.

#include <cmath>

#include "clpipe/background.hpp"

namespace clpipe {

std::size_t background_apply_cpu(ImageView frame, ModelView model, MaskView mask, float alpha,
                                 int threshold) {
    const float limit = static_cast<float>(threshold);
    std::size_t foreground = 0;

    for (int r = 0; r < frame.rows; ++r) {
        const std::uint8_t* src = frame.row(r);
        float* bg = model.row(r);
        std::uint8_t* out = mask.row(r);

        for (int c = 0; c < frame.cols; ++c) {
            const float pixel = static_cast<float>(src[c]);
            const float delta = pixel - bg[c];

            if (std::fabs(delta) > limit) {
                out[c] = 255;
                ++foreground;
                // Model deliberately left untouched: see the note in background.hpp.
            } else {
                out[c] = 0;
                bg[c] += alpha * delta;
            }
        }
    }

    return foreground;
}

}  // namespace clpipe
