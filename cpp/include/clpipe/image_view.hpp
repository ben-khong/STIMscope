#pragma once

// A borrowed, non-owning view onto a 2D pixel buffer.
//
// This type is the whole reason the Python/C++ boundary is free. A numpy array
// handed down from the pipeline is described by a pointer, a shape and a row
// stride; wrapping those three things in a struct lets C++ read the exact bytes
// the camera wrote, with no allocation and no memcpy. The view never owns the
// memory, so the Python object must outlive the call — which it does, because
// the call is synchronous.

#include <cstddef>
#include <cstdint>

namespace clpipe {

template <typename T>
struct View2D {
    T* data = nullptr;
    int rows = 0;
    int cols = 0;
    std::ptrdiff_t stride = 0;  // elements (not bytes) between the start of successive rows

    View2D() = default;
    View2D(T* data_, int rows_, int cols_, std::ptrdiff_t stride_)
        : data(data_), rows(rows_), cols(cols_), stride(stride_) {}

    View2D(T* data_, int rows_, int cols_) : View2D(data_, rows_, cols_, cols_) {}

    /// True when rows sit back to back, i.e. the buffer can be treated as flat.
    bool contiguous() const { return stride == static_cast<std::ptrdiff_t>(cols); }

    std::size_t count() const {
        return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    }

    T* row(int r) const { return data + static_cast<std::ptrdiff_t>(r) * stride; }

    bool same_shape_as(const View2D<T>& other) const {
        return rows == other.rows && cols == other.cols;
    }
};

using ImageView = View2D<const std::uint8_t>;  ///< incoming 8-bit frame
using MaskView = View2D<std::uint8_t>;         ///< 0/255 foreground mask
using ModelView = View2D<float>;               ///< running background estimate

}  // namespace clpipe
