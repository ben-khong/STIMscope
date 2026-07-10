// pybind11 bindings for the native core.
//
// The contract at this boundary is that nothing is copied. A frame that passed
// the diagnostic gate in Python is already sitting in a numpy buffer; the
// binding wraps that buffer's pointer, shape and stride in a View2D and calls
// straight into C++. No serialisation, no socket, no second process — the
// frame never moves.
//
// Two things enforce that rather than merely claiming it:
//   * every array argument is declared .noconvert(), so a dtype or layout that
//     would require a silent copy raises a TypeError instead;
//   * `probe_pointer` runs an array through the identical conversion path and
//     returns the address C++ saw, which the test suite compares against
//     `array.ctypes.data`.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>

#include "clpipe/background.hpp"

namespace py = pybind11;

namespace {

/// Validate a 2D uint8 array and describe it as a View2D without copying.
///
/// Rows may be padded (a stride greater than the width), which is what a crop
/// of a larger frame looks like, but elements within a row must be adjacent.
template <typename T, typename Array>
clpipe::View2D<T> as_view(Array& array, const char* name) {
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
    if (array.strides(1) != static_cast<py::ssize_t>(sizeof(std::uint8_t))) {
        throw std::invalid_argument(std::string(name) +
                                    " must have contiguous rows (stride 1 along axis 1)");
    }

    const py::ssize_t row_stride = array.strides(0) / static_cast<py::ssize_t>(sizeof(std::uint8_t));

    clpipe::View2D<T> view;
    view.rows = static_cast<int>(array.shape(0));
    view.cols = static_cast<int>(array.shape(1));
    view.stride = static_cast<std::ptrdiff_t>(row_stride);
    return view;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
    m.doc() = "Native core: background subtraction over borrowed numpy buffers.";

    m.attr("__version__") = "0.2.0";

    m.def("built_with_cuda", &clpipe::built_with_cuda,
          "True if the extension was compiled with the CUDA backend included.");
    m.def("cuda_available", &clpipe::cuda_available,
          "True if the CUDA backend was compiled in and a device is present.");

    m.def(
        "probe_pointer",
        [](py::array_t<std::uint8_t> array) {
            return reinterpret_cast<std::uintptr_t>(array.request().ptr);
        },
        py::arg("array").noconvert(),
        "Address of the buffer as seen from C++. Equal to array.ctypes.data iff no copy "
        "was made crossing the boundary.");

    py::class_<clpipe::BackgroundSubtractor>(m, "BackgroundSubtractor")
        .def(py::init([](int rows, int cols, float alpha, int threshold, bool prefer_cuda) {
                 return new clpipe::BackgroundSubtractor(rows, cols, alpha, threshold,
                                                         prefer_cuda);
             }),
             py::arg("rows"), py::arg("cols"), py::arg("alpha") = 0.05f,
             py::arg("threshold") = 25, py::arg("prefer_cuda") = true)
        .def(
            "apply",
            [](clpipe::BackgroundSubtractor& self, py::array_t<std::uint8_t> frame,
               py::array_t<std::uint8_t> mask) {
                if (!mask.writeable()) {
                    throw std::invalid_argument("mask must be writeable");
                }

                auto frame_view = as_view<const std::uint8_t>(frame, "frame");
                auto mask_view = as_view<std::uint8_t>(mask, "mask");
                frame_view.data = frame.data();
                mask_view.data = mask.mutable_data();

                // The compute touches no Python objects, so the GIL is dropped
                // for its duration. That is what lets the ingest thread keep
                // pulling frames while this one is in C++.
                py::gil_scoped_release unlocked;
                return self.apply(frame_view, mask_view);
            },
            py::arg("frame").noconvert(), py::arg("mask").noconvert(),
            "Write the foreground mask for `frame` into `mask`; return the foreground "
            "pixel count.")
        .def("reset", &clpipe::BackgroundSubtractor::reset,
             "Forget the background model; the next frame reseeds it.")
        .def_property_readonly(
            "backend",
            [](const clpipe::BackgroundSubtractor& self) {
                return std::string(clpipe::backend_name(self.backend()));
            },
            "Either \"cpu\" or \"cuda\".")
        .def_property_readonly("seeded", &clpipe::BackgroundSubtractor::seeded)
        .def_property_readonly("rows", &clpipe::BackgroundSubtractor::rows)
        .def_property_readonly("cols", &clpipe::BackgroundSubtractor::cols)
        .def_property_readonly("alpha", &clpipe::BackgroundSubtractor::alpha)
        .def_property_readonly("threshold", &clpipe::BackgroundSubtractor::threshold)
        .def("__repr__", [](const clpipe::BackgroundSubtractor& self) {
            return "<BackgroundSubtractor " + std::to_string(self.rows()) + "x" +
                   std::to_string(self.cols()) + " backend=" +
                   clpipe::backend_name(self.backend()) + ">";
        });
}
