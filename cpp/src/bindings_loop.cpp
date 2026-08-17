// Bindings for the inference and actuation half of the system.
//
// Same zero-copy contract as bindings.cpp: array arguments are .noconvert(),
// views are built from the caller's buffer, and the GIL is released around the
// cycle so acquisition keeps running while C++ is working.
//
// `last_pattern` goes the other way — it hands Python a read-only view onto the
// DMD's own buffer, with the controller as the array's base object so it cannot
// be freed underneath. That is what lets the simulated sample read what was
// projected without the pattern ever being copied across the boundary.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>

#include "clpipe/controller.hpp"
#include "clpipe/dmd.hpp"
#include "clpipe/segmenter.hpp"

namespace py = pybind11;

namespace {

/// Describe a 2D uint8 array as a View2D without copying. Rows may be padded;
/// elements within a row must be adjacent.
template <typename T, typename Array>
clpipe::View2D<T> as_view(Array& array, const char* name) {
    if (array.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2D array");
    }
    if (array.strides(1) != static_cast<py::ssize_t>(sizeof(std::uint8_t))) {
        throw std::invalid_argument(std::string(name) +
                                    " must have contiguous rows (stride 1 along axis 1)");
    }

    clpipe::View2D<T> view;
    view.rows = static_cast<int>(array.shape(0));
    view.cols = static_cast<int>(array.shape(1));
    view.stride = static_cast<std::ptrdiff_t>(array.strides(0) /
                                              static_cast<py::ssize_t>(sizeof(std::uint8_t)));
    return view;
}

}  // namespace

void bind_closed_loop(py::module_& m) {
    m.def("built_with_onnxruntime", &clpipe::built_with_onnxruntime,
          "True if the extension was compiled against ONNX Runtime.");
    m.def("onnxruntime_version", &clpipe::onnxruntime_version,
          "Version of the linked ONNX Runtime, or an empty string without it.");

    py::class_<clpipe::CycleResult>(m, "CycleResult")
        .def_readonly("active_pixels", &clpipe::CycleResult::active_pixels)
        .def_readonly("stimulated_pixels", &clpipe::CycleResult::stimulated_pixels)
        .def_readonly("peak_confidence", &clpipe::CycleResult::peak_confidence)
        .def_readonly("segment_ms", &clpipe::CycleResult::segment_ms)
        .def_readonly("actuate_ms", &clpipe::CycleResult::actuate_ms)
        .def_property_readonly("total_ms", &clpipe::CycleResult::total_ms)
        .def("__repr__", [](const clpipe::CycleResult& self) {
            return "<CycleResult active=" + std::to_string(self.active_pixels) +
                   " stimulated=" + std::to_string(self.stimulated_pixels) + " total=" +
                   std::to_string(self.total_ms()) + "ms>";
        });

    py::class_<clpipe::DmdStats>(m, "DmdStats")
        .def_readonly("patterns_written", &clpipe::DmdStats::patterns_written)
        .def_readonly("mirrors_on_total", &clpipe::DmdStats::mirrors_on_total)
        .def_readonly("last_mirrors_on", &clpipe::DmdStats::last_mirrors_on);

    py::class_<clpipe::ClosedLoopController>(m, "ClosedLoopController")
        .def(py::init([](int rows, int cols, const std::string& segmenter,
                         std::uint8_t activation_threshold) {
                 return new clpipe::ClosedLoopController(rows, cols, segmenter,
                                                         activation_threshold);
             }),
             py::arg("rows"), py::arg("cols"), py::arg("segmenter") = "builtin",
             py::arg("activation_threshold") = 128)
        .def(
            "process",
            [](clpipe::ClosedLoopController& self, py::array_t<std::uint8_t> frame,
               py::object motion, py::object confidence) {
                auto frame_view = as_view<const std::uint8_t>(frame, "frame");
                frame_view.data = frame.data();

                clpipe::PatternView motion_view;
                py::array_t<std::uint8_t> motion_array;
                if (!motion.is_none()) {
                    motion_array = py::cast<py::array_t<std::uint8_t>>(motion);
                    motion_view = as_view<const std::uint8_t>(motion_array, "motion");
                    motion_view.data = motion_array.data();
                }

                clpipe::ConfidenceView confidence_view;
                py::array_t<std::uint8_t> confidence_array;
                if (!confidence.is_none()) {
                    confidence_array = py::cast<py::array_t<std::uint8_t>>(confidence);
                    if (!confidence_array.writeable()) {
                        throw std::invalid_argument("confidence must be writeable");
                    }
                    confidence_view = as_view<std::uint8_t>(confidence_array, "confidence");
                    confidence_view.data = confidence_array.mutable_data();
                }

                py::gil_scoped_release unlocked;
                return self.process(frame_view, motion_view, confidence_view);
            },
            py::arg("frame").noconvert(), py::arg("motion") = py::none(),
            py::arg("confidence") = py::none(),
            "Segment `frame`, gate the result by `motion`, write the DMD pattern.")
        .def("blank", &clpipe::ClosedLoopController::blank,
             "Turn every mirror off. Called when there is no usable frame.")
        .def(
            "last_pattern",
            [](py::object self_object) {
                auto& self = self_object.cast<clpipe::ClosedLoopController&>();
                // self_object as base keeps the controller alive while the view
                // exists; the write flag is cleared because this aliases the
                // device's own buffer and must not be edited from Python.
                py::array_t<std::uint8_t> view(
                    {self.rows(), self.cols()},
                    {static_cast<py::ssize_t>(self.cols()), static_cast<py::ssize_t>(1)},
                    self.last_pattern(), self_object);
                view.attr("setflags")(py::arg("write") = false);
                return view;
            },
            "Read-only view of the pattern currently on the device. No copy.")
        .def_property_readonly("segmenter", &clpipe::ClosedLoopController::segmenter_name)
        .def_property_readonly("dmd_stats", &clpipe::ClosedLoopController::dmd_stats)
        .def_property_readonly("rows", &clpipe::ClosedLoopController::rows)
        .def_property_readonly("cols", &clpipe::ClosedLoopController::cols)
        .def_property("activation_threshold",
                      &clpipe::ClosedLoopController::activation_threshold,
                      &clpipe::ClosedLoopController::set_activation_threshold)
        .def("__repr__", [](const clpipe::ClosedLoopController& self) {
            return "<ClosedLoopController " + std::to_string(self.rows()) + "x" +
                   std::to_string(self.cols()) + " segmenter=" + self.segmenter_name() + ">";
        });
}
