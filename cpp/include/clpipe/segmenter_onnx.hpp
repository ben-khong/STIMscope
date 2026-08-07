#pragma once

// Internal seam to the ONNX Runtime translation unit. Only segmenter_onnx.cpp
// includes the ORT headers; everything else compiles with no knowledge that it
// exists. Guarded so a build without ORT has no dangling declaration to link.

#include <memory>
#include <string>

#include "clpipe/segmenter.hpp"

#ifdef CLPIPE_WITH_ONNXRUNTIME

namespace clpipe {

std::unique_ptr<Segmenter> make_onnx_segmenter(const std::string& model_path, int rows, int cols);

std::string onnx_runtime_version_string();

}  // namespace clpipe

#endif  // CLPIPE_WITH_ONNXRUNTIME
