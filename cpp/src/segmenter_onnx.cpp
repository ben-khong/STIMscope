// ONNX Runtime backend.
//
// This is the part that corresponds to "deploys and runs the model in real
// time": the model is somebody else's artefact, and the job here is to load it
// once, hand it pixels at frame rate, and get a map back without allocating or
// copying more than necessary.
//
// Three decisions worth stating, because they are the ones that bite in a
// real-time loop:
//
//   * The session, the allocator, the input tensor and the output buffer are
//     all created once in the constructor. Per-frame allocation inside a loop
//     with a deadline shows up as p99 latency long before it shows up in a mean.
//   * Intra-op threading is pinned to one thread. ORT defaults to filling the
//     machine, which on a Jetson means fighting the acquisition thread for
//     cores and making latency *worse* under load.
//   * Input geometry is validated against the frame at construction. A model
//     whose spatial dims disagree with the camera is a configuration error, and
//     it should fail at startup rather than halfway through an experiment.

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "clpipe/segmenter_onnx.hpp"

namespace clpipe {
namespace {

class OnnxSegmenter : public Segmenter {
public:
    OnnxSegmenter(const std::string& model_path, int rows, int cols)
        : rows_(rows),
          cols_(cols),
          env_(ORT_LOGGING_LEVEL_WARNING, "clpipe"),
          memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), options);
        } catch (const Ort::Exception& e) {
            throw std::runtime_error("failed to load ONNX model '" + model_path +
                                     "': " + e.what());
        }

        if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
            throw std::runtime_error(
                "expected a single-input, single-output segmentation model");
        }

        Ort::AllocatorWithDefaultOptions allocator;
        input_name_ = session_->GetInputNameAllocated(0, allocator).get();
        output_name_ = session_->GetOutputNameAllocated(0, allocator).get();
        input_names_[0] = input_name_.c_str();
        output_names_[0] = output_name_.c_str();

        const auto shape =
            session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4) {
            throw std::runtime_error("expected an NCHW input of rank 4, got rank " +
                                     std::to_string(shape.size()));
        }
        // Negative entries are ORT's marker for a dynamic axis, which we are free
        // to bind to the camera geometry. A fixed axis has to match it.
        check_axis(shape[2], rows, "height");
        check_axis(shape[3], cols, "width");

        input_shape_ = {1, 1, static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)};
        input_buffer_.assign(static_cast<std::size_t>(rows) * cols, 0.0f);

        description_ = "onnx:" + basename(model_path);
    }

    SegmentationResult run(ImageView frame, ConfidenceView confidence,
                           std::uint8_t activation_threshold) override {
        if (frame.rows != rows_ || frame.cols != cols_) {
            throw std::invalid_argument("frame shape does not match the model's geometry");
        }
        if (confidence.rows != rows_ || confidence.cols != cols_) {
            throw std::invalid_argument("confidence shape does not match the model's geometry");
        }

        // uint8 -> float in [0, 1], into the buffer allocated at construction.
        for (int r = 0; r < rows_; ++r) {
            const std::uint8_t* src = frame.row(r);
            float* dst = input_buffer_.data() + static_cast<std::size_t>(r) * cols_;
            for (int c = 0; c < cols_; ++c) {
                dst[c] = static_cast<float>(src[c]) * (1.0f / 255.0f);
            }
        }

        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info_, input_buffer_.data(), input_buffer_.size(), input_shape_.data(),
            input_shape_.size());

        auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), &input, 1,
                                     output_names_.data(), 1);
        if (outputs.empty() || !outputs[0].IsTensor()) {
            throw std::runtime_error("model returned no output tensor");
        }

        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const std::size_t expected = static_cast<std::size_t>(rows_) * cols_;
        if (info.GetElementCount() != expected) {
            throw std::runtime_error("model output has " +
                                     std::to_string(info.GetElementCount()) +
                                     " elements, expected " + std::to_string(expected));
        }

        const float* scores = outputs[0].GetTensorData<float>();

        SegmentationResult result;
        for (int r = 0; r < rows_; ++r) {
            std::uint8_t* out = confidence.row(r);
            const float* row = scores + static_cast<std::size_t>(r) * cols_;
            for (int c = 0; c < cols_; ++c) {
                const float clamped = std::min(1.0f, std::max(0.0f, row[c]));
                const std::uint8_t value =
                    static_cast<std::uint8_t>(std::lround(clamped * 255.0f));
                out[c] = value;
                if (value >= activation_threshold) ++result.active_pixels;
                if (value > result.peak_confidence) result.peak_confidence = value;
            }
        }
        return result;
    }

    const char* name() const override { return description_.c_str(); }

private:
    static void check_axis(std::int64_t declared, int actual, const char* label) {
        if (declared > 0 && declared != static_cast<std::int64_t>(actual)) {
            throw std::runtime_error(std::string("model ") + label + " is " +
                                     std::to_string(declared) + " but frames are " +
                                     std::to_string(actual));
        }
    }

    static std::string basename(const std::string& path) {
        const auto slash = path.find_last_of("/\\");
        return slash == std::string::npos ? path : path.substr(slash + 1);
    }

    int rows_;
    int cols_;
    Ort::Env env_;
    Ort::MemoryInfo memory_info_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;
    std::array<const char*, 1> input_names_{};
    std::array<const char*, 1> output_names_{};

    std::array<std::int64_t, 4> input_shape_{};
    std::vector<float> input_buffer_;
    std::string description_;
};

}  // namespace

std::unique_ptr<Segmenter> make_onnx_segmenter(const std::string& model_path, int rows,
                                               int cols) {
    return std::unique_ptr<Segmenter>(new OnnxSegmenter(model_path, rows, cols));
}

std::string onnx_runtime_version_string() { return std::string(OrtGetApiBase()->GetVersionString()); }

}  // namespace clpipe
