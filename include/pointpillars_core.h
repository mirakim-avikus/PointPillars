#pragma once

#include <torch/torch.h>
#include <tuple>
#include <vector>
#include <onnxruntime_cxx_api.h>

class PointPillarsCore {
    public:
        PointPillarsCore(const std::string& model_path, bool use_gpu);
        ~PointPillarsCore();

        std::vector<Ort::Value> runInference(torch::Tensor& pillars, torch::Tensor& coors, torch::Tensor& npoints);

        Ort::AllocatorWithDefaultOptions& getAllocator();
        Ort::MemoryInfo& getMemoryInfo();
        const std::vector<const char*>& getInputNames() const;
        const std::vector<const char*>& getOutputNames() const;
    
    private:
        bool core_use_gpu_;
        Ort::Env env_;
        std::unique_ptr<Ort::Session> session_;
        Ort::AllocatorWithDefaultOptions allocator_;
        Ort::MemoryInfo memory_info_;
        Ort::RunOptions run_options_;
        std::vector<const char*> input_names_;
        std::vector<const char*> output_names_;
};
