#include "pointpillars_core.h"
#include <iostream>

PointPillarsCore::PointPillarsCore(const std::string& model_path, bool use_gpu)
    : core_use_gpu_(use_gpu), 
        env_(ORT_LOGGING_LEVEL_WARNING, "PointPillarsONNX"),
        allocator_(Ort::AllocatorWithDefaultOptions()),
        memory_info_(core_use_gpu_
                        ? Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
                        : Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (core_use_gpu_) {
        OrtCUDAProviderOptions cuda_options;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        input_names_.push_back(session_->GetInputName(i, allocator_));
    }

    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        output_names_.push_back(session_->GetOutputName(i, allocator_));
    }

    run_options_.SetRunLogSeverityLevel(0);
}

PointPillarsCore::~PointPillarsCore() {
    std::cout << "[PointPillarsCore] Destructor called." << std::endl;
}

std::vector<Ort::Value> PointPillarsCore::runInference(torch::Tensor& pillars, torch::Tensor& coors, torch::Tensor& npoints) {
    if (core_use_gpu_)
    {   
        std::cout << "[onnx] : use cuda..." << std::endl;
        if (pillars.is_cpu()) {
            pillars = pillars.cuda();
            coors = coors.cuda();
            npoints = npoints.cuda();
        }
    } else {
        std::cout << "[onnx] : use CPU..." << std::endl;
        if (pillars.is_cuda()) {
            pillars = pillars.cpu();
            coors = coors.cpu();
            npoints = npoints.cpu();
        }
    }

    Ort::Value input_tensor_pillars = Ort::Value::CreateTensor<float>(memory_info_,
        (float*)pillars.contiguous().data_ptr(), pillars.numel(), pillars.sizes().data(), pillars.dim());

    Ort::Value input_tensor_coors = Ort::Value::CreateTensor<int64_t>(memory_info_,
        (int64_t*)coors.contiguous().data_ptr(), coors.numel(), coors.sizes().data(), coors.dim());

    Ort::Value input_tensor_npoints = Ort::Value::CreateTensor<int32_t>(memory_info_,
        (int32_t*)npoints.contiguous().data_ptr(), npoints.numel(), npoints.sizes().data(), npoints.dim());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor_pillars));
    ort_inputs.push_back(std::move(input_tensor_coors));
    ort_inputs.push_back(std::move(input_tensor_npoints));

    Ort::RunOptions run_options;
    run_options.SetRunLogSeverityLevel(0);

    return session_->Run(run_options, input_names_.data(), ort_inputs.data(), ort_inputs.size(),
                        output_names_.data(), output_names_.size());
}

Ort::AllocatorWithDefaultOptions& PointPillarsCore::getAllocator() {
    return allocator_;
}

Ort::MemoryInfo& PointPillarsCore::getMemoryInfo() {
    return memory_info_;
}

const std::vector<const char*>& PointPillarsCore::getInputNames() const {
    return input_names_;
}

const std::vector<const char*>& PointPillarsCore::getOutputNames() const {
    return output_names_;
}
