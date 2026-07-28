#include "pointpillars_pipeline.h"
#include <iostream>
#include <filesystem>


PointPillarsPipeline::PointPillarsPipeline(const std::string& model_path, 
                                const std::vector<float>& voxel_size,
                                const std::vector<float>& point_cloud_range,
                                int max_num_points,
                                std::pair<int, int> max_voxels,
                                bool training,
                                bool deterministic,
                                bool prepost_use_gpu,
                                bool core_use_gpu,
                                int nclasses,
                                float nms_thres,
                                float score_thres,
                                int max_num) 
    : prepost_use_gpu_(prepost_use_gpu), core_use_gpu_(core_use_gpu),
    preprocessor_(voxel_size, point_cloud_range, max_num_points, max_voxels, training, deterministic, prepost_use_gpu_),
    postprocessor_(nclasses, nms_thres, score_thres, max_num, prepost_use_gpu_),
    coreprocessor_(model_path, core_use_gpu_) {}

std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PointPillarsPipeline::run(torch::Tensor& pt_tensor) {
    std::cout << "prepost_use_gpu_ : " << static_cast<int64_t>(prepost_use_gpu_) << std::endl;
    if (prepost_use_gpu_) {
        pt_tensor = pt_tensor.cuda();
    }
    std::cout << "pt tensor cuda : " << pt_tensor.device() << std::endl;
    std::vector<torch::Tensor> pt_tensor_vec = {pt_tensor};
    auto [pillars, coors, npoints] = preprocessor_.forward(pt_tensor_vec);

    std::vector<Ort::Value> inference_results = coreprocessor_.runInference(pillars, coors, npoints);

    torch::Tensor output_tensor = OrtValueToTensor(inference_results[0]);

    return postprocessor_.forward(output_tensor);
}

torch::Tensor PointPillarsPipeline::OrtValueToTensor(Ort::Value& ort_value) {
    assert (ort_value.IsTensor());

    Ort::TensorTypeAndShapeInfo shape_info = ort_value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();

    float* data_ptr = ort_value.GetTensorMutableData<float>();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor tensor = torch::from_blob(data_ptr, shape, options).clone();
    return tensor;
}