#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <torch/torch.h>
#include <onnxruntime_cxx_api.h>
#include <optional>
#include <tuple>
#include <vector>
#include <string>

#include "pointpillars_pre.h"
#include "pointpillars_post.h"
#include "pointpillars_core.h"


class PointPillarsPipeline {
    public:
        PointPillarsPipeline(const std::string& model_path, 
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
                                int max_num);
        std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> run(torch::Tensor& pt_tensor);
        torch::Tensor OrtValueToTensor(Ort::Value& ort_value);

    private:
        bool prepost_use_gpu_;
        bool core_use_gpu_;
        PointPillarsPre preprocessor_;
        PointPillarsPost postprocessor_;
        PointPillarsCore coreprocessor_;

};
