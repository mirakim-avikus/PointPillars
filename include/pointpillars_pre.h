#pragma once 

#include <torch/torch.h>
#include "pillar_layer.h"

class PointPillarsPre {
    public:
        PointPillarsPre(std::vector<float> voxel_size,
                            std::vector<float> point_cloud_range, 
                            int max_num_points,
                            std::pair<int, int> max_voxels,
                            bool use_gpu);
        
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(std::vector<torch::Tensor> batched_pts);

    private:
        PillarLayer pillar_layer;
};
