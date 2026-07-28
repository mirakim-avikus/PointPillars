#pragma once 

#include <torch/torch.h>

class PillarLayer {
    public:
        PillarLayer(std::vector<float> voxel_size,
                        std::vector<float> point_cloud_range,
                        int max_num_points,
                        std::pair<int, int> max_voxels,
                        bool use_gpu,
                        bool deterministic, 
                        bool training);
        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const std::vector<torch::Tensor>& batched_pts);

    private:
        std::unique_ptr<Voxelization> voxel_layer;
};
