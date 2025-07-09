#pragma once 

#include <torch/torch.h>

class Voxelization {
    public:
        Voxelization(std::vector<float> voxel_size, 
                    std::vector<float> point_cloud_range,
                    int max_num_points,
                    std::pair<int, int> max_voxels,
                    bool is_training,
                    bool use_gpu,
                    bool deterministic = true);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor input);

        // for save / load
        std::vector<float> voxel_size_;
        std::vector<float> point_cloud_range_;
        int max_num_points_;
        std::pair<int, int> max_voxels_;
        bool deterministic_;
        bool use_gpu_;
        bool is_training_;
        torch::Tensor grid_size_;
        std::vector<int64_t> pcd_shape_;
};
