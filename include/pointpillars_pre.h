#pragma once 

#include <torch/torch.h>
#include <tuple>
#include <vector>

class PointPillarsPre {
    public:
        PointPillarsPre(const std::vector<float>& voxel_size,
                            const std::vector<float>& point_cloud_range, 
                            int max_num_points,
                            std::pair<int, int> max_voxels,
                            bool is_training,
                            bool deterministic,
                            bool use_gpu);
        ~PointPillarsPre();

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(std::vector<torch::Tensor>& batched_pts);

    private:
        std::vector<float> voxel_size_;
        std::vector<float> point_cloud_range_;
        int max_num_points_;
        std::pair<int, int> max_voxels_;
        bool is_training_;
        bool deterministic_;
        bool prepost_process_gpu_;
};
