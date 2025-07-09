#include "pointpillars_pre.h"

PointPillarsPre::PointPillarsPre(std::vector<float> voxel_size,
                                        std::vector<float> point_cloud_range,
                                        int max_num_points,
                                        std::pair<int, int> max_voxels,
                                        bool training, 
                                        bool deterministic,
                                        bool use_gpu)
    : pillar_layer(voxel_size, point_cloud_range, max_num_points, max_voxels, training, deterministic, use_gpu) {
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PointPillarsPre::forward(std::vector<torch::Tensor> batched_pts) {
    return pillar_layer.forward(batched_pts);
}