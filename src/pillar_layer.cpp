#include "pillar_layer.h"
#include <torch/nn/functional/padding.h>

PillarLayer::PillarLayer(std::vector<float> voxel_size,
                                std::vector<float> point_cloud_range,
                                int max_num_points,
                                std::pair<int, int> max_voxels,
                                bool training)
    : voxel_layer(voxel_size, point_cloud_range, max_num_points, max_voxels, training) {
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PillarLayer::forward(const std::vector<torch::Tensor>& batched_pts) {
    std::vector<torch::Tensor> pillars_list, coors_list, npoints_list;

    for (size_t i = 0; i < batched_pts.size(); i++) {
        torch::Tensor voxels_out, coors_out, npoints_out;
        std::tie(voxels_out, coors_out, npoints_out) = voxel_layer.forward(batched_pts[i]);
        pillars_list.push_back(voxels_out);
        coors_list.push_back(coors_out);
        npoints_list.push_back(npoints_out);
    }

    torch::Tensor pillars = torch::cat(pillars_list, 0);
    torch::Tensor npoints_per_pillar = torch::cat(npoints_list, 0);

    std::vector<torch::Tensor> coors_batch;
    for (size_t i = 0; i < coors_list.size(); i++)
    {
        // Pad with batch idx : shape becomes (N, 1+3)
        auto batch_idx = torch::full({coors_list[i].size(0), 1}, static_cast<int64_t>(i), torch::dtype(torch::kLong).device(coors_list[i].device()));
        auto padded = torch::cat({batch_idx, coors_list[i]}, 1);
        coors_batch.push_back(padded);
    }

    torch::Tensor coors_batch_tensor = torch::cat(coors_batch, 0);

    return std::make_tuple(pillars, coors_batch_tensor, npoints_per_pillar);
}