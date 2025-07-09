#include "voxelization_pre.h"
#include "voxelization.h"
#include <torch/torch.h>
#include <torch/extension.h>

Voxelization::Voxelization(std::vector<float> voxel_size,
                            std::vector<float> point_cloud_range,
                            int max_num_points,
                            std::pair<int, int> max_voxels,
                            bool is_training,
                            bool deterministic,
                            bool use_gpu)
    : voxel_size_(voxel_size),
    point_cloud_range_(point_cloud_range),
    max_num_points_(max_num_points),
    max_voxels_(max_voxels),
    deterministic_(deterministic),
    use_gpu_(use_gpu),
    is_training_(is_training) {

    auto pcr = torch::tensor(point_cloud_range, torch::kFloat32);
    auto vs = torch::tensor(voxel_size, torch::kFloat32);

    auto grid_size_f = (pcr.slice(0, 3, 6) - pcr.slice(0, 0, 3)) / vs;
    grid_size_ = torch::round(grid_size_f).to(torch::kLong);

    auto input_feat_shape = grid_size_.slice(0, 0, 2);
    pcd_shape_ = {1, input_feat_shape[1].item<int64_t>(), input_feat_shape[0].item<int64_t>()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> voxelization_forward(
    const torch::Tensor& points,
    std::vector<float> voxel_size,
    std::vector<float> coors_range,
    int max_points,
    int max_voxels, 
    bool deterministic,
    bool use_gpu
) 
{
    int ndim = points.size(1);
    
    auto options = points.options();
    auto int_options = options.dtype(torch::kInt32);

    // Allocate output tensors
    torch::Tensor voxels = torch::zeros({max_voxels, max_points, ndim}, options);
    torch::Tensor coors = torch::zeros({max_voxels, 3}, int_options);
    torch::Tensor num_points_per_voxel = torch::zeros({max_voxels}, int_options);

    // Call custom CUDA/CPU kernel
    int voxel_num;
    if (use_gpu) {
        if (deterministic) {
            std::cout << "preprocess : GPU - deterministic hard voxelization!" << std::endl;
            voxel_num = voxelization::hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3);
        } else {
            std::cout << "preprocess : GPU - nondeterministic hard voxelization!" << std::endl;
            voxel_num = voxelization::nondisterministic_hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3);
        }
    } else {
        std::cout << "preprocess : CPU - hard voxelization!" << std::endl;
        voxel_num = voxelization::hard_voxelize_cpu(points, voxels, coors, num_points_per_voxel, voxel_size, coors_range, max_points, max_voxels, 3);
    }

    // Select valid part
    torch::Tensor voxels_out = voxels.slice(0, 0, voxel_num);
    torch::Tensor coors_out = coors.slice(0, 0, voxel_num).flip(-1);
    torch::Tensor num_points_per_voxel_out = num_points_per_voxel.slice(0, 0, voxel_num);

    // Return
    return std::make_tuple(voxels_out, coors_out, num_points_per_voxel_out);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Voxelization::forward(torch::Tensor points) {
    int max_vox = is_training_ ? max_voxels_.first : max_voxels_.second;
    return voxelization_forward(points,
    voxel_size_,
    point_cloud_range_,
    max_num_points_,
    max_vox, 
    deterministic_, 
    use_gpu_);

    throw std::runtime_error("Voxelization::forward() not implemented. You need to bind the custom op.");
}