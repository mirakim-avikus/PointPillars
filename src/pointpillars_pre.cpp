#include "pointpillars_pre.h"
#include "voxelization.h"
#include <torch/nn/functional/padding.h>

int voxelization::hard_voxelize_cpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim);

int voxelization::hard_voxelize_gpu(const at::Tensor &points, at::Tensor &voxels,
                      at::Tensor &coors, at::Tensor &num_points_per_voxel,
                      const std::vector<float> voxel_size,
                      const std::vector<float> coors_range,
                      const int max_points, const int max_voxels,
                      const int NDim);

int voxelization::nondisterministic_hard_voxelize_gpu(const at::Tensor &points, at::Tensor &voxels,
                                        at::Tensor &coors, at::Tensor &num_points_per_voxel,
                                        const std::vector<float> voxel_size,
                                        const std::vector<float> coors_range,
                                        const int max_points, const int max_voxels,
                                        const int NDim);

PointPillarsPre::PointPillarsPre(const std::vector<float>& voxel_size,
                                        const std::vector<float>& point_cloud_range,
                                        int max_num_points,
                                        std::pair<int, int> max_voxels,
                                        bool is_training, 
                                        bool deterministic,
                                        bool use_gpu)
    : voxel_size_(voxel_size),
    point_cloud_range_(point_cloud_range),
    max_num_points_(max_num_points),
    max_voxels_(max_voxels),
    is_training_(is_training), 
    deterministic_(deterministic),
    prepost_process_gpu_(use_gpu) {
}

PointPillarsPre::~PointPillarsPre() {
    std::cout << "[PointPillarsPre] Destructor called." << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> PointPillarsPre::forward(std::vector<torch::Tensor>& batched_pts) {
    std::vector<torch::Tensor> pillars_list, coors_list, npoints_list;
    int max_vox = is_training_ ? max_voxels_.first : max_voxels_.second;

    for (size_t i = 0; i < batched_pts.size(); i++) {
        torch::Tensor points = batched_pts[i];
        int ndim = points.size(1);
        auto options = points.options();
        auto int_options = options.dtype(torch::kInt32);

        // Allocate output tensors
        torch::Tensor voxels = torch::zeros({max_vox, max_num_points_, ndim}, options);
        torch::Tensor coors = torch::zeros({max_vox, 3}, int_options);
        torch::Tensor num_points_per_voxel = torch::zeros({max_vox}, int_options);

        // Call custom CUDA/CPU kernel
        int voxel_num;
        std::cout << "prepost_process_gpu : " << static_cast<int64_t>(prepost_process_gpu_) << std::endl;
        if (prepost_process_gpu_) {
            if (deterministic_) {
                std::cout << "[preprocess] : use cuda - deterministic hard voxelization..." << std::endl;
                voxel_num = voxelization::hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel, voxel_size_, point_cloud_range_, max_num_points_, max_vox, 3);
            } else {
                std::cout << "[preprocess] : use cuda - nondeterministic hard voxelization..." << std::endl;
                voxel_num = voxelization::nondisterministic_hard_voxelize_gpu(points, voxels, coors, num_points_per_voxel, voxel_size_, point_cloud_range_, max_num_points_, max_vox, 3);
            }
        } else {
            std::cout << "[preprocess] : use CPU ..." << std::endl;
            voxel_num = voxelization::hard_voxelize_cpu(points, voxels, coors, num_points_per_voxel, voxel_size_, point_cloud_range_, max_num_points_, max_vox, 3);
        }

        // Select valid part
        torch::Tensor voxels_out = voxels.slice(0, 0, voxel_num);
        torch::Tensor coors_out = coors.slice(0, 0, voxel_num).flip(-1);
        torch::Tensor num_points_per_voxel_out = num_points_per_voxel.slice(0, 0, voxel_num);

        pillars_list.push_back(voxels_out);
        coors_list.push_back(coors_out);
        npoints_list.push_back(num_points_per_voxel_out);
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