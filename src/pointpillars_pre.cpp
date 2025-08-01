#include "pointpillars_pre.h"
#include "voxelization.h"
#include <torch/nn/functional/padding.h>

extern bool SAVE_TENSOR;

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

void saveTensorToBin(const torch::Tensor& tensor, const std::string& filename) {
    torch::Tensor contiguous = tensor.contiguous();
    torch::ScalarType dtype = contiguous.scalar_type();  // ✅ FIXED

    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "파일 열기 실패: " << filename << "\n";
        return;
    }

    size_t num_elements = contiguous.numel();

    if (dtype == torch::kFloat32) {
        float* data_ptr = contiguous.data_ptr<float>();
        out.write(reinterpret_cast<char*>(data_ptr), num_elements * sizeof(float));
    } else if (dtype == torch::kInt32) {
        int32_t* data_ptr = contiguous.data_ptr<int32_t>();
        out.write(reinterpret_cast<char*>(data_ptr), num_elements * sizeof(int32_t));
    } else if (dtype == torch::kInt64) {
        int64_t* data_ptr = contiguous.data_ptr<int64_t>();
        out.write(reinterpret_cast<char*>(data_ptr), num_elements * sizeof(int64_t));
    } else {
        std::cerr << "지원되지 않는 dtype입니다: " << dtype << "\n";
    }

    out.close();
}

template <typename T>
std::vector<T> loadTensorFromBin(const std::string& filename, const std::vector<int64_t>& shape, torch::Dtype dtype) {
    size_t num_elements = 1;
    for (auto s : shape) num_elements *= s;

    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "파일 열기 실패: " << filename << "\n";
        return {};
    }

    std::vector<T> buffer(num_elements);
    in.read(reinterpret_cast<char*>(buffer.data()), num_elements * sizeof(T));
    in.close();

    return buffer;
}

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
    torch::Tensor coors_batch_tensor = torch::cat(coors_list, 0);

    if (SAVE_TENSOR) {
        saveTensorToBin(pillars, "pillars.bin");
        saveTensorToBin(coors_batch_tensor, "coors.bin");
        saveTensorToBin(npoints_per_pillar, "npoints.bin");

        std::vector<int64_t> pillars_shape(pillars.sizes().begin(), pillars.sizes().end());
        std::vector<int64_t> coors_shape(coors_batch_tensor.sizes().begin(), coors_batch_tensor.sizes().end());
        std::vector<int64_t> npoints_shape(npoints_per_pillar.sizes().begin(), npoints_per_pillar.sizes().end());

        std::vector<float> t1 = loadTensorFromBin<float>("pillars.bin", pillars_shape, torch::kFloat32);
        std::vector<int64_t> t2 = loadTensorFromBin<int64_t>("coors.bin", coors_shape, torch::kInt64);
        std::vector<int32_t> t3 = loadTensorFromBin<int32_t>("npoints.bin", npoints_shape, torch::kInt32);
    }

    return std::make_tuple(pillars, coors_batch_tensor, npoints_per_pillar);
}