#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <array>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <limits>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <torch/torch.h>
#include "pointpillars_pipeline.h"
#include <assert.h>
#include <chrono>
#include <onnxruntime_cxx_api.h>

constexpr int MAX_PILLARS = 40000;
constexpr int MAX_POINTS_PER_PILLAR = 32;
constexpr int PILLAR_FEATURE_DIM = 4;
constexpr int COORS_DIM = 4;

bool PREPOST_PROCESS_GPU = false;
bool PREPOST_PROCESS_DETERMINISTIC = false;
bool CORE_PROCESS_GPU = false;

torch::Tensor OrtValueToTensor(Ort::Value& ort_value) {
    assert (ort_value.IsTensor());

    Ort::TensorTypeAndShapeInfo shape_info = ort_value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();

    float* data_ptr = ort_value.GetTensorMutableData<float>();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor tensor = torch::from_blob(data_ptr, shape, options).clone();
    return tensor;
}
    
std::vector<Ort::Value> run_onnx_inference(torch::Tensor& pillars, torch::Tensor& coors, torch::Tensor& npoints, const std::string& model_path, Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator, Ort::MemoryInfo& memory_info, std::vector<const char*>& input_names, std::vector<const char*>& output_names, Ort::RunOptions& run_options) {
    // onnx setting 
    if (CORE_PROCESS_GPU && pillars.is_cpu())
    {   
        std::cout << "[onnx] : use cuda..." << std::endl;
        pillars = pillars.cuda();
        coors = coors.cuda();
        npoints = npoints.cuda();
    } else if (!CORE_PROCESS_GPU && pillars.is_cuda()) {
        std::cout << "[onnx] : use CPU..." << std::endl;
        pillars = pillars.cpu();
        coors = coors.cpu();
        npoints = npoints.cpu();
    }

    Ort::Value input_tensor_pillars = Ort::Value::CreateTensor<float>(
        memory_info,
        (float*)pillars.contiguous().data_ptr(),
        pillars.numel(),
        pillars.sizes().data(),
        pillars.dim()
    );

    Ort::Value input_tensor_coors = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        (int64_t*)coors.contiguous().data_ptr(),
        coors.numel(),
        coors.sizes().data(),
        coors.dim()
    );

    Ort::Value input_tensor_npoints = Ort::Value::CreateTensor<int32_t>(
        memory_info,
        (int32_t*)npoints.contiguous().data_ptr(),
        npoints.numel(),
        npoints.sizes().data(),
        npoints.dim()
    );

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor_pillars));
    ort_inputs.push_back(std::move(input_tensor_coors));
    ort_inputs.push_back(std::move(input_tensor_npoints));

    std::vector<Ort::Value> output_tensors = session.Run(run_options, 
                                        input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                        output_names.data(), output_names.size());

    std::cout << "onnx inference done ..." << std::endl;
    return output_tensors;
}

void calculate_mean_std(std::vector<double>& vec, int total_size, std::string key) {
    // mean 
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double mean = sum / total_size;

    // std 
    double variance = 0.0;
    for (const auto& val : vec) {
        variance += (val - mean) * (val - mean);
    }
    variance /= total_size;

    double std_dev = std::sqrt(variance);

    std::cout << key << " : " << mean << " , " << std_dev << std::endl;
}

void save_vector_to_csv(const std::vector<double>& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "cannot open file : " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); i++) {
        file << data[i];
        if (i != data.size() - 1) {
            file << ","; 
        }
    }
    file << "\n";

    file.close();
    std::cout << "csv file saved complete! " << filename << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <onnx_model_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string root_path = argv[2];

    std::vector<std::string> pcd_files;
    for (const auto& entry : std::filesystem::directory_iterator(root_path)) {
        if (entry.path().extension() == ".pcd") {
            pcd_files.push_back(entry.path().string());
        }
    }

    std::sort(pcd_files.begin(), pcd_files.end());

    std::cout << "# total : " << pcd_files.size() << std::endl;

    std::chrono::duration<double, std::milli> min_pre_elapsed(std::numeric_limits<double>::infinity());
    std::chrono::duration<double, std::milli> min_onnx_elapsed(std::numeric_limits<double>::infinity());
    std::chrono::duration<double, std::milli> min_post_elapsed(std::numeric_limits<double>::infinity());

    std::vector<double> vec_pre;
    std::vector<double> vec_onnx;
    std::vector<double> vec_post;

    int num_failed = 0;

    // point cloud range
    int max_num_points = 32;
    std::vector<float> point_cloud_range = {0.0f, -50.0f, -10.0f, 250.0f, 50.0f, 10.0f};
    std::vector<float> voxel_size = {0.16f, 0.16, 4.0f};
    std::pair<int, int> max_voxels = {16000, 40000};
    bool training = false;

    std::cout << "Available providers: " << std::endl;
    for (auto& provider : Ort::GetAvailableProviders()) {
        std::cout << " - " << provider << std::endl; 
    }

    int nclasses = 3;
    float score_thres = 0.1;
    float nms_thres = 0.01;
    int max_num = 50;
    bool prepost_use_gpu = false;
    bool core_use_gpu = true;

    PointPillarsPipeline pointpillars_pipeline(model_path,
                                    voxel_size,
                                    point_cloud_range,
                                    max_num_points,
                                    max_voxels,
                                    training,
                                    PREPOST_PROCESS_DETERMINISTIC,
                                    prepost_use_gpu,
                                    core_use_gpu,
                                    nclasses,
                                    nms_thres,
                                    score_thres,
                                    max_num);

    for (const auto& file_path : pcd_files) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        std::filesystem::path pcd_path = file_path;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path, *cloud) == -1) 
        {
            PCL_ERROR("Couldn't read the file.\n");
            return -1;
        }

        std::cout << "Loaded " << cloud->width * cloud->height << " data points from PCD file." << std::endl;

        for (size_t i = 0; i < std::min((size_t)5, cloud->points.size()); ++i)
        {
            std::cout << "Point[" << i << "] : " << cloud->points[i].x << ", " 
                                                << cloud->points[i].y << ", " 
                                                << cloud->points[i].z << ", "
                                                << cloud->points[i].intensity << std::endl;
        }

        size_t num_points = cloud->points.size();
        torch::Tensor pt_tensor = torch::empty({(long)num_points, 4}, torch::kFloat32);
        for (size_t i = 0; i < num_points; i++)
        {
            pt_tensor[i][0] = cloud->points[i].x;
            pt_tensor[i][1] = cloud->points[i].y;
            pt_tensor[i][2] = cloud->points[i].z;
            pt_tensor[i][3] = cloud->points[i].intensity / 255.;
        }

        std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result = pointpillars_pipeline.run(pt_tensor);
        if (result.has_value()) {
            auto [bboxes, labels, scores] = *result;
            std::cout << "Postprocessed : " << bboxes.size(0) << " bboxes" << std::endl;
            std::cout << "bboxes : " << bboxes << std::endl;
            std::cout << "labels : " << labels << std::endl;
            std::cout << "scores : " << scores << std::endl;
        } else {
            std::cout << "PostProcess has been failed!" << std::endl;
            num_failed++;
        }
    }

    return 0;
}