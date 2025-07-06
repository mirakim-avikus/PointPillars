#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <iostream>
#include <array>
#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <numeric>
#include <torch/torch.h>
#include "pillar_layer.h"
#include "iou3d.h"
#include <assert.h>
#include <onnxruntime_cxx_api.h>

constexpr int MAX_PILLARS = 40000;
constexpr int MAX_POINTS_PER_PILLAR = 32;
constexpr int PILLAR_FEATURE_DIM = 4;
constexpr int COORS_DIM = 4;

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);

torch::Tensor OrtValueToTensor(Ort::Value& ort_value) {
    assert (ort_value.IsTensor());

    Ort::TensorTypeAndShapeInfo shape_info = ort_value.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();

    float* data_ptr = ort_value.GetTensorMutableData<float>();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor tensor = torch::from_blob(data_ptr, shape, options).clone();
    return tensor;
}

class PointPillarsPre {
    public:
    PointPillarsPre(const std::vector<float>& voxel_size,
                    const std::vector<float>& point_cloud_range,
                    int max_num_points = 32,
                    std::pair<int, int> max_voxels = {16000, 40000},
                    bool training = false)
        : pillar_layer_(voxel_size, point_cloud_range, max_num_points, max_voxels, training) {}

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(const std::vector<torch::Tensor>& batched_pts) {
        return pillar_layer_.forward(batched_pts);
    }
    private:
    PillarLayer pillar_layer_;
};

class PointPillarPost {
    public:
    PointPillarPost(int nclasses=3,
                        float nms_thrs=0.01,
                        float score_thrs=0.1,
                        int max_num=50)
        : nclasses_(nclasses), nms_thrs_(nms_thrs), score_thrs_(score_thrs), max_num_(max_num) {}

    std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> forward(const torch::Tensor&result) {
        torch::Tensor bbox_pred = result.index({torch::indexing::Slice(), torch::indexing::Slice({0, 7})}).cuda();
        torch::Tensor cls_pred = result.index({torch::indexing::Slice(), torch::indexing::Slice({7, 10})}).cuda();
        torch::Tensor dir_cls_pred = result.index({torch::indexing::Slice(), 10}).cuda();
        auto nms_result = nms_filter(bbox_pred, cls_pred, dir_cls_pred);
        if (nms_result.has_value()) {
            return nms_result;
        } else {
            std::cout << "No results for nms_filter" << std::endl;
            return std::nullopt;
        }
    }

    private:
    int nclasses_;
    float nms_thrs_;
    float score_thrs_;
    int max_num_;

    torch::Tensor nms_cuda(const torch::Tensor& boxes, const torch::Tensor& scores, float thresh,
                            c10::optional<int> pre_maxsize = c10::nullopt,
                            c10::optional<int> post_maxsize = c10::nullopt) {
        auto order_with_values = scores.sort(/*dim=*/0, /*descending=*/true);
        auto order = std::get<1>(order_with_values);

        if (pre_maxsize.has_value()) {
            order = order.index({torch::indexing::Slice(0, pre_maxsize.value())});
        }

        torch::Tensor boxes_ordered = boxes.index_select(0, order).contiguous();
        auto keep = torch::empty({boxes_ordered.size(0)}, torch::dtype(torch::kLong));
        
        std::cout << "thresh : " << thresh << std::endl;
        int kept = nms_gpu(boxes_ordered, keep, thresh, boxes.device().index());
        keep = keep.narrow(0, 0, kept).to(order.device());
        torch::Tensor keep_original = order.index_select(0, keep).contiguous();

        if (post_maxsize.has_value()) {
            keep_original = keep_original.index({torch::indexing::Slice(0, (post_maxsize.value()))});
        }
        return keep_original;
    }

    std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> nms_filter(const torch::Tensor& bbox_pred, const torch::Tensor& bbox_cls_pred, const torch::Tensor& bbox_dir_cls_pred) {
        torch::Tensor bbox_pred2d_xy = bbox_pred.index({torch::indexing::Slice(), torch::indexing::Slice({0, 2})});
        torch::Tensor bbox_pred2d_lw = bbox_pred.index({torch::indexing::Slice(), torch::indexing::Slice({3, 5})});
        torch::Tensor bbox_rot = bbox_pred.index({torch::indexing::Slice(), torch::indexing::Slice({6, torch::indexing::None})});

        torch::Tensor bbox_min = bbox_pred2d_xy - bbox_pred2d_lw / 2;
        torch::Tensor bbox_max = bbox_pred2d_xy + bbox_pred2d_lw / 2;
        torch::Tensor bbox_pred2d = torch::cat({bbox_min, bbox_max, bbox_rot}, /*dim=*/-1);
    
        std::vector<torch::Tensor> ret_bboxes, ret_labels, ret_scores;    
        for (int cls = 0; cls < nclasses_; cls++)
        {
            torch::Tensor cur_bbox_cls_pred = bbox_cls_pred.index({torch::indexing::Slice(), cls});
            torch::Tensor score_inds = cur_bbox_cls_pred > score_thrs_;
            if (score_inds.sum().item<int>() == 0) continue;
            
            torch::Tensor inds = score_inds.nonzero().squeeze(1);
            cur_bbox_cls_pred = cur_bbox_cls_pred.index_select(0, inds);
            torch::Tensor cur_bbox_pred2d = bbox_pred2d.index_select(0, inds);
            torch::Tensor cur_bbox_pred = bbox_pred.index_select(0, inds);
            torch::Tensor cur_bbox_dir_cls_pred = bbox_dir_cls_pred.index_select(0, inds);

            std::cout << "nms_thrs_ : " << nms_thrs_ << std::endl;
            torch::Tensor keep_inds = nms_cuda(cur_bbox_pred2d, cur_bbox_cls_pred, nms_thrs_);

            cur_bbox_cls_pred = cur_bbox_cls_pred.index_select(0, keep_inds);
            cur_bbox_pred = cur_bbox_pred.index_select(0, keep_inds);
            cur_bbox_dir_cls_pred = cur_bbox_dir_cls_pred.index_select(0, keep_inds);

            torch::Tensor rot = limit_period(cur_bbox_pred.index({torch::indexing::Slice(), -1}).detach().cpu(), 1, M_PI);
            rot += (1 - cur_bbox_dir_cls_pred).to(torch::kFloat32).to(rot.device()) * M_PI;
            cur_bbox_pred.index_put_({torch::indexing::Slice(), -1}, rot.to(cur_bbox_pred.device()));

            ret_bboxes.push_back(cur_bbox_pred);
            ret_labels.push_back(torch::full({cur_bbox_pred.size(0)}, cls, torch::dtype(torch::kLong).device(cur_bbox_pred.device())));
            ret_scores.push_back(cur_bbox_cls_pred);
        }
        
        if (ret_bboxes.empty()) return std::nullopt;

        auto bboxes = torch::cat(ret_bboxes, 0);
        auto labels = torch::cat(ret_labels, 0);
        auto scores = torch::cat(ret_scores, 0);

        if (bboxes.size(0) > max_num_) {
            auto topk = std::get<1>(scores.topk(max_num_));
            bboxes = bboxes.index_select(0, topk);
            labels = labels.index_select(0, topk);
            scores = scores.index_select(0, topk);
        }
        return std::make_tuple(bboxes, labels, scores);
    }

    torch::Tensor limit_period(const torch::Tensor& val, float offset, float period) {
        auto div = val / period + offset;
        auto floored = div.floor();
        return val - floored * period;
    }
};

    
std::vector<Ort::Value> run_onnx_inference(const torch::Tensor& pillars, const torch::Tensor& coors, const torch::Tensor& npoints, const std::string& model_path) {
    // onnx setting 
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PointPillarsONNX");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef USE_CUDA
    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif

    Ort::Session session(env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    size_t num_inputs = session.GetInputCount();
    std::vector<const char*> input_names;
    for (size_t i = 0; i < num_inputs; i++) {
        input_names.push_back(session.GetInputName(i, allocator));
    }

    size_t num_outputs = session.GetOutputCount();
    std::vector<const char*> output_names;
    for (size_t i = 0; i < num_outputs; i++) {
        output_names.push_back(session.GetOutputName(i, allocator));
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

    std::cout << "dim list : " << pillars.dim() << " , " << coors.dim() << " , " << npoints.dim() << std::endl;

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor_pillars));
    ort_inputs.push_back(std::move(input_tensor_coors));
    ort_inputs.push_back(std::move(input_tensor_npoints));

    Ort::RunOptions run_options;
    run_options.SetRunLogSeverityLevel(0);

    std::vector<Ort::Value> output_tensors = session.Run(run_options, 
                                        input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                        output_names.data(), output_names.size());

    const float* out_data = output_tensors[0].GetTensorData<float>();
    for (int i = 0; i < 10; ++i) {
        std::cout << out_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "onnx run finished!" << std::endl;
    return output_tensors;
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage : " << argv[0] << " <onnx_model_path>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>("../data/avikus/motorboat/007/lidar/flippedData/1736093516111.avikus.pcd", *cloud) == -1) 
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

    std::vector<torch::Tensor> pt_tensor_vec;
    pt_tensor_vec.push_back(pt_tensor);

    // point cloud range
    std::vector<float> point_cloud_range = {0.0f, -50.0f, -10.0f, 250.0f, 50.0f, 10.0f};
    std::vector<float> voxel_size = {0.16f, 0.16, 4.0f};

    PointPillarsPre preprocess(voxel_size, point_cloud_range);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> results = preprocess.forward(pt_tensor_vec);
    
    torch::Tensor pillars = std::get<0>(results);
    torch::Tensor coors = std::get<1>(results);
    torch::Tensor npoints = std::get<2>(results);

    std::vector<Ort::Value> inference_results = run_onnx_inference(pillars, coors, npoints, model_path);
    torch::Tensor output_tensors = OrtValueToTensor(inference_results[0]);
    std::cout << "output_tensors: " << output_tensors.index({torch::indexing::Slice({0, 5})}) << std::endl;
    int nclass = 3;
    float score_thres = 0.1;
    float nms_thres = 0.01;
    int max_num = 50;
    PointPillarPost postprocess(nclass, nms_thres, score_thres, max_num);
    auto result = postprocess.forward(output_tensors);
    if (result.has_value()) {
        auto [bboxes, labels, scores] = *result;
        std::cout << "Postprocessed : " << bboxes.size(0) << " bboxes" << std::endl;
        std::cout << "bboxes : " << bboxes << std::endl;
        std::cout << "labels : " << labels << std::endl;
        std::cout << "scores : " << scores << std::endl;
    } else {
        std::cout << "PostProcess has been failed!" << std::endl;
    }

    return 0;
}