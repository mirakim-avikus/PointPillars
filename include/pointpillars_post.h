#pragma once

#include <torch/torch.h>
#include <optional>
#include <tuple>

int nms_gpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);
int nms_cpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id);

class PointPillarsPost {
    public:
        PointPillarsPost(int nclasses = 3,
                        float nms_thres = 0.01,
                        float score_thres = 0.1,
                        int max_num = 50,
                        bool use_gpu = false);
        ~PointPillarsPost();

        std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> forward(torch::Tensor& result);
    private:
        int nclasses_;
        float nms_thrs_;
        float score_thrs_;
        int max_num_;
        bool prepost_use_gpu_;

        torch::Tensor nms(const torch::Tensor& boxes, const torch::Tensor& scores, float thresh,
                                    c10::optional<int> pre_maxsize = c10::nullopt,
                                    c10::optional<int> post_maxsize = c10::nullopt);

        std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> nms_filter(const torch::Tensor& bbox_pred, const torch::Tensor& bbox_cls_pred, const torch::Tensor& bbox_dir_cls_pred);
        torch::Tensor limit_period(const torch::Tensor& val, float offset, float period);
};