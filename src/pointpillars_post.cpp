#include "pointpillars_post.h"
#include <iostream>
#include "iou3d.h"

PointPillarsPost::PointPillarsPost(int nclasses,
                        float nms_thrs,
                        float score_thrs,
                        int max_num,
                        bool use_gpu)
        : nclasses_(nclasses), nms_thrs_(nms_thrs), score_thrs_(score_thrs), max_num_(max_num), prepost_use_gpu_(use_gpu) {
}

PointPillarsPost::~PointPillarsPost() {
    std::cout << "[PointPillarsPost] Destructor called." << std::endl;
}

std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PointPillarsPost::forward(torch::Tensor&result) {
    if (prepost_use_gpu_ && result.is_cpu()) {
        result = result.cuda();
    } else if (!prepost_use_gpu_ && result.is_cuda()) {
        result = result.cpu();
    }
    torch::Tensor bbox_pred = result.index({torch::indexing::Slice(), torch::indexing::Slice({0, 7})});
    torch::Tensor cls_pred = result.index({torch::indexing::Slice(), torch::indexing::Slice({7, 10})});
    torch::Tensor dir_cls_pred = result.index({torch::indexing::Slice(), 10});
    auto nms_result = nms_filter(bbox_pred, cls_pred, dir_cls_pred);
    if (nms_result.has_value()) {
        return nms_result;
    } else {
        std::cout << "No results for nms_filter" << std::endl;
        return std::nullopt;
    }
}

torch::Tensor PointPillarsPost::nms(const torch::Tensor& boxes, const torch::Tensor& scores, float thresh,
                        c10::optional<int> pre_maxsize,
                        c10::optional<int> post_maxsize) {
    auto order_with_values = scores.sort(/*dim=*/0, /*descending=*/true);
    auto order = std::get<1>(order_with_values);

    if (pre_maxsize.has_value()) {
        order = order.index({torch::indexing::Slice(0, pre_maxsize.value())});
    }

    torch::Tensor boxes_ordered = boxes.index_select(0, order).contiguous();
    auto keep = torch::empty({boxes_ordered.size(0)}, torch::dtype(torch::kLong));
    
    int kept;
    if (prepost_use_gpu_) {
        std::cout << "[postprocess] : use cuda..." << std::endl;
        kept = nms_gpu(boxes_ordered, keep, thresh, boxes.device().index());
    } else {
        std::cout << "[postprocess] : use CPU..." << std::endl;
        kept = nms_cpu(boxes_ordered, keep, thresh, boxes.device().index());
    }
    keep = keep.narrow(0, 0, kept).to(order.device());
    torch::Tensor keep_original = order.index_select(0, keep).contiguous();

    if (post_maxsize.has_value()) {
        keep_original = keep_original.index({torch::indexing::Slice(0, (post_maxsize.value()))});
    }
    return keep_original;
}

std::optional<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PointPillarsPost::nms_filter(const torch::Tensor& bbox_pred, const torch::Tensor& bbox_cls_pred, const torch::Tensor& bbox_dir_cls_pred) {
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

        torch::Tensor keep_inds = nms(cur_bbox_pred2d, cur_bbox_cls_pred, nms_thrs_);

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

torch::Tensor PointPillarsPost::limit_period(const torch::Tensor& val, float offset, float period) {
    auto div = val / period + offset;
    auto floored = div.floor();
    return val - floored * period;
}
