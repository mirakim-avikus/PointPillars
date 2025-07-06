#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <cstdint>
#include <vector>

int nms_gpu(at::Tensor boxes, at::Tensor keep,
	    float nms_overlap_thresh, int device_id);
int nms_normal_gpu(at::Tensor boxes, at::Tensor keep,
                   float nms_overlap_thresh, int device_id);
int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b,
                      at::Tensor ans_iou);
int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b,
                          at::Tensor ans_overlap);