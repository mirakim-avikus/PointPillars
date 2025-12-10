// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <cstdint>
#include <vector>
#include "iou3d.h"

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

void boxesoverlapLauncher(const int num_a, const float *boxes_a,
                          const int num_b, const float *boxes_b,
                          float *ans_overlap);
void boxesioubevLauncher(const int num_a, const float *boxes_a, const int num_b,
                         const float *boxes_b, float *ans_iou);
void nmsLauncher(const float *boxes, unsigned long long *mask, int boxes_num,
                 float nms_overlap_thresh);
void nmsNormalLauncher(const float *boxes, unsigned long long *mask,
                       int boxes_num, float nms_overlap_thresh);

int boxes_overlap_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b,
                          at::Tensor ans_overlap) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)

  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_overlap);

  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  const float *boxes_a_data = boxes_a.data_ptr<float>();
  const float *boxes_b_data = boxes_b.data_ptr<float>();
  float *ans_overlap_data = ans_overlap.data_ptr<float>();

  boxesoverlapLauncher(num_a, boxes_a_data, num_b, boxes_b_data,
                       ans_overlap_data);

  return 1;
}

int boxes_iou_bev_gpu(at::Tensor boxes_a, at::Tensor boxes_b,
                      at::Tensor ans_iou) {
  // params boxes_a: (N, 5) [x1, y1, x2, y2, ry]
  // params boxes_b: (M, 5)
  // params ans_overlap: (N, M)

  CHECK_INPUT(boxes_a);
  CHECK_INPUT(boxes_b);
  CHECK_INPUT(ans_iou);

  int num_a = boxes_a.size(0);
  int num_b = boxes_b.size(0);

  const float *boxes_a_data = boxes_a.data_ptr<float>();
  const float *boxes_b_data = boxes_b.data_ptr<float>();
  float *ans_iou_data = ans_iou.data_ptr<float>();

  boxesioubevLauncher(num_a, boxes_a_data, num_b, boxes_b_data, ans_iou_data);

  return 1;
}

int nms_gpu(at::Tensor boxes, at::Tensor keep,
	    float nms_overlap_thresh, int device_id) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)

  CHECK_INPUT(boxes);
  CHECK_CONTIGUOUS(keep);
  cudaSetDevice(device_id);

  int boxes_num = boxes.size(0);
  const float *boxes_data = boxes.data_ptr<float>();
  int64_t *keep_data = keep.data_ptr<int64_t>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  // unsigned long long mask_cpu[boxes_num * col_blocks];
  // unsigned long long *mask_cpu = new unsigned long long [boxes_num *
  // col_blocks];
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

  //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

  cudaFree(mask_data);

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return num_to_keep;
}

int nms_cpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh, int device_id=0) {
  // boxes : (N, 5) [x1, y1, x2, y2, ry]
  // keep : (N)
  CHECK_CONTIGUOUS(keep);

  const int boxes_num = boxes.size(0);
  auto boxes_acc = boxes.accessor<float, 2>(); //(N, 5)
  int64_t* keep_data = keep.data_ptr<int64_t>();

  const int col_blocks = (boxes_num + THREADS_PER_BLOCK_NMS - 1) / THREADS_PER_BLOCK_NMS;
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks, 0);

  // generate mask : nms_gpu(): mask[cur_box_idx * col_blocks + col_start] = t
  for (int i = 0; i < boxes_num; i++) {
    float x1_i = boxes_acc[i][0];
    float y1_i = boxes_acc[i][1];
    float x2_i = boxes_acc[i][2];
    float y2_i = boxes_acc[i][3];
    float area_i = (x2_i - x1_i + 1) * (y2_i - y1_i + 1);

    for (int j = i + 1; j < boxes_num; j++) {
      float x1_j = boxes_acc[j][0];
      float y1_j = boxes_acc[j][1];
      float x2_j = boxes_acc[j][2];
      float y2_j = boxes_acc[j][3];
      float area_j = (x2_j - x1_j + 1) * (y2_j - y1_j + 1);

      float xx1 = std::max(x1_i, x1_j);
      float yy1 = std::max(y1_i, y1_j);
      float xx2 = std::max(x2_i, x2_j);
      float yy2 = std::max(y2_i, y2_j);
      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      float iou = inter / (area_i + area_j - inter);

      if (iou > nms_overlap_thresh) {
        int block_j = j / THREADS_PER_BLOCK_NMS;
        int offset_j = j % THREADS_PER_BLOCK_NMS;
        mask_cpu[i * col_blocks + block_j] |= 1ULL << offset_j;
      }
    }
  }

  std::vector<unsigned long long> remv(col_blocks, 0);
  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int block_i = i / THREADS_PER_BLOCK_NMS;
    int offset_i = i % THREADS_PER_BLOCK_NMS;

    if (!(remv[block_i] & (1ULL << offset_i))) {
      keep_data[num_to_keep++] = i;
      for (int j = block_i; j < col_blocks; j++) {
        remv[j] |= mask_cpu[i * col_blocks + j];
      }
    }
  }
  return num_to_keep;
}

int nms_normal_gpu(at::Tensor boxes, at::Tensor keep,
                   float nms_overlap_thresh, int device_id) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)

  CHECK_INPUT(boxes);
  CHECK_CONTIGUOUS(keep);
  cudaSetDevice(device_id);

  int boxes_num = boxes.size(0);
  const float *boxes_data = boxes.data_ptr<float>();
  int64_t *keep_data = keep.data_ptr<int64_t>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_data = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  nmsNormalLauncher(boxes_data, mask_data, boxes_num, nms_overlap_thresh);

  // unsigned long long mask_cpu[boxes_num * col_blocks];
  // unsigned long long *mask_cpu = new unsigned long long [boxes_num *
  // col_blocks];
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);

  //    printf("boxes_num=%d, col_blocks=%d\n", boxes_num, col_blocks);
  CHECK_ERROR(cudaMemcpy(&mask_cpu[0], mask_data,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));

  cudaFree(mask_data);

  unsigned long long *remv_cpu = new unsigned long long[col_blocks]();

  int num_to_keep = 0;

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  delete[] remv_cpu;
  if (cudaSuccess != cudaGetLastError()) printf("Error!\n");

  return num_to_keep;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("boxes_overlap_bev_gpu", &boxes_overlap_bev_gpu,
        "oriented boxes overlap");
  m.def("boxes_iou_bev_gpu", &boxes_iou_bev_gpu, "oriented boxes iou");
  m.def("nms_gpu", &nms_gpu, "oriented nms gpu");
  m.def("nms_normal_gpu", &nms_normal_gpu, "nms gpu");
}
