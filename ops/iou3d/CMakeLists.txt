cmake_minimum_required(VERSION 3.10)
project(iou3d_nms)

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -std=c++14")

cuda_add_library(iou3d_lib SHARED
  iou3d.cpp
  iou3d_kernel.cu
)

target_link_libraries(iou3d_lib "${TORCH_LIBRARIES}" Python3::Python)
set_property(TARGET iou3d_lib PROPERTY CXX_STANDARD 14)
