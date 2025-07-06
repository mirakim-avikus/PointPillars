cmake_minimum_required(VERSION 3.10)
project(onnx_infer LANGUAGES C CXX CUDA)

# -------------------- [Global Settings] --------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Architecture: Ampere (e.g. RTX 30xx series)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(TORCH_CUDA_ARCH_LIST "8.6")

# CUDA flags (for debugging)
unset(CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_86,code=sm_86 -g -G")

# -------------------- [Path Setup] --------------------
set(CMAKE_PREFIX_PATH 
        /workspace/libtorch 
        ${CMAKE_PREFIX_PATH}
        /usr/local/cuda
)

set(ONNX_ROOT /workspace/onnxruntime-linux-x64-gpu-1.12.1)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -------------------- [Subdirectories] --------------------
add_subdirectory(ops/iou3d)

# -------------------- [Include Directories] --------------------
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ops/voxelization
    ${PROJECT_SOURCE_DIR}/ops/iou3d
    ${ONNX_ROOT}/include
)

# -------------------- [PCL Setup] --------------------
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# -------------------- [Torch Setup] --------------------
set(Torch_DIR "/workspace/libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# -------------------- [ONNX Runtime] --------------------
link_directories(${ONNX_ROOT}/lib)

# -------------------- [Other Dependencies] --------------------
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

# -------------------- [Main Executable] --------------------
add_executable(preprocess 
                    src/main.cpp 
                    src/pillar_layer.cpp 
                    src/voxelization.cpp 
                    ops/voxelization/voxelization_cpu.cpp 
                    ops/voxelization/voxelization_cuda.cu
)

target_link_libraries(preprocess PRIVATE
                                ${TORCH_LIBRARIES}
                                ${PCL_LIBRARIES} 
                                ${CMAKE_THREAD_LIBS_INIT} 
                                Python3::Python 
                                pybind11::module 
                                CUDA::cudart
                                iou3d_lib
                                "${ONNX_ROOT}/lib/libonnxruntime.so"
)

target_compile_definitions(preprocess PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=1")

# Optional: AVX / CUDA arch tuning 
set_target_properties(preprocess PROPERTIES CUDA_ARCHITECTURES 86)

# Optional: RPATH for ONNX shared lib
set_target_properties(preprocess PROPERTIES
    BUILD_RPATH ${ONNX_ROOT}/lib
    INSTALL_RPATH ${ONNX_ROOT}/lib
)

# -------------------- [Python Bindings: voxelization module] --------------------
pybind11_add_module(voxelization ops/voxelization/voxelization.cpp)
target_link_libraries(voxelization PRIVATE "${TORCH_LIBRARIES}")
