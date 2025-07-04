cmake_minimum_required(VERSION 3.10)
project(onnx_infer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "75") # or 86 (Ampere용), depending on what's supported
set(CMAKE_PREFIX_PATH /workspace/libtorch "${CMAKE_PREFIX_PATH};/usr/local/cuda")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(ONNX_ROOT /workspace/onnxruntime-linux-x64-gpu-1.12.1)

unset(CMAKE_CUDA_FLAGS)
set(CMAKE_CUDA_FLAGS "-gencode arch=compute_75,code=sm_75")

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/ops/voxelization)
include_directories(${TORCH_INCLUDE_DIRS})

# 1. PCL 
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 2. LibTorch 
set(Torch_DIR "/workspace/libtorch")
find_package(Torch REQUIRED)

# 3. execution file 
add_executable(preprocess src/main.cpp 
                            src/pillar_layer.cpp 
                            src/voxelization.cpp 
                            ops/voxelization/voxelization_cpu.cpp 
                            ops/voxelization/voxelization_cuda.cu)

# 6. Onnx 
include_directories(${ONNX_ROOT}/include)
link_directories(${ONNX_ROOT}/lib)
message(STATUS "ONNX_ROOT : ${ONNX_ROOT}")

# 4. library link
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(preprocess 
            PRIVATE 
                "${TORCH_LIBRARIES}" 
                ${PCL_LIBRARIES} 
                ${CMAKE_THREAD_LIBS_INIT} 
                Python3::Python 
                pybind11::module 
                CUDA::cudart 
                "${ONNX_ROOT}/lib/libonnxruntime.so")

pybind11_add_module(voxelization ops/voxelization/voxelization.cpp)
target_link_libraries(voxelization PRIVATE "${TORCH_LIBRARIES}")


# 5. LibTorch definition 
target_compile_options(preprocess PRIVATE "-D_GLIBCXX_USE_CXX11_ABI=1")

# 7. Python
find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

# 8. Threads 
find_package(Threads REQUIRED)

# [Optional] Compiler AVX 
set_property(TARGET preprocess PROPERTY CXX_STANDARD 17)

set_target_properties(preprocess PROPERTIES
    CUDA_ARCHITECTURES 75
    BUILD_RPATH ${ONNX_ROOT}/lib
    INSTALL_RPATH ${ONNX_ROOT}/lib
)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
find_package(CUDA REQUIRED)

# [Optional] Compiler AVX 
set_property(TARGET preprocess PROPERTY CXX_STANDARD 17)
