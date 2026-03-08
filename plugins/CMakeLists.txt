cmake_minimum_required(VERSION 3.16)
project(univ2x_plugins LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "-fPIC -O2")
set(CMAKE_CXX_FLAGS "-fPIC -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14 --expt-relaxed-constexpr --compiler-options -fPIC -O2")

# TensorRT root — override with: cmake -DTRT_ROOT=/path/to/tensorrt ..
set(TRT_ROOT "$ENV{TRT_ROOT}" CACHE STRING "Path to TensorRT installation")
if(NOT TRT_ROOT)
    set(TRT_ROOT "/usr/local/tensorrt")
endif()

find_package(CUDA REQUIRED)

# RTX 4090 = SM 89
set(ARCH 89)

set(TENSORRT_INCLUDE_DIRS ${TRT_ROOT}/include)
set(TENSORRT_LIBRARY_DIRS ${TRT_ROOT}/lib)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/multi_scale_deform_attn
    ${CMAKE_SOURCE_DIR}/rotate
    ${CMAKE_SOURCE_DIR}/inverse)

link_directories(${TENSORRT_LIBRARY_DIRS})

set(plugins_cu_srcs
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn_kernel.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotateKernel.cu)

cuda_add_library(plugins_cu STATIC ${plugins_cu_srcs} OPTIONS -arch=sm_${ARCH})
set_target_properties(plugins_cu PROPERTIES CUDA_ARCHITECTURES ${ARCH})
set_target_properties(plugins_cu PROPERTIES CUDA_STANDARD 14)
set_target_properties(plugins_cu PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(univ2x_plugins SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_scale_deform_attn/ms_deform_attn.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/rotate/rotate_plugin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/inverse/inversePlugin.cpp)

target_link_libraries(univ2x_plugins plugins_cu nvinfer ${CUDA_LIBRARIES} cublas)
