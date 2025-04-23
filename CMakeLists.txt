cmake_minimum_required(VERSION 3.18)
project(ParallelPrefixScan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Set architecture — tune to your GPU or leave default
set(CMAKE_CUDA_ARCHITECTURES 60 70 75 86)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add source files
set(SOURCES
    src/main.cu
    src/scan_kernels.cu
    src/utils.cu
)

# Define executable
add_executable(parallel_scan ${SOURCES})

# Use separate compilation for CUDA
set_target_properties(parallel_scan PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Optional: faster build with RelWithDebInfo
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
