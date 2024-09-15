cmake_minimum_required(VERSION 3.10)
project(VectorAdd CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)

# Add CUDA compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3)

# Add the CUDA source file
cuda_add_library(vector_add SHARED
    vector_add.cu
    vector_add_wrapper.cpp
)

# Set the output name of the library
set_target_properties(vector_add PROPERTIES OUTPUT_NAME "vector_add")

# Link against CUDA libraries
target_link_libraries(vector_add ${CUDA_LIBRARIES})