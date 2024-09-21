cmake_minimum_required(VERSION 3.13)

project(dfccl_extension LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# 寻找 Python 和 PyBind11
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

# 寻找 CUDA 工具包
find_package(CUDAToolkit REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)
include_directories(${pybind11_INCLUDE_DIRS})

# 设置 DFCCL_LIB_PATH 和 DFCCL_INCLUDE_DIR，可通过命令行指定
set(DFCCL_LIB_PATH "/workspace/dfccl/build/lib" CACHE PATH "Path to libdfccl.so")
set(DFCCL_INCLUDE_DIR "/workspace/dfccl/build/include" CACHE PATH "Path to dfccl.h and other headers")

# 添加共享库
add_library(dfccl_extension MODULE
    src/dfccl_extension.cu
    src/wrapper.cpp
)

# 链接库和依赖
target_include_directories(dfccl_extension PRIVATE
    include
    ${DFCCL_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}  # 添加 CUDA 的包含目录
)

target_link_libraries(dfccl_extension
    ${PYTHON_LIBRARIES}
    "${DFCCL_LIB_PATH}/libdfccl.so"
    CUDA::cudart                         # 链接 CUDA Runtime 库
    CUDA::cuda_driver                    # 链接 CUDA Driver 库
)

# 设置共享库属性
set_target_properties(dfccl_extension PROPERTIES
    PREFIX ""  # 去掉前缀
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# 添加 RPATH，以便在运行时找到 libdfccl.so
set_target_properties(dfccl_extension PROPERTIES
    BUILD_RPATH ${DFCCL_LIB_PATH}
    INSTALL_RPATH ${DFCCL_LIB_PATH}
)

# 添加 CUDA 的编译选项（可根据需要调整）
target_compile_options(dfccl_extension PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-extended-lambda
>)