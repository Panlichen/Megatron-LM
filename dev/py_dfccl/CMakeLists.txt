cmake_minimum_required(VERSION 3.13)

project(MyExtension LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

# 找到 Python 和 PyBind11
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)
include_directories(${pybind11_INCLUDE_DIRS})

add_library(my_extension SHARED
    src/wrapper.cpp
    src/dfccl_extension.cpp
)

target_link_libraries(my_extension
    ${PYTHON_LIBRARIES}
)

set_target_properties(my_extension PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""  # 去掉前缀
)

# 指定库的输出目录为 build 目录
set_target_properties(my_extension PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})