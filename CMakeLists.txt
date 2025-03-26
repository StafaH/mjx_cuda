cmake_minimum_required(VERSION 3.18)

project(mjx_cuda LANGUAGES CXX)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb

include(FetchContent)

FetchContent_Declare(
    mujoco
    GIT_REPOSITORY https://github.com/deepmind/mujoco.git
    GIT_TAG 3.2.7
)

set(MUJOCO_BUILD_EXAMPLES OFF)
set(MUJOCO_BUILD_TESTS OFF)
set(MUJOCO_TEST_PYTHON_UTIL OFF)

FetchContent_MakeAvailable(mujoco)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
