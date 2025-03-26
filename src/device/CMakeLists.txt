add_library(cuda_kernels STATIC
    types.h
    math.h
    util.h
    smooth.h
    smooth.cu
    io.cpp
)

target_link_libraries(cuda_kernels PRIVATE
    mujoco::mujoco
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

target_compile_features(cuda_kernels PUBLIC
    cxx_std_20
)

target_compile_options(cuda_kernels PRIVATE
    "-fPIC"
)

target_include_directories(cuda_kernels PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)