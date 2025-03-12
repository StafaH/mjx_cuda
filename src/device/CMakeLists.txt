add_library(cuda_kernels STATIC
    smooth.h
    smooth.cu
)

set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
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