add_subdirectory(device)

add_executable(mjx_cuda
    main.cpp
)

target_link_libraries(mjx_cuda PUBLIC
    cuda_kernels
    mujoco
)

target_include_directories(mjx_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}

)

set_target_properties(mjx_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_compile_features(mjx_cuda PUBLIC
    cxx_std_20
)