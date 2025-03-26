add_subdirectory(device)

# Common function to set up target properties
function(setup_target target_name)
    target_link_libraries(${target_name} PUBLIC
        cuda_kernels
        mujoco
    )

    target_include_directories(${target_name} PUBLIC
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )

    set_target_properties(${target_name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )

    target_compile_features(${target_name} PUBLIC
        cxx_std_20
    )
endfunction()

# Test executable
add_executable(test
    test.cpp
    tests/test_base.h
    tests/smooth_test.h
)
setup_target(test)

# Speed test executable
add_executable(testspeed
    testspeed.cpp
)
setup_target(testspeed)