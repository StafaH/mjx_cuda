#pragma once

#include <mujoco/mujoco.h>
#include "device/types.h"

class TestBase {
public:
    virtual ~TestBase() = default;
    
    // Initialize test with model, data, and batch size
    virtual void init(mjModel* m, mjData* d, int batch_size) = 0;
    
    // Run the test and return true if passed
    virtual bool run_test() = 0;
    
    // Get the name of the test
    virtual const char* get_name() const = 0;

protected:
    mjModel* model = nullptr;
    mjData* data = nullptr;
    CudaModel* cuda_model = nullptr;
    CudaData* cuda_data = nullptr;
    int batch_size = 1;
}; 