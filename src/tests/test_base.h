#pragma once

#include <mujoco/mujoco.h>
#include "device/types.h"

class TestBase {
public:
    virtual ~TestBase() = default;
    
    virtual void init(mjModel* m, mjData* d, int batch_size) = 0;
    
    virtual bool run_test() = 0;
    
    virtual const char* get_name() const = 0;

protected:
    mjModel* model = nullptr;
    mjData* data = nullptr;
    CudaModel* cuda_model = nullptr;
    CudaData* cuda_data = nullptr;
    int batch_size = 1;
}; 