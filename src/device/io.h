#pragma once

#include <mujoco/mujoco.h>
#include <cuda_runtime.h>

#include "types.h"

// Function declarations
CudaModel* put_model(const mjModel* m);
CudaData* put_data(const mjModel* m, const mjData* d, int nworlds);
void free_cuda_model(CudaModel* cm);
void free_cuda_data(CudaData* cd); 
