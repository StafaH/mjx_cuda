#pragma once

#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <mujoco/mujoco.h>

#include "math.h"
#include "util.h"
#include "io.h"
#include "types.h"

void LaunchKinematicsKernel(
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd);

__global__ void KinematicsKernel(
    unsigned int n,
    unsigned int nq,
    unsigned int njnt,
    unsigned int nbody,
    unsigned int ngeom,
    unsigned int nsite,
    unsigned int nmocap,
    const float* qpos0,
    const int* body_jntadr,
    const int* body_jntnum,
    const int* body_parentid,
    const int* body_mocapid,
    const float* body_pos,
    const float* body_quat,
    const int* jnt_type,
    const int* jnt_qposadr,
    const float* jnt_axis,
    const float* jnt_pos,
    const float* qpos,
    const float* mocap_pos,
    const float* mocap_quat,
    float* xanchor,
    float* xaxis,
    float* xmat,
    float* xpos,
    float* xquat,
    float* xipos,
    float* ximat);