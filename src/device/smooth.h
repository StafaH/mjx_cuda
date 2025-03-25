#pragma once

#include <cuda.h>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <mujoco/mujoco.h>

#include "math.h"
#include "util.h"
#include "io.h"
#include "types.h"

void LaunchNoiseKernel(
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd);

void LaunchKinematicsKernel(
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd);

__global__ void RootKernel(
    unsigned int n,
    float* xpos,
    float* xquat,
    float* xipos,
    float* xmat,
    float* ximat);

__global__ void LevelKernel(
    unsigned int n,
    unsigned int leveladr,
    unsigned int nq,
    unsigned int njnt,
    unsigned int nbody,
    const float* qpos0,
    const int* body_jntadr,
    const int* body_jntnum,
    const int* body_parentid,
    const int* body_mocapid,
    const float* body_pos,
    const float* body_quat,
    const float* body_ipos,
    const float* body_iquat,
    const int* jnt_type,
    const int* jnt_qposadr,
    const float* jnt_axis,
    const float* jnt_pos,
    const int* body_tree,
    float* qpos,
    float* mocap_pos,
    float* mocap_quat,
    float* xanchor,
    float* xaxis,
    float* xmat,
    float* xpos,
    float* xquat,
    float* xipos,
    float* ximat);

__global__ void GeomLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int ngeom,
    const int* geom_bodyid,
    const float* geom_pos,
    const float* geom_quat,
    const float* xpos,
    const float* xquat,
    float* geom_xpos,
    float* geom_xmat);

__global__ void SiteLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int nsite,
    const int* site_bodyid,
    const float* site_pos,
    const float* site_quat,
    const float* xpos,
    const float* xquat,
    float* site_xpos,
    float* site_xmat);

__global__ void NoiseInjectionKernel(
    unsigned int n,
    unsigned int nq,
    float* qpos,
    float noise_scale,
    unsigned int seed);