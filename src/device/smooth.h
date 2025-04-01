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
    cudaStream_t stream,
    unsigned int batch_size,
    CudaModel* cm,
    CudaData* cd);

__global__ void RootKernel(
    unsigned int n,
    vec3p* xpos,
    quat* xquat,
    vec3p* xipos,
    mat3p* xmat,
    mat3p* ximat);

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
    const vec3p* body_pos,
    const quat* body_quat,
    const vec3p* body_ipos,
    const quat* body_iquat,
    const int* jnt_type,
    const int* jnt_qposadr,
    const vec3p* jnt_axis,
    const vec3p* jnt_pos,
    const int* body_tree,
    float* qpos,
    vec3p* xanchor,
    vec3p* xaxis,
    mat3p* xmat,
    vec3p* xpos,
    quat* xquat,
    vec3p* xipos,
    mat3p* ximat);

__global__ void GeomLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int ngeom,
    const int* geom_bodyid,
    const vec3p* geom_pos,
    const quat* geom_quat,
    const vec3p* xpos,
    const quat* xquat,
    vec3p* geom_xpos,
    mat3p* geom_xmat);

__global__ void SiteLocalToGlobalKernel(
    unsigned int n,
    unsigned int nbody,
    unsigned int nsite,
    const int* site_bodyid,
    const vec3p* site_pos,
    const quat* site_quat,
    const vec3p* xpos,
    const quat* xquat,
    vec3p* site_xpos,
    mat3p* site_xmat);

__global__ void NoiseInjectionKernel(
    unsigned int n,
    unsigned int nq,
    float* qpos,
    float noise_scale,
    unsigned int seed);