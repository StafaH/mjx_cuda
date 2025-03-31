#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) getchar();
   }
}

inline __device__ void Normalize(float* x, int n) {
  float norm = 0.0f;
  for (int i = 0; i < n; ++i) {
    norm += x[i] * x[i];
  }
  norm = sqrt(norm);
  for (int i = 0; i < n; ++i) {
    x[i] /= norm;
  }
}

// multiply 3-by-3 matrix by vector
inline __device__ void MulMatVec3(float res[3], const float mat[9],
                                  const float vec[3]) {
  res[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  res[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  res[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline __device__ void AddTo(float* res, const float* vec, int n) {
  for (int i = 0; i < n; ++i) {
    res[i] += vec[i];
  }
}

inline __device__ void AddToScl(float* res, const float* vec, float scl,
                                int n) {
  for (int i = 0; i < n; ++i) {
    res[i] += vec[i] * scl;
  }
}

// res = vec1 - vec2
inline __device__ void Sub(float* res, const float* vec1, const float* vec2,
                           int n) {
  for (int i = 0; i < n; ++i) {
    res[i] = vec1[i] - vec2[i];
  }
}
