#pragma once

#include <cuda.h>

// multiply quaternions
inline __device__ void MulQuat(float res[4], const float qa[4],
                               const float qb[4]) {
  res[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3];
  res[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2];
  res[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1];
  res[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0];
}

// rotate vector by quaternion
inline __device__ void RotVecQuat(float res[3], const float vec[3],
                                  const float quat[4]) {
  res[0] = (quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]) * vec[0] + 
           2.0f * ((quat[1] * vec[0] + quat[2] * vec[1] + quat[3] * vec[2]) * quat[1] + 
                   quat[0] * (quat[2] * vec[2] - quat[3] * vec[1]));
  res[1] = (quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]) * vec[1] + 
           2.0f * ((quat[1] * vec[0] + quat[2] * vec[1] + quat[3] * vec[2]) * quat[2] + 
                   quat[0] * (quat[3] * vec[0] - quat[1] * vec[2]));
  res[2] = (quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]) * vec[2] + 
           2.0f * ((quat[1] * vec[0] + quat[2] * vec[1] + quat[3] * vec[2]) * quat[3] + 
                   quat[0] * (quat[1] * vec[1] - quat[2] * vec[0]));
}

// convert axisAngle to quaternion
inline __device__ void AxisAngle2Quat(float res[4], const float axis[3],
                                    float angle) {
  const float half_angle = 0.5f * angle;
  
  #ifdef __CUDA_ARCH__
  float sin_value, cos_value;
  __sincosf(half_angle, &sin_value, &cos_value);
  #else
  const float sin_value = sinf(half_angle);
  const float cos_value = cosf(half_angle);
  #endif
  
  res[0] = cos_value;
  res[1] = axis[0] * sin_value;
  res[2] = axis[1] * sin_value;
  res[3] = axis[2] * sin_value;
}

// convert quaternion to 3D rotation matrix
inline __device__ void Quat2Mat(float res[9], const float quat[4]) {
  res[0] = quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2] - quat[3]*quat[3];
  res[4] = quat[0]*quat[0] - quat[1]*quat[1] + quat[2]*quat[2] - quat[3]*quat[3];
  res[8] = quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2] + quat[3]*quat[3];

  res[1] = 2.0f * (quat[1]*quat[2] - quat[0]*quat[3]);
  res[2] = 2.0f * (quat[1]*quat[3] + quat[0]*quat[2]);
  res[3] = 2.0f * (quat[1]*quat[2] + quat[0]*quat[3]);
  res[5] = 2.0f * (quat[2]*quat[3] - quat[0]*quat[1]);
  res[6] = 2.0f * (quat[1]*quat[3] - quat[0]*quat[2]);
  res[7] = 2.0f * (quat[2]*quat[3] + quat[0]*quat[1]);
}
