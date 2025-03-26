#pragma once

#include <cuda.h>

// multiply quaternions
inline __device__ void MulQuat(float res[4], const float qa[4],
                               const float qb[4]) {
  const float qa0 = qa[0], qa1 = qa[1], qa2 = qa[2], qa3 = qa[3];
  const float qb0 = qb[0], qb1 = qb[1], qb2 = qb[2], qb3 = qb[3];
  
  res[0] = qa0*qb0 - qa1*qb1 - qa2*qb2 - qa3*qb3;
  res[1] = qa0*qb1 + qa1*qb0 + qa2*qb3 - qa3*qb2;
  res[2] = qa0*qb2 - qa1*qb3 + qa2*qb0 + qa3*qb1;
  res[3] = qa0*qb3 + qa1*qb2 - qa2*qb1 + qa3*qb0;
}

// rotate vector by quaternion
inline __device__ void RotVecQuat(float res[3], const float vec[3],
                                  const float quat[4]) {
  const float s = quat[0];
  const float qx = quat[1];
  const float qy = quat[2];
  const float qz = quat[3];
  
  const float s2 = s * s;
  const float qx2 = qx * qx;
  const float qy2 = qy * qy;
  const float qz2 = qz * qz;
  
  const float qv_dot_vec = qx * vec[0] + qy * vec[1] + qz * vec[2];
  const float factor = s2 - (qx2 + qy2 + qz2);
  
  res[0] = factor * vec[0] + 2.0f * (qv_dot_vec * qx + s * (qy * vec[2] - qz * vec[1]));
  res[1] = factor * vec[1] + 2.0f * (qv_dot_vec * qy + s * (qz * vec[0] - qx * vec[2]));
  res[2] = factor * vec[2] + 2.0f * (qv_dot_vec * qz + s * (qx * vec[1] - qy * vec[0]));
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
  const float w = quat[0];
  const float x = quat[1];
  const float y = quat[2];
  const float z = quat[3];
  
  const float ww = w * w;
  const float xx = x * x;
  const float yy = y * y;
  const float zz = z * z;
  
  const float wx2 = 2.0f * w * x;
  const float wy2 = 2.0f * w * y;
  const float wz2 = 2.0f * w * z;
  const float xy2 = 2.0f * x * y;
  const float xz2 = 2.0f * x * z;
  const float yz2 = 2.0f * y * z;
  
  res[0] = ww + xx - yy - zz;
  res[4] = ww - xx + yy - zz;
  res[8] = ww - xx - yy + zz;
  
  res[1] = xy2 - wz2;
  res[2] = xz2 + wy2;
  res[3] = xy2 + wz2;
  res[5] = yz2 - wx2;
  res[6] = xz2 - wy2;
  res[7] = yz2 + wx2;
}
