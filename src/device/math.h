#pragma once

#include <cuda.h>

// multiply quaternions
inline __device__ void MulQuat(float res[4], const float qa[4],
                               const float qb[4]) {
  const float tmp[4] = {
    qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3],
    qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2],
    qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1],
    qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
  };
  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
  res[3] = tmp[3];
}

// rotate vector by quaternion
inline __device__ void RotVecQuat(float res[3], const float vec[3],
                                  const float quat[4]) {
  // Extract scalar part and vector part from quaternion
  const float s = quat[0];
  const float u[3] = {quat[1], quat[2], quat[3]};
  
  // Calculate dot product of u and vec
  const float u_dot_vec = u[0] * vec[0] + u[1] * vec[1] + u[2] * vec[2];
  
  // Calculate dot product of u with itself
  const float u_dot_u = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  
  // First term: 2.0 * (wp.dot(u, vec) * u)
  const float term1[3] = {
    2.0f * u_dot_vec * u[0],
    2.0f * u_dot_vec * u[1],
    2.0f * u_dot_vec * u[2]
  };
  
  // Second term: (s * s - wp.dot(u, u)) * vec
  const float factor = s * s - u_dot_u;
  const float term2[3] = {
    factor * vec[0],
    factor * vec[1],
    factor * vec[2]
  };
  
  // Third term: 2.0 * s * wp.cross(u, vec)
  const float cross[3] = {
    u[1] * vec[2] - u[2] * vec[1],
    u[2] * vec[0] - u[0] * vec[2],
    u[0] * vec[1] - u[1] * vec[0]
  };
  
  const float term3[3] = {
    2.0f * s * cross[0],
    2.0f * s * cross[1],
    2.0f * s * cross[2]
  };
  
  // Combine all terms: term1 + term2 + term3
  res[0] = term1[0] + term2[0] + term3[0];
  res[1] = term1[1] + term2[1] + term3[1];
  res[2] = term1[2] + term2[2] + term3[2];
}

// convert axisAngle to quaternion
inline __device__ void AxisAngle2Quat(float res[4], const float axis[3],
                                      float angle) {
  const float s = sin(angle * 0.5);
  res[0] = cos(angle * 0.5);
  res[1] = axis[0] * s;
  res[2] = axis[1] * s;
  res[3] = axis[2] * s;
}

// convert quaternion to 3D rotation matrix
inline __device__ void Quat2Mat(float res[9], const float quat[4]) {
  const float q00 = quat[0] * quat[0];
  const float q01 = quat[0] * quat[1];
  const float q02 = quat[0] * quat[2];
  const float q03 = quat[0] * quat[3];
  const float q11 = quat[1] * quat[1];
  const float q12 = quat[1] * quat[2];
  const float q13 = quat[1] * quat[3];
  const float q22 = quat[2] * quat[2];
  const float q23 = quat[2] * quat[3];
  const float q33 = quat[3] * quat[3];

  res[0] = q00 + q11 - q22 - q33;
  res[4] = q00 - q11 + q22 - q33;
  res[8] = q00 - q11 - q22 + q33;

  res[1] = 2 * (q12 - q03);
  res[2] = 2 * (q13 + q02);
  res[3] = 2 * (q12 + q03);
  res[5] = 2 * (q23 - q01);
  res[6] = 2 * (q13 - q02);
  res[7] = 2 * (q23 + q01);
}
