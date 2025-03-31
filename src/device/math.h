#pragma once

#include <cuda.h>
#include "types.h"

// // multiply quaternions
// inline __device__ void MulQuat(float res[4], const float qa[4],
//                                const float qb[4]) {
//   res[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3];
//   res[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2];
//   res[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1];
//   res[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0];
// }

// // rotate vector by quaternion
// inline __device__ void RotVecQuat(float res[3], const float vec[3],
//                                   const float quat[4]) {
//   res[0] = (quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]) * vec[0] + 
//            2.0f * ((quat[1] * vec[0] + quat[2] * vec[1] + quat[3] * vec[2]) * quat[1] + 
//                    quat[0] * (quat[2] * vec[2] - quat[3] * vec[1]));
//   res[1] = (quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]) * vec[1] + 
//            2.0f * ((quat[1] * vec[0] + quat[2] * vec[1] + quat[3] * vec[2]) * quat[2] + 
//                    quat[0] * (quat[3] * vec[0] - quat[1] * vec[2]));
//   res[2] = (quat[0] * quat[0] - quat[1] * quat[1] - quat[2] * quat[2] - quat[3] * quat[3]) * vec[2] + 
//            2.0f * ((quat[1] * vec[0] + quat[2] * vec[1] + quat[3] * vec[2]) * quat[3] + 
//                    quat[0] * (quat[1] * vec[1] - quat[2] * vec[0]));
// }

// // convert axisAngle to quaternion
// inline __device__ void AxisAngle2Quat(float res[4], const float axis[3],
//                                     float angle) {
//   const float half_angle = 0.5f * angle;
  
//   #ifdef __CUDA_ARCH__
//   float sin_value, cos_value;
//   __sincosf(half_angle, &sin_value, &cos_value);
//   #else
//   const float sin_value = sinf(half_angle);
//   const float cos_value = cosf(half_angle);
//   #endif
  
//   res[0] = cos_value;
//   res[1] = axis[0] * sin_value;
//   res[2] = axis[1] * sin_value;
//   res[3] = axis[2] * sin_value;
// }

// // convert quaternion to 3D rotation matrix
// inline __device__ void Quat2Mat(float res[9], const float quat[4]) {
//   res[0] = quat[0]*quat[0] + quat[1]*quat[1] - quat[2]*quat[2] - quat[3]*quat[3];
//   res[4] = quat[0]*quat[0] - quat[1]*quat[1] + quat[2]*quat[2] - quat[3]*quat[3];
//   res[8] = quat[0]*quat[0] - quat[1]*quat[1] - quat[2]*quat[2] + quat[3]*quat[3];

//   res[1] = 2.0f * (quat[1]*quat[2] - quat[0]*quat[3]);
//   res[2] = 2.0f * (quat[1]*quat[3] + quat[0]*quat[2]);
//   res[3] = 2.0f * (quat[1]*quat[2] + quat[0]*quat[3]);
//   res[5] = 2.0f * (quat[2]*quat[3] - quat[0]*quat[1]);
//   res[6] = 2.0f * (quat[1]*quat[3] - quat[0]*quat[2]);
//   res[7] = 2.0f * (quat[2]*quat[3] + quat[0]*quat[1]);
// }

static __inline__ __host__ __device__ vec3p make_vec3p(float x, float y, float z)
{
  vec3p v; v.x = x; v.y = y; v.z = z; return v;
}

inline __host__ __device__ vec3p operator+(vec3p a, vec3p b)
{
    return make_vec3p(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ void operator+=(vec3p &a, vec3p b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
inline __host__ __device__ vec3p operator+(vec3p a, float b)
{
    return make_vec3p(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ vec3p operator+(float b, vec3p a)
{
    return make_vec3p(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(vec3p &a, float b)
{
    a.x += b;
    a.y += b;
    a.z += b;
}

inline __host__ __device__ vec3p operator-(vec3p a, vec3p b)
{
    return make_vec3p(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ void operator-=(vec3p &a, vec3p b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
inline __host__ __device__ vec3p operator-(vec3p a, float b)
{
    return make_vec3p(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(vec3p &a, float b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
}

inline __host__ __device__ vec3p operator*(vec3p a, vec3p b)
{
    return make_vec3p(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ void operator*=(vec3p &a, vec3p b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}
inline __host__ __device__ vec3p operator*(vec3p a, float b)
{
    return make_vec3p(a.x * b, a.y * b, a.z * b);
}
inline __host__ __device__ vec3p operator*(float b, vec3p a)
{
    return make_vec3p(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ void operator*=(vec3p &a, float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

inline __host__ __device__ vec3p operator/(vec3p a, vec3p b)
{
    return make_vec3p(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ void operator/=(vec3p &a, vec3p b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}
inline __host__ __device__ vec3p operator/(vec3p a, float b)
{
    return make_vec3p(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ void operator/=(vec3p &a, float b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
}
inline __host__ __device__ vec3p operator/(float b, vec3p a)
{
    return make_vec3p(b / a.x, b / a.y, b / a.z);
}

// multiply quaternions with quat type
inline __device__ void MulQuat(quat& res, const quat& qa, const quat& qb) {
  res.w = qa.w*qb.w - qa.x*qb.x - qa.y*qb.y - qa.z*qb.z;
  res.x = qa.w*qb.x + qa.x*qb.w + qa.y*qb.z - qa.z*qb.y;
  res.y = qa.w*qb.y - qa.x*qb.z + qa.y*qb.w + qa.z*qb.x;
  res.z = qa.w*qb.z + qa.x*qb.y - qa.y*qb.x + qa.z*qb.w;
}

// rotate vector by quaternion with vec3p and quat types
inline __device__ void RotVecQuat(vec3p& res, const vec3p& vec, const quat& q) {
  res.x = (q.w * q.w - q.x * q.x - q.y * q.y - q.z * q.z) * vec.x + 
          2.0f * ((q.x * vec.x + q.y * vec.y + q.z * vec.z) * q.x + 
                  q.w * (q.y * vec.z - q.z * vec.y));
  res.y = (q.w * q.w - q.x * q.x - q.y * q.y - q.z * q.z) * vec.y + 
          2.0f * ((q.x * vec.x + q.y * vec.y + q.z * vec.z) * q.y + 
                  q.w * (q.z * vec.x - q.x * vec.z));
  res.z = (q.w * q.w - q.x * q.x - q.y * q.y - q.z * q.z) * vec.z + 
          2.0f * ((q.x * vec.x + q.y * vec.y + q.z * vec.z) * q.z + 
                  q.w * (q.x * vec.y - q.y * vec.x));
}

// convert quaternion to 3D rotation matrix with quat type
inline __device__ void Quat2Mat(float* res, const quat& q) {
  res[0] = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z;
  res[4] = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z;
  res[8] = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z;

  res[1] = 2.0f * (q.x*q.y - q.w*q.z);
  res[2] = 2.0f * (q.x*q.z + q.w*q.y);
  res[3] = 2.0f * (q.x*q.y + q.w*q.z);
  res[5] = 2.0f * (q.y*q.z - q.w*q.x);
  res[6] = 2.0f * (q.x*q.z - q.w*q.y);
  res[7] = 2.0f * (q.y*q.z + q.w*q.x);
}

// multiply matrix and vector with vec3p type
inline __device__ void MulMatVec3(vec3p& res, const float* mat, const vec3p& vec) {
  res.x = mat[0]*vec.x + mat[1]*vec.y + mat[2]*vec.z;
  res.y = mat[3]*vec.x + mat[4]*vec.y + mat[5]*vec.z;
  res.z = mat[6]*vec.x + mat[7]*vec.y + mat[8]*vec.z;
}

// normalize quaternion
inline __device__ void NormalizeQuat(quat& q) {
  float norm = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
  float invNorm = 1.0f / norm;
  q.w *= invNorm;
  q.x *= invNorm;
  q.y *= invNorm;
  q.z *= invNorm;
}

// convert axis angle to quaternion with vec3p
inline __device__ void AxisAngle2Quat(quat& res, const vec3p& axis, float angle) {
  const float half_angle = 0.5f * angle;
  
  #ifdef __CUDA_ARCH__
  float sin_value, cos_value;
  __sincosf(half_angle, &sin_value, &cos_value);
  #else
  const float sin_value = sinf(half_angle);
  const float cos_value = cosf(half_angle);
  #endif
  
  res.w = cos_value;
  res.x = axis.x * sin_value;
  res.y = axis.y * sin_value;
  res.z = axis.z * sin_value;
}
