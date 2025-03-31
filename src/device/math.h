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

static __inline__ __host__ __device__ quat make_quat(float w, float x, float y, float z)
{
  quat q; q.w = w; q.x = x; q.y = y; q.z = z; return q;
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

inline __device__ quat MulQuat(const quat& qa, const quat& qb) {
    return {
        qa.w*qb.w - qa.x*qb.x - qa.y*qb.y - qa.z*qb.z,
        qa.w*qb.x + qa.x*qb.w + qa.y*qb.z - qa.z*qb.y,
        qa.w*qb.y - qa.x*qb.z + qa.y*qb.w + qa.z*qb.x,
        qa.w*qb.z + qa.x*qb.y - qa.y*qb.x + qa.z*qb.w
    };
}   

// rotate vector by quaternion with vec3p and quat types
inline __device__ vec3p RotVecQuat(const vec3p& v, const quat& q) {
    float qxx = q.x * q.x, qyy = q.y * q.y, qzz = q.z * q.z;
    float qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;
    float qxy = q.x * q.y, qxz = q.x * q.z, qyz = q.y * q.z;

    return {
        (1 - 2 * (qyy + qzz)) * v.x + 2 * (qxy - qwz) * v.y + 2 * (qxz + qwy) * v.z,
        2 * (qxy + qwz) * v.x + (1 - 2 * (qxx + qzz)) * v.y + 2 * (qyz - qwx) * v.z,
        2 * (qxz - qwy) * v.x + 2 * (qyz + qwx) * v.y + (1 - 2 * (qxx + qyy)) * v.z,
        0.0f
    };
}

// convert quaternion to 3D rotation matrix with quat type
inline __device__ void Quat2Mat(mat3p& res, const quat& q) {
  res.m[0] = q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z;
  res.m[4] = q.w*q.w - q.x*q.x + q.y*q.y - q.z*q.z;
  res.m[8] = q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z;

  res.m[1] = 2.0f * (q.x*q.y - q.w*q.z);
  res.m[2] = 2.0f * (q.x*q.z + q.w*q.y);
  res.m[3] = 2.0f * (q.x*q.y + q.w*q.z);
  res.m[5] = 2.0f * (q.y*q.z - q.w*q.x);
  res.m[6] = 2.0f * (q.x*q.z - q.w*q.y);
  res.m[7] = 2.0f * (q.y*q.z + q.w*q.x);
}

// multiply matrix and vector with vec3p type
inline __device__ void MulMatVec3(vec3p& res, const mat3p& mat, const vec3p& vec) {
    res.x = mat.m[0]*vec.x + mat.m[1]*vec.y + mat.m[2]*vec.z;
    res.y = mat.m[3]*vec.x + mat.m[4]*vec.y + mat.m[5]*vec.z;
    res.z = mat.m[6]*vec.x + mat.m[7]*vec.y + mat.m[8]*vec.z;
}

// normalize quaternion
inline __device__ void NormalizeQuat(quat& q) {
    float invNorm = 1.0f / sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
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
