#pragma once

#include <cuda.h>
#include "types.h"


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
inline __device__ mat3p Quat2Mat(const quat& q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float ww = q.w * q.w;

    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    return {
        ww + xx - yy - zz,
        2.0f * (xy - wz),
        2.0f * (xz + wy),

        2.0f * (xy + wz),
        ww - xx + yy - zz,
        2.0f * (yz - wx),

        2.0f * (xz - wy),
        2.0f * (yz + wx),
        ww - xx - yy + zz
    };
}

// multiply matrix and vector with vec3p type
inline __device__ vec3p MulMatVec3(const mat3p& m, const vec3p& v) {
    return {
        m.m[0]*v.x + m.m[1]*v.y + m.m[2]*v.z,
        m.m[3]*v.x + m.m[4]*v.y + m.m[5]*v.z,
        m.m[6]*v.x + m.m[7]*v.y + m.m[8]*v.z,
        0.0f
    };
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
