#pragma once

#include "Defs.cuh"

__host__ __device__
inline float Dot(const glm::vec2& v0, const glm::vec2& v1)
{
    return v0.x * v1.x + v0.y * v1.y;
}

__host__ __device__
inline float Dot(const glm::vec3& v0, const glm::vec3& v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

__host__ __device__
inline glm::vec3 Cross(const glm::vec3& a, const glm::vec3& b)
{
    glm::vec3 r;
    r.x = a.y*b.z - a.z*b.y;
    r.y = a.z*b.x - a.x*b.z;
    r.z = a.x*b.y - a.y*b.x;
    return r;
}

__host__ __device__
inline float Length(const glm::vec3& v)
{
    return sqrtf(Dot(v, v));
}