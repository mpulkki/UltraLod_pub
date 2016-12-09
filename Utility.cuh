#pragma once

#include "Defs.cuh"


inline void ThrowIfFailed(cudaError_t err)
{
    if (err != cudaSuccess)
        throw std::exception(std::to_string((int)err).c_str());
}

__device__ __host__
inline float max(glm::vec3 v)
{
    return __max(__max(v.x, v.y), v.z);
}

__device__ __host__
inline float min(glm::vec3 v)
{
    return __min(__min(v.x, v.y), v.z);
}

__device__ __host__
inline glm::vec2 min(const glm::vec2& v0, const glm::vec2& v1)
{
    glm::vec2 r;
    r.x = __min(v0.x, v1.x);
    r.y = __min(v0.y, v1.y);
    return r;
}

__device__ __host__
inline glm::vec2 max(const glm::vec2& v0, const glm::vec2& v1)
{
    glm::vec2 r;
    r.x = __max(v0.x, v1.x);
    r.y = __max(v0.y, v1.y);
    return r;
}

__device__ __host__
inline glm::vec3 min(const glm::vec3& v0, const glm::vec3& v1)
{
    glm::vec3 r;
    r.x = __min(v0.x, v1.x);
    r.y = __min(v0.y, v1.y);
    r.z = __min(v0.z, v1.z);
    return r;
}

__device__ __host__
inline glm::vec3 max(const glm::vec3& v0, const glm::vec3& v1)
{
    glm::vec3 r;
    r.x = __max(v0.x, v1.x);
    r.y = __max(v0.y, v1.y);
    r.z = __max(v0.z, v1.z);
    return r;
}

__device__ __host__
inline float max(float f0, float f1, float f2)
{
    return __max(f0, __max(f1, f2));
}

__device__ __host__
inline float min(float f0, float f1, float f2)
{
    return __min(f0, __min(f1, f2));
}

__device__ __host__
inline glm::ivec3 toIVec3(const glm::vec3& v, float unit)
{
    glm::ivec3 r;
    r.x = (int)floorf(v.x / unit);
    r.y = (int)floorf(v.y / unit);
    r.z = (int)floorf(v.z / unit);
    return r;
}

__device__ __host__
inline glm::vec3 Normalize(const glm::vec3& v)
{
    auto len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return v / len;
}

template <int TBase>
__device__ __host__
inline float Halton(int index)
{
    float result = 0.0f;
    float f = 1.0f;
    while (index > 0)
    {
        f /= TBase;
        result += f * (index % TBase);
        index = index / TBase;
    }
    return result;
}

template <typename T, T TValue>
inline T NextMultipleOf(T value)
{
    if (value % TValue != 0)
        value = value + TValue - (value % TValue);

    return value;
}

struct Color
{
    uint8_t r, g, b, a;
};

struct ColorRGB24
{
    uint8_t r, g, b;
};