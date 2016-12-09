#pragma once

#include "Defs.cuh"
#include "Utility.cuh"

//struct Voxel
//{
//    uint64_t mortonCode;
//    uint32_t color;
//};

using Voxel = uint64_t;

struct TrianglePos
{
    glm::vec3 v0;
    glm::vec3 v1;
    glm::vec3 v2;
};

struct TriangleUv
{
    glm::vec2 v0;
    glm::vec2 v1;
    glm::vec2 v2;
};

struct Texture
{
    int width, height;
    std::vector<uint8_t> data;
};

struct Aabb
{
    glm::vec3 min;
    glm::vec3 max;

    __device__ __host__
    inline glm::vec3 Center() const
    {
        return 0.5f * (min + max);
    }

    __device__ __host__
    inline glm::vec3 Size() const
    {
        return max - min;
    }
};

struct iAabb
{
    glm::ivec3 min;
    glm::ivec3 max;

    __device__ __host__
    inline glm::ivec3 Size() const
    {
        return max - min;
    }
};

struct Plane
{
    glm::vec3 n;
    float d;
};

struct Ray
{
    glm::vec3 pos;
    glm::vec3 dir;
};

struct Mesh
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec2> uvs;
    Texture texture;
};