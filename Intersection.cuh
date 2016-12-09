#pragma once

#include "Geometry.hpp"
#include "Math.cuh"
#include "SparseOctree.hpp"

#ifdef AXIS_CHECK
#   error "AXIS_CHECK already defined!"
#endif

#define AXIS_CHECK(boxNorm, face) \
    axis = Cross(boxNorm, face); \
    r = extent.x * abs(Dot(vec3(1, 0, 0), axis)) + extent.y * abs(Dot(vec3(0, 1, 0), axis)) + extent.z * abs(Dot(vec3(0, 0, 1), axis)); \
    p0 = Dot(v0, axis); \
    p1 = Dot(v1, axis); \
    p2 = Dot(v2, axis); \
    if (max(p0, p1, p2) < -r || min(p0, p1, p2) > r) \
        return false;

namespace UltraLod
{
    // Computes 2d barycentric triangle coordinates
    __device__ __host__
    inline glm::vec3 Barycentric(const glm::vec2& a, const glm::vec2& b, const glm::vec2& c, const glm::vec2& p)
    {
        auto v0 = b - a;
        auto v1 = c - a;
        auto v2 = p - a;

        float d00 = Dot(v0, v0);
        float d01 = Dot(v0, v1);
        float d11 = Dot(v1, v1);
        float d20 = Dot(v2, v0);
        float d21 = Dot(v2, v1);
        float denom = d00 * d11 - d01 * d01;

        glm::vec3 bc;

        bc.y = (d11 * d20 - d01 * d21) / denom;
        bc.z = (d00 * d21 - d01 * d20) / denom;
        bc.x = 1.0f - bc.y - bc.z;

        return bc;
    }

    __device__ __host__
    inline bool BarycentricInTriangle(const glm::vec3& bc)
    {
        return
            bc.x >= 0.0f && bc.x <= 1.0f &&
            bc.y >= 0.0f && bc.y <= 1.0f &&
            bc.z >= 0.0f && bc.z <= 1.0f;
    }

    __device__ __host__
    inline bool Intersects(const Aabb& aabb, const Plane& plane)
    {
        using namespace glm;

        vec3 center = aabb.Center();
        vec3 extent = aabb.max - center;

        float r = extent.x * abs(plane.n.x) + extent.y * abs(plane.n.y) + extent.z * abs(plane.n.z);
        float s = Dot(plane.n, center) - plane.d;

        return abs(s) <= r;
    }

    __device__ __host__
    inline float Intersects(const Plane& plane, const glm::vec3& p, const glm::vec3& d, float& outDist)
    {
        using namespace glm;

        // Compute t value along vector ab
        outDist = (plane.d - Dot(plane.n, p)) / Dot(plane.n, d);

        // collision if t is between [0.0f, 1.0f]
        return outDist >= 0.0f && outDist <= 1.0f;
    }

    __device__ __host__
    inline bool Intersects(const Aabb& aabb, const TrianglePos& tri)
    {
        using namespace glm;

        vec3 center = aabb.Center();
        vec3 extent = 0.5f * aabb.Size();

        // Triangle position relative to bounds
        vec3 v0 = tri.v0 - center;
        vec3 v1 = tri.v1 - center;
        vec3 v2 = tri.v2 - center;

        // Triangle edges
        vec3 f0 = v1 - v0;
        vec3 f1 = v2 - v1;
        vec3 f2 = v0 - v2;

        vec3 boxNorm, triNorm, axis;
        float p0, p1, p2, r;

        // Check separating axes
        AXIS_CHECK(vec3(1, 0, 0), f0);
        AXIS_CHECK(vec3(1, 0, 0), f1);
        AXIS_CHECK(vec3(1, 0, 0), f2);

        AXIS_CHECK(vec3(0, 1, 0), f0);
        AXIS_CHECK(vec3(0, 1, 0), f1);
        AXIS_CHECK(vec3(0, 1, 0), f2);

        AXIS_CHECK(vec3(0, 0, 1), f0);
        AXIS_CHECK(vec3(0, 0, 1), f1);
        AXIS_CHECK(vec3(0, 0, 1), f2);

        vec3 minv = min(min(v0, v1), v2);
        vec3 maxv = max(max(v0, v1), v2);

        if (maxv.x < -extent.x || maxv.y < -extent.y || maxv.z < -extent.z)
            return false;

        if (minv.x > extent.x || minv.y > extent.y || minv.z > extent.z)
            return false;

        // Execute bounds plane test
        Plane p;
        p.n = Cross(f0, f1);
        p.d = Dot(p.n, tri.v0);

        return Intersects(aabb, p);
    }


    __device__ __host__
    inline bool Intersects(const Aabb& bounds, const glm::vec3& pos, const glm::vec3& dir, float& t)
    {
        using namespace glm;

        float tMin = 0;
        float tMax = FLT_MAX;
        const float epsilon = 0.00001f;

        for (int i = 0; i < 3; i++)
        {
            if (fabsf(dir[i]) < epsilon)
            {
                // Parallel ray
                if (pos[i] < bounds.min[i] || pos[i] > bounds.max[i])
                    return false;
            }
            else
            {
                float ood = 1.0f / dir[i];
                float t1 = (bounds.min[i] - pos[i]) * ood;
                float t2 = (bounds.max[i] - pos[i]) * ood;
                if (t1 > t2)
                {
                    // Swap
                    float temp = t1;
                    t1 = t2;
                    t2 = temp;
                }
                if (t1 > tMin)
                    tMin = t1;
                if (t2 < tMax)
                    tMax = t2;
                if (tMin > tMax)
                    return false;
            }
        }

        t = tMin;
        return true;
    }


    // Finds closest point on the triangle. returns BARYCENTRIC coordinates of the closest point!
    __device__ __host__
    inline glm::vec3 ClosestPointBConTri(const glm::vec3& point, const TrianglePos& tri)
    {
        using namespace glm;

        auto& a = tri.v0;
        auto& b = tri.v1;
        auto& c = tri.v2;

        auto ab = b - a;
        auto ac = c - a;
        auto ap = point - a;
        auto d1 = Dot(ab, ap);
        auto d2 = Dot(ac, ap);

        if (d1 <= 0.0f && d2 <= 0.0f)
            return vec3(1, 0, 0);

        auto bp = point - b;
        float d3 = Dot(ab, bp);
        float d4 = Dot(ac, bp);
        if (d3 >= 0.0f && d4 <= d3)
            return vec3(0, 1, 0);

        float vc = d1*d4 - d3*d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
        {
            float v = d1 / (d1 - d3);
            return vec3(1 - v, v, 0);
        }

        auto cp = point - c;
        float d5 = Dot(ab, cp);
        float d6 = Dot(ac, cp);
        if (d6 >= 0.0f && d5 <= d6)
            return vec3(0, 0, 1);

        float vb = d5*d2 - d1*d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
        {
            float w = d2 / (d2 - d6);
            return vec3(1 - w, 0, w);
        }

        float va = d3*d6 - d5*d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f)
        {
            float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return vec3(0, 1 - w, w);
        }

        float denom = 1.0f / (va + vb + vc);
        float v = vb * denom;
        float w = vc * denom;

        return vec3(1.0f - v - w, v, w);
    }

    __device__ __host__
    inline Aabb ComputeOctant(const Aabb& aabb, int idx)
    {
        assert(idx >= 0 && idx < 8);

        auto min = aabb.min;
        auto max = aabb.max;
        auto ext = 0.5f * aabb.Size();

        // TODO: branchless
        if (idx % 2 == 0)
            max.x -= ext.x;
        else
            min.x += ext.x;

        if ((idx / 2) % 2 == 0)
            max.y -= ext.y;
        else
            min.y += ext.y;

        if (idx / 4 == 0)
            max.z -= ext.z;
        else
            min.z += ext.z;

        return{ min, max };
    }

    struct RCStackFrame
    {
        uint32_t nodeIdx;
        uint8_t  childrenVisited;
        uint8_t  visitOrder;
        uint8_t  firstOctant;
        Aabb     bounds;
    };

    __device__
    inline void NodeRayCast(const SparseOctree::Node* tree, const ColorRGB24* colors, const glm::vec3& pos, const glm::vec3& dir, float& tMin, float& tMax, ColorRGB24& outColor, const Aabb& bounds)
    {
        using namespace glm;

        RCStackFrame stack[21];
        int          stackSize = 0;

        // Start from the root
        stack[stackSize++] = { 0, 0, 0, 0, bounds };

        while (stackSize > 0)
        {
            auto& frame = stack[stackSize - 1];

            if (frame.childrenVisited == 8)
            {
                // This level has been checked
                stackSize--;
                continue;
            }

            // Get octree node
            auto& node = tree[frame.nodeIdx];

            // Go through children
            for (; frame.childrenVisited < 8; frame.childrenVisited++)
            {
                if (frame.childrenVisited == 0)
                {
                    // Find first node to visit
                    float xDist, yDist, zDist;

                    auto center = bounds.Center();

                    frame.visitOrder = 0;
                    frame.firstOctant =
                        ((pos.x >= center.x) << 0) |
                        ((pos.y >= center.y) << 1) |
                        ((pos.z >= center.z) << 2);

                    Intersects({ { 1, 0, 0 }, center.x }, pos, dir, xDist);
                    Intersects({ { 0, 1, 0 }, center.y }, pos, dir, yDist);
                    Intersects({ { 0, 0, 1 }, center.z }, pos, dir, zDist);

                    xDist = abs(xDist);
                    yDist = abs(yDist);
                    zDist = abs(zDist);

                    // Get closest plane
                    if (xDist <= yDist)
                    {
                        if (xDist <= zDist)
                        {
                            if (yDist <= zDist) frame.visitOrder = 0b00000110;  // x, y, z
                            else                frame.visitOrder = 0b00001001;  // x, z, y
                        }
                        else
                            frame.visitOrder = 0b00100001;  // z, x, y
                    }
                    else
                    {
                        if (yDist <= zDist)
                        {
                            if (xDist <= zDist) frame.visitOrder = 0b00010010;  // y, x, z
                            else                frame.visitOrder = 0b00011000;  // y, z, x
                        }
                        else
                            frame.visitOrder = 0b00100100;  // z, y, x
                    }
                }

                // Get next node to visit
                auto firstSplit = (frame.visitOrder >> 4) & 0b11;
                auto secondSplit = (frame.visitOrder >> 2) & 0b11;
                auto thirdSplit = (frame.visitOrder >> 0) & 0b11;

                uint8_t childIdx = frame.firstOctant;

                childIdx ^= (1 << firstSplit)  * (frame.childrenVisited % 2);
                childIdx ^= (1 << secondSplit) * ((frame.childrenVisited / 2) % 2);
                childIdx ^= (1 << thirdSplit)  * (frame.childrenVisited / 4);

                auto isValid = node.validMask & (1 << childIdx);
                auto isLeaf = node.leafMask  & (1 << childIdx);

                if (!isValid)
                    continue;

                auto childBounds = ComputeOctant(frame.bounds, childIdx);

                // Check for intersection
                if (!Intersects(childBounds, pos, dir, tMin))
                    continue;

                if (tMin >= tMax)
                    // Solid voxel was found earlier on the ray
                    continue;

                // Is this a leaf node?
                if (isLeaf)
                {
                    if (tMin < tMax)
                    {
                        tMax = tMin;

                        if (colors)
                        {
                            // Read color from the color tree
                            auto idxBits = (uint32_t)(node.leafMask << (8 - childIdx)) & 0xff;
                            auto color = colors[node.childOffset + __popc(idxBits)];

                            outColor.r = color.r;
                            outColor.g = color.g;
                            outColor.b = color.b;
                        }
                    }

                    continue;
                }

                // Continue recursion
                stack[stackSize++] = { frame.nodeIdx + node.childOffset + childIdx, 0, 0, 0, childBounds };
                frame.childrenVisited++;

                break;
            }
        }
    }
}

#undef AXIS_CHECK