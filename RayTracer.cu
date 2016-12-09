#include "RayTracer.cuh"

#include <assert.h>
#include "Intersection.cuh"

using namespace glm;


namespace UltraLod
{
    // RayTracer

    namespace
    {
        struct ColorConversion
        {
            __device__
                Color operator()(const float4& v) const
            {
                Color c;

                c.r = (uint8_t)(__min(__max(0.0f, v.x), 1.0f) * 255);
                c.g = (uint8_t)(__min(__max(0.0f, v.y), 1.0f) * 255);
                c.b = (uint8_t)(__min(__max(0.0f, v.z), 1.0f) * 255);
                c.a = (uint8_t)(__min(__max(0.0f, v.w), 1.0f) * 255);

                return c;
            }
        };

        __constant__ float3 c_dCameraPos;
        __constant__ float3 c_dFrustumTopLeft;
        __constant__ float3 c_dRight;
        __constant__ float3 c_dDown;
        __constant__ float3 c_dStep;
        __constant__ uint2  c_dGrid;
        __constant__ float3 c_dHalfPixel;
        __constant__ float2 c_dJitter;
        __constant__ float3 c_dBounds[2];
        __constant__ float2 c_dFloorSize;
        __constant__ float3 c_dFloorPos;

        __device__ vec3 GetRayDir(int x, int y, const vec3& pos)
        {
            vec3 rightStep = (vec3&)c_dRight / (float)c_dGrid.x;
            vec3 downStep = (vec3&)c_dDown / (float)c_dGrid.y;
            vec3 rayRight = rightStep * (float)x;
            vec3 rayDown = downStep * (float)y;

            vec3 pointOnFarPlane = (vec3&)c_dFrustumTopLeft + (vec3&)c_dHalfPixel + rayRight + rayDown;

            // Apply a small jitter
            vec3 jitter = (c_dJitter.x - 0.5f) * rightStep + (c_dJitter.y - 0.5f) * downStep;
            vec3 rayDir = glm::normalize(pointOnFarPlane + jitter - pos);

            return rayDir;
        }

        __device__ bool FloorRayCast(const vec3& pos, const vec3& dir, float& outDist, int& outEven)
        {
            if (dir.y == 0)
                return false;

            auto relativePos = pos - (const vec3&)c_dFloorPos;

            if (relativePos.y > 0 && dir.y >= 0)
                return false;

            if (relativePos.y < 0 && dir.y <= 0)
                return false;

            // Get steps required to hit the floor
            float steps = relativePos.y / -dir.y;
            vec3  posOnFloor = (pos + dir * steps) - (const vec3&)c_dFloorPos;

            // Hit the floor?
            if (posOnFloor.x < c_dFloorSize.x * -0.5f || posOnFloor.x > c_dFloorSize.x * 0.5f)
                return false;

            if (posOnFloor.z < c_dFloorSize.y * -0.5f || posOnFloor.z > c_dFloorSize.y * 0.5f)
                return false;

            // Store travel distance
            outDist = steps;

            int x = (int)fabsf(floorf(posOnFloor.x));
            int z = (int)fabsf(floorf(posOnFloor.z));

            // And extra grid value
            outEven = (x + z) % 2;

            return true;
        }

        __device__ bool OctreeRayCast(const SparseOctree::Node* tree, const ColorRGB24* colors, const vec3& pos, const vec3& dir, const Aabb& bounds, float& outDist, ColorRGB24& outColor)
        {
            // Check first against octree bounds
            if (!Intersects(bounds, pos, dir, outDist))
                return false;

            float tMin = 0;
            float tMax = FLT_MAX;

            // Continue recursion to nodes until a leaf is met
            NodeRayCast(tree, colors, pos, dir, tMin, tMax, outColor, bounds);

            // Set output
            outDist = tMax;

            return tMax != FLT_MAX;
        }

        __global__ void kernelRayTrace(const SparseOctree::Node* tree, const ColorRGB24* colors, int* outSampleCounts, float4* outColor)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;
            auto idx = y * c_dGrid.x + x;

            // Compute ray direction
            vec3 rayStart = (vec3&)c_dCameraPos;
            vec3 rayDir = GetRayDir(x, y, rayStart);

            float      tTree = FLT_MAX;
            float      tFloor = FLT_MAX;
            int        even = 0;
            ColorRGB24 color = { 0, 0, 0 };

            // Check collision distances to floor and tree
            OctreeRayCast(tree, colors, rayStart, rayDir, (const Aabb&)c_dBounds, tTree, color);
            FloorRayCast(rayStart, rayDir, tFloor, even);

            if (tTree != FLT_MAX && tTree <= tFloor)
            {
                outColor[idx].x += (float)color.r / 255.0f;
                outColor[idx].y += (float)color.g / 255.0f;
                outColor[idx].z += (float)color.b / 255.0f;
                outColor[idx].w += 1;

                atomicAdd(&outSampleCounts[idx], 1);
            }
            else if (tFloor != FLT_MAX)
            {
                if (even)
                {
                    outColor[idx].x += 1;
                    outColor[idx].y += 1;
                    outColor[idx].z += 1;
                    outColor[idx].w += 1;
                }
                else
                {
                    outColor[idx].x += 0.3f;
                    outColor[idx].y += 0.3f;
                    outColor[idx].z += 0.3f;
                    outColor[idx].w += 1.0f;
                }

                atomicAdd(&outSampleCounts[idx], 1);
            }
        }

        __global__ void kernelAverageColor(const int* sampleCounts, float4* outColors)
        {
            auto x = blockIdx.x * blockDim.x + threadIdx.x;
            auto y = blockIdx.y * blockDim.y + threadIdx.y;
            auto idx = y * c_dGrid.x + x;

            int sampleCount = sampleCounts[idx];

            if (sampleCount > 0)
            {
                outColors[idx].x /= sampleCount;
                outColors[idx].y /= sampleCount;
                outColors[idx].z /= sampleCount;
                outColors[idx].w /= sampleCount;
            }
        }
    }


    RayTracer::RayTracer(RenderTarget& renderTarget)
        : m_renderTarget(renderTarget)
        , m_floorSize(0, 0)
        , m_floorPos(0, 0, 0)
    {
        // Initialize vram buffers
        auto imageArea = m_renderTarget.GetWidth() * m_renderTarget.GetHeight();

        m_dBackBuffer.resize(imageArea);
        m_dSampleCounts.resize(imageArea);
        m_dImageBuffer.resize(imageArea);

        // Fill with empty values
        float4 initColor = { 0, 0, 0, 0 };

        thrust::fill(m_dBackBuffer.begin(), m_dBackBuffer.end(), initColor);
        thrust::fill(m_dSampleCounts.begin(), m_dSampleCounts.end(), 0);
    }

    void RayTracer::SetFloor(const vec2& size, const vec3& pos)
    {
        m_floorSize = size;
        m_floorPos = pos;
    }

    void RayTracer::Trace(const SparseOctree& tree, const Camera& camera)
    {
        // Get camera frustum in world space
        auto pos = camera.GetPosition();
        auto clipToWorld = camera.ClipToWorld();

        auto frustumTopLeft = clipToWorld * vec4(-1, 1, 1, 1);
        auto frustumTopRight = clipToWorld * vec4(1, 1, 1, 1);
        auto frustumBottomLeft = clipToWorld * vec4(-1, -1, 1, 1);

        if (frustumTopLeft.w > 0.0f)    frustumTopLeft /= frustumTopLeft.w;
        if (frustumTopRight.w > 0.0f)   frustumTopRight /= frustumTopRight.w;
        if (frustumBottomLeft.w > 0.0f) frustumBottomLeft /= frustumBottomLeft.w;

        auto rtWidth = m_renderTarget.GetWidth();
        auto rtHeight = m_renderTarget.GetHeight();
        assert(rtWidth >= 8 && rtHeight >= 8);

        auto right = vec3(frustumTopRight - frustumTopLeft);
        auto down = vec3(frustumBottomLeft - frustumTopLeft);
        auto step = right / (float)(rtWidth)+down / (float)(rtHeight);
        auto halfPixel = 0.5f * step;

        // Prepare gpu variables
        uint2 grid = { (uint32_t)rtWidth, (uint32_t)rtHeight };

        ThrowIfFailed(cudaMemcpyToSymbol(c_dCameraPos, &pos, sizeof(pos)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dFrustumTopLeft, &frustumTopLeft, sizeof(frustumTopLeft)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dRight, &right, sizeof(right)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dDown, &down, sizeof(down)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dStep, &step, sizeof(step)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dGrid, &grid, sizeof(grid)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dHalfPixel, &halfPixel, sizeof(halfPixel)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dBounds, &tree.GetBounds(), sizeof(Aabb)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dFloorSize, &m_floorSize, sizeof(vec2)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dFloorPos, &m_floorPos, sizeof(vec3)));

        auto& treeNodes = tree.GetTreeNodes();
        auto& treeColors = tree.GetTreeColors();

        // Upload tree to gpu
        m_dTree.resize(treeNodes.size());
        m_dTreeColors.resize(treeColors.size());

        /*vec3 start(10, 0, 0);
        vec3 dir(-1, 0.05f, 0.05f);
        dir = normalize(dir);
        float tMin = 0.0f;
        float tMax = FLT_MAX;
        ColorRGB24 outColor;
        auto bounds = tree.GetBounds();

        NodeRayCast(
            tree.GetTreeNodes().data(),
            tree.GetTreeColors().data(),
            start, dir,
            tMin, tMax,
            outColor, bounds,
            0);*/

        ThrowIfFailed(cudaMemcpy(m_dTree.data().get(), treeNodes.data(), treeNodes.size() * sizeof(SparseOctree::Node), cudaMemcpyHostToDevice));
        ThrowIfFailed(cudaMemcpy(m_dTreeColors.data().get(), treeColors.data(), treeColors.size() * sizeof(ColorRGB24), cudaMemcpyHostToDevice));

        // Evaluate dimensions for ray casting
        dim3 block(8, 8, 1);
        dim3 groups(rtWidth / block.x, rtHeight / block.y, 1);

        if (groups.x == 0 || groups.y == 0)
            return;

        // Launch ray tracer kernel using 8 samples
        for (int i = 0; i < 8; i++)
        {
            // Different jitter for each iteration
            float2 jitter = { Halton<2>(i + 1), Halton<3>(i + 1) };

            ThrowIfFailed(cudaMemcpyToSymbol(c_dJitter, &jitter, sizeof(jitter)));

            kernelRayTrace<<<groups, block>>> (
                m_dTree.data().get(),
                m_dTreeColors.data().get(),
                m_dSampleCounts.data().get(),
                m_dBackBuffer.data().get());

            ThrowIfFailed(cudaGetLastError());
        }

        // Average color values
        kernelAverageColor<<<groups, block>>> (
            m_dSampleCounts.data().get(),
            m_dBackBuffer.data().get());

        ThrowIfFailed(cudaDeviceSynchronize());

        // Convert to 32bit colors
        thrust::transform(m_dBackBuffer.begin(), m_dBackBuffer.end(), m_dImageBuffer.begin(), ColorConversion());

        // Read pixel data back
        ThrowIfFailed(cudaMemcpy(m_renderTarget.DataPtr(), m_dImageBuffer.data().get(), rtWidth * rtHeight * sizeof(Color), cudaMemcpyDeviceToHost));
    }
}