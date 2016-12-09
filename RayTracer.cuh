#pragma once

#include "Utility.cuh"
#include "SparseOctree.hpp"
#include "Camera.hpp"
#include "RenderTarget.hpp"
#include <thrust/device_vector.h>

namespace UltraLod
{
    class RayTracer
    {
    public:
        RayTracer(RenderTarget& renderTarget);

        // Enables floor on output image
        void SetFloor(const glm::vec2& size, const glm::vec3& pos);

        // Ray traces given tree
        void Trace(const SparseOctree& tree, const Camera& camera);

    private:
        RenderTarget&                             m_renderTarget;
        glm::vec2                                 m_floorSize;
        glm::vec3                                 m_floorPos;
        thrust::device_vector<float4>             m_dBackBuffer;
        thrust::device_vector<int>                m_dSampleCounts;
        thrust::device_vector<Color>              m_dImageBuffer;
        thrust::device_vector<SparseOctree::Node> m_dTree;
        thrust::device_vector<ColorRGB24>         m_dTreeColors;
    };
}