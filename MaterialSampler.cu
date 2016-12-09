#include "MaterialSampler.cuh"
#include "RenderTarget.hpp"
#include "SparseOctree.hpp"
#include "Intersection.cuh"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

using namespace std;
using namespace glm;
using namespace UltraLod;

#define WorkGroupSize (256)

namespace
{

    __global__
    void kernelRasterizeSB(const vec2* uvs, const int* indices, int count, int width, int height, int* outStencilBuffer)
    {
        int tIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (!(tIdx < count))
            return;

        // Compute constants
        float wToUv = 1.0f / width;
        float hToUv = 1.0f / height;

        // Get triangle uvs
        auto& uv0 = uvs[indices[tIdx * 3 + 0]];
        auto& uv1 = uvs[indices[tIdx * 3 + 1]];
        auto& uv2 = uvs[indices[tIdx * 3 + 2]];

        auto minUv = min(min(uv0, uv1), uv2);
        auto maxUv = max(max(uv0, uv1), uv2);

        // Convert to texel coordinates
        auto minTexel = ivec2((int)(minUv.x * width), (int)(minUv.y * height));
        auto maxTexel = ivec2((int)(maxUv.x * width), (int)(maxUv.y * height));

        minTexel.x = __min(__max(minTexel.x, 0), width - 1);
        minTexel.y = __min(__max(minTexel.y, 0), height - 1);

        maxTexel.x = __min(__max(maxTexel.x, 0), width - 1);
        maxTexel.y = __min(__max(maxTexel.y, 0), height - 1);

        // go through texels
        for (int y = minTexel.y; y <= maxTexel.y; y++)
        for (int x = minTexel.x; x <= maxTexel.x; x++)
        {
            // Test middle and all 4 corners
            vec2 offsets[5] =
            {
                { 0,    0    },     // top left
                { 1.0f, 0    },     // top right
                { 1.0f, 1.0f },     // bottom right
                { 0,    1.0f },     // bottom left
                { 0.5f, 0.5f }      // center
            };

            bool texelInTriangle = false;

            for (int i = 0; i < 5; i++)
            {
                auto texelUv = vec2(
                    (x + offsets[i].x) * wToUv,
                    (y + offsets[i].y) * hToUv);

                // Get barycentric coordinates
                auto bcCoords = Barycentric(uv0, uv1, uv2, texelUv);

                // Even one sample in triangle is enough
                if (BarycentricInTriangle(bcCoords))
                {
                    texelInTriangle = true;
                    break;
                }
            }

            if (texelInTriangle)
            {
                // Find output index
                int outIdx = y * width + x;

                // Store triangle index to the stencil buffer
                atomicExch(&outStencilBuffer[outIdx], tIdx);
            }
        }
    }

    __global__
    void kernelSampleMaterials(const SparseOctree::Node* tree, const ColorRGB24* colors, Aabb bounds, const vec3* positions, const vec2* uvs, const int* indices, const int* stencilBuffer, float voxelSize, int width, int height, Color* outTexture)
    {
        int idxX = blockIdx.x * blockDim.x + threadIdx.x;
        int idxY = blockIdx.y * blockDim.y + threadIdx.y;

        if (!(idxX < width) || !(idxY < height))
            return;

        int bufferIdx = idxY * width + idxX;

        // Read triangle index from the stencil buffer
        auto tIdx = stencilBuffer[bufferIdx];

        if (tIdx < 0)
            return;

        // Get triangle uv-coordinates
        auto& uv0 = uvs[indices[tIdx * 3 + 0]];
        auto& uv1 = uvs[indices[tIdx * 3 + 1]];
        auto& uv2 = uvs[indices[tIdx * 3 + 2]];

        // ...and barycentric coordinates of the texel (0.5f == texel middle)
        auto texelUvX = ((float)idxX + 0.5f) / width;
        auto texelUvY = ((float)idxY + 0.5f) / height;

        auto barycentric = Barycentric(uv0, uv1, uv2, vec2(texelUvX, texelUvY));

        // Use barycentric coordinates to find texel position in world
        auto& v0 = positions[indices[tIdx * 3 + 0]];
        auto& v1 = positions[indices[tIdx * 3 + 1]];
        auto& v2 = positions[indices[tIdx * 3 + 2]];

        auto texelPos    = barycentric.x * v0 + barycentric.y * v1 + barycentric.z * v2;
        auto texelNormal = Cross(v2 - v0, v1 - v0);

        auto len = Length(texelNormal);

        if (len == 0.0f)
            return;

        // Normalize normal
        texelNormal /= len;

        // Evaluate ray to shoot
        auto rayDir = texelNormal;
        auto rayPos = texelPos - texelNormal * voxelSize;
        auto tMin   = 0.0f;
        auto tMax   = 2.0f * voxelSize;
        auto rayLen = tMax;

        ColorRGB24 sampleColor = { 0, 0, 0 };

        NodeRayCast(tree, colors, rayPos, rayDir, tMin, tMax, sampleColor, bounds);

        // Store hit in output texture
        if (tMax < rayLen)
        {
            outTexture[bufferIdx].r = sampleColor.r;
            outTexture[bufferIdx].g = sampleColor.g;
            outTexture[bufferIdx].b = sampleColor.b;
            outTexture[bufferIdx].a = 255;
        }
        else
        {
            outTexture[bufferIdx] = { 255, 0, 0, 255 };
        }
    }
}


namespace UltraLod
{
    // Cuda sampler implementation

    class CudaMaterialSampler
    {
    public:
        CudaMaterialSampler(const SparseOctree& octree, RenderTarget& renderTarget, const vector<vec3>& positions, const vector<vec2>& uvs, const vector<int>& indices);

        void RasterizeToStencilBuffer(thrust::device_vector<int>& stencilBuffer) const;
        void Sample(float voxelSize);

    private:
        const SparseOctree& m_octree;
        RenderTarget&       m_renderTarget;
        const vector<vec3>& m_positions;
        const vector<vec2>& m_uvs;
        const vector<int>&  m_indices;

        thrust::device_vector<int>  m_dIndices;
        thrust::device_vector<vec2> m_dUvs;
        thrust::device_vector<vec3> m_dPositions;
    };


    CudaMaterialSampler::CudaMaterialSampler(const SparseOctree& octree, RenderTarget& renderTarget, const vector<vec3>& positions, const vector<vec2>& uvs, const vector<int>& indices)
        : m_octree(octree)
        , m_renderTarget(renderTarget)
        , m_positions(positions)
        , m_uvs(uvs)
        , m_indices(indices)
    { }

    void CudaMaterialSampler::RasterizeToStencilBuffer(thrust::device_vector<int>& stencilBuffer) const
    {
        auto tCount     = (int)m_dIndices.size() / 3;
        auto groupCount = NextMultipleOf<int, WorkGroupSize>(tCount) / WorkGroupSize;

        kernelRasterizeSB<<<groupCount, WorkGroupSize>>>(
            m_dUvs.data().get(),
            m_dIndices.data().get(),
            tCount,
            m_renderTarget.GetWidth(),
            m_renderTarget.GetHeight(),
            stencilBuffer.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());
    }

    void CudaMaterialSampler::Sample(float voxelSize)
    {
        // Populate device vectors with mesh data
        m_dUvs.resize(m_uvs.size());
        m_dIndices.resize(m_indices.size());
        m_dPositions.resize(m_positions.size());

        thrust::copy(m_uvs.data(), m_uvs.data() + m_uvs.size(), m_dUvs.begin());
        thrust::copy(m_indices.data(), m_indices.data() + m_indices.size(), m_dIndices.begin());
        thrust::copy(m_positions.data(), m_positions.data() + m_positions.size(), m_dPositions.begin());

        // Render triangles to stencil buffer so that each texel knows it's triangle
        thrust::device_vector<int> dStencilBuffer;

        dStencilBuffer.resize(m_renderTarget.GetWidth() * m_renderTarget.GetHeight());
        thrust::fill(dStencilBuffer.begin(), dStencilBuffer.end(), -1);

        // Execute rasterization
        RasterizeToStencilBuffer(dStencilBuffer);

        // Execute material sampling
        auto textureWidth = m_renderTarget.GetWidth();
        auto textureHeight = m_renderTarget.GetHeight();

        auto gridWidth  = NextMultipleOf<int, 4>(textureWidth);
        auto gridHeight = NextMultipleOf<int, 4>(textureHeight);

        dim3 block(4, 4, 1);
        dim3 groups(gridWidth / 4, gridHeight / 4, 1);

        // Load required variables to vram
        thrust::device_vector<SparseOctree::Node> dOctree;
        thrust::device_vector<ColorRGB24>         dOctreeColors;
        thrust::device_vector<Color>              dTexture;

        auto& nodes  = m_octree.GetTreeNodes();
        auto& colors = m_octree.GetTreeColors();

        dOctree.resize(nodes.size());
        dOctreeColors.resize(colors.size());
        dTexture.resize(textureWidth * textureHeight);

        thrust::copy(nodes.data(), nodes.data() + nodes.size(), dOctree.begin());
        thrust::copy(colors.data(), colors.data() + colors.size(), dOctreeColors.begin());

        kernelSampleMaterials<<<groups, block>>>(
            dOctree.data().get(),
            dOctreeColors.data().get(),
            m_octree.GetBounds(),
            m_dPositions.data().get(),
            m_dUvs.data().get(),
            m_dIndices.data().get(),
            dStencilBuffer.data().get(),
            voxelSize,
            textureWidth,
            textureHeight,
            dTexture.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());

        // Copy result texture back to cpu memory
        thrust::copy(dTexture.begin(), dTexture.end(), m_renderTarget.DataPtr());
    }


    // Public interface

    MaterialSampler::MaterialSampler(const SparseOctree& octree, RenderTarget& renderTarget, const vector<vec3>& positions, const vector<vec2>& uvs, const vector<int>& indices)
        : m_octree(octree)
        , m_renderTarget(renderTarget)
        , m_positions(positions)
        , m_uvs(uvs)
        , m_indices(indices)
    { }

    void MaterialSampler::Sample(float voxelSize)
    {
        // Sample materials using cuda implementation
        CudaMaterialSampler sampler(m_octree, m_renderTarget, m_positions, m_uvs, m_indices);

        sampler.Sample(voxelSize);
    }
}