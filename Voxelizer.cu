#include "Voxelizer.hpp"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "Utility.cuh"
#include "Intersection.cuh"
#include "Timer.hpp"
#include "ComputationParams.hpp"

using namespace thrust;
using namespace glm;
using namespace std;
using namespace UltraLod;

#define WorkGroupSize (256)
#define MaxVoxelizationBatchCount (8*1000192ULL)     // Has to be multiple of WorkGroupSize

static_assert(MaxVoxelizationBatchCount % WorkGroupSize == 0, "Invalid voxelization batch count. Must be multiple of WorkGroupSize");

#define STAT(expression) expression


// Gpu implementation of voxelizer

namespace
{
    __constant__ float c_dSceneBounds[6];        // Bounds of WHOLE scene
    __constant__ float c_dSplitBounds[6];        // Bounds of split nodes
    __constant__ float c_dVoxelSize;
    __constant__ int c_dDepth;
    __constant__ int c_dTextureWidth;
    __constant__ int c_dTextureHeight;
    __device__   int dIntersectionCounter;
    __device__   int dVoxelCounter;

    template <typename T, T value>
    struct PredEquals
    {
        __host__ __device__
        bool operator()(const T& v)
        {
            return v == value;
        }
    };

    __device__ __host__
    inline int LowerBound(const uint64_t* data, int len, uint64_t key)
    {
        int lo = 0;
        int hi = len - 1;

        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (data[mid] > key)
                hi = mid - 1;
            else if (data[mid] < key)
                lo = mid + 1;
            else
                return mid;
        }

        return lo - 1;
    }

    __device__
    inline uint64_t ComputeMortonValue(const ivec3& voxelIdx)
    {
        uint64_t value = 0;

        for (int i = 0; i < c_dDepth; i++)
        {
            uint64_t xBit = (voxelIdx.x & (1 << i)) != 0;
            uint64_t yBit = (voxelIdx.y & (1 << i)) != 0;
            uint64_t zBit = (voxelIdx.z & (1 << i)) != 0;

            // Interleave bits
            value |= xBit << (3 * i + 0);
            value |= yBit << (3 * i + 1);
            value |= zBit << (3 * i + 2);
        }

        return value;
    }

    __device__
    inline iAabb ComputeVoxelBounds(const TrianglePos& tri)
    {
        // Compute triangle bounds in world space
        vec3 triMin = min(min(tri.v0, tri.v1), tri.v2);
        vec3 triMax = max(max(tri.v0, tri.v1), tri.v2);

        // Clamp to volume bounds (and range)
        const Aabb& splitBounds = (const Aabb&)c_dSplitBounds;
        const Aabb& sceneBounds = (const Aabb&)c_dSceneBounds;

        triMax = min(triMax, splitBounds.max) - sceneBounds.min;
        triMin = max(triMin, splitBounds.min) - sceneBounds.min;

        // Convert to int voxel coordinates
        ivec3 vMin = toIVec3(triMin, c_dVoxelSize);
        ivec3 vMax = toIVec3(triMax, c_dVoxelSize);

        return { vMin, vMax };
    }

    __device__
    inline uint64_t ComputeVoxelCoverage(const TrianglePos& tri)
    {
        auto size = ComputeVoxelBounds(tri).Size() + ivec3(1, 1, 1);
        return (uint64_t)size.x * (uint64_t)size.y * (uint64_t)size.z;
    }

    __device__
    Aabb GetVoxelBounds(const ivec3& voxel)
    {
        const Aabb& bounds = (const Aabb&)c_dSceneBounds;

        // Compute world bounds of the voxel
        Aabb voxelBounds;

        voxelBounds.min = (vec3)voxel * c_dVoxelSize + bounds.min;
        voxelBounds.max = voxelBounds.min + c_dVoxelSize;

        return voxelBounds;
    }

    __device__
    inline ColorRGB24 SampleTexture(const vec2& uv, const ColorRGB24* texture)
    {
        // To texel coordinates
        auto texelX = (int)(uv.x * c_dTextureWidth);
        auto texelY = (int)(uv.y * c_dTextureHeight);

        // clamp mode
        texelX = __max(__min(texelX, c_dTextureWidth  - 1), 0);
        texelY = __max(__min(texelY, c_dTextureHeight - 1), 0);

        // Convert to 1D value
        auto texelIdx = texelY * c_dTextureWidth + texelX;

        if (texelIdx < 0 || texelIdx >= c_dTextureWidth * c_dTextureHeight)
            return { 0, 0, 0 };

        return texture[texelIdx];
    }

    __device__
    ivec3 To3DvoxelIdx(const iAabb& coverage, int voxelIdx)
    {
        ivec3 size = coverage.max - coverage.min + ivec3(1, 1, 1);

        int x = voxelIdx % size.x;
        int y = (voxelIdx / size.x) % size.y;
        int z = voxelIdx / (size.x * size.y);

        return coverage.min + ivec3(x, y, z);
    }

    __global__
    void kernelOctreeSplit(const TrianglePos* pos, const int* indices, uint8_t* outFlag)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Get triangle using index mapping
        int   triIdx = indices[idx];
        auto& tri    = pos[triIdx];

        // Write 1 if triangle is intersecting these bounds
        auto intersects = Intersects((const Aabb&)c_dSplitBounds, tri);
        outFlag[idx] = intersects;

        // Increment counter to count intersecting triangles
        if (intersects)
            atomicAdd(&dIntersectionCounter, 1);
    }

    __global__
    void kernelComputeVoxelCoverage(const TrianglePos* pos, const int* indices, uint64_t* outVoxelCoverage)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Get triangle using index mapping
        int   triIdx = indices[idx];
        auto& tri    = pos[triIdx];

        // Compute coverage
        outVoxelCoverage[idx] = ComputeVoxelCoverage(tri);
    }

    __global__
    void kernelVoxelize(const TrianglePos* tris, const TriangleUv* uvs, const ColorRGB24* texture, const int* indices, const int* triMap, const uint64_t* voxelCoverage,
        Voxel* outVoxels, ColorRGB24* outColors)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        // Get triangle and other input values
        auto& triIdx   = triMap[idx];
        auto& tri      = tris[indices[triIdx]];
        auto& triUv    = uvs[indices[triIdx]];
        auto  voxelIdx = (int)(idx - voxelCoverage[triIdx]);

        // Compute voxel bounds of the triangle
        auto  triVoxelCoverage = ComputeVoxelBounds(tri);
        ivec3 voxelPos         = To3DvoxelIdx(triVoxelCoverage, voxelIdx);

        // Execute triangle-box intersection
        auto voxelWorldBounds = GetVoxelBounds(voxelPos);

        // Voxel is actually a morton value
        Voxel voxel = 0;

        if (Intersects(voxelWorldBounds, tri))
        {
            voxel = ComputeMortonValue(voxelPos);

            // Sample texture color at this point
            vec3 bc = ClosestPointBConTri(voxelWorldBounds.Center(), tri);
            vec2 uv = triUv.v0 * bc.x + triUv.v1 * bc.y + triUv.v2 * bc.z;

            // Use voxel counter value as the index
            int outIdx = atomicAdd(&dVoxelCounter, 1);

            outVoxels[outIdx] = voxel;
            outColors[outIdx] = SampleTexture(uv, texture);
        }
    }

    __global__
    void kernelCreateMappings(const uint64_t* voxelCoverages, int coverageCount, int* outTriMap)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        outTriMap[idx] = LowerBound(voxelCoverages, coverageCount, idx);
    }
}


namespace UltraLod
{
    struct DeviceTexture
    {
        device_vector<ColorRGB24> Data;
        int Width;
        int Height;
    };

    class CudaVoxelizer
    {
    public:
        CudaVoxelizer(vector<Voxel>& resultVoxels, vector<ColorRGB24>& resultColors, Voxelizer::Stats& stats);
        ~CudaVoxelizer();

        void Voxelize(const Mesh& mesh, const ComputationParams& params);

    private:
        // copy not allowed
        CudaVoxelizer(const CudaVoxelizer&) = delete;
        CudaVoxelizer& operator=(const CudaVoxelizer&) = delete;

        // Splits input data set to octants to reduce workload
        void Split(const Aabb& bounds, int triangleCount, int currentDepth);

        // Executes triangle voxelization
        void Voxelize(const Aabb& bounds, int triangleCount, uint64_t totalCoverage);

    private:
        ComputationParams m_params;

        vector<Voxel>&      m_resultVoxels;     // Results of the voxelization process
        vector<ColorRGB24>& m_resultColors;
        Voxelizer::Stats&   m_stats;

        device_vector<int>         m_dTriIndices;       // Index mappings to TrianglePos and TriangleUv
        device_vector<uint64_t>    m_dVoxelCoverage;    // Number of voxels covered by triangles
        device_vector<uint8_t>     m_dVoxelFlags;
        device_vector<TrianglePos> m_dPositions;
        device_vector<TriangleUv>  m_dUvs;

        device_vector<int>        m_dTriMap;        // maps index => triangle idx
        device_vector<Voxel>      m_dVoxels;        // Morton values of voxels
        device_vector<ColorRGB24> m_dColors;        // Output colors of voxels

        DeviceTexture m_dTexture;
    };

    CudaVoxelizer::CudaVoxelizer(vector<Voxel>& resultVoxels, vector<ColorRGB24>& resultColors, Voxelizer::Stats& stats)
        : m_resultVoxels(resultVoxels)
        , m_resultColors(resultColors)
        , m_stats(stats)
    {
        m_dTexture.Width = 0;
        m_dTexture.Height = 0;
    }

    CudaVoxelizer::~CudaVoxelizer()
    { }

    void CudaVoxelizer::Voxelize(const Mesh& mesh, const ComputationParams& params)
    {
        if (!mesh.positions.size())
            return;

        STAT(Timer timer);
        STAT(Timer totalTimer);

        m_params = params;

        assert(mesh.positions.size() == mesh.uvs.size());
        assert(mesh.positions.size() % 3 == 0);
        assert(m_params.GetDepth() <= 21);      // Max 21 levels can be stored in uint64_t

        // Initialize buffers. Add extra space to buffers because gpu will work with group size of cWorkGroupSize
        auto triangleCount = (int)(mesh.positions.size() / 3);
        auto bufferItemCount = triangleCount + WorkGroupSize;

        m_dTriIndices.resize(bufferItemCount);
        m_dVoxelCoverage.resize(bufferItemCount);
        m_dVoxelFlags.resize(bufferItemCount);
        m_dPositions.resize(bufferItemCount);
        m_dUvs.resize(bufferItemCount);
        m_dTriMap.resize(MaxVoxelizationBatchCount * sizeof(int));  // These can get quite big!
        m_dVoxels.resize(MaxVoxelizationBatchCount * sizeof(Voxel));
        m_dColors.resize(MaxVoxelizationBatchCount * sizeof(ColorRGB24));

        auto posPtr = (const TrianglePos*)mesh.positions.data();
        auto uvsPtr = (const TriangleUv*)mesh.uvs.data();

        thrust::copy(posPtr, posPtr + triangleCount, m_dPositions.begin());
        thrust::copy(uvsPtr, uvsPtr + triangleCount, m_dUvs.begin());
        thrust::sequence(m_dTriIndices.begin(), m_dTriIndices.end());

        if (bufferItemCount > triangleCount)
        {
            // Fill extra slots with replicated data
            thrust::fill(m_dPositions.begin() + triangleCount, m_dPositions.end(), posPtr[0]);
            thrust::fill(m_dUvs.begin() + triangleCount, m_dUvs.end(), uvsPtr[0]);
            thrust::fill(m_dTriIndices.begin() + triangleCount, m_dTriIndices.end(), 0);
        }

        // Upload texture
        auto texturePtr = (const ColorRGB24*)mesh.texture.data.data();
        auto textureWidth = mesh.texture.width;
        auto textureHeight = mesh.texture.height;
        auto textureSize = textureWidth * textureHeight;
        assert(textureWidth * textureHeight * sizeof(ColorRGB24) == mesh.texture.data.size());

        m_dTexture.Width = mesh.texture.width;
        m_dTexture.Height = mesh.texture.height;
        m_dTexture.Data.resize(textureSize);

        thrust::copy(texturePtr, texturePtr + textureSize, m_dTexture.Data.begin());

        // Prepare gpu variables
        auto  voxelSize = m_params.GetVoxelSize();
        auto  depth     = m_params.GetDepth();
        auto& bounds    = m_params.GetBounds();

        ThrowIfFailed(cudaMemcpyToSymbol(c_dVoxelSize, &voxelSize, sizeof(voxelSize)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dDepth, &depth, sizeof(depth)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dSceneBounds, &bounds, sizeof(Aabb)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dTextureWidth, &m_dTexture.Width, sizeof(m_dTexture.Width)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dTextureHeight, &m_dTexture.Height, sizeof(m_dTexture.Height)));

        STAT(m_stats.initTime = timer.End());

        // Start recursive voxelization
        Split(bounds, triangleCount, 0);

        STAT(m_stats.totalTime += totalTimer.End());
    }

    void CudaVoxelizer::Split(const Aabb& bounds, int triangleCount, int currentDepth)
    {
        int resultTriCount = 0;

        STAT(Timer timer);

        // Prepare gpu variables
        ThrowIfFailed(cudaMemcpyToSymbol(dIntersectionCounter, &resultTriCount, sizeof(dIntersectionCounter)));
        ThrowIfFailed(cudaMemcpyToSymbol(c_dSplitBounds, &bounds, sizeof(Aabb)));

        // Check triangles that are colliding with provided bounds
        int groupCount = NextMultipleOf<int, WorkGroupSize>(triangleCount) / WorkGroupSize;

        assert(groupCount * WorkGroupSize >= triangleCount);

#ifdef _DEBUG
        // Validate that kernels are not accessing invalid memory
        auto maxIdx = groupCount * WorkGroupSize;
        assert(maxIdx <= m_dPositions.size());
        assert(maxIdx <= m_dTriIndices.size());
        assert(maxIdx <= m_dVoxelFlags.size());
        assert(maxIdx <= m_dVoxelCoverage.size());
#endif

        kernelOctreeSplit << <groupCount, WorkGroupSize >> > (
            m_dPositions.data().get(),
            m_dTriIndices.data().get(),
            m_dVoxelFlags.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());

        // Read back number of triangles intersecting these bounds
        ThrowIfFailed(cudaMemcpyFromSymbol(&resultTriCount, dIntersectionCounter, sizeof(dIntersectionCounter)));

        // Empty volume?
        if (resultTriCount == 0)
            return;

        STAT(Timer subTimer);

        // Sort index array so that indices of intersecting triangles are in front
        thrust::stable_partition(m_dTriIndices.begin(), m_dTriIndices.begin() + triangleCount, m_dVoxelFlags.begin(), PredEquals<uint8_t, 1>());

        STAT(m_stats.splitPartitionTime += subTimer.End());

        // Evaluate how many voxel intersection test would need to be executed with this set of triangles
        groupCount = NextMultipleOf<int, WorkGroupSize>(resultTriCount) / WorkGroupSize;

        kernelComputeVoxelCoverage << <groupCount, WorkGroupSize >> > (
            m_dPositions.data().get(),
            m_dTriIndices.data().get(),
            m_dVoxelCoverage.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());

        auto totalVoxelCoverage = thrust::reduce(m_dVoxelCoverage.begin(), m_dVoxelCoverage.begin() + resultTriCount, 0ULL, thrust::plus<uint64_t>());

        STAT(m_stats.splitCount++);
        STAT(m_stats.splitTime += timer.End());
        STAT(m_stats.triAabbTestCount += triangleCount);

        // Use total voxel coverage to determine if we should split more or start triangle voxelixation
        if (totalVoxelCoverage <= MaxVoxelizationBatchCount || currentDepth + 1 >= m_params.GetDepth())
        {
            Voxelize(bounds, resultTriCount, totalVoxelCoverage);
        }
        else
        {
            // Continue recursive splitting
            for (int i = 0; i < 8; i++)
                Split(ComputeOctant(bounds, i), resultTriCount, currentDepth + 1);
        }
    }

    void CudaVoxelizer::Voxelize(const Aabb& bounds, int triangleCount, uint64_t totalCoverage)
    {
        assert(triangleCount > 0);
        assert(totalCoverage <= MaxVoxelizationBatchCount);     // TODO: not implemented yet!
        assert(totalCoverage <= m_dVoxels.size());
        assert(totalCoverage <= m_dColors.size());

        STAT(Timer timer);
        STAT(Timer subTimer);

        // Evaluate starting indices for triangles' voxels, ie. create mappings from index to triangle idx and voxel idx
        thrust::exclusive_scan(m_dVoxelCoverage.begin(), m_dVoxelCoverage.begin() + triangleCount, m_dVoxelCoverage.begin());

        STAT(m_stats.exclusiveScanTime += subTimer.Reset());

        // Create triangle mappings for voxelization
        auto groupCount = NextMultipleOf<uint64_t, WorkGroupSize>(totalCoverage) / WorkGroupSize;

        kernelCreateMappings << <(int)groupCount, WorkGroupSize >> > (
            m_dVoxelCoverage.data().get(),
            triangleCount,
            m_dTriMap.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());

        STAT(m_stats.mappingKernelTime += subTimer.Reset());

        // Init gpu variables for voxelization
        int solidVoxels = 0;
        ThrowIfFailed(cudaMemcpyToSymbol(dVoxelCounter, &solidVoxels, sizeof(solidVoxels)));

        // Launch voxelization!
        groupCount = NextMultipleOf<uint64_t, WorkGroupSize>(totalCoverage) / WorkGroupSize;

        assert(groupCount * WorkGroupSize <= m_dTriMap.size());
        assert(groupCount * WorkGroupSize <= m_dVoxels.size());
        assert(groupCount * WorkGroupSize <= m_dColors.size());

        STAT(subTimer.Reset());

        kernelVoxelize<<<(int)groupCount, WorkGroupSize>>>(
            m_dPositions.data().get(),
            m_dUvs.data().get(),
            m_dTexture.Data.data().get(),
            m_dTriIndices.data().get(),
            m_dTriMap.data().get(),
            m_dVoxelCoverage.data().get(),
            m_dVoxels.data().get(),
            m_dColors.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());

        STAT(m_stats.voxelizationKernelTime += subTimer.Reset());

        // Get the number of solid voxels
        ThrowIfFailed(cudaMemcpyFromSymbol(&solidVoxels, dVoxelCounter, sizeof(dVoxelCounter)));

        // Download results to ram
        auto prevVoxelCount = m_resultVoxels.size();

        assert(m_resultVoxels.size() == m_resultColors.size());
        m_resultVoxels.resize(prevVoxelCount + solidVoxels);
        m_resultColors.resize(prevVoxelCount + solidVoxels);

        ThrowIfFailed(cudaMemcpy(m_resultVoxels.data() + prevVoxelCount, m_dVoxels.data().get(), solidVoxels * sizeof(Voxel), cudaMemcpyDeviceToHost));
        ThrowIfFailed(cudaMemcpy(m_resultColors.data() + prevVoxelCount, m_dColors.data().get(), solidVoxels * sizeof(ColorRGB24), cudaMemcpyDeviceToHost));

        STAT(m_stats.voxelizationTime += timer.End());
        STAT(m_stats.voxelizationCount++);
        STAT(m_stats.triAabbTestCount += totalCoverage);
    }



    // Voxelizer

    struct VoxelizerImpl
    {
        VoxelizerImpl(vector<Voxel>& voxels, vector<ColorRGB24>& colors)
            : voxelizer(voxels, colors, stats)
        {
            memset(&stats, 0, sizeof(stats));
        }

        CudaVoxelizer     voxelizer;
        ComputationParams params;
        Voxelizer::Stats  stats;
    };

    Voxelizer::Voxelizer(const ComputationParams& params, vector<Voxel>& voxels, vector<ColorRGB24>& colors)
    {
        m_impl = make_unique<VoxelizerImpl>(voxels, colors);

        // Set rest of the parameters
        m_impl->params = params;
    }

    Voxelizer::~Voxelizer()
    { }

    const Voxelizer::Stats& Voxelizer::GetStats() const
    {
        return m_impl->stats;
    }

    void Voxelizer::Voxelize(const Mesh& mesh)
    {
        // Execute cuda voxelization
        m_impl->voxelizer.Voxelize(mesh, m_impl->params);
    }
}