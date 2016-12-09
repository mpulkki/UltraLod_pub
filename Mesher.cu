#include "Mesher.hpp"
#include "SparseOctree.hpp"
#include "Intersection.cuh"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#define WorkGroupSize (256)
#define MaxMesherBatchSize (WorkGroupSize * 40000ULL)
#define MaxVertexBatchSize (50000)
#define EpsTreshold (0.0001f)

#define xBitPattern  (0x249249249249249ULL)     // only 20 levels
#define yBitPattern  (xBitPattern << 1)
#define zBitPattern  (xBitPattern << 2)
#define xFillPattern (~xBitPattern)
#define yFillPattern (~yBitPattern)
#define zFillPattern (~zBitPattern)

#define Voxel_xPlus(v)  ((((v | xFillPattern) + 0b001) & xBitPattern) | (v & yBitPattern) | (v & zBitPattern))
#define Voxel_xMinus(v) ((((v & xBitPattern)  - 0b001) & xBitPattern) | (v & yBitPattern) | (v & zBitPattern))
#define Voxel_yPlus(v)  ((((v | yFillPattern) + 0b010) & yBitPattern) | (v & xBitPattern) | (v & zBitPattern))
#define Voxel_yMinus(v) ((((v & yBitPattern)  - 0b010) & yBitPattern) | (v & xBitPattern) | (v & zBitPattern))
#define Voxel_zPlus(v)  ((((v | zFillPattern) + 0b100) & zBitPattern) | (v & xBitPattern) | (v & yBitPattern))
#define Voxel_zMinus(v) ((((v & zBitPattern)  - 0b100) & zBitPattern) | (v & xBitPattern) | (v & yBitPattern))

using namespace glm;
using namespace std;
using namespace thrust;
using namespace UltraLod;

struct Vec3Less     // Doesn't work if this is in anonymous struct
{
    __device__ __host__
    inline bool operator()(const vec3& v0, const vec3& v1) const
    {
        auto xd = v0.x - v1.x;
        auto yd = v0.y - v1.y;
        auto zd = v0.z - v1.z;

        if      (xd < -EpsTreshold) return true;
        else if (xd > EpsTreshold) return false;

        if      (yd < -EpsTreshold) return true;
        else if (yd > EpsTreshold) return false;

        if      (zd < -EpsTreshold) return true;
        else if (zd > EpsTreshold) return false;

        return false;
    }
};

struct Vec3Equals
{
    __device__ __host__
    inline bool operator()(const vec3& v0, const vec3& v1) const
    {
        return
            fabsf(v0.x - v1.x) <= EpsTreshold &&
            fabsf(v0.y - v1.y) <= EpsTreshold &&
            fabsf(v0.z - v1.z) <= EpsTreshold;
            //v0.x == v1.x &&
            //v0.y == v1.y &&
            //v0.z == v1.z;
    }
};

namespace
{


    struct FaceCountConversion
    {
        __device__
        inline int operator()(uint8_t face) const
        {
            return __popc((uint32_t)face);
        }
    };

    struct ShrinkToDepth
    {
    public:
        ShrinkToDepth(int depthReduce)
            : m_depthReduce(depthReduce)
        { }

        __device__ __host__
        inline Voxel operator()(const Voxel& voxel) const
        {
            // Strip flag bit
            Voxel stripVoxel = voxel & ~(1ULL << 63);

            return stripVoxel >> (m_depthReduce * 3);
        }

    private:
        int m_depthReduce;
    };

    // Binary search search
    __device__ __host__
    inline bool SolidVoxel(const Voxel* data, int len, const Voxel& voxel)
    {
        int lo = 0;
        int hi = len - 1;

        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            if (data[mid] > voxel)
                hi = mid - 1;
            else if (data[mid] < voxel)
                lo = mid + 1;
            else
                return true;
        }

        return false;
    }

    __device__ __host__
    inline Aabb GetVoxelBounds(const Voxel& voxel, const Aabb& rootBounds, float voxelSize)
    {
        // De-interleave bits
        uint64_t x = 0;
        uint64_t y = 0;
        uint64_t z = 0;

        for (int i = 0; i < 21; i++)
        {
            x |= (voxel >> (2 * i + 0)) & (1ULL << i);
            y |= (voxel >> (2 * i + 1)) & (1ULL << i);
            z |= (voxel >> (2 * i + 2)) & (1ULL << i);
        }

        vec3 min = vec3((int)x, (int)y, (int)z) * voxelSize;
        vec3 max = min + vec3(voxelSize, voxelSize, voxelSize);

        min += rootBounds.min;
        max += rootBounds.min;

        return { min, max };
    }

    __global__
    void kernelCreateAdjacencyInfo(const Voxel* voxels, int count, uint8_t* outAdjacency)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (!(idx < count))
            return;

        auto& voxel = voxels[idx];

        // Skip this if not solid (shouldn't be possible though)
        if (!SolidVoxel(voxels, count, voxel))
            return;

        uint8_t adjacency = 0;

        // Check each side for solid voxels
        adjacency |= (!SolidVoxel(voxels, count, Voxel_xMinus(voxel)) << 0);
        adjacency |= (!SolidVoxel(voxels, count, Voxel_xPlus(voxel))  << 1);
        adjacency |= (!SolidVoxel(voxels, count, Voxel_yMinus(voxel)) << 2);
        adjacency |= (!SolidVoxel(voxels, count, Voxel_yPlus(voxel))  << 3);
        adjacency |= (!SolidVoxel(voxels, count, Voxel_zMinus(voxel)) << 4);
        adjacency |= (!SolidVoxel(voxels, count, Voxel_zPlus(voxel))  << 5);

        outAdjacency[idx] = adjacency;
    }

    __global__
    void kernelCreateTriangleIndices(const int* vertexMap, int faceCount, int* outIndices)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (!(idx < faceCount))
            return;

        auto inIdx  = idx * 4;
        auto outIdx = idx * 6;

        outIndices[outIdx++] = vertexMap[inIdx + 0];
        outIndices[outIdx++] = vertexMap[inIdx + 2];
        outIndices[outIdx++] = vertexMap[inIdx + 1];

        outIndices[outIdx++] = vertexMap[inIdx + 2];
        outIndices[outIdx++] = vertexMap[inIdx + 0];
        outIndices[outIdx++] = vertexMap[inIdx + 3];
    }

    __global__
    void kernelSampleClosestDistances(
        const SparseOctree::Node* nodes,
        Aabb treeBounds,
        const vec3* vertices,
        int vertexCount,
        int sampleCount,
        float voxelSize,
        float sampleDist,
        vec3* outClosestPoints,
        float* outClosestDistances)
    {
        int vIdx = blockIdx.x * blockDim.x + threadIdx.x;

        if (!(vIdx < vertexCount))
            return;

        int sIdx   = threadIdx.y;
        int outIdx = vIdx * sampleCount + sIdx;

        // Get vertex to test
        auto& vertex = vertices[vIdx];

        // Use halton sequence to determine ray direction
        // TODO: This should be uniform distribution over unit sphere!
        vec3 dir =
        {
            Halton<3>(sIdx) * 2.0f - 1.0f,
            Halton<7>(sIdx) * 2.0f - 1.0f,
            Halton<11>(sIdx) * 2.0f - 1.0f
        };

        dir = Normalize(dir);

        // Shoot the ray!
        ColorRGB24 outColor;    // dummy
        float tMin = 0.0f;
        float tMax = sampleDist;

        NodeRayCast(nodes, nullptr, vertex, dir, tMin, tMax, outColor, treeBounds);

        // Store result if collision happened
        if (tMax < sampleDist)
        {
            auto np = vertex + dir * tMax;

            // Make sure vertex is not moved too much
            np.x = __min(__max(np.x, vertex.x - voxelSize), vertex.x + voxelSize);
            np.y = __min(__max(np.y, vertex.y - voxelSize), vertex.y + voxelSize);
            np.z = __min(__max(np.z, vertex.z - voxelSize), vertex.z + voxelSize);

            outClosestPoints[outIdx] = np;
            outClosestDistances[outIdx] = tMax;
        }
        else
            outClosestDistances[outIdx] = FLT_MAX;
    }

    __global__
    void kernelFilterClosestDistance(const vec3* closestPoints, const float* closestDistances, int vertexCount, int sampleCount, vec3* outPoints)
    {
        int vIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int sIdx = vIdx * sampleCount;

        if (!(vIdx < vertexCount))
            return;

        float closestDistance = FLT_MAX;
        vec3  closestPoint    = vec3(0, 0, 0);

        // Go through points
        for (int i = 0; i < sampleCount; i++)
        {
            auto& dist = closestDistances[sIdx + i];

            if (dist < closestDistance)
            {
                closestDistance = dist;

                // Store closest point candidate
                closestPoint = closestPoints[sIdx + i];
            }
        }

        if (closestDistance < FLT_MAX)
            outPoints[vIdx] = closestPoint;
    }

    __global__
    void kernelTriangulate(const Voxel* voxels, const uint8_t* adjacency, const int* faceIndices, int count, Aabb bounds, float voxelSize, vec3* outVertices)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (!(idx < count))
            return;

        // Get voxel information
        auto& voxel     = voxels[idx];
        auto  startIdx  = faceIndices[idx];
        auto  faces     = adjacency[idx];
        auto  faceCount = __popc(faces);
        auto  vertIdx   = startIdx * 4;

        if (!faceCount)
            return;

        // Evaluate voxel bounds
        auto voxelBounds = GetVoxelBounds(voxel, bounds, voxelSize);

        auto& min = voxelBounds.min;
        auto& vs  = voxelSize;

#define AppendTri() AppendIdx(0); AppendIdx(1); AppendIdx(2); AppendIdx(2); AppendIdx(3); AppendIdx(0);
#define AppendVert(x, y, z) (outVertices[vertIdx++] = min + vec3(x, y, z))

        // Go through each face
        if (faces & 0b000001)   // xm
        {
            AppendVert(0, 0,  vs);
            AppendVert(0, 0,  0);
            AppendVert(0, vs, 0);
            AppendVert(0, vs, vs);
        }

        if (faces & 0b000010) // xp
        {
            AppendVert(vs, 0,  0);
            AppendVert(vs, 0,  vs);
            AppendVert(vs, vs, vs);
            AppendVert(vs, vs, 0);
        }

        if (faces & 0b000100) // ym
        {
            AppendVert(0,  0, vs);
            AppendVert(vs, 0, vs);
            AppendVert(vs, 0, 0);
            AppendVert(0,  0, 0);
        }

        if (faces & 0b001000) // yp
        {
            AppendVert(0,  vs, 0);
            AppendVert(vs, vs, 0);
            AppendVert(vs, vs, vs);
            AppendVert(0,  vs, vs);
        }

        if (faces & 0b010000) // zm
        {
            AppendVert(0,  0,  0);
            AppendVert(vs, 0,  0);
            AppendVert(vs, vs, 0);
            AppendVert(0,  vs, 0);
        }

        if (faces & 0b100000) // zp
        {
            AppendVert(vs, 0,  vs);
            AppendVert(0,  0,  vs);
            AppendVert(0,  vs, vs);
            AppendVert(vs, vs, vs);
        }
    }

#undef AppendVert
#undef AppendTri
#undef AppendIdx
}

namespace UltraLod
{
    // CudaMesher

    class CudaMesher
    {
    public:
        CudaMesher(const SparseOctree& tree, const vector<Voxel>& voxels, vector<vec3>& outVertices, vector<int>& outIndices);

        void Meshify(int treeDepth);

    private:
        void GenerateAdjacencyInfo();
        void GenerateSolidVoxels(int targetDepth);
        void GenerateVoxelFaceGeometry(int depth);
        void MinimizePositionError();
        //void GenerateSolidVoxels(int targetDepth, int startDepth, Voxel parentVoxel);

    private:
        const SparseOctree&  m_tree;
        const vector<Voxel>& m_voxels;

        device_vector<Voxel>   m_dSolidVoxels;
        device_vector<uint8_t> m_dAdjacentVoxels;
        device_vector<vec3>    m_dVertices;
        vector<Voxel>          m_solidVoxels;

        float         m_voxelSize;
        vector<vec3>& m_outVertices;
        vector<int>&  m_outIndices;
    };


    CudaMesher::CudaMesher(const SparseOctree& tree, const vector<Voxel>& voxels, vector<vec3>& outVertices, vector<int>& outIndices)
        : m_tree(tree)
        , m_voxels(voxels)
        , m_outVertices(outVertices)
        , m_outIndices(outIndices)
    { }

    void CudaMesher::Meshify(int targetDepth)
    {
        // Validate depth
        auto treeDepth = m_tree.GetDepth();

        if (targetDepth >= treeDepth)
            return;

        // Generate solid voxels
        GenerateSolidVoxels(targetDepth);

        if (m_solidVoxels.size() == 0)
            return;

        // Generate adjacency information (check all sides for solid voxels)
        GenerateAdjacencyInfo();

        // Generate voxel face geometry
        GenerateVoxelFaceGeometry(targetDepth);

        // Finally try to minimize error of generated geometry by moving them closer to original surface
        MinimizePositionError();
    }

    void CudaMesher::GenerateAdjacencyInfo()
    {
        auto voxelCount = (int)m_solidVoxels.size();
        assert(voxelCount > 0);

        // Instantiate voxel bit flags
        m_dAdjacentVoxels.resize(voxelCount);
        thrust::fill(m_dAdjacentVoxels.begin(), m_dAdjacentVoxels.end(), 0);

        // Load voxels to gpu (although should alread be there)
        m_dSolidVoxels.resize(voxelCount);
        ThrowIfFailed(cudaMemcpy(m_dSolidVoxels.data().get(), m_solidVoxels.data(), voxelCount * sizeof(Voxel), cudaMemcpyHostToDevice));

        // Launch the kernel!
        auto workCount  = NextMultipleOf<int, WorkGroupSize>(voxelCount);
        auto groupCount = workCount / WorkGroupSize;

        kernelCreateAdjacencyInfo<<<(int)groupCount, WorkGroupSize>>>(
            m_dSolidVoxels.data().get(),
            voxelCount,
            m_dAdjacentVoxels.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());
    }

    void CudaMesher::GenerateSolidVoxels(int targetDepth)
    {
        auto depthDiff = m_tree.GetDepth() - targetDepth;
        assert(m_tree.GetDepth() > targetDepth);

        // Evalute voxel size of generated mesh
        m_voxelSize = m_tree.GetBounds().Size().x / (1 << targetDepth);

        // Init gpu variables
        m_dSolidVoxels.resize(std::min(m_voxels.size(), MaxMesherBatchSize));

        // Go through voxels as batches
        uint64_t startIdx   = 0;
        int      batchCount = 0;

        while (startIdx < m_voxels.size())
        {
            // Evaluate batch size
            auto batchSize = std::min(MaxMesherBatchSize, m_voxels.size() - startIdx);

            // Copy voxels of this batch to gpu
            ThrowIfFailed(cudaMemcpy(m_dSolidVoxels.data().get(), m_voxels.data() + startIdx, batchSize * sizeof(Voxel), cudaMemcpyHostToDevice));

            // Shrink them to the target depth
            thrust::transform(m_dSolidVoxels.begin(), m_dSolidVoxels.begin() + batchSize, m_dSolidVoxels.begin(), ShrinkToDepth(depthDiff));

            // Get unique only
            thrust::stable_sort(m_dSolidVoxels.begin(), m_dSolidVoxels.begin() + batchSize);
            auto newEnd = thrust::unique(m_dSolidVoxels.begin(), m_dSolidVoxels.begin() + batchSize);

            // Copy data back to cpu
            auto solidVoxelCount = newEnd - m_dSolidVoxels.begin();
            auto prevCount       = m_solidVoxels.size();

            m_solidVoxels.resize(m_solidVoxels.size() + solidVoxelCount);
            ThrowIfFailed(cudaMemcpy(m_solidVoxels.data() + prevCount, m_dSolidVoxels.data().get(), solidVoxelCount * sizeof(Voxel), cudaMemcpyDeviceToHost));

            // Update start index for the next batch
            startIdx += batchSize;
            batchCount++;
        }

        if (batchCount > 1)
        {
            // Load all back to gpu for final sort
            assert(m_solidVoxels.size() <= MaxMesherBatchSize);
            auto solidVoxelCount = std::min(m_solidVoxels.size(), MaxMesherBatchSize);

            if (m_dSolidVoxels.size() < solidVoxelCount)
                m_dSolidVoxels.resize(solidVoxelCount);

            ThrowIfFailed(cudaMemcpy(m_dSolidVoxels.data().get(), m_solidVoxels.data(), solidVoxelCount * sizeof(Voxel), cudaMemcpyHostToDevice));

            // Get unique solid voxels
            thrust::stable_sort(m_dSolidVoxels.begin(), m_dSolidVoxels.begin() + solidVoxelCount);
            auto newEnd = thrust::unique(m_dSolidVoxels.begin(), m_dSolidVoxels.begin() + solidVoxelCount);

            // Copy data back to cpu
            solidVoxelCount = newEnd - m_dSolidVoxels.begin();

            m_solidVoxels.resize(solidVoxelCount);
            ThrowIfFailed(cudaMemcpy(m_solidVoxels.data(), m_dSolidVoxels.data().get(), solidVoxelCount * sizeof(Voxel), cudaMemcpyDeviceToHost));
        }
    }

    void CudaMesher::GenerateVoxelFaceGeometry(int depth)
    {
        auto voxelCount = m_solidVoxels.size();
        assert(voxelCount > 0);
        assert(voxelCount == m_dAdjacentVoxels.size());

        // Read total number of faces
        auto faceCount = thrust::transform_reduce(m_dAdjacentVoxels.begin(), m_dAdjacentVoxels.begin() + voxelCount, FaceCountConversion(), 0, thrust::plus<int>());

        // Get starting indices for voxel face geometry
        device_vector<int> voxelFaceStartIndices(faceCount);

        thrust::transform(m_dAdjacentVoxels.begin(), m_dAdjacentVoxels.end(), voxelFaceStartIndices.begin(), FaceCountConversion());
        thrust::exclusive_scan(voxelFaceStartIndices.begin(), voxelFaceStartIndices.begin() + voxelCount, voxelFaceStartIndices.begin(), 0, thrust::plus<int>());

        // Allocate memory for output geometry
        device_vector<vec3> dOrigVertices(faceCount * 4);

        // Triangulate faces!
        auto  groupCount = (int)(NextMultipleOf<uint64_t, WorkGroupSize>(voxelCount) / WorkGroupSize);
        auto& bounds     = m_tree.GetBounds();
        auto  voxelSize  = bounds.Size().x / (1 << depth);

        kernelTriangulate<<<groupCount, WorkGroupSize>>>(
            m_dSolidVoxels.data().get(),
            m_dAdjacentVoxels.data().get(),
            voxelFaceStartIndices.data().get(),
            (int)voxelCount,
            m_tree.GetBounds(),
            voxelSize,
            dOrigVertices.data().get());

        ThrowIfFailed(cudaGetLastError());
        ThrowIfFailed(cudaDeviceSynchronize());

        // Remove duplicate vertices
        m_dVertices = dOrigVertices;

        // Gather duplicate vertices together
        thrust::sort(m_dVertices.begin(), m_dVertices.end(), Vec3Less());

        // Remove duplicates
        auto newEnd = thrust::unique(m_dVertices.begin(), m_dVertices.end(), Vec3Equals());
        m_dVertices.erase(newEnd, m_dVertices.end());

        // Create mappings to this new list
        device_vector<int> dIndexMapping(dOrigVertices.size());

        thrust::lower_bound(
            thrust::device,
            m_dVertices.begin(), m_dVertices.end(),
            dOrigVertices.begin(), dOrigVertices.end(),
            dIndexMapping.begin(),
            Vec3Less());

        // Create triangle indices
        device_vector<int> dIndices(faceCount * 6);

        groupCount = NextMultipleOf<int, WorkGroupSize>(faceCount) / WorkGroupSize;

        kernelCreateTriangleIndices<<<groupCount, WorkGroupSize>>>(
            dIndexMapping.data().get(),
            faceCount,
            dIndices.data().get());

        // Copy indices back to ram. Vertices are still being modified!
        m_outIndices.resize(dIndices.size());

        ThrowIfFailed(cudaMemcpy(m_outIndices.data(), dIndices.data().get(), dIndices.size() * sizeof(int), cudaMemcpyDeviceToHost));
    }

    void CudaMesher::MinimizePositionError()
    {
        const uint32_t cSampleCount = 256;

        auto& nodes      = m_tree.GetTreeNodes();
        auto  vCount     = (int)m_dVertices.size();
        auto  bufferSize = std::min(vCount, MaxVertexBatchSize) * cSampleCount;

        thrust::device_vector<SparseOctree::Node> dTree(nodes.size());
        thrust::device_vector<vec3>               dClosestPositions(bufferSize);
        thrust::device_vector<float>              dClosestDistances(bufferSize);

        // Resize output vertex array
        m_outVertices.resize(vCount);

        // Copy tree to vram
        thrust::copy(nodes.data(), nodes.data() + nodes.size(), dTree.begin());

        // Go through batches
        int startIdx = 0;

        while (startIdx < vCount)
        {
            // Evaluate batch size
            auto batchSize = std::min(MaxVertexBatchSize, (int)vCount - startIdx);

            const auto xDimSize = 2u;
            dim3 groupSize  = { xDimSize, cSampleCount, 1 };
            auto groupCount = NextMultipleOf<int, xDimSize>(batchSize) / xDimSize;

            // Launch kernel to compute closest distances
            kernelSampleClosestDistances<<<groupCount, groupSize>>>(
                dTree.data().get(),
                m_tree.GetBounds(),
                m_dVertices.data().get() + startIdx,
                batchSize,
                cSampleCount,
                m_voxelSize,
                m_voxelSize * 1.73f,            // Use voxel size as max sampling distance
                dClosestPositions.data().get(),
                dClosestDistances.data().get());

            ThrowIfFailed(cudaGetLastError());

            // Launch a second kernel to get minimum distance from candidates
            groupCount = NextMultipleOf<int, WorkGroupSize>(batchSize) / WorkGroupSize;

            kernelFilterClosestDistance<<<groupCount, WorkGroupSize>>>(
                dClosestPositions.data().get(),
                dClosestDistances.data().get(),
                batchSize,
                cSampleCount,
                m_dVertices.data().get() + startIdx);

            ThrowIfFailed(cudaGetLastError());
            ThrowIfFailed(cudaDeviceSynchronize());

            // Finally copy vertices back to ram
            thrust::copy(m_dVertices.begin() + startIdx, m_dVertices.begin() + startIdx + batchSize, m_outVertices.data() + startIdx);

            startIdx += batchSize;
        }
    }


    // Mesher

    Mesher::Mesher(const SparseOctree& tree, const vector<Voxel>& voxels, vector<vec3>& outVertices, vector<int>& outIndices)
        : m_tree(tree)
        , m_voxels(voxels)
        , m_outVertices(outVertices)
        , m_outIndices(outIndices)
    { }

    void Mesher::Meshify(int targetDepth)
    {
        assert(targetDepth <= 21 && targetDepth > 0);

        // Pass call forward to cuda mesher
        CudaMesher mesher(m_tree, m_voxels, m_outVertices, m_outIndices);

        mesher.Meshify(targetDepth);
    }
}