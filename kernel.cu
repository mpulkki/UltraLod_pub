#include "kernel.cuh"
#include <cuda_runtime.h>
#include <stdlib.h>

using namespace glm;

__constant__ float cTileBounds[6];
__constant__ float cVoxelSize;

//uint3 __device_builtin__ __STORAGE__ threadIdx;
//uint3 __device_builtin__ __STORAGE__ blockIdx;
//dim3 __device_builtin__ __STORAGE__ blockDim;
//dim3 __device_builtin__ __STORAGE__ gridDim;
//int __device_builtin__ __STORAGE__ warpSize;

inline __device__ __host__ Aabb triangleBounds(const glm::vec3* tri)
{
    glm::vec3 bMin = min(min(tri[0], tri[1]), tri[2]);
    glm::vec3 bMax = max(max(tri[0], tri[1]), tri[2]);
    return { bMin, bMax };
}

__global__ void kernelComputeVoxelCoverage(const vec3* inTriangles, int* outCoverage)
{
    // Compute triangle index
    int kernelIdx     = blockIdx.x * blockDim.x + threadIdx.x;
    int triangleIndex = kernelIdx * 3;
    int resultIndex   = kernelIdx;

    // Compute bounds for voxel
    Aabb triBounds = triangleBounds(&inTriangles[triangleIndex]);

    // Clamp to tile volume
    const vec3& tileMin = (const vec3&)cTileBounds[0];
    const vec3& tileMax = (const vec3&)cTileBounds[3];

    triBounds.min = max(triBounds.min, tileMin) - tileMin;
    triBounds.max = min(triBounds.max, tileMax) - tileMin;

    // Get volume from bounds
    ivec3 iMin = toIVec3(triBounds.min, cVoxelSize);
    ivec3 iMax = toIVec3(triBounds.max, cVoxelSize);

    ivec3 size = iMax - iMin + ivec3(1, 1, 1);

    // Store result
    outCoverage[resultIndex] = size.x * size.y * size.z;
}


void computeVoxelCoverage(const CoverageInput& input)
{
    // Upload constants
    if (cudaMemcpyToSymbol(cTileBounds, &input.bounds, sizeof(Aabb)) != cudaSuccess)
        return;
        //return;
    if (cudaMemcpyToSymbol(cVoxelSize, &input.voxelSize, sizeof(float)) != cudaSuccess)
        return;
        //return;

    // Launch the kernel!
    kernelComputeVoxelCoverage<<<input.groupCount, input.groupSize>>>(input.dPositions, input.dCoverages);

    // Wait to finish
    cudaDeviceSynchronize();
}