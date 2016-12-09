#pragma once

#include <memory>
#include "Geometry.hpp"

namespace UltraLod
{
    class ComputationParams;

    class Voxelizer
    {
    public:
        struct Stats
        {
            uint64_t splitCount;
            uint64_t voxelizationCount;
            uint64_t triAabbTestCount;
            double   initTime;

            // Split timings
            double splitTime;
            double splitPartitionTime;

            // Voxelization timing
            double totalTime;
            double voxelizationTime;
            double voxelizationKernelTime;
            double mappingKernelTime;
            double exclusiveScanTime;
            double voxelPartitionTime;
        };

    public:
        Voxelizer(const ComputationParams& params, std::vector<Voxel>& voxels, std::vector<ColorRGB24>& colors);
        ~Voxelizer();

        // Voxelises the given mesh
        void Voxelize(const Mesh& mesh);

    public:

        // Returns stats from previous voxelization
        const Stats& GetStats() const;

    private:
        std::unique_ptr<struct VoxelizerImpl> m_impl;
    };
}