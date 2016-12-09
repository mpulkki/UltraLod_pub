#pragma once

#include <vector>
#include <glm/vec2.hpp>

namespace UltraLod
{
    class RenderTarget;
    class SparseOctree;

    class MaterialSampler
    {
    public:
        MaterialSampler(
            const SparseOctree& octree,
            RenderTarget& renderTarget,
            const std::vector<glm::vec3>& positions,
            const std::vector<glm::vec2>& uvs,
            const std::vector<int>& indices);

        // Executes material sampling
        void Sample(float voxelSize);

    private:
        const SparseOctree&           m_octree;
        RenderTarget&                 m_renderTarget;
        const std::vector<glm::vec3>& m_positions;
        const std::vector<glm::vec2>& m_uvs;
        const std::vector<int>&       m_indices;
    };
}