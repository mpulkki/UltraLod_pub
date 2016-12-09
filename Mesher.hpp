#pragma once

#include "Geometry.hpp"

namespace UltraLod
{
    class SparseOctree;

    // Class for creating a mesh from octree
    class Mesher
    {
    public:
        Mesher(const SparseOctree& tree, const std::vector<Voxel>& voxels, std::vector<glm::vec3>& outVertices, std::vector<int>& outIndices);

        // Creates a mesh from the octree
        void Meshify(int targetDepth);

    private:
        const SparseOctree&       m_tree;
        const std::vector<Voxel>& m_voxels;
        std::vector<glm::vec3>&   m_outVertices;
        std::vector<int>&         m_outIndices;
    };
}