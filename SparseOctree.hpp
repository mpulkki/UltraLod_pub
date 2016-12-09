#pragma once

#include "Geometry.hpp"

namespace UltraLod
{
    class SparseOctree
    {
    public:
        struct Node
        {
            uint32_t childOffset;
            uint8_t  validMask;
            uint8_t  leafMask;
        };

        static_assert(sizeof(Node) == 8, "Unexpected node size");

    public:
        SparseOctree(int depth, const Aabb& bounds);

        // Constructs this tree from a list of voxels.
        void AddVoxels(const std::vector<Voxel>& voxels, const std::vector<ColorRGB24>& colors);

        // Returns bounds of the tree
        const Aabb& GetBounds() const;

        // Depth of the tree
        int GetDepth() const;

        // Returns colors of the tree
        const std::vector<ColorRGB24>& GetTreeColors() const;

        // Returns nodes of the tree
        const std::vector<Node>& GetTreeNodes() const;

        // Validates that all pointers are valid
        void Validate() const;

    private:
        //void AddColor(const Voxel& voxel, const ColorRGB24& color, int nodeIdx, int depth);
        void AddVoxel(const Voxel& voxel, const ColorRGB24& color, int nodeIdx, int depth);
        void Validate(int idx) const;

    private:
        int                     m_depth;
        Aabb                    m_bounds;
        std::vector<Node>       m_nodes;
        std::vector<ColorRGB24> m_colors;
    };
}