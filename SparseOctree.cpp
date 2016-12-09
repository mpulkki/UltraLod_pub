#include "SparseOctree.hpp"

#include <bitset>

using namespace std;

#define NullOffset ((uint32_t)-1)

namespace UltraLod
{
    SparseOctree::SparseOctree(int depth, const Aabb& bounds)
        : m_depth(depth)
        , m_bounds(bounds)
    {
        // Add empty root node
        m_nodes.push_back({ 0, 0, 0 });
    }

    void SparseOctree::AddVoxel(const Voxel& voxel, const ColorRGB24& color, int nodeIdx, int depth)
    {
        auto morton = voxel & ((1ULL << 63) - 1);
        int  bitPos = (m_depth - depth - 1) * 3;

        // Get coordinates at this level
        auto childIdx = (uint8_t)((voxel >> bitPos) & 0x7);
        auto node = m_nodes[nodeIdx];

        // Is next leaf node?
        if (depth == m_depth - 1)
        {
            auto firstChild = node.leafMask == 0;

            node.validMask |= 1 << childIdx;
            node.leafMask  |= 1 << childIdx;

            // Store color information
            if (!node.childOffset && firstChild)
                node.childOffset = (int)m_colors.size();

            m_colors.push_back(color);

            m_nodes[nodeIdx] = node;
        }
        else
        {
            // Create children?
            if (!node.childOffset)
            {
                auto newIdx = m_nodes.size();

                // Allocate space for children
                m_nodes.resize(m_nodes.size() + 8);
                memset(m_nodes.data() + newIdx, 0, sizeof(Node) * 8);

                assert(newIdx - nodeIdx > 0);
                node.childOffset = (uint32_t)(newIdx - nodeIdx);
            }

            node.validMask |= 1 << childIdx;

            m_nodes[nodeIdx] = node;

            // Continue recursion
            auto childPos = nodeIdx + node.childOffset + childIdx;

            AddVoxel(voxel, color, childPos, depth + 1);
        }
    }

    void SparseOctree::AddVoxels(const vector<Voxel>& voxels, const vector<ColorRGB24>& colors)
    {
        assert(voxels.size() == colors.size());

        Voxel prevVoxel = 0;

        // Allocate memory for colors. Node count is not known yet
        m_colors.reserve(colors.size());
        m_nodes.reserve(voxels.size());

        // First build the tree
        for (auto i = (size_t)0; i < voxels.size(); i++)
        {
            // Validate that voxels are sorted
            if (voxels[i] < prevVoxel)
                throw std::exception();

            AddVoxel(voxels[i], colors[i], 0, 0);
            prevVoxel = voxels[i];
        }
    }

    const Aabb& SparseOctree::GetBounds() const
    {
        return m_bounds;
    }

    int SparseOctree::GetDepth() const
    {
        return m_depth;
    }

    const vector<ColorRGB24>& SparseOctree::GetTreeColors() const
    {
        return m_colors;
    }

    const vector<SparseOctree::Node>& SparseOctree::GetTreeNodes() const
    {
        return m_nodes;
    }

    void SparseOctree::Validate() const
    {
        if (m_nodes.size() == 0)
            return;

        Validate(0);
    }

    void SparseOctree::Validate(int idx) const
    {
        if (idx >= m_nodes.size())
            throw std::exception();

        auto& node = m_nodes[idx];

        if (((node.validMask ^ node.leafMask) & node.leafMask) != 0)
            throw std::exception();

        if (node.validMask)
        {
            if (node.childOffset < 0)
                throw std::exception();

            for (int i = 0; i < 8; i++)
            {
                if (node.validMask & (1 << i) && !(node.leafMask & (1 << i)))
                {
                    Validate(idx + node.childOffset + i);
                }
            }
        }
    }
}
/*
#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

class Camera
{
public:
    Camera();


private:
    glm::vec4 m_orientation;
    glm::vec3 m_position;
    float     m_fov;
    float     m_width;
    float     m_height;
};

#include "Camera.cuh"
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace glm;

Camera::Camera()
    : m_position(0, 0, 0)
    , m_orientation(0, 0, 0, 1)
{


    //glm::perspectiveFov
}
*/