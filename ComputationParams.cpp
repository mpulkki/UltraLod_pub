#include "ComputationParams.hpp"

#include <glm/vec3.hpp>

using namespace glm;

namespace UltraLod
{
    ComputationParams::ComputationParams()
        : m_depth(0)
        , m_voxelSize(0.0f)
    { }

    const Aabb& ComputationParams::GetBounds() const
    {
        return m_bounds;
    }

    int ComputationParams::GetDepth() const
    {
        return m_depth;
    }

    float ComputationParams::GetVoxelSize() const
    {
        return m_voxelSize;
    }

    void ComputationParams::SetBounds(const Aabb& bounds)
    {
        auto center      = 0.5f * (bounds.min + bounds.max);
        auto longestSide = max(bounds.max - bounds.min);
        auto newExtents  = vec3(1, 1, 1) * 0.5f * longestSide;

        m_bounds.min = center - newExtents;
        m_bounds.max = center + newExtents;

        if (m_depth > 0)
            m_voxelSize = m_bounds.Size().x / (1 << m_depth);
    }

    void ComputationParams::SetDepth(int depth)
    {
        m_depth = depth;
        m_voxelSize = m_bounds.Size().x / (1 << m_depth);
    }
}