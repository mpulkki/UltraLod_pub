#pragma once

#include "Geometry.hpp"

namespace UltraLod
{
    // Describes aligned and formatted computation bounds and other parameters
    class ComputationParams
    {
    public:
        ComputationParams();

        const Aabb& GetBounds()    const;
        int         GetDepth()     const;
        float       GetVoxelSize() const;

        void SetBounds(const Aabb& bounds);
        void SetDepth(int depth);

    private:
        Aabb  m_bounds;
        int   m_depth;
        float m_voxelSize;
    };
}