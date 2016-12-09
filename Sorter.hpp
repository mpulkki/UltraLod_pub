#pragma once

#include "Geometry.hpp"

namespace UltraLod
{
    class Sorter
    {
    public:
        Sorter(std::vector<Voxel>& voxels, std::vector<ColorRGB24>& colors);

        void Sort();

    private:
        std::vector<Voxel>&      m_voxels;
        std::vector<ColorRGB24>& m_colors;
    };
}