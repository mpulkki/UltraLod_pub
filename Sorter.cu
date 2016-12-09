#include "Sorter.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

using namespace std;

namespace UltraLod
{
    Sorter::Sorter(vector<Voxel>& voxels, vector<ColorRGB24>& colors)
        : m_voxels(voxels)
        , m_colors(colors)
    { }

    void Sorter::Sort()
    {
        thrust::device_vector<Voxel>      dVoxels(m_voxels.size());
        thrust::device_vector<ColorRGB24> dColors(m_colors.size());

        // Copy data to gpu
        thrust::copy(m_voxels.data(), m_voxels.data() + m_voxels.size(), dVoxels.begin());
        thrust::copy(m_colors.data(), m_colors.data() + m_colors.size(), dColors.begin());

        // Execute sorting!
        thrust::sort_by_key(dVoxels.begin(), dVoxels.end(), dColors.begin());

        // Copy data back
        thrust::copy(dVoxels.begin(), dVoxels.end(), m_voxels.data());
        thrust::copy(dColors.begin(), dColors.end(), m_colors.data());
    }
}