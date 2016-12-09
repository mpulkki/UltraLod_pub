#pragma once

#include <vector>
#include <glm/vec3.hpp>

namespace UltraLod
{
    class Decimation
    {
    public:
        Decimation(std::vector<glm::vec3>& positions, std::vector<int>& indices);

        void SetVertexLayer(int idx, uint8_t layer);
        void Simplify();

    private:
        std::vector<glm::vec3>& m_positions;
        std::vector<int>&       m_indices;
        std::vector<uint8_t>    m_vertexLayers;
    };
}