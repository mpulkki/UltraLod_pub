#pragma once

#include <vector>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace UltraLod
{
    class UVMapper
    {
    public:
        UVMapper(std::vector<glm::vec3>& positions, std::vector<int>& indices, std::vector<glm::vec2>& uvs);

        // Generates texture atlas for the given mesh. Note: mesh is modified! Returns texture size
        glm::ivec2 CreateChart();

    private:
        std::vector<glm::vec3>& m_positions;
        std::vector<glm::vec2>& m_uvs;
        std::vector<int>&       m_indices;
    };
}