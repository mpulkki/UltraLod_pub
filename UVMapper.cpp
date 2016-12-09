#include "UVMapper.hpp"

#include "uvatlas\UVAtlasMesh.h"
#include "uvatlas\UVAtlasMapper.h"
#include "UVAtlas\UVAtlasPacker.h"

using namespace std;
using namespace glm;

namespace UltraLod
{
    // Private implementation

    class UVMapperImpl
    {
    public:
        UVMapperImpl(vector<vec3>& positions, vector<int>& indices, vector<vec2>& uvs);

        ivec2 CreateChart();

    private:
        vector<vec3>& m_positions;
        vector<int>&  m_indices;
        vector<vec2>& m_uvs;
    };


    UVMapperImpl::UVMapperImpl(vector<vec3>& positions, vector<int>& indices, vector<vec2>& uvs)
        : m_positions(positions)
        , m_indices(indices)
        , m_uvs(uvs)
    { }

    ivec2 UVMapperImpl::CreateChart()
    {
        auto atlasMesh = UVAtlas::Mesh((int)m_positions.size(), (int)m_indices.size() / 3);

        // Copy mesh data
        for (auto i = 0u; i < m_positions.size(); i++)
        {
            atlasMesh.m_vertices[i].pos    = (const Umbra::Vector3&)m_positions[i];
            atlasMesh.m_vertices[i].origId = (int)i;
        }

        for (auto i = 0u; i < m_indices.size() / 3; i++)
        {
            atlasMesh.m_triangles[i].v[0]   = m_indices[i * 3 + 0];
            atlasMesh.m_triangles[i].v[1]   = m_indices[i * 3 + 1];
            atlasMesh.m_triangles[i].v[2]   = m_indices[i * 3 + 2];
            atlasMesh.m_triangles[i].origId = (int)i;
        }

        // Generate mesh
        auto mappedMesh = UVAtlas::Mapper::map(atlasMesh, 32.0f);
        assert(mappedMesh);

        if (!mappedMesh)
            return { 0, 0 };

        // Generate uvs
        UVAtlas::Packer packer(*mappedMesh, 4, 4, false);

        // Copy results back to the mesh
        auto vCount = mappedMesh->m_numVertices;
        auto tCount = mappedMesh->m_numTriangles;

        m_positions.resize(vCount);
        m_indices.resize(tCount * 3);
        m_uvs.resize(vCount);

        for (int i = 0; i < vCount; i++)
        {
            m_positions[i] = (const vec3&)mappedMesh->m_vertices[i].pos;
            m_uvs[i]       = (const vec2&)mappedMesh->m_vertices[i].uv;
        }

        for (int i = 0; i < tCount; i++)
        {
            m_indices[i * 3 + 0] = mappedMesh->m_triangles[i].v[0];
            m_indices[i * 3 + 1] = mappedMesh->m_triangles[i].v[1];
            m_indices[i * 3 + 2] = mappedMesh->m_triangles[i].v[2];
        }

        return { packer.width(), packer.height() };
    }



    // Public interface

    UVMapper::UVMapper(vector<vec3>& positions, vector<int>& indices, vector<vec2>& uvs)
        : m_positions(positions)
        , m_indices(indices)
        , m_uvs(uvs)
    { }

    ivec2 UVMapper::CreateChart()
    {
        UVMapperImpl impl(m_positions, m_indices, m_uvs);

        // Generate the chart
        return impl.CreateChart();
    }
}