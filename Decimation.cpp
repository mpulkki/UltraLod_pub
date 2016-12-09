#include "Decimation.hpp"
#include "Heap.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_access.hpp>

#include <assert.h>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <numeric>
#include <algorithm>

using namespace glm;
using namespace std;


template<>
struct hash<vec3> : public _Bitwise_hash<vec3>
{
};

namespace UltraLod
{
    struct Triangle
    {
        int v0, v1, v2;
        vec4 plane;
    };

    struct Contraction
    {
        int    v;      // Index of other vertex
        double error;
    };

    class MultimapExt
    {
    public:

        static bool Contains(const multimap<int, int>& map, int key, int value)
        {
            auto mIt = map.equal_range(key);

            for (auto it = mIt.first; it != mIt.second; it++)
                if (it->second == value)
                    return true;

            return false;
        }

        static void RemoveValue(multimap<int, int>& map, int key, int value)
        {
            auto mIt = map.equal_range(key);

            // Check if vertices are neighbours
            for (auto it = mIt.first; it != mIt.second;)
            {
                if (it->second == value)
                    it = map.erase(it);
                else
                    it++;
            }
        }
    };


    // Private implementation

    class DecimationImpl
    {
    public:
        DecimationImpl(vector<vec3>& positions, vector<int>& indices, const vector<uint8_t>& vertexLayers);

        void BuildResult();

        void Simplify();

    private:
        void Initialize(const vector<vec3>& positions, const vector<int>& indices);

        bool   AboutToFlip(int v0Idx, int v1Idx, const vec3& newPos) const;
        bool   AreNeighbours(int vIdx0, int vIdx1) const;
        void   ComputeError(int vIdx);
        void   EvalContractionCost(int vIdx);
        double EvalEdgeContractionCost(int v0, int v1, const vec3& pos);
        void   EvalPlaneConstants(Triangle& tri);
        void   GetNeighbours(int vIdx, set<int>& vertices) const;
        void   FilterTriangles(int vIdx, set<int>& triList) const;
        void   GetTriangles(int vIdx, set<int>& triList) const;
        void   RemoveFace(int vIdx, int triIdx);
        void   UpdateNeighbours(int vIdx, int oldVIdx, int newVIdx);
        bool   UpdateTriangleVertex(int tIdx, int oldVIdx, int newVIdx);

    private:
        vector<vec3>&          m_origPositions;
        vector<int>&           m_origIndices;
        const vector<uint8_t>& m_vertexLayers;

        //list<int> m_decimatedVertices;      // Vertex indices
        MinHeapHash<double> m_decimationHeap;

        vector<dmat4x4>     m_errors;
        vector<Contraction> m_contractionCosts;
        vector<vec3>        m_positions;
        multimap<int, int>  m_vertTriMap;    // For each vertex, all triangles
        vector<Triangle>    m_triangles;
        multimap<int, int>  m_neighbours;    // For each vertex, all connected vertices (ie. edges)
    };


    DecimationImpl::DecimationImpl(vector<vec3>& positions, vector<int>& indices, const vector<uint8_t>& vertexLayers)
        : m_origPositions(positions)
        , m_origIndices(indices)
        , m_vertexLayers(vertexLayers)
    {
        Initialize(positions, indices);
    }

    bool DecimationImpl::AboutToFlip(int v0Idx, int v1Idx, const vec3& newPos) const
    {
        set<int> tris;

        GetTriangles(v0Idx, tris);
        GetTriangles(v1Idx, tris);

        for (auto tIdx : tris)
        {
            // Get old normal
            auto& tri        = m_triangles[tIdx];
            auto  prevNormal = vec3(tri.plane.x, tri.plane.y, tri.plane.z);

            // Compute what would be the new normal
            auto v0 = m_positions[tri.v0];
            auto v1 = m_positions[tri.v1];
            auto v2 = m_positions[tri.v2];

            if (tri.v0 == v0Idx || tri.v0 == v1Idx) v0 = newPos;
            if (tri.v1 == v0Idx || tri.v1 == v1Idx) v1 = newPos;
            if (tri.v2 == v0Idx || tri.v2 == v1Idx) v2 = newPos;

            auto newNormal = cross(v1 - v0, v2 - v0);

            // Flip if signs are different
            if (dot(prevNormal, newNormal) < 0.0f)
                return true;
        }

        return false;
    }

    bool DecimationImpl::AreNeighbours(int vIdx0, int vIdx1) const
    {
        return MultimapExt::Contains(m_neighbours, vIdx0, vIdx1);
    }

    void DecimationImpl::BuildResult()
    {
        m_origPositions.clear();
        m_origIndices.clear();

        unordered_map<vec3, int> vertices;
        set<int> triangles;

        // Get unique set of triangles
        while (m_decimationHeap.Size())
        {
            auto vIdx = m_decimationHeap.Pop().hash;
            auto tIt  = m_vertTriMap.equal_range(vIdx);

            for (auto it = tIt.first; it != tIt.second; it++)
                triangles.insert(it->second);
        }

        // Generate triangle mesh
        for (auto tIdx : triangles)
        {
            auto& tri = m_triangles[tIdx];
            auto& v0  = m_positions[tri.v0];
            auto& v1  = m_positions[tri.v1];
            auto& v2  = m_positions[tri.v2];

            int v0Idx, v1Idx, v2Idx;

            auto it = vertices.find(v0);
            if (it != vertices.end())
                v0Idx = it->second;
            else
            {
                v0Idx = (int)vertices.size();
                vertices.insert({ v0, v0Idx });
                m_origPositions.push_back(v0);
            }

            m_origIndices.push_back(v0Idx);


            it = vertices.find(v1);
            if (it != vertices.end())
                v1Idx = it->second;
            else
            {
                v1Idx = (int)vertices.size();
                vertices.insert({ v1, v1Idx });
                m_origPositions.push_back(v1);
            }

            m_origIndices.push_back(v1Idx);


            it = vertices.find(v2);
            if (it != vertices.end())
                v2Idx = it->second;
            else
            {
                v2Idx = (int)vertices.size();
                vertices.insert({ v2, v2Idx });
                m_origPositions.push_back(v2);
            }

            m_origIndices.push_back(v2Idx);
        }
    }

    void DecimationImpl::ComputeError(int vIdx)
    {
        assert(vIdx >= 0 && vIdx < m_positions.size());

        // Get and reset the vertex
        auto& v = m_positions[vIdx];

        memset(&m_errors[vIdx], 0, sizeof(dmat4x4));

        // All faces contribute in error value
        auto faceIt = m_vertTriMap.equal_range(vIdx);

        for (auto it = faceIt.first; it != faceIt.second; it++)
        {
            auto& tri = m_triangles[it->second];

            // Apply plane equation to error
            auto& a = tri.plane.x;
            auto& b = tri.plane.y;
            auto& c = tri.plane.z;
            auto& d = tri.plane.w;

            dmat4x4 kp = dmat4x4();

            kp = row(kp, 0, dvec4(a*a, a*b, a*c, a*d));
            kp = row(kp, 1, dvec4(b*a, b*b, b*c, b*d));
            kp = row(kp, 2, dvec4(c*a, c*b, c*c, c*d));
            kp = row(kp, 3, dvec4(d*a, d*b, d*c, d*d));

            m_errors[vIdx] += kp;
        }
    }

    void DecimationImpl::EvalContractionCost(int vIdx)
    {
        assert(vIdx >= 0 && vIdx < m_positions.size());

        auto&      vertex      = m_positions[vIdx];
        double     minError    = DBL_MAX;
        const auto flipPenalty = 1e10;
        const auto flagPenalty = 1e10;

        // Find edge with lowest contraction cost
        auto edgeIt = m_neighbours.equal_range(vIdx);

        for (auto it = edgeIt.first; it != edgeIt.second; it++)
        {
            auto& neighbour = m_positions[it->second];

            // contraction position is always in the middle (TODO: iterative solver)
            auto position = 0.5f * (vertex + neighbour);
            auto error    = EvalEdgeContractionCost(vIdx, it->second, position);

            // Allow contraction of vertex pairs that have same flags
            if (m_vertexLayers[vIdx] != m_vertexLayers[it->second])
                error += flagPenalty;

            // Add a huge penalty factor if triangles are about to flip
            if (AboutToFlip(vIdx, it->second, position))
                error += flipPenalty;

            if (error < minError)
            {
                minError = error;
                m_contractionCosts[vIdx] = { it->second, error };
            }
        }
    }

    double DecimationImpl::EvalEdgeContractionCost(int v0, int v1, const vec3& pos)
    {
        // Get error matrices of both vertices
        auto& m0 = m_errors[v0];
        auto& m1 = m_errors[v1];

        // Use addition operator for combining both matrices
        auto errorMatrix = m0 + m1;
        auto pos4        = dvec4(pos, 1);

        // Calculate position^T * errorMatrix * position
        auto r = dvec4(
            dot(pos4, (column(errorMatrix, 0))),
            dot(pos4, (column(errorMatrix, 1))),
            dot(pos4, (column(errorMatrix, 2))),
            dot(pos4, (column(errorMatrix, 3))));

        return dot(r, pos4);
    }

    void DecimationImpl::EvalPlaneConstants(Triangle& tri)
    {
        auto& v0 = m_positions[tri.v0];
        auto& v1 = m_positions[tri.v1];
        auto& v2 = m_positions[tri.v2];

        auto a = v1 - v0;
        auto b = v2 - v0;

        // Eval plane normal
        auto normal = normalize(cross(a, b));

        // And distance using scalar projection
        auto d = -dot(normal, v0);

        tri.plane = vec4(normal.x, normal.y, normal.z, d);
    }

    void DecimationImpl::GetNeighbours(int vIdx, set<int>& vertices) const
    {
        auto vIt = m_neighbours.equal_range(vIdx);

        for (auto it = vIt.first; it != vIt.second; it++)
            vertices.insert(it->second);
    }

    void DecimationImpl::FilterTriangles(int vIdx, set<int>& triList) const
    {
        // Remove all triangles that do not have given vertex in them
        for (auto it = triList.begin(); it != triList.end(); )
        {
            auto& tri = m_triangles[*it];

            if (tri.v0 != vIdx && tri.v1 != vIdx && tri.v2 != vIdx)
                it = triList.erase(it);
            else
                it++;
        }
    }

    void DecimationImpl::GetTriangles(int vIdx, set<int>& triList) const
    {
        auto triIt = m_vertTriMap.equal_range(vIdx);

        for (auto it = triIt.first; it != triIt.second; it++)
            triList.insert(it->second);
    }

    void DecimationImpl::Initialize(const vector<vec3>& positions, const vector<int>& indices)
    {
        // Copy positions
        m_positions = positions;

        auto pCount = m_positions.size();
        m_errors.resize(pCount);
        m_contractionCosts.resize(pCount);
        m_decimationHeap.Reserve(pCount);

        // Copy indices
        m_triangles.resize(indices.size() / 3);

        for (auto i = 0u; i < indices.size() / 3; i++)
        {
            auto& i0 = indices[i * 3 + 0];
            auto& i1 = indices[i * 3 + 1];
            auto& i2 = indices[i * 3 + 2];

            m_triangles[i] = { i0, i1, i2, {0, 0, 0, 0} };
        }

        for (auto i = 0u; i < m_triangles.size(); i++)
        {
            auto& tri = m_triangles[i];

            // Populate edge information
            if (!MultimapExt::Contains(m_neighbours, tri.v0, tri.v1)) m_neighbours.insert({ tri.v0, tri.v1 });
            if (!MultimapExt::Contains(m_neighbours, tri.v0, tri.v2)) m_neighbours.insert({ tri.v0, tri.v2 });

            if (!MultimapExt::Contains(m_neighbours, tri.v1, tri.v0)) m_neighbours.insert({ tri.v1, tri.v0 });
            if (!MultimapExt::Contains(m_neighbours, tri.v1, tri.v2)) m_neighbours.insert({ tri.v1, tri.v2 });

            if (!MultimapExt::Contains(m_neighbours, tri.v2, tri.v0)) m_neighbours.insert({ tri.v2, tri.v0 });
            if (!MultimapExt::Contains(m_neighbours, tri.v2, tri.v1)) m_neighbours.insert({ tri.v2, tri.v1 });

            // Populate face information
            m_vertTriMap.insert({ tri.v0, i });
            m_vertTriMap.insert({ tri.v1, i });
            m_vertTriMap.insert({ tri.v2, i });

            // Evaluate plane constants
            EvalPlaneConstants(tri);
        }

        // Compute errors
        for (auto i = 0u; i < pCount; i++)
            ComputeError(i);

        for (auto i = 0u; i < pCount; i++)
            EvalContractionCost(i);

        // Populate work queue
        for (auto i = 0u; i < pCount; i++)
            m_decimationHeap.Push(m_contractionCosts[i].error, (int)i);
    }

    void DecimationImpl::Simplify()
    {
        set<int> updatedTriangles;
        set<int> removedTriangles;
        set<int> updatedVertices;
        set<int> neighbourVertices;

        const auto maxAcceptedErrorTreshold = 0.01;

        while (m_decimationHeap.Size() > 3)
        {
            // Get next edge to contract. Lowest impact contractions on the back of queue
            auto  item = m_decimationHeap.Pop();
            auto& vIdx = item.hash;
            auto& pair = m_contractionCosts[vIdx];
            assert(item.value == pair.error);

            // Check that contraction cost is within given limits
            if (pair.error > maxAcceptedErrorTreshold)
                break;

            // Reuse one vertex and remove the other (must be this way because pair.v is still in the work queue!)
            auto vertexToRemove = vIdx;
            auto vertexToSpare  = pair.v;

            // Update position
            m_positions[vertexToSpare] = 0.5f * (m_positions[vertexToSpare] + m_positions[vertexToRemove]);

            // Find triangles and vertices that are affected by the contraction operation. Then remove any references to removed items
            updatedTriangles.clear();

            GetTriangles(vertexToRemove, updatedTriangles);
            GetTriangles(vertexToSpare,  updatedTriangles);

            // Get triangles shared by both vertices
            removedTriangles = updatedTriangles;

            FilterTriangles(vertexToRemove, removedTriangles);
            FilterTriangles(vertexToSpare,  removedTriangles);

            // Update triangles to point to the new vertex instead
            for (auto tIdx : updatedTriangles)
            {
                if (UpdateTriangleVertex(tIdx, vertexToRemove, vertexToSpare))
                    m_vertTriMap.insert({ vertexToSpare, tIdx });
            }

            updatedVertices.clear();

            // Update vertices
            {
                GetNeighbours(vertexToSpare,  updatedVertices);
                GetNeighbours(vertexToRemove, updatedVertices);

                for (auto vIdx : updatedVertices)
                {
                    if (vIdx == vertexToRemove)
                        continue;

                    UpdateNeighbours(vIdx, vertexToRemove, vertexToSpare);

                    // Remove all references to deleted triangles
                    for (auto tIdx : removedTriangles)
                        RemoveFace(vIdx, tIdx);
                }
            }

            // Remove mappings from dead items
            m_neighbours.erase(vertexToRemove);
            m_vertTriMap.erase(vertexToRemove);

            // Update plane equations and error matrices of changed triangles and vertices
            for (auto tIdx : updatedTriangles)
                EvalPlaneConstants(m_triangles[tIdx]);

            for (auto vIdx : updatedVertices)
                ComputeError(vIdx);

            // Update contraction costs of all edges whose error matrices were updated
            neighbourVertices.clear();

            for (auto vIdx : updatedVertices)
            {
                neighbourVertices.insert(vIdx);
                GetNeighbours(vIdx, neighbourVertices);
            }

            for (auto vIdx : neighbourVertices)
            {
                // Remove from heap for update
                m_decimationHeap.Pop(vIdx);

                EvalContractionCost(vIdx);

                // Reinsert updated vertex
                m_decimationHeap.Push(m_contractionCosts[vIdx].error, vIdx);
            }
        }
    }

    void DecimationImpl::RemoveFace(int vIdx, int triIdx)
    {
        MultimapExt::RemoveValue(m_vertTriMap, vIdx, triIdx);
    }

    void DecimationImpl::UpdateNeighbours(int vIdx, int oldVIdx, int newVIdx)
    {
        if (AreNeighbours(vIdx, oldVIdx))
        {
            assert(AreNeighbours(oldVIdx, vIdx));
            MultimapExt::RemoveValue(m_neighbours, vIdx, oldVIdx);
            MultimapExt::RemoveValue(m_neighbours, oldVIdx, vIdx);
        }

        if (vIdx != newVIdx && !AreNeighbours(vIdx, newVIdx))
        {
            assert(!AreNeighbours(newVIdx, vIdx));
            m_neighbours.insert({ vIdx, newVIdx });
            m_neighbours.insert({ newVIdx, vIdx });
        }
    }

    bool DecimationImpl::UpdateTriangleVertex(int tIdx, int oldVIdx, int newVIdx)
    {
        auto& tri = m_triangles[tIdx];
        bool updated = false;

        if (tri.v0 == oldVIdx)
        {
            tri.v0 = newVIdx;
            updated = true;
        }

        if (tri.v1 == oldVIdx)
        {
            tri.v1 = newVIdx;
            updated = true;
        }

        if (tri.v2 == oldVIdx)
        {
            tri.v2 = newVIdx;
            updated = true;
        }

        return updated;
    }


    // Public api

    Decimation::Decimation(vector<vec3>& positions, vector<int>& indices)
        : m_positions(positions)
        , m_indices(indices)
    {
        m_vertexLayers.resize(m_positions.size());

        // Every position belongs to layer 0 by default
        std::fill(m_vertexLayers.begin(), m_vertexLayers.end(), 0);
    }

    void Decimation::SetVertexLayer(int idx, uint8_t layer)
    {
        assert(idx >= 0 && idx < m_vertexLayers.size());
        m_vertexLayers[idx] = layer;
    }

    void Decimation::Simplify()
    {
        DecimationImpl decimator(m_positions, m_indices, m_vertexLayers);

        // Execute decimation
        decimator.Simplify();

        // Get results to original mesh
        decimator.BuildResult();
    }
}