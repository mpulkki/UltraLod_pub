#include <cuda_runtime.h>
#include <stdlib.h>

#include "Geometry.hpp"
#include "kernel.cuh"
#include "Utility.cuh"
#include "SparseOctree.hpp"
#include "Voxelizer.hpp"
#include "RayTracer.cuh"
#include "Mesher.hpp"
#include "Decimation.hpp"
#include "UVMapper.hpp"
#include "MaterialSampler.cuh"
#include "ComputationParams.hpp"
#include "Timer.hpp"
#include "Sorter.hpp"

#define STAT(expression) expression

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;
using namespace glm;
using namespace UltraLod;

using MeshVector = vector<unique_ptr<Mesh>>;

template <typename T>
T Read(ifstream& stream)
{
    T value;
    stream.read((char*)&value, sizeof(T));
    return value;
}

template <typename T>
vector<T> Read(ifstream& stream, int count)
{
    vector<T> data;
    data.resize(count);
    stream.read((char*)data.data(), sizeof(T) * count);
    return data;
}

template <typename T>
void Write(ofstream& file, const T& value)
{
    file.write((const char*)&value, sizeof(T));
}

template <typename T>
void Write(ofstream& file, const vector<T>& vector)
{
    file.write((const char*)vector.data(), vector.size() * sizeof(T));
}

template <typename T>
void Write(ofstream& file, const T* data, int count)
{
    file.write((const char*)data, count * sizeof(T));
}

Aabb ComputeBounds(const MeshVector& meshes)
{
    Aabb aabb = { vec3(FLT_MAX,FLT_MAX,FLT_MAX), vec3(-FLT_MAX,-FLT_MAX,-FLT_MAX) };

    for (auto& mesh : meshes)
    {
        for (int i = 0; i < mesh->positions.size(); i++)
        {
            auto& p = mesh->positions[i];

            aabb.min.x = __min(aabb.min.x, p.x);
            aabb.min.y = __min(aabb.min.y, p.y);
            aabb.min.z = __min(aabb.min.z, p.z);

            aabb.max.x = __max(aabb.max.x, p.x);
            aabb.max.y = __max(aabb.max.y, p.y);
            aabb.max.z = __max(aabb.max.z, p.z);
        }
    }

    return aabb;
}

Aabb GenerateTestCube(MeshVector& meshes)
{
    auto min = vec3(-0.5f, -0.5f, -0.5f);
    auto max = vec3( 0.5f,  0.5f,  0.5f);

    Aabb bounds = { vec3(-3, -3, -3), vec3(3, 3, 3) };

    // Create simple 3d cube
    vec3 vertexArray[8] =
    {
        {min.x, min.y, min.z},
        {max.x, min.y, min.z},
        {max.x, min.y, max.z},
        {min.x, min.y, max.z},

        {min.x, max.y, min.z},
        {max.x, max.y, min.z},
        {max.x, max.y, max.z},
        {min.x, max.y, max.z}
    };

    auto mesh = make_unique<Mesh>();

    auto addFace = [&](int i0, int i1, int i2, int i3)
    {
        mesh->positions.push_back(vertexArray[i0]);
        mesh->positions.push_back(vertexArray[i1]);
        mesh->positions.push_back(vertexArray[i2]);

        mesh->positions.push_back(vertexArray[i2]);
        mesh->positions.push_back(vertexArray[i3]);
        mesh->positions.push_back(vertexArray[i0]);
    };

    addFace(0, 1, 5, 4);
    addFace(1, 2, 6, 5);
    addFace(2, 3, 7, 6);
    addFace(3, 0, 4, 7);
    addFace(4, 5, 6, 7);
    addFace(3, 2, 1, 0);

    for (int i = 0; i < 6; i++)
    {
        mesh->uvs.push_back({ 0, 0 });
        mesh->uvs.push_back({ 0, 1 });
        mesh->uvs.push_back({ 1, 1 });
        mesh->uvs.push_back({ 1, 1 });
        mesh->uvs.push_back({ 1, 0 });
        mesh->uvs.push_back({ 0, 0 });
    }

    assert(mesh->positions.size() == mesh->uvs.size());

    mesh->texture.width = mesh->texture.height = 2;
    mesh->texture.data.push_back(255);
    mesh->texture.data.push_back(0);
    mesh->texture.data.push_back(0);

    mesh->texture.data.push_back(0);
    mesh->texture.data.push_back(255);
    mesh->texture.data.push_back(0);

    mesh->texture.data.push_back(0);
    mesh->texture.data.push_back(0);
    mesh->texture.data.push_back(255);

    mesh->texture.data.push_back(255);
    mesh->texture.data.push_back(0);
    mesh->texture.data.push_back(0);

    assert(mesh->texture.data.size() == mesh->texture.width * mesh->texture.height * sizeof(ColorRGB24));

    meshes.push_back(std::move(mesh));

    return bounds;
}

bool ImportMeshes(MeshVector& meshes, const string& path)
{
    ifstream file(path.c_str(), ios::binary);
    file.seekg(0);

    if (!file.is_open())
        return false;

    int meshCount = Read<int>(file);
    int materialCounter = 0;

    for (int m = 0; m < meshCount; m++)
    {
        vector<vec3> positions;
        vector<vec2> uvs;

        int vertexCount = Read<int>(file);// 0;

        positions = Read<vec3>(file, vertexCount);
        uvs = Read<vec2>(file, vertexCount);

        int subMeshCount = Read<int>(file);// 0;

        for (int i = 0; i < subMeshCount; i++)
        {
            auto mesh = unique_ptr<Mesh>(new Mesh);

            int indexCount = Read<int>(file);// 0;

            for (int j = 0; j < indexCount / 3; j++)
            {
                int i0 = Read<int>(file);
                int i1 = Read<int>(file);
                int i2 = Read<int>(file);

                mesh->positions.push_back(positions[i0]);
                mesh->positions.push_back(positions[i1]);
                mesh->positions.push_back(positions[i2]);

                mesh->uvs.push_back(uvs[i0]);
                mesh->uvs.push_back(uvs[i1]);
                mesh->uvs.push_back(uvs[i2]);
            }

            meshes.push_back(move(mesh));
        }

        int materialCount = Read<int>(file);// 0;

        for (int i = 0; i < materialCount; i++)
        {
            int width = Read<int>(file);
            int height = Read<int>(file);

            meshes[materialCounter]->texture.width = width;
            meshes[materialCounter]->texture.height = height;
            meshes[materialCounter]->texture.data = Read<uint8_t>(file, width * height * 3);

            materialCounter++;
        }
    }

    return true;
}


bool SaveMesh(const vector<vec3>& positions, const vector<int>& indices, const vector<vec2>& uvs, const RenderTarget& texture, const char* path)
{
    if (!path || !strlen(path))
        return false;

    ofstream file(path, ios::binary);

    if (file.bad())
        return false;

    // Write positions
    Write(file, (int)positions.size());
    Write(file, positions);

    // Write uvs
    Write(file, (int)uvs.size());
    Write(file, uvs);

    // Write indices
    Write(file, (int)indices.size());
    Write(file, indices);

    // Write texture
    Write(file, texture.GetWidth());
    Write(file, texture.GetHeight());
    Write(file, texture.DataPtr(), texture.GetWidth() * texture.GetHeight());

    return file.good();
}

int main(void)
{
    cout << "UltraLod v1.0 1.1.2017" << endl;
    cout << "----------------------" << endl;
    cout << endl;

    if (cudaSetDevice(0) != cudaSuccess)
    {
        cout << "Cuda device initialization failed" << endl;
        return -1;
    }

    // Request user to load input mesh
    string inputPath;

    //cout << "Input path:" << endl;
    //cin >> inputPath;                   // "F:\\Programming\\Cuda_octree\\UltraLod\\house_model.bin"

    inputPath = "F:\\Programming\\Cuda_octree\\UltraLod\\house_model.bin";

    // Load meshes
    MeshVector meshes;

#if 1
    if (!ImportMeshes(meshes, inputPath))
    {
        cout << "Failed to open input file or invalid format" << endl;
        return -1;
    }

    auto bounds = ComputeBounds(meshes);

    //int depth = 8;

#else
    auto bounds = GenerateTestCube(meshes);

    //int depth = 8;

#endif

    ComputationParams params;

    // Update computation params
    params.SetBounds(bounds);

    // Print bounds
    cout << "Computed bounds: " << endl;
    cout << " min: { " << bounds.min.x << ", " << bounds.min.y << ", " << bounds.min.z << " }" << endl;
    cout << " max: { " << bounds.max.x << ", " << bounds.max.y << ", " << bounds.max.z << " }" << endl;
    cout << " size: { " << bounds.Size().x << ", " << bounds.Size().y << ", " << bounds.Size().z << " }" << endl;
    cout << endl;

    // Prompt user to input depth values
    int depth        = 0;
    int meshifyDepth = 0;

    while (true)
    {
        cout << "Voxelization depth:" << endl;
        cin >> depth;

        if (depth <= 0 || depth > 12)
            cout << "Supported depth values are in range [1, 12]" << endl << endl;
        else
        {
            // Update params
            params.SetDepth(depth);

            cout << "Depth " << depth << " results in sampling frequency of " << params.GetVoxelSize() << endl;
            cout << "Ok? (y)es" << endl;

            char ok = '\0';
            cin >> ok;

            cout << endl;

            if (ok == 'y')
                break;
        }
    }

    // Second params for meshification
    ComputationParams meshifyParams;

    meshifyParams.SetBounds(bounds);

    while (true)
    {
        cout << "Meshification depth:" << endl;
        cin >> meshifyDepth;

        if (meshifyDepth <= 0 || meshifyDepth > depth)
            cout << "Meshification depth must be > 0 and <= depth" << endl << endl;
        else
        {
            // Update params
            meshifyParams.SetDepth(meshifyDepth);

            cout << "Meshification depth of " << meshifyDepth << " leads to voxelization result of " << meshifyParams.GetVoxelSize() << endl;
            cout << "Ok? (y)es" << endl;

            char ok = '\0';
            cin >> ok;

            cout << endl;

            if (ok == 'y')
                break;
        }
    }

    // Prompt to start!
    system("pause");

    vector<Voxel>      voxels;
    vector<ColorRGB24> colors;

    cout.precision(17);

    // Initialize timer values
    auto timeVoxelizer    = 0.0;
    auto timeSorting      = 0.0;
    auto timeTreeBuilding = 0.0;
    auto timeMesher       = 0.0;
    auto timeDecimation   = 0.0;
    auto timeUvMapping    = 0.0;
    auto timeSampling     = 0.0;
    auto timeTotal        = 0.0;

    // Extra stats
    Voxelizer::Stats stats;

    cout << "Starting voxelization..." << endl;
    {
        STAT(Timer timer);
        Voxelizer voxelizer(params, voxels, colors);

        for (int i = 0; i < (int)meshes.size(); i++)
            voxelizer.Voxelize(*meshes[i].get());

        stats = voxelizer.GetStats();
        STAT(timeVoxelizer = timer.End());
        STAT(timeTotal += timeVoxelizer);
    }
    cout << "Voxelization finished (" << (timeVoxelizer) << " s)" << endl;

    cout << "Starting voxel sorting..." << endl;
    {
        STAT(Timer timer);
        Sorter sorter(voxels, colors);
        sorter.Sort();
        //Sorting::QuickSort(voxels.data(), colors.data(), (int)voxels.size());
        STAT(timeSorting = timer.End());
        STAT(timeTotal += timeSorting);
    }
    cout << "Sorting finished (" << timeSorting << " s)" << endl;

    cout << "Starting sparse octree building..." << endl;
    SparseOctree octree(depth, params.GetBounds());
    {
        STAT(Timer timer);
        octree.AddVoxels(voxels, colors);
        STAT(timeTreeBuilding = timer.End());
        STAT(timeTotal += timeTreeBuilding);
    }
    cout << "Octree building finished (" << (timeTreeBuilding) << " s)" << endl;

    vector<vec3> positions;
    vector<int>  indices;
    vector<vec2> uvs;

    // Generate mesh from the voxel data
    cout << "Starting meshing..." << endl;
    {
        STAT(Timer timer);
        Mesher mesher(octree, voxels, positions, indices);
        mesher.Meshify(meshifyDepth);
        STAT(timeMesher = timer.End());
        STAT(timeTotal += timeMesher);
    }
    cout << "Meshing finished (" << (timeMesher) << " s)" << endl;

    // Simplify the mesh
    cout << "Starting mesh decimation..." << endl;
    {
        STAT(Timer timer);
        Decimation decimator(positions, indices);

        //// Assign test layers
        //for (auto i = 0u; i < positions.size(); i++)
        //    decimator.SetVertexLayer(i, positions[i].x <= -0.5f ? 0 : 1);

        decimator.Simplify();
        STAT(timeDecimation = timer.End());
        STAT(timeTotal += timeDecimation);
    }
    cout << "Mesh decimation finished (" << (timeDecimation) << " s)" << endl;

    ivec2 textureSize(0, 0);

    // Generate uvs
    cout << "Starting uv mapping..." << endl;
    {
        STAT(Timer timer);
        UVMapper mapper(positions, indices, uvs);
        textureSize = mapper.CreateChart();
        STAT(timeUvMapping = timer.End());
        STAT(timeTotal += timeUvMapping);
    }
    cout << "Uv mapping finished (" << (timeUvMapping) << " s)" << endl;

    RenderTarget texture(textureSize.x, textureSize.y);

    // Sample materials
    cout << "Starting material sampling..." << endl;
    {
        STAT(Timer timer);
        MaterialSampler sampler(octree, texture, positions, uvs, indices);
        sampler.Sample(meshifyParams.GetVoxelSize());
        STAT(timeSampling = timer.End());
        STAT(timeTotal += timeSampling);
    }
    cout << "Material sampling finished (" << (timeSampling) << " s)" << endl;

    // Print total time
    cout << endl << "Total time " << timeTotal << " s" << endl;

    int width = 1920, height = 1080;

    // Render to an image using raytracing
    RenderTarget buffer(width, height);
    RayTracer    rayTracer(buffer);
    Camera       camera((float)width, (float)height, 70);

    // Render floor too!
    auto floorPos = bounds.Center();
    floorPos.y = bounds.min.y;

    rayTracer.SetFloor({ 16, 16 }, floorPos);

    camera.SetPosition({ 4.6f, 3.2f, 0});
    camera.SetYaw(3.14f / 2);

    // Ray trace!
    rayTracer.Trace(octree, camera);

    // Save result to an image
    stbi_write_png("output.png", width, height, 4, buffer.DataPtr(), width * sizeof(Color));
    stbi_write_png("output_textureAtlas.png", texture.GetWidth(), texture.GetHeight(), 4, texture.DataPtr(), texture.GetWidth() * sizeof(Color));

    // Save optimized mesh
    SaveMesh(positions, indices, uvs, texture, "F:\\Programming\\Cuda_octree\\UltraLod\\optimized_model.bin");

    // Next line will invalidate all gpu allocations!
    //if (cudaDeviceReset() != cudaSuccess)
    //    return -1;

    return 0;
}