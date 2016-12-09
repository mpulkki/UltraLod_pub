#pragma once

#include "Geometry.hpp"

struct CoverageInput
{
    const glm::vec3* dPositions;
    int* dCoverages;
    int groupCount;
    int groupSize;
    const Aabb& bounds;
    float voxelSize;
};

void computeVoxelCoverage(const CoverageInput& input);