#pragma once

#include "Defs.cuh"

class Timer
{
public:
    Timer();
    double End() const;
    double Reset();

private:
    int64_t        m_start;
    static int64_t ticksPerSecond;
};
