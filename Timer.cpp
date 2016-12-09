#include "Timer.hpp"

#include <Windows.h>

int64_t Timer::ticksPerSecond;

Timer::Timer()

{
    static bool init = (QueryPerformanceFrequency((LARGE_INTEGER*)&ticksPerSecond), true);

    // Sample start time
    Reset();
}

double Timer::End() const
{
    int64_t end;

    // Get ticks
    QueryPerformanceCounter((LARGE_INTEGER*)&end);

    return (double)(end - m_start) / ticksPerSecond;
}

double Timer::Reset()
{
    auto value = End();
    QueryPerformanceCounter((LARGE_INTEGER*)&m_start);
    return value;
}