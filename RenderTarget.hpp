#pragma once

#include "Utility.cuh"

namespace UltraLod
{
    class RenderTarget
    {
    public:
        RenderTarget(int width, int height);

        int GetWidth() const;
        int GetHeight() const;

        Color* DataPtr();
        const Color* DataPtr() const;

    private:
        RenderTarget(const RenderTarget&) = delete;
        RenderTarget& operator=(const RenderTarget&) = delete;

    private:
        int m_width;
        int m_height;

        std::vector<Color> m_data;
    };
}