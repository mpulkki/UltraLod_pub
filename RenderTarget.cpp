#include "RenderTarget.hpp"
#include <assert.h>


namespace UltraLod
{
    RenderTarget::RenderTarget(int width, int height)
        : m_width(width)
        , m_height(height)
    {
        assert(m_width > 0 && m_height > 0);

        // Allocate buffer memory
        m_data.resize(m_width * m_height);
    }

    Color* RenderTarget::DataPtr()
    {
        return m_data.data();
    }

    const Color* RenderTarget::DataPtr() const
    {
        return m_data.data();
    }

    int RenderTarget::GetHeight() const
    {
        return m_height;
    }

    int RenderTarget::GetWidth() const
    {
        return m_width;
    }
}