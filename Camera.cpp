#include "Camera.hpp"

#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace glm;

Camera::Camera(float width, float height, float fov)
    : m_position(0, 0, 0)
    , m_yaw(0)
    , m_pitch(0)
{
    // Create projection matrix
    m_proj = glm::perspectiveFovRH(fov, width, height, 0.1f, 1000.0f);
}

mat4x4 Camera::ClipToWorld() const
{
    return inverse(WorldToClip());
}

mat4x4 Camera::WorldToClip() const
{
    auto translation   = glm::translate(-m_position);
    auto rotationYaw   = glm::rotate(-m_yaw, vec3(0, 1, 0));
    auto rotationPitch = glm::rotate(-m_pitch, vec3(1, 0, 0));

    return m_proj * rotationPitch * rotationYaw * translation;
}

const vec3& Camera::GetPosition() const
{
    return m_position;
}

Camera& Camera::SetPosition(const vec3& pos)
{
    m_position = pos;
    return *this;
}

Camera& Camera::SetPitch(float pitch)
{
    m_pitch = pitch;
    return *this;
}

Camera& Camera::SetYaw(float yaw)
{
    m_yaw = yaw;
    return *this;
}