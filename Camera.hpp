#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/matrix.hpp>

class Camera
{
public:
    Camera(float width, float height, float fov);

    Camera& SetPosition(const glm::vec3& pos);
    Camera& SetPitch(float pitch);
    Camera& SetYaw(float yaw);

    const glm::vec3& GetPosition() const;

    glm::mat4x4 WorldToClip() const;
    glm::mat4x4 ClipToWorld() const;

private:
    float     m_yaw;
    float     m_pitch;
    glm::vec3 m_position;

    glm::mat4x4 m_proj;
    //float     m_fov;
    //float     m_width;
    //float     m_height;
};
