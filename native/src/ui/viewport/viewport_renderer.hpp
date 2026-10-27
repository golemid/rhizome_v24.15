#pragma once

// =========================================================================
// RHIZOME VIEWPORT RENDERER — ui/viewport module (v23.29 Phase 4)
// =========================================================================
// OpenGL 3.3 Core MSAA framebuffer presenter, Z-up orbit/pan/zoom camera,
// 3Dconnexion SpaceNavigator hooks, seed picking, and the offscreen
// raymarch compositing blit. Migrated from include/app/viewport_renderer.hpp
// + src/app/viewport_renderer.cpp (kernels and GL code verbatim; CUDA
// viewport services cuda_viewport_gl.cu / cuda_viewport_packers.cu already
// live in this directory, compiled into rhizome_core since Phase 3).
//
// ViewportContext: v23.28's ~24-member context is replaced by the approved
// two-reference form (Phase 1 proposal Sec 5.2): AppState& + EngineResources&
// plus the two orchestration callbacks. Seed points / frozen flags / live
// flags / viewport toggles / specimen extent all resolve through
// app_state.* (shadow-state eradication, finding F4).
// =========================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <memory>
#include <functional>

#include "core/app_state.hpp"
#include "engine/io/mesh.hpp"
#include "engine/colonization/colonization.hpp"
#include "engine/sdf/cuda_rasterizer.hpp"

// (include-cycle break: EngineResources — defined in core/engine_resources.hpp,
// which includes THIS header for the GL buffer structs — is only referenced
// through ViewportContext::res, so a forward declaration suffices here.)

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace rhizome {

struct EngineResources; // core/engine_resources.hpp (fwd — see note above)

// 3D Camera with Smooth Orbit, Pan, and Zoom (Z-up coordinate system)
struct Camera3D {
    float pitch = 30.0f; // Isometric elevation angle above X-Y floor plane
    float yaw = 45.0f;   // Isometric azimuth (+X, -Y, +Z)
    float distance = 75.0f;
    glm::vec3 target = {0.0f, 0.0f, 20.0f};

    glm::vec3 get_eye_position() const {
        float p_rad = glm::radians(pitch);
        float y_rad = glm::radians(yaw);
        glm::vec3 eye;
        float h_dist = distance * std::cos(p_rad);
        eye.x = target.x + h_dist * std::sin(y_rad);
        eye.y = target.y - h_dist * std::cos(y_rad);
        eye.z = target.z + distance * std::sin(p_rad);
        return eye;
    }

    glm::vec3 get_world_up() const {
        if (std::abs(pitch) > 89.5f) {
            return (pitch > 0.0f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, -1.0f, 0.0f);
        }
        return glm::vec3(0.0f, 0.0f, 1.0f);
    }

    glm::mat4 get_view_matrix() const {
        glm::vec3 eye = get_eye_position();
        glm::vec3 forward = glm::normalize(target - eye);

        // In a Z-up system, default world up is (0, 0, 1)
        glm::vec3 world_up(0.0f, 0.0f, 1.0f);
        if (std::abs(pitch) > 89.5f) {
            // Looking almost straight down/up along +/- Z axis: use +Y/-Y as screen-up
            world_up = (pitch > 0.0f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, -1.0f, 0.0f);
        }

        glm::vec3 right = glm::cross(forward, world_up);
        if (glm::length(right) < 1e-4f) {
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            right = glm::normalize(right);
        }
        glm::vec3 up = glm::normalize(glm::cross(right, forward));

        return glm::lookAt(eye, target, up);
    }

    void pan(float dx, float dy) {
        float p_rad = glm::radians(pitch);
        float y_rad = glm::radians(yaw);

        glm::vec3 eye;
        float h_dist = distance * std::cos(p_rad);
        eye.x = target.x + h_dist * std::sin(y_rad);
        eye.y = target.y - h_dist * std::cos(y_rad);
        eye.z = target.z + distance * std::sin(p_rad);

        glm::vec3 forward = glm::normalize(target - eye);
        glm::vec3 world_up(0.0f, 0.0f, 1.0f);
        if (std::abs(pitch) > 89.5f) {
            world_up = (pitch > 0.0f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(0.0f, -1.0f, 0.0f);
        }

        glm::vec3 right = glm::cross(forward, world_up);
        if (glm::length(right) < 1e-4f) {
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            right = glm::normalize(right);
        }
        glm::vec3 up = glm::normalize(glm::cross(right, forward));

        float speed = distance * 0.0018f;
        target -= right * (dx * speed);
        target += up * (dy * speed);
    }
};

// 3Dconnexion SpaceNavigator 6-DOF Hardware Input State
struct SpaceNavState {
    float tx = 0.0f;
    float ty = 0.0f;
    float tz = 0.0f;
    float rx = 0.0f;
    float ry = 0.0f;
    float rz = 0.0f;
    bool connected = false;
    bool fit_requested = false;
    uint32_t buttons = 0;
    std::chrono::steady_clock::time_point last_input_time;
};

#ifdef _WIN32
void init_spacenav(GLFWwindow* window);
void shutdown_spacenav(GLFWwindow* window);
SpaceNavState& get_spacenav_state();
#endif

// OpenGL Mesh Buffers (VAO, VBO, EBO)
struct GLMeshBuffers {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    size_t index_count = 0;

    void cleanup();
    void upload(const rhizome::TriangleMesh& mesh, bool compute_normals = true);
};

// Bounding Box Line Buffers Helper
struct BoundingBoxBuffers {
    GLuint vao = 0;
    GLuint vbo = 0;
    bool valid = false;

    void cleanup();
    void update(glm::vec3 min_p, glm::vec3 max_p);
};

// Branch Skeleton Line Buffer Helper for Live Interactive Preview
struct BranchSkeletonBuffers {
    GLuint vao = 0;
    GLuint vbo = 0;
    size_t line_vertex_count = 0;

    void cleanup();
    void upload(const std::vector<ColonizationNode>& nodes, bool colorize_by_root = false);
};

// Spore Cloud Point Buffer Helper for 3D Nutrient Visualization
struct SporeCloudBuffers {
    GLuint vao = 0;
    GLuint vbo = 0;
    size_t point_count = 0;

    void cleanup();
    void upload(const std::vector<glm::vec3>& spores);
};

// Floor Grid Buffer Helper (flat on X-Y ground plane, Z = 0)
struct FloorGridBuffers {
    GLuint vao = 0;
    GLuint vbo = 0;
    size_t grid_line_count = 0;
    size_t axis_line_count = 0;

    void cleanup();
    void init(float half_size = 100.0f, float step = 10.0f);
};

// Viewport Renderer managing MSAA FBO, Resolved Texture, Raymarching, and Shaders
struct ViewportRenderer {
    // 4x MSAA Multisample Framebuffer
    GLuint msaa_fbo = 0;
    GLuint msaa_color_rbo = 0;
    GLuint msaa_depth_rbo = 0;

    // Resolved 2D Texture Framebuffer for ImGui presentation
    GLuint fbo = 0;
    GLuint fbo_texture = 0;

    // Direct GPU Raymarching RGBA Texture
    GLuint raymarch_tex = 0;

    // Fullscreen Textured Quad for Compositing Direct GPU Raymarched Implicit Volume
    GLuint screen_quad_vao = 0;
    GLuint screen_quad_vbo = 0;
    GLuint screen_quad_prog = 0;

    // Main 3D Mesh Shader Program and Uniform Locations
    GLuint shader_prog = 0;
    GLint uModelLoc = -1;
    GLint uViewLoc = -1;
    GLint uProjLoc = -1;
    GLint uColorLoc = -1;
    GLint uWireLoc = -1;
    GLint uUseVertexColorLoc = -1;
    GLint uMatteGlassLoc = -1;
    GLint uCameraPosLoc = -1;
    GLint uDiffuseStrengthLoc = -1;
    GLint uEdgeDarkeningLoc = -1;
    GLint uCreviceStrengthLoc = -1;
    GLint uSpecularStrengthLoc = -1;
    // v24.01 Phase 1 (lighting expansion): derived key/fill directions
    // (core/lighting_rig.hpp) + the five promoted scalar constants.
    GLint uLightDirKeyLoc = -1;
    GLint uLightDirFillLoc = -1;
    GLint uAmbientLoc = -1;
    GLint uFillLightLoc = -1;
    GLint uGlossinessLoc = -1;
    GLint uHighlightTintLoc = -1;
    GLint uFresnelRimLoc = -1;

    int fbo_width = 1280;
    int fbo_height = 720;
    std::vector<uint32_t> raymarch_rgba_buffer;
    GLuint raymarch_depth_tex = 0;
    std::vector<float> raymarch_depth_buffer;

    bool init(int width = 1280, int height = 720);
    void setup_fbo(int w, int h);
    void cleanup();
};

GLuint create_shader_program();

struct TestMeshObject {
    GLMeshBuffers buffers;
    glm::vec4 color{1.0f};
    std::string name;
    bool passed = false;
};

// Slim two-reference viewport context (Phase 1 proposal Sec 5.2):
// state through AppState, live GPU/GL resources through EngineResources,
// plus the two main.cpp orchestration commands.
struct ViewportContext {
    ViewportRenderer& renderer;
    AppState& app_state;
    EngineResources& res;
    std::function<ColonizationConfig()> build_colonization_config = nullptr;
    std::function<void(bool)> update_implicit_session = nullptr;
};

void render_viewport_window(ViewportContext& ctx);

} // namespace rhizome
