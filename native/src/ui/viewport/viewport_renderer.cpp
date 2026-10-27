// =========================================================================
// RHIZOME VIEWPORT RENDERER — implementation (v23.29 Phase 4)
// =========================================================================
// Migrated verbatim from src/app/viewport_renderer.cpp. The only changes:
// the include paths point at the relocated tree, and the ~24-member
// ViewportContext is resolved through AppState& + EngineResources&
// (deterministic rewrite, scripts/phase4_move_viewport.sh). The dual
// seed-visibility OR (Phase 2 finding F9: DisplayState::show_seeds was
// never written anywhere) is consolidated onto SeedState::show_seeds,
// which is the flag CTRL112 actually drives.
// =========================================================================

#include "ui/viewport/viewport_renderer.hpp"
#include "core/engine_resources.hpp"   // EngineResources (ViewportContext::res)
#include "core/lighting_rig.hpp"       // v24.01 Phase 1: key/fill dir derivation (shared with CUDA path call)

#include <imgui.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#endif

namespace rhizome {

#ifdef _WIN32
static SpaceNavState g_spacenav;
static WNDPROC g_prev_wndproc = nullptr;

static LRESULT CALLBACK SpaceNavWndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_INPUT) {
        UINT dwSize = 0;
        GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, nullptr, &dwSize, sizeof(RAWINPUTHEADER));
        if (dwSize > 0) {
            std::vector<BYTE> lpb(dwSize);
            if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lParam), RID_INPUT, lpb.data(), &dwSize, sizeof(RAWINPUTHEADER)) == dwSize) {
                RAWINPUT* raw = reinterpret_cast<RAWINPUT*>(lpb.data());
                if (raw->header.dwType == RIM_TYPEHID) {
                    const BYTE* bRawData = raw->data.hid.bRawData;
                    DWORD total_bytes = raw->data.hid.dwSizeHid * raw->data.hid.dwCount;

                    if (total_bytes > 0) {
                        g_spacenav.connected = true;
                        g_spacenav.last_input_time = std::chrono::steady_clock::now();

                        // 3Dconnexion SpaceNavigator HID report IDs:
                        // 0x01: Translation (Tx, Ty, Tz as signed 16-bit little-endian)
                        // 0x02: Rotation (Rx, Ry, Rz as signed 16-bit little-endian)
                        // 0x03: Buttons
                        if (total_bytes >= 7 && bRawData[0] == 0x01) {
                            int16_t x = static_cast<int16_t>(bRawData[1] | (bRawData[2] << 8));
                            int16_t y = static_cast<int16_t>(bRawData[3] | (bRawData[4] << 8));
                            int16_t z = static_cast<int16_t>(bRawData[5] | (bRawData[6] << 8));
                            g_spacenav.tx = static_cast<float>(x) / 350.0f;
                            g_spacenav.ty = static_cast<float>(y) / 350.0f;
                            g_spacenav.tz = static_cast<float>(z) / 350.0f;
                        } else if (total_bytes >= 7 && bRawData[0] == 0x02) {
                            int16_t rx = static_cast<int16_t>(bRawData[1] | (bRawData[2] << 8));
                            int16_t ry = static_cast<int16_t>(bRawData[3] | (bRawData[4] << 8));
                            int16_t rz = static_cast<int16_t>(bRawData[5] | (bRawData[6] << 8));
                            g_spacenav.rx = static_cast<float>(rx) / 350.0f;
                            g_spacenav.ry = static_cast<float>(ry) / 350.0f;
                            g_spacenav.rz = static_cast<float>(rz) / 350.0f;
                        } else if (total_bytes >= 3 && bRawData[0] == 0x03) {
                            uint32_t btn = static_cast<uint32_t>(bRawData[1] | (bRawData[2] << 8));
                            if ((btn & 0x01) && !(g_spacenav.buttons & 0x01)) {
                                g_spacenav.fit_requested = true;
                            }
                            g_spacenav.buttons = btn;
                        } else if (total_bytes >= 13) {
                            // Combined report format
                            int16_t x = static_cast<int16_t>(bRawData[1] | (bRawData[2] << 8));
                            int16_t y = static_cast<int16_t>(bRawData[3] | (bRawData[4] << 8));
                            int16_t z = static_cast<int16_t>(bRawData[5] | (bRawData[6] << 8));
                            int16_t rx = static_cast<int16_t>(bRawData[7] | (bRawData[8] << 8));
                            int16_t ry = static_cast<int16_t>(bRawData[9] | (bRawData[10] << 8));
                            int16_t rz = static_cast<int16_t>(bRawData[11] | (bRawData[12] << 8));
                            g_spacenav.tx = static_cast<float>(x) / 350.0f;
                            g_spacenav.ty = static_cast<float>(y) / 350.0f;
                            g_spacenav.tz = static_cast<float>(z) / 350.0f;
                            g_spacenav.rx = static_cast<float>(rx) / 350.0f;
                            g_spacenav.ry = static_cast<float>(ry) / 350.0f;
                            g_spacenav.rz = static_cast<float>(rz) / 350.0f;
                        } else if (total_bytes == 6) {
                            int16_t x = static_cast<int16_t>(bRawData[0] | (bRawData[1] << 8));
                            int16_t y = static_cast<int16_t>(bRawData[2] | (bRawData[3] << 8));
                            int16_t z = static_cast<int16_t>(bRawData[4] | (bRawData[5] << 8));
                            g_spacenav.tx = static_cast<float>(x) / 350.0f;
                            g_spacenav.ty = static_cast<float>(y) / 350.0f;
                            g_spacenav.tz = static_cast<float>(z) / 350.0f;
                        }
                    }
                }
            }
        }
    }
    return CallWindowProc(g_prev_wndproc, hwnd, uMsg, wParam, lParam);
}

void init_spacenav(GLFWwindow* window) {
    if (!window) return;
    HWND hwnd = glfwGetWin32Window(window);
    if (hwnd) {
        RAWINPUTDEVICE rid;
        rid.usUsagePage = 0x01; // Generic Desktop Controls
        rid.usUsage = 0x08;     // Multi-axis Controller (3Dconnexion SpaceNavigator / SpaceMouse)
        rid.dwFlags = RIDEV_INPUTSINK;
        rid.hwndTarget = hwnd;
        if (RegisterRawInputDevices(&rid, 1, sizeof(RAWINPUTDEVICE))) {
            g_prev_wndproc = reinterpret_cast<WNDPROC>(SetWindowLongPtr(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(SpaceNavWndProc)));
        }
    }
}

void shutdown_spacenav(GLFWwindow* window) {
    if (g_prev_wndproc && window) {
        HWND win_hwnd = glfwGetWin32Window(window);
        if (win_hwnd) {
            SetWindowLongPtr(win_hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(g_prev_wndproc));
        }
        g_prev_wndproc = nullptr;
    }
}

SpaceNavState& get_spacenav_state() {
    return g_spacenav;
}
#endif

// GLMeshBuffers Implementation
void GLMeshBuffers::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    if (ebo) { glDeleteBuffers(1, &ebo); ebo = 0; }
    index_count = 0;
}

void GLMeshBuffers::upload(const rhizome::TriangleMesh& mesh, bool compute_normals) {
    cleanup();
    if (mesh.vertex_count() == 0 || mesh.indices.empty()) return;

    std::vector<float> vertex_data;
    float pack_ms = 0.0f;
    rhizome::CudaRasterizer::pack_gl_mesh_vbo_gpu(mesh, compute_normals, vertex_data, pack_ms);

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertex_data.size() * sizeof(float), vertex_data.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(uint32_t), mesh.indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    index_count = mesh.indices.size();
}

// BoundingBoxBuffers Implementation
void BoundingBoxBuffers::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    valid = false;
}

void BoundingBoxBuffers::update(glm::vec3 min_p, glm::vec3 max_p) {
    cleanup();
    float lines[] = {
        min_p.x, min_p.y, min_p.z,  0,1,0,  max_p.x, min_p.y, min_p.z,  0,1,0,
        max_p.x, min_p.y, min_p.z,  0,1,0,  max_p.x, min_p.y, max_p.z,  0,1,0,
        max_p.x, min_p.y, max_p.z,  0,1,0,  min_p.x, min_p.y, max_p.z,  0,1,0,
        min_p.x, min_p.y, max_p.z,  0,1,0,  min_p.x, min_p.y, min_p.z,  0,1,0,

        min_p.x, max_p.y, min_p.z,  0,1,0,  max_p.x, max_p.y, min_p.z,  0,1,0,
        max_p.x, max_p.y, min_p.z,  0,1,0,  max_p.x, max_p.y, max_p.z,  0,1,0,
        max_p.x, max_p.y, max_p.z,  0,1,0,  min_p.x, max_p.y, max_p.z,  0,1,0,
        min_p.x, max_p.y, max_p.z,  0,1,0,  min_p.x, max_p.y, min_p.z,  0,1,0,

        min_p.x, min_p.y, min_p.z,  0,1,0,  min_p.x, max_p.y, min_p.z,  0,1,0,
        max_p.x, min_p.y, min_p.z,  0,1,0,  max_p.x, max_p.y, min_p.z,  0,1,0,
        max_p.x, min_p.y, max_p.z,  0,1,0,  max_p.x, max_p.y, max_p.z,  0,1,0,
        min_p.x, min_p.y, max_p.z,  0,1,0,  min_p.x, max_p.y, max_p.z,  0,1,0
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(lines), lines, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    valid = true;
}

// BranchSkeletonBuffers Implementation
void BranchSkeletonBuffers::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    line_vertex_count = 0;
}

void BranchSkeletonBuffers::upload(const std::vector<ColonizationNode>& nodes, bool colorize_by_root) {
    cleanup();
    if (nodes.empty()) return;

    std::vector<float> data;
    float pack_ms = 0.0f;
    rhizome::CudaRasterizer::pack_branch_skeleton_vbo_gpu(
        nodes.data(), nodes.size(), data, pack_ms, colorize_by_root
    );

    if (data.empty()) return;
    line_vertex_count = data.size() / 6;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
}

// SporeCloudBuffers Implementation
void SporeCloudBuffers::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    point_count = 0;
}

void SporeCloudBuffers::upload(const std::vector<glm::vec3>& spores) {
    cleanup();
    if (spores.empty()) return;

    std::vector<float> data;
    float pack_ms = 0.0f;
    rhizome::CudaRasterizer::pack_spore_cloud_vbo_gpu(spores, data, pack_ms);
    point_count = spores.size();

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
}

// FloorGridBuffers Implementation
void FloorGridBuffers::cleanup() {
    if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
    if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
    grid_line_count = 0;
    axis_line_count = 0;
}

void FloorGridBuffers::init(float half_size, float step) {
    cleanup();
    std::vector<float> data;
    float pack_ms = 0.0f;
    rhizome::CudaRasterizer::pack_floor_grid_vbo_gpu(
        half_size, step, data, grid_line_count, axis_line_count, pack_ms
    );
    if (data.empty()) return;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
}

// Shader Programs
static const char* vertex_shader_src = R"(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vNormal;
out vec3 vViewNormal;
out vec3 vPos;
out vec3 vAttrColor;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vPos = worldPos.xyz;
    vAttrColor = aNormal;
    vec3 worldNormal = normalize(mat3(uModel) * aNormal);
    vNormal = worldNormal;
    vViewNormal = normalize(mat3(uView) * worldNormal);
    gl_Position = uProj * uView * worldPos;
    gl_PointSize = 4.0;
}
)";

static const char* fragment_shader_src = R"(
#version 330 core
in vec3 vNormal;
in vec3 vViewNormal;
in vec3 vPos;
in vec3 vAttrColor;

uniform vec4 uColor;
uniform bool uWireframe;
uniform bool uUseVertexColor;
uniform bool uMatteGlass;
uniform float uGlassFrosting;
uniform vec3 uCameraPos;
uniform float uDiffuseStrength;
uniform float uEdgeDarkening;
uniform float uCreviceStrength;
uniform float uSpecularStrength;
// v24.01 Phase 1 (lighting expansion): key/fill directions derived on the
// CPU from Light Azimuth/Elevation (core/lighting_rig.hpp; both render
// paths share that single derivation) + the promoted v23.29 constants.
uniform vec3 uLightDirKey;
uniform vec3 uLightDirFill;
uniform float uAmbient;
uniform float uFillLight;
uniform float uGlossiness;
uniform float uHighlightTint;
uniform float uFresnelRim;

out vec4 FragColor;

void main() {
    if (uWireframe) {
        FragColor = uUseVertexColor ? vec4(vAttrColor, uColor.a) : uColor;
        return;
    }

    vec3 N = normalize(vNormal);
    vec3 nV = normalize(vViewNormal);
    if (!gl_FrontFacing) {
        N = -N;
        nV = -nV;
    }
    vec3 V = normalize(uCameraPos - vPos);

    if (uMatteGlass) {
        // Smooth Matte Frosted Glass Model
        float ndotv = clamp(abs(dot(N, V)), 0.0, 1.0);
        float fresnel = pow(1.0 - ndotv, 2.0);

        vec3 keyDir = normalize(vec3(0.5, 0.4, 0.8));
        float wrapKey = max(dot(N, keyDir) * 0.6 + 0.4, 0.0);
        vec3 diffuse = uColor.rgb * (wrapKey * 0.65);

        vec3 hKey = normalize(keyDir + V);
        float specRough = pow(max(dot(N, hKey), 0.0), mix(28.0, 8.0, uGlassFrosting));
        vec3 specular = vec3(1.0, 0.98, 0.95) * (specRough * 0.35);

        float alpha = clamp(mix(0.12, 0.35, uGlassFrosting) + fresnel * 0.55, 0.08, 0.92) * uColor.a;
        vec3 finalRgb = diffuse + specular + uColor.rgb * (fresnel * uFresnelRim);
        FragColor = vec4(finalRgb, alpha);
        return;
    }

    // Meshmixer View-Space Sculpt MatCap / Clay Shader with True-Color Preservation
    // In view space: +X is right, +Y is up, +Z points straight out toward the camera lens
    float nz = clamp(nV.z, 0.0, 1.0);

    // 1. Primary Meshmixer Headlight Key (slightly upper-left of camera lens) + Counter-Fill
    // v24.01 Phase 1: directions arrive as uniforms (CPU-derived from
    // Light Azimuth/Elevation; defaults reproduce (-0.22, 0.34, 0.91) /
    // (0.38, -0.28, 0.88) — see core/lighting_rig.hpp).
    vec3 keyDirV = uLightDirKey;
    float diffKey = max(dot(nV, keyDirV), 0.0);
    vec3 fillDirV = uLightDirFill;
    float diffFill = max(dot(nV, fillDirV), 0.0);

    // Normalized diffuse response so the lit body of each cylinder matches uColor.rgb at 100% when uDiffuseStrength = 1.0
    float baseIllum = clamp(uAmbient + 0.62 * diffKey + uFillLight * diffFill, 0.0, 1.15) * uDiffuseStrength;

    // 2. Outer Cylinder Edge Darkening (restricted to the outer rim so the main body keeps its true swatch color)
    float rimProfile = smoothstep(0.0, 0.72, nz);
    float edgeContour = mix(1.0, mix(0.08, 1.0, pow(rimProfile, 0.65)), clamp(uEdgeDarkening, 0.0, 1.0));

    // 3. Screen-Space Crevice / Valley Contact Occlusion where roots touch or cross
    vec3 dNdx = dFdx(nV);
    vec3 dNdy = dFdy(nV);
    float normalEdge = clamp(length(dNdx) + length(dNdy), 0.0, 1.4);
    float rawCrevice = clamp(normalEdge * 0.65, 0.0, 0.85);
    float creviceAO = 1.0 - rawCrevice * clamp(uCreviceStrength, 0.0, 1.0);

    // 4. Color-Preserving Satin Specular Highlight along the crown of each root cylinder
    vec3 halfV = normalize(keyDirV + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(nV, halfV), 0.0), uGlossiness) * uSpecularStrength * edgeContour * creviceAO;
    vec3 specColor = mix(uColor.rgb, vec3(1.0), uHighlightTint) * spec;

    vec3 finalRgb = uColor.rgb * (baseIllum * edgeContour * creviceAO) + specColor;

    FragColor = vec4(clamp(finalRgb, 0.0, 1.0), uColor.a);
}
)";

GLuint create_shader_program() {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader_src, nullptr);
    glCompileShader(vs);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader_src, nullptr);
    glCompileShader(fs);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

bool ViewportRenderer::init(int width, int height) {
    fbo_width = width;
    fbo_height = height;

    shader_prog = create_shader_program();
    uModelLoc = glGetUniformLocation(shader_prog, "uModel");
    uViewLoc = glGetUniformLocation(shader_prog, "uView");
    uProjLoc = glGetUniformLocation(shader_prog, "uProj");
    uColorLoc = glGetUniformLocation(shader_prog, "uColor");
    uWireLoc = glGetUniformLocation(shader_prog, "uWireframe");
    uUseVertexColorLoc = glGetUniformLocation(shader_prog, "uUseVertexColor");
    uMatteGlassLoc = glGetUniformLocation(shader_prog, "uMatteGlass");
    uCameraPosLoc = glGetUniformLocation(shader_prog, "uCameraPos");
    uDiffuseStrengthLoc = glGetUniformLocation(shader_prog, "uDiffuseStrength");
    uEdgeDarkeningLoc = glGetUniformLocation(shader_prog, "uEdgeDarkening");
    uCreviceStrengthLoc = glGetUniformLocation(shader_prog, "uCreviceStrength");
    uSpecularStrengthLoc = glGetUniformLocation(shader_prog, "uSpecularStrength");
    uLightDirKeyLoc = glGetUniformLocation(shader_prog, "uLightDirKey");
    uLightDirFillLoc = glGetUniformLocation(shader_prog, "uLightDirFill");
    uAmbientLoc = glGetUniformLocation(shader_prog, "uAmbient");
    uFillLightLoc = glGetUniformLocation(shader_prog, "uFillLight");
    uGlossinessLoc = glGetUniformLocation(shader_prog, "uGlossiness");
    uHighlightTintLoc = glGetUniformLocation(shader_prog, "uHighlightTint");
    uFresnelRimLoc = glGetUniformLocation(shader_prog, "uFresnelRim");

    // Fullscreen textured quad shader for compositing Direct GPU Raymarched Implicit Volume
    {
        const char* q_vs_src = R"(
        #version 330 core
        layout(location = 0) in vec2 aPos;
        layout(location = 1) in vec2 aUV;
        out vec2 vUV;
        void main() {
            vUV = aUV;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
        )";
        const char* q_fs_src = R"(
        #version 330 core
        in vec2 vUV;
        uniform sampler2D uTex;
        uniform sampler2D uDepthTex;
        out vec4 FragColor;
        void main() {
            vec4 col = texture(uTex, vUV);
            if (col.a <= 0.001) {
                discard;
            }
            FragColor = col;
            gl_FragDepth = texture(uDepthTex, vUV).r;
        }
        )";
        GLuint qvs = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(qvs, 1, &q_vs_src, nullptr);
        glCompileShader(qvs);
        GLuint qfs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(qfs, 1, &q_fs_src, nullptr);
        glCompileShader(qfs);
        screen_quad_prog = glCreateProgram();
        glAttachShader(screen_quad_prog, qvs);
        glAttachShader(screen_quad_prog, qfs);
        glLinkProgram(screen_quad_prog);
        glDeleteShader(qvs);
        glDeleteShader(qfs);

        float quad_verts[] = {
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f,  1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f,  0.0f, 1.0f
        };
        glGenVertexArrays(1, &screen_quad_vao);
        glGenBuffers(1, &screen_quad_vbo);
        glBindVertexArray(screen_quad_vao);
        glBindBuffer(GL_ARRAY_BUFFER, screen_quad_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
        glBindVertexArray(0);
    }

    setup_fbo(fbo_width, fbo_height);
    return true;
}

void ViewportRenderer::setup_fbo(int w, int h) {
    if (w <= 0 || h <= 0) return;
    if (msaa_fbo) {
        glDeleteFramebuffers(1, &msaa_fbo);
        glDeleteRenderbuffers(1, &msaa_color_rbo);
        glDeleteRenderbuffers(1, &msaa_depth_rbo);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &fbo_texture);
        if (raymarch_tex) glDeleteTextures(1, &raymarch_tex);
        if (raymarch_depth_tex) glDeleteTextures(1, &raymarch_depth_tex);
        msaa_fbo = 0;
        fbo = 0;
        raymarch_tex = 0;
        raymarch_depth_tex = 0;
    }
    fbo_width = w;
    fbo_height = h;

    // 1. Setup 4x MSAA Multisample Framebuffer
    glGenFramebuffers(1, &msaa_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, msaa_fbo);

    glGenRenderbuffers(1, &msaa_color_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, msaa_color_rbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, msaa_color_rbo);

    glGenRenderbuffers(1, &msaa_depth_rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, msaa_depth_rbo);
    glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH24_STENCIL8, w, h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, msaa_depth_rbo);

    // 2. Setup Resolved 2D Texture Framebuffer for ImGui presentation
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glGenTextures(1, &fbo_texture);
    glBindTexture(GL_TEXTURE_2D, fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);

    // 3. Setup Direct GPU Raymarching RGBA Texture
    glGenTextures(1, &raymarch_tex);
    glBindTexture(GL_TEXTURE_2D, raymarch_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 4. Setup Direct GPU Raymarching Depth Texture (GL_R32F)
    glGenTextures(1, &raymarch_depth_tex);
    glBindTexture(GL_TEXTURE_2D, raymarch_depth_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ViewportRenderer::cleanup() {
    if (msaa_fbo) {
        glDeleteFramebuffers(1, &msaa_fbo);
        glDeleteRenderbuffers(1, &msaa_color_rbo);
        glDeleteRenderbuffers(1, &msaa_depth_rbo);
        msaa_fbo = 0;
    }
    if (fbo) {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &fbo_texture);
        fbo = 0;
        fbo_texture = 0;
    }
    if (raymarch_tex) {
        glDeleteTextures(1, &raymarch_tex);
        raymarch_tex = 0;
    }
    if (raymarch_depth_tex) {
        glDeleteTextures(1, &raymarch_depth_tex);
        raymarch_depth_tex = 0;
    }
    if (screen_quad_vao) {
        glDeleteVertexArrays(1, &screen_quad_vao);
        screen_quad_vao = 0;
    }
    if (screen_quad_vbo) {
        glDeleteBuffers(1, &screen_quad_vbo);
        screen_quad_vbo = 0;
    }
    if (screen_quad_prog) {
        glDeleteProgram(screen_quad_prog);
        screen_quad_prog = 0;
    }
    if (shader_prog) {
        glDeleteProgram(shader_prog);
        shader_prog = 0;
    }
}

void render_viewport_window(ViewportContext& ctx) {
    static bool first_viewport_focus = true;
    if (first_viewport_focus) {
        ImGui::SetNextWindowFocus();
        first_viewport_focus = false;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::Begin("3D Viewport", nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImVec2 view_size = ImGui::GetContentRegionAvail();
    int cur_w = std::max(64, static_cast<int>(view_size.x));
    int cur_h = std::max(64, static_cast<int>(view_size.y));
    if (cur_w != ctx.renderer.fbo_width || cur_h != ctx.renderer.fbo_height) {
        ctx.renderer.setup_fbo(cur_w, cur_h);
    }

    ImGuiIO& io = ImGui::GetIO();
    bool is_vp_hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_None);
    ImVec2 vp_screen_pos = ImGui::GetCursorScreenPos();
    ImVec2 mouse_pos = ImGui::GetMousePos();

    float cam_min_dist = std::max(1.0f, ctx.app_state.specimen.specimen_extent * 0.05f);
    float cam_max_dist = std::max(3000.0f, ctx.app_state.specimen.specimen_extent * 25.0f);

    const DisplayState& disp = ctx.app_state.display;
    const bool show_original_mesh = disp.show_original_mesh;
    const bool show_converted_body = disp.show_converted_body;
    const bool specimen_transparent = disp.specimen_transparent;
    const float param_glass_frosting = disp.param_glass_frosting;
    const float specimen_opacity = disp.specimen_opacity;
    const bool show_roots = disp.show_roots;
    const bool colorize_roots = disp.colorize_roots;
    const bool colorize_strands = disp.colorize_strands;
    const bool show_branch_skeleton = disp.show_branch_skeleton;
    const bool show_wireframe = disp.show_wireframe;
    const bool show_bounding_box = disp.show_bounding_box;
    const bool show_spores = disp.show_spores;
    // F9 (Phase 2 finding): DisplayState::show_seeds was never written by any
    // control; CTRL112 drives SeedState::show_seeds. Consolidated onto the
    // live flag — single source of truth.
    const bool show_seeds = ctx.app_state.seed.show_seeds;
    const int param_export_target = ctx.app_state.mesh_export.param_export_target;

    const float param_shader_diffuse = disp.param_shader_diffuse;
    const float param_shader_edge_darkening = disp.param_shader_edge_darkening;
    const float param_shader_crevice_ao = disp.param_shader_crevice_ao;
    const float param_shader_specular = disp.param_shader_specular;
    const float param_shader_bark_strength = disp.param_shader_bark_strength;
    const float param_shader_bark_frequency = disp.param_shader_bark_frequency;

    // v24.01 Phase 1: the lighting-rig pack. The two directions are derived
    // ONCE here (core/lighting_rig.hpp) and feed BOTH render paths — the GL
    // uniforms below and the CUDA raymarch call further down — so the rigs
    // cannot drift (proposal 1.2/1.3).
    const float param_light_azimuth = disp.param_light_azimuth;
    const float param_light_elevation = disp.param_light_elevation;
    const float param_fill_light = disp.param_fill_light;
    const float param_ambient = disp.param_ambient;
    const float param_glossiness = disp.param_glossiness;
    const float param_highlight_tint = disp.param_highlight_tint;
    const float param_fresnel_rim = disp.param_fresnel_rim;
    const float param_volumetric_ao = disp.param_volumetric_ao;
    const float param_sss_strength = disp.param_sss_strength;
    const float param_sss_warmth = disp.param_sss_warmth;
    const glm::vec3 light_key_dir = light_dir_view(param_light_azimuth, param_light_elevation);
    const glm::vec3 light_fill_dir = fill_light_dir_view(param_light_azimuth, param_light_elevation);

    const glm::vec4& color_viewport_bg = disp.color_viewport_bg;
    const glm::vec4& color_floor_grid = disp.color_floor_grid;
    const glm::vec4& color_floor_axis_x = disp.color_floor_axis_x;
    const glm::vec4& color_floor_axis_y = disp.color_floor_axis_y;
    const glm::vec4& color_converted_body = disp.color_converted_body;
    const glm::vec4& color_roots = disp.color_roots;
    const glm::vec4& color_branch_skeleton = disp.color_branch_skeleton;
    const glm::vec4& color_wireframe = disp.color_wireframe;
    const glm::vec4& color_original_mesh = disp.color_original_mesh;
    const glm::vec4& color_bounding_box = disp.color_bounding_box;
    const glm::vec4& color_spores = disp.color_spores;
    const glm::vec4& color_seed_selected = disp.color_seed_selected;
    const glm::vec4& color_seed_unselected = disp.color_seed_unselected;
    const glm::vec4& color_test_pass = disp.color_test_pass;
    const glm::vec4& color_test_fail = disp.color_test_fail;

#ifdef _WIN32
    // 1. 3Dconnexion SpaceNavigator 6-DOF Hardware Input Processing
    auto& spacenav_state = rhizome::get_spacenav_state();
    if (spacenav_state.connected) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - spacenav_state.last_input_time).count();
        if (elapsed_ms > 200) {
            spacenav_state.tx = 0.0f;
            spacenav_state.ty = 0.0f;
            spacenav_state.tz = 0.0f;
            spacenav_state.rx = 0.0f;
            spacenav_state.ry = 0.0f;
            spacenav_state.rz = 0.0f;
        }

        auto deadzone = [](float val) -> float {
            const float dz = 0.04f;
            if (std::abs(val) < dz) return 0.0f;
            float sign = (val > 0.0f) ? 1.0f : -1.0f;
            return sign * (std::abs(val) - dz) / (1.0f - dz);
        };

        float dt = io.DeltaTime > 0.0f ? std::min(io.DeltaTime, 0.1f) : 0.016f;
        float stx = deadzone(spacenav_state.tx);
        float sty = deadzone(spacenav_state.ty);
        float stz = deadzone(spacenav_state.tz);
        float srx = deadzone(spacenav_state.rx);
        float sry = deadzone(spacenav_state.ry);
        float srz = deadzone(spacenav_state.rz);

        // Orbit Rotation (Object mode: tilting cap forward tilts object forward, twisting/tilting sideways spins)
        if (std::abs(srx) > 0.001f || std::abs(srz) > 0.001f || std::abs(sry) > 0.001f) {
            ctx.res.camera.pitch += srx * (85.0f * dt);
            ctx.res.camera.pitch = std::clamp(ctx.res.camera.pitch, -89.9f, 89.9f);
            ctx.res.camera.yaw += (srz + sry * 0.75f) * (100.0f * dt);
        }

        // Pan (push left/right -> pan left/right; pull up/down -> pan vertically)
        if (std::abs(stx) > 0.001f || std::abs(stz) > 0.001f) {
            ctx.res.camera.pan(stx * (35.0f * dt), stz * (35.0f * dt));
        }

        // Zoom (push forward -> zoom in; pull backward -> zoom out)
        if (std::abs(sty) > 0.001f) {
            ctx.res.camera.distance -= sty * (ctx.res.camera.distance * 1.5f * dt);
            ctx.res.camera.distance = std::clamp(ctx.res.camera.distance, cam_min_dist, cam_max_dist);
        }

        // Button 1: Fit View / Center
        if (spacenav_state.fit_requested) {
            spacenav_state.fit_requested = false;
            if (ctx.res.original_mesh.vertex_count() > 0) {
                glm::vec3 min_p(0.0f), max_p(0.0f);
                float bbox_ms = 0.0f;
                rhizome::CudaRasterizer::compute_bounding_box_gpu(ctx.res.original_mesh, min_p, max_p, bbox_ms);
                ctx.res.camera.target = (min_p + max_p) * 0.5f;
                float size = glm::length(max_p - min_p);
                ctx.res.camera.distance = std::max(size * 1.5f, 30.0f);
            }
        }
    }
#endif

    // 2. Auto Orbit update
    if (ctx.app_state.display.auto_orbit) {
        ctx.res.camera.yaw += 0.35f;
        if (ctx.res.camera.yaw > 360.0f) ctx.res.camera.yaw -= 360.0f;
    }

    // 3. Standard 3D CAD/DCC Mouse Controls
    static bool is_dragging_orbit = false;
    static bool is_dragging_pan = false;

    // Disambiguate left click between seed picking and view orbiting
    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left) && is_vp_hovered) {
        bool hit_seed = false;
        if (show_seeds && !ctx.app_state.seed.points.empty()) {
            float aspect_early = static_cast<float>(ctx.renderer.fbo_width) / ctx.renderer.fbo_height;
            float cam_far_early = std::max(5000.0f, ctx.res.camera.distance * 5.0f);
            glm::mat4 early_view(1.0f), early_proj(1.0f), early_vp(1.0f), early_inv_vp(1.0f);
            float early_gpu_ms = 0.0f;
            rhizome::CudaRasterizer::compute_camera_matrices_gpu(
                glm::radians(45.0f), aspect_early, 0.05f, cam_far_early,
                ctx.res.camera.get_eye_position(), ctx.res.camera.target, ctx.res.camera.get_world_up(),
                early_view, early_proj, early_vp, early_inv_vp, early_gpu_ms
            );

            float ndc_x = ((mouse_pos.x - vp_screen_pos.x) / view_size.x) * 2.0f - 1.0f;
            float ndc_y = 1.0f - ((mouse_pos.y - vp_screen_pos.y) / view_size.y) * 2.0f;
            glm::vec4 near4 = early_inv_vp * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
            glm::vec4 far4  = early_inv_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
            glm::vec3 r_orig = glm::vec3(near4) / near4.w;
            glm::vec3 r_end  = glm::vec3(far4)  / far4.w;
            glm::vec3 r_dir  = glm::normalize(r_end - r_orig);

            int hit_idx = -1;
            float pick_radius = std::max({ctx.app_state.growth.trunk_caliber * 1.80f, ctx.res.camera.distance * 0.022f, 0.80f});
            float pick_ms = 0.0f;
            rhizome::CudaRasterizer::pick_seed_ray_gpu(
                ctx.app_state.seed.points, r_orig, r_dir, pick_radius, hit_idx, pick_ms
            );

            if (hit_idx >= 0) {
                ctx.app_state.seed.is_dragging_seed = true;
                ctx.app_state.seed.dragged_seed_index = hit_idx;
                ctx.app_state.seed.selected_seed_index = hit_idx;
                hit_seed = true;
            }
        }
        if (!hit_seed) {
            is_dragging_orbit = true;
        }
    }

    // Delete key shortcut inside 3D viewport to delete selected seed and prune its grown roots
    if (is_vp_hovered && ImGui::IsKeyPressed(ImGuiKey_Delete) && show_seeds) {
        if (!ctx.app_state.seed.points.empty() && ctx.app_state.seed.selected_seed_index >= 0 && ctx.app_state.seed.selected_seed_index < static_cast<int>(ctx.app_state.seed.points.size())) {
            int removed_idx = ctx.app_state.seed.selected_seed_index;
            ctx.app_state.seed.points.erase(ctx.app_state.seed.points.begin() + removed_idx);
            if (removed_idx < static_cast<int>(ctx.app_state.seed.frozen.size())) {
                ctx.app_state.seed.frozen.erase(ctx.app_state.seed.frozen.begin() + removed_idx);
            }
            if (removed_idx < static_cast<int>(ctx.app_state.seed.snapped_to_root.size())) {
                ctx.app_state.seed.snapped_to_root.erase(ctx.app_state.seed.snapped_to_root.begin() + removed_idx);
            }
            if (removed_idx < static_cast<int>(ctx.app_state.seed.radii.size())) {
                ctx.app_state.seed.radii.erase(ctx.app_state.seed.radii.begin() + removed_idx);
            }

            if (ctx.app_state.growth.has_live_session) {
                ctx.res.live_colonizer.delete_seed(removed_idx);
            }

            if (ctx.app_state.seed.points.empty()) {
                glm::vec3 min_p(0.0f), max_p(0.0f);
                float bbox_ms = 0.0f;
                if (ctx.res.original_mesh.vertex_count() > 0) {
                    rhizome::CudaRasterizer::compute_bounding_box_gpu(ctx.res.original_mesh, min_p, max_p, bbox_ms);
                }
                glm::vec3 center = (min_p + max_p) * 0.5f;
                glm::vec3 default_seed(center.x, center.y, min_p.z + 2.0f * ctx.app_state.growth.detail_level);
                if (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) {
                    float desired_margin = -(ctx.app_state.growth.trunk_caliber + ctx.app_state.growth.seed_offset);
                    float seed_ms = 0.0f;
                    rhizome::CudaRasterizer::find_default_seed_gpu(*ctx.res.current_sdf_grid, min_p, max_p, desired_margin, default_seed, seed_ms);
                }
                ctx.app_state.seed.points.push_back(default_seed);
                ctx.app_state.seed.frozen.push_back(0);
                ctx.app_state.seed.snapped_to_root.push_back(0);
                ctx.app_state.seed.radii.push_back(ctx.app_state.growth.trunk_caliber);
                ctx.app_state.seed.selected_seed_index = 0;
                if (ctx.app_state.growth.has_live_session) {
                    ctx.res.live_colonizer.clear();
                    ctx.app_state.growth.has_live_session = false;
                    if (ctx.update_implicit_session) ctx.update_implicit_session(false);
                }
            } else {
                if (ctx.app_state.seed.selected_seed_index >= static_cast<int>(ctx.app_state.seed.points.size())) {
                    ctx.app_state.seed.selected_seed_index = static_cast<int>(ctx.app_state.seed.points.size()) - 1;
                }
                if (ctx.app_state.growth.has_live_session && ctx.res.current_sdf_grid && ctx.build_colonization_config) {
                    auto c_cfg = ctx.build_colonization_config();
                    ctx.res.live_colonizer.sync_seeds_session(ctx.res.current_sdf_grid, ctx.app_state.seed.points, c_cfg);
                    if (ctx.update_implicit_session) ctx.update_implicit_session(!ctx.app_state.growth.is_growing_live);
                }
            }
        }
    }

    static bool s_was_dragging_seed = false;
    if (ctx.app_state.seed.is_dragging_seed) {
        s_was_dragging_seed = true;
    }

    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (s_was_dragging_seed) {
            s_was_dragging_seed = false;
            if (ctx.app_state.growth.has_live_session && ctx.res.current_sdf_grid && ctx.build_colonization_config && ctx.update_implicit_session) {
                ColonizationConfig c_cfg = ctx.build_colonization_config();
                ctx.res.live_colonizer.sync_seeds_session(ctx.res.current_sdf_grid, ctx.app_state.seed.points, c_cfg);
                ctx.update_implicit_session(!ctx.app_state.growth.is_growing_live);
            }
        }
        is_dragging_orbit = false;
        ctx.app_state.seed.is_dragging_seed = false;
        ctx.app_state.seed.dragged_seed_index = -1;
    }

    if ((ImGui::IsMouseClicked(ImGuiMouseButton_Right) || ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) && is_vp_hovered) {
        is_dragging_pan = true;
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Right) && !ImGui::IsMouseDown(ImGuiMouseButton_Middle)) {
        is_dragging_pan = false;
    }

    if (is_dragging_orbit) {
        ctx.res.camera.yaw -= io.MouseDelta.x * 0.35f;
        ctx.res.camera.pitch += io.MouseDelta.y * 0.35f;
        ctx.res.camera.pitch = std::clamp(ctx.res.camera.pitch, -89.9f, 89.9f);
    }

    if (is_dragging_pan) {
        ctx.res.camera.pan(io.MouseDelta.x, io.MouseDelta.y);
    }

    if (is_vp_hovered && io.MouseWheel != 0.0f) {
        ctx.res.camera.distance -= io.MouseWheel * (ctx.res.camera.distance * 0.10f);
        ctx.res.camera.distance = std::clamp(ctx.res.camera.distance, cam_min_dist, cam_max_dist);
    }

    // 4. Compute 100% Synchronized View, Projection, and Inverse matrices on CUDA GPU
    glm::mat4 view(1.0f), proj(1.0f), view_proj(1.0f), inv_vp(1.0f);
    float aspect = static_cast<float>(ctx.renderer.fbo_width) / ctx.renderer.fbo_height;
    float cam_far = std::max(5000.0f, ctx.res.camera.distance * 5.0f);
    float cam_near = 0.05f;
    float gpu_mat_ms = 0.0f;

    rhizome::CudaRasterizer::compute_camera_matrices_gpu(
        glm::radians(45.0f), aspect, cam_near, cam_far,
        ctx.res.camera.get_eye_position(), ctx.res.camera.target, ctx.res.camera.get_world_up(),
        view, proj, view_proj, inv_vp, gpu_mat_ms
    );

    // 5. Seed Dragging Update with fully synchronized inverse view-projection matrix
    if (show_seeds && ctx.app_state.seed.is_dragging_seed) {
        if (ctx.app_state.seed.dragged_seed_index >= 0 && ctx.app_state.seed.dragged_seed_index < static_cast<int>(ctx.app_state.seed.points.size()) &&
            (ctx.app_state.seed.dragged_seed_index >= static_cast<int>(ctx.app_state.seed.frozen.size()) || ctx.app_state.seed.frozen[ctx.app_state.seed.dragged_seed_index] == 0)) {
            float ndc_x = ((mouse_pos.x - vp_screen_pos.x) / view_size.x) * 2.0f - 1.0f;
            float ndc_y = 1.0f - ((mouse_pos.y - vp_screen_pos.y) / view_size.y) * 2.0f;

            glm::vec4 near4 = inv_vp * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
            glm::vec4 far4 = inv_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
            glm::vec3 r_orig = glm::vec3(near4) / near4.w;
            glm::vec3 r_end = glm::vec3(far4) / far4.w;
            glm::vec3 r_dir = glm::normalize(r_end - r_orig);

            glm::vec3 cur_seed = ctx.app_state.seed.points[ctx.app_state.seed.dragged_seed_index];
            glm::vec3 cam_forward = glm::normalize(ctx.res.camera.target - ctx.res.camera.get_eye_position());

            float denom = glm::dot(r_dir, cam_forward);
            if (std::abs(denom) > 1e-4f) {
                float t = glm::dot(cur_seed - r_orig, cam_forward) / denom;
                glm::vec3 target_pos = r_orig + t * r_dir;

                glm::vec3 min_p(0.0f), max_p(0.0f);
                if (ctx.res.original_mesh.vertex_count() > 0) {
                    float bbox_ms = 0.0f;
                    rhizome::CudaRasterizer::compute_bounding_box_gpu(ctx.res.original_mesh, min_p, max_p, bbox_ms);
                    target_pos = glm::clamp(target_pos, min_p, max_p);
                }

                int drag_idx = ctx.app_state.seed.dragged_seed_index;
                bool seed_is_snapped = (drag_idx >= 0 && drag_idx < static_cast<int>(ctx.app_state.seed.snapped_to_root.size()) && ctx.app_state.seed.snapped_to_root[drag_idx]);
                bool snap_to_root = seed_is_snapped || (ctx.app_state.growth.algorithm_mode == 3 || ctx.app_state.growth.surface_adherence > 0.5f);
                glm::vec3 root_snapped(0.0f), root_norm(0.0f, 0.0f, 1.0f);
                if (snap_to_root && ctx.res.live_colonizer.snap_seed_to_root_surface(target_pos, ctx.app_state.growth.seed_offset, root_snapped, root_norm)) {
                    if (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) {
                        float cur_sdf = ctx.res.current_sdf_grid->sample_trilinear(root_snapped);
                        if (cur_sdf > -0.05f) {
                            root_snapped -= root_norm * (cur_sdf + 0.15f);
                        }
                    }
                    ctx.app_state.seed.points[ctx.app_state.seed.dragged_seed_index] = root_snapped;
                } else if (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) {
                    float desired_margin = (ctx.app_state.growth.trunk_caliber + ctx.app_state.growth.seed_offset);

                    // 100% CUDA GPU Ray-SDF Volume Intersection (Zero Mesh Required)
                    glm::vec3 hit_pt(0.0f), hit_norm(0.0f, 0.0f, 1.0f);
                    float hit_t = -1.0f, gpu_ms = 0.0f;

                    if (rhizome::CudaRasterizer::raycast_sdf_volume_gpu(*ctx.res.current_sdf_grid, r_orig, r_dir, hit_pt, hit_norm, hit_t, gpu_ms)) {
                        glm::vec3 inward_seed = hit_pt - hit_norm * desired_margin;
                        ctx.app_state.seed.points[ctx.app_state.seed.dragged_seed_index] = inward_seed;
                    } else if (ctx.res.current_sdf_grid->sample_trilinear(target_pos) <= -desired_margin) {
                        ctx.app_state.seed.points[ctx.app_state.seed.dragged_seed_index] = target_pos;
                    }
                } else {
                    ctx.app_state.seed.points[ctx.app_state.seed.dragged_seed_index] = target_pos;
                }
            }
        }
    }

    // Render 3D Scene into 4x MSAA Multisample Framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, ctx.renderer.msaa_fbo);
    glViewport(0, 0, ctx.renderer.fbo_width, ctx.renderer.fbo_height);
    glEnable(GL_MULTISAMPLE);
    glClearColor(color_viewport_bg.r, color_viewport_bg.g, color_viewport_bg.b, color_viewport_bg.a);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glUseProgram(ctx.renderer.shader_prog);
    glUniformMatrix4fv(ctx.renderer.uViewLoc, 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(ctx.renderer.uProjLoc, 1, GL_FALSE, &proj[0][0]);
    glm::mat4 default_model(1.0f);
    glUniformMatrix4fv(ctx.renderer.uModelLoc, 1, GL_FALSE, &default_model[0][0]);
    glm::vec3 cam_pos = ctx.res.camera.get_eye_position();
    glUniform3f(ctx.renderer.uCameraPosLoc, cam_pos.x, cam_pos.y, cam_pos.z);
    glUniform1f(ctx.renderer.uDiffuseStrengthLoc, param_shader_diffuse);
    glUniform1f(ctx.renderer.uEdgeDarkeningLoc, param_shader_edge_darkening);
    glUniform1f(ctx.renderer.uCreviceStrengthLoc, param_shader_crevice_ao);
    glUniform1f(ctx.renderer.uSpecularStrengthLoc, param_shader_specular);
    // v24.01 Phase 1: lighting-rig uniforms (already unit vectors —
    // core/lighting_rig.hpp normalizes).
    glUniform3f(ctx.renderer.uLightDirKeyLoc, light_key_dir.x, light_key_dir.y, light_key_dir.z);
    glUniform3f(ctx.renderer.uLightDirFillLoc, light_fill_dir.x, light_fill_dir.y, light_fill_dir.z);
    glUniform1f(ctx.renderer.uAmbientLoc, param_ambient);
    glUniform1f(ctx.renderer.uFillLightLoc, param_fill_light);
    glUniform1f(ctx.renderer.uGlossinessLoc, param_glossiness);
    glUniform1f(ctx.renderer.uHighlightTintLoc, param_highlight_tint);
    glUniform1f(ctx.renderer.uFresnelRimLoc, param_fresnel_rim);

    // 0. Render Floor Grid flat on X-Y ground plane (Z = 0)
    if (ctx.res.floor_grid_buffers.vao != 0) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glUniform1i(ctx.renderer.uWireLoc, 1);
        glBindVertexArray(ctx.res.floor_grid_buffers.vao);

        // Floor grid lines
        if (ctx.res.floor_grid_buffers.grid_line_count > 0) {
            glUniform4f(ctx.renderer.uColorLoc, color_floor_grid.r, color_floor_grid.g, color_floor_grid.b, color_floor_grid.a);
            glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(ctx.res.floor_grid_buffers.grid_line_count));
        }
        // Primary X and Y axis lines on floor
        if (ctx.res.floor_grid_buffers.axis_line_count > 0) {
            glUniform4f(ctx.renderer.uColorLoc, color_floor_axis_x.r, color_floor_axis_x.g, color_floor_axis_x.b, color_floor_axis_x.a);
            glDrawArrays(GL_LINES, static_cast<GLint>(ctx.res.floor_grid_buffers.grid_line_count), 2);
            glUniform4f(ctx.renderer.uColorLoc, color_floor_axis_y.r, color_floor_axis_y.g, color_floor_axis_y.b, color_floor_axis_y.a);
            glDrawArrays(GL_LINES, static_cast<GLint>(ctx.res.floor_grid_buffers.grid_line_count + 2), 2);
        }
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // 1 & 2. Direct GPU Raymarching of Implicit Specimen SDF & Live Root Volume/Capsules (Zero Viewport Triangles)
    bool roots_active = show_roots;
    if ((show_converted_body && ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) || roots_active) {
        std::vector<rhizome::CudaVolSection> gpu_vol_secs;
        const auto& vol_secs = ctx.res.live_colonizer.get_volumetric_sections();
        const auto& b_nodes = ctx.res.live_colonizer.get_nodes();
        const void* classic_nodes_ptr = nullptr;
        size_t classic_nodes_cnt = 0;

        const bool active_colorize_roots = colorize_roots;
        if (roots_active && (!ctx.res.current_roots_grid || !ctx.res.current_roots_grid->is_valid() || colorize_strands || active_colorize_roots)) {
            if ((ctx.app_state.growth.algorithm_mode == 1 || ctx.app_state.growth.algorithm_mode == 2) && !vol_secs.empty()) {
                gpu_vol_secs.resize(vol_secs.size());
                std::memcpy(gpu_vol_secs.data(), vol_secs.data(), vol_secs.size() * sizeof(rhizome::CudaVolSection));
            } else if (!b_nodes.empty()) {
                classic_nodes_ptr = b_nodes.data();
                classic_nodes_cnt = b_nodes.size();
            }
        }

        float rm_ms = 0.0f;
        static int s_last_rm_w = 0, s_last_rm_h = 0;
        static glm::mat4 s_last_view(0.0f), s_last_inv_vp(0.0f);
        static bool s_last_ortho = false, s_last_show_spec = false, s_last_spec_trans = false;
        static float s_last_frost = -1.0f, s_last_opac = -1.0f;
        static glm::vec4 s_last_col_spec(0.0f), s_last_col_roots(0.0f);
        static uint64_t s_last_spec_ver = 0, s_last_roots_ver = 0;
        static size_t s_last_vol_secs = 0, s_last_b_nodes = 0;
        static int s_last_iter = -1;
        static bool s_last_show_roots = false, s_last_col_strands = false, s_last_col_roots_by_seed = false;
        static float s_last_diff = -1.0f, s_last_edge = -1.0f, s_last_ao = -1.0f, s_last_spec = -1.0f;
        static float s_last_blend = -1.0f, s_last_wall_cut = -1.0f;
        static float s_last_bark_str = -1.0f, s_last_bark_freq = -1.0f;
        static float s_last_trunk_caliber = -1.0f, s_last_tendril_caliber = -1.0f, s_last_taper = -1.0f;
        // v24.01 Phase 1: lighting-rig cache (derived dirs + promoted
        // scalars; all ten feed the raymarch path).
        static glm::vec3 s_last_lkey(0.0f), s_last_lfill(0.0f);
        static float s_last_ambient = -1.0f, s_last_fill_lt = -1.0f, s_last_gloss = -1.0f;
        static float s_last_tint = -1.0f, s_last_frim = -1.0f, s_last_volao = -1.0f;
        static float s_last_sss = -1.0f, s_last_sssw = -1.0f;

        uint64_t cur_spec_ver = (ctx.res.current_sdf_grid && ctx.res.current_sdf_grid->is_valid()) ? ctx.res.current_sdf_grid->version : 0;
        uint64_t cur_roots_ver = (ctx.res.current_roots_grid && ctx.res.current_roots_grid->is_valid()) ? ctx.res.current_roots_grid->version : 0;
        int cur_iter = ctx.res.live_colonizer.get_current_iteration();

        bool need_raymarch = ctx.app_state.growth.is_growing_live ||
            ctx.renderer.raymarch_rgba_buffer.size() != static_cast<size_t>(ctx.renderer.fbo_width) * ctx.renderer.fbo_height ||
            s_last_rm_w != ctx.renderer.fbo_width || s_last_rm_h != ctx.renderer.fbo_height ||
            s_last_view != view || s_last_inv_vp != inv_vp ||
            s_last_ortho != ctx.app_state.viewport.is_orthographic ||
            s_last_show_spec != show_converted_body ||
            s_last_spec_trans != specimen_transparent ||
            s_last_frost != param_glass_frosting ||
            s_last_opac != specimen_opacity ||
            s_last_col_spec != color_converted_body ||
            s_last_col_roots != color_roots ||
            s_last_spec_ver != cur_spec_ver ||
            s_last_roots_ver != cur_roots_ver ||
            s_last_vol_secs != vol_secs.size() ||
            s_last_b_nodes != b_nodes.size() ||
            s_last_iter != cur_iter ||
            s_last_show_roots != roots_active ||
            s_last_col_strands != colorize_strands ||
            s_last_col_roots_by_seed != active_colorize_roots ||
            s_last_diff != param_shader_diffuse ||
            s_last_edge != param_shader_edge_darkening ||
            s_last_ao != param_shader_crevice_ao ||
            s_last_spec != param_shader_specular ||
            s_last_lkey != light_key_dir ||
            s_last_lfill != light_fill_dir ||
            s_last_ambient != param_ambient ||
            s_last_fill_lt != param_fill_light ||
            s_last_gloss != param_glossiness ||
            s_last_tint != param_highlight_tint ||
            s_last_frim != param_fresnel_rim ||
            s_last_volao != param_volumetric_ao ||
            s_last_sss != param_sss_strength ||
            s_last_sssw != param_sss_warmth ||
            s_last_blend != ctx.app_state.growth.branch_smooth_k ||
            s_last_wall_cut != ctx.app_state.growth.wall_cut_smooth ||
            s_last_bark_str != param_shader_bark_strength ||
            s_last_bark_freq != param_shader_bark_frequency ||
            s_last_trunk_caliber != ctx.app_state.growth.trunk_caliber ||
            s_last_tendril_caliber != ctx.app_state.growth.tendril_caliber ||
            s_last_taper != ctx.app_state.growth.taper_curve;

        bool rm_ok = !need_raymarch;
        if (need_raymarch) {
            ctx.renderer.raymarch_depth_buffer.resize(static_cast<size_t>(ctx.renderer.fbo_width) * ctx.renderer.fbo_height);
            rm_ok = rhizome::CudaRasterizer::raymarch_implicit_viewport_gpu(
                ctx.renderer.fbo_width, ctx.renderer.fbo_height,
                view, inv_vp, cam_pos, ctx.app_state.viewport.is_orthographic,
                ctx.res.current_sdf_grid.get(), show_converted_body, specimen_transparent,
                param_glass_frosting, specimen_opacity, color_converted_body,
                ctx.res.current_roots_grid.get(),
                gpu_vol_secs.empty() ? nullptr : &gpu_vol_secs,
                classic_nodes_ptr, classic_nodes_cnt,
                roots_active, colorize_strands, color_roots,
                param_shader_diffuse, param_shader_edge_darkening,
                param_shader_crevice_ao, param_shader_specular,
                ctx.renderer.raymarch_rgba_buffer, rm_ms,
                ctx.app_state.growth.branch_smooth_k, ctx.app_state.growth.wall_cut_smooth, false, active_colorize_roots,
                param_shader_bark_strength, param_shader_bark_frequency,
                &view_proj, &ctx.renderer.raymarch_depth_buffer,
                // v24.01 Phase 1: lighting rig (dirs pre-derived above — the
                // same vectors the GL uniforms carry)
                light_key_dir, light_fill_dir,
                param_ambient, param_fill_light, param_glossiness,
                param_highlight_tint, param_fresnel_rim,
                param_volumetric_ao, param_sss_strength, param_sss_warmth);
            if (rm_ok) {
                glBindTexture(GL_TEXTURE_2D, ctx.renderer.raymarch_tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ctx.renderer.fbo_width, ctx.renderer.fbo_height, GL_RGBA, GL_UNSIGNED_BYTE, ctx.renderer.raymarch_rgba_buffer.data());

                glBindTexture(GL_TEXTURE_2D, ctx.renderer.raymarch_depth_tex);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ctx.renderer.fbo_width, ctx.renderer.fbo_height, GL_RED, GL_FLOAT, ctx.renderer.raymarch_depth_buffer.data());

                s_last_rm_w = ctx.renderer.fbo_width; s_last_rm_h = ctx.renderer.fbo_height;
                s_last_view = view; s_last_inv_vp = inv_vp;
                s_last_ortho = ctx.app_state.viewport.is_orthographic;
                s_last_show_spec = show_converted_body;
                s_last_spec_trans = specimen_transparent;
                s_last_frost = param_glass_frosting;
                s_last_opac = specimen_opacity;
                s_last_col_spec = color_converted_body;
                s_last_col_roots = color_roots;
                s_last_spec_ver = cur_spec_ver;
                s_last_roots_ver = cur_roots_ver;
                s_last_vol_secs = vol_secs.size();
                s_last_b_nodes = b_nodes.size();
                s_last_iter = cur_iter;
                s_last_show_roots = roots_active;
                s_last_col_strands = colorize_strands;
                s_last_col_roots_by_seed = active_colorize_roots;
                s_last_diff = param_shader_diffuse;
                s_last_edge = param_shader_edge_darkening;
                s_last_ao = param_shader_crevice_ao;
                s_last_spec = param_shader_specular;
                s_last_lkey = light_key_dir;
                s_last_lfill = light_fill_dir;
                s_last_ambient = param_ambient;
                s_last_fill_lt = param_fill_light;
                s_last_gloss = param_glossiness;
                s_last_tint = param_highlight_tint;
                s_last_frim = param_fresnel_rim;
                s_last_volao = param_volumetric_ao;
                s_last_sss = param_sss_strength;
                s_last_sssw = param_sss_warmth;
                s_last_blend = ctx.app_state.growth.branch_smooth_k;
                s_last_wall_cut = ctx.app_state.growth.wall_cut_smooth;
                s_last_bark_str = param_shader_bark_strength;
                s_last_bark_freq = param_shader_bark_frequency;
                s_last_trunk_caliber = ctx.app_state.growth.trunk_caliber;
                s_last_tendril_caliber = ctx.app_state.growth.tendril_caliber;
                s_last_taper = ctx.app_state.growth.taper_curve;
            }
        }

        if (rm_ok) {
            glEnable(GL_BLEND);
            // v24.01 fix (Task 7d / report Sec 6.2): keep the MSAA FBO's alpha
            // channel at 1.0. The quad is the first alpha-blended draw into a
            // buffer cleared to alpha 1.0, so (ONE, ONE) on the alpha channel
            // is a no-op for clay/root/empty texels and only lifts matte-glass
            // texels from a^2 (alpha would otherwise be applied twice: once
            // here, once more by the ImGui::Image blit of fbo_texture) back to
            // the raymarcher's intended alpha.
            glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE);
            glEnable(GL_DEPTH_TEST);
            // v24.01 fix (Task 7d / report Sec 6.2): the raymarcher deliberately
            // writes depth 1.0 for matte-glass texels (glass must not occlude
            // the GL mesh passes drawn after this quad). GL_LESS rejected those
            // fragments outright (1.0 < 1.0 fails against the cleared depth
            // buffer), so the glass color was discarded before it ever reached
            // the framebuffer and the specimen vanished in glass mode - a
            // latent v23.28-era bug (byte-identical code in v23.29; no test
            // ever exercised glass mode). GL_LEQUAL lets the color through; the
            // depth written for those texels stays 1.0, so the mesh passes'
            // occlusion semantics are unchanged. GL_LESS is restored below for
            // everything drawn after the quad.
            glDepthFunc(GL_LEQUAL);
            glDepthMask(GL_TRUE);

            glUseProgram(ctx.renderer.screen_quad_prog);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ctx.renderer.raymarch_tex);
            glUniform1i(glGetUniformLocation(ctx.renderer.screen_quad_prog, "uTex"), 0);

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, ctx.renderer.raymarch_depth_tex);
            glUniform1i(glGetUniformLocation(ctx.renderer.screen_quad_prog, "uDepthTex"), 1);

            glBindVertexArray(ctx.renderer.screen_quad_vao);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            glBindVertexArray(0);

            // v24.01 fix (Task 7d): restore v23.29 depth semantics for every
            // pass drawn after the raymarch quad (meshes, wireframes, seeds).
            glDepthFunc(GL_LESS);
            glDisable(GL_BLEND);
            glUseProgram(ctx.renderer.shader_prog);
        }
    }

    // 2b. Render Branch Skeleton Tendril Graph (Live Interactive Preview)
    if (show_branch_skeleton && ctx.res.branch_gl_buffers.vao != 0 && ctx.res.branch_gl_buffers.line_vertex_count > 0) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(2.5f);
        glUniform1i(ctx.renderer.uWireLoc, 1);
        glUniform1i(ctx.renderer.uUseVertexColorLoc, (colorize_strands || colorize_roots) ? 1 : 0);
        glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
        glUniform4f(ctx.renderer.uColorLoc, color_branch_skeleton.r, color_branch_skeleton.g, color_branch_skeleton.b, color_branch_skeleton.a);
        glBindVertexArray(ctx.res.branch_gl_buffers.vao);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(ctx.res.branch_gl_buffers.line_vertex_count));
        glUniform1i(ctx.renderer.uUseVertexColorLoc, 0);
        glLineWidth(1.0f);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // Generated Mesh Result Rendering (Solid Shaded + Wireframe Edges when 'Generated mesh' toggle is active)
    if (show_wireframe) {
        const GLMeshBuffers* gen_buf = nullptr;
        glm::vec4 gen_col = color_roots;
        if (param_export_target == 0 && ctx.res.roots_gl_buffers.vao != 0 && ctx.res.roots_gl_buffers.index_count > 0) {
            gen_buf = &ctx.res.roots_gl_buffers;
            gen_col = color_roots;
        } else if (param_export_target == 1 && ctx.res.body_gl_buffers.vao != 0 && ctx.res.body_gl_buffers.index_count > 0) {
            gen_buf = &ctx.res.body_gl_buffers;
            gen_col = color_converted_body;
        } else if (param_export_target == 2 && ctx.res.merged_gl_buffers.vao != 0 && ctx.res.merged_gl_buffers.index_count > 0) {
            gen_buf = &ctx.res.merged_gl_buffers;
            gen_col = color_roots;
        } else if (ctx.res.roots_gl_buffers.vao != 0 && ctx.res.roots_gl_buffers.index_count > 0) {
            gen_buf = &ctx.res.roots_gl_buffers;
            gen_col = color_roots;
        } else if (ctx.res.body_gl_buffers.vao != 0 && ctx.res.body_gl_buffers.index_count > 0) {
            gen_buf = &ctx.res.body_gl_buffers;
            gen_col = color_converted_body;
        }

        if (gen_buf != nullptr) {
            // 1. Solid shaded generated triangle mesh
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glUniform1i(ctx.renderer.uWireLoc, 0);
            glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
            glUniform4f(ctx.renderer.uColorLoc, gen_col.r, gen_col.g, gen_col.b, 1.0f);
            glBindVertexArray(gen_buf->vao);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(gen_buf->index_count), GL_UNSIGNED_INT, 0);

            // 2. Crisp wireframe triangle edges over the generated mesh
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(-1.5f, -1.5f);
            glUniform1i(ctx.renderer.uWireLoc, 1);
            glUniform4f(ctx.renderer.uColorLoc, color_wireframe.r, color_wireframe.g, color_wireframe.b, color_wireframe.a);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(gen_buf->index_count), GL_UNSIGNED_INT, 0);
            glDisable(GL_POLYGON_OFFSET_LINE);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
    }

    // 3. Render Original Mesh Wireframe
    if (show_original_mesh && ctx.res.orig_gl_buffers.vao != 0 && ctx.res.orig_gl_buffers.index_count > 0) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonOffset(-1.0f, -1.0f);
        glUniform1i(ctx.renderer.uWireLoc, 1);
        glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
        glUniform4f(ctx.renderer.uColorLoc, color_original_mesh.r, color_original_mesh.g, color_original_mesh.b, color_original_mesh.a);
        glBindVertexArray(ctx.res.orig_gl_buffers.vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ctx.res.orig_gl_buffers.index_count), GL_UNSIGNED_INT, 0);
        glDisable(GL_POLYGON_OFFSET_LINE);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // 4. Render Test Result Meshes (Green for PASS, Red for FAIL)
    for (const auto& obj : ctx.res.test_objects) {
        if (obj.buffers.vao != 0 && obj.buffers.index_count > 0) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glUniform1i(ctx.renderer.uWireLoc, 0);
            glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
            const glm::vec4& tcol = obj.passed ? color_test_pass : color_test_fail;
            glUniform4f(ctx.renderer.uColorLoc, tcol.r, tcol.g, tcol.b, tcol.a);
            glBindVertexArray(obj.buffers.vao);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(obj.buffers.index_count), GL_UNSIGNED_INT, 0);
        }
    }

    // 5. Render Bounding Box
    if (show_bounding_box && ctx.res.bbox_buffers.valid) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glUniform1i(ctx.renderer.uWireLoc, 1);
        glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
        glUniform4f(ctx.renderer.uColorLoc, color_bounding_box.r, color_bounding_box.g, color_bounding_box.b, color_bounding_box.a);
        glBindVertexArray(ctx.res.bbox_buffers.vao);
        glDrawArrays(GL_LINES, 0, 24);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    // 5b. Render Spores (3D Nutrient Cloud inside Vessel)
    if (show_spores && ctx.res.spore_gl_buffers.vao != 0 && ctx.res.spore_gl_buffers.point_count > 0) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glPointSize(4.0f);
        glUniformMatrix4fv(ctx.renderer.uModelLoc, 1, GL_FALSE, &default_model[0][0]);
        glUniform1i(ctx.renderer.uWireLoc, 1);
        glUniform1i(ctx.renderer.uUseVertexColorLoc, 0);
        glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
        glUniform4f(ctx.renderer.uColorLoc, color_spores.r, color_spores.g, color_spores.b, color_spores.a);
        glBindVertexArray(ctx.res.spore_gl_buffers.vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(ctx.res.spore_gl_buffers.point_count));
        glBindVertexArray(0);
        glPointSize(1.0f);
        glDisable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_BLEND);
    }

    // 6. Render Seeds (Visual size = Trunk Caliber or locked frozen radius, colorized per seed with true 3D depth occlusion)
    if (show_seeds && ctx.res.seed_sphere_buffer.vao != 0 && ctx.res.seed_sphere_buffer.index_count > 0) {
        if (ctx.app_state.seed.radii.size() < ctx.app_state.seed.points.size()) {
            ctx.app_state.seed.radii.resize(ctx.app_state.seed.points.size(), ctx.app_state.growth.trunk_caliber);
        }

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glUniform1i(ctx.renderer.uWireLoc, 0);
        glUniform1i(ctx.renderer.uMatteGlassLoc, 0);
        glBindVertexArray(ctx.res.seed_sphere_buffer.vao);

        const bool colorize_by_seed = (colorize_roots || colorize_strands);

        for (size_t si = 0; si < ctx.app_state.seed.points.size(); ++si) {
            glm::vec3 spos = ctx.app_state.seed.points[si];
            bool is_frozen = (si < ctx.app_state.seed.frozen.size() && ctx.app_state.seed.frozen[si]);
            float s_rad = (si < ctx.app_state.seed.radii.size() && is_frozen)
                ? ctx.app_state.seed.radii[si]
                : ctx.app_state.growth.trunk_caliber;

            // Keep unfrozen seed radius tracked so locking captures the exact current caliber
            if (!is_frozen && si < ctx.app_state.seed.radii.size()) {
                ctx.app_state.seed.radii[si] = ctx.app_state.growth.trunk_caliber;
            }

            glm::mat4 model = glm::translate(glm::mat4(1.0f), spos) * glm::scale(glm::mat4(1.0f), glm::vec3(s_rad));
            glUniformMatrix4fv(ctx.renderer.uModelLoc, 1, GL_FALSE, &model[0][0]);

            glm::vec4 base_seed_col = colorize_by_seed
                ? get_seed_color_vec4(static_cast<int>(si))
                : color_seed_unselected;

            if (static_cast<int>(si) == ctx.app_state.seed.selected_seed_index) {
                glm::vec4 sel_col = colorize_by_seed
                    ? glm::mix(base_seed_col, color_seed_selected, 0.40f)
                    : color_seed_selected;
                glUniform4f(ctx.renderer.uColorLoc, sel_col.r, sel_col.g, sel_col.b, 1.0f);
            } else {
                glUniform4f(ctx.renderer.uColorLoc, base_seed_col.r, base_seed_col.g, base_seed_col.b, 1.0f);
            }

            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(ctx.res.seed_sphere_buffer.index_count), GL_UNSIGNED_INT, 0);
        }
        // Reset uModel to Identity for subsequent renders
        glm::mat4 ident(1.0f);
        glUniformMatrix4fv(ctx.renderer.uModelLoc, 1, GL_FALSE, &ident[0][0]);
    }

    glBindVertexArray(0);

    // Resolve 4x MSAA buffer into resolved FBO texture for ImGui viewport
    glBindFramebuffer(GL_READ_FRAMEBUFFER, ctx.renderer.msaa_fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ctx.renderer.fbo);
    glBlitFramebuffer(0, 0, ctx.renderer.fbo_width, ctx.renderer.fbo_height, 0, 0, ctx.renderer.fbo_width, ctx.renderer.fbo_height, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    ImGui::Image(reinterpret_cast<void*>(static_cast<uintptr_t>(ctx.renderer.fbo_texture)), view_size, ImVec2(0, 1), ImVec2(1, 0));
    ImGui::End();
    ImGui::PopStyleVar();
}

} // namespace rhizome
