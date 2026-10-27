#pragma once
// =========================================================================
// LIGHTING RIG DERIVATION — core module (v24.01 Phase 1)
// =========================================================================
// Single host-side derivation shared by BOTH render paths (V2401_PROPOSAL.md
// Sec 4, items 1.2/1.3; risk-table mitigation: "single host-side
// lighting-parameter pack feeding both paths"). The OpenGL mesh shader
// (ui/viewport/viewport_renderer.cpp) uploads the derived directions as
// uniforms, and the CUDA raymarcher (ui/viewport/cuda_viewport_gl.cu)
// receives them through CudaRasterizer::raymarch_implicit_viewport_gpu —
// both fed from the same call site in render_viewport_window(), so the two
// rigs cannot drift. The verification harness re-uses this header for the
// --verify lighting-defaults sub-check (Test 20).
//
// View-space convention (matches the v23.28/v23.29 hardcoded constants):
// +X right, +Y up, +Z toward the camera lens. The azimuth/elevation
// parameterization reproduces the original constants:
//   key  (-0.22, 0.34, 0.91) -> az = atan2(-0.22, 0.91) ~ -13.6 deg,
//                               el = asin(0.34/|v|)      ~ +20.0 deg
//   fill ( 0.38,-0.28, 0.88) -> fixed angular offset from the key light
//                               (dAz = +36.94 deg, dEl = -36.24 deg)
// The default slider values (-13.6, +20.0) re-derive both vectors to within
// 0.4% per component (rounded, human-friendly defaults) — documented as an
// accepted deviation in V2401_PHASE1_REPORT.md.
// =========================================================================
#include <cmath>
#include <glm/glm.hpp>

namespace rhizome {

// Unit view-space direction from azimuth (deg; 0 = camera axis, +right)
// and elevation (deg; +up). Inverse of the spherical constants above.
inline glm::vec3 light_dir_view(float azimuth_deg, float elevation_deg) {
    const float az = glm::radians(azimuth_deg);
    const float el = glm::radians(elevation_deg);
    return glm::normalize(glm::vec3(glm::cos(el) * glm::sin(az),
                                    glm::sin(el),
                                    glm::cos(el) * glm::cos(az)));
}

// Fill-light angular offset from the key light, derived from the v23.29
// hardcoded pair (header comment): the fill rides along when the key
// rotates, preserving the rig's key/fill relationship.
inline constexpr float kFillAzOffsetDeg = 36.94f;
inline constexpr float kFillElOffsetDeg = -36.24f;

inline glm::vec3 fill_light_dir_view(float key_azimuth_deg, float key_elevation_deg) {
    return light_dir_view(key_azimuth_deg + kFillAzOffsetDeg,
                          key_elevation_deg + kFillElOffsetDeg);
}

} // namespace rhizome
