#pragma once

// =========================================================================
// RHIZOME MATH UTILITIES — core module (v23.29 Phase 2)
// =========================================================================
// Shared host/device math helpers, atomic CAS guards, and rasterization
// tiling constants for the 100% CUDA GPU pipeline. Migrated verbatim from
// v23.28 src/sdf/cuda_common.cuh in Phase 2. Memory management helpers
// (CUDA_CHECK, VRAM alloc/copy wrappers, DeviceBuffer RAII) and the shared
// cross-TU kernel declarations now live in engine/sdf/cuda_common.cuh
// (Phase 3, AGENTS.md Sec A.2) — this header stays pure math, no CUDA
// runtime API usage.
// =========================================================================

#include "engine/sdf/cuda_types.hpp"   // CudaCapsule / CudaTriangle / CudaVolSection (math helpers only —
                                       // no dependency on the CudaRasterizer driver header)
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#define HD_FUNC __host__ __device__ __forceinline__
#else
#define HD_FUNC inline
#endif

namespace rhizome {

// --- 0. Shared Rasterization Tiling Constants (AGENTS.md Sec A.2) ---
// Single Source of Truth for shared-memory rasterization tile sizing.
// Previously a local constant inside cuda_rasterizer.cu; promoted here in
// Phase 2 (Phase 1 finding F6) so every CUDA engine module shares it.
constexpr int RASTER_TILE_SIZE = 128;

// Ceiling division helper for CUDA grid sizing (additive, Phase 2).
#ifndef RHIZOME_DIV_UP
#define RHIZOME_DIV_UP(a, b) (((a) + (b) - 1) / (b))
#endif

// --- 1. Exact Triangle Point Distance (Host & Device Inline) ---
HD_FUNC float point_triangle_distance_sq(
    const glm::vec3& p,
    const glm::vec3& a,
    const glm::vec3& b,
    const glm::vec3& c,
    glm::vec3& out_closest
) {
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;
    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        out_closest = a;
        return glm::dot(ap, ap);
    }

    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) {
        out_closest = b;
        return glm::dot(bp, bp);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        out_closest = a + v * ab;
        glm::vec3 diff = p - out_closest;
        return glm::dot(diff, diff);
    }

    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) {
        out_closest = c;
        return glm::dot(cp, cp);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        out_closest = a + w * ac;
        glm::vec3 diff = p - out_closest;
        return glm::dot(diff, diff);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        out_closest = b + w * (c - b);
        glm::vec3 diff = p - out_closest;
        return glm::dot(diff, diff);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    out_closest = a + ab * v + ac * w;
    glm::vec3 diff = p - out_closest;
    return glm::dot(diff, diff);
}

// --- 1b. Exact Triangle Solid Angle for Generalized Winding Number (Host & Device Inline) ---
HD_FUNC float triangle_solid_angle(
    const glm::vec3& p,
    const glm::vec3& v0,
    const glm::vec3& v1,
    const glm::vec3& v2
) {
    glm::vec3 a = v0 - p;
    glm::vec3 b = v1 - p;
    glm::vec3 c = v2 - p;
    float la = sqrtf(glm::dot(a, a));
    float lb = sqrtf(glm::dot(b, b));
    float lc = sqrtf(glm::dot(c, c));
    if (la <= 1e-6f || lb <= 1e-6f || lc <= 1e-6f) {
        return 0.0f;
    }
    float det = a.x * (b.y * c.z - b.z * c.y) -
                a.y * (b.x * c.z - b.z * c.x) +
                a.z * (b.x * c.y - b.y * c.x);
    float denom = la * lb * lc +
                  glm::dot(a, b) * lc +
                  glm::dot(b, c) * la +
                  glm::dot(c, a) * lb;
    return 2.0f * atan2f(det, denom);
}

struct GpuColonizationNodeLayout {
    glm::vec3 position;
    int parent_index;
    float radius;
    int depth;
    float energy;
    int strand_id;
};

#ifdef __CUDACC__

__device__ __forceinline__ float dev_atomicMin_float(float* addr, float value) {
    if (isnan(value)) return *addr;
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float current_f = __int_as_float(assumed);
        if (isnan(current_f)) {
            old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        } else {
            old = atomicCAS(addr_as_int, assumed,
                __float_as_int(fminf(value, current_f)));
        }
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float dev_atomicMax_float(float* addr, float value) {
    if (isnan(value)) return *addr;
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        float current_f = __int_as_float(assumed);
        if (isnan(current_f)) {
            old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
        } else {
            old = atomicCAS(addr_as_int, assumed,
                __float_as_int(fmaxf(value, current_f)));
        }
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float dev_clamp(float v, float lo, float hi) {
    return fminf(fmaxf(v, lo), hi);
}

__device__ __forceinline__ glm::vec3 dev_safe_norm(const glm::vec3& v, const glm::vec3& fallback) {
    float l2 = glm::dot(v, v);
    return (l2 > 1e-10f) ? (v * rsqrtf(l2)) : fallback;
}

__device__ __forceinline__ glm::vec3 dev_clamp_to_cone(
    const glm::vec3& cand_dir,
    const glm::vec3& axis_dir,
    float max_angle_deg
) {
    if (max_angle_deg >= 179.0f) return cand_dir;
    float max_angle_rad = max_angle_deg * (3.14159265358979323846f / 180.0f);
    float cos_limit = cosf(max_angle_rad);
    float sin_limit = sinf(max_angle_rad);

    float d = glm::dot(cand_dir, axis_dir);
    if (d >= cos_limit) {
        return cand_dir;
    }

    glm::vec3 perp = cand_dir - axis_dir * d;
    float perp_l2 = glm::dot(perp, perp);
    if (perp_l2 > 1e-10f) {
        perp *= rsqrtf(perp_l2);
        return dev_safe_norm(axis_dir * cos_limit + perp * sin_limit, axis_dir);
    }
    return axis_dir;
}

__device__ __forceinline__ uint32_t dev_hash_coords(int cx, int cy, int cz, uint32_t mask) {
    uint32_t h = (static_cast<uint32_t>(cx) * 73856093u) ^
                 (static_cast<uint32_t>(cy) * 19349663u) ^
                 (static_cast<uint32_t>(cz) * 83492791u);
    return h & mask;
}

__device__ __forceinline__ float dev_hash_01(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return static_cast<float>(x & 0x00FFFFFFu) / 16777216.0f;
}

// 3D Procedural Simplex Noise for organic bark striations and veins on GPU
__device__ __forceinline__ float dev_simplex_noise_3d(const glm::vec3& p) {
    const float F3 = 1.0f / 3.0f;
    const float G3 = 1.0f / 6.0f;

    float s = (p.x + p.y + p.z) * F3;
    int i = static_cast<int>(floorf(p.x + s));
    int j = static_cast<int>(floorf(p.y + s));
    int k = static_cast<int>(floorf(p.z + s));

    float t = static_cast<float>(i + j + k) * G3;
    float x0 = p.x - (static_cast<float>(i) - t);
    float y0 = p.y - (static_cast<float>(j) - t);
    float z0 = p.z - (static_cast<float>(k) - t);

    int i1, j1, k1;
    int i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        } else if (x0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
        } else {
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
        }
    } else {
        if (y0 < z0) {
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
        } else if (x0 < z0) {
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
        } else {
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
        }
    }

    float x1 = x0 - static_cast<float>(i1) + G3;
    float y1 = y0 - static_cast<float>(j1) + G3;
    float z1 = z0 - static_cast<float>(k1) + G3;

    float x2 = x0 - static_cast<float>(i2) + 2.0f * G3;
    float y2 = y0 - static_cast<float>(j2) + 2.0f * G3;
    float z2 = z0 - static_cast<float>(k2) + 2.0f * G3;

    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    auto grad_dot = [](uint32_t hash, float x, float y, float z) -> float {
        int h = hash & 15;
        float u = h < 8 ? x : y;
        float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
        return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
    };

    uint32_t mask = 0x0FFFFFFFu;
    uint32_t h0 = dev_hash_coords(i, j, k, mask);
    uint32_t h1 = dev_hash_coords(i + i1, j + j1, k + k1, mask);
    uint32_t h2 = dev_hash_coords(i + i2, j + j2, k + k2, mask);
    uint32_t h3 = dev_hash_coords(i + 1, j + 1, k + 1, mask);

    float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;

    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 > 0.0f) { t0 *= t0; n0 = t0 * t0 * grad_dot(h0, x0, y0, z0); }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 > 0.0f) { t1 *= t1; n1 = t1 * t1 * grad_dot(h1, x1, y1, z1); }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 > 0.0f) { t2 *= t2; n2 = t2 * t2 * grad_dot(h2, x2, y2, z2); }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 > 0.0f) { t3 *= t3; n3 = t3 * t3 * grad_dot(h3, x3, y3, z3); }

    return 32.0f * (n0 + n1 + n2 + n3);
}

__device__ __forceinline__ float dev_bark_texture_3d(const glm::vec3& p, float freq) {
    // Anisotropic scaling creates elongated root bark grooves and striations along growth axis
    glm::vec3 sp(p.x * freq * 1.5f, p.y * freq * 1.5f, p.z * freq * 0.45f);
    float n1 = dev_simplex_noise_3d(sp);
    float n2 = dev_simplex_noise_3d(sp * 2.3f) * 0.5f;
    float n3 = dev_simplex_noise_3d(sp * 4.7f) * 0.25f;
    return (n1 + n2 + n3) / 1.75f;
}

__device__ __forceinline__ unsigned int hash_u64(unsigned long long k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (unsigned int)k;
}

__device__ __forceinline__ unsigned int dev_float_to_ordered_uint(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
}

__device__ __forceinline__ float dev_sample_sdf_nearest(
    const float* __restrict__ sdf_grid,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    const glm::vec3& p
) {
    int gx = static_cast<int>(floorf(p.x / voxel_size)) - min_coord.x;
    int gy = static_cast<int>(floorf(p.y / voxel_size)) - min_coord.y;
    int gz = static_cast<int>(floorf(p.z / voxel_size)) - min_coord.z;
    if (gx < 0 || gx >= dims.x || gy < 0 || gy >= dims.y || gz < 0 || gz >= dims.z) {
        return 1000.0f;
    }
    return sdf_grid[(size_t)gz * dims.y * dims.x + (size_t)gy * dims.x + (size_t)gx];
}

__device__ __forceinline__ float dev_sample_sdf_trilinear(
    const float* __restrict__ sdf_grid,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    const glm::vec3& p
) {
    float gx = (p.x / voxel_size) - static_cast<float>(min_coord.x);
    float gy = (p.y / voxel_size) - static_cast<float>(min_coord.y);
    float gz = (p.z / voxel_size) - static_cast<float>(min_coord.z);

    int x0 = static_cast<int>(floorf(gx));
    int y0 = static_cast<int>(floorf(gy));
    int z0 = static_cast<int>(floorf(gz));

    if (x0 < 0 || x0 >= dims.x - 1 || y0 < 0 || y0 >= dims.y - 1 || z0 < 0 || z0 >= dims.z - 1) {
        return dev_sample_sdf_nearest(sdf_grid, min_coord, dims, voxel_size, p);
    }

    float fx = gx - static_cast<float>(x0);
    float fy = gy - static_cast<float>(y0);
    float fz = gz - static_cast<float>(z0);

    const size_t sy = static_cast<size_t>(dims.x);
    const size_t sz = static_cast<size_t>(dims.y) * sy;
    const size_t base = static_cast<size_t>(z0) * sz + static_cast<size_t>(y0) * sy + static_cast<size_t>(x0);

    float v000 = sdf_grid[base];
    float v100 = sdf_grid[base + 1];
    float v010 = sdf_grid[base + sy];
    float v110 = sdf_grid[base + sy + 1];
    float v001 = sdf_grid[base + sz];
    float v101 = sdf_grid[base + sz + 1];
    float v011 = sdf_grid[base + sz + sy];
    float v111 = sdf_grid[base + sz + sy + 1];

    float c00 = v000 * (1.0f - fx) + v100 * fx;
    float c10 = v010 * (1.0f - fx) + v110 * fx;
    float c01 = v001 * (1.0f - fx) + v101 * fx;
    float c11 = v011 * (1.0f - fx) + v111 * fx;

    float c0 = c00 * (1.0f - fy) + c10 * fy;
    float c1 = c01 * (1.0f - fy) + c11 * fy;

    return c0 * (1.0f - fz) + c1 * fz;
}

__device__ __forceinline__ glm::vec3 dev_sample_sdf_grad(
    const float* __restrict__ sdf_grid,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    const glm::vec3& p
) {
    float eps = fmaxf(voxel_size, 0.1f);
    float gx = dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p + glm::vec3(eps, 0.0f, 0.0f)) -
               dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p - glm::vec3(eps, 0.0f, 0.0f));
    float gy = dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p + glm::vec3(0.0f, eps, 0.0f)) -
               dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p - glm::vec3(0.0f, eps, 0.0f));
    float gz = dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p + glm::vec3(0.0f, 0.0f, eps)) -
               dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p - glm::vec3(0.0f, 0.0f, eps));
    return dev_safe_norm(glm::vec3(gx, gy, gz), glm::vec3(0.0f, 0.0f, 1.0f));
}

__device__ __forceinline__ void dev_confine_sphere_to_vessel(
    const float* __restrict__ sdf_grid,
    glm::ivec3 min_coord,
    glm::ivec3 dims,
    float voxel_size,
    float min_vessel_sdf,
    glm::vec3& p,
    float r,
    glm::vec3* dir_inout,
    float wall_clearance_scale = 0.85f
) {
    float target_sdf = (min_vessel_sdf < 0.0f)
        ? fmaxf(-r * wall_clearance_scale, min_vessel_sdf * 0.55f)
        : -0.05f;
    for (int k = 0; k < 4; ++k) {
        float sdf = dev_sample_sdf_trilinear(sdf_grid, min_coord, dims, voxel_size, p);
        if (sdf <= target_sdf) break;
        glm::vec3 n = dev_sample_sdf_grad(sdf_grid, min_coord, dims, voxel_size, p);
        p -= n * (sdf - target_sdf + 0.02f * voxel_size);
        if (dir_inout && k == 0) {
            float dn = glm::dot(*dir_inout, n);
            if (dn > 0.0f) {
                glm::vec3 tang = *dir_inout - dn * n;
                float tl2 = glm::dot(tang, tang);
                if (tl2 > 1e-10f) {
                    *dir_inout = tang * rsqrtf(tl2);
                }
            }
        }
    }
}

__device__ __forceinline__ glm::vec3 dev_get_strand_color(int strand_id) {
    const int idx = (strand_id >= 0 ? strand_id : 0) % 10;
    switch (idx) {
        case 0: return glm::vec3(0.098f, 0.827f, 0.635f); // Primary Teal (#19D3A2)
        case 1: return glm::vec3(1.000f, 0.541f, 0.122f); // Warm Orange (#FF8A1F)
        case 2: return glm::vec3(0.220f, 0.741f, 0.973f); // Sky Cyan (#38BDF8)
        case 3: return glm::vec3(0.957f, 0.247f, 0.369f); // Coral Rose (#F43F5E)
        case 4: return glm::vec3(0.659f, 0.333f, 0.969f); // Vivid Violet (#A855F7)
        case 5: return glm::vec3(0.918f, 0.702f, 0.031f); // Amber Gold (#EAB308)
        case 6: return glm::vec3(0.518f, 0.800f, 0.086f); // Lime Green (#84CC16)
        case 7: return glm::vec3(0.925f, 0.282f, 0.600f); // Hot Magenta (#EC4899)
        case 8: return glm::vec3(0.024f, 0.714f, 0.831f); // Deep Aqua (#06B6D4)
        default: return glm::vec3(0.976f, 0.451f, 0.086f); // Tangerine (#F97316)
    }
}

__device__ __forceinline__ int dev_trace_node_root_seed_index(
    const GpuColonizationNodeLayout* __restrict__ nodes,
    int num_nodes,
    int idx
) {
    int cur = idx;
    for (int hop = 0; hop < 4096; ++hop) {
        int p = nodes[cur].parent_index;
        if (p < 0 || p >= num_nodes || p == cur) break;
        cur = p;
    }
    return (nodes[cur].strand_id & 0xFFFF);
}

__device__ __forceinline__ int dev_trace_vol_section_root_seed_index(
    const CudaVolSection* __restrict__ sections,
    int num_sections,
    int idx
) {
    int cur = idx;
    for (int hop = 0; hop < 4096; ++hop) {
        int p = sections[cur].parent_sec;
        if (p < 0 || p >= num_sections || p == cur) break;
        cur = p;
    }
    return sections[cur].root_id;
}

__device__ __forceinline__ bool dev_try_build_cuda_triangle(
    const float* __restrict__ vertices,
    int num_verts,
    const uint32_t* __restrict__ indices,
    int t,
    CudaTriangle& out_tri
) {
    uint32_t i0 = indices[t * 3 + 0];
    uint32_t i1 = indices[t * 3 + 1];
    uint32_t i2 = indices[t * 3 + 2];
    if (i0 >= static_cast<uint32_t>(num_verts) ||
        i1 >= static_cast<uint32_t>(num_verts) ||
        i2 >= static_cast<uint32_t>(num_verts)) {
        return false;
    }

    glm::vec3 v0(vertices[i0 * 3 + 0], vertices[i0 * 3 + 1], vertices[i0 * 3 + 2]);
    glm::vec3 v1(vertices[i1 * 3 + 0], vertices[i1 * 3 + 1], vertices[i1 * 3 + 2]);
    glm::vec3 v2(vertices[i2 * 3 + 0], vertices[i2 * 3 + 1], vertices[i2 * 3 + 2]);

    glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
    float len2 = glm::dot(n, n);
    if (len2 > 1e-12f) {
        out_tri.v0 = v0;
        out_tri.v1 = v1;
        out_tri.v2 = v2;
        out_tri.normal = n * rsqrtf(len2);
        return true;
    }
    return false;
}

// --- Shared cross-TU kernel declarations: RELOCATED in Phase 3 -------------
// gpu_accumulate_face_normals_kernel, gpu_normalize_normals_kernel,
// gpu_reduce_mesh_bbox_kernel, gpu_reduce_min_float_kernel,
// gpu_mark_node_children_kernel, gpu_build_node_capsules_kernel,
// gpu_build_vol_section_capsules_kernel
// now live in engine/sdf/cuda_common.cuh (memory/launch-safety layer,
// AGENTS.md Sec A.2) so this header remains pure host/device math.

#endif // __CUDACC__

} // namespace rhizome
